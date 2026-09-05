/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

use std::collections::HashMap;

use tvm_ffi::{
    structural_visit, structural_walk, AnyView, DefRegionKind, Error, ObjectIdentity,
    ObjectRefCore, Result, VisitCallbacks, VisitContext, VisitInterrupt, WalkOrder, WalkResult,
    VALUE_ERROR,
};

pub use crate::generated::arith::{
    Analyzer, AnalyzerObj, ConstIntBound, ConstIntBoundObj, ModularSet, ModularSetObj,
};
pub use crate::generated::ir::{IntSet, IntSetObj};

use crate::ir::prim::{AddObj, MulObj, SubObj};
use crate::ir::{CallObj, ExprObj, IntImmObj, OpObj, PrimExpr, TensorLoadObj, VarObj};
use crate::tirx::{
    AssertStmtObj, BufferStoreObj, EvaluateObj, ForObj, IfThenElseObj, PrimVar, SeqStmtObj, StmtObj,
};

/// Decompose `expression` as one linear coefficient per variable plus a base.
///
/// TVM returns an empty array when the expression is not linear in the given
/// variables.  On success the final array element is the variable-independent
/// base term.
pub fn detect_linear_equation(
    expression: &PrimExpr,
    variables: Vec<PrimVar>,
) -> Result<tvm_ffi::Array<PrimExpr>> {
    tvm_ffi::cached_global_func!("arith.DetectLinearEquation")
        .call_tuple((expression, tvm_ffi::Array::new(variables)))?
        .try_into()
}

/// ABI-compatible Rust representation of TVM's `tirx::CallEffectKind`.
///
/// This is an open integer newtype rather than a closed Rust enum, so a value
/// added by a newer native library remains representable and memory-safe.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct CallEffectKind(i32);

#[allow(non_upper_case_globals)]
impl CallEffectKind {
    /// The call is an expression annotation that behaves like an identity.
    pub const kExprAnnotation: Self = Self(0);
    /// The expression does not interact with external state.
    pub const kPure: Self = Self(1);
    /// The expression may read external state but does not update it.
    pub const kReadState: Self = Self(2);
    /// The expression may update state or has unknown behavior.
    pub const kUpdateState: Self = Self(3);
    /// C++ `kOpaque` is an alias of `kUpdateState`.
    pub const kOpaque: Self = Self::kUpdateState;
    /// The call carries special argument information.
    pub const kSpecialCallArg: Self = Self(4);
    /// The call embeds opaque information and cannot be generated as code.
    pub const kEmbedInfo: Self = Self(5);
    /// The call changes control flow.
    pub const kControlJump: Self = Self(6);

    /// Preserve an enumerator not yet known by this Rust binding.
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }

    /// Return the native integer representation.
    pub const fn as_raw(self) -> i32 {
        self.0
    }

    /// Return whether discarding evaluation could remove a state update.
    pub fn may_update_state(self) -> bool {
        self.0 >= Self::kUpdateState.0
    }
}

impl TryFrom<i64> for CallEffectKind {
    type Error = Error;

    fn try_from(value: i64) -> Result<Self> {
        i32::try_from(value).map(Self).map_err(|_| {
            Error::new(
                VALUE_ERROR,
                &format!(
                    "tirx.CallEffectKind value {value} does not fit its native i32 representation"
                ),
                "",
            )
        })
    }
}

/// Read a boolean attribute from a registry-owned operator.
///
/// Operator attributes belong to the native operator registry rather than to
/// the reflected `Call` fields, so Rust intentionally uses TVM's existing
/// language-independent lookup here.
pub(crate) fn operator_bool_attr(operator: &crate::ir::Expr, name: &str) -> Result<Option<bool>> {
    if operator.as_node::<OpObj>().is_none() {
        return Ok(None);
    }
    tvm_ffi::cached_global_func!("ir.OpGetAttr")
        .call_tuple((operator, tvm_ffi::String::from(name)))?
        .try_into()
}

struct SideEffectAnalyzer {
    kind: CallEffectKind,
    operator_effects: HashMap<ObjectIdentity, CallEffectKind>,
}

impl SideEffectAnalyzer {
    fn update(&mut self, kind: CallEffectKind) -> WalkResult {
        let kind = if kind > CallEffectKind::kUpdateState {
            CallEffectKind::kUpdateState
        } else {
            kind
        };
        self.kind = self.kind.max(kind);
        if self.kind.may_update_state() {
            WalkResult::Interrupt
        } else {
            WalkResult::Advance
        }
    }
}

#[tvm_ffi::dispatch(walk)]
impl SideEffectAnalyzer {
    fn walk_tensor_load(&mut self, _node: &TensorLoadObj) -> WalkResult {
        self.update(CallEffectKind::kReadState)
    }

    fn walk_call(&mut self, node: &CallObj) -> Result<WalkResult> {
        if node.op.as_node::<OpObj>().is_none() {
            return Ok(self.update(CallEffectKind::kUpdateState));
        }
        let identity = ObjectIdentity::of(&node.op);
        if let Some(kind) = self.operator_effects.get(&identity).copied() {
            return Ok(self.update(kind));
        }
        let raw: i64 = tvm_ffi::cached_global_func!("ir.OpGetAttr")
            .call_tuple((&node.op, tvm_ffi::String::from("TCallEffectKind")))?
            .try_into()?;
        let kind = raw.try_into()?;
        self.operator_effects.insert(identity, kind);
        Ok(self.update(kind))
    }
}

/// Classify whether evaluating a primitive expression reads or updates external state.
pub fn side_effect(expression: &PrimExpr) -> Result<CallEffectKind> {
    let mut analyzer = SideEffectAnalyzer {
        kind: CallEffectKind::kPure,
        operator_effects: HashMap::new(),
    };
    structural_walk(expression, &mut analyzer, WalkOrder::PreOrder)?;
    Ok(analyzer.kind)
}

/// Count expression nodes using TVM's language-agnostic structural protocol.
pub fn expr_complexity<R>(root: &R) -> Result<usize>
where
    for<'a> AnyView<'a>: From<&'a R>,
{
    let mut count = 0;
    structural_walk(
        root,
        |_: &ExprObj| {
            count += 1;
            WalkResult::Advance
        },
        WalkOrder::PreOrder,
    )?;
    Ok(count)
}

/// Counts of representative IR node categories observed during a walk.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct NodeStatistics {
    pub expressions: usize,
    pub statements: usize,
    pub int_immediates: usize,
    pub additions: usize,
    pub subtractions: usize,
    pub multiplications: usize,
    pub calls: usize,
    pub variables: usize,
    pub variable_definitions: usize,
    pub variable_uses: usize,
    pub assertions: usize,
    pub evaluations: usize,
    pub sequences: usize,
    pub conditionals: usize,
    pub loops: usize,
    pub buffer_loads: usize,
    pub buffer_stores: usize,
}

#[tvm_ffi::dispatch(walk)]
impl NodeStatistics {
    fn walk_assert(&mut self, _node: &AssertStmtObj) -> WalkResult {
        self.statements += 1;
        self.assertions += 1;
        WalkResult::Advance
    }

    fn walk_evaluate(&mut self, _node: &EvaluateObj) -> WalkResult {
        self.statements += 1;
        self.evaluations += 1;
        WalkResult::Advance
    }

    fn walk_sequence(&mut self, _node: &SeqStmtObj) -> WalkResult {
        self.statements += 1;
        self.sequences += 1;
        WalkResult::Advance
    }

    fn walk_conditional(&mut self, _node: &IfThenElseObj) -> WalkResult {
        self.statements += 1;
        self.conditionals += 1;
        WalkResult::Advance
    }

    fn walk_loop(&mut self, _node: &ForObj) -> WalkResult {
        self.statements += 1;
        self.loops += 1;
        WalkResult::Advance
    }

    fn walk_buffer_store(&mut self, _node: &BufferStoreObj) -> WalkResult {
        self.statements += 1;
        self.buffer_stores += 1;
        WalkResult::Advance
    }

    fn walk_other_statement(&mut self, _node: &StmtObj) -> WalkResult {
        self.statements += 1;
        WalkResult::Advance
    }

    fn walk_integer(&mut self, _node: &IntImmObj) -> WalkResult {
        self.expressions += 1;
        self.int_immediates += 1;
        WalkResult::Advance
    }

    fn walk_addition(&mut self, _node: &AddObj) -> WalkResult {
        self.expressions += 1;
        self.additions += 1;
        WalkResult::Advance
    }

    fn walk_subtraction(&mut self, _node: &SubObj) -> WalkResult {
        self.expressions += 1;
        self.subtractions += 1;
        WalkResult::Advance
    }

    fn walk_multiplication(&mut self, _node: &MulObj) -> WalkResult {
        self.expressions += 1;
        self.multiplications += 1;
        WalkResult::Advance
    }

    fn walk_call(&mut self, _node: &CallObj) -> WalkResult {
        self.expressions += 1;
        self.calls += 1;
        WalkResult::Advance
    }

    fn walk_tensor_load(&mut self, _node: &TensorLoadObj) -> WalkResult {
        self.expressions += 1;
        self.buffer_loads += 1;
        WalkResult::Advance
    }

    fn walk_variable(&mut self, _node: &VarObj, kind: DefRegionKind) -> WalkResult {
        self.expressions += 1;
        self.variables += 1;
        match kind {
            DefRegionKind::None => self.variable_uses += 1,
            DefRegionKind::Recursive | DefRegionKind::NonRecursive => {
                self.variable_definitions += 1
            }
        }
        WalkResult::Advance
    }

    fn walk_other_expression(&mut self, _node: &ExprObj) -> WalkResult {
        self.expressions += 1;
        WalkResult::Advance
    }
}

/// Collect representative node counts with generated typed dispatch.
pub fn node_statistics<R>(root: &R) -> Result<NodeStatistics>
where
    for<'a> AnyView<'a>: From<&'a R>,
{
    let mut statistics = NodeStatistics::default();
    structural_walk(root, &mut statistics, WalkOrder::PreOrder)?;
    Ok(statistics)
}

/// Summary of concrete buffer accesses in a TIR subtree.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MemoryAccessStatistics {
    pub loads: usize,
    pub stores: usize,
    pub maximum_load_rank: usize,
    pub maximum_store_rank: usize,
}

#[tvm_ffi::dispatch(walk)]
impl MemoryAccessStatistics {
    fn walk_load(&mut self, node: &TensorLoadObj) -> WalkResult {
        self.loads += 1;
        self.maximum_load_rank = self.maximum_load_rank.max(node.indices.len());
        WalkResult::Advance
    }

    fn walk_store(&mut self, node: &BufferStoreObj) -> WalkResult {
        self.stores += 1;
        self.maximum_store_rank = self.maximum_store_rank.max(node.indices.len());
        WalkResult::Advance
    }
}

/// Collect read/write counts and maximum access rank.
pub fn memory_access_statistics<R>(root: &R) -> Result<MemoryAccessStatistics>
where
    for<'a> AnyView<'a>: From<&'a R>,
{
    let mut statistics = MemoryAccessStatistics::default();
    structural_walk(root, &mut statistics, WalkOrder::PreOrder)?;
    Ok(statistics)
}

/// Lexical loop nesting measured by explicitly controlling `For.body` recursion.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LoopNesting {
    pub loops: usize,
    pub maximum_depth: usize,
    current_depth: usize,
}

/// Return the number of loops and maximum lexical nesting depth.
///
/// Unlike `structural_walk`, this visitor can bracket the recursive call for a
/// loop body with state updates.  Loop bounds are visited at their enclosing
/// depth, while the body is visited one level deeper.
pub fn loop_nesting<R>(root: &R) -> Result<LoopNesting>
where
    for<'a> AnyView<'a>: From<&'a R>,
{
    let mut visitor = VisitCallbacks::new(LoopNesting::default(), visit_loop_body);
    structural_visit(root, &mut visitor)?;
    Ok(visitor.into_state())
}

fn visit_loop_body(
    node: &ForObj,
    visitor: &mut VisitContext<'_, LoopNesting>,
) -> Result<Option<VisitInterrupt>> {
    visitor.state_mut().loops += 1;

    if let Some(interrupt) = visitor.visit_with(&node.loop_var, DefRegionKind::Recursive)? {
        return Ok(Some(interrupt));
    }
    if let Some(interrupt) = visitor.visit(&node.min)? {
        return Ok(Some(interrupt));
    }
    if let Some(interrupt) = visitor.visit(&node.extent)? {
        return Ok(Some(interrupt));
    }

    {
        let state = visitor.state_mut();
        state.current_depth += 1;
        state.maximum_depth = state.maximum_depth.max(state.current_depth);
    }
    let body_result = visitor.visit(&node.body);
    visitor.state_mut().current_depth -= 1;
    if let Some(interrupt) = body_result? {
        return Ok(Some(interrupt));
    }

    if let Some(interrupt) = visitor.visit(&node.thread_binding)? {
        return Ok(Some(interrupt));
    }
    if let Some(interrupt) = visitor.visit(&node.annotations)? {
        return Ok(Some(interrupt));
    }
    if let Some(step) = &node.step {
        if let Some(interrupt) = visitor.visit(step)? {
            return Ok(Some(interrupt));
        }
    }
    Ok(None)
}

/// One event in a typed expression traversal.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExprTraceEvent {
    Add,
    Int(i64),
}

#[derive(Default)]
struct ExprTrace {
    events: Vec<ExprTraceEvent>,
}

#[tvm_ffi::dispatch(walk)]
impl ExprTrace {
    fn walk_addition(&mut self, _node: &AddObj) -> WalkResult {
        self.events.push(ExprTraceEvent::Add);
        WalkResult::Advance
    }

    fn walk_integer(&mut self, node: &IntImmObj) -> WalkResult {
        self.events.push(ExprTraceEvent::Int(node.value));
        WalkResult::Advance
    }
}

/// Return the typed callback order for additions and integer literals.
pub fn expression_trace<R>(root: &R, order: WalkOrder) -> Result<Vec<ExprTraceEvent>>
where
    for<'a> AnyView<'a>: From<&'a R>,
{
    let mut trace = ExprTrace::default();
    structural_walk(root, &mut trace, order)?;
    Ok(trace.events)
}

/// Return early when an integer literal with `target` is found.
pub fn contains_int<R>(root: &R, target: i64) -> Result<bool>
where
    for<'a> AnyView<'a>: From<&'a R>,
{
    let outcome = structural_walk(
        root,
        |node: &IntImmObj| {
            if node.value == target {
                WalkResult::Interrupt
            } else {
                WalkResult::Advance
            }
        },
        WalkOrder::PreOrder,
    )?;
    Ok(outcome.is_some())
}

/// Return the first integer literal in pre-order using an interrupt payload.
pub fn first_int<R>(root: &R) -> Result<Option<i64>>
where
    for<'a> AnyView<'a>: From<&'a R>,
{
    structural_walk(
        root,
        |node: &IntImmObj| WalkResult::interrupt_with(node.value),
        WalkOrder::PreOrder,
    )?
    .map(|interrupt| i64::try_from(interrupt.value))
    .transpose()
}
