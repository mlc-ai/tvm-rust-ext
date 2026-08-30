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

use tvm_ffi::derive::{Object, ObjectRef};
use tvm_ffi::{
    structural_visit, structural_walk, AnyView, DefRegionKind, Error, ObjectArc, ObjectRefCast,
    Result, VisitCallbacks, VisitContext, VisitInterrupt, WalkOrder, WalkResult, VALUE_ERROR,
};

use crate::ir::{CallObj, ExprObj, IntImmObj, PrimExpr, VarObj};
use crate::tirx::{
    AddObj, AssertStmtObj, BufferLoadObj, BufferStoreObj, EvaluateObj, ForObj, IfThenElseObj,
    MulObj, SBlockObj, SBlockRealizeObj, SeqStmtObj, StmtObj, SubObj,
};

/// Opaque Rust view of TVM's stateful arithmetic analyzer.
///
/// Unlike an IR node, the analyzer has private C++ implementation state, so
/// Rust owns only its reference-counted FFI handle and uses the registered
/// analysis functions for operations.
#[repr(C)]
#[derive(Object)]
#[type_key = "arith.Analyzer"]
#[type_final]
pub struct AnalyzerObj {
    base: tvm_ffi::Object,
}

/// Shared handle to one TVM arithmetic-analysis context.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct Analyzer {
    data: ObjectArc<AnalyzerObj>,
}

impl std::ops::Deref for Analyzer {
    type Target = AnalyzerObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl Analyzer {
    /// Construct a fresh native analyzer context.
    pub fn new() -> Result<Self> {
        tvm_ffi::cached_global_func!("arith.Analyzer")
            .call_tuple(())?
            .try_into()
    }

    /// Simplify a primitive expression with TVM's standard two analysis steps.
    pub fn simplify(&self, expression: &PrimExpr) -> Result<PrimExpr> {
        self.simplify_with_steps(expression, 2)
    }

    /// Simplify a primitive expression with an explicit analysis-step count.
    pub fn simplify_with_steps(&self, expression: &PrimExpr, steps: i32) -> Result<PrimExpr> {
        tvm_ffi::cached_global_func!("arith.AnalyzerSimplify")
            .call_tuple((self, expression, steps))?
            .try_into()
    }

    /// Prove that two primitive expressions are equal.
    pub fn can_prove_equal(&self, lhs: &PrimExpr, rhs: &PrimExpr) -> Result<bool> {
        tvm_ffi::cached_global_func!("arith.AnalyzerCanProveEqual")
            .call_tuple((self, lhs, rhs))?
            .try_into()
    }
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
    pub const ExprAnnotation: Self = Self(0);
    /// The expression does not interact with external state.
    pub const Pure: Self = Self(1);
    /// The expression may read external state but does not update it.
    pub const ReadState: Self = Self(2);
    /// The expression may update state or has unknown behavior.
    pub const UpdateState: Self = Self(3);
    /// The call carries special argument information.
    pub const SpecialCallArg: Self = Self(4);
    /// The call embeds opaque information and cannot be generated as code.
    pub const EmbedInfo: Self = Self(5);
    /// The call changes control flow.
    pub const ControlJump: Self = Self(6);
    /// C++ `kOpaque` is an alias of `kUpdateState`.
    pub const OPAQUE: Self = Self::UpdateState;

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
        self.0 >= Self::UpdateState.0
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

/// Opaque view of a registry-owned TVM operator.
#[repr(C)]
#[derive(Object)]
#[type_key = "ir.Op"]
#[type_final]
struct OpObj {
    base: ExprObj,
}

#[repr(C)]
#[derive(ObjectRef, Clone)]
struct Op {
    data: ObjectArc<OpObj>,
}

struct SideEffectAnalyzer {
    kind: CallEffectKind,
}

impl SideEffectAnalyzer {
    fn update(&mut self, kind: CallEffectKind) -> WalkResult {
        let kind = if kind > CallEffectKind::UpdateState {
            CallEffectKind::UpdateState
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
    fn walk_buffer_load(&mut self, _node: &BufferLoadObj) -> WalkResult {
        self.update(CallEffectKind::ReadState)
    }

    fn walk_call(&mut self, node: &CallObj) -> Result<WalkResult> {
        let Ok(op) = node.op.clone().try_cast::<Op>() else {
            return Ok(self.update(CallEffectKind::UpdateState));
        };
        let raw: i64 = tvm_ffi::cached_global_func!("ir.OpGetAttr")
            .call_tuple((op, tvm_ffi::String::from("TCallEffectKind")))?
            .try_into()?;
        Ok(self.update(raw.try_into()?))
    }
}

/// Classify whether evaluating a primitive expression reads or updates external state.
pub fn side_effect(expression: &PrimExpr) -> Result<CallEffectKind> {
    let mut analyzer = SideEffectAnalyzer {
        kind: CallEffectKind::Pure,
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
    pub blocks: usize,
    pub block_realizations: usize,
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

    fn walk_block(&mut self, _node: &SBlockObj) -> WalkResult {
        self.statements += 1;
        self.blocks += 1;
        WalkResult::Advance
    }

    fn walk_block_realization(&mut self, _node: &SBlockRealizeObj) -> WalkResult {
        self.statements += 1;
        self.block_realizations += 1;
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

    fn walk_buffer_load(&mut self, _node: &BufferLoadObj) -> WalkResult {
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
    pub predicated_loads: usize,
    pub predicated_stores: usize,
    pub maximum_load_rank: usize,
    pub maximum_store_rank: usize,
}

#[tvm_ffi::dispatch(walk)]
impl MemoryAccessStatistics {
    fn walk_load(&mut self, node: &BufferLoadObj) -> Result<WalkResult> {
        self.loads += 1;
        self.maximum_load_rank = self.maximum_load_rank.max(node.indices.len());
        self.predicated_loads += usize::from(node.predicate.is_some());
        Ok(WalkResult::Advance)
    }

    fn walk_store(&mut self, node: &BufferStoreObj) -> Result<WalkResult> {
        self.stores += 1;
        self.maximum_store_rank = self.maximum_store_rank.max(node.indices.len());
        self.predicated_stores += usize::from(node.predicate.is_some());
        Ok(WalkResult::Advance)
    }
}

/// Collect read/write counts, predicate counts, and maximum access rank.
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

    fn walk_integer(&mut self, node: &IntImmObj) -> Result<WalkResult> {
        self.events.push(ExprTraceEvent::Int(node.value));
        Ok(WalkResult::Advance)
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
        |node: &IntImmObj| -> Result<WalkResult> {
            if node.value == target {
                Ok(WalkResult::Interrupt)
            } else {
                Ok(WalkResult::Advance)
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
        |node: &IntImmObj| -> Result<WalkResult> { Ok(WalkResult::interrupt_with(node.value)) },
        WalkOrder::PreOrder,
    )?
    .map(|interrupt| i64::try_from(interrupt.value))
    .transpose()
}
