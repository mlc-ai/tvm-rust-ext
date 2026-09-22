/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

use std::collections::HashMap;

#[cfg(test)]
#[path = "../../tests/unit/constraint_cleanup.rs"]
mod constraint_cleanup_tests;

use tvm_ffi::derive::{Object, ObjectRef};
use tvm_ffi::{
    structural_mutate, Any, AnyCompatible, Array, FieldGetter, Function, Map, Mutator, ObjectArc,
    ObjectCore, ObjectIdentity, ObjectRefCast, ObjectRefCore, Result, String as FfiString,
    StructuralView, RUNTIME_ERROR,
};

use super::utils::{
    array_same_as, finish_constraint_contexts, get_operator, int_value, is_call,
    is_evaluate_zero as is_no_op, option_same_as, with_prim_func_body, BufferRemaps,
};
use super::{create_prim_func_pass_with_context, Pass, PassContext};
use crate::analysis::{side_effect, Analyzer, CallEffectKind, IntSet};
use crate::ir::{Call, CallObj, Expr, IntImm, PrimExpr, Range, TensorLoad, TensorLoadObj, Var};
use crate::prim::{Let, Not, Select, Sub, GT, LE};
use crate::te::Reduce;
use crate::tirx::{
    AssertStmt, AssertStmtObj, AttrStmt, Bind, BufferStore, BufferVar, Evaluate, For, IfThenElse,
    IterVar, PrimFunc, SeqStmt, Stmt,
};

const DEBUG_SKIP_REGION: &str = "pragma_debug_skip_region";
const ASYNC_WAIT_QUEUE_SCOPE: &str = "async_wait_queue_scope";
const ASYNC_WAIT_INFLIGHT_COUNT: &str = "async_wait_inflight_count";
const THREAD_EXTENT: &str = "thread_extent";
const VIRTUAL_THREAD: &str = "virtual_thread";

/// Opaque read-only view of TVM's `tirx.transform.RemoveNoOpConfig`.
#[repr(C)]
#[derive(Object)]
#[type_key = "tirx.transform.RemoveNoOpConfig"]
#[type_final]
struct RemoveNoOpConfigObj {
    base: tvm_ffi::Object,
}

#[repr(C)]
#[derive(ObjectRef, Clone)]
struct RemoveNoOpConfig {
    data: ObjectArc<RemoveNoOpConfigObj>,
}

impl std::ops::Deref for RemoveNoOpConfig {
    type Target = RemoveNoOpConfigObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl RemoveNoOpConfig {
    fn field<T>(&self, name: &str) -> Result<T>
    where
        T: TryFrom<Any, Error = tvm_ffi::Error>,
    {
        FieldGetter::new(RemoveNoOpConfigObj::type_index(), name)?.get(&**self)
    }

    fn max_simplification_steps(&self) -> Result<i64> {
        self.field("max_simplification_steps")
    }

    fn ignore_profiler_call(&self) -> Result<bool> {
        self.field("ignore_profiler_call")
    }
}

#[derive(Clone, Copy, Default)]
struct RemoveNoOpOptions {
    max_simplification_steps: i64,
    ignore_profiler_call: bool,
}

impl RemoveNoOpOptions {
    fn from_context(context: &PassContext) -> Result<Self> {
        let Some(raw) = context.config()?.get(&FfiString::from("tirx.RemoveNoOp"))? else {
            return Ok(Self::default());
        };
        let config = RemoveNoOpConfig::try_from(raw)?;
        Ok(Self {
            max_simplification_steps: config.max_simplification_steps()?,
            ignore_profiler_call: config.ignore_profiler_call()?,
        })
    }
}

/// Remove no-op statements using the same non-SBlock rewrite as TVM.
pub fn remove_no_op_prim_func(function: PrimFunc) -> Result<PrimFunc> {
    remove_no_op_with_options(function, RemoveNoOpOptions::default())
}

fn remove_no_op_with_options(function: PrimFunc, options: RemoveNoOpOptions) -> Result<PrimFunc> {
    let mut remover = NoOpRemover::new(options)?;
    let body = structural_mutate(function.body().clone(), &mut remover)?.try_into()?;
    Ok(with_prim_func_body(function, body))
}

/// Build TVM's `tirx.RemoveNoOp` PrimFunc pass in Rust.
pub fn remove_no_op() -> Result<Pass> {
    create_prim_func_pass_with_context(
        "tirx.RemoveNoOp",
        0,
        Vec::new(),
        false,
        |function, context| {
            let options = RemoveNoOpOptions::from_context(&context)?;
            remove_no_op_with_options(function, options)
        },
    )
}

struct NoOpRemover {
    analyzer: Analyzer,
    ignore_profiler_call: bool,
    profiler_operators: Vec<Expr>,
    variable_domains: HashMap<ObjectIdentity, (Var, IntSet)>,
    buffer_remaps: BufferRemaps,
    likely_operator: Expr,
    if_then_else_operator: Expr,
    bitwise_and_operator: Expr,
}

#[tvm_ffi::dispatch(mutate)]
impl NoOpRemover {
    fn mutate_bind(&mut self, value: Bind, mutator: &mut Mutator) -> Result<Bind> {
        // Bind.var is a definition, not a recursive use.
        let bound_value: Expr = mutator.mutate(self, &value.value)?.try_into()?;
        if let Ok(bound_value) = PrimExpr::try_from(&bound_value) {
            if side_effect(&bound_value)? <= CallEffectKind::kPure {
                self.analyzer.bind_expression(&value.var, &bound_value)?;
            }
        }
        if bound_value.same_as(&value.value) {
            return Ok(value);
        }
        Ok(value.copy_with(value.var.clone(), bound_value))
    }

    fn mutate_let(&mut self, value: Let, mutator: &mut Mutator) -> Result<Let> {
        let bound_value: PrimExpr = mutator.mutate(self, &value.value)?.try_into()?;
        if side_effect(&bound_value)? <= CallEffectKind::kPure {
            self.analyzer.bind_expression(&value.var, &bound_value)?;
        }
        let body: PrimExpr = mutator.mutate(self, &value.body)?.try_into()?;
        if bound_value.same_as(&value.value) && body.same_as(&value.body) {
            return Ok(value);
        }
        Ok(value.copy_with(value.var.clone(), bound_value, body))
    }

    fn mutate_reduce(&mut self, value: Reduce, mutator: &mut Mutator) -> Result<Reduce> {
        // IRMutatorWithAnalyzer binds every reduction axis before visiting any
        // reduction child.
        for axis in value.axis.iter() {
            if let Some(domain) = axis.dom()? {
                self.analyzer.bind(axis.var()?.as_var(), &domain)?;
            }
        }

        let mut axes = Vec::with_capacity(value.axis.len());
        for axis in value.axis.iter() {
            let original_domain = axis.dom()?;
            let domain = original_domain
                .as_ref()
                .map(|domain| -> Result<Range> {
                    let minimum: PrimExpr = mutator.mutate(self, &domain.min)?.try_into()?;
                    let extent: PrimExpr = mutator.mutate(self, &domain.extent)?.try_into()?;
                    if minimum.same_as(&domain.min) && extent.same_as(&domain.extent) {
                        Ok(domain.clone())
                    } else {
                        Ok(domain.copy_with(minimum, extent))
                    }
                })
                .transpose()?;
            if option_same_as(&domain, &original_domain) {
                axes.push(axis);
            } else {
                axes.push(IterVar::with_metadata(
                    domain,
                    axis.var()?.as_var().clone(),
                    axis.iter_type()?,
                    axis.thread_tag()?.as_str(),
                    axis.span()?.as_ref(),
                )?);
            }
        }

        let source: Array<PrimExpr> = mutator.mutate(self, &value.source)?.try_into()?;
        let init: Array<PrimExpr> = mutator.mutate(self, &value.init)?.try_into()?;
        let condition: PrimExpr = mutator.mutate(self, &value.condition)?.try_into()?;
        let axes = Array::new(axes);
        if array_same_as(&source, &value.source)
            && array_same_as(&init, &value.init)
            && array_same_as(&axes, &value.axis)
            && condition.same_as(&value.condition)
        {
            return Ok(value);
        }
        Ok(Reduce::from_complete_fields(
            value.span.clone(),
            value.ty.clone().try_cast()?,
            value.combiner.clone(),
            source,
            init,
            axes,
            condition,
            value.value_index,
        ))
    }

    fn mutate_assertion(&mut self, value: AssertStmt, mutator: &mut Mutator) -> Result<AssertStmt> {
        let condition: PrimExpr = mutator.mutate(self, &value.condition)?.try_into()?;
        if condition.same_as(&value.condition) {
            return Ok(value);
        }
        Ok(value.copy_with(
            condition,
            value.error_kind.clone(),
            value.message_parts.clone(),
        ))
    }

    fn mutate_select(&mut self, value: Select, mutator: &mut Mutator) -> Result<PrimExpr> {
        let condition: PrimExpr = mutator.mutate(self, &value.condition)?.try_into()?;
        let true_value =
            mutate_under_constraint_with_facts(self, mutator, &value.true_value, &condition)?;
        let negative = self
            .analyzer
            .simplify(&PrimExpr::from(Not::new(condition.clone())?))?;
        let false_value = mutate_under_constraint(self, mutator, &value.false_value, &negative)?;
        if let Some(condition) = int_value(&condition) {
            return if condition != 0 {
                Ok(true_value)
            } else {
                Ok(false_value)
            };
        }
        if condition.same_as(&value.condition)
            && true_value.same_as(&value.true_value)
            && false_value.same_as(&value.false_value)
        {
            return Ok(value.into());
        }
        Ok(value.copy_with(condition, true_value, false_value).into())
    }

    fn mutate_call(&mut self, value: Call, mutator: &mut Mutator) -> Result<Expr> {
        if !value.op.same_as(&self.if_then_else_operator) || value.args.len() != 3 {
            let args: Array<Expr> = mutator.mutate(self, &value.args)?.try_into()?;
            if array_same_as(&args, &value.args) {
                return Ok(value.into());
            }
            return Ok(value
                .copy_with(value.ty.clone(), value.op.clone(), args)
                .into());
        }

        let original_condition = value.args.get(0).expect("condition argument is present");
        let original_true = value.args.get(1).expect("true argument is present");
        let original_false = value.args.get(2).expect("false argument is present");
        let condition: PrimExpr = mutator.mutate(self, &original_condition)?.try_into()?;
        let true_value =
            mutate_under_constraint_with_facts(self, mutator, &original_true, &condition)?;
        let negative: PrimExpr = Not::new(condition.clone())?.into();
        let false_value = mutate_under_constraint(self, mutator, &original_false, &negative)?;
        if let Some(condition) = int_value(&condition) {
            return if condition != 0 {
                Ok(true_value)
            } else {
                Ok(false_value)
            };
        }
        if condition.same_as(&PrimExpr::try_from(original_condition)?)
            && true_value.same_as(&original_true)
            && false_value.same_as(&original_false)
        {
            return Ok(value.into());
        }
        Ok(value
            .copy_with(
                value.ty.clone(),
                value.op.clone(),
                tvm_ffi::Array::new(vec![condition.into(), true_value, false_value]),
            )
            .into())
    }

    fn mutate_attribute(&mut self, value: AttrStmt, mutator: &mut Mutator) -> Result<Stmt> {
        if value.attr_key.as_str() == DEBUG_SKIP_REGION {
            return evaluate_zero();
        }

        if value.attr_key.as_str() == ASYNC_WAIT_QUEUE_SCOPE {
            let inner = value.body.clone().try_cast::<AttrStmt>().map_err(|_| {
                tvm_ffi::Error::new(
                    RUNTIME_ERROR,
                    "async_wait_queue_scope must contain async_wait_inflight_count",
                    "",
                )
            })?;
            if inner.attr_key.as_str() != ASYNC_WAIT_INFLIGHT_COUNT {
                return Err(tvm_ffi::Error::new(
                    RUNTIME_ERROR,
                    "async_wait_queue_scope must contain async_wait_inflight_count",
                    "",
                ));
            }
            let zero = zero_like(&PrimExpr::try_from(inner.value.clone())?);
            let negative: PrimExpr = crate::prim::LT::new(inner.value.clone(), zero)?.into();
            if Analyzer::new()?.can_prove(&negative)? {
                return mutator.mutate(self, &inner.body)?.try_into();
            }
        }

        if matches!(value.attr_key.as_str(), THREAD_EXTENT | VIRTUAL_THREAD) {
            let iteration = IterVar::try_from(value.node.clone())?;
            let variable = iteration.var()?;
            let domain = Range::from_min_extent(
                zero_like(&PrimExpr::try_from(value.value.clone())?),
                value.value.clone(),
            )?;
            self.analyzer.bind(variable.as_var(), &domain)?;
        }

        // Match StmtExprMutator: AttrStmt.node is metadata and must not be
        // recursively rewritten.
        let attr_value: Expr = mutator.mutate(self, &value.value)?.try_into()?;
        let body: Stmt = mutator.mutate(self, &value.body)?.try_into()?;
        let mutated = if attr_value.same_as(&value.value) && body.same_as(&value.body) {
            value
        } else {
            value.copy_with(value.node.clone(), value.attr_key.clone(), attr_value, body)
        };
        if is_no_op(&mutated.body) {
            self.make_evaluate(PrimExpr::try_from(mutated.value.clone())?)
        } else {
            Ok(mutated.into())
        }
    }

    fn mutate_conditional(&mut self, value: IfThenElse, mutator: &mut Mutator) -> Result<Stmt> {
        let condition: PrimExpr = mutator.mutate(self, &value.condition)?.try_into()?;
        let real_condition = self.unwrap_likely(&condition)?;
        let then_case =
            mutate_under_constraint_with_facts(self, mutator, &value.then_case, &real_condition)?;
        let negative = self
            .analyzer
            .simplify(&PrimExpr::from(Not::new(real_condition.clone())?))?;
        let else_case = value
            .else_case
            .as_ref()
            .map(|branch| mutate_under_constraint(self, mutator, branch, &negative))
            .transpose()?;

        if let Some(condition) = int_value(&self.analyzer.simplify(&real_condition)?) {
            return if condition != 0 {
                Ok(then_case)
            } else {
                else_case.map_or_else(evaluate_zero, Ok)
            };
        }

        let conditional =
            IfThenElse::from_complete_fields(value.span.clone(), condition, then_case, else_case);

        let then_is_no_op = is_no_op(&conditional.then_case);
        match conditional.else_case.clone() {
            Some(else_case) if then_is_no_op && is_no_op(&else_case) => {
                self.make_evaluate(conditional.condition.clone())
            }
            Some(else_case) if is_no_op(&else_case) => Ok(IfThenElse::new(
                conditional.condition.clone(),
                conditional.then_case.clone(),
            )?
            .into()),
            Some(else_case) if then_is_no_op => {
                Ok(IfThenElse::new(Not::new(conditional.condition.clone())?, else_case)?.into())
            }
            None if then_is_no_op => self.make_evaluate(conditional.condition.clone()),
            _ => Ok(conditional.into()),
        }
    }

    fn mutate_loop(&mut self, value: For, mutator: &mut Mutator) -> Result<Stmt> {
        let domains: Map<Var, IntSet> = self
            .variable_domains
            .values()
            .map(|(variable, domain)| (variable.clone(), domain.clone()))
            .collect();
        let extent_set = self.analyzer.int_set(&value.extent, &domains)?;
        let maximum = extent_set.maximum()?;
        let non_positive: PrimExpr = LE::new(maximum.clone(), zero_like(&maximum))?.into();
        if self.analyzer.can_prove(&non_positive)? {
            return evaluate_zero();
        }

        let domain = Range::from_min_extent(value.min.clone(), value.extent.clone())?;
        self.analyzer.bind(value.loop_var.as_var(), &domain)?;

        let one = one_like(&value.extent);
        let extent_minus_one: PrimExpr = Sub::new(value.extent.clone(), one)?.into();
        let maximum: PrimExpr = crate::prim::Add::new(value.min.clone(), extent_minus_one)?.into();
        let integer_domain = IntSet::interval(value.min.clone(), maximum)?;
        let identity = ObjectIdentity::of(&value.loop_var);
        let previous = self.variable_domains.insert(
            identity.clone(),
            (value.loop_var.as_var().clone(), integer_domain),
        );
        let mutated = (|| -> Result<For> {
            let minimum: PrimExpr = mutator.mutate(self, &value.min)?.try_into()?;
            let extent: PrimExpr = mutator.mutate(self, &value.extent)?.try_into()?;
            let step: Option<PrimExpr> = mutator.mutate(self, &value.step)?.try_into()?;
            let positive: PrimExpr = GT::new(extent.clone(), zero_like(&extent))?.into();
            let body = mutate_under_constraint_with_facts(self, mutator, &value.body, &positive)?;
            if minimum.same_as(&value.min)
                && extent.same_as(&value.extent)
                && option_same_as(&step, &value.step)
                && body.same_as(&value.body)
            {
                return Ok(value.clone());
            }
            Ok(For::from_complete_fields(
                value.span.clone(),
                value.loop_var.clone(),
                minimum,
                extent,
                value.kind,
                body,
                value.thread_binding.clone(),
                value.annotations.clone(),
                step,
            ))
        })();
        match previous {
            Some(previous) => {
                self.variable_domains.insert(identity, previous);
            }
            None => {
                self.variable_domains.remove(&identity);
            }
        }
        let mutated = mutated?;
        if int_value(&mutated.extent) == Some(0) {
            return evaluate_zero();
        }
        if is_no_op(&mutated.body) {
            return self.make_evaluate_values([mutated.min.clone(), mutated.extent.clone()]);
        }
        Ok(mutated.into())
    }

    fn mutate_evaluate(&mut self, value: Evaluate) -> Result<Evaluate> {
        if self.has_side_effect(&value.value)? {
            Ok(value)
        } else {
            Evaluate::from_i64(0)
        }
    }

    fn mutate_store(&mut self, value: BufferStore) -> Result<Stmt> {
        let load = TensorLoad::from_buffer(
            value.buffer.as_var().clone(),
            value.indices.iter().map(Expr::from).collect(),
        )?;
        let difference: PrimExpr = Sub::new(value.value.clone(), load)?.into();
        let equal_to_zero: PrimExpr =
            crate::prim::EQ::new(difference.clone(), zero_like(&difference))?.into();
        if int_value(&self.analyzer.simplify(&equal_to_zero)?) == Some(1) {
            return self.store_side_effects(&value);
        }

        if let Some(load) = value.value.as_node::<TensorLoadObj>() {
            let source: BufferVar = (&load.source).try_into()?;
            if source.same_as(&value.buffer)
                && self.buffer_geometry_equal(&source, &value.buffer)?
                && self.array_equal(&load.indices, &value.indices)?
            {
                return self.store_side_effects(&value);
            }
        }

        Ok(value.into())
    }

    fn mutate_sequence(&mut self, value: SeqStmt, mutator: &mut Mutator) -> Result<Stmt> {
        let mut exits = Vec::<Function>::new();
        let result = (|| -> Result<Stmt> {
            let mut statements = Vec::with_capacity(value.seq.len());
            for statement in value.seq.iter() {
                let mutated: Stmt = mutator.mutate(self, &statement)?.try_into()?;
                if let Some(assertion) = mutated.as_node::<AssertStmtObj>() {
                    exits.push(self.analyzer.enter_constraint(&assertion.condition)?);
                }
                statements.push(mutated);
            }
            Stmt::sequence(statements)
        })();
        finish_constraint_contexts(result, exits)
    }

    fn mutate_default(&mut self, value: &StructuralView, mutator: &mut Mutator) -> Result<Any> {
        BufferRemaps::mutate_default(self, mutator, value, |state| &mut state.buffer_remaps)
    }
}

impl NoOpRemover {
    fn new(options: RemoveNoOpOptions) -> Result<Self> {
        let analyzer = Analyzer::new()?;
        analyzer.set_maximum_rewrite_steps(options.max_simplification_steps)?;
        let profiler_operators = if options.ignore_profiler_call {
            [
                "tirx.timer_init_cuda",
                "tirx.timer_start_cuda",
                "tirx.timer_end_cuda",
                "tirx.timer_finalize_cuda",
            ]
            .into_iter()
            .map(get_operator)
            .collect::<Result<Vec<_>>>()?
        } else {
            Vec::new()
        };
        Ok(Self {
            analyzer,
            ignore_profiler_call: options.ignore_profiler_call,
            profiler_operators,
            variable_domains: HashMap::new(),
            buffer_remaps: BufferRemaps::default(),
            likely_operator: get_operator("prim.likely")?,
            if_then_else_operator: get_operator("prim.if_then_else")?,
            bitwise_and_operator: get_operator("prim.bitwise_and")?,
        })
    }

    fn unwrap_likely(&self, condition: &PrimExpr) -> Result<PrimExpr> {
        let Some(call) = condition.as_node::<CallObj>() else {
            return Ok(condition.clone());
        };
        if !call.op.same_as(&self.likely_operator) || call.args.len() != 1 {
            return Ok(condition.clone());
        }
        PrimExpr::try_from(call.args.get(0).expect("one likely argument is present"))
    }

    fn has_side_effect(&self, value: &Expr) -> Result<bool> {
        if let Ok(primitive) = PrimExpr::try_from(value) {
            if self.ignore_profiler_call && is_profiler_call(&primitive, &self.profiler_operators) {
                return Ok(false);
            }
            return Ok(side_effect(&primitive)? > CallEffectKind::kReadState);
        }
        Ok(is_call(value))
    }

    fn make_evaluate(&self, value: PrimExpr) -> Result<Stmt> {
        if side_effect(&value)? > CallEffectKind::kReadState {
            Ok(Evaluate::new(value)?.into())
        } else {
            evaluate_zero()
        }
    }

    fn make_evaluate_values<I>(&self, values: I) -> Result<Stmt>
    where
        I: IntoIterator<Item = PrimExpr>,
    {
        let mut statements = Vec::new();
        for value in values {
            if side_effect(&value)? > CallEffectKind::kReadState {
                statements.push(Evaluate::new(value)?.into());
            }
        }
        Stmt::sequence(statements)
    }

    fn store_side_effects(&self, store: &BufferStore) -> Result<Stmt> {
        self.make_evaluate_values(std::iter::once(store.value.clone()).chain(store.indices.iter()))
    }

    fn buffer_geometry_equal(&self, lhs: &BufferVar, rhs: &BufferVar) -> Result<bool> {
        let lhs_type = lhs.type_annotation();
        let rhs_type = rhs.type_annotation();
        Ok(self
            .analyzer
            .can_prove_equal(&lhs_type.elem_offset, &rhs_type.elem_offset)?
            && self.array_equal(&lhs_type.shape, &rhs_type.shape)?
            && self.array_equal(&lhs_type.strides, &rhs_type.strides)?)
    }

    fn array_equal(
        &self,
        lhs: &tvm_ffi::Array<PrimExpr>,
        rhs: &tvm_ffi::Array<PrimExpr>,
    ) -> Result<bool> {
        if lhs.len() != rhs.len() {
            return Ok(false);
        }
        for (lhs, rhs) in lhs.iter().zip(rhs.iter()) {
            if !self.analyzer.can_prove_equal(&lhs, &rhs)? {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

fn mutate_under_constraint<T>(
    remover: &mut NoOpRemover,
    mutator: &mut Mutator,
    value: &T,
    constraint: &PrimExpr,
) -> Result<T>
where
    T: AnyCompatible + TryFrom<Any, Error = tvm_ffi::Error>,
{
    let analyzer = remover.analyzer.clone();
    analyzer.with_constraint(constraint, || mutator.mutate(remover, value)?.try_into())
}

fn mutate_under_constraint_with_facts<T>(
    remover: &mut NoOpRemover,
    mutator: &mut Mutator,
    value: &T,
    constraint: &PrimExpr,
) -> Result<T>
where
    T: AnyCompatible + TryFrom<Any, Error = tvm_ffi::Error>,
{
    let exits = super::analyzer_constraints::enter_constraint_facts(
        &remover.analyzer,
        constraint,
        &remover.bitwise_and_operator,
    )?;
    let result = mutator.mutate(remover, value).and_then(T::try_from);
    finish_constraint_contexts(result, exits)
}

fn is_profiler_call(value: &PrimExpr, profiler_operators: &[Expr]) -> bool {
    let Some(call) = value.as_node::<CallObj>() else {
        return false;
    };
    profiler_operators
        .iter()
        .any(|operator| call.op.same_as(operator))
}

fn evaluate_zero() -> Result<Stmt> {
    Ok(Evaluate::from_i64(0)?.into())
}

fn zero_like(value: &PrimExpr) -> PrimExpr {
    IntImm::from_complete_fields(None, value.type_annotation(), 0.into()).into()
}

fn one_like(value: &PrimExpr) -> PrimExpr {
    IntImm::from_complete_fields(None, value.type_annotation(), 1.into()).into()
}
