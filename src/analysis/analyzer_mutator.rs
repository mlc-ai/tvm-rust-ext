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

use tvm_ffi::{
    structural_mutate, Any, AnyCompatible, AnyView, Array, Error, Function, Map, MutateDispatch,
    Mutator, ObjectCore, ObjectRefCast, ObjectRefCore, Result, String as FfiString,
};

use super::{side_effect, Analyzer, CallEffectKind};
use crate::ir::{Call, Expr, IntImm, OpaqueExprObj, PointerType, PrimExpr, Range, Var};
use crate::tirx::{
    Add, And, AssertStmt, AttrStmt, Bind, BufferType, BufferVar, FloorDiv, For, IfThenElse,
    IterVar, Let, Mul, Not, PrimFunc, PrimVar, Reduce, Select, SeqStmt, Stmt, EQ, GE, GT, LE, LT,
};

const THREAD_EXTENT: &str = "thread_extent";
const VIRTUAL_THREAD: &str = "virtual_thread";

/// Shared arithmetic state and constraint scopes for an [`AnalyzerMutator`].
pub struct AnalyzerMutatorState {
    analyzer: Analyzer,
    bitwise_and_operator: Expr,
    likely_operator: Expr,
    if_then_else_operator: Expr,
    buffer_data_operator: Expr,
    constraint_scopes: Vec<Vec<Function>>,
    iter_vars: Map<PrimVar, Range>,
    iter_predicates: Vec<PrimExpr>,
}

impl AnalyzerMutatorState {
    /// Create analyzer-aware mutation state around an existing analyzer.
    pub fn new(analyzer: Analyzer) -> Result<Self> {
        Ok(Self {
            analyzer,
            bitwise_and_operator: get_operator("tirx.bitwise_and")?,
            likely_operator: get_operator("tirx.likely")?,
            if_then_else_operator: get_operator("tirx.if_then_else")?,
            buffer_data_operator: get_operator("tirx.buffer_data")?,
            constraint_scopes: Vec::new(),
            iter_vars: Map::new(),
            iter_predicates: Vec::new(),
        })
    }

    /// Active loop and thread-binding domains visible to iterator-map analysis.
    pub fn iter_vars(&self) -> &Map<PrimVar, Range> {
        &self.iter_vars
    }

    /// Active branch predicates that depend on an iterator variable.
    pub fn iter_predicates(&self) -> &[PrimExpr] {
        &self.iter_predicates
    }

    pub(crate) fn buffer_data_operator(&self) -> &Expr {
        &self.buffer_data_operator
    }
}

impl Drop for AnalyzerMutatorState {
    fn drop(&mut self) {
        while let Some(exits) = self.constraint_scopes.pop() {
            for exit in exits.into_iter().rev() {
                let _ = exit.call_tuple(());
            }
        }
    }
}

impl std::ops::Deref for AnalyzerMutatorState {
    type Target = Analyzer;

    fn deref(&self) -> &Self::Target {
        &self.analyzer
    }
}

/// A typed structural mutator whose recursive rewrites use an arithmetic analyzer.
///
/// The pass keeps the analyzer and all other pass state on `Self`; [`Mutator`]
/// continues to contain only structural-recursion state.  These helpers mirror
/// the constraint-scoped child visits in C++ `IRMutatorWithAnalyzer` without
/// adding another FFI protocol.
pub trait AnalyzerMutator: MutateDispatch {
    /// Analyzer state owned by this pass.
    fn analyzer_state(&self) -> &AnalyzerMutatorState;

    /// Mutable analyzer state owned by this pass.
    fn analyzer_state_mut(&mut self) -> &mut AnalyzerMutatorState;

    /// Mutate one root while keeping all analyzer facts scoped to this call.
    fn mutate_root<R>(&mut self, root: R) -> Result<Any>
    where
        R: Into<Any>,
    {
        self.with_analyzer_scope(|this| structural_mutate(root, &mut *this))
    }

    /// Run `operation` in a nested analyzer constraint scope.
    fn with_analyzer_scope<T>(
        &mut self,
        operation: impl FnOnce(&mut Self) -> Result<T>,
    ) -> Result<T> {
        self.analyzer_state_mut().constraint_scopes.push(Vec::new());
        let result = operation(self);
        let exits = self
            .analyzer_state_mut()
            .constraint_scopes
            .pop()
            .expect("the nested analyzer scope is present");
        finish_constraint_contexts(result, exits)
    }

    /// Keep `constraint` active until the current analyzer scope exits.
    fn enter_analyzer_constraint(&mut self, constraint: &PrimExpr) -> Result<()> {
        if self.analyzer_state().constraint_scopes.is_empty() {
            return Err(Error::new(
                tvm_ffi::RUNTIME_ERROR,
                "analyzer-aware mutation must start with AnalyzerMutator::mutate_root",
                "",
            ));
        }
        let exit = self
            .analyzer_state()
            .analyzer
            .enter_constraint(constraint)?;
        self.analyzer_state_mut()
            .constraint_scopes
            .last_mut()
            .expect("an analyzer scope is active")
            .push(exit);
        Ok(())
    }

    /// Mark every buffer-parameter shape as non-negative for this mutation.
    fn mark_buffer_parameter_shapes(&mut self, function: &PrimFunc) -> Result<()> {
        for parameter in function.params.iter() {
            let Ok(buffer) = BufferVar::try_from(&parameter) else {
                continue;
            };
            for extent in buffer.type_annotation().shape.iter() {
                let non_negative: PrimExpr = GE::new(extent.clone(), zero_like(&extent))?.into();
                self.enter_analyzer_constraint(&non_negative)?;
            }
        }
        Ok(())
    }

    /// Run `operation` with one loop or thread-binding variable visible to iterator analysis.
    fn with_analyzer_iter_var<T>(
        &mut self,
        variable: PrimVar,
        domain: Range,
        operation: impl FnOnce(&mut Self) -> Result<T>,
    ) -> Result<T> {
        self.analyzer_state()
            .analyzer
            .bind(variable.as_var(), &domain)?;
        let previous = self.analyzer_state().iter_vars.clone();
        let mut entries = previous.iter().collect::<Vec<_>>();
        entries.retain(|(existing, _)| !existing.same_as(&variable));
        entries.push((variable, domain));
        self.analyzer_state_mut().iter_vars = Map::from_iter(entries);
        let result = operation(self);
        self.analyzer_state_mut().iter_vars = previous;
        result
    }

    /// Run `operation` while recording an iterator-dependent branch predicate.
    fn with_analyzer_iter_predicate<T>(
        &mut self,
        condition: PrimExpr,
        operation: impl FnOnce(&mut Self) -> Result<T>,
    ) -> Result<T> {
        if !self.condition_uses_analyzer_iter_var(&condition)? {
            return operation(self);
        }
        self.analyzer_state_mut().iter_predicates.push(condition);
        let result = operation(self);
        self.analyzer_state_mut().iter_predicates.pop();
        result
    }

    /// Simplify iterator indices using the active loop domains and predicates.
    fn iter_map_simplify_with_context(
        &self,
        indices: &Array<PrimExpr>,
        non_trivial_only: bool,
    ) -> Result<Array<PrimExpr>> {
        let mut predicate: PrimExpr = IntImm::new("bool", 1)?.into();
        for condition in self.analyzer_state().iter_predicates() {
            predicate = And::new(predicate, condition.clone())?.into();
        }
        let simplified: Array<PrimExpr> = tvm_ffi::cached_global_func!("arith.IterMapSimplify")
            .call_tuple((
                indices.clone(),
                self.analyzer_state().iter_vars.clone(),
                predicate,
                1_i32,
                false,
                Some(self.analyzer_state().analyzer.clone()),
            ))?
            .try_into()?;
        if !non_trivial_only {
            return Ok(simplified);
        }

        Ok(Array::new(
            simplified
                .iter()
                .zip(indices.iter())
                .map(|(simplified, original)| {
                    if int_value(&simplified).is_some() && is_primitive_variable(&original) {
                        original
                    } else {
                        simplified
                    }
                })
                .collect(),
        ))
    }

    /// Whether `condition` references any active iterator variable.
    fn condition_uses_analyzer_iter_var(&self, condition: &PrimExpr) -> Result<bool> {
        if self.analyzer_state().iter_vars.is_empty() {
            return Ok(false);
        }
        let undefined: Array<Var> = tvm_ffi::cached_global_func!("tirx.analysis.UndefinedVars")
            .call_tuple((condition, Array::<Var>::new(Vec::new())))?
            .try_into()?;
        Ok(undefined.iter().any(|variable| {
            self.analyzer_state()
                .iter_vars
                .iter()
                .any(|(iterator, _)| iterator.as_var().same_as(&variable))
        }))
    }

    /// Recursively mutate `value` while `constraint` is known to be true.
    fn mutate_with_constraint<Value, Output>(
        &mut self,
        mutator: &mut Mutator,
        value: &Value,
        constraint: &PrimExpr,
    ) -> Result<Output>
    where
        for<'a> AnyView<'a>: From<&'a Value>,
        Output: TryFrom<Any, Error = Error>,
    {
        self.with_analyzer_scope(|this| {
            this.enter_analyzer_constraint(constraint)?;
            mutator.mutate(this, value)?.try_into()
        })
    }

    /// Recursively mutate `value` with `constraint` and its implied floor-div facts.
    fn mutate_with_constraint_facts<Value, Output>(
        &mut self,
        mutator: &mut Mutator,
        value: &Value,
        constraint: &PrimExpr,
    ) -> Result<Output>
    where
        for<'a> AnyView<'a>: From<&'a Value>,
        Output: TryFrom<Any, Error = Error>,
    {
        let mut constraints = vec![constraint.clone()];
        collect_derived_constraint_facts(
            constraint,
            &self.analyzer_state().bitwise_and_operator,
            &mut constraints,
        )?;
        self.with_analyzer_scope(|this| {
            for constraint in &constraints {
                this.enter_analyzer_constraint(constraint)?;
            }
            mutator.mutate(this, value)?.try_into()
        })
    }

    /// Mutate with implied facts while recording an iterator-dependent predicate.
    fn mutate_with_constraint_facts_and_predicate<Value, Output>(
        &mut self,
        mutator: &mut Mutator,
        value: &Value,
        constraint: &PrimExpr,
    ) -> Result<Output>
    where
        for<'a> AnyView<'a>: From<&'a Value>,
        Output: TryFrom<Any, Error = Error>,
    {
        let mut constraints = vec![constraint.clone()];
        collect_derived_constraint_facts(
            constraint,
            &self.analyzer_state().bitwise_and_operator,
            &mut constraints,
        )?;
        self.with_analyzer_scope(|this| {
            for constraint in &constraints {
                this.enter_analyzer_constraint(constraint)?;
            }
            this.with_analyzer_iter_predicate(constraint.clone(), |this| {
                mutator.mutate(this, value)?.try_into()
            })
        })
    }

    /// Mutate with one fact while recording an iterator-dependent predicate.
    fn mutate_with_constraint_and_predicate<Value, Output>(
        &mut self,
        mutator: &mut Mutator,
        value: &Value,
        constraint: &PrimExpr,
    ) -> Result<Output>
    where
        for<'a> AnyView<'a>: From<&'a Value>,
        Output: TryFrom<Any, Error = Error>,
    {
        self.with_analyzer_scope(|this| {
            this.enter_analyzer_constraint(constraint)?;
            this.with_analyzer_iter_predicate(constraint.clone(), |this| {
                mutator.mutate(this, value)?.try_into()
            })
        })
    }

    /// Apply the analyzer-aware default recursion for `Bind`.
    fn default_mutate_bind(&mut self, mutator: &mut Mutator, value: Bind) -> Result<Bind> {
        let bound_value: Expr = mutator.mutate(self, &value.value)?.try_into()?;
        if let Ok(bound_value) = PrimExpr::try_from(&bound_value) {
            if side_effect(&bound_value)? <= CallEffectKind::kPure {
                self.analyzer_state()
                    .bind_expression(&value.var, &bound_value)?;
            }
        }
        if bound_value.same_as(&value.value) {
            return Ok(value);
        }
        Ok(value.copy_with(value.var.clone(), bound_value))
    }

    /// Apply the analyzer-aware default recursion for a primitive `Let`.
    fn default_mutate_let(&mut self, mutator: &mut Mutator, value: Let) -> Result<Let> {
        let bound_value: PrimExpr = mutator.mutate(self, &value.value)?.try_into()?;
        if side_effect(&bound_value)? <= CallEffectKind::kPure {
            self.analyzer_state()
                .bind_expression(&value.var, &bound_value)?;
        }
        let body: PrimExpr = mutator.mutate(self, &value.body)?.try_into()?;
        if bound_value.same_as(&value.value) && body.same_as(&value.body) {
            return Ok(value);
        }
        Ok(value.copy_with(value.var.clone(), bound_value, body))
    }

    /// Bind reduction axes before recursively mutating reduction children.
    fn default_mutate_reduce(&mut self, mutator: &mut Mutator, value: Reduce) -> Result<Reduce> {
        for axis in value.axis.iter() {
            if let Some(domain) = axis.dom()? {
                self.analyzer_state().bind(axis.var()?.as_var(), &domain)?;
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

    /// Mutate an assertion condition. Its enclosing sequence owns fact lifetime.
    fn default_mutate_assertion(
        &mut self,
        mutator: &mut Mutator,
        value: AssertStmt,
    ) -> Result<AssertStmt> {
        let condition: PrimExpr = mutator.mutate(self, &value.condition)?.try_into()?;
        self.enter_analyzer_constraint(&condition)?;
        if condition.same_as(&value.condition) {
            return Ok(value);
        }
        Ok(value.copy_with(
            condition,
            value.error_kind.clone(),
            value.message_parts.clone(),
        ))
    }

    /// Mutate sequence elements in order so assertions constrain later siblings.
    fn default_mutate_sequence(&mut self, mutator: &mut Mutator, value: SeqStmt) -> Result<Stmt> {
        let mut statements = Vec::with_capacity(value.seq.len());
        for statement in value.seq.iter() {
            statements.push(mutator.mutate(self, &statement)?.try_into()?);
        }
        let flattened = Stmt::sequence(statements)?;
        if let Ok(sequence) = flattened.clone().try_cast::<SeqStmt>() {
            if array_same_as(&sequence.seq, &value.seq) {
                return Ok(value.into());
            }
        }
        Ok(flattened)
    }

    /// Apply branch constraints while mutating a `Select`.
    fn default_mutate_select(&mut self, mutator: &mut Mutator, value: Select) -> Result<PrimExpr> {
        let condition: PrimExpr = mutator.mutate(self, &value.condition)?.try_into()?;
        let true_value =
            self.mutate_with_constraint_facts(mutator, &value.true_value, &condition)?;
        let negative = self
            .analyzer_state()
            .simplify(&PrimExpr::from(Not::new(condition.clone())?))?;
        let false_value = self.mutate_with_constraint(mutator, &value.false_value, &negative)?;
        match int_value(&condition) {
            Some(1) => return Ok(true_value),
            Some(0) => return Ok(false_value),
            _ => {}
        }
        if condition.same_as(&value.condition)
            && true_value.same_as(&value.true_value)
            && false_value.same_as(&value.false_value)
        {
            return Ok(value.into());
        }
        Ok(value.copy_with(condition, true_value, false_value).into())
    }

    /// Apply lazy branch constraints to `tirx.if_then_else` calls.
    fn default_mutate_call(&mut self, mutator: &mut Mutator, value: Call) -> Result<Expr> {
        if !value
            .op
            .same_as(&self.analyzer_state().if_then_else_operator)
            || value.args.len() != 3
        {
            let call_operator =
                if AnyView::from(&value.op).type_index() == OpaqueExprObj::type_index() {
                    mutator.mutate(self, &value.op)?.try_into()?
                } else {
                    value.op.clone()
                };
            let args: Array<Expr> = mutator.mutate(self, &value.args)?.try_into()?;
            if call_operator.same_as(&value.op) && array_same_as(&args, &value.args) {
                return Ok(value.into());
            }

            let result_type = if value
                .op
                .same_as(&self.analyzer_state().buffer_data_operator)
            {
                if args.len() != 1 {
                    return Err(Error::new(
                        tvm_ffi::RUNTIME_ERROR,
                        "buffer_data requires exactly one buffer variable",
                        "",
                    ));
                }
                let variable = args
                    .get(0)
                    .expect("buffer_data has one argument")
                    .try_cast::<Var>()?;
                let buffer_type = variable.ty.clone().try_cast::<BufferType>()?;
                PointerType::new(
                    buffer_type.dtype.clone(),
                    buffer_type.storage_scope.as_str(),
                )?
                .into()
            } else {
                value.ty.clone()
            };
            return Ok(value.copy_with(result_type, call_operator, args).into());
        }

        let original_condition = value.args.get(0).expect("condition argument is present");
        let original_true = value.args.get(1).expect("true argument is present");
        let original_false = value.args.get(2).expect("false argument is present");
        let condition: PrimExpr = mutator.mutate(self, &original_condition)?.try_into()?;
        let true_value =
            self.mutate_with_constraint_facts_and_predicate(mutator, &original_true, &condition)?;
        let negative: PrimExpr = Not::new(condition.clone())?.into();
        let false_value =
            self.mutate_with_constraint_and_predicate(mutator, &original_false, &negative)?;
        match int_value(&condition) {
            Some(1) => return Ok(true_value),
            Some(0) => return Ok(false_value),
            _ => {}
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
                Array::new(vec![condition.into(), true_value, false_value]),
            )
            .into())
    }

    /// Bind thread-domain metadata and mutate the recursive `AttrStmt` fields.
    fn default_mutate_attribute(
        &mut self,
        mutator: &mut Mutator,
        value: AttrStmt,
    ) -> Result<AttrStmt> {
        self.with_analyzer_scope(|this| {
            if matches!(value.attr_key.as_str(), THREAD_EXTENT | VIRTUAL_THREAD) {
                let iteration = IterVar::try_from(value.node.clone())?;
                if iteration.thread_tag()?.is_empty() {
                    return Err(Error::new(
                        tvm_ffi::RUNTIME_ERROR,
                        "thread extent requires an IterVar with a non-empty thread tag",
                        "",
                    ));
                }
                let variable = iteration.var()?;
                let domain = Range::from_min_extent(zero_like(&value.value), value.value.clone())?;
                return this.with_analyzer_iter_var(variable, domain, |this| {
                    mutate_attribute_fields(this, mutator, value)
                });
            }
            mutate_attribute_fields(this, mutator, value)
        })
    }

    /// Apply scoped branch facts and condition folding to an `IfThenElse`.
    fn default_mutate_conditional(
        &mut self,
        mutator: &mut Mutator,
        value: IfThenElse,
    ) -> Result<Stmt> {
        self.with_analyzer_scope(|this| {
            let condition: PrimExpr = mutator.mutate(this, &value.condition)?.try_into()?;
            let real_condition = unwrap_likely(&condition, &this.analyzer_state().likely_operator)?;
            let then_case: Stmt = this.mutate_with_constraint_facts_and_predicate(
                mutator,
                &value.then_case,
                &real_condition,
            )?;
            let negative = this
                .analyzer_state()
                .simplify(&PrimExpr::from(Not::new(real_condition.clone())?))?;
            let else_case = value
                .else_case
                .as_ref()
                .map(|branch| this.mutate_with_constraint(mutator, branch, &negative))
                .transpose()?;

            match int_value(&real_condition) {
                Some(1) => return Ok(then_case),
                Some(0) => return else_case.map_or_else(evaluate_zero, Ok),
                _ => {}
            }
            if condition.same_as(&value.condition)
                && then_case.same_as(&value.then_case)
                && option_same_as(&else_case, &value.else_case)
            {
                return Ok(value.into());
            }
            Ok(IfThenElse::from_complete_fields(
                value.span.clone(),
                condition,
                then_case,
                else_case,
            )
            .into())
        })
    }

    /// Bind a loop domain and mutate its body under `extent > 0`.
    fn default_mutate_loop(&mut self, mutator: &mut Mutator, value: For) -> Result<For> {
        self.with_analyzer_scope(|this| {
            let domain = Range::from_min_extent(value.min.clone(), value.extent.clone())?;
            this.with_analyzer_iter_var(value.loop_var.clone(), domain, |this| {
                let minimum: PrimExpr = mutator.mutate(this, &value.min)?.try_into()?;
                let extent: PrimExpr = mutator.mutate(this, &value.extent)?.try_into()?;
                let step: Option<PrimExpr> = mutator.mutate(this, &value.step)?.try_into()?;
                let positive: PrimExpr = GT::new(extent.clone(), zero_like(&extent))?.into();
                let body: Stmt =
                    this.mutate_with_constraint_facts(mutator, &value.body, &positive)?;
                if minimum.same_as(&value.min)
                    && extent.same_as(&value.extent)
                    && option_same_as(&step, &value.step)
                    && body.same_as(&value.body)
                {
                    return Ok(value);
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
            })
        })
    }
}

fn mutate_attribute_fields<D: AnalyzerMutator>(
    driver: &mut D,
    mutator: &mut Mutator,
    value: AttrStmt,
) -> Result<AttrStmt> {
    let attr_value: PrimExpr = mutator.mutate(driver, &value.value)?.try_into()?;
    let body: Stmt = mutator.mutate(driver, &value.body)?.try_into()?;
    if attr_value.same_as(&value.value) && body.same_as(&value.body) {
        return Ok(value);
    }
    Ok(value.copy_with(value.node.clone(), value.attr_key.clone(), attr_value, body))
}

fn array_same_as<T: AnyCompatible + ObjectRefCore>(lhs: &Array<T>, rhs: &Array<T>) -> bool {
    lhs.len() == rhs.len()
        && lhs
            .iter()
            .zip(rhs.iter())
            .all(|(lhs, rhs)| lhs.same_as(&rhs))
}

fn option_same_as<T: ObjectRefCore>(lhs: &Option<T>, rhs: &Option<T>) -> bool {
    match (lhs, rhs) {
        (Some(lhs), Some(rhs)) => lhs.same_as(rhs),
        (None, None) => true,
        _ => false,
    }
}

fn unwrap_likely(condition: &PrimExpr, likely_operator: &Expr) -> Result<PrimExpr> {
    let Ok(call) = condition.as_expr().clone().try_cast::<Call>() else {
        return Ok(condition.clone());
    };
    if !call.op.same_as(likely_operator) || call.args.len() != 1 {
        return Ok(condition.clone());
    }
    PrimExpr::try_from(call.args.get(0).expect("one likely argument is present"))
}

fn zero_like(value: &PrimExpr) -> PrimExpr {
    IntImm::from_complete_fields(None, value.type_annotation(), 0).into()
}

fn evaluate_zero() -> Result<Stmt> {
    Ok(crate::tirx::Evaluate::from_i64(0)?.into())
}

#[derive(Clone, Copy)]
enum CompareKind {
    Equal,
    LessThan,
    LessEqual,
    GreaterThan,
    GreaterEqual,
}

fn collect_derived_constraint_facts(
    condition: &PrimExpr,
    bitwise_and_operator: &Expr,
    output: &mut Vec<PrimExpr>,
) -> Result<()> {
    if let Ok(and) = condition.as_expr().clone().try_cast::<And>() {
        collect_derived_constraint_facts(&and.a, bitwise_and_operator, output)?;
        collect_derived_constraint_facts(&and.b, bitwise_and_operator, output)?;
        return Ok(());
    }
    if let Ok(call) = condition.as_expr().clone().try_cast::<Call>() {
        if call.op.same_as(bitwise_and_operator) && call.args.len() == 2 {
            let lhs = PrimExpr::try_from(call.args.get(0).expect("two arguments are present"))?;
            let rhs = PrimExpr::try_from(call.args.get(1).expect("two arguments are present"))?;
            if is_bool8(&lhs) && is_bool8(&rhs) {
                collect_derived_constraint_facts(&lhs, bitwise_and_operator, output)?;
                collect_derived_constraint_facts(&rhs, bitwise_and_operator, output)?;
                return Ok(());
            }
        }
    }

    if let Ok(compare) = condition.as_expr().clone().try_cast::<EQ>() {
        collect_floor_div_constraints(&compare.a, &compare.b, CompareKind::Equal, output)?;
    } else if let Ok(compare) = condition.as_expr().clone().try_cast::<LT>() {
        collect_floor_div_constraints(&compare.a, &compare.b, CompareKind::LessThan, output)?;
    } else if let Ok(compare) = condition.as_expr().clone().try_cast::<LE>() {
        collect_floor_div_constraints(&compare.a, &compare.b, CompareKind::LessEqual, output)?;
    } else if let Ok(compare) = condition.as_expr().clone().try_cast::<GT>() {
        collect_floor_div_constraints(&compare.a, &compare.b, CompareKind::GreaterThan, output)?;
    } else if let Ok(compare) = condition.as_expr().clone().try_cast::<GE>() {
        collect_floor_div_constraints(&compare.a, &compare.b, CompareKind::GreaterEqual, output)?;
    }
    Ok(())
}

fn collect_floor_div_constraints(
    lhs: &PrimExpr,
    rhs: &PrimExpr,
    kind: CompareKind,
    output: &mut Vec<PrimExpr>,
) -> Result<()> {
    if let (Ok(division), Some(value)) =
        (lhs.as_expr().clone().try_cast::<FloorDiv>(), int_value(rhs))
    {
        append_floor_div_constraints(&division, value, kind, output)?;
    }
    if let (Ok(division), Some(value)) =
        (rhs.as_expr().clone().try_cast::<FloorDiv>(), int_value(lhs))
    {
        append_floor_div_constraints(&division, value, invert_compare(kind), output)?;
    }
    Ok(())
}

fn append_floor_div_constraints(
    division: &FloorDiv,
    value: i64,
    kind: CompareKind,
    output: &mut Vec<PrimExpr>,
) -> Result<()> {
    let Some(divisor_value) = int_value(&division.b) else {
        return Ok(());
    };
    if divisor_value <= 0 {
        return Ok(());
    }

    let dtype = division.a.type_annotation().dtype;
    let divisor: PrimExpr = IntImm::from_dtype(dtype, divisor_value)?.into();
    let k: PrimExpr = IntImm::from_dtype(dtype, value)?.into();
    let one: PrimExpr = IntImm::from_dtype(dtype, 1)?.into();
    let lower: PrimExpr = Mul::new(k.clone(), divisor.clone())?.into();
    let upper: PrimExpr = Mul::new(Add::new(k, one)?, divisor)?.into();

    match kind {
        CompareKind::Equal => {
            output.push(GE::new(division.a.clone(), lower)?.into());
            output.push(LT::new(division.a.clone(), upper)?.into());
        }
        CompareKind::LessThan => output.push(LT::new(division.a.clone(), lower)?.into()),
        CompareKind::LessEqual => output.push(LT::new(division.a.clone(), upper)?.into()),
        CompareKind::GreaterThan => output.push(GE::new(division.a.clone(), upper)?.into()),
        CompareKind::GreaterEqual => output.push(GE::new(division.a.clone(), lower)?.into()),
    }
    Ok(())
}

fn invert_compare(kind: CompareKind) -> CompareKind {
    match kind {
        CompareKind::Equal => CompareKind::Equal,
        CompareKind::LessThan => CompareKind::GreaterThan,
        CompareKind::LessEqual => CompareKind::GreaterEqual,
        CompareKind::GreaterThan => CompareKind::LessThan,
        CompareKind::GreaterEqual => CompareKind::LessEqual,
    }
}

fn int_value(value: &PrimExpr) -> Option<i64> {
    value
        .as_expr()
        .clone()
        .try_cast::<IntImm>()
        .ok()
        .map(|literal| literal.value)
}

fn is_primitive_variable(value: &PrimExpr) -> bool {
    value
        .as_expr()
        .clone()
        .try_cast::<Var>()
        .is_ok_and(|variable| PrimVar::try_from(&variable).is_ok())
}

fn is_bool8(value: &PrimExpr) -> bool {
    let dtype = value.type_annotation().dtype;
    dtype.code == tvm_ffi::DLDataTypeCode::kDLBool as u8 && dtype.bits == 8
}

fn get_operator(name: &str) -> Result<Expr> {
    tvm_ffi::cached_global_func!("ir.GetOp")
        .call_tuple((FfiString::from(name),))?
        .try_into()
}

pub(crate) fn finish_constraint_contexts<T>(result: Result<T>, exits: Vec<Function>) -> Result<T> {
    let mut exit_error = None;
    for exit in exits.into_iter().rev() {
        if let Err(error) = exit.call_tuple(()) {
            exit_error.get_or_insert(error);
        }
    }
    match (result, exit_error) {
        (Err(error), _) | (Ok(_), Some(error)) => Err(error),
        (Ok(value), None) => Ok(value),
    }
}
