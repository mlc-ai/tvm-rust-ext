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
    structural_mutate, Any, Array, FieldGetter, Function, Map, MapValue, Mutator, ObjectArc,
    ObjectCore, ObjectRefCast, ObjectRefCore, Result, String as FfiString,
};

use super::utils::{
    array_same_as, finish_constraint_contexts, get_operator, int_value, mutate_stmt_expr_default,
    with_prim_func_body,
};
use super::{create_prim_func_pass_with_context, Pass, PassContext};
use crate::analysis::{side_effect, Analyzer, CallEffectKind};
use crate::ir::prim::{Add, Not, GE, LT};
use crate::ir::{Call, Expr, PrimExpr, Range, TensorLoad, Var};
use crate::tirx::{
    AssertStmt, AttrStmt, Bind, BufferStore, BufferVar, Evaluate, For, IfThenElse, IterVar,
    PrimFunc, Stmt,
};

const THREAD_EXTENT: &str = "thread_extent";
const VIRTUAL_THREAD: &str = "virtual_thread";

const TRANSITIVELY_PROVE_INEQUALITIES: i64 = 1 << 0;
const CONVERT_BOOLEAN_TO_AND_OF_ORS: i64 = 1 << 1;
const APPLY_CONSTRAINTS_TO_BOOLEAN_BRANCHES: i64 = 1 << 2;

#[repr(C)]
#[derive(Object)]
#[type_key = "tirx.transform.StmtSimplifyConfig"]
#[type_final]
struct StmtSimplifyConfigObj {
    base: tvm_ffi::Object,
}

#[repr(C)]
#[derive(ObjectRef, Clone)]
struct StmtSimplifyConfig {
    data: ObjectArc<StmtSimplifyConfigObj>,
}

impl std::ops::Deref for StmtSimplifyConfig {
    type Target = StmtSimplifyConfigObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl StmtSimplifyConfig {
    fn field(&self, name: &str) -> Result<bool> {
        FieldGetter::new(StmtSimplifyConfigObj::type_index(), name)?.get(&**self)
    }
}

#[derive(Clone, Copy, Default)]
struct StmtSimplifyOptions {
    transitively_prove_inequalities: bool,
    convert_boolean_to_and_of_ors: bool,
    apply_constraints_to_boolean_branches: bool,
}

impl StmtSimplifyOptions {
    fn from_context(context: &PassContext) -> Result<Self> {
        let Some(raw) = context
            .config()?
            .get(&FfiString::from("tirx.StmtSimplify"))?
        else {
            return Ok(Self::default());
        };
        let config = StmtSimplifyConfig::try_from(raw)?;
        Ok(Self {
            transitively_prove_inequalities: config.field("transitively_prove_inequalities")?,
            convert_boolean_to_and_of_ors: config.field("convert_boolean_to_and_of_ors")?,
            apply_constraints_to_boolean_branches: config
                .field("apply_constraints_to_boolean_branches")?,
        })
    }

    fn extensions(self) -> i64 {
        (if self.transitively_prove_inequalities {
            TRANSITIVELY_PROVE_INEQUALITIES
        } else {
            0
        }) | (if self.convert_boolean_to_and_of_ors {
            CONVERT_BOOLEAN_TO_AND_OF_ORS
        } else {
            0
        }) | (if self.apply_constraints_to_boolean_branches {
            APPLY_CONSTRAINTS_TO_BOOLEAN_BRANCHES
        } else {
            0
        })
    }
}

/// Simplify one PrimFunc using the default `tirx.StmtSimplify` configuration.
pub fn stmt_simplify_prim_func(function: PrimFunc) -> Result<PrimFunc> {
    stmt_simplify_with_options(function, StmtSimplifyOptions::default())
}

fn stmt_simplify_with_options(
    function: PrimFunc,
    options: StmtSimplifyOptions,
) -> Result<PrimFunc> {
    let analyzer = Analyzer::new()?;
    analyzer.set_enabled_extensions(options.extensions())?;

    let if_then_else_operator = get_operator("ir.prim.if_then_else")?;
    let mut simplifier = StmtSimplifier {
        analyzer,
        if_then_else_operator,
        non_inlined_bindings: Map::new(),
        constraint_scopes: vec![Vec::new()],
    };

    // C++ marks buffer-parameter shapes as globally non-negative.  The public
    // Analyzer ABI exposes the equivalent scoped constraint operation, so keep
    // those facts active for the complete mutation.
    for parameter in function.params.iter() {
        if let Ok(buffer) = BufferVar::try_from(&parameter) {
            for shape in buffer.type_annotation().shape.iter() {
                let zero = crate::ir::IntImm::from_dtype(shape.dtype(), 0)?;
                let non_negative: PrimExpr = GE::new(shape, zero)?.into();
                simplifier.enter_persistent_constraint(&non_negative)?;
            }
        }
    }

    let body = structural_mutate(function.body.clone(), &mut simplifier).and_then(Stmt::try_from);
    let body = simplifier.finish_root_scope(body)?;
    Ok(with_prim_func_body(function, body))
}

/// Build TVM's `tirx.StmtSimplify` PrimFunc pass in Rust.
pub fn stmt_simplify() -> Result<Pass> {
    create_prim_func_pass_with_context(
        "tirx.StmtSimplify",
        0,
        Vec::new(),
        false,
        |function, context| {
            stmt_simplify_with_options(function, StmtSimplifyOptions::from_context(&context)?)
        },
    )
}

struct StmtSimplifier {
    analyzer: Analyzer,
    if_then_else_operator: Expr,
    non_inlined_bindings: Map<Var, Expr>,
    constraint_scopes: Vec<Vec<Function>>,
}

impl StmtSimplifier {
    fn prove_condition(&self, condition: &PrimExpr) -> Result<Option<bool>> {
        let substituted: PrimExpr = tvm_ffi::cached_global_func!("tirx.Substitute")
            .call_tuple((condition, &self.non_inlined_bindings))?
            .try_into()?;
        let simplified = self.analyzer.simplify(&substituted)?;
        Ok(int_value(&simplified).map(|value| value != 0))
    }

    fn record_binding(&mut self, variable: Var, value: PrimExpr) {
        let identity = tvm_ffi::ObjectIdentity::of(&variable);
        let mut bindings = self
            .non_inlined_bindings
            .iter()
            .filter(|(existing, _)| tvm_ffi::ObjectIdentity::of(existing) != identity)
            .collect::<Vec<_>>();
        bindings.push((variable, value.into()));
        self.non_inlined_bindings = Map::from_iter(bindings);
    }

    fn enter_persistent_constraint(&mut self, constraint: &PrimExpr) -> Result<()> {
        let exit = self.analyzer.enter_constraint(constraint)?;
        self.constraint_scopes
            .last_mut()
            .expect("a constraint scope is always active")
            .push(exit);
        Ok(())
    }

    fn with_new_scope<T>(&mut self, operation: impl FnOnce(&mut Self) -> Result<T>) -> Result<T> {
        self.constraint_scopes.push(Vec::new());
        let result = operation(self);
        let exits = self
            .constraint_scopes
            .pop()
            .expect("the newly-created scope is present");
        finish_constraint_contexts(result, exits)
    }

    fn with_constraint<T>(
        &mut self,
        constraint: &PrimExpr,
        operation: impl FnOnce(&mut Self) -> Result<T>,
    ) -> Result<T> {
        self.with_new_scope(|simplifier| {
            simplifier.enter_persistent_constraint(constraint)?;
            operation(simplifier)
        })
    }

    fn finish_root_scope<T>(&mut self, result: Result<T>) -> Result<T> {
        let exits = self
            .constraint_scopes
            .pop()
            .expect("the root constraint scope is present");
        finish_constraint_contexts(result, exits)
    }

    fn expressions_equal(&self, lhs: &PrimExpr, rhs: &PrimExpr) -> Result<bool> {
        tvm_ffi::cached_global_func!("ffi.StructuralEqual")
            .call_tuple((lhs, rhs, false, false))?
            .try_into()
    }

    fn arrays_equal(&self, lhs: &Array<PrimExpr>, rhs: &Array<PrimExpr>) -> Result<bool> {
        if lhs.len() != rhs.len() {
            return Ok(false);
        }
        for (lhs, rhs) in lhs.iter().zip(rhs.iter()) {
            if !self.expressions_equal(&lhs, &rhs)? {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

#[tvm_ffi::dispatch(mutate)]
impl StmtSimplifier {
    fn mutate_loop(&mut self, value: For, mutator: &mut Mutator) -> Result<For> {
        self.with_new_scope(|simplifier| {
            let domain = Range::from_min_extent(value.min.clone(), value.extent.clone())?;
            simplifier.analyzer.bind(value.loop_var.as_var(), &domain)?;

            let lower: PrimExpr = GE::new(value.loop_var.clone(), value.min.clone())?.into();
            let upper_bound: PrimExpr = Add::new(value.min.clone(), value.extent.clone())?.into();
            let upper: PrimExpr = LT::new(value.loop_var.clone(), upper_bound)?.into();
            simplifier.enter_persistent_constraint(&lower)?;
            simplifier.enter_persistent_constraint(&upper)?;

            let minimum: PrimExpr = mutator.mutate(simplifier, &value.min)?.try_into()?;
            let extent: PrimExpr = mutator.mutate(simplifier, &value.extent)?.try_into()?;
            let step: Option<PrimExpr> = mutator.mutate(simplifier, &value.step)?.try_into()?;
            let zero = crate::ir::IntImm::from_dtype(extent.dtype(), 0)?;
            let positive: PrimExpr = crate::ir::prim::GT::new(extent.clone(), zero)?.into();
            let body: Stmt = simplifier.with_constraint(&positive, |simplifier| {
                mutator.mutate(simplifier, &value.body)?.try_into()
            })?;

            if minimum.same_as(&value.min)
                && extent.same_as(&value.extent)
                && super::utils::option_same_as(&step, &value.step)
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
    }

    fn mutate_bind(&mut self, value: Bind, mutator: &mut Mutator) -> Result<Bind> {
        let bound_value: Expr = mutator.mutate(self, &value.value)?.try_into()?;
        if let Ok(primitive) = PrimExpr::try_from(&bound_value) {
            if side_effect(&primitive)? <= CallEffectKind::kPure {
                self.analyzer.bind_expression(&value.var, &primitive)?;
                self.record_binding(value.var.clone(), primitive);
            }
        }
        if bound_value.same_as(&value.value) {
            return Ok(value);
        }
        Ok(value.copy_with(value.var.clone(), bound_value))
    }

    fn mutate_conditional(&mut self, value: IfThenElse, mutator: &mut Mutator) -> Result<Stmt> {
        if let Some(condition) = self.prove_condition(&value.condition)? {
            return if condition {
                mutator.mutate(self, &value.then_case)?.try_into()
            } else if let Some(else_case) = &value.else_case {
                mutator.mutate(self, else_case)?.try_into()
            } else {
                Ok(Evaluate::from_i64(0)?.into())
            };
        }

        self.with_new_scope(|simplifier| {
            let condition: PrimExpr = mutator.mutate(simplifier, &value.condition)?.try_into()?;
            let then_case: Stmt = simplifier.with_constraint(&condition, |simplifier| {
                mutator.mutate(simplifier, &value.then_case)?.try_into()
            })?;
            let negative = simplifier
                .analyzer
                .simplify(&PrimExpr::from(Not::new(condition.clone())?))?;
            let else_case = value
                .else_case
                .as_ref()
                .map(|branch| {
                    simplifier.with_constraint(&negative, |simplifier| {
                        mutator.mutate(simplifier, branch)?.try_into()
                    })
                })
                .transpose()?;

            if int_value(&condition) == Some(1) {
                return Ok(then_case);
            }
            if int_value(&condition) == Some(0) {
                return Ok(else_case.unwrap_or(Evaluate::from_i64(0)?.into()));
            }
            if condition.same_as(&value.condition)
                && then_case.same_as(&value.then_case)
                && super::utils::option_same_as(&else_case, &value.else_case)
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

    fn mutate_attribute(&mut self, value: AttrStmt, mutator: &mut Mutator) -> Result<AttrStmt> {
        self.with_new_scope(|simplifier| {
            if value.attr_key.as_str() == THREAD_EXTENT || value.attr_key.as_str() == VIRTUAL_THREAD
            {
                let iteration = IterVar::try_from(value.node.clone())?;
                let variable = iteration.var()?;
                let zero = crate::ir::IntImm::from_dtype(value.value.dtype(), 0)?;
                let domain = Range::from_min_extent(zero, value.value.clone())?;
                simplifier.analyzer.bind(variable.as_var(), &domain)?;
            }

            let attr_value: PrimExpr = mutator.mutate(simplifier, &value.value)?.try_into()?;
            let body: Stmt = mutator.mutate(simplifier, &value.body)?.try_into()?;
            if attr_value.same_as(&value.value) && body.same_as(&value.body) {
                return Ok(value);
            }
            Ok(value.copy_with(value.node.clone(), value.attr_key.clone(), attr_value, body))
        })
    }

    fn mutate_assertion(&mut self, value: AssertStmt, mutator: &mut Mutator) -> Result<AssertStmt> {
        let condition: PrimExpr = mutator.mutate(self, &value.condition)?.try_into()?;
        self.enter_persistent_constraint(&condition)?;
        if condition.same_as(&value.condition) {
            return Ok(value);
        }
        Ok(value.copy_with(
            condition,
            value.error_kind.clone(),
            value.message_parts.clone(),
        ))
    }

    fn mutate_store(&mut self, value: BufferStore, mutator: &mut Mutator) -> Result<Stmt> {
        let stored_value: PrimExpr = mutator.mutate(self, &value.value)?.try_into()?;
        let indices: Array<PrimExpr> = mutator.mutate(self, &value.indices)?.try_into()?;
        let store = if stored_value.same_as(&value.value) && array_same_as(&indices, &value.indices)
        {
            value
        } else {
            value.copy_with(value.buffer.clone(), stored_value, indices)
        };

        if let Ok(load) = store.value.clone().try_cast::<TensorLoad>() {
            let load_buffer = BufferVar::try_from(&load.source)?;
            let load_type = load_buffer.type_annotation();
            let store_type = store.buffer.type_annotation();
            if load_buffer.same_as(&store.buffer)
                && self.arrays_equal(&load.indices, &store.indices)?
                && self.expressions_equal(&load_type.elem_offset, &store_type.elem_offset)?
                && self.arrays_equal(&load_type.shape, &store_type.shape)?
                && self.arrays_equal(&load_type.strides, &store_type.strides)?
            {
                return Ok(Evaluate::from_i64(0)?.into());
            }
        }
        Ok(store.into())
    }

    fn mutate_call(&mut self, value: Call, mutator: &mut Mutator) -> Result<Expr> {
        if value.op.same_as(&self.if_then_else_operator) && value.args.len() == 3 {
            let original_condition =
                PrimExpr::try_from(value.args.get(0).expect("three call arguments are present"))?;
            if let Some(condition) = self.prove_condition(&original_condition)? {
                let selected = if condition { 1 } else { 2 };
                let branch = value
                    .args
                    .get(selected)
                    .expect("three call arguments are present");
                return mutator.mutate(self, &branch)?.try_into();
            }

            let condition: PrimExpr = mutator.mutate(self, &original_condition)?.try_into()?;
            let true_value: Expr = self.with_constraint(&condition, |simplifier| {
                let branch = value.args.get(1).expect("three call arguments are present");
                mutator.mutate(simplifier, &branch)?.try_into()
            })?;
            let negative: PrimExpr = Not::new(condition.clone())?.into();
            let false_value: Expr = self.with_constraint(&negative, |simplifier| {
                let branch = value.args.get(2).expect("three call arguments are present");
                mutator.mutate(simplifier, &branch)?.try_into()
            })?;
            if int_value(&condition) == Some(1) {
                return Ok(true_value);
            }
            if int_value(&condition) == Some(0) {
                return Ok(false_value);
            }
            return Ok(value
                .copy_with(
                    value.ty.clone(),
                    value.op.clone(),
                    Array::new(vec![condition.into(), true_value, false_value]),
                )
                .into());
        }
        super::utils::mutate_expr_default(self, mutator, value.into())
    }

    fn mutate_expression(&mut self, value: Expr) -> Result<Expr> {
        if let Ok(primitive) = PrimExpr::try_from(&value) {
            return Ok(self.analyzer.simplify(&primitive)?.into());
        }
        Ok(value)
    }

    fn mutate_default(&mut self, value: &MapValue, mutator: &mut Mutator) -> Result<Any> {
        mutate_stmt_expr_default(self, mutator, value)
    }
}
