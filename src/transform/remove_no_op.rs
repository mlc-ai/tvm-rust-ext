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

use tvm_ffi::derive::{Object, ObjectRef};
use tvm_ffi::{
    Any, Array, FieldGetter, Map, MapValue, Mutator, ObjectArc, ObjectCore, ObjectIdentity,
    ObjectRefCast, ObjectRefCore, Result, String as FfiString, RUNTIME_ERROR,
};

use super::utils::{array_same_as, mutate_stmt_expr_default, with_prim_func_body, BufferRemaps};
use super::{create_prim_func_pass_with_context, Pass, PassContext};
use crate::analysis::{
    side_effect, Analyzer, AnalyzerMutator, AnalyzerMutatorState, CallEffectKind, IntSet,
};
use crate::ir::{Call, Expr, IntImm, PrimExpr, TensorLoad, Var};
use crate::tirx::{
    AllocBuffer, AssertStmt, AttrStmt, Bind, BufferStore, BufferVar, DeclBuffer, Evaluate, For,
    IfThenElse, Let, Not, PrimFunc, Reduce, Select, SeqStmt, Stmt, Sub, LE,
};

const DEBUG_SKIP_REGION: &str = "pragma_debug_skip_region";
const ASYNC_WAIT_QUEUE_SCOPE: &str = "async_wait_queue_scope";
const ASYNC_WAIT_INFLIGHT_COUNT: &str = "async_wait_inflight_count";
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
    let mut remover = NoOpRemover {
        analysis: AnalyzerMutatorState::new(analyzer)?,
        ignore_profiler_call: options.ignore_profiler_call,
        profiler_operators,
        variable_domains: HashMap::new(),
        buffer_remaps: BufferRemaps::default(),
    };
    let body = remover
        .mutate_root(function.body.clone())
        .and_then(Stmt::try_from)?;
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
    analysis: AnalyzerMutatorState,
    ignore_profiler_call: bool,
    profiler_operators: Vec<Expr>,
    variable_domains: HashMap<ObjectIdentity, (Var, IntSet)>,
    buffer_remaps: BufferRemaps,
}

impl AnalyzerMutator for NoOpRemover {
    fn analyzer_state(&self) -> &AnalyzerMutatorState {
        &self.analysis
    }

    fn analyzer_state_mut(&mut self) -> &mut AnalyzerMutatorState {
        &mut self.analysis
    }
}

#[tvm_ffi::dispatch(mutate)]
impl NoOpRemover {
    fn mutate_bind(&mut self, value: Bind, mutator: &mut Mutator) -> Result<Bind> {
        self.default_mutate_bind(mutator, value)
    }

    fn mutate_let(&mut self, value: Let, mutator: &mut Mutator) -> Result<Let> {
        self.default_mutate_let(mutator, value)
    }

    fn mutate_reduce(&mut self, value: Reduce, mutator: &mut Mutator) -> Result<Reduce> {
        self.default_mutate_reduce(mutator, value)
    }

    fn mutate_assertion(&mut self, value: AssertStmt, mutator: &mut Mutator) -> Result<AssertStmt> {
        self.default_mutate_assertion(mutator, value)
    }

    fn mutate_select(&mut self, value: Select, mutator: &mut Mutator) -> Result<PrimExpr> {
        self.default_mutate_select(mutator, value)
    }

    fn mutate_call(&mut self, value: Call, mutator: &mut Mutator) -> Result<Expr> {
        self.default_mutate_call(mutator, value)
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
            let zero = zero_like(&inner.value);
            let negative: PrimExpr = crate::tirx::LT::new(inner.value.clone(), zero)?.into();
            if Analyzer::new()?.can_prove(&negative)? {
                return mutator.mutate(self, &inner.body)?.try_into();
            }
        }

        let mutated = self.default_mutate_attribute(mutator, value)?;
        if is_no_op(&mutated.body) {
            self.make_evaluate(mutated.value.clone())
        } else {
            Ok(mutated.into())
        }
    }

    fn mutate_conditional(&mut self, value: IfThenElse, mutator: &mut Mutator) -> Result<Stmt> {
        let statement = self.default_mutate_conditional(mutator, value)?;
        let Ok(conditional) = statement.clone().try_cast::<IfThenElse>() else {
            return Ok(statement);
        };

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
        let extent_set = self.analysis.int_set(&value.extent, &domains)?;
        let maximum = extent_set.maximum()?;
        let non_positive: PrimExpr = LE::new(maximum.clone(), zero_like(&maximum))?.into();
        if self.analysis.can_prove(&non_positive)? {
            return evaluate_zero();
        }

        let one = one_like(&value.extent);
        let extent_minus_one: PrimExpr = Sub::new(value.extent.clone(), one)?.into();
        let maximum: PrimExpr = crate::tirx::Add::new(value.min.clone(), extent_minus_one)?.into();
        let integer_domain = IntSet::interval(value.min.clone(), maximum)?;
        let identity = ObjectIdentity::of(&value.loop_var);
        let previous = self.variable_domains.insert(
            identity.clone(),
            (value.loop_var.as_var().clone(), integer_domain),
        );
        let mutated = self.default_mutate_loop(mutator, value);
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
            crate::tirx::EQ::new(difference.clone(), zero_like(&difference))?.into();
        if int_value(&self.analysis.simplify(&equal_to_zero)?) == Some(1) {
            return self.store_side_effects(&value);
        }

        if let Ok(load) = value.value.clone().try_cast::<TensorLoad>() {
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
        self.default_mutate_sequence(mutator, value)
    }

    fn mutate_variable(&mut self, value: Var) -> Var {
        self.buffer_remaps.use_variable(&value)
    }

    fn mutate_load(&mut self, value: TensorLoad, mutator: &mut Mutator) -> Result<TensorLoad> {
        let old_source: BufferVar = (&value.source).try_into()?;
        let source = self.buffer_remaps.use_buffer(&old_source).as_var().clone();
        let indices: Array<PrimExpr> = mutator.mutate(self, &value.indices)?.try_into()?;
        if source.same_as(old_source.as_var()) && array_same_as(&indices, &value.indices) {
            return Ok(value);
        }
        Ok(value.copy_with(source.into(), indices))
    }

    fn mutate_allocation(
        &mut self,
        value: AllocBuffer,
        mutator: &mut Mutator,
    ) -> Result<AllocBuffer> {
        let buffer = mutate_buffer_definition(self, mutator, &value.buffer)?;
        if buffer.same_as(&value.buffer) {
            return Ok(value);
        }
        Ok(value.copy_with(buffer))
    }

    fn mutate_declaration(
        &mut self,
        value: DeclBuffer,
        mutator: &mut Mutator,
    ) -> Result<DeclBuffer> {
        let data: Expr = mutator.mutate(self, &value.data)?.try_into()?;
        let buffer = mutate_buffer_definition(self, mutator, &value.buffer)?;
        if data.same_as(&value.data) && buffer.same_as(&value.buffer) {
            return Ok(value);
        }
        Ok(value.copy_with(buffer, data))
    }

    fn mutate_default(&mut self, value: &MapValue, mutator: &mut Mutator) -> Result<Any> {
        mutate_stmt_expr_default(self, mutator, value)
    }
}

impl NoOpRemover {
    fn has_side_effect(&self, value: &Expr) -> Result<bool> {
        if let Ok(primitive) = PrimExpr::try_from(value) {
            if self.ignore_profiler_call && is_profiler_call(&primitive, &self.profiler_operators) {
                return Ok(false);
            }
            return Ok(side_effect(&primitive)? > CallEffectKind::kReadState);
        }
        Ok(value.clone().try_cast::<Call>().is_ok())
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
            .analysis
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
            if !self.analysis.can_prove_equal(&lhs, &rhs)? {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

fn mutate_buffer_definition(
    remover: &mut NoOpRemover,
    mutator: &mut Mutator,
    buffer: &BufferVar,
) -> Result<BufferVar> {
    let mut remaps = std::mem::take(&mut remover.buffer_remaps);
    let result = remaps.mutate_definition(buffer, |expression| {
        mutator.mutate(remover, expression)?.try_into()
    });
    remover.buffer_remaps = remaps;
    result
}

fn get_operator(name: &str) -> Result<Expr> {
    tvm_ffi::cached_global_func!("ir.GetOp")
        .call_tuple((FfiString::from(name),))?
        .try_into()
}

fn is_profiler_call(value: &PrimExpr, profiler_operators: &[Expr]) -> bool {
    let Ok(call) = value.clone().try_cast::<Call>() else {
        return false;
    };
    profiler_operators
        .iter()
        .any(|operator| call.op.same_as(operator))
}

fn evaluate_zero() -> Result<Stmt> {
    Ok(Evaluate::from_i64(0)?.into())
}

fn is_no_op(statement: &Stmt) -> bool {
    statement
        .clone()
        .try_cast::<Evaluate>()
        .ok()
        .and_then(|evaluate| evaluate.value.clone().try_cast::<IntImm>().ok())
        .is_some_and(|literal| literal.value == 0)
}

fn int_value(value: &PrimExpr) -> Option<i64> {
    value
        .clone()
        .try_cast::<IntImm>()
        .ok()
        .map(|literal| literal.value)
}

fn zero_like(value: &PrimExpr) -> PrimExpr {
    IntImm::from_complete_fields(None, value.type_annotation(), 0).into()
}

fn one_like(value: &PrimExpr) -> PrimExpr {
    IntImm::from_complete_fields(None, value.type_annotation(), 1).into()
}
