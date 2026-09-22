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
    structural_mutate, structural_visit, Any, Array, Mutator, ObjectIdentity, ObjectRefCast,
    ObjectRefCore, Result, StructuralView, VisitCallbacks, VisitContext, VisitInterrupt,
};

use super::utils::{
    get_operator, is_evaluate_zero, mutate_stmt_expr_default, option_same_as,
    visit_stmt_expr_default, with_prim_func_body,
};
use super::{create_prim_func_pass, Pass};
use crate::ir::StringImm;
use crate::ir::{CallObj, Expr, PrimExpr, Var};
use crate::tirx::{Bind, Evaluate, For, IfThenElse, PrimFunc, SeqStmt, Stmt, While};

const ENCODE_TILED_FUNCTION: &str = "runtime.cuTensorMapEncodeTiled";

/// Deduplicate equivalent cuTensorMap allocations and encode calls.
pub fn lower_tirx_dedup_cu_tensor_maps_prim_func(function: PrimFunc) -> Result<PrimFunc> {
    let operators = TensorMapOperators::new()?;
    let mut analyzer = VisitCallbacks::new(
        DedupAnalysis::new(operators),
        (
            analyze_loop,
            analyze_while,
            analyze_conditional,
            analyze_evaluate,
            analyze_default,
        ),
    );
    structural_visit(function.body(), &mut analyzer)?;
    let analysis = analyzer.into_state();
    if analysis.variable_remaps.is_empty() {
        return Ok(function);
    }
    let mut rewriter = DedupRewriter::new(analysis.operators, analysis.variable_remaps);
    let body: Stmt = structural_mutate(function.body().clone(), &mut rewriter)?.try_into()?;
    Ok(with_prim_func_body(function, body))
}

/// Build TVM's `tirx.LowerTIRxDedupCuTensorMaps` pass in Rust.
pub fn lower_tirx_dedup_cu_tensor_maps() -> Result<Pass> {
    create_prim_func_pass(
        "tirx.LowerTIRxDedupCuTensorMaps",
        0,
        Vec::new(),
        false,
        lower_tirx_dedup_cu_tensor_maps_prim_func,
    )
}

#[derive(Clone)]
struct TensorMapOperators {
    stack_alloca: Expr,
    call_packed: Expr,
}

impl TensorMapOperators {
    fn new() -> Result<Self> {
        Ok(Self {
            stack_alloca: get_operator("tirx.tvm_stack_alloca")?,
            call_packed: get_operator("tirx.tvm_call_packed")?,
        })
    }
}

struct DedupAnalysis {
    operators: TensorMapOperators,
    canonical_frames: Vec<Vec<(Array<Expr>, Var)>>,
    variable_remaps: HashMap<ObjectIdentity, Var>,
}

impl DedupAnalysis {
    fn new(operators: TensorMapOperators) -> Self {
        Self {
            operators,
            canonical_frames: vec![Vec::new()],
            variable_remaps: HashMap::new(),
        }
    }

    fn record_encode(&mut self, call: &CallObj) -> Result<()> {
        let Some((variable, key)) = extract_encode_key(call, &self.operators) else {
            return Ok(());
        };
        for frame in &self.canonical_frames {
            for (existing_key, canonical) in frame {
                if structurally_equal(existing_key, &key)? {
                    if !canonical.same_as(&variable) {
                        self.variable_remaps
                            .insert(ObjectIdentity::of(&variable), canonical.clone());
                    }
                    return Ok(());
                }
            }
        }
        self.canonical_frames
            .last_mut()
            .expect("a root canonical frame is always present")
            .push((key, variable));
        Ok(())
    }
}

fn analyze_loop(value: For, visitor: &mut VisitContext<'_, DedupAnalysis>) -> Result<()> {
    visitor.visit(&value.min)?;
    visitor.visit(&value.extent)?;
    visitor.state_mut().canonical_frames.push(Vec::new());
    let result = visitor.visit(&value.body);
    visitor.state_mut().canonical_frames.pop();
    result.map(|_| ())
}

fn analyze_while(value: While, visitor: &mut VisitContext<'_, DedupAnalysis>) -> Result<()> {
    visitor.visit(&value.condition)?;
    visitor.state_mut().canonical_frames.push(Vec::new());
    let result = visitor.visit(&value.body);
    visitor.state_mut().canonical_frames.pop();
    result.map(|_| ())
}

fn analyze_conditional(
    value: IfThenElse,
    visitor: &mut VisitContext<'_, DedupAnalysis>,
) -> Result<()> {
    visitor.visit(&value.condition)?;
    visitor.state_mut().canonical_frames.push(Vec::new());
    let then_result = visitor.visit(&value.then_case);
    visitor.state_mut().canonical_frames.pop();
    then_result?;
    if let Some(else_case) = &value.else_case {
        visitor.state_mut().canonical_frames.push(Vec::new());
        let else_result = visitor.visit(else_case);
        visitor.state_mut().canonical_frames.pop();
        else_result?;
    }
    Ok(())
}

fn analyze_evaluate(value: Evaluate, visitor: &mut VisitContext<'_, DedupAnalysis>) -> Result<()> {
    if let Some(call) = value.value.as_node::<CallObj>() {
        visitor.state_mut().record_encode(call)?;
    }
    visitor.visit(&value.value)?;
    Ok(())
}

fn analyze_default(
    value: &StructuralView,
    visitor: &mut VisitContext<'_, DedupAnalysis>,
) -> Result<Option<VisitInterrupt>> {
    visit_stmt_expr_default(visitor, value)
}

struct DedupRewriter {
    operators: TensorMapOperators,
    variable_remaps: HashMap<ObjectIdentity, Var>,
    emitted_key_frames: Vec<Vec<Array<Expr>>>,
}

impl DedupRewriter {
    fn new(operators: TensorMapOperators, variable_remaps: HashMap<ObjectIdentity, Var>) -> Self {
        Self {
            operators,
            variable_remaps,
            emitted_key_frames: vec![Vec::new()],
        }
    }

    fn mutate_scoped_body(&mut self, mutator: &mut Mutator, body: &Stmt) -> Result<Stmt> {
        self.emitted_key_frames.push(Vec::new());
        let result = mutator.mutate(self, body).and_then(Stmt::try_from);
        self.emitted_key_frames.pop();
        result
    }

    fn key_was_emitted(&self, key: &Array<Expr>) -> Result<bool> {
        for frame in &self.emitted_key_frames {
            for existing in frame {
                if structurally_equal(existing, key)? {
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }
}

#[tvm_ffi::dispatch(mutate)]
impl DedupRewriter {
    fn mutate_variable(&mut self, value: Var) -> Var {
        self.variable_remaps
            .get(&ObjectIdentity::of(&value))
            .cloned()
            .unwrap_or(value)
    }

    fn mutate_sequence(&mut self, value: SeqStmt, mutator: &mut Mutator) -> Result<Stmt> {
        let mut changed = false;
        let mut statements = Vec::with_capacity(value.seq.len());
        for statement in value.seq.iter() {
            let mapped: Stmt = mutator.mutate(self, &statement)?.try_into()?;
            if is_evaluate_zero(&mapped) {
                changed = true;
                continue;
            }
            changed |= !mapped.same_as(&statement);
            statements.push(mapped);
        }
        if !changed {
            return Ok(value.into());
        }
        Stmt::sequence(statements)
    }

    fn mutate_loop(&mut self, value: For, mutator: &mut Mutator) -> Result<For> {
        let minimum: PrimExpr = mutator.mutate(self, &value.min)?.try_into()?;
        let extent: PrimExpr = mutator.mutate(self, &value.extent)?.try_into()?;
        let body = self.mutate_scoped_body(mutator, &value.body)?;
        if minimum.same_as(&value.min) && extent.same_as(&value.extent) && body.same_as(&value.body)
        {
            return Ok(value);
        }
        Ok(value.copy_with(value.loop_var.clone(), minimum, extent, body))
    }

    fn mutate_while(&mut self, value: While, mutator: &mut Mutator) -> Result<While> {
        let condition: PrimExpr = mutator.mutate(self, &value.condition)?.try_into()?;
        let body = self.mutate_scoped_body(mutator, &value.body)?;
        if condition.same_as(&value.condition) && body.same_as(&value.body) {
            return Ok(value);
        }
        Ok(value.copy_with(condition, body))
    }

    fn mutate_conditional(
        &mut self,
        value: IfThenElse,
        mutator: &mut Mutator,
    ) -> Result<IfThenElse> {
        let condition: PrimExpr = mutator.mutate(self, &value.condition)?.try_into()?;
        let then_case = self.mutate_scoped_body(mutator, &value.then_case)?;
        let else_case = value
            .else_case
            .as_ref()
            .map(|branch| self.mutate_scoped_body(mutator, branch))
            .transpose()?;
        if condition.same_as(&value.condition)
            && then_case.same_as(&value.then_case)
            && option_same_as(&else_case, &value.else_case)
        {
            return Ok(value);
        }
        Ok(IfThenElse::from_complete_fields(
            value.span.clone(),
            condition,
            then_case,
            else_case,
        ))
    }

    fn mutate_binding(&mut self, value: Bind, mutator: &mut Mutator) -> Result<Stmt> {
        let mapped_value: Expr = mutator.mutate(self, &value.value)?.try_into()?;
        if is_tensor_map_alloca(&value, &self.operators)
            && self
                .variable_remaps
                .contains_key(&ObjectIdentity::of(&value.var))
        {
            return Ok(Evaluate::from_i64(0)?.into());
        }
        if mapped_value.same_as(&value.value) {
            return Ok(value.into());
        }
        Ok(value.copy_with(value.var.clone(), mapped_value).into())
    }

    fn mutate_evaluate(&mut self, value: Evaluate, mutator: &mut Mutator) -> Result<Evaluate> {
        let mapped_value: Expr = mutator.mutate(self, &value.value)?.try_into()?;
        let mapped = if mapped_value.same_as(&value.value) {
            value
        } else {
            value.copy_with(mapped_value)
        };
        let Some(call) = mapped.value.as_node::<CallObj>() else {
            return Ok(mapped);
        };
        let Some((_variable, key)) = extract_encode_key(call, &self.operators) else {
            return Ok(mapped);
        };
        if self.key_was_emitted(&key)? {
            return Evaluate::from_i64(0);
        }
        self.emitted_key_frames
            .last_mut()
            .expect("a root emitted-key frame is always present")
            .push(key);
        Ok(mapped)
    }

    fn mutate_default(&mut self, value: &StructuralView, mutator: &mut Mutator) -> Result<Any> {
        mutate_stmt_expr_default(self, mutator, value)
    }
}

fn extract_encode_key(
    call: &CallObj,
    operators: &TensorMapOperators,
) -> Option<(Var, Array<Expr>)> {
    if !call.op.same_as(&operators.call_packed) || call.args.len() < 2 {
        return None;
    }
    let function = call.args.get(0).ok()?.try_cast::<StringImm>().ok()?;
    if function.value.as_str() != ENCODE_TILED_FUNCTION {
        return None;
    }
    let variable = call.args.get(1).ok()?.try_cast::<Var>().ok()?;
    let key = Array::new(call.args.iter().skip(2).collect());
    Some((variable, key))
}

fn is_tensor_map_alloca(binding: &Bind, operators: &TensorMapOperators) -> bool {
    let Some(call) = binding.value.as_node::<CallObj>() else {
        return false;
    };
    if !call.op.same_as(&operators.stack_alloca) || call.args.len() != 2 {
        return false;
    }
    call.args
        .get(0)
        .ok()
        .and_then(|value| value.try_cast::<StringImm>().ok())
        .is_some_and(|value| value.value.as_str() == "tensormap")
}

fn structurally_equal(lhs: &Array<Expr>, rhs: &Array<Expr>) -> Result<bool> {
    tvm_ffi::cached_global_func!("ffi.StructuralEqual")
        .call_tuple((lhs, rhs, false, false))?
        .try_into()
}
