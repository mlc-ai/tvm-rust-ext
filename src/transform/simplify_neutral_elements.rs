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
    structural_map, structural_mutate, Any, AnyCompatible, DefRegionKind, Map, Result, String,
    StructuralMutator, WalkOrder,
};

use super::utils::int_value;
use crate::ir::{Expr, Var};
use crate::tirx::{Add, For as TirFor, Mul, PrimFunc, Stmt, Sub};

/// Simplify arithmetic identity operations using framework-controlled mapping.
///
/// The post-order map rewrites children first, then removes `x + 0`, `0 + x`,
/// `x - 0`, `x * 1`, and `1 * x` throughout the expression graph.
pub fn simplify_neutral_elements_expr(expr: Expr) -> Result<Expr> {
    let mut mapper = NeutralElementSimplifier;
    structural_map(expr, &mut mapper, WalkOrder::PostOrder)?.try_into()
}

/// Apply neutral-element simplification throughout a TIR PrimFunc.
pub fn simplify_neutral_elements_prim_func(func: PrimFunc) -> Result<PrimFunc> {
    let mut mapper = NeutralElementSimplifier;
    structural_map(func, &mut mapper, WalkOrder::PostOrder)?.try_into()
}

struct NeutralElementSimplifier;

#[tvm_ffi::dispatch(map)]
impl NeutralElementSimplifier {
    fn map_add(&mut self, value: Add) -> Result<Any> {
        if int_value(&value.a) == Some(0) {
            return Ok(value.b.to_any());
        }
        if int_value(&value.b) == Some(0) {
            return Ok(value.a.to_any());
        }
        Ok(Any::from(value))
    }

    fn map_subtract(&mut self, value: Sub) -> Result<Any> {
        if int_value(&value.b) == Some(0) {
            return Ok(value.a.to_any());
        }
        Ok(Any::from(value))
    }

    fn map_multiply(&mut self, value: Mul) -> Result<Any> {
        if int_value(&value.a) == Some(1) {
            return Ok(value.b.to_any());
        }
        if int_value(&value.b) == Some(1) {
            return Ok(value.a.to_any());
        }
        Ok(Any::from(value))
    }
}

#[derive(Default)]
struct LoopBodyMutator {
    depth: usize,
}

/// Simplify neutral arithmetic operations only while inside a loop body.
///
/// This uses callback-controlled mutation because the state transition must
/// surround one specific child (`For.body`).  Bounds of a top-level loop are
/// left unchanged, while bounds of a nested loop are transformed because that
/// loop itself occurs inside its parent's body.
pub fn simplify_neutral_elements_in_loop_bodies(statement: Stmt) -> Result<Stmt> {
    let mut mutator = LoopBodyMutator::default();
    structural_mutate(statement, &mut mutator)?.try_into()
}

#[tvm_ffi::dispatch(mutate)]
impl LoopBodyMutator {
    fn mutate_loop(&mut self, value: TirFor, region: DefRegionKind) -> Result<Any> {
        let loop_var = Var::try_from(self.mutate(&value.loop_var, DefRegionKind::Recursive)?)?;
        let minimum = Expr::try_from(self.mutate(&value.min, region)?)?;
        let extent = Expr::try_from(self.mutate(&value.extent, region)?)?;

        self.depth += 1;
        let body_result = self.mutate(&value.body, region);
        self.depth -= 1;
        let body = Stmt::try_from(body_result?)?;

        let thread_binding =
            Option::<crate::tirx::IterVar>::try_from(self.mutate(&value.thread_binding, region)?)?;
        let annotations = Map::<String, Any>::try_from(self.mutate(&value.annotations, region)?)?;
        let step = value
            .step
            .as_ref()
            .map(|step| self.mutate(step, region).and_then(Expr::try_from))
            .transpose()?;

        Ok(Any::from(TirFor::with_metadata(
            loop_var,
            minimum,
            extent,
            value.kind,
            body,
            thread_binding,
            annotations,
            step,
            value.span.as_ref(),
        )?))
    }

    fn mutate_scoped_add(&mut self, value: Add, region: DefRegionKind) -> Result<Any> {
        let value = Add::try_from(self.default_mutate_value(&value, region)?)?;
        if self.depth == 0 {
            return Ok(Any::from(value));
        }
        if int_value(&value.a) == Some(0) {
            return Ok(value.b.to_any());
        }
        if int_value(&value.b) == Some(0) {
            return Ok(value.a.to_any());
        }
        Ok(Any::from(value))
    }

    fn mutate_scoped_subtract(&mut self, value: Sub, region: DefRegionKind) -> Result<Any> {
        let value = Sub::try_from(self.default_mutate_value(&value, region)?)?;
        if self.depth > 0 && int_value(&value.b) == Some(0) {
            return Ok(value.a.to_any());
        }
        Ok(Any::from(value))
    }

    fn mutate_scoped_multiply(&mut self, value: Mul, region: DefRegionKind) -> Result<Any> {
        let value = Mul::try_from(self.default_mutate_value(&value, region)?)?;
        if self.depth == 0 {
            return Ok(Any::from(value));
        }
        if int_value(&value.a) == Some(1) {
            return Ok(value.b.to_any());
        }
        if int_value(&value.b) == Some(1) {
            return Ok(value.a.to_any());
        }
        Ok(Any::from(value))
    }
}
