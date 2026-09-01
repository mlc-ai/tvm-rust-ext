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
    structural_map, structural_mutate, DefRegionKind, Result, StructuralMutator, WalkOrder,
};

use super::utils::int_value;
use crate::ir::{Expr, PrimExpr};
use crate::tirx::{Add, For, Mul, PrimFunc, Stmt, Sub};

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
    fn map_add(&mut self, value: Add) -> PrimExpr {
        if int_value(&value.a) == Some(0) {
            return value.b.clone();
        }
        if int_value(&value.b) == Some(0) {
            return value.a.clone();
        }
        value.into()
    }

    fn map_subtract(&mut self, value: Sub) -> PrimExpr {
        if int_value(&value.b) == Some(0) {
            return value.a.clone();
        }
        value.into()
    }

    fn map_multiply(&mut self, value: Mul) -> PrimExpr {
        if int_value(&value.a) == Some(1) {
            return value.b.clone();
        }
        if int_value(&value.b) == Some(1) {
            return value.a.clone();
        }
        value.into()
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
    fn mutate_for(&mut self, value: For, region: DefRegionKind) -> Result<Stmt> {
        // A matched `mutate_for` owns recursion. Mutate executable loop
        // expressions at the current depth and enter the new scope only for
        // `body`; structural metadata remains unchanged.
        let minimum = PrimExpr::try_from(self.mutate(&value.min, region)?)?;
        let extent = PrimExpr::try_from(self.mutate(&value.extent, region)?)?;

        self.depth += 1;
        let body_result = self.mutate(&value.body, region);
        self.depth -= 1;
        let body = Stmt::try_from(body_result?)?;

        let step = Option::<PrimExpr>::try_from(self.mutate(&value.step, region)?)?;

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
        )
        .into())
    }

    fn mutate_scoped_add(&mut self, value: Add, region: DefRegionKind) -> Result<PrimExpr> {
        let value = Add::try_from(self.default_mutate_value(&value, region)?)?;
        if self.depth == 0 {
            return Ok(value.into());
        }
        if int_value(&value.a) == Some(0) {
            return Ok(value.b.clone());
        }
        if int_value(&value.b) == Some(0) {
            return Ok(value.a.clone());
        }
        Ok(value.into())
    }

    fn mutate_scoped_subtract(&mut self, value: Sub, region: DefRegionKind) -> Result<PrimExpr> {
        let value = Sub::try_from(self.default_mutate_value(&value, region)?)?;
        if self.depth > 0 && int_value(&value.b) == Some(0) {
            return Ok(value.a.clone());
        }
        Ok(value.into())
    }

    fn mutate_scoped_multiply(&mut self, value: Mul, region: DefRegionKind) -> Result<PrimExpr> {
        let value = Mul::try_from(self.default_mutate_value(&value, region)?)?;
        if self.depth == 0 {
            return Ok(value.into());
        }
        if int_value(&value.a) == Some(1) {
            return Ok(value.b.clone());
        }
        if int_value(&value.b) == Some(1) {
            return Ok(value.a.clone());
        }
        Ok(value.into())
    }
}
