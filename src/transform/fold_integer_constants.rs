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

use tvm_ffi::{structural_map, Any, DLDataType, DLDataTypeCode, ObjectRefCast, Result, WalkOrder};

use super::utils::LazyAnalyzer;
use crate::ir::{Expr, IntImm, PrimExpr, PrimType, Span};
use crate::tirx::{Add, Mul, PrimFunc, Sub};

/// Fold binary integer expressions with checked Rust arithmetic and TVM's analyzer.
pub fn fold_integer_constants_expr(expr: Expr) -> Result<Expr> {
    let mut mapper = IntegerConstantFolder::default();
    structural_map(expr, &mut mapper, WalkOrder::PostOrder)?.try_into()
}

/// Fold binary integer constants throughout a TIR PrimFunc.
pub fn fold_integer_constants_prim_func(function: PrimFunc) -> Result<PrimFunc> {
    let mut mapper = IntegerConstantFolder::default();
    structural_map(function, &mut mapper, WalkOrder::PostOrder)?.try_into()
}

#[derive(Default)]
struct IntegerConstantFolder {
    analyzer: LazyAnalyzer,
}

impl IntegerConstantFolder {
    fn fold_or_analyze(
        &mut self,
        original: PrimExpr,
        lhs: &Expr,
        rhs: &Expr,
        span: Option<&Span>,
        operation: fn(i64, i64) -> Option<i64>,
    ) -> Result<Any> {
        if let Some(folded) = try_fold_binary(lhs, rhs, span, operation) {
            return Ok(Any::from(folded));
        }
        if is_matching_integer_pair(lhs, rhs) {
            return Ok(Any::from(self.analyzer.get()?.simplify(&original)?));
        }
        Ok(Any::from(original))
    }
}

#[tvm_ffi::dispatch(map)]
impl IntegerConstantFolder {
    fn map_add(&mut self, value: Add) -> Result<Any> {
        self.fold_or_analyze(
            value.clone().into(),
            &value.a,
            &value.b,
            value.span.as_ref(),
            i64::checked_add,
        )
    }

    fn map_subtract(&mut self, value: Sub) -> Result<Any> {
        self.fold_or_analyze(
            value.clone().into(),
            &value.a,
            &value.b,
            value.span.as_ref(),
            i64::checked_sub,
        )
    }

    fn map_multiply(&mut self, value: Mul) -> Result<Any> {
        self.fold_or_analyze(
            value.clone().into(),
            &value.a,
            &value.b,
            value.span.as_ref(),
            i64::checked_mul,
        )
    }
}

fn try_fold_binary(
    lhs: &Expr,
    rhs: &Expr,
    span: Option<&Span>,
    operation: fn(i64, i64) -> Option<i64>,
) -> Option<Expr> {
    let (lhs, rhs, dtype) = matching_integer_literals(lhs, rhs)?;
    let value = operation(lhs.value, rhs.value)?;
    IntImm::from_dtype_with_span(dtype, value, span)
        .ok()
        .map(Expr::from)
}

fn is_matching_integer_pair(lhs: &Expr, rhs: &Expr) -> bool {
    matching_integer_literals(lhs, rhs).is_some()
}

fn matching_integer_literals(lhs: &Expr, rhs: &Expr) -> Option<(IntImm, IntImm, DLDataType)> {
    let lhs = lhs.clone().try_cast::<IntImm>().ok()?;
    let rhs = rhs.clone().try_cast::<IntImm>().ok()?;
    let lhs_dtype = lhs.ty.clone().try_cast::<PrimType>().ok()?.dtype;
    let rhs_dtype = rhs.ty.clone().try_cast::<PrimType>().ok()?.dtype;
    if lhs_dtype != rhs_dtype
        || (lhs_dtype.code != DLDataTypeCode::kDLInt as u8
            && lhs_dtype.code != DLDataTypeCode::kDLUInt as u8)
    {
        return None;
    }
    Some((lhs, rhs, lhs_dtype))
}
