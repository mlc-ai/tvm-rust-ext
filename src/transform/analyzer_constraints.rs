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

use super::utils::{finish_constraint_contexts, int_value};
use crate::analysis::Analyzer;
use crate::ir::{CallObj, Expr, IntImm, PrimExpr};
use crate::prim::{Add, AndObj, EQObj, FloorDivObj, GEObj, GTObj, LEObj, LTObj, Mul, GE, LT};
use tvm_ffi::{Function, ObjectRefCore, Result};

/// Enter the condition and the extra facts used by C++ IRMutatorWithAnalyzer.
pub(super) fn enter_constraint_facts(
    analyzer: &Analyzer,
    condition: &PrimExpr,
    bitwise_and_operator: &Expr,
) -> Result<Vec<Function>> {
    let mut constraints = vec![condition.clone()];
    collect_derived_constraint_facts(condition, bitwise_and_operator, &mut constraints)?;
    let mut exits = Vec::with_capacity(constraints.len());
    for constraint in constraints {
        match analyzer.enter_constraint(&constraint) {
            Ok(exit) => exits.push(exit),
            Err(error) => return finish_constraint_contexts(Err(error), exits),
        }
    }
    Ok(exits)
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
    if let Some(and) = condition.as_node::<AndObj>() {
        collect_derived_constraint_facts(&and.a, bitwise_and_operator, output)?;
        collect_derived_constraint_facts(&and.b, bitwise_and_operator, output)?;
        return Ok(());
    }
    if let Some(call) = condition.as_node::<CallObj>() {
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

    if let Some(compare) = condition.as_node::<EQObj>() {
        collect_floor_div_constraints(&compare.a, &compare.b, CompareKind::Equal, output)?;
    } else if let Some(compare) = condition.as_node::<LTObj>() {
        collect_floor_div_constraints(&compare.a, &compare.b, CompareKind::LessThan, output)?;
    } else if let Some(compare) = condition.as_node::<LEObj>() {
        collect_floor_div_constraints(&compare.a, &compare.b, CompareKind::LessEqual, output)?;
    } else if let Some(compare) = condition.as_node::<GTObj>() {
        collect_floor_div_constraints(&compare.a, &compare.b, CompareKind::GreaterThan, output)?;
    } else if let Some(compare) = condition.as_node::<GEObj>() {
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
    if let (Some(div), Some(value)) = (lhs.as_node::<FloorDivObj>(), int_value(rhs)) {
        append_floor_div_constraints(div, value, kind, output)?;
    }
    if let (Some(div), Some(value)) = (rhs.as_node::<FloorDivObj>(), int_value(lhs)) {
        append_floor_div_constraints(div, value, invert_compare(kind), output)?;
    }
    Ok(())
}

fn append_floor_div_constraints(
    division: &FloorDivObj,
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
    let dtype = division.a.dtype();
    let divisor: PrimExpr = IntImm::from_dtype(dtype, divisor_value)?.into();
    let k: PrimExpr = IntImm::from_dtype(dtype, value)?.into();
    let one: PrimExpr = IntImm::from_dtype(dtype, 1)?.into();
    let lower: PrimExpr = Mul::new(k.clone(), divisor.clone())?.into();
    let next: PrimExpr = Add::new(k, one)?.into();
    let upper: PrimExpr = Mul::new(next, divisor)?.into();

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

fn is_bool8(value: &PrimExpr) -> bool {
    let dtype = value.dtype();
    dtype.code == tvm_ffi::DLDataTypeCode::kDLBool as u8 && dtype.bits == 8
}
