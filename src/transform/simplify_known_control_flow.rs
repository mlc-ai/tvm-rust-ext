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

use tvm_ffi::{structural_map, Any, ObjectRefCast, Result, WalkOrder};

use super::utils::{int_value, LazyAnalyzer};
use crate::analysis::side_effect;
use crate::ir::{Call, Expr, PrimExpr};
use crate::tirx::{Evaluate, For as TirFor, IfThenElse, PrimFunc, SeqStmt, Stmt};

/// Remove control flow proven inactive by literals or TVM's arithmetic analyzer.
pub fn simplify_known_control_flow_prim_func(function: PrimFunc) -> Result<PrimFunc> {
    let mut mapper = KnownControlFlowSimplifier::default();
    structural_map(function, &mut mapper, WalkOrder::PostOrder)?.try_into()
}

#[derive(Default)]
struct KnownControlFlowSimplifier {
    analyzer: LazyAnalyzer,
}

impl KnownControlFlowSimplifier {
    fn known_integer(&mut self, expression: &PrimExpr) -> Result<Option<i64>> {
        if let Some(value) = int_value(expression) {
            return Ok(Some(value));
        }
        let simplified = self.analyzer.get()?.simplify(expression)?;
        Ok(int_value(&simplified))
    }

    fn preserve_update_or_no_op(&self, expression: &Expr) -> Result<Stmt> {
        if expression_may_update_state(expression)? {
            Ok(Evaluate::new(expression.clone())?.into())
        } else {
            no_op()
        }
    }
}

#[tvm_ffi::dispatch(map)]
impl KnownControlFlowSimplifier {
    fn map_evaluation(&mut self, value: Evaluate) -> Result<Any> {
        if int_value(&value.value).is_some() {
            Ok(Any::from(no_op()?))
        } else if expression_may_update_state(&value.value)? {
            Ok(Any::from(value))
        } else {
            Ok(Any::from(no_op()?))
        }
    }

    fn map_conditional(&mut self, value: IfThenElse) -> Result<Any> {
        if let Some(condition) = self.known_integer(&value.condition)? {
            let selected = if condition != 0 {
                value.then_case.clone()
            } else {
                match &value.else_case {
                    Some(else_case) => else_case.clone(),
                    None => no_op()?,
                }
            };
            return Ok(Any::from(canonical_no_op(selected)?));
        }

        let no_op_then = is_no_op(&value.then_case);
        let no_op_else = match &value.else_case {
            Some(else_case) => is_no_op(else_case),
            None => true,
        };
        if no_op_then && no_op_else {
            return Ok(Any::from(self.preserve_update_or_no_op(&value.condition)?));
        }
        if let Some(else_case) = &value.else_case {
            if is_no_op(else_case) {
                return Ok(Any::from(Stmt::from(IfThenElse::with_span(
                    &value.condition,
                    &value.then_case,
                    None,
                    value.span.as_ref(),
                )?)));
            }
        }
        Ok(Any::from(value))
    }

    fn map_loop(&mut self, value: TirFor) -> Result<Any> {
        if self.known_integer(&value.extent)? == Some(0) {
            return Ok(Any::from(no_op()?));
        }
        if is_no_op(&value.body)
            && self.known_integer(&value.min)?.is_some()
            && self.known_integer(&value.extent)?.is_some()
        {
            return Ok(Any::from(no_op()?));
        }
        Ok(Any::from(value))
    }

    fn map_sequence(&mut self, value: SeqStmt) -> Result<Any> {
        Ok(Any::from(Stmt::sequence_with_span(
            value.seq.iter().collect(),
            value.span.as_ref(),
        )?))
    }
}

/// Match C++ `RemoveNoOp` handling for the general `Expr` stored by `Evaluate`.
fn expression_may_update_state(expression: &Expr) -> Result<bool> {
    if let Ok(primitive) = PrimExpr::try_from(expression) {
        return Ok(side_effect(&primitive)?.may_update_state());
    }
    Ok(expression.clone().try_cast::<Call>().is_ok())
}

fn no_op() -> Result<Stmt> {
    Ok(Evaluate::from_i64(0)?.into())
}

fn canonical_no_op(statement: Stmt) -> Result<Stmt> {
    if is_no_op(&statement) {
        no_op()
    } else {
        Ok(statement)
    }
}

fn is_no_op(statement: &Stmt) -> bool {
    let Ok(evaluate) = statement.clone().try_cast::<Evaluate>() else {
        return false;
    };
    int_value(&evaluate.value).is_some()
}
