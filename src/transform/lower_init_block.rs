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

use tvm_ffi::{structural_map, Any, Error, Result, WalkOrder, RUNTIME_ERROR};

use super::{create_prim_func_pass, Pass};
use crate::ir::{Expr, PrimExpr};
use crate::tirx::{And, IfThenElse, IterVar, IterVarType, PrimFunc, SBlock, Stmt, EQ};

/// Lower every block `init` into a guarded statement at the start of its body.
pub fn lower_init_block_prim_func(function: PrimFunc) -> Result<PrimFunc> {
    let mut lower = InitBlockLower;
    structural_map(function, &mut lower, WalkOrder::PostOrder)?.try_into()
}

struct InitBlockLower;

#[tvm_ffi::dispatch(map)]
impl InitBlockLower {
    fn map_block(&mut self, block: SBlock) -> Result<Any> {
        let Some(init) = block.init.clone() else {
            return Ok(Any::from(block));
        };

        let init = lower_init(init, &block.iter_vars.iter().collect::<Vec<_>>())?;
        let body = Stmt::sequence(vec![init, block.body.clone()])?;
        Ok(Any::from(SBlock::from_complete_fields(
            block.span.clone(),
            block.iter_vars.clone(),
            block.reads.clone(),
            block.writes.clone(),
            block.name_hint.clone(),
            block.alloc_buffers.clone(),
            block.match_buffers.clone(),
            block.annotations.clone(),
            None,
            body,
        )))
    }
}

fn lower_init(init: Stmt, iter_vars: &[IterVar]) -> Result<Stmt> {
    let mut conditions = Vec::new();
    for iter_var in iter_vars {
        if iter_var.iter_type()? != IterVarType::CommutativeReduction {
            continue;
        }
        let domain = iter_var.dom()?.ok_or_else(|| {
            Error::new(
                RUNTIME_ERROR,
                "reduction block iterator must have a domain",
                "",
            )
        })?;
        let variable: Expr = iter_var.var()?.into();
        conditions.push(PrimExpr::from(EQ::new(variable, domain.min.clone())?));
    }

    let mut conditions = conditions.into_iter();
    let Some(mut condition) = conditions.next() else {
        return Ok(init);
    };
    for next in conditions {
        condition = And::new(condition, next)?.into();
    }
    Ok(IfThenElse::new(Expr::from(condition), init, None)?.into())
}

/// Build TVM's `s_tir.LowerInitBlock` PrimFunc pass in Rust.
pub fn lower_init_block() -> Result<Pass> {
    create_prim_func_pass(
        "s_tir.LowerInitBlock",
        0,
        Vec::new(),
        false,
        |function, _module, _context| lower_init_block_prim_func(function),
    )
}
