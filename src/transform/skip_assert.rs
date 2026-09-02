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

use tvm_ffi::{structural_mutate, Any, MapValue, Mutator, Result};

use super::utils::{mutate_stmt_expr_default, with_prim_func_body};
use super::{create_prim_func_pass, Pass};
use crate::tirx::{AssertStmt, Evaluate, PrimFunc};

/// Replace every `AssertStmt` in a PrimFunc with `Evaluate(0)`.
pub fn skip_assert_prim_func(func: PrimFunc) -> Result<PrimFunc> {
    let body = structural_mutate(func.body.clone(), AssertSkipper)?.try_into()?;
    Ok(with_prim_func_body(func, body))
}

struct AssertSkipper;

#[tvm_ffi::dispatch(mutate)]
impl AssertSkipper {
    fn mutate_assertion(&mut self, _value: AssertStmt) -> Result<Evaluate> {
        Evaluate::from_i64(0)
    }

    fn mutate_default(&mut self, value: &MapValue, mutator: &mut Mutator) -> Result<Any> {
        mutate_stmt_expr_default(self, mutator, value)
    }
}

/// Build the Rust implementation of `tirx.SkipAssert` as a normal TVM pass.
pub fn skip_assert() -> Result<Pass> {
    create_prim_func_pass(
        "tirx.SkipAssert",
        0,
        Vec::new(),
        false,
        skip_assert_prim_func,
    )
}
