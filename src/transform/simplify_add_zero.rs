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

use tvm_ffi::{structural_map, Any, AnyCompatible, Result, WalkOrder};

use super::utils::int_value;
use crate::ir::{BaseFunc, Expr, IRModule};
use crate::tirx::{Add, PrimFunc};

/// Remove additions whose left or right operand is integer zero.
pub fn simplify_add_zero_expr(expr: Expr) -> Result<Expr> {
    let mut mapper = AddZeroSimplifier;
    structural_map(expr, &mut mapper, WalkOrder::PostOrder)?.try_into()
}

/// Apply the add-zero simplifier throughout a PrimFunc.
pub fn simplify_add_zero_prim_func(func: PrimFunc) -> Result<PrimFunc> {
    let mut mapper = AddZeroSimplifier;
    structural_map(func, &mut mapper, WalkOrder::PostOrder)?.try_into()
}

/// Apply the add-zero simplifier to every function reachable from a module.
pub fn simplify_add_zero_module(module: IRModule) -> Result<IRModule> {
    let mut mapper = AddZeroSimplifier;
    let functions = module.functions.clone();
    // Rebuild an independent module so callers that retained another handle to
    // `module` do not observe this pass's updates.
    let mut output = module.copy_for_update()?;
    for (global_var, function) in functions.iter() {
        let mapped = structural_map(function, &mut mapper, WalkOrder::PostOrder)?;
        output = output.update_function_owned(&global_var, &BaseFunc::try_from(mapped)?)?;
    }
    Ok(output)
}

struct AddZeroSimplifier;

#[tvm_ffi::dispatch(map)]
impl AddZeroSimplifier {
    fn map_add(&mut self, value: Add) -> Result<Any> {
        if int_value(&value.a) == Some(0) {
            return Ok(value.b.to_any());
        }
        if int_value(&value.b) == Some(0) {
            return Ok(value.a.to_any());
        }
        Ok(Any::from(value))
    }
}
