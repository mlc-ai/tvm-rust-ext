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
    structural_mutate, DefRegionKind, ObjectIdentity, ObjectRefCast, Result, String,
    StructuralMutator,
};

use super::utils::with_prim_func_body;
use super::{create_prim_func_pass, Pass};
use crate::ir::{Call, Expr};
use crate::tirx::{Evaluate, PrimFunc, Stmt};

/// Remove `Evaluate(tirx.assume(...))` using TVM's operator-identity rule.
pub fn remove_assume_prim_func(function: PrimFunc) -> Result<PrimFunc> {
    let assume_op: Expr = tvm_ffi::cached_global_func!("ir.GetOp")
        .call_tuple((String::from("tirx.assume"),))?
        .try_into()?;
    let mut remover = AssumeRemover {
        assume_op: ObjectIdentity::of(&assume_op),
    };
    let body = structural_mutate(function.body.clone(), &mut remover)?.try_into()?;
    Ok(with_prim_func_body(function, body))
}

/// Build the callback pass used as the first stage of TVM's
/// `tirx.RemoveAssume` sequential pass.
pub fn remove_assume_internal() -> Result<Pass> {
    create_prim_func_pass(
        "tirx.RemoveAssumeInternal",
        0,
        Vec::new(),
        false,
        |function, _module, _context| remove_assume_prim_func(function),
    )
}

struct AssumeRemover {
    assume_op: ObjectIdentity,
}

#[tvm_ffi::dispatch(mutate)]
impl AssumeRemover {
    fn mutate_evaluate(&mut self, value: Evaluate, region: DefRegionKind) -> Result<Stmt> {
        if let Ok(call) = value.value.clone().try_cast::<Call>() {
            if ObjectIdentity::of(&call.op) == self.assume_op {
                return Ok(Evaluate::from_i64(0)?.into());
            }
        }
        self.default_mutate_value(&value, region)
            .and_then(Stmt::try_from)
    }
}
