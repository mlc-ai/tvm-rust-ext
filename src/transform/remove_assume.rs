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
    structural_mutate, Any, DefRegionKind, MapValue, ObjectIdentity, ObjectRefCast, ObjectRefCore,
    Result, String, StructuralMutator,
};

use super::utils::{mutate_stmt_expr_default, with_prim_func_body};
use super::{create_prim_func_pass, remove_no_op, remove_no_op_prim_func, sequential, Pass};
use crate::ir::{Call, Expr};
use crate::tirx::{Evaluate, PrimFunc};

/// Remove `Evaluate(tirx.assume(...))` using TVM's operator-identity rule.
pub fn remove_assume_prim_func(function: PrimFunc) -> Result<PrimFunc> {
    remove_no_op_prim_func(remove_assume_nodes(function)?)
}

fn remove_assume_nodes(function: PrimFunc) -> Result<PrimFunc> {
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
        remove_assume_nodes,
    )
}

/// Build TVM's full `tirx.RemoveAssume` sequence in Rust.
pub fn remove_assume() -> Result<Pass> {
    sequential(
        vec![remove_assume_internal()?, remove_no_op()?],
        "tirx.RemoveAssume",
    )
}

struct AssumeRemover {
    assume_op: ObjectIdentity,
}

#[tvm_ffi::dispatch(mutate)]
impl AssumeRemover {
    fn mutate_evaluate(&mut self, value: Evaluate, region: DefRegionKind) -> Result<Evaluate> {
        if let Ok(call) = value.value.clone().try_cast::<Call>() {
            if ObjectIdentity::of(&call.op) == self.assume_op {
                return Evaluate::from_i64(0);
            }
        }
        let evaluated: Expr = self.mutate(&value.value, region)?.try_into()?;
        if evaluated.same_as(&value.value) {
            return Ok(value);
        }
        Ok(Evaluate::from_complete_fields(
            value.span.clone(),
            evaluated,
        ))
    }

    fn mutate_stmt_expr_default(&mut self, value: &MapValue, region: DefRegionKind) -> Result<Any> {
        mutate_stmt_expr_default(self, value, region)
    }
}
