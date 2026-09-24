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

use tvm::ir::{Expr, Span, Var};
use tvm::prim::{BitwiseAnd, BitwiseNot, BitwiseOr, BitwiseXor, LShift, RShift};
use tvm::tvm_ffi::{Function, Result};
mod common;
use common::{assert_structural_equal, load_tvm_compiler};

#[test]
fn bitwise_nodes_match_native_constructors_and_reject_invalid_types() -> Result<()> {
    load_tvm_compiler();
    macro_rules! binary {
        ($node:ident, $allow_bool:literal) => {
            for dtype in [
                "int8", "int32", "int64", "uint8", "uint64", "int32x4", "bool",
            ] {
                let a = Var::new("a", dtype)?;
                let b = Var::new("b", dtype)?;
                let rust = $node::new(a.clone(), b.clone());
                let native = Function::get_global(concat!("prim.", stringify!($node)))?
                    .call_tuple((a, b, Option::<Span>::None));
                if dtype == "bool" && !$allow_bool {
                    assert!(rust.is_err());
                    assert!(native.is_err());
                } else {
                    let rust: Expr = rust?.into();
                    let native: Expr = native?.try_into()?;
                    assert_structural_equal(&rust, &native);
                }
            }
            assert!($node::new(Var::new("a", "float32")?, Var::new("b", "float32")?).is_err());
            assert!($node::new(Var::new("a", "int32")?, Var::new("b", "int64")?).is_err());
        };
    }
    binary!(LShift, false);
    binary!(RShift, false);
    binary!(BitwiseAnd, true);
    binary!(BitwiseOr, true);
    binary!(BitwiseXor, true);
    for dtype in ["int8", "uint64", "int32x4", "bool"] {
        let a = Var::new("a", dtype)?;
        let rust = BitwiseNot::new(a.clone())?;
        let native: Expr = Function::get_global("prim.BitwiseNot")?
            .call_tuple((a, Option::<Span>::None))?
            .try_into()?;
        assert_structural_equal(&rust, &native);
    }
    assert!(BitwiseNot::new(Var::new("a", "float32")?).is_err());
    Ok(())
}
