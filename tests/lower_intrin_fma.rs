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

// A separate test binary keeps the temporary FMA rule out of other pass tests.
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use tvm::ir::prim::{Add, Broadcast, Cast, Mul};
use tvm::ir::{Call, CallObj, DictAttrs, IRModule, IntImm, Op, PrimExpr, PrimType, Type, Var};
use tvm::target::Target;
use tvm::tirx::{Evaluate, EvaluateObj, PrimFunc};
use tvm::transform::{self, Pass};
use tvm::tvm_ffi::{Any, DLDataTypeCode, Function, Map, ObjectRefCore, Result, String};

mod common;
use common::{assert_structural_equal, load_tvm_compiler};

#[test]
fn rust_lower_intrin_matches_cpp_for_fused_multiply_add() -> Result<()> {
    load_tvm_compiler();
    let a = Var::new("a", "float32")?;
    let b = Var::new("b", "float32")?;
    let c = Var::new("c", "float32")?;
    let expression = Add::new(Mul::new(&a, &b)?, &c)?;
    let mut cases = vec![(vec![a, b, c], expression)];
    let four: PrimExpr = IntImm::new("int32", 4)?.into();
    let scalable = Mul::new(
        four.clone(),
        Call::new(
            PrimType::new("int32")?,
            Op::get("ir.prim.vscale")?,
            Vec::new(),
        ),
    )?
    .into();
    for lanes in [four, scalable] {
        for (source_dtype, dtype) in [("int8", "int16"), ("float16", "float32")] {
            let a = Var::new("a", source_dtype)?;
            let b = Var::new("b", dtype)?;
            let c = Var::new("c", dtype)?;
            let lhs = Broadcast::new(Cast::new(PrimType::new(dtype)?, &a)?, &lanes)?;
            let rhs = Broadcast::new(&b, &lanes)?;
            let added = Broadcast::new(&c, &lanes)?;
            let expression = Add::new(Mul::new(lhs, rhs)?, added)?;
            cases.push((vec![a, b, c], expression));
        }
    }
    let attrs = DictAttrs::from_dictionary(Map::from_iter([(
        "target".into(),
        Any::from(Target::new("llvm")?),
    )]));
    let cpp: Pass = Function::get_global("tirx.transform.LowerIntrin")?
        .call_tuple(())?
        .try_into()?;
    let fma = Op::get("tirx.fma")?;
    let attribute = String::from("llvm.FLowerIntrinsic");
    let calls = Arc::new(AtomicUsize::new(0));
    for (priority, decline) in [(1000, false), (1001, true)] {
        let rule_calls = calls.clone();
        // Keep the fused call, or decline it. This works without an LLVM backend.
        let rule = Function::from_typed(move |call: Call| -> Result<Option<Call>> {
            rule_calls.fetch_add(1, Ordering::Relaxed);
            Ok((!decline).then_some(call))
        });
        Function::get_global("ir.OpSetAttr")?.call_tuple((&fma, &attribute, rule, priority))?;
        for (params, expression) in &cases {
            let floating = expression.a.dtype().code == DLDataTypeCode::kDLFloat as u8;
            // A fused Call is visited once more; a declined fusion falls back to Add.
            let expected_calls = if floating {
                if decline {
                    1
                } else {
                    2
                }
            } else {
                0
            };
            let function = PrimFunc::with_metadata(
                params.clone(),
                Evaluate::new(expression.clone())?,
                Type::missing(),
                attrs.clone(),
                None,
            )?;
            let module = IRModule::from_expr(&function)?;
            let actual = transform::lower_intrin_prim_func(function.clone())?;
            assert_eq!(calls.swap(0, Ordering::Relaxed), expected_calls, "Rust");
            let expected = cpp.run(module)?;
            assert_eq!(calls.swap(0, Ordering::Relaxed), expected_calls, "C++");
            assert_structural_equal(&IRModule::from_expr(&actual)?, &expected);
            if floating {
                if decline {
                    // In particular, do not swap Broadcast(Cast) after the rule declines.
                    assert!(actual.body.same_as(&function.body));
                } else {
                    let evaluation = actual.body.as_node::<EvaluateObj>().unwrap();
                    let fused = evaluation.value.as_node::<CallObj>().unwrap();
                    assert!(fused.op.same_as(&fma));
                }
            }
        }
    }
    Ok(())
}
