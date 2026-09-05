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

use super::*;
use tvm::analysis::{verify_memory, verify_ssa};
use tvm::ir::Op;
use tvm::target::Target;
use tvm::tvm_ffi::String as FfiString;

#[test]
fn rust_verify_ssa_matches_cpp_definition_rules() -> Result<()> {
    load_tvm_compiler();
    let variable = Var::new("x", "int32")?;
    let other = Var::new("y", "int32")?;
    let z = Var::new("z", "int32")?;
    let bind: Stmt = Bind::new(variable.clone(), int_expression(1))?.into();
    let loop_node: Stmt = For::new(
        &variable,
        int_expression(0),
        int_expression(4),
        Evaluate::new(&variable)?,
    )?
    .into();
    let let_one = Let::new(variable.clone(), int_expression(1), variable.clone())?;
    let buffer = BufferType::new(
        "global",
        "int32",
        vec![variable.clone().into(), variable.clone().into()],
    )?
    .new_var("buffer");
    let allocation: Stmt = AllocBuffer::new(&buffer)?.into();
    let empty: Stmt = Evaluate::from_i64(0)?.into();
    let inner_left = Let::new(other.clone(), int_expression(1), other.clone())?;
    let inner_right = Let::new(z.clone(), int_expression(1), z.clone())?;
    let cases = [
        (
            "free use",
            Vec::new(),
            Evaluate::new(&variable)?.into(),
            true,
        ),
        (
            "duplicate parameters",
            vec![variable.clone(), variable.clone()],
            empty.clone(),
            false,
        ),
        (
            "parameter redefinition",
            vec![variable.clone()],
            bind.clone(),
            false,
        ),
        (
            "repeated Bind",
            Vec::new(),
            SeqStmt::new(vec![bind.clone(), bind])?.into(),
            false,
        ),
        (
            "repeated For",
            Vec::new(),
            SeqStmt::new(vec![loop_node.clone(), loop_node])?.into(),
            false,
        ),
        (
            "repeated allocation",
            Vec::new(),
            SeqStmt::new(vec![allocation.clone(), allocation])?.into(),
            false,
        ),
        (
            "shared Let",
            Vec::new(),
            Evaluate::new(Add::new(let_one.clone(), let_one.clone())?)?.into(),
            true,
        ),
        (
            "equal Let values",
            Vec::new(),
            Evaluate::new(Add::new(
                let_one.clone(),
                Let::new(variable.clone(), int_expression(1), variable.clone())?,
            )?)?
            .into(),
            true,
        ),
        (
            "different Let values",
            Vec::new(),
            Evaluate::new(Add::new(
                let_one,
                Let::new(variable.clone(), int_expression(2), variable.clone())?,
            )?)?
            .into(),
            false,
        ),
        (
            "no alpha remapping",
            Vec::new(),
            Evaluate::new(Add::new(
                Let::new(variable.clone(), inner_left, variable.clone())?,
                Let::new(variable.clone(), inner_right, variable.clone())?,
            )?)?
            .into(),
            false,
        ),
        (
            "shared shape symbols",
            vec![buffer.as_var().clone()],
            empty,
            true,
        ),
        (
            "shape symbol redefinition",
            vec![buffer.as_var().clone()],
            Bind::new(variable, int_expression(1))?.into(),
            false,
        ),
    ];
    let native_check = Function::get_global("tirx.analysis.verify_ssa")?;
    let rust_pass = transform::verify_ssa()?;
    let native_pass = cpp_pass("tirx.transform.VerifySSA");
    for (name, params, body, expected) in cases {
        let function = PrimFunc::new(params, body)?;
        let native: bool = native_check.call_tuple((&function,))?.try_into()?;
        assert_eq!(native, expected, "{name}");
        assert_eq!(verify_ssa(&function)?, expected, "{name}");
        let module = IRModule::from_expr(&function)?;
        let rust = rust_pass.run(module.clone());
        let native = native_pass.run(module.clone());
        assert_eq!(rust.is_ok(), native.is_ok(), "{name}");
        if expected {
            assert!(rust?.same_as(&module), "{name}");
            assert!(native?.same_as(&module), "{name}");
        }
    }
    let module = IRModule::new(Map::new())?;
    assert!(rust_pass.run(module.clone())?.same_as(&module));
    Ok(())
}

#[test]
fn rust_verify_memory_matches_cpp_target_scope_and_argument_rules() -> Result<()> {
    load_tvm_compiler();
    let buffer_type = BufferType::new("global", "int32", vec![int_expression(8)])?;
    let buffer = buffer_type.new_var("data");
    let internal = buffer_type.new_var("internal");
    let thread = Var::new("thread", "int32")?;
    let store: Stmt = BufferStore::new(&buffer, int_expression(1), vec![int_expression(0)])?.into();
    let load: Stmt =
        Evaluate::new(TensorLoad::from_buffer(&buffer, vec![int_expression(0)])?)?.into();
    let bound: Stmt = AttrStmt::new(
        thread.clone(),
        "thread_extent",
        int_expression(8),
        store.clone(),
    )?
    .into();
    let pointer_type = PointerType::new(PrimType::void(), "")?;
    let first = Var::with_type("packed", pointer_type.clone());
    let struct_get = Op::get("tirx.tvm_struct_get")?;
    let middle = Var::with_type("middle", pointer_type.clone());
    let derived = Var::with_type("derived", pointer_type.clone());
    let extract = |source: &Var| {
        Call::new(
            pointer_type.clone(),
            &struct_get,
            vec![source.clone().into(), int_expression(0), int_expression(1)],
        )
    };
    let derived_load: Stmt = Evaluate::new(Call::new(
        PrimType::new("int32")?,
        Op::get("tirx.masked_load")?,
        vec![
            derived.clone().into(),
            int_expression(0),
            typed_int_expression("bool", 1),
        ],
    ))?
    .into();
    let chain = SeqStmt::new(vec![
        Bind::new(middle.clone(), extract(&first))?.into(),
        Bind::new(derived.clone(), extract(&middle))?.into(),
        derived_load.clone(),
    ])?;
    let default_params = vec![buffer.as_var().clone()];
    let cases = [
        (
            "unbound store",
            default_params.clone(),
            store.clone(),
            false,
        ),
        ("unbound load", default_params.clone(), load.clone(), false),
        (
            "buffer is not first parameter",
            vec![first.clone(), buffer.as_var().clone()],
            load.clone(),
            false,
        ),
        ("thread extent", default_params.clone(), bound.clone(), true),
        (
            "nested thread extent",
            default_params.clone(),
            AttrStmt::new(
                thread.clone(),
                "thread_extent",
                int_expression(8),
                bound.clone(),
            )?
            .into(),
            true,
        ),
        (
            "thread scope exit",
            default_params.clone(),
            SeqStmt::new(vec![bound, store.clone()])?.into(),
            false,
        ),
        (
            "attribute expression",
            default_params.clone(),
            AttrStmt::new(
                thread.clone(),
                "thread_extent",
                TensorLoad::from_buffer(&buffer, vec![int_expression(0)])?,
                Evaluate::from_i64(0)?,
            )?
            .into(),
            true,
        ),
        (
            "virtual thread is not thread extent",
            default_params.clone(),
            AttrStmt::new(
                thread.clone(),
                "virtual_thread",
                int_expression(8),
                store.clone(),
            )?
            .into(),
            false,
        ),
        (
            "For is not thread extent",
            default_params.clone(),
            For::with_metadata(
                thread.clone(),
                int_expression(0),
                int_expression(8),
                ForKind::kThreadBinding,
                store.clone(),
                Some(IterVar::with_metadata(
                    None,
                    thread.clone(),
                    IterVarType::kThreadIndex,
                    "threadIdx.x",
                    None,
                )?),
                Map::new(),
                None,
                None,
            )?
            .into(),
            false,
        ),
        (
            "internal buffer",
            default_params.clone(),
            Evaluate::new(TensorLoad::from_buffer(&internal, vec![int_expression(0)])?)?.into(),
            true,
        ),
        (
            "packed argument chain",
            vec![first.clone()],
            chain.into(),
            false,
        ),
        (
            "plain alias is not struct_get",
            vec![first.clone()],
            SeqStmt::new(vec![
                Bind::new(derived.clone(), &first)?.into(),
                derived_load.clone(),
            ])?
            .into(),
            true,
        ),
        (
            "struct_get from non-first pointer",
            vec![first.clone(), middle.clone()],
            SeqStmt::new(vec![
                Bind::new(derived.clone(), extract(&middle))?.into(),
                derived_load.clone(),
            ])?
            .into(),
            true,
        ),
        (
            "masked load",
            default_params.clone(),
            Evaluate::new(Call::new(
                PrimType::new("int32")?,
                Op::get("tirx.masked_load")?,
                vec![
                    buffer.as_var().clone().into(),
                    int_expression(0),
                    typed_int_expression("bool", 1),
                ],
            ))?
            .into(),
            false,
        ),
        (
            "masked store",
            default_params.clone(),
            Evaluate::new(Call::new(
                PrimType::void(),
                Op::get("tirx.masked_store")?,
                vec![
                    buffer.as_var().clone().into(),
                    int_expression(1),
                    int_expression(0),
                    typed_int_expression("bool", 1),
                ],
            ))?
            .into(),
            false,
        ),
    ];
    let cuda = Target::new("cuda")?;
    let llvm = Target::new("llvm")?;
    let native_check = Function::get_global("tirx.analysis.verify_memory")?;
    let rust_pass = transform::verify_memory()?;
    let native_pass = cpp_pass("tirx.transform.VerifyMemory");
    for (name, params, body, gpu_expected) in cases {
        for (target, calling_conv, expected) in [
            (Some(&cuda), 0, gpu_expected),
            (Some(&llvm), 0, true),
            (None, 0, true),
            (Some(&cuda), 2, true),
        ] {
            let mut attrs = vec![(FfiString::from("calling_conv"), Any::from(calling_conv))];
            if let Some(target) = target {
                attrs.push((FfiString::from("target"), Any::from(target.clone())));
            }
            let function = PrimFunc::with_metadata(
                params.clone(),
                body.clone(),
                Type::missing(),
                DictAttrs::from_dictionary(Map::from_iter(attrs)),
                None,
            )?;
            let native: bool = native_check.call_tuple((&function,))?.try_into()?;
            assert_eq!(native, expected, "{name}, calling_conv={calling_conv}");
            assert_eq!(
                verify_memory(&function)?,
                expected,
                "{name}, calling_conv={calling_conv}"
            );
            let module = IRModule::from_expr(&function)?;
            let rust = rust_pass.run(module.clone());
            let native = native_pass.run(module.clone());
            assert_eq!(rust.is_ok(), native.is_ok(), "{name}");
            if expected {
                assert!(rust?.same_as(&module), "{name}");
                assert!(native?.same_as(&module), "{name}");
            } else {
                assert!(
                    rust.err()
                        .unwrap()
                        .to_string()
                        .contains("Memory verification failed"),
                    "{name}"
                );
            }
        }
    }
    // Malformed cyclic definitions must fail, not loop forever. Do not run
    // this input through the native verifier, whose chain traversal is unbounded.
    let cycle = PrimFunc::with_metadata(
        vec![first],
        SeqStmt::new(vec![
            Bind::new(derived.clone(), extract(&derived))?.into(),
            derived_load,
        ])?,
        Type::missing(),
        DictAttrs::from_dictionary(Map::from_iter([(
            FfiString::from("target"),
            Any::from(cuda),
        )])),
        None,
    )?;
    assert!(verify_memory(&cycle)
        .unwrap_err()
        .to_string()
        .contains("cyclic tvm_struct_get"));
    Ok(())
}
