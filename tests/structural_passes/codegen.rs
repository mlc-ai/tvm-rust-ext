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

#[test]
fn native_pass_can_move_a_rust_function_body_and_fail() -> Result<()> {
    load_tvm_compiler();

    fn warp_function(dtype: &str, attrs: DictAttrs) -> Result<PrimFunc> {
        let buffer =
            BufferType::new("warp", dtype, vec![int_expression(64)])?.new_var("warp_buffer");
        let thread = Var::new("thread_idx", "int32")?;
        let axis = IterVar::with_metadata(
            None,
            thread.clone(),
            IterVarType::kThreadIndex,
            "threadIdx.x",
            None,
        )?;
        let value = Var::new("value", dtype)?;
        let index = Mul::new(thread, IntImm::new("int32", 2)?)?;
        let store = BufferStore::new(buffer.clone(), value.clone(), vec![index.into()])?;
        let scope = AttrStmt::new(axis, "thread_extent", IntImm::new("int32", 32)?, store)?;
        let body = Stmt::sequence(vec![AllocBuffer::new(buffer)?.into(), scope.into()])?;
        PrimFunc::with_metadata(vec![value], body, Type::missing(), attrs, None)
    }

    let native = cpp_pass("tirx.transform.LowerWarpMemory");
    let attrs = DictAttrs::from_dictionary(Map::from_iter([(
        "target".into(),
        tvm::target::Target::new("cuda")?.into(),
    )]));
    for dtype in ["int32", "int32x4"] {
        for in_pipeline in [false, true] {
            let function = if in_pipeline {
                PrimFunc::with_metadata(
                    Vec::new(),
                    Evaluate::from_i64(0)?,
                    Type::missing(),
                    attrs.clone(),
                    None,
                )?
            } else {
                warp_function(dtype, attrs.clone())?
            };
            // Do not retain an owning identity key: that would force native COW.
            let pointer = object_pointer(&function);
            assert_eq!(
                tvm::tvm_ffi::ObjectArc::strong_count(PrimFunc::data(&function)),
                1,
            );
            let module = IRModule::from_expr(function)?;
            let result = if in_pipeline {
                let create = transform::create_prim_func_pass(
                    "testing.CreateWarpFunction",
                    0,
                    Vec::new(),
                    false,
                    move |input| warp_function(dtype, input.attrs.clone()),
                )?;
                let pipeline = transform::sequential(vec![create, native.clone()], "testing.Warp")?;
                // Retaining the initial module must not hide an error while
                // destroying a fresh Rust allocation created inside Sequential.
                let result = pipeline.run(module.clone());
                drop(module);
                result
            } else {
                native.run(module)
            };
            if dtype == "int32" {
                let output = result?;
                let function: PrimFunc = output.functions.iter().next().unwrap().1.try_cast()?;
                assert!(function.body().as_node::<tvm::tirx::SeqStmtObj>().is_some());
                if !in_pipeline {
                    assert_eq!(object_pointer(&function), pointer);
                }
            } else {
                // LowerWarpMemory moves body before rejecting this vector store.
                let error = result.err().expect("native pass should reject the store");
                assert!(error
                    .to_string()
                    .contains("can only handle continuous store"));
            }
        }
    }
    Ok(())
}

#[test]
fn rust_lower_warp_memory_matches_cpp_for_flat_buffer_access() {
    load_tvm_compiler();
    let buffer_type = BufferType::new("warp", "int32", vec![int_expression(64)]).unwrap();
    let buffer = buffer_type.new_var("warp_buffer");
    let thread = Var::new("thread_idx", "int32").unwrap();
    let thread_axis = IterVar::with_metadata(
        None,
        thread.clone(),
        IterVarType::kThreadIndex,
        "threadIdx.x",
        None,
    )
    .unwrap();
    let index = Mul::new(thread, IntImm::new("int32", 2).unwrap()).unwrap();
    let store = BufferStore::new(
        buffer.clone(),
        IntImm::new("int32", 1).unwrap(),
        vec![index.clone().into()],
    )
    .unwrap();
    let load = TensorLoad::from_buffer(buffer.clone(), vec![index.into()]).unwrap();
    let scope = AttrStmt::new(
        thread_axis,
        "thread_extent",
        IntImm::new("int32", 32).unwrap(),
        Stmt::sequence(vec![store.into(), Evaluate::new(load).unwrap().into()]).unwrap(),
    )
    .unwrap();
    let body =
        Stmt::sequence(vec![AllocBuffer::new(buffer).unwrap().into(), scope.into()]).unwrap();
    let attrs = DictAttrs::from_dictionary(Map::from_iter([(
        tvm::tvm_ffi::String::from("target"),
        Any::from(tvm::target::Target::new("cuda").unwrap()),
    )]));
    let function = PrimFunc::with_metadata(Vec::new(), body, Type::missing(), attrs, None).unwrap();
    let module = IRModule::from_expr(function.clone()).unwrap();

    let rust_result =
        IRModule::from_expr(transform::lower_warp_memory_prim_func(function).unwrap()).unwrap();
    let cpp_result = cpp_pass("tirx.transform.LowerWarpMemory")
        .run(module)
        .unwrap();

    assert_structural_equal(&rust_result, &cpp_result);
}

#[test]
fn rust_lower_warp_memory_matches_cpp_for_cross_thread_shuffle() {
    load_tvm_compiler();
    let buffer_type = BufferType::new("warp", "int32", vec![int_expression(64)]).unwrap();
    let buffer = buffer_type.new_var("warp_buffer");
    let thread = Var::new("thread_idx", "int32").unwrap();
    let thread_axis = IterVar::with_metadata(
        None,
        thread.clone(),
        IterVarType::kThreadIndex,
        "threadIdx.x",
        None,
    )
    .unwrap();
    let store_index = Mul::new(thread, IntImm::new("int32", 2).unwrap()).unwrap();
    let store = BufferStore::new(
        buffer.clone(),
        IntImm::new("int32", 1).unwrap(),
        vec![store_index.into()],
    )
    .unwrap();
    let load = TensorLoad::from_buffer(buffer.clone(), vec![int_expression(6)]).unwrap();
    let scope = AttrStmt::new(
        thread_axis,
        "thread_extent",
        IntImm::new("int32", 32).unwrap(),
        Stmt::sequence(vec![store.into(), Evaluate::new(load).unwrap().into()]).unwrap(),
    )
    .unwrap();
    let body =
        Stmt::sequence(vec![AllocBuffer::new(buffer).unwrap().into(), scope.into()]).unwrap();
    let attrs = DictAttrs::from_dictionary(Map::from_iter([(
        tvm::tvm_ffi::String::from("target"),
        Any::from(tvm::target::Target::new("cuda").unwrap()),
    )]));
    let function = PrimFunc::with_metadata(Vec::new(), body, Type::missing(), attrs, None).unwrap();
    let module = IRModule::from_expr(function.clone()).unwrap();

    let rust_result =
        IRModule::from_expr(transform::lower_warp_memory_prim_func(function).unwrap()).unwrap();
    let cpp_result = cpp_pass("tirx.transform.LowerWarpMemory")
        .run(module)
        .unwrap();

    assert_structural_equal(&rust_result, &cpp_result);
}

#[test]
fn rust_lower_tvm_builtin_matches_cpp_for_context_id() {
    load_tvm_compiler();
    let context_id: Expr = tvm::ir::Op::get("tirx.tvm_context_id").unwrap().into();
    let body = Evaluate::new(Call::new(
        PrimType::new("int32").unwrap(),
        context_id,
        Vec::new(),
    ))
    .unwrap();
    let attrs = DictAttrs::from_dictionary(Map::from_iter([(
        tvm::tvm_ffi::String::from("target"),
        Any::from(tvm::target::Target::new("llvm").unwrap()),
    )]));
    let function = PrimFunc::with_metadata(Vec::new(), body, Type::missing(), attrs, None).unwrap();
    let module = IRModule::from_expr(function.clone()).unwrap();

    let rust_result =
        IRModule::from_expr(transform::lower_tvm_builtin_prim_func(function).unwrap()).unwrap();
    let cpp_result = cpp_pass("tirx.transform.LowerTVMBuiltin")
        .run(module)
        .unwrap();

    assert_structural_equal(&rust_result, &cpp_result);
}

#[test]
fn rust_lower_tvm_builtin_matches_cpp_for_packed_call_stack() {
    load_tvm_compiler();
    let call_packed: Expr = tvm::ir::Op::get("tirx.tvm_call_packed").unwrap().into();
    let body = Evaluate::new(Call::new(
        PrimType::new("int32").unwrap(),
        call_packed,
        vec![
            StringImm::new("testing.consume").into(),
            IntImm::new("int32", 4).unwrap().into(),
        ],
    ))
    .unwrap();
    let attrs = DictAttrs::from_dictionary(Map::from_iter([(
        tvm::tvm_ffi::String::from("target"),
        Any::from(tvm::target::Target::new("llvm").unwrap()),
    )]));
    let function = PrimFunc::with_metadata(Vec::new(), body, Type::missing(), attrs, None).unwrap();
    let module = IRModule::from_expr(function.clone()).unwrap();

    let rust_result =
        IRModule::from_expr(transform::lower_tvm_builtin_prim_func(function).unwrap()).unwrap();
    let cpp_result = cpp_pass("tirx.transform.LowerTVMBuiltin")
        .run(module)
        .unwrap();

    assert_structural_equal(&rust_result, &cpp_result);
}

#[test]
fn rust_lower_tvm_builtin_matches_cpp_for_shape_and_array_stack() -> Result<()> {
    load_tvm_compiler();
    let operator = |name: &str| tvm::ir::Op::get(name);
    let shape = Call::new(
        PointerType::new(PrimType::new("int64")?, "global")?,
        operator("tirx.tvm_stack_make_shape")?,
        vec![int_expression(4), int_expression(8)],
    );
    let attrs = DictAttrs::from_dictionary(Map::from_iter([(
        "target".into(),
        Any::from(tvm::target::Target::new("llvm")?),
    )]));
    let native = cpp_pass("tirx.transform.LowerTVMBuiltin");
    for (dtype, offset, supported) in [
        (None, 0, true),
        (Some("float32"), 0, true),
        (Some("float32x4"), 3, true),
        (Some("float32xvscalex4"), 0, false),
    ] {
        let mut parameters = Vec::new();
        let argument: Expr = if let Some(dtype) = dtype {
            let pointer = PointerType::new(PrimType::void(), "global")?;
            let data = Var::with_type("data", pointer.clone());
            let marker = Var::new("marker", dtype)?;
            let array = Call::new(
                pointer,
                operator("tirx.tvm_stack_make_array")?,
                vec![
                    data.clone().into(),
                    shape.clone().into(),
                    int_expression(0),
                    int_expression(2),
                    marker.clone().into(),
                    int_expression(offset),
                ],
            );
            parameters.extend([data, marker]);
            array.into()
        } else {
            shape.clone().into()
        };
        let body = Evaluate::new(Call::new(
            PrimType::new("int32")?,
            operator("tirx.tvm_call_packed")?,
            vec![StringImm::new("testing.consume").into(), argument],
        ))?;
        let body = AttrStmt::new(0_i64, "device_id", int_expression(0), body)?;
        let function =
            PrimFunc::with_metadata(parameters, body, Type::missing(), attrs.clone(), None)?;
        let module = IRModule::from_expr(function)?;
        let expected = native.run(module.clone());
        let actual = transform::lower_tvm_builtin()?.run(module);
        assert_eq!(
            expected.is_ok(),
            supported,
            "C++: {dtype:?}, offset={offset}"
        );
        assert_eq!(
            actual.is_ok(),
            supported,
            "Rust: {dtype:?}, offset={offset}"
        );
        if supported {
            assert_structural_equal(&actual?, &expected?);
        }
    }
    Ok(())
}

#[test]
fn rust_lower_tvm_builtin_matches_cpp_for_workspace_allocation() {
    load_tvm_compiler();
    let buffer = BufferType::new("local", "int32", vec![int_expression(1024)])
        .unwrap()
        .new_var("workspace");
    let body = AttrStmt::new(
        tvm::tvm_ffi::String::from("default"),
        "device_id",
        IntImm::new("int32", 0).unwrap(),
        Stmt::sequence(vec![
            AllocBuffer::new(buffer).unwrap().into(),
            Evaluate::from_i64(0).unwrap().into(),
        ])
        .unwrap(),
    )
    .unwrap();
    let attrs = DictAttrs::from_dictionary(Map::from_iter([(
        tvm::tvm_ffi::String::from("target"),
        Any::from(tvm::target::Target::new("llvm").unwrap()),
    )]));
    let function = PrimFunc::with_metadata(Vec::new(), body, Type::missing(), attrs, None).unwrap();
    let module = IRModule::from_expr(function.clone()).unwrap();

    let rust_result =
        IRModule::from_expr(transform::lower_tvm_builtin_prim_func(function).unwrap()).unwrap();
    let cpp_result = cpp_pass("tirx.transform.LowerTVMBuiltin")
        .run(module)
        .unwrap();

    assert_structural_equal(&rust_result, &cpp_result);
}

#[test]
fn rust_pointer_value_type_rewrite_matches_cpp_for_buffer_accesses() -> Result<()> {
    load_tvm_compiler();
    let cpp = cpp_pass("tirx.transform.PointerValueTypeRewrite");
    for (scope, scalar_read, allocated) in [
        ("global", false, false),
        ("global", true, false),
        ("local", false, true),
    ] {
        let buffer = BufferType::new(scope, "float32", vec![int_expression(16)])?.new_var("data");
        let ramp = tvm::prim::Ramp::new(
            prim_int_expression(0),
            prim_int_expression(1),
            prim_int_expression(4),
        )?;
        let load = TensorLoad::from_buffer(&buffer, vec![ramp.into()])?;
        let mut statements = Vec::new();
        if allocated {
            statements.push(AllocBuffer::new(&buffer)?.into());
        }
        statements.push(Evaluate::new(load)?.into());
        if scalar_read {
            let scalar = TensorLoad::from_buffer(&buffer, vec![int_expression(1)])?;
            statements.push(Evaluate::new(scalar)?.into());
        }
        let params = if allocated {
            Vec::new()
        } else {
            vec![buffer.as_var().clone()]
        };
        let function = PrimFunc::new(params, Stmt::sequence(statements)?)?;
        let module = IRModule::from_expr(&function)?;
        let rust_result =
            IRModule::from_expr(transform::pointer_value_type_rewrite_prim_func(function)?)?;
        assert_structural_equal(&rust_result, &cpp.run(module)?);
    }
    Ok(())
}

#[test]
fn rust_vectorize_loop_matches_cpp_for_buffer_update() {
    load_tvm_compiler();
    let buffer = BufferType::new("global", "float32", vec![int_expression(16)])
        .unwrap()
        .new_var("data");
    let lane = Var::new("lane", "int32").unwrap();
    let index: Expr = lane.clone().into();
    let load = TensorLoad::from_buffer(buffer.clone(), vec![index.clone()]).unwrap();
    let updated = Add::new(load, FloatImm::new("float32", 1.0).unwrap()).unwrap();
    let store = BufferStore::new(&buffer, updated, vec![index]).unwrap();
    let loop_node = For::with_metadata(
        lane,
        int_expression(0),
        int_expression(4),
        ForKind::kVectorized,
        store.into(),
        None,
        Map::new(),
        None,
        None,
    )
    .unwrap();
    let function = PrimFunc::new(vec![buffer.as_var().clone()], loop_node).unwrap();
    let module = IRModule::from_expr(function.clone()).unwrap();

    let rust_result =
        IRModule::from_expr(transform::vectorize_loop_prim_func(function, true).unwrap()).unwrap();
    let native_pass: transform::Pass = Function::get_global("tirx.transform.VectorizeLoop")
        .unwrap()
        .call_tuple((true,))
        .unwrap()
        .try_into()
        .unwrap();
    let cpp_result = native_pass.run(module).unwrap();

    assert_structural_equal(&rust_result, &cpp_result);
}

#[test]
fn rust_vectorize_loop_matches_cpp_when_disabled() {
    load_tvm_compiler();
    let lane = Var::new("lane", "int32").unwrap();
    let loop_node = For::with_metadata(
        lane,
        int_expression(0),
        int_expression(4),
        ForKind::kVectorized,
        Evaluate::from_i64(0).unwrap().into(),
        None,
        Map::new(),
        None,
        None,
    )
    .unwrap();
    let function = PrimFunc::from_body(loop_node).unwrap();
    let module = IRModule::from_expr(function.clone()).unwrap();

    let rust_result =
        IRModule::from_expr(transform::vectorize_loop_prim_func(function, false).unwrap()).unwrap();
    let native_pass: transform::Pass = Function::get_global("tirx.transform.VectorizeLoop")
        .unwrap()
        .call_tuple((false,))
        .unwrap()
        .try_into()
        .unwrap();
    let cpp_result = native_pass.run(module).unwrap();

    assert_structural_equal(&rust_result, &cpp_result);
}

#[test]
fn rust_storage_rewrite_matches_cpp_across_loop_and_thread_scopes() -> Result<()> {
    load_tvm_compiler();
    for case in ["direct_use", "loop_carried", "parallel", "thread_extent"] {
        let a = BufferType::new("global", "int32", vec![int_expression(16)])?.new_var("a");
        let b = BufferType::new("global", "int32", vec![int_expression(16)])?.new_var("b");
        let output = BufferType::new("global", "int32", vec![int_expression(16)])?.new_var("out");
        let index = Var::new("i", "int32")?;
        let body: Stmt = match case {
            "parallel" => For::with_metadata(
                index.clone(),
                int_expression(0),
                int_expression(16),
                ForKind::kParallel,
                Stmt::sequence(vec![
                    AllocBuffer::new(&a)?.into(),
                    BufferStore::new(&a, int_expression(1), vec![index.into()])?.into(),
                ])?,
                None,
                Map::new(),
                None,
                None,
            )?
            .into(),
            "loop_carried" => Stmt::sequence(vec![
                AllocBuffer::new(&a)?.into(),
                AllocBuffer::new(&b)?.into(),
                BufferStore::new(&a, int_expression(1), vec![int_expression(0)])?.into(),
                For::new(
                    index.clone(),
                    int_expression(0),
                    int_expression(16),
                    Stmt::sequence(vec![
                        BufferStore::new(
                            &output,
                            TensorLoad::from_buffer(&a, vec![int_expression(0)])?,
                            vec![index.into()],
                        )?
                        .into(),
                        BufferStore::new(&b, int_expression(7), vec![int_expression(0)])?.into(),
                    ])?,
                )?
                .into(),
            ])?,
            "direct_use" => Stmt::sequence(vec![
                AllocBuffer::new(&a)?.into(),
                Evaluate::new(Call::new(
                    PrimType::new("void")?,
                    GlobalVar::new("consume"),
                    vec![tvm::ir::Tuple::new(vec![a.into()]).into()],
                ))?
                .into(),
            ])?,
            "thread_extent" => AttrStmt::new(
                IterVar::with_metadata(
                    None,
                    index.clone(),
                    IterVarType::kThreadIndex,
                    "threadIdx.x",
                    None,
                )?,
                "thread_extent",
                int_expression(16),
                Stmt::sequence(vec![
                    AllocBuffer::new(&a)?.into(),
                    BufferStore::new(&a, int_expression(1), vec![index.into()])?.into(),
                ])?,
            )?
            .into(),
            _ => unreachable!(),
        };
        let module = IRModule::from_expr(PrimFunc::new(vec![output.as_var().clone()], body)?)?;
        let native = cpp_pass("tirx.transform.StorageRewrite").run(module.clone())?;
        let rust = transform::storage_rewrite()?.run(module)?;
        assert_structural_equal(&rust, &native);
    }
    Ok(())
}

#[test]
fn rust_storage_rewrite_rejects_cpp_unsupported_inputs() -> Result<()> {
    load_tvm_compiler();
    let buffer = BufferType::new(
        "global",
        "int32",
        vec![int_expression(4), int_expression(4)],
    )?
    .new_var("matrix");
    let multidimensional = Stmt::sequence(vec![
        AllocBuffer::new(&buffer)?.into(),
        BufferStore::new(
            &buffer,
            int_expression(1),
            vec![int_expression(0), int_expression(0)],
        )?
        .into(),
    ])?;
    let vectorized: Stmt = For::with_metadata(
        Var::new("i", "int32")?,
        int_expression(0),
        int_expression(4),
        ForKind::kVectorized,
        Evaluate::from_i64(0)?.into(),
        None,
        Map::new(),
        None,
        None,
    )?
    .into();
    let source = BufferType::new("global", "int32", vec![int_expression(4)])?.new_var("source");
    let alias = source.type_annotation().new_var("alias");
    let unregistered_alias: Stmt = DeclBuffer::new(alias, source.data()?)?.into();
    for body in [multidimensional, vectorized, unregistered_alias] {
        let module = IRModule::from_expr(PrimFunc::from_body(body)?)?;
        assert!(cpp_pass("tirx.transform.StorageRewrite")
            .run(module.clone())
            .is_err());
        assert!(transform::storage_rewrite()?.run(module).is_err());
    }
    Ok(())
}

#[test]
fn rust_storage_rewrite_matches_cpp_for_reuse_policy() -> Result<()> {
    use tvm::tvm_ffi::String;

    load_tvm_compiler();
    let cases = [
        ("global", 16, "int32", None, false),
        ("global", 1, "int32", None, false),
        ("local", 16, "int32", None, false),
        ("warp", 16, "int32", None, false),
        ("wmma.matrix_a", 16, "int32", None, false),
        ("shared.dyn", 16, "int32", None, false),
        ("shared", 16, "int32", None, true),
        ("shared", 16, "float32", Some("vulkan"), false),
        ("shared", 16, "float32", Some("webgpu"), false),
    ];
    for (scope, size, dtype, target, merge_static_smem) in cases {
        let a = BufferType::new(scope, "int32", vec![int_expression(size)])?.new_var("a");
        let b = BufferType::new(scope, dtype, vec![int_expression(size)])?.new_var("b");
        let stored: Expr = if dtype == "float32" {
            FloatImm::new(dtype, 2.0)?.into()
        } else {
            int_expression(2)
        };
        let body = Stmt::sequence(vec![
            AllocBuffer::new(&a)?.into(),
            BufferStore::new(&a, int_expression(1), vec![int_expression(0)])?.into(),
            AllocBuffer::new(&b)?.into(),
            BufferStore::new(&b, stored, vec![int_expression(0)])?.into(),
        ])?;
        let attrs = target.map(tvm::target::Target::new).transpose()?;
        let attrs = DictAttrs::from_dictionary(Map::from_iter(
            attrs
                .into_iter()
                .map(|target| (String::from("target"), Any::from(target))),
        ));
        let module = IRModule::from_expr(PrimFunc::with_metadata(
            Vec::new(),
            body,
            Type::missing(),
            attrs,
            None,
        )?)?;
        let context: transform::PassContext = Function::get_global("transform.PassContext")?
            .call_tuple((
                2_i64,
                Array::<String>::new(Vec::new()),
                Array::<String>::new(Vec::new()),
                Array::<Any>::new(Vec::new()),
                Some(Map::from_iter([(
                    String::from("tirx.merge_static_smem"),
                    Any::from(merge_static_smem),
                )])),
            ))?
            .try_into()?;
        Function::get_global("transform.EnterPassContext")?.call_tuple((&context,))?;
        let native = cpp_pass("tirx.transform.StorageRewrite").run(module.clone());
        let rust = transform::storage_rewrite().and_then(|pass| pass.run(module));
        Function::get_global("transform.ExitPassContext")?.call_tuple((&context,))?;
        assert_structural_equal(&rust?, &native?);
    }
    Ok(())
}

#[test]
fn rust_storage_rewrite_matches_cpp_for_inplace_and_symbolic_reuse() -> Result<()> {
    load_tvm_compiler();
    for case in [
        "inplace",
        "inplace_metadata",
        "shifted",
        "reduction",
        "indirect_read",
        "indirect_write",
        "opaque",
        "loop",
        "extern",
        "volatile_body",
        "symbolic",
        "symbolic_inplace",
        "symbolic_inplace_different_sizes",
    ] {
        let n = Var::new("n", "int32")?;
        let m = Var::new("m", "int32")?;
        let symbolic = case.starts_with("symbolic");
        let a_size = if symbolic {
            n.clone().into()
        } else {
            int_expression(16)
        };
        let b_size = if matches!(case, "symbolic" | "symbolic_inplace_different_sizes") {
            m.clone().into()
        } else {
            a_size.clone()
        };
        let a = BufferType::new("global", "int32", vec![a_size])?.new_var("a");
        let b = BufferType::new("global", "int32", vec![b_size])?.new_var("b");
        let indices =
            BufferType::new("global", "int32", vec![int_expression(16)])?.new_var("indices");
        let read: Expr = TensorLoad::from_buffer(&a, vec![int_expression(0)])?.into();
        let indirect: Expr = TensorLoad::from_buffer(&indices, vec![int_expression(0)])?.into();
        let load = match case {
            "symbolic" => int_expression(7),
            "reduction" => {
                Add::new(read, TensorLoad::from_buffer(&b, vec![int_expression(0)])?)?.into()
            }
            "shifted" => TensorLoad::from_buffer(&a, vec![int_expression(1)])?.into(),
            "indirect_read" => TensorLoad::from_buffer(&a, vec![indirect.clone()])?.into(),
            "opaque" => Call::new(
                PrimType::new("int32")?,
                GlobalVar::new("consume"),
                vec![a.clone().into()],
            )
            .into(),
            _ => read,
        };
        let index = if case == "indirect_write" {
            indirect
        } else {
            int_expression(0)
        };
        let copy: Stmt = BufferStore::new(&b, load, vec![index])?.into();
        let copy = match case {
            "loop" => For::new(
                Var::new("i", "int32")?,
                int_expression(0),
                int_expression(1),
                copy,
            )?
            .into(),
            "extern" => {
                AttrStmt::new(a.as_var().clone(), "extern_scope", int_expression(0), copy)?.into()
            }
            "volatile_body" => {
                let temporary = a.type_annotation().new_var("volatile_tmp");
                let allocation = AllocBuffer::with_metadata(
                    temporary.into(),
                    Map::from_iter([(
                        tvm::tvm_ffi::String::from("tirx.volatile"),
                        Any::from(true),
                    )]),
                    None,
                )?;
                For::new(
                    Var::new("i", "int32")?,
                    int_expression(0),
                    int_expression(1),
                    Stmt::sequence(vec![allocation.into(), copy])?,
                )?
                .into()
            }
            _ => copy,
        };
        let span = Span::new(SourceName::get("storage_rewrite.rs")?, 1, 1, 1, 8)?;
        let allocation = |buffer: &BufferVar| {
            if case == "inplace_metadata" {
                AllocBuffer::with_metadata(
                    buffer.as_var().clone(),
                    Map::from_iter([
                        (tvm::tvm_ffi::String::from("tirx.volatile"), Any::from(true)),
                        (
                            tvm::tvm_ffi::String::from("custom_annotation"),
                            Any::from(1i64),
                        ),
                    ]),
                    Some(&span),
                )
            } else {
                AllocBuffer::new(buffer)
            }
        };
        let body = Stmt::sequence(vec![
            allocation(&a)?.into(),
            allocation(&b)?.into(),
            BufferStore::new(&a, int_expression(1), vec![int_expression(0)])?.into(),
            copy,
            Evaluate::new(TensorLoad::from_buffer(&b, vec![int_expression(0)])?)?.into(),
        ])?;
        let module = IRModule::from_expr(PrimFunc::new(vec![n, m, indices.into()], body)?)?;
        let native = cpp_pass("tirx.transform.StorageRewrite").run(module.clone())?;
        let rust = transform::storage_rewrite()?.run(module)?;
        eprintln!("storage reuse case: {case}");
        assert_structural_equal(&rust, &native);
        if case == "inplace_metadata" {
            // Source spans are not compared by StructuralEqual.
            let mut allocations = 0;
            let mut declarations = 0;
            structural_walk(
                &rust,
                (
                    |allocation: AllocBuffer| {
                        allocations += 1;
                        assert!(allocation.span.is_none());
                        WalkResult::Skip
                    },
                    |declaration: DeclBuffer| {
                        declarations += 1;
                        assert!(declaration.span.is_none());
                        WalkResult::Skip
                    },
                ),
                WalkOrder::PreOrder,
            )?;
            assert_eq!((allocations, declarations), (1, 1));
        }
    }
    Ok(())
}

#[test]
fn rust_storage_rewrite_matches_cpp_for_free_block_selection() -> Result<()> {
    load_tvm_compiler();
    // The first two buffers die together; the third probes size ordering and FIFO.
    for (scope, a_size, b_size, c_size, dtype) in [
        ("global", 64, 16, 16, "int32"),
        ("global", 16, 64, 32, "int32"),
        ("global", 16, 24, 32, "int32"),
        ("global", 2, 1024, 64, "int32"),
        ("global", 64, 16, 16, "float32"),
        ("global", 16, 24, 32, "float32"),
        ("global", 0, 0, 0, "int32"),
        ("global", 0, 0, 0, "float32"),
        ("local.L0A", 64, 16, 16, "int32"),
        ("local.L0A", 16, 24, 32, "int32"),
        ("shared.dyn", 0, 0, 0, "int32"),
    ] {
        let n = Var::new("n", "int32")?;
        let size = |value| {
            if value == 0 {
                Expr::from(n.clone())
            } else {
                int_expression(value)
            }
        };
        let a = BufferType::new(scope, "int32", vec![size(a_size)])?.new_var("a");
        let b = BufferType::new(scope, "int32", vec![size(b_size)])?.new_var("b");
        let c = BufferType::new(scope, dtype, vec![size(c_size)])?.new_var("c");
        let consume = |args| {
            Evaluate::new(Call::new(
                PrimType::new("void")?,
                GlobalVar::new("consume"),
                args,
            ))
        };
        let body = Stmt::sequence(vec![
            AllocBuffer::new(&a)?.into(),
            AllocBuffer::new(&b)?.into(),
            AllocBuffer::new(&c)?.into(),
            consume(vec![
                TensorLoad::from_buffer(&a, vec![int_expression(0)])?.into(),
                TensorLoad::from_buffer(&b, vec![int_expression(0)])?.into(),
            ])?
            .into(),
            consume(vec![
                TensorLoad::from_buffer(&c, vec![int_expression(0)])?.into()
            ])?
            .into(),
        ])?;
        let module = IRModule::from_expr(PrimFunc::new(vec![n], body)?)?;
        let native = cpp_pass("tirx.transform.StorageRewrite").run(module.clone())?;
        let rust = transform::storage_rewrite()?.run(module)?;
        eprintln!("free block case: {scope}, {a_size}, {b_size}, {c_size}, {dtype}");
        assert_structural_equal(&rust, &native);
    }
    Ok(())
}

#[test]
fn rust_storage_rewrite_matches_cpp_for_intrinsic_accesses() -> Result<()> {
    use tvm::ir::Op;

    load_tvm_compiler();
    for (scope, masked) in [
        ("global", true),
        ("local.L0A", true),
        ("global", false),
        ("local.L0A", false),
    ] {
        let a = BufferType::new(scope, "int32", vec![int_expression(16)])?.new_var("a");
        let b = a.type_annotation().new_var("b");
        let load = |buffer: &BufferVar| -> Result<Expr> {
            if masked {
                Ok(Call::new(
                    PrimType::new("int32")?,
                    Op::get("tirx.masked_load")?,
                    vec![
                        buffer.as_var().clone().into(),
                        int_expression(0),
                        typed_int_expression("bool", 1),
                    ],
                )
                .into())
            } else {
                Ok(Call::new(
                    PointerType::new(PrimType::new("int32")?, scope)?,
                    Op::get("tirx.tvm_access_ptr")?,
                    vec![
                        int_expression(0),
                        buffer.data()?,
                        int_expression(0),
                        int_expression(16),
                        int_expression(1),
                    ],
                )
                .into())
            }
        };
        let mut statements = vec![
            AllocBuffer::new(&a)?.into(),
            AllocBuffer::new(&b)?.into(),
            Evaluate::new(Call::new(
                PrimType::void(),
                GlobalVar::new("consume"),
                vec![load(&a)?, load(&b)?],
            ))?
            .into(),
        ];
        if masked {
            statements.push(
                Evaluate::new(Call::new(
                    PrimType::void(),
                    Op::get("tirx.masked_store")?,
                    vec![
                        b.into(),
                        int_expression(3),
                        int_expression(0),
                        typed_int_expression("bool", 1),
                    ],
                ))?
                .into(),
            );
        }
        let module = IRModule::from_expr(PrimFunc::from_body(Stmt::sequence(statements)?)?)?;
        let native = cpp_pass("tirx.transform.StorageRewrite").run(module.clone())?;
        let rust = transform::storage_rewrite()?.run(module)?;
        eprintln!("intrinsic case: {scope}, masked={masked}");
        assert_structural_equal(&rust, &native);
    }
    Ok(())
}

#[test]
fn rust_storage_rewrite_matches_cpp_for_sequential_tagged_allocations() {
    load_tvm_compiler();
    let buffer_a = BufferType::new("local.L0A", "float32", vec![int_expression(200)])
        .unwrap()
        .new_var("a");
    let buffer_b = BufferType::new("local.L0A", "float32", vec![int_expression(200)])
        .unwrap()
        .new_var("b");
    let lane_a = Var::new("i", "int32").unwrap();
    let lane_b = Var::new("j", "int32").unwrap();
    let store_a = BufferStore::new(
        &buffer_a,
        FloatImm::new("float32", 1.2).unwrap(),
        vec![Expr::from(lane_a.clone())],
    )
    .unwrap();
    let store_b = BufferStore::new(
        &buffer_b,
        FloatImm::new("float32", 1.3).unwrap(),
        vec![Expr::from(lane_b.clone())],
    )
    .unwrap();
    let loop_a = For::new(lane_a, int_expression(0), int_expression(10), store_a).unwrap();
    let loop_b = For::new(lane_b, int_expression(0), int_expression(10), store_b).unwrap();
    let body = Stmt::sequence(vec![
        AllocBuffer::new(buffer_a).unwrap().into(),
        loop_a.into(),
        AllocBuffer::new(buffer_b).unwrap().into(),
        loop_b.into(),
    ])
    .unwrap();
    let function = PrimFunc::from_body(body).unwrap();
    let module = IRModule::from_expr(function.clone()).unwrap();

    let rust_result =
        IRModule::from_expr(transform::storage_rewrite_prim_func(function).unwrap()).unwrap();
    let cpp_result = cpp_pass("tirx.transform.StorageRewrite")
        .run(module)
        .unwrap();

    assert_structural_equal(&rust_result, &cpp_result);
}

#[test]
fn rust_make_packed_api_matches_cpp_for_scalar_arguments_and_return() {
    load_tvm_compiler();
    let host = tvm::target::Target::new("llvm").unwrap();
    let target = tvm::target::Target::new("cuda")
        .unwrap()
        .with_host(&host)
        .unwrap();
    let value = Var::new("value", "int32").unwrap();
    let attrs = DictAttrs::from_dictionary(Map::from_iter([
        (
            tvm::tvm_ffi::String::from("global_symbol"),
            Any::from(tvm::tvm_ffi::String::from("add_one")),
        ),
        (tvm::tvm_ffi::String::from("target"), Any::from(target)),
    ]));
    let function = PrimFunc::with_metadata(
        vec![value.clone()],
        Return::new(Add::new(value, int_expression(1)).unwrap()),
        PrimType::new("int32").unwrap(),
        attrs.clone(),
        None,
    )
    .unwrap();
    let module = IRModule::from_expr(function).unwrap();

    let rust_result = transform::make_packed_api_module(module.clone()).unwrap();
    let cpp_result = cpp_pass("tirx.transform.MakePackedAPI")
        .run(module)
        .unwrap();

    assert_structural_equal(&rust_result, &cpp_result);

    let native = cpp_pass("tirx.transform.MakePackedAPI");
    for (dtype, bound, supported) in [
        ("int32", true, true),
        ("bool", true, true),
        ("int32", false, false),
        ("int32x4", true, false),
        ("float32x4", true, false),
        ("boolx4", true, false),
        ("int32xvscalex4", true, false),
    ] {
        let value = Var::new("value", dtype).unwrap();
        let params = if bound {
            vec![value.clone()]
        } else {
            Vec::new()
        };
        let function = PrimFunc::with_metadata(
            params,
            Evaluate::new(value).unwrap(),
            Type::missing(),
            attrs.clone(),
            None,
        )
        .unwrap();
        let module = IRModule::from_expr(function).unwrap();
        let expected = native.run(module.clone());
        let actual = transform::make_packed_api_module(module);
        assert_eq!(expected.is_ok(), supported, "C++: {dtype}, bound={bound}");
        assert_eq!(actual.is_ok(), supported, "Rust: {dtype}, bound={bound}");
        if supported {
            assert_structural_equal(&actual.unwrap(), &expected.unwrap());
        } else if !bound {
            assert!(actual
                .err()
                .unwrap()
                .to_string()
                .contains("not passed in as API arguments"));
        }
    }
}

#[test]
fn rust_make_packed_api_matches_cpp_for_buffer_argument() {
    load_tvm_compiler();
    let host = tvm::target::Target::new("llvm").unwrap();
    let target = tvm::target::Target::new("cuda")
        .unwrap()
        .with_host(&host)
        .unwrap();
    let extent = Var::new("n", "int64").unwrap();
    let buffer = BufferType::new(
        "global",
        "float32",
        vec![Expr::from(extent), typed_int_expression("int64", 4)],
    )
    .unwrap()
    .new_var("buffer");
    let attrs = DictAttrs::from_dictionary(Map::from_iter([
        (
            tvm::tvm_ffi::String::from("global_symbol"),
            Any::from(tvm::tvm_ffi::String::from("read_buffer")),
        ),
        (tvm::tvm_ffi::String::from("target"), Any::from(target)),
    ]));
    let load = TensorLoad::from_buffer(
        buffer.as_var().clone(),
        vec![
            typed_int_expression("int64", 0),
            typed_int_expression("int64", 0),
        ],
    )
    .unwrap();
    let function = PrimFunc::with_metadata(
        vec![buffer.as_var().clone()],
        Return::new(load),
        PrimType::new("float32").unwrap(),
        attrs.clone(),
        None,
    )
    .unwrap();
    let module = IRModule::from_expr(function).unwrap();

    let rust_result = transform::make_packed_api_module(module.clone()).unwrap();
    let cpp_result = cpp_pass("tirx.transform.MakePackedAPI")
        .run(module)
        .unwrap();

    assert_structural_equal(&rust_result, &cpp_result);
    for (dtype, symbolic_shape, aligned_offset, supported) in [
        ("float32x4", false, false, true),
        ("float32xvscalex4", false, false, false),
        ("int32", false, true, true),
        ("uint1", false, false, true),
        ("uint1", true, true, true),
        ("int4x4", false, false, true),
        ("uint4x4", false, false, true),
        ("int1x4", false, false, true),
        ("uint1x4", false, false, true),
        ("int4", false, false, true),
        ("int4", true, false, false),
        ("uint4", true, false, false),
        ("int1", true, false, false),
    ] {
        let shape = if symbolic_shape {
            Var::new("n", "int64").unwrap().into()
        } else {
            typed_int_expression("int64", 16)
        };
        let offset = if aligned_offset {
            Var::new("offset", "int64").unwrap().into()
        } else {
            typed_int_expression("int64", 0)
        };
        let buffer = BufferType::with_metadata(
            "global",
            PrimType::new(dtype).unwrap(),
            vec![shape],
            Vec::new(),
            offset,
            64,
            if aligned_offset { 4 } else { 1 },
            None,
            Vec::new(),
            None,
        )
        .unwrap()
        .new_var("buffer");
        let function = PrimFunc::with_metadata(
            vec![buffer.as_var().clone()],
            Evaluate::from_i64(0).unwrap(),
            Type::missing(),
            attrs.clone(),
            None,
        )
        .unwrap();
        let module = IRModule::from_expr(function).unwrap();
        let native = cpp_pass("tirx.transform.MakePackedAPI").run(module.clone());
        let rust = transform::make_packed_api_module(module);
        assert_eq!(native.is_ok(), supported, "C++: {dtype}");
        assert_eq!(rust.is_ok(), supported, "Rust: {dtype}");
        if supported {
            assert_structural_equal(&rust.unwrap(), &native.unwrap());
        }
    }
}

#[test]
fn rust_split_host_device_matches_cpp_for_cuda_thread_extent() {
    load_tvm_compiler();
    let host = tvm::target::Target::new("llvm").unwrap();
    let target = tvm::target::Target::new("cuda")
        .unwrap()
        .with_host(&host)
        .unwrap();
    let thread_var = Var::new("threadIdx.x", "int32").unwrap();
    let thread = IterVar::with_metadata(
        None,
        thread_var,
        IterVarType::kThreadIndex,
        "threadIdx.x",
        None,
    )
    .unwrap();
    let body = AttrStmt::new(
        thread,
        "thread_extent",
        int_expression(32),
        Evaluate::from_i64(0).unwrap(),
    )
    .unwrap();
    let attrs = DictAttrs::from_dictionary(Map::from_iter([
        (
            tvm::tvm_ffi::String::from("global_symbol"),
            Any::from(tvm::tvm_ffi::String::from("main.with.dots")),
        ),
        (tvm::tvm_ffi::String::from("target"), Any::from(target)),
    ]));
    let function = PrimFunc::with_metadata(Vec::new(), body, Type::missing(), attrs, None).unwrap();
    let module = IRModule::from_expr(function).unwrap();

    let rust_result = transform::split_host_device_module(module.clone()).unwrap();
    let cpp_result = cpp_pass("tirx.transform.SplitHostDevice")
        .run(module)
        .unwrap();

    assert_structural_equal(&rust_result, &cpp_result);
    for result in [&rust_result, &cpp_result] {
        assert!(result
            .functions
            .iter()
            .any(|(global, _)| global.name_hint.as_str() == "main_with_dots_kernel"));
    }
}

#[test]
fn rust_split_host_device_matches_cpp_for_dynamic_shared_memory() -> Result<()> {
    load_tvm_compiler();
    let host = tvm::target::Target::new("llvm")?;
    let target = tvm::target::Target::new(r#"{"kind":"cuda","arch":"sm_80"}"#)?.with_host(&host)?;
    let attrs = DictAttrs::from_dictionary(Map::from_iter([("target".into(), target.into())]));
    let native = cpp_pass("tirx.transform.SplitHostDevice");
    for (index_type, dtype, extent, declared_bytes, supported) in [
        ("int32", "float32", 16, None, true),
        ("int64", "float32", 16, None, true),
        ("int32", "float32", 0, None, false),
        ("int32", "float32", 0, Some(128), true),
        ("int32", "float32xvscalex4", 16, None, false),
    ] {
        let buffer = BufferType::new(
            "shared.dyn",
            dtype,
            vec![IntImm::new(index_type, extent)?.into()],
        )?
        .new_var("shared");
        let mut body: Stmt = AllocBuffer::new(buffer)?.into();
        if let Some(bytes) = declared_bytes {
            body = AttrStmt::new(
                0_i64,
                "tirx.dyn_smem_bytes",
                IntImm::new("int64", bytes)?,
                body,
            )?
            .into();
        }
        let thread = IterVar::with_metadata(
            None,
            Var::new("threadIdx.x", "int32")?,
            IterVarType::kThreadIndex,
            "threadIdx.x",
            None,
        )?;
        let body = AttrStmt::new(thread, "thread_extent", int_expression(32), body)?;
        let function =
            PrimFunc::with_metadata(Vec::new(), body, Type::missing(), attrs.clone(), None)?;
        let module = IRModule::from_expr(function)?;
        let expected = native.run(module.clone());
        let actual = transform::split_host_device_module(module);
        assert_eq!(
            expected.is_ok(),
            supported,
            "C++: {index_type}, {dtype}, {extent}"
        );
        assert_eq!(
            actual.is_ok(),
            supported,
            "Rust: {index_type}, {dtype}, {extent}"
        );
        if supported {
            assert_structural_equal(&actual?, &expected?);
        }
    }
    Ok(())
}

#[test]
fn rust_split_host_device_matches_cpp_for_buffer_capture() {
    load_tvm_compiler();
    let host = tvm::target::Target::new("llvm").unwrap();
    let target = tvm::target::Target::new("cuda")
        .unwrap()
        .with_host(&host)
        .unwrap();
    let buffer = BufferType::new("global", "int32", vec![int_expression(16)])
        .unwrap()
        .new_var("buffer");
    let thread_var = Var::new("threadIdx.x", "int32").unwrap();
    let thread = IterVar::with_metadata(
        None,
        thread_var.clone(),
        IterVarType::kThreadIndex,
        "threadIdx.x",
        None,
    )
    .unwrap();
    let store = BufferStore::new(&buffer, int_expression(1), vec![Expr::from(thread_var)]).unwrap();
    let body = AttrStmt::new(thread, "thread_extent", int_expression(16), store).unwrap();
    let attrs = DictAttrs::from_dictionary(Map::from_iter([
        (
            tvm::tvm_ffi::String::from("global_symbol"),
            Any::from(tvm::tvm_ffi::String::from("write_buffer")),
        ),
        (tvm::tvm_ffi::String::from("target"), Any::from(target)),
    ]));
    let function = PrimFunc::with_metadata(
        vec![buffer.as_var().clone()],
        body,
        Type::missing(),
        attrs,
        None,
    )
    .unwrap();
    let module = IRModule::from_expr(function).unwrap();

    let rust_result = transform::split_host_device_module(module.clone()).unwrap();
    let cpp_result = cpp_pass("tirx.transform.SplitHostDevice")
        .run(module)
        .unwrap();

    assert_structural_equal(&rust_result, &cpp_result);
}

#[test]
fn rust_split_host_device_matches_cpp_for_cpu_device_scope() {
    load_tvm_compiler();
    let host = tvm::target::Target::new("llvm").unwrap();
    let target = tvm::target::Target::new("c")
        .unwrap()
        .with_host(&host)
        .unwrap();
    let body = AttrStmt::new(
        0_i64,
        "device_scope",
        int_expression(0),
        Evaluate::from_i64(0).unwrap(),
    )
    .unwrap();
    let attrs = DictAttrs::from_dictionary(Map::from_iter([
        (
            tvm::tvm_ffi::String::from("global_symbol"),
            Any::from(tvm::tvm_ffi::String::from("host_compute")),
        ),
        (tvm::tvm_ffi::String::from("target"), Any::from(target)),
    ]));
    let function = PrimFunc::with_metadata(Vec::new(), body, Type::missing(), attrs, None).unwrap();
    let module = IRModule::from_expr(function).unwrap();

    let rust_result = transform::split_host_device_module(module.clone()).unwrap();
    let cpp_result = cpp_pass("tirx.transform.SplitHostDevice")
        .run(module)
        .unwrap();

    assert_structural_equal(&rust_result, &cpp_result);
}

#[test]
fn rust_tile_primitive_dispatch_matches_cpp_for_scope_ids() {
    load_tvm_compiler();
    let block_x: tvm::tirx::PrimVar = Var::new("bx", "int32").unwrap().try_into().unwrap();
    let block_y: tvm::tirx::PrimVar = Var::new("by", "int32").unwrap().try_into().unwrap();
    let block_z: tvm::tirx::PrimVar = Var::new("bz", "int32").unwrap().try_into().unwrap();
    let lane: tvm::tirx::PrimVar = Var::new("lane", "int32").unwrap().try_into().unwrap();
    let blocks = ScopeIdDef::new(
        vec![block_x, block_y, block_z],
        Some(vec![
            prim_int_expression(1),
            prim_int_expression(1),
            prim_int_expression(1),
        ]),
        ScopeBinding::KERNEL_CTA,
        None,
    )
    .unwrap();
    let threads = ScopeIdDef::new(
        vec![lane.clone()],
        Some(vec![prim_int_expression(32)]),
        ScopeBinding::CTA_THREAD,
        None,
    )
    .unwrap();
    let body = Stmt::sequence(vec![
        ScopeIdDefStmt::new(blocks, None).into(),
        ScopeIdDefStmt::new(threads, None).into(),
        Evaluate::new(lane).unwrap().into(),
    ])
    .unwrap();
    let body = AttrStmt::new(0_i64, "tirx.device_entry", int_expression(1), body).unwrap();
    let attrs = DictAttrs::from_dictionary(Map::from_iter([(
        tvm::tvm_ffi::String::from("target"),
        Any::from(tvm::target::Target::new("cuda").unwrap()),
    )]));
    let function = PrimFunc::with_metadata(Vec::new(), body, Type::missing(), attrs, None).unwrap();
    let module = IRModule::from_expr(&function).unwrap();
    let lower_input = module.clone();

    let rust_function = transform::tile_primitive_dispatch_prim_func(function).unwrap();
    let rust_result = IRModule::from_expr(rust_function).unwrap();
    let cpp_result = cpp_pass("tirx.transform.TilePrimitiveDispatch")
        .run(module)
        .unwrap();

    assert_structural_equal(&rust_result, &cpp_result);

    let rust_lowered = transform::lower_tirx()
        .unwrap()
        .run(lower_input.clone())
        .unwrap();
    let cpp_lowered = cpp_pass("tirx.transform.LowerTIRx")
        .run(lower_input)
        .unwrap();
    assert_structural_equal(&rust_lowered, &cpp_lowered);
}

#[test]
fn rust_tile_primitive_dispatch_matches_cpp_for_registered_dispatcher() {
    load_tvm_compiler();
    Function::register_global(
        "tirx.f_op_dispatcher",
        Function::from_typed(
            |_call: TilePrimitiveCall, context: DispatchContext| -> Result<PrimFunc> {
                let lane = context
                    .inter
                    .get(&tvm::tvm_ffi::String::from("laneid"))?
                    .expect("thread dispatch exposes laneid");
                PrimFunc::from_body(Evaluate::new(lane.get(1)?)?)
            },
        ),
    )
    .unwrap();
    let operator = tvm::ir::Op::get("tirx.tile.zero").unwrap();
    let call = TilePrimitiveCall::new(
        operator,
        Vec::new(),
        Map::new(),
        Map::new(),
        None,
        ExecScope::new(ScopeKind::THREAD).unwrap(),
    )
    .unwrap();
    let block: tvm::tirx::PrimVar = Var::new("block", "int32").unwrap().try_into().unwrap();
    let lane: tvm::tirx::PrimVar = Var::new("lane", "int32").unwrap().try_into().unwrap();
    let blocks = ScopeIdDef::new(
        vec![block],
        Some(vec![prim_int_expression(1)]),
        ScopeBinding::KERNEL_CTA,
        None,
    )
    .unwrap();
    let threads = ScopeIdDef::new(
        vec![lane.clone()],
        Some(vec![prim_int_expression(32)]),
        ScopeBinding::CTA_THREAD,
        None,
    )
    .unwrap();
    let filtered_call =
        IfThenElse::new(EQ::new(lane, prim_int_expression(3)).unwrap(), call).unwrap();
    let body = Stmt::sequence(vec![
        ScopeIdDefStmt::new(blocks, None).into(),
        ScopeIdDefStmt::new(threads, None).into(),
        filtered_call.into(),
    ])
    .unwrap();
    let body = AttrStmt::new(0_i64, "tirx.device_entry", int_expression(1), body).unwrap();
    let attrs = DictAttrs::from_dictionary(Map::from_iter([(
        tvm::tvm_ffi::String::from("target"),
        Any::from(tvm::target::Target::new("cuda").unwrap()),
    )]));
    let function = PrimFunc::with_metadata(Vec::new(), body, Type::missing(), attrs, None).unwrap();
    let module = IRModule::from_expr(&function).unwrap();

    let rust_function = transform::tile_primitive_dispatch_prim_func(function).unwrap();
    let rust_result = IRModule::from_expr(rust_function).unwrap();
    let cpp_result = cpp_pass("tirx.transform.TilePrimitiveDispatch")
        .run(module)
        .unwrap();

    assert_structural_equal(&rust_result, &cpp_result);
}
