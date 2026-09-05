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
fn rust_lower_tvm_builtin_matches_cpp_for_shape_stack() {
    load_tvm_compiler();
    let operator = |name: &str| -> Expr { tvm::ir::Op::get(name).unwrap().into() };
    let shape = Call::new(
        PointerType::new(PrimType::new("int64").unwrap(), "global").unwrap(),
        operator("tirx.tvm_stack_make_shape"),
        vec![int_expression(4), int_expression(8)],
    );
    let body = Evaluate::new(Call::new(
        PrimType::new("int32").unwrap(),
        operator("tirx.tvm_call_packed"),
        vec![StringImm::new("testing.consume_shape").into(), shape.into()],
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
fn rust_pointer_value_type_rewrite_matches_cpp_for_vector_buffer_load() {
    load_tvm_compiler();
    let buffer = BufferType::new("global", "float32", vec![int_expression(16)])
        .unwrap()
        .new_var("data");
    let ramp: PrimExpr = Function::get_global("ir.prim.Ramp")
        .unwrap()
        .call_tuple((
            prim_int_expression(0),
            prim_int_expression(1),
            prim_int_expression(4),
            Option::<Span>::None,
        ))
        .unwrap()
        .try_into()
        .unwrap();
    let load = TensorLoad::from_buffer(buffer.clone(), vec![ramp.into()]).unwrap();
    let function =
        PrimFunc::new(vec![buffer.as_var().clone()], Evaluate::new(load).unwrap()).unwrap();
    let module = IRModule::from_expr(function.clone()).unwrap();

    let rust_result =
        IRModule::from_expr(transform::pointer_value_type_rewrite_prim_func(function).unwrap())
            .unwrap();
    let cpp_result = cpp_pass("tirx.transform.PointerValueTypeRewrite")
        .run(module)
        .unwrap();

    assert_structural_equal(&rust_result, &cpp_result);
}

#[test]
fn rust_pointer_value_type_rewrite_matches_cpp_for_scalar_shuffle_read() {
    load_tvm_compiler();
    let buffer = BufferType::new("global", "float32", vec![int_expression(16)])
        .unwrap()
        .new_var("data");
    let ramp: PrimExpr = Function::get_global("ir.prim.Ramp")
        .unwrap()
        .call_tuple((
            prim_int_expression(0),
            prim_int_expression(1),
            prim_int_expression(4),
            Option::<Span>::None,
        ))
        .unwrap()
        .try_into()
        .unwrap();
    let vector_load = TensorLoad::from_buffer(buffer.clone(), vec![ramp.into()]).unwrap();
    let scalar_load = TensorLoad::from_buffer(buffer.clone(), vec![int_expression(1)]).unwrap();
    let body = Stmt::sequence(vec![
        Evaluate::new(vector_load).unwrap().into(),
        Evaluate::new(scalar_load).unwrap().into(),
    ])
    .unwrap();
    let function = PrimFunc::new(vec![buffer.as_var().clone()], body).unwrap();
    let module = IRModule::from_expr(function.clone()).unwrap();

    let rust_result =
        IRModule::from_expr(transform::pointer_value_type_rewrite_prim_func(function).unwrap())
            .unwrap();
    let cpp_result = cpp_pass("tirx.transform.PointerValueTypeRewrite")
        .run(module)
        .unwrap();

    assert_structural_equal(&rust_result, &cpp_result);
}

#[test]
fn rust_pointer_value_type_rewrite_matches_cpp_for_allocated_buffer() {
    load_tvm_compiler();
    let buffer = BufferType::new("local", "float32", vec![int_expression(16)])
        .unwrap()
        .new_var("temporary");
    let ramp: PrimExpr = Function::get_global("ir.prim.Ramp")
        .unwrap()
        .call_tuple((
            prim_int_expression(0),
            prim_int_expression(1),
            prim_int_expression(4),
            Option::<Span>::None,
        ))
        .unwrap()
        .try_into()
        .unwrap();
    let load = TensorLoad::from_buffer(buffer.clone(), vec![ramp.into()]).unwrap();
    let body = Stmt::sequence(vec![
        AllocBuffer::new(buffer).unwrap().into(),
        Evaluate::new(load).unwrap().into(),
    ])
    .unwrap();
    let function = PrimFunc::from_body(body).unwrap();
    let module = IRModule::from_expr(function.clone()).unwrap();

    let rust_result =
        IRModule::from_expr(transform::pointer_value_type_rewrite_prim_func(function).unwrap())
            .unwrap();
    let cpp_result = cpp_pass("tirx.transform.PointerValueTypeRewrite")
        .run(module)
        .unwrap();

    assert_structural_equal(&rust_result, &cpp_result);
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
        attrs,
        None,
    )
    .unwrap();
    let module = IRModule::from_expr(function).unwrap();

    let rust_result = transform::make_packed_api_module(module.clone()).unwrap();
    let cpp_result = cpp_pass("tirx.transform.MakePackedAPI")
        .run(module)
        .unwrap();

    assert_structural_equal(&rust_result, &cpp_result);
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
        attrs,
        None,
    )
    .unwrap();
    let module = IRModule::from_expr(function).unwrap();

    let rust_result = transform::make_packed_api_module(module.clone()).unwrap();
    let cpp_result = cpp_pass("tirx.transform.MakePackedAPI")
        .run(module)
        .unwrap();

    assert_structural_equal(&rust_result, &cpp_result);
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
