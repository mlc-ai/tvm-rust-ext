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
fn rust_remove_no_op_matches_cpp_on_non_sblock_control_and_effects() {
    load_tvm_compiler();
    let condition = Var::new("condition", "bool").unwrap();
    let effect_operator: Expr = GlobalVar::new("effect").into();
    let effect: Expr =
        Call::new(PrimType::new("int32").unwrap(), effect_operator, Vec::new()).into();
    let conditional: Stmt = IfThenElse::with_span(
        condition.clone(),
        Evaluate::from_i64(0).unwrap(),
        Some(Evaluate::new(effect.clone()).unwrap().into()),
        None,
    )
    .unwrap()
    .into();
    let loop_var = Var::new("i", "int32").unwrap();
    let empty_loop: Stmt = For::new(
        loop_var,
        int_expression(0),
        int_expression(0),
        Evaluate::new(effect.clone()).unwrap(),
    )
    .unwrap()
    .into();
    let body = SeqStmt::new(vec![
        Evaluate::new(Add::new(int_expression(1), int_expression(2)).unwrap())
            .unwrap()
            .into(),
        conditional,
        empty_loop,
        Evaluate::new(effect).unwrap().into(),
    ])
    .unwrap();
    let function = PrimFunc::new(vec![condition], body).unwrap();
    let module = IRModule::from_expr(&function).unwrap();

    let rust_function = transform::remove_no_op_prim_func(function).unwrap();
    let rust_result = IRModule::from_expr(&rust_function).unwrap();
    let cpp_result = cpp_pass("tirx.transform.RemoveNoOp").run(module).unwrap();

    assert_structural_equal(&rust_result, &cpp_result);
}

#[test]
fn rust_remove_no_op_selects_constant_branches_and_preserves_effect_spans() {
    load_tvm_compiler();
    let span = Span::new(SourceName::get("remove_no_op.rs").unwrap(), 7, 7, 3, 19).unwrap();
    let effect: Expr = Call::new(
        PrimType::new("int32").unwrap(),
        GlobalVar::new("effect"),
        Vec::new(),
    )
    .into();
    let effect_statement = Evaluate::with_span(effect, Some(&span)).unwrap();
    let effect_pointer = object_pointer(&effect_statement);
    let conditional = IfThenElse::with_span(
        typed_int_expression("bool", 1),
        effect_statement,
        Some(Evaluate::from_i64(0).unwrap().into()),
        None,
    )
    .unwrap();
    let function = PrimFunc::from_body(conditional).unwrap();
    let module = IRModule::from_expr(&function).unwrap();

    let rust_function = transform::remove_no_op_prim_func(function).unwrap();
    let rust_evaluate = rust_function.body.clone().try_cast::<Evaluate>().unwrap();
    let cpp_result = cpp_pass("tirx.transform.RemoveNoOp").run(module).unwrap();

    assert_eq!(object_pointer(&rust_evaluate), effect_pointer);
    assert_eq!(rust_evaluate.span.as_ref().unwrap().line, 7);
    assert_structural_equal(&IRModule::from_expr(rust_function).unwrap(), &cpp_result);
}

#[test]
fn rust_remove_no_op_matches_cpp_on_redundant_buffer_store() {
    load_tvm_compiler();
    let buffer_type =
        BufferType::new("global", "int32", vec![typed_int_expression("int64", 16)]).unwrap();
    let buffer = buffer_type.new_var("buffer");
    let index = Var::new("index", "int64").unwrap();
    let load = TensorLoad::from_buffer(&buffer, vec![index.clone().into()]).unwrap();
    let store = BufferStore::new(&buffer, load, vec![index.clone().into()]).unwrap();
    let function = PrimFunc::new(vec![buffer.as_var().clone(), index], store).unwrap();
    let module = IRModule::from_expr(&function).unwrap();

    let rust_result =
        IRModule::from_expr(transform::remove_no_op_prim_func(function).unwrap()).unwrap();
    let cpp_result = cpp_pass("tirx.transform.RemoveNoOp").run(module).unwrap();

    assert_structural_equal(&rust_result, &cpp_result);
}

#[test]
fn rust_stmt_simplify_matches_cpp_for_flat_bind_conditions() {
    load_tvm_compiler();
    let variable = Var::new("condition", "int32").unwrap();
    let condition = EQ::new(variable.clone(), int_expression(1)).unwrap();
    let body = SeqStmt::new(vec![
        Bind::new(variable, int_expression(1)).unwrap().into(),
        IfThenElse::with_span(
            condition,
            Evaluate::from_i64(2).unwrap(),
            Some(Evaluate::from_i64(3).unwrap().into()),
            None,
        )
        .unwrap()
        .into(),
    ])
    .unwrap();
    let function = PrimFunc::from_body(body).unwrap();
    let module = IRModule::from_expr(&function).unwrap();

    let rust_result =
        IRModule::from_expr(transform::stmt_simplify_prim_func(function).unwrap()).unwrap();
    let cpp_result = cpp_pass("tirx.transform.StmtSimplify").run(module).unwrap();

    assert_structural_equal(&rust_result, &cpp_result);
}

#[test]
fn rust_stmt_simplify_matches_cpp_for_loop_constraints_and_redundant_store() {
    load_tvm_compiler();
    let buffer_type =
        BufferType::new("global", "int32", vec![typed_int_expression("int32", 4)]).unwrap();
    let buffer = buffer_type.new_var("buffer");
    let index = Var::new("index", "int32").unwrap();
    let load = TensorLoad::from_buffer(&buffer, vec![index.clone().into()]).unwrap();
    let redundant_store = BufferStore::new(&buffer, load, vec![index.clone().into()]).unwrap();
    let conditional = IfThenElse::with_span(
        LT::new(index.clone(), int_expression(4)).unwrap(),
        redundant_store,
        Some(Evaluate::from_i64(7).unwrap().into()),
        None,
    )
    .unwrap();
    let body = For::new(index, int_expression(0), int_expression(4), conditional).unwrap();
    let function = PrimFunc::new(vec![buffer.as_var().clone()], body).unwrap();
    let module = IRModule::from_expr(&function).unwrap();

    let rust_result =
        IRModule::from_expr(transform::stmt_simplify_prim_func(function).unwrap()).unwrap();
    let cpp_result = cpp_pass("tirx.transform.StmtSimplify").run(module).unwrap();

    assert_structural_equal(&rust_result, &cpp_result);
}

#[test]
fn rust_stmt_simplify_matches_cpp_for_scope_definition_extents() {
    load_tvm_compiler();
    let extent_variable = Var::new("extent", "int32").unwrap();
    let scope_id: PrimVar = Var::new("lane", "int32").unwrap().try_into().unwrap();
    let redundant_zero = prim_int_expression(0);
    let extent: PrimExpr = Add::new(extent_variable.clone(), redundant_zero.clone())
        .unwrap()
        .into();
    let preferred: PrimExpr = Add::new(extent_variable.clone(), redundant_zero)
        .unwrap()
        .into();
    let definition = ScopeIdDef::new(
        vec![scope_id],
        Some(vec![extent]),
        ScopeBinding::CTA_THREAD,
        Some(vec![preferred]),
    )
    .unwrap();
    let body = ScopeIdDefStmt::new(definition, None);
    let function = PrimFunc::new(vec![extent_variable], body).unwrap();
    let module = IRModule::from_expr(&function).unwrap();

    let rust_result =
        IRModule::from_expr(transform::stmt_simplify_prim_func(function).unwrap()).unwrap();
    let cpp_result = cpp_pass("tirx.transform.StmtSimplify").run(module).unwrap();

    assert_structural_equal(&rust_result, &cpp_result);
}

#[test]
fn rust_stmt_simplify_matches_cpp_inside_tile_primitive_values() {
    load_tvm_compiler();
    let variable = Var::new("value", "int32").unwrap();
    let redundant: PrimExpr = Add::new(variable.clone(), prim_int_expression(0))
        .unwrap()
        .into();
    let operator = tvm::ir::Op::get("tirx.tile.zero").unwrap();
    let call = TilePrimitiveCall::new(
        operator,
        vec![
            Any::from(redundant.clone()),
            Any::from(Array::new(vec![Any::from(redundant.clone())])),
        ],
        Map::new(),
        Map::from_iter([(tvm::tvm_ffi::String::from("value"), Any::from(redundant))]),
        None,
        ExecScope::new(ScopeKind::THREAD).unwrap(),
    )
    .unwrap();
    let function = PrimFunc::new(vec![variable], call).unwrap();
    let module = IRModule::from_expr(&function).unwrap();

    let rust_result =
        IRModule::from_expr(transform::stmt_simplify_prim_func(function).unwrap()).unwrap();
    let cpp_result = cpp_pass("tirx.transform.StmtSimplify").run(module).unwrap();

    assert_structural_equal(&rust_result, &cpp_result);
}

#[test]
fn rust_stmt_simplify_preserves_unchanged_tile_metadata() {
    load_tvm_compiler();
    let source = SourceName::get("tile-call.tvm").unwrap();
    let span = Span::new(&source, 1, 1, 1, 8).unwrap();
    let variable = Var::new("value", "int32").unwrap();
    let redundant: PrimExpr = Add::new(variable.clone(), prim_int_expression(0))
        .unwrap()
        .into();
    let config = Map::from_iter([(tvm::tvm_ffi::String::from("unchanged"), Any::from(1i64))]);
    let call = TilePrimitiveCall::with_span(
        tvm::ir::Op::get("tirx.tile.zero").unwrap(),
        vec![Any::from(redundant)],
        Map::new(),
        config.clone(),
        None,
        ExecScope::new(ScopeKind::THREAD).unwrap(),
        Some(&span),
    )
    .unwrap();
    let result =
        transform::stmt_simplify_prim_func(PrimFunc::new(vec![variable], call).unwrap()).unwrap();
    let result_call = result.body.clone().try_cast::<TilePrimitiveCall>().unwrap();

    assert!(result_call.span.as_ref().unwrap().same_as(&span));
    assert!(result_call.config.same_as(&config));
}

#[test]
fn rust_flatten_buffer_matches_cpp_for_parameter_buffer_and_nested_loops() {
    load_tvm_compiler();
    let buffer_type = BufferType::new(
        "global",
        "int32",
        vec![
            typed_int_expression("int64", 4),
            typed_int_expression("int64", 8),
        ],
    )
    .unwrap();
    let buffer = buffer_type.new_var("matrix");
    let row = Var::new("row", "int64").unwrap();
    let column = Var::new("column", "int64").unwrap();
    let store = BufferStore::new(
        &buffer,
        int_expression(7),
        vec![row.clone().into(), column.clone().into()],
    )
    .unwrap();
    let body = For::new(
        row,
        typed_int_expression("int64", 0),
        typed_int_expression("int64", 4),
        For::new(
            column,
            typed_int_expression("int64", 0),
            typed_int_expression("int64", 8),
            store,
        )
        .unwrap(),
    )
    .unwrap();
    let function = PrimFunc::new(vec![buffer.as_var().clone()], body).unwrap();
    let module = IRModule::from_expr(&function).unwrap();

    let rust_result =
        IRModule::from_expr(transform::flatten_buffer_prim_func(function).unwrap()).unwrap();
    let cpp_result = cpp_pass("tirx.transform.FlattenBuffer")
        .run(module)
        .unwrap();

    assert_structural_equal(&rust_result, &cpp_result);
}

#[test]
fn rust_flatten_buffer_matches_cpp_for_local_definition_load_and_store() {
    load_tvm_compiler();
    let buffer_type = BufferType::new(
        "local",
        "int32",
        vec![
            typed_int_expression("int64", 2),
            typed_int_expression("int64", 3),
        ],
    )
    .unwrap();
    let buffer = buffer_type.new_var("scratch");
    let row = typed_int_expression("int64", 1);
    let column = typed_int_expression("int64", 2);
    let load = TensorLoad::from_buffer(&buffer, vec![row.clone(), column.clone()]).unwrap();
    let body = SeqStmt::new(vec![
        AllocBuffer::new(&buffer).unwrap().into(),
        BufferStore::new(&buffer, int_expression(11), vec![row, column])
            .unwrap()
            .into(),
        Evaluate::new(load).unwrap().into(),
    ])
    .unwrap();
    let function = PrimFunc::from_body(body).unwrap();
    let module = IRModule::from_expr(&function).unwrap();

    let rust_result =
        IRModule::from_expr(transform::flatten_buffer_prim_func(function).unwrap()).unwrap();
    let cpp_result = cpp_pass("tirx.transform.FlattenBuffer")
        .run(module)
        .unwrap();

    assert_structural_equal(&rust_result, &cpp_result);
}

#[test]
fn rust_force_narrow_index_to_int32_matches_cpp_for_buffer_indices() {
    load_tvm_compiler();
    let buffer_type =
        BufferType::new("global", "int32", vec![typed_int_expression("int64", 16)]).unwrap();
    let buffer = buffer_type.new_var("buffer");
    let index = Var::new("index", "int64").unwrap();
    let store = BufferStore::new(
        &buffer,
        int_expression(7),
        vec![Add::new(index.clone(), typed_int_expression("int64", 1))
            .unwrap()
            .into()],
    )
    .unwrap();
    let body = For::new(
        index,
        typed_int_expression("int64", 0),
        typed_int_expression("int64", 15),
        store,
    )
    .unwrap();
    let function = PrimFunc::new(vec![buffer.as_var().clone()], body).unwrap();
    let module = IRModule::from_expr(&function).unwrap();

    let rust_result =
        IRModule::from_expr(transform::force_narrow_index_to_int32_prim_func(function).unwrap())
            .unwrap();
    let cpp_result = cpp_pass("tirx.transform.ForceNarrowIndexToInt32")
        .run(module)
        .unwrap();

    assert_structural_equal(&rust_result, &cpp_result);
}

#[test]
fn rust_force_narrow_remaps_buffer_regions_inside_tile_calls() {
    load_tvm_compiler();
    let buffer_type =
        BufferType::new("global", "int32", vec![typed_int_expression("int64", 16)]).unwrap();
    let buffer = buffer_type.new_var("scratch");
    let region = BufferRegion::new(
        &buffer,
        vec![Range::from_min_extent(
            typed_int_expression("int64", 0),
            typed_int_expression("int64", 16),
        )
        .unwrap()],
    )
    .unwrap();
    let tile_call = TilePrimitiveCall::new(
        tvm::ir::Op::get("tirx.tile.zero").unwrap(),
        vec![Any::from(region)],
        Map::new(),
        Map::new(),
        None,
        ExecScope::new(ScopeKind::THREAD).unwrap(),
    )
    .unwrap();
    let body = SeqStmt::new(vec![
        AllocBuffer::new(&buffer).unwrap().into(),
        tile_call.into(),
    ])
    .unwrap();
    let function = PrimFunc::from_body(body).unwrap();
    let module = IRModule::from_expr(&function).unwrap();

    let rust_function = transform::force_narrow_index_to_int32_prim_func(function).unwrap();
    let rust_result = IRModule::from_expr(&rust_function).unwrap();
    let cpp_result = cpp_pass("tirx.transform.ForceNarrowIndexToInt32")
        .run(module)
        .unwrap();
    assert_structural_equal(&rust_result, &cpp_result);

    let sequence = rust_function.body.clone().try_cast::<SeqStmt>().unwrap();
    let allocation = sequence
        .seq
        .get(0)
        .unwrap()
        .try_cast::<AllocBuffer>()
        .unwrap();
    let call = sequence
        .seq
        .get(1)
        .unwrap()
        .try_cast::<TilePrimitiveCall>()
        .unwrap();
    let region = BufferRegion::try_from(call.args.get(0).unwrap()).unwrap();
    assert!(region.buffer.same_as(&allocation.buffer));
}

#[test]
fn rust_narrow_data_type_matches_cpp_for_proven_and_unproven_ranges() {
    load_tvm_compiler();
    let build = |name: &str, extent: i64| {
        let buffer_type = BufferType::new(
            "global",
            "int32",
            vec![typed_int_expression("int64", extent)],
        )
        .unwrap();
        let buffer = buffer_type.new_var(name);
        let index = Var::new(&format!("{name}_index"), "int64").unwrap();
        let body = For::new(
            index.clone(),
            typed_int_expression("int64", 0),
            typed_int_expression("int64", extent),
            BufferStore::new(&buffer, int_expression(1), vec![index.into()]).unwrap(),
        )
        .unwrap();
        PrimFunc::new(vec![buffer.as_var().clone()], body).unwrap()
    };
    let safe = build("safe", 16);
    let unsafe_range = build("wide", 70_000);
    let module =
        module_from_named_prim_funcs(vec![("safe", safe.clone()), ("wide", unsafe_range.clone())]);

    let rust_module = module_from_named_prim_funcs(vec![
        (
            "safe",
            transform::narrow_data_type_prim_func(safe, 16).unwrap(),
        ),
        (
            "wide",
            transform::narrow_data_type_prim_func(unsafe_range, 16).unwrap(),
        ),
    ]);
    let cpp_narrow: transform::Pass = Function::get_global("tirx.transform.NarrowDataType")
        .unwrap()
        .call_tuple((16_i64,))
        .unwrap()
        .try_into()
        .unwrap();
    let cpp_module = cpp_narrow.run(module).unwrap();

    assert_structural_equal(&rust_module, &cpp_module);
}

#[test]
fn rust_bind_target_matches_cpp_for_mixed_host_and_device_calls() {
    load_tvm_compiler();
    let host = tvm::target::Target::new("llvm").unwrap();
    let target = tvm::target::Target::new("cuda")
        .unwrap()
        .with_host(&host)
        .unwrap();
    assert!(host.without_host().unwrap().same_as(&host));
    let device = target.without_host().unwrap();
    assert!(device.host().unwrap().is_none());
    assert!(device.without_host().unwrap().same_as(&device));
    assert_structural_equal(&device.with_host(&host).unwrap(), &target);
    assert!(target.host().unwrap().unwrap().same_as(&host));
    assert!(host.host().unwrap().is_none());
    assert!(host.has_key("cpu").unwrap());
    assert!(!host.has_key("cuda").unwrap());
    let callee_global = GlobalVar::new("worker.with.dots");
    let callee = PrimFunc::with_metadata(
        Vec::new(),
        Evaluate::from_i64(0).unwrap(),
        PrimType::new("int32").unwrap(),
        DictAttrs::empty(),
        None,
    )
    .unwrap();
    let call = || {
        Evaluate::new(Call::new(
            PrimType::new("int32").unwrap(),
            callee_global.clone(),
            Vec::new(),
        ))
        .unwrap()
    };
    let device_call = AttrStmt::new(
        0_i64,
        "tirx.device_entry",
        typed_int_expression("bool", 1),
        call(),
    )
    .unwrap();
    let caller = PrimFunc::with_metadata(
        Vec::new(),
        SeqStmt::new(vec![call().into(), device_call.into()]).unwrap(),
        Type::missing(),
        DictAttrs::from_dictionary(Map::from_iter([(
            tvm::tvm_ffi::String::from("global_symbol"),
            Any::from(tvm::tvm_ffi::String::from("main")),
        )])),
        None,
    )
    .unwrap();
    let module = IRModule::new(Map::from_iter([
        (callee_global, BaseFunc::from(callee)),
        (GlobalVar::new("main"), BaseFunc::from(caller)),
        (
            GlobalVar::new("worker_with_dots_host"),
            PrimFunc::from_body(Evaluate::from_i64(0).unwrap())
                .unwrap()
                .into(),
        ),
    ]))
    .unwrap();

    let rust_result = transform::bind_target_module(module.clone(), target.clone()).unwrap();
    let cpp_bind: transform::Pass = Function::get_global("tirx.transform.BindTarget")
        .unwrap()
        .call_tuple((target,))
        .unwrap()
        .try_into()
        .unwrap();
    let cpp_result = cpp_bind.run(module).unwrap();

    assert_structural_equal(&rust_result, &cpp_result);
    for result in [&rust_result, &cpp_result] {
        assert!(result
            .functions
            .iter()
            .any(|(global, _)| global.name_hint.as_str() == "worker_with_dots_host_1"));
    }
}

#[test]
fn rust_lower_tirx_cleanup_matches_cpp_for_a_layout_buffer_parameter() {
    load_tvm_compiler();
    let extent = typed_int_expression("int64", 8);
    let layout: Layout = TileLayout::new(
        vec![Iter::new(
            &extent,
            typed_int_expression("int64", 1),
            Axis::get("m").unwrap(),
        )
        .unwrap()],
        Vec::new(),
        Map::new(),
    )
    .unwrap()
    .into();
    let buffer_type = BufferType::with_metadata(
        "global",
        PrimType::new("int32").unwrap(),
        vec![extent],
        Vec::new(),
        typed_int_expression("int64", 0),
        64,
        1,
        Some(layout),
        Vec::new(),
        None,
    )
    .unwrap();
    let buffer = buffer_type.new_var("buffer");
    let body = BufferStore::new(
        &buffer,
        int_expression(7),
        vec![typed_int_expression("int64", 3)],
    )
    .unwrap();
    let attrs = DictAttrs::from_dictionary(Map::from_iter([(
        tvm::tvm_ffi::String::from("target"),
        Any::from(tvm::target::Target::new("llvm").unwrap()),
    )]));
    let function = PrimFunc::with_metadata(
        vec![buffer.as_var().clone()],
        body,
        Type::missing(),
        attrs,
        None,
    )
    .unwrap();
    let module = IRModule::from_expr(&function).unwrap();

    let rust_result =
        IRModule::from_expr(transform::lower_tirx_cleanup_prim_func(function).unwrap()).unwrap();
    let cpp_result = cpp_pass("tirx.transform.LowerTIRxCleanup")
        .run(module)
        .unwrap();

    assert_structural_equal(&rust_result, &cpp_result);
}

#[test]
fn rust_bf16_storage_legalize_matches_cpp_for_local_buffer_storage() {
    load_tvm_compiler();
    let buffer_type =
        BufferType::new("local", "bfloat16", vec![typed_int_expression("int64", 4)]).unwrap();
    let buffer = buffer_type.new_var("values");
    let stored: PrimExpr = Function::get_global("tirx.reinterpret")
        .unwrap()
        .call_tuple((
            PrimType::new("bfloat16").unwrap(),
            IntImm::new("uint16", 0x3f80).unwrap(),
            Option::<Span>::None,
        ))
        .unwrap()
        .try_into()
        .unwrap();
    let body = SeqStmt::new(vec![
        AllocBuffer::new(&buffer).unwrap().into(),
        BufferStore::new(&buffer, stored, vec![typed_int_expression("int64", 0)])
            .unwrap()
            .into(),
    ])
    .unwrap();
    let function = PrimFunc::from_body(body).unwrap();
    let module = IRModule::from_expr(&function).unwrap();

    let rust_result =
        IRModule::from_expr(transform::bf16_storage_legalize_prim_func(function).unwrap()).unwrap();
    let cpp_result = cpp_pass("tirx.transform.BF16StorageLegalize")
        .run(module)
        .unwrap();

    assert_structural_equal(&rust_result, &cpp_result);
}

#[test]
fn rust_fp8_storage_legalize_matches_cpp_for_local_buffer_storage() {
    load_tvm_compiler();
    let buffer_type = BufferType::new(
        "local",
        "float8_e4m3fn",
        vec![typed_int_expression("int64", 4)],
    )
    .unwrap();
    let buffer = buffer_type.new_var("values");
    let stored: PrimExpr = Function::get_global("tirx.reinterpret")
        .unwrap()
        .call_tuple((
            PrimType::new("float8_e4m3fn").unwrap(),
            IntImm::new("uint8", 0x38).unwrap(),
            Option::<Span>::None,
        ))
        .unwrap()
        .try_into()
        .unwrap();
    let body = SeqStmt::new(vec![
        AllocBuffer::new(&buffer).unwrap().into(),
        BufferStore::new(&buffer, stored, vec![typed_int_expression("int64", 0)])
            .unwrap()
            .into(),
    ])
    .unwrap();
    let function = PrimFunc::from_body(body).unwrap();
    let module = IRModule::from_expr(&function).unwrap();

    let rust_result =
        IRModule::from_expr(transform::fp8_storage_legalize_prim_func(function).unwrap()).unwrap();
    let cpp_result = cpp_pass("tirx.transform.FP8StorageLegalize")
        .run(module)
        .unwrap();

    assert_structural_equal(&rust_result, &cpp_result);
}

#[test]
fn rust_compute_legalize_matches_cpp_across_buffer_boundaries() {
    load_tvm_compiler();
    let cpp_fp8: transform::Pass = Function::get_global("tirx.transform.FP8ComputeLegalize")
        .unwrap()
        .call_tuple((tvm::tvm_ffi::String::from("float16"),))
        .unwrap()
        .try_into()
        .unwrap();
    let cases = [
        (
            "bfloat16",
            transform::bf16_compute_legalize().unwrap(),
            cpp_pass("tirx.transform.BF16ComputeLegalize"),
        ),
        (
            "float8_e4m3fn",
            transform::fp8_compute_legalize("float16").unwrap(),
            cpp_fp8,
        ),
    ];
    for (dtype, rust_pass, cpp_pass) in cases {
        let buffer_type =
            BufferType::new("global", dtype, vec![typed_int_expression("int64", 4)]).unwrap();
        let buffer = buffer_type.new_var("values");
        let index = typed_int_expression("int64", 0);
        let loaded = TensorLoad::from_buffer(&buffer, vec![index.clone()]).unwrap();
        let increment = FloatImm::new(dtype, 1.0).unwrap();
        let updated = Add::new(loaded, increment).unwrap();
        let body = BufferStore::new(&buffer, updated, vec![index]).unwrap();
        let function = PrimFunc::new(vec![buffer.as_var().clone()], body).unwrap();
        let module = IRModule::from_expr(function).unwrap();

        let rust_result = rust_pass.run(module.clone()).unwrap();
        let cpp_result = cpp_pass.run(module).unwrap();
        assert_structural_equal(&rust_result, &cpp_result);
    }
}

#[test]
fn rust_remove_no_op_uses_branch_constraints_like_cpp() {
    load_tvm_compiler();
    let extent = Var::new("n", "int32").unwrap();
    let loop_var = Var::new("i", "int32").unwrap();
    let effect: Expr = Call::new(
        PrimType::new("int32").unwrap(),
        GlobalVar::new("effect"),
        Vec::new(),
    )
    .into();
    let loop_statement = For::new(
        loop_var,
        int_expression(0),
        extent.clone(),
        Evaluate::new(effect).unwrap(),
    )
    .unwrap();
    let body = IfThenElse::with_span(
        LE::new(extent.clone(), int_expression(0)).unwrap(),
        loop_statement,
        Some(Evaluate::from_i64(0).unwrap().into()),
        None,
    )
    .unwrap();
    let function = PrimFunc::new(vec![extent], body).unwrap();
    let module = IRModule::from_expr(&function).unwrap();

    let rust_result =
        IRModule::from_expr(transform::remove_no_op_prim_func(function).unwrap()).unwrap();
    let cpp_result = cpp_pass("tirx.transform.RemoveNoOp").run(module).unwrap();

    assert_structural_equal(&rust_result, &cpp_result);
}
