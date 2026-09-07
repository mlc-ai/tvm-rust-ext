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
    let rust_evaluate = rust_function.body().clone().try_cast::<Evaluate>().unwrap();
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
fn rust_stmt_simplify_matches_cpp_for_nested_expression_dispatch() -> Result<()> {
    use tvm::ir::{Op, Tuple, TupleGetItem};

    load_tvm_compiler();
    let sum: Expr = Add::new(int_expression(1), int_expression(2))?.into();
    let tuple = Tuple::new(vec![sum.clone(), Tuple::new(vec![sum]).into()]);
    let call = Call::new(
        PrimType::new("int32")?,
        Op::get("ir.prim.shift_left")?,
        vec![int_expression(3), int_expression(1)],
    );
    let body = SeqStmt::new(vec![
        Evaluate::new(tuple.clone())?.into(),
        Evaluate::new(TupleGetItem::new(tuple, 0)?)?.into(),
        Evaluate::new(call)?.into(),
    ])?;
    let module = IRModule::from_expr(PrimFunc::from_body(body)?)?;
    let native = cpp_pass("tirx.transform.StmtSimplify").run(module.clone())?;
    let rust = transform::stmt_simplify()?.run(module)?;
    assert_structural_equal(&rust, &native);
    Ok(())
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
    let result_call = result
        .body()
        .clone()
        .try_cast::<TilePrimitiveCall>()
        .unwrap();

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
    let broadcast: PrimExpr = tvm::ir::prim::Broadcast::new(
        IntImm::new("int64", 9).unwrap(),
        IntImm::new("int32", 4).unwrap(),
    )
    .unwrap()
    .into();
    let shuffle = tvm::ir::prim::Shuffle::new(
        Array::new(vec![broadcast.clone()]),
        Array::new(vec![prim_int_expression(1)]),
    )
    .unwrap();
    let body = For::new(
        index,
        typed_int_expression("int64", 0),
        typed_int_expression("int64", 15),
        Stmt::sequence(vec![
            store.into(),
            Evaluate::new(broadcast).unwrap().into(),
            Evaluate::new(shuffle).unwrap().into(),
        ])
        .unwrap(),
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
fn rust_force_narrow_preserves_buffer_remaps_inside_later_definitions() -> Result<()> {
    load_tvm_compiler();
    let first =
        BufferType::new("local", "int32", vec![typed_int_expression("int64", 8)])?.new_var("first");
    let extent = TensorLoad::from_buffer(&first, vec![int_expression(0)])?;
    let second = BufferType::new("local", "int32", vec![extent.into()])?.new_var("second");
    let function = PrimFunc::from_body(SeqStmt::new(vec![
        AllocBuffer::new(&first)?.into(),
        BufferStore::new(&first, int_expression(4), vec![int_expression(0)])?.into(),
        AllocBuffer::new(&second)?.into(),
        BufferStore::new(&second, int_expression(1), vec![int_expression(0)])?.into(),
    ])?)?;
    let module = IRModule::from_expr(function)?;
    let native = cpp_pass("tirx.transform.ForceNarrowIndexToInt32").run(module.clone())?;
    let rust = transform::force_narrow_index_to_int32()?.run(module)?;
    assert_structural_equal(&rust, &native);
    Ok(())
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

    let sequence = rust_function.body().clone().try_cast::<SeqStmt>().unwrap();
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
fn rust_narrow_data_type_matches_cpp_for_ranges_and_shared_uses() -> Result<()> {
    load_tvm_compiler();
    let index = Var::new("i", "int64")?;
    let buffer = BufferType::new(
        "global",
        "int32",
        vec![typed_int_expression("int64", 70_000)],
    )?
    .new_var("data");
    let narrow_use: Stmt =
        BufferStore::new(&buffer, int_expression(1), vec![index.clone().into()])?.into();
    let wide_expression = Mul::new(&index, typed_int_expression("int64", 1_000_000))?;
    let wide_use: Stmt = Evaluate::new(wide_expression.clone())?.into();
    let cases = [
        ("proven_range", 16, narrow_use.clone(), Map::new()),
        ("unproven_range", 70_000, narrow_use.clone(), Map::new()),
        (
            "narrow_then_wide",
            16,
            SeqStmt::new(vec![narrow_use.clone(), wide_use.clone()])?.into(),
            Map::new(),
        ),
        (
            "wide_then_narrow",
            16,
            SeqStmt::new(vec![wide_use, narrow_use.clone()])?.into(),
            Map::new(),
        ),
        (
            "ignored_annotation",
            16,
            narrow_use,
            Map::from_iter([("test_annotation".into(), Any::from(wide_expression))]),
        ),
    ];
    let mut functions = Vec::new();
    for (name, extent, body, annotations) in cases {
        let function = PrimFunc::new(
            vec![buffer.as_var().clone()],
            For::with_metadata(
                index.clone(),
                typed_int_expression("int64", 0),
                typed_int_expression("int64", extent),
                ForKind::kSerial,
                body,
                None,
                annotations,
                None,
                None,
            )?,
        )?;
        functions.push((name, function));
    }
    let module = module_from_named_prim_funcs(functions);
    let native_pass: transform::Pass = Function::get_global("tirx.transform.NarrowDataType")?
        .call_tuple((16_i64,))?
        .try_into()?;
    let native = native_pass.run(module.clone())?;
    let rust = transform::narrow_data_type(16)?.run(module)?;
    assert_structural_equal(&rust, &native);
    Ok(())
}

#[test]
fn rust_index_narrowing_matches_cpp_for_arithmetic_rebuilding() -> Result<()> {
    use tvm::ir::prim::{Div, Max, Min, Mod, Sub};

    load_tvm_compiler();
    let index = Var::new("i", "int64")?;
    let one = typed_int_expression("int64", 1);
    let arithmetic: Vec<PrimExpr> = vec![
        Add::new(&index, &one)?.into(),
        Add::new(&index, typed_int_expression("int64", 0))?.into(),
        Sub::new(&index, &index)?.into(),
        Mul::new(&index, &one)?.into(),
        Div::new(&index, &one)?.into(),
        Mod::new(&index, &one)?.into(),
        FloorDiv::new(&index, &one)?.into(),
        FloorMod::new(&index, &one)?.into(),
        Min::new(&index, &index)?.into(),
        Max::new(&index, &index)?.into(),
        EQ::new(&index, &one)?.into(),
        NE::new(&index, &one)?.into(),
        LT::new(&index, &one)?.into(),
        LE::new(&index, &one)?.into(),
        GT::new(&index, &one)?.into(),
        GE::new(&index, &one)?.into(),
    ];
    let body = arithmetic
        .into_iter()
        .map(|expression| Evaluate::new(expression).map(Stmt::from))
        .collect::<Result<Vec<_>>>()?;
    let function = PrimFunc::from_body(For::new(
        index,
        typed_int_expression("int64", 0),
        typed_int_expression("int64", 16),
        SeqStmt::new(body)?,
    )?)?;
    let module = IRModule::from_expr(function)?;
    let native_narrow: transform::Pass = Function::get_global("tirx.transform.NarrowDataType")?
        .call_tuple((16_i64,))?
        .try_into()?;
    for (rust_pass, native_pass) in [
        (transform::narrow_data_type(16)?, native_narrow),
        (
            transform::force_narrow_index_to_int32()?,
            cpp_pass("tirx.transform.ForceNarrowIndexToInt32"),
        ),
    ] {
        let native = native_pass.run(module.clone())?;
        let rust = rust_pass.run(module.clone())?;
        assert_structural_equal(&rust, &native);
    }
    Ok(())
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
fn rust_storage_legalize_matches_cpp_for_local_and_masked_accesses() -> Result<()> {
    use tvm::ir::Op;

    load_tvm_compiler();
    let span = Span::new(SourceName::get("storage.rs")?, 1, 1, 1, 10)?;
    for (dtype, uint_dtype, bits, rust_pass, native_pass) in [
        (
            "bfloat16",
            "uint16",
            0x3f80,
            transform::bf16_storage_legalize()?,
            cpp_pass("tirx.transform.BF16StorageLegalize"),
        ),
        (
            "float8_e4m3fn",
            "uint8",
            0x38,
            transform::fp8_storage_legalize()?,
            cpp_pass("tirx.transform.FP8StorageLegalize"),
        ),
    ] {
        let buffer = BufferType::new("local", dtype, vec![int_expression(4)])?.new_var("values");
        let stored: PrimExpr = Function::get_global("tirx.reinterpret")?
            .call_tuple((
                PrimType::new(dtype)?,
                IntImm::new(uint_dtype, bits)?,
                Option::<Span>::None,
            ))?
            .try_into()?;
        let index = int_expression(0);
        let predicate = typed_int_expression("bool", 1);
        let load = Call::new(
            PrimType::new(dtype)?,
            Op::get("tirx.masked_load")?,
            vec![
                buffer.as_var().clone().into(),
                index.clone(),
                predicate.clone(),
            ],
        );
        let store = Call::new(
            PrimType::void(),
            Op::get("tirx.masked_store")?,
            vec![
                buffer.as_var().clone().into(),
                stored.clone().into(),
                index.clone(),
                predicate,
            ],
        );
        let access = AttrStmt::with_span(
            buffer.clone(),
            "test.buffer",
            int_expression(0),
            SeqStmt::new(vec![
                BufferStore::new(&buffer, stored, vec![index])?.into(),
                Evaluate::new(load)?.into(),
                Evaluate::new(store)?.into(),
            ])?,
            Some(&span),
        )?;
        let function = PrimFunc::from_body(SeqStmt::new(vec![
            AllocBuffer::new(&buffer)?.into(),
            access.into(),
        ])?)?;
        let module = IRModule::from_expr(function)?;
        let rust = rust_pass.run(module.clone())?;
        let native = native_pass.run(module)?;
        assert_structural_equal(&rust, &native);
        for result in [rust, native] {
            let function: PrimFunc = result.functions.iter().next().unwrap().1.try_cast()?;
            let body: SeqStmt = function.body().clone().try_cast()?;
            let attribute: AttrStmt = body.seq.get(1)?.try_cast()?;
            // Native legalization reconstructs attributes when their node is remapped.
            assert!(attribute.span.is_none());
        }
    }
    Ok(())
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
    let mut cases = vec![(
        "bfloat16",
        transform::bf16_compute_legalize().unwrap(),
        cpp_pass("tirx.transform.BF16ComputeLegalize"),
    )];
    for dtype in [
        "float8_e3m4",
        "float8_e4m3",
        "float8_e4m3b11fnuz",
        "float8_e4m3fn",
        "float8_e4m3fnuz",
        "float8_e5m2",
        "float8_e5m2fnuz",
        "float8_e8m0fnu",
    ] {
        cases.push((
            dtype,
            transform::fp8_compute_legalize("float16").unwrap(),
            cpp_fp8.clone(),
        ));
    }
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
fn rust_compute_legalize_recurses_into_allocated_buffer_metadata() -> Result<()> {
    use tvm::ir::prim::Cast;

    load_tvm_compiler();
    let extent: Expr = Cast::new(PrimType::new("int32")?, FloatImm::new("bfloat16", 8.0)?)?.into();
    let buffer = BufferType::new("local", "int32", vec![extent])?.new_var("buffer");
    let function = PrimFunc::from_body(SeqStmt::new(vec![
        AllocBuffer::new(&buffer)?.into(),
        Evaluate::new(TensorLoad::from_buffer(&buffer, vec![int_expression(0)])?)?.into(),
    ])?)?;
    let module = IRModule::from_expr(function)?;
    let native = cpp_pass("tirx.transform.BF16ComputeLegalize").run(module.clone())?;
    let rust = transform::bf16_compute_legalize()?.run(module)?;
    assert_structural_equal(&rust, &native);
    Ok(())
}

#[test]
fn rust_compute_legalize_preserves_buffers_referenced_by_layouts() -> Result<()> {
    use tvm::ir::Op;

    load_tvm_compiler();
    let buffer = BufferType::new("local", "bfloat16", vec![int_expression(8)])?.new_var("data");
    let extent = Call::new(
        PrimType::new("int32")?,
        Op::get("tirx.call_extern")?,
        vec![
            StringImm::new("layout_extent").into(),
            Call::new(
                PointerType::new(PrimType::new("bfloat16")?, "local")?,
                Op::get("tirx.buffer_data")?,
                vec![buffer.as_var().clone().into()],
            )
            .into(),
        ],
    );
    let iteration = Iter::new(extent, int_expression(1), Axis::get("m")?)?;
    let rust_pass = transform::bf16_compute_legalize()?;
    let native_pass = cpp_pass("tirx.transform.BF16ComputeLegalize");
    for (shard, replica) in [
        (vec![iteration.clone()], Vec::new()),
        (Vec::new(), vec![iteration]),
    ] {
        let layout = TileLayout::new(shard, replica, Map::new())?;
        let layout_buffer = BufferType::with_metadata(
            "local",
            PrimType::new("int32")?,
            vec![int_expression(8)],
            Vec::new(),
            int_expression(0),
            64,
            1,
            Some(layout.into()),
            Vec::new(),
            None,
        )?
        .new_var("layout_buffer");
        let function = PrimFunc::from_body(SeqStmt::new(vec![
            AllocBuffer::new(&buffer)?.into(),
            AllocBuffer::new(layout_buffer)?.into(),
            Evaluate::new(TensorLoad::from_buffer(&buffer, vec![int_expression(0)])?)?.into(),
        ])?)?;
        let module = IRModule::from_expr(function)?;
        let native = native_pass.run(module.clone())?;
        let rust = rust_pass.run(module.clone())?;
        assert_structural_equal(&rust, &native);
    }
    Ok(())
}

#[test]
fn rust_compute_legalize_matches_cpp_for_fixed_and_scalable_vector_stores() -> Result<()> {
    use tvm::ir::prim::{Broadcast, Ramp};
    use tvm::ir::Op;

    load_tvm_compiler();
    let scalable: Expr = Mul::new(
        Call::new(
            PrimType::new("int32")?,
            Op::get("ir.prim.vscale")?,
            Vec::new(),
        ),
        IntImm::new("int32", 4)?,
    )?
    .into();
    let cpp_fp8: transform::Pass = Function::get_global("tirx.transform.FP8ComputeLegalize")?
        .call_tuple((tvm::tvm_ffi::String::from("float16"),))?
        .try_into()?;
    for (dtype, rust_pass, native_pass) in [
        (
            "bfloat16",
            transform::bf16_compute_legalize()?,
            cpp_pass("tirx.transform.BF16ComputeLegalize"),
        ),
        (
            "float8_e4m3fn",
            transform::fp8_compute_legalize("float16")?,
            cpp_fp8,
        ),
    ] {
        for (lanes, supported) in [(int_expression(4), true), (scalable.clone(), false)] {
            let buffer =
                BufferType::new("global", dtype, vec![int_expression(64)])?.new_var("data");
            let index = Ramp::new(int_expression(0), int_expression(1), &lanes)?;
            let value = Broadcast::new(FloatImm::new(dtype, 1.0)?, &lanes)?;
            let function = PrimFunc::new(
                vec![buffer.as_var().clone()],
                BufferStore::new(&buffer, value, vec![index.into()])?,
            )?;
            let module = IRModule::from_expr(function)?;
            let native = native_pass.run(module.clone());
            let rust = rust_pass.run(module);
            assert_eq!(native.is_ok(), supported, "{dtype}");
            assert_eq!(rust.is_ok(), supported, "{dtype}");
            if supported {
                assert_structural_equal(&rust?, &native?);
            } else {
                assert!(native.err().unwrap().to_string().contains("scalable"));
                assert!(rust.err().unwrap().to_string().contains("scalable"));
            }
        }
    }
    Ok(())
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
