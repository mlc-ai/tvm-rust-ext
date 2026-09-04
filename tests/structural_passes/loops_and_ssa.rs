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
fn rust_unroll_loop_matches_cpp_for_explicit_loop() {
    load_tvm_compiler();
    let loop_var = Var::new("i", "int32").unwrap();
    let temporary = Var::new("temporary", "int32").unwrap();
    let body = SeqStmt::new(vec![
        Bind::new(temporary.clone(), loop_var.clone())
            .unwrap()
            .into(),
        Evaluate::new(temporary).unwrap().into(),
    ])
    .unwrap();
    let loop_node = For::with_metadata(
        loop_var,
        int_expression(2),
        int_expression(3),
        ForKind::kUnrolled,
        body.into(),
        None,
        Map::new(),
        None,
        None,
    )
    .unwrap();
    let module = IRModule::from_expr(PrimFunc::from_body(loop_node).unwrap()).unwrap();

    let rust_result = transform::unroll_loop()
        .unwrap()
        .run(module.clone())
        .unwrap();
    let cpp_result = cpp_pass("tirx.transform.UnrollLoop").run(module).unwrap();

    assert_structural_equal(&rust_result, &cpp_result);
}

#[test]
fn rust_unroll_loop_matches_cpp_for_scoped_auto_unroll_pragma() {
    load_tvm_compiler();
    let loop_var = Var::new("i", "int32").unwrap();
    let loop_node = For::new(
        loop_var.clone(),
        int_expression(0),
        int_expression(3),
        Evaluate::new(loop_var.clone()).unwrap(),
    )
    .unwrap();
    let body = AttrStmt::new(
        loop_var,
        "pragma_auto_unroll_max_step",
        int_expression(8),
        loop_node,
    )
    .unwrap();
    let module = IRModule::from_expr(PrimFunc::from_body(body).unwrap()).unwrap();

    let rust_result = transform::unroll_loop()
        .unwrap()
        .run(module.clone())
        .unwrap();
    let cpp_result = cpp_pass("tirx.transform.UnrollLoop").run(module).unwrap();

    assert_structural_equal(&rust_result, &cpp_result);
}

#[test]
fn rust_unroll_loop_matches_cpp_when_explicit_expansion_is_disabled() {
    load_tvm_compiler();
    let loop_var = Var::new("i", "int32").unwrap();
    let loop_node = For::new(
        loop_var.clone(),
        int_expression(0),
        int_expression(3),
        Evaluate::new(loop_var.clone()).unwrap(),
    )
    .unwrap();
    let explicit = AttrStmt::new(
        loop_var.clone(),
        "pragma_unroll_explicit",
        int_expression(0),
        loop_node,
    )
    .unwrap();
    let body = AttrStmt::new(
        loop_var,
        "pragma_auto_unroll_max_step",
        int_expression(8),
        explicit,
    )
    .unwrap();
    let module = IRModule::from_expr(PrimFunc::from_body(body).unwrap()).unwrap();

    let rust_result = transform::unroll_loop()
        .unwrap()
        .run(module.clone())
        .unwrap();
    let cpp_result = cpp_pass("tirx.transform.UnrollLoop").run(module).unwrap();

    assert_structural_equal(&rust_result, &cpp_result);
    let function = rust_result
        .functions
        .iter()
        .next()
        .unwrap()
        .1
        .try_cast::<PrimFunc>()
        .unwrap();
    assert_eq!(
        function.body.clone().try_cast::<For>().unwrap().kind,
        ForKind::kUnrolled
    );
}

#[test]
fn rust_unroll_loop_matches_cpp_for_local_buffer_indices() {
    load_tvm_compiler();
    let buffer_type =
        BufferType::new("local", "int32", vec![typed_int_expression("int64", 4)]).unwrap();
    let buffer = buffer_type.new_var("local_buffer");
    let loop_var = Var::new("i", "int32").unwrap();
    let store =
        BufferStore::new(&buffer, int_expression(1), vec![loop_var.clone().into()]).unwrap();
    let loop_node = For::new(loop_var, int_expression(0), int_expression(4), store).unwrap();
    let function = PrimFunc::new(vec![buffer.as_var().clone()], loop_node).unwrap();
    let module = IRModule::from_expr(function).unwrap();

    let unroll_config = Map::<tvm::tvm_ffi::String, Any>::from_iter([(
        tvm::tvm_ffi::String::from("unroll_local_access"),
        Any::from(true),
    )]);
    let pass_config = Map::<tvm::tvm_ffi::String, Any>::from_iter([(
        tvm::tvm_ffi::String::from("tirx.UnrollLoop"),
        Any::from(unroll_config),
    )]);
    let context: transform::PassContext = Function::get_global("transform.PassContext")
        .unwrap()
        .call_tuple((
            2_i64,
            Array::<tvm::tvm_ffi::String>::new(Vec::new()),
            Array::<tvm::tvm_ffi::String>::new(Vec::new()),
            Array::<Any>::new(Vec::new()),
            Some(pass_config),
        ))
        .unwrap()
        .try_into()
        .unwrap();
    Function::get_global("transform.EnterPassContext")
        .unwrap()
        .call_tuple((&context,))
        .unwrap();
    let rust_result = transform::unroll_loop().and_then(|pass| pass.run(module.clone()));
    let cpp_result = cpp_pass("tirx.transform.UnrollLoop").run(module);
    Function::get_global("transform.ExitPassContext")
        .unwrap()
        .call_tuple((&context,))
        .unwrap();

    assert_structural_equal(&rust_result.unwrap(), &cpp_result.unwrap());
}

#[test]
fn rust_convert_ssa_matches_cpp_for_flat_redefinitions() {
    load_tvm_compiler();
    let variable = Var::new("value", "int32").unwrap();
    let body = SeqStmt::new(vec![
        Bind::new(variable.clone(), int_expression(1))
            .unwrap()
            .into(),
        Evaluate::new(variable.clone()).unwrap().into(),
        Bind::new(variable.clone(), int_expression(2))
            .unwrap()
            .into(),
        Evaluate::new(variable).unwrap().into(),
    ])
    .unwrap();
    let module = IRModule::from_expr(PrimFunc::from_body(body).unwrap()).unwrap();

    let rust_result = transform::convert_ssa()
        .unwrap()
        .run(module.clone())
        .unwrap();
    let cpp_result = cpp_pass("tirx.transform.ConvertSSA").run(module).unwrap();

    assert_structural_equal(&rust_result, &cpp_result);

    let unchanged =
        IRModule::from_expr(PrimFunc::from_body(Evaluate::from_i64(0).unwrap()).unwrap()).unwrap();
    let unchanged_pointer = object_pointer(&unchanged);
    let converted = transform::convert_ssa_module(unchanged).unwrap();
    assert_eq!(object_pointer(&converted), unchanged_pointer);
}

#[test]
fn rust_convert_ssa_matches_cpp_for_repeated_let_binders() {
    load_tvm_compiler();
    let variable = Var::new("value", "int32").unwrap();
    let first = Let::new(variable.clone(), int_expression(1), variable.clone()).unwrap();
    let second = Let::new(variable.clone(), int_expression(2), variable).unwrap();
    let body = SeqStmt::new(vec![
        Evaluate::new(first).unwrap().into(),
        Evaluate::new(second).unwrap().into(),
    ])
    .unwrap();
    let module = IRModule::from_expr(PrimFunc::from_body(body).unwrap()).unwrap();

    let rust_result = transform::convert_ssa()
        .unwrap()
        .run(module.clone())
        .unwrap();
    let cpp_result = cpp_pass("tirx.transform.ConvertSSA").run(module).unwrap();

    assert_structural_equal(&rust_result, &cpp_result);
    let rust_function = rust_result
        .functions
        .iter()
        .next()
        .unwrap()
        .1
        .try_cast::<PrimFunc>()
        .unwrap();
    let valid_ssa: bool = Function::get_global("tirx.analysis.verify_ssa")
        .unwrap()
        .call_tuple((rust_function,))
        .unwrap()
        .try_into()
        .unwrap();
    assert!(valid_ssa);
}

#[test]
fn rust_convert_ssa_keeps_while_bindings_scoped_like_cpp() {
    load_tvm_compiler();
    let condition = Var::new("condition", "bool").unwrap();
    let variable = Var::new("value", "int32").unwrap();
    let loop_body = Bind::new(variable.clone(), int_expression(1)).unwrap();
    let body = SeqStmt::new(vec![
        Bind::new(variable.clone(), int_expression(0))
            .unwrap()
            .into(),
        While::new(condition.clone(), loop_body).unwrap().into(),
        Evaluate::new(variable).unwrap().into(),
    ])
    .unwrap();
    let module = IRModule::from_expr(PrimFunc::new(vec![condition], body).unwrap()).unwrap();

    let rust_result = transform::convert_ssa()
        .unwrap()
        .run(module.clone())
        .unwrap();
    let cpp_result = cpp_pass("tirx.transform.ConvertSSA").run(module).unwrap();

    assert_structural_equal(&rust_result, &cpp_result);
}

#[test]
fn rust_convert_ssa_matches_cpp_for_reused_dynamic_buffer_parameters() {
    load_tvm_compiler();
    let shape = Var::new("n", "int64").unwrap();
    let buffer_type = BufferType::new("global", "int32", vec![shape.into()]).unwrap();
    let buffer = buffer_type.new_var("buffer");
    let make_function = || {
        PrimFunc::new(
            vec![buffer.as_var().clone()],
            BufferStore::new(
                &buffer,
                int_expression(1),
                vec![typed_int_expression("int64", 0)],
            )
            .unwrap(),
        )
        .unwrap()
    };
    let module = IRModule::new(Map::from_iter([
        (GlobalVar::new("first"), BaseFunc::from(make_function())),
        (GlobalVar::new("second"), BaseFunc::from(make_function())),
    ]))
    .unwrap();

    let rust_result = transform::convert_ssa()
        .unwrap()
        .run(module.clone())
        .unwrap();
    let cpp_result = cpp_pass("tirx.transform.ConvertSSA").run(module).unwrap();

    assert_structural_equal(&rust_result, &cpp_result);
}

#[test]
fn rust_convert_ssa_keeps_for_annotations_outside_the_ssa_scope() {
    load_tvm_compiler();
    let variable = Var::new("i", "int32").unwrap();
    let annotation_key = tvm::tvm_ffi::String::from("test.metadata");
    let annotations = Map::from_iter([(annotation_key.clone(), Any::from(variable.clone()))]);
    let loop_node = For::with_metadata(
        variable.clone(),
        int_expression(0),
        int_expression(1),
        ForKind::kSerial,
        Evaluate::new(variable.clone()).unwrap().into(),
        None,
        annotations,
        None,
        None,
    )
    .unwrap();
    let body = SeqStmt::new(vec![
        Bind::new(variable.clone(), int_expression(0))
            .unwrap()
            .into(),
        loop_node.into(),
    ])
    .unwrap();
    let module = IRModule::from_expr(PrimFunc::from_body(body).unwrap()).unwrap();

    let rust_result = transform::convert_ssa()
        .unwrap()
        .run(module.clone())
        .unwrap();
    let cpp_result = cpp_pass("tirx.transform.ConvertSSA").run(module).unwrap();

    assert_structural_equal(&rust_result, &cpp_result);
    let function = rust_result
        .functions
        .iter()
        .next()
        .unwrap()
        .1
        .try_cast::<PrimFunc>()
        .unwrap();
    let sequence = function.body.clone().try_cast::<SeqStmt>().unwrap();
    let loop_node = sequence.seq.get(1).unwrap().try_cast::<For>().unwrap();
    let annotation =
        Var::try_from(loop_node.annotations.get(&annotation_key).unwrap().unwrap()).unwrap();
    assert!(annotation.same_as(&variable));
    assert!(!loop_node.loop_var.as_var().same_as(&variable));
}

#[test]
fn new_non_sblock_bindings_match_native_constructors() {
    load_tvm_compiler();
    let lhs = prim_int_expression(1);
    let rhs = prim_int_expression(2);

    macro_rules! assert_binary_constructor {
        ($rust:expr, $name:literal) => {{
            let native: Expr = Function::get_global($name)
                .unwrap()
                .call_tuple((lhs.clone(), rhs.clone(), Option::<Span>::None))
                .unwrap()
                .try_into()
                .unwrap();
            assert_structural_equal(&$rust, &native);
        }};
    }

    assert_binary_constructor!(
        Expr::from(NE::new(lhs.clone(), rhs.clone()).unwrap()),
        "ir.prim.NE"
    );
    assert_binary_constructor!(
        Expr::from(LT::new(lhs.clone(), rhs.clone()).unwrap()),
        "ir.prim.LT"
    );
    assert_binary_constructor!(
        Expr::from(LE::new(lhs.clone(), rhs.clone()).unwrap()),
        "ir.prim.LE"
    );
    assert_binary_constructor!(
        Expr::from(GT::new(lhs.clone(), rhs.clone()).unwrap()),
        "ir.prim.GT"
    );
    assert_binary_constructor!(
        Expr::from(GE::new(lhs.clone(), rhs.clone()).unwrap()),
        "ir.prim.GE"
    );

    let condition = PrimExpr::try_from(typed_int_expression("bool", 1)).unwrap();
    let native_not: Expr = Function::get_global("ir.prim.Not")
        .unwrap()
        .call_tuple((condition.clone(), Option::<Span>::None))
        .unwrap()
        .try_into()
        .unwrap();
    assert_structural_equal(
        &Expr::from(Not::new(condition.clone()).unwrap()),
        &native_not,
    );

    let rust_select = Select::new(condition.clone(), lhs.clone(), rhs.clone()).unwrap();
    let native_select: Expr = Function::get_global("ir.prim.Select")
        .unwrap()
        .call_tuple((
            condition.clone(),
            lhs.clone(),
            rhs.clone(),
            Option::<Span>::None,
        ))
        .unwrap()
        .try_into()
        .unwrap();
    assert_structural_equal(&Expr::from(rust_select), &native_select);

    let let_variable = Var::new("let_bound", "int32").unwrap();
    let rust_let = Let::new(let_variable.clone(), lhs.clone(), rhs.clone()).unwrap();
    let native_let: Expr = Function::get_global("ir.prim.Let")
        .unwrap()
        .call_tuple((let_variable, lhs, rhs, Option::<Span>::None))
        .unwrap()
        .try_into()
        .unwrap();
    assert_structural_equal(&Expr::from(rust_let), &native_let);

    let rust_while = While::new(condition.clone(), Evaluate::from_i64(0).unwrap()).unwrap();
    let native_while: Stmt = Function::get_global("tirx.While")
        .unwrap()
        .call_tuple((
            condition,
            Stmt::from(Evaluate::from_i64(0).unwrap()),
            Option::<Span>::None,
        ))
        .unwrap()
        .try_into()
        .unwrap();
    assert_structural_equal(&Stmt::from(rust_while), &native_while);

    let variable = Var::new("bound", "int32").unwrap();
    let rust_bind = Bind::new(variable.clone(), int_expression(1)).unwrap();
    let native_bind: Bind = Function::get_global("tirx.Bind")
        .unwrap()
        .call_tuple((variable, int_expression(1), Option::<Span>::None))
        .unwrap()
        .try_into()
        .unwrap();
    assert_structural_equal(&rust_bind, &native_bind);

    let buffer_type =
        BufferType::new("global", "int32", vec![typed_int_expression("int64", 4)]).unwrap();
    let buffer = buffer_type.new_var("buffer");
    let data = int_expression(0);
    let rust_decl = DeclBuffer::new(&buffer, data.clone()).unwrap();
    let native_decl: DeclBuffer = Function::get_global("tirx.DeclBuffer")
        .unwrap()
        .call_tuple((buffer.clone(), data, Option::<Span>::None))
        .unwrap()
        .try_into()
        .unwrap();
    assert_structural_equal(&rust_decl, &native_decl);

    let rust_alloc = AllocBuffer::new(&buffer).unwrap();
    let native_alloc: AllocBuffer = Function::get_global("tirx.AllocBuffer")
        .unwrap()
        .call_tuple((
            buffer,
            Option::<Map<tvm::tvm_ffi::String, Any>>::None,
            Option::<Span>::None,
        ))
        .unwrap()
        .try_into()
        .unwrap();
    assert_structural_equal(&rust_alloc, &native_alloc);
}
