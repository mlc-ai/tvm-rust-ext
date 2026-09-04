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
fn common_subexpr_elim_matches_cpp_for_repeated_arithmetic() {
    load_tvm_compiler();
    let variable = Var::new("x", "int32").unwrap();
    let lhs = Add::new(variable.clone(), int_expression(1)).unwrap();
    let rhs = Add::new(variable.clone(), int_expression(1)).unwrap();
    let body = Evaluate::new(Mul::new(lhs, rhs).unwrap()).unwrap();
    let function = PrimFunc::new(vec![variable], body).unwrap();
    let module = IRModule::from_expr(&function).unwrap();

    let rust_function = transform::common_subexpr_elim_prim_func(function).unwrap();
    let rust_result = IRModule::from_expr(&rust_function).unwrap();
    let cpp_result = cpp_pass("tirx.transform.CommonSubexprElim")
        .run(module)
        .unwrap();

    assert_structural_equal(&rust_result, &cpp_result);
    let sequence = rust_function.body.clone().try_cast::<SeqStmt>().unwrap();
    assert_eq!(sequence.seq.len(), 2);
    let binding = sequence.seq.get(0).unwrap().try_cast::<Bind>().unwrap();
    assert_eq!(binding.var.name.as_str(), "cse_v1");
}

#[test]
fn common_subexpr_elim_matches_cpp_scope_and_forbidden_call_rules() {
    load_tvm_compiler();
    let variable = Var::new("x", "int32").unwrap();
    let condition = Var::new("condition", "bool").unwrap();
    let then_value = Add::new(variable.clone(), int_expression(1)).unwrap();
    let else_value = Add::new(variable.clone(), int_expression(1)).unwrap();
    let conditional: Stmt = IfThenElse::with_span(
        condition.clone(),
        Evaluate::new(then_value).unwrap(),
        Some(Evaluate::new(else_value).unwrap().into()),
        None,
    )
    .unwrap()
    .into();

    let callee = GlobalVar::new("callee");
    let call_type = PrimType::new("int32").unwrap();
    let call_lhs = Call::new(call_type.clone(), callee.clone(), Vec::new());
    let call_rhs = Call::new(call_type, callee, Vec::new());
    let effectful_lhs = Add::new(call_lhs, int_expression(1)).unwrap();
    let effectful_rhs = Add::new(call_rhs, int_expression(1)).unwrap();
    let effectful: Stmt = Evaluate::new(Mul::new(effectful_lhs, effectful_rhs).unwrap())
        .unwrap()
        .into();

    let let_variable = Var::new("let_value", "int32").unwrap();
    let let_lhs = Add::new(let_variable.clone(), variable.clone()).unwrap();
    let let_rhs = Add::new(let_variable.clone(), variable.clone()).unwrap();
    let let_body = Add::new(let_lhs, let_rhs).unwrap();
    let let_expression: Expr = Function::get_global("ir.prim.Let")
        .unwrap()
        .call_tuple((
            let_variable,
            prim_int_expression(0),
            PrimExpr::try_from(Expr::from(let_body)).unwrap(),
            Option::<Span>::None,
        ))
        .unwrap()
        .try_into()
        .unwrap();
    let let_statement: Stmt = Evaluate::new(let_expression).unwrap().into();

    let first_predicate = LT::new(variable.clone(), int_expression(10)).unwrap();
    let second_predicate = LT::new(variable.clone(), int_expression(10)).unwrap();
    let first_bool_use: Stmt = IfThenElse::new(first_predicate, Evaluate::from_i64(1).unwrap())
        .unwrap()
        .into();
    let second_bool_use: Stmt = IfThenElse::new(second_predicate, Evaluate::from_i64(2).unwrap())
        .unwrap()
        .into();

    let loop_variable = Var::new("i", "int32").unwrap();
    let loop_lhs = Add::new(loop_variable.clone(), variable.clone()).unwrap();
    let loop_rhs = Add::new(loop_variable.clone(), variable.clone()).unwrap();
    let loop_body = SeqStmt::new(vec![
        Evaluate::new(loop_lhs).unwrap().into(),
        Evaluate::new(loop_rhs).unwrap().into(),
    ])
    .unwrap();
    let loop_statement: Stmt = For::new(
        loop_variable,
        int_expression(0),
        int_expression(4),
        loop_body,
    )
    .unwrap()
    .into();

    let while_lhs = Add::new(variable.clone(), int_expression(2)).unwrap();
    let while_rhs = Add::new(variable.clone(), int_expression(2)).unwrap();
    let while_body = SeqStmt::new(vec![
        Evaluate::new(while_lhs).unwrap().into(),
        Evaluate::new(while_rhs).unwrap().into(),
    ])
    .unwrap();
    let while_statement: Stmt = Function::get_global("tirx.While")
        .unwrap()
        .call_tuple((
            PrimExpr::try_from(Expr::from(condition.clone())).unwrap(),
            Stmt::from(while_body),
            Option::<Span>::None,
        ))
        .unwrap()
        .try_into()
        .unwrap();
    let body = SeqStmt::new(vec![
        conditional,
        effectful,
        let_statement,
        first_bool_use,
        second_bool_use,
        loop_statement,
        while_statement,
    ])
    .unwrap();
    let function = PrimFunc::new(vec![variable, condition], body).unwrap();
    let module = IRModule::from_expr(&function).unwrap();

    let rust_function = transform::common_subexpr_elim_prim_func(function).unwrap();
    let rust_result = IRModule::from_expr(&rust_function).unwrap();
    let cpp_result = cpp_pass("tirx.transform.CommonSubexprElim")
        .run(module)
        .unwrap();

    assert_structural_equal(&rust_result, &cpp_result);
    let mut bindings = 0;
    structural_walk(
        &rust_function,
        |_: &BindObj| {
            bindings += 1;
            WalkResult::Advance
        },
        WalkOrder::PreOrder,
    )
    .unwrap();
    assert_eq!(bindings, 3);
}

#[test]
fn common_subexpr_elim_matches_cpp_for_shared_statement_identity() {
    load_tvm_compiler();
    let variable = Var::new("x", "int32").unwrap();
    let lhs = Add::new(variable.clone(), int_expression(1)).unwrap();
    let rhs = Add::new(variable.clone(), int_expression(1)).unwrap();
    let shared: Stmt = Evaluate::new(Mul::new(lhs, rhs).unwrap()).unwrap().into();
    let body = SeqStmt::new(vec![shared.clone(), shared]).unwrap();
    let function = PrimFunc::new(vec![variable], body).unwrap();
    let module = IRModule::from_expr(&function).unwrap();

    let rust_result = transform::common_subexpr_elim()
        .unwrap()
        .run(module.clone())
        .unwrap();
    let cpp_result = cpp_pass("tirx.transform.CommonSubexprElim")
        .run(module)
        .unwrap();

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
fn common_subexpr_elim_matches_cpp_for_decl_buffer_data() {
    load_tvm_compiler();
    let variable = Var::new("x", "int32").unwrap();
    let lhs = Add::new(variable.clone(), int_expression(1)).unwrap();
    let rhs = Add::new(variable.clone(), int_expression(1)).unwrap();
    let data = Mul::new(lhs, rhs).unwrap();
    let buffer_type =
        BufferType::new("global", "int32", vec![typed_int_expression("int64", 1)]).unwrap();
    let declaration = DeclBuffer::new(buffer_type.new_var("buffer"), data).unwrap();
    let function = PrimFunc::new(vec![variable], declaration).unwrap();
    let module = IRModule::from_expr(&function).unwrap();

    let rust_result =
        IRModule::from_expr(transform::common_subexpr_elim_prim_func(function).unwrap()).unwrap();
    let cpp_result = cpp_pass("tirx.transform.CommonSubexprElim")
        .run(module)
        .unwrap();

    assert_structural_equal(&rust_result, &cpp_result);
}

#[test]
fn common_subexpr_elim_remaps_buffer_definitions_and_uses_together() {
    load_tvm_compiler();
    let variable = Var::new("n", "int64").unwrap();
    let first: Expr = Add::new(variable.clone(), typed_int_expression("int64", 1))
        .unwrap()
        .into();
    let second: Expr = Add::new(variable.clone(), typed_int_expression("int64", 1))
        .unwrap()
        .into();
    let buffer_type = BufferType::new("local", "int32", vec![first, second]).unwrap();
    let buffer = buffer_type.new_var("buffer");
    let body = SeqStmt::new(vec![
        AllocBuffer::new(&buffer).unwrap().into(),
        BufferStore::new(
            &buffer,
            int_expression(1),
            vec![
                typed_int_expression("int64", 0),
                typed_int_expression("int64", 0),
            ],
        )
        .unwrap()
        .into(),
    ])
    .unwrap();
    let function = PrimFunc::new(vec![variable], body).unwrap();
    let module = IRModule::from_expr(function.clone()).unwrap();

    let rust_result =
        IRModule::from_expr(transform::common_subexpr_elim_prim_func(function).unwrap()).unwrap();
    let cpp_result = cpp_pass("tirx.transform.CommonSubexprElim")
        .run(module)
        .unwrap();

    assert_structural_equal(&rust_result, &cpp_result);
}

#[test]
fn rust_lower_tirx_dedup_cu_tensor_maps_matches_cpp() {
    load_tvm_compiler();
    let get_operator = |name: &str| -> Expr { tvm::ir::Op::get(name).unwrap().into() };
    let stack_alloca = get_operator("tirx.tvm_stack_alloca");
    let call_packed = get_operator("tirx.tvm_call_packed");
    let handle_type = PointerType::new(PrimType::void(), "global").unwrap();
    let first = Var::with_type("first_tensormap", handle_type.clone());
    let duplicate = Var::with_type("duplicate_tensormap", handle_type.clone());
    let allocation = |variable: Var| {
        Bind::new(
            variable,
            Call::new(
                handle_type.clone(),
                stack_alloca.clone(),
                vec![StringImm::new("tensormap").into(), int_expression(1)],
            ),
        )
        .unwrap()
    };
    let encode = |variable: Var| {
        Evaluate::new(Call::new(
            PrimType::new("int32").unwrap(),
            call_packed.clone(),
            vec![
                StringImm::new("runtime.cuTensorMapEncodeTiled").into(),
                variable.into(),
                int_expression(16),
                int_expression(32),
            ],
        ))
        .unwrap()
    };
    let use_handle = Evaluate::new(Call::new(
        PrimType::new("int32").unwrap(),
        call_packed.clone(),
        vec![
            StringImm::new("testing.consume_tensormap").into(),
            duplicate.clone().into(),
        ],
    ))
    .unwrap();
    let body = Stmt::sequence(vec![
        allocation(first.clone()).into(),
        encode(first.clone()).into(),
        allocation(duplicate.clone()).into(),
        encode(duplicate).into(),
        use_handle.into(),
    ])
    .unwrap();
    let function = PrimFunc::from_body(body).unwrap();
    let expected_body = Stmt::sequence(vec![
        allocation(first.clone()).into(),
        encode(first.clone()).into(),
        Evaluate::new(Call::new(
            PrimType::new("int32").unwrap(),
            call_packed,
            vec![
                StringImm::new("testing.consume_tensormap").into(),
                first.into(),
            ],
        ))
        .unwrap()
        .into(),
    ])
    .unwrap();
    let expected = IRModule::from_expr(PrimFunc::from_body(expected_body).unwrap()).unwrap();

    let rust_result = IRModule::from_expr(
        transform::lower_tirx_dedup_cu_tensor_maps_prim_func(function).unwrap(),
    )
    .unwrap();

    assert_structural_equal(&rust_result, &expected);
}

#[test]
fn rust_lower_intrin_matches_cpp_for_integer_floor_division() {
    load_tvm_compiler();
    let value = Var::new("value", "int32").unwrap();
    let body =
        Evaluate::new(FloorDiv::new(value.clone(), IntImm::new("int32", 8).unwrap()).unwrap())
            .unwrap();
    let attrs = DictAttrs::from_dictionary(Map::from_iter([(
        tvm::tvm_ffi::String::from("target"),
        Any::from(tvm::target::Target::new("llvm").unwrap()),
    )]));
    let function =
        PrimFunc::with_metadata(vec![value], body, Type::missing(), attrs, None).unwrap();
    let module = IRModule::from_expr(function.clone()).unwrap();

    let rust_result =
        IRModule::from_expr(transform::lower_intrin_prim_func(function).unwrap()).unwrap();
    let cpp_result = cpp_pass("tirx.transform.LowerIntrin").run(module).unwrap();

    assert_structural_equal(&rust_result, &cpp_result);
}

#[test]
fn rust_lower_intrin_matches_cpp_for_access_pointer() {
    load_tvm_compiler();
    let element_type = PrimType::new("float32").unwrap();
    let pointer_type = PointerType::new(element_type, "global").unwrap();
    let data = Var::with_type("data", pointer_type.clone());
    let access_ptr: Expr = tvm::ir::Op::get("tirx.tvm_access_ptr").unwrap().into();
    let access = Call::new(
        pointer_type,
        access_ptr,
        vec![
            FloatImm::new("float32", 0.0).unwrap().into(),
            data.clone().into(),
            int_expression(3),
            int_expression(8),
            int_expression(1),
        ],
    );
    let attrs = DictAttrs::from_dictionary(Map::from_iter([(
        tvm::tvm_ffi::String::from("target"),
        Any::from(tvm::target::Target::new("llvm").unwrap()),
    )]));
    let function = PrimFunc::with_metadata(
        vec![data],
        Evaluate::new(access).unwrap(),
        Type::missing(),
        attrs,
        None,
    )
    .unwrap();
    let module = IRModule::from_expr(function.clone()).unwrap();

    let rust_result =
        IRModule::from_expr(transform::lower_intrin_prim_func(function).unwrap()).unwrap();
    let cpp_result = cpp_pass("tirx.transform.LowerIntrin").run(module).unwrap();

    assert_structural_equal(&rust_result, &cpp_result);
}

#[test]
fn rust_lower_intrin_matches_cpp_for_signed_floor_remainder() {
    load_tvm_compiler();
    let value = Var::new("value", "int32").unwrap();
    let body =
        Evaluate::new(FloorMod::new(value.clone(), IntImm::new("int32", 3).unwrap()).unwrap())
            .unwrap();
    let attrs = DictAttrs::from_dictionary(Map::from_iter([(
        tvm::tvm_ffi::String::from("target"),
        Any::from(tvm::target::Target::new("llvm").unwrap()),
    )]));
    let function =
        PrimFunc::with_metadata(vec![value], body, Type::missing(), attrs, None).unwrap();
    let module = IRModule::from_expr(function.clone()).unwrap();

    let rust_result =
        IRModule::from_expr(transform::lower_intrin_prim_func(function).unwrap()).unwrap();
    let cpp_result = cpp_pass("tirx.transform.LowerIntrin").run(module).unwrap();

    assert_structural_equal(&rust_result, &cpp_result);
}

#[test]
fn rust_lower_intrin_matches_cpp_for_registered_target_rule() {
    load_tvm_compiler();
    let value = Var::new("value", "float32").unwrap();
    let exponential: Expr = tvm::ir::Op::get("tirx.exp").unwrap().into();
    let body = Evaluate::new(Call::new(
        PrimType::new("float32").unwrap(),
        exponential,
        vec![value.clone().into()],
    ))
    .unwrap();
    let attrs = DictAttrs::from_dictionary(Map::from_iter([(
        tvm::tvm_ffi::String::from("target"),
        Any::from(tvm::target::Target::new("llvm").unwrap()),
    )]));
    let function =
        PrimFunc::with_metadata(vec![value], body, Type::missing(), attrs, None).unwrap();
    let module = IRModule::from_expr(function.clone()).unwrap();

    let rust_result =
        IRModule::from_expr(transform::lower_intrin_prim_func(function).unwrap()).unwrap();
    let cpp_result = cpp_pass("tirx.transform.LowerIntrin").run(module).unwrap();

    assert_structural_equal(&rust_result, &cpp_result);
}

#[test]
fn rust_lower_intrin_matches_cpp_for_fused_multiply_add() {
    load_tvm_compiler();
    let a = Var::new("a", "float32").unwrap();
    let b = Var::new("b", "float32").unwrap();
    let c = Var::new("c", "float32").unwrap();
    let expression = Add::new(Mul::new(a.clone(), b.clone()).unwrap(), c.clone()).unwrap();
    let attrs = DictAttrs::from_dictionary(Map::from_iter([(
        tvm::tvm_ffi::String::from("target"),
        Any::from(tvm::target::Target::new("llvm").unwrap()),
    )]));
    let function = PrimFunc::with_metadata(
        vec![a, b, c],
        Evaluate::new(expression).unwrap(),
        Type::missing(),
        attrs,
        None,
    )
    .unwrap();
    let module = IRModule::from_expr(function.clone()).unwrap();

    let rust_result =
        IRModule::from_expr(transform::lower_intrin_prim_func(function).unwrap()).unwrap();
    let cpp_result = cpp_pass("tirx.transform.LowerIntrin").run(module).unwrap();

    assert_structural_equal(&rust_result, &cpp_result);
}
