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
    let sequence = rust_function.body().clone().try_cast::<SeqStmt>().unwrap();
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
fn rust_lower_intrin_matches_cpp_for_scalar_and_vector_floor_operations() -> Result<()> {
    use tvm::ir::prim::{Broadcast, Max};

    load_tvm_compiler();
    let scalable: PrimExpr = Mul::new(
        int_expression(4),
        Call::new(
            PrimType::new("int32")?,
            tvm::ir::Op::get("ir.prim.vscale")?,
            Vec::new(),
        ),
    )?
    .into();
    let attrs = DictAttrs::from_dictionary(Map::from_iter([(
        "target".into(),
        Any::from(tvm::target::Target::new("llvm")?),
    )]));
    let cpp = cpp_pass("tirx.transform.LowerIntrin");
    for dtype in ["int16", "int32", "int64", "uint32"] {
        for lanes in [None, Some(prim_int_expression(4)), Some(scalable.clone())] {
            let constant = |number| -> Result<PrimExpr> {
                let scalar = IntImm::new(dtype, number)?;
                match &lanes {
                    Some(lanes) => Ok(Broadcast::new(scalar, lanes)?.into()),
                    None => Ok(scalar.into()),
                }
            };
            let ty = constant(0)?.type_annotation();
            let value = Var::with_type("value", ty.clone());
            let divisor = Var::with_type("divisor", ty);
            let variable: PrimExpr = Expr::from(&value).try_into()?;
            let mut cases = vec![
                ("by_one", variable.clone(), constant(1)?),
                ("by_three", variable.clone(), constant(3)?),
                ("by_eight", variable.clone(), constant(8)?),
                (
                    "by_variable",
                    variable.clone(),
                    Expr::from(&divisor).try_into()?,
                ),
                ("constants", constant(10)?, constant(3)?),
            ];
            if dtype.starts_with("int") {
                cases.push(("negative_divisor", variable, constant(-3)?));
                cases.push(("negative_dividend", constant(-10)?, constant(3)?));
            }
            for (name, lhs, rhs) in cases {
                let division = FloorDiv::new(&lhs, &rhs)?;
                let remainder = FloorMod::new(&lhs, &rhs)?;
                let operations: [PrimExpr; 5] = [
                    division.clone().into(),
                    remainder.clone().into(),
                    Max::new(division, constant(0)?)?.into(),
                    EQ::new(&remainder, constant(0)?)?.into(),
                    NE::new(remainder, constant(0)?)?.into(),
                ];
                for operation in operations {
                    eprintln!(
                        "LowerIntrin floor operation: {name}, {dtype}, lanes={}",
                        operation.dtype().lanes
                    );
                    let function = PrimFunc::with_metadata(
                        vec![value.clone(), divisor.clone()],
                        Evaluate::new(operation)?,
                        Type::missing(),
                        attrs.clone(),
                        None,
                    )?;
                    let module = IRModule::from_expr(&function)?;
                    let rust_function = transform::lower_intrin_prim_func(function)?;
                    structural_walk(
                        &rust_function,
                        |literal: IntImm| {
                            assert_eq!(
                                PrimExpr::from(literal).dtype().lanes,
                                1,
                                "IntImm must be scalar"
                            );
                            WalkResult::Advance
                        },
                        WalkOrder::PreOrder,
                    )?;
                    assert_structural_equal(
                        &IRModule::from_expr(rust_function)?,
                        &cpp.run(module)?,
                    );
                }
            }
        }
    }
    Ok(())
}

#[test]
fn rust_lower_intrin_matches_cpp_analyzer_scopes() -> Result<()> {
    load_tvm_compiler();
    let n = Var::new("n", "int32")?;
    let i = Var::new("i", "int32")?;
    let thread = Var::new("thread", "int32")?;
    let quotient = |value: Var| -> Result<Stmt> {
        Ok(Evaluate::new(FloorDiv::new(value, int_expression(3))?)?.into())
    };
    let condition: PrimExpr = GE::new(n.clone(), int_expression(0))?.into();
    let negative: PrimExpr = tvm::ir::prim::Sub::new(int_expression(0), n.clone())?.into();
    let division: PrimExpr = FloorDiv::new(n.clone(), int_expression(3))?.into();
    let negative_division: PrimExpr = FloorDiv::new(negative, int_expression(3))?.into();
    let assertion = AssertStmt::new(condition.clone(), "ValueError", "nonnegative")?;
    let iter = IterVar::with_metadata(
        None,
        thread.clone(),
        IterVarType::kThreadIndex,
        "threadIdx.x",
        None,
    )?;
    let bound = Var::new("bound", "int32")?;
    let cases: Vec<(&str, Stmt)> = vec![
        (
            "loop",
            For::new(
                i.clone(),
                int_expression(0),
                int_expression(10),
                quotient(i.clone())?,
            )?
            .into(),
        ),
        (
            "branch",
            IfThenElse::with_span(
                condition.clone(),
                quotient(n.clone())?,
                Some(quotient(n.clone())?),
                None,
            )?
            .into(),
        ),
        (
            "loop_extent_and_exit",
            SeqStmt::new(vec![
                For::new(i, int_expression(0), n.clone(), quotient(n.clone())?)?.into(),
                quotient(n.clone())?,
            ])?
            .into(),
        ),
        (
            "likely",
            IfThenElse::new(
                Call::new(
                    PrimType::new("bool")?,
                    tvm::ir::Op::get("ir.prim.likely")?,
                    vec![condition.clone().into()],
                ),
                quotient(n.clone())?,
            )?
            .into(),
        ),
        (
            "select",
            Evaluate::new(Select::new(
                condition.clone(),
                division.clone(),
                negative_division.clone(),
            )?)?
            .into(),
        ),
        (
            "if_then_else_call",
            Evaluate::new(Call::new(
                PrimType::new("int32")?,
                tvm::ir::Op::get("ir.prim.if_then_else")?,
                vec![condition.into(), division.into(), negative_division.into()],
            ))?
            .into(),
        ),
        (
            "assert_siblings",
            SeqStmt::new(vec![assertion.clone().into(), quotient(n.clone())?])?.into(),
        ),
        (
            "nested_sequence",
            SeqStmt::new(vec![
                SeqStmt::new(vec![
                    assertion.clone().into(),
                    Evaluate::from_i64(1)?.into(),
                ])?
                .into(),
                quotient(n.clone())?,
            ])?
            .into(),
        ),
        (
            "branch_exit",
            SeqStmt::new(vec![
                IfThenElse::new(GE::new(n.clone(), int_expression(-10))?, assertion.clone())?
                    .into(),
                quotient(n.clone())?,
            ])?
            .into(),
        ),
        (
            "attribute_exit",
            SeqStmt::new(vec![
                AttrStmt::new(
                    n.clone(),
                    "scope",
                    int_expression(1),
                    SeqStmt::new(vec![assertion.into(), quotient(n.clone())?])?,
                )?
                .into(),
                quotient(n.clone())?,
            ])?
            .into(),
        ),
        (
            "thread_extent",
            AttrStmt::new(
                iter.clone(),
                "thread_extent",
                int_expression(32),
                quotient(thread.clone())?,
            )?
            .into(),
        ),
        (
            "virtual_thread",
            AttrStmt::new(
                iter,
                "virtual_thread",
                int_expression(32),
                quotient(thread)?,
            )?
            .into(),
        ),
        (
            "bind",
            SeqStmt::new(vec![
                Bind::new(bound.clone(), int_expression(6))?.into(),
                quotient(bound.clone())?,
            ])?
            .into(),
        ),
        (
            "let",
            Evaluate::new(Let::new(
                bound.clone(),
                int_expression(6),
                FloorDiv::new(bound.clone(), int_expression(3))?,
            )?)?
            .into(),
        ),
        (
            "constant_branch",
            IfThenElse::new(IntImm::new("bool", 1)?, quotient(n.clone())?)?.into(),
        ),
        (
            "constant_select",
            Evaluate::new(Select::new(
                IntImm::new("bool", 0)?,
                n.clone(),
                int_expression(4),
            )?)?
            .into(),
        ),
        (
            "constant_false_without_else",
            IfThenElse::new(IntImm::new("bool", 0)?, quotient(n.clone())?)?.into(),
        ),
        (
            "effectful_binding",
            SeqStmt::new(vec![
                Bind::new(
                    bound.clone(),
                    Call::new(
                        PrimType::new("int32")?,
                        tvm::ir::Op::get("tirx.call_extern")?,
                        vec![StringImm::new("read_value").into()],
                    ),
                )?
                .into(),
                quotient(bound.clone())?,
            ])?
            .into(),
        ),
        (
            "ordinary_function_binding",
            SeqStmt::new(vec![
                Bind::new(
                    bound.clone(),
                    Call::new(
                        PrimType::new("int32")?,
                        GlobalVar::new("callee"),
                        vec![n.clone().into()],
                    ),
                )?
                .into(),
                quotient(bound)?,
            ])?
            .into(),
        ),
        (
            "rewritten_assertion",
            SeqStmt::new(vec![
                AssertStmt::new(
                    GE::new(
                        FloorDiv::new(n.clone(), int_expression(8))?,
                        int_expression(0),
                    )?,
                    "ValueError",
                    "quotient is nonnegative",
                )?
                .into(),
                quotient(n.clone())?,
            ])?
            .into(),
        ),
    ];
    let attrs = DictAttrs::from_dictionary(Map::from_iter([(
        "target".into(),
        Any::from(tvm::target::Target::new("llvm")?),
    )]));
    let native = cpp_pass("tirx.transform.LowerIntrin");
    for (name, body) in cases {
        eprintln!("LowerIntrin analyzer case: {name}");
        let function =
            PrimFunc::with_metadata(vec![n.clone()], body, Type::missing(), attrs.clone(), None)?;
        let module = IRModule::from_expr(function)?;
        let expected = native.run(module.clone())?;
        let actual = transform::lower_intrin()?.run(module)?;
        assert_structural_equal(&actual, &expected);
    }
    Ok(())
}

#[test]
fn rust_lower_intrin_rewrites_buffer_definitions_and_uses() -> Result<()> {
    load_tvm_compiler();
    let n = Var::new("n", "int32")?;
    let extent: Expr = FloorDiv::new(n.clone(), int_expression(8))?.into();
    let iteration = Iter::new(extent.clone(), extent.clone(), Axis::get("m")?)?;
    let buffer = BufferType::with_metadata(
        "local",
        PrimType::new("int32")?,
        vec![extent.clone()],
        vec![extent.clone()],
        extent,
        64,
        1,
        Some(TileLayout::new(vec![iteration], Vec::new(), Map::new())?.into()),
        Vec::new(),
        None,
    )?
    .new_var("buffer");
    let data = Var::with_type("data", PointerType::new(PrimType::new("int32")?, "local")?);
    let attrs = DictAttrs::from_dictionary(Map::from_iter([(
        "target".into(),
        Any::from(tvm::target::Target::new("llvm")?),
    )]));
    let native = cpp_pass("tirx.transform.LowerIntrin");
    for definition in [
        Stmt::from(AllocBuffer::new(&buffer)?),
        DeclBuffer::new(&buffer, &data)?.into(),
    ] {
        let body = SeqStmt::new(vec![
            definition,
            BufferStore::new(&buffer, int_expression(1), vec![int_expression(0)])?.into(),
            Evaluate::new(TensorLoad::from_buffer(&buffer, vec![int_expression(0)])?)?.into(),
            Evaluate::new(Call::new(
                PrimType::new("int32")?,
                tvm::ir::Op::get("tirx.call_extern")?,
                vec![
                    StringImm::new("use_buffer").into(),
                    BufferRegion::new(
                        &buffer,
                        vec![Range::from_min_extent(
                            int_expression(0),
                            int_expression(1),
                        )?],
                    )?
                    .into(),
                    buffer.clone().into(),
                ],
            ))?
            .into(),
        ])?;
        let function = PrimFunc::with_metadata(
            vec![n.clone(), data.clone()],
            body,
            Type::missing(),
            attrs.clone(),
            None,
        )?;
        let module = IRModule::from_expr(function)?;
        let expected = native.run(module.clone())?;
        let actual = transform::lower_intrin()?.run(module)?;
        assert_structural_equal(&actual, &expected);
    }
    Ok(())
}

#[test]
fn rust_lower_intrin_matches_cpp_for_access_pointer() -> Result<()> {
    load_tvm_compiler();
    let pointer_type = PointerType::new(PrimType::new("float32")?, "global")?;
    let data = Var::with_type("data", pointer_type.clone());
    let access = Call::new(
        pointer_type,
        tvm::ir::Op::get("tirx.tvm_access_ptr")?,
        vec![
            FloatImm::new("float32", 0.0)?.into(),
            data.clone().into(),
            int_expression(3),
            int_expression(8),
            int_expression(1),
        ],
    );
    let use_pointer = |dtype: &str| -> Result<Call> {
        Ok(Call::new(
            PrimType::new(dtype)?,
            tvm::ir::Op::get("tirx.call_extern")?,
            vec![StringImm::new("use_pointer").into(), access.clone().into()],
        ))
    };
    // Aliases must also surround statements handled by typed callbacks.
    let bodies: Vec<Stmt> = vec![
        Evaluate::new(access.clone())?.into(),
        For::new(
            Var::new("i", "int32")?,
            int_expression(0),
            use_pointer("int32")?,
            Evaluate::from_i64(0)?,
        )?
        .into(),
        IfThenElse::new(use_pointer("bool")?, Evaluate::from_i64(0)?)?.into(),
        AttrStmt::new(
            data.clone(),
            "scope",
            use_pointer("int32")?,
            Evaluate::from_i64(0)?,
        )?
        .into(),
        AssertStmt::new(use_pointer("bool")?, "ValueError", "pointer check")?.into(),
        Bind::new(Var::new("bound", "int32")?, use_pointer("int32")?)?.into(),
    ];
    let attrs = DictAttrs::from_dictionary(Map::from_iter([(
        "target".into(),
        Any::from(tvm::target::Target::new("llvm")?),
    )]));
    let native = cpp_pass("tirx.transform.LowerIntrin");
    for body in bodies {
        let function = PrimFunc::with_metadata(
            vec![data.clone()],
            body,
            Type::missing(),
            attrs.clone(),
            None,
        )?;
        let module = IRModule::from_expr(function)?;
        let expected = native.run(module.clone())?;
        let actual = transform::lower_intrin()?.run(module)?;
        assert_structural_equal(&actual, &expected);
    }
    for (dtype, supported) in [("float32x4", true), ("float32xvscalex4", false)] {
        let marker = Var::new("marker", dtype)?;
        let access = Call::new(
            data.ty.clone(),
            tvm::ir::Op::get("tirx.tvm_access_ptr")?,
            vec![
                marker.clone().into(),
                data.clone().into(),
                int_expression(3),
                int_expression(8),
                int_expression(1),
            ],
        );
        let function = PrimFunc::with_metadata(
            vec![data.clone(), marker],
            Evaluate::new(access)?,
            Type::missing(),
            attrs.clone(),
            None,
        )?;
        let module = IRModule::from_expr(function)?;
        let expected = native.run(module.clone());
        let actual = transform::lower_intrin()?.run(module);
        assert_eq!(expected.is_ok(), supported, "C++: {dtype}");
        assert_eq!(actual.is_ok(), supported, "Rust: {dtype}");
        if supported {
            assert_structural_equal(&actual?, &expected?);
        }
    }
    Ok(())
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
