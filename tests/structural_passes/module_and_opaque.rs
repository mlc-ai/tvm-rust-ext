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
fn rust_inline_private_functions_matches_cpp() {
    load_tvm_compiler();
    let callee_global = GlobalVar::new("private_add_one");
    let parameter = Var::new("value", "int32").unwrap();
    let callee = PrimFunc::new(
        vec![parameter.clone()],
        Evaluate::new(Add::new(parameter, int_expression(1)).unwrap()).unwrap(),
    )
    .unwrap();
    let call = Call::new(
        callee.ret_type.clone(),
        callee_global.clone(),
        vec![int_expression(2)],
    );
    let caller = PrimFunc::with_metadata(
        Vec::new(),
        Evaluate::new(call).unwrap(),
        Type::missing(),
        DictAttrs::from_dictionary(Map::from_iter([(
            tvm::tvm_ffi::String::from("global_symbol"),
            Any::from(tvm::tvm_ffi::String::from("main")),
        )])),
        None,
    )
    .unwrap();
    let recursive_global = GlobalVar::new("recursive");
    let recursive = PrimFunc::from_body(
        Evaluate::new(Call::new(
            callee.ret_type.clone(),
            recursive_global.clone(),
            Vec::new(),
        ))
        .unwrap(),
    )
    .unwrap();
    let module = IRModule::new(Map::from_iter([
        (callee_global, BaseFunc::from(callee)),
        (recursive_global, BaseFunc::from(recursive)),
        (GlobalVar::new("main"), BaseFunc::from(caller)),
    ]))
    .unwrap();

    let rust_result = transform::inline_private_functions()
        .unwrap()
        .run(module.clone())
        .unwrap();
    let cpp_result = cpp_pass("tirx.transform.InlinePrivateFunctions")
        .run(module)
        .unwrap();

    assert_structural_equal(&rust_result, &cpp_result);
    assert_eq!(rust_result.functions.len(), 2);
}

#[test]
fn rust_inline_private_functions_matches_cpp_for_targets_and_expression_calls() {
    load_tvm_compiler();
    let make_target = |kind: &str| {
        Function::get_global("target.Target")
            .unwrap()
            .call_packed(&[AnyView::from(&tvm::tvm_ffi::String::from(kind))])
            .unwrap()
    };
    let llvm = make_target("llvm");
    let cuda = make_target("cuda");
    let int_type = PrimType::new("int32").unwrap();

    let target_global = GlobalVar::new("private_cuda");
    let target_callee = PrimFunc::with_metadata(
        Vec::new(),
        Evaluate::from_i64(1).unwrap(),
        int_type.clone(),
        DictAttrs::from_dictionary(Map::from_iter([(
            tvm::tvm_ffi::String::from("target"),
            cuda,
        )])),
        None,
    )
    .unwrap();

    let expression_global = GlobalVar::new("private_expression");
    let expression_callee = PrimFunc::with_metadata(
        Vec::new(),
        Evaluate::from_i64(2).unwrap(),
        int_type.clone(),
        DictAttrs::from_dictionary(Map::from_iter([(
            tvm::tvm_ffi::String::from("target"),
            llvm.clone(),
        )])),
        None,
    )
    .unwrap();

    let target_call = Call::new(int_type.clone(), target_global.clone(), Vec::new());
    let expression_call = Call::new(int_type, expression_global.clone(), Vec::new());
    let caller_body = SeqStmt::new(vec![
        Evaluate::new(target_call).unwrap().into(),
        Evaluate::new(Add::new(expression_call, int_expression(1)).unwrap())
            .unwrap()
            .into(),
    ])
    .unwrap();
    let caller = PrimFunc::with_metadata(
        Vec::new(),
        caller_body,
        Type::missing(),
        DictAttrs::from_dictionary(Map::from_iter([
            (
                tvm::tvm_ffi::String::from("global_symbol"),
                Any::from(tvm::tvm_ffi::String::from("main")),
            ),
            (tvm::tvm_ffi::String::from("target"), llvm),
        ])),
        None,
    )
    .unwrap();
    let module = IRModule::new(Map::from_iter([
        (target_global, BaseFunc::from(target_callee)),
        (expression_global, BaseFunc::from(expression_callee)),
        (GlobalVar::new("main"), BaseFunc::from(caller)),
    ]))
    .unwrap();

    let rust_result = transform::inline_private_functions()
        .unwrap()
        .run(module.clone())
        .unwrap();
    let cpp_result = cpp_pass("tirx.transform.InlinePrivateFunctions")
        .run(module)
        .unwrap();

    assert_structural_equal(&rust_result, &cpp_result);
    assert_eq!(rust_result.functions.len(), 3);
}

#[test]
fn annotate_entry_func_matches_cpp_branch_for_branch() {
    load_tvm_compiler();

    let single =
        module_from_named_prim_funcs(vec![("only", prim_func_with_global_symbol(1, None))]);
    let rust_single = transform::annotate_entry_func()
        .unwrap()
        .run(single.clone())
        .unwrap();
    let cpp_single = cpp_pass("tirx.transform.AnnotateEntryFunc")
        .run(single)
        .unwrap();
    assert_structural_equal(&rust_single, &cpp_single);
    assert!(has_nonzero_function_attr(
        &rust_single.functions.iter().next().unwrap().1,
        "tirx.is_entry_func"
    ));

    let unique_external = module_from_named_prim_funcs(vec![
        ("internal", prim_func_with_global_symbol(2, None)),
        (
            "exported",
            prim_func_with_global_symbol(3, Some("exported_symbol")),
        ),
    ]);
    let rust_unique = transform::annotate_entry_func()
        .unwrap()
        .run(unique_external.clone())
        .unwrap();
    let cpp_unique = cpp_pass("tirx.transform.AnnotateEntryFunc")
        .run(unique_external)
        .unwrap();
    assert_structural_equal(&rust_unique, &cpp_unique);
    for (global, function) in rust_unique.functions.iter() {
        assert_eq!(
            has_nonzero_function_attr(&function, "tirx.is_entry_func"),
            global.name_hint.as_str() == "exported"
        );
    }

    let ambiguous = module_from_named_prim_funcs(vec![
        (
            "first",
            prim_func_with_global_symbol(4, Some("first_symbol")),
        ),
        (
            "second",
            prim_func_with_global_symbol(5, Some("second_symbol")),
        ),
    ]);
    let rust_ambiguous = transform::annotate_entry_func()
        .unwrap()
        .run(ambiguous.clone())
        .unwrap();
    let cpp_ambiguous = cpp_pass("tirx.transform.AnnotateEntryFunc")
        .run(ambiguous)
        .unwrap();
    assert_structural_equal(&rust_ambiguous, &cpp_ambiguous);
    assert!(rust_ambiguous
        .functions
        .iter()
        .all(|(_, function)| !has_nonzero_function_attr(&function, "tirx.is_entry_func")));
}

#[test]
fn filter_matches_cpp_and_removes_rejected_prim_funcs() {
    load_tvm_compiler();
    let module = module_from_named_prim_funcs(vec![
        ("one", prim_func_with_global_symbol(1, None)),
        ("two", prim_func_with_global_symbol(2, None)),
        ("three", prim_func_with_global_symbol(3, None)),
    ]);

    let rust_result = transform::filter(|function| Ok(prim_func_body_integer(function)? % 2 == 1))
        .unwrap()
        .run(module.clone())
        .unwrap();

    let cpp_condition = Function::from_typed(|function: PrimFunc| -> Result<bool> {
        Ok(prim_func_body_integer(function)? % 2 == 1)
    });
    let cpp_filter: transform::Pass = Function::get_global("tirx.transform.Filter")
        .unwrap()
        .call_tuple((cpp_condition,))
        .unwrap()
        .try_into()
        .unwrap();
    let cpp_result = cpp_filter.run(module.clone()).unwrap();

    assert_structural_equal(&rust_result, &cpp_result);
    let mut names = rust_result
        .functions
        .iter()
        .map(|(global, _)| global.name_hint.as_str().to_owned())
        .collect::<Vec<_>>();
    names.sort();
    assert_eq!(names, vec!["one", "three"]);

    // Mirror TVM's native regression case: removing every PrimFunc must also
    // clear IRModule's derived global-name index.
    let rust_empty = transform::filter(|_function| Ok(false))
        .unwrap()
        .run(module.clone())
        .unwrap();
    let cpp_reject_all = Function::from_typed(|_function: PrimFunc| -> Result<bool> { Ok(false) });
    let cpp_empty_pass: transform::Pass = Function::get_global("tirx.transform.Filter")
        .unwrap()
        .call_tuple((cpp_reject_all,))
        .unwrap()
        .try_into()
        .unwrap();
    let cpp_empty = cpp_empty_pass.run(module).unwrap();
    assert_structural_equal(&rust_empty, &cpp_empty);
    assert!(rust_empty.functions.is_empty());
    assert!(rust_empty.global_var_map.is_empty());
}

#[test]
fn unit_loop_elimination_preserves_annotated_loops() {
    load_tvm_compiler();
    let loop_var = Var::new("i", "int64").unwrap();
    let body: Stmt = Evaluate::new(Expr::from(loop_var.clone())).unwrap().into();
    let annotations: Map<tvm::tvm_ffi::String, Any> = [(
        tvm::tvm_ffi::String::from("keep_unit_loop"),
        Any::from(1i64),
    )]
    .into_iter()
    .collect();
    let loop_statement = For::with_metadata(
        loop_var,
        typed_int_expression("int64", 7),
        typed_int_expression("int64", 1),
        tvm::tirx::ForKind::kSerial,
        body,
        None,
        annotations,
        None,
        None,
    )
    .unwrap();
    let function = PrimFunc::from_body(&loop_statement).unwrap();
    let module = IRModule::from_expr(&function).unwrap();

    let rust_function = transform::lower_tirx_opaque_prim_func(function).unwrap();
    let rust_result = IRModule::from_expr(&rust_function).unwrap();
    let cpp_result = cpp_pass("tirx.transform.LowerTIRxOpaque")
        .run(module)
        .unwrap();

    assert_eq!(node_statistics(&rust_result).unwrap().loops, 1);
    assert_structural_equal(&rust_result, &cpp_result);
}

#[test]
fn rust_lower_tirx_opaque_remaps_buffer_definitions_and_uses_together() {
    load_tvm_compiler();
    let loop_var = Var::new("i", "int64").unwrap();
    let buffer_type = BufferType::new("local", "int32", vec![loop_var.clone().into()]).unwrap();
    let buffer = buffer_type.new_var("buffer");
    let body = SeqStmt::new(vec![
        AllocBuffer::new(&buffer).unwrap().into(),
        BufferStore::new(
            &buffer,
            int_expression(1),
            vec![typed_int_expression("int64", 0)],
        )
        .unwrap()
        .into(),
    ])
    .unwrap();
    let loop_node = For::new(
        loop_var,
        typed_int_expression("int64", 4),
        typed_int_expression("int64", 1),
        body,
    )
    .unwrap();
    let function = PrimFunc::from_body(loop_node).unwrap();
    let module = IRModule::from_expr(function.clone()).unwrap();

    let rust_function = transform::lower_tirx_opaque_prim_func(function).unwrap();
    let rust_result = IRModule::from_expr(rust_function).unwrap();
    let cpp_result = cpp_pass("tirx.transform.LowerTIRxOpaque")
        .run(module)
        .unwrap();

    assert_structural_equal(&rust_result, &cpp_result);
}

#[test]
fn rust_lower_tirx_opaque_matches_cpp_for_thread_binding_and_pragmas() {
    load_tvm_compiler();
    let loop_var = Var::new("tx", "int64").unwrap();
    let minimum = typed_int_expression("int64", 0);
    let extent = typed_int_expression("int64", 8);
    let thread_axis = IterVar::with_metadata(
        Some(Range::from_min_extent(minimum.clone(), extent.clone()).unwrap()),
        loop_var.clone(),
        IterVarType::kThreadIndex,
        "threadIdx.x",
        None,
    )
    .unwrap();
    let annotations: Map<tvm::tvm_ffi::String, Any> = [
        (
            tvm::tvm_ffi::String::from("pragma_zeta"),
            Any::from(tvm::tirx::StringImm::new("z")),
        ),
        (
            tvm::tvm_ffi::String::from("pragma_alpha"),
            Any::from(IntImm::new("int32", 3).unwrap()),
        ),
        (tvm::tvm_ffi::String::from("pragma_unroll"), Any::from(1i64)),
        (
            tvm::tvm_ffi::String::from("software_pipeline_stage"),
            Any::from(2i64),
        ),
    ]
    .into_iter()
    .collect();
    let body = Evaluate::new(Expr::from(loop_var.clone())).unwrap();
    let loop_statement = For::with_metadata(
        loop_var,
        minimum,
        extent,
        tvm::tirx::ForKind::kThreadBinding,
        body.into(),
        Some(thread_axis),
        annotations,
        None,
        None,
    )
    .unwrap();
    let function = PrimFunc::from_body(&loop_statement).unwrap();
    let module = IRModule::from_expr(&function).unwrap();

    let rust_function = transform::lower_tirx_opaque_prim_func(function).unwrap();
    let rust_result = IRModule::from_expr(&rust_function).unwrap();
    let cpp_result = cpp_pass("tirx.transform.LowerTIRxOpaque")
        .run(module)
        .unwrap();

    assert_structural_equal(&rust_result, &cpp_result);
    let pragma_alpha = rust_function.body.clone().try_cast::<AttrStmt>().unwrap();
    assert_eq!(pragma_alpha.attr_key.as_str(), "pragma_alpha");
    let pragma_zeta = pragma_alpha.body.clone().try_cast::<AttrStmt>().unwrap();
    assert_eq!(pragma_zeta.attr_key.as_str(), "pragma_zeta");
    let launch = pragma_zeta.body.clone().try_cast::<AttrStmt>().unwrap();
    assert_eq!(launch.attr_key.as_str(), "thread_extent");
}

#[test]
fn rust_remap_thread_axis_matches_cpp() {
    load_tvm_compiler();
    let extent = typed_int_expression("int64", 16);
    let old_var = Var::new("tx", "int64").unwrap();
    let old_axis = IterVar::with_metadata(
        Some(Range::from_min_extent(typed_int_expression("int64", 0), extent.clone()).unwrap()),
        old_var.clone(),
        IterVarType::kThreadIndex,
        "threadIdx.x",
        None,
    )
    .unwrap();
    let new_var = Var::new("ty", "int64").unwrap();
    let new_axis = IterVar::with_metadata(
        Some(Range::from_min_extent(typed_int_expression("int64", 0), extent.clone()).unwrap()),
        new_var.clone(),
        IterVarType::kThreadIndex,
        "threadIdx.y",
        None,
    )
    .unwrap();
    let body = Evaluate::new(Expr::from(old_var)).unwrap();
    let attr = AttrStmt::new(old_axis.clone(), "thread_extent", extent, body).unwrap();
    let attrs = DictAttrs::from_dictionary(
        [(
            tvm::tvm_ffi::String::from("tirx.kernel_launch_params"),
            Any::from(Array::new(vec![old_axis])),
        )]
        .into_iter()
        .collect(),
    );
    let function = PrimFunc::with_metadata(Vec::new(), attr, Type::missing(), attrs, None).unwrap();
    let module = IRModule::from_expr(&function).unwrap();
    let thread_map: Map<tvm::tvm_ffi::String, IterVar> =
        [(tvm::tvm_ffi::String::from("threadIdx.x"), new_axis.clone())]
            .into_iter()
            .collect();

    let rust_function = transform::remap_thread_axis_prim_func(function, &thread_map).unwrap();
    let rust_result = IRModule::from_expr(&rust_function).unwrap();
    let cpp_pass: transform::Pass = Function::get_global("tirx.transform.RemapThreadAxis")
        .unwrap()
        .call_tuple((thread_map,))
        .unwrap()
        .try_into()
        .unwrap();
    let cpp_result = cpp_pass.run(module).unwrap();

    assert_structural_equal(&rust_result, &cpp_result);
    let remapped_attr = rust_function.body.clone().try_cast::<AttrStmt>().unwrap();
    let remapped_axis = IterVar::try_from(remapped_attr.node.clone()).unwrap();
    assert!(remapped_axis.same_as(&new_axis));
    let remapped_body = remapped_attr.body.clone().try_cast::<Evaluate>().unwrap();
    let remapped_var = remapped_body.value.clone().try_cast::<Var>().unwrap();
    assert!(remapped_var.same_as(&new_var));
    let launch_params = rust_function
        .attrs
        .dict
        .get(&tvm::tvm_ffi::String::from("tirx.kernel_launch_params"))
        .unwrap()
        .unwrap();
    let launch_params = Array::<IterVar>::try_from(launch_params).unwrap();
    assert!(launch_params.get(0).unwrap().same_as(&new_axis));
}

#[test]
fn rust_remap_thread_axis_does_not_rewrite_loop_annotations() {
    load_tvm_compiler();
    let extent = typed_int_expression("int64", 8);
    let old_var = Var::new("tx", "int64").unwrap();
    let old_axis = IterVar::with_metadata(
        Some(Range::from_min_extent(typed_int_expression("int64", 0), extent.clone()).unwrap()),
        old_var.clone(),
        IterVarType::kThreadIndex,
        "threadIdx.x",
        None,
    )
    .unwrap();
    let new_var = Var::new("ty", "int64").unwrap();
    let new_axis = IterVar::with_metadata(
        Some(Range::from_min_extent(typed_int_expression("int64", 0), extent.clone()).unwrap()),
        new_var.clone(),
        IterVarType::kThreadIndex,
        "threadIdx.y",
        None,
    )
    .unwrap();
    let annotation_key = tvm::tvm_ffi::String::from("test.metadata");
    let loop_var = Var::new("i", "int64").unwrap();
    let annotations = Map::from_iter([(annotation_key.clone(), Any::from(old_var.clone()))]);
    let loop_node = For::with_metadata(
        loop_var,
        typed_int_expression("int64", 0),
        typed_int_expression("int64", 1),
        ForKind::kSerial,
        Evaluate::new(old_var.clone()).unwrap().into(),
        None,
        annotations,
        None,
        None,
    )
    .unwrap();
    let body = AttrStmt::new(old_axis, "thread_extent", extent, loop_node).unwrap();
    let function = PrimFunc::from_body(body).unwrap();
    let module = IRModule::from_expr(function.clone()).unwrap();
    let thread_map = Map::from_iter([(tvm::tvm_ffi::String::from("threadIdx.x"), new_axis)]);

    let rust_function = transform::remap_thread_axis_prim_func(function, &thread_map).unwrap();
    let rust_result = IRModule::from_expr(rust_function.clone()).unwrap();
    let cpp_pass: transform::Pass = Function::get_global("tirx.transform.RemapThreadAxis")
        .unwrap()
        .call_tuple((thread_map,))
        .unwrap()
        .try_into()
        .unwrap();
    let cpp_result = cpp_pass.run(module).unwrap();

    assert_structural_equal(&rust_result, &cpp_result);
    let attribute = rust_function.body.clone().try_cast::<AttrStmt>().unwrap();
    let loop_node = attribute.body.clone().try_cast::<For>().unwrap();
    let annotation =
        Var::try_from(loop_node.annotations.get(&annotation_key).unwrap().unwrap()).unwrap();
    assert!(annotation.same_as(&old_var));
    let evaluated = loop_node.body.clone().try_cast::<Evaluate>().unwrap();
    assert!(evaluated
        .value
        .clone()
        .try_cast::<Var>()
        .unwrap()
        .same_as(&new_var));
}

#[test]
fn rust_remove_assume_matches_cpp_for_a_root_assume() {
    load_tvm_compiler();
    let assume_op: Expr = Function::get_global("ir.GetOp")
        .unwrap()
        .call_tuple((tvm::tvm_ffi::String::from("tirx.assume"),))
        .unwrap()
        .try_into()
        .unwrap();
    let condition = typed_int_expression("bool", 1);
    let call = Call::new(PrimType::new("bool").unwrap(), assume_op, vec![condition]);
    let function = PrimFunc::from_body(Evaluate::new(call).unwrap()).unwrap();
    let module = IRModule::from_expr(&function).unwrap();

    let rust_result = transform::remove_assume()
        .unwrap()
        .run(module.clone())
        .unwrap();
    let cpp_result = cpp_pass("tirx.transform.RemoveAssume").run(module).unwrap();

    assert_structural_equal(&rust_result, &cpp_result);
    let rust_function = rust_result
        .functions
        .iter()
        .next()
        .unwrap()
        .1
        .try_cast::<PrimFunc>()
        .unwrap();
    let evaluate = rust_function.body.clone().try_cast::<Evaluate>().unwrap();
    assert_eq!(
        evaluate.value.clone().try_cast::<IntImm>().unwrap().value,
        0
    );
}
