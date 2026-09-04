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
fn source_and_module_metadata_round_trip_cpp_objects() {
    load_tvm_compiler();
    let source_name = SourceName::get("contract-test.tvm").unwrap();
    let same_source_name = SourceName::get("contract-test.tvm").unwrap();
    let cpp_source_name: SourceName = Function::get_global("ir.SourceName")
        .unwrap()
        .call_packed(&[AnyView::from(&tvm::tvm_ffi::String::from(
            "contract-test.tvm",
        ))])
        .unwrap()
        .try_into()
        .unwrap();
    let span = Span::new(&source_name, 2, 4, 3, 5).unwrap();

    assert_eq!(source_name.name().unwrap().as_str(), "contract-test.tvm");
    assert_eq!(
        object_pointer(&source_name),
        object_pointer(&same_source_name)
    );
    assert_structural_equal(&source_name, &same_source_name);
    assert_eq!(
        object_pointer(&source_name),
        object_pointer(&cpp_source_name)
    );
    assert_structural_equal(&source_name, &cpp_source_name);
    assert_eq!(
        object_pointer(span.source_name.as_ref().unwrap()),
        object_pointer(&source_name)
    );
    assert_eq!(span.line, 2);
    assert_eq!(span.column, 3);
    assert_eq!(span.end_line, 4);
    assert_eq!(span.end_column, 5);
    let complete_span = Span::from_complete_fields(Some(source_name), 11, 22, 33, 44);
    assert_eq!(complete_span.line, 11);
    assert_eq!(complete_span.column, 22);
    assert_eq!(complete_span.end_line, 33);
    assert_eq!(complete_span.end_column, 44);

    let sequential = SequentialSpan::new(vec![span.clone(), complete_span.clone()]);
    assert!(sequential.source_name.is_none());
    assert_eq!(sequential.spans.len(), 2);
    let native_sequential: SequentialSpan = Function::get_global("ir.SequentialSpan")
        .unwrap()
        .call_tuple((Array::new(vec![span.clone(), complete_span]),))
        .unwrap()
        .try_into()
        .unwrap();
    assert_structural_equal(&sequential, &native_sequential);
    let nested = SequentialSpan::new(vec![Span::from(sequential), span]);
    assert_eq!(nested.spans.len(), 3);

    let int_type = PrimType::new("int32").unwrap();
    assert!(int_type.span.is_none());
    let function = PrimFunc::from_body(Evaluate::from_i64(0).unwrap()).unwrap();
    let module = IRModule::from_expr(&function).unwrap();
    assert_eq!(module.functions.len(), 1);
    assert_eq!(module.global_var_map.len(), 1);
    assert_eq!(module.source_map.source_map.len(), 0);

    let mut source_map = SourceMap::new();
    let source_name = source_map
        .add("module.tvm", "first line\nsecond line")
        .unwrap();
    let sources = source_map.source_map.clone();
    let source = sources.get(&source_name).unwrap().unwrap();
    let cpp_lookup_name: SourceName = Function::get_global("ir.SourceName")
        .unwrap()
        .call_packed(&[AnyView::from(&tvm::tvm_ffi::String::from("module.tvm"))])
        .unwrap()
        .try_into()
        .unwrap();
    assert_eq!(
        object_pointer(&sources.get(&cpp_lookup_name).unwrap().unwrap()),
        object_pointer(&source)
    );
    assert_eq!(
        source.source_name().unwrap().name().unwrap().as_str(),
        "module.tvm"
    );
    assert_eq!(source.text().unwrap().as_str(), "first line\nsecond line");

    let dictionary: Map<tvm::tvm_ffi::String, Any> = [
        (tvm::tvm_ffi::String::from("number"), Any::from(7i64)),
        (
            tvm::tvm_ffi::String::from("text"),
            Any::from(tvm::tvm_ffi::String::from("value")),
        ),
    ]
    .into_iter()
    .collect();
    let attrs = DictAttrs::from_dictionary(dictionary);
    let dictionary = attrs.dict.clone();
    assert_eq!(
        i64::try_from(
            dictionary
                .get(&tvm::tvm_ffi::String::from("number"))
                .unwrap()
                .unwrap()
        )
        .unwrap(),
        7
    );
    assert_eq!(
        tvm::tvm_ffi::String::try_from(
            dictionary
                .get(&tvm::tvm_ffi::String::from("text"))
                .unwrap()
                .unwrap()
        )
        .unwrap()
        .as_str(),
        "value"
    );

    assert!(module.global_infos.is_empty());
    let dummy = DummyGlobalInfo::new();
    let updated = module
        .with_updated_global_info("dummy", vec![dummy.clone().into()])
        .unwrap();
    assert!(module.global_infos.is_empty());
    let group = updated
        .global_infos
        .get(&tvm::tvm_ffi::String::from("dummy"))
        .unwrap()
        .unwrap();
    assert_eq!(group.len(), 1);
    assert_eq!(
        object_pointer(&group.get(0).unwrap()),
        object_pointer(&dummy)
    );
}

#[test]
fn full_direct_constructors_preserve_source_spans() {
    load_tvm_compiler();
    let source_name = SourceName::get("constructors.tvm").unwrap();
    let span = Span::new(&source_name, 1, 3, 2, 4).unwrap();
    let int_type: Type = PrimType::new("int32").unwrap().into();
    let missing = Type::missing();
    let same_missing = Type::missing();
    assert!(missing.is_missing());
    assert_eq!(
        object_pointer(&missing),
        object_pointer(&same_missing),
        "the missing type is a native singleton"
    );

    let variable = Var::with_type_and_span("x", &int_type, Some(&span));
    let iter_domain = Range::from_min_extent(
        IntImm::new("int32", 0).unwrap(),
        IntImm::new("int32", 4).unwrap(),
    )
    .unwrap();
    let iter_var = IterVar::with_metadata(
        Some(iter_domain),
        variable.clone(),
        IterVarType::kDataPar,
        "",
        Some(&span),
    )
    .unwrap();
    let literal =
        IntImm::from_dtype_with_span(PrimType::new("int32").unwrap().dtype, 1, Some(&span))
            .unwrap();
    let addition = Add::with_span(variable.clone(), literal.clone(), Some(&span)).unwrap();
    let callee = GlobalVar::with_span("callee", Some(&span));
    let call = Call::with_metadata(
        &int_type,
        callee,
        vec![addition.clone().into()],
        None,
        Vec::new(),
        Some(&span),
    );
    let evaluation = Evaluate::with_span(call.clone(), Some(&span)).unwrap();
    let sequence = SeqStmt::with_span(
        vec![
            evaluation.clone().into(),
            Evaluate::with_span(literal.clone(), Some(&span))
                .unwrap()
                .into(),
        ],
        Some(&span),
    )
    .unwrap();
    for actual in [
        variable.span.as_ref(),
        literal.span.as_ref(),
        addition.span.as_ref(),
        call.span.as_ref(),
        evaluation.span.as_ref(),
        sequence.span.as_ref(),
    ] {
        assert_eq!(object_pointer(actual.unwrap()), object_pointer(&span));
    }
    assert_eq!(
        object_pointer(iter_var.span().unwrap().as_ref().unwrap()),
        object_pointer(&span)
    );
}

#[test]
fn statement_sequence_normalizes_empty_single_and_nested_inputs() {
    load_tvm_compiler();

    let empty = Stmt::sequence(Vec::new()).unwrap();
    let empty = empty.try_cast::<Evaluate>().unwrap();
    assert_eq!(empty.value.clone().try_cast::<IntImm>().unwrap().value, 0);

    let single: Stmt = Evaluate::from_i64(7).unwrap().into();
    let single_pointer = object_pointer(&single);
    let normalized = Stmt::sequence(vec![single]).unwrap();
    assert_eq!(object_pointer(&normalized), single_pointer);

    let canonical = SeqStmt::new(vec![
        Evaluate::from_i64(1).unwrap().into(),
        Evaluate::from_i64(2).unwrap().into(),
    ])
    .unwrap();
    let canonical_pointer = object_pointer(&canonical);
    let flattened = canonical.flatten().unwrap();
    assert_eq!(object_pointer(&flattened), canonical_pointer);

    let nested = SeqStmt::new(vec![
        Evaluate::from_i64(1).unwrap().into(),
        Evaluate::from_i64(0).unwrap().into(),
    ])
    .unwrap();
    let normalized = Stmt::sequence(vec![
        nested.into(),
        Evaluate::from_i64(0).unwrap().into(),
        Evaluate::from_i64(2).unwrap().into(),
    ])
    .unwrap()
    .try_cast::<SeqStmt>()
    .unwrap();
    assert_eq!(normalized.seq.len(), 2);
}

#[test]
fn buffer_defaults_match_the_cpp_constructor() {
    load_tvm_compiler();
    let extent = typed_int_expression("int32", 8);
    let buffer_type = BufferType::new("", "float32", vec![extent.clone()]).unwrap();
    let element_offset = buffer_type
        .elem_offset
        .clone()
        .try_cast::<IntImm>()
        .unwrap();

    assert_eq!(buffer_type.storage_scope.as_str(), "global");
    assert_eq!(element_offset.value, 0);
    assert_eq!(
        element_offset
            .ty
            .clone()
            .try_cast::<PrimType>()
            .unwrap()
            .dtype
            .bits,
        32
    );
    assert!(buffer_type.data_alignment > 0);
    assert_eq!(buffer_type.offset_factor, 1);

    let dtype = PrimType::new("float32").unwrap();
    let normalized_defaults = BufferType::with_metadata(
        "global",
        dtype.clone(),
        vec![extent],
        Vec::new(),
        element_offset.into(),
        0,
        0,
        None,
        Vec::new(),
        None,
    )
    .unwrap();
    assert_eq!(normalized_defaults.data_alignment, 64);
    assert_eq!(normalized_defaults.offset_factor, 1);

    let cpp_buffer: BufferType = Function::get_global("tirx.BufferType")
        .unwrap()
        .call_packed(&[
            AnyView::from(&tvm::tvm_ffi::String::from("")),
            AnyView::from(&dtype),
            AnyView::from(&Array::new(vec![typed_int_expression("int32", 8)])),
            AnyView::from(&Array::<Expr>::new(Vec::new())),
            AnyView::from(&()),
            AnyView::from(&0_i64),
            AnyView::from(&0_i64),
            AnyView::from(&()),
            AnyView::from(&Array::<Expr>::new(Vec::new())),
            AnyView::from(&()),
        ])
        .unwrap()
        .try_into()
        .unwrap();
    assert_structural_equal(&buffer_type, &cpp_buffer);
}

#[test]
fn every_layout_registered_operation_is_callable() {
    load_tvm_compiler();

    let axis = Axis::get("m").unwrap();
    let eight = prim_int_expression(8);
    let one = prim_int_expression(1);
    let three = prim_int_expression(3);
    let tile = TileLayout::new(
        vec![Iter::new(&eight, &one, &axis).unwrap()],
        Vec::new(),
        Map::new(),
    )
    .unwrap();
    let layout: Layout = tile.clone().into();
    let shape = Array::new(vec![eight.clone()]);
    let coordinate = Array::new(vec![three.clone()]);

    assert!(layout.compatible_with_shape(&shape).unwrap());
    assert!(layout.verify_well_formed().unwrap());
    let cpp_verified = Function::get_global("tirx.LayoutVerifyWellFormed")
        .unwrap()
        .call_packed(&[AnyView::from(&layout)])
        .unwrap();
    assert!(bool::try_from(cpp_verified).unwrap());
    assert_structural_equal(&layout.get_size(None).unwrap(), &eight);
    assert_structural_equal(&layout.get_size(Some("m")).unwrap(), &eight);
    assert_structural_equal(&layout.get_span(None).unwrap(), &eight);
    assert_structural_equal(
        &layout
            .apply(&coordinate)
            .unwrap()
            .get(&tvm::tvm_ffi::String::from("m"))
            .unwrap()
            .unwrap(),
        &three,
    );
    assert_structural_equal(
        &layout
            .apply_linear(&three)
            .unwrap()
            .get(&tvm::tvm_ffi::String::from("m"))
            .unwrap()
            .unwrap(),
        &three,
    );
    assert_structural_equal(
        &layout
            .apply_with_shape(&coordinate, &shape)
            .unwrap()
            .get(&tvm::tvm_ffi::String::from("m"))
            .unwrap()
            .unwrap(),
        &three,
    );
    assert_structural_equal(&layout.canonicalize().unwrap(), &layout);

    let compose = ComposeLayout::new(0, 0, 0, tile.clone(), false).unwrap();
    assert_eq!(compose.per_element().unwrap(), 0);
    assert_eq!(compose.swizzle_len().unwrap(), 0);
    assert_eq!(compose.atom_len().unwrap(), 0);
    assert!(!compose.swizzle_inner().unwrap());
    assert_eq!(compose.inner_mask().unwrap(), 0);
    assert_eq!(compose.outer_mask().unwrap(), 0);
    assert!(compose.tile_layout().unwrap().same_as(&tile));
    let compose: Layout = compose.into();
    assert!(compose.verify_well_formed().unwrap());
    assert_structural_equal(
        &compose
            .apply_linear(&three)
            .unwrap()
            .get(&tvm::tvm_ffi::String::from("m"))
            .unwrap()
            .unwrap(),
        &three,
    );

    let tiled = layout.tile(&tile, &shape, &shape).unwrap();
    let tiled_shape = Array::new(vec![prim_int_expression(64)]);
    assert!(layout
        .is_tile_inner(&tiled, &tiled_shape, &shape)
        .unwrap()
        .is_some());
    assert!(layout
        .is_tile_outer(&tiled, &tiled_shape, &shape)
        .unwrap()
        .is_some());

    let region = Array::new(vec![Range::from_min_extent(
        int_expression(2),
        int_expression(3),
    )
    .unwrap()]);
    assert!(layout.slice(&shape, &region).unwrap().is_some());

    let left = TileLayout::new(
        vec![
            Iter::new(int_expression(2), int_expression(8), &axis).unwrap(),
            Iter::new(int_expression(2), int_expression(2), &axis).unwrap(),
        ],
        Vec::new(),
        Map::new(),
    )
    .unwrap();
    let right = TileLayout::new(
        vec![
            Iter::new(int_expression(2), int_expression(4), &axis).unwrap(),
            Iter::new(int_expression(2), one, &axis).unwrap(),
        ],
        Vec::new(),
        Map::new(),
    )
    .unwrap();
    let left_shape = Array::new(vec![prim_int_expression(2), prim_int_expression(2)]);
    let right_shape = left_shape.clone();
    let right_layout: Layout = right.into();
    let direct_sum = right_layout
        .direct_sum(&left, &left_shape, &right_shape)
        .unwrap();
    let interleaved_shape = Array::new(vec![
        prim_int_expression(2),
        prim_int_expression(2),
        prim_int_expression(2),
        prim_int_expression(2),
    ]);
    assert!(right_layout
        .is_direct_sum_right(&direct_sum, &interleaved_shape, &right_shape)
        .unwrap()
        .is_some());
    let left_layout: Layout = left.into();
    assert!(left_layout
        .is_direct_sum_left(&direct_sum, &interleaved_shape, &left_shape)
        .unwrap()
        .is_some());
}

#[test]
fn index_maps_and_tensor_intrinsics_cross_the_native_abi() {
    load_tvm_compiler();

    let index = PrimVar::try_from(Var::new("i", "int32").unwrap()).unwrap();
    let identity_map = IndexMap::new(vec![index.clone()], vec![index.clone().into()], None);
    assert_eq!(
        identity_map
            .map_shape(vec![prim_int_expression(4)], None)
            .unwrap()
            .len(),
        1
    );
    let one = IntImm::new("int32", 1).unwrap();
    let output = Add::new(index.clone(), one).unwrap();
    let index_map = IndexMap::new(vec![index], vec![output.into()], None);
    let mapped = index_map
        .map_indices(vec![prim_int_expression(3)], None)
        .unwrap();
    assert_structural_equal(&mapped.get(0).unwrap(), &prim_int_expression(4));
    let domain = vec![Range::from_min_extent(int_expression(0), int_expression(4)).unwrap()];
    let inverse = index_map.inverse(domain.clone(), None).unwrap();
    let recovered = inverse
        .map_indices(vec![prim_int_expression(4)], None)
        .unwrap();
    assert_structural_equal(&recovered.get(0).unwrap(), &prim_int_expression(3));
    let (non_surjective_inverse, _) = index_map.non_surjective_inverse(domain, None).unwrap();
    assert_eq!(non_surjective_inverse.initial_indices.len(), 1);

    let handle_type = PointerType::new(PrimType::void(), "global").unwrap();
    let parameter = Var::with_type("data", handle_type);
    let function = PrimFunc::new(vec![parameter], Evaluate::from_i64(0).unwrap()).unwrap();
    let intrinsic = TensorIntrin::new(function.clone(), function).unwrap();
    intrinsic
        .register("testing.rust_tensor_intrin", true)
        .unwrap();
    assert!(TensorIntrin::get("testing.rust_tensor_intrin")
        .unwrap()
        .same_as(&intrinsic));
    assert!(TensorIntrin::try_get("testing.missing_tensor_intrin")
        .unwrap()
        .is_none());

    let invalid = PrimFunc::new(
        vec![Var::new("scalar", "int32").unwrap()],
        Evaluate::from_i64(0).unwrap(),
    )
    .unwrap();
    assert!(TensorIntrin::new(invalid.clone(), invalid).is_err());
}

#[test]
fn rust_skip_assert_rebuilds_conditional_branches_like_cpp() {
    load_tvm_compiler();
    let condition = typed_int_expression("bool", 1);
    let assertion = || -> Stmt {
        AssertStmt::new(&condition, "RuntimeError", "failed")
            .unwrap()
            .into()
    };
    let then_case: Stmt = SeqStmt::new(vec![
        assertion(),
        Evaluate::new(int_expression(1)).unwrap().into(),
    ])
    .unwrap()
    .into();
    let else_case: Stmt = SeqStmt::new(vec![
        assertion(),
        Evaluate::new(int_expression(2)).unwrap().into(),
    ])
    .unwrap()
    .into();
    let conditional = IfThenElse::with_span(&condition, &then_case, Some(else_case), None).unwrap();
    assert!(conditional.else_case.is_some());

    let function = PrimFunc::from_body(&conditional).unwrap();
    let module = IRModule::from_expr(&function).unwrap();
    let rust_result = transform::skip_assert()
        .unwrap()
        .run(module.clone())
        .unwrap();
    let cpp_result = cpp_pass("tirx.transform.SkipAssert").run(module).unwrap();

    let statistics = node_statistics(&rust_result).unwrap();
    assert_eq!(statistics.assertions, 0);
    assert_eq!(statistics.conditionals, 1);
    assert_eq!(statistics.evaluations, 2);
    assert_structural_equal(&rust_result, &cpp_result);
}

#[test]
fn rust_skip_assert_does_not_rewrite_for_annotations() {
    load_tvm_compiler();
    let condition = typed_int_expression("bool", 1);
    let hidden_assertion: Stmt = AssertStmt::new(&condition, "RuntimeError", "metadata")
        .unwrap()
        .into();
    let annotation_key = tvm::tvm_ffi::String::from("test.metadata");
    let annotations = Map::from_iter([(annotation_key.clone(), Any::from(hidden_assertion))]);
    let loop_var = Var::new("i", "int32").unwrap();
    let loop_node = For::with_metadata(
        loop_var,
        int_expression(0),
        int_expression(1),
        ForKind::kSerial,
        Evaluate::from_i64(1).unwrap().into(),
        None,
        annotations,
        None,
        None,
    )
    .unwrap();
    let module = IRModule::from_expr(PrimFunc::from_body(loop_node).unwrap()).unwrap();

    let rust_result = transform::skip_assert()
        .unwrap()
        .run(module.clone())
        .unwrap();
    let cpp_result = cpp_pass("tirx.transform.SkipAssert").run(module).unwrap();

    assert_structural_equal(&rust_result, &cpp_result);
    let function = rust_result
        .functions
        .iter()
        .next()
        .unwrap()
        .1
        .try_cast::<PrimFunc>()
        .unwrap();
    let loop_node = function.body.clone().try_cast::<For>().unwrap();
    let annotation = loop_node.annotations.get(&annotation_key).unwrap().unwrap();
    assert!(AssertStmt::try_from(annotation).is_ok());
}

#[test]
fn decorate_device_scope_matches_cpp() {
    load_tvm_compiler();
    let function = PrimFunc::from_body(Evaluate::from_i64(7).unwrap()).unwrap();
    let module = IRModule::from_expr(&function).unwrap();

    let rust_result = transform::decorate_device_scope()
        .unwrap()
        .run(module.clone())
        .unwrap();
    let cpp_result = cpp_pass("s_tir.transform.DecorateDeviceScope")
        .run(module)
        .unwrap();
    assert_structural_equal(&rust_result, &cpp_result);

    let mapped_function = rust_result
        .functions
        .iter()
        .next()
        .unwrap()
        .1
        .try_cast::<PrimFunc>()
        .unwrap();
    let attribute = mapped_function.body.clone().try_cast::<AttrStmt>().unwrap();
    assert_eq!(attribute.attr_key.as_str(), "device_scope");
    assert_eq!(i64::try_from(attribute.node.clone()).unwrap(), 0);
    assert_eq!(
        attribute.value.clone().try_cast::<IntImm>().unwrap().value,
        0
    );
}

fn nested_loop_statement() -> Stmt {
    let outer = Var::new("i", "int32").unwrap();
    let inner = Var::new("j", "int32").unwrap();
    let sum = Add::new(outer.clone(), inner.clone()).unwrap();
    let inner_body: Stmt = Evaluate::new(Expr::from(sum)).unwrap().into();
    let inner_loop: Stmt = For::new(&inner, int_expression(1), int_expression(3), &inner_body)
        .unwrap()
        .into();
    For::new(&outer, int_expression(0), int_expression(4), &inner_loop)
        .unwrap()
        .into()
}

#[test]
fn walk_and_visit_handle_real_tir_loop_scopes() {
    load_tvm_compiler();
    let statement = nested_loop_statement();

    assert!(statement
        .clone()
        .try_cast::<For>()
        .unwrap()
        .thread_binding
        .is_none());

    let statistics = node_statistics(&statement).unwrap();
    assert_eq!(statistics.loops, 2);
    assert_eq!(statistics.statements, 3);
    assert_eq!(statistics.additions, 1);
    assert_eq!(statistics.variable_definitions, 2);
    assert_eq!(statistics.variable_uses, 2);
    let nesting = loop_nesting(&statement).unwrap();
    assert_eq!(nesting.loops, 2);
    assert_eq!(nesting.maximum_depth, 2);
}

#[test]
fn buffer_bindings_round_trip_cpp_objects() {
    load_tvm_compiler();

    // A C++-created Tensor is consumed through its OpaqueExpr base, then used
    // as the callee of TVM's standard tensor-load operation.
    let tensor_dtype = PrimType::new("float32").unwrap();
    let tensor: OpaqueExpr = Function::get_global("te.Placeholder")
        .unwrap()
        .call_packed(&[
            AnyView::from(&Array::<Expr>::new(Vec::new())),
            AnyView::from(&tensor_dtype),
            AnyView::from(&tvm::tvm_ffi::String::from("scalar_input")),
        ])
        .unwrap()
        .try_into()
        .unwrap();
    let tensor_expr: PrimExpr = Function::get_global("te.TensorLoad")
        .unwrap()
        .call_tuple((&tensor, Array::<PrimExpr>::new(Vec::new())))
        .unwrap()
        .try_into()
        .unwrap();
    assert_eq!(
        tensor_expr.ty.clone().try_cast::<PrimType>().unwrap().dtype,
        tensor_dtype.dtype
    );

    let extent = typed_int_expression("int64", 8);
    let stride = typed_int_expression("int64", 1);
    let axis_name = Axis::get("m").unwrap();
    let layout_iter = Iter::new(&extent, &stride, &axis_name).unwrap();
    let tile_layout = TileLayout::new(vec![layout_iter], Vec::new(), Map::new()).unwrap();
    let layout = Layout::from(tile_layout);
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
    let reflected_layout = buffer_type
        .layout
        .as_ref()
        .unwrap()
        .clone()
        .try_cast::<TileLayout>()
        .unwrap();
    let layout: tvm::tirx::Layout = reflected_layout.clone().into();
    let layout_is_valid = layout.verify_well_formed().unwrap();
    assert!(layout_is_valid);
    assert_eq!(reflected_layout.shard().unwrap().len(), 1);
    assert!(reflected_layout.replica().unwrap().is_empty());
    assert!(reflected_layout.offset().unwrap().is_empty());
    let reflected_iter = reflected_layout.shard().unwrap().get(0).unwrap();
    assert_eq!(
        reflected_iter
            .extent
            .clone()
            .try_cast::<IntImm>()
            .unwrap()
            .value,
        8
    );
    assert_eq!(
        reflected_iter
            .stride
            .clone()
            .try_cast::<IntImm>()
            .unwrap()
            .value,
        1
    );
    assert_eq!(reflected_iter.axis.name().unwrap().as_str(), "m");
    let buffer = buffer_type.new_var("A");
    let axis = Var::new("vi", "int64").unwrap();
    let axis_domain = Range::from_min_extent(
        typed_int_expression("int64", 0),
        typed_int_expression("int64", 8),
    )
    .unwrap();
    let iter_var = IterVar::new(&axis_domain, &axis).unwrap();
    let converted_axis = PrimExprConvertible::from(iter_var.clone())
        .to_prim_expr()
        .unwrap();
    assert_eq!(object_pointer(&converted_axis), object_pointer(&axis));
    let converted_by_cpp: Add = Function::get_global("ir.prim.Add")
        .unwrap()
        .call_packed(&[
            AnyView::from(&iter_var),
            AnyView::from(&typed_int_expression("int64", 1)),
            AnyView::from(&()),
        ])
        .unwrap()
        .try_into()
        .unwrap();
    assert_eq!(object_pointer(&converted_by_cpp.a), object_pointer(&axis));
    let domainless_iter = IterVar::with_metadata(
        None,
        axis.clone(),
        IterVarType::kThreadIndex,
        "threadIdx.x",
        None,
    )
    .unwrap();
    assert!(domainless_iter.dom().unwrap().is_none());
    let converted_domainless = PrimExprConvertible::from(domainless_iter)
        .to_prim_expr()
        .unwrap();
    assert_eq!(object_pointer(&converted_domainless), object_pointer(&axis));
    assert!(TensorLoad::from_buffer(&axis, vec![typed_int_expression("int64", 0)]).is_err());
    let load = TensorLoad::from_buffer(&buffer, vec![axis.clone().into()]).unwrap();
    let explicit_load_type = PrimType::new("int32").unwrap();
    let complete_load = TensorLoad::from_complete_fields(
        None,
        explicit_load_type.clone(),
        buffer.as_var().clone().into(),
        load.indices.clone(),
    );
    let explicit_load_type: Type = explicit_load_type.into();
    assert_eq!(
        object_pointer(&complete_load.ty),
        object_pointer(&explicit_load_type)
    );
    let store = BufferStore::new(&buffer, load.clone(), vec![axis.clone().into()]).unwrap();
    let cpp_indices = tvm::tvm_ffi::Array::new(load.indices.iter().collect());
    let cpp_load: TensorLoad = Function::get_global("tirx.BufferLoad")
        .unwrap()
        .call_packed(&[
            AnyView::from(&buffer),
            AnyView::from(&cpp_indices),
            AnyView::from(&()),
        ])
        .unwrap()
        .try_into()
        .unwrap();
    assert_structural_equal(&load, &cpp_load);

    let load_expr: Expr = load.clone().into();
    let cpp_store: BufferStore = Function::get_global("tirx.BufferStore")
        .unwrap()
        .call_packed(&[
            AnyView::from(&buffer),
            AnyView::from(&load_expr),
            AnyView::from(&cpp_indices),
            AnyView::from(&()),
        ])
        .unwrap()
        .try_into()
        .unwrap();
    assert_structural_equal(&store, &cpp_store);

    let region = BufferRegion::new(&buffer, vec![axis_domain]).unwrap();
    let _: Expr = region.clone().into();
    let _: BufferRegionType = region.ty.clone().try_cast().unwrap();
    let cpp_region: BufferRegion = Function::get_global("tirx.BufferRegion")
        .unwrap()
        .call_tuple((&buffer, &region.region))
        .unwrap()
        .try_into()
        .unwrap();
    assert_structural_equal(&region, &cpp_region);
    let match_buffer = MatchBufferRegion::new(&buffer, &region).unwrap();
    let cpp_match_buffer: MatchBufferRegion = Function::get_global("tirx.MatchBufferRegion")
        .unwrap()
        .call_packed(&[AnyView::from(&buffer), AnyView::from(&region)])
        .unwrap()
        .try_into()
        .unwrap();
    assert_structural_equal(&match_buffer, &cpp_match_buffer);
    assert_eq!(
        object_pointer(&match_buffer.buffer),
        object_pointer(&buffer)
    );
    assert_eq!(
        object_pointer(&match_buffer.source),
        object_pointer(&region)
    );
    let function = PrimFunc::new(vec![buffer.clone().into()], store.clone()).unwrap();

    assert_eq!(buffer_type.dtype.dtype.bits, 32);
    assert_eq!(buffer_type.storage_scope.as_str(), "global");
    assert_eq!(buffer_type.shape.len(), 1);
    assert!(buffer_type.strides.is_empty());
    assert!(buffer_type.data_alignment > 0);
    assert_eq!(buffer_type.offset_factor, 1);
    assert!(buffer_type.allocated_addr.is_empty());
    assert_eq!(
        buffer
            .ty
            .clone()
            .try_cast::<BufferType>()
            .unwrap()
            .shape
            .len(),
        1
    );
    assert_eq!(object_pointer(&load.source), object_pointer(&buffer));
    assert_eq!(load.indices.len(), 1);
    assert_eq!(object_pointer(&store.buffer), object_pointer(&buffer));
    assert_eq!(iter_var.iter_type().unwrap(), IterVarType::kDataPar);
    assert_eq!(
        object_pointer(&iter_var.var().unwrap()),
        object_pointer(&axis)
    );
    assert_eq!(
        iter_var
            .dom()
            .unwrap()
            .as_ref()
            .unwrap()
            .extent
            .clone()
            .try_cast::<IntImm>()
            .unwrap()
            .value,
        8
    );

    let statistics = node_statistics(&function).unwrap();
    assert_eq!(statistics.buffer_loads, 1);
    assert_eq!(statistics.buffer_stores, 1);
    assert_eq!(
        memory_access_statistics(&function).unwrap(),
        tvm::analysis::MemoryAccessStatistics {
            loads: 1,
            stores: 1,
            maximum_load_rank: 1,
            maximum_store_rank: 1,
        }
    );
}

#[test]
fn rust_unit_loop_elimination_matches_cpp_on_buffer_indices() {
    load_tvm_compiler();
    let buffer_type =
        BufferType::new("global", "int32", vec![typed_int_expression("int64", 16)]).unwrap();
    let buffer = buffer_type.new_var("A");
    let outer_var = Var::new("i", "int64").unwrap();
    let unit_var = Var::new("j", "int64").unwrap();
    let index: Expr = Add::new(outer_var.clone(), unit_var.clone())
        .unwrap()
        .into();
    let load: Expr = TensorLoad::from_buffer(&buffer, vec![index.clone()])
        .unwrap()
        .into();
    let store: Stmt = BufferStore::new(&buffer, &load, vec![index])
        .unwrap()
        .into();
    let unit_loop: Stmt = For::new(
        &unit_var,
        typed_int_expression("int64", 2),
        typed_int_expression("int64", 1),
        &store,
    )
    .unwrap()
    .into();
    let outer_loop = For::new(
        &outer_var,
        typed_int_expression("int64", 0),
        typed_int_expression("int64", 4),
        &unit_loop,
    )
    .unwrap();
    let function = PrimFunc::new(vec![buffer.into()], &outer_loop).unwrap();
    let module = IRModule::from_expr(&function).unwrap();

    let rust_function = transform::lower_tirx_opaque_prim_func(function).unwrap();
    let rust_result = IRModule::from_expr(&rust_function).unwrap();
    let cpp_result = cpp_pass("tirx.transform.LowerTIRxOpaque")
        .run(module.clone())
        .unwrap();

    assert_eq!(node_statistics(&module).unwrap().loops, 2);
    assert_eq!(node_statistics(&rust_result).unwrap().loops, 1);
    assert_structural_equal(&rust_result, &cpp_result);

    let mapped_function = rust_result
        .functions
        .iter()
        .next()
        .unwrap()
        .1
        .try_cast::<PrimFunc>()
        .unwrap();
    let mapped_outer = mapped_function.body.clone().try_cast::<For>().unwrap();
    let mapped_store = mapped_outer.body.clone().try_cast::<BufferStore>().unwrap();
    let mapped_index = mapped_store
        .indices
        .get(0)
        .unwrap()
        .try_cast::<Add>()
        .unwrap();
    assert_eq!(
        mapped_index.b.clone().try_cast::<IntImm>().unwrap().value,
        2
    );
    assert_eq!(memory_access_statistics(&rust_result).unwrap().loads, 1);
    assert_eq!(memory_access_statistics(&rust_result).unwrap().stores, 1);
}

#[test]
fn call_effect_kind_preserves_future_native_values() {
    let future_value = CallEffectKind::from_raw(17);
    assert_eq!(future_value.as_raw(), 17);
    assert!(future_value.may_update_state());
    assert_eq!(CallEffectKind::try_from(17).unwrap(), future_value);
    assert!(CallEffectKind::try_from(i64::MAX).is_err());
}
