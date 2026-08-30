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

use tvm::analysis::{
    contains_int, expression_trace, first_int, loop_nesting, memory_access_statistics,
    node_statistics, side_effect, CallEffectKind, ExprTraceEvent,
};
use tvm::ir::{
    BaseFunc, Call, DictAttrs, DummyGlobalInfo, Expr, GlobalVar, IRModule, IntImm, OpaqueExpr,
    PrimExpr, PrimExprConvertible, PrimType, Range, SourceMap, SourceName, Span, Type, Var,
};
use tvm::tirx::{
    Add, AddObj, And, AssertStmt, AssertStmtObj, AttrStmt, Axis, BufferLoad, BufferRegion,
    BufferStore, BufferType, Evaluate, EvaluateObj, For as TirFor, IfThenElse, Iter, IterVar,
    IterVarType, Layout, MatchBufferRegion, Mul, PrimFunc, SBlock, SBlockRealize, SeqStmt, Stmt,
    Sub, TileLayout, EQ,
};
use tvm::transform;
use tvm::tvm_ffi::{
    dispatch, structural_map, structural_walk, Any, AnyView, Array, DefRegionKind, Function, Map,
    ObjectRefCast, Result, WalkOrder, WalkResult,
};

mod common;
use common::{assert_structural_equal, load_tvm_compiler, object_pointer};

fn node_counts<R>(root: &R) -> (usize, usize)
where
    for<'a> AnyView<'a>: From<&'a R>,
{
    let mut asserts = 0;
    let mut evaluates = 0;
    structural_walk(
        root,
        (
            |_: &AssertStmtObj| {
                asserts += 1;
                WalkResult::Advance
            },
            |_: &EvaluateObj| {
                evaluates += 1;
                WalkResult::Advance
            },
        ),
        WalkOrder::PreOrder,
    )
    .unwrap();
    (asserts, evaluates)
}

fn int_expression(value: i64) -> Expr {
    typed_int_expression("int32", value)
}

fn typed_int_expression(dtype: &str, value: i64) -> Expr {
    IntImm::new(dtype, value).unwrap().into()
}

fn prim_int_expression(value: i64) -> PrimExpr {
    IntImm::new("int32", value).unwrap().into()
}

fn add_expression<L, R>(lhs: L, rhs: R) -> Expr
where
    L: Into<Expr>,
    R: Into<Expr>,
{
    Add::new(lhs, rhs).unwrap().into()
}

fn sample_sum() -> Expr {
    let one = int_expression(1);
    let two = int_expression(2);
    let three = int_expression(3);
    add_expression(add_expression(&one, &two), &three)
}

fn cpp_pass(name: &str) -> transform::Pass {
    Function::get_global(name)
        .unwrap()
        .call_packed(&[])
        .unwrap()
        .try_into()
        .unwrap()
}

fn has_nonzero_function_attr(function: &BaseFunc, key: &str) -> bool {
    function
        .attrs
        .dict
        .get(&tvm::tvm_ffi::String::from(key))
        .unwrap()
        .map(i64::try_from)
        .transpose()
        .unwrap()
        .unwrap_or(0)
        != 0
}

fn module_from_named_prim_funcs(functions: Vec<(&str, PrimFunc)>) -> IRModule {
    IRModule::with_metadata(
        functions
            .into_iter()
            .map(|(name, function)| (GlobalVar::new(name), BaseFunc::from(function)))
            .collect(),
        SourceMap::new(),
        DictAttrs::empty(),
        Map::new(),
    )
    .unwrap()
}

fn prim_func_with_global_symbol(value: i64, symbol: Option<&str>) -> PrimFunc {
    let attrs = symbol
        .map(|symbol| {
            DictAttrs::from_dictionary(Map::from_iter([(
                tvm::tvm_ffi::String::from("global_symbol"),
                Any::from(tvm::tvm_ffi::String::from(symbol)),
            )]))
        })
        .unwrap_or_else(DictAttrs::empty);
    PrimFunc::with_metadata(
        Vec::new(),
        Evaluate::from_i64(value).unwrap(),
        Type::missing(),
        attrs,
        None,
    )
    .unwrap()
}

fn prim_func_body_integer(function: PrimFunc) -> Result<i64> {
    Ok(function
        .body
        .clone()
        .try_cast::<Evaluate>()?
        .value
        .clone()
        .try_cast::<IntImm>()?
        .value)
}

#[test]
fn structural_walk_honors_pre_and_post_order() {
    load_tvm_compiler();
    let sum = sample_sum();

    assert_eq!(
        expression_trace(&sum, WalkOrder::PreOrder).unwrap(),
        vec![
            ExprTraceEvent::Add,
            ExprTraceEvent::Add,
            ExprTraceEvent::Int(1),
            ExprTraceEvent::Int(2),
            ExprTraceEvent::Int(3),
        ]
    );
    assert_eq!(
        expression_trace(&sum, WalkOrder::PostOrder).unwrap(),
        vec![
            ExprTraceEvent::Int(1),
            ExprTraceEvent::Int(2),
            ExprTraceEvent::Add,
            ExprTraceEvent::Int(3),
            ExprTraceEvent::Add,
        ]
    );
}

#[test]
fn structural_walk_interrupts_with_the_requested_payload() {
    load_tvm_compiler();
    let sum = sample_sum();

    assert!(contains_int(&sum, 2).unwrap());
    assert!(!contains_int(&sum, 9).unwrap());
    assert_eq!(first_int(&sum).unwrap(), Some(1));
    assert_eq!(
        first_int(&GlobalVar::new("without_literals")).unwrap(),
        None
    );
}

#[test]
fn structural_walk_skip_prunes_only_the_selected_subtree() {
    load_tvm_compiler();
    let left = add_expression(int_expression(1), int_expression(2));
    let root = add_expression(&left, int_expression(3));
    let left_pointer = object_pointer(&left);
    let mut seen_additions = 0;
    let mut seen_integers = Vec::new();

    structural_walk(
        &root,
        (
            |node: &AddObj| {
                seen_additions += 1;
                if node as *const AddObj as *const () == left_pointer {
                    WalkResult::Skip
                } else {
                    WalkResult::Advance
                }
            },
            |node: &tvm::ir::IntImmObj| -> Result<WalkResult> {
                seen_integers.push(node.value);
                Ok(WalkResult::Advance)
            },
        ),
        WalkOrder::PreOrder,
    )
    .unwrap();

    assert_eq!(seen_additions, 2);
    assert_eq!(seen_integers, vec![3]);
}

#[test]
fn direct_fields_borrow_rust_allocated_nodes() {
    load_tvm_compiler();
    let one = IntImm::new("int32", 1).unwrap();
    let two = IntImm::new("int32", 2).unwrap();
    let add = Add::new(one.clone(), two.clone()).unwrap();
    let assertion = AssertStmt::new(typed_int_expression("bool", 1), "ValueError", "bad").unwrap();
    let leaf_conditional = IfThenElse::new(
        typed_int_expression("bool", 1),
        Evaluate::from_i64(1).unwrap(),
        None,
    )
    .unwrap();

    assert_eq!(one.value, 1);
    assert_eq!(
        one.ty
            .clone()
            .try_cast::<tvm::ir::PrimType>()
            .unwrap()
            .dtype
            .bits,
        32
    );
    assert_eq!(add.a.clone().try_cast::<IntImm>().unwrap().value, 1);
    assert_eq!(add.b.clone().try_cast::<IntImm>().unwrap().value, 2);
    assert_eq!(assertion.error_kind.value.as_str(), "ValueError");
    assert_eq!(
        assertion.message_parts.get(0).unwrap().value.as_str(),
        "bad"
    );
    assert_eq!(
        leaf_conditional
            .condition
            .clone()
            .try_cast::<IntImm>()
            .unwrap()
            .value,
        1
    );
    assert!(leaf_conditional
        .then_case
        .clone()
        .try_cast::<Evaluate>()
        .is_ok());
    assert!(leaf_conditional.else_case.is_none());
}

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
        object_pointer(&span.source_name),
        object_pointer(&source_name)
    );
    assert_eq!(span.line, 2);
    assert_eq!(span.column, 3);
    assert_eq!(span.end_line, 4);
    assert_eq!(span.end_column, 5);
    let complete_span = Span::from_complete_fields(source_name.clone(), 11, 22, 33, 44);
    assert_eq!(complete_span.line, 11);
    assert_eq!(complete_span.column, 22);
    assert_eq!(complete_span.end_line, 33);
    assert_eq!(complete_span.end_column, 44);

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
        IterVarType::DataParallel,
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
fn complete_field_allocators_preserve_supplied_inherited_fields() {
    load_tvm_compiler();
    let missing = Type::missing();
    let lhs = typed_int_expression("int32", 1);
    let rhs = typed_int_expression("int32", 2);

    // These deliberately supplied annotations differ from the convenience
    // constructors' derived defaults.  A lossless stubgen path must store them
    // verbatim instead of silently invoking semantic construction again.
    let global = GlobalVar::from_complete_fields(None, missing.clone(), "typed".into());
    assert_eq!(object_pointer(&global.ty), object_pointer(&missing));

    let explicit_add_type = PrimType::new("int32").unwrap();
    let addition = Add::from_complete_fields(
        None,
        explicit_add_type.clone(),
        PrimExpr::try_from(lhs.clone()).unwrap(),
        PrimExpr::try_from(rhs.clone()).unwrap(),
    );
    assert_eq!(
        object_pointer(&addition.ty),
        object_pointer(&explicit_add_type)
    );
}

#[test]
fn nested_sequence_construction_matches_cpp_flattening() {
    load_tvm_compiler();
    let first = Evaluate::from_i64(1).unwrap();
    let noop = Evaluate::from_i64(0).unwrap();
    let second = Evaluate::from_i64(2).unwrap();
    let nested = SeqStmt::new(vec![first.clone().into(), noop.into()]).unwrap();
    let rust_sequence = SeqStmt::new(vec![nested.into(), second.clone().into()]).unwrap();

    let cpp_input = tvm::tvm_ffi::Array::<Stmt>::new(vec![
        SeqStmt::new(vec![first.into(), Evaluate::from_i64(0).unwrap().into()])
            .unwrap()
            .into(),
        second.into(),
    ]);
    let cpp_sequence = tvm::tvm_ffi::Function::get_global("tirx.SeqStmt")
        .unwrap()
        .call_packed(&[AnyView::from(&cpp_input), AnyView::from(&())])
        .and_then(Stmt::try_from)
        .unwrap();

    assert_eq!(rust_sequence.seq.len(), 2);
    assert_structural_equal(&rust_sequence, &cpp_sequence);
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

    // ComposeLayout is not part of the handwritten Rust slice, but it shares
    // the reflected Layout method contract. Construct a no-op swizzle through
    // the reference API and verify its concrete methods.
    let compose: Layout = Function::get_global("tirx.ComposeLayout")
        .unwrap()
        .call_packed(&[
            AnyView::from(&0_i64),
            AnyView::from(&0_i64),
            AnyView::from(&0_i64),
            AnyView::from(&tile),
            AnyView::from(&false),
        ])
        .unwrap()
        .try_into()
        .unwrap();
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
    let right_layout: Layout = right.clone().into();
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
fn rust_skip_assert_matches_the_cpp_pass() {
    load_tvm_compiler();
    let condition = typed_int_expression("bool", 1);
    let assertion: Stmt = AssertStmt::new(&condition, "RuntimeError", "failed")
        .unwrap()
        .into();
    let evaluation: Stmt = Evaluate::new(sample_sum()).unwrap().into();
    let body = SeqStmt::new(vec![assertion, evaluation]).unwrap();
    let function = PrimFunc::from_body(&body).unwrap();
    let module = IRModule::from_expr(&function).unwrap();

    let rust_result = transform::skip_assert()
        .unwrap()
        .run(module.clone())
        .unwrap();
    assert_eq!(node_counts(&module), (1, 1));
    let cpp_pass = cpp_pass("tirx.transform.SkipAssert");
    let cpp_result = tvm::transform::Pass::run(&cpp_pass, module).unwrap();

    assert_eq!(node_counts(&rust_result), (0, 1));
    assert_structural_equal(&rust_result, &cpp_result);
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
    let conditional = IfThenElse::new(&condition, &then_case, Some(else_case)).unwrap();
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

#[test]
fn lower_init_block_matches_cpp() {
    load_tvm_compiler();
    let first_axis = Var::new("k", "int64").unwrap();
    let first_domain = Range::from_min_extent(
        typed_int_expression("int64", 0),
        typed_int_expression("int64", 8),
    )
    .unwrap();
    let second_axis = Var::new("l", "int64").unwrap();
    let second_domain = Range::from_min_extent(
        typed_int_expression("int64", 2),
        typed_int_expression("int64", 4),
    )
    .unwrap();
    let reduction_block = SBlock::with_metadata(
        vec![
            IterVar::new(
                &first_domain,
                &first_axis,
                IterVarType::CommutativeReduction,
            )
            .unwrap(),
            IterVar::new(
                &second_domain,
                &second_axis,
                IterVarType::CommutativeReduction,
            )
            .unwrap(),
        ],
        Vec::new(),
        Vec::new(),
        "reduction",
        Evaluate::from_i64(2).unwrap().into(),
        Some(Evaluate::from_i64(1).unwrap().into()),
        Vec::new(),
        Vec::new(),
        Map::new(),
        None,
    );
    let reduction_realize = SBlockRealize::new(
        vec![
            typed_int_expression("int64", 3),
            typed_int_expression("int64", 4),
        ],
        typed_int_expression("bool", 1),
        reduction_block,
    )
    .unwrap();

    let data_axis = Var::new("i", "int64").unwrap();
    let data_domain = Range::from_min_extent(
        typed_int_expression("int64", 0),
        typed_int_expression("int64", 8),
    )
    .unwrap();
    let data_block = SBlock::with_metadata(
        vec![IterVar::new(&data_domain, &data_axis, IterVarType::DataParallel).unwrap()],
        Vec::new(),
        Vec::new(),
        "data_parallel",
        Evaluate::from_i64(4).unwrap().into(),
        Some(Evaluate::from_i64(3).unwrap().into()),
        Vec::new(),
        Vec::new(),
        Map::new(),
        None,
    );
    let data_realize = SBlockRealize::new(
        vec![typed_int_expression("int64", 5)],
        typed_int_expression("bool", 1),
        data_block,
    )
    .unwrap();
    let no_init_realize = SBlockRealize::new(
        Vec::new(),
        typed_int_expression("bool", 1),
        SBlock::new("no_init", Evaluate::from_i64(6).unwrap()),
    )
    .unwrap();
    let module = module_from_named_prim_funcs(vec![
        (
            "reduction",
            PrimFunc::from_body(&reduction_realize).unwrap(),
        ),
        ("data_parallel", PrimFunc::from_body(&data_realize).unwrap()),
        ("no_init", PrimFunc::from_body(&no_init_realize).unwrap()),
    ]);

    let rust_result = transform::lower_init_block()
        .unwrap()
        .run(module.clone())
        .unwrap();
    let cpp_result = cpp_pass("s_tir.transform.LowerInitBlock")
        .run(module)
        .unwrap();
    assert_structural_equal(&rust_result, &cpp_result);

    let mapped_function = rust_result
        .functions
        .iter()
        .find(|(global, _)| global.name_hint.as_str() == "reduction")
        .unwrap()
        .1
        .try_cast::<PrimFunc>()
        .unwrap();
    let mapped_realize = mapped_function
        .body
        .clone()
        .try_cast::<SBlockRealize>()
        .unwrap();
    assert!(mapped_realize.block.init.is_none());
    let sequence = mapped_realize
        .block
        .body
        .clone()
        .try_cast::<SeqStmt>()
        .unwrap();
    let conditional = sequence
        .seq
        .get(0)
        .unwrap()
        .try_cast::<IfThenElse>()
        .unwrap();
    let condition = conditional.condition.clone().try_cast::<And>().unwrap();
    assert!(condition.a.clone().try_cast::<EQ>().is_ok());
    assert!(condition.b.clone().try_cast::<EQ>().is_ok());
}

#[test]
fn convert_blocks_to_opaque_matches_cpp() {
    load_tvm_compiler();
    let buffer_type =
        BufferType::new("global", "int32", vec![typed_int_expression("int64", 8)]).unwrap();
    let buffer = buffer_type.new_var("A");
    let axis = Var::new("vi", "int64").unwrap();
    let domain = Range::from_min_extent(
        typed_int_expression("int64", 0),
        typed_int_expression("int64", 8),
    )
    .unwrap();
    let iter_var = IterVar::new(&domain, &axis, IterVarType::DataParallel).unwrap();
    let load = BufferLoad::new(&buffer, vec![axis.clone().into()], None).unwrap();
    let store = BufferStore::new(&buffer, load, vec![axis.clone().into()], None).unwrap();
    let region = BufferRegion::new(
        &buffer,
        vec![Range::from_min_extent(axis.clone(), typed_int_expression("int64", 1)).unwrap()],
    )
    .unwrap();
    let block = SBlock::with_metadata(
        vec![iter_var],
        vec![region.clone()],
        vec![region],
        "elementwise",
        store.into(),
        None,
        Vec::new(),
        Vec::new(),
        Map::new(),
        None,
    );
    let realize = SBlockRealize::new(
        vec![typed_int_expression("int64", 4)],
        typed_int_expression("bool", 1),
        block,
    )
    .unwrap();
    let function = PrimFunc::new(vec![buffer.into()], &realize).unwrap();
    let module = IRModule::from_expr(&function).unwrap();

    let rust_result = transform::convert_blocks_to_opaque()
        .unwrap()
        .run(module.clone())
        .unwrap();
    let cpp_result = cpp_pass("s_tir.transform.ConvertBlocksToOpaque")
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
    let mapped_realize = mapped_function
        .body
        .clone()
        .try_cast::<SBlockRealize>()
        .unwrap();
    assert!(mapped_realize.iter_values.is_empty());
    assert!(mapped_realize.block.iter_vars.is_empty());
    let store = mapped_realize
        .block
        .body
        .clone()
        .try_cast::<BufferStore>()
        .unwrap();
    assert_eq!(
        store
            .indices
            .get(0)
            .unwrap()
            .try_cast::<IntImm>()
            .unwrap()
            .value,
        4
    );
    let read_region = mapped_realize.block.reads.get(0).unwrap();
    assert_eq!(
        read_region
            .region()
            .unwrap()
            .get(0)
            .unwrap()
            .min
            .clone()
            .try_cast::<IntImm>()
            .unwrap()
            .value,
        4
    );
}

#[test]
fn structural_map_preserves_map_keys_and_maps_only_values() {
    load_tvm_compiler();
    let key = GlobalVar::new("main");
    let key_pointer = object_pointer(&key);
    let input = Map::from_iter([(key, int_expression(7))]);
    let mapped = structural_map(
        input,
        |value: IntImm| -> Result<Any> { Ok(Any::from(IntImm::new("int32", value.value + 1)?)) },
        WalkOrder::PostOrder,
    )
    .and_then(Map::<GlobalVar, Expr>::try_from)
    .unwrap();

    let (mapped_key, mapped_value) = mapped.iter().next().unwrap();
    assert_eq!(object_pointer(&mapped_key), key_pointer);
    assert_eq!(mapped_key.name_hint.as_str(), "main");
    assert_eq!(mapped_value.try_cast::<IntImm>().unwrap().value, 8);
}

#[test]
fn structural_map_reuses_a_uniquely_owned_array_container() {
    load_tvm_compiler();
    let input = tvm::tvm_ffi::Array::new(vec![int_expression(1), int_expression(2)]);
    let input_pointer = object_pointer(&input);
    let mapped = structural_map(
        input,
        |value: IntImm| -> Result<Any> { Ok(Any::from(IntImm::new("int32", value.value + 1)?)) },
        WalkOrder::PostOrder,
    )
    .and_then(tvm::tvm_ffi::Array::<Expr>::try_from)
    .unwrap();

    assert_eq!(object_pointer(&mapped), input_pointer);
    assert_eq!(
        mapped
            .iter()
            .map(|value| value.try_cast::<IntImm>().unwrap().value)
            .collect::<Vec<_>>(),
        vec![2, 3]
    );
}

#[test]
fn rust_add_zero_transformation_matches_cpp_stmt_simplify() {
    load_tvm_compiler();
    let one = int_expression(1);
    let zero = int_expression(0);
    let inner = add_expression(&zero, &one);
    let expression = add_expression(&inner, int_expression(0));
    let expected = int_expression(1);
    assert_structural_equal(
        &transform::examples::simplify_add_zero_expr(expression.clone()).unwrap(),
        &expected,
    );

    let function = PrimFunc::from_body(Evaluate::new(&expression).unwrap()).unwrap();
    let module = IRModule::from_expr(&function).unwrap();
    let rust_function = transform::examples::simplify_add_zero_prim_func(function).unwrap();
    let rust_result = IRModule::from_expr(&rust_function).unwrap();
    assert_eq!(node_statistics(&module).unwrap().additions, 2);
    assert_eq!(node_statistics(&rust_result).unwrap().additions, 0);
    let cpp_result = cpp_pass("tirx.transform.StmtSimplify").run(module).unwrap();
    assert_structural_equal(&rust_result, &cpp_result);
}

#[test]
fn rust_module_transformation_maps_function_values_like_cpp() {
    load_tvm_compiler();
    let expression = add_expression(
        add_expression(int_expression(0), int_expression(4)),
        int_expression(0),
    );
    let function = PrimFunc::from_body(Evaluate::new(&expression).unwrap()).unwrap();
    let main_module = IRModule::from_expr(&function).unwrap();
    let helper_var = GlobalVar::new("helper");
    let helper_function: tvm::ir::BaseFunc = PrimFunc::from_body(
        Evaluate::new(add_expression(int_expression(0), int_expression(9))).unwrap(),
    )
    .unwrap()
    .into();
    let module = main_module
        .with_updated_function(&helper_var, &helper_function)
        .unwrap();
    assert_eq!(main_module.functions.len(), 1);
    assert_eq!(module.functions.len(), 2);

    let rust_result = transform::examples::simplify_add_zero_module(module.clone()).unwrap();
    assert_eq!(node_statistics(&module).unwrap().additions, 3);
    assert_eq!(node_statistics(&rust_result).unwrap().additions, 0);
    let cpp_result = cpp_pass("tirx.transform.StmtSimplify").run(module).unwrap();

    assert_eq!(rust_result.functions.len(), 2);
    assert_structural_equal(&rust_result, &cpp_result);
}

#[derive(Default)]
struct RenameVariables {
    callback_calls: usize,
    regions: Vec<DefRegionKind>,
}

#[dispatch(map)]
impl RenameVariables {
    fn map_variable(&mut self, variable: Var, kind: DefRegionKind) -> Result<Any> {
        self.callback_calls += 1;
        self.regions.push(kind);
        let name = format!("{}_mapped", variable.name.as_str());
        Ok(Any::from(Var::with_type(&name, &variable.ty)))
    }
}

#[test]
fn structural_map_memoizes_free_var_identity() {
    load_tvm_compiler();
    let variable = Var::new("x", "int32").unwrap();
    let body = Evaluate::new(Expr::from(variable.clone())).unwrap();
    let function = PrimFunc::new(vec![variable.clone()], &body).unwrap();
    let mut mapper = RenameVariables::default();
    let mapped = structural_map(function.clone(), &mut mapper, WalkOrder::PostOrder)
        .and_then(PrimFunc::try_from)
        .unwrap();

    assert_eq!(mapper.callback_calls, 1);
    assert_eq!(mapper.regions, vec![DefRegionKind::Recursive]);
    let renamed = Var::new("x_mapped", "int32").unwrap();
    let expected_body = Evaluate::new(Expr::from(renamed.clone())).unwrap();
    let expected = PrimFunc::new(vec![renamed], &expected_body).unwrap();
    assert_structural_equal(&mapped, &expected);
    assert_eq!(variable.name.as_str(), "x");
}

#[derive(Default)]
struct ReplaceAddProbe {
    additions: usize,
    integers: usize,
}

#[dispatch(map)]
impl ReplaceAddProbe {
    fn map_add(&mut self, _value: Add) -> Result<Any> {
        self.additions += 1;
        Ok(Any::from(IntImm::new("int32", 0)?))
    }

    fn map_integer(&mut self, value: IntImm) -> Any {
        self.integers += 1;
        Any::from(value)
    }
}

#[test]
fn structural_map_preorder_replacement_prunes_original_children() {
    load_tvm_compiler();
    let expression = add_expression(int_expression(1), int_expression(2));

    let mut pre = ReplaceAddProbe::default();
    let pre_result = structural_map(expression.clone(), &mut pre, WalkOrder::PreOrder)
        .and_then(Expr::try_from)
        .unwrap();
    let mut post = ReplaceAddProbe::default();
    let post_result = structural_map(expression, &mut post, WalkOrder::PostOrder)
        .and_then(Expr::try_from)
        .unwrap();

    assert_eq!((pre.additions, pre.integers), (1, 0));
    assert_eq!((post.additions, post.integers), (1, 2));
    assert_structural_equal(&pre_result, &int_expression(0));
    assert_structural_equal(&post_result, &int_expression(0));
}

fn nested_loop_statement() -> Stmt {
    let outer = Var::new("i", "int32").unwrap();
    let inner = Var::new("j", "int32").unwrap();
    let sum = Add::new(outer.clone(), inner.clone()).unwrap();
    let inner_body: Stmt = Evaluate::new(Expr::from(sum)).unwrap().into();
    let inner_loop: Stmt =
        TirFor::serial(&inner, int_expression(1), int_expression(3), &inner_body)
            .unwrap()
            .into();
    TirFor::serial(&outer, int_expression(0), int_expression(4), &inner_loop)
        .unwrap()
        .into()
}

#[test]
fn walk_and_visit_handle_real_tir_loop_scopes() {
    load_tvm_compiler();
    let statement = nested_loop_statement();

    assert!(statement
        .clone()
        .try_cast::<TirFor>()
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
fn neutral_element_map_matches_cpp_stmt_simplify() {
    load_tvm_compiler();
    let variable = Var::new("x", "int32").unwrap();
    let add = Add::new(variable.clone(), int_expression(0)).unwrap();
    let multiply = Mul::new(add, int_expression(1)).unwrap();
    let expression: Expr = Sub::new(multiply, int_expression(0)).unwrap().into();

    let mapped = transform::examples::simplify_neutral_elements_expr(expression.clone()).unwrap();
    assert_structural_equal(&mapped, &Expr::from(variable.clone()));

    let function = PrimFunc::new(vec![variable], Evaluate::new(&expression).unwrap()).unwrap();
    let rust_function =
        transform::examples::simplify_neutral_elements_prim_func(function.clone()).unwrap();
    let cpp_module = cpp_pass("tirx.transform.StmtSimplify")
        .run(IRModule::from_expr(&function).unwrap())
        .unwrap();
    let rust_module = IRModule::from_expr(&rust_function).unwrap();
    assert_structural_equal(&rust_module, &cpp_module);
}

#[test]
fn mutate_can_limit_a_rewrite_to_loop_bodies() {
    load_tvm_compiler();
    let outer_var = Var::new("i", "int32").unwrap();
    let inner_var = Var::new("j", "int32").unwrap();
    let sum: Expr = Add::new(outer_var.clone(), inner_var.clone())
        .unwrap()
        .into();
    let inner_value: Expr = Sub::new(&sum, int_expression(0)).unwrap().into();
    let inner_body: Stmt = Evaluate::new(&inner_value).unwrap().into();
    let inner_loop: Stmt = TirFor::serial(
        &inner_var,
        Sub::new(int_expression(1), int_expression(0)).unwrap(),
        Mul::new(int_expression(3), int_expression(1)).unwrap(),
        &inner_body,
    )
    .unwrap()
    .into();
    let statement: Stmt = TirFor::serial(
        &outer_var,
        Add::new(int_expression(0), int_expression(0)).unwrap(),
        Add::new(int_expression(4), int_expression(0)).unwrap(),
        &inner_loop,
    )
    .unwrap()
    .into();
    let original = statement.clone();
    let mapped = transform::examples::simplify_neutral_elements_in_loop_bodies(statement)
        .unwrap()
        .try_cast::<TirFor>()
        .unwrap();

    assert!(mapped.min.clone().try_cast::<Add>().is_ok());
    assert!(mapped.extent.clone().try_cast::<Add>().is_ok());
    let inner = mapped.body.clone().try_cast::<TirFor>().unwrap();
    assert_eq!(inner.min.clone().try_cast::<IntImm>().unwrap().value, 1);
    assert_eq!(inner.extent.clone().try_cast::<IntImm>().unwrap().value, 3);

    let mapped_outer_var = mapped.loop_var.clone();
    let mapped_inner_var = inner.loop_var.clone();
    let inner_value = inner
        .body
        .clone()
        .try_cast::<Evaluate>()
        .unwrap()
        .value
        .clone()
        .try_cast::<Add>()
        .unwrap();
    assert_eq!(
        object_pointer(&inner_value.a),
        object_pointer(&mapped_outer_var)
    );
    assert_eq!(
        object_pointer(&inner_value.b),
        object_pointer(&mapped_inner_var)
    );

    let original_outer = original.try_cast::<TirFor>().unwrap();
    assert!(original_outer.min.clone().try_cast::<Add>().is_ok());
    let original_inner = original_outer.body.clone();
    let original_inner = original_inner.try_cast::<TirFor>().unwrap();
    assert!(original_inner.min.clone().try_cast::<Sub>().is_ok());
}

#[test]
fn buffer_and_block_bindings_round_trip_cpp_objects() {
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
    let tile_layout = TileLayout::new(vec![layout_iter.clone()], Vec::new(), Map::new()).unwrap();
    let layout = Layout::from(tile_layout.clone());
    let buffer_type = BufferType::with_metadata(
        "global",
        PrimType::new("int32").unwrap(),
        vec![extent.clone()],
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
    let iter_var = IterVar::new(&axis_domain, &axis, IterVarType::DataParallel).unwrap();
    let converted_axis = PrimExprConvertible::from(iter_var.clone())
        .to_prim_expr()
        .unwrap();
    assert_eq!(object_pointer(&converted_axis), object_pointer(&axis));
    let converted_by_cpp: Add = Function::get_global("tirx.Add")
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
        IterVarType::ThreadIndex,
        "threadIdx.x",
        None,
    )
    .unwrap();
    assert!(domainless_iter.dom().unwrap().is_none());
    let converted_domainless = PrimExprConvertible::from(domainless_iter)
        .to_prim_expr()
        .unwrap();
    assert_eq!(object_pointer(&converted_domainless), object_pointer(&axis));
    assert!(BufferLoad::new(&axis, vec![typed_int_expression("int64", 0)], None).is_err());
    let predicate = typed_int_expression("bool", 1);
    let load =
        BufferLoad::new(&buffer, vec![axis.clone().into()], Some(predicate.clone())).unwrap();
    let explicit_load_type = PrimType::new("int32").unwrap();
    let complete_load = BufferLoad::from_complete_fields(
        None,
        explicit_load_type.clone(),
        buffer.clone(),
        load.indices.clone(),
        load.predicate.clone(),
    );
    let explicit_load_type: Type = explicit_load_type.into();
    assert_eq!(
        object_pointer(&complete_load.ty),
        object_pointer(&explicit_load_type)
    );
    let store = BufferStore::new(
        &buffer,
        load.clone(),
        vec![axis.clone().into()],
        Some(predicate),
    )
    .unwrap();
    let cpp_indices = tvm::tvm_ffi::Array::new(load.indices.iter().collect());
    let cpp_predicate = load.predicate.clone();
    let cpp_load: BufferLoad = Function::get_global("tirx.BufferLoad")
        .unwrap()
        .call_packed(&[
            AnyView::from(&buffer),
            AnyView::from(&cpp_indices),
            AnyView::from(&cpp_predicate),
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
            AnyView::from(&cpp_predicate),
            AnyView::from(&()),
        ])
        .unwrap()
        .try_into()
        .unwrap();
    assert_structural_equal(&store, &cpp_store);

    let region = BufferRegion::new(&buffer, vec![axis_domain.clone()]).unwrap();
    let _: Expr = PrimExprConvertible::from(region.clone())
        .to_prim_expr()
        .unwrap();
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
    let annotations: Map<tvm::tvm_ffi::String, Any> = [
        (tvm::tvm_ffi::String::from("pipeline"), Any::from(2i64)),
        (
            tvm::tvm_ffi::String::from("tag"),
            Any::from(tvm::tvm_ffi::String::from("copy")),
        ),
    ]
    .into_iter()
    .collect();
    let block = SBlock::with_metadata(
        vec![iter_var],
        vec![region.clone()],
        vec![region],
        "copy",
        store.clone().into(),
        None,
        Vec::new(),
        Vec::new(),
        annotations,
        None,
    );
    assert_eq!(block.annotations.len(), 2);
    let realization = SBlockRealize::new(
        vec![typed_int_expression("int64", 0)],
        typed_int_expression("bool", 1),
        &block,
    )
    .unwrap();
    let function = PrimFunc::new(vec![buffer.clone().into()], &realization).unwrap();

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
    assert_eq!(object_pointer(&load.buffer), object_pointer(&buffer));
    assert_eq!(load.indices.len(), 1);
    assert!(load.predicate.is_some());
    assert_eq!(object_pointer(&store.buffer), object_pointer(&buffer));
    assert!(store.predicate.is_some());
    assert_eq!(block.name_hint.as_str(), "copy");
    assert_eq!(block.iter_vars.len(), 1);
    let reflected_axis = block.iter_vars.get(0).unwrap();
    assert_eq!(
        reflected_axis.iter_type().unwrap(),
        IterVarType::DataParallel
    );
    assert_eq!(
        object_pointer(&reflected_axis.var().unwrap()),
        object_pointer(&axis)
    );
    assert_eq!(
        reflected_axis
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
    assert_eq!(block.reads.len(), 1);
    assert_eq!(block.writes.len(), 1);
    assert_eq!(realization.block.name_hint.as_str(), "copy");

    let statistics = node_statistics(&function).unwrap();
    assert_eq!(statistics.blocks, 1);
    assert_eq!(statistics.block_realizations, 1);
    assert_eq!(statistics.buffer_loads, 1);
    assert_eq!(statistics.buffer_stores, 1);
    assert_eq!(
        memory_access_statistics(&function).unwrap(),
        tvm::analysis::MemoryAccessStatistics {
            loads: 1,
            stores: 1,
            predicated_loads: 1,
            predicated_stores: 1,
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
    let load: Expr = BufferLoad::new(&buffer, vec![index.clone()], None)
        .unwrap()
        .into();
    let store: Stmt = BufferStore::new(&buffer, &load, vec![index], None)
        .unwrap()
        .into();
    let unit_loop: Stmt = TirFor::serial(
        &unit_var,
        typed_int_expression("int64", 2),
        typed_int_expression("int64", 1),
        &store,
    )
    .unwrap()
    .into();
    let outer_loop = TirFor::serial(
        &outer_var,
        typed_int_expression("int64", 0),
        typed_int_expression("int64", 4),
        &unit_loop,
    )
    .unwrap();
    let function = PrimFunc::new(vec![buffer.into()], &outer_loop).unwrap();
    let module = IRModule::from_expr(&function).unwrap();

    let rust_function = transform::examples::eliminate_unit_loops_prim_func(function).unwrap();
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
    let mapped_outer = mapped_function.body.clone().try_cast::<TirFor>().unwrap();
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
fn integer_constant_folding_matches_cpp_on_fast_and_analyzer_paths() {
    load_tvm_compiler();
    let sum: Expr = Add::new(int_expression(2), int_expression(3))
        .unwrap()
        .into();
    let product: Expr = Mul::new(&sum, int_expression(4)).unwrap().into();
    let expression: Expr = Sub::new(&product, int_expression(1)).unwrap().into();

    let folded = transform::examples::fold_integer_constants_expr(expression.clone()).unwrap();
    assert_eq!(folded.clone().try_cast::<IntImm>().unwrap().value, 19);

    let function = PrimFunc::from_body(Evaluate::new(&expression).unwrap()).unwrap();
    let module = IRModule::from_expr(&function).unwrap();
    let rust_function = transform::examples::fold_integer_constants_prim_func(function).unwrap();
    let rust_result = IRModule::from_expr(&rust_function).unwrap();
    let cpp_result = cpp_pass("tirx.transform.StmtSimplify").run(module).unwrap();
    assert_structural_equal(&rust_result, &cpp_result);

    let maximum = typed_int_expression("int8", 127);
    let one = typed_int_expression("int8", 1);
    let overflow: Expr = Add::new(&maximum, &one).unwrap().into();
    let overflow_module =
        IRModule::from_expr(PrimFunc::from_body(Evaluate::new(&overflow).unwrap()).unwrap())
            .unwrap();
    let overflow_function = overflow_module
        .functions
        .iter()
        .next()
        .unwrap()
        .1
        .try_cast::<PrimFunc>()
        .unwrap();
    let rust_overflow = IRModule::from_expr(
        transform::examples::fold_integer_constants_prim_func(overflow_function).unwrap(),
    )
    .unwrap();
    let cpp_overflow = cpp_pass("tirx.transform.StmtSimplify")
        .run(overflow_module)
        .unwrap();
    assert_structural_equal(&rust_overflow, &cpp_overflow);
}

#[test]
fn call_effect_kind_preserves_future_native_values() {
    let future_value = CallEffectKind::from_raw(17);
    assert_eq!(future_value.as_raw(), 17);
    assert!(future_value.may_update_state());
    assert_eq!(CallEffectKind::try_from(17).unwrap(), future_value);
    assert!(CallEffectKind::try_from(i64::MAX).is_err());
}

#[test]
fn known_control_flow_simplification_matches_cpp_on_analyzed_constants() {
    load_tvm_compiler();
    let condition: Expr = Add::new(
        typed_int_expression("bool", 1),
        typed_int_expression("bool", 0),
    )
    .unwrap()
    .into();
    let then_case: Stmt = Evaluate::from_i64(7).unwrap().into();
    let else_case: Stmt = Evaluate::from_i64(9).unwrap().into();
    let conditional: Stmt = IfThenElse::new(&condition, &then_case, Some(else_case))
        .unwrap()
        .into();
    let standalone = IRModule::from_expr(PrimFunc::from_body(&conditional).unwrap()).unwrap();
    let standalone_function = standalone
        .functions
        .iter()
        .next()
        .unwrap()
        .1
        .try_cast::<PrimFunc>()
        .unwrap();
    let rust_standalone = IRModule::from_expr(
        transform::examples::simplify_known_control_flow_prim_func(standalone_function).unwrap(),
    )
    .unwrap();
    let cpp_standalone = cpp_pass("tirx.transform.RemoveNoOp")
        .run(standalone)
        .unwrap();
    assert_structural_equal(&rust_standalone, &cpp_standalone);

    let loop_var = Var::new("i", "int32").unwrap();
    let zero_extent: Expr = Sub::new(int_expression(2), int_expression(2))
        .unwrap()
        .into();
    let empty_loop: Stmt = TirFor::serial(
        &loop_var,
        int_expression(0),
        &zero_extent,
        Evaluate::from_i64(11).unwrap(),
    )
    .unwrap()
    .into();
    let body = SeqStmt::new(vec![conditional, empty_loop]).unwrap();
    let function = PrimFunc::from_body(&body).unwrap();
    let module = IRModule::from_expr(&function).unwrap();

    let rust_result = IRModule::from_expr(
        transform::examples::simplify_known_control_flow_prim_func(function).unwrap(),
    )
    .unwrap();
    let cpp_result = cpp_pass("tirx.transform.RemoveNoOp").run(module).unwrap();
    assert_structural_equal(&rust_result, &cpp_result);
    assert_eq!(node_statistics(&rust_result).unwrap().conditionals, 0);
    assert_eq!(node_statistics(&rust_result).unwrap().loops, 0);

    let pure_expression: Expr = Add::new(int_expression(3), int_expression(4))
        .unwrap()
        .into();
    assert_eq!(
        side_effect(&PrimExpr::try_from(&pure_expression).unwrap()).unwrap(),
        CallEffectKind::Pure
    );
    let read_buffer_type =
        BufferType::new("global", "int32", vec![typed_int_expression("int64", 1)]).unwrap();
    let read_buffer = read_buffer_type.new_var("read_buffer");
    let read_expression: Expr =
        BufferLoad::new(&read_buffer, vec![typed_int_expression("int64", 0)], None)
            .unwrap()
            .into();
    assert_eq!(
        side_effect(&PrimExpr::try_from(&read_expression).unwrap()).unwrap(),
        CallEffectKind::ReadState
    );
    let opaque_operator: Expr = GlobalVar::new("opaque_function").into();
    let opaque_call: Expr =
        Call::new(PrimType::new("int32").unwrap(), opaque_operator, Vec::new()).into();
    assert_eq!(
        side_effect(&PrimExpr::try_from(&opaque_call).unwrap()).unwrap(),
        CallEffectKind::UpdateState
    );
    let evaluations = SeqStmt::new(vec![
        Evaluate::new(&pure_expression).unwrap().into(),
        Evaluate::new(&read_expression).unwrap().into(),
        Evaluate::new(&opaque_call).unwrap().into(),
    ])
    .unwrap();
    let effects_module = IRModule::from_expr(PrimFunc::from_body(&evaluations).unwrap()).unwrap();
    let effects_function = effects_module
        .functions
        .iter()
        .next()
        .unwrap()
        .1
        .try_cast::<PrimFunc>()
        .unwrap();
    let rust_effects = IRModule::from_expr(
        transform::examples::simplify_known_control_flow_prim_func(effects_function).unwrap(),
    )
    .unwrap();
    let cpp_effects = cpp_pass("tirx.transform.RemoveNoOp")
        .run(effects_module)
        .unwrap();
    assert_structural_equal(&rust_effects, &cpp_effects);
    assert_eq!(node_statistics(&rust_effects).unwrap().evaluations, 1);
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
fn module_reachability_pruning_keeps_callees_and_external_roots() {
    load_tvm_compiler();
    let result_type: Type = PrimType::new("int32").unwrap().into();
    let main_global = GlobalVar::new("main");
    let helper_global = GlobalVar::new("helper");
    let exported_global = GlobalVar::new("exported");
    let dead_global = GlobalVar::new("dead");
    let helper_call: Expr = Call::new(&result_type, helper_global.clone(), Vec::new()).into();

    let main_function: tvm::ir::BaseFunc =
        PrimFunc::from_body(Evaluate::new(&helper_call).unwrap())
            .unwrap()
            .into();
    let helper_function: tvm::ir::BaseFunc = PrimFunc::from_body(Evaluate::from_i64(42).unwrap())
        .unwrap()
        .into();
    let exported_attributes = DictAttrs::from_dictionary(
        [(
            tvm::tvm_ffi::String::from("global_symbol"),
            Any::from(tvm::tvm_ffi::String::from("exported_symbol")),
        )]
        .into_iter()
        .collect(),
    );
    let exported_function: tvm::ir::BaseFunc = PrimFunc::with_metadata(
        Vec::new(),
        Evaluate::from_i64(7).unwrap(),
        Type::missing(),
        exported_attributes,
        None,
    )
    .unwrap()
    .into();
    let dead_function: tvm::ir::BaseFunc = PrimFunc::from_body(Evaluate::from_i64(99).unwrap())
        .unwrap()
        .into();
    let module = IRModule::with_metadata(
        [
            (main_global, main_function),
            (helper_global, helper_function),
            (exported_global, exported_function),
            (dead_global, dead_function),
        ]
        .into_iter()
        .collect(),
        SourceMap::new(),
        DictAttrs::empty(),
        Map::new(),
    )
    .unwrap();

    let direct =
        transform::examples::prune_unreachable_functions_from_main(module.clone()).unwrap();
    assert_eq!(direct.functions.len(), 3);
    let mut retained_names = direct
        .functions
        .iter()
        .map(|(global, _)| global.name_hint.as_str().to_owned())
        .collect::<Vec<_>>();
    retained_names.sort();
    assert_eq!(retained_names, vec!["exported", "helper", "main"]);

    let external_only = transform::examples::prune_unreachable_functions(module, &[]).unwrap();
    let external_names = external_only
        .functions
        .iter()
        .map(|(global, _)| global.name_hint.as_str().to_owned())
        .collect::<Vec<_>>();
    assert_eq!(external_names, vec!["exported"]);
    assert!(transform::examples::prune_unreachable_functions(direct, &["missing"]).is_err());
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
    let loop_statement = TirFor::with_metadata(
        loop_var,
        typed_int_expression("int64", 7),
        typed_int_expression("int64", 1),
        tvm::tirx::ForKind::Serial,
        body,
        None,
        annotations,
        None,
        None,
    )
    .unwrap();
    let function = PrimFunc::from_body(&loop_statement).unwrap();
    let module = IRModule::from_expr(&function).unwrap();

    let rust_function = transform::examples::eliminate_unit_loops_prim_func(function).unwrap();
    let rust_result = IRModule::from_expr(&rust_function).unwrap();
    let cpp_result = cpp_pass("tirx.transform.LowerTIRxOpaque")
        .run(module)
        .unwrap();

    assert_eq!(node_statistics(&rust_result).unwrap().loops, 1);
    assert_structural_equal(&rust_result, &cpp_result);
}
