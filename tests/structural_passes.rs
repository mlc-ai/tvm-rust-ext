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
    node_statistics, Analyzer, AnalyzerMutator, AnalyzerMutatorState, CallEffectKind,
    ExprTraceEvent,
};
use tvm::ir::{
    BaseFunc, Call, DictAttrs, DummyGlobalInfo, Expr, FloatImm, GlobalVar, IRModule, IntImm,
    OpaqueExpr, PointerType, PrimExpr, PrimExprConvertible, PrimType, Range, SourceMap, SourceName,
    Span, TensorLoad, Type, Var,
};
use tvm::tirx::{
    Add, AddObj, AllocBuffer, AssertStmt, AssertStmtObj, AttrStmt, Axis, Bind, BindObj,
    BufferRegion, BufferStore, BufferType, DeclBuffer, DispatchContext, Evaluate, EvaluateObj,
    ExecScope, FloorDiv, FloorMod, For, ForKind, IfThenElse, Iter, IterVar, IterVarType, Layout,
    Let, MatchBufferRegion, Mul, Not, PrimFunc, Return, ScopeBinding, ScopeIdDef, ScopeIdDefStmt,
    ScopeKind, Select, SeqStmt, Stmt, StringImm, Sub, TileLayout, TilePrimitiveCall, While, EQ, GE,
    GT, LE, LT, NE,
};
use tvm::transform;
use tvm::tvm_ffi::{
    dispatch, structural_map, structural_mutate, structural_walk, Any, AnyView, Array,
    DefRegionKind, Function, Map, MapValue, Mutator, ObjectRefCast, ObjectRefCore, Result,
    WalkOrder, WalkResult,
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

struct AnalyzerMutationProbe {
    analysis: AnalyzerMutatorState,
    old_variable: Option<Var>,
    new_variable: Option<Var>,
}

impl AnalyzerMutationProbe {
    fn new(old_variable: Option<Var>, new_variable: Option<Var>) -> Self {
        Self {
            analysis: AnalyzerMutatorState::new(Analyzer::new().unwrap()).unwrap(),
            old_variable,
            new_variable,
        }
    }
}

impl AnalyzerMutator for AnalyzerMutationProbe {
    fn analyzer_state(&self) -> &AnalyzerMutatorState {
        &self.analysis
    }

    fn analyzer_state_mut(&mut self) -> &mut AnalyzerMutatorState {
        &mut self.analysis
    }
}

#[dispatch(mutate)]
impl AnalyzerMutationProbe {
    fn mutate_variable(&mut self, value: Var) -> Var {
        match (&self.old_variable, &self.new_variable) {
            (Some(old), Some(new)) if value.same_as(old) => new.clone(),
            _ => value,
        }
    }

    fn mutate_call(&mut self, value: Call, mutator: &mut Mutator) -> Result<Expr> {
        self.default_mutate_call(mutator, value)
    }

    fn mutate_assertion(&mut self, value: AssertStmt, mutator: &mut Mutator) -> Result<AssertStmt> {
        self.default_mutate_assertion(mutator, value)
    }

    fn mutate_default(&mut self, _value: &MapValue, mutator: &mut Mutator) -> Result<Any> {
        mutator.default_mutate(self)
    }
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
    IRModule::new(
        functions
            .into_iter()
            .map(|(name, function)| (GlobalVar::new(name), BaseFunc::from(function)))
            .collect(),
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
    let annotations =
        Map::from_iter([(annotation_key.clone(), Any::from(hidden_assertion.clone()))]);
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

#[test]
fn structural_map_preserves_map_keys_and_maps_only_values() {
    load_tvm_compiler();
    let key = GlobalVar::new("main");
    let key_pointer = object_pointer(&key);
    let input = Map::from_iter([(key, int_expression(7))]);
    let mapped = structural_map(
        input,
        |value: IntImm| -> Result<IntImm> { IntImm::new("int32", value.value + 1) },
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
        |value: IntImm| -> Result<IntImm> { IntImm::new("int32", value.value + 1) },
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

#[derive(Default)]
struct RenameVariables {
    callback_calls: usize,
    regions: Vec<DefRegionKind>,
}

#[dispatch(map)]
impl RenameVariables {
    fn map_variable(&mut self, variable: Var, kind: DefRegionKind) -> Result<Var> {
        self.callback_calls += 1;
        self.regions.push(kind);
        let name = format!("{}_mapped", variable.name.as_str());
        Ok(Var::with_type(&name, &variable.ty))
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
    fn map_add(&mut self, _value: Add) -> Result<IntImm> {
        self.additions += 1;
        IntImm::new("int32", 0)
    }

    fn map_integer(&mut self, value: IntImm) -> IntImm {
        self.integers += 1;
        value
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
    let iter_var = IterVar::new(&axis_domain, &axis).unwrap();
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
    let cpp_result = cpp_pass("tirx.transform.RemoveNoOp")
        .run(module.clone())
        .unwrap();

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
        Bind::new(variable.clone(), int_expression(1))
            .unwrap()
            .into(),
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
    let callee_global = GlobalVar::new("worker");
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
fn rust_bf16_compute_legalize_matches_cpp_across_a_buffer_boundary() {
    load_tvm_compiler();
    let buffer_type =
        BufferType::new("global", "bfloat16", vec![typed_int_expression("int64", 4)]).unwrap();
    let buffer = buffer_type.new_var("values");
    let index = typed_int_expression("int64", 0);
    let loaded = TensorLoad::from_buffer(&buffer, vec![index.clone()]).unwrap();
    let increment = tvm::ir::FloatImm::new("bfloat16", 1.0).unwrap();
    let updated = Add::new(loaded, increment).unwrap();
    let body = BufferStore::new(&buffer, updated, vec![index]).unwrap();
    let function = PrimFunc::new(vec![buffer.as_var().clone()], body).unwrap();
    let module = IRModule::from_expr(&function).unwrap();

    let rust_result =
        IRModule::from_expr(transform::bf16_compute_legalize_prim_func(function).unwrap()).unwrap();
    let cpp_result = cpp_pass("tirx.transform.BF16ComputeLegalize")
        .run(module)
        .unwrap();

    assert_structural_equal(&rust_result, &cpp_result);
}

#[test]
fn rust_fp8_compute_legalize_matches_cpp_across_a_buffer_boundary() {
    load_tvm_compiler();
    let buffer_type = BufferType::new(
        "global",
        "float8_e4m3fn",
        vec![typed_int_expression("int64", 4)],
    )
    .unwrap();
    let buffer = buffer_type.new_var("values");
    let index = typed_int_expression("int64", 0);
    let loaded = TensorLoad::from_buffer(&buffer, vec![index.clone()]).unwrap();
    let increment = tvm::ir::FloatImm::new("float8_e4m3fn", 1.0).unwrap();
    let updated = Add::new(loaded, increment).unwrap();
    let body = BufferStore::new(&buffer, updated, vec![index]).unwrap();
    let function = PrimFunc::new(vec![buffer.as_var().clone()], body).unwrap();
    let module = IRModule::from_expr(&function).unwrap();

    let rust_result = IRModule::from_expr(
        transform::fp8_compute_legalize_prim_func(function, "float16").unwrap(),
    )
    .unwrap();
    let cpp_pass: transform::Pass = Function::get_global("tirx.transform.FP8ComputeLegalize")
        .unwrap()
        .call_tuple((tvm::tvm_ffi::String::from("float16"),))
        .unwrap()
        .try_into()
        .unwrap();
    let cpp_result = cpp_pass.run(module).unwrap();

    assert_structural_equal(&rust_result, &cpp_result);
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

#[test]
fn rust_remove_no_op_keeps_nested_sequence_facts_like_cpp() {
    load_tvm_compiler();
    let value = Var::new("value", "int32").unwrap();
    let condition = EQ::new(value.clone(), int_expression(8)).unwrap();
    let assertion: Stmt = AssertStmt::new(condition, "ValueError", "value must be in range")
        .unwrap()
        .into();
    // Keep a physical nested sequence: the canonical constructor would flatten it.
    let nested = SeqStmt::from_complete_fields(None, Array::new(vec![assertion]));

    let buffer_type = BufferType::new("global", "int32", vec![int_expression(1)]).unwrap();
    let buffer = buffer_type.new_var("buffer");
    let load = TensorLoad::from_buffer(&buffer, vec![int_expression(0)]).unwrap();
    let difference = Sub::new(value.clone(), int_expression(8)).unwrap();
    let stored_value = Add::new(load, difference).unwrap();
    let store = BufferStore::new(&buffer, stored_value, vec![int_expression(0)]).unwrap();
    let body = SeqStmt::from_complete_fields(None, Array::new(vec![nested.into(), store.into()]));
    let function = PrimFunc::new(vec![buffer.as_var().clone(), value], body).unwrap();
    let module = IRModule::from_expr(&function).unwrap();

    let rust_function = transform::remove_no_op_prim_func(function).unwrap();
    assert!(rust_function.body.clone().try_cast::<AssertStmt>().is_ok());
    let rust_result = IRModule::from_expr(rust_function).unwrap();
    let cpp_result = cpp_pass("tirx.transform.RemoveNoOp").run(module).unwrap();

    assert_structural_equal(&rust_result, &cpp_result);
}

#[test]
fn analyzer_mutator_recomputes_buffer_data_type_after_variable_remap() {
    load_tvm_compiler();
    let old_buffer = BufferType::new("global", "int32", vec![int_expression(4)])
        .unwrap()
        .new_var("old_buffer");
    let new_buffer = BufferType::new("shared", "float32", vec![int_expression(4)])
        .unwrap()
        .new_var("new_buffer");
    let data: Expr = Function::get_global("tirx.BufferData")
        .unwrap()
        .call_packed(&[AnyView::from(&old_buffer)])
        .unwrap()
        .try_into()
        .unwrap();

    let mut probe = AnalyzerMutationProbe::new(
        Some(old_buffer.as_var().clone()),
        Some(new_buffer.as_var().clone()),
    );
    let rewritten = probe
        .mutate_root(data)
        .and_then(Expr::try_from)
        .unwrap()
        .try_cast::<Call>()
        .unwrap();

    let source = rewritten.args.get(0).unwrap().try_cast::<Var>().unwrap();
    assert!(source.same_as(new_buffer.as_var()));
    let pointer = rewritten.ty.clone().try_cast::<PointerType>().unwrap();
    assert_eq!(pointer.storage_scope().unwrap().as_str(), "shared");
    let element_type = pointer
        .element_type()
        .unwrap()
        .try_cast::<PrimType>()
        .unwrap();
    assert_eq!(element_type.dtype, new_buffer.type_annotation().dtype.dtype);
}

#[test]
fn analyzer_mutator_reports_missing_root_scope_without_panicking() {
    load_tvm_compiler();
    let assertion = AssertStmt::new(
        EQ::new(int_expression(1), int_expression(1)).unwrap(),
        "ValueError",
        "condition must hold",
    )
    .unwrap();
    let mut probe = AnalyzerMutationProbe::new(None, None);

    let Err(error) = structural_mutate(assertion, &mut probe) else {
        panic!("mutation without an analyzer root scope unexpectedly succeeded");
    };
    assert!(error
        .to_string()
        .contains("must start with AnalyzerMutator::mutate_root"));
}

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
    let second = Let::new(variable.clone(), int_expression(2), variable.clone()).unwrap();
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
        "tirx.NE"
    );
    assert_binary_constructor!(
        Expr::from(LT::new(lhs.clone(), rhs.clone()).unwrap()),
        "tirx.LT"
    );
    assert_binary_constructor!(
        Expr::from(LE::new(lhs.clone(), rhs.clone()).unwrap()),
        "tirx.LE"
    );
    assert_binary_constructor!(
        Expr::from(GT::new(lhs.clone(), rhs.clone()).unwrap()),
        "tirx.GT"
    );
    assert_binary_constructor!(
        Expr::from(GE::new(lhs.clone(), rhs.clone()).unwrap()),
        "tirx.GE"
    );

    let condition = PrimExpr::try_from(typed_int_expression("bool", 1)).unwrap();
    let native_not: Expr = Function::get_global("tirx.Not")
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
    let native_select: Expr = Function::get_global("tirx.Select")
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
    let native_let: Expr = Function::get_global("tirx.Let")
        .unwrap()
        .call_tuple((let_variable, lhs.clone(), rhs.clone(), Option::<Span>::None))
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
    let expression_call = Call::new(int_type.clone(), expression_global.clone(), Vec::new());
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
fn rust_lower_tirx_opaque_matches_cpp_for_a_unit_loop() {
    load_tvm_compiler();
    let loop_var = Var::new("i", "int64").unwrap();
    let value = Add::new(loop_var.clone(), typed_int_expression("int64", 1)).unwrap();
    let body = Evaluate::new(value).unwrap();
    let loop_statement = For::new(
        loop_var,
        typed_int_expression("int64", 2),
        typed_int_expression("int64", 1),
        body,
    )
    .unwrap();
    let function = PrimFunc::from_body(loop_statement).unwrap();
    let module = IRModule::from_expr(&function).unwrap();

    let rust_function = transform::lower_tirx_opaque_prim_func(function).unwrap();
    let rust_result = IRModule::from_expr(&rust_function).unwrap();
    let cpp_result = cpp_pass("tirx.transform.LowerTIRxOpaque")
        .run(module)
        .unwrap();

    assert!(rust_function.body.clone().try_cast::<For>().is_err());
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
    let let_expression: Expr = Function::get_global("tirx.Let")
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
    let get_operator = |name: &str| -> Expr {
        Function::get_global("ir.GetOp")
            .unwrap()
            .call_tuple((tvm::tvm_ffi::String::from(name),))
            .unwrap()
            .try_into()
            .unwrap()
    };
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
    let pointer_type = PointerType::new(element_type.clone(), "global").unwrap();
    let data = Var::with_type("data", pointer_type.clone());
    let access_ptr: Expr = Function::get_global("ir.GetOp")
        .unwrap()
        .call_tuple((tvm::tvm_ffi::String::from("tirx.tvm_access_ptr"),))
        .unwrap()
        .try_into()
        .unwrap();
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
    let exponential: Expr = Function::get_global("ir.GetOp")
        .unwrap()
        .call_tuple((tvm::tvm_ffi::String::from("tirx.exp"),))
        .unwrap()
        .try_into()
        .unwrap();
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
    let index = Mul::new(thread.clone(), IntImm::new("int32", 2).unwrap()).unwrap();
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
    let context_id: Expr = Function::get_global("ir.GetOp")
        .unwrap()
        .call_tuple((tvm::tvm_ffi::String::from("tirx.tvm_context_id"),))
        .unwrap()
        .try_into()
        .unwrap();
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
    let call_packed: Expr = Function::get_global("ir.GetOp")
        .unwrap()
        .call_tuple((tvm::tvm_ffi::String::from("tirx.tvm_call_packed"),))
        .unwrap()
        .try_into()
        .unwrap();
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
    let operator = |name: &str| -> Expr {
        Function::get_global("ir.GetOp")
            .unwrap()
            .call_tuple((tvm::tvm_ffi::String::from(name),))
            .unwrap()
            .try_into()
            .unwrap()
    };
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
    let ramp: PrimExpr = Function::get_global("tirx.Ramp")
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
    let ramp: PrimExpr = Function::get_global("tirx.Ramp")
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
    let ramp: PrimExpr = Function::get_global("tirx.Ramp")
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
        vec![Expr::from(extent.clone()), typed_int_expression("int64", 4)],
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
            Any::from(tvm::tvm_ffi::String::from("main")),
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
        ScopeIdDefStmt::new(blocks, None).unwrap().into(),
        ScopeIdDefStmt::new(threads, None).unwrap().into(),
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
                    .inter()?
                    .get(&tvm::tvm_ffi::String::from("laneid"))?
                    .expect("thread dispatch exposes laneid");
                PrimFunc::from_body(Evaluate::new(lane.get(1)?)?)
            },
        ),
    )
    .unwrap();
    let operator: Expr = Function::get_global("ir.GetOp")
        .unwrap()
        .call_tuple((tvm::tvm_ffi::String::from("tirx.tile.zero"),))
        .unwrap()
        .try_into()
        .unwrap();
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
        ScopeIdDefStmt::new(blocks, None).unwrap().into(),
        ScopeIdDefStmt::new(threads, None).unwrap().into(),
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
