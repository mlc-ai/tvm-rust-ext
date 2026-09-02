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

//! Each shorthand constructor must produce exactly what the full constructor
//! produces with its default metadata, and must keep the full constructor's
//! validation.  Each rebuild shorthand (`with_children` and friends) must
//! produce exactly what the hand-written `from_complete_fields` rebuild in a
//! pass produced, keeping every field it does not replace.

use tvm::ir::{
    BaseFunc, Call, DictAttrs, DummyGlobalInfo, Expr, GlobalInfo, GlobalVar, IRModule, IntImm,
    PrimExpr, PrimType, Range, SourceMap, SourceName, Span, TensorLoad, Type, Var,
};
use tvm::tirx::{
    Add, AllocBuffer, AssertStmt, AttrStmt, BufferStore, BufferType, BufferVar, CommReducer,
    DeclBuffer, Evaluate, For, ForKind, IfThenElse, IterVar, IterVarType, Let, Not, PrimFunc,
    Reduce, Select, Stmt, StringImm, LT,
};
use tvm::tvm_ffi::{Any, Array, Function, Map, ObjectRefCore, String};

mod common;
use common::{assert_structural_equal, load_tvm_compiler, object_pointer};

fn int_expression(dtype: &str, value: i64) -> Expr {
    IntImm::new(dtype, value).unwrap().into()
}

fn loop_body() -> Stmt {
    Evaluate::from_i64(1).unwrap().into()
}

fn attrs_with_global_symbol(symbol: &str) -> DictAttrs {
    DictAttrs::from_dictionary(Map::from_iter([(
        String::from("global_symbol"),
        Any::from(String::from(symbol)),
    )]))
}

#[test]
fn for_new_matches_with_metadata_defaults() {
    load_tvm_compiler();
    let loop_var = Var::new("i", "int32").unwrap();

    let shorthand = For::new(
        &loop_var,
        int_expression("int32", 0),
        int_expression("int32", 4),
        ForKind::kParallel,
        loop_body(),
    )
    .unwrap();
    let full = For::with_metadata(
        loop_var.clone(),
        int_expression("int32", 0),
        int_expression("int32", 4),
        ForKind::kParallel,
        loop_body(),
        None,
        Map::new(),
        None,
        None,
    )
    .unwrap();

    assert_eq!(shorthand.kind, ForKind::kParallel);
    assert!(shorthand.thread_binding.is_none());
    assert_eq!(shorthand.annotations.len(), 0);
    assert!(shorthand.step.is_none());
    assert!(shorthand.span.is_none());
    assert_structural_equal(&shorthand, &full);
}

#[test]
fn for_new_keeps_full_constructor_validation() {
    load_tvm_compiler();
    let float_var = Var::new("f", "float32").unwrap();
    let error = For::new(
        float_var,
        int_expression("int32", 0),
        int_expression("int32", 4),
        ForKind::kSerial,
        loop_body(),
    )
    .err()
    .expect("a float loop variable must be rejected");
    assert!(error.to_string().contains("loop_var"), "{error}");
}

#[test]
fn for_thread_bound_sets_kind_and_binding() {
    load_tvm_compiler();
    let loop_var = Var::new("tx", "int64").unwrap();
    let extent = int_expression("int64", 8);
    let domain = Range::from_min_extent(int_expression("int64", 0), extent.clone()).unwrap();
    let axis = IterVar::with_metadata(
        Some(domain),
        loop_var.clone(),
        IterVarType::kThreadIndex,
        "threadIdx.x",
        None,
    )
    .unwrap();

    let shorthand = For::thread_bound(
        &loop_var,
        int_expression("int64", 0),
        extent.clone(),
        axis.clone(),
        loop_body(),
    )
    .unwrap();
    let full = For::with_metadata(
        loop_var.clone(),
        int_expression("int64", 0),
        extent,
        ForKind::kThreadBinding,
        loop_body(),
        Some(axis.clone()),
        Map::new(),
        None,
        None,
    )
    .unwrap();

    assert_eq!(shorthand.kind, ForKind::kThreadBinding);
    let binding = shorthand.thread_binding.as_ref().unwrap();
    assert_eq!(object_pointer(binding), object_pointer(&axis));
    assert_eq!(shorthand.annotations.len(), 0);
    assert!(shorthand.step.is_none());
    assert_structural_equal(&shorthand, &full);
}

#[test]
fn iter_var_thread_index_matches_with_metadata() {
    load_tvm_compiler();
    let variable = Var::new("tx", "int32").unwrap();
    let domain =
        Range::from_min_extent(int_expression("int32", 0), int_expression("int32", 32)).unwrap();

    let shorthand = IterVar::thread_index(Some(domain.clone()), &variable, "threadIdx.x").unwrap();
    let full = IterVar::with_metadata(
        Some(domain),
        variable.clone(),
        IterVarType::kThreadIndex,
        "threadIdx.x",
        None,
    )
    .unwrap();
    assert_eq!(shorthand.iter_type().unwrap(), IterVarType::kThreadIndex);
    assert_eq!(shorthand.thread_tag().unwrap().as_str(), "threadIdx.x");
    assert!(shorthand.span().unwrap().is_none());
    assert_structural_equal(&shorthand, &full);

    let domainless = IterVar::thread_index(None, &variable, "blockIdx.x").unwrap();
    let full_domainless = IterVar::with_metadata(
        None,
        variable,
        IterVarType::kThreadIndex,
        "blockIdx.x",
        None,
    )
    .unwrap();
    assert!(domainless.dom().unwrap().is_none());
    assert_structural_equal(&domainless, &full_domainless);
}

#[test]
fn prim_func_with_attrs_matches_with_metadata() {
    load_tvm_compiler();
    let parameter = Var::new("n", "int32").unwrap();
    let body = Evaluate::new(&parameter).unwrap();

    let shorthand = PrimFunc::with_attrs(
        vec![parameter.clone()],
        &body,
        attrs_with_global_symbol("main"),
    )
    .unwrap();
    let full = PrimFunc::with_metadata(
        vec![parameter],
        &body,
        Type::missing(),
        attrs_with_global_symbol("main"),
        None,
    )
    .unwrap();

    let symbol = shorthand
        .attrs
        .dict
        .get(&String::from("global_symbol"))
        .unwrap()
        .map(String::try_from)
        .transpose()
        .unwrap()
        .unwrap();
    assert_eq!(symbol.as_str(), "main");
    assert!(!shorthand.ret_type.is_missing());
    assert!(shorthand.span.is_none());
    assert_structural_equal(&shorthand, &full);
}

#[test]
fn ir_module_from_functions_matches_with_metadata() {
    load_tvm_compiler();
    let first = PrimFunc::from_body(Evaluate::from_i64(1).unwrap()).unwrap();
    let second = PrimFunc::from_body(Evaluate::from_i64(2).unwrap()).unwrap();
    let first_global = GlobalVar::new("first");
    let second_global = GlobalVar::new("second");

    let from_pairs = IRModule::from_functions([
        (first_global.clone(), first.clone()),
        (second_global.clone(), second.clone()),
    ])
    .unwrap();
    let functions: Map<GlobalVar, BaseFunc> = Map::from_iter([
        (first_global.clone(), BaseFunc::from(first)),
        (second_global.clone(), BaseFunc::from(second)),
    ]);
    let from_map = IRModule::from_functions(&functions).unwrap();
    let full = IRModule::with_metadata(functions, SourceMap::new(), DictAttrs::empty(), Map::new())
        .unwrap();

    assert_eq!(from_pairs.functions.len(), 2);
    assert_eq!(from_pairs.source_map.source_map.len(), 0);
    assert_eq!(from_pairs.attrs.dict.len(), 0);
    assert_eq!(from_pairs.global_infos.len(), 0);
    let indexed = from_pairs
        .global_var_map
        .get(&String::from("second"))
        .unwrap()
        .unwrap();
    assert_eq!(object_pointer(&indexed), object_pointer(&second_global));
    assert_structural_equal(&from_pairs, &full);
    assert_structural_equal(&from_map, &full);
}

#[test]
fn ir_module_from_functions_keeps_duplicate_name_rejection() {
    load_tvm_compiler();
    let function = PrimFunc::from_body(Evaluate::from_i64(1).unwrap()).unwrap();
    let error = IRModule::from_functions([
        (GlobalVar::new("main"), function.clone()),
        (GlobalVar::new("main"), function),
    ])
    .err()
    .expect("duplicate global names must be rejected");
    assert!(error.to_string().contains("duplicate"), "{error}");
}

// ---------------------------------------------------------------------------
// Rebuilding an existing node with replaced children.
// ---------------------------------------------------------------------------

fn prim(value: i64) -> PrimExpr {
    IntImm::new("int32", value).unwrap().into()
}

fn prim64(value: i64) -> PrimExpr {
    IntImm::new("int64", value).unwrap().into()
}

fn span() -> Span {
    Span::new(
        SourceName::get("shorthand_constructors.rs").unwrap(),
        1,
        1,
        1,
        8,
    )
    .unwrap()
}

fn buffer_var(name: &str, extent: i64) -> (Var, BufferVar) {
    let buffer_type =
        BufferType::new("global", "int32", vec![int_expression("int32", extent)]).unwrap();
    let var = Var::with_type(name, buffer_type);
    let typed = BufferVar::try_from(var.clone()).unwrap();
    (var, typed)
}

#[test]
fn for_with_children_keeps_loop_metadata() {
    load_tvm_compiler();
    let loop_var = Var::new("i", "int32").unwrap();
    let annotations: Map<String, Any> =
        Map::from_iter([(String::from("pragma_test"), Any::from(1_i64))]);
    let original = For::with_metadata(
        loop_var.clone(),
        int_expression("int32", 0),
        int_expression("int32", 4),
        ForKind::kParallel,
        loop_body(),
        None,
        annotations,
        None,
        Some(&span()),
    )
    .unwrap();

    let (min, extent, body, step) = (prim(2), prim(8), loop_body(), Some(prim(2)));
    let rebuilt = original.with_children(min.clone(), extent.clone(), body.clone(), step.clone());
    let expected = For::from_complete_fields(
        original.span.clone(),
        original.loop_var.clone(),
        min,
        extent,
        original.kind,
        body,
        original.thread_binding.clone(),
        original.annotations.clone(),
        step,
    );

    assert_eq!(rebuilt.kind, ForKind::kParallel);
    assert_eq!(
        object_pointer(rebuilt.loop_var.as_var()),
        object_pointer(&loop_var)
    );
    assert_eq!(rebuilt.annotations.len(), 1);
    assert!(rebuilt.span.is_some());
    assert!(rebuilt.step.is_some());
    assert_structural_equal(&rebuilt, &expected);
}

#[test]
fn for_with_kind_changes_only_the_kind() {
    load_tvm_compiler();
    let loop_var = Var::new("i", "int32").unwrap();
    let original = For::serial(
        &loop_var,
        int_expression("int32", 0),
        int_expression("int32", 4),
        loop_body(),
    )
    .unwrap();

    let rebuilt = original.with_kind(ForKind::kUnrolled);
    assert_eq!(rebuilt.kind, ForKind::kUnrolled);
    assert_eq!(
        object_pointer(rebuilt.min.as_expr()),
        object_pointer(original.min.as_expr())
    );
    assert!(rebuilt.body.same_as(&original.body));
    let expected = For::from_complete_fields(
        None,
        original.loop_var.clone(),
        original.min.clone(),
        original.extent.clone(),
        ForKind::kUnrolled,
        original.body.clone(),
        None,
        Map::new(),
        None,
    );
    assert_structural_equal(&rebuilt, &expected);
}

#[test]
fn attr_stmt_with_children_keeps_node_and_key() {
    load_tvm_compiler();
    let variable = Var::new("tx", "int32").unwrap();
    let original = AttrStmt::with_span(
        variable.clone(),
        "thread_extent",
        int_expression("int32", 4),
        loop_body(),
        Some(&span()),
    )
    .unwrap();

    let (value, body): (PrimExpr, Stmt) = (prim(8), Evaluate::from_i64(2).unwrap().into());
    let rebuilt = original.with_children(value.clone(), body.clone());
    assert_eq!(rebuilt.attr_key.as_str(), "thread_extent");
    assert!(rebuilt.span.is_some());
    let expected = AttrStmt::from_complete_fields(
        original.span.clone(),
        original.node.clone(),
        original.attr_key.clone(),
        value,
        body,
    );
    assert_structural_equal(&rebuilt, &expected);
}

#[test]
fn conditional_statements_with_children_keep_span() {
    load_tvm_compiler();
    let condition = LT::new(Var::new("i", "int32").unwrap(), int_expression("int32", 4)).unwrap();
    let new_condition: PrimExpr = Not::new(&condition).unwrap().into();

    let conditional = IfThenElse::with_span(&condition, loop_body(), None, Some(&span())).unwrap();
    let else_case: Option<Stmt> = Some(Evaluate::from_i64(3).unwrap().into());
    let rebuilt = conditional.with_children(new_condition.clone(), loop_body(), else_case.clone());
    assert!(rebuilt.span.is_some());
    assert!(rebuilt.else_case.is_some());
    let expected = IfThenElse::from_complete_fields(
        conditional.span.clone(),
        new_condition.clone(),
        loop_body(),
        else_case,
    );
    assert_structural_equal(&rebuilt, &expected);

    let assertion = AssertStmt::with_metadata(
        &condition,
        StringImm::new("RuntimeError"),
        vec![StringImm::new("message")],
        Some(&span()),
    )
    .unwrap();
    let rebuilt = assertion.with_children(
        new_condition.clone(),
        assertion.error_kind.clone(),
        assertion.message_parts.clone(),
    );
    assert!(rebuilt.span.is_some());
    let expected = AssertStmt::from_complete_fields(
        assertion.span.clone(),
        new_condition,
        assertion.error_kind.clone(),
        assertion.message_parts.clone(),
    );
    assert_structural_equal(&rebuilt, &expected);
}

#[test]
fn buffer_nodes_with_children_keep_span_and_annotations() {
    load_tvm_compiler();
    let (a_var, _a) = buffer_var("A", 16);
    let (_b_var, b) = buffer_var("B", 16);
    let index = Expr::from(prim(1));

    let store = BufferStore::with_span(
        a_var.clone(),
        int_expression("int32", 7),
        vec![index.clone()],
        Some(&span()),
    )
    .unwrap();
    let rebuilt = store.with_children(b.clone(), prim(9), Array::new(vec![prim(2)]));
    assert!(rebuilt.span.is_some());
    assert_eq!(
        object_pointer(rebuilt.buffer.as_var()),
        object_pointer(b.as_var())
    );
    let expected = BufferStore::from_complete_fields(
        store.span.clone(),
        b.clone(),
        prim(9),
        Array::new(vec![prim(2)]),
    );
    assert_structural_equal(&rebuilt, &expected);

    let annotations: Map<String, Any> = Map::from_iter([(String::from("k"), Any::from(1_i64))]);
    let allocation = AllocBuffer::with_metadata(a_var.clone(), annotations, Some(&span())).unwrap();
    let rebuilt = allocation.with_buffer(b.clone());
    assert_eq!(rebuilt.annotations.len(), 1);
    assert!(rebuilt.span.is_some());
    let expected = AllocBuffer::from_complete_fields(
        allocation.span.clone(),
        b.clone(),
        allocation.annotations.clone(),
    );
    assert_structural_equal(&rebuilt, &expected);

    let data = Expr::from(Var::new("data", "int32").unwrap());
    let declaration = DeclBuffer::with_span(a_var.clone(), data.clone(), Some(&span())).unwrap();
    let new_data = Expr::from(Var::new("other", "int32").unwrap());
    let rebuilt = declaration.with_children(b.clone(), new_data.clone());
    assert!(rebuilt.span.is_some());
    let expected = DeclBuffer::from_complete_fields(declaration.span.clone(), b.clone(), new_data);
    assert_structural_equal(&rebuilt, &expected);

    let load = TensorLoad::from_buffer_with_span(a_var, vec![index], Some(&span())).unwrap();
    let new_source = Expr::from(b.as_var().clone());
    let rebuilt = load.with_children(new_source.clone(), Array::new(vec![prim(3)]));
    assert!(rebuilt.span.is_some());
    assert_structural_equal(
        &PrimExpr::from(&rebuilt).type_annotation(),
        &PrimType::new("int32").unwrap(),
    );
    let expected = TensorLoad::from_complete_fields(
        load.span.clone(),
        PrimExpr::from(&load).type_annotation(),
        new_source,
        Array::new(vec![prim(3)]),
    );
    assert_structural_equal(&rebuilt, &expected);
}

#[test]
fn expressions_with_children_keep_metadata_and_retype() {
    load_tvm_compiler();
    let int_type: Type = PrimType::new("int32").unwrap().into();
    let callee = GlobalVar::new("f");
    let new_callee = GlobalVar::new("g");
    let call = Call::with_metadata(
        &int_type,
        &callee,
        vec![int_expression("int32", 1)],
        Some(DictAttrs::empty().into()),
        vec![int_type.clone()],
        Some(&span()),
    );
    let rebuilt = call.with_children(
        Expr::from(&new_callee),
        Array::new(vec![int_expression("int32", 2)]),
    );
    assert!(rebuilt.span.is_some());
    assert!(rebuilt.attrs.is_some());
    assert_eq!(rebuilt.ty_args.len(), 1);
    let expected = Call::from_complete_fields(
        call.span.clone(),
        call.ty.clone(),
        Expr::from(&new_callee),
        Array::new(vec![int_expression("int32", 2)]),
        call.attrs.clone(),
        call.ty_args.clone(),
    );
    assert_structural_equal(&rebuilt, &expected);

    let var = Var::new("x", "int32").unwrap();
    let let_expr =
        Let::with_span(var.clone(), int_expression("int32", 1), &var, Some(&span())).unwrap();
    let body = prim64(5);
    let rebuilt = let_expr.with_children(prim(2), body.clone());
    assert!(rebuilt.span.is_some());
    assert_structural_equal(
        &PrimExpr::from(&rebuilt).type_annotation(),
        &PrimType::new("int64").unwrap(),
    );
    let expected = Let::from_complete_fields(
        let_expr.span.clone(),
        body.type_annotation(),
        var.clone(),
        prim(2),
        body,
    );
    assert_structural_equal(&rebuilt, &expected);

    let select = Select::with_span(
        LT::new(&var, int_expression("int32", 1)).unwrap(),
        int_expression("int32", 1),
        int_expression("int32", 2),
        Some(&span()),
    )
    .unwrap();
    let rebuilt = select.with_children(select.condition.clone(), prim64(3), prim64(4));
    assert!(rebuilt.span.is_some());
    assert_structural_equal(
        &PrimExpr::from(&rebuilt).type_annotation(),
        &PrimType::new("int64").unwrap(),
    );
    let expected = Select::from_complete_fields(
        select.span.clone(),
        PrimType::new("int64").unwrap(),
        select.condition.clone(),
        prim64(3),
        prim64(4),
    );
    assert_structural_equal(&rebuilt, &expected);
}

#[test]
fn reduce_with_children_keeps_combiner_type_and_index() {
    load_tvm_compiler();
    let x = Var::new("x", "int32").unwrap();
    let y = Var::new("y", "int32").unwrap();
    let combiner: CommReducer = Function::get_global("tirx.CommReducer")
        .unwrap()
        .call_tuple((
            Array::new(vec![x.clone()]),
            Array::new(vec![y.clone()]),
            Array::new(vec![PrimExpr::from(Add::new(&x, &y).unwrap())]),
            Array::new(vec![prim(0)]),
            Option::<Span>::None,
        ))
        .unwrap()
        .try_into()
        .unwrap();
    let axis_var = Var::new("k", "int32").unwrap();
    let axis = IterVar::new(
        Range::from_min_extent(int_expression("int32", 0), int_expression("int32", 4)).unwrap(),
        &axis_var,
        IterVarType::kCommReduce,
    )
    .unwrap();
    let condition: PrimExpr = IntImm::new("bool", 1).unwrap().into();
    let original = Reduce::from_complete_fields(
        Some(span()),
        PrimType::new("int32").unwrap(),
        combiner.clone(),
        Array::new(vec![PrimExpr::try_from(Expr::from(&x)).unwrap()]),
        Array::<PrimExpr>::new(Vec::new()),
        Array::new(vec![axis.clone()]),
        condition.clone(),
        0,
    );

    let new_source = Array::new(vec![PrimExpr::try_from(Expr::from(&y)).unwrap()]);
    let rebuilt = original.with_children(
        new_source.clone(),
        Array::new(vec![prim(0)]),
        Array::new(vec![axis.clone()]),
        condition.clone(),
    );
    assert_eq!(rebuilt.value_index, 0);
    assert!(rebuilt.span.is_some());
    assert_eq!(object_pointer(&rebuilt.combiner), object_pointer(&combiner));
    let expected = Reduce::from_complete_fields(
        original.span.clone(),
        PrimType::new("int32").unwrap(),
        combiner,
        new_source,
        Array::new(vec![prim(0)]),
        Array::new(vec![axis]),
        condition,
        0,
    );
    assert_structural_equal(&rebuilt, &expected);
}

#[test]
fn iter_var_with_children_keeps_type_tag_and_span() {
    load_tvm_compiler();
    let var = Var::new("tx", "int32").unwrap();
    let domain =
        Range::from_min_extent(int_expression("int32", 0), int_expression("int32", 8)).unwrap();
    let original = IterVar::with_metadata(
        Some(domain),
        var,
        IterVarType::kThreadIndex,
        "threadIdx.x",
        Some(&span()),
    )
    .unwrap();

    let new_var = Var::new("ty", "int32").unwrap();
    let new_domain =
        Range::from_min_extent(int_expression("int32", 0), int_expression("int32", 16)).unwrap();
    let rebuilt = original
        .with_children(Some(new_domain.clone()), new_var.clone())
        .unwrap();
    assert_eq!(rebuilt.iter_type().unwrap(), IterVarType::kThreadIndex);
    assert_eq!(rebuilt.thread_tag().unwrap().as_str(), "threadIdx.x");
    assert!(rebuilt.span().unwrap().is_some());
    let expected = IterVar::with_metadata(
        Some(new_domain),
        new_var,
        IterVarType::kThreadIndex,
        "threadIdx.x",
        original.span().unwrap().as_ref(),
    )
    .unwrap();
    assert_structural_equal(&rebuilt, &expected);
}

#[test]
fn buffer_type_with_children_keeps_scope_dtype_and_alignment() {
    load_tvm_compiler();
    let original = BufferType::with_metadata(
        "shared",
        PrimType::new("float16").unwrap(),
        vec![int_expression("int32", 16)],
        Vec::new(),
        int_expression("int32", 0),
        64,
        2,
        None,
        Vec::new(),
        Some(&span()),
    )
    .unwrap();

    let shape = Array::new(vec![prim(32)]);
    let rebuilt = original.with_children(
        shape.clone(),
        original.strides.clone(),
        original.elem_offset.clone(),
        None,
        original.allocated_addr.clone(),
    );
    assert_eq!(rebuilt.storage_scope.as_str(), "shared");
    assert_eq!(rebuilt.data_alignment, 64);
    assert_eq!(rebuilt.offset_factor, 2);
    assert!(rebuilt.span.is_some());
    let expected = BufferType::from_complete_fields(
        original.span.clone(),
        original.dtype.clone(),
        original.storage_scope.clone(),
        shape,
        original.strides.clone(),
        original.elem_offset.clone(),
        64,
        2,
        None,
        original.allocated_addr.clone(),
    );
    assert_structural_equal(&rebuilt, &expected);
}

#[test]
fn prim_func_rebuilds_keep_type_and_metadata() {
    load_tvm_compiler();
    let parameter = Var::new("n", "int32").unwrap();
    let int_type: Type = PrimType::new("int32").unwrap().into();
    let original = PrimFunc::with_metadata(
        vec![parameter.clone()],
        Evaluate::new(&parameter).unwrap(),
        int_type,
        attrs_with_global_symbol("main"),
        Some(&span()),
    )
    .unwrap();

    let new_body: Stmt = Evaluate::from_i64(1).unwrap().into();
    let with_body = original.with_body(new_body.clone());
    assert_eq!(object_pointer(&with_body.ty), object_pointer(&original.ty));
    assert_eq!(
        object_pointer(&with_body.attrs),
        object_pointer(&original.attrs)
    );
    assert!(with_body.span.is_some());
    let expected = PrimFunc::from_complete_fields(
        original.span.clone(),
        original.ty.clone(),
        original.attrs.clone(),
        original.params.clone(),
        original.ret_type.clone(),
        new_body.clone(),
    );
    assert_structural_equal(&with_body, &expected);

    let replaced = original.with_attr("global_symbol", String::from("other"));
    assert_eq!(replaced.attrs.dict.len(), 1);
    let symbol = replaced
        .attrs
        .dict
        .get(&String::from("global_symbol"))
        .unwrap()
        .map(String::try_from)
        .transpose()
        .unwrap()
        .unwrap();
    assert_eq!(symbol.as_str(), "other");
    let added = original.with_attr("tir.noalias", true);
    assert_eq!(added.attrs.dict.len(), 2);
    assert_eq!(original.attrs.dict.len(), 1);
    assert!(added.body.same_as(&original.body));

    let new_parameter = Var::new("m", "int32").unwrap();
    let with_children = original
        .with_children(
            vec![new_parameter.clone()],
            new_body.clone(),
            DictAttrs::empty(),
        )
        .unwrap();
    assert_eq!(
        object_pointer(&with_children.ret_type),
        object_pointer(&original.ret_type)
    );
    assert!(with_children.span.is_some());
    assert_eq!(with_children.attrs.dict.len(), 0);
    let expected = PrimFunc::with_metadata(
        vec![new_parameter],
        new_body,
        original.ret_type.clone(),
        DictAttrs::empty(),
        original.span.as_ref(),
    )
    .unwrap();
    assert_structural_equal(&with_children, &expected);
}

#[test]
fn ir_module_with_functions_keeps_module_metadata() {
    load_tvm_compiler();
    let first = PrimFunc::from_body(Evaluate::from_i64(1).unwrap()).unwrap();
    let second = PrimFunc::from_body(Evaluate::from_i64(2).unwrap()).unwrap();
    let first_global = GlobalVar::new("first");
    let second_global = GlobalVar::new("second");
    let global_infos: Map<String, Array<GlobalInfo>> = Map::from_iter([(
        String::from("infos"),
        Array::new(vec![GlobalInfo::from(DummyGlobalInfo::new())]),
    )]);
    let original = IRModule::with_metadata(
        Map::from_iter([(first_global.clone(), BaseFunc::from(first.clone()))]),
        SourceMap::new(),
        attrs_with_global_symbol("module"),
        global_infos,
    )
    .unwrap();

    let rebuilt = original
        .with_functions([
            (first_global.clone(), first.clone()),
            (second_global.clone(), second.clone()),
        ])
        .unwrap();
    assert_eq!(rebuilt.functions.len(), 2);
    assert_eq!(rebuilt.attrs.dict.len(), 1);
    assert_eq!(rebuilt.global_infos.len(), 1);
    assert_eq!(
        object_pointer(&rebuilt.source_map),
        object_pointer(&original.source_map)
    );
    let expected = IRModule::with_metadata(
        Map::from_iter([
            (first_global, BaseFunc::from(first)),
            (second_global, BaseFunc::from(second)),
        ]),
        original.source_map.clone(),
        original.attrs.clone(),
        original.global_infos.clone(),
    )
    .unwrap();
    assert_structural_equal(&rebuilt, &expected);
}
