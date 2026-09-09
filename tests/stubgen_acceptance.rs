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

//! Minimal acceptance tests for generated TVM Rust IR bindings.
//!
//! These tests intentionally use only the small handwritten binding slice that
//! stubgen should replace first.  Once those bindings are generated, deleting
//! their handwritten definitions must not require changing this file.

use tvm::ir::prim::{Add, AddObj};
use tvm::ir::{Expr, IntImm, IntImmObj, PrimExpr, PrimType, PrimTypeObj, Type, Var, VarObj};
use tvm::tirx::{Evaluate, EvaluateObj, PrimFunc};
use tvm::tvm_ffi::{
    structural_map, structural_walk, DefRegionKind, FieldGetter, Function, ObjectArc, ObjectCore,
    ObjectRefCast, ObjectRefCore, WalkOrder, WalkResult,
};

mod common;
use common::{
    assert_structural_equal as assert_cpp_structural_equal, load_tvm_compiler, object_pointer,
};

fn typed_int_expression(dtype: &str, value: i64) -> Expr {
    IntImm::new(dtype, value).unwrap().into()
}

fn sample_function() -> PrimFunc {
    let variable = Var::new("x", "int32").unwrap();
    let zero = IntImm::new("int32", 0).unwrap();
    let value = Add::new(variable.clone(), zero).unwrap();
    let body = Evaluate::new(value).unwrap();
    PrimFunc::new(vec![variable], body).unwrap()
}

#[test]
fn direct_and_semantic_constructors_round_trip() {
    load_tvm_compiler();
    let missing = Type::missing();
    assert!(missing.is_missing());
    let cpp_recognizes_missing = Function::get_global("ir.TypeIsMissing")
        .unwrap()
        .call_tuple((&missing,))
        .unwrap();
    assert!(bool::try_from(cpp_recognizes_missing).unwrap());

    // PrimFunc's handwritten semantic constructor derives its fields, while
    // both it and the lossless path allocate the final node in Rust.
    let function = sample_function();
    let cpp_function: PrimFunc = Function::get_global("tirx.PrimFunc")
        .expect("missing reference semantic constructor")
        .call_tuple((
            &function.params,
            function.body(),
            &Type::missing(),
            &function.attrs,
            (),
        ))
        .unwrap()
        .try_into()
        .unwrap();
    assert!(!function.same_as(&cpp_function));
    assert_cpp_structural_equal(&function, &cpp_function);

    let rust_rebuilt = PrimFunc::from_complete_fields(
        function.span.clone(),
        function.ty.clone(),
        function.attrs.clone(),
        function.params.clone(),
        function.ret_type.clone(),
        Some(function.body().clone()),
    );
    assert_cpp_structural_equal(&function, &rust_rebuilt);

    let parameter = function.params.get(0).unwrap();
    assert_eq!(parameter.name.as_str(), "x");
    assert!(parameter.span.is_none());
    assert_eq!(
        parameter.ty.as_node::<PrimTypeObj>().unwrap().dtype.bits,
        32
    );

    let body = function.body().as_node::<EvaluateObj>().unwrap();
    assert!(body.span.is_none());
    let addition = body.value.clone().try_cast::<Add>().unwrap();
    let lhs_count = ObjectArc::strong_count(<PrimExpr as ObjectRefCore>::data(&addition.a));
    let borrowed_lhs: &Expr = &addition.a;
    assert!(borrowed_lhs.same_as(&addition.a));
    assert_eq!(
        ObjectArc::strong_count(<PrimExpr as ObjectRefCore>::data(&addition.a)),
        lhs_count,
        "borrowing a generated public field must not clone its object handle"
    );
    let rhs = addition.b.as_node::<IntImmObj>().unwrap();
    assert!(addition.a.same_as(&parameter));
    assert_eq!(rhs.value, 0);
    assert!(rhs.span.is_none());

    // Exercise the C++ field getter on Rust-owned storage through standard FFI.
    let reflected_lhs: Expr = FieldGetter::new(AddObj::type_index(), "a")
        .unwrap()
        .get(&addition)
        .unwrap();
    assert!(reflected_lhs.same_as(&parameter));
    let cpp_add: Add = Function::get_global("ir.prim.Add")
        .unwrap()
        .call_tuple((&addition.a, &addition.b, Option::<tvm::ir::Span>::None))
        .unwrap()
        .try_into()
        .unwrap();
    assert_cpp_structural_equal(&addition, &cpp_add);

    let moved_lhs = typed_int_expression("int32", 3);
    let lhs_tracker = moved_lhs.clone();
    let lhs_count = ObjectArc::strong_count(<Expr as ObjectRefCore>::data(&lhs_tracker));
    let moved_rhs = typed_int_expression("int32", 4);
    let result_type = moved_lhs.ty.clone().try_cast::<PrimType>().unwrap();
    let direct_add = Add::from_complete_fields(
        None,
        result_type,
        PrimExpr::try_from(moved_lhs).unwrap(),
        PrimExpr::try_from(moved_rhs).unwrap(),
    );
    assert!(direct_add.a.same_as(&lhs_tracker));
    assert_eq!(
        ObjectArc::strong_count(<Expr as ObjectRefCore>::data(&lhs_tracker)),
        lhs_count,
        "moving a handle into a complete-field allocator must not clone it"
    );

    assert!(function.span.is_none());
    assert!(IntImm::new("int8", 128).is_err());

    // C++ accepts values stored in IntImm's i64 payload for wider integer
    // types.  The generated Rust validation must not invent a 64-bit type
    // limit merely because the payload itself is i64.
    let wide = IntImm::new("int128", 42).unwrap();
    let cpp_wide: IntImm = Function::get_global("ir.IntImm")
        .unwrap()
        .call_tuple((
            &wide.ty.as_node::<PrimTypeObj>().unwrap().dtype,
            &42_i64,
            (),
        ))
        .unwrap()
        .try_into()
        .unwrap();
    assert_cpp_structural_equal(&wide, &cpp_wide);
}

#[test]
fn object_upcast_preserves_pointer_and_reference_count() {
    load_tvm_compiler();
    let literal = IntImm::new("int32", 7).unwrap();
    let pointer = object_pointer(&literal);
    let strong_count = ObjectArc::strong_count(<IntImm as ObjectRefCore>::data(&literal));

    let expression = Expr::from(literal);

    assert_eq!(object_pointer(&expression), pointer);
    assert_eq!(
        ObjectArc::strong_count(<Expr as ObjectRefCore>::data(&expression)),
        strong_count,
        "upcasting must move the same ObjectArc without changing its reference count"
    );
}

#[test]
fn generated_bindings_support_structural_walk() {
    load_tvm_compiler();
    let function = sample_function();
    let mut additions = 0;
    let mut integer_literals = Vec::new();
    let mut variable_regions = Vec::new();

    structural_walk(
        &function,
        (
            |_: Add| {
                additions += 1;
                WalkResult::Advance
            },
            |value: IntImm| {
                integer_literals.push(value.value);
                WalkResult::Advance
            },
            |_: Var, kind: DefRegionKind| {
                variable_regions.push(kind);
                WalkResult::Advance
            },
        ),
        WalkOrder::PreOrder,
    )
    .unwrap();

    assert_eq!(additions, 1);
    assert_eq!(integer_literals, vec![0]);
    assert_eq!(
        variable_regions,
        vec![DefRegionKind::Pattern, DefRegionKind::None]
    );
}

#[test]
fn generated_bindings_support_structural_map() {
    load_tvm_compiler();
    let original = sample_function();

    let mapped = structural_map(
        original.clone(),
        |addition: Add| -> PrimExpr {
            if addition
                .b
                .as_node::<IntImmObj>()
                .is_some_and(|rhs| rhs.value == 0)
            {
                addition.a.clone()
            } else {
                addition.into()
            }
        },
        WalkOrder::PostOrder,
    )
    .and_then(PrimFunc::try_from)
    .unwrap();

    let mapped_body = mapped.body().as_node::<EvaluateObj>().unwrap();
    let mapped_variable = mapped_body.value.as_node::<VarObj>().unwrap();
    assert_eq!(mapped_variable.name.as_str(), "x");
    assert!(mapped_body.value.same_as(&mapped.params.get(0).unwrap()));

    let original_body = original.body().as_node::<EvaluateObj>().unwrap();
    assert!(original_body.value.as_node::<AddObj>().is_some());
}
