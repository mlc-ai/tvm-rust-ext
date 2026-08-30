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

use tvm::ir::{
    BaseFuncObj, Expr, ExprObj, IntImm, IntImmObj, PrimExpr, PrimType, Type, Var, VarObj,
};
use tvm::tirx::{Add, AddObj, Evaluate, EvaluateObj, PrimFunc, PrimFuncObj, StmtObj};
use tvm::tvm_ffi::tvm_ffi_sys::{TVMFFIFieldFlagBitMask, TVMFFIFieldInfo};
use tvm::tvm_ffi::{
    structural_map, structural_walk, Any, AnyView, DefRegionKind, Function, Object, ObjectArc,
    ObjectCore, ObjectRefCast, ObjectRefCore, Result, WalkOrder, WalkResult,
};

mod common;
use common::{
    assert_structural_equal as assert_cpp_structural_equal, direct_fields, load_tvm_compiler,
    runtime_type_info,
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

fn direct_field<N: ObjectCore>(name: &str) -> &'static TVMFFIFieldInfo {
    direct_fields::<N>()
        .iter()
        .find(|field| field.name.as_str() == name)
        .unwrap_or_else(|| panic!("missing reflected field {}.{name}", N::TYPE_KEY))
}

fn assert_type_contract<N: ObjectCore, P: ObjectCore>(
    expected_final: bool,
    expected_fields: &[&str],
) {
    let info = runtime_type_info::<N>();
    assert_eq!(info.type_index, N::type_index());
    assert_eq!(info.type_key.as_str(), N::TYPE_KEY);
    assert_eq!(info.type_depth, N::TYPE_DEPTH);
    assert_eq!(N::TYPE_DEPTH, P::TYPE_DEPTH + 1);
    assert_eq!(N::TYPE_FINAL, expected_final);

    assert!(!info.type_acenstors.is_null());
    let parent = unsafe { *info.type_acenstors.add(P::TYPE_DEPTH as usize) };
    assert!(!parent.is_null());
    assert_eq!(unsafe { (*parent).type_index }, P::type_index());

    let fields = direct_fields::<N>();
    assert_eq!(
        fields
            .iter()
            .map(|field| field.name.as_str())
            .collect::<Vec<_>>(),
        expected_fields
    );
    for field in fields {
        assert!(
            field.getter.is_some(),
            "reflected field {}.{} has no getter",
            N::TYPE_KEY,
            field.name.as_str()
        );
    }
}

fn assert_field_flag<N: ObjectCore>(name: &str, flag: TVMFFIFieldFlagBitMask) {
    assert_ne!(direct_field::<N>(name).flags & flag as i64, 0);
}

use common::object_pointer;

fn cpp_reflected_field<O: ObjectRefCore>(value: &O, name: &str) -> Any {
    let field = direct_field::<O::ContainerType>(name);
    let getter = field
        .getter
        .expect("reflected field must have a C ABI getter");
    let object = object_pointer(value).cast::<u8>();
    let address = unsafe { object.add(field.offset as usize).cast_mut().cast() };
    let mut result = Any::new();
    assert_eq!(unsafe { getter(address, Any::as_data_ptr(&mut result)) }, 0);
    result
}

#[test]
fn minimal_bindings_match_the_runtime_contract() {
    load_tvm_compiler();

    assert_type_contract::<ExprObj, Object>(false, &["span", "ty"]);
    assert_type_contract::<VarObj, ExprObj>(false, &["name"]);
    assert_type_contract::<IntImmObj, ExprObj>(true, &["value"]);
    assert_type_contract::<AddObj, ExprObj>(true, &["a", "b"]);
    assert_type_contract::<StmtObj, Object>(false, &["span"]);
    assert_type_contract::<EvaluateObj, StmtObj>(true, &["value"]);
    assert_type_contract::<BaseFuncObj, ExprObj>(false, &["attrs"]);
    assert_type_contract::<PrimFuncObj, BaseFuncObj>(true, &["params", "ret_type", "body"]);

    assert_field_flag::<ExprObj>(
        "span",
        TVMFFIFieldFlagBitMask::kTVMFFIFieldFlagBitMaskSEqHashIgnore,
    );
    assert_field_flag::<VarObj>(
        "name",
        TVMFFIFieldFlagBitMask::kTVMFFIFieldFlagBitMaskSEqHashIgnore,
    );
    assert_field_flag::<StmtObj>(
        "span",
        TVMFFIFieldFlagBitMask::kTVMFFIFieldFlagBitMaskSEqHashIgnore,
    );
    assert_field_flag::<PrimFuncObj>(
        "params",
        TVMFFIFieldFlagBitMask::kTVMFFIFieldFlagBitMaskSEqHashDefRecursive,
    );
}

#[test]
fn direct_and_semantic_constructors_round_trip() {
    load_tvm_compiler();
    let missing = Type::missing();
    assert!(missing.is_missing());
    let cpp_recognizes_missing = Function::get_global("ir.TypeIsMissing")
        .unwrap()
        .call_packed(&[AnyView::from(&missing)])
        .unwrap();
    assert!(bool::try_from(cpp_recognizes_missing).unwrap());

    // PrimFunc's handwritten semantic constructor derives its fields, while
    // both it and the lossless path allocate the final node in Rust.
    let function = sample_function();
    let cpp_function: PrimFunc = Function::get_global("tirx.PrimFunc")
        .expect("missing reference semantic constructor")
        .call_packed(&[
            AnyView::from(&function.params),
            AnyView::from(&function.body),
            AnyView::from(&Type::missing()),
            AnyView::from(&function.attrs),
            AnyView::from(&()),
        ])
        .unwrap()
        .try_into()
        .unwrap();
    assert_ne!(object_pointer(&function), object_pointer(&cpp_function));
    assert_cpp_structural_equal(&function, &cpp_function);

    let rust_rebuilt = PrimFunc::from_complete_fields(
        function.span.clone(),
        function.ty.clone(),
        function.attrs.clone(),
        function.params.clone(),
        function.ret_type.clone(),
        function.body.clone(),
    );
    assert_cpp_structural_equal(&function, &rust_rebuilt);

    let parameter = function.params.get(0).unwrap();
    assert_eq!(parameter.name.as_str(), "x");
    assert!(parameter.span.is_none());
    assert_eq!(
        parameter
            .ty
            .clone()
            .try_cast::<PrimType>()
            .unwrap()
            .dtype
            .bits,
        32
    );

    let body = function.body.clone().try_cast::<Evaluate>().unwrap();
    assert!(body.span.is_none());
    let addition = body.value.clone().try_cast::<Add>().unwrap();
    let lhs_count = ObjectArc::strong_count(<PrimExpr as ObjectRefCore>::data(&addition.a));
    let borrowed_lhs: &Expr = &addition.a;
    assert_eq!(object_pointer(borrowed_lhs), object_pointer(&addition.a));
    assert_eq!(
        ObjectArc::strong_count(<PrimExpr as ObjectRefCore>::data(&addition.a)),
        lhs_count,
        "borrowing a generated public field must not clone its object handle"
    );
    let lhs = addition.a.clone().try_cast::<Var>().unwrap();
    let rhs = addition.b.clone().try_cast::<IntImm>().unwrap();
    assert_eq!(object_pointer(&lhs), object_pointer(&parameter));
    assert_eq!(rhs.value, 0);
    assert!(rhs.span.is_none());

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
    assert_eq!(object_pointer(&direct_add.a), object_pointer(&lhs_tracker));
    assert_eq!(
        ObjectArc::strong_count(<Expr as ObjectRefCore>::data(&lhs_tracker)),
        lhs_count,
        "moving a handle into a complete-field allocator must not clone it"
    );

    let _borrowed_attrs = &function.attrs;
    let _borrowed_return_type = &function.ret_type;
    let _borrowed_checked_type = &function.ty;
    assert!(function.span.is_none());
    assert!(IntImm::new("int8", 128).is_err());

    // C++ accepts values stored in IntImm's i64 payload for wider integer
    // types.  The generated Rust validation must not invent a 64-bit type
    // limit merely because the payload itself is i64.
    let wide = IntImm::new("int128", 42).unwrap();
    let cpp_wide: IntImm = Function::get_global("ir.IntImm")
        .unwrap()
        .call_packed(&[
            AnyView::from(&wide.ty.clone().try_cast::<PrimType>().unwrap().dtype),
            AnyView::from(&42_i64),
            AnyView::from(&()),
        ])
        .unwrap()
        .try_into()
        .unwrap();
    assert_cpp_structural_equal(&wide, &cpp_wide);
}

#[test]
fn rust_allocated_nodes_are_consumed_by_cpp_abi() {
    load_tvm_compiler();

    let lhs = Expr::from(IntImm::new("int32", 7).unwrap());
    let rhs = Expr::from(IntImm::new("int32", 9).unwrap());
    let rust_add = Add::new(&lhs, &rhs).unwrap();

    // Invoke the field getter registered by C++ on a Rust allocation.  A wrong
    // Rust field offset or representation would return the wrong object here.
    let reflected_lhs = Expr::try_from(cpp_reflected_field(&rust_add, "a")).unwrap();
    assert_eq!(object_pointer(&reflected_lhs), object_pointer(&lhs));

    // Build the equivalent node through C++ and compare through C++'s
    // structural-equality implementation.
    let none = ();
    let cpp_lhs = Function::get_global("ir.IntImm")
        .unwrap()
        .call_packed(&[
            AnyView::from(&PrimType::new("int32").unwrap().dtype),
            AnyView::from(&7_i64),
            AnyView::from(&none),
        ])
        .unwrap();
    let cpp_rhs = Function::get_global("ir.IntImm")
        .unwrap()
        .call_packed(&[
            AnyView::from(&PrimType::new("int32").unwrap().dtype),
            AnyView::from(&9_i64),
            AnyView::from(&none),
        ])
        .unwrap();
    let cpp_add: Add = Function::get_global("tirx.Add")
        .unwrap()
        .call_packed(&[
            AnyView::from(&cpp_lhs),
            AnyView::from(&cpp_rhs),
            AnyView::from(&none),
        ])
        .unwrap()
        .try_into()
        .unwrap();
    assert_cpp_structural_equal(&rust_add, &cpp_add);
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
            |_: &AddObj| {
                additions += 1;
                WalkResult::Advance
            },
            |value: &tvm::ir::IntImmObj| -> Result<WalkResult> {
                integer_literals.push(value.value);
                Ok(WalkResult::Advance)
            },
            |_: &VarObj, kind: DefRegionKind| {
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
        vec![DefRegionKind::Recursive, DefRegionKind::None]
    );
}

#[test]
fn generated_bindings_support_structural_map() {
    load_tvm_compiler();
    let original = sample_function();

    let mapped = structural_map(
        original.clone(),
        |addition: Add| -> Result<Any> {
            let lhs = addition.a.clone();
            let rhs = addition.b.clone();
            let rhs_is_zero = rhs
                .clone()
                .try_cast::<IntImm>()
                .ok()
                .map(|value| value.value)
                == Some(0);
            Ok(Any::from(if rhs_is_zero { lhs } else { addition.into() }))
        },
        WalkOrder::PostOrder,
    )
    .and_then(PrimFunc::try_from)
    .unwrap();

    let mapped_value = mapped
        .body
        .clone()
        .try_cast::<Evaluate>()
        .unwrap()
        .value
        .clone();
    let mapped_variable = mapped_value.try_cast::<Var>().unwrap();
    assert_eq!(mapped_variable.name.as_str(), "x");
    assert_eq!(
        object_pointer(&mapped_variable),
        object_pointer(&mapped.params.get(0).unwrap())
    );

    let original_value = original
        .body
        .clone()
        .try_cast::<Evaluate>()
        .unwrap()
        .value
        .clone();
    assert!(original_value.try_cast::<Add>().is_ok());
}
