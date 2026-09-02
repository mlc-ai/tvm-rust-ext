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

use tvm::tvm_ffi::tvm_ffi_sys::{TVMFFIFieldInfo, TVMFFIGetTypeInfo, TVMFFITypeInfo};
use tvm::tvm_ffi::ObjectCore;
use tvm::tvm_ffi::{AnyCompatible, AnyView, Function, ObjectArc, ObjectRefCore, String};

/// Load `libtvm_compiler` from the `apache-tvm` pip package (see `tvm::libinfo`).
pub fn load_tvm_compiler() {
    tvm::libinfo::load_compiler().expect("failed to load the TVM compiler library");
}

#[allow(dead_code)]
pub fn object_pointer<O: ObjectRefCore>(value: &O) -> *const () {
    unsafe { ObjectArc::as_raw(O::data(value)).cast() }
}

#[allow(dead_code)]
pub fn assert_structural_equal<L: AnyCompatible, R: AnyCompatible>(lhs: &L, rhs: &R) {
    let equal = Function::get_global("ffi.StructuralEqual")
        .unwrap()
        .call_packed(&[
            AnyView::from(lhs),
            AnyView::from(rhs),
            AnyView::from(&false),
            AnyView::from(&false),
        ])
        .unwrap();
    if !bool::try_from(equal).unwrap() {
        let repr = Function::get_global("ffi.ReprPrint").unwrap();
        let mismatch = Function::get_global("ffi.GetFirstStructuralMismatch")
            .unwrap()
            .call_packed(&[
                AnyView::from(lhs),
                AnyView::from(rhs),
                AnyView::from(&false),
                AnyView::from(&false),
            ])
            .unwrap();
        let mismatch =
            String::try_from(repr.call_packed(&[AnyView::from(&mismatch)]).unwrap()).unwrap();
        let lhs = String::try_from(repr.call_packed(&[AnyView::from(lhs)]).unwrap()).unwrap();
        let rhs = String::try_from(repr.call_packed(&[AnyView::from(rhs)]).unwrap()).unwrap();
        panic!("structural mismatch at {mismatch}:\nleft:  {lhs}\nright: {rhs}");
    }
}

#[allow(dead_code)]
pub fn runtime_type_info<N: ObjectCore>() -> &'static TVMFFITypeInfo {
    let pointer = unsafe { TVMFFIGetTypeInfo(N::type_index()) };
    assert!(!pointer.is_null(), "missing type info for {}", N::TYPE_KEY);
    unsafe { &*pointer }
}

#[allow(dead_code)]
pub fn direct_fields<N: ObjectCore>() -> &'static [TVMFFIFieldInfo] {
    let info = runtime_type_info::<N>();
    if info.num_fields == 0 {
        return &[];
    }
    assert!(!info.fields.is_null(), "missing fields for {}", N::TYPE_KEY);
    unsafe { std::slice::from_raw_parts(info.fields, info.num_fields as usize) }
}
