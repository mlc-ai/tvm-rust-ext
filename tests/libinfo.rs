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

//! The native libraries are resolved from the pip packages installed in the
//! active Python environment, not from a TVM source tree.

use tvm::libinfo;
use tvm::tvm_ffi::Function;

#[test]
fn compiler_library_is_resolved_from_the_python_environment() {
    let dir = libinfo::build_time_library_dir()
        .expect("build.rs should have located the tvm package's library directory");
    assert!(dir.is_absolute(), "{}", dir.display());
    let path = libinfo::compiler_library_path().unwrap();
    assert!(path.is_file(), "{}", path.display());
    assert_eq!(path.file_name().unwrap(), libinfo::COMPILER_LIBRARY);
    assert_eq!(path.parent().unwrap(), dir);
    let runtime = libinfo::runtime_library_path().unwrap();
    assert_eq!(runtime.parent().unwrap(), dir);
}

#[test]
fn tvm_ffi_library_directory_is_recorded() {
    let dir = libinfo::tvm_ffi_library_dir()
        .expect("build.rs should have recorded `tvm-ffi-config --libdir`");
    let found = ["libtvm_ffi.so", "libtvm_ffi.dylib", "tvm_ffi.dll"]
        .iter()
        .any(|name| dir.join(name).is_file());
    assert!(found, "no tvm_ffi library in {}", dir.display());
}

#[test]
fn loading_the_compiler_registers_compiler_functions() {
    // Compiler-only services are absent until the library is loaded.
    let module = libinfo::load_compiler().unwrap();
    let again = libinfo::load_compiler().unwrap();
    assert!(std::ptr::eq(module, again));
    Function::get_global("ir.OpGetAttr").unwrap();
    Function::get_global("tirx.PrimFunc").unwrap();
}

#[test]
fn missing_library_reports_searched_directories() {
    let error = libinfo::find_library("libtvm_does_not_exist.so").unwrap_err();
    assert_eq!(error.file_name, "libtvm_does_not_exist.so");
    assert!(!error.searched.is_empty());
    let message = error.to_string();
    assert!(message.contains("cannot find libtvm_does_not_exist.so"), "{message}");
    assert!(message.contains("searched:"), "{message}");
}
