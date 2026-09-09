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

//! Native libraries use explicit path overrides or build-time package discovery.

use std::path::PathBuf;

use tvm::libinfo;
use tvm::tvm_ffi::Function;

#[test]
fn native_libraries_resolve_from_configured_paths() {
    let path = libinfo::compiler_library_path().unwrap();
    assert!(path.is_file(), "{}", path.display());
    if let Some(explicit) = std::env::var_os("TVM_COMPILER_LIBRARY") {
        assert_eq!(path, PathBuf::from(explicit));
    } else {
        assert!(libinfo::library_dirs()
            .iter()
            .any(|dir| dir.join(libinfo::COMPILER_LIBRARY) == path));
    }
    let runtime = libinfo::runtime_library_path().unwrap();
    assert!(runtime.is_file(), "{}", runtime.display());
    assert!(libinfo::library_dirs()
        .iter()
        .any(|dir| dir.join(libinfo::RUNTIME_LIBRARY) == runtime));
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
    let threads = (0..4)
        .map(|_| std::thread::spawn(|| libinfo::load_compiler().unwrap()))
        .collect::<Vec<_>>();
    for thread in threads {
        thread.join().unwrap();
    }
    libinfo::load_compiler().unwrap();
    Function::get_global("ir.OpGetAttr").unwrap();
    Function::get_global("tirx.PrimFunc").unwrap();
}

#[test]
fn missing_library_reports_searched_directories() {
    let error = libinfo::find_library("libtvm_does_not_exist.so").unwrap_err();
    assert_eq!(error.file_name, "libtvm_does_not_exist.so");
    assert!(!error.searched.is_empty());
    let message = error.to_string();
    assert!(
        message.contains("cannot find libtvm_does_not_exist.so"),
        "{message}"
    );
    assert!(message.contains("searched:"), "{message}");
}
