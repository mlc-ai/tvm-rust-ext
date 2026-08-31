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

//! Functions exported from the shared-library build of this crate.
//!
//! `Cargo.toml` builds the crate both as an `rlib` (for Rust consumers and
//! the tests) and as a `cdylib`, `target/<profile>/libtvm.so`.  Every
//! `tvm_ffi_dll_export_typed_func!(name, f)` below emits a `#[no_mangle]`
//! `__tvm_ffi_<name>` symbol that follows tvm-ffi's DSO export ABI, so any
//! tvm-ffi host can load the library and call `name` — for example from
//! Python:
//!
//! ```python
//! import tvm                                  # loads libtvm_compiler first
//! import tvm_ffi
//! lib = tvm_ffi.load_module("target/debug/libtvm.so")
//! simplified = lib["simplify_add_zero"](prim_func)
//! ```
//!
//! Arguments and results cross the boundary as TVM objects: a
//! `tvm.tirx.PrimFunc` goes in, a `PrimFunc` comes back.  The host must have
//! loaded `libtvm_compiler` before calling (importing `tvm` in Python does),
//! because several passes use registered compiler services such as
//! `arith.Analyzer` and `tirx.transform.CreatePrimFuncPass`.
//!
//! See `python/demo.py` for an end-to-end example.

use tvm_ffi::tvm_ffi_dll_export_typed_func;

use crate::analysis;
use crate::tirx::PrimFunc;
use crate::transform::{self, examples};

// PrimFunc -> PrimFunc transformations.
tvm_ffi_dll_export_typed_func!(simplify_add_zero, examples::simplify_add_zero_prim_func);
tvm_ffi_dll_export_typed_func!(fold_integer_constants, examples::fold_integer_constants_prim_func);
tvm_ffi_dll_export_typed_func!(
    simplify_neutral_elements,
    examples::simplify_neutral_elements_prim_func
);
tvm_ffi_dll_export_typed_func!(
    simplify_known_control_flow,
    examples::simplify_known_control_flow_prim_func
);
tvm_ffi_dll_export_typed_func!(eliminate_unit_loops, examples::eliminate_unit_loops_prim_func);
tvm_ffi_dll_export_typed_func!(skip_assert, transform::skip_assert_prim_func);
tvm_ffi_dll_export_typed_func!(
    convert_blocks_to_opaque,
    transform::convert_blocks_to_opaque_prim_func
);
tvm_ffi_dll_export_typed_func!(lower_init_block, transform::lower_init_block_prim_func);
tvm_ffi_dll_export_typed_func!(
    decorate_device_scope,
    transform::decorate_device_scope_prim_func
);

// IRModule -> IRModule transformations.
tvm_ffi_dll_export_typed_func!(simplify_add_zero_module, examples::simplify_add_zero_module);
tvm_ffi_dll_export_typed_func!(
    prune_unreachable_functions_from_main,
    examples::prune_unreachable_functions_from_main
);

// Pass factories: each returns a `transform.Pass` object that composes with
// TVM's pass infrastructure (`tvm.transform.Sequential`, `PassContext`, ...).
tvm_ffi_dll_export_typed_func!(skip_assert_pass, transform::skip_assert);
tvm_ffi_dll_export_typed_func!(annotate_entry_func_pass, transform::annotate_entry_func);
tvm_ffi_dll_export_typed_func!(convert_blocks_to_opaque_pass, transform::convert_blocks_to_opaque);
tvm_ffi_dll_export_typed_func!(lower_init_block_pass, transform::lower_init_block);
tvm_ffi_dll_export_typed_func!(decorate_device_scope_pass, transform::decorate_device_scope);

// Analyses.
tvm_ffi_dll_export_typed_func!(expr_complexity, |func: PrimFunc| {
    analysis::expr_complexity(&func).map(|count| count as i64)
});
