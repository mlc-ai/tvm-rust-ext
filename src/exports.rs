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
//! simplified = lib["remove_no_op"](prim_func)
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
use crate::transform;

// PrimFunc -> PrimFunc transformations.
tvm_ffi_dll_export_typed_func!(skip_assert, transform::skip_assert_prim_func);
tvm_ffi_dll_export_typed_func!(lower_tirx_opaque, transform::lower_tirx_opaque_prim_func);
tvm_ffi_dll_export_typed_func!(remove_no_op, transform::remove_no_op_prim_func);
tvm_ffi_dll_export_typed_func!(remove_assume, transform::remove_assume_prim_func);
tvm_ffi_dll_export_typed_func!(unroll_loop, transform::unroll_loop_prim_func);
tvm_ffi_dll_export_typed_func!(
    common_subexpr_elim,
    transform::common_subexpr_elim_prim_func
);
tvm_ffi_dll_export_typed_func!(
    decorate_device_scope,
    transform::decorate_device_scope_prim_func
);

// IRModule -> IRModule transformations.
tvm_ffi_dll_export_typed_func!(
    inline_private_functions,
    transform::inline_private_functions_module
);
tvm_ffi_dll_export_typed_func!(convert_ssa, transform::convert_ssa_module);

// Pass factories: each returns a `transform.Pass` object that composes with
// TVM's pass infrastructure (`tvm.transform.Sequential`, `PassContext`, ...).
tvm_ffi_dll_export_typed_func!(skip_assert_pass, transform::skip_assert);
tvm_ffi_dll_export_typed_func!(annotate_entry_func_pass, transform::annotate_entry_func);
tvm_ffi_dll_export_typed_func!(decorate_device_scope_pass, transform::decorate_device_scope);
tvm_ffi_dll_export_typed_func!(lower_tirx_opaque_pass, transform::lower_tirx_opaque);
tvm_ffi_dll_export_typed_func!(remap_thread_axis_pass, transform::remap_thread_axis);
tvm_ffi_dll_export_typed_func!(remove_no_op_pass, transform::remove_no_op);
tvm_ffi_dll_export_typed_func!(remove_assume_pass, transform::remove_assume);
tvm_ffi_dll_export_typed_func!(unroll_loop_pass, transform::unroll_loop);
tvm_ffi_dll_export_typed_func!(
    inline_private_functions_pass,
    transform::inline_private_functions
);
tvm_ffi_dll_export_typed_func!(convert_ssa_pass, transform::convert_ssa);
tvm_ffi_dll_export_typed_func!(common_subexpr_elim_pass, transform::common_subexpr_elim);

// Analyses.
tvm_ffi_dll_export_typed_func!(expr_complexity, |func: PrimFunc| {
    analysis::expr_complexity(&func).map(|count| count as i64)
});
