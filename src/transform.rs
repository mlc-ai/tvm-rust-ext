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

use tvm_ffi::derive::{Object, ObjectRef};
use tvm_ffi::{
    Any, Array, FieldGetter, Function, Map, ObjectArc, ObjectCore, RValueRef, Result, String,
};

use crate::ir::IRModule;
use crate::tirx::PrimFunc;

mod annotate_entry_func;
mod common_subexpr_elim;
mod convert_ssa;
mod decorate_device_scope;
mod filter;
mod inline_private_functions;
mod lower_tirx_opaque;
mod remap_thread_axis;
mod remove_assume;
mod remove_no_op;
mod skip_assert;
mod unroll_loop;
mod utils;

pub use annotate_entry_func::annotate_entry_func;
pub use common_subexpr_elim::{common_subexpr_elim, common_subexpr_elim_prim_func};
pub use convert_ssa::{convert_ssa, convert_ssa_module, convert_ssa_prim_func};
pub use decorate_device_scope::{decorate_device_scope, decorate_device_scope_prim_func};
pub use filter::filter;
pub use inline_private_functions::{inline_private_functions, inline_private_functions_module};
pub use lower_tirx_opaque::{lower_tirx_opaque, lower_tirx_opaque_prim_func};
pub use remap_thread_axis::{remap_thread_axis, remap_thread_axis_prim_func};
pub use remove_assume::{remove_assume, remove_assume_internal, remove_assume_prim_func};
pub use remove_no_op::{remove_no_op, remove_no_op_prim_func};
pub use skip_assert::{skip_assert, skip_assert_prim_func};
pub use unroll_loop::{unroll_loop, unroll_loop_prim_func};

/// Opaque Rust view of TVM's `PassNode` prefix.
#[repr(C)]
#[derive(Object)]
#[type_key = "transform.Pass"]
pub struct PassObj {
    base: tvm_ffi::Object,
}

/// Reference-counted handle to a TVM pass.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct Pass {
    data: ObjectArc<PassObj>,
}

impl std::ops::Deref for Pass {
    type Target = PassObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

/// Opaque Rust view of TVM's `PassContextNode`.
#[repr(C)]
#[derive(Object)]
#[type_key = "transform.PassContext"]
#[type_final]
pub struct PassContextObj {
    base: tvm_ffi::Object,
}

/// Reference-counted handle to the active TVM pass context.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct PassContext {
    data: ObjectArc<PassContextObj>,
}

impl std::ops::Deref for PassContext {
    type Target = PassContextObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl PassContext {
    /// Return the language-independent pass-configuration map.
    pub fn config(&self) -> Result<Map<String, Any>> {
        FieldGetter::new(PassContextObj::type_index(), "config")?.get(&**self)
    }
}

impl Pass {
    /// Run this pass on an IRModule using TVM's current PassContext.
    ///
    /// This consumes the Rust module handle and transfers its strong reference
    /// through the same rvalue-reference ABI used by C++ passes.
    pub fn run(&self, module: IRModule) -> Result<IRModule> {
        tvm_ffi::cached_global_func!("transform.RunPass")
            .call_tuple((self, RValueRef::new(module)))?
            .try_into()
    }
}

/// Compose passes in order using TVM's language-independent pass container.
pub fn sequential(passes: Vec<Pass>, name: &str) -> Result<Pass> {
    let passes = Array::new(passes);
    let required = Array::<String>::new(Vec::new());
    tvm_ffi::cached_global_func!("transform.Sequential")
        .call_tuple((passes, 0_i64, String::from(name), required, false))?
        .try_into()
}

/// Construct a TVM PrimFunc pass backed by a Rust callback.
///
/// The callback signature mirrors C++ `CreatePrimFuncPass`: the function is
/// passed as an ABI rvalue reference, followed by the module and pass context.
pub fn create_prim_func_pass<F>(
    name: &str,
    opt_level: i64,
    required: Vec<&str>,
    traceable: bool,
    pass_func: F,
) -> Result<Pass>
where
    F: Fn(PrimFunc, IRModule, PassContext) -> Result<PrimFunc> + 'static,
{
    let pass_func = Function::from_typed(
        move |func: RValueRef<PrimFunc>, module: IRModule, context: PassContext| {
            pass_func(func.into_inner(), module, context)
        },
    );

    let pass_info = create_pass_info(name, opt_level, required, traceable)?;

    tvm_ffi::cached_global_func!("tirx.transform.CreatePrimFuncPass")
        .call_tuple((pass_func, pass_info))?
        .try_into()
}

/// Construct a TVM PrimFunc pass whose callback may remove the current function.
///
/// A `None` result has the same meaning as a null `PrimFunc` returned by C++:
/// the native PrimFunc pass removes that function from its module.
pub(super) fn create_optional_prim_func_pass<F>(
    name: &str,
    opt_level: i64,
    required: Vec<&str>,
    traceable: bool,
    pass_func: F,
) -> Result<Pass>
where
    F: Fn(PrimFunc, IRModule, PassContext) -> Result<Option<PrimFunc>> + 'static,
{
    let pass_func = Function::from_typed(
        move |func: RValueRef<PrimFunc>, module: IRModule, context: PassContext| {
            pass_func(func.into_inner(), module, context)
        },
    );
    let pass_info = create_pass_info(name, opt_level, required, traceable)?;

    tvm_ffi::cached_global_func!("tirx.transform.CreatePrimFuncPass")
        .call_tuple((pass_func, pass_info))?
        .try_into()
}

/// Construct a TVM module pass backed by a Rust callback.
pub fn create_module_pass<F>(
    name: &str,
    opt_level: i64,
    required: Vec<&str>,
    traceable: bool,
    pass_func: F,
) -> Result<Pass>
where
    F: Fn(IRModule, PassContext) -> Result<IRModule> + 'static,
{
    let pass_func =
        Function::from_typed(move |module: RValueRef<IRModule>, context: PassContext| {
            pass_func(module.into_inner(), context)
        });
    let pass_info = create_pass_info(name, opt_level, required, traceable)?;

    tvm_ffi::cached_global_func!("transform.MakeModulePass")
        .call_tuple((pass_func, pass_info))?
        .try_into()
}

fn create_pass_info(
    name: &str,
    opt_level: i64,
    required: Vec<&str>,
    traceable: bool,
) -> Result<Any> {
    let required = Array::<String>::new(required.into_iter().map(String::from).collect());
    let name = String::from(name);
    tvm_ffi::cached_global_func!("transform.PassInfo")
        .call_tuple((opt_level, name, required, traceable))
}
