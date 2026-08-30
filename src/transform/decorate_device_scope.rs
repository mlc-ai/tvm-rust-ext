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

use tvm_ffi::{Any, Result};

use super::{create_prim_func_pass, Pass};
use crate::ir::IntImm;
use crate::tirx::{AttrStmt, PrimFunc};

const DEVICE_SCOPE: &str = "device_scope";

/// Wrap a PrimFunc body in the device-scope attribute used by TVM.
pub fn decorate_device_scope_prim_func(function: PrimFunc) -> Result<PrimFunc> {
    let body = AttrStmt::new(
        Any::from(0i64),
        DEVICE_SCOPE,
        IntImm::new("int32", 0)?,
        function.body.clone(),
    )?;
    Ok(PrimFunc::from_complete_fields(
        function.span.clone(),
        function.ty.clone(),
        function.attrs.clone(),
        function.params.clone(),
        function.ret_type.clone(),
        body.into(),
    ))
}

/// Build TVM's `s_tir.DecorateDeviceScope` PrimFunc pass in Rust.
pub fn decorate_device_scope() -> Result<Pass> {
    create_prim_func_pass(
        "s_tir.DecorateDeviceScope",
        0,
        Vec::new(),
        false,
        |function, _module, _context| decorate_device_scope_prim_func(function),
    )
}
