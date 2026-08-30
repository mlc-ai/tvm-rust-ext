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

use tvm_ffi::Result;

use super::{create_optional_prim_func_pass, Pass};
use crate::tirx::PrimFunc;

/// Keep only PrimFuncs accepted by `condition`.
///
/// This is the Rust translation of TVM's C++ `Filter` pass in
/// `src/tirx/transform/primfunc_utils.cc`.
pub fn filter<F>(condition: F) -> Result<Pass>
where
    F: Fn(PrimFunc) -> Result<bool> + 'static,
{
    create_optional_prim_func_pass(
        "tirx.Filter",
        0,
        Vec::new(),
        false,
        move |function, _module, _context| {
            if condition(function.clone())? {
                Ok(Some(function))
            } else {
                Ok(None)
            }
        },
    )
}
