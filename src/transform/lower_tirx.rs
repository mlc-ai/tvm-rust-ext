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

use tvm_ffi::{Result, String};

use super::{lower_tirx_cleanup, sequential, tile_primitive_dispatch, Pass};

/// Compose TVM's standard TIRx lowering pipeline.
pub fn lower_tirx() -> Result<Pass> {
    let mut passes = vec![tile_primitive_dispatch()?];
    if std::env::var_os("TVM_PRINT_AFTER_TIRX_DISPATCH_OPS").is_some() {
        passes.push(
            tvm_ffi::cached_global_func!("transform.PrintIR")
                .call_tuple((String::from(""),))?
                .try_into()?,
        );
    }
    passes.push(lower_tirx_cleanup()?);
    sequential(passes, "tirx.LowerTIRx")
}
