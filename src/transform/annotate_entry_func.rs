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

use tvm_ffi::{Any, Map, ObjectRefCast, Result, String};

use super::{create_module_pass, Pass};
use crate::ir::{BaseFunc, DictAttrs, IRModule};
use crate::tirx::PrimFunc;

const GLOBAL_SYMBOL: &str = "global_symbol";
const IS_ENTRY_FUNC: &str = "tirx.is_entry_func";

/// Infer and annotate the unique TIR entry function.
///
/// This is the Rust translation of TVM's C++ `AnnotateEntryFunc` algorithm in
/// `src/tirx/transform/primfunc_utils.cc`.
fn annotate_entry_func_module(module: IRModule) -> Result<IRModule> {
    if module.functions.len() == 1 {
        let (global, function) = module
            .functions
            .iter()
            .next()
            .expect("a one-function module has one entry");
        if !has_nonzero_attr(&function, IS_ENTRY_FUNC)? {
            if let Ok(prim_func) = function.try_cast::<PrimFunc>() {
                let annotated = with_attr(prim_func, IS_ENTRY_FUNC, Any::from(true));
                return module.with_updated_function(&global, &BaseFunc::from(annotated));
            }
        }
        return Ok(module);
    }

    let mut external_prim_func = None;
    let mut external_prim_func_count = 0;
    let mut has_external_non_prim_func = false;
    for (global, function) in module.functions.iter() {
        if !has_string_attr(&function, GLOBAL_SYMBOL)? {
            continue;
        }
        match function.try_cast::<PrimFunc>() {
            Ok(prim_func) => {
                external_prim_func_count += 1;
                external_prim_func = Some((global, prim_func));
            }
            Err(_) => has_external_non_prim_func = true,
        }
    }

    if external_prim_func_count == 1 && !has_external_non_prim_func {
        let (global, function) = external_prim_func.expect("one external PrimFunc was counted");
        let annotated = with_attr(function, IS_ENTRY_FUNC, Any::from(true));
        module.with_updated_function(&global, &BaseFunc::from(annotated))
    } else {
        Ok(module)
    }
}

/// Build TVM's `tirx.AnnotateEntryFunc` module pass from the Rust translation.
pub fn annotate_entry_func() -> Result<Pass> {
    create_module_pass(
        "tirx.AnnotateEntryFunc",
        0,
        Vec::new(),
        false,
        |module, _context| annotate_entry_func_module(module),
    )
}

fn has_nonzero_attr(function: &BaseFunc, key: &str) -> Result<bool> {
    let Some(value) = function.attrs.dict.get(&String::from(key))? else {
        return Ok(false);
    };
    Ok(i64::try_from(value)? != 0)
}

fn has_string_attr(function: &BaseFunc, key: &str) -> Result<bool> {
    let Some(value) = function.attrs.dict.get(&String::from(key))? else {
        return Ok(false);
    };
    String::try_from(value)?;
    Ok(true)
}

fn with_attr(function: PrimFunc, key: &str, value: Any) -> PrimFunc {
    let key = String::from(key);
    let mut attributes = function
        .attrs
        .dict
        .iter()
        .filter(|(existing, _)| existing.as_str() != key.as_str())
        .collect::<Vec<_>>();
    attributes.push((key, value));
    let attrs = DictAttrs::from_dictionary(Map::from_iter(attributes));

    PrimFunc::from_complete_fields(
        function.span.clone(),
        function.ty.clone(),
        attrs,
        function.params.clone(),
        function.ret_type.clone(),
        function.body.clone(),
    )
}
