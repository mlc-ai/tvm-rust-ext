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

use std::collections::{HashMap, HashSet};

use tvm_ffi::{
    structural_walk, Map, Result, String as FfiString, WalkOrder, WalkResult, VALUE_ERROR,
};

use crate::ir::{BaseFunc, GlobalVar, GlobalVarObj, IRModule};

/// Keep functions reachable from explicit entry names or external linkage.
///
/// If a reachable function refers to a global that is not defined by the
/// module, this conservative prototype returns the original module unchanged.
pub fn prune_unreachable_functions(module: IRModule, entry_names: &[&str]) -> Result<IRModule> {
    let functions = module
        .functions
        .iter()
        .map(|(global, function)| (global.name_hint.as_str().to_owned(), (global, function)))
        .collect::<HashMap<_, _>>();

    let mut reachable = HashSet::new();
    let mut pending = Vec::new();
    for (name, (_, function)) in &functions {
        if has_external_linkage(function)? && reachable.insert(name.clone()) {
            pending.push(name.clone());
        }
    }
    for entry in entry_names {
        if !functions.contains_key(*entry) {
            return Err(tvm_ffi::Error::new(
                VALUE_ERROR,
                &format!("module has no entry function named `{entry}`"),
                "",
            ));
        }
        if reachable.insert((*entry).to_owned()) {
            pending.push((*entry).to_owned());
        }
    }
    if pending.is_empty() {
        return Err(tvm_ffi::Error::new(
            VALUE_ERROR,
            "function pruning requires an explicit entry or a function with `global_symbol`",
            "",
        ));
    }

    while let Some(name) = pending.pop() {
        let (_, function) = functions
            .get(&name)
            .expect("pending functions originate from the module");
        let mut callees = Vec::new();
        structural_walk(
            function,
            |global: &GlobalVarObj| {
                callees.push(global.name_hint.as_str().to_owned());
                WalkResult::Advance
            },
            WalkOrder::PreOrder,
        )?;
        for callee in callees {
            if !functions.contains_key(&callee) {
                return Ok(module);
            }
            if reachable.insert(callee.clone()) {
                pending.push(callee);
            }
        }
    }

    let retained: Map<GlobalVar, BaseFunc> = module
        .functions
        .iter()
        .filter(|(global, _)| reachable.contains(global.name_hint.as_str()))
        .collect();
    IRModule::with_metadata(
        retained,
        module.source_map.clone(),
        module.attrs.clone(),
        module.global_infos.clone(),
    )
}

/// Keep functions reachable from `main` together with all external roots.
pub fn prune_unreachable_functions_from_main(module: IRModule) -> Result<IRModule> {
    prune_unreachable_functions(module, &["main"])
}

fn has_external_linkage(function: &BaseFunc) -> Result<bool> {
    let Some(symbol) = function.attrs.dict.get(&FfiString::from("global_symbol"))? else {
        return Ok(false);
    };
    FfiString::try_from(symbol)?;
    Ok(true)
}
