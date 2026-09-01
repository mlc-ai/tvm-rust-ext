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

use std::collections::HashMap;

use tvm_ffi::{
    structural_mutate, Array, DefRegionKind, Error, Map, ObjectIdentity, ObjectRefCore, Result,
    String as FfiString, StructuralMutator, RUNTIME_ERROR,
};

use super::utils::{with_prim_func_attr, with_prim_func_body};
use super::{create_prim_func_pass, Pass};
use crate::ir::{Expr, Var};
use crate::tirx::{AttrStmt, IterVar, PrimFunc, Stmt};

const KERNEL_LAUNCH_PARAMS: &str = "tirx.kernel_launch_params";
const THREAD_EXTENT: &str = "thread_extent";

/// Remap thread axes using the same function-attribute and statement rewrite
/// performed by TVM's `tirx::RemapThreadAxis`.
pub fn remap_thread_axis_prim_func(
    function: PrimFunc,
    thread_map: &Map<FfiString, IterVar>,
) -> Result<PrimFunc> {
    let thread_map = collect_thread_map(thread_map);
    let function = remap_launch_params(function, &thread_map)?;
    let mut rewriter = ThreadAxisRewriter {
        thread_map,
        variable_map: HashMap::new(),
    };
    let body = structural_mutate(function.body.clone(), &mut rewriter)?.try_into()?;
    Ok(with_prim_func_body(function, body))
}

/// Build TVM's `tirx.RemapThreadAxis` PrimFunc pass in Rust.
pub fn remap_thread_axis(thread_map: Map<FfiString, IterVar>) -> Result<Pass> {
    create_prim_func_pass(
        "tirx.RemapThreadAxis",
        0,
        Vec::new(),
        false,
        move |function, _module, _context| remap_thread_axis_prim_func(function, &thread_map),
    )
}

struct ThreadAxisRewriter {
    thread_map: HashMap<std::string::String, IterVar>,
    variable_map: HashMap<ObjectIdentity, Var>,
}

#[tvm_ffi::dispatch(mutate)]
impl ThreadAxisRewriter {
    fn mutate_thread_extent(&mut self, value: AttrStmt, region: DefRegionKind) -> Result<Stmt> {
        if value.attr_key.as_str() != THREAD_EXTENT {
            return self
                .default_mutate_value(&value, region)
                .and_then(Stmt::try_from);
        }

        let iter_var = IterVar::try_from(value.node.clone())?;
        let thread_tag = iter_var.thread_tag()?;
        if thread_tag.as_str().is_empty() {
            return Err(Error::new(
                RUNTIME_ERROR,
                "thread_extent IterVar must have a non-empty thread tag",
                "",
            ));
        }
        let Some(new_iter_var) = self.thread_map.get(thread_tag.as_str()).cloned() else {
            return self
                .default_mutate_value(&value, region)
                .and_then(Stmt::try_from);
        };

        let old_variable = iter_var.var()?;
        let new_variable: Var = new_iter_var.var()?.into();
        let identity = ObjectIdentity::of(&old_variable);
        if let Some(existing) = self.variable_map.get(&identity) {
            if !existing.same_as(&new_variable) {
                return Err(Error::new(
                    RUNTIME_ERROR,
                    "one thread variable cannot be remapped to two different axes",
                    "",
                ));
            }
        } else {
            self.variable_map.insert(identity, new_variable);
        }

        let body = Stmt::try_from(self.mutate(&value.body, region)?)?;
        Ok(AttrStmt::new(
            new_iter_var,
            value.attr_key.as_str(),
            value.value.clone(),
            body,
        )?
        .into())
    }

    fn mutate_thread_variable(&mut self, value: Var, region: DefRegionKind) -> Result<Expr> {
        if let Some(replacement) = self.variable_map.get(&ObjectIdentity::of(&value)) {
            return Ok(replacement.clone().into());
        }
        self.default_mutate_value(&value, region)
            .and_then(Expr::try_from)
    }
}

fn collect_thread_map(
    thread_map: &Map<FfiString, IterVar>,
) -> HashMap<std::string::String, IterVar> {
    let mut collected = HashMap::with_capacity(thread_map.len());
    for (tag, iter_var) in thread_map.iter() {
        collected.insert(tag.as_str().to_owned(), iter_var);
    }
    collected
}

fn remap_launch_params(
    function: PrimFunc,
    thread_map: &HashMap<std::string::String, IterVar>,
) -> Result<PrimFunc> {
    let Some(value) = function
        .attrs
        .dict
        .get(&FfiString::from(KERNEL_LAUNCH_PARAMS))?
    else {
        return Ok(function);
    };
    let launch_params = Array::<IterVar>::try_from(value)?;
    let mut remapped = Vec::with_capacity(launch_params.len());
    for iter_var in launch_params.iter() {
        let tag = iter_var.thread_tag()?;
        remapped.push(thread_map.get(tag.as_str()).cloned().unwrap_or(iter_var));
    }
    Ok(with_prim_func_attr(
        function,
        KERNEL_LAUNCH_PARAMS,
        Array::new(remapped),
    ))
}
