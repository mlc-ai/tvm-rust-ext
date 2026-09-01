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

use tvm_ffi::extra::structural_mutate::MutateContextDriver;
use tvm_ffi::{
    structural_mutate, Any, Array, Error, Map, MapValue, MutateCallbacks, Mutator, ObjectIdentity,
    ObjectRefCast, ObjectRefCore, Result, String as FfiString, RUNTIME_ERROR,
};

use super::utils::{
    array_same_as, mutate_stmt_expr_default, option_same_as, with_prim_func_attr,
    with_prim_func_body, BufferRemaps,
};
use super::{create_prim_func_pass, Pass};
use crate::ir::{Expr, PrimExpr, TensorLoad, Var};
use crate::tirx::{
    AllocBuffer, AttrStmt, BufferStore, BufferVar, DeclBuffer, For, IterVar, PrimFunc, Stmt,
};

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
    let state = ThreadAxisRewriteState {
        thread_map,
        variable_map: HashMap::new(),
        buffer_remaps: BufferRemaps::default(),
    };
    let mut rewriter = MutateCallbacks::new(state, ThreadAxisRewriter);
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
        move |function| remap_thread_axis_prim_func(function, &thread_map),
    )
}

struct ThreadAxisRewriteState {
    thread_map: HashMap<std::string::String, IterVar>,
    variable_map: HashMap<ObjectIdentity, Var>,
    buffer_remaps: BufferRemaps,
}

struct ThreadAxisRewriter;

#[tvm_ffi::dispatch(mutate)]
impl ThreadAxisRewriter {
    fn mutate_thread_extent(
        &self,
        value: AttrStmt,
        mutator: &mut Mutator<ThreadAxisRewriteState>,
    ) -> Result<AttrStmt> {
        if value.attr_key.as_str() != THREAD_EXTENT {
            return mutate_regular_attribute(mutator, value);
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
        let Some(new_iter_var) = mutator.state().thread_map.get(thread_tag.as_str()).cloned()
        else {
            return mutate_regular_attribute(mutator, value);
        };

        let old_variable = iter_var.var()?;
        let new_variable: Var = new_iter_var.var()?.into();
        let identity = ObjectIdentity::of(&old_variable);
        if let Some(existing) = mutator.state().variable_map.get(&identity) {
            if !existing.same_as(&new_variable) {
                return Err(Error::new(
                    RUNTIME_ERROR,
                    "one thread variable cannot be remapped to two different axes",
                    "",
                ));
            }
        } else {
            mutator
                .state_mut()
                .variable_map
                .insert(identity, new_variable);
        }

        let body = Stmt::try_from(mutator.mutate(&value.body)?)?;
        AttrStmt::new(
            new_iter_var,
            value.attr_key.as_str(),
            value.value.clone(),
            body,
        )
    }

    fn mutate_loop(
        &self,
        value: For,
        mutator: &mut Mutator<ThreadAxisRewriteState>,
    ) -> Result<For> {
        // Match StmtExprMutator::VisitStmt_(For): loop metadata and the binder
        // are not recursive uses of thread variables.
        let minimum: PrimExpr = mutator.mutate(&value.min)?.try_into()?;
        let extent: PrimExpr = mutator.mutate(&value.extent)?.try_into()?;
        let step: Option<PrimExpr> = mutator.mutate(&value.step)?.try_into()?;
        let body: Stmt = mutator.mutate(&value.body)?.try_into()?;
        if minimum.same_as(&value.min)
            && extent.same_as(&value.extent)
            && option_same_as(&step, &value.step)
            && body.same_as(&value.body)
        {
            return Ok(value);
        }
        Ok(For::from_complete_fields(
            value.span.clone(),
            value.loop_var.clone(),
            minimum,
            extent,
            value.kind,
            body,
            value.thread_binding.clone(),
            value.annotations.clone(),
            step,
        ))
    }

    fn mutate_thread_variable(
        &self,
        value: Var,
        mutator: &mut Mutator<ThreadAxisRewriteState>,
    ) -> Var {
        if let Some(replacement) = mutator
            .state()
            .variable_map
            .get(&ObjectIdentity::of(&value))
        {
            return replacement.clone();
        }
        mutator.state().buffer_remaps.use_variable(&value)
    }

    fn mutate_load(
        &self,
        value: TensorLoad,
        mutator: &mut Mutator<ThreadAxisRewriteState>,
    ) -> Result<TensorLoad> {
        let source = BufferVar::try_from(value.source.clone().try_cast::<Var>()?)?;
        let source = mutator.state().buffer_remaps.use_buffer(&source);
        let indices: Array<PrimExpr> = mutator.mutate(&value.indices)?.try_into()?;
        if source.as_var().same_as(&value.source) && array_same_as(&indices, &value.indices) {
            return Ok(value);
        }
        Ok(TensorLoad::from_complete_fields(
            value.span.clone(),
            value.ty.clone().try_cast()?,
            source.into(),
            indices,
        ))
    }

    fn mutate_store(
        &self,
        value: BufferStore,
        mutator: &mut Mutator<ThreadAxisRewriteState>,
    ) -> Result<BufferStore> {
        let buffer = mutator.state().buffer_remaps.use_buffer(&value.buffer);
        let stored_value: PrimExpr = mutator.mutate(&value.value)?.try_into()?;
        let indices: Array<PrimExpr> = mutator.mutate(&value.indices)?.try_into()?;
        if buffer.same_as(&value.buffer)
            && stored_value.same_as(&value.value)
            && array_same_as(&indices, &value.indices)
        {
            return Ok(value);
        }
        Ok(BufferStore::from_complete_fields(
            value.span.clone(),
            buffer,
            stored_value,
            indices,
        ))
    }

    fn mutate_alloc_buffer(
        &self,
        value: AllocBuffer,
        mutator: &mut Mutator<ThreadAxisRewriteState>,
    ) -> Result<AllocBuffer> {
        let buffer = mutate_buffer_definition(mutator, &value.buffer)?;
        if buffer.same_as(&value.buffer) {
            return Ok(value);
        }
        Ok(AllocBuffer::from_complete_fields(
            value.span.clone(),
            buffer,
            value.annotations.clone(),
        ))
    }

    fn mutate_decl_buffer(
        &self,
        value: DeclBuffer,
        mutator: &mut Mutator<ThreadAxisRewriteState>,
    ) -> Result<DeclBuffer> {
        let data: Expr = mutator.mutate(&value.data)?.try_into()?;
        let buffer = mutate_buffer_definition(mutator, &value.buffer)?;
        if data.same_as(&value.data) && buffer.same_as(&value.buffer) {
            return Ok(value);
        }
        Ok(DeclBuffer::from_complete_fields(
            value.span.clone(),
            buffer,
            data,
        ))
    }

    fn mutate_stmt_expr_default(
        &self,
        value: &MapValue,
        mutator: &mut Mutator<ThreadAxisRewriteState>,
    ) -> Result<Any> {
        mutate_stmt_expr_default(mutator, value)
    }
}

fn mutate_buffer_definition<Driver>(
    mutator: &mut Mutator<ThreadAxisRewriteState, Driver>,
    buffer: &BufferVar,
) -> Result<BufferVar>
where
    Driver: MutateContextDriver<ThreadAxisRewriteState> + ?Sized,
{
    let mut remaps = std::mem::take(&mut mutator.state_mut().buffer_remaps);
    let result =
        remaps.mutate_definition(buffer, |expression| mutator.mutate(expression)?.try_into());
    mutator.state_mut().buffer_remaps = remaps;
    result
}

fn mutate_regular_attribute<Driver>(
    mutator: &mut Mutator<ThreadAxisRewriteState, Driver>,
    value: AttrStmt,
) -> Result<AttrStmt>
where
    Driver: MutateContextDriver<ThreadAxisRewriteState> + ?Sized,
{
    let attr_value: PrimExpr = mutator.mutate(&value.value)?.try_into()?;
    let body: Stmt = mutator.mutate(&value.body)?.try_into()?;
    if attr_value.same_as(&value.value) && body.same_as(&value.body) {
        return Ok(value);
    }
    Ok(AttrStmt::from_complete_fields(
        value.span.clone(),
        value.node.clone(),
        value.attr_key.clone(),
        attr_value,
        body,
    ))
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
