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
    structural_mutate, Any, Array, DefRegionKind, Error, Map, MapValue, ObjectIdentity,
    ObjectRefCast, ObjectRefCore, Result, String as FfiString, StructuralMutator, RUNTIME_ERROR,
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
    let mut rewriter = ThreadAxisRewriter {
        thread_map,
        variable_map: HashMap::new(),
        buffer_remaps: BufferRemaps::default(),
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
    buffer_remaps: BufferRemaps,
}

#[tvm_ffi::dispatch(mutate)]
impl ThreadAxisRewriter {
    fn mutate_thread_extent(&mut self, value: AttrStmt, region: DefRegionKind) -> Result<AttrStmt> {
        if value.attr_key.as_str() != THREAD_EXTENT {
            return self.mutate_regular_attribute(value, region);
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
            return self.mutate_regular_attribute(value, region);
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
        AttrStmt::new(
            new_iter_var,
            value.attr_key.as_str(),
            value.value.clone(),
            body,
        )
    }

    fn mutate_loop(&mut self, value: For, region: DefRegionKind) -> Result<For> {
        // Match StmtExprMutator::VisitStmt_(For): loop metadata and the binder
        // are not recursive uses of thread variables.
        let minimum: PrimExpr = self.mutate(&value.min, region)?.try_into()?;
        let extent: PrimExpr = self.mutate(&value.extent, region)?.try_into()?;
        let step: Option<PrimExpr> = self.mutate(&value.step, region)?.try_into()?;
        let body: Stmt = self.mutate(&value.body, region)?.try_into()?;
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

    fn mutate_thread_variable(&mut self, value: Var) -> Var {
        if let Some(replacement) = self.variable_map.get(&ObjectIdentity::of(&value)) {
            return replacement.clone();
        }
        self.buffer_remaps.use_variable(&value)
    }

    fn mutate_load(&mut self, value: TensorLoad, region: DefRegionKind) -> Result<TensorLoad> {
        let source = BufferVar::try_from(value.source.clone().try_cast::<Var>()?)?;
        let source = self.buffer_remaps.use_buffer(&source);
        let indices: Array<PrimExpr> = self.mutate(&value.indices, region)?.try_into()?;
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

    fn mutate_store(&mut self, value: BufferStore, region: DefRegionKind) -> Result<BufferStore> {
        let buffer = self.buffer_remaps.use_buffer(&value.buffer);
        let stored_value: PrimExpr = self.mutate(&value.value, region)?.try_into()?;
        let indices: Array<PrimExpr> = self.mutate(&value.indices, region)?.try_into()?;
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
        &mut self,
        value: AllocBuffer,
        region: DefRegionKind,
    ) -> Result<AllocBuffer> {
        let buffer = self.mutate_buffer_definition(&value.buffer, region)?;
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
        &mut self,
        value: DeclBuffer,
        region: DefRegionKind,
    ) -> Result<DeclBuffer> {
        let data: Expr = self.mutate(&value.data, region)?.try_into()?;
        let buffer = self.mutate_buffer_definition(&value.buffer, region)?;
        if data.same_as(&value.data) && buffer.same_as(&value.buffer) {
            return Ok(value);
        }
        Ok(DeclBuffer::from_complete_fields(
            value.span.clone(),
            buffer,
            data,
        ))
    }

    fn mutate_stmt_expr_default(&mut self, value: &MapValue, region: DefRegionKind) -> Result<Any> {
        mutate_stmt_expr_default(self, value, region)
    }
}

impl ThreadAxisRewriter {
    fn mutate_buffer_definition(
        &mut self,
        buffer: &BufferVar,
        region: DefRegionKind,
    ) -> Result<BufferVar> {
        let mut remaps = std::mem::take(&mut self.buffer_remaps);
        let result = remaps.mutate_definition(buffer, |expression| {
            self.mutate(expression, region)?.try_into()
        });
        self.buffer_remaps = remaps;
        result
    }

    fn mutate_regular_attribute(
        &mut self,
        value: AttrStmt,
        region: DefRegionKind,
    ) -> Result<AttrStmt> {
        let attr_value: PrimExpr = self.mutate(&value.value, region)?.try_into()?;
        let body: Stmt = self.mutate(&value.body, region)?.try_into()?;
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
