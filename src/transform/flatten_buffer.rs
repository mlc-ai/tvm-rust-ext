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
    structural_mutate, Any, Array, MapValue, Mutator, ObjectIdentity, ObjectRefCast, ObjectRefCore,
    Result,
};

use super::utils::{mutate_stmt_expr_default, with_prim_func_body};
use super::{create_prim_func_pass, Pass};
use crate::analysis::{Analyzer, AnalyzerMutator, AnalyzerMutatorState};
use crate::ir::{Call, Expr, IntImm, PrimExpr, TensorLoad, Var};
use crate::tirx::{
    AllocBuffer, AttrStmt, BufferStore, BufferType, BufferVar, DeclBuffer, For, IfThenElse,
    PrimFunc, Reduce, Select, Stmt,
};

#[derive(Clone)]
struct FlatInfo {
    fold_view: BufferVar,
    flattened: BufferVar,
}

/// Flatten all non-SBlock buffer accesses in one PrimFunc.
pub fn flatten_buffer_prim_func(function: PrimFunc) -> Result<PrimFunc> {
    let mut flattener = BufferFlattener {
        analysis: AnalyzerMutatorState::new(Analyzer::new()?)?,
        flat_map: HashMap::new(),
        buffers_used: HashSet::new(),
        extern_buffers: HashSet::new(),
        masked_load_operator: get_operator("tirx.masked_load")?,
        masked_store_operator: get_operator("tirx.masked_store")?,
    };

    flattener.with_analyzer_scope(|flattener| {
        flattener.mark_buffer_parameter_shapes(&function)?;
        for parameter in function.params.iter() {
            if let Ok(buffer) = BufferVar::try_from(&parameter) {
                flattener
                    .extern_buffers
                    .insert(ObjectIdentity::of(buffer.as_var()));
                flattener.define(&buffer)?;
            }
        }

        let mut body: Stmt =
            structural_mutate(function.body.clone(), &mut *flattener)?.try_into()?;

        // PrimFunc parameters retain their public N-D contract.  The flattened
        // view used by the body aliases each parameter through a DeclBuffer.
        for parameter in function.params.iter().collect::<Vec<_>>().into_iter().rev() {
            let Ok(original) = BufferVar::try_from(&parameter) else {
                continue;
            };
            if !flattener
                .buffers_used
                .contains(&ObjectIdentity::of(original.as_var()))
            {
                continue;
            }
            let flattened = flattener.lookup(&original)?.flattened.clone();
            if !flattened.same_as(&original) {
                let data = buffer_data(&original)?;
                body = Stmt::sequence(vec![DeclBuffer::new(flattened, data)?.into(), body])?;
            }
        }

        Ok(with_prim_func_body(function, body))
    })
}

/// Build TVM's `tirx.FlattenBuffer` PrimFunc pass in Rust.
pub fn flatten_buffer() -> Result<Pass> {
    create_prim_func_pass(
        "tirx.FlattenBuffer",
        0,
        Vec::new(),
        false,
        flatten_buffer_prim_func,
    )
}

struct BufferFlattener {
    analysis: AnalyzerMutatorState,
    flat_map: HashMap<ObjectIdentity, FlatInfo>,
    buffers_used: HashSet<ObjectIdentity>,
    extern_buffers: HashSet<ObjectIdentity>,
    masked_load_operator: Expr,
    masked_store_operator: Expr,
}

impl AnalyzerMutator for BufferFlattener {
    fn analyzer_state(&self) -> &AnalyzerMutatorState {
        &self.analysis
    }

    fn analyzer_state_mut(&mut self) -> &mut AnalyzerMutatorState {
        &mut self.analysis
    }
}

impl BufferFlattener {
    fn define(&mut self, buffer: &BufferVar) -> Result<FlatInfo> {
        let identity = ObjectIdentity::of(buffer.as_var());
        if let Some(info) = self.flat_map.get(&identity) {
            return Ok(info.clone());
        }

        let old_type = buffer.type_annotation();
        let shape = self.mutate_prim_array(&old_type.shape)?;
        let strides = self.mutate_prim_array(&old_type.strides)?;
        let elem_offset = self.mutate_prim(&old_type.elem_offset)?;
        let allocated_addr = self.mutate_prim_array(&old_type.allocated_addr)?;
        let layout = super::utils::mutate_layout(&old_type.layout, |value| {
            structural_mutate(value.clone(), &mut *self)?.try_into()
        })?;
        let fold_type = BufferType::from_complete_fields(
            old_type.span.clone(),
            old_type.dtype.clone(),
            old_type.storage_scope.clone(),
            shape,
            strides,
            elem_offset,
            old_type.data_alignment,
            old_type.offset_factor,
            layout,
            allocated_addr,
        );
        let fold_view = rebuild_buffer(buffer, fold_type)?;

        let native_flattened = native_flatten_buffer(&fold_view)?;
        let flat_type = native_flattened.type_annotation();
        let shape = flat_type
            .shape
            .iter()
            .map(|extent| self.analysis.canonical_simplify(&extent))
            .collect::<Result<Vec<_>>>()?;
        let elem_offset = if is_zero(&flat_type.elem_offset) {
            flat_type.elem_offset.clone()
        } else {
            IntImm::from_dtype(flat_type.elem_offset.type_annotation().dtype, 0)?.into()
        };
        let flat_type = BufferType::from_complete_fields(
            flat_type.span.clone(),
            flat_type.dtype.clone(),
            flat_type.storage_scope.clone(),
            Array::new(shape),
            flat_type.strides.clone(),
            elem_offset,
            flat_type.data_alignment,
            flat_type.offset_factor,
            None,
            flat_type.allocated_addr.clone(),
        );

        let is_external = self.extern_buffers.contains(&identity);
        let flattened = if !is_external && structural_equal(&flat_type, &old_type)? {
            buffer.clone()
        } else {
            rebuild_buffer(buffer, flat_type)?
        };
        let info = FlatInfo {
            fold_view,
            flattened,
        };
        self.flat_map.insert(identity, info.clone());
        Ok(info)
    }

    fn lookup(&self, buffer: &BufferVar) -> Result<&FlatInfo> {
        self.flat_map
            .get(&ObjectIdentity::of(buffer.as_var()))
            .ok_or_else(|| {
                tvm_ffi::Error::new(
                    tvm_ffi::VALUE_ERROR,
                    &format!(
                        "buffer {} is used before its AllocBuffer, DeclBuffer, or PrimFunc definition",
                        buffer.name.as_str()
                    ),
                    "",
                )
            })
    }

    fn mutate_prim(&mut self, value: &PrimExpr) -> Result<PrimExpr> {
        structural_mutate(value.clone(), &mut *self)?.try_into()
    }

    fn mutate_prim_array(&mut self, values: &Array<PrimExpr>) -> Result<Array<PrimExpr>> {
        values
            .iter()
            .map(|value| self.mutate_prim(&value))
            .collect::<Result<Vec<_>>>()
            .map(Array::new)
    }

    fn mark_used(&mut self, buffer: &BufferVar) {
        self.buffers_used
            .insert(ObjectIdentity::of(buffer.as_var()));
    }

    fn fold_indices(&self, info: &FlatInfo, indices: Array<PrimExpr>) -> Result<Array<PrimExpr>> {
        let offsets = buffer_offset_of(&info.fold_view, indices)?;
        self.iter_map_simplify_with_context(&offsets, false)
    }
}

#[tvm_ffi::dispatch(mutate)]
impl BufferFlattener {
    fn mutate_variable(&mut self, value: Var) -> Result<Var> {
        let Ok(buffer) = BufferVar::try_from(&value) else {
            return Ok(value);
        };
        Ok(self
            .flat_map
            .get(&ObjectIdentity::of(buffer.as_var()))
            .map(|info| info.flattened.as_var().clone())
            .unwrap_or(value))
    }

    fn mutate_allocation(&mut self, value: AllocBuffer) -> Result<AllocBuffer> {
        let flattened = self.define(&value.buffer)?.flattened;
        if flattened.same_as(&value.buffer) {
            return Ok(value);
        }
        Ok(value.copy_with(flattened))
    }

    fn mutate_declaration(
        &mut self,
        value: DeclBuffer,
        mutator: &mut Mutator,
    ) -> Result<DeclBuffer> {
        let external_source = value
            .data
            .clone()
            .try_cast::<Call>()
            .ok()
            .filter(|call| {
                call.op.same_as(self.analysis.buffer_data_operator()) && call.args.len() == 1
            })
            .and_then(|call| call.args.get(0).ok())
            .and_then(|argument| argument.try_cast::<Var>().ok())
            .and_then(|variable| BufferVar::try_from(variable).ok())
            .is_some_and(|buffer| {
                self.extern_buffers
                    .contains(&ObjectIdentity::of(buffer.as_var()))
            });
        let data = if external_source {
            value.data.clone()
        } else {
            mutator.mutate(self, &value.data)?.try_into()?
        };
        let flattened = self.define(&value.buffer)?.flattened;
        if flattened.same_as(&value.buffer) && data.same_as(&value.data) {
            return Ok(value);
        }
        Ok(value.copy_with(flattened, data))
    }

    fn mutate_store(&mut self, value: BufferStore, mutator: &mut Mutator) -> Result<BufferStore> {
        let stored_value: PrimExpr = mutator.mutate(self, &value.value)?.try_into()?;
        let indices: Array<PrimExpr> = mutator.mutate(self, &value.indices)?.try_into()?;
        self.mark_used(&value.buffer);
        let info = self.lookup(&value.buffer)?.clone();
        let indices = self.fold_indices(&info, indices)?;
        Ok(value.copy_with(info.flattened, stored_value, indices))
    }

    fn mutate_load(&mut self, value: TensorLoad, mutator: &mut Mutator) -> Result<TensorLoad> {
        let original = BufferVar::try_from(&value.source)?;
        let indices: Array<PrimExpr> = mutator.mutate(self, &value.indices)?.try_into()?;
        self.mark_used(&original);
        let info = self.lookup(&original)?.clone();
        let indices = self.fold_indices(&info, indices)?;
        TensorLoad::from_buffer_with_span(
            info.flattened.as_var().clone(),
            indices.iter().map(Into::into).collect(),
            value.span.as_ref(),
        )
    }

    fn mutate_loop(&mut self, value: For, mutator: &mut Mutator) -> Result<For> {
        self.default_mutate_loop(mutator, value)
    }

    fn mutate_attribute(&mut self, value: AttrStmt, mutator: &mut Mutator) -> Result<AttrStmt> {
        self.default_mutate_attribute(mutator, value)
    }

    fn mutate_conditional(&mut self, value: IfThenElse, mutator: &mut Mutator) -> Result<Stmt> {
        self.default_mutate_conditional(mutator, value)
    }

    fn mutate_select(&mut self, value: Select, mutator: &mut Mutator) -> Result<PrimExpr> {
        self.default_mutate_select(mutator, value)
    }

    fn mutate_reduce(&mut self, value: Reduce, mutator: &mut Mutator) -> Result<Reduce> {
        self.default_mutate_reduce(mutator, value)
    }

    fn mutate_call(&mut self, value: Call, mutator: &mut Mutator) -> Result<Expr> {
        if value.op.same_as(self.analysis.buffer_data_operator()) && value.args.len() == 1 {
            let variable = value.args.get(0)?.try_cast::<Var>()?;
            if let Ok(buffer) = BufferVar::try_from(variable) {
                self.mark_used(&buffer);
                return buffer_data(&self.lookup(&buffer)?.flattened);
            }
        }

        if value.op.same_as(&self.masked_load_operator)
            || value.op.same_as(&self.masked_store_operator)
        {
            return self.mutate_masked_access(value, mutator);
        }

        self.default_mutate_call(mutator, value)
    }

    fn mutate_default(&mut self, value: &MapValue, mutator: &mut Mutator) -> Result<Any> {
        mutate_stmt_expr_default(self, mutator, value)
    }
}

impl BufferFlattener {
    fn mutate_masked_access(&mut self, value: Call, mutator: &mut Mutator) -> Result<Expr> {
        let is_load = value.op.same_as(&self.masked_load_operator);
        let variable = value.args.get(0)?.try_cast::<Var>()?;
        let original = BufferVar::try_from(variable)?;
        let start = if is_load { 1 } else { 2 };
        let mut indices = Vec::new();
        for index in start..value.args.len().saturating_sub(1) {
            let argument = value.args.get(index)?;
            indices.push(mutator.mutate(self, &argument)?.try_into()?);
        }
        self.mark_used(&original);
        let info = self.lookup(&original)?.clone();
        let indices = self.fold_indices(&info, Array::new(indices))?;

        let mut arguments = vec![info.flattened.as_var().clone().into()];
        if !is_load {
            let stored = value.args.get(1)?;
            arguments.push(mutator.mutate(self, &stored)?.try_into()?);
        }
        arguments.extend(indices.iter().map(Into::into));
        let mask = value.args.get(value.args.len() - 1)?;
        arguments.push(mutator.mutate(self, &mask)?.try_into()?);
        Ok(value
            .copy_with(value.ty.clone(), value.op.clone(), Array::new(arguments))
            .into())
    }
}

fn native_flatten_buffer(buffer: &BufferVar) -> Result<BufferVar> {
    tvm_ffi::cached_global_func!("tirx.BufferGetFlattenedBuffer")
        .call_tuple((buffer,))?
        .try_into()
}

fn buffer_offset_of(buffer: &BufferVar, indices: Array<PrimExpr>) -> Result<Array<PrimExpr>> {
    tvm_ffi::cached_global_func!("tirx.BufferOffsetOf")
        .call_tuple((buffer, indices))?
        .try_into()
}

fn buffer_data(buffer: &BufferVar) -> Result<Expr> {
    tvm_ffi::cached_global_func!("tirx.BufferData")
        .call_tuple((buffer,))?
        .try_into()
}

fn rebuild_buffer(buffer: &BufferVar, ty: BufferType) -> Result<BufferVar> {
    BufferVar::try_from(buffer.copy_with(buffer.name.clone(), ty.into()))
}

fn structural_equal(lhs: &BufferType, rhs: &BufferType) -> Result<bool> {
    tvm_ffi::cached_global_func!("ffi.StructuralEqual")
        .call_tuple((lhs, rhs, false, false))?
        .try_into()
}

fn is_zero(value: &PrimExpr) -> bool {
    value
        .as_expr()
        .clone()
        .try_cast::<IntImm>()
        .is_ok_and(|literal| literal.value == 0)
}

fn get_operator(name: &str) -> Result<Expr> {
    tvm_ffi::cached_global_func!("ir.GetOp")
        .call_tuple((tvm_ffi::String::from(name),))?
        .try_into()
}
