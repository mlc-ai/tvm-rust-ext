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
    structural_mutate, Any, Array, MapValue, Mutator, ObjectIdentity, ObjectRefCast, ObjectRefCore,
    Result,
};

use super::utils::{mutate_expr_default, mutate_stmt_expr_default};
use super::{create_prim_func_pass, Pass};
use crate::analysis::Analyzer;
use crate::ir::{Call, Expr, IntImm, PrimExpr, TensorLoad, Var};
use crate::target::Target;
use crate::tirx::{
    Add, AllocBuffer, BufferStore, BufferType, BufferVar, DeclBuffer, Mul, PrimFunc, Stmt, Sub,
    TileLayout,
};

/// Apply layouts and remove logical buffer offsets from one PrimFunc.
pub fn lower_tirx_cleanup_prim_func(function: PrimFunc) -> Result<PrimFunc> {
    let target = function_target(&function)?;
    let mut applier = LayoutApplier::new(target)?;
    let mut parameters = Vec::with_capacity(function.params.len());
    let mut parameter_views = Vec::new();

    for parameter in function.params.iter() {
        let Ok(buffer) = BufferVar::try_from(&parameter) else {
            parameters.push(parameter);
            continue;
        };
        applier
            .buffer_aliases
            .insert(ObjectIdentity::of(buffer.as_var()), buffer.clone());
        if buffer.type_annotation().layout.is_some() {
            let flattened = applier.flatten_buffer(&buffer, false)?;
            let mut source_type = buffer.type_annotation();
            source_type = BufferType::from_complete_fields(
                source_type.span.clone(),
                source_type.dtype.clone(),
                source_type.storage_scope.clone(),
                source_type.shape.clone(),
                source_type.strides.clone(),
                source_type.elem_offset.clone(),
                source_type.data_alignment,
                source_type.offset_factor,
                None,
                source_type.allocated_addr.clone(),
            );
            let source = rebuild_buffer(&buffer, source_type)?;
            parameters.push(source.as_var().clone());
            parameter_views.push((flattened, source));
        } else {
            parameters.push(parameter);
        }
    }

    let mut body: Stmt = structural_mutate(function.body.clone(), &mut applier)?.try_into()?;
    for (flattened, source) in parameter_views {
        body = Stmt::sequence(vec![
            DeclBuffer::new(flattened, buffer_data(&source)?)?.into(),
            body,
        ])?;
    }
    let mut remover = BufferOffsetRemover::new()?;
    body = structural_mutate(body, &mut remover)?.try_into()?;

    Ok(PrimFunc::from_complete_fields(
        function.span.clone(),
        function.ty.clone(),
        function.attrs.clone(),
        Array::new(parameters),
        function.ret_type.clone(),
        body,
    ))
}

/// Build TVM's `tirx.LowerTIRxCleanup` PrimFunc pass in Rust.
pub fn lower_tirx_cleanup() -> Result<Pass> {
    create_prim_func_pass(
        "tirx.LowerTIRxCleanup",
        0,
        Vec::new(),
        false,
        lower_tirx_cleanup_prim_func,
    )
}

struct LayoutApplier {
    analyzer: Analyzer,
    target_kind: String,
    remaps: HashMap<ObjectIdentity, BufferVar>,
    buffer_aliases: HashMap<ObjectIdentity, BufferVar>,
    buffer_data_operator: Expr,
}

impl LayoutApplier {
    fn new(target: Target) -> Result<Self> {
        Ok(Self {
            analyzer: Analyzer::new()?,
            target_kind: target.kind_name()?.as_str().to_owned(),
            remaps: HashMap::new(),
            buffer_aliases: HashMap::new(),
            buffer_data_operator: get_operator("tirx.buffer_data")?,
        })
    }

    fn use_buffer(&self, buffer: &BufferVar) -> BufferVar {
        self.remaps
            .get(&ObjectIdentity::of(buffer.as_var()))
            .cloned()
            .unwrap_or_else(|| buffer.clone())
    }

    fn mutate_primitive(&mut self, value: &PrimExpr) -> Result<PrimExpr> {
        structural_mutate(value.clone(), &mut *self)?.try_into()
    }

    fn flatten_buffer(&mut self, buffer: &BufferVar, is_allocation: bool) -> Result<BufferVar> {
        let identity = ObjectIdentity::of(buffer.as_var());
        if let Some(mapped) = self.remaps.get(&identity) {
            return Ok(mapped.clone());
        }
        let old_type = buffer.type_annotation();
        if self.target_kind == "trn" && old_type.layout.is_none() {
            return Ok(buffer.clone());
        }

        let native = if let Some(tile) = old_type
            .layout
            .as_ref()
            .and_then(|layout| layout.clone().try_cast::<TileLayout>().ok())
        {
            if tile.is_trainium()? {
                let shape = if old_type.storage_scope.as_str() == "trn.psum" {
                    vec![
                        tile.get_span(Some("Bank"))?,
                        tile.get_size(Some("P"))?,
                        tile.get_span(Some("F"))?,
                    ]
                } else {
                    vec![tile.get_size(Some("P"))?, tile.get_span(Some("F"))?]
                };
                rebuild_buffer(
                    buffer,
                    BufferType::from_complete_fields(
                        old_type.span.clone(),
                        old_type.dtype.clone(),
                        old_type.storage_scope.clone(),
                        Array::new(shape),
                        Array::new(Vec::new()),
                        old_type.elem_offset.clone(),
                        old_type.data_alignment,
                        old_type.offset_factor,
                        old_type.layout.clone(),
                        old_type.allocated_addr.clone(),
                    ),
                )?
            } else if is_allocation && tile.has_thread_axis()? {
                let mut span: PrimExpr = IntImm::new("int32", 1)?.into();
                for iteration in tile.shard()?.iter().chain(tile.replica()?.iter()) {
                    if iteration.axis.is_memory_axis()? {
                        let extent_minus_one: PrimExpr =
                            Sub::new(iteration.extent.clone(), IntImm::new("int32", 1)?)?.into();
                        let term: PrimExpr =
                            Mul::new(extent_minus_one, iteration.stride.clone())?.into();
                        span = Add::new(span, term)?.into();
                    }
                }
                for (axis, offset) in tile.offset()?.iter() {
                    if axis.is_memory_axis()? {
                        span = Add::new(span, offset)?.into();
                    }
                }
                let span = self.analyzer.simplify(&span)?;
                rebuild_buffer(
                    buffer,
                    BufferType::from_complete_fields(
                        old_type.span.clone(),
                        old_type.dtype.clone(),
                        old_type.storage_scope.clone(),
                        Array::new(vec![span]),
                        Array::new(Vec::new()),
                        old_type.elem_offset.clone(),
                        old_type.data_alignment,
                        old_type.offset_factor,
                        old_type.layout.clone(),
                        old_type.allocated_addr.clone(),
                    ),
                )?
            } else {
                native_flatten_buffer(buffer)?
            }
        } else {
            native_flatten_buffer(buffer)?
        };

        let native_type = native.type_annotation();
        let shape = native_type
            .shape
            .iter()
            .map(|value| {
                self.mutate_primitive(&value)
                    .and_then(|value| self.analyzer.canonical_simplify(&value))
            })
            .collect::<Result<Vec<_>>>()?;
        let strides = native_type
            .strides
            .iter()
            .map(|value| self.mutate_primitive(&value))
            .collect::<Result<Vec<_>>>()?;
        let elem_offset = self.mutate_primitive(&old_type.elem_offset)?;
        let new_type = BufferType::from_complete_fields(
            native_type.span.clone(),
            native_type.dtype.clone(),
            native_type.storage_scope.clone(),
            Array::new(shape),
            Array::new(strides),
            elem_offset,
            native_type.data_alignment,
            native_type.offset_factor,
            None,
            native_type.allocated_addr.clone(),
        );
        if structural_equal(&old_type, &new_type)? {
            return Ok(buffer.clone());
        }
        let mapped = rebuild_buffer(&native, new_type)?;
        self.remaps.insert(identity, mapped.clone());
        Ok(mapped)
    }

    fn register_alias(&mut self, buffer: &BufferVar, data: &Expr) -> Result<()> {
        let mut root = buffer.clone();
        if let Ok(call) = data.clone().try_cast::<Call>() {
            if call.op.same_as(&self.buffer_data_operator) && call.args.len() == 1 {
                if let Ok(source) = call.args.get(0)?.try_cast::<Var>() {
                    if let Ok(source) = BufferVar::try_from(source) {
                        root = self
                            .buffer_aliases
                            .get(&ObjectIdentity::of(source.as_var()))
                            .cloned()
                            .ok_or_else(|| {
                                tvm_ffi::Error::new(
                                    tvm_ffi::VALUE_ERROR,
                                    "buffer alias source must be defined before its alias",
                                    "",
                                )
                            })?;
                    }
                }
            }
        }
        self.buffer_aliases
            .insert(ObjectIdentity::of(buffer.as_var()), root);
        Ok(())
    }

    fn flattened_indices(
        &mut self,
        buffer: &BufferVar,
        indices: &Array<PrimExpr>,
    ) -> Result<Array<PrimExpr>> {
        let old_type = buffer.type_annotation();
        let offsets = if let Some(layout) = &old_type.layout {
            let tile = layout.clone().try_cast::<TileLayout>().ok();
            let is_trainium = tile
                .as_ref()
                .map(TileLayout::is_trainium)
                .transpose()?
                .unwrap_or(false);
            if is_trainium {
                let coordinates = layout.apply_with_shape(indices, &old_type.shape)?;
                let axes = if old_type.storage_scope.as_str() == "trn.psum" {
                    vec!["Bank", "P", "F"]
                } else {
                    vec!["P", "F"]
                };
                let mut result = Vec::with_capacity(axes.len());
                for axis in axes {
                    let value = coordinates
                        .get(&tvm_ffi::String::from(axis))?
                        .unwrap_or(IntImm::new("int32", 0)?.into());
                    result.push(self.analyzer.simplify(&value)?);
                }
                Array::new(result)
            } else {
                if tile
                    .as_ref()
                    .map(TileLayout::has_thread_axis)
                    .transpose()?
                    .unwrap_or(false)
                {
                    return Err(tvm_ffi::Error::new(
                        tvm_ffi::VALUE_ERROR,
                        "direct buffer access cannot lower a thread-axis layout",
                        "",
                    ));
                }
                let coordinates = layout
                    .canonicalize()?
                    .apply_with_shape(indices, &old_type.shape)?;
                if coordinates.len() != 1 {
                    return Err(tvm_ffi::Error::new(
                        tvm_ffi::VALUE_ERROR,
                        "layout buffer access must produce one element offset",
                        "",
                    ));
                }
                let offset = coordinates.iter().next().expect("one coordinate").1;
                Array::new(vec![self.analyzer.simplify(&offset)?])
            }
        } else {
            buffer_offset_of(buffer, indices.clone())?
        };
        if offsets.len() != 1 && old_type.layout.is_none() {
            return Err(tvm_ffi::Error::new(
                tvm_ffi::VALUE_ERROR,
                "flattened buffer access must produce one element offset",
                "",
            ));
        }
        offsets
            .iter()
            .map(|offset| self.analyzer.simplify(&offset))
            .collect::<Result<Vec<_>>>()
            .map(Array::new)
    }
}

#[tvm_ffi::dispatch(mutate)]
impl LayoutApplier {
    fn mutate_variable(&mut self, value: Var) -> Var {
        BufferVar::try_from(&value)
            .ok()
            .map(|buffer| self.use_buffer(&buffer).as_var().clone())
            .unwrap_or(value)
    }

    fn mutate_call(&mut self, value: Call, mutator: &mut Mutator) -> Result<Expr> {
        if value.op.same_as(&self.buffer_data_operator) && value.args.len() == 1 {
            let variable = value.args.get(0)?.try_cast::<Var>()?;
            if let Ok(buffer) = BufferVar::try_from(variable) {
                let root = self
                    .buffer_aliases
                    .get(&ObjectIdentity::of(buffer.as_var()))
                    .cloned()
                    .ok_or_else(|| {
                        tvm_ffi::Error::new(
                            tvm_ffi::VALUE_ERROR,
                            "buffer_data source has no visible definition",
                            "",
                        )
                    })?;
                return buffer_data(&self.use_buffer(&root));
            }
        }
        mutate_expr_default(self, mutator, value.into())
    }

    fn mutate_allocation(&mut self, value: AllocBuffer) -> Result<AllocBuffer> {
        self.buffer_aliases.insert(
            ObjectIdentity::of(value.buffer.as_var()),
            value.buffer.clone(),
        );
        let buffer = self.flatten_buffer(&value.buffer, true)?;
        if buffer.same_as(&value.buffer) {
            return Ok(value);
        }
        Ok(AllocBuffer::from_complete_fields(
            value.span.clone(),
            buffer,
            value.annotations.clone(),
        ))
    }

    fn mutate_declaration(
        &mut self,
        value: DeclBuffer,
        mutator: &mut Mutator,
    ) -> Result<DeclBuffer> {
        self.register_alias(&value.buffer, &value.data)?;
        let data: Expr = mutator.mutate(self, &value.data)?.try_into()?;
        let buffer = self.flatten_buffer(&value.buffer, false)?;
        if data.same_as(&value.data) && buffer.same_as(&value.buffer) {
            return Ok(value);
        }
        Ok(DeclBuffer::from_complete_fields(
            value.span.clone(),
            buffer,
            data,
        ))
    }

    fn mutate_store(&mut self, value: BufferStore, mutator: &mut Mutator) -> Result<BufferStore> {
        let stored: PrimExpr = mutator.mutate(self, &value.value)?.try_into()?;
        let indices: Array<PrimExpr> = mutator.mutate(self, &value.indices)?.try_into()?;
        if self.target_kind == "trn" && value.buffer.type_annotation().layout.is_none() {
            if stored.same_as(&value.value) && indices_same(&indices, &value.indices) {
                return Ok(value);
            }
            return Ok(BufferStore::from_complete_fields(
                value.span.clone(),
                value.buffer.clone(),
                stored,
                indices,
            ));
        }
        let flat_indices = self.flattened_indices(&value.buffer, &indices)?;
        let buffer = self.flatten_buffer(&value.buffer, false)?;
        Ok(BufferStore::from_complete_fields(
            value.span.clone(),
            buffer,
            stored,
            flat_indices,
        ))
    }

    fn mutate_load(&mut self, value: TensorLoad, mutator: &mut Mutator) -> Result<TensorLoad> {
        let buffer = BufferVar::try_from(&value.source)?;
        let indices: Array<PrimExpr> = mutator.mutate(self, &value.indices)?.try_into()?;
        if self.target_kind == "trn" && buffer.type_annotation().layout.is_none() {
            if indices_same(&indices, &value.indices) {
                return Ok(value);
            }
            return TensorLoad::from_buffer_with_span(
                buffer.as_var().clone(),
                indices.iter().map(Into::into).collect(),
                value.span.as_ref(),
            );
        }
        let flat_indices = self.flattened_indices(&buffer, &indices)?;
        let buffer = self.flatten_buffer(&buffer, false)?;
        TensorLoad::from_buffer_with_span(
            buffer.as_var().clone(),
            flat_indices.iter().map(Into::into).collect(),
            value.span.as_ref(),
        )
    }

    fn mutate_default(&mut self, value: &MapValue, mutator: &mut Mutator) -> Result<Any> {
        mutate_stmt_expr_default(self, mutator, value)
    }
}

struct BufferOffsetRemover {
    remaps: HashMap<ObjectIdentity, BufferVar>,
    buffer_offset_operator: Expr,
}

impl BufferOffsetRemover {
    fn new() -> Result<Self> {
        Ok(Self {
            remaps: HashMap::new(),
            buffer_offset_operator: get_operator("tirx.buffer_offset")?,
        })
    }

    fn use_buffer(&self, buffer: &BufferVar) -> BufferVar {
        self.remaps
            .get(&ObjectIdentity::of(buffer.as_var()))
            .cloned()
            .unwrap_or_else(|| buffer.clone())
    }
}

#[tvm_ffi::dispatch(mutate)]
impl BufferOffsetRemover {
    fn mutate_variable(&mut self, value: Var) -> Var {
        BufferVar::try_from(&value)
            .ok()
            .map(|buffer| self.use_buffer(&buffer).as_var().clone())
            .unwrap_or(value)
    }

    fn mutate_call(&mut self, value: Call, mutator: &mut Mutator) -> Result<Expr> {
        if value.op.same_as(&self.buffer_offset_operator) && value.args.len() == 1 {
            let load = value.args.get(0)?.try_cast::<TensorLoad>()?;
            if load.indices.len() != 1 {
                return Err(tvm_ffi::Error::new(
                    tvm_ffi::VALUE_ERROR,
                    "buffer_offset requires a one-dimensional load",
                    "",
                ));
            }
            return Ok(load.indices.get(0)?.into());
        }
        mutate_expr_default(self, mutator, value.into())
    }

    fn mutate_declaration(
        &mut self,
        value: DeclBuffer,
        mutator: &mut Mutator,
    ) -> Result<DeclBuffer> {
        let data: Expr = mutator.mutate(self, &value.data)?.try_into()?;
        let old_type = value.buffer.type_annotation();
        let elem_offset: PrimExpr = mutator.mutate(self, &old_type.elem_offset)?.try_into()?;
        let buffer = if elem_offset.same_as(&old_type.elem_offset) {
            value.buffer.clone()
        } else {
            let mapped = rebuild_buffer(
                &value.buffer,
                BufferType::from_complete_fields(
                    old_type.span.clone(),
                    old_type.dtype.clone(),
                    old_type.storage_scope.clone(),
                    old_type.shape.clone(),
                    old_type.strides.clone(),
                    elem_offset,
                    old_type.data_alignment,
                    old_type.offset_factor,
                    old_type.layout.clone(),
                    old_type.allocated_addr.clone(),
                ),
            )?;
            self.remaps
                .insert(ObjectIdentity::of(value.buffer.as_var()), mapped.clone());
            mapped
        };
        if buffer.same_as(&value.buffer) && data.same_as(&value.data) {
            return Ok(value);
        }
        Ok(DeclBuffer::from_complete_fields(
            value.span.clone(),
            buffer,
            data,
        ))
    }

    fn mutate_store(&mut self, value: BufferStore, mutator: &mut Mutator) -> Result<BufferStore> {
        let stored: PrimExpr = mutator.mutate(self, &value.value)?.try_into()?;
        let indices: Array<PrimExpr> = mutator.mutate(self, &value.indices)?.try_into()?;
        let buffer = self.use_buffer(&value.buffer);
        if buffer.same_as(&value.buffer)
            && stored.same_as(&value.value)
            && indices_same(&indices, &value.indices)
        {
            return Ok(value);
        }
        Ok(BufferStore::from_complete_fields(
            value.span.clone(),
            buffer,
            stored,
            indices,
        ))
    }

    fn mutate_load(&mut self, value: TensorLoad, mutator: &mut Mutator) -> Result<TensorLoad> {
        let original = BufferVar::try_from(&value.source)?;
        let buffer = self.use_buffer(&original);
        let indices: Array<PrimExpr> = mutator.mutate(self, &value.indices)?.try_into()?;
        if buffer.same_as(&original) && indices_same(&indices, &value.indices) {
            return Ok(value);
        }
        TensorLoad::from_buffer_with_span(
            buffer.as_var().clone(),
            indices.iter().map(Into::into).collect(),
            value.span.as_ref(),
        )
    }

    fn mutate_default(&mut self, value: &MapValue, mutator: &mut Mutator) -> Result<Any> {
        mutate_stmt_expr_default(self, mutator, value)
    }
}

fn function_target(function: &PrimFunc) -> Result<Target> {
    if let Some(target) = function.attrs.dict.get(&tvm_ffi::String::from("target"))? {
        return Target::try_from(target);
    }
    let current: Option<Target> = tvm_ffi::cached_global_func!("target.TargetCurrent")
        .call_tuple((false,))?
        .try_into()?;
    current.ok_or_else(|| {
        tvm_ffi::Error::new(
            tvm_ffi::VALUE_ERROR,
            "LowerTIRxCleanup requires a function target",
            "",
        )
    })
}

fn rebuild_buffer(buffer: &BufferVar, ty: BufferType) -> Result<BufferVar> {
    BufferVar::try_from(Var::from_complete_fields(
        buffer.span.clone(),
        ty.into(),
        buffer.name.clone(),
    ))
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

fn structural_equal(lhs: &BufferType, rhs: &BufferType) -> Result<bool> {
    tvm_ffi::cached_global_func!("ffi.StructuralEqual")
        .call_tuple((lhs, rhs, false, false))?
        .try_into()
}

fn get_operator(name: &str) -> Result<Expr> {
    tvm_ffi::cached_global_func!("ir.GetOp")
        .call_tuple((tvm_ffi::String::from(name),))?
        .try_into()
}

fn indices_same(lhs: &Array<PrimExpr>, rhs: &Array<PrimExpr>) -> bool {
    lhs.len() == rhs.len()
        && lhs
            .iter()
            .zip(rhs.iter())
            .all(|(lhs, rhs)| lhs.same_as(&rhs))
}
