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
    structural_mutate, structural_visit, Any, Array, DLDataType, DLDataTypeCode, MapValue, Mutator,
    ObjectIdentity, ObjectRefCast, ObjectRefCore, Result, VisitCallbacks, VisitContext,
    VisitInterrupt, VisitValue,
};

use super::utils::{
    array_same_as, mutate_expr_default, mutate_stmt_default, visit_stmt_expr_default,
};
use super::{create_prim_func_pass, Pass};
use crate::analysis::Analyzer;
use crate::ir::{Call, Expr, IntImm, PointerType, PrimExpr, PrimType, TensorLoad, Var};
use crate::tirx::{
    AllocBuffer, AttrStmt, Bind, BufferStore, BufferType, BufferVar, DeclBuffer, Let, PrimFunc,
    Ramp, Stmt,
};

const PRIM_FUNC_BUFFER_PARAM: u8 = 1 << 0;
const PRIM_FUNC_POINTER_PARAM: u8 = 1 << 1;
const ALLOC_BUFFER_NODE: u8 = 1 << 2;
const LET_NODE: u8 = 1 << 3;
const DECL_BUFFER_NODE: u8 = 1 << 4;

/// Rewrite pointer element types to the vector types consistently used by their accesses.
pub fn pointer_value_type_rewrite_prim_func(function: PrimFunc) -> Result<PrimFunc> {
    pointer_value_type_rewrite_with_options(function, RewriteOptions::default())
}

/// Build TVM's `tirx.PointerValueTypeRewrite` pass in Rust.
pub fn pointer_value_type_rewrite() -> Result<Pass> {
    create_prim_func_pass(
        "tirx.PointerValueTypeRewrite",
        0,
        Vec::new(),
        false,
        pointer_value_type_rewrite_prim_func,
    )
}

#[derive(Clone, Copy)]
pub(super) struct RewriteOptions {
    pub(super) allow_untyped_pointers: bool,
    pub(super) rewrite_buffer_params: bool,
    pub(super) rewrite_pointer_params: bool,
    pub(super) rewrite_alloc_buffers: bool,
    pub(super) rewrite_indices: bool,
    pub(super) rewrite_let_bindings: bool,
    pub(super) detect_scalar_read_patterns: bool,
}

impl Default for RewriteOptions {
    fn default() -> Self {
        Self {
            allow_untyped_pointers: false,
            rewrite_buffer_params: true,
            rewrite_pointer_params: true,
            rewrite_alloc_buffers: true,
            rewrite_indices: true,
            rewrite_let_bindings: true,
            detect_scalar_read_patterns: true,
        }
    }
}

pub(super) fn pointer_value_type_rewrite_with_options(
    function: PrimFunc,
    options: RewriteOptions,
) -> Result<PrimFunc> {
    let mut checker = AccessChecker::new(options)?;
    for parameter in function.params.iter() {
        checker.register_parameter(parameter)?;
    }
    let aliases = checker.aliases.clone();
    let analyzer = checker.analyzer.clone();
    let mut callbacks = VisitCallbacks::new(
        checker,
        (
            check_load,
            check_store,
            check_call,
            check_allocation,
            check_declaration,
            check_let,
            check_binding,
            check_default,
        ),
    );
    structural_visit(&function.body, &mut callbacks)?;
    let checker = callbacks.into_state();

    let mut rewriter = VectorTypeRewriter::new(checker.infos, aliases, analyzer, options)?;
    let body: Stmt = structural_mutate(function.body.clone(), &mut rewriter)?.try_into()?;
    let substitutions = rewriter.variable_substitutions();
    let mut substituter = PointerVarSubstituter::new(substitutions.clone());
    let body: Stmt = structural_mutate(body, &mut substituter)?.try_into()?;
    let params = function
        .params
        .iter()
        .map(|parameter| {
            substitutions
                .get(&ObjectIdentity::of(&parameter))
                .cloned()
                .unwrap_or(parameter)
        })
        .collect::<Vec<_>>();
    Ok(PrimFunc::from_complete_fields(
        function.span.clone(),
        function.ty.clone(),
        function.attrs.clone(),
        Array::new(params),
        function.ret_type.clone(),
        body,
    ))
}

struct BufferInfo {
    variable: Var,
    element_dtype: PrimType,
    extent: PrimExpr,
    declaration_location: u8,
    access_dtypes: Vec<DLDataType>,
    scalar_read_dtypes: Vec<DLDataType>,
}

impl BufferInfo {
    fn preferred_dtype(&self, analyzer: &Analyzer) -> Result<PrimType> {
        let mut base_types = Vec::new();
        for dtype in self
            .access_dtypes
            .iter()
            .chain(self.scalar_read_dtypes.iter())
        {
            insert_unique_dtype(&mut base_types, with_dtype_lanes(*dtype, 1));
        }
        if base_types.len() != 1 {
            return Ok(self.element_dtype.clone());
        }

        let mut preferred = base_types[0];
        preferred.lanes = self.element_dtype.dtype.lanes;
        if self.element_dtype.dtype.lanes == 1 && self.access_dtypes.len() == 1 {
            let lanes = self.access_dtypes[0].lanes;
            if self
                .scalar_read_dtypes
                .iter()
                .any(|dtype| dtype.lanes % lanes != 0)
            {
                return Ok(self.element_dtype.clone());
            }
            let modular = analyzer.modular_set(&self.extent)?;
            if modular.coeff % i64::from(lanes) == 0 && modular.base_value % i64::from(lanes) == 0 {
                preferred.lanes = lanes;
            }
        }
        PrimType::from_dtype(preferred)
    }
}

struct AccessChecker {
    infos: HashMap<ObjectIdentity, BufferInfo>,
    aliases: HashMap<ObjectIdentity, Var>,
    analyzer: Analyzer,
    options: RewriteOptions,
    masked_load: ObjectIdentity,
    masked_store: ObjectIdentity,
    access_ptr: ObjectIdentity,
    address_of: ObjectIdentity,
    buffer_data: ObjectIdentity,
}

impl AccessChecker {
    fn new(options: RewriteOptions) -> Result<Self> {
        Ok(Self {
            infos: HashMap::new(),
            aliases: HashMap::new(),
            analyzer: Analyzer::new()?,
            options,
            masked_load: operator_identity("tirx.masked_load")?,
            masked_store: operator_identity("tirx.masked_store")?,
            access_ptr: operator_identity("tirx.tvm_access_ptr")?,
            address_of: operator_identity("tirx.address_of")?,
            buffer_data: operator_identity("tirx.buffer_data")?,
        })
    }

    fn register_parameter(&mut self, variable: Var) -> Result<()> {
        if let Ok(buffer) = BufferVar::try_from(&variable) {
            self.aliases
                .insert(ObjectIdentity::of(&variable), variable.clone());
            let ty = buffer.type_annotation();
            let extent = ty
                .shape
                .iter()
                .last()
                .unwrap_or(IntImm::new("int32", 0)?.into());
            return self.declare(variable, ty.dtype.clone(), extent, PRIM_FUNC_BUFFER_PARAM);
        }
        if let Some(element) = pointer_element_type(&variable)? {
            if !is_void(&element) || self.options.allow_untyped_pointers {
                self.declare(
                    variable,
                    element,
                    IntImm::new("int32", 0)?.into(),
                    PRIM_FUNC_POINTER_PARAM,
                )?;
            }
        }
        Ok(())
    }

    fn declare(
        &mut self,
        variable: Var,
        mut element_dtype: PrimType,
        extent: PrimExpr,
        declaration_location: u8,
    ) -> Result<()> {
        let identity = ObjectIdentity::of(&variable);
        if self.infos.contains_key(&identity) {
            return Err(value_error(&format!(
                "array declaration of {} occurred multiple times",
                variable.name.as_str()
            )));
        }
        if element_dtype.dtype.code == DLDataTypeCode::kDLBool as u8 {
            let mut dtype = element_dtype.dtype;
            dtype.code = DLDataTypeCode::kDLInt as u8;
            dtype.bits = 8;
            element_dtype = PrimType::from_dtype(dtype)?;
        }
        self.infos.insert(
            identity,
            BufferInfo {
                variable,
                element_dtype,
                extent,
                declaration_location,
                access_dtypes: Vec::new(),
                scalar_read_dtypes: Vec::new(),
            },
        );
        Ok(())
    }

    fn register_alias(&mut self, buffer: &BufferVar, data: &Expr) -> Result<()> {
        let mut root = buffer.as_var().clone();
        if let Some(source) = get_buffer_data_var(data, &self.buffer_data)? {
            if BufferVar::try_from(&source).is_ok() {
                root = self
                    .aliases
                    .get(&ObjectIdentity::of(&source))
                    .cloned()
                    .ok_or_else(|| value_error("buffer alias source was not declared"))?;
            }
        }
        self.aliases
            .insert(ObjectIdentity::of(buffer.as_var()), root);
        Ok(())
    }

    fn declare_pointer_binding(&mut self, variable: Var) -> Result<()> {
        if let Some(element) = pointer_element_type(&variable)? {
            if !is_void(&element) || self.options.allow_untyped_pointers {
                self.declare(variable, element, IntImm::new("int32", 0)?.into(), LET_NODE)?;
            }
        }
        Ok(())
    }

    fn record_access(
        &mut self,
        mut value_dtype: PrimType,
        buffer: &Var,
        indices: &Array<PrimExpr>,
        is_buffer_load: bool,
    ) -> Result<()> {
        if is_scalable_vector(&value_dtype) {
            return Ok(());
        }
        let original_identity = ObjectIdentity::of(buffer);
        let root = self
            .aliases
            .get(&original_identity)
            .cloned()
            .unwrap_or_else(|| buffer.clone());
        let root_identity = ObjectIdentity::of(&root);
        let identity = if self.infos.contains_key(&root_identity) {
            root_identity
        } else {
            original_identity
        };
        let info = self.infos.get_mut(&identity).ok_or_else(|| {
            value_error(&format!(
                "load/store of buffer {} occurred before its declaration",
                buffer.name.as_str()
            ))
        })?;
        if value_dtype.dtype.code == DLDataTypeCode::kDLBool as u8 {
            let mut dtype = value_dtype.dtype;
            dtype.code = DLDataTypeCode::kDLInt as u8;
            dtype.bits = 8;
            value_dtype = PrimType::from_dtype(dtype)?;
        }
        if is_void(&info.element_dtype) {
            if !self.options.allow_untyped_pointers {
                return Err(value_error(
                    "pointer declaration is missing an element type",
                ));
            }
            info.element_dtype = PrimType::from_dtype(with_dtype_lanes(value_dtype.dtype, 1))?;
        }
        for index in indices.iter().take(indices.len().saturating_sub(1)) {
            if index.type_annotation().dtype.lanes != 1 {
                return Err(value_error(
                    "only the last index of a buffer access may be vector-valued",
                ));
            }
        }
        let index_lanes = indices
            .iter()
            .last()
            .map(|index| index.type_annotation().dtype.lanes)
            .unwrap_or(1);
        let mut lanes_used = info.element_dtype.dtype.lanes;
        if u32::from(index_lanes) * u32::from(info.element_dtype.dtype.lanes)
            != u32::from(value_dtype.dtype.lanes)
        {
            if index_lanes != value_dtype.dtype.lanes {
                return Err(value_error(
                    "buffer access lane count is incompatible with its element type",
                ));
            }
            lanes_used = 1;
            info.element_dtype =
                PrimType::from_dtype(with_dtype_lanes(info.element_dtype.dtype, 1))?;
        }

        if let Some(index) = indices.iter().last() {
            if let Ok(ramp) = index.clone().try_cast::<Ramp>() {
                if int_value(&ramp.stride) == Some(1) {
                    if let Some(lanes) = int_value(&ramp.lanes) {
                        let modular = self.analyzer.modular_set(&ramp.base)?;
                        if lanes > 0
                            && modular.coeff % lanes == 0
                            && modular.base_value % lanes == 0
                        {
                            lanes_used = u16::try_from(lanes)
                                .map_err(|_| value_error("ramp lane count exceeds u16"))?;
                        }
                    }
                }
            }
        }
        let access_dtype = with_dtype_lanes(value_dtype.dtype, lanes_used);
        if self.options.detect_scalar_read_patterns && is_buffer_load {
            if let Some(index) = indices.iter().last() {
                if index.type_annotation().dtype.lanes == 1 {
                    let modular = self.analyzer.modular_set(&index)?;
                    if let Ok(lanes) = u16::try_from(modular.coeff) {
                        insert_unique_dtype(
                            &mut info.scalar_read_dtypes,
                            with_dtype_lanes(access_dtype, lanes),
                        );
                        return Ok(());
                    }
                }
            }
        }
        insert_unique_dtype(&mut info.access_dtypes, access_dtype);
        Ok(())
    }
}

fn check_load(value: TensorLoad, visitor: &mut VisitContext<'_, AccessChecker>) -> Result<()> {
    let buffer = BufferVar::try_from(&value.source)?;
    visitor.state_mut().record_access(
        value.ty.clone().try_cast()?,
        buffer.as_var(),
        &value.indices,
        true,
    )?;
    visit_values(visitor, value.indices.iter().map(Into::into))
}

fn check_store(value: BufferStore, visitor: &mut VisitContext<'_, AccessChecker>) -> Result<()> {
    visitor.state_mut().record_access(
        value.value.type_annotation(),
        value.buffer.as_var(),
        &value.indices,
        false,
    )?;
    visitor.visit(&value.value)?;
    visit_values(visitor, value.indices.iter().map(Into::into))
}

fn check_call(value: Call, visitor: &mut VisitContext<'_, AccessChecker>) -> Result<()> {
    let operator = ObjectIdentity::of(&value.op);
    let (masked_load, masked_store, access_ptr, address_of) = {
        let state = visitor.state();
        (
            state.masked_load.clone(),
            state.masked_store.clone(),
            state.access_ptr.clone(),
            state.address_of.clone(),
        )
    };
    if operator == masked_load || operator == masked_store {
        let is_load = operator == masked_load;
        let buffer = BufferVar::try_from(value.args.get(0)?)?;
        let dtype = if is_load {
            value.ty.clone().try_cast::<PrimType>()?
        } else {
            value.args.get(1)?.try_cast::<PrimExpr>()?.type_annotation()
        };
        let begin = if is_load { 1 } else { 2 };
        let indices = Array::new(
            value
                .args
                .iter()
                .skip(begin)
                .take(value.args.len().saturating_sub(begin + 1))
                .map(PrimExpr::try_from)
                .collect::<Result<Vec<_>>>()?,
        );
        visitor
            .state_mut()
            .record_access(dtype, buffer.as_var(), &indices, is_load)?;
    } else if operator == access_ptr && value.args.len() >= 3 {
        let dtype = value.args.get(0)?.try_cast::<PrimExpr>()?.type_annotation();
        if let Some(buffer) =
            get_buffer_data_var(&value.args.get(1)?, &visitor.state().buffer_data)?
        {
            let index = value.args.get(2)?.try_cast::<PrimExpr>()?;
            visitor
                .state_mut()
                .record_access(dtype, &buffer, &Array::new(vec![index]), false)?;
        }
    } else if operator == address_of && !value.args.is_empty() {
        if let Ok(load) = value.args.get(0)?.try_cast::<TensorLoad>() {
            let buffer = BufferVar::try_from(&load.source)?;
            visitor.state_mut().record_access(
                load.ty.clone().try_cast()?,
                buffer.as_var(),
                &load.indices,
                false,
            )?;
        }
    }
    if value.op.clone().try_cast::<crate::ir::OpaqueExpr>().is_ok() {
        visitor.visit(&value.op)?;
    }
    visit_values(visitor, value.args.iter())
}

fn check_allocation(
    value: AllocBuffer,
    visitor: &mut VisitContext<'_, AccessChecker>,
) -> Result<()> {
    let variable = value.buffer.as_var().clone();
    visitor
        .state_mut()
        .aliases
        .insert(ObjectIdentity::of(&variable), variable.clone());
    let ty = value.buffer.type_annotation();
    let extent = ty
        .shape
        .iter()
        .last()
        .unwrap_or(IntImm::new("int32", 0)?.into());
    visitor
        .state_mut()
        .declare(variable, ty.dtype.clone(), extent, ALLOC_BUFFER_NODE)?;
    visit_buffer_definition(visitor, &value.buffer)
}

fn check_declaration(
    value: DeclBuffer,
    visitor: &mut VisitContext<'_, AccessChecker>,
) -> Result<()> {
    visitor
        .state_mut()
        .register_alias(&value.buffer, &value.data)?;
    let ty = value.buffer.type_annotation();
    let extent = ty
        .shape
        .iter()
        .last()
        .unwrap_or(IntImm::new("int32", 0)?.into());
    visitor.state_mut().declare(
        value.buffer.as_var().clone(),
        ty.dtype.clone(),
        extent,
        DECL_BUFFER_NODE,
    )?;
    visitor.visit(&value.data)?;
    visit_buffer_definition(visitor, &value.buffer)
}

fn check_let(value: Let, visitor: &mut VisitContext<'_, AccessChecker>) -> Result<()> {
    visitor
        .state_mut()
        .declare_pointer_binding(value.var.clone())?;
    visitor.visit(&value.value)?;
    visitor.visit(&value.body)?;
    Ok(())
}

fn check_binding(value: Bind, visitor: &mut VisitContext<'_, AccessChecker>) -> Result<()> {
    visitor
        .state_mut()
        .declare_pointer_binding(value.var.clone())?;
    visitor.visit(&value.value)?;
    Ok(())
}

fn check_default(
    value: &VisitValue,
    visitor: &mut VisitContext<'_, AccessChecker>,
) -> Result<Option<VisitInterrupt>> {
    visit_stmt_expr_default(visitor, value)
}

fn visit_buffer_definition(
    visitor: &mut VisitContext<'_, AccessChecker>,
    buffer: &BufferVar,
) -> Result<()> {
    let ty = buffer.type_annotation();
    for expression in ty.shape.iter().chain(ty.strides.iter()) {
        visitor.visit(&expression)?;
    }
    visitor.visit(&ty.elem_offset)?;
    for expression in ty.allocated_addr.iter() {
        visitor.visit(&expression)?;
    }
    if let Some(layout) = &ty.layout {
        if let Ok(tile) = layout.clone().try_cast::<crate::tirx::TileLayout>() {
            for iteration in tile.shard()?.iter().chain(tile.replica()?.iter()) {
                visitor.visit(&iteration.extent)?;
                visitor.visit(&iteration.stride)?;
            }
        }
    }
    Ok(())
}

fn visit_values<I>(visitor: &mut VisitContext<'_, AccessChecker>, values: I) -> Result<()>
where
    I: IntoIterator<Item = Expr>,
{
    for value in values {
        visitor.visit(&value)?;
    }
    Ok(())
}

#[derive(Clone)]
struct RewriteInfo {
    new_variable: Var,
    old_element_dtype: PrimType,
    new_element_dtype: PrimType,
}

impl RewriteInfo {
    fn factor(&self) -> Result<u16> {
        let old_lanes = self.old_element_dtype.dtype.lanes;
        let new_lanes = self.new_element_dtype.dtype.lanes;
        if old_lanes == 0 || !new_lanes.is_multiple_of(old_lanes) {
            return Err(value_error(
                "rewritten vector lanes are not an integral multiple",
            ));
        }
        Ok(new_lanes / old_lanes)
    }
}

struct VectorTypeRewriter {
    rewrite_indices: bool,
    rewrites: HashMap<ObjectIdentity, RewriteInfo>,
    buffer_cache: HashMap<ObjectIdentity, BufferVar>,
    aliases: HashMap<ObjectIdentity, Var>,
    analyzer: Analyzer,
    masked_load: ObjectIdentity,
    masked_store: ObjectIdentity,
    access_ptr: ObjectIdentity,
    buffer_data: ObjectIdentity,
}

impl VectorTypeRewriter {
    fn new(
        infos: HashMap<ObjectIdentity, BufferInfo>,
        aliases: HashMap<ObjectIdentity, Var>,
        analyzer: Analyzer,
        options: RewriteOptions,
    ) -> Result<Self> {
        let mut rewrite_mask = 0;
        if options.rewrite_buffer_params {
            rewrite_mask |= PRIM_FUNC_BUFFER_PARAM;
        }
        if options.rewrite_pointer_params {
            rewrite_mask |= PRIM_FUNC_POINTER_PARAM;
        }
        if options.rewrite_alloc_buffers {
            rewrite_mask |= ALLOC_BUFFER_NODE;
        }
        if options.rewrite_let_bindings {
            rewrite_mask |= LET_NODE;
        }

        let mut rewrites = HashMap::new();
        for (identity, info) in infos {
            let preferred = info.preferred_dtype(&analyzer)?;
            if preferred.dtype == info.element_dtype.dtype
                || rewrite_mask & info.declaration_location == 0
            {
                continue;
            }
            let new_variable = if let Ok(buffer) = BufferVar::try_from(&info.variable) {
                let old_type = buffer.type_annotation();
                let factor = preferred.dtype.lanes / info.element_dtype.dtype.lanes;
                let mut shape = old_type.shape.iter().collect::<Vec<_>>();
                if let Some(last) = shape.last_mut() {
                    *last = divide_by_factor(last.clone(), factor)?;
                }
                rebuild_buffer(
                    &buffer,
                    BufferType::from_complete_fields(
                        old_type.span.clone(),
                        preferred.clone(),
                        old_type.storage_scope.clone(),
                        Array::new(shape),
                        old_type.strides.clone(),
                        old_type.elem_offset.clone(),
                        old_type.data_alignment,
                        old_type.offset_factor,
                        None,
                        old_type.allocated_addr.clone(),
                    ),
                )?
                .as_var()
                .clone()
            } else {
                let pointer = info.variable.ty.clone().try_cast::<PointerType>()?;
                Var::from_complete_fields(
                    info.variable.span.clone(),
                    PointerType::new(preferred.clone(), pointer.storage_scope()?.as_str())?.into(),
                    info.variable.name.clone(),
                )
            };
            rewrites.insert(
                identity,
                RewriteInfo {
                    new_variable,
                    old_element_dtype: info.element_dtype,
                    new_element_dtype: preferred,
                },
            );
        }

        Ok(Self {
            rewrite_indices: options.rewrite_indices,
            rewrites,
            buffer_cache: HashMap::new(),
            aliases,
            analyzer,
            masked_load: operator_identity("tirx.masked_load")?,
            masked_store: operator_identity("tirx.masked_store")?,
            access_ptr: operator_identity("tirx.tvm_access_ptr")?,
            buffer_data: operator_identity("tirx.buffer_data")?,
        })
    }

    fn variable_substitutions(&self) -> HashMap<ObjectIdentity, Var> {
        self.rewrites
            .iter()
            .map(|(identity, info)| (identity.clone(), info.new_variable.clone()))
            .collect()
    }

    fn rewrite_access(
        &mut self,
        buffer: BufferVar,
        indices: Array<PrimExpr>,
    ) -> Result<(BufferVar, Array<PrimExpr>, Option<i64>)> {
        if !self.rewrite_indices {
            return Ok((buffer, indices, None));
        }
        let root = self
            .aliases
            .get(&ObjectIdentity::of(buffer.as_var()))
            .cloned()
            .unwrap_or_else(|| buffer.as_var().clone());
        let Some(info) = self.rewrites.get(&ObjectIdentity::of(&root)).cloned() else {
            return Ok((buffer, indices, None));
        };
        let Some(last_index) = indices.iter().last() else {
            return Err(value_error(
                "rewritten buffer access requires at least one index",
            ));
        };
        if is_scalable_vector(&buffer.type_annotation().dtype)
            || is_scalable_vector(&last_index.type_annotation())
        {
            return Ok((buffer, indices, None));
        }

        let factor = info.factor()?;
        let mut rewritten = indices.iter().collect::<Vec<_>>();
        let mut shuffle_index = None;
        if let Ok(ramp) = last_index.clone().try_cast::<Ramp>() {
            if int_value(&ramp.stride) == Some(1) {
                if let Some(lanes) = int_value(&ramp.lanes) {
                    let lanes = u16::try_from(lanes)
                        .map_err(|_| value_error("ramp lane count exceeds u16"))?;
                    let mut new_index = divide_by_factor(ramp.base.clone(), lanes)?;
                    if lanes != factor {
                        if factor == 0 || lanes % factor != 0 {
                            return Err(value_error(
                                "ramp lane count is incompatible with rewritten buffer type",
                            ));
                        }
                        let new_lanes = lanes / factor;
                        new_index = make_ramp(
                            binary_op(
                                "tirx._OpMul",
                                new_index,
                                IntImm::from_dtype(
                                    ramp.base.type_annotation().dtype,
                                    i64::from(new_lanes),
                                )?
                                .into(),
                            )?,
                            ramp.stride.clone(),
                            IntImm::new("int32", i64::from(new_lanes))?.into(),
                            ramp.span.as_ref(),
                        )?;
                    }
                    *rewritten.last_mut().unwrap() = new_index;
                }
            }
        } else if last_index.type_annotation().dtype.lanes == 1 && factor > 1 {
            let modular = self.analyzer.modular_set(&last_index)?;
            if modular.coeff != 0 && i64::from(factor) % modular.coeff != 0 {
                return Err(value_error(
                    "scalar access alignment is incompatible with rewritten buffer type",
                ));
            }
            *rewritten.last_mut().unwrap() = divide_by_factor(last_index, factor)?;
            shuffle_index = Some(modular.base_value % i64::from(factor));
        }
        Ok((
            self.remap_buffer(buffer)?,
            Array::new(rewritten),
            shuffle_index,
        ))
    }

    fn remap_buffer(&mut self, buffer: BufferVar) -> Result<BufferVar> {
        let cache_key = ObjectIdentity::of(buffer.as_var());
        if let Some(mapped) = self.buffer_cache.get(&cache_key) {
            return Ok(mapped.clone());
        }
        let root = self
            .aliases
            .get(&cache_key)
            .cloned()
            .unwrap_or_else(|| buffer.as_var().clone());
        let mut mapped = buffer.clone();
        if let Some(info) = self.rewrites.get(&ObjectIdentity::of(&root)).cloned() {
            if root.same_as(buffer.as_var()) {
                mapped = BufferVar::try_from(info.new_variable)?;
            } else {
                let old_type = buffer.type_annotation();
                let mut shape = old_type.shape.iter().collect::<Vec<_>>();
                if let Some(last) = shape.last_mut() {
                    *last = divide_by_factor(last.clone(), info.factor()?)?;
                }
                mapped = rebuild_buffer(
                    &buffer,
                    BufferType::from_complete_fields(
                        old_type.span.clone(),
                        info.new_element_dtype,
                        old_type.storage_scope.clone(),
                        Array::new(shape),
                        old_type.strides.clone(),
                        old_type.elem_offset.clone(),
                        old_type.data_alignment,
                        old_type.offset_factor,
                        None,
                        old_type.allocated_addr.clone(),
                    ),
                )?;
            }
        }
        self.buffer_cache.insert(cache_key, mapped.clone());
        Ok(mapped)
    }

    fn rewrite_masked_call(&mut self, value: Call, mutator: &mut Mutator) -> Result<Expr> {
        let is_load = ObjectIdentity::of(&value.op) == self.masked_load;
        let buffer = BufferVar::try_from(value.args.get(0)?)?;
        if is_load {
            let indices = value
                .args
                .iter()
                .skip(1)
                .take(value.args.len().saturating_sub(2))
                .map(|index| mutator.mutate(self, &index).and_then(PrimExpr::try_from))
                .collect::<Result<Vec<_>>>()?;
            let predicate: Expr = mutator
                .mutate(self, &value.args.get(value.args.len() - 1)?)?
                .try_into()?;
            let load = TensorLoad::from_buffer_with_span(
                buffer.as_var().clone(),
                indices.iter().cloned().map(Into::into).collect(),
                value.span.as_ref(),
            )?;
            let (buffer, indices, shuffle) = self.rewrite_access(buffer, Array::new(indices))?;
            if shuffle.is_some() {
                return Err(value_error(
                    "a masked vector load cannot be rewritten into a scalar shuffle",
                ));
            }
            let mut arguments = vec![buffer.as_var().clone().into()];
            arguments.extend(indices.iter().map(Into::into));
            arguments.push(predicate);
            return Ok(Call::from_complete_fields(
                value.span.clone(),
                load.ty.clone(),
                value.op.clone(),
                Array::new(arguments),
                value.attrs.clone(),
                value.ty_args.clone(),
            )
            .into());
        }

        let stored_value: PrimExpr = mutator.mutate(self, &value.args.get(1)?)?.try_into()?;
        let indices = value
            .args
            .iter()
            .skip(2)
            .take(value.args.len().saturating_sub(3))
            .map(|index| mutator.mutate(self, &index).and_then(PrimExpr::try_from))
            .collect::<Result<Vec<_>>>()?;
        let predicate: Expr = mutator
            .mutate(self, &value.args.get(value.args.len() - 1)?)?
            .try_into()?;
        let (buffer, indices, shuffle) = self.rewrite_access(buffer, Array::new(indices))?;
        if shuffle.is_some() {
            return Err(value_error(
                "a masked vector store cannot be rewritten into a scalar shuffle",
            ));
        }
        let mut arguments = vec![buffer.as_var().clone().into(), stored_value.into()];
        arguments.extend(indices.iter().map(Into::into));
        arguments.push(predicate);
        Ok(Call::from_complete_fields(
            value.span.clone(),
            PrimType::void().into(),
            value.op.clone(),
            Array::new(arguments),
            value.attrs.clone(),
            value.ty_args.clone(),
        )
        .into())
    }
}

struct PointerVarSubstituter {
    variables: HashMap<ObjectIdentity, Var>,
    buffer_data: ObjectIdentity,
}

impl PointerVarSubstituter {
    fn new(variables: HashMap<ObjectIdentity, Var>) -> Self {
        Self {
            variables,
            buffer_data: operator_identity("tirx.buffer_data")
                .expect("tirx.buffer_data must be registered"),
        }
    }

    fn variable(&self, value: &Var) -> Var {
        self.variables
            .get(&ObjectIdentity::of(value))
            .cloned()
            .unwrap_or_else(|| value.clone())
    }

    fn buffer(&self, value: &BufferVar) -> Result<BufferVar> {
        let mapped = self.variable(value.as_var());
        if mapped.same_as(value.as_var()) {
            Ok(value.clone())
        } else {
            BufferVar::try_from(mapped)
        }
    }
}

#[tvm_ffi::dispatch(mutate)]
impl PointerVarSubstituter {
    fn mutate_variable(&mut self, value: Var) -> Expr {
        self.variable(&value).into()
    }

    fn mutate_load(&mut self, value: TensorLoad, mutator: &mut Mutator) -> Result<PrimExpr> {
        let old_buffer = BufferVar::try_from(&value.source)?;
        let buffer = self.buffer(&old_buffer)?;
        let indices: Array<PrimExpr> = mutator.mutate(self, &value.indices)?.try_into()?;
        if buffer.same_as(&old_buffer) && array_same_as(&indices, &value.indices) {
            return Ok(value.into());
        }
        TensorLoad::from_buffer_with_span(
            buffer.as_var().clone(),
            indices.iter().map(Into::into).collect(),
            value.span.as_ref(),
        )
        .map(Into::into)
    }

    fn mutate_store(&mut self, value: BufferStore, mutator: &mut Mutator) -> Result<Stmt> {
        let buffer = self.buffer(&value.buffer)?;
        let stored_value: PrimExpr = mutator.mutate(self, &value.value)?.try_into()?;
        let indices: Array<PrimExpr> = mutator.mutate(self, &value.indices)?.try_into()?;
        if buffer.same_as(&value.buffer)
            && stored_value.same_as(&value.value)
            && array_same_as(&indices, &value.indices)
        {
            return Ok(value.into());
        }
        Ok(
            BufferStore::from_complete_fields(value.span.clone(), buffer, stored_value, indices)
                .into(),
        )
    }

    fn mutate_call(&mut self, value: Call, mutator: &mut Mutator) -> Result<Expr> {
        let mapped = mutate_expr_default(self, mutator, value.into())?;
        let call = mapped.clone().try_cast::<Call>()?;
        if ObjectIdentity::of(&call.op) == self.buffer_data && call.args.len() == 1 {
            if let Ok(variable) = call.args.get(0)?.try_cast::<Var>() {
                if let Ok(buffer) = BufferVar::try_from(variable) {
                    return buffer_data(&buffer);
                }
            }
        }
        Ok(mapped)
    }

    fn mutate_allocation(&mut self, value: AllocBuffer) -> Result<Stmt> {
        let buffer = self.buffer(&value.buffer)?;
        if buffer.same_as(&value.buffer) {
            return Ok(value.into());
        }
        Ok(
            AllocBuffer::from_complete_fields(
                value.span.clone(),
                buffer,
                value.annotations.clone(),
            )
            .into(),
        )
    }

    fn mutate_declaration(&mut self, value: DeclBuffer, mutator: &mut Mutator) -> Result<Stmt> {
        let data: Expr = mutator.mutate(self, &value.data)?.try_into()?;
        let buffer = self.buffer(&value.buffer)?;
        if data.same_as(&value.data) && buffer.same_as(&value.buffer) {
            return Ok(value.into());
        }
        Ok(DeclBuffer::from_complete_fields(value.span.clone(), buffer, data).into())
    }

    fn mutate_attribute(&mut self, value: AttrStmt, mutator: &mut Mutator) -> Result<Stmt> {
        let mapped = mutate_stmt_default(self, mutator, value.clone().into())?;
        let attribute = mapped.clone().try_cast::<AttrStmt>()?;
        let Some(variable) = attribute.node.try_as::<Var>() else {
            return Ok(mapped);
        };
        let variable = self.variable(&variable);
        if attribute
            .node
            .try_as::<Var>()
            .is_some_and(|old| variable.same_as(&old))
        {
            return Ok(mapped);
        }
        Ok(AttrStmt::from_complete_fields(
            attribute.span.clone(),
            Any::from(variable),
            attribute.attr_key.clone(),
            attribute.value.clone(),
            attribute.body.clone(),
        )
        .into())
    }

    fn mutate_default(&mut self, value: &MapValue, mutator: &mut Mutator) -> Result<Any> {
        super::utils::mutate_stmt_expr_default(self, mutator, value)
    }
}

fn pointer_element_type(variable: &Var) -> Result<Option<PrimType>> {
    let Ok(pointer) = variable.ty.clone().try_cast::<PointerType>() else {
        return Ok(None);
    };
    Ok(pointer.element_type()?.try_cast::<PrimType>().ok())
}

fn is_void(value: &PrimType) -> bool {
    value.dtype.code == DLDataTypeCode::kDLOpaqueHandle as u8
        && value.dtype.bits == 0
        && value.dtype.lanes == 0
}

fn is_scalable_vector(value: &PrimType) -> bool {
    (value.dtype.lanes as i16) < 0
}

fn with_dtype_lanes(mut dtype: DLDataType, lanes: u16) -> DLDataType {
    dtype.lanes = lanes;
    dtype
}

fn insert_unique_dtype(values: &mut Vec<DLDataType>, value: DLDataType) {
    if !values.contains(&value) {
        values.push(value);
    }
}

fn int_value(value: &PrimExpr) -> Option<i64> {
    value
        .clone()
        .try_cast::<IntImm>()
        .ok()
        .map(|literal| literal.value)
}

fn divide_by_factor(value: PrimExpr, factor: u16) -> Result<PrimExpr> {
    if factor == 1 {
        return Ok(value);
    }
    let divisor = IntImm::from_dtype(value.type_annotation().dtype, i64::from(factor))?;
    binary_op("tirx._OpDiv", value, divisor.into())
}

fn make_ramp(
    base: PrimExpr,
    stride: PrimExpr,
    lanes: PrimExpr,
    span: Option<&crate::ir::Span>,
) -> Result<PrimExpr> {
    tvm_ffi::cached_global_func!("tirx.Ramp")
        .call_tuple((base, stride, lanes, span.cloned()))?
        .try_into()
}

fn extract_element(
    vector: PrimExpr,
    index: i64,
    span: Option<&crate::ir::Span>,
) -> Result<PrimExpr> {
    tvm_ffi::cached_global_func!("tirx.Shuffle")
        .call_tuple((
            Array::new(vec![vector]),
            Array::<PrimExpr>::new(vec![IntImm::new("int32", index)?.into()]),
            span.cloned(),
        ))?
        .try_into()
}

fn type_annotation(dtype: &PrimType) -> Result<Expr> {
    Ok(Call::new(
        dtype.clone(),
        get_operator("tirx.type_annotation")?,
        Vec::new(),
    )
    .into())
}

fn buffer_data(buffer: &BufferVar) -> Result<Expr> {
    let ty = buffer.type_annotation();
    let pointer = PointerType::new(ty.dtype.clone(), ty.storage_scope.as_str())?;
    Ok(Call::new(
        pointer,
        get_operator("tirx.buffer_data")?,
        vec![buffer.as_var().clone().into()],
    )
    .into())
}

fn get_buffer_data_var(value: &Expr, buffer_data: &ObjectIdentity) -> Result<Option<Var>> {
    if let Ok(variable) = value.clone().try_cast::<Var>() {
        return Ok(Some(variable));
    }
    if let Ok(call) = value.clone().try_cast::<Call>() {
        if ObjectIdentity::of(&call.op) == *buffer_data && call.args.len() == 1 {
            return Ok(call.args.get(0)?.try_cast::<Var>().ok());
        }
    }
    Ok(None)
}

fn rebuild_buffer(buffer: &BufferVar, ty: BufferType) -> Result<BufferVar> {
    BufferVar::try_from(Var::from_complete_fields(
        buffer.span.clone(),
        ty.into(),
        buffer.name.clone(),
    ))
}

fn operator_identity(name: &str) -> Result<ObjectIdentity> {
    Ok(ObjectIdentity::of(&get_operator(name)?))
}

fn get_operator(name: &str) -> Result<Expr> {
    tvm_ffi::cached_global_func!("ir.GetOp")
        .call_tuple((tvm_ffi::String::from(name),))?
        .try_into()
}

fn binary_op(name: &str, lhs: PrimExpr, rhs: PrimExpr) -> Result<PrimExpr> {
    tvm_ffi::Function::get_global(name)?
        .call_tuple((lhs, rhs, Option::<crate::ir::Span>::None))?
        .try_into()
}

fn value_error(message: &str) -> tvm_ffi::Error {
    tvm_ffi::Error::new(tvm_ffi::VALUE_ERROR, message, "")
}

#[tvm_ffi::dispatch(mutate)]
impl VectorTypeRewriter {
    fn mutate_load(&mut self, value: TensorLoad, mutator: &mut Mutator) -> Result<PrimExpr> {
        let buffer = BufferVar::try_from(&value.source)?;
        let indices: Array<PrimExpr> = mutator.mutate(self, &value.indices)?.try_into()?;
        let (buffer, indices, shuffle_index) = self.rewrite_access(buffer, indices)?;
        if buffer.same_as(&BufferVar::try_from(&value.source)?)
            && array_same_as(&indices, &value.indices)
        {
            return Ok(value.into());
        }
        let load: PrimExpr = TensorLoad::from_buffer_with_span(
            buffer.as_var().clone(),
            indices.iter().map(Into::into).collect(),
            value.span.as_ref(),
        )?
        .into();
        if let Some(index) = shuffle_index {
            return extract_element(load, index, value.span.as_ref());
        }
        Ok(load)
    }

    fn mutate_store(&mut self, value: BufferStore, mutator: &mut Mutator) -> Result<Stmt> {
        let stored_value: PrimExpr = mutator.mutate(self, &value.value)?.try_into()?;
        let indices: Array<PrimExpr> = mutator.mutate(self, &value.indices)?.try_into()?;
        let (buffer, indices, shuffle_index) =
            self.rewrite_access(value.buffer.clone(), indices)?;
        if shuffle_index.is_some() {
            return Err(value_error("a buffer store cannot use a scalar shuffle"));
        }
        if buffer.same_as(&value.buffer)
            && stored_value.same_as(&value.value)
            && array_same_as(&indices, &value.indices)
        {
            return Ok(value.into());
        }
        Ok(
            BufferStore::from_complete_fields(value.span.clone(), buffer, stored_value, indices)
                .into(),
        )
    }

    fn mutate_call(&mut self, value: Call, mutator: &mut Mutator) -> Result<Expr> {
        let operator = ObjectIdentity::of(&value.op);
        if operator == self.masked_load || operator == self.masked_store {
            return self.rewrite_masked_call(value, mutator);
        }
        if operator == self.buffer_data && value.args.len() == 1 {
            if let Ok(variable) = value.args.get(0)?.try_cast::<Var>() {
                if let Ok(buffer) = BufferVar::try_from(variable) {
                    return buffer_data(&self.remap_buffer(buffer)?);
                }
            }
        }
        if operator != self.access_ptr {
            return mutate_expr_default(self, mutator, value.into());
        }

        let mapped = mutate_expr_default(self, mutator, value.into())?;
        let call = mapped.clone().try_cast::<Call>()?;
        if !self.rewrite_indices || call.args.len() < 5 {
            return Ok(mapped);
        }
        let Some(buffer) = get_buffer_data_var(&call.args.get(1)?, &self.buffer_data)? else {
            return Ok(mapped);
        };
        let root = self
            .aliases
            .get(&ObjectIdentity::of(&buffer))
            .cloned()
            .unwrap_or(buffer);
        let Some(info) = self.rewrites.get(&ObjectIdentity::of(&root)).cloned() else {
            return Ok(mapped);
        };
        let factor = info.factor()?;
        let index = divide_by_factor(call.args.get(2)?.try_cast::<PrimExpr>()?, factor)?;
        let extent = divide_by_factor(call.args.get(3)?.try_cast::<PrimExpr>()?, factor)?;
        let data = if BufferVar::try_from(&info.new_variable).is_ok() {
            buffer_data(&BufferVar::try_from(info.new_variable.clone())?)?
        } else {
            info.new_variable.clone().into()
        };
        let old_pointer = call.ty.clone().try_cast::<PointerType>()?;
        let new_pointer = PointerType::new(
            info.new_element_dtype.clone(),
            old_pointer.storage_scope()?.as_str(),
        )?;
        Ok(Call::from_complete_fields(
            call.span.clone(),
            new_pointer.into(),
            call.op.clone(),
            Array::new(vec![
                type_annotation(&info.new_element_dtype)?,
                data,
                index.into(),
                extent.into(),
                call.args.get(4)?,
            ]),
            call.attrs.clone(),
            call.ty_args.clone(),
        )
        .into())
    }

    fn mutate_binding(&mut self, value: Bind, mutator: &mut Mutator) -> Result<Stmt> {
        let mut mapped_value: Expr = mutator.mutate(self, &value.value)?.try_into()?;
        let variable = self
            .rewrites
            .get(&ObjectIdentity::of(&value.var))
            .map(|info| info.new_variable.clone())
            .unwrap_or_else(|| value.var.clone());
        if !variable.same_as(&value.var) {
            let call = mapped_value.clone().try_cast::<Call>()?;
            mapped_value = Call::from_complete_fields(
                call.span.clone(),
                variable.ty.clone(),
                call.op.clone(),
                call.args.clone(),
                call.attrs.clone(),
                call.ty_args.clone(),
            )
            .into();
        }
        if variable.same_as(&value.var) && mapped_value.same_as(&value.value) {
            return Ok(value.into());
        }
        Ok(Bind::from_complete_fields(value.span.clone(), variable, mapped_value).into())
    }

    fn mutate_allocation(&mut self, value: AllocBuffer) -> Result<Stmt> {
        let buffer = self.remap_buffer(value.buffer.clone())?;
        if buffer.same_as(&value.buffer) {
            return Ok(value.into());
        }
        Ok(
            AllocBuffer::from_complete_fields(
                value.span.clone(),
                buffer,
                value.annotations.clone(),
            )
            .into(),
        )
    }

    fn mutate_declaration(&mut self, value: DeclBuffer, mutator: &mut Mutator) -> Result<Stmt> {
        let data: Expr = mutator.mutate(self, &value.data)?.try_into()?;
        let buffer = self.remap_buffer(value.buffer.clone())?;
        if data.same_as(&value.data) && buffer.same_as(&value.buffer) {
            return Ok(value.into());
        }
        Ok(DeclBuffer::from_complete_fields(value.span.clone(), buffer, data).into())
    }

    fn mutate_default(&mut self, value: &MapValue, mutator: &mut Mutator) -> Result<Any> {
        super::utils::mutate_stmt_expr_default(self, mutator, value)
    }
}
