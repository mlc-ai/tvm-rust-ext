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

use tvm_ffi::extra::structural_mutate::MutateContextDriver;
use tvm_ffi::{
    structural_mutate, structural_walk, Any, Array, Map, MapValue, MutateCallbacks, Mutator,
    ObjectIdentity, ObjectRefCast, ObjectRefCore, Result, WalkOrder, WalkResult,
};

use super::utils::{array_same_as, mutate_stmt_expr_default, option_same_as};
use super::{create_module_pass, Pass};
use crate::ir::{BaseFunc, DictAttrs, Expr, IRModule, PrimExpr, Range, TensorLoad, Type, Var};
use crate::tirx::{
    AllocBuffer, AttrStmt, Bind, BufferStore, BufferType, BufferVar, DeclBuffer, For, IfThenElse,
    Iter, IterVar, Layout, Let, PrimFunc, Stmt, TileLayout, While,
};

#[derive(Default)]
struct Scope {
    variables: Vec<ObjectIdentity>,
    buffers: Vec<ObjectIdentity>,
}

/// Convert repeated variable definitions to SSA form without requiring SBlock bindings.
pub fn convert_ssa_module(module: IRModule) -> Result<IRModule> {
    let mut state = SsaState::default();
    let mut changed = false;
    let mut functions = Vec::with_capacity(module.functions.len());
    for (global, function) in module.functions.iter() {
        let function = if let Ok(primitive) = function.clone().try_cast::<PrimFunc>() {
            let converted = convert_function(&mut state, primitive)?;
            changed |= !converted.same_as(&function);
            BaseFunc::from(converted)
        } else {
            function
        };
        functions.push((global, function));
    }
    if !changed {
        return Ok(module);
    }
    IRModule::with_metadata(
        Map::from_iter(functions),
        module.source_map.clone(),
        module.attrs.clone(),
        module.global_infos.clone(),
    )
}

/// Convert repeated definitions in a single PrimFunc to SSA form.
pub fn convert_ssa_prim_func(function: PrimFunc) -> Result<PrimFunc> {
    convert_function(&mut SsaState::default(), function)
}

/// Build TVM's `tirx.ConvertSSA` module pass in Rust.
pub fn convert_ssa() -> Result<Pass> {
    create_module_pass("tirx.ConvertSSA", 0, Vec::new(), false, convert_ssa_module)
}

/// Convert repeated definitions in an isolated statement.
pub(crate) fn convert_ssa_stmt(statement: Stmt) -> Result<Stmt> {
    let mut converter = MutateCallbacks::new(SsaState::default(), SsaConverter);
    converter.state_mut().enter_scope();
    let converted = structural_mutate(statement, &mut converter).and_then(Stmt::try_from);
    converter.state_mut().exit_scope();
    converted
}

#[derive(Default)]
struct SsaState {
    defined: HashSet<ObjectIdentity>,
    variable_remaps: HashMap<ObjectIdentity, Vec<Var>>,
    function_remaps: HashMap<ObjectIdentity, Var>,
    buffer_remaps: HashMap<ObjectIdentity, Vec<BufferVar>>,
    scopes: Vec<Scope>,
}

impl SsaState {
    fn enter_scope(&mut self) {
        self.scopes.push(Scope::default());
    }

    fn exit_scope(&mut self) {
        let scope = self.scopes.pop().expect("SSA scope stack is balanced");
        for identity in scope.buffers.into_iter().rev() {
            let remove = self.buffer_remaps.get_mut(&identity).is_some_and(|stack| {
                stack.pop();
                stack.is_empty()
            });
            if remove {
                self.buffer_remaps.remove(&identity);
            }
        }
        for identity in scope.variables.into_iter().rev() {
            let remove = self
                .variable_remaps
                .get_mut(&identity)
                .is_some_and(|stack| {
                    stack.pop();
                    stack.is_empty()
                });
            if remove {
                self.variable_remaps.remove(&identity);
            }
        }
    }

    fn current_variable(&self, variable: &Var) -> Var {
        let identity = ObjectIdentity::of(variable);
        self.variable_remaps
            .get(&identity)
            .and_then(|stack| stack.last())
            .or_else(|| self.function_remaps.get(&identity))
            .cloned()
            .unwrap_or_else(|| variable.clone())
    }

    fn define_local(&mut self, variable: &Var) -> Var {
        let identity = ObjectIdentity::of(variable);
        if self.defined.insert(identity) {
            variable.clone()
        } else {
            let replacement = fresh_variable(variable);
            self.push_variable_remap(variable, replacement.clone());
            replacement
        }
    }

    fn push_variable_remap(&mut self, old: &Var, replacement: Var) {
        let identity = ObjectIdentity::of(old);
        self.variable_remaps
            .entry(identity.clone())
            .or_default()
            .push(replacement);
        self.scopes
            .last_mut()
            .expect("variable remaps require an active scope")
            .variables
            .push(identity);
    }

    fn replace_current_remap(&mut self, old: &Var, replacement: Var) {
        let identity = ObjectIdentity::of(old);
        if let Some(current) = self
            .variable_remaps
            .get_mut(&identity)
            .and_then(|stack| stack.last_mut())
        {
            *current = replacement;
        } else if let Some(current) = self.function_remaps.get_mut(&identity) {
            *current = replacement;
        } else {
            self.push_variable_remap(old, replacement);
        }
    }

    fn finish_buffer_remap(
        &mut self,
        buffer: &BufferVar,
        shape: Array<PrimExpr>,
        strides: Array<PrimExpr>,
        elem_offset: PrimExpr,
        layout: Option<Layout>,
    ) -> Result<BufferVar> {
        let old_variable = buffer.as_var();
        let mapped_variable = self.current_variable(old_variable);
        let old_type = buffer.type_annotation();

        let identity = ObjectIdentity::of(old_variable);
        if let Some(candidate) = self
            .buffer_remaps
            .get(&identity)
            .and_then(|stack| stack.last())
        {
            let candidate_type = candidate.type_annotation();
            if candidate.as_var().same_as(&mapped_variable)
                && buffer_metadata_matches(&candidate_type, &shape, &strides, &elem_offset, &layout)
            {
                return Ok(candidate.clone());
            }
        }

        if let Ok(mapped_buffer) = BufferVar::try_from(&mapped_variable) {
            let mapped_type = mapped_buffer.type_annotation();
            if buffer_metadata_matches(&mapped_type, &shape, &strides, &elem_offset, &layout) {
                return Ok(mapped_buffer);
            }
        }

        let replacement_type = BufferType::from_complete_fields(
            old_type.span.clone(),
            old_type.dtype.clone(),
            old_type.storage_scope.clone(),
            shape,
            strides,
            elem_offset,
            old_type.data_alignment,
            old_type.offset_factor,
            layout,
            old_type.allocated_addr.clone(),
        );
        let replacement = Var::from_complete_fields(
            None,
            Type::from(replacement_type),
            mapped_variable.name.clone(),
        );
        let replacement = BufferVar::try_from(replacement)?;
        self.replace_current_remap(old_variable, replacement.as_var().clone());
        self.buffer_remaps
            .entry(identity.clone())
            .or_default()
            .push(replacement.clone());
        self.scopes
            .last_mut()
            .expect("buffer remaps require an active scope")
            .buffers
            .push(identity);
        Ok(replacement)
    }

    fn register_implicit_buffer_variables(&mut self, function: &PrimFunc) -> Result<()> {
        let explicit = function
            .params
            .iter()
            .map(|parameter| ObjectIdentity::of(&parameter))
            .collect::<HashSet<_>>();
        let mut matched = HashSet::new();
        for parameter in function.params.iter() {
            let Ok(buffer) = BufferVar::try_from(parameter) else {
                continue;
            };
            let buffer_type = buffer.type_annotation();
            let mut record = |variable: Var| {
                let identity = ObjectIdentity::of(&variable);
                if explicit.contains(&identity) || !matched.insert(identity.clone()) {
                    return;
                }
                if self.defined.insert(identity.clone()) {
                    return;
                }
                self.function_remaps
                    .entry(identity)
                    .or_insert_with(|| fresh_variable(&variable));
            };
            for expression in buffer_type.shape.iter() {
                structural_walk(
                    &expression,
                    |variable: Var| {
                        record(variable);
                        WalkResult::Advance
                    },
                    WalkOrder::PostOrder,
                )?;
            }
            for stride in buffer_type.strides.iter() {
                if let Ok(variable) = stride.try_cast::<Var>() {
                    record(variable);
                }
            }
            if let Ok(variable) = buffer_type.elem_offset.clone().try_cast::<Var>() {
                record(variable);
            }
        }
        Ok(())
    }
}

struct SsaConverter;

fn convert_function(state: &mut SsaState, function: PrimFunc) -> Result<PrimFunc> {
    let mut converter = MutateCallbacks::new(std::mem::take(state), SsaConverter);
    converter.state_mut().enter_scope();
    let converted = (|| -> Result<PrimFunc> {
        let mut params = Vec::with_capacity(function.params.len());
        for parameter in function.params.iter() {
            let identity = ObjectIdentity::of(&parameter);
            if converter.state_mut().defined.insert(identity.clone()) {
                params.push(parameter);
            } else {
                let replacement = fresh_variable(&parameter);
                converter
                    .state_mut()
                    .function_remaps
                    .insert(identity, replacement.clone());
                params.push(replacement);
            }
        }
        converter
            .state_mut()
            .register_implicit_buffer_variables(&function)?;

        let mut converted_params = Vec::with_capacity(params.len());
        for (original, parameter) in function.params.iter().zip(params) {
            if let Ok(buffer) = BufferVar::try_from(&original) {
                converted_params.push(remap_buffer_root(&mut converter, &buffer)?.into());
            } else {
                converted_params.push(parameter);
            }
        }
        let attrs = mutate_function_attrs(&mut converter, &function.attrs)?;
        let body: Stmt = structural_mutate(function.body.clone(), &mut converter)?.try_into()?;
        let params_changed = converted_params.len() != function.params.len()
            || converted_params
                .iter()
                .zip(function.params.iter())
                .any(|(converted, original)| !converted.same_as(&original));
        if !params_changed && attrs.same_as(&function.attrs) && body.same_as(&function.body) {
            return Ok(function);
        }
        PrimFunc::with_metadata(
            converted_params,
            body,
            function.ret_type.clone(),
            attrs,
            function.span.as_ref(),
        )
    })();
    converter.state_mut().exit_scope();
    converter.state_mut().function_remaps.clear();
    converter.state_mut().buffer_remaps.clear();
    *state = converter.into_state();
    converted
}

fn mutate_function_attrs<State, Link, Marker>(
    converter: &mut MutateCallbacks<State, Link, Marker>,
    attrs: &DictAttrs,
) -> Result<DictAttrs>
where
    MutateCallbacks<State, Link, Marker>: tvm_ffi::StructuralMutator,
{
    let mut changed = false;
    let mut converted = Vec::with_capacity(attrs.dict.len());
    for (key, original) in attrs.dict.iter() {
        let (value, value_changed) = if let Ok(expression) = PrimExpr::try_from(original.clone()) {
            let mapped: PrimExpr =
                structural_mutate(expression.clone(), &mut *converter)?.try_into()?;
            let changed = !mapped.same_as(&expression);
            (Any::from(mapped), changed)
        } else if let Ok(statement) = Stmt::try_from(original.clone()) {
            let mapped: Stmt = structural_mutate(statement.clone(), &mut *converter)?.try_into()?;
            let changed = !mapped.same_as(&statement);
            (Any::from(mapped), changed)
        } else {
            (original.clone(), false)
        };
        changed |= value_changed;
        converted.push((key, value));
    }
    if changed {
        Ok(DictAttrs::from_dictionary(Map::from_iter(converted)))
    } else {
        Ok(attrs.clone())
    }
}

fn remap_buffer<Driver>(
    mutator: &mut Mutator<SsaState, Driver>,
    buffer: &BufferVar,
) -> Result<BufferVar>
where
    Driver: MutateContextDriver<SsaState> + ?Sized,
{
    let old_type = buffer.type_annotation();
    let shape: Array<PrimExpr> = mutator.mutate(&old_type.shape)?.try_into()?;
    let strides: Array<PrimExpr> = mutator.mutate(&old_type.strides)?.try_into()?;
    let elem_offset: PrimExpr = mutator.mutate(&old_type.elem_offset)?.try_into()?;
    let layout = remap_tile_layout(&old_type.layout, |expression| {
        mutator.mutate(expression)?.try_into()
    })?;
    mutator
        .state_mut()
        .finish_buffer_remap(buffer, shape, strides, elem_offset, layout)
}

fn remap_buffer_root<Link, Marker>(
    converter: &mut MutateCallbacks<SsaState, Link, Marker>,
    buffer: &BufferVar,
) -> Result<BufferVar>
where
    MutateCallbacks<SsaState, Link, Marker>: tvm_ffi::StructuralMutator,
{
    let old_type = buffer.type_annotation();
    let shape: Array<PrimExpr> =
        structural_mutate(old_type.shape.clone(), &mut *converter)?.try_into()?;
    let strides: Array<PrimExpr> =
        structural_mutate(old_type.strides.clone(), &mut *converter)?.try_into()?;
    let elem_offset: PrimExpr =
        structural_mutate(old_type.elem_offset.clone(), &mut *converter)?.try_into()?;
    let layout = remap_tile_layout(&old_type.layout, |expression| {
        structural_mutate(expression.clone(), &mut *converter)?.try_into()
    })?;
    converter
        .state_mut()
        .finish_buffer_remap(buffer, shape, strides, elem_offset, layout)
}

fn remap_tile_layout(
    layout: &Option<Layout>,
    mut mutate: impl FnMut(&PrimExpr) -> Result<PrimExpr>,
) -> Result<Option<Layout>> {
    let Some(original) = layout else {
        return Ok(None);
    };
    let Ok(tile) = original.clone().try_cast::<TileLayout>() else {
        return Ok(Some(original.clone()));
    };
    let old_shard = tile.shard()?;
    let old_replica = tile.replica()?;
    let mut remap = |iter: Iter| -> Result<Iter> {
        let extent = mutate(&iter.extent)?;
        let stride = mutate(&iter.stride)?;
        if extent.same_as(&iter.extent) && stride.same_as(&iter.stride) {
            Ok(iter)
        } else {
            Ok(Iter::from_complete_fields(
                extent,
                stride,
                iter.axis.clone(),
            ))
        }
    };
    let shard = old_shard
        .iter()
        .map(&mut remap)
        .collect::<Result<Vec<_>>>()?;
    let replica = old_replica
        .iter()
        .map(&mut remap)
        .collect::<Result<Vec<_>>>()?;
    let shard = Array::new(shard);
    let replica = Array::new(replica);
    if array_same_as(&shard, &old_shard) && array_same_as(&replica, &old_replica) {
        return Ok(Some(original.clone()));
    }
    Ok(Some(
        TileLayout::new(
            shard.iter().collect(),
            replica.iter().collect(),
            tile.offset()?,
        )?
        .into(),
    ))
}

fn mutate_scoped_statement<Driver>(
    mutator: &mut Mutator<SsaState, Driver>,
    statement: &Stmt,
) -> Result<Stmt>
where
    Driver: MutateContextDriver<SsaState> + ?Sized,
{
    mutator.state_mut().enter_scope();
    let result = mutator.mutate(statement).and_then(Stmt::try_from);
    mutator.state_mut().exit_scope();
    result
}

#[tvm_ffi::dispatch(mutate)]
impl SsaConverter {
    fn mutate_variable(&self, value: Var, mutator: &mut Mutator<SsaState>) -> Var {
        mutator.state().current_variable(&value)
    }

    fn mutate_bind(&self, value: Bind, mutator: &mut Mutator<SsaState>) -> Result<Bind> {
        // The RHS sees the previous definition; the new definition starts afterwards.
        let bound_value: Expr = mutator.mutate(&value.value)?.try_into()?;
        let variable = mutator.state_mut().define_local(&value.var);
        if variable.same_as(&value.var) && bound_value.same_as(&value.value) {
            return Ok(value);
        }
        Ok(Bind::from_complete_fields(
            value.span.clone(),
            variable,
            bound_value,
        ))
    }

    fn mutate_let(&self, value: Let, mutator: &mut Mutator<SsaState>) -> Result<Let> {
        // The bound value sees the old definition.  Only the body sees this binder.
        let bound_value: PrimExpr = mutator.mutate(&value.value)?.try_into()?;
        mutator.state_mut().enter_scope();
        let variable = mutator.state_mut().define_local(&value.var);
        let body = mutator.mutate(&value.body).and_then(PrimExpr::try_from);
        mutator.state_mut().exit_scope();
        let body = body?;
        if variable.same_as(&value.var)
            && bound_value.same_as(&value.value)
            && body.same_as(&value.body)
        {
            return Ok(value);
        }
        Ok(Let::from_complete_fields(
            value.span.clone(),
            body.type_annotation(),
            variable,
            bound_value,
            body,
        ))
    }

    fn mutate_loop(&self, value: For, mutator: &mut Mutator<SsaState>) -> Result<For> {
        mutator.state_mut().enter_scope();
        let variable = mutator.state_mut().define_local(value.loop_var.as_var());
        let converted = (|| -> Result<For> {
            let minimum: PrimExpr = mutator.mutate(&value.min)?.try_into()?;
            let extent: PrimExpr = mutator.mutate(&value.extent)?.try_into()?;
            let step: Option<PrimExpr> = mutator.mutate(&value.step)?.try_into()?;
            let body: Stmt = mutator.mutate(&value.body)?.try_into()?;
            if variable.same_as(value.loop_var.as_var())
                && minimum.same_as(&value.min)
                && extent.same_as(&value.extent)
                && body.same_as(&value.body)
                && option_same_as(&step, &value.step)
            {
                return Ok(value);
            }
            Ok(For::from_complete_fields(
                value.span.clone(),
                variable.try_into()?,
                minimum,
                extent,
                value.kind,
                body,
                value.thread_binding.clone(),
                value.annotations.clone(),
                step,
            ))
        })();
        mutator.state_mut().exit_scope();
        converted
    }

    fn mutate_while(&self, value: While, mutator: &mut Mutator<SsaState>) -> Result<While> {
        mutator.state_mut().enter_scope();
        let converted = (|| -> Result<While> {
            let condition: PrimExpr = mutator.mutate(&value.condition)?.try_into()?;
            let body: Stmt = mutator.mutate(&value.body)?.try_into()?;
            if condition.same_as(&value.condition) && body.same_as(&value.body) {
                return Ok(value);
            }
            Ok(While::from_complete_fields(
                value.span.clone(),
                condition,
                body,
            ))
        })();
        mutator.state_mut().exit_scope();
        converted
    }

    fn mutate_conditional(
        &self,
        value: IfThenElse,
        mutator: &mut Mutator<SsaState>,
    ) -> Result<IfThenElse> {
        let condition: PrimExpr = mutator.mutate(&value.condition)?.try_into()?;
        let then_case = mutate_scoped_statement(mutator, &value.then_case)?;
        let else_case = value
            .else_case
            .as_ref()
            .map(|branch| mutate_scoped_statement(mutator, branch))
            .transpose()?;
        if condition.same_as(&value.condition)
            && then_case.same_as(&value.then_case)
            && option_same_as(&else_case, &value.else_case)
        {
            return Ok(value);
        }
        Ok(IfThenElse::from_complete_fields(
            value.span.clone(),
            condition,
            then_case,
            else_case,
        ))
    }

    fn mutate_attribute(
        &self,
        value: AttrStmt,
        mutator: &mut Mutator<SsaState>,
    ) -> Result<AttrStmt> {
        if let Ok(iteration) = IterVar::try_from(value.node.clone()) {
            return mutate_iter_var_attribute(mutator, value, iteration);
        }

        let (node, node_unchanged) = if let Ok(variable) = Var::try_from(value.node.clone()) {
            let mapped = mutator.state().current_variable(&variable);
            let unchanged = mapped.same_as(&variable);
            (Any::from(mapped), unchanged)
        } else {
            // Match C++ StmtMutator: an AttrStmt's node is metadata, not a
            // recursive expression child.
            (value.node.clone(), true)
        };
        let attr_value: PrimExpr = mutator.mutate(&value.value)?.try_into()?;
        let body = mutate_scoped_statement(mutator, &value.body)?;
        if node_unchanged && attr_value.same_as(&value.value) && body.same_as(&value.body) {
            return Ok(value);
        }
        Ok(AttrStmt::from_complete_fields(
            value.span.clone(),
            node,
            value.attr_key.clone(),
            attr_value,
            body,
        ))
    }

    fn mutate_decl_buffer(
        &self,
        value: DeclBuffer,
        mutator: &mut Mutator<SsaState>,
    ) -> Result<DeclBuffer> {
        mutator.state_mut().define_local(value.buffer.as_var());
        let buffer = remap_buffer(mutator, &value.buffer)?;
        let data: Expr = mutator.mutate(&value.data)?.try_into()?;
        if buffer.same_as(&value.buffer) && data.same_as(&value.data) {
            return Ok(value);
        }
        Ok(DeclBuffer::from_complete_fields(
            value.span.clone(),
            buffer,
            data,
        ))
    }

    fn mutate_alloc_buffer(
        &self,
        value: AllocBuffer,
        mutator: &mut Mutator<SsaState>,
    ) -> Result<AllocBuffer> {
        mutator.state_mut().define_local(value.buffer.as_var());
        let buffer = remap_buffer(mutator, &value.buffer)?;
        if buffer.same_as(&value.buffer) {
            return Ok(value);
        }
        Ok(AllocBuffer::from_complete_fields(
            value.span.clone(),
            buffer,
            value.annotations.clone(),
        ))
    }

    fn mutate_store(
        &self,
        value: BufferStore,
        mutator: &mut Mutator<SsaState>,
    ) -> Result<BufferStore> {
        let buffer = remap_buffer(mutator, &value.buffer)?;
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

    fn mutate_load(
        &self,
        value: TensorLoad,
        mutator: &mut Mutator<SsaState>,
    ) -> Result<TensorLoad> {
        let source: BufferVar = (&value.source).try_into()?;
        let buffer = remap_buffer(mutator, &source)?;
        let indices = mutator.mutate(&value.indices)?.try_into()?;
        if buffer.same_as(&source) && array_same_as(&indices, &value.indices) {
            return Ok(value);
        }
        Ok(TensorLoad::from_complete_fields(
            value.span.clone(),
            value.ty.clone().try_cast()?,
            Expr::from(buffer),
            indices,
        ))
    }

    fn mutate_stmt_expr_default(
        &self,
        value: &MapValue,
        mutator: &mut Mutator<SsaState>,
    ) -> Result<Any> {
        mutate_stmt_expr_default(mutator, value)
    }
}

fn mutate_iter_var_attribute<Driver>(
    mutator: &mut Mutator<SsaState, Driver>,
    value: AttrStmt,
    iteration: IterVar,
) -> Result<AttrStmt>
where
    Driver: MutateContextDriver<SsaState> + ?Sized,
{
    let original_iteration = iteration.clone();
    let original_domain = iteration.dom()?;
    let domain = original_domain
        .as_ref()
        .map(|domain| -> Result<Range> {
            let minimum: PrimExpr = mutator.mutate(&domain.min)?.try_into()?;
            let extent: PrimExpr = mutator.mutate(&domain.extent)?.try_into()?;
            if minimum.same_as(&domain.min) && extent.same_as(&domain.extent) {
                Ok(domain.clone())
            } else {
                Ok(Range::from_complete_fields(
                    minimum,
                    extent,
                    domain.span.clone(),
                ))
            }
        })
        .transpose()?;
    let original_variable = iteration.var()?.as_var().clone();
    let identity = ObjectIdentity::of(&original_variable);
    let mut delayed_definition = false;
    let variable = if let Some(mapped) = mutator.state().function_remaps.get(&identity) {
        mapped.clone()
    } else if mutator.state().defined.contains(&identity) {
        let replacement = fresh_variable(&original_variable);
        mutator
            .state_mut()
            .function_remaps
            .insert(identity.clone(), replacement.clone());
        replacement
    } else {
        delayed_definition = true;
        original_variable.clone()
    };
    let iteration =
        if option_same_as(&original_domain, &domain) && variable.same_as(&original_variable) {
            iteration
        } else {
            IterVar::with_metadata(
                domain,
                variable.clone(),
                iteration.iter_type()?,
                iteration.thread_tag()?.as_str(),
                iteration.span()?.as_ref(),
            )?
        };
    let attr_value: PrimExpr = mutator.mutate(&value.value)?.try_into()?;
    let body = mutate_scoped_statement(mutator, &value.body)?;

    if delayed_definition && !mutator.state().defined.contains(&identity) {
        mutator.state_mut().defined.insert(identity.clone());
        mutator
            .state_mut()
            .function_remaps
            .insert(identity, variable);
    }

    if iteration.same_as(&original_iteration)
        && attr_value.same_as(&value.value)
        && body.same_as(&value.body)
    {
        return Ok(value);
    }

    Ok(AttrStmt::from_complete_fields(
        value.span.clone(),
        Any::from(iteration),
        value.attr_key.clone(),
        attr_value,
        body,
    ))
}

fn fresh_variable(variable: &Var) -> Var {
    Var::from_complete_fields(None, variable.ty.clone(), variable.name.clone())
}

fn buffer_metadata_matches(
    buffer: &BufferType,
    shape: &Array<PrimExpr>,
    strides: &Array<PrimExpr>,
    elem_offset: &PrimExpr,
    layout: &Option<Layout>,
) -> bool {
    array_same_as(&buffer.shape, shape)
        && array_same_as(&buffer.strides, strides)
        && buffer.elem_offset.same_as(elem_offset)
        && option_same_as(&buffer.layout, layout)
}
