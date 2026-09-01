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
    structural_mutate, structural_walk, Any, Array, DefRegionKind, Map, ObjectIdentity,
    ObjectRefCast, ObjectRefCore, Result, StructuralMutator, WalkOrder, WalkResult,
};

use super::{create_module_pass, Pass};
use crate::ir::{BaseFunc, DictAttrs, Expr, IRModule, PrimExpr, Range, TensorLoad, Type, Var};
use crate::tirx::{
    AllocBuffer, AttrStmt, Bind, BufferStore, BufferType, BufferVar, DeclBuffer, For, IfThenElse,
    IterVar, PrimFunc, Stmt,
};

#[derive(Default)]
struct Scope {
    variables: Vec<ObjectIdentity>,
    buffers: Vec<ObjectIdentity>,
}

/// Convert repeated variable definitions to SSA form without requiring SBlock bindings.
pub fn convert_ssa_module(module: IRModule) -> Result<IRModule> {
    let mut converter = SsaConverter::default();
    let mut changed = false;
    let mut functions = Vec::with_capacity(module.functions.len());
    for (global, function) in module.functions.iter() {
        let function = if let Ok(primitive) = function.clone().try_cast::<PrimFunc>() {
            let converted = converter.convert_function(primitive)?;
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
    SsaConverter::default().convert_function(function)
}

/// Build TVM's `tirx.ConvertSSA` module pass in Rust.
pub fn convert_ssa() -> Result<Pass> {
    create_module_pass(
        "tirx.ConvertSSA",
        0,
        Vec::new(),
        false,
        |module, _context| convert_ssa_module(module),
    )
}

/// Convert repeated definitions in an isolated statement.
pub(crate) fn convert_ssa_stmt(statement: Stmt) -> Result<Stmt> {
    let mut converter = SsaConverter::default();
    converter.enter_scope();
    let converted = structural_mutate(statement, &mut converter).and_then(Stmt::try_from);
    converter.exit_scope();
    converted
}

#[derive(Default)]
struct SsaConverter {
    defined: HashSet<ObjectIdentity>,
    variable_remaps: HashMap<ObjectIdentity, Vec<Var>>,
    function_remaps: HashMap<ObjectIdentity, Var>,
    buffer_remaps: HashMap<ObjectIdentity, Vec<BufferVar>>,
    scopes: Vec<Scope>,
}

impl SsaConverter {
    fn convert_function(&mut self, function: PrimFunc) -> Result<PrimFunc> {
        self.enter_scope();
        let converted = (|| -> Result<PrimFunc> {
            let mut params = Vec::with_capacity(function.params.len());
            for parameter in function.params.iter() {
                let identity = ObjectIdentity::of(&parameter);
                if self.defined.insert(identity.clone()) {
                    params.push(parameter);
                } else {
                    let replacement = fresh_variable(&parameter);
                    self.function_remaps.insert(identity, replacement.clone());
                    params.push(replacement);
                }
            }
            self.register_implicit_buffer_variables(&function)?;

            let attrs: DictAttrs =
                structural_mutate(function.attrs.clone(), &mut *self)?.try_into()?;
            let body: Stmt = structural_mutate(function.body.clone(), &mut *self)?.try_into()?;
            let mut converted_params = Vec::with_capacity(params.len());
            for (original, parameter) in function.params.iter().zip(params) {
                if let Ok(buffer) = BufferVar::try_from(&original) {
                    let buffer_type = buffer.ty.clone().try_cast::<BufferType>()?;
                    let mapped_type: BufferType =
                        structural_mutate(buffer_type, &mut *self)?.try_into()?;
                    converted_params.push(
                        self.finish_buffer_remap(
                            &buffer,
                            mapped_type.shape.clone(),
                            mapped_type.strides.clone(),
                            mapped_type.elem_offset.clone(),
                            mapped_type.layout.clone(),
                            mapped_type.allocated_addr.clone(),
                        )?
                        .into(),
                    );
                } else {
                    converted_params.push(parameter);
                }
            }
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

        self.exit_scope();
        self.function_remaps.clear();
        self.buffer_remaps.clear();
        converted
    }

    fn enter_scope(&mut self) {
        self.scopes.push(Scope::default());
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
            let buffer_type = buffer.ty.clone().try_cast::<BufferType>()?;
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

    fn remap_buffer(&mut self, buffer: &BufferVar) -> Result<BufferVar> {
        let old_type = buffer.ty.clone().try_cast::<BufferType>()?;
        let shape: Array<PrimExpr> = self
            .mutate(&old_type.shape, DefRegionKind::None)?
            .try_into()?;
        let strides: Array<PrimExpr> = self
            .mutate(&old_type.strides, DefRegionKind::None)?
            .try_into()?;
        let elem_offset: PrimExpr = self
            .mutate(&old_type.elem_offset, DefRegionKind::None)?
            .try_into()?;
        let allocated_addr: Array<PrimExpr> = self
            .mutate(&old_type.allocated_addr, DefRegionKind::None)?
            .try_into()?;
        let layout = self
            .mutate(&old_type.layout, DefRegionKind::None)?
            .try_into()?;

        self.finish_buffer_remap(buffer, shape, strides, elem_offset, layout, allocated_addr)
    }

    fn finish_buffer_remap(
        &mut self,
        buffer: &BufferVar,
        shape: Array<PrimExpr>,
        strides: Array<PrimExpr>,
        elem_offset: PrimExpr,
        layout: Option<crate::tirx::Layout>,
        allocated_addr: Array<PrimExpr>,
    ) -> Result<BufferVar> {
        let old_variable = buffer.as_var();
        let mapped_variable = self.current_variable(old_variable);
        let old_type = buffer.ty.clone().try_cast::<BufferType>()?;

        let identity = ObjectIdentity::of(old_variable);
        if let Some(candidate) = self
            .buffer_remaps
            .get(&identity)
            .and_then(|stack| stack.last())
        {
            let candidate_type = candidate.ty.clone().try_cast::<BufferType>()?;
            if candidate.as_var().same_as(&mapped_variable)
                && buffer_metadata_matches(
                    &candidate_type,
                    &shape,
                    &strides,
                    &elem_offset,
                    &layout,
                    &allocated_addr,
                )
            {
                return Ok(candidate.clone());
            }
        }

        if let Ok(mapped_buffer) = BufferVar::try_from(&mapped_variable) {
            let mapped_type = mapped_buffer.ty.clone().try_cast::<BufferType>()?;
            if buffer_metadata_matches(
                &mapped_type,
                &shape,
                &strides,
                &elem_offset,
                &layout,
                &allocated_addr,
            ) {
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
            allocated_addr,
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

    fn mutate_scoped_statement(&mut self, statement: &Stmt) -> Result<Stmt> {
        self.enter_scope();
        let result = self
            .mutate(statement, DefRegionKind::None)
            .and_then(Stmt::try_from);
        self.exit_scope();
        result
    }
}

#[tvm_ffi::dispatch(mutate)]
impl SsaConverter {
    fn mutate_variable(&mut self, value: Var) -> Expr {
        self.current_variable(&value).into()
    }

    fn mutate_bind(&mut self, value: Bind) -> Result<Stmt> {
        // The RHS sees the previous definition; the new definition starts afterwards.
        let bound_value: Expr = self.mutate(&value.value, DefRegionKind::None)?.try_into()?;
        let variable = self.define_local(&value.var);
        if variable.same_as(&value.var) && bound_value.same_as(&value.value) {
            return Ok(value.into());
        }
        Ok(Bind::from_complete_fields(value.span.clone(), variable, bound_value).into())
    }

    fn mutate_loop(&mut self, value: For) -> Result<Stmt> {
        self.enter_scope();
        let variable = self.define_local(value.loop_var.as_var());
        let converted = (|| -> Result<Stmt> {
            let minimum: PrimExpr = self.mutate(&value.min, DefRegionKind::None)?.try_into()?;
            let extent: PrimExpr = self
                .mutate(&value.extent, DefRegionKind::None)?
                .try_into()?;
            let body: Stmt = self.mutate(&value.body, DefRegionKind::None)?.try_into()?;
            let thread_binding: Option<IterVar> = self
                .mutate(&value.thread_binding, DefRegionKind::None)?
                .try_into()?;
            let annotations: Map<tvm_ffi::String, Any> = self
                .mutate(&value.annotations, DefRegionKind::None)?
                .try_into()?;
            let step: Option<PrimExpr> =
                self.mutate(&value.step, DefRegionKind::None)?.try_into()?;
            if variable.same_as(value.loop_var.as_var())
                && minimum.same_as(&value.min)
                && extent.same_as(&value.extent)
                && body.same_as(&value.body)
                && option_same_as(&thread_binding, &value.thread_binding)
                && annotations.same_as(&value.annotations)
                && option_same_as(&step, &value.step)
            {
                return Ok(value.into());
            }
            Ok(For::from_complete_fields(
                value.span.clone(),
                variable.try_into()?,
                minimum,
                extent,
                value.kind,
                body,
                thread_binding,
                annotations,
                step,
            )
            .into())
        })();
        self.exit_scope();
        converted
    }

    fn mutate_conditional(&mut self, value: IfThenElse) -> Result<Stmt> {
        let condition: PrimExpr = self
            .mutate(&value.condition, DefRegionKind::None)?
            .try_into()?;
        let then_case = self.mutate_scoped_statement(&value.then_case)?;
        let else_case = value
            .else_case
            .as_ref()
            .map(|branch| self.mutate_scoped_statement(branch))
            .transpose()?;
        if condition.same_as(&value.condition)
            && then_case.same_as(&value.then_case)
            && option_same_as(&else_case, &value.else_case)
        {
            return Ok(value.into());
        }
        Ok(
            IfThenElse::from_complete_fields(value.span.clone(), condition, then_case, else_case)
                .into(),
        )
    }

    fn mutate_attribute(&mut self, value: AttrStmt) -> Result<Stmt> {
        if let Ok(iteration) = IterVar::try_from(value.node.clone()) {
            return self.mutate_iter_var_attribute(value, iteration);
        }

        let (node, node_unchanged) = if let Ok(variable) = Var::try_from(value.node.clone()) {
            let mapped = self.current_variable(&variable);
            let unchanged = mapped.same_as(&variable);
            (Any::from(mapped), unchanged)
        } else {
            (self.mutate(&value.node, DefRegionKind::None)?, false)
        };
        let attr_value: PrimExpr = self.mutate(&value.value, DefRegionKind::None)?.try_into()?;
        let body = self.mutate_scoped_statement(&value.body)?;
        if node_unchanged && attr_value.same_as(&value.value) && body.same_as(&value.body) {
            return Ok(value.into());
        }
        Ok(AttrStmt::from_complete_fields(
            value.span.clone(),
            node,
            value.attr_key.clone(),
            attr_value,
            body,
        )
        .into())
    }

    fn mutate_decl_buffer(&mut self, value: DeclBuffer) -> Result<Stmt> {
        self.define_local(value.buffer.as_var());
        let buffer = self.remap_buffer(&value.buffer)?;
        let data: Expr = self.mutate(&value.data, DefRegionKind::None)?.try_into()?;
        if buffer.same_as(&value.buffer) && data.same_as(&value.data) {
            return Ok(value.into());
        }
        Ok(DeclBuffer::from_complete_fields(value.span.clone(), buffer, data).into())
    }

    fn mutate_alloc_buffer(&mut self, value: AllocBuffer) -> Result<Stmt> {
        self.define_local(value.buffer.as_var());
        let buffer = self.remap_buffer(&value.buffer)?;
        let annotations: Map<tvm_ffi::String, Any> = self
            .mutate(&value.annotations, DefRegionKind::None)?
            .try_into()?;
        if buffer.same_as(&value.buffer) && annotations.same_as(&value.annotations) {
            return Ok(value.into());
        }
        Ok(AllocBuffer::from_complete_fields(value.span.clone(), buffer, annotations).into())
    }

    fn mutate_store(&mut self, value: BufferStore) -> Result<Stmt> {
        let buffer = self.remap_buffer(&value.buffer)?;
        let stored_value: PrimExpr = self.mutate(&value.value, DefRegionKind::None)?.try_into()?;
        let indices: Array<PrimExpr> = self
            .mutate(&value.indices, DefRegionKind::None)?
            .try_into()?;
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

    fn mutate_load(&mut self, value: TensorLoad) -> Result<Expr> {
        let source = value.source.clone().try_cast::<Var>()?;
        let buffer = self.remap_buffer(&BufferVar::try_from(&source)?)?;
        let indices = self
            .mutate(&value.indices, DefRegionKind::None)?
            .try_into()?;
        if buffer.as_var().same_as(&source) && array_same_as(&indices, &value.indices) {
            return Ok(value.into());
        }
        Ok(TensorLoad::from_complete_fields(
            value.span.clone(),
            value.ty.clone().try_cast()?,
            Expr::from(buffer),
            indices,
        )
        .into())
    }
}

impl SsaConverter {
    fn mutate_iter_var_attribute(&mut self, value: AttrStmt, iteration: IterVar) -> Result<Stmt> {
        let original_iteration = iteration.clone();
        let original_domain = iteration.dom()?;
        let domain = original_domain
            .as_ref()
            .map(|domain| -> Result<Range> {
                let minimum: PrimExpr =
                    self.mutate(&domain.min, DefRegionKind::None)?.try_into()?;
                let extent: PrimExpr = self
                    .mutate(&domain.extent, DefRegionKind::None)?
                    .try_into()?;
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
        let variable = if let Some(mapped) = self.function_remaps.get(&identity) {
            mapped.clone()
        } else if self.defined.contains(&identity) {
            let replacement = fresh_variable(&original_variable);
            self.function_remaps
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
        let attr_value: PrimExpr = self.mutate(&value.value, DefRegionKind::None)?.try_into()?;
        let body = self.mutate_scoped_statement(&value.body)?;

        if delayed_definition && !self.defined.contains(&identity) {
            self.defined.insert(identity.clone());
            self.function_remaps.insert(identity, variable);
        }

        if iteration.same_as(&original_iteration)
            && attr_value.same_as(&value.value)
            && body.same_as(&value.body)
        {
            return Ok(value.into());
        }

        Ok(AttrStmt::from_complete_fields(
            value.span.clone(),
            Any::from(iteration),
            value.attr_key.clone(),
            attr_value,
            body,
        )
        .into())
    }
}

fn fresh_variable(variable: &Var) -> Var {
    Var::from_complete_fields(None, variable.ty.clone(), variable.name.clone())
}

fn buffer_metadata_matches(
    buffer: &BufferType,
    shape: &Array<PrimExpr>,
    strides: &Array<PrimExpr>,
    elem_offset: &PrimExpr,
    layout: &Option<crate::tirx::Layout>,
    allocated_addr: &Array<PrimExpr>,
) -> bool {
    array_same_as(&buffer.shape, shape)
        && array_same_as(&buffer.strides, strides)
        && buffer.elem_offset.same_as(elem_offset)
        && option_same_as(&buffer.layout, layout)
        && array_same_as(&buffer.allocated_addr, allocated_addr)
}

fn array_same_as<T: ObjectRefCore + tvm_ffi::AnyCompatible + Clone>(
    lhs: &Array<T>,
    rhs: &Array<T>,
) -> bool {
    lhs.len() == rhs.len()
        && lhs
            .iter()
            .zip(rhs.iter())
            .all(|(lhs, rhs)| lhs.same_as(&rhs))
}

fn option_same_as<T: ObjectRefCore>(lhs: &Option<T>, rhs: &Option<T>) -> bool {
    match (lhs, rhs) {
        (Some(lhs), Some(rhs)) => lhs.same_as(rhs),
        (None, None) => true,
        _ => false,
    }
}
