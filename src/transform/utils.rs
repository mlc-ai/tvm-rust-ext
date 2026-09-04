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
    Any, Array, DLDataType, Map, MapValue, MutateDispatch, Mutator, ObjectIdentity, ObjectRefCast,
    ObjectRefCore, Result, String, VisitContext, VisitInterrupt, VisitValue,
};

use crate::ir::prim::{Let, LetObj, Select, SelectObj, StringImm, StringImmObj};
use crate::ir::{
    Call, CallObj, DictAttrs, Expr, IntImmObj, OpaqueExprObj, PointerTypeObj, PrimExpr, PrimType,
    PrimTypeObj, Range, TensorLoad, TensorLoadObj, Type, Var, VarObj,
};
use crate::te::{Reduce, ReduceObj};
use crate::tirx::{
    AllocBufferObj, AssertStmt, AssertStmtObj, AttrStmt, AttrStmtObj, Bind, BindObj, BufferStore,
    BufferStoreObj, BufferType, BufferTypeObj, BufferVar, DeclBuffer, DeclBufferObj, Evaluate,
    EvaluateObj, For, ForObj, IfThenElse, IfThenElseObj, Iter, IterVar, Layout, PrimFunc, SeqStmt,
    SeqStmtObj, Stmt, TileLayout, While, WhileObj,
};

pub(super) fn int_value<T: ObjectRefCore>(expr: &T) -> Option<i64> {
    expr.as_node::<IntImmObj>().map(|value| value.value)
}

pub(super) fn get_operator(name: &str) -> Result<Expr> {
    tvm_ffi::cached_global_func!("ir.GetOp")
        .call_tuple((String::from(name),))?
        .try_into()
}

pub(super) fn operator_identity(name: &str) -> Result<ObjectIdentity> {
    Ok(ObjectIdentity::of(&get_operator(name)?))
}

pub(super) fn value_error(message: &str) -> tvm_ffi::Error {
    tvm_ffi::Error::new(tvm_ffi::VALUE_ERROR, message, "")
}

pub(super) fn is_call<T: ObjectRefCore>(value: &T) -> bool {
    value.as_node::<CallObj>().is_some()
}

pub(super) fn is_opaque_expr(value: &Expr) -> bool {
    value.as_node::<OpaqueExprObj>().is_some()
}

pub(super) fn is_pointer_type(value: &Type) -> bool {
    value.as_node::<PointerTypeObj>().is_some()
}

pub(super) fn is_primitive_type(value: &Type) -> bool {
    value.as_node::<PrimTypeObj>().is_some()
}

pub(super) fn is_buffer_type(value: &Type) -> bool {
    value.as_node::<BufferTypeObj>().is_some()
}

pub(super) fn is_buffer_var(value: &Var) -> bool {
    is_buffer_type(&value.ty)
}

pub(super) fn is_string_imm<T: ObjectRefCore>(value: &T) -> bool {
    value.as_node::<StringImmObj>().is_some()
}

pub(super) fn int_dtype_and_value<T: ObjectRefCore>(expr: &T) -> Option<(DLDataType, i64)> {
    let literal = expr.as_node::<IntImmObj>()?;
    let dtype = literal.ty.as_node::<PrimTypeObj>()?.dtype;
    Some((dtype, literal.value))
}

pub(super) fn variable_name<T: ObjectRefCore>(expr: &T) -> Option<&str> {
    expr.as_node::<VarObj>()
        .map(|variable| variable.name.as_str())
}

pub(super) fn is_evaluate_zero(statement: &Stmt) -> bool {
    statement
        .as_node::<EvaluateObj>()
        .is_some_and(|evaluate| int_value(&evaluate.value) == Some(0))
}

pub(super) fn with_prim_func_body(function: PrimFunc, body: Stmt) -> PrimFunc {
    PrimFunc::from_complete_fields(
        function.span.clone(),
        function.ty.clone(),
        function.attrs.clone(),
        function.params.clone(),
        function.ret_type.clone(),
        body,
    )
}

pub(super) fn with_prim_func_attr(
    function: PrimFunc,
    key: &str,
    value: impl Into<Any>,
) -> PrimFunc {
    let key = String::from(key);
    let mut attributes = function
        .attrs
        .dict
        .iter()
        .filter(|(existing, _)| existing.as_str() != key.as_str())
        .collect::<Vec<_>>();
    attributes.push((key, value.into()));
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

pub(super) fn without_prim_func_attr(function: PrimFunc, key: &str) -> PrimFunc {
    let attrs = DictAttrs::from_dictionary(Map::from_iter(
        function
            .attrs
            .dict
            .iter()
            .filter(|(existing, _)| existing.as_str() != key),
    ));
    PrimFunc::from_complete_fields(
        function.span.clone(),
        function.ty.clone(),
        attrs,
        function.params.clone(),
        function.ret_type.clone(),
        function.body.clone(),
    )
}

pub(super) fn cast_prim_expr(value: PrimExpr, target: PrimType) -> Result<PrimExpr> {
    tvm_ffi::cached_global_func!("tirx._cast")
        .call_tuple((target, value, Option::<crate::ir::Span>::None))?
        .try_into()
}

/// Identity-preserving buffer-definition remaps used by semantic TIR mutators.
#[derive(Default)]
pub(super) struct BufferRemaps(HashMap<ObjectIdentity, BufferVar>);

impl BufferRemaps {
    pub(super) fn clear(&mut self) {
        self.0.clear();
    }

    pub(super) fn use_buffer(&self, buffer: &BufferVar) -> BufferVar {
        self.0
            .get(&ObjectIdentity::of(buffer.as_var()))
            .cloned()
            .unwrap_or_else(|| buffer.clone())
    }

    pub(super) fn use_variable(&self, variable: &Var) -> Var {
        self.0
            .get(&ObjectIdentity::of(variable))
            .map(|buffer| buffer.as_var().clone())
            .unwrap_or_else(|| variable.clone())
    }

    pub(super) fn mutate_definition<F>(
        &mut self,
        buffer: &BufferVar,
        mut mutate: F,
    ) -> Result<BufferVar>
    where
        F: FnMut(&PrimExpr) -> Result<PrimExpr>,
    {
        let identity = ObjectIdentity::of(buffer.as_var());
        if let Some(mapped) = self.0.get(&identity) {
            return Ok(mapped.clone());
        }
        let old_type = buffer.type_annotation();
        let shape = old_type
            .shape
            .iter()
            .map(|expression| mutate(&expression))
            .collect::<Result<Vec<_>>>()?;
        let strides = old_type
            .strides
            .iter()
            .map(|expression| mutate(&expression))
            .collect::<Result<Vec<_>>>()?;
        let elem_offset = mutate(&old_type.elem_offset)?;
        let allocated_addr = old_type
            .allocated_addr
            .iter()
            .map(|expression| mutate(&expression))
            .collect::<Result<Vec<_>>>()?;
        let layout = mutate_layout(&old_type.layout, &mut mutate)?;
        let shape = Array::new(shape);
        let strides = Array::new(strides);
        let allocated_addr = Array::new(allocated_addr);
        if array_same_as(&shape, &old_type.shape)
            && array_same_as(&strides, &old_type.strides)
            && elem_offset.same_as(&old_type.elem_offset)
            && array_same_as(&allocated_addr, &old_type.allocated_addr)
            && option_same_as(&layout, &old_type.layout)
        {
            return Ok(buffer.clone());
        }
        let new_type = BufferType::from_complete_fields(
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
        let mapped = BufferVar::try_from(buffer.copy_with(buffer.name.clone(), new_type.into()))?;
        self.0.insert(identity, mapped.clone());
        Ok(mapped)
    }
}

pub(super) fn mutate_layout<F>(layout: &Option<Layout>, mut mutate: F) -> Result<Option<Layout>>
where
    F: FnMut(&PrimExpr) -> Result<PrimExpr>,
{
    let Some(original) = layout else {
        return Ok(None);
    };
    let Ok(tile) = original.clone().try_cast::<TileLayout>() else {
        return Ok(Some(original.clone()));
    };
    let old_shard = tile.shard()?;
    let old_replica = tile.replica()?;
    let remap = |iter: Iter, mutate: &mut F| -> Result<Iter> {
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
        .map(|iter| remap(iter, &mut mutate))
        .collect::<Result<Vec<_>>>()?;
    let replica = old_replica
        .iter()
        .map(|iter| remap(iter, &mut mutate))
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

/// Apply TVM's `StmtExprVisitor` child policy to a structural value.
pub(super) fn visit_stmt_expr_default<State>(
    visitor: &mut VisitContext<'_, State>,
    value: &VisitValue,
) -> Result<Option<VisitInterrupt>> {
    if value.as_node::<VarObj>().is_some() {
        return Ok(None);
    }
    if let Some(load) = value.as_node::<TensorLoadObj>() {
        for index in load.indices.iter() {
            if let Some(interrupt) = visitor.visit(&index)? {
                return Ok(Some(interrupt));
            }
        }
        return Ok(None);
    }
    if let Some(call) = value.as_node::<CallObj>() {
        if call.op.as_node::<OpaqueExprObj>().is_some() {
            if let Some(interrupt) = visitor.visit(&call.op)? {
                return Ok(Some(interrupt));
            }
        }
        for argument in call.args.iter() {
            if let Some(interrupt) = visitor.visit(&argument)? {
                return Ok(Some(interrupt));
            }
        }
        return Ok(None);
    }
    if let Some(let_expr) = value.as_node::<LetObj>() {
        if let Some(interrupt) = visitor.visit(&let_expr.value)? {
            return Ok(Some(interrupt));
        }
        return visitor.visit(&let_expr.body);
    }
    if let Some(select) = value.as_node::<SelectObj>() {
        if let Some(interrupt) = visitor.visit(&select.condition)? {
            return Ok(Some(interrupt));
        }
        if let Some(interrupt) = visitor.visit(&select.true_value)? {
            return Ok(Some(interrupt));
        }
        return visitor.visit(&select.false_value);
    }
    if let Some(reduce) = value.as_node::<ReduceObj>() {
        for axis in reduce.axis.iter() {
            if let Some(domain) = axis.dom()? {
                if let Some(interrupt) = visitor.visit(&domain.min)? {
                    return Ok(Some(interrupt));
                }
                if let Some(interrupt) = visitor.visit(&domain.extent)? {
                    return Ok(Some(interrupt));
                }
            }
        }
        for source in reduce.source.iter() {
            if let Some(interrupt) = visitor.visit(&source)? {
                return Ok(Some(interrupt));
            }
        }
        for init in reduce.init.iter() {
            if let Some(interrupt) = visitor.visit(&init)? {
                return Ok(Some(interrupt));
            }
        }
        return visitor.visit(&reduce.condition);
    }
    if let Some(bind) = value.as_node::<BindObj>() {
        return visitor.visit(&bind.value);
    }
    if let Some(attribute) = value.as_node::<AttrStmtObj>() {
        if let Some(interrupt) = visitor.visit(&attribute.value)? {
            return Ok(Some(interrupt));
        }
        return visitor.visit(&attribute.body);
    }
    if let Some(loop_node) = value.as_node::<ForObj>() {
        if let Some(interrupt) = visitor.visit(&loop_node.min)? {
            return Ok(Some(interrupt));
        }
        if let Some(interrupt) = visitor.visit(&loop_node.extent)? {
            return Ok(Some(interrupt));
        }
        if let Some(step) = &loop_node.step {
            if let Some(interrupt) = visitor.visit(step)? {
                return Ok(Some(interrupt));
            }
        }
        return visitor.visit(&loop_node.body);
    }
    if let Some(while_node) = value.as_node::<WhileObj>() {
        if let Some(interrupt) = visitor.visit(&while_node.condition)? {
            return Ok(Some(interrupt));
        }
        return visitor.visit(&while_node.body);
    }
    if let Some(allocation) = value.as_node::<AllocBufferObj>() {
        return visit_buffer_definition(visitor, &allocation.buffer);
    }
    if let Some(declaration) = value.as_node::<DeclBufferObj>() {
        if let Some(interrupt) = visitor.visit(&declaration.data)? {
            return Ok(Some(interrupt));
        }
        return visit_buffer_definition(visitor, &declaration.buffer);
    }
    if let Some(store) = value.as_node::<BufferStoreObj>() {
        if let Some(interrupt) = visitor.visit(&store.value)? {
            return Ok(Some(interrupt));
        }
        for index in store.indices.iter() {
            if let Some(interrupt) = visitor.visit(&index)? {
                return Ok(Some(interrupt));
            }
        }
        return Ok(None);
    }
    if let Some(conditional) = value.as_node::<IfThenElseObj>() {
        if let Some(interrupt) = visitor.visit(&conditional.condition)? {
            return Ok(Some(interrupt));
        }
        if let Some(interrupt) = visitor.visit(&conditional.then_case)? {
            return Ok(Some(interrupt));
        }
        if let Some(branch) = &conditional.else_case {
            return visitor.visit(branch);
        }
        return Ok(None);
    }
    if let Some(assertion) = value.as_node::<AssertStmtObj>() {
        if let Some(interrupt) = visitor.visit(&assertion.condition)? {
            return Ok(Some(interrupt));
        }
        if let Some(interrupt) = visitor.visit(&assertion.error_kind)? {
            return Ok(Some(interrupt));
        }
        for part in assertion.message_parts.iter() {
            if let Some(interrupt) = visitor.visit(&part)? {
                return Ok(Some(interrupt));
            }
        }
        return Ok(None);
    }
    if let Some(sequence) = value.as_node::<SeqStmtObj>() {
        for child in sequence.seq.iter() {
            if let Some(interrupt) = visitor.visit(&child)? {
                return Ok(Some(interrupt));
            }
        }
        return Ok(None);
    }
    if let Some(evaluate) = value.as_node::<EvaluateObj>() {
        return visitor.visit(&evaluate.value);
    }

    visitor.visit_children()
}

fn visit_buffer_definition<State>(
    visitor: &mut VisitContext<'_, State>,
    buffer: &crate::tirx::BufferVar,
) -> Result<Option<VisitInterrupt>> {
    let buffer_type = buffer.buffer_type();
    for expression in buffer_type.shape.iter().chain(buffer_type.strides.iter()) {
        if let Some(interrupt) = visitor.visit(&expression)? {
            return Ok(Some(interrupt));
        }
    }
    if let Some(interrupt) = visitor.visit(&buffer_type.elem_offset)? {
        return Ok(Some(interrupt));
    }
    for expression in buffer_type.allocated_addr.iter() {
        if let Some(interrupt) = visitor.visit(&expression)? {
            return Ok(Some(interrupt));
        }
    }
    if let Some(layout) = &buffer_type.layout {
        if let Ok(tile) = layout.clone().try_cast::<crate::tirx::TileLayout>() {
            for iter in tile.shard()?.iter().chain(tile.replica()?.iter()) {
                if let Some(interrupt) = visitor.visit(&iter.extent)? {
                    return Ok(Some(interrupt));
                }
                if let Some(interrupt) = visitor.visit(&iter.stride)? {
                    return Ok(Some(interrupt));
                }
            }
        }
    }
    Ok(None)
}

/// Apply TVM's `StmtExprMutator` child policy to a generic structural value.
///
/// Structural mutation normally follows reflected structural fields.  TIR
/// mutation is intentionally narrower: binders, annotations, call metadata,
/// expression types, and source metadata are not ordinary recursive children.
/// Pass-specific handlers run first; this catch-all supplies the matching TIR
/// default for the remaining handwritten node set.
pub(super) fn mutate_stmt_expr_default<D: MutateDispatch>(
    dispatch: &mut D,
    mutator: &mut Mutator,
    value: &MapValue,
) -> Result<Any> {
    if let Some(expression) = value.cast::<Expr>() {
        return mutate_expr_default(dispatch, mutator, expression).map(Into::into);
    }
    if let Some(statement) = value.cast::<Stmt>() {
        return mutate_stmt_default(dispatch, mutator, statement).map(Into::into);
    }
    mutator.default_mutate(dispatch)
}

pub(super) fn mutate_expr_default<D: MutateDispatch>(
    dispatch: &mut D,
    mutator: &mut Mutator,
    value: Expr,
) -> Result<Expr> {
    if value.as_node::<VarObj>().is_some() {
        return Ok(value);
    }
    if let Ok(load) = value.clone().try_cast::<TensorLoad>() {
        let indices: Array<PrimExpr> = mutator.mutate(dispatch, &load.indices)?.try_into()?;
        if array_same_as(&indices, &load.indices) {
            return Ok(value);
        }
        return Ok(load.copy_with(load.source.clone(), indices).into());
    }
    if let Ok(call) = value.clone().try_cast::<Call>() {
        let op = if is_opaque_expr(&call.op) {
            mutator.mutate(dispatch, &call.op)?.try_into()?
        } else {
            call.op.clone()
        };
        let args: Array<Expr> = mutator.mutate(dispatch, &call.args)?.try_into()?;
        if op.same_as(&call.op) && array_same_as(&args, &call.args) {
            return Ok(value);
        }
        return Ok(call.copy_with(call.ty.clone(), op, args).into());
    }
    if let Ok(let_expr) = value.clone().try_cast::<Let>() {
        let bound_value: PrimExpr = mutator.mutate(dispatch, &let_expr.value)?.try_into()?;
        let body: PrimExpr = mutator.mutate(dispatch, &let_expr.body)?.try_into()?;
        if bound_value.same_as(&let_expr.value) && body.same_as(&let_expr.body) {
            return Ok(value);
        }
        return Ok(let_expr
            .copy_with(let_expr.var.clone(), bound_value, body)
            .into());
    }
    if let Ok(select) = value.clone().try_cast::<Select>() {
        let condition: PrimExpr = mutator.mutate(dispatch, &select.condition)?.try_into()?;
        let true_value: PrimExpr = mutator.mutate(dispatch, &select.true_value)?.try_into()?;
        let false_value: PrimExpr = mutator.mutate(dispatch, &select.false_value)?.try_into()?;
        if condition.same_as(&select.condition)
            && true_value.same_as(&select.true_value)
            && false_value.same_as(&select.false_value)
        {
            return Ok(value);
        }
        return Ok(select.copy_with(condition, true_value, false_value).into());
    }
    if let Ok(reduce) = value.clone().try_cast::<Reduce>() {
        let mut axes = Vec::with_capacity(reduce.axis.len());
        for axis in reduce.axis.iter() {
            let old_domain = axis.dom()?;
            let domain = old_domain
                .as_ref()
                .map(|domain| mutate_range(dispatch, mutator, domain))
                .transpose()?;
            if option_same_as(&domain, &old_domain) {
                axes.push(axis);
            } else {
                axes.push(IterVar::with_metadata(
                    domain,
                    axis.var()?.as_var().clone(),
                    axis.iter_type()?,
                    axis.thread_tag()?.as_str(),
                    axis.span()?.as_ref(),
                )?);
            }
        }
        let source: Array<PrimExpr> = mutator.mutate(dispatch, &reduce.source)?.try_into()?;
        let init: Array<PrimExpr> = mutator.mutate(dispatch, &reduce.init)?.try_into()?;
        let condition: PrimExpr = mutator.mutate(dispatch, &reduce.condition)?.try_into()?;
        let axes = Array::new(axes);
        if array_same_as(&source, &reduce.source)
            && array_same_as(&init, &reduce.init)
            && array_same_as(&axes, &reduce.axis)
            && condition.same_as(&reduce.condition)
        {
            return Ok(value);
        }
        return Ok(Reduce::from_complete_fields(
            reduce.span.clone(),
            reduce.ty.clone().try_cast()?,
            reduce.combiner.clone(),
            source,
            init,
            axes,
            condition,
            reduce.value_index,
        )
        .into());
    }
    mutator.default_mutate(dispatch).and_then(Expr::try_from)
}

pub(super) fn mutate_stmt_default<D: MutateDispatch>(
    dispatch: &mut D,
    mutator: &mut Mutator,
    value: Stmt,
) -> Result<Stmt> {
    if let Ok(bind) = value.clone().try_cast::<Bind>() {
        let bound_value: Expr = mutator.mutate(dispatch, &bind.value)?.try_into()?;
        if bound_value.same_as(&bind.value) {
            return Ok(value);
        }
        return Ok(bind.copy_with(bind.var.clone(), bound_value).into());
    }
    if let Ok(attribute) = value.clone().try_cast::<AttrStmt>() {
        let attr_value: PrimExpr = mutator.mutate(dispatch, &attribute.value)?.try_into()?;
        let body: Stmt = mutator.mutate(dispatch, &attribute.body)?.try_into()?;
        if attr_value.same_as(&attribute.value) && body.same_as(&attribute.body) {
            return Ok(value);
        }
        return Ok(attribute
            .copy_with(
                attribute.node.clone(),
                attribute.attr_key.clone(),
                attr_value,
                body,
            )
            .into());
    }
    if let Ok(loop_node) = value.clone().try_cast::<For>() {
        let minimum: PrimExpr = mutator.mutate(dispatch, &loop_node.min)?.try_into()?;
        let extent: PrimExpr = mutator.mutate(dispatch, &loop_node.extent)?.try_into()?;
        let step: Option<PrimExpr> = mutator.mutate(dispatch, &loop_node.step)?.try_into()?;
        let body: Stmt = mutator.mutate(dispatch, &loop_node.body)?.try_into()?;
        if minimum.same_as(&loop_node.min)
            && extent.same_as(&loop_node.extent)
            && option_same_as(&step, &loop_node.step)
            && body.same_as(&loop_node.body)
        {
            return Ok(value);
        }
        return Ok(For::from_complete_fields(
            loop_node.span.clone(),
            loop_node.loop_var.clone(),
            minimum,
            extent,
            loop_node.kind,
            body,
            loop_node.thread_binding.clone(),
            loop_node.annotations.clone(),
            step,
        )
        .into());
    }
    if let Ok(while_node) = value.clone().try_cast::<While>() {
        let condition: PrimExpr = mutator
            .mutate(dispatch, &while_node.condition)?
            .try_into()?;
        let body: Stmt = mutator.mutate(dispatch, &while_node.body)?.try_into()?;
        if condition.same_as(&while_node.condition) && body.same_as(&while_node.body) {
            return Ok(value);
        }
        return Ok(while_node.copy_with(condition, body).into());
    }
    if value.as_node::<AllocBufferObj>().is_some() {
        // Buffer-definition recursion requires a pass-specific remap table.
        // A pass that changes buffer metadata supplies its own handlers.
        return Ok(value);
    }
    if let Ok(declaration) = value.clone().try_cast::<DeclBuffer>() {
        let data: Expr = mutator.mutate(dispatch, &declaration.data)?.try_into()?;
        if data.same_as(&declaration.data) {
            return Ok(value);
        }
        return Ok(declaration
            .copy_with(declaration.buffer.clone(), data)
            .into());
    }
    if let Ok(store) = value.clone().try_cast::<BufferStore>() {
        let stored_value: PrimExpr = mutator.mutate(dispatch, &store.value)?.try_into()?;
        let indices: Array<PrimExpr> = mutator.mutate(dispatch, &store.indices)?.try_into()?;
        if stored_value.same_as(&store.value) && array_same_as(&indices, &store.indices) {
            return Ok(value);
        }
        return Ok(store
            .copy_with(store.buffer.clone(), stored_value, indices)
            .into());
    }
    if let Ok(conditional) = value.clone().try_cast::<IfThenElse>() {
        let condition: PrimExpr = mutator
            .mutate(dispatch, &conditional.condition)?
            .try_into()?;
        let then_case: Stmt = mutator
            .mutate(dispatch, &conditional.then_case)?
            .try_into()?;
        let else_case: Option<Stmt> = mutator
            .mutate(dispatch, &conditional.else_case)?
            .try_into()?;
        if condition.same_as(&conditional.condition)
            && then_case.same_as(&conditional.then_case)
            && option_same_as(&else_case, &conditional.else_case)
        {
            return Ok(value);
        }
        return Ok(IfThenElse::from_complete_fields(
            conditional.span.clone(),
            condition,
            then_case,
            else_case,
        )
        .into());
    }
    if let Ok(assertion) = value.clone().try_cast::<AssertStmt>() {
        let condition: PrimExpr = mutator.mutate(dispatch, &assertion.condition)?.try_into()?;
        let error_kind: StringImm = mutator
            .mutate(dispatch, &assertion.error_kind)?
            .try_into()?;
        let message_parts: Array<StringImm> = mutator
            .mutate(dispatch, &assertion.message_parts)?
            .try_into()?;
        if condition.same_as(&assertion.condition)
            && error_kind.same_as(&assertion.error_kind)
            && array_same_as(&message_parts, &assertion.message_parts)
        {
            return Ok(value);
        }
        return Ok(assertion
            .copy_with(condition, error_kind, message_parts)
            .into());
    }
    if let Ok(sequence) = value.clone().try_cast::<SeqStmt>() {
        let statements: Array<Stmt> = mutator.mutate(dispatch, &sequence.seq)?.try_into()?;
        return Stmt::sequence_with_span(statements.iter().collect(), sequence.span.as_ref());
    }
    if let Ok(evaluate) = value.clone().try_cast::<Evaluate>() {
        let evaluated: Expr = mutator.mutate(dispatch, &evaluate.value)?.try_into()?;
        if evaluated.same_as(&evaluate.value) {
            return Ok(value);
        }
        return Ok(evaluate.copy_with(evaluated).into());
    }
    mutator.default_mutate(dispatch).and_then(Stmt::try_from)
}

fn mutate_range<D: MutateDispatch>(
    dispatch: &mut D,
    mutator: &mut Mutator,
    value: &Range,
) -> Result<Range> {
    let minimum: PrimExpr = mutator.mutate(dispatch, &value.min)?.try_into()?;
    let extent: PrimExpr = mutator.mutate(dispatch, &value.extent)?.try_into()?;
    if minimum.same_as(&value.min) && extent.same_as(&value.extent) {
        Ok(value.clone())
    } else {
        Ok(value.copy_with(minimum, extent))
    }
}

pub(super) fn array_same_as<T>(lhs: &Array<T>, rhs: &Array<T>) -> bool
where
    T: ObjectRefCore + tvm_ffi::AnyCompatible + Clone,
{
    lhs.len() == rhs.len()
        && lhs
            .iter()
            .zip(rhs.iter())
            .all(|(lhs, rhs)| lhs.same_as(&rhs))
}

pub(super) fn option_same_as<T: ObjectRefCore>(lhs: &Option<T>, rhs: &Option<T>) -> bool {
    match (lhs, rhs) {
        (Some(lhs), Some(rhs)) => lhs.same_as(rhs),
        (None, None) => true,
        _ => false,
    }
}
