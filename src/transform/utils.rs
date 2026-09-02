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
    Any, Array, Map, MapValue, MutateDispatch, Mutator, ObjectIdentity, ObjectRefCast,
    ObjectRefCore, Result, String, VisitContext, VisitInterrupt, VisitValue,
};

use crate::ir::{
    Call, DictAttrs, Expr, IntImm, OpaqueExpr, PrimExpr, PrimType, Range, TensorLoad, Var,
};
use crate::tirx::{
    AllocBuffer, AssertStmt, AttrStmt, Bind, BufferStore, BufferType, BufferVar, DeclBuffer,
    Evaluate, For, IfThenElse, Iter, IterVar, Layout, Let, PrimFunc, Reduce, Select, SeqStmt, Stmt,
    StringImm, TileLayout, While,
};

pub(super) fn int_value(expr: &Expr) -> Option<i64> {
    expr.clone()
        .try_cast::<IntImm>()
        .ok()
        .map(|value| value.value)
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

pub(super) fn cast_prim_expr(value: PrimExpr, target: PrimType) -> Result<PrimExpr> {
    tvm_ffi::cached_global_func!("tirx.Cast")
        .call_tuple((target, value, Option::<crate::ir::Span>::None))?
        .try_into()
}

/// Identity-preserving buffer-definition remaps used by semantic TIR mutators.
#[derive(Default)]
pub(super) struct BufferRemaps(HashMap<ObjectIdentity, BufferVar>);

impl BufferRemaps {
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
        let layout = mutate_tile_layout(&old_type.layout, &mut mutate)?;
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
        let mapped = BufferVar::try_from(Var::from_complete_fields(
            buffer.span.clone(),
            new_type.into(),
            buffer.name.clone(),
        ))?;
        self.0.insert(identity, mapped.clone());
        Ok(mapped)
    }
}

fn mutate_tile_layout<F>(layout: &Option<Layout>, mutate: &mut F) -> Result<Option<Layout>>
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
        .map(|iter| remap(iter, mutate))
        .collect::<Result<Vec<_>>>()?;
    let replica = old_replica
        .iter()
        .map(|iter| remap(iter, mutate))
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
    if let Some(expression) = value.cast::<Expr>() {
        if expression.clone().try_cast::<Var>().is_ok() {
            return Ok(None);
        }
        if let Ok(load) = expression.clone().try_cast::<TensorLoad>() {
            for index in load.indices.iter() {
                if let Some(interrupt) = visitor.visit(&index)? {
                    return Ok(Some(interrupt));
                }
            }
            return Ok(None);
        }
        if let Ok(call) = expression.clone().try_cast::<Call>() {
            if call.op.clone().try_cast::<OpaqueExpr>().is_ok() {
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
        if let Ok(let_expr) = expression.clone().try_cast::<Let>() {
            if let Some(interrupt) = visitor.visit(&let_expr.value)? {
                return Ok(Some(interrupt));
            }
            return visitor.visit(&let_expr.body);
        }
        if let Ok(select) = expression.clone().try_cast::<Select>() {
            if let Some(interrupt) = visitor.visit(&select.condition)? {
                return Ok(Some(interrupt));
            }
            if let Some(interrupt) = visitor.visit(&select.true_value)? {
                return Ok(Some(interrupt));
            }
            return visitor.visit(&select.false_value);
        }
        if let Ok(reduce) = expression.try_cast::<Reduce>() {
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
        return visitor.visit_children();
    }

    if let Some(statement) = value.cast::<Stmt>() {
        if let Ok(bind) = statement.clone().try_cast::<Bind>() {
            return visitor.visit(&bind.value);
        }
        if let Ok(attribute) = statement.clone().try_cast::<AttrStmt>() {
            if let Some(interrupt) = visitor.visit(&attribute.value)? {
                return Ok(Some(interrupt));
            }
            return visitor.visit(&attribute.body);
        }
        if let Ok(loop_node) = statement.clone().try_cast::<For>() {
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
        if let Ok(while_node) = statement.clone().try_cast::<While>() {
            if let Some(interrupt) = visitor.visit(&while_node.condition)? {
                return Ok(Some(interrupt));
            }
            return visitor.visit(&while_node.body);
        }
        if let Ok(allocation) = statement.clone().try_cast::<AllocBuffer>() {
            return visit_buffer_definition(visitor, &allocation.buffer);
        }
        if let Ok(declaration) = statement.clone().try_cast::<DeclBuffer>() {
            if let Some(interrupt) = visitor.visit(&declaration.data)? {
                return Ok(Some(interrupt));
            }
            return visit_buffer_definition(visitor, &declaration.buffer);
        }
        if let Ok(store) = statement.clone().try_cast::<BufferStore>() {
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
        if let Ok(conditional) = statement.clone().try_cast::<IfThenElse>() {
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
        if let Ok(assertion) = statement.clone().try_cast::<AssertStmt>() {
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
        if let Ok(sequence) = statement.clone().try_cast::<SeqStmt>() {
            for child in sequence.seq.iter() {
                if let Some(interrupt) = visitor.visit(&child)? {
                    return Ok(Some(interrupt));
                }
            }
            return Ok(None);
        }
        if let Ok(evaluate) = statement.try_cast::<Evaluate>() {
            return visitor.visit(&evaluate.value);
        }
        return visitor.visit_children();
    }

    visitor.visit_children()
}

fn visit_buffer_definition<State>(
    visitor: &mut VisitContext<'_, State>,
    buffer: &crate::tirx::BufferVar,
) -> Result<Option<VisitInterrupt>> {
    let buffer_type = buffer.type_annotation();
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
    if value.clone().try_cast::<Var>().is_ok() {
        return Ok(value);
    }
    if let Ok(load) = value.clone().try_cast::<TensorLoad>() {
        let indices: Array<PrimExpr> = mutator.mutate(dispatch, &load.indices)?.try_into()?;
        if array_same_as(&indices, &load.indices) {
            return Ok(value);
        }
        return Ok(TensorLoad::from_complete_fields(
            load.span.clone(),
            load.ty.clone().try_cast()?,
            load.source.clone(),
            indices,
        )
        .into());
    }
    if let Ok(call) = value.clone().try_cast::<Call>() {
        let op = if call.op.clone().try_cast::<OpaqueExpr>().is_ok() {
            mutator.mutate(dispatch, &call.op)?.try_into()?
        } else {
            call.op.clone()
        };
        let args: Array<Expr> = mutator.mutate(dispatch, &call.args)?.try_into()?;
        if op.same_as(&call.op) && array_same_as(&args, &call.args) {
            return Ok(value);
        }
        return Ok(Call::from_complete_fields(
            call.span.clone(),
            call.ty.clone(),
            op,
            args,
            call.attrs.clone(),
            call.ty_args.clone(),
        )
        .into());
    }
    if let Ok(let_expr) = value.clone().try_cast::<Let>() {
        let bound_value: PrimExpr = mutator.mutate(dispatch, &let_expr.value)?.try_into()?;
        let body: PrimExpr = mutator.mutate(dispatch, &let_expr.body)?.try_into()?;
        if bound_value.same_as(&let_expr.value) && body.same_as(&let_expr.body) {
            return Ok(value);
        }
        return Ok(Let::from_complete_fields(
            let_expr.span.clone(),
            body.type_annotation(),
            let_expr.var.clone(),
            bound_value,
            body,
        )
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
        return Ok(Select::from_complete_fields(
            select.span.clone(),
            true_value.type_annotation(),
            condition,
            true_value,
            false_value,
        )
        .into());
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
        return Ok(
            Bind::from_complete_fields(bind.span.clone(), bind.var.clone(), bound_value).into(),
        );
    }
    if let Ok(attribute) = value.clone().try_cast::<AttrStmt>() {
        let attr_value: PrimExpr = mutator.mutate(dispatch, &attribute.value)?.try_into()?;
        let body: Stmt = mutator.mutate(dispatch, &attribute.body)?.try_into()?;
        if attr_value.same_as(&attribute.value) && body.same_as(&attribute.body) {
            return Ok(value);
        }
        return Ok(AttrStmt::from_complete_fields(
            attribute.span.clone(),
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
        return Ok(While::from_complete_fields(while_node.span.clone(), condition, body).into());
    }
    if value.clone().try_cast::<AllocBuffer>().is_ok() {
        // Buffer-definition recursion requires a pass-specific remap table.
        // A pass that changes buffer metadata supplies its own handlers.
        return Ok(value);
    }
    if let Ok(declaration) = value.clone().try_cast::<DeclBuffer>() {
        let data: Expr = mutator.mutate(dispatch, &declaration.data)?.try_into()?;
        if data.same_as(&declaration.data) {
            return Ok(value);
        }
        return Ok(DeclBuffer::from_complete_fields(
            declaration.span.clone(),
            declaration.buffer.clone(),
            data,
        )
        .into());
    }
    if let Ok(store) = value.clone().try_cast::<BufferStore>() {
        let stored_value: PrimExpr = mutator.mutate(dispatch, &store.value)?.try_into()?;
        let indices: Array<PrimExpr> = mutator.mutate(dispatch, &store.indices)?.try_into()?;
        if stored_value.same_as(&store.value) && array_same_as(&indices, &store.indices) {
            return Ok(value);
        }
        return Ok(BufferStore::from_complete_fields(
            store.span.clone(),
            store.buffer.clone(),
            stored_value,
            indices,
        )
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
        return Ok(AssertStmt::from_complete_fields(
            assertion.span.clone(),
            condition,
            error_kind,
            message_parts,
        )
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
        return Ok(Evaluate::from_complete_fields(evaluate.span.clone(), evaluated).into());
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
        Ok(Range::from_complete_fields(
            minimum,
            extent,
            value.span.clone(),
        ))
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
