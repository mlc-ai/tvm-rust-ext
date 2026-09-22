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
    Any, AnyCompatible, Array, DLDataType, Function, Map, MutateDispatch, Mutator, ObjectIdentity,
    ObjectRefCast, ObjectRefCore, Result, String, StructuralView, VisitContext, VisitInterrupt,
};

use crate::ir::{
    Call, CallObj, DictAttrs, Expr, FloatImmObj, GlobalVarObj, IntImm, IntImmObj, Op, OpObj,
    OpaqueExprObj, PointerType, PointerTypeObj, PrimExpr, PrimType, PrimTypeObj, Range, TensorLoad,
    TensorLoadObj, Tuple, TupleGetItem, TupleGetItemObj, TupleObj, Type, UniqueNameSupply, Var,
    VarObj,
};
use crate::ir::{StringImm, StringImmObj};
use crate::ir::{TensorRegion, TensorRegionObj};
use crate::prim::{
    AddObj, AndObj, Broadcast, BroadcastObj, CastObj, DivObj, EQObj, FloorDivObj, FloorModObj,
    GEObj, GTObj, LEObj, LTObj, Let, LetObj, MaxObj, MinObj, ModObj, MulObj, NEObj, NotObj, OrObj,
    Ramp, RampObj, SelectObj, Shuffle, ShuffleObj, SubObj,
};
use crate::tirx::{
    AllocBuffer, AllocBufferObj, AssertStmt, AssertStmtObj, AttrStmt, AttrStmtObj, Bind, BindObj,
    BreakObj, BufferStore, BufferStoreObj, BufferType, BufferTypeObj, BufferVar, ContinueObj,
    DeclBuffer, DeclBufferObj, Evaluate, EvaluateObj, For, ForObj, IfThenElse, IfThenElseObj, Iter,
    Layout, PrimFunc, Return, ReturnObj, ScopeIdDef, ScopeIdDefStmt, SeqStmt, SeqStmtObj, Stmt,
    TileLayout, TilePrimitiveCall, While, WhileObj,
};

pub(super) fn int_value<T: ObjectRefCore>(expr: &T) -> Option<i64> {
    expr.as_node::<IntImmObj>().map(|value| value.value_i64())
}

pub(super) fn get_operator(name: &str) -> Result<Expr> {
    Op::get(name).map(Into::into)
}

pub(super) fn substitute_vars(node: Any, replacements: &Map<Var, Expr>) -> Result<Any> {
    // Map callbacks run for each occurrence; default descent does not cache
    // callback replacements as variable bindings.
    tvm_ffi::structural_map(
        node,
        |variable: Var| -> Result<Any> {
            Ok(match replacements.get(&variable)? {
                Some(replacement) => replacement.into(),
                None => variable.into(),
            })
        },
        tvm_ffi::WalkOrder::PostOrder,
    )
}

pub(super) fn const_handle(value: i64) -> Result<Expr> {
    Ok(Call::new(
        PointerType::new(PrimType::void(), "")?,
        get_operator("tirx.reinterpret")?,
        vec![IntImm::new("uint64", value)?.into()],
    )
    .into())
}

pub(super) fn global_name_supply(module: &crate::ir::IRModule) -> Result<UniqueNameSupply> {
    let names = UniqueNameSupply::new("")?;
    for (global, _) in module.functions.iter() {
        names.reserve_name(global.name_hint.as_str(), false)?;
    }
    Ok(names)
}

pub(super) fn value_error(message: &str) -> tvm_ffi::Error {
    tvm_ffi::Error::new(tvm_ffi::VALUE_ERROR, message, "")
}

/// Decode a fixed lane count; negative encodings describe scalable vectors.
pub(super) fn fixed_lanes(ty: &PrimType) -> Result<i64> {
    i16::try_from(ty.dtype.lanes)
        .map(i64::from)
        .map_err(|_| value_error("scalable vector has no fixed lane count"))
}

pub(super) fn storage_bytes(ty: &PrimType) -> Result<i64> {
    Ok((i64::from(ty.dtype.bits) * fixed_lanes(ty)? + 7) / 8)
}

/// Exit nested analyzer constraints inside-out, preserving the original error.
pub(super) fn finish_constraint_contexts<T>(result: Result<T>, exits: Vec<Function>) -> Result<T> {
    let mut exit_error = None;
    for exit in exits.into_iter().rev() {
        if let Err(error) = exit.call_tuple(()) {
            exit_error.get_or_insert(error);
        }
    }
    match (result, exit_error) {
        (Err(error), _) | (Ok(_), Some(error)) => Err(error),
        (Ok(value), None) => Ok(value),
    }
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
    Some((dtype, literal.value_i64()))
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
        function.body().clone(),
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
        function.body().clone(),
    )
}

pub(super) fn cast_prim_expr(value: PrimExpr, target: PrimType) -> Result<PrimExpr> {
    tvm_ffi::cached_global_func!("prim._cast")
        .call_tuple((target, value, Option::<crate::ir::Span>::None))?
        .try_into()
}

/// Use TVM's operator builders for type matching and constant folding.
pub(super) fn binary_op(name: &str, lhs: PrimExpr, rhs: PrimExpr) -> Result<PrimExpr> {
    Function::get_global(name)?
        .call_tuple((lhs, rhs, Option::<crate::ir::Span>::None))?
        .try_into()
}

/// Clone and downcast an object only after a borrowed node check succeeds.
///
/// A direct `value.clone().try_cast()` changes the reference count even when
/// the dynamic type does not match. Default mutators test several node types
/// in sequence, so keep failed probes borrowed and create one owning handle
/// only for the matching branch.
pub(super) fn clone_downcast<B>(value: &impl ObjectRefCast) -> Result<Option<B>>
where
    B: ObjectRefCore + AnyCompatible,
{
    if value.as_node::<B::ContainerType>().is_none() {
        return Ok(None);
    }
    value.clone().try_cast().map(Some)
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

    /// Match StmtExprMutator: rewrite buffer definitions once, then reuse them.
    pub(super) fn mutate_stmt<D: MutateDispatch>(
        dispatch: &mut D,
        mutator: &mut Mutator,
        value: Stmt,
        remaps: impl Fn(&mut D) -> &mut Self,
    ) -> Result<Stmt> {
        if let Some(allocation) = clone_downcast::<AllocBuffer>(&value)? {
            let buffer = Self::mutate_definition(
                dispatch,
                &allocation.buffer,
                remaps,
                |dispatch, expression| mutator.mutate(dispatch, expression)?.try_into(),
            )?;
            if buffer.same_as(&allocation.buffer) {
                return Ok(value);
            }
            return Ok(allocation.copy_with(buffer).into());
        }
        if let Some(declaration) = clone_downcast::<DeclBuffer>(&value)? {
            let data: Expr = mutator.mutate(dispatch, &declaration.data)?.try_into()?;
            let buffer = Self::mutate_definition(
                dispatch,
                &declaration.buffer,
                remaps,
                |dispatch, expression| mutator.mutate(dispatch, expression)?.try_into(),
            )?;
            if data.same_as(&declaration.data) && buffer.same_as(&declaration.buffer) {
                return Ok(value);
            }
            return Ok(declaration.copy_with(buffer, data).into());
        }
        if let Some(store) = clone_downcast::<BufferStore>(&value)? {
            let buffer = remaps(dispatch).use_buffer(&store.buffer);
            let stored_value: PrimExpr = mutator.mutate(dispatch, &store.value)?.try_into()?;
            let indices = mutator.mutate(dispatch, &store.indices)?.try_into()?;
            if buffer.same_as(&store.buffer)
                && stored_value.same_as(&store.value)
                && array_same_as(&indices, &store.indices)
            {
                return Ok(value);
            }
            return Ok(store.copy_with(buffer, stored_value, indices).into());
        }
        mutate_stmt_default(dispatch, mutator, value)
    }

    pub(super) fn mutate_expr<D: MutateDispatch>(
        dispatch: &mut D,
        mutator: &mut Mutator,
        value: Expr,
        remaps: impl Fn(&mut D) -> &mut Self,
    ) -> Result<Expr> {
        if let Some(variable) = clone_downcast::<Var>(&value)? {
            return Ok(remaps(dispatch).use_variable(&variable).into());
        }
        if let Some(load) = clone_downcast::<TensorLoad>(&value)? {
            let source: BufferVar = (&load.source).try_into()?;
            let source = remaps(dispatch).use_buffer(&source);
            let indices: Array<PrimExpr> = mutator.mutate(dispatch, &load.indices)?.try_into()?;
            if source.as_var().same_as(&load.source) && array_same_as(&indices, &load.indices) {
                return Ok(value);
            }
            return Ok(TensorLoad::from_buffer_with_span(
                source.into(),
                indices.iter().map(Into::into).collect(),
                load.span.as_ref(),
            )?
            .into());
        }
        if let Some(region) = clone_downcast::<TensorRegion>(&value)? {
            let buffer = remaps(dispatch).use_buffer(&BufferVar::try_from(region.source.clone())?);
            return mutate_buffer_region_with_buffer(dispatch, mutator, region, buffer)
                .map(Into::into);
        }
        mutate_expr_default(dispatch, mutator, value)
    }

    pub(super) fn mutate_default<D: MutateDispatch>(
        dispatch: &mut D,
        mutator: &mut Mutator,
        value: &StructuralView,
        remaps: impl Fn(&mut D) -> &mut Self,
    ) -> Result<Any> {
        if let Some(statement) = value.cast::<Stmt>() {
            return Self::mutate_stmt(dispatch, mutator, statement, remaps).map(Into::into);
        }
        if let Some(expression) = value.cast::<Expr>() {
            return Self::mutate_expr(dispatch, mutator, expression, remaps).map(Into::into);
        }
        mutator.default_mutate(dispatch, value)
    }

    pub(super) fn mutate_definition<D, F>(
        dispatch: &mut D,
        buffer: &BufferVar,
        remaps: impl Fn(&mut D) -> &mut Self,
        mut mutate: F,
    ) -> Result<BufferVar>
    where
        F: FnMut(&mut D, &PrimExpr) -> Result<PrimExpr>,
    {
        let identity = ObjectIdentity::of(buffer.as_var());
        if let Some(mapped) = remaps(dispatch).0.get(&identity) {
            return Ok(mapped.clone());
        }
        // Child expressions may use earlier buffer definitions. Keep their
        // remaps in the driver and borrow the table only before/after recursion.
        let old_type = buffer.type_annotation();
        let shape = old_type
            .shape
            .iter()
            .map(|expression| mutate(dispatch, &expression))
            .collect::<Result<Vec<_>>>()?;
        let strides = old_type
            .strides
            .iter()
            .map(|expression| mutate(dispatch, &expression))
            .collect::<Result<Vec<_>>>()?;
        let elem_offset = mutate(dispatch, &old_type.elem_offset)?;
        let allocated_addr = old_type
            .allocated_addr
            .iter()
            .map(|expression| mutate(dispatch, &expression))
            .collect::<Result<Vec<_>>>()?;
        let layout = mutate_layout(&old_type.layout, |expression| mutate(dispatch, expression))?;
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
        remaps(dispatch).0.insert(identity, mapped.clone());
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
    let Some(tile) = clone_downcast::<TileLayout>(original)? else {
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
    value: &StructuralView,
) -> Result<Option<VisitInterrupt>> {
    if value.as_node::<VarObj>().is_some() {
        return Ok(None);
    }
    if value.as_node::<OpaqueExprObj>().is_some()
        || value.as_node::<GlobalVarObj>().is_some()
        || value.as_node::<OpObj>().is_some()
        || value.as_node::<IntImmObj>().is_some()
        || value.as_node::<FloatImmObj>().is_some()
        || value.as_node::<StringImmObj>().is_some()
    {
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
    if let Some(region) = value.as_node::<TensorRegionObj>() {
        for range in region.region.iter() {
            if let Some(interrupt) = visitor.visit(&range.min)? {
                return Ok(Some(interrupt));
            }
            if let Some(interrupt) = visitor.visit(&range.extent)? {
                return Ok(Some(interrupt));
            }
        }
        return Ok(None);
    }
    if let Some(tuple) = value.as_node::<TupleObj>() {
        for field in tuple.fields.iter() {
            if let Some(interrupt) = visitor.visit(&field)? {
                return Ok(Some(interrupt));
            }
        }
        return Ok(None);
    }
    if let Some(projection) = value.as_node::<TupleGetItemObj>() {
        return visitor.visit(&projection.tuple_value);
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
    macro_rules! visit_binary {
        ($node:ty) => {
            if let Some(binary) = value.as_node::<$node>() {
                if let Some(interrupt) = visitor.visit(&binary.a)? {
                    return Ok(Some(interrupt));
                }
                return visitor.visit(&binary.b);
            }
        };
    }
    visit_binary!(AddObj);
    visit_binary!(SubObj);
    visit_binary!(MulObj);
    visit_binary!(DivObj);
    visit_binary!(ModObj);
    visit_binary!(FloorDivObj);
    visit_binary!(FloorModObj);
    visit_binary!(MinObj);
    visit_binary!(MaxObj);
    visit_binary!(EQObj);
    visit_binary!(NEObj);
    visit_binary!(LTObj);
    visit_binary!(LEObj);
    visit_binary!(GTObj);
    visit_binary!(GEObj);
    visit_binary!(AndObj);
    visit_binary!(OrObj);
    if let Some(cast) = value.as_node::<CastObj>() {
        return visitor.visit(&cast.value);
    }
    if let Some(not) = value.as_node::<NotObj>() {
        return visitor.visit(&not.a);
    }
    if let Some(ramp) = value.as_node::<RampObj>() {
        if let Some(interrupt) = visitor.visit(&ramp.base_)? {
            return Ok(Some(interrupt));
        }
        return visitor.visit(&ramp.stride);
    }
    if let Some(broadcast) = value.as_node::<BroadcastObj>() {
        return visitor.visit(&broadcast.value);
    }
    if let Some(shuffle) = value.as_node::<ShuffleObj>() {
        for index in shuffle.indices.iter() {
            if let Some(interrupt) = visitor.visit(&index)? {
                return Ok(Some(interrupt));
            }
        }
        for vector in shuffle.vectors.iter() {
            if let Some(interrupt) = visitor.visit(&vector)? {
                return Ok(Some(interrupt));
            }
        }
        return Ok(None);
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
    if let Some(return_node) = value.as_node::<ReturnObj>() {
        return visitor.visit(&return_node.value);
    }
    if value.as_node::<BreakObj>().is_some() || value.as_node::<ContinueObj>().is_some() {
        return Ok(None);
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
    if let Some(scope_definition) = value.cast::<ScopeIdDefStmt>() {
        return visit_scope_id_definition(visitor, &scope_definition.def);
    }
    if let Some(tile_call) = value.cast::<TilePrimitiveCall>() {
        for argument in tile_call.args.iter() {
            if let Some(interrupt) = visit_tile_value(visitor, &argument)? {
                return Ok(Some(interrupt));
            }
        }
        for (_, configured) in tile_call.config.iter() {
            if let Some(interrupt) = visit_tile_value(visitor, &configured)? {
                return Ok(Some(interrupt));
            }
        }
        return Ok(None);
    }

    visitor.visit_children()
}

fn visit_scope_id_definition<State>(
    visitor: &mut VisitContext<'_, State>,
    definition: &ScopeIdDef,
) -> Result<Option<VisitInterrupt>> {
    if let Some(extents) = &definition.extents {
        for extent in extents.iter() {
            if let Some(interrupt) = visitor.visit(&extent)? {
                return Ok(Some(interrupt));
            }
        }
    }
    if let Some(extents) = &definition.preferred_extents {
        for extent in extents.iter() {
            if let Some(interrupt) = visitor.visit(&extent)? {
                return Ok(Some(interrupt));
            }
        }
    }
    Ok(None)
}

fn visit_tile_value<State>(
    visitor: &mut VisitContext<'_, State>,
    value: &Any,
) -> Result<Option<VisitInterrupt>> {
    if let Some(region) = value.try_as::<TensorRegion>() {
        for range in region.region.iter() {
            if let Some(interrupt) = visitor.visit(&range.min)? {
                return Ok(Some(interrupt));
            }
            if let Some(interrupt) = visitor.visit(&range.extent)? {
                return Ok(Some(interrupt));
            }
        }
        return Ok(None);
    }
    if value.try_as::<Var>().is_some_and(|var| is_buffer_var(&var)) {
        return Ok(None);
    }
    if let Some(expression) = value.try_as::<PrimExpr>() {
        return visitor.visit(&expression);
    }
    if let Some(statement) = value.try_as::<Stmt>() {
        return visitor.visit(&statement);
    }
    if let Some(array) = value.try_as::<Array<Any>>() {
        for item in array.iter() {
            if let Some(interrupt) = visit_tile_value(visitor, &item)? {
                return Ok(Some(interrupt));
            }
        }
    }
    Ok(None)
}

pub(super) fn visit_buffer_definition<State>(
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
        if let Some(tile) = clone_downcast::<TileLayout>(layout)? {
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
    value: &StructuralView,
) -> Result<Any> {
    if let Some(expression) = value.cast::<Expr>() {
        return mutate_expr_default(dispatch, mutator, expression).map(Into::into);
    }
    if let Some(statement) = value.cast::<Stmt>() {
        return mutate_stmt_default(dispatch, mutator, statement).map(Into::into);
    }
    mutator.default_mutate(dispatch, value)
}

pub(super) fn mutate_expr_default<D: MutateDispatch>(
    dispatch: &mut D,
    mutator: &mut Mutator,
    value: Expr,
) -> Result<Expr> {
    if value.as_node::<VarObj>().is_some() {
        return Ok(value);
    }
    if let Some(load) = clone_downcast::<TensorLoad>(&value)? {
        let indices: Array<PrimExpr> = mutator.mutate(dispatch, &load.indices)?.try_into()?;
        if array_same_as(&indices, &load.indices) {
            return Ok(value);
        }
        return Ok(load.copy_with(load.source.clone(), indices).into());
    }
    if let Some(region) = clone_downcast::<TensorRegion>(&value)? {
        let ranges = region
            .region
            .iter()
            .map(|range| mutate_range(dispatch, mutator, &range))
            .collect::<Result<Vec<_>>>()?;
        let ranges = Array::new(ranges);
        if array_same_as(&ranges, &region.region) {
            return Ok(value);
        }
        return Ok(TensorRegion::from_complete_fields(
            region.span.clone(),
            region.ty.clone(),
            region.source.clone(),
            ranges,
        )
        .into());
    }
    if let Some(tuple) = clone_downcast::<Tuple>(&value)? {
        let fields: Array<Expr> = mutator.mutate(dispatch, &tuple.fields)?.try_into()?;
        if array_same_as(&fields, &tuple.fields) {
            return Ok(value);
        }
        return Ok(tuple.copy_with(fields).into());
    }
    if let Some(projection) = clone_downcast::<TupleGetItem>(&value)? {
        let tuple: Expr = mutator
            .mutate(dispatch, &projection.tuple_value)?
            .try_into()?;
        if tuple.same_as(&projection.tuple_value) {
            return Ok(value);
        }
        return Ok(projection.copy_with(tuple)?.into());
    }
    if let Some(call) = clone_downcast::<Call>(&value)? {
        let op = if is_opaque_expr(&call.op) {
            mutator.mutate(dispatch, &call.op)?.try_into()?
        } else {
            call.op.clone()
        };
        let args: Array<Expr> = mutator.mutate(dispatch, &call.args)?.try_into()?;
        if op.same_as(&call.op) && array_same_as(&args, &call.args) {
            return Ok(value);
        }
        let result_type = if call.op.same_as(&get_operator("tirx.buffer_data")?) {
            let source = args.get(0)?;
            let buffer = BufferVar::try_from(&source)?;
            let buffer_type = buffer.buffer_type();
            PointerType::new(
                buffer_type.dtype.clone(),
                buffer_type.storage_scope.as_str(),
            )?
            .into()
        } else {
            call.ty.clone()
        };
        return Ok(call.copy_with(result_type, op, args).into());
    }
    if let Some(let_expr) = clone_downcast::<Let>(&value)? {
        let bound_value: PrimExpr = mutator.mutate(dispatch, &let_expr.value)?.try_into()?;
        let body: PrimExpr = mutator.mutate(dispatch, &let_expr.body)?.try_into()?;
        if bound_value.same_as(&let_expr.value) && body.same_as(&let_expr.body) {
            return Ok(value);
        }
        return Ok(let_expr
            .copy_with(let_expr.var.clone(), bound_value, body)
            .into());
    }
    if value.as_node::<OpaqueExprObj>().is_some()
        || value.as_node::<GlobalVarObj>().is_some()
        || value.as_node::<OpObj>().is_some()
        || value.as_node::<IntImmObj>().is_some()
        || value.as_node::<FloatImmObj>().is_some()
        || value.as_node::<StringImmObj>().is_some()
    {
        return Ok(value);
    }
    if let Some(ramp) = clone_downcast::<Ramp>(&value)? {
        let base: PrimExpr = mutator.mutate(dispatch, &ramp.base_)?.try_into()?;
        let stride: PrimExpr = mutator.mutate(dispatch, &ramp.stride)?.try_into()?;
        let lanes: PrimExpr = mutator.mutate(dispatch, &ramp.lanes)?.try_into()?;
        if base.same_as(&ramp.base_) && stride.same_as(&ramp.stride) && lanes.same_as(&ramp.lanes) {
            return Ok(value);
        }
        return Ok(Ramp::new(base, stride, lanes)?.into());
    }
    if let Some(broadcast) = clone_downcast::<Broadcast>(&value)? {
        let broadcast_value: PrimExpr = mutator.mutate(dispatch, &broadcast.value)?.try_into()?;
        let lanes: PrimExpr = mutator.mutate(dispatch, &broadcast.lanes)?.try_into()?;
        if broadcast_value.same_as(&broadcast.value) && lanes.same_as(&broadcast.lanes) {
            return Ok(value);
        }
        return Ok(Broadcast::new(broadcast_value, lanes)?.into());
    }
    if let Some(shuffle) = clone_downcast::<Shuffle>(&value)? {
        let vectors: Array<PrimExpr> = mutator.mutate(dispatch, &shuffle.vectors)?.try_into()?;
        let indices: Array<PrimExpr> = mutator.mutate(dispatch, &shuffle.indices)?.try_into()?;
        if array_same_as(&vectors, &shuffle.vectors) && array_same_as(&indices, &shuffle.indices) {
            return Ok(value);
        }
        return Ok(Shuffle::new(vectors, indices)?.into());
    }
    mutator
        .default_mutate(dispatch, &value)
        .and_then(Expr::try_from)
}

pub(super) fn mutate_stmt_default<D: MutateDispatch>(
    dispatch: &mut D,
    mutator: &mut Mutator,
    value: Stmt,
) -> Result<Stmt> {
    if let Some(bind) = clone_downcast::<Bind>(&value)? {
        let bound_value: Expr = mutator.mutate(dispatch, &bind.value)?.try_into()?;
        if bound_value.same_as(&bind.value) {
            return Ok(value);
        }
        return Ok(bind.copy_with(bind.var.clone(), bound_value).into());
    }
    if let Some(attribute) = clone_downcast::<AttrStmt>(&value)? {
        let attr_value: Expr = mutator.mutate(dispatch, &attribute.value)?.try_into()?;
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
    if let Some(loop_node) = clone_downcast::<For>(&value)? {
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
    if let Some(while_node) = clone_downcast::<While>(&value)? {
        let condition: PrimExpr = mutator
            .mutate(dispatch, &while_node.condition)?
            .try_into()?;
        let body: Stmt = mutator.mutate(dispatch, &while_node.body)?.try_into()?;
        if condition.same_as(&while_node.condition) && body.same_as(&while_node.body) {
            return Ok(value);
        }
        return Ok(while_node.copy_with(condition, body).into());
    }
    if let Some(return_node) = clone_downcast::<Return>(&value)? {
        let returned: Expr = mutator.mutate(dispatch, &return_node.value)?.try_into()?;
        if returned.same_as(&return_node.value) {
            return Ok(value);
        }
        return Ok(return_node.copy_with(returned).into());
    }
    if value.as_node::<BreakObj>().is_some() || value.as_node::<ContinueObj>().is_some() {
        return Ok(value);
    }
    if value.as_node::<AllocBufferObj>().is_some() {
        // Buffer-definition recursion requires a pass-specific remap table.
        // A pass that changes buffer metadata supplies its own handlers.
        return Ok(value);
    }
    if let Some(declaration) = clone_downcast::<DeclBuffer>(&value)? {
        let data: Expr = mutator.mutate(dispatch, &declaration.data)?.try_into()?;
        if data.same_as(&declaration.data) {
            return Ok(value);
        }
        return Ok(declaration
            .copy_with(declaration.buffer.clone(), data)
            .into());
    }
    if let Some(store) = clone_downcast::<BufferStore>(&value)? {
        let stored_value: PrimExpr = mutator.mutate(dispatch, &store.value)?.try_into()?;
        let indices: Array<PrimExpr> = mutator.mutate(dispatch, &store.indices)?.try_into()?;
        if stored_value.same_as(&store.value) && array_same_as(&indices, &store.indices) {
            return Ok(value);
        }
        return Ok(store
            .copy_with(store.buffer.clone(), stored_value, indices)
            .into());
    }
    if let Some(conditional) = clone_downcast::<IfThenElse>(&value)? {
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
    if let Some(assertion) = clone_downcast::<AssertStmt>(&value)? {
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
    if let Some(sequence) = clone_downcast::<SeqStmt>(&value)? {
        let statements: Array<Stmt> = mutator.mutate(dispatch, &sequence.seq)?.try_into()?;
        if array_same_as(&statements, &sequence.seq) {
            return sequence.flatten();
        }
        return Stmt::sequence_with_span(statements.iter().collect(), sequence.span.as_ref());
    }
    if let Some(evaluate) = clone_downcast::<Evaluate>(&value)? {
        let evaluated: Expr = mutator.mutate(dispatch, &evaluate.value)?.try_into()?;
        if evaluated.same_as(&evaluate.value) {
            return Ok(value);
        }
        return Ok(evaluate.copy_with(evaluated).into());
    }
    if let Some(scope_statement) = clone_downcast::<ScopeIdDefStmt>(&value)? {
        let definition = &scope_statement.def;
        let old_extents = definition.extents.clone();
        let old_preferred_extents = definition.preferred_extents.clone();
        let extents = mutate_optional_prim_exprs(dispatch, mutator, old_extents.clone())?;
        let preferred_extents =
            mutate_optional_prim_exprs(dispatch, mutator, old_preferred_extents.clone())?;
        if option_array_same_as(&extents, &old_extents)
            && option_array_same_as(&preferred_extents, &old_preferred_extents)
        {
            return Ok(value);
        }
        let definition = ScopeIdDef::new(
            definition.def_ids.iter().collect(),
            extents.map(|values| values.iter().collect()),
            definition.scope,
            preferred_extents.map(|values| values.iter().collect()),
        )?;
        return Ok(ScopeIdDefStmt::new(definition, scope_statement.span.as_ref()).into());
    }
    if let Some(tile_call) = clone_downcast::<TilePrimitiveCall>(&value)? {
        let old_args = tile_call.args.clone();
        let old_config = tile_call.config.clone();
        let (args, args_changed) = mutate_tile_values(dispatch, mutator, &old_args)?;
        let mut config_changed = false;
        let mut config = Vec::with_capacity(old_config.len());
        for (key, configured) in old_config.iter() {
            let (configured, changed) = mutate_tile_value(dispatch, mutator, &configured)?;
            config_changed |= changed;
            config.push((key, configured));
        }
        if !args_changed && !config_changed {
            return Ok(value);
        }
        let config = if config_changed {
            Map::from_iter(config)
        } else {
            old_config
        };
        return Ok(tile_call.copy_with(args, config)?.into());
    }
    mutator
        .default_mutate(dispatch, &value)
        .and_then(Stmt::try_from)
}

fn mutate_tile_values<D: MutateDispatch>(
    dispatch: &mut D,
    mutator: &mut Mutator,
    values: &Array<Any>,
) -> Result<(Array<Any>, bool)> {
    let mut changed = false;
    let mut mapped = Vec::with_capacity(values.len());
    for value in values.iter() {
        let (value, value_changed) = mutate_tile_value(dispatch, mutator, &value)?;
        changed |= value_changed;
        mapped.push(value);
    }
    Ok((
        if changed {
            Array::new(mapped)
        } else {
            values.clone()
        },
        changed,
    ))
}

fn mutate_tile_value<D: MutateDispatch>(
    dispatch: &mut D,
    mutator: &mut Mutator,
    value: &Any,
) -> Result<(Any, bool)> {
    if let Some(region) = value.try_as::<TensorRegion>() {
        let mapped = mutator.mutate(dispatch, &region)?;
        let mapped_region = mapped
            .try_as::<TensorRegion>()
            .ok_or_else(|| value_error("mutating a TensorRegion must return a TensorRegion"))?;
        let changed = !mapped_region.same_as(&region);
        return Ok((mapped, changed));
    }
    if let Some(variable) = value.try_as::<Var>() {
        if is_buffer_var(&variable) {
            let mapped = mutator.mutate(dispatch, &variable)?;
            let mapped_variable = mapped
                .try_as::<Var>()
                .ok_or_else(|| value_error("mutating a buffer variable must return a Var"))?;
            BufferVar::try_from(&mapped_variable)?;
            let changed = !mapped_variable.same_as(&variable);
            return Ok((mapped, changed));
        }
    }
    if let Some(expression) = value.try_as::<PrimExpr>() {
        let mapped = mutator.mutate(dispatch, &expression)?;
        let mapped_expression = PrimExpr::try_from(mapped.clone())?;
        let changed = !mapped_expression.same_as(&expression);
        return Ok((mapped, changed));
    }
    if let Some(statement) = value.try_as::<Stmt>() {
        let mapped = mutator.mutate(dispatch, &statement)?;
        let mapped_statement = Stmt::try_from(mapped.clone())?;
        let changed = !mapped_statement.same_as(&statement);
        return Ok((mapped, changed));
    }
    if let Some(array) = value.try_as::<Array<Any>>() {
        let (mapped, changed) = mutate_tile_values(dispatch, mutator, &array)?;
        return Ok((Any::from(mapped), changed));
    }
    Ok((value.clone(), false))
}

fn mutate_optional_prim_exprs<D: MutateDispatch>(
    dispatch: &mut D,
    mutator: &mut Mutator,
    values: Option<Array<PrimExpr>>,
) -> Result<Option<Array<PrimExpr>>> {
    values
        .map(|values| mutator.mutate(dispatch, &values)?.try_into())
        .transpose()
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

/// Mutate a buffer region after its buffer definition has been remapped.
pub(super) fn mutate_buffer_region_with_buffer<D: MutateDispatch>(
    dispatch: &mut D,
    mutator: &mut Mutator,
    value: TensorRegion,
    buffer: BufferVar,
) -> Result<TensorRegion> {
    let ranges = value
        .region
        .iter()
        .map(|range| mutate_range(dispatch, mutator, &range))
        .collect::<Result<Vec<_>>>()?;
    let ranges = Array::new(ranges);
    if buffer.same_as(&BufferVar::try_from(value.source.clone())?)
        && array_same_as(&ranges, &value.region)
    {
        return Ok(value);
    }
    Ok(TensorRegion::from_complete_fields(
        value.span.clone(),
        value.ty.clone(),
        buffer.into(),
        ranges,
    ))
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

fn option_array_same_as<T>(lhs: &Option<Array<T>>, rhs: &Option<Array<T>>) -> bool
where
    T: ObjectRefCore + tvm_ffi::AnyCompatible + Clone,
{
    match (lhs, rhs) {
        (Some(lhs), Some(rhs)) => array_same_as(lhs, rhs),
        (None, None) => true,
        _ => false,
    }
}
