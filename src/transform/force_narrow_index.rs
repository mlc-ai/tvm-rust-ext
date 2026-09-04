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
    structural_mutate, Any, Array, DLDataType, DLDataTypeCode, DLDataTypeExt, MapValue, Mutator,
    ObjectIdentity, ObjectRefCast, ObjectRefCore, Result,
};

use super::utils::{
    array_same_as, cast_prim_expr, get_operator, is_primitive_type, mutate_expr_default,
    mutate_stmt_expr_default, option_same_as, with_prim_func_body, BufferRemaps,
};
use super::{create_prim_func_pass, Pass};
use crate::ir::{Call, Expr, IntImm, PrimExpr, PrimType, Range, TensorLoad, Var};
use crate::tirx::{
    Add, AllocBuffer, AttrStmt, Bind, BufferStore, BufferVar, Cast, Div, FloorDiv, FloorMod, For,
    IfThenElse, IterVar, Let, Max, Min, Mod, Mul, PrimFunc, Ramp, Select, Stmt, Sub, EQ, GE, GT,
    LE, LT, NE,
};

const THREAD_EXTENT: &str = "thread_extent";
const VIRTUAL_THREAD: &str = "virtual_thread";

/// Force all non-SBlock index expressions in one PrimFunc to signed int32.
pub fn force_narrow_index_to_int32_prim_func(function: PrimFunc) -> Result<PrimFunc> {
    for parameter in function.params.iter() {
        if let Ok(buffer) = BufferVar::try_from(&parameter) {
            let dtype = buffer.dtype().dtype;
            if dtype.code == DLDataTypeCode::kDLInt as u8 && dtype.bits > 32 {
                return Err(tvm_ffi::Error::new(
                    tvm_ffi::TYPE_ERROR,
                    &format!(
                        "buffer parameter {} has dtype {}; ForceNarrowIndexToInt32 only narrows indices",
                        buffer.name.as_str(),
                        dtype.to_string()
                    ),
                    "",
                ));
            }
        }
    }

    let mut narrower = IndexDataTypeNormalizer::new()?;

    // The first traversal discovers every variable that appears in an index
    // context.  Discarding its output matches IndexDataTypeNormalizer's C++
    // pre-pass and makes remapping independent of definition/use order.
    structural_mutate(function.body.clone(), &mut narrower)?;
    narrower.buffer_remaps.clear();
    narrower.iter_var_remap.clear();

    let old_enabled = narrower.enabled;
    narrower.enabled = true;
    let mut params = Vec::with_capacity(function.params.len());
    for parameter in function.params.iter() {
        if let Ok(buffer) = BufferVar::try_from(&parameter) {
            params.push(narrower.mutate_buffer_definition(&buffer)?.as_var().clone());
        } else if parameter
            .ty
            .clone()
            .try_cast::<PrimType>()
            .is_ok_and(|ty| is_signed_integer(ty.dtype))
        {
            params.push(narrower.rewrite_variable(parameter));
        } else {
            params.push(parameter);
        }
    }
    narrower.enabled = old_enabled;

    let body: Stmt = structural_mutate(function.body.clone(), &mut narrower)?.try_into()?;
    let mut result = with_prim_func_body(function, body);
    if !array_same_as(&Array::new(params.clone()), &result.params) {
        result = result.copy_with(params, result.body.clone())?;
    }
    Ok(result)
}

/// Build TVM's `tirx.ForceNarrowIndexToInt32` PrimFunc pass in Rust.
pub fn force_narrow_index_to_int32() -> Result<Pass> {
    create_prim_func_pass(
        // Keep the upstream pass-info name, including its historical alias.
        "tirx.NarrowDataType",
        0,
        Vec::new(),
        false,
        force_narrow_index_to_int32_prim_func,
    )
}

pub(super) struct IndexDataTypeNormalizer {
    target: PrimType,
    selected_types: Option<HashMap<ObjectIdentity, PrimType>>,
    enabled: bool,
    condition: bool,
    var_remap: HashMap<ObjectIdentity, Var>,
    buffer_remaps: BufferRemaps,
    iter_var_remap: HashMap<ObjectIdentity, IterVar>,
    shift_right_operator: Expr,
    shift_left_operator: Expr,
    bitwise_and_operator: Expr,
    bitwise_or_operator: Expr,
    bitwise_xor_operator: Expr,
    pow_operator: Expr,
    clz_operator: Expr,
    if_then_else_operator: Expr,
}

impl IndexDataTypeNormalizer {
    fn new() -> Result<Self> {
        Ok(Self {
            target: PrimType::new("int32")?,
            selected_types: None,
            enabled: false,
            condition: false,
            var_remap: HashMap::new(),
            buffer_remaps: BufferRemaps::default(),
            iter_var_remap: HashMap::new(),
            shift_right_operator: get_operator("tirx.shift_right")?,
            shift_left_operator: get_operator("tirx.shift_left")?,
            bitwise_and_operator: get_operator("tirx.bitwise_and")?,
            bitwise_or_operator: get_operator("tirx.bitwise_or")?,
            bitwise_xor_operator: get_operator("tirx.bitwise_xor")?,
            pow_operator: get_operator("tirx.pow")?,
            clz_operator: get_operator("tirx.clz")?,
            if_then_else_operator: get_operator("tirx.if_then_else")?,
        })
    }

    pub(super) fn from_selected_types(
        target: PrimType,
        selected_types: HashMap<ObjectIdentity, PrimType>,
    ) -> Result<Self> {
        let mut result = Self::new()?;
        result.target = target;
        result.selected_types = Some(selected_types);
        Ok(result)
    }

    fn selected_type<T: ObjectRefCore>(&self, value: &T) -> Option<PrimType> {
        self.selected_types
            .as_ref()
            .and_then(|types| types.get(&ObjectIdentity::of(value)).cloned())
    }

    fn rewrite_variable(&mut self, value: Var) -> Var {
        let identity = ObjectIdentity::of(&value);
        if let Some(mapped) = self.var_remap.get(&identity) {
            return mapped.clone();
        }
        let Ok(ty) = value.ty.clone().try_cast::<PrimType>() else {
            return value;
        };
        let replacement = if self.selected_types.is_some() {
            self.selected_type(&value)
        } else if self.enabled && can_rewrite(ty.dtype) {
            Some(self.target.clone())
        } else {
            None
        };
        if let Some(replacement) = replacement.filter(|replacement| replacement.dtype != ty.dtype) {
            let mapped = value.copy_with(value.name.clone(), replacement.into());
            self.var_remap.insert(identity, mapped.clone());
            mapped
        } else {
            value
        }
    }

    fn mutate_buffer_definition(&mut self, buffer: &BufferVar) -> Result<BufferVar> {
        let old_enabled = self.enabled;
        self.enabled = true;
        let mut remaps = std::mem::take(&mut self.buffer_remaps);
        let result = remaps.mutate_definition(buffer, |expression| {
            structural_mutate(expression.clone(), &mut *self)?.try_into()
        });
        self.buffer_remaps = remaps;
        self.enabled = old_enabled;
        result
    }

    fn mutate_indices(
        &mut self,
        mutator: &mut Mutator,
        values: &Array<PrimExpr>,
    ) -> Result<Array<PrimExpr>> {
        let old_enabled = self.enabled;
        self.enabled = true;
        let result = mutator.mutate(self, values)?.try_into();
        self.enabled = old_enabled;
        result
    }

    fn mutate_condition(&mut self, mutator: &mut Mutator, value: &PrimExpr) -> Result<PrimExpr> {
        let old_condition = self.condition;
        self.condition = true;
        let result = mutator.mutate(self, value)?.try_into();
        self.condition = old_condition;
        result
    }

    fn mutate_binary<T>(
        &mut self,
        mutator: &mut Mutator,
        original_a: &PrimExpr,
        original_b: &PrimExpr,
        original: &T,
        construct: impl FnOnce(PrimExpr, PrimExpr) -> Result<PrimExpr>,
    ) -> Result<PrimExpr>
    where
        T: Clone + Into<PrimExpr>,
    {
        let a: PrimExpr = mutator.mutate(self, original_a)?.try_into()?;
        let b: PrimExpr = mutator.mutate(self, original_b)?.try_into()?;
        if a.same_as(original_a) && b.same_as(original_b) && a.dtype() == b.dtype() {
            return Ok(original.clone().into());
        }
        construct(a, b)
    }

    fn mutate_comparison<T>(
        &mut self,
        mutator: &mut Mutator,
        original_a: &PrimExpr,
        original_b: &PrimExpr,
        original: &T,
        construct: impl FnOnce(PrimExpr, PrimExpr) -> Result<PrimExpr>,
    ) -> Result<PrimExpr>
    where
        T: Clone + Into<PrimExpr>,
    {
        let old_enabled = self.enabled;
        self.enabled = self.condition
            && is_signed_integer(original_a.dtype())
            && is_signed_integer(original_b.dtype());
        let result = self.mutate_binary(mutator, original_a, original_b, original, construct);
        self.enabled = old_enabled;
        result
    }

    fn mutate_thread_iter(&mut self, iteration: &IterVar) -> Result<IterVar> {
        let identity = ObjectIdentity::of(iteration);
        if let Some(mapped) = self.iter_var_remap.get(&identity) {
            return Ok(mapped.clone());
        }
        let variable = self.rewrite_variable(iteration.var()?.as_var().clone());
        let domain = iteration
            .dom()?
            .map(|domain| -> Result<Range> {
                let minimum: PrimExpr =
                    structural_mutate(domain.min.clone(), &mut *self)?.try_into()?;
                let extent: PrimExpr =
                    structural_mutate(domain.extent.clone(), &mut *self)?.try_into()?;
                let ty = variable.ty.clone().try_cast::<PrimType>()?;
                Ok(domain.copy_with(cast_if_needed(minimum, &ty)?, cast_if_needed(extent, &ty)?))
            })
            .transpose()?;
        let mapped = IterVar::with_metadata(
            domain,
            variable,
            iteration.iter_type()?,
            iteration.thread_tag()?.as_str(),
            iteration.span()?.as_ref(),
        )?;
        self.iter_var_remap.insert(identity, mapped.clone());
        Ok(mapped)
    }
}

#[tvm_ffi::dispatch(mutate)]
impl IndexDataTypeNormalizer {
    fn mutate_integer(&mut self, value: IntImm) -> Result<IntImm> {
        let dtype = value.ty.clone().try_cast::<PrimType>()?.dtype;
        let replacement = if let Some(selected) = self.selected_type(&value) {
            self.enabled.then_some(selected)
        } else if self.selected_types.is_none()
            && dtype.code == DLDataTypeCode::kDLInt as u8
            && dtype.bits == 64
        {
            Some(self.target.clone())
        } else {
            None
        };
        if let Some(replacement) = replacement.filter(|replacement| replacement.dtype != dtype) {
            if self.selected_types.is_none() && value.value > i64::from(i32::MAX) {
                return Err(tvm_ffi::Error::new(
                    tvm_ffi::VALUE_ERROR,
                    "int64 index literal does not fit in int32",
                    "",
                ));
            }
            IntImm::from_dtype(replacement.dtype, value.value)
        } else {
            Ok(value)
        }
    }

    fn mutate_variable(&mut self, value: Var) -> Result<Var> {
        if let Ok(buffer) = BufferVar::try_from(&value) {
            return Ok(self.buffer_remaps.use_buffer(&buffer).as_var().clone());
        }
        Ok(self.rewrite_variable(value))
    }

    fn mutate_cast(&mut self, value: Cast, mutator: &mut Mutator) -> Result<PrimExpr> {
        let dtype = value.ty.clone().try_cast::<PrimType>()?.dtype;
        let replacement = if let Some(selected) = self.selected_type(&value) {
            self.enabled.then_some(selected)
        } else if self.selected_types.is_none() && self.enabled && can_rewrite(dtype) {
            Some(self.target.clone())
        } else {
            None
        };
        if let Some(replacement) = replacement {
            let inner: PrimExpr = mutator.mutate(self, &value.value)?.try_into()?;
            if inner.dtype() == replacement.dtype {
                Ok(inner)
            } else {
                Ok(Cast::new(replacement, inner)?.into())
            }
        } else {
            mutate_expr_default(self, mutator, value.into()).and_then(PrimExpr::try_from)
        }
    }

    fn mutate_add(&mut self, value: Add, mutator: &mut Mutator) -> Result<PrimExpr> {
        self.mutate_binary(mutator, &value.a, &value.b, &value, |a, b| {
            Ok(Add::new(a, b)?.into())
        })
    }

    fn mutate_subtract(&mut self, value: Sub, mutator: &mut Mutator) -> Result<PrimExpr> {
        self.mutate_binary(mutator, &value.a, &value.b, &value, |a, b| {
            Ok(Sub::new(a, b)?.into())
        })
    }

    fn mutate_multiply(&mut self, value: Mul, mutator: &mut Mutator) -> Result<PrimExpr> {
        self.mutate_binary(mutator, &value.a, &value.b, &value, |a, b| {
            Ok(Mul::new(a, b)?.into())
        })
    }

    fn mutate_divide(&mut self, value: Div, mutator: &mut Mutator) -> Result<PrimExpr> {
        self.mutate_binary(mutator, &value.a, &value.b, &value, |a, b| {
            Ok(Div::new(a, b)?.into())
        })
    }

    fn mutate_modulo(&mut self, value: Mod, mutator: &mut Mutator) -> Result<PrimExpr> {
        self.mutate_binary(mutator, &value.a, &value.b, &value, |a, b| {
            Ok(Mod::new(a, b)?.into())
        })
    }

    fn mutate_floor_divide(&mut self, value: FloorDiv, mutator: &mut Mutator) -> Result<PrimExpr> {
        self.mutate_binary(mutator, &value.a, &value.b, &value, |a, b| {
            Ok(FloorDiv::new(a, b)?.into())
        })
    }

    fn mutate_floor_modulo(&mut self, value: FloorMod, mutator: &mut Mutator) -> Result<PrimExpr> {
        self.mutate_binary(mutator, &value.a, &value.b, &value, |a, b| {
            Ok(FloorMod::new(a, b)?.into())
        })
    }

    fn mutate_minimum(&mut self, value: Min, mutator: &mut Mutator) -> Result<PrimExpr> {
        self.mutate_binary(mutator, &value.a, &value.b, &value, |a, b| {
            Ok(Min::new(a, b)?.into())
        })
    }

    fn mutate_maximum(&mut self, value: Max, mutator: &mut Mutator) -> Result<PrimExpr> {
        self.mutate_binary(mutator, &value.a, &value.b, &value, |a, b| {
            Ok(Max::new(a, b)?.into())
        })
    }

    fn mutate_equal(&mut self, value: EQ, mutator: &mut Mutator) -> Result<PrimExpr> {
        self.mutate_comparison(mutator, &value.a, &value.b, &value, |a, b| {
            Ok(EQ::new(a, b)?.into())
        })
    }

    fn mutate_not_equal(&mut self, value: NE, mutator: &mut Mutator) -> Result<PrimExpr> {
        self.mutate_comparison(mutator, &value.a, &value.b, &value, |a, b| {
            Ok(NE::new(a, b)?.into())
        })
    }

    fn mutate_less_than(&mut self, value: LT, mutator: &mut Mutator) -> Result<PrimExpr> {
        self.mutate_comparison(mutator, &value.a, &value.b, &value, |a, b| {
            Ok(LT::new(a, b)?.into())
        })
    }

    fn mutate_less_equal(&mut self, value: LE, mutator: &mut Mutator) -> Result<PrimExpr> {
        self.mutate_comparison(mutator, &value.a, &value.b, &value, |a, b| {
            Ok(LE::new(a, b)?.into())
        })
    }

    fn mutate_greater_than(&mut self, value: GT, mutator: &mut Mutator) -> Result<PrimExpr> {
        self.mutate_comparison(mutator, &value.a, &value.b, &value, |a, b| {
            Ok(GT::new(a, b)?.into())
        })
    }

    fn mutate_greater_equal(&mut self, value: GE, mutator: &mut Mutator) -> Result<PrimExpr> {
        self.mutate_comparison(mutator, &value.a, &value.b, &value, |a, b| {
            Ok(GE::new(a, b)?.into())
        })
    }

    fn mutate_ramp(&mut self, value: Ramp, mutator: &mut Mutator) -> Result<PrimExpr> {
        let base: PrimExpr = mutator.mutate(self, &value.base)?.try_into()?;
        let stride: PrimExpr = mutator.mutate(self, &value.stride)?.try_into()?;
        if base.same_as(&value.base)
            && stride.same_as(&value.stride)
            && base.dtype() == stride.dtype()
        {
            return Ok(value.into());
        }
        let bits = base.dtype().bits.max(stride.dtype().bits);
        let dtype = PrimType::from_dtype(DLDataType {
            bits,
            ..base.dtype()
        })?;
        let base = cast_if_needed(base, &dtype)?;
        let stride = cast_if_needed(stride, &dtype)?;
        let result_type = PrimType::from_dtype(DLDataType {
            lanes: value.ty.clone().try_cast::<PrimType>()?.dtype.lanes,
            ..dtype.dtype
        })?;
        Ok(Ramp::from_complete_fields(None, result_type, base, stride, value.lanes.clone()).into())
    }

    fn mutate_select(&mut self, value: Select, mutator: &mut Mutator) -> Result<PrimExpr> {
        let condition = self.mutate_condition(mutator, &value.condition)?;
        let mut true_value: PrimExpr = mutator.mutate(self, &value.true_value)?.try_into()?;
        let mut false_value: PrimExpr = mutator.mutate(self, &value.false_value)?.try_into()?;
        if condition.same_as(&value.condition)
            && true_value.same_as(&value.true_value)
            && false_value.same_as(&value.false_value)
            && true_value.dtype() == false_value.dtype()
        {
            return Ok(value.into());
        }
        let bits = true_value.dtype().bits.max(false_value.dtype().bits);
        let dtype = PrimType::from_dtype(DLDataType {
            bits,
            ..true_value.dtype()
        })?;
        true_value = cast_if_needed(true_value, &dtype)?;
        false_value = cast_if_needed(false_value, &dtype)?;
        Ok(Select::new(condition, true_value, false_value)?.into())
    }

    fn mutate_let(&mut self, value: Let, mutator: &mut Mutator) -> Result<PrimExpr> {
        let bound_value: PrimExpr = mutator.mutate(self, &value.value)?.try_into()?;
        let variable = if bound_value.dtype() != value.var.ty.clone().try_cast::<PrimType>()?.dtype
        {
            let variable = value
                .var
                .copy_with(value.var.name.clone(), bound_value.type_annotation().into());
            self.var_remap
                .insert(ObjectIdentity::of(&value.var), variable.clone());
            variable
        } else {
            value.var.clone()
        };
        let body: PrimExpr = mutator.mutate(self, &value.body)?.try_into()?;
        if bound_value.same_as(&value.value) && body.same_as(&value.body) {
            return Ok(value.into());
        }
        Ok(value.copy_with(variable, bound_value, body).into())
    }

    fn mutate_binding(&mut self, value: Bind, mutator: &mut Mutator) -> Result<Bind> {
        let bound_value: Expr = mutator.mutate(self, &value.value)?.try_into()?;
        let variable = if let Ok(primitive) = PrimExpr::try_from(&bound_value) {
            if primitive.dtype() != value.var.ty.clone().try_cast::<PrimType>()?.dtype {
                let variable = value
                    .var
                    .copy_with(value.var.name.clone(), primitive.type_annotation().into());
                self.var_remap
                    .insert(ObjectIdentity::of(&value.var), variable.clone());
                variable
            } else {
                value.var.clone()
            }
        } else {
            value.var.clone()
        };
        if variable.same_as(&value.var) && bound_value.same_as(&value.value) {
            return Ok(value);
        }
        Ok(value.copy_with(variable, bound_value))
    }

    fn mutate_loop(&mut self, value: For, mutator: &mut Mutator) -> Result<For> {
        let old_enabled = self.enabled;
        self.enabled = true;
        let variable = self.rewrite_variable(value.loop_var.as_var().clone());
        let variable = crate::tirx::PrimVar::try_from(variable)?;
        let minimum: PrimExpr = mutator.mutate(self, &value.min)?.try_into()?;
        let extent: PrimExpr = mutator.mutate(self, &value.extent)?.try_into()?;
        self.enabled = old_enabled;
        let body: Stmt = mutator.mutate(self, &value.body)?.try_into()?;
        if variable.same_as(&value.loop_var)
            && minimum.same_as(&value.min)
            && extent.same_as(&value.extent)
            && body.same_as(&value.body)
        {
            return Ok(value);
        }
        let ty = variable.type_annotation();
        let thread_binding = value
            .thread_binding
            .as_ref()
            .map(|binding| {
                IterVar::with_metadata(
                    binding.dom()?,
                    Var::from_complete_fields(
                        binding.var()?.span.clone(),
                        ty.clone().into(),
                        binding.var()?.name.clone(),
                    ),
                    binding.iter_type()?,
                    binding.thread_tag()?.as_str(),
                    binding.span()?.as_ref(),
                )
            })
            .transpose()?;
        Ok(For::from_complete_fields(
            value.span.clone(),
            variable,
            cast_if_needed(minimum, &ty)?,
            cast_if_needed(extent, &ty)?,
            value.kind,
            body,
            thread_binding,
            value.annotations.clone(),
            value.step.clone(),
        ))
    }

    fn mutate_conditional(
        &mut self,
        value: IfThenElse,
        mutator: &mut Mutator,
    ) -> Result<IfThenElse> {
        let condition = self.mutate_condition(mutator, &value.condition)?;
        let then_case: Stmt = mutator.mutate(self, &value.then_case)?.try_into()?;
        let else_case: Option<Stmt> = mutator.mutate(self, &value.else_case)?.try_into()?;
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

    fn mutate_attribute(&mut self, value: AttrStmt, mutator: &mut Mutator) -> Result<AttrStmt> {
        if !matches!(value.attr_key.as_str(), THREAD_EXTENT | VIRTUAL_THREAD) {
            return super::utils::mutate_stmt_default(self, mutator, value.into())?
                .try_cast::<AttrStmt>();
        }
        let old_enabled = self.enabled;
        self.enabled = true;
        let iteration = IterVar::try_from(value.node.clone())?;
        let iteration = self.mutate_thread_iter(&iteration)?;
        let attr_value: PrimExpr = mutator.mutate(self, &value.value)?.try_into()?;
        let body: Stmt = mutator.mutate(self, &value.body)?.try_into()?;
        self.enabled = old_enabled;
        let ty = iteration.var()?.type_annotation();
        Ok(value.copy_with(
            iteration.into(),
            value.attr_key.clone(),
            cast_if_needed(attr_value, &ty)?,
            body,
        ))
    }

    fn mutate_allocation(&mut self, value: AllocBuffer) -> Result<AllocBuffer> {
        let buffer = self.mutate_buffer_definition(&value.buffer)?;
        if buffer.same_as(&value.buffer) {
            return Ok(value);
        }
        Ok(value.copy_with(buffer))
    }

    fn mutate_declaration(
        &mut self,
        value: crate::tirx::DeclBuffer,
        mutator: &mut Mutator,
    ) -> Result<crate::tirx::DeclBuffer> {
        let data: Expr = mutator.mutate(self, &value.data)?.try_into()?;
        let buffer = self.mutate_buffer_definition(&value.buffer)?;
        if buffer.same_as(&value.buffer) && data.same_as(&value.data) {
            return Ok(value);
        }
        Ok(value.copy_with(buffer, data))
    }

    fn mutate_store(&mut self, value: BufferStore, mutator: &mut Mutator) -> Result<BufferStore> {
        let buffer = self.buffer_remaps.use_buffer(&value.buffer);
        let mut stored_value: PrimExpr = mutator.mutate(self, &value.value)?.try_into()?;
        let buffer_dtype = buffer.dtype().clone();
        if stored_value.dtype() != buffer_dtype.dtype && stored_value.dtype().lanes == 1 {
            stored_value = cast_prim_expr(stored_value, buffer_dtype)?;
        }
        let indices = self.mutate_indices(mutator, &value.indices)?;
        if buffer.same_as(&value.buffer)
            && stored_value.same_as(&value.value)
            && array_same_as(&indices, &value.indices)
        {
            return Ok(value);
        }
        Ok(value.copy_with(buffer, stored_value, indices))
    }

    fn mutate_load(&mut self, value: TensorLoad, mutator: &mut Mutator) -> Result<TensorLoad> {
        let original = BufferVar::try_from(&value.source)?;
        let buffer = self.buffer_remaps.use_buffer(&original);
        let indices = self.mutate_indices(mutator, &value.indices)?;
        if buffer.same_as(&original) && array_same_as(&indices, &value.indices) {
            return Ok(value);
        }
        TensorLoad::from_buffer_with_span(
            buffer.as_var().clone(),
            indices.iter().map(Into::into).collect(),
            value.span.as_ref(),
        )
    }

    fn mutate_call(&mut self, value: Call, mutator: &mut Mutator) -> Result<Expr> {
        if value.op.same_as(&self.if_then_else_operator) && value.args.len() == 3 {
            let condition =
                self.mutate_condition(mutator, &PrimExpr::try_from(value.args.get(0)?)?)?;
            let mut true_value: PrimExpr = mutator.mutate(self, &value.args.get(1)?)?.try_into()?;
            let mut false_value: PrimExpr =
                mutator.mutate(self, &value.args.get(2)?)?.try_into()?;
            let bits = true_value.dtype().bits.max(false_value.dtype().bits);
            let dtype = PrimType::from_dtype(DLDataType {
                bits,
                ..true_value.dtype()
            })?;
            true_value = cast_if_needed(true_value, &dtype)?;
            false_value = cast_if_needed(false_value, &dtype)?;
            return Ok(value
                .copy_with(
                    dtype.into(),
                    value.op.clone(),
                    Array::new(vec![
                        condition.into(),
                        true_value.into(),
                        false_value.into(),
                    ]),
                )
                .into());
        }

        let before_lhs_type = value
            .args
            .get(0)
            .ok()
            .and_then(|argument| PrimExpr::try_from(argument).ok())
            .map(|argument| argument.type_annotation());
        let expression = mutate_expr_default(self, mutator, value.into())?;
        let call = expression.clone().try_cast::<Call>()?;
        if !is_primitive_type(&call.ty) {
            return Ok(expression);
        }
        let is_shift = call.op.same_as(&self.shift_right_operator)
            || call.op.same_as(&self.shift_left_operator);
        let is_binary = is_shift
            || call.op.same_as(&self.bitwise_and_operator)
            || call.op.same_as(&self.bitwise_or_operator)
            || call.op.same_as(&self.bitwise_xor_operator)
            || call.op.same_as(&self.pow_operator);
        let is_clz = call.op.same_as(&self.clz_operator);
        if (!is_binary && !is_clz) || call.args.len() < if is_binary { 2 } else { 1 } {
            return Ok(expression);
        }
        let lhs = PrimExpr::try_from(call.args.get(0)?)?;
        let mut rhs = if is_binary {
            Some(PrimExpr::try_from(call.args.get(1)?)?)
        } else {
            None
        };
        if is_shift
            && before_lhs_type.as_ref().is_some_and(|before| {
                is_signed_integer(before.dtype)
                    && is_signed_integer(lhs.dtype())
                    && before.dtype.bits > lhs.dtype().bits
            })
        {
            let rhs_value = rhs.take().expect("binary call must have a right operand");
            let limit = IntImm::from_dtype(rhs_value.dtype(), i64::from(lhs.dtype().bits) - 1)?;
            let rhs_value: PrimExpr = Min::new(rhs_value, limit)?.into();
            return Ok(Call::new(
                lhs.type_annotation(),
                call.op.clone(),
                vec![lhs.into(), rhs_value.into()],
            )
            .into());
        }
        if is_shift
            || call.op.same_as(&self.bitwise_and_operator)
            || call.op.same_as(&self.bitwise_or_operator)
            || call.op.same_as(&self.bitwise_xor_operator)
            || call.op.same_as(&self.pow_operator)
        {
            return Ok(Call::new(
                lhs.type_annotation(),
                call.op.clone(),
                vec![
                    lhs.into(),
                    rhs.expect("binary call must have a right operand").into(),
                ],
            )
            .into());
        }
        if is_clz {
            if let Some(before) = before_lhs_type {
                let after = lhs.type_annotation();
                if before.dtype.bits != after.dtype.bits {
                    let sub: PrimExpr = Sub::new(
                        call,
                        IntImm::from_dtype(after.dtype, i64::from(after.dtype.bits))?,
                    )?
                    .into();
                    return Ok(Add::new(
                        sub,
                        IntImm::from_dtype(after.dtype, i64::from(before.dtype.bits))?,
                    )?
                    .into());
                }
            }
        }
        Ok(expression)
    }

    fn mutate_default(&mut self, value: &MapValue, mutator: &mut Mutator) -> Result<Any> {
        mutate_stmt_expr_default(self, mutator, value)
    }
}

fn cast_if_needed(value: PrimExpr, target: &PrimType) -> Result<PrimExpr> {
    if value.dtype() == target.dtype {
        Ok(value)
    } else {
        cast_prim_expr(value, target.clone())
    }
}

fn can_rewrite(dtype: DLDataType) -> bool {
    is_signed_integer(dtype) && dtype.bits >= 32
}

fn is_signed_integer(dtype: DLDataType) -> bool {
    dtype.code == DLDataTypeCode::kDLInt as u8
}
