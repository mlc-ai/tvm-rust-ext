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
    structural_mutate, structural_visit, Any, Array, DLDataType, DLDataTypeCode, DLDataTypeExt,
    Function, MapValue, Mutator, ObjectIdentity, ObjectRefCast, ObjectRefCore, Result,
    VisitCallbacks, VisitContext, VisitInterrupt, VisitValue,
};

use super::utils::{
    array_same_as, binary_op, cast_prim_expr, get_operator, is_opaque_expr, is_pointer_type,
    mutate_buffer_region_with_buffer, mutate_expr_default, mutate_stmt_default,
    mutate_stmt_expr_default, visit_buffer_definition, visit_stmt_expr_default,
    with_prim_func_body, BufferRemaps,
};
use super::{create_prim_func_pass, Pass};
use crate::ir::TensorRegion;
use crate::ir::{
    Call, CallObj, Expr, FloatImm, PointerType, PointerTypeObj, PrimExpr, PrimType, PrimTypeObj,
    TensorLoad, Type, Var,
};
use crate::prim::{
    Add, Broadcast, Cast, CastObj, Div, Let, Max, Min, Mul, Select, Shuffle, Sub, EQ, GE, GT, LE,
    LT, NE,
};
use crate::target::Target;
use crate::te::CommReducer;
use crate::tirx::{
    AllocBuffer, AttrStmt, Bind, BufferStore, BufferVar, DeclBuffer, PrimFunc, PrimVar, Stmt,
};

/// Promote BF16 computations to float32 while preserving external storage.
pub fn bf16_compute_legalize_prim_func(function: PrimFunc) -> Result<PrimFunc> {
    compute_legalize_prim_func(
        function,
        UnsupportedFloat::BFloat16,
        PrimType::new("float32")?,
    )
}

/// Promote FP8 computations to `promote_dtype` while preserving external storage.
pub fn fp8_compute_legalize_prim_func(function: PrimFunc, promote_dtype: &str) -> Result<PrimFunc> {
    compute_legalize_prim_func(
        function,
        UnsupportedFloat::Float8,
        PrimType::new(promote_dtype)?,
    )
}

/// Build TVM's `tirx.BF16ComputeLegalize` PrimFunc pass in Rust.
pub fn bf16_compute_legalize() -> Result<Pass> {
    create_prim_func_pass(
        "tirx.BF16ComputeLegalize",
        0,
        Vec::new(),
        false,
        bf16_compute_legalize_prim_func,
    )
}

/// Build TVM's `tirx.FP8ComputeLegalize` PrimFunc pass in Rust.
pub fn fp8_compute_legalize(promote_dtype: &str) -> Result<Pass> {
    let promote_dtype = promote_dtype.to_owned();
    create_prim_func_pass(
        "tirx.FP8ComputeLegalize",
        0,
        Vec::new(),
        false,
        move |function| fp8_compute_legalize_prim_func(function, &promote_dtype),
    )
}

/// Replace BF16 storage with equal-width unsigned-integer storage.
pub fn bf16_storage_legalize_prim_func(function: PrimFunc) -> Result<PrimFunc> {
    storage_legalize_prim_func(function, UnsupportedFloat::BFloat16)
}

/// Replace FP8 storage with equal-width unsigned-integer storage.
pub fn fp8_storage_legalize_prim_func(function: PrimFunc) -> Result<PrimFunc> {
    storage_legalize_prim_func(function, UnsupportedFloat::Float8)
}

/// Build TVM's `tirx.BF16StorageLegalize` PrimFunc pass in Rust.
pub fn bf16_storage_legalize() -> Result<Pass> {
    create_prim_func_pass(
        "tirx.BF16StorageLegalize",
        0,
        Vec::new(),
        false,
        bf16_storage_legalize_prim_func,
    )
}

/// Build TVM's `tirx.FP8StorageLegalize` PrimFunc pass in Rust.
pub fn fp8_storage_legalize() -> Result<Pass> {
    create_prim_func_pass(
        "tirx.FP8StorageLegalize",
        0,
        Vec::new(),
        false,
        fp8_storage_legalize_prim_func,
    )
}

#[derive(Clone, Copy)]
enum UnsupportedFloat {
    BFloat16,
    Float8,
}

impl UnsupportedFloat {
    fn matches(self, ty: &PrimTypeObj) -> bool {
        let code = ty.dtype.code;
        match self {
            Self::BFloat16 => code == DLDataTypeCode::kDLBfloat as u8 && ty.dtype.bits == 16,
            Self::Float8 => matches!(
                code,
                x if x == DLDataTypeCode::kDLFloat8_e3m4 as u8
                    || x == DLDataTypeCode::kDLFloat8_e4m3 as u8
                    || x == DLDataTypeCode::kDLFloat8_e4m3b11fnuz as u8
                    || x == DLDataTypeCode::kDLFloat8_e4m3fn as u8
                    || x == DLDataTypeCode::kDLFloat8_e4m3fnuz as u8
                    || x == DLDataTypeCode::kDLFloat8_e5m2 as u8
                    || x == DLDataTypeCode::kDLFloat8_e5m2fnuz as u8
                    || x == DLDataTypeCode::kDLFloat8_e8m0fnu as u8
            ),
        }
    }

    fn support_function(self) -> &'static str {
        match self {
            Self::BFloat16 => "tvm.support.nvcc.supports_bf16",
            Self::Float8 => "tvm.support.nvcc.supports_fp8",
        }
    }
}

fn compute_legalize_prim_func(
    function: PrimFunc,
    unsupported: UnsupportedFloat,
    promote_type: PrimType,
) -> Result<PrimFunc> {
    if target_has_native_support(&function, unsupported)? {
        return Ok(function);
    }
    let mut planner = VisitCallbacks::new(
        ComputePlan::new(unsupported, promote_type.clone())?,
        (
            plan_allocation,
            plan_declaration,
            plan_store,
            plan_load,
            plan_call,
            plan_variable,
            plan_default,
        ),
    );
    structural_visit(function.body(), &mut planner)?;
    let mut plan = planner.into_state();
    plan.finish();
    let mut legalizer = ComputeLegalizer::new(
        unsupported,
        promote_type,
        plan.buffer_remaps,
        plan.variable_remaps,
    )?;
    let body: Stmt = structural_mutate(function.body().clone(), &mut legalizer)?.try_into()?;
    Ok(with_prim_func_body(function, body))
}

struct ComputePlan {
    unsupported: UnsupportedFloat,
    promote_type: PrimType,
    buffer_remaps: HashMap<ObjectIdentity, BufferVar>,
    variable_remaps: HashMap<ObjectIdentity, Var>,
    opaque_variables: HashSet<ObjectIdentity>,
    buffer_data_operator: Expr,
}

impl ComputePlan {
    fn new(unsupported: UnsupportedFloat, promote_type: PrimType) -> Result<Self> {
        Ok(Self {
            unsupported,
            promote_type,
            buffer_remaps: HashMap::new(),
            variable_remaps: HashMap::new(),
            opaque_variables: HashSet::new(),
            buffer_data_operator: get_operator("tirx.buffer_data")?,
        })
    }

    fn plan_allocation(&mut self, buffer: &BufferVar) -> Result<()> {
        let old_type = buffer.type_annotation();
        if !self.unsupported.matches(&old_type.dtype) {
            return Ok(());
        }
        let new_type = old_type.copy_with(
            old_type.storage_scope.clone(),
            with_lanes(&self.promote_type, old_type.dtype.dtype.lanes)?,
            old_type.shape.clone(),
        );
        let mapped = BufferVar::try_from(buffer.copy_with(buffer.name.clone(), new_type.into()))?;
        self.variable_remaps
            .insert(ObjectIdentity::of(buffer.as_var()), mapped.as_var().clone());
        Ok(())
    }

    fn populate_buffer_remap(&mut self, buffer: &BufferVar) -> Result<()> {
        let identity = ObjectIdentity::of(buffer.as_var());
        if let Some(mapped) = self.variable_remaps.get(&identity) {
            self.buffer_remaps
                .insert(identity, BufferVar::try_from(mapped.clone())?);
        }
        Ok(())
    }

    fn finish(&mut self) {
        for identity in &self.opaque_variables {
            self.variable_remaps.remove(identity);
            self.buffer_remaps.remove(identity);
        }
    }
}

fn plan_allocation(value: AllocBuffer, visitor: &mut VisitContext<'_, ComputePlan>) -> Result<()> {
    visitor.state_mut().plan_allocation(&value.buffer)?;
    visit_buffer_definition(visitor, &value.buffer)?;
    Ok(())
}

fn plan_declaration(value: DeclBuffer, visitor: &mut VisitContext<'_, ComputePlan>) -> Result<()> {
    visitor.visit(&value.data)?;
    visit_buffer_definition(visitor, &value.buffer)?;
    visitor.state_mut().populate_buffer_remap(&value.buffer)
}

fn plan_store(value: BufferStore, visitor: &mut VisitContext<'_, ComputePlan>) -> Result<()> {
    visitor.visit(&value.value)?;
    visitor.visit(&value.indices)?;
    visitor.state_mut().populate_buffer_remap(&value.buffer)
}

fn plan_load(value: TensorLoad, visitor: &mut VisitContext<'_, ComputePlan>) -> Result<()> {
    visitor.visit(&value.indices)?;
    let buffer = BufferVar::try_from(&value.source)?;
    visitor.state_mut().populate_buffer_remap(&buffer)
}

fn plan_call(value: Call, visitor: &mut VisitContext<'_, ComputePlan>) -> Result<()> {
    if value.op.same_as(&visitor.state().buffer_data_operator) && value.args.len() == 1 {
        if let Ok(variable) = value.args.get(0)?.try_cast::<Var>() {
            visitor
                .state_mut()
                .opaque_variables
                .insert(ObjectIdentity::of(&variable));
        }
    }
    if is_opaque_expr(&value.op) {
        visitor.visit(&value.op)?;
    }
    visitor.visit(&value.args)?;
    Ok(())
}

fn plan_variable(value: Var, visitor: &mut VisitContext<'_, ComputePlan>) -> Result<()> {
    if let Ok(buffer) = BufferVar::try_from(&value) {
        visitor.state_mut().populate_buffer_remap(&buffer)?;
    } else if is_pointer_type(&value.ty) {
        visitor
            .state_mut()
            .opaque_variables
            .insert(ObjectIdentity::of(&value));
    }
    Ok(())
}

fn plan_default(
    value: &VisitValue,
    visitor: &mut VisitContext<'_, ComputePlan>,
) -> Result<Option<VisitInterrupt>> {
    visit_stmt_expr_default(visitor, value)
}

struct ComputeLegalizer {
    unsupported: UnsupportedFloat,
    promote_type: PrimType,
    buffer_remaps: HashMap<ObjectIdentity, BufferVar>,
    variable_remaps: HashMap<ObjectIdentity, Var>,
    // StmtExprMutator's definition rewrites are separate from the dtype plan.
    definition_remaps: BufferRemaps,
    conversion: DTypeConverter,
    masked_load_operator: Expr,
    masked_store_operator: Expr,
    reinterpret_operator: Expr,
}

impl ComputeLegalizer {
    fn new(
        unsupported: UnsupportedFloat,
        promote_type: PrimType,
        buffer_remaps: HashMap<ObjectIdentity, BufferVar>,
        variable_remaps: HashMap<ObjectIdentity, Var>,
    ) -> Result<Self> {
        Ok(Self {
            unsupported,
            promote_type,
            buffer_remaps,
            variable_remaps,
            definition_remaps: BufferRemaps::default(),
            conversion: DTypeConverter,
            masked_load_operator: get_operator("tirx.masked_load")?,
            masked_store_operator: get_operator("tirx.masked_store")?,
            reinterpret_operator: get_operator("tirx.reinterpret")?,
        })
    }

    fn remap_buffer(&self, buffer: &BufferVar) -> BufferVar {
        self.buffer_remaps
            .get(&ObjectIdentity::of(buffer.as_var()))
            .cloned()
            .unwrap_or_else(|| buffer.clone())
    }

    fn promote_type_for(&self, ty: &PrimType) -> Result<PrimType> {
        with_lanes(&self.promote_type, ty.dtype.lanes)
    }

    fn promote(&self, value: PrimExpr) -> Result<PrimExpr> {
        let ty = value.type_annotation();
        if !self.unsupported.matches(&ty) {
            return Ok(value);
        }
        if let Some(cast) = value.as_node::<CastObj>() {
            if cast.value.dtype() == self.promote_type_for(&ty)?.dtype {
                return Ok(cast.value.clone());
            }
        }
        self.conversion.convert(value, self.promote_type_for(&ty)?)
    }

    fn cast_from_promoted(&self, value: PrimExpr, target: PrimType) -> Result<PrimExpr> {
        if value.dtype().code != DLDataTypeCode::kDLFloat as u8 {
            return Ok(value);
        }
        if value.dtype() != self.promote_type_for(&value.type_annotation())?.dtype {
            return Err(tvm_ffi::Error::new(
                tvm_ffi::TYPE_ERROR,
                "stored value does not have the configured promoted dtype",
                "",
            ));
        }
        self.conversion.convert(value, target)
    }

    fn mutate_binary<F>(
        &mut self,
        mutator: &mut Mutator,
        original: PrimExpr,
        original_a: &PrimExpr,
        original_b: &PrimExpr,
        rebuild: F,
    ) -> Result<PrimExpr>
    where
        F: FnOnce(PrimExpr, PrimExpr) -> Result<PrimExpr>,
    {
        let a: PrimExpr = mutator.mutate(self, original_a)?.try_into()?;
        let b: PrimExpr = mutator.mutate(self, original_b)?.try_into()?;
        let a = self.promote(a)?;
        let b = self.promote(b)?;
        if a.same_as(original_a) && b.same_as(original_b) {
            return Ok(original);
        }
        rebuild(a, b)
    }

    fn mutate_masked_call(&mut self, value: Call, mutator: &mut Mutator) -> Result<Expr> {
        let is_load = value.op.same_as(&self.masked_load_operator);
        let original = BufferVar::try_from(value.args.get(0)?)?;
        let buffer = self.remap_buffer(&original);
        let mut arguments = vec![Expr::from(buffer.as_var().clone())];
        let mut indices = Vec::new();
        let index_begin = if is_load { 1 } else { 2 };
        let mut stored = None;
        if !is_load {
            let value: PrimExpr = mutator.mutate(self, &value.args.get(1)?)?.try_into()?;
            stored = Some(value);
        }
        for index in index_begin..value.args.len() - 1 {
            let index: PrimExpr = mutator.mutate(self, &value.args.get(index)?)?.try_into()?;
            indices.push(index);
        }
        let predicate: Expr = mutator
            .mutate(self, &value.args.get(value.args.len() - 1)?)?
            .try_into()?;
        let access_type: PrimType =
            TensorLoad::from_buffer(&buffer, indices.iter().cloned().map(Into::into).collect())?
                .ty
                .clone()
                .try_cast()?;
        if is_load {
            arguments.extend(indices.into_iter().map(Into::into));
            arguments.push(predicate);
            return Ok(value
                .copy_with(access_type.into(), value.op.clone(), Array::new(arguments))
                .into());
        }
        let mut stored = stored.expect("masked store has a value");
        if self.unsupported.matches(buffer.dtype()) {
            stored = self.cast_from_promoted(stored, access_type.clone())?;
        }
        if stored.dtype() != access_type.dtype {
            stored = self.conversion.convert(stored, access_type)?;
        }
        arguments.push(stored.into());
        arguments.extend(indices.into_iter().map(Into::into));
        arguments.push(predicate);
        Ok(value
            .copy_with(
                PrimType::void().into(),
                value.op.clone(),
                Array::new(arguments),
            )
            .into())
    }
}

#[tvm_ffi::dispatch(mutate)]
impl ComputeLegalizer {
    fn mutate_variable(&mut self, value: Var) -> Var {
        self.variable_remaps
            .get(&ObjectIdentity::of(&value))
            .cloned()
            .unwrap_or(value)
    }

    fn mutate_cast(&mut self, value: Cast, mutator: &mut Mutator) -> Result<PrimExpr> {
        let inner: PrimExpr = mutator.mutate(self, &value.value)?.try_into()?;
        let inner = self.promote(inner)?;
        let target: PrimType = value.ty.clone().try_cast()?;
        if self.unsupported.matches(&target) {
            return cast_prim_expr(inner, self.promote_type_for(&target)?);
        }
        if inner.same_as(&value.value) {
            return Ok(value.into());
        }
        cast_prim_expr(inner, target)
    }

    fn mutate_select(&mut self, value: Select, mutator: &mut Mutator) -> Result<PrimExpr> {
        let condition: PrimExpr = mutator.mutate(self, &value.condition)?.try_into()?;
        let true_value: PrimExpr = mutator.mutate(self, &value.true_value)?.try_into()?;
        let false_value: PrimExpr = mutator.mutate(self, &value.false_value)?.try_into()?;
        let true_value = self.promote(true_value)?;
        let false_value = self.promote(false_value)?;
        if condition.same_as(&value.condition)
            && true_value.same_as(&value.true_value)
            && false_value.same_as(&value.false_value)
        {
            return Ok(value.into());
        }
        Ok(Select::new(condition, true_value, false_value)?.into())
    }

    fn mutate_broadcast(&mut self, value: Broadcast, mutator: &mut Mutator) -> Result<PrimExpr> {
        let element: PrimExpr = mutator.mutate(self, &value.value)?.try_into()?;
        let element = self.promote(element)?;
        if element.same_as(&value.value) {
            return Ok(value.into());
        }
        Ok(Broadcast::new(element, value.lanes.clone())?.into())
    }

    fn mutate_shuffle(&mut self, value: Shuffle, mutator: &mut Mutator) -> Result<PrimExpr> {
        let mut vectors = Vec::with_capacity(value.vectors.len());
        for vector in value.vectors.iter() {
            let vector: PrimExpr = mutator.mutate(self, &vector)?.try_into()?;
            vectors.push(self.promote(vector)?);
        }
        let vectors = Array::new(vectors);
        if array_same_as(&vectors, &value.vectors) {
            return Ok(value.into());
        }
        Ok(Shuffle::new(vectors, value.indices.clone())?.into())
    }

    fn mutate_call(&mut self, value: Call, mutator: &mut Mutator) -> Result<Expr> {
        if value.op.same_as(&self.masked_load_operator)
            || value.op.same_as(&self.masked_store_operator)
        {
            return self.mutate_masked_call(value, mutator);
        }
        if value.op.same_as(&self.reinterpret_operator) {
            return mutate_expr_default(self, mutator, value.into());
        }
        let Ok(output_type) = value.ty.clone().try_cast::<PrimType>() else {
            return mutate_expr_default(self, mutator, value.into());
        };
        let original_dtype = output_type.dtype;
        let mut arguments = Vec::with_capacity(value.args.len());
        for argument in value.args.iter() {
            let argument: Expr = mutator.mutate(self, &argument)?.try_into()?;
            arguments.push(
                if let Ok(primitive) = PrimExpr::try_from(argument.clone()) {
                    self.promote(primitive)?.into()
                } else {
                    argument
                },
            );
        }
        let arguments = Array::new(arguments);
        let output_type = if self.unsupported.matches(&output_type) {
            self.promote_type_for(&output_type)?
        } else {
            output_type
        };
        if array_same_as(&arguments, &value.args) && output_type.dtype == original_dtype {
            return Ok(value.into());
        }
        Ok(Call::from_complete_fields(
            value.span.clone(),
            output_type.into(),
            value.op.clone(),
            arguments,
            value.attrs.clone(),
            Array::new(Vec::new()),
        )
        .into())
    }

    fn mutate_float(&mut self, value: FloatImm) -> Result<PrimExpr> {
        let ty: PrimType = value.ty.clone().try_cast()?;
        if !self.unsupported.matches(&ty) {
            return Ok(value.into());
        }
        Ok(FloatImm::from_complete_fields(None, self.promote_type.clone(), value.value).into())
    }

    fn mutate_let(&mut self, value: Let, mutator: &mut Mutator) -> Result<PrimExpr> {
        let bound_value = self.promote(value.value.clone())?;
        let variable = if bound_value.dtype() != value.value.dtype() {
            let mapped = value
                .var
                .copy_with(value.var.name.clone(), value.value.type_annotation().into());
            self.variable_remaps
                .insert(ObjectIdentity::of(&value.var), mapped.clone());
            mapped
        } else {
            value.var.clone()
        };
        let body: PrimExpr = mutator.mutate(self, &value.body)?.try_into()?;
        if bound_value.same_as(&value.value)
            && variable.same_as(&value.var)
            && body.same_as(&value.body)
        {
            return Ok(value.into());
        }
        Ok(Let::new(variable, bound_value, body)?.into())
    }

    fn mutate_add(&mut self, value: Add, mutator: &mut Mutator) -> Result<PrimExpr> {
        self.mutate_binary(mutator, value.clone().into(), &value.a, &value.b, |a, b| {
            Ok(Add::new(a, b)?.into())
        })
    }

    fn mutate_subtract(&mut self, value: Sub, mutator: &mut Mutator) -> Result<PrimExpr> {
        self.mutate_binary(mutator, value.clone().into(), &value.a, &value.b, |a, b| {
            Ok(Sub::new(a, b)?.into())
        })
    }

    fn mutate_multiply(&mut self, value: Mul, mutator: &mut Mutator) -> Result<PrimExpr> {
        self.mutate_binary(mutator, value.clone().into(), &value.a, &value.b, |a, b| {
            Ok(Mul::new(a, b)?.into())
        })
    }

    fn mutate_divide(&mut self, value: Div, mutator: &mut Mutator) -> Result<PrimExpr> {
        self.mutate_binary(mutator, value.clone().into(), &value.a, &value.b, |a, b| {
            Ok(Div::new(a, b)?.into())
        })
    }

    fn mutate_minimum(&mut self, value: Min, mutator: &mut Mutator) -> Result<PrimExpr> {
        self.mutate_binary(mutator, value.clone().into(), &value.a, &value.b, |a, b| {
            Ok(Min::new(a, b)?.into())
        })
    }

    fn mutate_maximum(&mut self, value: Max, mutator: &mut Mutator) -> Result<PrimExpr> {
        self.mutate_binary(mutator, value.clone().into(), &value.a, &value.b, |a, b| {
            Ok(Max::new(a, b)?.into())
        })
    }

    fn mutate_less_than(&mut self, value: LT, mutator: &mut Mutator) -> Result<PrimExpr> {
        self.mutate_binary(mutator, value.clone().into(), &value.a, &value.b, |a, b| {
            Ok(LT::new(a, b)?.into())
        })
    }

    fn mutate_less_equal(&mut self, value: LE, mutator: &mut Mutator) -> Result<PrimExpr> {
        self.mutate_binary(mutator, value.clone().into(), &value.a, &value.b, |a, b| {
            Ok(LE::new(a, b)?.into())
        })
    }

    fn mutate_greater_than(&mut self, value: GT, mutator: &mut Mutator) -> Result<PrimExpr> {
        self.mutate_binary(mutator, value.clone().into(), &value.a, &value.b, |a, b| {
            Ok(GT::new(a, b)?.into())
        })
    }

    fn mutate_greater_equal(&mut self, value: GE, mutator: &mut Mutator) -> Result<PrimExpr> {
        self.mutate_binary(mutator, value.clone().into(), &value.a, &value.b, |a, b| {
            Ok(GE::new(a, b)?.into())
        })
    }

    fn mutate_equal(&mut self, value: EQ, mutator: &mut Mutator) -> Result<PrimExpr> {
        self.mutate_binary(mutator, value.clone().into(), &value.a, &value.b, |a, b| {
            Ok(EQ::new(a, b)?.into())
        })
    }

    fn mutate_not_equal(&mut self, value: NE, mutator: &mut Mutator) -> Result<PrimExpr> {
        self.mutate_binary(mutator, value.clone().into(), &value.a, &value.b, |a, b| {
            Ok(NE::new(a, b)?.into())
        })
    }

    fn mutate_binding(&mut self, value: Bind, mutator: &mut Mutator) -> Result<Bind> {
        let Ok(primitive) = PrimExpr::try_from(value.value.clone()) else {
            return mutate_stmt_default(self, mutator, value.into())?.try_cast();
        };
        let original_dtype = primitive.dtype();
        let bound_value = self.promote(primitive)?;
        let variable = if bound_value.dtype() != original_dtype {
            let mapped = value
                .var
                .copy_with(value.var.name.clone(), value.value.ty.clone());
            self.variable_remaps
                .insert(ObjectIdentity::of(&value.var), mapped.clone());
            mapped
        } else {
            value.var.clone()
        };
        if bound_value.same_as(&value.value) && variable.same_as(&value.var) {
            return Ok(value);
        }
        Bind::new(variable, bound_value)
    }

    fn mutate_store(&mut self, value: BufferStore, mutator: &mut Mutator) -> Result<BufferStore> {
        let mut stored: PrimExpr = mutator.mutate(self, &value.value)?.try_into()?;
        let indices: Array<PrimExpr> = mutator.mutate(self, &value.indices)?.try_into()?;
        let buffer = self.remap_buffer(&value.buffer);
        if stored.same_as(&value.value)
            && array_same_as(&indices, &value.indices)
            && buffer.same_as(&value.buffer)
        {
            return Ok(value);
        }
        let storage_type: PrimType =
            TensorLoad::from_buffer(&buffer, indices.iter().map(Into::into).collect())?
                .ty
                .clone()
                .try_cast()?;
        if self.unsupported.matches(buffer.dtype()) {
            stored = self.cast_from_promoted(stored, storage_type.clone())?;
        }
        if stored.dtype() != storage_type.dtype {
            stored = self.conversion.convert(stored, storage_type)?;
        }
        BufferStore::new(buffer, stored, indices.iter().map(Into::into).collect())
    }

    fn mutate_attribute(&mut self, value: AttrStmt, mutator: &mut Mutator) -> Result<AttrStmt> {
        let mutated: AttrStmt = mutate_stmt_default(self, mutator, value.into())?.try_cast()?;
        let node = if let Some(mapped) =
            remap_attribute_variable(&mutated.node, &self.buffer_remaps, &self.variable_remaps)
        {
            mapped
        } else if let Some(reducer) = mutated.node.try_as::<CommReducer>() {
            let identities: Array<PrimExpr> = mutator
                .mutate(self, &reducer.identity_element)?
                .try_into()?;
            for index in 0..identities.len() {
                let identity_type = identities.get(index)?.type_annotation();
                for variable in [reducer.lhs.get(index)?, reducer.rhs.get(index)?] {
                    if variable.dtype() != identity_type.dtype {
                        let mapped =
                            variable.copy_with(variable.name.clone(), identity_type.clone().into());
                        self.variable_remaps
                            .insert(ObjectIdentity::of(variable.as_var()), mapped);
                    }
                }
            }
            let results: Array<PrimExpr> = mutator.mutate(self, &reducer.result)?.try_into()?;
            let lhs = remap_primitive_variables(&reducer.lhs, &self.variable_remaps)?;
            let rhs = remap_primitive_variables(&reducer.rhs, &self.variable_remaps)?;
            CommReducer::from_complete_fields(lhs, rhs, results, identities, reducer.span.clone())
                .into()
        } else {
            return Ok(mutated);
        };
        AttrStmt::new(
            node,
            mutated.attr_key.as_str(),
            mutated.value.clone(),
            mutated.body.clone(),
        )
    }

    fn mutate_declaration(
        &mut self,
        value: DeclBuffer,
        mutator: &mut Mutator,
    ) -> Result<DeclBuffer> {
        let data: Expr = mutator.mutate(self, &value.data)?.try_into()?;
        let buffer = self.remap_buffer(&value.buffer);
        if data.same_as(&value.data) && buffer.same_as(&value.buffer) {
            return Ok(value);
        }
        Ok(value.copy_with(buffer, data))
    }

    fn mutate_allocation(
        &mut self,
        value: AllocBuffer,
        mutator: &mut Mutator,
    ) -> Result<AllocBuffer> {
        let buffer = BufferRemaps::mutate_definition(
            self,
            &value.buffer,
            |state| &mut state.definition_remaps,
            |state, expression| mutator.mutate(state, expression)?.try_into(),
        )?;
        let buffer = self.remap_buffer(&buffer);
        if buffer.same_as(&value.buffer) {
            return Ok(value);
        }
        Ok(value.copy_with(buffer))
    }

    fn mutate_load(&mut self, value: TensorLoad, mutator: &mut Mutator) -> Result<PrimExpr> {
        let original = BufferVar::try_from(&value.source)?;
        let defined = self.definition_remaps.use_buffer(&original);
        let indices: Array<PrimExpr> = mutator.mutate(self, &value.indices)?.try_into()?;
        let buffer = self.remap_buffer(&defined);
        if array_same_as(&indices, &value.indices) && buffer.same_as(&original) {
            return Ok(value.into());
        }
        let span = if buffer.same_as(&defined) {
            value.span.as_ref()
        } else {
            None
        };
        TensorLoad::from_buffer_with_span(
            buffer.as_var().clone(),
            indices.iter().map(Into::into).collect(),
            span,
        )
        .map(Into::into)
    }

    fn mutate_buffer_region(
        &mut self,
        value: TensorRegion,
        mutator: &mut Mutator,
    ) -> Result<TensorRegion> {
        let buffer = self
            .definition_remaps
            .use_buffer(&BufferVar::try_from(value.source.clone())?);
        mutate_buffer_region_with_buffer(self, mutator, value, buffer)
    }

    fn mutate_default(&mut self, value: &MapValue, mutator: &mut Mutator) -> Result<Any> {
        mutate_stmt_expr_default(self, mutator, value)
    }
}

struct DTypeConverter;

impl DTypeConverter {
    fn convert(&self, source: PrimExpr, target: PrimType) -> Result<PrimExpr> {
        let source_type = source.type_annotation();
        if source_type.dtype.lanes != target.dtype.lanes {
            return Err(tvm_ffi::Error::new(
                tvm_ffi::TYPE_ERROR,
                "dtype conversion requires matching lane counts",
                "",
            ));
        }
        let source_config = FloatConfig::from_type(&source_type)?;
        let target_config = FloatConfig::from_type(&target)?;
        let exponent_delta = target_config.exponent - source_config.exponent;
        let bias_delta = target_config.bias - source_config.bias;
        let mantissa_delta = target_config.mantissa - source_config.mantissa;
        let source_uint = storage_uint_type(&source_type)?;
        let target_uint = storage_uint_type(&target)?;
        let mut source_bits = reinterpret_value(source_uint.clone(), source)?;

        if mantissa_delta < 0 {
            let shift = i64::from(-mantissa_delta);
            let least_kept = binary_op(
                "tirx.bitwise_and",
                shift_global("tirx.right_shift", source_bits.clone(), shift)?,
                typed_constant(&source_uint, 1)?,
            )?;
            let bias = Add::new(
                least_kept,
                typed_constant(&source_uint, (1_i64 << (shift - 1)) - 1)?,
            )?;
            source_bits = Add::new(source_bits, bias)?.into();
        }

        if exponent_delta == 0 {
            let mut result = if mantissa_delta >= 0 {
                shift_global(
                    "tirx.left_shift",
                    cast_prim_expr(source_bits, target_uint.clone())?,
                    i64::from(mantissa_delta),
                )?
            } else {
                cast_prim_expr(
                    shift_global("tirx.right_shift", source_bits, i64::from(-mantissa_delta))?,
                    target_uint.clone(),
                )?
            };
            if bias_delta > 0 {
                let bias = shift_global(
                    "tirx.left_shift",
                    typed_constant(&target_uint, i64::from(bias_delta))?,
                    i64::from(target_config.mantissa),
                )?;
                result = Add::new(result, bias)?.into();
            } else if bias_delta < 0 {
                let bias = shift_global(
                    "tirx.left_shift",
                    typed_constant(&target_uint, i64::from(-bias_delta))?,
                    i64::from(target_config.mantissa),
                )?;
                result = Sub::new(result, bias)?.into();
            }
            return reinterpret_value(target, result);
        }

        let mantissa = if mantissa_delta >= 0 {
            shift_global(
                "tirx.left_shift",
                cast_prim_expr(source_bits.clone(), target_uint.clone())?,
                i64::from(mantissa_delta),
            )?
        } else {
            cast_prim_expr(
                shift_global(
                    "tirx.right_shift",
                    source_bits.clone(),
                    i64::from(-mantissa_delta),
                )?,
                target_uint.clone(),
            )?
        };
        let mantissa = binary_op(
            "tirx.bitwise_and",
            mantissa,
            typed_constant(&target_uint, (1_i64 << target_config.mantissa) - 1)?,
        )?;
        let exponent_before_delta = shift_global(
            "tirx.right_shift",
            shift_global("tirx.left_shift", source_bits.clone(), 1)?,
            i64::from(source_config.mantissa + 1),
        )?;
        let sign = shift_global(
            "tirx.left_shift",
            cast_prim_expr(
                shift_global(
                    "tirx.right_shift",
                    source_bits,
                    i64::from(source_config.mantissa + source_config.exponent),
                )?,
                target_uint.clone(),
            )?,
            i64::from(target_config.mantissa + target_config.exponent),
        )?;

        let result = if bias_delta >= 0 {
            let exponent = if bias_delta > 0 {
                Add::new(
                    exponent_before_delta,
                    typed_constant(&source_uint, i64::from(bias_delta))?,
                )?
                .into()
            } else {
                exponent_before_delta
            };
            let exponent = shift_global(
                "tirx.left_shift",
                cast_prim_expr(exponent, target_uint)?,
                i64::from(target_config.mantissa),
            )?;
            binary_op(
                "tirx.bitwise_or",
                binary_op("tirx.bitwise_or", mantissa, exponent)?,
                sign,
            )?
        } else {
            let underflows = LT::new(
                exponent_before_delta.clone(),
                typed_constant(&source_uint, i64::from(-bias_delta))?,
            )?;
            let exponent = Sub::new(
                exponent_before_delta,
                typed_constant(&source_uint, i64::from(-bias_delta))?,
            )?;
            let exponent = shift_global(
                "tirx.left_shift",
                cast_prim_expr(exponent.into(), target_uint.clone())?,
                i64::from(target_config.mantissa),
            )?;
            let populated = binary_op(
                "tirx.bitwise_or",
                binary_op("tirx.bitwise_or", mantissa, exponent)?,
                sign,
            )?;
            if_then_else(
                underflows.into(),
                typed_constant(&target_uint, 0)?,
                populated,
            )?
        };
        reinterpret_value(target, result)
    }
}

struct FloatConfig {
    exponent: i32,
    mantissa: i32,
    bias: i32,
}

impl FloatConfig {
    fn from_type(ty: &PrimType) -> Result<Self> {
        let code = ty.dtype.code;
        let bits = ty.dtype.bits;
        let result = if code == DLDataTypeCode::kDLFloat as u8 {
            match bits {
                16 => Self::new(5, 10, 15),
                32 => Self::new(8, 23, 127),
                64 => Self::new(11, 52, 1023),
                _ => return Err(unsupported_float_type(ty)),
            }
        } else if code == DLDataTypeCode::kDLBfloat as u8 && bits == 16 {
            Self::new(8, 7, 127)
        } else if code == DLDataTypeCode::kDLFloat8_e3m4 as u8 {
            Self::new(3, 4, 3)
        } else if matches!(
            code,
            x if x == DLDataTypeCode::kDLFloat8_e4m3 as u8
                || x == DLDataTypeCode::kDLFloat8_e4m3b11fnuz as u8
                || x == DLDataTypeCode::kDLFloat8_e4m3fn as u8
                || x == DLDataTypeCode::kDLFloat8_e4m3fnuz as u8
        ) {
            Self::new(4, 3, 7)
        } else if matches!(
            code,
            x if x == DLDataTypeCode::kDLFloat8_e5m2 as u8
                || x == DLDataTypeCode::kDLFloat8_e5m2fnuz as u8
        ) {
            Self::new(5, 2, 15)
        } else if code == DLDataTypeCode::kDLFloat8_e8m0fnu as u8 {
            Self::new(8, 0, 127)
        } else {
            return Err(unsupported_float_type(ty));
        };
        Ok(result)
    }

    const fn new(exponent: i32, mantissa: i32, bias: i32) -> Self {
        Self {
            exponent,
            mantissa,
            bias,
        }
    }
}

fn unsupported_float_type(ty: &PrimType) -> tvm_ffi::Error {
    tvm_ffi::Error::new(
        tvm_ffi::TYPE_ERROR,
        &format!("unsupported floating-point dtype {}", ty.dtype.to_string()),
        "",
    )
}

fn storage_uint_type(ty: &PrimType) -> Result<PrimType> {
    PrimType::from_dtype(DLDataType {
        code: DLDataTypeCode::kDLUInt as u8,
        bits: ty.dtype.bits,
        lanes: ty.dtype.lanes,
    })
}

fn with_lanes(ty: &PrimType, lanes: u16) -> Result<PrimType> {
    if (lanes as i16) < 0 {
        return Err(tvm_ffi::Error::new(
            tvm_ffi::TYPE_ERROR,
            "compute legalization requires a fixed lane count, not a scalable vector",
            "",
        ));
    }
    PrimType::from_dtype(DLDataType { lanes, ..ty.dtype })
}

fn typed_constant(ty: &PrimType, value: i64) -> Result<PrimExpr> {
    let scalar_type = with_lanes(ty, 1)?;
    let scalar: PrimExpr = crate::ir::IntImm::from_dtype(scalar_type.dtype, value)?.into();
    cast_prim_expr(scalar, ty.clone())
}

fn reinterpret_value(target: PrimType, value: PrimExpr) -> Result<PrimExpr> {
    tvm_ffi::cached_global_func!("tirx.reinterpret")
        .call_tuple((target, value, Option::<crate::ir::Span>::None))?
        .try_into()
}

fn shift_global(name: &str, lhs: PrimExpr, amount: i64) -> Result<PrimExpr> {
    Function::get_global(name)?
        .call_tuple((lhs, amount, Option::<crate::ir::Span>::None))?
        .try_into()
}

fn if_then_else(
    condition: PrimExpr,
    true_value: PrimExpr,
    false_value: PrimExpr,
) -> Result<PrimExpr> {
    tvm_ffi::cached_global_func!("tirx._OpIfThenElse")
        .call_tuple((
            condition,
            true_value,
            false_value,
            Option::<crate::ir::Span>::None,
        ))?
        .try_into()
}

fn remap_primitive_variables(
    variables: &Array<PrimVar>,
    remaps: &HashMap<ObjectIdentity, Var>,
) -> Result<Array<PrimVar>> {
    variables
        .iter()
        .map(|variable| {
            remaps
                .get(&ObjectIdentity::of(variable.as_var()))
                .cloned()
                .map(PrimVar::try_from)
                .unwrap_or_else(|| Ok(variable))
        })
        .collect::<Result<Vec<_>>>()
        .map(Array::new)
}

fn storage_legalize_prim_func(
    function: PrimFunc,
    unsupported: UnsupportedFloat,
) -> Result<PrimFunc> {
    if target_has_native_support(&function, unsupported)? {
        return Ok(function);
    }
    for parameter in function.params.iter() {
        if BufferVar::try_from(&parameter).is_ok() {
            return Err(tvm_ffi::Error::new(
                tvm_ffi::VALUE_ERROR,
                "storage legalization must run after MakePackedAPI",
                "",
            ));
        }
    }

    let mut legalizer = StorageLegalizer::new(unsupported)?;
    let params = function
        .params
        .iter()
        .map(|parameter| legalizer.remap_variable_definition(parameter))
        .collect::<Result<Vec<_>>>()?;
    let body: Stmt = structural_mutate(function.body().clone(), &mut legalizer)?.try_into()?;
    function.copy_with(params, body)
}

struct StorageLegalizer {
    unsupported: UnsupportedFloat,
    variable_remaps: HashMap<ObjectIdentity, Var>,
    buffer_remaps: HashMap<ObjectIdentity, BufferVar>,
    reinterpret_operator: Expr,
    masked_load_operator: Expr,
    masked_store_operator: Expr,
}

impl StorageLegalizer {
    fn new(unsupported: UnsupportedFloat) -> Result<Self> {
        Ok(Self {
            unsupported,
            variable_remaps: HashMap::new(),
            buffer_remaps: HashMap::new(),
            reinterpret_operator: get_operator("tirx.reinterpret")?,
            masked_load_operator: get_operator("tirx.masked_load")?,
            masked_store_operator: get_operator("tirx.masked_store")?,
        })
    }

    fn storage_type(&self, ty: &PrimTypeObj) -> Result<PrimType> {
        PrimType::from_dtype(DLDataType {
            code: DLDataTypeCode::kDLUInt as u8,
            bits: ty.dtype.bits,
            lanes: ty.dtype.lanes,
        })
    }

    fn remap_variable_definition(&mut self, variable: Var) -> Result<Var> {
        let Some(pointer) = variable.ty.as_node::<PointerTypeObj>() else {
            return Ok(variable);
        };
        let Some(element) = pointer.element_type.as_node::<PrimTypeObj>() else {
            return Ok(variable);
        };
        if !self.unsupported.matches(element) {
            return Ok(variable);
        }
        let mapped = Var::with_type(
            variable.name.as_str(),
            PointerType::new(self.storage_type(element)?, pointer.storage_scope.as_str())?,
        );
        self.variable_remaps
            .insert(ObjectIdentity::of(&variable), mapped.clone());
        Ok(mapped)
    }

    fn remap_buffer(&mut self, buffer: &BufferVar, allow_definition: bool) -> Result<BufferVar> {
        let identity = ObjectIdentity::of(buffer.as_var());
        if let Some(mapped) = self.buffer_remaps.get(&identity) {
            return Ok(mapped.clone());
        }
        let mapped = if let Some(variable) = self.variable_remaps.get(&identity) {
            BufferVar::try_from(variable.clone())?
        } else {
            if !allow_definition && self.unsupported.matches(buffer.dtype()) {
                return Err(tvm_ffi::Error::new(
                    tvm_ffi::VALUE_ERROR,
                    &format!(
                        "cannot find storage remap for buffer {}",
                        buffer.name.as_str()
                    ),
                    "",
                ));
            }
            buffer.clone()
        };
        self.buffer_remaps.insert(identity, mapped.clone());
        Ok(mapped)
    }

    fn force_buffer_storage_type(&mut self, buffer: BufferVar) -> Result<BufferVar> {
        let current = self.remap_buffer(&buffer, true)?;
        if !self.unsupported.matches(current.dtype()) {
            return Ok(current);
        }
        let old_type = current.type_annotation();
        let new_type = old_type.copy_with(
            old_type.storage_scope.clone(),
            self.storage_type(&old_type.dtype)?,
            old_type.shape.clone(),
        );
        let mapped = BufferVar::try_from(current.copy_with(current.name.clone(), new_type.into()))?;
        self.variable_remaps
            .insert(ObjectIdentity::of(buffer.as_var()), mapped.as_var().clone());
        self.buffer_remaps
            .insert(ObjectIdentity::of(buffer.as_var()), mapped.clone());
        Ok(mapped)
    }

    fn reinterpret(&self, target: PrimType, value: PrimExpr) -> Result<PrimExpr> {
        tvm_ffi::cached_global_func!("tirx.reinterpret")
            .call_tuple((target, value, Option::<crate::ir::Span>::None))?
            .try_into()
    }

    fn change_to_uint(&self, value: PrimExpr) -> Result<PrimExpr> {
        if !self.unsupported.matches(&value.type_annotation()) {
            return Ok(value);
        }
        let Some(call) = value.as_node::<CallObj>() else {
            return Ok(value);
        };
        if !call.op.same_as(&self.reinterpret_operator) || call.args.len() != 1 {
            return Ok(value);
        }
        let source: PrimExpr = call.args.get(0)?.try_into()?;
        self.reinterpret(self.storage_type(&value.type_annotation())?, source)
    }
}

#[tvm_ffi::dispatch(mutate)]
impl StorageLegalizer {
    fn mutate_variable(&mut self, value: Var) -> Var {
        self.variable_remaps
            .get(&ObjectIdentity::of(&value))
            .cloned()
            .unwrap_or(value)
    }

    fn mutate_allocation(&mut self, value: AllocBuffer) -> Result<AllocBuffer> {
        let buffer = self.force_buffer_storage_type(value.buffer.clone())?;
        if buffer.same_as(&value.buffer) {
            return Ok(value);
        }
        Ok(value.copy_with(buffer))
    }

    fn mutate_declaration(
        &mut self,
        value: DeclBuffer,
        mutator: &mut Mutator,
    ) -> Result<DeclBuffer> {
        let buffer = self.force_buffer_storage_type(value.buffer.clone())?;
        let data: Expr = mutator.mutate(self, &value.data)?.try_into()?;
        if buffer.same_as(&value.buffer) && data.same_as(&value.data) {
            return Ok(value);
        }
        Ok(value.copy_with(buffer, data))
    }

    fn mutate_let(&mut self, value: Let, mutator: &mut Mutator) -> Result<Let> {
        let bound_value: PrimExpr = mutator.mutate(self, &value.value)?.try_into()?;
        let variable = self.remap_variable_definition(value.var.clone())?;
        let body: PrimExpr = mutator.mutate(self, &value.body)?.try_into()?;
        if bound_value.same_as(&value.value)
            && variable.same_as(&value.var)
            && body.same_as(&value.body)
        {
            return Ok(value);
        }
        Ok(value.copy_with(variable, bound_value, body))
    }

    fn mutate_binding(&mut self, value: Bind, mutator: &mut Mutator) -> Result<Bind> {
        let bound_value: Expr = mutator.mutate(self, &value.value)?.try_into()?;
        let variable = self.remap_variable_definition(value.var.clone())?;
        if bound_value.same_as(&value.value) && variable.same_as(&value.var) {
            return Ok(value);
        }
        Ok(value.copy_with(variable, bound_value))
    }

    fn mutate_store(&mut self, value: BufferStore, mutator: &mut Mutator) -> Result<BufferStore> {
        let stored: PrimExpr = mutator.mutate(self, &value.value)?.try_into()?;
        let stored = self.change_to_uint(stored)?;
        let buffer = self.remap_buffer(&value.buffer, false)?;
        let indices: Array<PrimExpr> = mutator.mutate(self, &value.indices)?.try_into()?;
        if buffer.same_as(&value.buffer)
            && stored.same_as(&value.value)
            && array_same_as(&indices, &value.indices)
        {
            return Ok(value);
        }
        Ok(value.copy_with(buffer, stored, indices))
    }

    fn mutate_attribute(&mut self, value: AttrStmt, mutator: &mut Mutator) -> Result<AttrStmt> {
        let value: AttrStmt = mutate_stmt_default(self, mutator, value.into())?.try_cast()?;
        let Some(node) =
            remap_attribute_variable(&value.node, &self.buffer_remaps, &self.variable_remaps)
        else {
            return Ok(value);
        };
        AttrStmt::new(
            node,
            value.attr_key.as_str(),
            value.value.clone(),
            value.body.clone(),
        )
    }

    fn mutate_load(&mut self, value: TensorLoad, mutator: &mut Mutator) -> Result<PrimExpr> {
        let original = BufferVar::try_from(&value.source)?;
        let indices: Array<PrimExpr> = mutator.mutate(self, &value.indices)?.try_into()?;
        let buffer = self.remap_buffer(&original, false)?;
        if buffer.same_as(&original) && array_same_as(&indices, &value.indices) {
            return Ok(value.into());
        }
        TensorLoad::from_buffer_with_span(
            buffer.as_var().clone(),
            indices.iter().map(Into::into).collect(),
            value.span.as_ref(),
        )
        .map(Into::into)
    }

    fn mutate_call(&mut self, value: Call, mutator: &mut Mutator) -> Result<Expr> {
        if value.op.same_as(&self.masked_load_operator)
            || value.op.same_as(&self.masked_store_operator)
        {
            return self.mutate_masked_call(value, mutator);
        }

        if let Ok(pointer) = value.ty.clone().try_cast::<PointerType>() {
            let mutated = mutate_expr_default(self, mutator, value.clone().into())?;
            let Ok(element) = pointer.element_type().clone().try_cast::<PrimType>() else {
                return Ok(mutated);
            };
            if !self.unsupported.matches(&element) {
                return Ok(mutated);
            }
            let call = mutated.try_cast::<Call>()?;
            let pointer = PointerType::new(self.storage_type(&element)?, pointer.storage_scope())?;
            return Ok(call
                .copy_with(pointer.into(), call.op.clone(), call.args.clone())
                .into());
        }

        let Ok(output_type) = value.ty.clone().try_cast::<PrimType>() else {
            return mutate_expr_default(self, mutator, value.into());
        };
        if value.op.same_as(&self.reinterpret_operator) && value.args.len() == 1 {
            let original_source: PrimExpr = value.args.get(0)?.try_into()?;
            let source: PrimExpr = mutator.mutate(self, &original_source)?.try_into()?;
            if source.dtype() == output_type.dtype {
                return Ok(source.into());
            }
            if self.unsupported.matches(&output_type) {
                return self
                    .reinterpret(self.storage_type(&output_type)?, source)
                    .map(Into::into);
            }
            if source.same_as(&original_source) {
                return Ok(value.into());
            }
            return self.reinterpret(output_type, source).map(Into::into);
        }
        mutate_expr_default(self, mutator, value.into())
    }

    fn mutate_default(&mut self, value: &MapValue, mutator: &mut Mutator) -> Result<Any> {
        mutate_stmt_expr_default(self, mutator, value)
    }
}

impl StorageLegalizer {
    fn mutate_masked_call(&mut self, value: Call, mutator: &mut Mutator) -> Result<Expr> {
        let is_load = value.op.same_as(&self.masked_load_operator);
        let original = BufferVar::try_from(value.args.get(0)?)?;
        let buffer = self.remap_buffer(&original, false)?;
        let mut arguments = vec![Expr::from(buffer.as_var().clone())];
        let index_begin = if is_load { 1 } else { 2 };
        if !is_load {
            let original_value: PrimExpr = value.args.get(1)?.try_into()?;
            let stored: PrimExpr = mutator.mutate(self, &original_value)?.try_into()?;
            arguments.push(self.change_to_uint(stored)?.into());
        }
        for index in index_begin..value.args.len() - 1 {
            let index: PrimExpr = mutator.mutate(self, &value.args.get(index)?)?.try_into()?;
            arguments.push(index.into());
        }
        let predicate: Expr = mutator
            .mutate(self, &value.args.get(value.args.len() - 1)?)?
            .try_into()?;
        arguments.push(predicate);
        let ty: Type = if is_load {
            TensorLoad::from_buffer(&buffer, arguments[1..arguments.len() - 1].to_vec())?
                .ty
                .clone()
        } else {
            PrimType::void().into()
        };
        Ok(value
            .copy_with(ty, value.op.clone(), Array::new(arguments))
            .into())
    }
}

fn remap_attribute_variable(
    node: &Any,
    buffers: &HashMap<ObjectIdentity, BufferVar>,
    variables: &HashMap<ObjectIdentity, Var>,
) -> Option<Any> {
    let variable = node.try_as::<Var>()?;
    let identity = ObjectIdentity::of(&variable);
    if BufferVar::try_from(&variable).is_ok() {
        buffers.get(&identity).cloned().map(Into::into)
    } else {
        variables.get(&identity).cloned().map(Into::into)
    }
}

fn target_has_native_support(function: &PrimFunc, unsupported: UnsupportedFloat) -> Result<bool> {
    let Some(target) = function
        .attrs
        .dict
        .get(&tvm_ffi::String::from("target"))?
        .map(Target::try_from)
        .transpose()?
    else {
        return Ok(false);
    };
    if target.kind_name()?.as_str() != "cuda" {
        return Ok(false);
    }
    let Ok(get_version) = Function::get_global("tvm.support.nvcc.get_compute_version") else {
        return Ok(false);
    };
    let Ok(check_support) = Function::get_global(unsupported.support_function()) else {
        return Ok(false);
    };
    let version: tvm_ffi::String = get_version.call_tuple((target,))?.try_into()?;
    check_support.call_tuple((version,))?.try_into()
}
