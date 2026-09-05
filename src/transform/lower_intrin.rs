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

use tvm_ffi::{
    Any, Array, DLDataTypeCode, DLDataTypeExt, Function, MapValue, Mutator, ObjectRefCast,
    ObjectRefCore, Result, String,
};

use super::utils::{
    get_operator, int_value, mutate_expr_default, mutate_stmt_default, value_error,
    with_prim_func_body,
};
use super::{create_prim_func_pass_with_context, Pass, PassContext};
use crate::analysis::Analyzer;
use crate::ir::prim::{
    Add, And, Broadcast, BroadcastObj, Cast, CastObj, Div, FloorDiv, FloorDivObj, FloorMod,
    FloorModObj, Let, Max, Mod, MulObj, Or, Select, EQ, GE, LE, LT, NE,
};
use crate::ir::{Call, CallObj, Expr, IntImm, PrimExpr, PrimType, TensorLoad, Var};
use crate::target::Target;
use crate::tirx::{BufferType, BufferVar, DeclBuffer, PrimFunc, Stmt};

/// Lower target-independent intrinsics using the function's target metadata.
pub fn lower_intrin_prim_func(function: PrimFunc) -> Result<PrimFunc> {
    lower_intrin_prim_func_with_config(function, false)
}

/// Build TVM's `tirx.LowerIntrin` pass in Rust.
pub fn lower_intrin() -> Result<Pass> {
    create_prim_func_pass_with_context(
        "tirx.LowerIntrin",
        0,
        Vec::new(),
        false,
        |function, context| {
            let enable_fast_math = pass_config_bool(&context, "tirx.enable_fast_math")?;
            lower_intrin_prim_func_with_config(function, enable_fast_math)
        },
    )
}

fn lower_intrin_prim_func_with_config(
    function: PrimFunc,
    enable_fast_math: bool,
) -> Result<PrimFunc> {
    let target: Target = function
        .attrs
        .dict
        .get(&String::from("target"))?
        .ok_or_else(|| {
            tvm_ffi::Error::new(
                tvm_ffi::VALUE_ERROR,
                "LowerIntrin requires the target attribute",
                "",
            )
        })?
        .try_into()?;
    let mut injecter = IntrinInjecter::new(&target, enable_fast_math)?;
    let body: Stmt =
        tvm_ffi::structural_mutate(function.body.clone(), &mut injecter)?.try_into()?;
    Ok(with_prim_func_body(function, body))
}

struct AccessPtrAlias {
    buffer: BufferVar,
    data: Expr,
}

struct IntrinInjecter {
    analyzer: Analyzer,
    attribute_names: Vec<String>,
    fma_rule: Option<Function>,
    aliases: Vec<AccessPtrAlias>,
    access_ptr_operator: Expr,
    buffer_data_operator: Expr,
    address_of_operator: Expr,
    fma_operator: Expr,
    floor_operator: Expr,
    shift_right_operator: Expr,
    bitwise_and_operator: Expr,
}

impl IntrinInjecter {
    fn new(target: &Target, enable_fast_math: bool) -> Result<Self> {
        let target_kind = target.kind_name()?;
        let export = target.export()?;
        let triple = export
            .get(&String::from("mtriple"))?
            .map(String::try_from)
            .transpose()?
            .unwrap_or_else(|| String::from(""));
        let mut attribute_names = Vec::new();
        if enable_fast_math {
            attribute_names.push(String::from(format!(
                "{}.fastmath.FLowerIntrinsic",
                target_kind.as_str()
            )));
            attribute_names.push(String::from(format!(
                "{}.fastmath.FLegalize",
                target_kind.as_str()
            )));
        }
        attribute_names.push(String::from(format!(
            "{}.FLowerIntrinsic",
            target_kind.as_str()
        )));
        attribute_names.push(String::from(format!("{}.FLegalize", target_kind.as_str())));
        if triple.as_str().contains("aarch64") {
            attribute_names.push(String::from(format!(
                "{}.aarch64.FLowerIntrinsic",
                target_kind.as_str()
            )));
            attribute_names.push(String::from(format!(
                "{}.aarch64.FLegalize",
                target_kind.as_str()
            )));
        }
        attribute_names.push(String::from("default.FLowerIntrinsic"));
        attribute_names.push(String::from("default.FLegalize"));

        let fma_operator = get_operator("tirx.fma")?;
        let fma_rule = first_operator_rule(&fma_operator, &attribute_names)?;
        Ok(Self {
            analyzer: Analyzer::new()?,
            attribute_names,
            fma_rule,
            aliases: Vec::new(),
            access_ptr_operator: get_operator("tirx.tvm_access_ptr")?,
            buffer_data_operator: get_operator("tirx.buffer_data")?,
            address_of_operator: get_operator("tirx.address_of")?,
            fma_operator,
            floor_operator: get_operator("tirx.floor")?,
            shift_right_operator: get_operator("ir.prim.shift_right")?,
            bitwise_and_operator: get_operator("ir.prim.bitwise_and")?,
        })
    }

    fn lower_access_ptr(&mut self, call: &Call) -> Result<Expr> {
        if call.args.len() != 5 {
            return Err(value_error("tvm_access_ptr expects five arguments"));
        }
        let dtype = call.args.get(0)?.try_cast::<PrimExpr>()?.type_annotation();
        let mut offset = call.args.get(2)?.try_cast::<PrimExpr>()?;
        let mut source = call.args.get(1)?;
        while let Some(inner) = source.as_node::<CallObj>() {
            if !inner.op.same_as(&self.access_ptr_operator) {
                break;
            }
            if inner.args.len() != 5 {
                return Err(value_error("nested tvm_access_ptr expects five arguments"));
            }
            let inner_dtype = inner.args.get(0)?.try_cast::<PrimExpr>()?.type_annotation();
            if inner_dtype.dtype != dtype.dtype {
                return Err(value_error(
                    "nested tvm_access_ptr calls must use the same element type",
                ));
            }
            let mut inner_offset = inner.args.get(2)?.try_cast::<PrimExpr>()?;
            if inner_offset.dtype() != offset.dtype() {
                inner_offset = semantic_cast(offset.type_annotation(), inner_offset)?;
            }
            offset = binary_op("tirx._OpAdd", inner_offset, offset)?;
            let inner_source = inner.args.get(1)?;
            source = inner_source;
        }
        if let Some(buffer_data) = source.as_node::<CallObj>() {
            if buffer_data.op.same_as(&self.buffer_data_operator) && buffer_data.args.len() == 1 {
                let buffer_source = buffer_data.args.get(0)?;
                source = buffer_source;
            }
        }
        let source_var = source.clone().try_cast::<Var>().map_err(|_| {
            value_error("tvm_access_ptr expects a buffer variable or nested tvm_access_ptr")
        })?;

        let scalar_dtype = with_lanes(&dtype, 1)?;
        let mut scalar_extent = binary_op("tirx._OpAdd", offset.clone(), int_like(&offset, 1))?;
        if dtype.dtype.lanes != 1 {
            let lanes = i64::from(dtype.dtype.lanes);
            offset = binary_op("tirx._OpMul", offset, int_like(&scalar_extent, lanes))?;
            scalar_extent = binary_op("tirx._OpAdd", offset.clone(), int_like(&offset, lanes))?;
            offset = ramp(
                offset,
                int_like(&scalar_extent, 1),
                int_like(&scalar_extent, lanes),
            )?;
        }

        let mut access_buffer = None;
        let mut storage_scope = String::from("");
        let mut access_data = source;
        if let Ok(buffer) = BufferVar::try_from(&source_var) {
            let buffer_type = buffer.type_annotation();
            if buffer_type.dtype.dtype == scalar_dtype.dtype && buffer_type.shape.len() == 1 {
                access_buffer = Some(buffer);
            } else {
                if with_lanes(&buffer_type.dtype, 1)?.dtype != scalar_dtype.dtype {
                    return Err(value_error(
                        "tvm_access_ptr element type must match the source buffer",
                    ));
                }
                storage_scope = buffer_type.storage_scope.clone();
                access_data = buffer.data()?;
            }
        } else {
            let pointer = source_var.ty.clone().try_cast::<crate::ir::PointerType>()?;
            storage_scope = tvm_ffi::String::from(pointer.storage_scope());
        }
        let access_buffer = if let Some(buffer) = access_buffer {
            buffer
        } else {
            let buffer_type = BufferType::new(
                storage_scope.as_str(),
                scalar_dtype.dtype.to_string().as_str(),
                vec![scalar_extent.into()],
            )?;
            let buffer = buffer_type.new_var(&format!("{}_access", source_var.name.as_str()));
            self.aliases.push(AccessPtrAlias {
                buffer: buffer.clone(),
                data: access_data,
            });
            buffer
        };
        let load = TensorLoad::from_buffer(&access_buffer, vec![offset.into()])?;
        Ok(Call::from_complete_fields(
            call.span.clone(),
            call.ty.clone(),
            self.address_of_operator.clone(),
            Array::new(vec![load.into()]),
            None,
            Array::new(Vec::new()),
        )
        .into())
    }

    fn apply_intrinsic_rule(&mut self, call: &Call, mutator: &mut Mutator) -> Result<Option<Expr>> {
        let Ok(primitive) = PrimExpr::try_from(Expr::from(call.clone())) else {
            return Ok(None);
        };
        for attribute_name in &self.attribute_names {
            let Some(rule) = operator_rule(&call.op, attribute_name)? else {
                continue;
            };
            let lowered: PrimExpr = rule.call_tuple((&primitive,))?.try_into()?;
            if !lowered.same_as(&primitive) {
                return mutator
                    .mutate(self, &lowered)
                    .and_then(Expr::try_from)
                    .map(Some);
            }
        }
        Ok(None)
    }

    fn make_fma(
        &mut self,
        a: &PrimExpr,
        b: &PrimExpr,
        c: &PrimExpr,
        original: &Add,
        mutator: &mut Mutator,
    ) -> Result<PrimExpr> {
        let lhs = swap_broadcast_cast(a)?;
        let rhs = swap_broadcast_cast(b)?;
        if let Some(rule) = &self.fma_rule {
            if original.a.dtype().code == DLDataTypeCode::kDLFloat as u8 {
                let fma = Call::new(
                    original.ty.clone(),
                    self.fma_operator.clone(),
                    vec![lhs.into(), rhs.into(), c.clone().into()],
                );
                let fma = PrimExpr::try_from(Expr::from(fma))?;
                let lowered: PrimExpr = rule.call_tuple((fma,))?.try_into()?;
                return mutator.mutate(self, &lowered)?.try_into();
            }
        }
        if !lhs.same_as(a) || !rhs.same_as(b) {
            let product: PrimExpr = mutator
                .mutate(self, &crate::ir::prim::Mul::new(lhs, rhs)?)?
                .try_into()?;
            let c: PrimExpr = mutator.mutate(self, c)?.try_into()?;
            return Ok(Add::new(product, c)?.into());
        }
        mutate_expr_default(self, mutator, original.clone().into())?.try_into()
    }

    fn try_find_shift_coefficient(&self, a: &PrimExpr, divisor: i64) -> Result<Option<i64>> {
        if divisor <= 0 {
            return Ok(None);
        }
        let bounds = self.analyzer.const_int_bound(a)?;
        if bounds.min_value >= 0 {
            return Ok(None);
        }
        let dtype = a.dtype();
        let max_value = if dtype.code == DLDataTypeCode::kDLUInt as u8 {
            if dtype.bits >= 63 {
                i64::MAX
            } else {
                (1_i64 << dtype.bits) - 1
            }
        } else if dtype.bits >= 64 {
            i64::MAX
        } else {
            (1_i64 << (dtype.bits - 1)) - 1
        };
        let Some(available) = max_value.checked_add(bounds.min_value) else {
            return Ok(None);
        };
        if divisor - 1 > available {
            return Ok(None);
        }
        let coefficient = ((divisor - 1) - bounds.min_value) / divisor;
        if coefficient <= 0 || coefficient > max_value / divisor {
            return Ok(None);
        }
        if bounds.max_value > max_value - divisor * coefficient {
            return Ok(None);
        }
        Ok(Some(coefficient))
    }

    fn can_prove_nonnegative(&self, value: &PrimExpr) -> Result<bool> {
        self.analyzer
            .can_prove(&GE::new(value.clone(), int_like(value, 0))?.into())
    }
}

#[tvm_ffi::dispatch(mutate)]
impl IntrinInjecter {
    fn mutate_statement(&mut self, value: Stmt, mutator: &mut Mutator) -> Result<Stmt> {
        let alias_begin = self.aliases.len();
        let mut mapped = mutate_stmt_default(self, mutator, value)?;
        for alias in self.aliases.drain(alias_begin..).rev() {
            mapped = Stmt::sequence(vec![
                DeclBuffer::new(alias.buffer, alias.data)?.into(),
                mapped,
            ])?;
        }
        Ok(mapped)
    }

    fn mutate_call(&mut self, value: Call, mutator: &mut Mutator) -> Result<Expr> {
        if value.op.same_as(&self.access_ptr_operator) {
            let lowered = self.lower_access_ptr(&value)?;
            return mutator.mutate(self, &lowered)?.try_into();
        }
        if let Some(lowered) = self.apply_intrinsic_rule(&value, mutator)? {
            return Ok(lowered);
        }
        mutate_expr_default(self, mutator, value.into())
    }

    fn mutate_add(&mut self, value: Add, mutator: &mut Mutator) -> Result<PrimExpr> {
        if let Some(product) = value.b.as_node::<MulObj>() {
            return self.make_fma(&product.a, &product.b, &value.a, &value, mutator);
        }
        if let Some(product) = value.a.as_node::<MulObj>() {
            return self.make_fma(&product.a, &product.b, &value.b, &value, mutator);
        }
        mutate_expr_default(self, mutator, value.into())?.try_into()
    }

    fn mutate_floor_divide(&mut self, value: FloorDiv, mutator: &mut Mutator) -> Result<PrimExpr> {
        let original: PrimExpr = value.clone().into();
        let mapped: PrimExpr = mutate_expr_default(self, mutator, value.into())?.try_into()?;
        let Some(mapped_node) = mapped.as_node::<FloorDivObj>() else {
            return Ok(mapped);
        };
        let mapped = mapped_node;
        let dtype = mapped.a.type_annotation();
        if let Some(shift) = constant_power_of_two(&mapped.b) {
            return shift_right(
                &self.shift_right_operator,
                mapped.a.clone(),
                int_like(&mapped.a, i64::from(shift)),
            );
        }
        if self.can_prove_nonnegative(&mapped.b)? {
            if self.can_prove_nonnegative(&mapped.a)? || self.can_prove_nonnegative(&original)? {
                return Ok(Div::new(mapped.a.clone(), mapped.b.clone())?.into());
            }
            if let Some(divisor) = int_value(&mapped.b) {
                if let Some(coefficient) = self.try_find_shift_coefficient(&mapped.a, divisor)? {
                    let shifted =
                        Add::new(mapped.a.clone(), int_like(&mapped.a, divisor * coefficient))?;
                    return Ok(crate::ir::prim::Sub::new(
                        Div::new(shifted, mapped.b.clone())?,
                        int_like(&mapped.a, coefficient),
                    )?
                    .into());
                }
            }
            let quotient: PrimExpr = Div::new(mapped.a.clone(), mapped.b.clone())?.into();
            let remainder: PrimExpr = Mod::new(mapped.a.clone(), mapped.b.clone())?.into();
            if matches!(dtype.dtype.bits, 32 | 64) {
                let correction = shift_right(
                    &self.shift_right_operator,
                    remainder,
                    int_like(&mapped.a, i64::from(dtype.dtype.bits - 1)),
                )?;
                return Ok(Add::new(quotient, correction)?.into());
            }
            return Ok(Select::new(
                GE::new(remainder.clone(), int_like(&remainder, 0))?,
                quotient.clone(),
                crate::ir::prim::Sub::new(quotient, int_like(&mapped.a, 1))?,
            )?
            .into());
        }
        if dtype.dtype.code == DLDataTypeCode::kDLFloat as u8 {
            let division: PrimExpr = Div::new(mapped.a.clone(), mapped.b.clone())?.into();
            let floor = Call::new(dtype, self.floor_operator.clone(), vec![division.into()]);
            return mutator.mutate(self, &floor)?.try_into();
        }
        let remainder = Var::with_type("rmod", dtype.clone());
        let quotient = Var::with_type("rdiv", dtype);
        let zero = int_like(&mapped.a, 0);
        let condition = Or::new(
            And::new(
                GE::new(mapped.b.clone(), zero.clone())?,
                GE::new(remainder.clone(), zero.clone())?,
            )?,
            And::new(
                LT::new(mapped.b.clone(), zero.clone())?,
                LE::new(remainder.clone(), zero)?,
            )?,
        )?;
        let selected = Select::new(
            condition,
            quotient.clone(),
            crate::ir::prim::Sub::new(quotient.clone(), int_like(&mapped.a, 1))?,
        )?;
        Ok(Let::new(
            remainder,
            Mod::new(mapped.a.clone(), mapped.b.clone())?,
            Let::new(
                quotient,
                Div::new(mapped.a.clone(), mapped.b.clone())?,
                selected,
            )?,
        )?
        .into())
    }

    fn mutate_floor_remainder(
        &mut self,
        value: FloorMod,
        mutator: &mut Mutator,
    ) -> Result<PrimExpr> {
        let mapped: PrimExpr = mutate_expr_default(self, mutator, value.into())?.try_into()?;
        let Some(mapped_node) = mapped.as_node::<FloorModObj>() else {
            return Ok(mapped);
        };
        let mapped = mapped_node;
        let dtype = mapped.a.type_annotation();
        if let Some(shift) = constant_power_of_two(&mapped.b) {
            let mask = (1_i64 << shift) - 1;
            return bitwise_and(
                &self.bitwise_and_operator,
                mapped.a.clone(),
                int_like(&mapped.a, mask),
            );
        }
        if self.can_prove_nonnegative(&mapped.b)? {
            if self.can_prove_nonnegative(&mapped.a)? {
                return Ok(Mod::new(mapped.a.clone(), mapped.b.clone())?.into());
            }
            if let Some(divisor) = int_value(&mapped.b) {
                if let Some(coefficient) = self.try_find_shift_coefficient(&mapped.a, divisor)? {
                    return Ok(Mod::new(
                        Add::new(mapped.a.clone(), int_like(&mapped.a, divisor * coefficient))?,
                        mapped.b.clone(),
                    )?
                    .into());
                }
            }
            let remainder: PrimExpr = Mod::new(mapped.a.clone(), mapped.b.clone())?.into();
            if matches!(dtype.dtype.bits, 32 | 64) {
                let sign = shift_right(
                    &self.shift_right_operator,
                    remainder.clone(),
                    int_like(&mapped.a, i64::from(dtype.dtype.bits - 1)),
                )?;
                let correction = bitwise_and(&self.bitwise_and_operator, mapped.b.clone(), sign)?;
                return Ok(Add::new(remainder, correction)?.into());
            }
            return Ok(Select::new(
                GE::new(remainder.clone(), int_like(&remainder, 0))?,
                remainder.clone(),
                Add::new(remainder, mapped.b.clone())?,
            )?
            .into());
        }
        if dtype.dtype.code == DLDataTypeCode::kDLFloat as u8 {
            let division: PrimExpr = Div::new(mapped.a.clone(), mapped.b.clone())?.into();
            let floor: PrimExpr = mutator
                .mutate(
                    self,
                    &Call::new(dtype, self.floor_operator.clone(), vec![division.into()]),
                )?
                .try_into()?;
            return Ok(crate::ir::prim::Sub::new(
                mapped.a.clone(),
                crate::ir::prim::Mul::new(floor, mapped.b.clone())?,
            )?
            .into());
        }
        let remainder = Var::with_type("rmod", dtype);
        let zero = int_like(&mapped.a, 0);
        let condition = Or::new(
            And::new(
                GE::new(mapped.b.clone(), zero.clone())?,
                GE::new(remainder.clone(), zero.clone())?,
            )?,
            And::new(
                LT::new(mapped.b.clone(), zero.clone())?,
                LE::new(remainder.clone(), zero)?,
            )?,
        )?;
        Ok(Let::new(
            remainder.clone(),
            Mod::new(mapped.a.clone(), mapped.b.clone())?,
            Select::new(
                condition,
                remainder.clone(),
                Add::new(remainder, mapped.b.clone())?,
            )?,
        )?
        .into())
    }

    fn mutate_maximum(&mut self, value: Max, mutator: &mut Mutator) -> Result<PrimExpr> {
        if let Some(divide) = value.a.as_node::<FloorDivObj>() {
            if int_value(&value.b).is_some()
                && self.can_prove_nonnegative(&divide.b)?
                && int_value(&value.b).is_some_and(|value| value >= 0)
            {
                let lowered = Max::new(
                    Div::new(divide.a.clone(), divide.b.clone())?,
                    value.b.clone(),
                )?;
                return mutator.mutate(self, &lowered)?.try_into();
            }
        }
        mutate_expr_default(self, mutator, value.into())?.try_into()
    }

    fn mutate_equal(&mut self, value: EQ, mutator: &mut Mutator) -> Result<PrimExpr> {
        if let Some(remainder) = value.a.as_node::<FloorModObj>() {
            if int_value(&value.b) == Some(0) {
                let lowered = EQ::new(
                    Mod::new(remainder.a.clone(), remainder.b.clone())?,
                    value.b.clone(),
                )?;
                return mutator.mutate(self, &lowered)?.try_into();
            }
        }
        mutate_expr_default(self, mutator, value.into())?.try_into()
    }

    fn mutate_not_equal(&mut self, value: NE, mutator: &mut Mutator) -> Result<PrimExpr> {
        if let Some(remainder) = value.a.as_node::<FloorModObj>() {
            if int_value(&value.b) == Some(0) {
                let lowered = NE::new(
                    Mod::new(remainder.a.clone(), remainder.b.clone())?,
                    value.b.clone(),
                )?;
                return mutator.mutate(self, &lowered)?.try_into();
            }
        }
        mutate_expr_default(self, mutator, value.into())?.try_into()
    }

    fn mutate_default(&mut self, value: &MapValue, mutator: &mut Mutator) -> Result<Any> {
        if let Some(expression) = value.cast::<Expr>() {
            return mutate_expr_default(self, mutator, expression).map(Into::into);
        }
        mutator.default_mutate(self)
    }
}

fn first_operator_rule(operator: &Expr, names: &[String]) -> Result<Option<Function>> {
    for name in names {
        if let Some(rule) = operator_rule(operator, name)? {
            return Ok(Some(rule));
        }
    }
    Ok(None)
}

fn operator_rule(operator: &Expr, name: &String) -> Result<Option<Function>> {
    let registered: bool = tvm_ffi::cached_global_func!("ir.OpHasAttr")
        .call_tuple((operator, name))?
        .try_into()?;
    if !registered {
        return Ok(None);
    }
    tvm_ffi::cached_global_func!("ir.OpGetAttr")
        .call_tuple((operator, name))?
        .try_into()
}

fn swap_broadcast_cast(value: &PrimExpr) -> Result<PrimExpr> {
    let Some(broadcast) = value.as_node::<BroadcastObj>() else {
        return Ok(value.clone());
    };
    let Some(cast) = broadcast.value.as_node::<CastObj>() else {
        return Ok(value.clone());
    };
    let cast_dtype = value.dtype();
    let value_type = cast.value.type_annotation();
    let integer_like = matches!(
        cast_dtype.code,
        x if x == DLDataTypeCode::kDLInt as u8 || x == DLDataTypeCode::kDLUInt as u8
    ) && matches!(
        value_type.dtype.code,
        x if x == DLDataTypeCode::kDLInt as u8 || x == DLDataTypeCode::kDLUInt as u8
    );
    if cast_dtype.bits != value_type.dtype.bits * 2
        && !(integer_like && cast_dtype.bits > value_type.dtype.bits)
    {
        return Ok(value.clone());
    }
    let lanes = int_value(&broadcast.lanes)
        .ok_or_else(|| value_error("broadcast(cast(...)) must have a constant lane count"))?;
    let vector_type = with_lanes(
        &value_type,
        u16::try_from(lanes).map_err(|_| {
            value_error("broadcast lane count does not fit the dtype representation")
        })?,
    )?;
    let broadcast = Broadcast::from_complete_fields(
        broadcast.span.clone(),
        vector_type,
        cast.value.clone(),
        broadcast.lanes.clone(),
    );
    Ok(
        Cast::from_complete_fields(cast.span.clone(), value.type_annotation(), broadcast.into())
            .into(),
    )
}

fn ramp(base: PrimExpr, stride: PrimExpr, lanes: PrimExpr) -> Result<PrimExpr> {
    tvm_ffi::cached_global_func!("ir.prim.Ramp")
        .call_tuple((base, stride, lanes, Option::<crate::ir::Span>::None))?
        .try_into()
}

fn semantic_cast(ty: PrimType, value: PrimExpr) -> Result<PrimExpr> {
    tvm_ffi::cached_global_func!("ir.prim.Cast")
        .call_tuple((ty, value, Option::<crate::ir::Span>::None))?
        .try_into()
}

fn binary_op(name: &str, lhs: PrimExpr, rhs: PrimExpr) -> Result<PrimExpr> {
    Function::get_global(name)?
        .call_tuple((lhs, rhs, Option::<crate::ir::Span>::None))?
        .try_into()
}

fn with_lanes(ty: &PrimType, lanes: u16) -> Result<PrimType> {
    let mut dtype = ty.dtype;
    dtype.lanes = lanes;
    PrimType::from_dtype(dtype)
}

fn shift_right(operator: &Expr, lhs: PrimExpr, rhs: PrimExpr) -> Result<PrimExpr> {
    PrimExpr::try_from(Expr::from(Call::new(
        lhs.type_annotation(),
        operator.clone(),
        vec![lhs.into(), rhs.into()],
    )))
}

fn bitwise_and(operator: &Expr, lhs: PrimExpr, rhs: PrimExpr) -> Result<PrimExpr> {
    PrimExpr::try_from(Expr::from(Call::new(
        lhs.type_annotation(),
        operator.clone(),
        vec![lhs.into(), rhs.into()],
    )))
}

fn constant_power_of_two(value: &PrimExpr) -> Option<u32> {
    let value = int_value(value)?;
    (value > 0 && (value as u64).is_power_of_two()).then(|| (value as u64).trailing_zeros())
}

fn int_like(value: &PrimExpr, literal: i64) -> PrimExpr {
    IntImm::from_complete_fields(None, value.type_annotation(), literal).into()
}

fn pass_config_bool(context: &PassContext, key: &str) -> Result<bool> {
    context
        .config()?
        .get(&String::from(key))?
        .map(bool::try_from)
        .transpose()
        .map(|value| value.unwrap_or(false))
}
