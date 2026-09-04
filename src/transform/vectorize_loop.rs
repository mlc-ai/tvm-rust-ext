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
    structural_mutate, Any, Array, DLDataType, Map, MapValue, Mutator, ObjectIdentity,
    ObjectRefCast, ObjectRefCore, Result,
};

use super::utils::{
    array_same_as, get_operator, int_value, mutate_stmt_expr_default, operator_identity,
    option_same_as, value_error, with_prim_func_body,
};
use super::{create_prim_func_pass, Pass};
use crate::analysis::{operator_bool_attr, Analyzer};
use crate::ir::{Call, Expr, IntImm, PrimExpr, PrimType, TensorLoad, Var};
use crate::tirx::{
    Add, And, Bind, Broadcast, BufferStore, Cast, Div, FloorDiv, FloorMod, For, ForKind,
    IfThenElse, Let, Max, Min, Mod, Mul, Not, Or, PrimFunc, PrimVar, Ramp, Select, Shuffle, Stmt,
    Sub, While, EQ, GE, GT, LE, LT, NE,
};

/// Vectorize loops marked with `ForKind::kVectorized`, or turn them into
/// serial loops when vectorization is disabled.
pub fn vectorize_loop_prim_func(function: PrimFunc, enable_vectorize: bool) -> Result<PrimFunc> {
    let body = if enable_vectorize {
        let mut vectorizer = LoopVectorizer;
        structural_mutate(function.body.clone(), &mut vectorizer)?.try_into()?
    } else {
        let mut skipper = VectorizeSkipper;
        structural_mutate(function.body.clone(), &mut skipper)?.try_into()?
    };
    Ok(with_prim_func_body(function, body))
}

/// Build TVM's `tirx.VectorizeLoop` PrimFunc pass in Rust.
pub fn vectorize_loop(enable_vectorize: bool) -> Result<Pass> {
    create_prim_func_pass(
        "tirx.VectorizeLoop",
        0,
        Vec::new(),
        false,
        move |function| vectorize_loop_prim_func(function, enable_vectorize),
    )
}

struct LoopVectorizer;

#[tvm_ffi::dispatch(mutate)]
impl LoopVectorizer {
    fn mutate_loop(&mut self, value: For, mutator: &mut Mutator) -> Result<Stmt> {
        if value.kind != ForKind::kVectorized {
            return super::utils::mutate_stmt_default(self, mutator, value.into());
        }

        if int_value(&value.min) != Some(0) {
            return Err(value_error("a vectorized loop must start at zero"));
        }
        let lanes = int_value(&value.extent).ok_or_else(|| {
            value_error("a fixed-width vectorized loop requires a constant integer extent")
        })?;
        if lanes < 1 {
            return Err(value_error("a vectorized loop extent must be positive"));
        }

        let mut vectorizer =
            Vectorizer::new(value.loop_var.as_var().clone(), value.extent.clone())?;
        structural_mutate(value.body.clone(), &mut vectorizer)?.try_into()
    }

    fn mutate_default(&mut self, value: &MapValue, mutator: &mut Mutator) -> Result<Any> {
        mutate_stmt_expr_default(self, mutator, value)
    }
}

struct VectorizeSkipper;

#[tvm_ffi::dispatch(mutate)]
impl VectorizeSkipper {
    fn mutate_loop(&mut self, value: For, mutator: &mut Mutator) -> Result<For> {
        let mapped =
            super::utils::mutate_stmt_default(self, mutator, value.into())?.try_cast::<For>()?;
        if mapped.kind != ForKind::kVectorized {
            return Ok(mapped);
        }
        Ok(For::from_complete_fields(
            mapped.span.clone(),
            mapped.loop_var.clone(),
            mapped.min.clone(),
            mapped.extent.clone(),
            ForKind::kSerial,
            mapped.body.clone(),
            mapped.thread_binding.clone(),
            mapped.annotations.clone(),
            mapped.step.clone(),
        ))
    }

    fn mutate_default(&mut self, value: &MapValue, mutator: &mut Mutator) -> Result<Any> {
        mutate_stmt_expr_default(self, mutator, value)
    }
}

struct Vectorizer {
    analyzer: Analyzer,
    variable: Var,
    lanes: PrimExpr,
    ramp: PrimExpr,
    need_scalarize: bool,
    let_bindings: HashMap<ObjectIdentity, PrimExpr>,
    if_then_else: ObjectIdentity,
    reinterpret: ObjectIdentity,
    call_llvm_pure_intrin: ObjectIdentity,
}

impl Vectorizer {
    fn new(variable: Var, lanes: PrimExpr) -> Result<Self> {
        let ty = primitive_type(&variable.clone().into())?;
        let zero = IntImm::from_dtype(ty.dtype, 0)?;
        let one = IntImm::from_dtype(ty.dtype, 1)?;
        let ramp = make_ramp(zero.into(), one.into(), lanes.clone(), None)?;
        Ok(Self {
            analyzer: Analyzer::new()?,
            variable,
            lanes,
            ramp,
            need_scalarize: false,
            let_bindings: HashMap::new(),
            if_then_else: operator_identity("tirx.if_then_else")?,
            reinterpret: operator_identity("tirx.reinterpret")?,
            call_llvm_pure_intrin: operator_identity("tirx.call_llvm_pure_intrin")?,
        })
    }

    fn mutate_prim(&mut self, mutator: &mut Mutator, value: &PrimExpr) -> Result<PrimExpr> {
        mutator.mutate(self, value)?.try_into()
    }

    fn mutate_expr(&mut self, mutator: &mut Mutator, value: &Expr) -> Result<Expr> {
        mutator.mutate(self, value)?.try_into()
    }

    fn mutate_statement(&mut self, mutator: &mut Mutator, value: &Stmt) -> Result<Stmt> {
        if self.need_scalarize {
            return Err(value_error(
                "internal vectorizer state leaked across statement boundaries",
            ));
        }
        let mapped = mutator.mutate(self, value)?.try_into()?;
        if self.need_scalarize {
            self.need_scalarize = false;
            self.scalarize(value.clone())
        } else {
            Ok(mapped)
        }
    }

    fn scalarize(&self, statement: Stmt) -> Result<Stmt> {
        let ty = primitive_type(&self.variable.clone().into())?;
        let scalar = Var::with_type(&format!("{}.s", self.variable.name.as_str()), ty.clone());
        let substitutions =
            Map::from_iter(vec![(self.variable.clone(), Expr::from(scalar.clone()))]);
        let body: Stmt = tvm_ffi::cached_global_func!("tirx.Substitute")
            .call_tuple((statement, substitutions))?
            .try_into()?;
        For::new(
            scalar,
            IntImm::from_dtype(ty.dtype, 0)?,
            self.lanes.clone(),
            body,
        )
        .map(Into::into)
    }

    fn binary_vec(
        &mut self,
        mutator: &mut Mutator,
        lhs: &PrimExpr,
        rhs: &PrimExpr,
        operation: &'static str,
    ) -> Result<PrimExpr> {
        let lhs = self.mutate_prim(mutator, lhs)?;
        let rhs = self.mutate_prim(mutator, rhs)?;
        let lanes = lane_count(&lhs).max(lane_count(&rhs));
        let scalable = is_scalable(&lhs) || is_scalable(&rhs);
        semantic_binary(
            operation,
            broadcast_to(lhs, lanes, scalable)?,
            broadcast_to(rhs, lanes, scalable)?,
        )
    }

    fn add_sub_vec(
        &mut self,
        mutator: &mut Mutator,
        lhs: &PrimExpr,
        rhs: &PrimExpr,
        operation: &'static str,
    ) -> Result<PrimExpr> {
        let lhs = self.mutate_prim(mutator, lhs)?;
        let rhs = self.mutate_prim(mutator, rhs)?;
        let lanes = lane_count(&lhs).max(lane_count(&rhs));
        if lanes != 1 {
            if is_scalar(&lhs) {
                if let Ok(ramp) = rhs.clone().try_cast::<Ramp>() {
                    let base = semantic_binary(operation, lhs, ramp.base.clone())?;
                    let zero = IntImm::from_dtype(ramp.stride.dtype(), 0)?;
                    let stride = semantic_binary(operation, zero.into(), ramp.stride.clone())?;
                    return make_ramp(base, stride, ramp.lanes.clone(), None);
                }
            }
            if is_scalar(&rhs) {
                if let Ok(ramp) = lhs.clone().try_cast::<Ramp>() {
                    let base = semantic_binary(operation, ramp.base.clone(), rhs)?;
                    return make_ramp(base, ramp.stride.clone(), ramp.lanes.clone(), None);
                }
            }
        }
        let scalable = is_scalable(&lhs) || is_scalable(&rhs);
        semantic_binary(
            operation,
            broadcast_to(lhs, lanes, scalable)?,
            broadcast_to(rhs, lanes, scalable)?,
        )
    }

    fn mutate_array(
        &mut self,
        mutator: &mut Mutator,
        values: &Array<PrimExpr>,
    ) -> Result<(Array<PrimExpr>, u16)> {
        let mut lanes = 0;
        let mut mapped = Vec::with_capacity(values.len());
        for value in values.iter() {
            let value = self.mutate_prim(mutator, &value)?;
            lanes = lanes.max(lane_count(&value));
            mapped.push(value);
        }
        for value in &mut mapped {
            if lane_count(value) != lanes {
                *value = broadcast_to(value.clone(), lanes, false)?;
            }
        }
        Ok((Array::new(mapped), lanes))
    }

    fn mutate_call_args(
        &mut self,
        mutator: &mut Mutator,
        values: impl Iterator<Item = Expr>,
    ) -> Result<(Vec<Expr>, u16)> {
        let mut lanes = 0;
        let mut mapped = Vec::new();
        for value in values {
            let value = self.mutate_expr(mutator, &value)?;
            if let Ok(primitive) = PrimExpr::try_from(value.clone()) {
                lanes = lanes.max(lane_count(&primitive));
            }
            mapped.push(value);
        }
        for value in &mut mapped {
            let Ok(primitive) = PrimExpr::try_from(value.clone()) else {
                continue;
            };
            if lane_count(&primitive) != lanes {
                *value = broadcast_to(primitive, lanes, false)?.into();
            }
        }
        Ok((mapped, lanes))
    }

    fn mutate_non_vectorizable_call(&mut self, mutator: &mut Mutator, value: Call) -> Result<Expr> {
        let mut arguments = Vec::with_capacity(value.args.len());
        for argument in value.args.iter() {
            let mapped = self.mutate_expr(mutator, &argument)?;
            if PrimExpr::try_from(mapped.clone()).is_ok_and(|primitive| is_vector(&primitive)) {
                self.need_scalarize = true;
                return Ok(value.into());
            }
            arguments.push(mapped);
        }
        let arguments = Array::new(arguments);
        if array_same_as(&arguments, &value.args) {
            return Ok(value.into());
        }
        Ok(value
            .copy_with(value.ty.clone(), value.op.clone(), arguments)
            .into())
    }
}

#[tvm_ffi::dispatch(mutate)]
impl Vectorizer {
    fn mutate_add(&mut self, value: Add, mutator: &mut Mutator) -> Result<PrimExpr> {
        self.add_sub_vec(mutator, &value.a, &value.b, "tirx._OpAdd")
    }

    fn mutate_subtract(&mut self, value: Sub, mutator: &mut Mutator) -> Result<PrimExpr> {
        self.add_sub_vec(mutator, &value.a, &value.b, "tirx._OpSub")
    }

    fn mutate_multiply(&mut self, value: Mul, mutator: &mut Mutator) -> Result<PrimExpr> {
        let lhs = self.mutate_prim(mutator, &value.a)?;
        let rhs = self.mutate_prim(mutator, &value.b)?;
        if !is_vector(&lhs) && !is_vector(&rhs) {
            return semantic_binary("tirx._OpMul", lhs, rhs);
        }
        if is_scalable(&lhs) != is_scalable(&rhs) && is_vector(&lhs) && is_vector(&rhs) {
            return Err(value_error(
                "fixed-length and scalable vectors cannot be mixed in multiplication",
            ));
        }
        if let Ok(ramp) = lhs.clone().try_cast::<Ramp>() {
            if is_scalar(&rhs) && self.is_positive(&rhs)? {
                return make_ramp(
                    semantic_binary("tirx._OpMul", ramp.base.clone(), rhs.clone())?,
                    semantic_binary("tirx._OpMul", ramp.stride.clone(), rhs)?,
                    ramp.lanes.clone(),
                    None,
                );
            }
        }
        if let Ok(ramp) = rhs.clone().try_cast::<Ramp>() {
            if is_scalar(&lhs) && self.is_positive(&lhs)? {
                return make_ramp(
                    semantic_binary("tirx._OpMul", ramp.base.clone(), lhs.clone())?,
                    semantic_binary("tirx._OpMul", ramp.stride.clone(), lhs)?,
                    ramp.lanes.clone(),
                    None,
                );
            }
        }
        let lanes = lane_count(&lhs).max(lane_count(&rhs));
        let scalable = is_scalable(&lhs) || is_scalable(&rhs);
        semantic_binary(
            "tirx._OpMul",
            broadcast_to(lhs, lanes, scalable)?,
            broadcast_to(rhs, lanes, scalable)?,
        )
    }

    fn mutate_divide(&mut self, value: Div, mutator: &mut Mutator) -> Result<PrimExpr> {
        self.binary_vec(mutator, &value.a, &value.b, "tirx._OpDiv")
    }

    fn mutate_modulo(&mut self, value: Mod, mutator: &mut Mutator) -> Result<PrimExpr> {
        self.binary_vec(mutator, &value.a, &value.b, "tirx._OpMod")
    }

    fn mutate_floor_divide(&mut self, value: FloorDiv, mutator: &mut Mutator) -> Result<PrimExpr> {
        self.binary_vec(mutator, &value.a, &value.b, "tirx._OpFloorDiv")
    }

    fn mutate_floor_modulo(&mut self, value: FloorMod, mutator: &mut Mutator) -> Result<PrimExpr> {
        self.binary_vec(mutator, &value.a, &value.b, "tirx._OpFloorMod")
    }

    fn mutate_minimum(&mut self, value: Min, mutator: &mut Mutator) -> Result<PrimExpr> {
        self.binary_vec(mutator, &value.a, &value.b, "tirx._OpMin")
    }

    fn mutate_maximum(&mut self, value: Max, mutator: &mut Mutator) -> Result<PrimExpr> {
        self.binary_vec(mutator, &value.a, &value.b, "tirx._OpMax")
    }

    fn mutate_equal(&mut self, value: EQ, mutator: &mut Mutator) -> Result<PrimExpr> {
        self.binary_vec(mutator, &value.a, &value.b, "tirx._OpEQ")
    }

    fn mutate_not_equal(&mut self, value: NE, mutator: &mut Mutator) -> Result<PrimExpr> {
        self.binary_vec(mutator, &value.a, &value.b, "tirx._OpNE")
    }

    fn mutate_less_than(&mut self, value: LT, mutator: &mut Mutator) -> Result<PrimExpr> {
        self.binary_vec(mutator, &value.a, &value.b, "tirx._OpLT")
    }

    fn mutate_less_equal(&mut self, value: LE, mutator: &mut Mutator) -> Result<PrimExpr> {
        self.binary_vec(mutator, &value.a, &value.b, "tirx._OpLE")
    }

    fn mutate_greater_than(&mut self, value: GT, mutator: &mut Mutator) -> Result<PrimExpr> {
        self.binary_vec(mutator, &value.a, &value.b, "tirx._OpGT")
    }

    fn mutate_greater_equal(&mut self, value: GE, mutator: &mut Mutator) -> Result<PrimExpr> {
        self.binary_vec(mutator, &value.a, &value.b, "tirx._OpGE")
    }

    fn mutate_and(&mut self, value: And, mutator: &mut Mutator) -> Result<PrimExpr> {
        self.binary_vec(mutator, &value.a, &value.b, "tirx._OpAnd")
    }

    fn mutate_or(&mut self, value: Or, mutator: &mut Mutator) -> Result<PrimExpr> {
        self.binary_vec(mutator, &value.a, &value.b, "tirx._OpOr")
    }

    fn mutate_not(&mut self, value: Not, mutator: &mut Mutator) -> Result<PrimExpr> {
        let operand = self.mutate_prim(mutator, &value.a)?;
        if operand.same_as(&value.a) {
            Ok(value.into())
        } else {
            Not::new(operand).map(Into::into)
        }
    }

    fn mutate_ramp(&mut self, value: Ramp, mutator: &mut Mutator) -> Result<PrimExpr> {
        let base = self.mutate_prim(mutator, &value.base)?;
        let stride = self.mutate_prim(mutator, &value.stride)?;
        if is_scalable(&base) || is_scalable(&stride) {
            return Err(value_error(
                "vectorizing an existing scalable Ramp is not supported",
            ));
        }
        if is_vector(&base) && is_scalar(&stride) {
            if let Ok(base_ramp) = base.clone().try_cast::<Ramp>() {
                let new_lanes = int_value(&value.lanes).ok_or_else(|| {
                    value_error("vectorizing a fixed-width Ramp requires constant lanes")
                })?;
                let base_lanes = int_value(&base_ramp.lanes).ok_or_else(|| {
                    value_error("vectorizing over a Ramp requires constant lanes")
                })?;
                let expected = semantic_binary(
                    "tirx._OpMul",
                    stride.clone(),
                    IntImm::from_dtype(stride.dtype(), base_lanes)?.into(),
                )?;
                if self
                    .analyzer
                    .can_prove_equal(&base_ramp.stride, &expected)?
                {
                    return make_ramp(
                        base_ramp.base.clone(),
                        stride,
                        IntImm::from_dtype(value.lanes.dtype(), new_lanes * base_lanes)?.into(),
                        None,
                    );
                }
            }
        }
        let lanes = lane_count(&base).max(lane_count(&stride));
        let base = broadcast_to(base, lanes, false)?;
        let stride = broadcast_to(stride, lanes, false)?;
        let mut ramps = Vec::with_capacity(usize::from(lanes));
        for lane in 0..lanes {
            ramps.push(make_ramp(
                extract_element(base.clone(), i64::from(lane))?,
                extract_element(stride.clone(), i64::from(lane))?,
                value.lanes.clone(),
                None,
            )?);
        }
        concat_vectors(ramps)
    }

    fn mutate_broadcast(&mut self, value: Broadcast, mutator: &mut Mutator) -> Result<PrimExpr> {
        let mapped = self.mutate_prim(mutator, &value.value)?;
        if is_vector(&mapped) {
            self.need_scalarize = true;
            return Ok(value.into());
        }
        if mapped.same_as(&value.value) {
            Ok(value.into())
        } else {
            // Match the native vectorizer: the original scalar value and lane
            // expression are retained when the recursive result remains scalar.
            make_broadcast(
                value.value.clone(),
                value.lanes.clone(),
                value.span.as_ref(),
            )
        }
    }

    fn mutate_select(&mut self, value: Select, mutator: &mut Mutator) -> Result<PrimExpr> {
        let condition = self.mutate_prim(mutator, &value.condition)?;
        let true_value = self.mutate_prim(mutator, &value.true_value)?;
        let false_value = self.mutate_prim(mutator, &value.false_value)?;
        if condition.same_as(&value.condition)
            && true_value.same_as(&value.true_value)
            && false_value.same_as(&value.false_value)
        {
            return Ok(value.into());
        }
        let lanes = lane_count(&condition)
            .max(lane_count(&true_value))
            .max(lane_count(&false_value));
        let scalable =
            is_scalable(&condition) || is_scalable(&true_value) || is_scalable(&false_value);
        Select::new(
            broadcast_to(condition, lanes, scalable)?,
            broadcast_to(true_value, lanes, scalable)?,
            broadcast_to(false_value, lanes, scalable)?,
        )
        .map(Into::into)
    }

    fn mutate_cast(&mut self, value: Cast, mutator: &mut Mutator) -> Result<PrimExpr> {
        let operand = self.mutate_prim(mutator, &value.value)?;
        if operand.same_as(&value.value) {
            return Ok(value.into());
        }
        let result_type = primitive_type(&Expr::from(value.clone()))?;
        Cast::new(with_lanes(&result_type, &operand)?, operand).map(Into::into)
    }

    fn mutate_variable(&mut self, value: Var) -> PrimExpr {
        if value.same_as(&self.variable) {
            return self.ramp.clone();
        }
        self.let_bindings
            .get(&ObjectIdentity::of(&value))
            .cloned()
            .unwrap_or_else(|| {
                PrimExpr::from(PrimVar::try_from(value).expect("a vectorizer Var is primitive"))
            })
    }

    fn mutate_call(&mut self, value: Call, mutator: &mut Mutator) -> Result<Expr> {
        let Ok(return_type) = value.ty.clone().try_cast::<PrimType>() else {
            return self.mutate_non_vectorizable_call(mutator, value);
        };
        let operator = ObjectIdentity::of(&value.op);
        if operator == self.if_then_else && value.args.len() == 3 {
            let condition = self.mutate_prim(mutator, &PrimExpr::try_from(value.args.get(0)?)?)?;
            if is_vector(&condition) {
                self.need_scalarize = true;
                return Ok(value.into());
            }
            let true_value = self.mutate_prim(mutator, &PrimExpr::try_from(value.args.get(1)?)?)?;
            let false_value =
                self.mutate_prim(mutator, &PrimExpr::try_from(value.args.get(2)?)?)?;
            let lanes = lane_count(&true_value).max(lane_count(&false_value));
            let scalable = is_scalable(&true_value) || is_scalable(&false_value);
            return Ok(Call::from_complete_fields(
                value.span.clone(),
                with_lane_count(&return_type, lanes, scalable)?.into(),
                value.op.clone(),
                Array::new(vec![
                    condition.into(),
                    broadcast_to(true_value, lanes, scalable)?.into(),
                    broadcast_to(false_value, lanes, scalable)?.into(),
                ]),
                value.attrs.clone(),
                Array::new(Vec::new()),
            )
            .into());
        }
        if operator == self.reinterpret && value.args.len() == 1 {
            let input = PrimExpr::try_from(value.args.get(0)?)?;
            let mapped = self.mutate_prim(mutator, &input)?;
            if mapped.same_as(&input) {
                return Ok(value.into());
            }
            let lanes = if is_scalable(&mapped) {
                lane_count(&mapped)
            } else if return_type.dtype.code != tvm_ffi::DLDataTypeCode::kDLFloat4_e2m1fn as u8
                && input.dtype().code != tvm_ffi::DLDataTypeCode::kDLFloat4_e2m1fn as u8
            {
                let mapped_ty = mapped.dtype();
                u16::from(mapped_ty.bits) * mapped_ty.lanes / u16::from(return_type.dtype.bits)
            } else {
                lane_count(&mapped)
            };
            return Ok(Call::from_complete_fields(
                value.span.clone(),
                with_lane_count(&return_type, lanes, is_scalable(&mapped))?.into(),
                value.op.clone(),
                Array::new(vec![mapped.into()]),
                value.attrs.clone(),
                Array::new(Vec::new()),
            )
            .into());
        }

        let vectorizable = operator_bool_attr(&value.op, "TVectorizable")?.unwrap_or(false)
            && !is_scalable_type(&return_type);
        if !vectorizable {
            return self.mutate_non_vectorizable_call(mutator, value);
        }

        let (arguments, lanes) = if operator == self.call_llvm_pure_intrin {
            let first = value.args.get(0)?;
            let (mut rest, lanes) = self.mutate_call_args(mutator, value.args.iter().skip(1))?;
            rest.insert(0, first);
            (rest, lanes)
        } else {
            self.mutate_call_args(mutator, value.args.iter())?
        };
        let arguments = Array::new(arguments);
        if array_same_as(&arguments, &value.args) {
            return Ok(value.into());
        }
        Ok(Call::from_complete_fields(
            value.span.clone(),
            with_lane_count(&return_type, lanes, false)?.into(),
            value.op.clone(),
            arguments,
            value.attrs.clone(),
            Array::new(Vec::new()),
        )
        .into())
    }

    fn mutate_load(&mut self, value: TensorLoad, mutator: &mut Mutator) -> Result<PrimExpr> {
        let indices: Array<PrimExpr> = mutator.mutate(self, &value.indices)?.try_into()?;
        if array_same_as(&indices, &value.indices) {
            return Ok(value.into());
        }
        tvm_ffi::cached_global_func!("tirx.BufferLoad")
            .call_tuple((value.source.clone(), indices, value.span.clone()))?
            .try_into()
    }

    fn mutate_let(&mut self, value: Let, mutator: &mut Mutator) -> Result<PrimExpr> {
        let bound_value = self.mutate_prim(mutator, &value.value)?;
        let identity = ObjectIdentity::of(&value.var);
        if lane_count(&bound_value) != lane_count(&value.value) {
            let mapped_var = Var::with_type(value.var.name.as_str(), bound_value.type_annotation());
            self.let_bindings.insert(
                identity,
                PrimExpr::from(PrimVar::try_from(mapped_var.clone())?),
            );
            let body = self.mutate_prim(mutator, &value.body)?;
            Let::new(mapped_var, bound_value, body).map(Into::into)
        } else {
            self.let_bindings.insert(
                identity,
                PrimExpr::from(PrimVar::try_from(value.var.clone())?),
            );
            let body = self.mutate_prim(mutator, &value.body)?;
            if bound_value.same_as(&value.value) && body.same_as(&value.body) {
                Ok(value.into())
            } else {
                Let::new(value.var.clone(), bound_value, body).map(Into::into)
            }
        }
    }

    fn mutate_shuffle(&mut self, value: Shuffle, mutator: &mut Mutator) -> Result<PrimExpr> {
        if value.vectors.len() != 1 || value.indices.len() != 1 {
            return Err(value_error(
                "vectorizing Shuffle with multiple vectors or indices is not supported",
            ));
        }
        let (vectors, _) = self.mutate_array(mutator, &value.vectors)?;
        let (indices, _) = self.mutate_array(mutator, &value.indices)?;
        if array_same_as(&vectors, &value.vectors) && array_same_as(&indices, &value.indices) {
            return Ok(value.into());
        }
        let total_lanes = int_value(&self.lanes).ok_or_else(|| {
            value_error("Shuffle vectorization requires a constant vectorized-loop extent")
        })?;
        let source_lanes = i64::from(lane_count(&value.vectors.get(0)?));
        let new_length = total_lanes / source_lanes;
        if new_length == 1 {
            let ty = primitive_type(&self.variable.clone().into())?;
            let substitutions = Map::from_iter(vec![(
                self.variable.clone(),
                Expr::from(IntImm::from_dtype(ty.dtype, 0)?),
            )]);
            return tvm_ffi::cached_global_func!("tirx.Substitute")
                .call_tuple((value.vectors.get(0)?, substitutions))?
                .try_into();
        }
        // This is the same narrowed re-vectorization used by C++ for the
        // supported one-vector/one-index shuffle forms.
        let previous_ramp = self.ramp.clone();
        let previous_lanes = self.lanes.clone();
        let ty = primitive_type(&self.variable.clone().into())?;
        self.lanes = IntImm::from_dtype(previous_lanes.dtype(), new_length)?.into();
        self.ramp = make_ramp(
            IntImm::from_dtype(ty.dtype, 0)?.into(),
            IntImm::from_dtype(ty.dtype, 2)?.into(),
            self.lanes.clone(),
            None,
        )?;
        let result = self
            .mutate_array(mutator, &value.vectors)
            .and_then(|(vectors, _)| vectors.get(0));
        self.ramp = previous_ramp;
        self.lanes = previous_lanes;
        result
    }

    fn mutate_store(&mut self, value: BufferStore, mutator: &mut Mutator) -> Result<Stmt> {
        let indices: Array<PrimExpr> = mutator.mutate(self, &value.indices)?.try_into()?;
        let stored_value = self.mutate_prim(mutator, &value.value)?;
        if array_same_as(&indices, &value.indices) && stored_value.same_as(&value.value) {
            return Ok(value.into());
        }
        if indices.is_empty() {
            return Err(value_error("a BufferStore requires at least one index"));
        }
        let buffer_lanes = lane_count_type(&value.buffer.dtype().dtype);
        let mut other_index_lanes = buffer_lanes;
        for index in indices.iter().take(indices.len() - 1) {
            if is_scalable(&index) {
                return Err(value_error("only the last buffer index may be scalable"));
            }
            other_index_lanes = other_index_lanes.saturating_mul(lane_count(&index));
        }
        let last = indices.get(indices.len() - 1)?;
        let index_lanes = other_index_lanes.saturating_mul(lane_count(&last));
        let total_lanes = index_lanes.max(lane_count(&stored_value));
        if !total_lanes.is_multiple_of(other_index_lanes) {
            return Err(value_error(
                "vectorized store lanes cannot be represented by its final index",
            ));
        }
        let mut mapped_indices = indices.iter().collect::<Vec<_>>();
        mapped_indices[indices.len() - 1] = broadcast_to(
            last.clone(),
            total_lanes / other_index_lanes,
            is_scalable(&last),
        )?;
        Ok(value
            .copy_with(
                value.buffer.clone(),
                broadcast_to(stored_value, total_lanes, is_scalable(&last))?,
                Array::new(mapped_indices),
            )
            .into())
    }

    fn mutate_loop(&mut self, value: For, mutator: &mut Mutator) -> Result<Stmt> {
        if int_value(&value.min) != Some(0) {
            return Err(value_error(
                "a loop nested inside vectorization must start at zero",
            ));
        }
        if is_vector(&value.extent) {
            return Err(value_error("a nested loop extent must initially be scalar"));
        }
        let extent = self.mutate_prim(mutator, &value.extent)?;
        if is_vector(&extent) {
            return self.scalarize(value.into());
        }
        let body = self.mutate_statement(mutator, &value.body)?;
        if extent.same_as(&value.extent) && body.same_as(&value.body) {
            return Ok(value.into());
        }
        Ok(value
            .copy_with(value.loop_var.clone(), value.min.clone(), extent, body)
            .into())
    }

    fn mutate_conditional(&mut self, value: IfThenElse, mutator: &mut Mutator) -> Result<Stmt> {
        if is_vector(&value.condition) {
            return Err(value_error(
                "an IfThenElse condition must initially be scalar",
            ));
        }
        let condition = self.mutate_prim(mutator, &value.condition)?;
        let condition_requested_scalarization = std::mem::take(&mut self.need_scalarize);
        let then_case = self.mutate_statement(mutator, &value.then_case)?;
        let else_case = value
            .else_case
            .as_ref()
            .map(|statement| self.mutate_statement(mutator, statement))
            .transpose()?;
        if condition_requested_scalarization || is_vector(&condition) {
            return self.scalarize(value.into());
        }
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

    fn mutate_while(&mut self, _value: While) -> Result<Stmt> {
        Err(value_error(
            "a while loop inside a vectorized loop is not supported",
        ))
    }

    fn mutate_binding(&mut self, value: Bind, mutator: &mut Mutator) -> Result<Stmt> {
        let Ok(original) = PrimExpr::try_from(value.value.clone()) else {
            return super::utils::mutate_stmt_default(self, mutator, value.into());
        };
        let mapped = self.mutate_prim(mutator, &original)?;
        if self.need_scalarize {
            self.need_scalarize = false;
            return self.scalarize(value.into());
        }
        let identity = ObjectIdentity::of(&value.var);
        if self.let_bindings.contains_key(&identity) {
            return Err(value_error("a Bind variable may only be defined once"));
        }
        if lane_count(&mapped) != lane_count(&original) {
            let variable = Var::with_type(value.var.name.as_str(), mapped.type_annotation());
            self.let_bindings.insert(
                identity,
                PrimExpr::from(PrimVar::try_from(variable.clone())?),
            );
            Bind::new(variable, mapped).map(Into::into)
        } else {
            self.let_bindings.insert(
                identity,
                PrimExpr::from(PrimVar::try_from(value.var.clone())?),
            );
            if mapped.same_as(&original) {
                Ok(value.into())
            } else {
                Bind::with_span(value.var.clone(), mapped, value.span.as_ref()).map(Into::into)
            }
        }
    }

    fn mutate_default(&mut self, value: &MapValue, mutator: &mut Mutator) -> Result<Any> {
        let original_statement = value.cast::<Stmt>();
        let mapped = mutate_stmt_expr_default(self, mutator, value)?;
        if let Some(statement) = original_statement {
            if self.need_scalarize {
                self.need_scalarize = false;
                return self.scalarize(statement).map(Into::into);
            }
        }
        Ok(mapped)
    }
}

impl Vectorizer {
    fn is_positive(&self, value: &PrimExpr) -> Result<bool> {
        let zero = IntImm::from_dtype(value.dtype(), 0)?;
        self.analyzer
            .can_prove(&semantic_binary("tirx._OpGT", value.clone(), zero.into())?)
    }
}

fn primitive_type(value: &Expr) -> Result<PrimType> {
    value.ty.clone().try_cast()
}

fn is_scalable_type(value: &PrimType) -> bool {
    (value.dtype.lanes as i16) < 0
}

fn lane_count_type(dtype: &DLDataType) -> u16 {
    let signed = dtype.lanes as i16;
    if signed < 0 {
        signed.unsigned_abs()
    } else {
        dtype.lanes
    }
}

fn lane_count(value: &PrimExpr) -> u16 {
    lane_count_type(&value.dtype())
}

fn is_scalable(value: &PrimExpr) -> bool {
    is_scalable_type(&value.type_annotation())
}

fn is_vector(value: &PrimExpr) -> bool {
    is_scalable(value) || lane_count(value) > 1
}

fn is_scalar(value: &PrimExpr) -> bool {
    !is_vector(value)
}

fn with_lane_count(value: &PrimType, lanes: u16, scalable: bool) -> Result<PrimType> {
    if lanes == 0 {
        return Err(value_error("a vector lane count must be positive"));
    }
    let encoded_lanes = if scalable {
        let lanes = i16::try_from(lanes)
            .map_err(|_| value_error("a scalable-vector factor must fit i16"))?;
        (-lanes) as u16
    } else {
        lanes
    };
    PrimType::from_dtype(DLDataType {
        lanes: encoded_lanes,
        ..value.dtype
    })
}

fn with_lanes(value: &PrimType, lanes_from: &PrimExpr) -> Result<PrimType> {
    with_lane_count(value, lane_count(lanes_from), is_scalable(lanes_from))
}

fn broadcast_to(value: PrimExpr, lanes: u16, scalable: bool) -> Result<PrimExpr> {
    if lane_count(&value) == lanes && is_scalable(&value) == scalable {
        return Ok(value);
    }
    if let Ok(broadcast) = value.clone().try_cast::<Broadcast>() {
        if is_scalable(&broadcast.value) != scalable && is_scalable(&value) != scalable {
            return Err(value_error(
                "cannot broadcast between scalable and fixed-length vectors",
            ));
        }
        let old_lanes = lane_count(&value);
        if lanes.is_multiple_of(old_lanes) {
            return make_broadcast(
                broadcast.value.clone(),
                lane_expression(lanes, scalable)?,
                None,
            );
        }
    }
    if !is_scalar(&value) {
        return Err(value_error(
            "only scalar values may be broadcast to new lanes",
        ));
    }
    make_broadcast(value, lane_expression(lanes, scalable)?, None)
}

fn lane_expression(lanes: u16, scalable: bool) -> Result<PrimExpr> {
    if !scalable {
        return Ok(IntImm::new("int32", i64::from(lanes))?.into());
    }
    let vscale = Call::new(
        PrimType::new("int32")?,
        get_operator("tirx.vscale")?,
        Vec::new(),
    );
    let vscale: Expr = vscale.into();
    semantic_binary(
        "tirx._OpMul",
        PrimExpr::try_from(vscale)?,
        IntImm::new("int32", i64::from(lanes))?.into(),
    )
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

fn make_broadcast(
    value: PrimExpr,
    lanes: PrimExpr,
    span: Option<&crate::ir::Span>,
) -> Result<PrimExpr> {
    tvm_ffi::cached_global_func!("tirx.Broadcast")
        .call_tuple((value, lanes, span.cloned()))?
        .try_into()
}

fn extract_element(vector: PrimExpr, index: i64) -> Result<PrimExpr> {
    tvm_ffi::cached_global_func!("tirx.Shuffle")
        .call_tuple((
            Array::new(vec![vector]),
            Array::new(vec![PrimExpr::from(IntImm::new("int32", index)?)]),
            Option::<crate::ir::Span>::None,
        ))?
        .try_into()
}

fn concat_vectors(vectors: Vec<PrimExpr>) -> Result<PrimExpr> {
    let mut indices = Vec::new();
    let mut offset = 0_i64;
    for vector in &vectors {
        for lane in 0..lane_count(vector) {
            indices.push(IntImm::new("int32", offset + i64::from(lane))?.into());
        }
        offset += i64::from(lane_count(vector));
    }
    tvm_ffi::cached_global_func!("tirx.Shuffle")
        .call_tuple((
            Array::new(vectors),
            Array::<PrimExpr>::new(indices),
            Option::<crate::ir::Span>::None,
        ))?
        .try_into()
}

fn semantic_binary(name: &str, lhs: PrimExpr, rhs: PrimExpr) -> Result<PrimExpr> {
    tvm_ffi::Function::get_global(name)?
        .call_tuple((lhs, rhs, Option::<crate::ir::Span>::None))?
        .try_into()
}
