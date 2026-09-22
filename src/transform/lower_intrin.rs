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
    Any, Array, DLDataType, DLDataTypeCode, DLDataTypeExt, Function, Mutator, ObjectRefCast,
    ObjectRefCore, Result, String, StructuralView,
};

#[cfg(test)]
#[path = "../../tests/unit/lower_intrin_scopes.rs"]
mod scope_tests;

use super::utils::{
    binary_op, cast_prim_expr, finish_constraint_contexts, fixed_lanes, get_operator, int_value,
    mutate_expr_default, option_same_as, value_error, with_prim_func_body, BufferRemaps,
};
use super::{create_prim_func_pass_with_context, Pass, PassContext};
use crate::analysis::{side_effect, Analyzer, CallEffectKind};
use crate::ir::{Call, CallObj, Expr, IntImm, PrimExpr, PrimType, Range, TensorLoad, Var};
use crate::prim::{
    Add, Broadcast, BroadcastObj, Cast, CastObj, FloorDiv, FloorDivObj, FloorMod, FloorModObj, Let,
    Max, MulObj, Not, Ramp, Select, EQ, GT, NE,
};
use crate::target::Target;
use crate::tirx::{
    AssertStmt, AttrStmt, Bind, BufferType, BufferVar, DeclBuffer, Evaluate, For, IfThenElse,
    IterVar, PrimFunc, PrimVar, Stmt,
};

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
    let body = injecter.with_scope(|injecter| {
        tvm_ffi::structural_mutate(function.body().clone(), injecter)?.try_into()
    })?;
    Ok(with_prim_func_body(function, body))
}

struct AccessPtrAlias {
    buffer: BufferVar,
    data: Expr,
}

struct IntrinInjecter {
    analyzer: Analyzer,
    buffer_remaps: BufferRemaps,
    constraint_exits: Vec<Function>,
    attribute_names: Vec<String>,
    fma_rule: Option<Function>,
    aliases: Vec<AccessPtrAlias>,
    access_ptr_operator: Expr,
    buffer_data_operator: Expr,
    address_of_operator: Expr,
    fma_operator: Expr,
    floor_operator: Expr,
    bitwise_and_operator: Expr,
    likely_operator: Expr,
    if_then_else_operator: Expr,
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
            buffer_remaps: BufferRemaps::default(),
            constraint_exits: Vec::new(),
            attribute_names,
            fma_rule,
            aliases: Vec::new(),
            access_ptr_operator: get_operator("tirx.tvm_access_ptr")?,
            buffer_data_operator: get_operator("tirx.buffer_data")?,
            address_of_operator: get_operator("tirx.address_of")?,
            fma_operator,
            floor_operator: get_operator("tirx.floor")?,
            bitwise_and_operator: get_operator("prim.bitwise_and")?,
            likely_operator: get_operator("prim.likely")?,
            if_then_else_operator: get_operator("prim.if_then_else")?,
        })
    }

    fn with_scope<T>(&mut self, operation: impl FnOnce(&mut Self) -> Result<T>) -> Result<T> {
        let begin = self.constraint_exits.len();
        let result = operation(self);
        finish_constraint_contexts(result, self.constraint_exits.split_off(begin))
    }

    fn with_constraint<T>(
        &mut self,
        condition: &PrimExpr,
        operation: impl FnOnce(&mut Self) -> Result<T>,
    ) -> Result<T> {
        self.with_scope(|this| {
            this.constraint_exits
                .push(this.analyzer.enter_constraint(condition)?);
            operation(this)
        })
    }

    fn with_constraint_facts<T>(
        &mut self,
        condition: &PrimExpr,
        operation: impl FnOnce(&mut Self) -> Result<T>,
    ) -> Result<T> {
        self.with_scope(|this| {
            let exits = super::analyzer_constraints::enter_constraint_facts(
                &this.analyzer,
                condition,
                &this.bitwise_and_operator,
            )?;
            this.constraint_exits.extend(exits);
            operation(this)
        })
    }

    // The native VisitStmt wrapper emits access-pointer aliases around every
    // statement, including statements with custom Analyzer scope handling.
    fn with_aliases(&mut self, operation: impl FnOnce(&mut Self) -> Result<Stmt>) -> Result<Stmt> {
        let begin = self.aliases.len();
        let result = operation(self);
        let aliases = self.aliases.split_off(begin);
        let mut mapped = result?;
        for alias in aliases.into_iter().rev() {
            mapped = Stmt::sequence(vec![
                DeclBuffer::new(alias.buffer, alias.data)?.into(),
                mapped,
            ])?;
        }
        Ok(mapped)
    }

    fn bind_pure_value(&self, variable: &Var, value: &PrimExpr) -> Result<()> {
        if side_effect(value)? <= CallEffectKind::kPure {
            self.analyzer.bind_expression(variable, value)?;
        }
        Ok(())
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
                inner_offset = Cast::new(offset.type_annotation(), inner_offset)?.into();
            }
            offset = binary_op("prim._OpAdd", inner_offset, offset)?;
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
        let mut scalar_extent = binary_op("prim._OpAdd", offset.clone(), int_like(&offset, 1)?)?;
        if dtype.dtype.lanes != 1 {
            let lanes = fixed_lanes(&dtype)?;
            offset = binary_op("prim._OpMul", offset, int_like(&scalar_extent, lanes)?)?;
            scalar_extent = binary_op("prim._OpAdd", offset.clone(), int_like(&offset, lanes)?)?;
            offset = Ramp::new(
                offset,
                int_like(&scalar_extent, 1)?,
                int_like(&scalar_extent, lanes)?,
            )?
            .into();
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
        if call.op.as_node::<crate::ir::OpObj>().is_none() {
            return Ok(None);
        }
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
                let lowered: Option<PrimExpr> = rule.call_tuple((fma,))?.try_into()?;
                if let Some(lowered) = lowered {
                    return mutator.mutate(self, &lowered)?.try_into();
                }
                // Like C++ MakeFMA, a declined rule visits the original Add,
                // without applying the Broadcast/Cast swaps attempted above.
                return mutate_expr_default(self, mutator, original.clone().into())?.try_into();
            }
        }
        if !lhs.same_as(a) || !rhs.same_as(b) {
            let product: PrimExpr = mutator
                .mutate(self, &crate::prim::Mul::new(lhs, rhs)?)?
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
        if let Some(literal) = int_value(value) {
            return Ok(literal >= 0);
        }
        // Match Analyzer::CanProveGreaterEqual, including vector element bounds.
        let simplified = self.analyzer.rewrite_simplify(value)?;
        Ok(self.analyzer.const_int_bound(&simplified)?.min_value >= 0)
    }
}

#[tvm_ffi::dispatch(mutate)]
impl IntrinInjecter {
    fn mutate_loop(&mut self, value: For, mutator: &mut Mutator) -> Result<Stmt> {
        self.with_aliases(|this| {
            this.with_scope(|this| {
                this.analyzer.bind(
                    value.loop_var.as_var(),
                    &Range::from_min_extent(value.min.clone(), value.extent.clone())?,
                )?;
                let minimum: PrimExpr = mutator.mutate(this, &value.min)?.try_into()?;
                let extent: PrimExpr = mutator.mutate(this, &value.extent)?.try_into()?;
                let step = mutator.mutate(this, &value.step)?.try_into()?;
                let positive = GT::new(extent.clone(), int_like(&extent, 0)?)?.into();
                let body: Stmt = this.with_constraint_facts(&positive, |this| {
                    mutator.mutate(this, &value.body)?.try_into()
                })?;
                if minimum.same_as(&value.min)
                    && extent.same_as(&value.extent)
                    && option_same_as(&step, &value.step)
                    && body.same_as(&value.body)
                {
                    return Ok(value.into());
                }
                Ok(For::from_complete_fields(
                    value.span.clone(),
                    value.loop_var.clone(),
                    minimum,
                    extent,
                    value.kind,
                    body,
                    value.thread_binding.clone(),
                    value.annotations.clone(),
                    step,
                )
                .into())
            })
        })
    }

    fn mutate_conditional(&mut self, value: IfThenElse, mutator: &mut Mutator) -> Result<Stmt> {
        self.with_aliases(|this| {
            this.with_scope(|this| {
                let condition: PrimExpr = mutator.mutate(this, &value.condition)?.try_into()?;
                let real_condition = if let Some(call) = condition.as_node::<CallObj>() {
                    if call.op.same_as(&this.likely_operator) {
                        call.args.get(0)?.try_into()?
                    } else {
                        condition.clone()
                    }
                } else {
                    condition.clone()
                };
                let then_case: Stmt = this.with_constraint_facts(&real_condition, |this| {
                    mutator.mutate(this, &value.then_case)?.try_into()
                })?;
                let else_case = value
                    .else_case
                    .as_ref()
                    .map(|branch| {
                        let negative = this
                            .analyzer
                            .rewrite_simplify(&Not::new(real_condition.clone())?.into())?;
                        this.with_constraint(&negative, |this| {
                            mutator.mutate(this, branch)?.try_into()
                        })
                    })
                    .transpose()?;
                match int_value(&real_condition) {
                    Some(1) => return Ok(then_case),
                    Some(0) => {
                        return else_case
                            .map(Ok)
                            .unwrap_or_else(|| Evaluate::from_i64(0).map(Into::into))
                    }
                    _ => {}
                }
                if condition.same_as(&value.condition)
                    && then_case.same_as(&value.then_case)
                    && option_same_as(&else_case, &value.else_case)
                {
                    return Ok(value.into());
                }
                Ok(IfThenElse::from_complete_fields(
                    value.span.clone(),
                    condition,
                    then_case,
                    else_case,
                )
                .into())
            })
        })
    }

    fn mutate_attribute(&mut self, value: AttrStmt, mutator: &mut Mutator) -> Result<Stmt> {
        self.with_aliases(|this| {
            this.with_scope(|this| {
                if matches!(value.attr_key.as_str(), "thread_extent" | "virtual_thread") {
                    let iteration: IterVar = value.node.clone().try_into()?;
                    if iteration.thread_tag()?.is_empty() {
                        return Err(value_error("thread extent requires a thread tag"));
                    }
                    this.analyzer.bind(
                        iteration.var()?.as_var(),
                        &Range::from_min_extent(
                            int_like(&PrimExpr::try_from(value.value.clone())?, 0)?,
                            value.value.clone(),
                        )?,
                    )?;
                }
                BufferRemaps::mutate_stmt(this, mutator, value.into(), |state| {
                    &mut state.buffer_remaps
                })
            })
        })
    }

    fn mutate_binding(&mut self, value: Bind, mutator: &mut Mutator) -> Result<Stmt> {
        self.with_aliases(|this| {
            let bound: Expr = mutator.mutate(this, &value.value)?.try_into()?;
            if let Ok(primitive) = PrimExpr::try_from(&bound) {
                this.bind_pure_value(&value.var, &primitive)?;
            }
            if bound.same_as(&value.value) {
                return Ok(value.into());
            }
            Ok(value.copy_with(value.var.clone(), bound).into())
        })
    }

    fn mutate_assertion(&mut self, value: AssertStmt, mutator: &mut Mutator) -> Result<Stmt> {
        self.with_aliases(|this| {
            let condition: PrimExpr = mutator.mutate(this, &value.condition)?.try_into()?;
            // Assert facts remain active for following siblings until the enclosing scope ends.
            this.constraint_exits
                .push(this.analyzer.enter_constraint(&condition)?);
            if condition.same_as(&value.condition) {
                return Ok(value.into());
            }
            Ok(value
                .copy_with(
                    condition,
                    value.error_kind.clone(),
                    value.message_parts.clone(),
                )
                .into())
        })
    }

    fn mutate_let(&mut self, value: Let, mutator: &mut Mutator) -> Result<Let> {
        let bound: PrimExpr = mutator.mutate(self, &value.value)?.try_into()?;
        self.bind_pure_value(&value.var, &bound)?;
        let body: PrimExpr = mutator.mutate(self, &value.body)?.try_into()?;
        if bound.same_as(&value.value) && body.same_as(&value.body) {
            return Ok(value);
        }
        Let::new(value.var.clone(), bound, body)
    }

    fn mutate_select(&mut self, value: Select, mutator: &mut Mutator) -> Result<PrimExpr> {
        let condition: PrimExpr = mutator.mutate(self, &value.condition)?.try_into()?;
        let true_value: PrimExpr = self.with_constraint_facts(&condition, |this| {
            mutator.mutate(this, &value.true_value)?.try_into()
        })?;
        let negative = self
            .analyzer
            .rewrite_simplify(&Not::new(condition.clone())?.into())?;
        let false_value: PrimExpr = self.with_constraint(&negative, |this| {
            mutator.mutate(this, &value.false_value)?.try_into()
        })?;
        match int_value(&condition) {
            Some(0) => return Ok(false_value),
            Some(1) => return Ok(true_value),
            _ => {}
        }
        if condition.same_as(&value.condition)
            && true_value.same_as(&value.true_value)
            && false_value.same_as(&value.false_value)
        {
            return Ok(value.into());
        }
        Select::new(condition, true_value, false_value).map(Into::into)
    }

    fn mutate_call(&mut self, value: Call, mutator: &mut Mutator) -> Result<Expr> {
        if value.op.same_as(&self.access_ptr_operator) {
            let lowered = self.lower_access_ptr(&value)?;
            return mutator.mutate(self, &lowered)?.try_into();
        }
        if let Some(lowered) = self.apply_intrinsic_rule(&value, mutator)? {
            return Ok(lowered);
        }
        if value.op.same_as(&self.if_then_else_operator) {
            let condition: PrimExpr = mutator.mutate(self, &value.args.get(0)?)?.try_into()?;
            let old_true = value.args.get(1)?;
            let old_false = value.args.get(2)?;
            let true_value: Expr = self.with_constraint_facts(&condition, |this| {
                mutator.mutate(this, &old_true)?.try_into()
            })?;
            let negative = Not::new(condition.clone())?.into();
            let false_value: Expr = self.with_constraint(&negative, |this| {
                mutator.mutate(this, &old_false)?.try_into()
            })?;
            match int_value(&condition) {
                Some(0) => return Ok(false_value),
                Some(1) => return Ok(true_value),
                _ => {}
            }
            if condition.same_as(&value.args.get(0)?)
                && true_value.same_as(&old_true)
                && false_value.same_as(&old_false)
            {
                return Ok(value.into());
            }
            return Ok(Call::with_metadata(
                value.ty.clone(),
                value.op.clone(),
                vec![condition.into(), true_value, false_value],
                value.attrs.clone(),
                Vec::new(),
                value.span.as_ref(),
            )
            .into());
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
            return binary_op(
                "prim.right_shift",
                mapped.a.clone(),
                int_like(&mapped.a, i64::from(shift))?,
            );
        }
        if self.can_prove_nonnegative(&mapped.b)? {
            if self.can_prove_nonnegative(&mapped.a)? || self.can_prove_nonnegative(&original)? {
                return binary_op("prim._OpTruncDiv", mapped.a.clone(), mapped.b.clone());
            }
            if let Some(divisor) = int_value(&mapped.b) {
                if let Some(coefficient) = self.try_find_shift_coefficient(&mapped.a, divisor)? {
                    let shifted = binary_op(
                        "prim._OpAdd",
                        mapped.a.clone(),
                        int_like(&mapped.a, divisor * coefficient)?,
                    )?;
                    return binary_op(
                        "prim._OpSub",
                        binary_op("prim._OpTruncDiv", shifted, mapped.b.clone())?,
                        int_like(&mapped.a, coefficient)?,
                    );
                }
            }
            let quotient = binary_op("prim._OpTruncDiv", mapped.a.clone(), mapped.b.clone())?;
            let remainder = binary_op("prim._OpTruncMod", mapped.a.clone(), mapped.b.clone())?;
            if dtype.dtype.code == DLDataTypeCode::kDLInt as u8
                && dtype.dtype.lanes == 1
                && matches!(dtype.dtype.bits, 32 | 64)
            {
                let correction = binary_op(
                    "prim.right_shift",
                    remainder,
                    int_like(&mapped.a, i64::from(dtype.dtype.bits - 1))?,
                )?;
                return binary_op("prim._OpAdd", quotient, correction);
            }
            return Ok(Select::new(
                binary_op("prim._OpGE", remainder.clone(), int_like(&remainder, 0)?)?,
                quotient.clone(),
                binary_op("prim._OpSub", quotient, int_like(&mapped.a, 1)?)?,
            )?
            .into());
        }
        if dtype.dtype.code == DLDataTypeCode::kDLFloat as u8 {
            let division = binary_op("prim._OpDiv", mapped.a.clone(), mapped.b.clone())?;
            let floor = Call::new(dtype, self.floor_operator.clone(), vec![division.into()]);
            return mutator.mutate(self, &floor)?.try_into();
        }
        let remainder: PrimVar = Var::with_type("rmod", dtype.clone()).try_into()?;
        let quotient: PrimVar = Var::with_type("rdiv", dtype).try_into()?;
        let condition = floor_remainder_has_correct_sign(mapped.b.clone(), (&remainder).into())?;
        let selected = Select::new(
            condition,
            quotient.clone(),
            binary_op("prim._OpSub", (&quotient).into(), int_like(&mapped.a, 1)?)?,
        )?;
        Ok(Let::new(
            remainder.into(),
            binary_op("prim._OpTruncMod", mapped.a.clone(), mapped.b.clone())?,
            Let::new(
                quotient.into(),
                binary_op("prim._OpTruncDiv", mapped.a.clone(), mapped.b.clone())?,
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
            return binary_op(
                "prim.bitwise_and",
                mapped.a.clone(),
                int_like(&mapped.a, mask)?,
            );
        }
        if self.can_prove_nonnegative(&mapped.b)? {
            if self.can_prove_nonnegative(&mapped.a)? {
                return binary_op("prim._OpTruncMod", mapped.a.clone(), mapped.b.clone());
            }
            if let Some(divisor) = int_value(&mapped.b) {
                if let Some(coefficient) = self.try_find_shift_coefficient(&mapped.a, divisor)? {
                    return binary_op(
                        "prim._OpTruncMod",
                        binary_op(
                            "prim._OpAdd",
                            mapped.a.clone(),
                            int_like(&mapped.a, divisor * coefficient)?,
                        )?,
                        mapped.b.clone(),
                    );
                }
            }
            let remainder = binary_op("prim._OpTruncMod", mapped.a.clone(), mapped.b.clone())?;
            if dtype.dtype.code == DLDataTypeCode::kDLInt as u8
                && dtype.dtype.lanes == 1
                && matches!(dtype.dtype.bits, 32 | 64)
            {
                let sign = binary_op(
                    "prim.right_shift",
                    remainder.clone(),
                    int_like(&mapped.a, i64::from(dtype.dtype.bits - 1))?,
                )?;
                let correction = binary_op("prim.bitwise_and", mapped.b.clone(), sign)?;
                return binary_op("prim._OpAdd", remainder, correction);
            }
            return Ok(Select::new(
                binary_op("prim._OpGE", remainder.clone(), int_like(&remainder, 0)?)?,
                remainder.clone(),
                binary_op("prim._OpAdd", remainder, mapped.b.clone())?,
            )?
            .into());
        }
        if dtype.dtype.code == DLDataTypeCode::kDLFloat as u8 {
            let division = binary_op("prim._OpDiv", mapped.a.clone(), mapped.b.clone())?;
            let floor: PrimExpr = mutator
                .mutate(
                    self,
                    &Call::new(dtype, self.floor_operator.clone(), vec![division.into()]),
                )?
                .try_into()?;
            return binary_op(
                "prim._OpSub",
                mapped.a.clone(),
                binary_op("prim._OpMul", floor, mapped.b.clone())?,
            );
        }
        let remainder: PrimVar = Var::with_type("rmod", dtype).try_into()?;
        let condition = floor_remainder_has_correct_sign(mapped.b.clone(), (&remainder).into())?;
        Ok(Let::new(
            (&remainder).into(),
            binary_op("prim._OpTruncMod", mapped.a.clone(), mapped.b.clone())?,
            Select::new(
                condition,
                remainder.clone(),
                binary_op("prim._OpAdd", remainder.into(), mapped.b.clone())?,
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
                let quotient = binary_op("prim._OpTruncDiv", divide.a.clone(), divide.b.clone())?;
                let quotient = mutator.mutate(self, &quotient)?.try_into()?;
                return binary_op("prim._OpMax", quotient, value.b.clone());
            }
        }
        mutate_expr_default(self, mutator, value.into())?.try_into()
    }

    fn mutate_equal(&mut self, value: EQ, mutator: &mut Mutator) -> Result<PrimExpr> {
        if let Some(remainder) = value.a.as_node::<FloorModObj>() {
            if int_value(&value.b) == Some(0) {
                let lowered = binary_op(
                    "prim._OpEQ",
                    binary_op("prim._OpTruncMod", remainder.a.clone(), remainder.b.clone())?,
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
                let lowered = binary_op(
                    "prim._OpNE",
                    binary_op("prim._OpTruncMod", remainder.a.clone(), remainder.b.clone())?,
                    value.b.clone(),
                )?;
                return mutator.mutate(self, &lowered)?.try_into();
            }
        }
        mutate_expr_default(self, mutator, value.into())?.try_into()
    }

    // Dispatch uses the first match, so keep this after the specific statement handlers.
    fn mutate_statement(&mut self, value: Stmt, mutator: &mut Mutator) -> Result<Stmt> {
        self.with_aliases(|this| {
            BufferRemaps::mutate_stmt(this, mutator, value, |state| &mut state.buffer_remaps)
        })
    }

    fn mutate_default(&mut self, value: &StructuralView, mutator: &mut Mutator) -> Result<Any> {
        BufferRemaps::mutate_default(self, mutator, value, |state| &mut state.buffer_remaps)
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
    let value_dtype = cast.value.dtype();
    let integer_like = matches!(
        cast_dtype.code,
        x if x == DLDataTypeCode::kDLInt as u8 || x == DLDataTypeCode::kDLUInt as u8
    ) && matches!(
        value_dtype.code,
        x if x == DLDataTypeCode::kDLInt as u8 || x == DLDataTypeCode::kDLUInt as u8
    );
    if cast_dtype.bits != value_dtype.bits * 2
        && !(integer_like && cast_dtype.bits > value_dtype.bits)
    {
        return Ok(value.clone());
    }
    let broadcast = Broadcast::new(cast.value.clone(), broadcast.lanes.clone())?;
    Ok(Cast::new(value.type_annotation(), broadcast)?.into())
}

fn with_lanes(ty: &PrimType, lanes: u16) -> Result<PrimType> {
    let mut dtype = ty.dtype;
    dtype.lanes = lanes;
    PrimType::from_dtype(dtype)
}

fn floor_remainder_has_correct_sign(divisor: PrimExpr, remainder: PrimExpr) -> Result<PrimExpr> {
    let zero = int_like(&divisor, 0)?;
    binary_op(
        "prim._OpOr",
        binary_op(
            "prim._OpAnd",
            binary_op("prim._OpGE", divisor.clone(), zero.clone())?,
            binary_op("prim._OpGE", remainder.clone(), zero.clone())?,
        )?,
        binary_op(
            "prim._OpAnd",
            binary_op("prim._OpLT", divisor, zero.clone())?,
            binary_op("prim._OpLE", remainder, zero)?,
        )?,
    )
}

fn constant_power_of_two(value: &PrimExpr) -> Option<u32> {
    let value = int_value(value)?;
    (value > 0 && (value as u64).is_power_of_two()).then(|| (value as u64).trailing_zeros())
}

fn int_like(value: &PrimExpr, literal: i64) -> Result<PrimExpr> {
    let dtype = value.dtype();
    let scalar = IntImm::from_dtype(DLDataType { lanes: 1, ..dtype }, literal)?.into();
    if dtype.lanes == 1 {
        Ok(scalar)
    } else {
        // The semantic cast broadcasts a scalar to fixed or scalable lanes.
        cast_prim_expr(scalar, value.type_annotation())
    }
}

fn pass_config_bool(context: &PassContext, key: &str) -> Result<bool> {
    context
        .config()?
        .get(&String::from(key))?
        .map(bool::try_from)
        .transpose()
        .map(|value| value.unwrap_or(false))
}
