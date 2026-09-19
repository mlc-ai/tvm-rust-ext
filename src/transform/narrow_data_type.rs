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
    structural_mutate, structural_visit, DLDataType, DLDataTypeCode, ObjectIdentity, ObjectRefCast,
    ObjectRefCore, Result, VisitCallbacks, VisitContext, VisitInterrupt, VisitValue,
};

use super::force_narrow_index::IndexDataTypeNormalizer;
use super::utils::{visit_stmt_expr_default, with_prim_func_body};
use super::{create_prim_func_pass, Pass};
use crate::analysis::Analyzer;
use crate::ir::{Expr, IntImm, PrimExpr, PrimType, Range, TensorLoad, Var};
use crate::prim::Cast;
use crate::te::Reduce;
use crate::tirx::{AttrStmt, For, IterVar, PrimFunc, Stmt};

const THREAD_EXTENT: &str = "thread_extent";
const VIRTUAL_THREAD: &str = "virtual_thread";

/// Narrow index components whose analyzer-proven ranges fit in `target_bits`.
pub fn narrow_data_type_prim_func(function: PrimFunc, target_bits: u8) -> Result<PrimFunc> {
    let target = PrimType::from_dtype(DLDataType {
        code: DLDataTypeCode::kDLInt as u8,
        bits: target_bits,
        lanes: 1,
    })?;
    let mut collector = VisitCallbacks::new(
        NarrowPlan::new(target_bits)?,
        (
            visit_tensor_load,
            visit_loop,
            visit_attribute,
            visit_reduce,
            visit_variable,
            visit_integer,
            visit_cast,
            visit_default,
        ),
    );
    structural_visit(function.body(), &mut collector)?;
    let selected_types = collector.into_state().selected_types()?;
    let mut normalizer = IndexDataTypeNormalizer::from_selected_types(target, selected_types)?;
    let body: Stmt = structural_mutate(function.body().clone(), &mut normalizer)?.try_into()?;
    Ok(with_prim_func_body(function, body))
}

/// Build TVM's `tirx.NarrowDataType` PrimFunc pass in Rust.
pub fn narrow_data_type(target_bits: u8) -> Result<Pass> {
    create_prim_func_pass(
        "tirx.NarrowDataType",
        0,
        Vec::new(),
        false,
        move |function| narrow_data_type_prim_func(function, target_bits),
    )
}

struct NarrowPlan {
    analyzer: Analyzer,
    target_bits: u8,
    current_bits: u8,
    variable_extent_types: HashMap<ObjectIdentity, PrimType>,
    // Original type and maximum required width across every use of a node.
    required_types: HashMap<ObjectIdentity, (PrimType, u8)>,
}

impl NarrowPlan {
    fn new(target_bits: u8) -> Result<Self> {
        // Constructing the type validates the bit width using TVM's dtype rules.
        PrimType::from_dtype(DLDataType {
            code: DLDataTypeCode::kDLInt as u8,
            bits: target_bits,
            lanes: 1,
        })?;
        Ok(Self {
            analyzer: Analyzer::new()?,
            target_bits,
            current_bits: target_bits,
            variable_extent_types: HashMap::new(),
            required_types: HashMap::new(),
        })
    }

    fn record_type<T: ObjectRefCore>(&mut self, value: &T, original: &PrimType, bits: u8) {
        let bits = original.dtype.bits.min(bits);
        self.required_types
            .entry(ObjectIdentity::of(value))
            .and_modify(|(_, required)| *required = (*required).max(bits))
            .or_insert_with(|| (original.clone(), bits));
    }

    fn selected_types(self) -> Result<HashMap<ObjectIdentity, PrimType>> {
        self.required_types
            .into_iter()
            .filter(|(_, (original, bits))| original.dtype.bits != *bits)
            .map(|(identity, (original, bits))| {
                let ty = PrimType::from_dtype(DLDataType {
                    bits,
                    ..original.dtype
                })?;
                Ok((identity, ty))
            })
            .collect()
    }
}

fn visit_tensor_load(value: TensorLoad, visitor: &mut VisitContext<'_, NarrowPlan>) -> Result<()> {
    visit_with_expression_context(value.clone().into(), visitor, |visitor| {
        let old_bits = visitor.state().current_bits;
        visitor.state_mut().current_bits = visitor.state().target_bits;
        let result = value
            .indices
            .iter()
            .try_for_each(|index| visitor.visit(&index).map(|_| ()));
        visitor.state_mut().current_bits = old_bits;
        result
    })
}

fn visit_loop(value: For, visitor: &mut VisitContext<'_, NarrowPlan>) -> Result<()> {
    let range = Range::from_min_extent(value.min.clone(), value.extent.clone())?;
    visitor
        .state()
        .analyzer
        .bind(value.loop_var.as_var(), &range)?;
    visitor.state_mut().variable_extent_types.insert(
        ObjectIdentity::of(value.loop_var.as_var()),
        value.extent.type_annotation(),
    );
    visitor.visit(&value.min)?;
    visitor.visit(&value.extent)?;
    if let Some(step) = &value.step {
        visitor.visit(step)?;
    }
    visitor.visit(&value.body)?;
    Ok(())
}

fn visit_attribute(value: AttrStmt, visitor: &mut VisitContext<'_, NarrowPlan>) -> Result<()> {
    if matches!(value.attr_key.as_str(), THREAD_EXTENT | VIRTUAL_THREAD) {
        let iteration = IterVar::try_from(value.node.clone())?;
        let variable = iteration.var()?;
        if iteration.thread_tag()?.is_empty() {
            return Err(tvm_ffi::Error::new(
                tvm_ffi::VALUE_ERROR,
                "thread extent requires a tagged IterVar",
                "",
            ));
        }
        let value_type = PrimExpr::try_from(value.value.clone())?.type_annotation();
        let zero = IntImm::from_dtype(value_type.dtype, 0)?;
        let range = Range::from_min_extent(zero, value.value.clone())?;
        visitor.state().analyzer.bind(variable.as_var(), &range)?;
        visitor
            .state_mut()
            .variable_extent_types
            .insert(ObjectIdentity::of(variable.as_var()), value_type);
    }
    visitor.visit(&value.value)?;
    visitor.visit(&value.body)?;
    Ok(())
}

fn visit_reduce(value: Reduce, visitor: &mut VisitContext<'_, NarrowPlan>) -> Result<()> {
    visit_with_expression_context(value.clone().into(), visitor, |visitor| {
        for iteration in value.axis.iter() {
            let domain = iteration.dom()?.ok_or_else(|| {
                tvm_ffi::Error::new(tvm_ffi::VALUE_ERROR, "reduction axis requires a domain", "")
            })?;
            let variable = iteration.var()?;
            visitor.state().analyzer.bind(variable.as_var(), &domain)?;
            visitor.state_mut().variable_extent_types.insert(
                ObjectIdentity::of(variable.as_var()),
                domain.extent.type_annotation(),
            );
        }
        visitor.visit_children()?;
        Ok(())
    })
}

fn visit_variable(value: Var, visitor: &mut VisitContext<'_, NarrowPlan>) -> Result<()> {
    visit_with_expression_context(value.clone().into(), visitor, |visitor| {
        let identity = ObjectIdentity::of(&value);
        if let Some(extent_type) = visitor
            .state()
            .variable_extent_types
            .get(&identity)
            .cloned()
        {
            let original = value.ty.clone().try_cast::<PrimType>()?;
            let bits = extent_type.dtype.bits.min(visitor.state().current_bits);
            visitor.state_mut().record_type(&value, &original, bits);
        }
        Ok(())
    })
}

fn visit_integer(value: IntImm, visitor: &mut VisitContext<'_, NarrowPlan>) -> Result<()> {
    visit_with_expression_context(value.clone().into(), visitor, |visitor| {
        let original = value.ty.clone().try_cast::<PrimType>()?;
        if is_signed_integer(&original) {
            let bits = visitor.state().current_bits;
            visitor.state_mut().record_type(&value, &original, bits);
        }
        Ok(())
    })
}

fn visit_cast(value: Cast, visitor: &mut VisitContext<'_, NarrowPlan>) -> Result<()> {
    visit_with_expression_context(value.clone().into(), visitor, |visitor| {
        let original = value.ty.clone().try_cast::<PrimType>()?;
        if is_signed_integer(&original) {
            let bits = visitor.state().current_bits;
            visitor.state_mut().record_type(&value, &original, bits);
        }
        visitor.visit(&value.value)?;
        Ok(())
    })
}

fn visit_default(
    value: &VisitValue,
    visitor: &mut VisitContext<'_, NarrowPlan>,
) -> Result<Option<VisitInterrupt>> {
    if let Some(expression) = value.cast::<Expr>() {
        visit_with_expression_context(expression, visitor, |visitor| {
            visit_stmt_expr_default(visitor, value)
        })
    } else {
        visit_stmt_expr_default(visitor, value)
    }
}

fn visit_with_expression_context<T>(
    value: Expr,
    visitor: &mut VisitContext<'_, NarrowPlan>,
    operation: impl FnOnce(&mut VisitContext<'_, NarrowPlan>) -> Result<T>,
) -> Result<T> {
    let old_bits = visitor.state().current_bits;
    let mut active_bits = old_bits;
    if let Ok(primitive) = PrimExpr::try_from(value) {
        let ty = primitive.type_annotation();
        if is_signed_integer(&ty) {
            let bound = visitor.state().analyzer.const_int_bound(&primitive)?;
            let target_bits = visitor.state().target_bits;
            let (minimum, maximum) = signed_limits(target_bits);
            let fits = ty.dtype.bits <= target_bits
                || (i128::from(bound.min_value) >= minimum
                    && i128::from(bound.max_value) <= maximum);
            active_bits = old_bits.max(if fits { target_bits } else { 64 });
        }
    }
    visitor.state_mut().current_bits = active_bits;
    let result = operation(visitor);
    visitor.state_mut().current_bits = old_bits;
    result
}

fn signed_limits(bits: u8) -> (i128, i128) {
    if bits >= 64 {
        (i128::from(i64::MIN), i128::from(i64::MAX))
    } else {
        let magnitude = 1_i128 << (bits - 1);
        (-magnitude, magnitude - 1)
    }
}

fn is_signed_integer(ty: &PrimType) -> bool {
    ty.dtype.code == DLDataTypeCode::kDLInt as u8
}
