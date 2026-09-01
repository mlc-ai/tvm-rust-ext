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

use std::collections::HashSet;

use tvm_ffi::derive::{Object, ObjectRef};
use tvm_ffi::{
    structural_mutate, structural_walk, Any, Array, DefRegionKind, Error, FieldGetter, Map,
    MapValue, ObjectArc, ObjectCore, ObjectIdentity, ObjectRefCast, ObjectRefCore, Result, String,
    StructuralMutator, WalkOrder, WalkResult, VALUE_ERROR,
};

use super::utils::{array_same_as, mutate_stmt_expr_default, option_same_as, with_prim_func_body};
use super::{create_prim_func_pass_with_context, Pass, PassContext};
use crate::analysis::Analyzer;
use crate::ir::{Expr, IntImm, PrimExpr, PrimType, TensorLoad, Var};
use crate::tirx::{
    Add, AttrStmt, BufferStore, BufferType, BufferVar, Evaluate, For, ForKind, PrimFunc, SeqStmt,
    Stmt,
};

const AUTO_UNROLL_MAX_STEP: &str = "pragma_auto_unroll_max_step";
const UNROLL_EXPLICIT: &str = "pragma_unroll_explicit";

/// Opaque read-only view of TVM's `tirx.transform.UnrollLoopConfig`.
#[repr(C)]
#[derive(Object)]
#[type_key = "tirx.transform.UnrollLoopConfig"]
#[type_final]
struct UnrollLoopConfigObj {
    base: tvm_ffi::Object,
}

#[repr(C)]
#[derive(ObjectRef, Clone)]
struct UnrollLoopConfig {
    data: ObjectArc<UnrollLoopConfigObj>,
}

impl std::ops::Deref for UnrollLoopConfig {
    type Target = UnrollLoopConfigObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl UnrollLoopConfig {
    fn field<T>(&self, name: &str) -> Result<T>
    where
        T: TryFrom<Any, Error = tvm_ffi::Error>,
    {
        FieldGetter::new(UnrollLoopConfigObj::type_index(), name)?.get(&**self)
    }
}

#[derive(Clone, Copy)]
struct UnrollOptions {
    auto_max_step: i32,
    auto_max_depth: i32,
    auto_max_extent: i32,
    explicit_unroll: bool,
    unroll_local_access: bool,
}

impl Default for UnrollOptions {
    fn default() -> Self {
        Self {
            auto_max_step: 0,
            auto_max_depth: 8,
            auto_max_extent: 0,
            explicit_unroll: true,
            unroll_local_access: false,
        }
    }
}

impl UnrollOptions {
    fn from_context(context: &PassContext) -> Result<Self> {
        let Some(raw) = context.config()?.get(&String::from("tirx.UnrollLoop"))? else {
            return Ok(Self::default());
        };
        let config = UnrollLoopConfig::try_from(raw)?;
        Ok(Self {
            auto_max_step: checked_i32_config(config.field("auto_max_step")?, "auto_max_step")?,
            auto_max_depth: checked_i32_config(config.field("auto_max_depth")?, "auto_max_depth")?,
            auto_max_extent: checked_i32_config(
                config.field("auto_max_extent")?,
                "auto_max_extent",
            )?,
            explicit_unroll: config.field::<i64>("explicit_unroll")? != 0,
            unroll_local_access: config.field::<i64>("unroll_local_access")? != 0,
        })
    }
}

/// Unroll loops in one PrimFunc using TVM's default configuration.
pub fn unroll_loop_prim_func(function: PrimFunc) -> Result<PrimFunc> {
    unroll_loop_with_options(function, UnrollOptions::default())
}

fn unroll_loop_with_options(function: PrimFunc, options: UnrollOptions) -> Result<PrimFunc> {
    let original_body = function.body.clone();
    let mut unroller = LoopUnroller {
        analyzer: Analyzer::new()?,
        options,
        normal_loop_depth: 0,
        unroll_depth: 0,
        step_count: 0,
        variables_touching_local: HashSet::new(),
        changed: false,
    };
    let body: Stmt = structural_mutate(function.body.clone(), &mut unroller)?.try_into()?;
    if !unroller.changed && body.same_as(&original_body) {
        return Ok(function);
    }
    let body = super::convert_ssa::convert_ssa_stmt(body)?;
    Ok(with_prim_func_body(function, body))
}

/// Build TVM's `tirx.UnrollLoop` PrimFunc pass in Rust.
pub fn unroll_loop() -> Result<Pass> {
    create_prim_func_pass_with_context(
        "tirx.UnrollLoop",
        0,
        Vec::new(),
        false,
        |function, context| {
            unroll_loop_with_options(function, UnrollOptions::from_context(&context)?)
        },
    )
}

struct LoopUnroller {
    analyzer: Analyzer,
    options: UnrollOptions,
    normal_loop_depth: i32,
    unroll_depth: i32,
    step_count: i32,
    variables_touching_local: HashSet<ObjectIdentity>,
    changed: bool,
}

#[tvm_ffi::dispatch(mutate)]
impl LoopUnroller {
    fn mutate_attribute(&mut self, value: AttrStmt, region: DefRegionKind) -> Result<Stmt> {
        match value.attr_key.as_str() {
            AUTO_UNROLL_MAX_STEP => {
                let replacement = literal_value(&value.value).ok_or_else(|| {
                    Error::new(
                        VALUE_ERROR,
                        "pragma_auto_unroll_max_step must be an integer literal",
                        "",
                    )
                })? as i32;
                let previous = std::mem::replace(&mut self.options.auto_max_step, replacement);
                let body = self.mutate(&value.body, region).and_then(Stmt::try_from);
                self.options.auto_max_step = previous;
                self.changed = true;
                body
            }
            UNROLL_EXPLICIT => {
                let replacement = literal_value(&value.value).ok_or_else(|| {
                    Error::new(
                        VALUE_ERROR,
                        "pragma_unroll_explicit must be an integer literal",
                        "",
                    )
                })? != 0;
                let previous = std::mem::replace(&mut self.options.explicit_unroll, replacement);
                let body = self.mutate(&value.body, region).and_then(Stmt::try_from);
                self.options.explicit_unroll = previous;
                self.changed = true;
                body
            }
            _ => self
                .rewrite_regular_attribute(value, region)
                .map(Into::into),
        }
    }

    fn mutate_for(&mut self, value: For, region: DefRegionKind) -> Result<Stmt> {
        let mutated = self.rewrite_loop_children(value, region)?;
        let extent = self.constant_extent(&mutated)?;
        let mut automatic = mutated.kind == ForKind::kSerial
            && extent >= 0
            && self.normal_loop_depth == 0
            && self.unroll_depth <= self.options.auto_max_depth;
        automatic = automatic
            && (extent * self.step_count <= self.options.auto_max_step
                || extent <= self.options.auto_max_extent);

        if mutated.kind == ForKind::kUnrolled {
            if extent < 0 {
                return Err(Error::new(
                    VALUE_ERROR,
                    "cannot unroll a loop with a non-constant extent",
                    "",
                ));
            }
            automatic = true;
        }

        if extent > 0
            && self.options.unroll_local_access
            && self
                .variables_touching_local
                .contains(&ObjectIdentity::of(mutated.loop_var.as_var()))
        {
            automatic = true;
        }

        if automatic {
            self.step_count *= extent;
            self.unroll_depth += 1;
        } else {
            self.normal_loop_depth += 1;
        }

        if (automatic && self.options.explicit_unroll)
            || (0 <= extent
                && extent <= self.options.auto_max_extent
                && self.options.auto_max_extent == 1)
        {
            self.changed = true;
            return self.unroll(&mutated, extent);
        }

        if automatic && mutated.kind != ForKind::kUnrolled {
            self.changed = true;
            return Ok(For::from_complete_fields(
                mutated.span.clone(),
                mutated.loop_var.clone(),
                mutated.min.clone(),
                mutated.extent.clone(),
                ForKind::kUnrolled,
                mutated.body.clone(),
                mutated.thread_binding.clone(),
                mutated.annotations.clone(),
                mutated.step.clone(),
            )
            .into());
        }
        Ok(mutated.into())
    }

    fn mutate_load(&mut self, value: TensorLoad) -> Result<TensorLoad> {
        if self.options.unroll_local_access {
            let variable = value.source.clone().try_cast::<Var>()?;
            let buffer = BufferVar::try_from(variable)?;
            if is_local_or_warp(&buffer)? {
                self.record_index_variables(&value.indices)?;
            }
        }
        Ok(value)
    }

    fn mutate_store(&mut self, value: BufferStore, region: DefRegionKind) -> Result<BufferStore> {
        self.step_count += 1;
        if self.options.unroll_local_access && is_local_or_warp(&value.buffer)? {
            self.record_index_variables(&value.indices)?;
        }
        let stored_value: PrimExpr = self.mutate(&value.value, region)?.try_into()?;
        let indices: Array<PrimExpr> = self.mutate(&value.indices, region)?.try_into()?;
        if stored_value.same_as(&value.value) && array_same_as(&indices, &value.indices) {
            return Ok(value);
        }
        Ok(BufferStore::from_complete_fields(
            value.span.clone(),
            value.buffer.clone(),
            stored_value,
            indices,
        ))
    }

    fn mutate_evaluate(&mut self, value: Evaluate, region: DefRegionKind) -> Result<Evaluate> {
        self.step_count += 1;
        let expression: Expr = self.mutate(&value.value, region)?.try_into()?;
        if expression.same_as(&value.value) {
            return Ok(value);
        }
        Ok(Evaluate::from_complete_fields(
            value.span.clone(),
            expression,
        ))
    }

    fn mutate_sequence(&mut self, value: SeqStmt, region: DefRegionKind) -> Result<SeqStmt> {
        let mut changed = false;
        let mut sequence = Vec::with_capacity(value.seq.len());
        for statement in value.seq.iter() {
            let saved_steps = self.step_count;
            let saved_unroll_depth = self.unroll_depth;
            let saved_normal_depth = self.normal_loop_depth;
            self.step_count = 0;
            self.unroll_depth = 0;
            self.normal_loop_depth = 0;

            let mutated: Stmt = self.mutate(&statement, region)?.try_into()?;
            changed |= !mutated.same_as(&statement);
            sequence.push(mutated);

            self.step_count += saved_steps;
            self.unroll_depth = self.unroll_depth.max(saved_unroll_depth);
            self.normal_loop_depth = self.normal_loop_depth.max(saved_normal_depth);
        }
        if changed {
            Ok(SeqStmt::from_complete_fields(
                value.span.clone(),
                Array::new(sequence),
            ))
        } else {
            Ok(value)
        }
    }

    fn mutate_stmt_expr_default(&mut self, value: &MapValue, region: DefRegionKind) -> Result<Any> {
        mutate_stmt_expr_default(self, value, region)
    }
}

impl LoopUnroller {
    fn rewrite_regular_attribute(
        &mut self,
        value: AttrStmt,
        region: DefRegionKind,
    ) -> Result<AttrStmt> {
        let attr_value: PrimExpr = self.mutate(&value.value, region)?.try_into()?;
        let body: Stmt = self.mutate(&value.body, region)?.try_into()?;
        if attr_value.same_as(&value.value) && body.same_as(&value.body) {
            return Ok(value);
        }
        Ok(AttrStmt::from_complete_fields(
            value.span.clone(),
            value.node.clone(),
            value.attr_key.clone(),
            attr_value,
            body,
        ))
    }

    fn rewrite_loop_children(&mut self, value: For, region: DefRegionKind) -> Result<For> {
        let minimum: PrimExpr = self.mutate(&value.min, region)?.try_into()?;
        let extent: PrimExpr = self.mutate(&value.extent, region)?.try_into()?;
        let step: Option<PrimExpr> = self.mutate(&value.step, region)?.try_into()?;
        let body: Stmt = self.mutate(&value.body, region)?.try_into()?;
        if minimum.same_as(&value.min)
            && extent.same_as(&value.extent)
            && option_same_as(&step, &value.step)
            && body.same_as(&value.body)
        {
            return Ok(value);
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
        ))
    }
}

impl LoopUnroller {
    fn constant_extent(&self, loop_node: &For) -> Result<i32> {
        let simplified = self.analyzer.simplify(&loop_node.extent)?;
        Ok(literal_value(&simplified)
            .and_then(|value| i32::try_from(value).ok())
            .unwrap_or(-1))
    }

    fn unroll(&self, loop_node: &For, extent: i32) -> Result<Stmt> {
        if extent < 0 {
            return Err(Error::new(
                VALUE_ERROR,
                "loop does not have a constant integer extent",
                "",
            ));
        }
        if extent == 0 {
            return Ok(Evaluate::from_i64(0)?.into());
        }
        let loop_type = loop_node.loop_var.ty.clone().try_cast::<PrimType>()?;
        let mut unrolled = Vec::with_capacity(extent as usize);
        for offset in 0..extent {
            let offset =
                IntImm::from_complete_fields(None, loop_type.clone(), i64::from(offset)).into();
            let replacement = add_with_constant_folding(&loop_node.min, offset)?;
            let replacements = Map::<Var, Expr>::from_iter([(
                loop_node.loop_var.as_var().clone(),
                replacement.into(),
            )]);
            let step: Stmt = tvm_ffi::cached_global_func!("tirx.Substitute")
                .call_tuple((loop_node.body.clone(), replacements))?
                .try_into()?;
            unrolled.push(step);
        }
        Stmt::sequence(unrolled)
    }

    fn record_index_variables(&mut self, indices: &Array<PrimExpr>) -> Result<()> {
        for index in indices.iter() {
            structural_walk(
                &index,
                |variable: Var| {
                    self.variables_touching_local
                        .insert(ObjectIdentity::of(&variable));
                    WalkResult::Advance
                },
                WalkOrder::PreOrder,
            )?;
        }
        Ok(())
    }
}

fn literal_value(value: &PrimExpr) -> Option<i64> {
    value
        .clone()
        .try_cast::<IntImm>()
        .ok()
        .map(|literal| literal.value)
}

fn checked_i32_config(value: i64, field: &str) -> Result<i32> {
    i32::try_from(value).map_err(|_| {
        Error::new(
            VALUE_ERROR,
            &format!("tirx.UnrollLoop {field} does not fit the native i32 field"),
            "",
        )
    })
}

fn add_with_constant_folding(lhs: &PrimExpr, rhs: PrimExpr) -> Result<PrimExpr> {
    let lhs_value = literal_value(lhs);
    let rhs_value = literal_value(&rhs);
    if rhs_value == Some(0) {
        return Ok(lhs.clone());
    }
    if lhs_value == Some(0) {
        return Ok(rhs);
    }
    if let (Some(lhs_value), Some(rhs_value)) = (lhs_value, rhs_value) {
        let result_type = lhs.ty.clone().try_cast::<PrimType>()?;
        let dtype = result_type.dtype;
        let mut result = lhs_value.wrapping_add(rhs_value);
        if dtype.bits < 64 {
            result &= (1_i64 << dtype.bits) - 1;
        }
        if dtype.bits < 64 && dtype.code == tvm_ffi::DLDataTypeCode::kDLInt as u8 {
            let sign = 1_i64 << (dtype.bits - 1);
            result = (result ^ sign) - sign;
        }
        return Ok(IntImm::from_complete_fields(None, result_type, result).into());
    }
    Ok(Add::new(lhs.clone(), rhs)?.into())
}

fn is_local_or_warp(buffer: &BufferVar) -> Result<bool> {
    let buffer_type = buffer.ty.clone().try_cast::<BufferType>()?;
    let scope = buffer_type.storage_scope.as_str();
    Ok(scope.starts_with("local") || scope.starts_with("warp"))
}
