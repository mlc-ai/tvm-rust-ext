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
    structural_mutate, Any, DefRegionKind, Error, Map, ObjectIdentity, ObjectRefCast, Result,
    String as FfiString, StructuralMutator, TYPE_ERROR,
};

use super::utils::{cast_prim_expr, int_value, with_prim_func_body};
use super::{create_prim_func_pass, Pass};
use crate::ir::{Expr, PrimExpr, PrimType, Range, Var};
use crate::tirx::{AttrStmt, For, ForKind, IterVar, IterVarType, PrimFunc, Stmt, StringImm};

const PRAGMA_UNROLL: &str = "pragma_unroll";
const IRREGULAR_LOOP_MARK: &str = "irregular_loop_mark";
const THREAD_EXTENT: &str = "thread_extent";
const VIRTUAL_THREAD: &str = "virtual_thread";

/// Lower opaque loop constructs using the same algorithm as TVM's
/// `tirx::TIRxOpaqueLower`.
pub fn lower_tirx_opaque_prim_func(function: PrimFunc) -> Result<PrimFunc> {
    let mut lowerer = TIRxOpaqueLower::default();
    let body = structural_mutate(function.body.clone(), &mut lowerer)?.try_into()?;
    Ok(with_prim_func_body(function, body))
}

/// Build TVM's `tirx.LowerTIRxOpaque` PrimFunc pass in Rust.
pub fn lower_tirx_opaque() -> Result<Pass> {
    create_prim_func_pass(
        "tirx.LowerTIRxOpaque",
        0,
        Vec::new(),
        false,
        |function, _module, _context| lower_tirx_opaque_prim_func(function),
    )
}

#[derive(Default)]
struct TIRxOpaqueLower {
    unit_loop_values: HashMap<ObjectIdentity, PrimExpr>,
}

#[tvm_ffi::dispatch(mutate)]
impl TIRxOpaqueLower {
    fn mutate_for(&mut self, value: For, region: DefRegionKind) -> Result<Stmt> {
        let minimum = PrimExpr::try_from(self.mutate(&value.min, region)?)?;
        let extent = PrimExpr::try_from(self.mutate(&value.extent, region)?)?;
        let is_unit_loop = int_value(&extent) == Some(1);

        if is_unit_loop && value.annotations.is_empty() {
            self.unit_loop_values
                .insert(ObjectIdentity::of(&value.loop_var), minimum.clone());
        }

        let body = Stmt::try_from(self.mutate(&value.body, region)?)?;
        let LoweredAnnotations {
            preserved: annotations,
            mut pragmas,
        } = lower_annotations(&value.annotations)?;

        let mut lowered = if value.kind == ForKind::kThreadBinding {
            let thread_binding = value.thread_binding.as_ref().ok_or_else(|| {
                Error::new(
                    TYPE_ERROR,
                    "a thread-binding For node requires an IterVar",
                    "",
                )
            })?;
            let thread_tag = thread_binding.thread_tag()?;
            make_launch_thread(
                minimum,
                extent,
                value.loop_var.as_var().clone(),
                thread_tag.as_str(),
                body,
            )?
        } else if is_unit_loop
            && value.annotations.is_empty()
            && !value
                .annotations
                .contains_key(&FfiString::from(IRREGULAR_LOOP_MARK))
        {
            body
        } else {
            For::from_complete_fields(
                None,
                value.loop_var.clone(),
                minimum,
                extent,
                value.kind,
                body,
                None,
                annotations,
                value.step.clone(),
            )
            .into()
        };

        while let Some((key, pragma_value)) = pragmas.pop() {
            lowered =
                AttrStmt::new(value.loop_var.clone(), key.as_str(), pragma_value, lowered)?.into();
        }
        Ok(lowered)
    }

    fn mutate_variable(&mut self, value: Var, region: DefRegionKind) -> Result<Expr> {
        let Some(replacement) = self
            .unit_loop_values
            .get(&ObjectIdentity::of(&value))
            .cloned()
        else {
            return self
                .default_mutate_value(&value, region)
                .and_then(Expr::try_from);
        };

        let variable_type = value.ty.clone().try_cast::<PrimType>()?;
        let replacement_type = replacement.ty.clone().try_cast::<PrimType>()?;
        if variable_type.dtype == replacement_type.dtype {
            Ok(replacement.into())
        } else {
            Ok(cast_prim_expr(replacement, variable_type)?.into())
        }
    }
}

fn make_launch_thread(
    minimum: PrimExpr,
    extent: PrimExpr,
    variable: Var,
    thread_tag: &str,
    body: Stmt,
) -> Result<Stmt> {
    let domain = Range::from_min_extent(minimum, extent.clone())?;
    let iter_var = IterVar::with_metadata(
        Some(domain),
        variable,
        IterVarType::kThreadIndex,
        thread_tag,
        None,
    )?;
    let attr_key = match thread_tag {
        "vthread" | "vthread.x" | "vthread.y" | "vthread.z" => VIRTUAL_THREAD,
        _ => THREAD_EXTENT,
    };
    Ok(AttrStmt::new(iter_var, attr_key, extent, body)?.into())
}

struct LoweredAnnotations {
    preserved: Map<FfiString, Any>,
    pragmas: Vec<(FfiString, PrimExpr)>,
}

fn lower_annotations(annotations: &Map<FfiString, Any>) -> Result<LoweredAnnotations> {
    let mut preserved = Vec::new();
    let mut pragmas = Vec::new();
    for (key, value) in annotations.iter() {
        if key.as_str() == PRAGMA_UNROLL {
            preserved.push((key, value));
        } else if key.as_str().starts_with("pragma_") {
            pragmas.push((key, annotation_value_to_prim_expr(value)?));
        } else {
            preserved.push((key, value));
        }
    }
    pragmas.sort_by(|(lhs, _), (rhs, _)| lhs.as_str().cmp(rhs.as_str()));
    Ok(LoweredAnnotations {
        preserved: Map::from_iter(preserved),
        pragmas,
    })
}

fn annotation_value_to_prim_expr(value: Any) -> Result<PrimExpr> {
    if let Ok(expression) = PrimExpr::try_from(value.clone()) {
        return Ok(expression);
    }
    if let Ok(string) = FfiString::try_from(value) {
        return Ok(StringImm::new(string.as_str()).into());
    }
    Err(Error::new(
        TYPE_ERROR,
        "TIRX pragma values must be primitive expressions or strings",
        "",
    ))
}
