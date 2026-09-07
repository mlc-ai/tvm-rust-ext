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

use std::collections::{hash_map::Entry, HashMap};

use tvm_ffi::{
    structural_visit, Error, ObjectIdentity, ObjectRefCast, Result, VisitCallbacks, VisitContext,
    VisitInterrupt, VisitValue, RUNTIME_ERROR,
};

use super::utils::{visit_buffer_definition, visit_stmt_expr_default};
use super::{create_module_pass, Pass};
use crate::ir::prim::Let;
use crate::ir::{Expr, PrimExpr, Var};
use crate::tirx::{AllocBuffer, Bind, BufferVar, For, PrimFunc};

/// Check TVM's SSA rules, including repeated Let bindings to deeply equal values.
pub fn verify_ssa_prim_func(function: &PrimFunc) -> Result<bool> {
    let mut verifier = SsaVerifier::default();
    for parameter in function.params.iter() {
        if !verifier.define(&parameter, parameter.clone().into()) {
            return Ok(false);
        }
    }
    let mut callbacks = VisitCallbacks::new(
        verifier,
        (
            visit_let,
            visit_bind,
            visit_loop,
            visit_allocation,
            visit_variable,
            visit_default,
        ),
    );
    callbacks.state_mut().match_scope = true;
    for parameter in function.params.iter() {
        if let Ok(buffer) = BufferVar::try_from(parameter) {
            if structural_visit(buffer.as_var(), &mut callbacks)?.is_some() {
                return Ok(false);
            }
            let ty = buffer.type_annotation();
            for expression in ty
                .shape
                .iter()
                .chain(ty.strides.iter())
                .chain([ty.elem_offset.clone()])
            {
                if structural_visit(&expression, &mut callbacks)?.is_some() {
                    return Ok(false);
                }
            }
        }
    }
    callbacks.state_mut().match_scope = false;
    Ok(structural_visit(function.body(), &mut callbacks)?.is_none())
}

/// Build the read-only `tirx.VerifySSA` module pass.
pub fn verify_ssa() -> Result<Pass> {
    create_module_pass("tirx.VerifySSA", 0, Vec::new(), false, |module| {
        for (_, function) in module.functions.iter() {
            if let Ok(function) = function.try_cast::<PrimFunc>() {
                if !verify_ssa_prim_func(&function)? {
                    return Err(Error::new(RUNTIME_ERROR, "IR is not in SSA form", ""));
                }
            }
        }
        Ok(module)
    })
}

#[derive(Default)]
struct SsaVerifier {
    definitions: HashMap<ObjectIdentity, Expr>,
    match_scope: bool,
}

impl SsaVerifier {
    fn define(&mut self, variable: &Var, value: Expr) -> bool {
        match self.definitions.entry(ObjectIdentity::of(variable)) {
            Entry::Vacant(entry) => {
                entry.insert(value);
                true
            }
            Entry::Occupied(_) => false,
        }
    }
}

fn visit_let(
    value: Let,
    visitor: &mut VisitContext<'_, SsaVerifier>,
) -> Result<Option<VisitInterrupt>> {
    if let Some(previous) = visitor
        .state()
        .definitions
        .get(&ObjectIdentity::of(&value.var))
    {
        // StructuralEqual may remap bound variables; native VerifySSA must not.
        let previous: PrimExpr = previous.try_into()?;
        let equal: bool = tvm_ffi::cached_global_func!("tirx.analysis.expr_deep_equal")
            .call_tuple((previous, &value.value))?
            .try_into()?;
        if !equal {
            return Ok(Some(VisitInterrupt::with(false)));
        }
    } else {
        visitor
            .state_mut()
            .define(&value.var, value.value.clone().into());
    }
    if let Some(interrupt) = visitor.visit(&value.value)? {
        return Ok(Some(interrupt));
    }
    visitor.visit(&value.body)
}

fn visit_bind(
    value: Bind,
    visitor: &mut VisitContext<'_, SsaVerifier>,
) -> Result<Option<VisitInterrupt>> {
    if !visitor.state_mut().define(&value.var, value.value.clone()) {
        return Ok(Some(VisitInterrupt::with(false)));
    }
    visitor.visit(&value.value)
}

fn visit_loop(
    value: For,
    visitor: &mut VisitContext<'_, SsaVerifier>,
) -> Result<Option<VisitInterrupt>> {
    let variable = value.loop_var.as_var();
    if !visitor
        .state_mut()
        .define(variable, variable.clone().into())
    {
        return Ok(Some(VisitInterrupt::with(false)));
    }
    for expression in [&value.min, &value.extent]
        .into_iter()
        .chain(value.step.iter())
    {
        if let Some(interrupt) = visitor.visit(expression)? {
            return Ok(Some(interrupt));
        }
    }
    visitor.visit(&value.body)
}

fn visit_allocation(
    value: AllocBuffer,
    visitor: &mut VisitContext<'_, SsaVerifier>,
) -> Result<Option<VisitInterrupt>> {
    let variable = value.buffer.as_var();
    if !visitor
        .state_mut()
        .define(variable, variable.clone().into())
    {
        return Ok(Some(VisitInterrupt::with(false)));
    }
    visit_buffer_definition(visitor, &value.buffer)
}

fn visit_variable(value: Var, visitor: &mut VisitContext<'_, SsaVerifier>) {
    if visitor.state().match_scope {
        visitor.state_mut().define(&value, value.clone().into());
    }
}

fn visit_default(
    value: &VisitValue,
    visitor: &mut VisitContext<'_, SsaVerifier>,
) -> Result<Option<VisitInterrupt>> {
    visit_stmt_expr_default(visitor, value)
}
