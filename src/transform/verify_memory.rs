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
    structural_visit, DLDeviceType, Error, ObjectIdentity, ObjectRefCast, ObjectRefCore, Result,
    String, VisitCallbacks, VisitContext, VisitInterrupt, VisitValue, RUNTIME_ERROR,
};

use super::utils::{get_operator, is_opaque_expr, value_error, visit_stmt_expr_default};
use super::{create_module_pass, Pass};
use crate::ir::{Call, CallObj, Expr, TensorLoad, Var};
use crate::target::Target;
use crate::tirx::{AttrStmt, Bind, BufferStore, BufferVar, PrimFunc};

/// Check that host code does not directly access a GPU function's argument buffers.
pub fn verify_memory_prim_func(function: &PrimFunc) -> Result<bool> {
    Ok(memory_errors(function)?.is_empty())
}

/// Build the read-only `tirx.VerifyMemory` module pass.
pub fn verify_memory() -> Result<Pass> {
    create_module_pass("tirx.VerifyMemory", 0, Vec::new(), false, |module| {
        for (_, function) in module.functions.iter() {
            if let Ok(function) = function.try_cast::<PrimFunc>() {
                let errors = memory_errors(&function)?;
                if !errors.is_empty() {
                    return Err(Error::new(
                        RUNTIME_ERROR,
                        &format!("Memory verification failed with the following errors:\n    {}\n  Did you forget to bind?",
                            errors.join("\n    ")),
                        "",
                    ));
                }
            }
        }
        Ok(module)
    })
}

fn memory_errors(function: &PrimFunc) -> Result<Vec<std::string::String>> {
    let Some(target) = function.attrs.dict.get(&String::from("target"))? else {
        return Ok(Vec::new());
    };
    let target = Target::try_from(target)?;
    let calling_conv = function
        .attrs
        .dict
        .get(&String::from("calling_conv"))?
        .map(i64::try_from)
        .transpose()?
        .unwrap_or(0);
    if calling_conv != 0 {
        return Ok(Vec::new());
    }
    let device_type = target.device_type()?;
    let is_gpu = [
        DLDeviceType::kDLCUDA,
        DLDeviceType::kDLOpenCL,
        DLDeviceType::kDLVulkan,
        DLDeviceType::kDLMetal,
        DLDeviceType::kDLROCM,
    ]
    .into_iter()
    .any(|kind| kind as i32 == device_type);
    if !is_gpu {
        return Ok(Vec::new());
    }
    let mut argument_buffers = HashSet::new();
    for parameter in function.params.iter() {
        if BufferVar::try_from(&parameter).is_ok() {
            argument_buffers.insert(ObjectIdentity::of(&parameter));
        }
    }
    let verifier = MemoryVerifier {
        argument_buffers,
        first_parameter: function
            .params
            .iter()
            .next()
            .map(|parameter| ObjectIdentity::of(&parameter)),
        definitions: HashMap::new(),
        in_thread_env: false,
        errors: Vec::new(),
        struct_get: get_operator("tirx.tvm_struct_get")?,
        masked_load: get_operator("tirx.masked_load")?,
        masked_store: get_operator("tirx.masked_store")?,
    };
    let mut callbacks = VisitCallbacks::new(
        verifier,
        (
            visit_bind,
            visit_attribute,
            visit_load,
            visit_store,
            visit_call,
            visit_default,
        ),
    );
    structural_visit(function.body(), &mut callbacks)?;
    Ok(callbacks.into_state().errors)
}

struct MemoryVerifier {
    argument_buffers: HashSet<ObjectIdentity>,
    first_parameter: Option<ObjectIdentity>,
    definitions: HashMap<ObjectIdentity, Expr>,
    in_thread_env: bool,
    errors: Vec<std::string::String>,
    struct_get: Expr,
    masked_load: Expr,
    masked_store: Expr,
}

impl MemoryVerifier {
    fn is_from_function_args(&self, variable: &Var) -> Result<bool> {
        let mut identity = ObjectIdentity::of(variable);
        if self.argument_buffers.contains(&identity) {
            return Ok(true);
        }
        // A valid chain visits each definition at most once.
        for _ in 0..=self.definitions.len() {
            if self.first_parameter.as_ref() == Some(&identity) {
                return Ok(true);
            }
            let Some(definition) = self.definitions.get(&identity) else {
                return Ok(false);
            };
            let Some(call) = definition.as_node::<CallObj>() else {
                return Ok(false);
            };
            if !call.op.same_as(&self.struct_get) {
                return Ok(false);
            }
            let Some(argument) = call.args.iter().next() else {
                return Ok(false);
            };
            let Ok(variable) = argument.try_cast::<Var>() else {
                return Ok(false);
            };
            identity = ObjectIdentity::of(&variable);
        }
        Err(value_error(
            "cyclic tvm_struct_get definitions in memory verification",
        ))
    }

    fn check_access(&mut self, variable: &Var) -> Result<()> {
        if !self.in_thread_env && self.is_from_function_args(variable)? {
            self.errors.push(format!(
                "Variable `{}` is directly accessed by host memory outside a thread environment.",
                variable.name.as_str(),
            ));
        }
        Ok(())
    }
}

fn visit_bind(value: Bind, visitor: &mut VisitContext<'_, MemoryVerifier>) -> Result<()> {
    visitor
        .state_mut()
        .definitions
        .insert(ObjectIdentity::of(&value.var), value.value.clone());
    visitor.visit(&value.value)?;
    Ok(())
}

fn visit_attribute(value: AttrStmt, visitor: &mut VisitContext<'_, MemoryVerifier>) -> Result<()> {
    let previous = visitor.state().in_thread_env;
    visitor.state_mut().in_thread_env |= value.attr_key.as_str() == "thread_extent";
    let result = (|| {
        visitor.visit(&value.value)?;
        visitor.visit(&value.body)?;
        Ok(())
    })();
    visitor.state_mut().in_thread_env = previous;
    result
}

fn visit_load(value: TensorLoad, visitor: &mut VisitContext<'_, MemoryVerifier>) -> Result<()> {
    let source: BufferVar = (&value.source).try_into()?;
    visitor.state_mut().check_access(source.as_var())?;
    for index in value.indices.iter() {
        visitor.visit(&index)?;
    }
    Ok(())
}

fn visit_store(value: BufferStore, visitor: &mut VisitContext<'_, MemoryVerifier>) -> Result<()> {
    visitor.state_mut().check_access(value.buffer.as_var())?;
    visitor.visit(&value.value)?;
    for index in value.indices.iter() {
        visitor.visit(&index)?;
    }
    Ok(())
}

fn visit_call(value: Call, visitor: &mut VisitContext<'_, MemoryVerifier>) -> Result<()> {
    if value.op.same_as(&visitor.state().masked_load)
        || value.op.same_as(&visitor.state().masked_store)
    {
        if let Some(argument) = value.args.iter().next() {
            visitor
                .state_mut()
                .check_access(&argument.try_cast::<Var>()?)?;
        }
    }
    if is_opaque_expr(&value.op) {
        visitor.visit(&value.op)?;
    }
    for argument in value.args.iter() {
        visitor.visit(&argument)?;
    }
    Ok(())
}

fn visit_default(
    value: &VisitValue,
    visitor: &mut VisitContext<'_, MemoryVerifier>,
) -> Result<Option<VisitInterrupt>> {
    visit_stmt_expr_default(visitor, value)
}
