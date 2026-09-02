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
    structural_mutate, structural_visit, Any, Map, MapValue, Mutator, ObjectIdentity,
    ObjectRefCast, Result, String as FfiString, VisitCallbacks, VisitContext, VisitInterrupt,
    VisitValue,
};

use super::utils::{
    mutate_expr_default, mutate_stmt_default, mutate_stmt_expr_default, with_prim_func_attr,
    with_prim_func_body, without_prim_func_attr,
};
use super::{create_module_pass, Pass};
use crate::ir::{BaseFunc, Call, GlobalVar, IRModule, OpaqueExpr};
use crate::target::Target;
use crate::tirx::{AttrStmt, For, ForKind, PrimFunc, Stmt};

const GLOBAL_SYMBOL: &str = "global_symbol";
const TARGET: &str = "target";
const IS_HOST_FUNC: &str = "tirx.is_host_func";
const THREAD_EXTENT: &str = "thread_extent";
const VIRTUAL_THREAD: &str = "virtual_thread";
const DEVICE_ENTRY: &str = "tirx.device_entry";

/// Bind host and device targets to a module using TIRx call-context rules.
pub fn bind_target_module(module: IRModule, target: Target) -> Result<IRModule> {
    let target_host = target.host()?.unwrap_or(Target::new("llvm")?);
    let target_without_host = target.without_host()?;
    let calls = classify_calls(&module)?;
    let mut replacements = HashMap::<ObjectIdentity, GlobalVar>::new();
    let mut used_names = module
        .functions
        .iter()
        .map(|(global, _)| global.name_hint.as_str().to_owned())
        .collect::<HashSet<_>>();
    let mut functions = Vec::with_capacity(module.functions.len());
    let mut additions = Vec::new();

    for (global, function) in module.functions.iter() {
        let Ok(mut primitive) = function.clone().try_cast::<PrimFunc>() else {
            functions.push((global, function));
            continue;
        };
        let externally_exposed = function_attr(&primitive, GLOBAL_SYMBOL)?.is_some();

        if let Some(existing) = function_attr(&primitive, TARGET)? {
            let existing = Target::try_from(existing)?;
            if externally_exposed && existing.host()?.is_none() && target.host()?.is_some() {
                primitive =
                    with_prim_func_attr(primitive, TARGET, existing.with_host(&target_host)?);
            }
            functions.push((global, primitive.into()));
            continue;
        }

        if has_nonzero_attr(&primitive, IS_HOST_FUNC)? {
            primitive = without_prim_func_attr(primitive, IS_HOST_FUNC);
            primitive =
                with_prim_func_attr(primitive, TARGET, target_host.with_host(&target_host)?);
            functions.push((global, primitive.into()));
            continue;
        }

        if externally_exposed {
            primitive = with_prim_func_attr(primitive, TARGET, target.clone());
        } else {
            let identity = ObjectIdentity::of(&global);
            let called_by_host = calls.host.contains(&identity);
            let called_by_device = calls.device.contains(&identity);
            if called_by_host && called_by_device {
                let host_function: PrimFunc = tvm_ffi::cached_global_func!("s_tir.RenewDefs")
                    .call_tuple((primitive.clone(),))?
                    .try_into()?;
                primitive = with_prim_func_attr(primitive, TARGET, target_without_host.clone());
                let host_function = with_prim_func_attr(host_function, TARGET, target_host.clone());
                let base = format!("{}_host", global.name_hint.as_str());
                let name = fresh_name(&base, &mut used_names);
                let host_global = GlobalVar::new(&name);
                replacements.insert(identity, host_global.clone());
                additions.push((host_global, BaseFunc::from(host_function)));
            } else if called_by_host {
                primitive = with_prim_func_attr(primitive, TARGET, target_host.clone());
            } else {
                // Device-only and currently unreferenced private functions both
                // receive the target without its host, matching native TIRx.
                primitive = with_prim_func_attr(primitive, TARGET, target_without_host.clone());
            }
        }
        functions.push((global, primitive.into()));
    }
    functions.extend(additions);

    if !replacements.is_empty() {
        for (_, function) in &mut functions {
            let Ok(primitive) = function.clone().try_cast::<PrimFunc>() else {
                continue;
            };
            if function_attr(&primitive, GLOBAL_SYMBOL)?.is_none() {
                continue;
            }
            let mut substitutor = CallSubstitutor {
                replacements: &replacements,
                under_gpu_scope: false,
            };
            let body: Stmt =
                structural_mutate(primitive.body.clone(), &mut substitutor)?.try_into()?;
            *function = with_prim_func_body(primitive, body).into();
        }
    }

    IRModule::with_metadata(
        Map::from_iter(functions),
        module.source_map.clone(),
        module.attrs.clone(),
        module.global_infos.clone(),
    )
}

/// Build TVM's `tirx.BindTarget` module pass in Rust.
pub fn bind_target(target: Target) -> Result<Pass> {
    create_module_pass("tirx.BindTarget", 0, Vec::new(), false, move |module| {
        bind_target_module(module, target.clone())
    })
}

#[derive(Default)]
struct ClassifiedCalls {
    under_gpu_scope: bool,
    host: HashSet<ObjectIdentity>,
    device: HashSet<ObjectIdentity>,
}

fn classify_calls(module: &IRModule) -> Result<ClassifiedCalls> {
    let mut visitor = VisitCallbacks::new(
        ClassifiedCalls::default(),
        (visit_call, visit_loop, visit_attribute, visit_default),
    );
    for (_, function) in module.functions.iter() {
        let Ok(function) = function.try_cast::<PrimFunc>() else {
            continue;
        };
        if function_attr(&function, GLOBAL_SYMBOL)?.is_some() {
            structural_visit(&function.body, &mut visitor)?;
        }
    }
    Ok(visitor.into_state())
}

fn visit_call(value: Call, visitor: &mut VisitContext<'_, ClassifiedCalls>) -> Result<()> {
    if let Ok(global) = value.op.clone().try_cast::<GlobalVar>() {
        let identity = ObjectIdentity::of(&global);
        if visitor.state().under_gpu_scope {
            visitor.state_mut().device.insert(identity);
        } else {
            visitor.state_mut().host.insert(identity);
        }
    }
    if value.op.clone().try_cast::<OpaqueExpr>().is_ok() {
        visitor.visit(&value.op)?;
    }
    for argument in value.args.iter() {
        visitor.visit(&argument)?;
    }
    Ok(())
}

fn visit_loop(value: For, visitor: &mut VisitContext<'_, ClassifiedCalls>) -> Result<()> {
    visit_scoped(value.kind == ForKind::kThreadBinding, visitor, |visitor| {
        visitor.visit(&value.min)?;
        visitor.visit(&value.extent)?;
        if let Some(step) = &value.step {
            visitor.visit(step)?;
        }
        visitor.visit(&value.body)?;
        Ok(())
    })
}

fn visit_attribute(value: AttrStmt, visitor: &mut VisitContext<'_, ClassifiedCalls>) -> Result<()> {
    visit_scoped(is_gpu_attribute(&value), visitor, |visitor| {
        visitor.visit(&value.value)?;
        visitor.visit(&value.body)?;
        Ok(())
    })
}

fn visit_default(
    value: &VisitValue,
    visitor: &mut VisitContext<'_, ClassifiedCalls>,
) -> Result<Option<VisitInterrupt>> {
    super::utils::visit_stmt_expr_default(visitor, value)
}

fn visit_scoped<F>(
    enter_gpu: bool,
    visitor: &mut VisitContext<'_, ClassifiedCalls>,
    operation: F,
) -> Result<()>
where
    F: FnOnce(&mut VisitContext<'_, ClassifiedCalls>) -> Result<()>,
{
    let old = visitor.state().under_gpu_scope;
    if enter_gpu {
        visitor.state_mut().under_gpu_scope = true;
    }
    let result = operation(visitor);
    visitor.state_mut().under_gpu_scope = old;
    result
}

struct CallSubstitutor<'a> {
    replacements: &'a HashMap<ObjectIdentity, GlobalVar>,
    under_gpu_scope: bool,
}

#[tvm_ffi::dispatch(mutate)]
impl CallSubstitutor<'_> {
    fn mutate_call(&mut self, value: Call, mutator: &mut Mutator) -> Result<Call> {
        let expression = mutate_expr_default(self, mutator, value.clone().into())?;
        let call = expression.try_cast::<Call>()?;
        if self.under_gpu_scope {
            return Ok(call);
        }
        let Ok(global) = call.op.clone().try_cast::<GlobalVar>() else {
            return Ok(call);
        };
        let Some(replacement) = self.replacements.get(&ObjectIdentity::of(&global)) else {
            return Ok(call);
        };
        Ok(Call::from_complete_fields(
            call.span.clone(),
            call.ty.clone(),
            replacement.clone().into(),
            call.args.clone(),
            call.attrs.clone(),
            call.ty_args.clone(),
        ))
    }

    fn mutate_loop(&mut self, value: For, mutator: &mut Mutator) -> Result<For> {
        let old = self.under_gpu_scope;
        if value.kind == ForKind::kThreadBinding {
            self.under_gpu_scope = true;
        }
        let result = mutate_stmt_default(self, mutator, value.into())?.try_cast();
        self.under_gpu_scope = old;
        result
    }

    fn mutate_attribute(&mut self, value: AttrStmt, mutator: &mut Mutator) -> Result<AttrStmt> {
        let old = self.under_gpu_scope;
        if is_gpu_attribute(&value) {
            self.under_gpu_scope = true;
        }
        let result = mutate_stmt_default(self, mutator, value.into())?.try_cast();
        self.under_gpu_scope = old;
        result
    }

    fn mutate_default(&mut self, value: &MapValue, mutator: &mut Mutator) -> Result<Any> {
        mutate_stmt_expr_default(self, mutator, value)
    }
}

fn is_gpu_attribute(value: &AttrStmt) -> bool {
    matches!(
        value.attr_key.as_str(),
        THREAD_EXTENT | VIRTUAL_THREAD | DEVICE_ENTRY
    )
}

fn function_attr(function: &PrimFunc, key: &str) -> Result<Option<Any>> {
    function.attrs.dict.get(&FfiString::from(key))
}

fn has_nonzero_attr(function: &PrimFunc, key: &str) -> Result<bool> {
    function_attr(function, key)?
        .map(i64::try_from)
        .transpose()
        .map(|value| value.unwrap_or(0) != 0)
}

fn fresh_name(base: &str, used: &mut HashSet<String>) -> String {
    if used.insert(base.to_owned()) {
        return base.to_owned();
    }
    let mut suffix = 1_u64;
    loop {
        let candidate = format!("{base}_{suffix}");
        if used.insert(candidate.clone()) {
            return candidate;
        }
        suffix += 1;
    }
}
