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
    structural_mutate, structural_visit, Any, Array, Error, Map, MapValue, Mutator, ObjectIdentity,
    ObjectRefCast, ObjectRefCore, Result, VisitCallbacks, VisitContext, VisitInterrupt, VisitValue,
    VALUE_ERROR,
};

use super::utils::{
    array_same_as, is_opaque_expr, mutate_buffer_region_with_buffer, mutate_stmt_expr_default,
    visit_stmt_expr_default, with_prim_func_body, BufferRemaps,
};
use super::{create_module_pass, Pass};
use crate::ir::{
    BaseFunc, Call, Expr, GlobalVar, GlobalVarObj, IRModule, PrimExpr, TensorLoad, Var,
};
use crate::tirx::{
    AllocBuffer, BufferRegion, BufferStore, BufferVar, DeclBuffer, Evaluate, PrimFunc, Stmt,
};

type FunctionTable = HashMap<ObjectIdentity, (GlobalVar, PrimFunc)>;

/// Inline non-recursive private PrimFuncs in a module.
pub fn inline_private_functions_module(module: IRModule) -> Result<IRModule> {
    let functions = collect_prim_funcs(&module);
    let recursive = collect_recursive_functions(&functions)?;
    let mut inlinable = FunctionTable::new();
    for (identity, (global, function)) in functions {
        if is_inlinable(&function, &identity, &recursive)? {
            inlinable.insert(identity, (global, function));
        }
    }
    if inlinable.is_empty() {
        return Ok(module);
    }

    let mut inliner = PrimFuncInliner {
        removable: inlinable.keys().cloned().collect(),
        inlinable,
        current_target: None,
        buffer_remaps: BufferRemaps::default(),
    };
    let mut changed = false;
    let mut updated_functions = Vec::with_capacity(module.functions.len());
    for (global, base_function) in module.functions.iter() {
        let updated = if let Ok(function) = base_function.clone().try_cast::<PrimFunc>() {
            let previous =
                std::mem::replace(&mut inliner.current_target, function_target(&function)?);
            let body =
                structural_mutate(function.body.clone(), &mut inliner).and_then(Stmt::try_from);
            inliner.current_target = previous;
            let body = body?;
            let updated = if body.same_as(&function.body) {
                function
            } else {
                with_prim_func_body(function, body)
            };
            changed |= !updated.same_as(&base_function);
            BaseFunc::from(updated)
        } else {
            base_function
        };
        updated_functions.push((global, updated));
    }
    if !changed {
        return Ok(module);
    }

    updated_functions
        .retain(|(global, _)| !inliner.removable.contains(&ObjectIdentity::of(global)));
    let updated = IRModule::with_metadata(
        Map::from_iter(updated_functions),
        module.source_map.clone(),
        module.attrs.clone(),
        module.global_infos.clone(),
    )?;
    super::convert_ssa::convert_ssa_module(updated)
}

/// Build TVM's `tirx.InlinePrivateFunctions` module pass in Rust.
pub fn inline_private_functions() -> Result<Pass> {
    create_module_pass(
        "tirx.InlinePrivateFunctions",
        0,
        Vec::new(),
        false,
        inline_private_functions_module,
    )
}

fn collect_prim_funcs(module: &IRModule) -> FunctionTable {
    module
        .functions
        .iter()
        .filter_map(|(global, function)| {
            function
                .try_cast::<PrimFunc>()
                .ok()
                .map(|function| (ObjectIdentity::of(&global), (global, function)))
        })
        .collect()
}

fn collect_recursive_functions(functions: &FunctionTable) -> Result<HashSet<ObjectIdentity>> {
    let mut call_graph = HashMap::<ObjectIdentity, HashSet<ObjectIdentity>>::new();
    for (caller_identity, (_, function)) in functions {
        let mut collector =
            VisitCallbacks::new(CallGraphState::default(), (visit_call, visit_default));
        structural_visit(&function.body, &mut collector)?;
        call_graph.insert(caller_identity.clone(), collector.into_state().callees);
    }

    let mut recursive = HashSet::new();
    for start in call_graph.keys() {
        let mut pending = call_graph
            .get(start)
            .into_iter()
            .flat_map(|callees| callees.iter().cloned())
            .collect::<Vec<_>>();
        let mut visited = HashSet::new();
        while let Some(current) = pending.pop() {
            if current == *start {
                recursive.insert(start.clone());
                break;
            }
            if visited.insert(current.clone()) {
                if let Some(callees) = call_graph.get(&current) {
                    pending.extend(callees.iter().cloned());
                }
            }
        }
    }
    Ok(recursive)
}

#[derive(Default)]
struct CallGraphState {
    callees: HashSet<ObjectIdentity>,
}

fn visit_call(call: Call, visitor: &mut VisitContext<'_, CallGraphState>) -> Result<()> {
    if call.op.as_node::<GlobalVarObj>().is_some() {
        visitor
            .state_mut()
            .callees
            .insert(ObjectIdentity::of(&call.op));
    }
    if is_opaque_expr(&call.op) {
        visitor.visit(&call.op)?;
    }
    for argument in call.args.iter() {
        visitor.visit(&argument)?;
    }
    Ok(())
}

fn visit_default(
    value: &VisitValue,
    visitor: &mut VisitContext<'_, CallGraphState>,
) -> Result<Option<VisitInterrupt>> {
    visit_stmt_expr_default(visitor, value)
}

fn is_inlinable(
    function: &PrimFunc,
    identity: &ObjectIdentity,
    recursive: &HashSet<ObjectIdentity>,
) -> Result<bool> {
    if function
        .attrs
        .dict
        .get(&tvm_ffi::String::from("global_symbol"))?
        .is_some()
        || recursive.contains(identity)
    {
        return Ok(false);
    }
    for parameter in function.params.iter() {
        if BufferVar::try_from(&parameter).is_ok() {
            return Ok(false);
        }
    }
    Ok(true)
}

struct PrimFuncInliner {
    inlinable: FunctionTable,
    removable: HashSet<ObjectIdentity>,
    current_target: Option<Any>,
    buffer_remaps: BufferRemaps,
}

impl PrimFuncInliner {
    fn targets_match(&self, callee: &PrimFunc) -> Result<bool> {
        let callee_target = function_target(callee)?;
        match (&self.current_target, callee_target) {
            (Some(caller), Some(callee)) => tvm_ffi::cached_global_func!("ffi.StructuralEqual")
                .call_tuple((caller, callee, false, false))?
                .try_into(),
            _ => Ok(true),
        }
    }
}

#[tvm_ffi::dispatch(mutate)]
impl PrimFuncInliner {
    fn mutate_variable(&mut self, value: Var) -> Var {
        self.buffer_remaps.use_variable(&value)
    }

    fn mutate_load(&mut self, value: TensorLoad, mutator: &mut Mutator) -> Result<TensorLoad> {
        let source: BufferVar = (&value.source).try_into()?;
        let source = self.buffer_remaps.use_buffer(&source);
        let indices: Array<PrimExpr> = mutator.mutate(self, &value.indices)?.try_into()?;
        if source.as_var().same_as(&value.source) && array_same_as(&indices, &value.indices) {
            return Ok(value);
        }
        Ok(value.copy_with(source.into(), indices))
    }

    fn mutate_store(&mut self, value: BufferStore, mutator: &mut Mutator) -> Result<BufferStore> {
        let buffer = self.buffer_remaps.use_buffer(&value.buffer);
        let stored_value: PrimExpr = mutator.mutate(self, &value.value)?.try_into()?;
        let indices: Array<PrimExpr> = mutator.mutate(self, &value.indices)?.try_into()?;
        if buffer.same_as(&value.buffer)
            && stored_value.same_as(&value.value)
            && array_same_as(&indices, &value.indices)
        {
            return Ok(value);
        }
        Ok(value.copy_with(buffer, stored_value, indices))
    }

    fn mutate_buffer_region(
        &mut self,
        value: BufferRegion,
        mutator: &mut Mutator,
    ) -> Result<BufferRegion> {
        let buffer = self.buffer_remaps.use_buffer(&value.buffer);
        mutate_buffer_region_with_buffer(self, mutator, value, buffer)
    }

    fn mutate_allocation(
        &mut self,
        value: AllocBuffer,
        mutator: &mut Mutator,
    ) -> Result<AllocBuffer> {
        let buffer = mutate_buffer_definition(self, mutator, &value.buffer)?;
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
        let data: Expr = mutator.mutate(self, &value.data)?.try_into()?;
        let buffer = mutate_buffer_definition(self, mutator, &value.buffer)?;
        if data.same_as(&value.data) && buffer.same_as(&value.buffer) {
            return Ok(value);
        }
        Ok(value.copy_with(buffer, data))
    }

    fn mutate_evaluate(&mut self, value: Evaluate, mutator: &mut Mutator) -> Result<Stmt> {
        if let Some(call) = value.value.as_node::<crate::ir::CallObj>() {
            if let Some(global) = call.op.as_node::<GlobalVarObj>() {
                let identity = ObjectIdentity::of(&call.op);
                if let Some((_, callee)) = self.inlinable.get(&identity).cloned() {
                    if self.targets_match(&callee)? {
                        if callee.params.len() != call.args.len() {
                            return Err(Error::new(
                                VALUE_ERROR,
                                &format!(
                                    "callee {} accepts {} parameters, but received {} arguments",
                                    global.name_hint.as_str(),
                                    callee.params.len(),
                                    call.args.len()
                                ),
                                "",
                            ));
                        }
                        let parameters: Map<Var, Any> = callee
                            .params
                            .iter()
                            .zip(call.args.iter())
                            .map(|(parameter, argument)| (parameter, argument.into()))
                            .collect();
                        let specialized: PrimFunc = tvm_ffi::cached_global_func!("tirx.Specialize")
                            .call_tuple((callee, parameters))?
                            .try_into()?;
                        return mutator.mutate(self, &specialized.body)?.try_into();
                    }
                }
            }
        }
        let evaluated: Expr = mutator.mutate(self, &value.value)?.try_into()?;
        if evaluated.same_as(&value.value) {
            return Ok(value.into());
        }
        Ok(value.copy_with(evaluated).into())
    }

    fn mutate_call(&mut self, value: Call, mutator: &mut Mutator) -> Result<Call> {
        if value.op.as_node::<GlobalVarObj>().is_some() {
            self.removable.remove(&ObjectIdentity::of(&value.op));
        }
        let op = if is_opaque_expr(&value.op) {
            mutator.mutate(self, &value.op)?.try_into()?
        } else {
            value.op.clone()
        };
        let args: Array<Expr> = mutator.mutate(self, &value.args)?.try_into()?;
        if op.same_as(&value.op) && array_same_as(&args, &value.args) {
            return Ok(value);
        }
        Ok(value.copy_with(value.ty.clone(), op, args))
    }

    fn mutate_default(&mut self, value: &MapValue, mutator: &mut Mutator) -> Result<Any> {
        mutate_stmt_expr_default(self, mutator, value)
    }
}

fn mutate_buffer_definition(
    inliner: &mut PrimFuncInliner,
    mutator: &mut Mutator,
    buffer: &BufferVar,
) -> Result<BufferVar> {
    BufferRemaps::mutate_definition(
        inliner,
        buffer,
        |state| &mut state.buffer_remaps,
        |state, expression| mutator.mutate(state, expression)?.try_into(),
    )
}

fn function_target(function: &PrimFunc) -> Result<Option<Any>> {
    function.attrs.dict.get(&tvm_ffi::String::from("target"))
}
