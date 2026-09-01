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
    structural_mutate, structural_visit, Any, Array, DefRegionKind, Error, Map, MapValue,
    ObjectIdentity, ObjectRefCast, ObjectRefCore, Result, StructuralMutator, StructuralVisitor,
    VisitInterrupt, VisitValue, VALUE_ERROR,
};

use super::utils::{
    array_same_as, mutate_stmt_expr_default, visit_stmt_expr_default, BufferRemaps,
};
use super::{create_module_pass, Pass};
use crate::ir::{BaseFunc, Call, Expr, GlobalVar, IRModule, OpaqueExpr, PrimExpr, TensorLoad, Var};
use crate::tirx::{
    AllocBuffer, AttrStmt, BufferStore, BufferType, BufferVar, DeclBuffer, Evaluate, For, PrimFunc,
    Stmt,
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
            let updated = inliner.visit_function(function)?;
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
        let mut collector = CallGraphCollector::default();
        structural_visit(&function.body, &mut collector)?;
        call_graph.insert(caller_identity.clone(), collector.callees);
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
struct CallGraphCollector {
    callees: HashSet<ObjectIdentity>,
}

#[tvm_ffi::dispatch(visit)]
impl CallGraphCollector {
    fn visit_call(&mut self, call: Call, region: DefRegionKind) -> Result<()> {
        if let Ok(global) = call.op.clone().try_cast::<GlobalVar>() {
            self.callees.insert(ObjectIdentity::of(&global));
        }
        if call.op.clone().try_cast::<OpaqueExpr>().is_ok() {
            self.visit_child(&call.op, region)?;
        }
        for argument in call.args.iter() {
            self.visit_child(&argument, region)?;
        }
        Ok(())
    }

    fn visit_attribute(&mut self, value: AttrStmt, region: DefRegionKind) -> Result<()> {
        self.visit_child(&value.value, region)?;
        self.visit_child(&value.body, region)?;
        Ok(())
    }

    fn visit_loop(&mut self, value: For, region: DefRegionKind) -> Result<()> {
        self.visit_child(&value.min, region)?;
        self.visit_child(&value.extent, region)?;
        if let Some(step) = &value.step {
            self.visit_child(step, region)?;
        }
        self.visit_child(&value.body, region)?;
        Ok(())
    }

    fn visit_stmt_expr_default(
        &mut self,
        value: &VisitValue,
        region: DefRegionKind,
    ) -> Result<Option<VisitInterrupt>> {
        visit_stmt_expr_default(self, value, region)
    }
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
        if parameter.ty.clone().try_cast::<BufferType>().is_ok() {
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
    fn mutate_buffer_definition(
        &mut self,
        buffer: &BufferVar,
        region: DefRegionKind,
    ) -> Result<BufferVar> {
        let mut remaps = std::mem::take(&mut self.buffer_remaps);
        let result = remaps.mutate_definition(buffer, |expression| {
            self.mutate(expression, region)?.try_into()
        });
        self.buffer_remaps = remaps;
        result
    }

    fn visit_function(&mut self, function: PrimFunc) -> Result<PrimFunc> {
        let previous = std::mem::replace(&mut self.current_target, function_target(&function)?);
        let body = structural_mutate(function.body.clone(), &mut *self).and_then(Stmt::try_from);
        self.current_target = previous;
        let body = body?;
        if body.same_as(&function.body) {
            Ok(function)
        } else {
            Ok(super::utils::with_prim_func_body(function, body))
        }
    }

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

    fn mutate_load(&mut self, value: TensorLoad, region: DefRegionKind) -> Result<TensorLoad> {
        let source = BufferVar::try_from(value.source.clone().try_cast::<Var>()?)?;
        let source = self.buffer_remaps.use_buffer(&source);
        let indices: Array<PrimExpr> = self.mutate(&value.indices, region)?.try_into()?;
        if source.as_var().same_as(&value.source) && array_same_as(&indices, &value.indices) {
            return Ok(value);
        }
        Ok(TensorLoad::from_complete_fields(
            value.span.clone(),
            value.ty.clone().try_cast()?,
            source.into(),
            indices,
        ))
    }

    fn mutate_store(&mut self, value: BufferStore, region: DefRegionKind) -> Result<BufferStore> {
        let buffer = self.buffer_remaps.use_buffer(&value.buffer);
        let stored_value: PrimExpr = self.mutate(&value.value, region)?.try_into()?;
        let indices: Array<PrimExpr> = self.mutate(&value.indices, region)?.try_into()?;
        if buffer.same_as(&value.buffer)
            && stored_value.same_as(&value.value)
            && array_same_as(&indices, &value.indices)
        {
            return Ok(value);
        }
        Ok(BufferStore::from_complete_fields(
            value.span.clone(),
            buffer,
            stored_value,
            indices,
        ))
    }

    fn mutate_alloc_buffer(
        &mut self,
        value: AllocBuffer,
        region: DefRegionKind,
    ) -> Result<AllocBuffer> {
        let buffer = self.mutate_buffer_definition(&value.buffer, region)?;
        if buffer.same_as(&value.buffer) {
            return Ok(value);
        }
        Ok(AllocBuffer::from_complete_fields(
            value.span.clone(),
            buffer,
            value.annotations.clone(),
        ))
    }

    fn mutate_decl_buffer(
        &mut self,
        value: DeclBuffer,
        region: DefRegionKind,
    ) -> Result<DeclBuffer> {
        let data: Expr = self.mutate(&value.data, region)?.try_into()?;
        let buffer = self.mutate_buffer_definition(&value.buffer, region)?;
        if data.same_as(&value.data) && buffer.same_as(&value.buffer) {
            return Ok(value);
        }
        Ok(DeclBuffer::from_complete_fields(
            value.span.clone(),
            buffer,
            data,
        ))
    }

    fn mutate_evaluate(&mut self, value: Evaluate, region: DefRegionKind) -> Result<Stmt> {
        if let Ok(call) = value.value.clone().try_cast::<Call>() {
            if let Ok(global) = call.op.clone().try_cast::<GlobalVar>() {
                let identity = ObjectIdentity::of(&global);
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
                        let parameters = Map::<Var, Any>::from_iter(
                            callee
                                .params
                                .iter()
                                .zip(call.args.iter())
                                .map(|(parameter, argument)| (parameter, Any::from(argument))),
                        );
                        let specialized: PrimFunc = tvm_ffi::cached_global_func!("tirx.Specialize")
                            .call_tuple((callee, parameters))?
                            .try_into()?;
                        return self.mutate(&specialized.body, region)?.try_into();
                    }
                }
            }
        }
        let evaluated: Expr = self.mutate(&value.value, region)?.try_into()?;
        if evaluated.same_as(&value.value) {
            return Ok(value.into());
        }
        Ok(Evaluate::from_complete_fields(value.span.clone(), evaluated).into())
    }

    fn mutate_call(&mut self, value: Call, region: DefRegionKind) -> Result<Call> {
        if let Ok(global) = value.op.clone().try_cast::<GlobalVar>() {
            self.removable.remove(&ObjectIdentity::of(&global));
        }
        let op = if value.op.clone().try_cast::<OpaqueExpr>().is_ok() {
            self.mutate(&value.op, region)?.try_into()?
        } else {
            value.op.clone()
        };
        let args: Array<Expr> = self.mutate(&value.args, region)?.try_into()?;
        if op.same_as(&value.op) && array_same_as(&args, &value.args) {
            return Ok(value);
        }
        Ok(Call::from_complete_fields(
            value.span.clone(),
            value.ty.clone(),
            op,
            args,
            value.attrs.clone(),
            value.ty_args.clone(),
        ))
    }

    fn mutate_stmt_expr_default(&mut self, value: &MapValue, region: DefRegionKind) -> Result<Any> {
        mutate_stmt_expr_default(self, value, region)
    }
}

fn function_target(function: &PrimFunc) -> Result<Option<Any>> {
    function.attrs.dict.get(&tvm_ffi::String::from("target"))
}
