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
    structural_mutate, structural_walk, Any, Array, Map, MapValue, Mutator, ObjectIdentity,
    ObjectRefCast, ObjectRefCore, Result, String as FfiString, WalkOrder, WalkResult,
};

use super::utils::{
    get_operator, int_value, is_buffer_type, is_pointer_type, mutate_stmt_default,
    mutate_stmt_expr_default, value_error, with_prim_func_attr, with_prim_func_body,
};
use super::{convert_ssa_module, create_module_pass, Pass};
use crate::ir::prim::StringImm;
use crate::ir::{
    BaseFunc, Call, Expr, GlobalVar, IRModule, IntImm, PointerType, PrimExpr, PrimType, TupleType,
    Type, Var,
};
use crate::target::Target;
use crate::tirx::{
    AllocBuffer, AssertStmt, AttrStmt, Bind, BufferVar, DeclBuffer, Evaluate, IterVar, PrimFunc,
    Return, Stmt,
};

const TARGET: &str = "target";
const GLOBAL_SYMBOL: &str = "global_symbol";
const CALLING_CONV: &str = "calling_conv";
const NUM_INPUTS: &str = "num_inputs";
const S_TIR: &str = "s_tir";
const THREAD_EXTENT: &str = "thread_extent";
const DEVICE_SCOPE: &str = "device_scope";
const KERNEL_LAUNCH_PARAMS: &str = "tirx.kernel_launch_params";
const NO_ALIAS: &str = "tirx.noalias";
const IS_GLOBAL_FUNC: &str = "tirx.is_global_func";
const DYNAMIC_SHARED_BYTES: &str = "tirx.dyn_smem_bytes";
const MIN_BLOCKS_PER_SM: &str = "tirx.launch_bounds_min_blocks_per_sm";
const MAX_BLOCKS_PER_CLUSTER: &str = "tirx.launch_bounds_max_blocks_per_cluster";
const MAX_REGISTERS: &str = "tirx.max_registers";
const REQUIRED_BLOCK_SIZE: &str = "tirx.required_block_size";
const DEVICE_KERNEL_LAUNCH: i64 = 2;
const USE_DYNAMIC_SHARED_MEMORY: &str = "tirx.use_dyn_shared_memory";
const USE_PROGRAMMATIC_DEPENDENT_LAUNCH: &str = "tirx.use_programtic_dependent_launch";
const USE_COOPERATIVE_LAUNCH: &str = "tirx.use_cooperative_launch";
const USE_REQUIRED_BLOCK_DIMENSION: &str = "tirx.use_required_block_dimension";

/// Split target-annotated device regions and lower cross-target kernel calls.
pub fn split_host_device_module(module: IRModule) -> Result<IRModule> {
    let mut used_names = module
        .functions
        .iter()
        .map(|(global, _)| global.name_hint.as_str().to_owned())
        .collect::<HashSet<_>>();
    let mut functions = Vec::with_capacity(module.functions.len());
    let mut device_functions = Vec::new();

    for (global, function) in module.functions.iter() {
        let Ok(function) = function.clone().try_cast::<PrimFunc>() else {
            functions.push((global, function));
            continue;
        };
        let function = annotate_device_regions(function)?;
        let prefix = function_string_attr(&function, GLOBAL_SYMBOL)?
            .unwrap_or_else(|| global.name_hint.clone());
        let mut splitter = HostDeviceSplitter {
            current_function: &function,
            kernel_name: format!("{}_kernel", prefix.as_str()),
            used_names: &mut used_names,
            device_functions: &mut device_functions,
        };
        let body: Stmt = structural_mutate(function.body.clone(), &mut splitter)?.try_into()?;
        functions.push((global, BaseFunc::from(with_prim_func_body(function, body))));
    }
    functions.extend(device_functions);
    let split = IRModule::with_metadata(
        Map::from_iter(functions),
        module.source_map.clone(),
        module.attrs.clone(),
        module.global_infos.clone(),
    )?;
    lower_device_kernel_launches(convert_ssa_module(split)?)
}

/// Build TVM's `tirx.SplitHostDevice` module pass in Rust.
pub fn split_host_device() -> Result<Pass> {
    create_module_pass(
        "tirx.SplitHostDevice",
        0,
        Vec::new(),
        false,
        split_host_device_module,
    )
}

fn annotate_device_regions(function: PrimFunc) -> Result<PrimFunc> {
    let target = function_target(&function)?;
    if target.host()?.is_none() {
        return Ok(function);
    }
    let mut annotator = DeviceRegionAnnotator {
        device_target: target.without_host()?,
    };
    let body: Stmt = structural_mutate(function.body.clone(), &mut annotator)?.try_into()?;
    Ok(with_prim_func_body(function, body))
}

struct DeviceRegionAnnotator {
    device_target: Target,
}

#[tvm_ffi::dispatch(mutate)]
impl DeviceRegionAnnotator {
    fn mutate_attribute(&mut self, value: AttrStmt, mutator: &mut Mutator) -> Result<Stmt> {
        match value.attr_key.as_str() {
            TARGET => Ok(value.into()),
            THREAD_EXTENT | DEVICE_SCOPE => Ok(AttrStmt::new(
                self.device_target.clone(),
                TARGET,
                IntImm::new("int32", 0)?,
                value,
            )?
            .into()),
            _ => mutate_stmt_default(self, mutator, value.into()),
        }
    }

    fn mutate_default(&mut self, value: &MapValue, mutator: &mut Mutator) -> Result<Any> {
        mutate_stmt_expr_default(self, mutator, value)
    }
}

struct HostDeviceSplitter<'a> {
    current_function: &'a PrimFunc,
    kernel_name: String,
    used_names: &'a mut HashSet<String>,
    device_functions: &'a mut Vec<(GlobalVar, BaseFunc)>,
}

#[tvm_ffi::dispatch(mutate)]
impl HostDeviceSplitter<'_> {
    fn mutate_attribute(&mut self, value: AttrStmt, mutator: &mut Mutator) -> Result<Stmt> {
        if value.attr_key.as_str() != TARGET {
            return mutate_stmt_default(self, mutator, value.into());
        }
        let target = Target::try_from(value.node.clone())?.without_host()?;
        self.extract_device_function(value.body.clone(), target)
    }

    fn mutate_default(&mut self, value: &MapValue, mutator: &mut Mutator) -> Result<Any> {
        mutate_stmt_expr_default(self, mutator, value)
    }
}

impl HostDeviceSplitter<'_> {
    fn extract_device_function(&mut self, body: Stmt, target: Target) -> Result<Stmt> {
        let is_cuda = target.kind_name()?.as_str() == "cuda";
        let parameters: Array<Var> = tvm_ffi::cached_global_func!("tirx.analysis.UndefinedVars")
            .call_tuple((&body, Array::<Var>::new(Vec::new())))?
            .try_into()?;
        let mut parameters = parameters.iter().collect::<Vec<_>>();
        if target.kind_name()?.as_str() != "trn" {
            parameters.sort_by(|lhs, rhs| {
                let lhs_handle = is_handle_parameter(lhs);
                let rhs_handle = is_handle_parameter(rhs);
                (!lhs_handle, lhs.name.as_str()).cmp(&(!rhs_handle, rhs.name.as_str()))
            });
        }

        let mut kernel_parameters = Vec::with_capacity(parameters.len());
        let mut call_arguments = Vec::with_capacity(parameters.len());
        let mut substitutions = Vec::new();
        let mut declarations = Vec::new();
        for parameter in parameters {
            if let Ok(buffer) = BufferVar::try_from(&parameter) {
                let buffer_type = buffer.type_annotation();
                let kernel_buffer = buffer_type.new_var(buffer.name.as_str());
                let data_parameter = Var::with_type(
                    &format!("{}_ptr", buffer.name.as_str()),
                    PointerType::new(
                        buffer_type.dtype.clone(),
                        buffer_type.storage_scope.as_str(),
                    )?,
                );
                call_arguments.push(buffer_data(&buffer)?);
                kernel_parameters.push(data_parameter.clone());
                substitutions.push((buffer.as_var().clone(), kernel_buffer.clone().into()));
                declarations.push((kernel_buffer, data_parameter));
            } else {
                call_arguments.push(parameter.clone().into());
                kernel_parameters.push(parameter);
            }
        }

        let mut body: Stmt = if substitutions.is_empty() {
            body
        } else {
            tvm_ffi::cached_global_func!("tirx.Substitute")
                .call_tuple((body, Map::<Var, Expr>::from_iter(substitutions)))?
                .try_into()?
        };
        let can_propagate_errors = matches!(target.device_type()?, 1 | 12 | 16);
        let success = IntImm::new("int32", 0)?;
        let return_type: Type = if can_propagate_errors {
            body = Stmt::sequence(vec![body, Return::new(success.clone()).into()])?;
            PrimType::new("int32")?.into()
        } else {
            TupleType::empty().into()
        };
        for (buffer, data) in declarations {
            body = Stmt::sequence(vec![DeclBuffer::new(buffer, data)?.into(), body])?;
        }

        let (body, launch_bounds) = extract_launch_bounds(body)?;
        let mut device_function = PrimFunc::with_metadata(
            kernel_parameters,
            body,
            return_type,
            crate::ir::DictAttrs::empty(),
            None,
        )?;
        device_function = with_prim_func_attr(device_function, TARGET, target);
        device_function = with_prim_func_attr(device_function, NO_ALIAS, true);
        device_function = with_prim_func_attr(device_function, IS_GLOBAL_FUNC, true);
        for key in [S_TIR, KERNEL_LAUNCH_PARAMS, NUM_INPUTS] {
            if let Some(attribute) = self
                .current_function
                .attrs
                .dict
                .get(&FfiString::from(key))?
            {
                device_function = with_prim_func_attr(device_function, key, attribute);
            }
        }
        if is_cuda {
            for (key, value) in launch_bounds {
                device_function = with_prim_func_attr(device_function, key, value);
            }
        }

        let name = fresh_name(&self.kernel_name, self.used_names);
        let global = GlobalVar::new(&name);
        self.device_functions
            .push((global.clone(), BaseFunc::from(device_function)));
        if can_propagate_errors {
            let error_code = Var::new("kernel_error_code", "int32")?;
            let call: PrimExpr =
                Call::new(PrimType::new("int32")?, global, call_arguments).try_cast()?;
            Stmt::sequence(vec![
                Bind::new(error_code.clone(), call)?.into(),
                AssertStmt::new(
                    crate::ir::prim::EQ::new(error_code, success)?,
                    "RuntimeError",
                    "Error executing compute kernel",
                )?
                .into(),
            ])
        } else {
            Evaluate::new(Call::new(PrimType::void(), global, call_arguments)).map(Into::into)
        }
    }
}

fn extract_launch_bounds(body: Stmt) -> Result<(Stmt, Vec<(&'static str, i64)>)> {
    let mut extractor = LaunchBoundsExtractor::default();
    let body: Stmt = structural_mutate(body, &mut extractor)?.try_into()?;
    if extractor.max_blocks_per_cluster.is_some() && extractor.min_blocks_per_sm.is_none() {
        return Err(value_error(
            "tirx.launch_bounds_max_blocks_per_cluster requires tirx.launch_bounds_min_blocks_per_sm",
        ));
    }
    if extractor.max_registers.is_some()
        && (extractor.min_blocks_per_sm.is_some() || extractor.max_blocks_per_cluster.is_some())
    {
        return Err(value_error(
            "tirx.max_registers cannot be combined with CUDA launch bounds",
        ));
    }
    if extractor.required_block_size.is_some() && extractor.max_registers.is_some() {
        return Err(value_error(
            "tirx.required_block_size cannot be combined with maximum registers",
        ));
    }
    let mut attributes = Vec::new();
    for (key, value) in [
        (MIN_BLOCKS_PER_SM, extractor.min_blocks_per_sm),
        (MAX_BLOCKS_PER_CLUSTER, extractor.max_blocks_per_cluster),
        (MAX_REGISTERS, extractor.max_registers),
        (REQUIRED_BLOCK_SIZE, extractor.required_block_size),
    ] {
        if let Some(value) = value {
            attributes.push((key, value));
        }
    }
    Ok((body, attributes))
}

#[derive(Default)]
struct LaunchBoundsExtractor {
    min_blocks_per_sm: Option<i64>,
    max_blocks_per_cluster: Option<i64>,
    max_registers: Option<i64>,
    required_block_size: Option<i64>,
}

#[tvm_ffi::dispatch(mutate)]
impl LaunchBoundsExtractor {
    fn mutate_attribute(&mut self, value: AttrStmt, mutator: &mut Mutator) -> Result<Stmt> {
        let slot = match value.attr_key.as_str() {
            MIN_BLOCKS_PER_SM => Some((&mut self.min_blocks_per_sm, false)),
            MAX_BLOCKS_PER_CLUSTER => Some((&mut self.max_blocks_per_cluster, false)),
            MAX_REGISTERS => Some((&mut self.max_registers, false)),
            REQUIRED_BLOCK_SIZE => Some((&mut self.required_block_size, true)),
            _ => None,
        };
        let Some((slot, must_equal_one)) = slot else {
            return mutate_stmt_default(self, mutator, value.into());
        };
        let integer = int_value(&value.value)
            .ok_or_else(|| value_error("launch bound expects an integer value"))?;
        if (must_equal_one && integer != 1) || (!must_equal_one && integer <= 0) {
            return Err(value_error("invalid launch-bound value"));
        }
        if slot.is_some_and(|existing| existing != integer) {
            return Err(value_error("conflicting launch-bound values"));
        }
        *slot = Some(integer);
        mutator.mutate(self, &value.body)?.try_into()
    }

    fn mutate_default(&mut self, value: &MapValue, mutator: &mut Mutator) -> Result<Any> {
        mutate_stmt_expr_default(self, mutator, value)
    }
}

#[derive(Clone)]
struct KernelInfo {
    target: Target,
    global_symbol: FfiString,
    parameters: Array<Var>,
    launch_parameters: Array<FfiString>,
    launch_arguments: Array<PrimExpr>,
}

fn lower_device_kernel_launches(module: IRModule) -> Result<IRModule> {
    let called = collect_called_globals(&module)?;
    let mut kernel_info = HashMap::new();
    for (global, function) in module.functions.iter() {
        if !called.contains(&ObjectIdentity::of(&global)) {
            continue;
        }
        if let Ok(function) = function.try_cast::<PrimFunc>() {
            kernel_info.insert(
                ObjectIdentity::of(&global),
                collect_kernel_info(&global, &function)?,
            );
        }
    }

    let mut rewriter = KernelLaunchRewriter {
        kernel_info: &kernel_info,
        launched: HashSet::new(),
        extern_calls: HashSet::new(),
        current_target: None,
        current_host_target: None,
    };
    let mut rewritten = Vec::with_capacity(module.functions.len());
    for (global, function) in module.functions.iter() {
        let Ok(function) = function.clone().try_cast::<PrimFunc>() else {
            rewritten.push((global, function));
            continue;
        };
        let target = function_target(&function)?;
        rewriter.current_host_target = target.host()?;
        rewriter.current_target = Some(target.without_host()?);
        let body: Stmt = structural_mutate(function.body.clone(), &mut rewriter)?.try_into()?;
        rewritten.push((global, BaseFunc::from(with_prim_func_body(function, body))));
        rewriter.current_target = None;
        rewriter.current_host_target = None;
    }

    let mut finalized = Vec::with_capacity(rewritten.len());
    for (global, function) in rewritten {
        let Ok(mut function) = function.clone().try_cast::<PrimFunc>() else {
            finalized.push((global, function));
            continue;
        };
        let identity = ObjectIdentity::of(&global);
        let launched = rewriter.launched.contains(&identity);
        let external = rewriter.extern_calls.contains(&identity);
        if launched && external {
            return Err(value_error(
                "one function cannot be both a device kernel launch and call_extern target",
            ));
        }
        if launched || external {
            function = with_prim_func_attr(function, IS_GLOBAL_FUNC, true);
        }
        if launched {
            let info = kernel_info
                .get(&identity)
                .ok_or_else(|| value_error("missing device kernel information"))?;
            let body = rewrite_kernel_returns(
                function.body.clone(),
                info.target.kind_name()?.as_str() != "cuda",
            )?;
            function = PrimFunc::from_complete_fields(
                function.span.clone(),
                function.ty.clone(),
                function.attrs.clone(),
                function.params.clone(),
                TupleType::empty().into(),
                body,
            );
            function = with_prim_func_attr(function, CALLING_CONV, DEVICE_KERNEL_LAUNCH);
            function = with_prim_func_attr(
                function,
                KERNEL_LAUNCH_PARAMS,
                info.launch_parameters.clone(),
            );
            function = with_prim_func_attr(function, GLOBAL_SYMBOL, info.global_symbol.clone());
        } else if external && function_string_attr(&function, GLOBAL_SYMBOL)?.is_none() {
            function = with_prim_func_attr(function, GLOBAL_SYMBOL, global.name_hint.clone());
        }
        finalized.push((global, BaseFunc::from(function)));
    }
    IRModule::with_metadata(
        Map::from_iter(finalized),
        module.source_map.clone(),
        module.attrs.clone(),
        module.global_infos.clone(),
    )
}

fn collect_called_globals(module: &IRModule) -> Result<HashSet<ObjectIdentity>> {
    let mut called = HashSet::new();
    for (_, function) in module.functions.iter() {
        let Ok(function) = function.try_cast::<PrimFunc>() else {
            continue;
        };
        structural_walk(
            &function.body,
            |call: Call| {
                if let Ok(global) = call.op.clone().try_cast::<GlobalVar>() {
                    called.insert(ObjectIdentity::of(&global));
                }
                WalkResult::Advance
            },
            WalkOrder::PreOrder,
        )?;
    }
    Ok(called)
}

fn collect_kernel_info(global: &GlobalVar, function: &PrimFunc) -> Result<KernelInfo> {
    let target = function_target(function)?.without_host()?;
    let requested = function
        .attrs
        .dict
        .get(&FfiString::from(KERNEL_LAUNCH_PARAMS))?
        .map(Array::<FfiString>::try_from)
        .transpose()?
        .unwrap_or_else(|| Array::new(Vec::new()));
    let mut collector = KernelInfoCollector {
        launch_parameters: Vec::new(),
        thread_extents: HashMap::new(),
        seen_threads: HashSet::new(),
        bindings: Map::new(),
        dynamic_shared_bytes: None,
        inferred_shared_bytes: None,
        saw_dynamic_shared_allocation: false,
        use_programmatic_dependent_launch: requested
            .iter()
            .any(|tag| tag.as_str() == USE_PROGRAMMATIC_DEPENDENT_LAUNCH),
        use_cooperative_launch: requested
            .iter()
            .any(|tag| tag.as_str() == USE_COOPERATIVE_LAUNCH),
        use_required_block_dimension: function
            .attrs
            .dict
            .get(&FfiString::from(REQUIRED_BLOCK_SIZE))?
            .map(i64::try_from)
            .transpose()?
            .unwrap_or(0)
            == 1,
    };
    structural_walk(&function.body, &mut collector, WalkOrder::PreOrder)?;
    if collector.dynamic_shared_bytes.is_none() {
        collector.dynamic_shared_bytes = collector.inferred_shared_bytes;
    }
    if collector.use_programmatic_dependent_launch {
        collector
            .launch_parameters
            .push(FfiString::from(USE_PROGRAMMATIC_DEPENDENT_LAUNCH));
    }
    if collector.use_cooperative_launch {
        collector
            .launch_parameters
            .push(FfiString::from(USE_COOPERATIVE_LAUNCH));
    }
    if collector.use_required_block_dimension {
        collector
            .launch_parameters
            .push(FfiString::from(USE_REQUIRED_BLOCK_DIMENSION));
    }
    if collector.dynamic_shared_bytes.is_some() {
        collector
            .launch_parameters
            .push(FfiString::from(USE_DYNAMIC_SHARED_MEMORY));
    }
    let mut launch_arguments = Vec::new();
    for parameter in &collector.launch_parameters {
        match parameter.as_str() {
            USE_PROGRAMMATIC_DEPENDENT_LAUNCH
            | USE_COOPERATIVE_LAUNCH
            | USE_REQUIRED_BLOCK_DIMENSION => {}
            USE_DYNAMIC_SHARED_MEMORY => launch_arguments.push(
                collector
                    .dynamic_shared_bytes
                    .clone()
                    .ok_or_else(|| value_error("missing dynamic shared-memory launch size"))?,
            ),
            tag => launch_arguments.push(
                collector
                    .thread_extents
                    .get(tag)
                    .cloned()
                    .ok_or_else(|| value_error("missing thread extent for kernel launch"))?,
            ),
        }
    }
    let global_symbol =
        function_string_attr(function, GLOBAL_SYMBOL)?.unwrap_or_else(|| global.name_hint.clone());
    Ok(KernelInfo {
        target,
        global_symbol,
        parameters: function.params.clone(),
        launch_parameters: Array::new(collector.launch_parameters),
        launch_arguments: Array::new(launch_arguments),
    })
}

struct KernelInfoCollector {
    launch_parameters: Vec<FfiString>,
    thread_extents: HashMap<String, PrimExpr>,
    seen_threads: HashSet<String>,
    bindings: Map<Var, Expr>,
    dynamic_shared_bytes: Option<PrimExpr>,
    inferred_shared_bytes: Option<PrimExpr>,
    saw_dynamic_shared_allocation: bool,
    use_programmatic_dependent_launch: bool,
    use_cooperative_launch: bool,
    use_required_block_dimension: bool,
}

#[tvm_ffi::dispatch(walk)]
impl KernelInfoCollector {
    fn walk_binding(&mut self, value: Bind) -> Result<WalkResult> {
        let Ok(mut expression) = value.value.clone().try_cast::<PrimExpr>() else {
            return Ok(WalkResult::Advance);
        };
        if !self.bindings.is_empty() {
            expression = substitute_prim(&expression, &self.bindings)?;
        }
        let mut bindings = self.bindings.iter().collect::<Vec<_>>();
        bindings.retain(|(variable, _)| !variable.same_as(&value.var));
        bindings.push((value.var.clone(), expression.into()));
        self.bindings = bindings.into_iter().collect();
        Ok(WalkResult::Advance)
    }

    fn walk_attribute(&mut self, value: AttrStmt) -> Result<WalkResult> {
        if value.attr_key.as_str() == DYNAMIC_SHARED_BYTES {
            if self.dynamic_shared_bytes.is_some() {
                return Err(value_error(
                    "only one tirx.dyn_smem_bytes declaration is allowed per kernel",
                ));
            }
            int_value(&value.value)
                .ok_or_else(|| value_error("tirx.dyn_smem_bytes must be an IntImm"))?;
            self.dynamic_shared_bytes = Some(value.value.clone());
        }
        if value.attr_key.as_str() == THREAD_EXTENT {
            let tag = if let Ok(iteration) = IterVar::try_from(value.node.clone()) {
                let tag = iteration.thread_tag()?;
                if tag.as_str().is_empty() {
                    return Err(value_error("thread_extent IterVar must have a thread tag"));
                }
                tag
            } else {
                Var::try_from(value.node.clone())?.name.clone()
            };
            if self.seen_threads.insert(tag.as_str().to_owned()) {
                let extent = if self.bindings.is_empty() {
                    value.value.clone()
                } else {
                    substitute_prim(&value.value, &self.bindings)?
                };
                self.thread_extents.insert(tag.as_str().to_owned(), extent);
                self.launch_parameters.push(tag);
            }
        }
        Ok(WalkResult::Advance)
    }

    fn walk_allocation(&mut self, value: AllocBuffer) -> Result<WalkResult> {
        let ty = value.buffer.type_annotation();
        if ty.storage_scope.as_str() != "shared.dyn" {
            return Ok(WalkResult::Advance);
        }
        if self.saw_dynamic_shared_allocation {
            return Err(value_error(
                "only one dynamic shared-memory allocation is allowed",
            ));
        }
        self.saw_dynamic_shared_allocation = true;
        if ty.shape.is_empty() {
            return Err(value_error(
                "dynamic shared-memory allocation needs a shape",
            ));
        }
        let mut size: PrimExpr = IntImm::new("int32", 1)?.into();
        for extent in ty.shape.iter() {
            size = crate::ir::prim::Mul::new(size, extent)?.into();
        }
        let bytes = (i64::from(ty.dtype.dtype.bits) * i64::from(ty.dtype.dtype.lanes) + 7) / 8;
        size = crate::ir::prim::Mul::new(size, IntImm::new("int64", bytes)?)?.into();
        if !self.bindings.is_empty() {
            size = substitute_prim(&size, &self.bindings)?;
        }
        if int_value(&size) == Some(0) && self.dynamic_shared_bytes.is_none() {
            return Err(value_error(
                "a placeholder shared.dyn allocation requires tirx.dyn_smem_bytes",
            ));
        }
        self.inferred_shared_bytes = Some(size);
        Ok(WalkResult::Advance)
    }
}

struct KernelLaunchRewriter<'a> {
    kernel_info: &'a HashMap<ObjectIdentity, KernelInfo>,
    launched: HashSet<ObjectIdentity>,
    extern_calls: HashSet<ObjectIdentity>,
    current_target: Option<Target>,
    current_host_target: Option<Target>,
}

#[tvm_ffi::dispatch(mutate)]
impl KernelLaunchRewriter<'_> {
    fn mutate_call(&mut self, _value: Call, mutator: &mut Mutator) -> Result<Expr> {
        let value: Call = mutator.default_mutate(self)?.try_into()?;
        let Ok(global) = value.op.clone().try_cast::<GlobalVar>() else {
            return Ok(value.into());
        };
        let identity = ObjectIdentity::of(&global);
        let info = self
            .kernel_info
            .get(&identity)
            .ok_or_else(|| value_error("global call target is absent from the module"))?;
        let caller_is_host = self.current_host_target.is_some();
        let caller_target = self
            .current_host_target
            .as_ref()
            .or(self.current_target.as_ref())
            .ok_or_else(|| value_error("kernel call has no caller target"))?;
        let is_kernel = !info.launch_parameters.is_empty();
        if !(caller_is_host && is_kernel) {
            if target_equal(caller_target, &info.target)? {
                return Ok(value.into());
            }
            if caller_target.device_type()? == info.target.device_type()? {
                self.extern_calls.insert(identity);
                let mut arguments = vec![StringImm::new(global.name_hint.as_str()).into()];
                arguments.extend(value.args.iter());
                return Ok(Call::new(
                    value.ty.clone(),
                    get_operator("tirx.call_extern")?,
                    arguments,
                )
                .into());
            }
        }

        if value.args.len() != info.parameters.len() {
            return Err(value_error(
                "kernel call argument count does not match its PrimFunc",
            ));
        }
        let substitutions = info
            .parameters
            .iter()
            .zip(value.args.iter())
            .filter_map(|(parameter, argument)| {
                PrimExpr::try_from(argument)
                    .ok()
                    .map(|argument| (parameter, Expr::from(argument)))
            })
            .collect::<Map<Var, Expr>>();
        let mut arguments = vec![StringImm::new(info.global_symbol.as_str()).into()];
        arguments.extend(value.args.iter());
        for launch_argument in info.launch_arguments.iter() {
            let substituted: PrimExpr = tvm_ffi::cached_global_func!("tirx.Substitute")
                .call_tuple((launch_argument, substitutions.clone()))?
                .try_into()?;
            arguments.push(substituted.into());
        }
        self.launched.insert(identity);
        let primitive = value.ty.clone().try_cast::<PrimType>()?;
        let return_type = if primitive.dtype.code == tvm_ffi::DLDataTypeCode::kDLOpaqueHandle as u8
            && primitive.dtype.bits == 0
            && primitive.dtype.lanes == 0
        {
            PrimType::new("int32")?
        } else {
            primitive
        };
        Ok(Call::new(
            return_type,
            get_operator("tirx.tvm_call_packed")?,
            arguments,
        )
        .into())
    }

    fn mutate_default(&mut self, value: &MapValue, mutator: &mut Mutator) -> Result<Any> {
        mutate_stmt_expr_default(self, mutator, value)
    }
}

fn rewrite_kernel_returns(body: Stmt, remove: bool) -> Result<Stmt> {
    let mut rewriter = KernelReturnRewriter { remove };
    structural_mutate(body, &mut rewriter)?.try_into()
}

struct KernelReturnRewriter {
    remove: bool,
}

#[tvm_ffi::dispatch(mutate)]
impl KernelReturnRewriter {
    fn mutate_return(&mut self, value: Return) -> Result<Stmt> {
        let returned = int_value(&value.value)
            .ok_or_else(|| value_error("device kernel may only return 0"))?;
        if returned != 0 {
            return Err(value_error("device kernel may only return 0"));
        }
        if self.remove {
            Evaluate::from_i64(0).map(Into::into)
        } else {
            Ok(value.into())
        }
    }

    fn mutate_attribute(&mut self, value: AttrStmt, mutator: &mut Mutator) -> Result<Stmt> {
        if value.attr_key.as_str() == DYNAMIC_SHARED_BYTES {
            return mutator.mutate(self, &value.body)?.try_into();
        }
        mutate_stmt_default(self, mutator, value.into())
    }

    fn mutate_default(&mut self, value: &MapValue, mutator: &mut Mutator) -> Result<Any> {
        mutate_stmt_expr_default(self, mutator, value)
    }
}

fn function_target(function: &PrimFunc) -> Result<Target> {
    function
        .attrs
        .dict
        .get(&FfiString::from(TARGET))?
        .ok_or_else(|| value_error("SplitHostDevice requires the target attribute"))?
        .try_into()
}

fn function_string_attr(function: &PrimFunc, key: &str) -> Result<Option<FfiString>> {
    function
        .attrs
        .dict
        .get(&FfiString::from(key))?
        .map(FfiString::try_from)
        .transpose()
}

fn is_handle_parameter(variable: &Var) -> bool {
    is_pointer_type(&variable.ty) || is_buffer_type(&variable.ty)
}

fn fresh_name(base: &str, used: &mut HashSet<String>) -> String {
    if used.insert(base.to_owned()) {
        return base.to_owned();
    }
    let mut suffix = 1_usize;
    loop {
        let candidate = format!("{base}_{suffix}");
        if used.insert(candidate.clone()) {
            return candidate;
        }
        suffix += 1;
    }
}

fn target_equal(lhs: &Target, rhs: &Target) -> Result<bool> {
    tvm_ffi::cached_global_func!("ffi.StructuralEqual")
        .call_tuple((lhs, rhs, false, false))?
        .try_into()
}

fn buffer_data(buffer: &BufferVar) -> Result<Expr> {
    let ty = buffer.type_annotation();
    Ok(Call::new(
        PointerType::new(ty.dtype.clone(), ty.storage_scope.as_str())?,
        get_operator("tirx.buffer_data")?,
        vec![buffer.as_var().clone().into()],
    )
    .into())
}

fn substitute_prim(expression: &PrimExpr, substitutions: &Map<Var, Expr>) -> Result<PrimExpr> {
    tvm_ffi::cached_global_func!("tirx.Substitute")
        .call_tuple((expression, substitutions))?
        .try_into()
}
