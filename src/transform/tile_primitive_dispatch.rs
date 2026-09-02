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
    structural_map, structural_mutate, structural_walk, Any, Array, DLDataTypeCode, Function, Map,
    MapValue, Mutator, ObjectIdentity, ObjectRefCast, ObjectRefCore, Result, String as FfiString,
    WalkOrder, WalkResult,
};

use super::exec_context::{encode_split, ExecContext};
use super::scope_id::{
    compute_warp_id_in_cta, resolve_scope_id, LaunchParams, ScopeDefinition, ScopeIdSet,
};
use super::utils::{mutate_stmt_default, mutate_stmt_expr_default};
use super::{create_prim_func_pass, Pass};
use crate::analysis::{detect_linear_equation, Analyzer};
use crate::ir::{Call, Expr, IntImm, PrimExpr, Range, Var};
use crate::target::Target;
use crate::tirx::{
    AllocBuffer, And, AttrStmt, Bind, BufferVar, DeclBuffer, DispatchContext, Evaluate, FloorMod,
    For, IfThenElse, IterVar, Mod, PrimFunc, ScopeBinding, ScopeIdDef, ScopeIdDefStmt, SeqStmt,
    Stmt, TilePrimitiveCall, EQ, GE, GT, LE, LT, NE,
};

const DEVICE_ENTRY: &str = "tirx.device_entry";
const THREAD_EXTENT: &str = "thread_extent";
const PRIVATE_ALLOC: &str = "private_alloc";
const DEVICE_INIT: &str = "device_init_stmt";
const HOST_INIT: &str = "host_init_stmt";
const POST_BUFFER_DEF: &str = "post_buffer_def_stmt";
const NEGATIVE_INFINITY: i64 = i64::MIN / 4;
const POSITIVE_INFINITY: i64 = i64::MAX / 4;

/// Lower registered tile-primitive calls in one TIRx function.
pub fn tile_primitive_dispatch_prim_func(function: PrimFunc) -> Result<PrimFunc> {
    let target = function_target(&function)?;
    let mut dispatcher = TileDispatcher::new(target)?;
    let body = structural_mutate(function.body.clone(), &mut dispatcher)?.try_into()?;
    ensure_no_tile_calls(&body)?;
    Ok(function.with_body(body))
}

/// Build TVM's `tirx.TilePrimitiveDispatch` PrimFunc pass in Rust.
pub fn tile_primitive_dispatch() -> Result<Pass> {
    create_prim_func_pass(
        "tirx.TilePrimitiveDispatch",
        0,
        Vec::new(),
        false,
        tile_primitive_dispatch_prim_func,
    )
}

struct PendingBufferStatements {
    source: BufferVar,
    statements: Vec<Stmt>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct ScopeTarget {
    binding: ScopeBinding,
    dimension: usize,
    dimensions: usize,
}

struct ScopeRange {
    target: ScopeTarget,
    low: i64,
    high: i64,
}

#[derive(Clone, Copy)]
enum Comparison {
    Equal,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
}

struct TileDispatcher {
    target: Target,
    target_kind: String,
    analyzer: Analyzer,
    launch_params: LaunchParams,
    variable_ranges: HashMap<ObjectIdentity, (Var, Range)>,
    scope_levels: Vec<Vec<ScopeDefinition>>,
    execution_contexts: Vec<ExecContext>,
    cluster_axes: Vec<(String, i64)>,
    allocations: Vec<BufferVar>,
    device_initializers: Vec<Stmt>,
    host_initializers: Vec<Stmt>,
    post_buffer_definitions: HashMap<ObjectIdentity, PendingBufferStatements>,
    shared_state: Map<FfiString, Any>,
    storage_roots: HashMap<ObjectIdentity, BufferVar>,
    device_depth: usize,
    filter_operator: Expr,
    bitwise_and_operator: Expr,
    elect_sync_operator: Expr,
    selector_operator: Expr,
}

impl TileDispatcher {
    fn new(target: Target) -> Result<Self> {
        Ok(Self {
            target_kind: target.kind_name()?.as_str().to_owned(),
            target,
            analyzer: Analyzer::new()?,
            launch_params: HashMap::new(),
            variable_ranges: HashMap::new(),
            scope_levels: Vec::new(),
            execution_contexts: Vec::new(),
            cluster_axes: Vec::new(),
            allocations: Vec::new(),
            device_initializers: Vec::new(),
            host_initializers: Vec::new(),
            post_buffer_definitions: HashMap::new(),
            shared_state: Map::new(),
            storage_roots: HashMap::new(),
            device_depth: 0,
            filter_operator: get_operator("tirx.filter")?,
            bitwise_and_operator: get_operator("tirx.bitwise_and")?,
            elect_sync_operator: get_operator("tirx.cuda.elect_sync")?,
            selector_operator: get_operator("tirx.selector")?,
        })
    }

    fn process_device_entry(&mut self, entry: AttrStmt, mutator: &mut Mutator) -> Result<Stmt> {
        let outermost = self.device_depth == 0;
        self.device_depth += 1;
        self.launch_params.clear();

        let original_defs = gather_scope_defs(&entry.body)?;
        let original_set = ScopeIdSet::verify(&original_defs)?;
        let mut bindings = self.extract_launch_params(&original_set)?;
        let tracking = self.push_kernel_context()?;

        self.scope_levels.push(Vec::new());
        let mutated: Result<Stmt> = mutator.mutate(self, &entry.body)?.try_into();
        self.scope_levels.pop();
        let mut body = mutated?;

        let lowered_defs = gather_scope_defs(&body)?;
        let lowered_set = ScopeIdSet::verify(&lowered_defs)?;
        let mut implicit_scope_ids = Vec::new();
        for definition in &lowered_defs {
            let definition = lowered_set.resolve_deferred(definition)?;
            let values = resolve_scope_id(
                definition.scope,
                definition.def_ids.len(),
                &self.target_kind,
                &self.launch_params,
            )?;
            for (variable, mut value) in definition.def_ids.iter().zip(values) {
                let variable_type = variable.type_annotation();
                if variable_type.dtype != value.type_annotation().dtype {
                    value = crate::tirx::Cast::new(variable_type, value)?.into();
                }
                if variable.as_var().name.as_str().is_empty() {
                    implicit_scope_ids.push(variable.as_var().clone());
                }
                bindings.push((variable.as_var().clone(), value));
            }
        }

        if !outermost {
            self.device_depth -= 1;
            if tracking {
                self.execution_contexts.pop();
            }
            if body.same_as(&entry.body) {
                return Ok(entry.into());
            }
            return Ok(entry.with_children(entry.value.clone(), body).into());
        }

        for initializer in self.device_initializers.drain(..).rev() {
            body = replace_kernel_point(initializer, &body)?;
        }
        if !self.allocations.is_empty() {
            let mut statements = self
                .allocations
                .drain(..)
                .map(AllocBuffer::new)
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .map(Into::into)
                .collect::<Vec<_>>();
            statements.push(body);
            body = Stmt::sequence(statements)?;
        }
        if !implicit_scope_ids.is_empty() {
            let mut statements = implicit_scope_ids
                .into_iter()
                .map(Evaluate::new)
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .map(Stmt::from)
                .collect::<Vec<_>>();
            statements.push(body);
            body = Stmt::sequence(statements)?;
        }
        body = remove_scope_definitions(body)?;

        let mut prefix = bindings
            .into_iter()
            .map(|(variable, value)| Bind::new(variable, value).map(Into::into))
            .collect::<Result<Vec<Stmt>>>()?;
        prefix.push(body);
        let mut result = Stmt::sequence(prefix)?;

        for iter_var in launch_attribute_order(&self.launch_params)
            .into_iter()
            .rev()
        {
            let domain = iter_var
                .dom()?
                .ok_or_else(|| value_error("launch parameter has no domain"))?;
            result = AttrStmt::new(iter_var, THREAD_EXTENT, domain.extent.clone(), result)?.into();
        }
        for initializer in self.host_initializers.drain(..) {
            let initializer = resolve_storage_roots(initializer, &self.storage_roots)?;
            result = replace_kernel_point(initializer, &result)?;
        }

        self.device_depth -= 1;
        if tracking {
            self.execution_contexts.pop();
        }
        Ok(result)
    }

    fn extract_launch_params(&mut self, definitions: &ScopeIdSet) -> Result<Vec<(Var, PrimExpr)>> {
        self.add_launch_parameters(definitions.get(ScopeBinding::CLUSTER_CTA), "clusterCtaIdx.")?;
        self.add_preferred_cluster_parameters(definitions.get(ScopeBinding::CLUSTER_CTA))?;
        self.add_launch_parameters(definitions.get(ScopeBinding::KERNEL_CTA), "blockIdx.")?;
        self.add_launch_parameters(definitions.get(ScopeBinding::CTA_THREAD), "threadIdx.")?;
        if !definitions.is_empty() && !self.launch_params.contains_key("threadIdx.x") {
            return Err(value_error(
                "a kernel with ScopeIdDefs must declare thread launch parameters",
            ));
        }
        let mut bindings = Vec::new();
        if self.launch_params.contains_key("threadIdx.x") {
            let value = compute_warp_id_in_cta(&self.launch_params)?;
            let variable = Var::with_type("warp_id_in_cta", value.type_annotation());
            let domain =
                Range::from_min_extent(IntImm::new("int32", 0)?, IntImm::new("int32", 1)?)?;
            let iter_var = IterVar::thread_index(Some(domain), variable.clone(), "warp_id_in_cta")?;
            self.launch_params.insert("warp_id_in_cta".into(), iter_var);
            bindings.push((variable, value));
        }
        Ok(bindings)
    }

    fn add_launch_parameters(
        &mut self,
        definition: Option<&ScopeDefinition>,
        prefix: &str,
    ) -> Result<()> {
        let Some(definition) = definition else {
            return Ok(());
        };
        let extents = definition
            .extents
            .as_ref()
            .ok_or_else(|| value_error("launch parameter cannot use a deferred extent"))?;
        if extents.len() > 3 {
            return Err(value_error(
                "launch parameters support at most three dimensions",
            ));
        }
        for (dimension, extent) in extents.iter().enumerate() {
            let tag = format!("{prefix}{}", axis_name(dimension));
            let variable = Var::with_type(&tag, extent.type_annotation());
            let domain = Range::from_min_extent(IntImm::new("int32", 0)?, extent)?;
            let iter_var = IterVar::thread_index(Some(domain), variable, &tag)?;
            self.launch_params.insert(tag, iter_var);
        }
        Ok(())
    }

    fn add_preferred_cluster_parameters(
        &mut self,
        definition: Option<&ScopeDefinition>,
    ) -> Result<()> {
        let Some(extents) = definition.and_then(|definition| definition.preferred_extents.as_ref())
        else {
            return Ok(());
        };
        for (dimension, extent) in extents.iter().enumerate() {
            let tag = format!("preferredClusterCtaIdx.{}", axis_name(dimension));
            let variable = Var::with_type(&tag, extent.type_annotation());
            let domain = Range::from_min_extent(IntImm::new("int32", 0)?, extent)?;
            self.launch_params.insert(
                tag.clone(),
                IterVar::thread_index(Some(domain), variable, &tag)?,
            );
        }
        Ok(())
    }

    fn push_kernel_context(&mut self) -> Result<bool> {
        let Some(thread_extent) =
            self.product_extent(["threadIdx.x", "threadIdx.y", "threadIdx.z"])?
        else {
            return Ok(false);
        };
        if thread_extent <= 0 {
            return Ok(false);
        }
        let cluster_axes = self.constant_axes("clusterCtaIdx.")?;
        let cta_axes = if cluster_axes.is_empty() {
            self.constant_axes("blockIdx.")?
        } else {
            cluster_axes.clone()
        };
        self.cluster_axes = cluster_axes;
        let cta_extent = cta_axes.iter().map(|(_, extent)| extent).product();
        let factored = if cta_axes.len() > 1 {
            cta_axes
        } else {
            Vec::new()
        };
        self.execution_contexts.push(ExecContext::at_kernel_entry(
            32,
            thread_extent / 32,
            cta_extent,
            &factored,
        )?);
        Ok(true)
    }

    fn product_extent<const N: usize>(&self, tags: [&str; N]) -> Result<Option<i64>> {
        let mut product = 1_i64;
        for tag in tags {
            let Some(iter_var) = self.launch_params.get(tag) else {
                continue;
            };
            let domain = iter_var
                .dom()?
                .ok_or_else(|| value_error("launch parameter has no domain"))?;
            let Some(extent) = integer_value(&domain.extent) else {
                return Ok(None);
            };
            product *= extent;
        }
        Ok(Some(product))
    }

    fn constant_axes(&self, prefix: &str) -> Result<Vec<(String, i64)>> {
        let axis_names = if prefix == "clusterCtaIdx." {
            ["cbx", "cby", "cbz"]
        } else {
            ["bx", "by", "bz"]
        };
        let mut result = Vec::new();
        for (dimension, axis) in axis_names.into_iter().enumerate() {
            let tag = format!("{prefix}{}", axis_name(dimension));
            let Some(iter_var) = self.launch_params.get(&tag) else {
                continue;
            };
            let domain = iter_var
                .dom()?
                .ok_or_else(|| value_error("launch parameter has no domain"))?;
            let Some(extent) = integer_value(&domain.extent) else {
                return Ok(Vec::new());
            };
            result.push((axis.into(), extent));
        }
        Ok(result)
    }

    fn dispatch_tile_call(&mut self, call: TilePrimitiveCall) -> Result<Stmt> {
        let scope = call.scope()?;
        let split = match self.execution_contexts.last() {
            Some(context) => context.split(scope.kind()?)?,
            None => None,
        };
        let (inter, intra) = split
            .map(|split| (encode_split(&split.inter), encode_split(&split.intra)))
            .unwrap_or_else(|| (Map::new(), Map::new()));
        let context = DispatchContext::with_metadata(
            self.target.clone(),
            scope.clone(),
            Map::from_iter(
                self.launch_params
                    .iter()
                    .map(|(tag, value)| (FfiString::from(tag.as_str()), value.clone())),
            ),
            Map::from_iter(
                self.variable_ranges
                    .values()
                    .map(|(variable, range)| (variable.clone(), range.clone())),
            ),
            false,
            Map::new(),
            self.shared_state.clone(),
            inter,
            intra,
            FfiString::from(scope.name()?),
        )?;
        let dispatcher = Function::get_global("tirx.f_op_dispatcher")?;
        let implementation: PrimFunc = dispatcher.call_tuple((&call, &context))?.try_into()?;

        let callbacks = context.callbacks()?;
        if let Some(value) = callbacks.get(&FfiString::from(PRIVATE_ALLOC))? {
            self.allocations
                .extend(Array::<BufferVar>::try_from(value)?.iter());
        }
        if let Some(value) = callbacks.get(&FfiString::from(DEVICE_INIT))? {
            self.device_initializers
                .extend(Array::<Stmt>::try_from(value)?.iter());
        }
        if let Some(value) = callbacks.get(&FfiString::from(HOST_INIT))? {
            self.host_initializers
                .extend(Array::<Stmt>::try_from(value)?.iter());
        }
        if let Some(value) = callbacks.get(&FfiString::from(POST_BUFFER_DEF))? {
            for (buffer, statements) in Map::<BufferVar, Array<Stmt>>::try_from(value)?.iter() {
                let identity = ObjectIdentity::of(buffer.as_var());
                self.post_buffer_definitions
                    .entry(identity)
                    .or_insert_with(|| PendingBufferStatements {
                        source: buffer.clone(),
                        statements: Vec::new(),
                    })
                    .statements
                    .extend(statements.iter());
            }
        }
        self.shared_state = context.shared_state()?;
        Ok(implementation.body.clone())
    }

    fn append_buffer_statements(
        &mut self,
        original: &BufferVar,
        mapped: &BufferVar,
        result: &mut Vec<Stmt>,
    ) -> Result<()> {
        let mut identities = vec![ObjectIdentity::of(original.as_var())];
        if !mapped.same_as(original) {
            identities.push(ObjectIdentity::of(mapped.as_var()));
        }
        for identity in identities {
            let Some(pending) = self.post_buffer_definitions.remove(&identity) else {
                continue;
            };
            for statement in pending.statements {
                let remapped = remap_buffer(statement, &pending.source, mapped)?;
                result.push(replace_kernel_point(
                    remapped,
                    &Evaluate::from_i64(0)?.into(),
                )?);
            }
        }
        Ok(())
    }

    fn resolve_scope_target(&self, expression: &PrimExpr) -> Option<ScopeTarget> {
        let variable = expression.clone().try_cast::<Var>().ok()?;
        for level in self.scope_levels.iter().rev() {
            for definition in level {
                for (dimension, candidate) in definition.def_ids.iter().enumerate() {
                    if candidate.as_var().same_as(&variable) {
                        return Some(ScopeTarget {
                            binding: definition.scope,
                            dimension,
                            dimensions: definition.def_ids.len(),
                        });
                    }
                }
            }
        }
        None
    }

    fn target_overlaps_cluster_axes(&self, target: ScopeTarget) -> bool {
        target.binding == ScopeBinding::KERNEL_CTA && !self.cluster_axes.is_empty()
    }

    fn push_range(&mut self, range: &ScopeRange) -> Result<bool> {
        let Some(current) = self.execution_contexts.last().cloned() else {
            return Ok(false);
        };
        if range.target.binding == ScopeBinding::CLUSTER_CTA_PAIR {
            return if range.high == range.low + 1 && (0..=1).contains(&range.low) {
                self.push_cta_pair_value(range.low)
            } else {
                Ok(false)
            };
        }
        if self.target_overlaps_cluster_axes(range.target) {
            return Ok(false);
        }
        let narrowed = if range.target.dimensions == 1 {
            current.with_filter(range.target.binding, range.low, range.high)?
        } else {
            current.with_axis_filter(scope_axis(&range.target)?, range.low, range.high)?
        };
        if let Some(narrowed) = narrowed {
            self.execution_contexts.push(narrowed);
            Ok(true)
        } else {
            Ok(false)
        }
    }

    fn push_modulo(&mut self, target: ScopeTarget, modulus: i64, residue: i64) -> Result<bool> {
        let Some(current) = self.execution_contexts.last().cloned() else {
            return Ok(false);
        };
        if target.binding == ScopeBinding::CLUSTER_CTA_PAIR
            || self.target_overlaps_cluster_axes(target)
        {
            return Ok(false);
        }
        let axis = if target.dimensions != 1 {
            scope_axis(&target)?
        } else if target.binding == ScopeBinding::KERNEL_CTA
            || target.binding == ScopeBinding::CLUSTER_CTA
        {
            "cta_id"
        } else {
            return Ok(false);
        };
        if let Some(narrowed) = current.with_axis_modulo(axis, modulus, residue)? {
            self.execution_contexts.push(narrowed);
            Ok(true)
        } else {
            Ok(false)
        }
    }

    fn push_cta_pair_value(&mut self, value: i64) -> Result<bool> {
        let Some(current) = self.execution_contexts.last().cloned() else {
            return Ok(false);
        };
        if self.cluster_axes.is_empty() {
            return Ok(false);
        }
        if self.cluster_axes.len() == 1 {
            if let Some(narrowed) = current.with_axis_modulo("cta_id", 2, value)? {
                self.execution_contexts.push(narrowed);
                return Ok(true);
            }
            return Ok(false);
        }

        let mut parity_axis = None;
        let mut coefficient = 1_i64;
        let mut fixed = 0_i64;
        for (axis, extent) in &self.cluster_axes {
            let Some(range) = current.active.get(axis) else {
                return Ok(false);
            };
            let (Some(active_extent), Some(active_offset), Some(active_stride)) = (
                integer_value(&range.extent),
                integer_value(&range.offset),
                integer_value(&range.stride),
            ) else {
                return Ok(false);
            };
            fixed += coefficient * active_offset;
            if active_extent > 1 && (coefficient * active_stride) % 2 != 0 {
                if parity_axis.is_some() {
                    return Ok(false);
                }
                parity_axis = Some(axis.as_str());
            }
            coefficient *= extent;
        }
        let residue = (value - fixed).rem_euclid(2);
        let Some(parity_axis) = parity_axis else {
            if residue == 0 {
                self.execution_contexts.push(current);
                return Ok(true);
            }
            return Ok(false);
        };
        if let Some(narrowed) = current.with_axis_modulo(parity_axis, 2, residue)? {
            self.execution_contexts.push(narrowed);
            Ok(true)
        } else {
            Ok(false)
        }
    }

    fn push_selector(&mut self, target: ScopeTarget, selector: PrimExpr) -> Result<bool> {
        let Some(current) = self.execution_contexts.last().cloned() else {
            return Ok(false);
        };
        if target.dimensions != 1 || self.target_overlaps_cluster_axes(target) {
            return Ok(false);
        }
        if let Some(narrowed) = current.with_selector(target.binding, selector)? {
            self.execution_contexts.push(narrowed);
            Ok(true)
        } else {
            Ok(false)
        }
    }

    fn push_predicate_context(&mut self, predicate: &PrimExpr) -> Result<usize> {
        if self.execution_contexts.is_empty() {
            return Ok(0);
        }
        if predicate.clone().try_cast::<And>().is_ok() || self.bitwise_and_call(predicate).is_some()
        {
            return self.push_conjunction(predicate);
        }
        if let Some(call) = self.filter_call(predicate) {
            return self.push_filter_context(&call);
        }
        if let Some(range) = self.comparison_range(predicate)? {
            return Ok(usize::from(self.push_range(&range)?));
        }
        if let Some((target, modulus, residue)) = self.modulo_equality(predicate)? {
            return Ok(usize::from(self.push_modulo(target, modulus, residue)?));
        }
        if self.contains_elect_sync(predicate)? {
            if let Some(variable) = self.lane_scope_variable() {
                let target = ScopeTarget {
                    binding: ScopeBinding::WARP_THREAD,
                    dimension: 0,
                    dimensions: 1,
                };
                let selector = self.selector(variable, predicate.clone())?;
                return Ok(usize::from(self.push_selector(target, selector)?));
            }
        }
        Ok(0)
    }

    fn comparison_range(&self, predicate: &PrimExpr) -> Result<Option<ScopeRange>> {
        let (lhs, rhs, relation) = if let Ok(compare) = predicate.clone().try_cast::<EQ>() {
            (compare.a.clone(), compare.b.clone(), Comparison::Equal)
        } else if let Ok(compare) = predicate.clone().try_cast::<LT>() {
            (compare.a.clone(), compare.b.clone(), Comparison::Less)
        } else if let Ok(compare) = predicate.clone().try_cast::<LE>() {
            (compare.a.clone(), compare.b.clone(), Comparison::LessEqual)
        } else if let Ok(compare) = predicate.clone().try_cast::<GT>() {
            (compare.a.clone(), compare.b.clone(), Comparison::Greater)
        } else if let Ok(compare) = predicate.clone().try_cast::<GE>() {
            (
                compare.a.clone(),
                compare.b.clone(),
                Comparison::GreaterEqual,
            )
        } else {
            return Ok(None);
        };
        let difference: PrimExpr = crate::tirx::Sub::new(lhs, rhs)?.into();
        for (variable, target) in self.scope_targets() {
            let linear = detect_linear_equation(&difference, vec![variable.clone()])?;
            if linear.len() != 2 {
                continue;
            }
            let coefficient = self.analyzer.simplify(&linear.get(0)?)?;
            let base = self.analyzer.simplify(&linear.get(1)?)?;
            let (Some(coefficient), Some(base)) =
                (integer_value(&coefficient), integer_value(&base))
            else {
                continue;
            };
            if coefficient != 1 && coefficient != -1 {
                continue;
            }
            let exact = if coefficient == 1 { -base } else { base };
            let range = match (relation, coefficient) {
                (Comparison::Equal, _) => (exact, exact + 1),
                (Comparison::Less, 1) => (NEGATIVE_INFINITY, -base),
                (Comparison::Less, -1) => (base + 1, POSITIVE_INFINITY),
                (Comparison::LessEqual, 1) => (NEGATIVE_INFINITY, -base + 1),
                (Comparison::LessEqual, -1) => (base, POSITIVE_INFINITY),
                (Comparison::Greater, 1) => (-base + 1, POSITIVE_INFINITY),
                (Comparison::Greater, -1) => (NEGATIVE_INFINITY, base),
                (Comparison::GreaterEqual, 1) => (-base, POSITIVE_INFINITY),
                (Comparison::GreaterEqual, -1) => (NEGATIVE_INFINITY, base + 1),
                _ => unreachable!("linear predicate coefficients are restricted to -1 and 1"),
            };
            return Ok(Some(ScopeRange {
                target,
                low: range.0,
                high: range.1,
            }));
        }
        Ok(None)
    }

    fn modulo_equality(&self, predicate: &PrimExpr) -> Result<Option<(ScopeTarget, i64, i64)>> {
        let Ok(equal) = predicate.clone().try_cast::<EQ>() else {
            return Ok(None);
        };
        if let Some((target, modulus)) = self.modulo_target(&equal.a)? {
            if let Some(residue) = integer_value(&self.analyzer.simplify(&equal.b)?) {
                return Ok(Some((target, modulus, residue)));
            }
        }
        if let Some((target, modulus)) = self.modulo_target(&equal.b)? {
            if let Some(residue) = integer_value(&self.analyzer.simplify(&equal.a)?) {
                return Ok(Some((target, modulus, residue)));
            }
        }
        Ok(None)
    }

    fn modulo_target(&self, expression: &PrimExpr) -> Result<Option<(ScopeTarget, i64)>> {
        let operands = if let Ok(modulo) = expression.clone().try_cast::<Mod>() {
            Some((modulo.a.clone(), modulo.b.clone()))
        } else if let Ok(modulo) = expression.clone().try_cast::<FloorMod>() {
            Some((modulo.a.clone(), modulo.b.clone()))
        } else {
            None
        };
        let Some((value, modulus)) = operands else {
            return Ok(None);
        };
        let Some(target) = self.resolve_scope_target(&value) else {
            return Ok(None);
        };
        let Some(modulus) = integer_value(&self.analyzer.simplify(&modulus)?) else {
            return Ok(None);
        };
        Ok((modulus > 0).then_some((target, modulus)))
    }

    fn push_conjunction(&mut self, predicate: &PrimExpr) -> Result<usize> {
        let mut terms = Vec::new();
        self.flatten_conjunction(predicate, &mut terms)?;
        let mut consumed = vec![false; terms.len()];
        let mut ranges: Vec<(ScopeRange, Vec<usize>)> = Vec::new();

        for (index, term) in terms.iter().enumerate() {
            let Some(range) = self.comparison_range(term)? else {
                continue;
            };
            if let Some((merged, indices)) = ranges
                .iter_mut()
                .find(|(merged, _)| merged.target == range.target)
            {
                merged.low = merged.low.max(range.low);
                merged.high = merged.high.min(range.high);
                indices.push(index);
            } else {
                ranges.push((range, vec![index]));
            }
        }

        let mut pushed = 0;
        let mut progress = true;
        while progress {
            progress = false;
            for (range, indices) in &ranges {
                if indices.iter().all(|index| consumed[*index]) || range.low >= range.high {
                    continue;
                }
                if self.push_range(range)? {
                    for index in indices {
                        consumed[*index] = true;
                    }
                    pushed += 1;
                    progress = true;
                }
            }
            for (index, term) in terms.iter().enumerate() {
                if consumed[index] || ranges.iter().any(|(_, indices)| indices.contains(&index)) {
                    continue;
                }
                let pushed_term =
                    if let Some((target, modulus, residue)) = self.modulo_equality(term)? {
                        self.push_modulo(target, modulus, residue)?
                    } else {
                        false
                    };
                if pushed_term {
                    consumed[index] = true;
                    pushed += 1;
                    progress = true;
                }
            }
        }

        for (index, term) in terms.iter().enumerate() {
            if consumed[index] || ranges.iter().any(|(_, indices)| indices.contains(&index)) {
                continue;
            }
            pushed += self.push_predicate_context(term)?;
        }
        Ok(pushed)
    }

    fn flatten_conjunction(&self, predicate: &PrimExpr, terms: &mut Vec<PrimExpr>) -> Result<()> {
        if let Ok(and) = predicate.clone().try_cast::<And>() {
            self.flatten_conjunction(&and.a, terms)?;
            self.flatten_conjunction(&and.b, terms)?;
        } else if let Some(call) = self.bitwise_and_call(predicate) {
            self.flatten_conjunction(&PrimExpr::try_from(call.args.get(0)?)?, terms)?;
            self.flatten_conjunction(&PrimExpr::try_from(call.args.get(1)?)?, terms)?;
        } else {
            terms.push(predicate.clone());
        }
        Ok(())
    }

    fn filter_call(&self, predicate: &PrimExpr) -> Option<Call> {
        let call = predicate.clone().try_cast::<Call>().ok()?;
        (call.op.same_as(&self.filter_operator) && call.args.len() == 2).then_some(call)
    }

    fn bitwise_and_call(&self, predicate: &PrimExpr) -> Option<Call> {
        let call = predicate.clone().try_cast::<Call>().ok()?;
        (call.op.same_as(&self.bitwise_and_operator) && call.args.len() == 2).then_some(call)
    }

    fn push_filter_context(&mut self, call: &Call) -> Result<usize> {
        let variable = PrimExpr::try_from(call.args.get(0)?)?;
        let condition = PrimExpr::try_from(call.args.get(1)?)?;
        let mut pushed = 0;
        if let Some(target) = self.resolve_scope_target(&variable) {
            if self.contains_elect_sync(&condition)? {
                let selector = self.selector(variable, condition.clone())?;
                pushed += usize::from(self.push_selector(target, selector)?);
            }
        }
        Ok(pushed + self.push_predicate_context(&condition)?)
    }

    fn selector(&self, variable: PrimExpr, predicate: PrimExpr) -> Result<PrimExpr> {
        PrimExpr::try_from(Expr::from(Call::new(
            variable.type_annotation(),
            self.selector_operator.clone(),
            vec![variable.into(), predicate.into()],
        )))
    }

    fn contains_elect_sync(&self, predicate: &PrimExpr) -> Result<bool> {
        let mut found = false;
        structural_walk(
            predicate,
            |call: Call| {
                if call.op.same_as(&self.elect_sync_operator) {
                    found = true;
                    WalkResult::Interrupt
                } else {
                    WalkResult::Advance
                }
            },
            WalkOrder::PreOrder,
        )?;
        Ok(found)
    }

    fn lane_scope_variable(&self) -> Option<PrimExpr> {
        for level in self.scope_levels.iter().rev() {
            for definition in level {
                if definition.scope == ScopeBinding::WARP_THREAD && definition.def_ids.len() == 1 {
                    return definition.def_ids.iter().next().map(PrimExpr::from);
                }
            }
        }
        None
    }

    fn rewrite_filter_calls(&self, predicate: &PrimExpr) -> Result<PrimExpr> {
        if let Ok(and) = predicate.clone().try_cast::<And>() {
            let lhs = self.rewrite_filter_calls(&and.a)?;
            let rhs = self.rewrite_filter_calls(&and.b)?;
            if lhs.same_as(&and.a) && rhs.same_as(&and.b) {
                return Ok(predicate.clone());
            }
            return Ok(And::with_span(lhs, rhs, and.span.as_ref())?.into());
        }
        let Ok(call) = predicate.clone().try_cast::<Call>() else {
            return Ok(predicate.clone());
        };
        if call.op.same_as(&self.filter_operator) {
            if call.args.len() != 2 {
                return Err(value_error("tirx.filter expects exactly two arguments"));
            }
            let condition = PrimExpr::try_from(call.args.get(1)?)?;
            return self.as_boolean(self.rewrite_filter_calls(&condition)?);
        }

        let mut changed = false;
        let mut arguments = Vec::with_capacity(call.args.len());
        for argument in call.args.iter() {
            let mapped = match PrimExpr::try_from(argument.clone()) {
                Ok(argument) => self.rewrite_filter_calls(&argument)?.into(),
                Err(_) => argument.clone(),
            };
            changed |= !mapped.same_as(&argument);
            arguments.push(mapped);
        }
        if !changed {
            return Ok(predicate.clone());
        }
        PrimExpr::try_from(Expr::from(
            call.with_children(call.op.clone(), Array::new(arguments)),
        ))
    }

    fn as_boolean(&self, predicate: PrimExpr) -> Result<PrimExpr> {
        let dtype = predicate.type_annotation().dtype;
        if dtype.code == DLDataTypeCode::kDLBool as u8 {
            return Ok(predicate);
        }
        Ok(NE::new(predicate, IntImm::from_dtype(dtype, 0)?)?.into())
    }

    fn scope_targets(&self) -> Vec<(crate::tirx::PrimVar, ScopeTarget)> {
        let mut result = Vec::new();
        for level in self.scope_levels.iter().rev() {
            for definition in level {
                for (dimension, variable) in definition.def_ids.iter().enumerate() {
                    result.push((
                        variable,
                        ScopeTarget {
                            binding: definition.scope,
                            dimension,
                            dimensions: definition.def_ids.len(),
                        },
                    ));
                }
            }
        }
        result
    }
}

#[tvm_ffi::dispatch(mutate)]
impl TileDispatcher {
    fn mutate_attribute(&mut self, value: AttrStmt, mutator: &mut Mutator) -> Result<Stmt> {
        if value.attr_key.as_str() == DEVICE_ENTRY {
            return self.process_device_entry(value, mutator);
        }
        mutate_stmt_default(self, mutator, value.into())
    }

    fn mutate_scope_definition(
        &mut self,
        value: ScopeIdDefStmt,
        mutator: &mut Mutator,
    ) -> Result<Stmt> {
        if let Some(level) = self.scope_levels.last_mut() {
            level.push(ScopeDefinition::read(value.definition()?)?);
        }
        mutate_stmt_default(self, mutator, value.into())
    }

    fn mutate_binding(&mut self, value: Bind, mutator: &mut Mutator) -> Result<Stmt> {
        let mapped = mutate_stmt_default(self, mutator, value.into())?;
        let binding = mapped.clone().try_cast::<Bind>()?;
        if let Ok(expression) = binding.value.clone().try_cast::<PrimExpr>() {
            self.variable_ranges.insert(
                ObjectIdentity::of(&binding.var),
                (
                    binding.var.clone(),
                    Range::from_min_extent(expression, IntImm::new("int32", 1)?)?,
                ),
            );
        }
        Ok(mapped)
    }

    fn mutate_loop(&mut self, value: For, mutator: &mut Mutator) -> Result<Stmt> {
        self.variable_ranges.insert(
            ObjectIdentity::of(value.loop_var.as_var()),
            (
                value.loop_var.as_var().clone(),
                Range::from_min_extent(value.min.clone(), value.extent.clone())?,
            ),
        );
        mutate_stmt_default(self, mutator, value.into())
    }

    fn mutate_conditional(&mut self, value: IfThenElse, mutator: &mut Mutator) -> Result<Stmt> {
        let pushed = self.push_predicate_context(&value.condition)?;
        let condition: PrimExpr = mutator.mutate(self, &value.condition)?.try_into()?;
        let condition = self.rewrite_filter_calls(&condition)?;
        let then_case: Stmt = mutator.mutate(self, &value.then_case)?.try_into()?;
        for _ in 0..pushed {
            self.execution_contexts.pop();
        }
        let else_case = value
            .else_case
            .as_ref()
            .map(|branch| mutator.mutate(self, branch).and_then(Stmt::try_from))
            .transpose()?;
        if condition.same_as(&value.condition)
            && then_case.same_as(&value.then_case)
            && options_same(&else_case, &value.else_case)
        {
            return Ok(value.into());
        }
        Ok(IfThenElse::with_span(condition, then_case, else_case, value.span.as_ref())?.into())
    }

    fn mutate_tile_call(&mut self, value: TilePrimitiveCall) -> Result<Stmt> {
        self.dispatch_tile_call(value)
    }

    fn mutate_allocation(&mut self, value: AllocBuffer, mutator: &mut Mutator) -> Result<Stmt> {
        let original = value.buffer.clone();
        let mapped = mutate_stmt_default(self, mutator, value.into())?;
        let mapped_buffer = mapped.clone().try_cast::<AllocBuffer>()?.buffer.clone();
        self.storage_roots.insert(
            ObjectIdentity::of(mapped_buffer.as_var()),
            mapped_buffer.clone(),
        );
        if !mapped_buffer.same_as(&original) {
            self.storage_roots
                .insert(ObjectIdentity::of(original.as_var()), mapped_buffer.clone());
        }
        let mut statements = vec![mapped];
        self.append_buffer_statements(&original, &mapped_buffer, &mut statements)?;
        Stmt::sequence(statements)
    }

    fn mutate_declaration(&mut self, value: DeclBuffer, mutator: &mut Mutator) -> Result<Stmt> {
        let original = value.buffer.clone();
        let mapped = mutate_stmt_default(self, mutator, value.into())?;
        let declaration = mapped.clone().try_cast::<DeclBuffer>()?;
        let mapped_buffer = declaration.buffer.clone();
        let root = buffer_data_source(&declaration.data)
            .and_then(|source| self.storage_roots.get(&ObjectIdentity::of(source.as_var())))
            .cloned()
            .unwrap_or_else(|| mapped_buffer.clone());
        self.storage_roots
            .insert(ObjectIdentity::of(mapped_buffer.as_var()), root.clone());
        if !mapped_buffer.same_as(&original) {
            self.storage_roots
                .insert(ObjectIdentity::of(original.as_var()), root);
        }
        let mut statements = vec![mapped];
        self.append_buffer_statements(&original, &mapped_buffer, &mut statements)?;
        Stmt::sequence(statements)
    }

    fn mutate_sequence(&mut self, value: SeqStmt, mutator: &mut Mutator) -> Result<Stmt> {
        let mapped = mutate_stmt_default(self, mutator, value.into())?;
        let Ok(sequence) = mapped.clone().try_cast::<SeqStmt>() else {
            return Ok(mapped);
        };
        let mut statements = Vec::new();
        for statement in sequence.seq.iter() {
            statements.push(statement.clone());
            if let Ok(allocation) = statement.clone().try_cast::<AllocBuffer>() {
                self.append_buffer_statements(
                    &allocation.buffer,
                    &allocation.buffer,
                    &mut statements,
                )?;
            } else if let Ok(declaration) = statement.clone().try_cast::<DeclBuffer>() {
                self.append_buffer_statements(
                    &declaration.buffer,
                    &declaration.buffer,
                    &mut statements,
                )?;
            }
        }
        Stmt::sequence_with_span(statements, sequence.span.as_ref())
    }

    fn mutate_default(&mut self, value: &MapValue, mutator: &mut Mutator) -> Result<Any> {
        mutate_stmt_expr_default(self, mutator, value)
    }
}

fn gather_scope_defs(statement: &Stmt) -> Result<Vec<ScopeIdDef>> {
    let mut definitions = Vec::new();
    structural_walk(
        statement,
        |value: ScopeIdDefStmt| -> Result<WalkResult> {
            definitions.push(value.definition()?);
            Ok(WalkResult::Advance)
        },
        WalkOrder::PreOrder,
    )?;
    Ok(definitions)
}

fn remove_scope_definitions(statement: Stmt) -> Result<Stmt> {
    structural_map(
        statement,
        |_: ScopeIdDefStmt| Evaluate::from_i64(0).map(Stmt::from),
        WalkOrder::PostOrder,
    )?
    .try_into()
}

fn replace_kernel_point(statement: Stmt, body: &Stmt) -> Result<Stmt> {
    let operator = get_operator("tirx.tvm_kernel_replace_point")?;
    let replacement = body.clone();
    structural_map(
        statement,
        move |value: Evaluate| -> Result<Stmt> {
            let Ok(call) = value.value.clone().try_cast::<Call>() else {
                return Ok(value.into());
            };
            if call.op.same_as(&operator) {
                Ok(replacement.clone())
            } else {
                Ok(value.into())
            }
        },
        WalkOrder::PostOrder,
    )?
    .try_into()
}

fn remap_buffer(statement: Stmt, source: &BufferVar, target: &BufferVar) -> Result<Stmt> {
    if source.same_as(target) {
        return Ok(statement);
    }
    let source_identity = ObjectIdentity::of(source.as_var());
    let target = target.clone();
    structural_map(
        statement,
        move |value: BufferVar| {
            if ObjectIdentity::of(value.as_var()) == source_identity {
                target.clone()
            } else {
                value
            }
        },
        WalkOrder::PostOrder,
    )?
    .try_into()
}

fn resolve_storage_roots(
    statement: Stmt,
    roots: &HashMap<ObjectIdentity, BufferVar>,
) -> Result<Stmt> {
    if roots.is_empty() {
        return Ok(statement);
    }
    let buffer_data = get_operator("tirx.buffer_data")?;
    let roots = roots.clone();
    structural_map(
        statement,
        move |call: Call| -> Result<Expr> {
            if !call.op.same_as(&buffer_data) || call.args.len() != 1 {
                return Ok(call.into());
            }
            let Ok(buffer) = BufferVar::try_from(call.args.get(0)?) else {
                return Ok(call.into());
            };
            let Some(root) = roots.get(&ObjectIdentity::of(buffer.as_var())) else {
                return Ok(call.into());
            };
            tvm_ffi::cached_global_func!("tirx.BufferData")
                .call_tuple((root,))?
                .try_into()
        },
        WalkOrder::PostOrder,
    )?
    .try_into()
}

fn ensure_no_tile_calls(statement: &Stmt) -> Result<()> {
    let mut found = false;
    structural_walk(
        statement,
        |_: TilePrimitiveCall| {
            found = true;
            WalkResult::Interrupt
        },
        WalkOrder::PreOrder,
    )?;
    if found {
        Err(value_error("failed to lower a TilePrimitiveCall"))
    } else {
        Ok(())
    }
}

fn function_target(function: &PrimFunc) -> Result<Target> {
    if let Some(target) = function.attrs.dict.get(&FfiString::from("target"))? {
        return target.try_into();
    }
    Target::current(false)?.ok_or_else(|| value_error("TilePrimitiveDispatch requires a target"))
}

fn buffer_data_source(value: &Expr) -> Option<BufferVar> {
    let call = value.clone().try_cast::<Call>().ok()?;
    if !call.op.same_as(&get_operator("tirx.buffer_data").ok()?) || call.args.len() != 1 {
        return None;
    }
    BufferVar::try_from(call.args.get(0).ok()?).ok()
}

fn scope_axis(target: &ScopeTarget) -> Result<&'static str> {
    let axes = if target.binding == ScopeBinding::KERNEL_CTA {
        ["bx", "by", "bz"]
    } else if target.binding == ScopeBinding::CLUSTER_CTA {
        ["cbx", "cby", "cbz"]
    } else {
        return Err(value_error("multi-dimensional scope is not a CTA scope"));
    };
    axes.get(target.dimension)
        .copied()
        .ok_or_else(|| value_error("scope dimension exceeds three"))
}

fn options_same<T: ObjectRefCore>(lhs: &Option<T>, rhs: &Option<T>) -> bool {
    match (lhs, rhs) {
        (None, None) => true,
        (Some(lhs), Some(rhs)) => lhs.same_as(rhs),
        _ => false,
    }
}

fn get_operator(name: &str) -> Result<Expr> {
    tvm_ffi::cached_global_func!("ir.GetOp")
        .call_tuple((FfiString::from(name),))?
        .try_into()
}

fn integer_value(expression: &PrimExpr) -> Option<i64> {
    expression
        .clone()
        .try_cast::<IntImm>()
        .ok()
        .map(|value| value.value)
}

fn axis_name(dimension: usize) -> char {
    char::from(b'x' + u8::try_from(dimension).expect("launch dimensions are limited to three"))
}

fn launch_attribute_order(launch_params: &LaunchParams) -> Vec<IterVar> {
    const ORDINARY_ORDER: [&str; 6] = [
        "blockIdx.x",
        "threadIdx.x",
        "blockIdx.y",
        "blockIdx.z",
        "threadIdx.y",
        "threadIdx.z",
    ];
    const CLUSTER_ORDER: [&str; 12] = [
        "clusterCtaIdx.x",
        "blockIdx.z",
        "clusterCtaIdx.y",
        "clusterCtaIdx.z",
        "preferredClusterCtaIdx.x",
        "preferredClusterCtaIdx.y",
        "preferredClusterCtaIdx.z",
        "blockIdx.x",
        "threadIdx.x",
        "blockIdx.y",
        "threadIdx.y",
        "threadIdx.z",
    ];
    let order: &[&str] = if launch_params.contains_key("clusterCtaIdx.x") {
        &CLUSTER_ORDER
    } else {
        &ORDINARY_ORDER
    };
    let mut result = Vec::new();
    for tag in order {
        if let Some(iter_var) = launch_params.get(*tag) {
            result.push(iter_var.clone());
        }
    }
    result
}

fn value_error(message: &str) -> tvm_ffi::Error {
    tvm_ffi::Error::new(tvm_ffi::VALUE_ERROR, message, "")
}
