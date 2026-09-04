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
    structural_mutate, structural_visit, Any, Array, Map, MapValue, Mutator, ObjectIdentity,
    ObjectRefCast, ObjectRefCore, Result, String as FfiString, VisitCallbacks, VisitContext,
    VisitInterrupt, VisitValue,
};

use super::pointer_value_type_rewrite::{pointer_value_type_rewrite_with_options, RewriteOptions};
use super::utils::{
    array_same_as, get_operator, int_value, mutate_stmt_expr_default, operator_identity,
    value_error, visit_stmt_expr_default, with_prim_func_body,
};
use super::{create_prim_func_pass, Pass};
use crate::ir::{Call, Expr, IntImm, PointerType, PrimExpr, PrimType, TensorLoad, Var};
use crate::tirx::{
    AllocBuffer, BufferStore, BufferType, BufferVar, DeclBuffer, Evaluate, PrimFunc, Stmt,
};

/// Rewrite local storage using the same liveness and buffer-view model as
/// TVM's native `StorageRewrite` pass.
pub fn storage_rewrite_prim_func(function: PrimFunc) -> Result<PrimFunc> {
    let mut analysis = StorageAnalysis::new(&function.params)?;
    let mut callbacks = VisitCallbacks::new(
        analysis,
        (
            analyze_allocation,
            analyze_declaration,
            analyze_store,
            analyze_load,
            analyze_call,
            analyze_default,
        ),
    );
    structural_visit(&function.body, &mut callbacks)?;
    analysis = callbacks.into_state();

    let plan = StoragePlan::build(analysis)?;
    let root_allocations = plan.root_allocations()?;
    let mut rewriter = StoragePlanRewriter::new(plan)?;
    let rewritten: Stmt = structural_mutate(function.body.clone(), &mut rewriter)?.try_into()?;
    let body = if root_allocations.is_empty() {
        rewritten
    } else {
        let mut statements = root_allocations;
        statements.push(rewritten);
        Stmt::sequence(statements)?
    };
    let function = with_prim_func_body(function, body);

    // Match the native pass's final storage-type normalization.  Parameters
    // stay unchanged; only internal allocation/view element types and indices
    // are eligible for rewriting.
    pointer_value_type_rewrite_with_options(
        function,
        RewriteOptions {
            allow_untyped_pointers: true,
            rewrite_buffer_params: false,
            rewrite_pointer_params: false,
            rewrite_alloc_buffers: true,
            rewrite_indices: true,
            rewrite_let_bindings: true,
            detect_scalar_read_patterns: false,
        },
    )
}

/// Build TVM's `tirx.StorageRewrite` PrimFunc pass in Rust.
pub fn storage_rewrite() -> Result<Pass> {
    create_prim_func_pass(
        "tirx.StorageRewrite",
        0,
        Vec::new(),
        false,
        storage_rewrite_prim_func,
    )
}

#[derive(Clone)]
struct AllocationInfo {
    buffer: BufferVar,
    annotations: Map<FfiString, Any>,
    ordinal: usize,
    first_access: Option<usize>,
    last_access: Option<usize>,
}

struct StorageAnalysis {
    event: usize,
    allocations: Vec<AllocationInfo>,
    allocation_by_root: HashMap<ObjectIdentity, usize>,
    aliases: HashMap<ObjectIdentity, ObjectIdentity>,
}

impl StorageAnalysis {
    fn new(parameters: &Array<Var>) -> Result<Self> {
        let mut aliases = HashMap::new();
        for parameter in parameters.iter() {
            let identity = ObjectIdentity::of(&parameter);
            if BufferVar::try_from(parameter).is_ok() {
                aliases.insert(identity.clone(), identity);
            }
        }
        Ok(Self {
            event: 0,
            allocations: Vec::new(),
            allocation_by_root: HashMap::new(),
            aliases,
        })
    }

    fn tick(&mut self) -> usize {
        let event = self.event;
        self.event += 1;
        event
    }

    fn root(&self, variable: &Var) -> ObjectIdentity {
        let identity = ObjectIdentity::of(variable);
        self.aliases.get(&identity).cloned().unwrap_or(identity)
    }

    fn register_allocation(&mut self, value: &AllocBuffer) {
        let identity = ObjectIdentity::of(value.buffer.as_var());
        self.aliases.insert(identity.clone(), identity.clone());
        let ordinal = self.tick();
        self.allocation_by_root
            .insert(identity, self.allocations.len());
        self.allocations.push(AllocationInfo {
            buffer: value.buffer.clone(),
            annotations: value.annotations.clone(),
            ordinal,
            first_access: None,
            last_access: None,
        });
    }

    fn register_alias(&mut self, value: &DeclBuffer) {
        let identity = ObjectIdentity::of(value.buffer.as_var());
        let root = buffer_data_var(&value.data)
            .map(|variable| self.root(&variable))
            .unwrap_or_else(|| identity.clone());
        self.aliases.insert(identity, root);
        self.tick();
    }

    fn access(&mut self, buffer: &BufferVar) {
        let event = self.tick();
        let root = self.root(buffer.as_var());
        let Some(index) = self.allocation_by_root.get(&root).copied() else {
            return;
        };
        let allocation = &mut self.allocations[index];
        allocation.first_access.get_or_insert(event);
        allocation.last_access = Some(event);
    }

    fn access_variable(&mut self, variable: &Var) {
        let root = self.root(variable);
        let Some(index) = self.allocation_by_root.get(&root).copied() else {
            return;
        };
        let event = self.tick();
        let allocation = &mut self.allocations[index];
        allocation.first_access.get_or_insert(event);
        allocation.last_access = Some(event);
    }
}

fn analyze_allocation(
    value: AllocBuffer,
    visitor: &mut VisitContext<'_, StorageAnalysis>,
) -> Result<()> {
    visitor.state_mut().register_allocation(&value);
    visitor.visit_children().map(|_| ())
}

fn analyze_declaration(
    value: DeclBuffer,
    visitor: &mut VisitContext<'_, StorageAnalysis>,
) -> Result<()> {
    visitor.state_mut().register_alias(&value);
    visitor.visit_children().map(|_| ())
}

fn analyze_store(
    value: BufferStore,
    visitor: &mut VisitContext<'_, StorageAnalysis>,
) -> Result<()> {
    visitor.state_mut().access(&value.buffer);
    visitor.visit_children().map(|_| ())
}

fn analyze_load(value: TensorLoad, visitor: &mut VisitContext<'_, StorageAnalysis>) -> Result<()> {
    if let Ok(buffer) = BufferVar::try_from(&value.source) {
        visitor.state_mut().access(&buffer);
    }
    visitor.visit_children().map(|_| ())
}

fn analyze_call(value: Call, visitor: &mut VisitContext<'_, StorageAnalysis>) -> Result<()> {
    for argument in value.args.iter() {
        if let Ok(variable) = argument.clone().try_cast::<Var>() {
            visitor.state_mut().access_variable(&variable);
        }
    }
    visitor.visit_children().map(|_| ())
}

fn analyze_default(
    value: &VisitValue,
    visitor: &mut VisitContext<'_, StorageAnalysis>,
) -> Result<Option<VisitInterrupt>> {
    visit_stmt_expr_default(visitor, value)
}

#[derive(Clone)]
struct PlannedStorage {
    backing: BufferVar,
    allocation: AllocBuffer,
}

#[derive(Clone)]
struct BufferRemap {
    storage: usize,
    bit_offset: u64,
}

struct StoragePlan {
    storage: Vec<PlannedStorage>,
    remaps: HashMap<ObjectIdentity, BufferRemap>,
    aliases: HashMap<ObjectIdentity, ObjectIdentity>,
}

impl StoragePlan {
    fn build(mut analysis: StorageAnalysis) -> Result<Self> {
        analysis
            .allocations
            .sort_by_key(|allocation| allocation.ordinal);
        let mut storage = Vec::<PlannedStorage>::new();
        let mut remaps = HashMap::new();
        let mut active = Vec::<(usize, usize)>::new();

        // Special tagged memories (for example `local.L0A`) are represented
        // by one aligned allocation per storage scope, matching the native
        // merge rule even when constituent lifetimes overlap.
        let mut tagged_groups: HashMap<String, Vec<AllocationInfo>> = HashMap::new();
        let mut regular = Vec::new();
        for allocation in analysis.allocations {
            if allocation.first_access.is_none() {
                continue;
            }
            let scope = allocation.buffer.type_annotation().storage_scope.clone();
            if is_special_tagged_scope(scope.as_str()) {
                tagged_groups
                    .entry(scope.as_str().to_owned())
                    .or_default()
                    .push(allocation);
            } else {
                regular.push(allocation);
            }
        }

        for (_, group) in tagged_groups {
            let mut entries = Vec::<(Vec<AllocationInfo>, usize, u64)>::new();
            for allocation in group {
                let first_access = allocation.first_access.expect("used allocation");
                let last_access = allocation.last_access.expect("used allocation");
                let bits = constant_allocation_bits(&allocation.buffer)?.ok_or_else(|| {
                    value_error("special tagged storage requires a constant allocation size")
                })?;
                if let Some(entry) = entries
                    .iter_mut()
                    .find(|(_, free_after, _)| *free_after < first_access)
                {
                    entry.0.push(allocation);
                    entry.1 = last_access;
                    entry.2 = entry.2.max(bits);
                } else {
                    entries.push((vec![allocation], last_access, bits));
                }
            }

            let first = entries[0].0[0].clone();
            let first_type = first.buffer.type_annotation();
            let mut total_bits = 0_u64;
            let mut entry_offsets = Vec::with_capacity(entries.len());
            for (_, _, bits) in &entries {
                total_bits = align_to(total_bits, 32);
                entry_offsets.push(total_bits);
                total_bits = total_bits
                    .checked_add(*bits)
                    .ok_or_else(|| value_error("merged storage size overflow"))?;
            }
            let backing_dtype = if entries.len() == 1 {
                entries[0]
                    .0
                    .iter()
                    .map(|allocation| allocation.buffer.dtype().clone())
                    .max_by_key(|dtype| lane_count(dtype.dtype.lanes))
                    .expect("a tagged storage entry is non-empty")
            } else {
                PrimType::from_dtype(tvm_ffi::DLDataType {
                    lanes: 1,
                    ..first_type.dtype.dtype
                })?
            };
            let element_bits = u64::from(backing_dtype.dtype.bits)
                * u64::from(lane_count(backing_dtype.dtype.lanes));
            let elements = total_bits.div_ceil(element_bits);
            let extent = IntImm::from_dtype(
                first_type.shape.get(0)?.dtype(),
                i64::try_from(elements)
                    .map_err(|_| value_error("merged storage extent does not fit i64"))?,
            )?;
            let backing = if entries.len() == 1
                && total_bits == constant_allocation_bits(&first.buffer)?.expect("tagged size")
                && backing_dtype.dtype == first_type.dtype.dtype
            {
                first.buffer.clone()
            } else {
                rebuild_buffer(
                    &first.buffer,
                    BufferType::from_complete_fields(
                        first_type.span.clone(),
                        backing_dtype,
                        first_type.storage_scope.clone(),
                        Array::new(vec![extent.into()]),
                        Array::new(Vec::new()),
                        first_type.elem_offset.clone(),
                        first_type.data_alignment,
                        first_type.offset_factor,
                        first_type.layout.clone(),
                        first_type.allocated_addr.clone(),
                    ),
                    first.buffer.name.as_str(),
                )?
            };
            let annotations = merge_annotations(
                entries
                    .iter()
                    .flat_map(|(allocations, _, _)| allocations.iter())
                    .map(|entry| &entry.annotations),
            );
            let index = storage.len();
            storage.push(PlannedStorage {
                allocation: AllocBuffer::from_complete_fields(
                    first.buffer.span.clone(),
                    backing.clone(),
                    annotations,
                ),
                backing,
            });
            for ((allocations, _, _), bit_offset) in entries.into_iter().zip(entry_offsets) {
                for allocation in allocations {
                    remaps.insert(
                        ObjectIdentity::of(allocation.buffer.as_var()),
                        BufferRemap {
                            storage: index,
                            bit_offset,
                        },
                    );
                }
            }
        }

        for allocation in regular {
            let first_access = allocation.first_access.expect("used allocation");
            let last_access = allocation.last_access.expect("used allocation");
            let allocation_bits = constant_allocation_bits(&allocation.buffer)?;
            let ty = allocation.buffer.type_annotation();
            let reusable = allocation_bits.is_some()
                && ty.shape.len() == 1
                && !is_scalable_dtype(ty.dtype.dtype.lanes)
                && !(storage_base_scope(ty.storage_scope.as_str()) == "local"
                    && allocation_bits.is_some_and(|bits| bits <= 32));

            let candidate = if reusable {
                active.iter().position(|(storage_index, free_after)| {
                    *free_after < first_access
                        && compatible_storage(&storage[*storage_index].backing, &allocation.buffer)
                })
            } else {
                None
            };

            let storage_index = if let Some(candidate) = candidate {
                let (storage_index, _) = active.remove(candidate);
                let existing = storage[storage_index].backing.clone();
                let merged = merge_reused_storage(&existing, &allocation.buffer)?;
                storage[storage_index].backing = merged.clone();
                storage[storage_index].allocation = AllocBuffer::from_complete_fields(
                    allocation.buffer.span.clone(),
                    merged,
                    merge_annotations([
                        &storage[storage_index].allocation.annotations,
                        &allocation.annotations,
                    ]),
                );
                storage_index
            } else {
                let storage_index = storage.len();
                storage.push(PlannedStorage {
                    backing: allocation.buffer.clone(),
                    allocation: allocation.clone().into_alloc_buffer(),
                });
                storage_index
            };
            active.push((storage_index, last_access));
            remaps.insert(
                ObjectIdentity::of(allocation.buffer.as_var()),
                BufferRemap {
                    storage: storage_index,
                    bit_offset: 0,
                },
            );
        }

        Ok(Self {
            storage,
            remaps,
            aliases: analysis.aliases,
        })
    }

    fn root_allocations(&self) -> Result<Vec<Stmt>> {
        Ok(self
            .storage
            .iter()
            .map(|entry| Stmt::from(entry.allocation.clone()))
            .collect())
    }

    fn root(&self, variable: &Var) -> ObjectIdentity {
        let identity = ObjectIdentity::of(variable);
        self.aliases.get(&identity).cloned().unwrap_or(identity)
    }

    fn remap(&self, buffer: &BufferVar) -> Option<(&PlannedStorage, &BufferRemap)> {
        let root = self.root(buffer.as_var());
        let remap = self.remaps.get(&root)?;
        Some((&self.storage[remap.storage], remap))
    }
}

impl AllocationInfo {
    fn into_alloc_buffer(self) -> AllocBuffer {
        AllocBuffer::from_complete_fields(self.buffer.span.clone(), self.buffer, self.annotations)
    }
}

struct StoragePlanRewriter {
    plan: StoragePlan,
    buffer_views: HashMap<ObjectIdentity, BufferVar>,
}

impl StoragePlanRewriter {
    fn new(plan: StoragePlan) -> Result<Self> {
        Ok(Self {
            plan,
            buffer_views: HashMap::new(),
        })
    }

    fn remap_buffer(&mut self, buffer: &BufferVar) -> Result<Option<(BufferVar, u64)>> {
        let Some((storage, remap)) = self.plan.remap(buffer) else {
            return Ok(None);
        };
        let identity = ObjectIdentity::of(buffer.as_var());
        if let Some(existing) = self.buffer_views.get(&identity) {
            return Ok(Some((existing.clone(), remap.bit_offset)));
        }
        let mapped = if buffer.same_as(&storage.backing) {
            buffer.clone()
        } else {
            rebuild_buffer(
                buffer,
                buffer.type_annotation(),
                storage.backing.name.as_str(),
            )?
        };
        self.buffer_views.insert(identity, mapped.clone());
        Ok(Some((mapped, remap.bit_offset)))
    }

    fn remap_indices(
        &self,
        buffer: &BufferVar,
        mut indices: Array<PrimExpr>,
        bit_offset: u64,
    ) -> Result<Array<PrimExpr>> {
        if bit_offset == 0 {
            return Ok(indices);
        }
        if indices.is_empty() {
            return Err(value_error("a remapped buffer access requires an index"));
        }
        let element_bits = u64::from(buffer.dtype().dtype.bits);
        if !bit_offset.is_multiple_of(element_bits) {
            return Err(value_error(
                "a merged storage offset is not aligned to the accessed element type",
            ));
        }
        let last = indices.get(indices.len() - 1)?;
        let offset = IntImm::from_dtype(
            last.dtype(),
            i64::try_from(bit_offset / element_bits)
                .map_err(|_| value_error("a merged storage offset does not fit i64"))?,
        )?;
        let last = semantic_add(offset.into(), last)?;
        let mut values = indices.iter().collect::<Vec<_>>();
        values[indices.len() - 1] = last;
        indices = Array::new(values);
        Ok(indices)
    }
}

#[tvm_ffi::dispatch(mutate)]
impl StoragePlanRewriter {
    fn mutate_allocation(&mut self, value: AllocBuffer) -> Result<Stmt> {
        // Every used allocation has been hoisted to its planned root storage;
        // unused allocations are intentionally removed as well.
        if self.plan.remap(&value.buffer).is_some() {
            if let Some((mapped, _)) = self.remap_buffer(&value.buffer)? {
                let (_, remap) = self.plan.remap(&value.buffer).expect("checked above");
                let storage = &self.plan.storage[remap.storage];
                if value.buffer.same_as(&storage.backing) || mapped.same_as(&storage.backing) {
                    return Evaluate::from_i64(0).map(Into::into);
                }
                return Ok(DeclBuffer::from_complete_fields(
                    value.span.clone(),
                    mapped,
                    buffer_data(&storage.backing)?,
                )
                .into());
            }
        }
        Evaluate::from_i64(0).map(Into::into)
    }

    fn mutate_declaration(&mut self, value: DeclBuffer, mutator: &mut Mutator) -> Result<Stmt> {
        let data: Expr = mutator.mutate(self, &value.data)?.try_into()?;
        let Some((buffer, _)) = self.remap_buffer(&value.buffer)? else {
            if data.same_as(&value.data) {
                return Ok(value.into());
            }
            return Ok(value.copy_with(value.buffer.clone(), data).into());
        };
        let (_, remap) = self.plan.remap(&value.buffer).expect("remapped buffer");
        let storage = &self.plan.storage[remap.storage];
        Ok(value
            .copy_with(buffer, buffer_data(&storage.backing)?)
            .into())
    }

    fn mutate_load(&mut self, value: TensorLoad, mutator: &mut Mutator) -> Result<PrimExpr> {
        let old_buffer = BufferVar::try_from(&value.source)?;
        let indices: Array<PrimExpr> = mutator.mutate(self, &value.indices)?.try_into()?;
        let Some((buffer, bit_offset)) = self.remap_buffer(&old_buffer)? else {
            if array_same_as(&indices, &value.indices) {
                return Ok(value.into());
            }
            return tvm_ffi::cached_global_func!("tirx.BufferLoad")
                .call_tuple((value.source.clone(), indices, value.span.clone()))?
                .try_into();
        };
        let indices = self.remap_indices(&old_buffer, indices, bit_offset)?;
        tvm_ffi::cached_global_func!("tirx.BufferLoad")
            .call_tuple((buffer.as_var().clone(), indices, value.span.clone()))?
            .try_into()
    }

    fn mutate_store(&mut self, value: BufferStore, mutator: &mut Mutator) -> Result<Stmt> {
        let stored_value: PrimExpr = mutator.mutate(self, &value.value)?.try_into()?;
        let indices: Array<PrimExpr> = mutator.mutate(self, &value.indices)?.try_into()?;
        let Some((buffer, bit_offset)) = self.remap_buffer(&value.buffer)? else {
            if stored_value.same_as(&value.value) && array_same_as(&indices, &value.indices) {
                return Ok(value.into());
            }
            return Ok(value
                .copy_with(value.buffer.clone(), stored_value, indices)
                .into());
        };
        let indices = self.remap_indices(&value.buffer, indices, bit_offset)?;
        Ok(value.copy_with(buffer, stored_value, indices).into())
    }

    fn mutate_variable(&mut self, value: Var) -> Result<Expr> {
        let Ok(buffer) = BufferVar::try_from(value.clone()) else {
            return Ok(value.into());
        };
        let Some((_, remap)) = self.plan.remap(&buffer) else {
            return Ok(value.into());
        };
        if remap.bit_offset != 0 {
            // Preserve the native warning-worthy behavior: an opaque use of an
            // offset view can only refer to the backing allocation itself.
        }
        Ok(self.plan.storage[remap.storage]
            .backing
            .as_var()
            .clone()
            .into())
    }

    fn mutate_default(&mut self, value: &MapValue, mutator: &mut Mutator) -> Result<Any> {
        mutate_stmt_expr_default(self, mutator, value)
    }
}

fn compatible_storage(lhs: &BufferVar, rhs: &BufferVar) -> bool {
    let lhs = lhs.type_annotation();
    let rhs = rhs.type_annotation();
    lhs.storage_scope == rhs.storage_scope && lhs.shape.len() == 1 && rhs.shape.len() == 1
}

fn merge_reused_storage(lhs: &BufferVar, rhs: &BufferVar) -> Result<BufferVar> {
    let lhs_type = lhs.type_annotation();
    let rhs_type = rhs.type_annotation();
    let lhs_bits = constant_allocation_bits(lhs)?;
    let rhs_bits = constant_allocation_bits(rhs)?;
    if lhs_type.dtype.dtype == rhs_type.dtype.dtype && lhs_bits >= rhs_bits {
        return Ok(lhs.clone());
    }
    let (dtype, bits) =
        if lane_count(rhs_type.dtype.dtype.lanes) > lane_count(lhs_type.dtype.dtype.lanes) {
            (rhs_type.dtype.clone(), lhs_bits.max(rhs_bits))
        } else {
            (lhs_type.dtype.clone(), lhs_bits.max(rhs_bits))
        };
    let bits = bits.ok_or_else(|| value_error("reused storage must have constant size"))?;
    let element_bits = u64::from(dtype.dtype.bits) * u64::from(lane_count(dtype.dtype.lanes));
    let extent = IntImm::from_dtype(
        lhs_type.shape.get(0)?.dtype(),
        i64::try_from(bits.div_ceil(element_bits))
            .map_err(|_| value_error("reused storage extent does not fit i64"))?,
    )?;
    rebuild_buffer(
        lhs,
        BufferType::from_complete_fields(
            lhs_type.span.clone(),
            dtype,
            lhs_type.storage_scope.clone(),
            Array::new(vec![extent.into()]),
            Array::new(Vec::new()),
            lhs_type.elem_offset.clone(),
            lhs_type.data_alignment,
            lhs_type.offset_factor,
            lhs_type.layout.clone(),
            lhs_type.allocated_addr.clone(),
        ),
        lhs.name.as_str(),
    )
}

fn constant_allocation_bits(buffer: &BufferVar) -> Result<Option<u64>> {
    let ty = buffer.type_annotation();
    let mut elements = 1_u64;
    for extent in ty.shape.iter() {
        let Some(extent) = int_value(&extent).and_then(|literal| u64::try_from(literal).ok())
        else {
            return Ok(None);
        };
        elements = elements
            .checked_mul(extent)
            .ok_or_else(|| value_error("allocation size overflow"))?;
    }
    let element_bits = u64::from(ty.dtype.dtype.bits)
        .checked_mul(u64::from(lane_count(ty.dtype.dtype.lanes)))
        .ok_or_else(|| value_error("allocation element size overflow"))?;
    elements
        .checked_mul(element_bits)
        .map(Some)
        .ok_or_else(|| value_error("allocation size overflow"))
}

fn rebuild_buffer(buffer: &BufferVar, ty: BufferType, name: &str) -> Result<BufferVar> {
    BufferVar::try_from(buffer.copy_with(FfiString::from(name), ty.into()))
}

fn buffer_data(buffer: &BufferVar) -> Result<Expr> {
    let ty = buffer.type_annotation();
    let pointer = PointerType::new(ty.dtype.clone(), ty.storage_scope.as_str())?;
    Ok(Call::new(
        pointer,
        get_operator("tirx.buffer_data")?,
        vec![buffer.as_var().clone().into()],
    )
    .into())
}

fn buffer_data_var(value: &Expr) -> Option<Var> {
    if let Ok(variable) = value.clone().try_cast::<Var>() {
        return Some(variable);
    }
    let call = value.clone().try_cast::<Call>().ok()?;
    if ObjectIdentity::of(&call.op) != operator_identity("tirx.buffer_data").ok()?
        || call.args.len() != 1
    {
        return None;
    }
    call.args.get(0).ok()?.try_cast().ok()
}

fn merge_annotations<'a>(
    annotations: impl IntoIterator<Item = &'a Map<FfiString, Any>>,
) -> Map<FfiString, Any> {
    let mut values = Vec::new();
    let mut volatile = false;
    for annotation in annotations {
        for (key, value) in annotation.iter() {
            if key.as_str() == "volatile" {
                volatile = true;
            } else if !values
                .iter()
                .any(|(existing, _): &(FfiString, Any)| existing == &key)
            {
                values.push((key, value));
            }
        }
    }
    if volatile {
        values.push((FfiString::from("volatile"), Any::from(true)));
    }
    Map::from_iter(values)
}

fn is_special_tagged_scope(scope: &str) -> bool {
    scope
        .split_once('.')
        .is_some_and(|(_, tag)| !matches!(tag, "" | "dyn" | "workspace" | "vtcm"))
}

fn storage_base_scope(scope: &str) -> &str {
    scope.split_once('.').map_or(scope, |(base, _)| base)
}

fn is_scalable_dtype(lanes: u16) -> bool {
    (lanes as i16) < 0
}

fn lane_count(lanes: u16) -> u16 {
    let signed = lanes as i16;
    if signed < 0 {
        signed.unsigned_abs()
    } else {
        lanes
    }
}

fn align_to(value: u64, alignment: u64) -> u64 {
    value.div_ceil(alignment) * alignment
}

fn semantic_add(lhs: PrimExpr, rhs: PrimExpr) -> Result<PrimExpr> {
    tvm_ffi::cached_global_func!("tirx._OpAdd")
        .call_tuple((lhs, rhs, Option::<crate::ir::Span>::None))?
        .try_into()
}
