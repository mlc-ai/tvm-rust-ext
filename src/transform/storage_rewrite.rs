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
    structural_mutate, structural_visit, Any, Array, DLDataTypeExt, Map, MapValue, Mutator,
    ObjectIdentity, ObjectRefCast, ObjectRefCore, Result, String as FfiString, VisitCallbacks,
    VisitContext, VisitInterrupt, VisitValue,
};

use super::pointer_value_type_rewrite::{pointer_value_type_rewrite_with_options, RewriteOptions};
use super::utils::{
    array_same_as, binary_op, get_operator, int_value, mutate_stmt_expr_default, value_error,
    visit_buffer_definition, visit_stmt_expr_default, with_prim_func_body,
};
use super::{create_prim_func_pass_with_context, Pass};
use crate::analysis::Analyzer;
use crate::ir::{Call, CallObj, Expr, IntImm, PrimExpr, PrimType, TensorLoad, Var, VarObj};
use crate::tirx::{
    AllocBuffer, AssertStmt, AttrStmt, Bind, BufferStore, BufferType, BufferVar, DeclBuffer,
    Evaluate, For, ForKind, IfThenElse, PrimFunc, Return, Stmt, While,
};

/// Rewrite local storage using the same liveness and buffer-view model as
/// TVM's native `StorageRewrite` pass.
pub fn storage_rewrite_prim_func(function: PrimFunc) -> Result<PrimFunc> {
    storage_rewrite_with_reuse(function, true)
}

fn storage_rewrite_with_reuse(function: PrimFunc, enable_reuse: bool) -> Result<PrimFunc> {
    let target = function
        .attrs
        .dict
        .get(&FfiString::from("target"))?
        .map(crate::target::Target::try_from)
        .transpose()?;
    let require_exact_dtype = target
        .as_ref()
        .map(|target| target.kind_name())
        .transpose()?
        .is_some_and(|kind| matches!(kind.as_str(), "vulkan" | "webgpu"));
    let mut analysis = StorageAnalysis::new(&function.params);
    let mut callbacks = VisitCallbacks::new(
        analysis,
        (
            analyze_allocation,
            analyze_declaration,
            analyze_store,
            analyze_load,
            analyze_variable,
            analyze_default,
        ),
    );
    structural_visit(function.body(), &mut callbacks)?;
    analysis = callbacks.into_state();

    let plan = StoragePlan::build(analysis, enable_reuse, require_exact_dtype)?;
    let root_allocations = plan.allocations_at(None);
    let mut rewriter = StoragePlanRewriter::new(plan)?;
    let rewritten: Stmt = structural_mutate(function.body().clone(), &mut rewriter)?.try_into()?;
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
    // SAFETY: the callback has no captured state.
    unsafe {
        create_prim_func_pass_with_context(
            "tirx.StorageRewrite",
            0,
            Vec::new(),
            false,
            |function, context| {
                let merge_static_smem = context
                    .config()?
                    .get(&FfiString::from("tirx.merge_static_smem"))?
                    .map(bool::try_from)
                    .transpose()?
                    .unwrap_or(false);
                storage_rewrite_with_reuse(function, !merge_static_smem)
            },
        )
    }
}

struct AllocationInfo {
    buffer: BufferVar,
    storage_scope: StorageScope,
    annotations: Map<FfiString, Any>,
    scope_depth: usize,
    attach_scope: Option<ObjectIdentity>,
    first_access: Option<usize>,
    last_access: Option<usize>,
}

struct StorageAnalysis {
    event: usize,
    allocations: Vec<AllocationInfo>,
    allocation_by_root: HashMap<ObjectIdentity, usize>,
    aliases: HashMap<ObjectIdentity, ObjectIdentity>,
    scopes: Vec<AccessScope>,
    completed_scopes: Vec<AccessScope>,
    thread_scope: Option<ObjectIdentity>,
    in_thread_env: bool,
}

struct AccessScope {
    // A buffer allocated outside a loop remains live across the whole loop,
    // not merely until the last access encountered during one AST traversal.
    begin: usize,
    end: usize,
    statement: Stmt,
    touched: Vec<usize>,
    attach_scope: Option<ObjectIdentity>,
}

impl StorageAnalysis {
    fn new(parameters: &Array<Var>) -> Self {
        let mut aliases = HashMap::new();
        for parameter in parameters.iter() {
            let identity = ObjectIdentity::of(&parameter);
            if BufferVar::try_from(parameter).is_ok() {
                aliases.insert(identity.clone(), identity);
            }
        }
        Self {
            event: 0,
            allocations: Vec::new(),
            allocation_by_root: HashMap::new(),
            aliases,
            scopes: Vec::new(),
            completed_scopes: Vec::new(),
            thread_scope: None,
            in_thread_env: false,
        }
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

    fn register_allocation(&mut self, value: &AllocBuffer) -> Result<()> {
        let storage_scope =
            StorageScope::parse(value.buffer.type_annotation().storage_scope.as_str())?;
        let identity = ObjectIdentity::of(value.buffer.as_var());
        self.aliases.insert(identity.clone(), identity.clone());
        self.allocation_by_root
            .insert(identity, self.allocations.len());
        self.allocations.push(AllocationInfo {
            buffer: value.buffer.clone(),
            storage_scope,
            annotations: value.annotations.clone(),
            scope_depth: self.scopes.len(),
            attach_scope: None,
            first_access: None,
            last_access: None,
        });
        Ok(())
    }

    fn register_alias(&mut self, value: &DeclBuffer) -> Result<()> {
        let identity = ObjectIdentity::of(value.buffer.as_var());
        let root = if let Some(source) =
            buffer_data_var(&value.data).filter(|source| BufferVar::try_from(source).is_ok())
        {
            self.aliases
                .get(&ObjectIdentity::of(&source))
                .cloned()
                .ok_or_else(|| {
                    value_error("buffer alias source must be registered before its DeclBuffer")
                })?
        } else {
            identity.clone()
        };
        self.aliases.insert(identity, root);
        Ok(())
    }

    fn access_buffer(&mut self, buffer: &BufferVar) -> Result<()> {
        if let Some(&index) = self.allocation_by_root.get(&self.root(buffer.as_var())) {
            if self.allocations[index].buffer.type_annotation().shape.len() != 1 {
                return Err(value_error(
                    "StorageRewrite requires flattened buffer allocations",
                ));
            }
        }
        self.access_variable(buffer.as_var())
    }

    fn access_variable(&mut self, variable: &Var) -> Result<()> {
        let root = self.root(variable);
        let Some(index) = self.allocation_by_root.get(&root).copied() else {
            return Ok(());
        };
        let depth = self.allocations[index].scope_depth;
        let scope = self.scopes.get_mut(depth).ok_or_else(|| {
            value_error("buffer access occurs outside its allocation's statement scope")
        })?;
        if !scope.touched.contains(&index) {
            scope.touched.push(index);
        }
        Ok(())
    }

    fn enter_scope(&mut self, statement: Stmt) {
        let begin = self.tick();
        self.scopes.push(AccessScope {
            begin,
            end: begin,
            statement,
            touched: Vec::new(),
            attach_scope: self.thread_scope.clone(),
        });
    }

    fn exit_scope(&mut self) {
        let end = self.tick();
        let mut scope = self.scopes.pop().expect("storage scopes are balanced");
        scope.end = end;
        for &index in &scope.touched {
            let allocation = &mut self.allocations[index];
            if allocation.first_access.is_none() {
                allocation.first_access = Some(scope.begin);
                allocation.attach_scope = scope.attach_scope.clone();
            }
            allocation.last_access = Some(end);
        }
        self.completed_scopes.push(scope);
    }
}

fn analyze_allocation(
    value: AllocBuffer,
    visitor: &mut VisitContext<'_, StorageAnalysis>,
) -> Result<()> {
    visitor.state_mut().register_allocation(&value)?;
    visit_buffer_definition(visitor, &value.buffer).map(|_| ())
}

fn analyze_declaration(
    value: DeclBuffer,
    visitor: &mut VisitContext<'_, StorageAnalysis>,
) -> Result<()> {
    visitor.state_mut().register_alias(&value)
}

fn analyze_store(
    value: BufferStore,
    visitor: &mut VisitContext<'_, StorageAnalysis>,
) -> Result<()> {
    visitor.state_mut().enter_scope(value.clone().into());
    let result = (|| {
        visitor.visit(&value.value)?;
        for index in value.indices.iter() {
            visitor.visit(&index)?;
        }
        visitor.state_mut().access_buffer(&value.buffer)
    })();
    visitor.state_mut().exit_scope();
    result
}

fn analyze_load(value: TensorLoad, visitor: &mut VisitContext<'_, StorageAnalysis>) -> Result<()> {
    for index in value.indices.iter() {
        visitor.visit(&index)?;
    }
    let buffer = BufferVar::try_from(&value.source)?;
    visitor.state_mut().access_buffer(&buffer)
}

fn analyze_variable(value: Var, visitor: &mut VisitContext<'_, StorageAnalysis>) -> Result<()> {
    visitor.state_mut().access_variable(&value)
}

fn is_node<T: ObjectRefCore>(value: &VisitValue) -> bool {
    value.as_node::<T::ContainerType>().is_some()
}

struct InplaceVerifier {
    destination: BufferVar,
    source: BufferVar,
    store: Option<BufferStore>,
    memory_depth: usize,
    at_root: bool,
}

impl InplaceVerifier {
    fn check(statement: &Stmt, destination: &BufferVar, source: &BufferVar) -> Result<bool> {
        let state = Self {
            destination: destination.clone(),
            source: source.clone(),
            store: None,
            memory_depth: 0,
            at_root: true,
        };
        let mut callbacks = VisitCallbacks::new(state, (verify_inplace,));
        Ok(structural_visit(statement, &mut callbacks)?.is_none())
    }
}

fn verify_inplace(
    value: &VisitValue,
    visitor: &mut VisitContext<'_, InplaceVerifier>,
) -> Result<Option<VisitInterrupt>> {
    let reject = || Ok(Some(VisitInterrupt::with(false)));
    if visitor.state().at_root {
        visitor.state_mut().at_root = false;
        if !(is_node::<AttrStmt>(value)
            || is_node::<For>(value)
            || is_node::<IfThenElse>(value)
            || is_node::<While>(value)
            || is_node::<BufferStore>(value))
        {
            return reject();
        }
    }
    if let Some(variable) = value.cast::<Var>() {
        return if variable.same_as(&visitor.state().source)
            || variable.same_as(&visitor.state().destination)
        {
            reject()
        } else {
            Ok(None)
        };
    }
    if let Some(store) = value.cast::<BufferStore>() {
        visitor.state_mut().memory_depth += 1;
        for index in store.indices.iter() {
            if let Some(interrupt) = visitor.visit(&index)? {
                return Ok(Some(interrupt));
            }
        }
        visitor.state_mut().memory_depth -= 1;
        let is_destination = store.buffer.same_as(&visitor.state().destination);
        if is_destination {
            visitor.state_mut().store = Some(store.clone());
        }
        let result = visitor.visit(&store.value);
        if is_destination {
            visitor.state_mut().store = None;
        }
        return result;
    }
    if let Some(load) = value.cast::<TensorLoad>() {
        if load.source.same_as(&visitor.state().destination) || visitor.state().memory_depth != 0 {
            return reject();
        }
        if load.source.same_as(&visitor.state().source) {
            let Some(store) = &visitor.state().store else {
                return reject();
            };
            if !store.value.ty.same_as(&load.ty) {
                return reject();
            }
            if store.indices.len() != load.indices.len() {
                return Err(value_error(
                    "in-place store/load have different index counts",
                ));
            }
            for (lhs, rhs) in store.indices.iter().zip(load.indices.iter()) {
                if !expr_deep_equal(&lhs, &rhs)? {
                    return reject();
                }
            }
        }
        visitor.state_mut().memory_depth += 1;
        let result = visit_stmt_expr_default(visitor, value);
        visitor.state_mut().memory_depth -= 1;
        return result;
    }
    if value
        .cast::<AttrStmt>()
        .is_some_and(|attr| attr.attr_key.as_str() == "extern_scope")
    {
        return reject();
    }
    if let Some(allocation) = value.cast::<AllocBuffer>() {
        if allocation
            .annotations
            .get(&FfiString::from("tirx.volatile"))?
            .is_some()
        {
            return reject();
        }
    }
    visit_stmt_expr_default(visitor, value)
}

fn analyze_default(
    value: &VisitValue,
    visitor: &mut VisitContext<'_, StorageAnalysis>,
) -> Result<Option<VisitInterrupt>> {
    let Some(statement) = value.cast::<Stmt>() else {
        return visit_stmt_expr_default(visitor, value);
    };
    let attribute = value.cast::<AttrStmt>();
    let is_thread = attribute.as_ref().is_some_and(|attribute| {
        attribute.attr_key.as_str() == "thread_extent" && !visitor.state().in_thread_env
    });
    let is_virtual_thread = attribute
        .as_ref()
        .is_some_and(|attribute| attribute.attr_key.as_str() == "virtual_thread");
    let is_parallel = value.cast::<For>().is_some_and(|loop_node| {
        loop_node.kind == ForKind::kParallel && visitor.state().thread_scope.is_none()
    });
    let is_scope = is_thread
        || is_virtual_thread
        || attribute
            .as_ref()
            .is_some_and(|attr| attr.attr_key.as_str() == "extern_scope")
        || is_node::<For>(value)
        || is_node::<While>(value)
        || is_node::<IfThenElse>(value)
        || is_node::<AssertStmt>(value)
        || is_node::<Evaluate>(value)
        || is_node::<Return>(value)
        || is_node::<Bind>(value);
    if !is_scope {
        return visit_stmt_expr_default(visitor, value);
    }
    let old_thread_scope = visitor.state().thread_scope.clone();
    let old_thread_env = visitor.state().in_thread_env;
    if (is_thread || is_virtual_thread) && old_thread_scope.is_some() {
        return Err(value_error(
            "nested storage attachment scopes are not supported",
        ));
    }
    visitor.state_mut().enter_scope(statement.clone());
    if is_thread || is_virtual_thread || is_parallel {
        visitor.state_mut().thread_scope = Some(ObjectIdentity::of(&statement));
    }
    visitor.state_mut().in_thread_env |= is_thread;
    let result = visit_stmt_expr_default(visitor, value);
    visitor.state_mut().exit_scope();
    visitor.state_mut().thread_scope = old_thread_scope;
    visitor.state_mut().in_thread_env = old_thread_env;
    result
}

struct PlannedStorage {
    backing: BufferVar,
    allocation: AllocBuffer,
    attach_scope: Option<ObjectIdentity>,
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
    fn build(
        analysis: StorageAnalysis,
        enable_reuse: bool,
        require_exact_dtype: bool,
    ) -> Result<Self> {
        let allocations = &analysis.allocations;
        let mut entries = Vec::<StorageEntry>::new();
        let mut assigned = HashMap::<usize, usize>::new();
        let mut free = Vec::<usize>::new();
        let mut replaced_inplace = HashSet::new();
        let mut events = Vec::new();
        for scope in &analysis.completed_scopes {
            events.push((scope.begin, true, scope));
            events.push((scope.end, false, scope));
        }
        events.sort_by_key(|(position, _, _)| *position);

        for (_, is_begin, scope) in events {
            if !is_begin {
                // Freed storage inside a thread/parallel scope cannot escape it.
                let identity = ObjectIdentity::of(&scope.statement);
                free.retain(|&index| entries[index].attach_scope.as_ref() != Some(&identity));
                for &index in &scope.touched {
                    if allocations[index].last_access == Some(scope.end)
                        && !replaced_inplace.contains(&index)
                    {
                        let entry = assigned[&index];
                        if entries[entry].can_free(allocations) {
                            free.push(entry);
                        }
                    }
                }
                continue;
            }

            let generated: Vec<_> = scope
                .touched
                .iter()
                .copied()
                .filter(|&index| allocations[index].first_access == Some(scope.begin))
                .collect();
            for &index in &generated {
                let allocation = &allocations[index];
                let dtype = allocation.buffer.dtype();
                let bits = constant_allocation_bits(&allocation.buffer)?.unwrap_or(0);
                let mut candidate = None;
                // The native planner considers at most two newly-live buffers,
                // and transfers ownership only after checking the actual accesses.
                if generated.len() <= 2 && !is_scalable_dtype(dtype.dtype.lanes) {
                    for &source in &scope.touched {
                        if allocations[source].last_access != Some(scope.end)
                            || replaced_inplace.contains(&source)
                        {
                            continue;
                        }
                        let Some(&entry) = assigned.get(&source) else {
                            continue;
                        };
                        if entries[entry].matches(allocation, allocations)
                            && entries[entry].element_type.dtype == scalar_dtype(dtype)
                            && entries[entry].constant_bits == bits
                            && InplaceVerifier::check(
                                &scope.statement,
                                &allocation.buffer,
                                &allocations[source].buffer,
                            )?
                        {
                            replaced_inplace.insert(source);
                            candidate = Some(entry);
                            break;
                        }
                    }
                }
                if candidate.is_none() && enable_reuse && reusable_allocation(allocation, bits) {
                    candidate = find_free_entry(
                        &mut free,
                        &entries,
                        allocations,
                        allocation,
                        bits,
                        require_exact_dtype,
                    );
                }
                let entry = if let Some(entry) = candidate {
                    entries[entry].constant_bits = entries[entry].constant_bits.max(bits);
                    entry
                } else {
                    let entry = entries.len();
                    entries.push(StorageEntry {
                        allocations: Vec::new(),
                        attach_scope: allocation.attach_scope.clone(),
                        constant_bits: bits,
                        element_type: PrimType::from_dtype(scalar_dtype(dtype))?,
                    });
                    entry
                };
                entries[entry].allocations.push(index);
                assigned.insert(index, entry);
            }
        }

        let analyzer = Analyzer::new()?;
        let mut storage = Vec::new();
        let mut remaps = HashMap::new();
        let mut merged = HashSet::new();
        for (index, entry) in entries.iter().enumerate() {
            if merged.contains(&index) {
                continue;
            }
            let first = &allocations[entry.allocations[0]];
            let mut group = vec![index];
            if first.storage_scope.is_special_tagged() {
                for (other, next) in entries.iter().enumerate().skip(index + 1) {
                    if next.matches(first, allocations) {
                        group.push(other);
                        merged.insert(other);
                    }
                }
            }
            let mut offsets = Vec::new();
            let backing = if group.len() == 1 {
                offsets.push(0);
                prepare_backing(entry, allocations, &analyzer)?
            } else {
                let mut total_bits = 0_u64;
                for &index in &group {
                    let bits = entries[index].constant_bits;
                    if bits == 0 {
                        return Err(value_error(
                            "special tagged storage requires a constant allocation size",
                        ));
                    }
                    offsets.push(total_bits);
                    total_bits = align_to(
                        total_bits
                            .checked_add(bits)
                            .ok_or_else(|| value_error("merged storage size overflow"))?,
                        32,
                    )?;
                }
                let extent = IntImm::from_dtype(
                    first.buffer.type_annotation().shape.get(0)?.dtype(),
                    i64::try_from(total_bits.div_ceil(u64::from(entry.element_type.dtype.bits)))
                        .map_err(|_| value_error("merged storage extent does not fit i64"))?,
                )?;
                resized_buffer(&first.buffer, entry.element_type.clone(), extent.into())?
            };
            let storage_index = storage.len();
            let annotations = merge_annotations(
                group
                    .iter()
                    .flat_map(|&index| &entries[index].allocations)
                    .map(|&index| &allocations[index].annotations),
            );
            storage.push(PlannedStorage {
                allocation: AllocBuffer::from_complete_fields(None, backing.clone(), annotations),
                backing,
                attach_scope: entry.attach_scope.clone(),
            });
            for (&index, bit_offset) in group.iter().zip(offsets) {
                for &allocation in &entries[index].allocations {
                    remaps.insert(
                        ObjectIdentity::of(&allocations[allocation].buffer),
                        BufferRemap {
                            storage: storage_index,
                            bit_offset,
                        },
                    );
                }
            }
        }
        Ok(Self {
            storage,
            remaps,
            aliases: analysis.aliases,
        })
    }

    fn allocations_at(&self, scope: Option<&ObjectIdentity>) -> Vec<Stmt> {
        self.storage
            .iter()
            .filter(|entry| entry.attach_scope.as_ref() == scope)
            .map(|entry| Stmt::from(entry.allocation.clone()))
            .collect()
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

struct StoragePlanRewriter {
    plan: StoragePlan,
    buffer_views: HashMap<ObjectIdentity, BufferVar>,
    masked_load: Expr,
    masked_store: Expr,
    access_ptr: Expr,
}

impl StoragePlanRewriter {
    fn new(plan: StoragePlan) -> Result<Self> {
        Ok(Self {
            plan,
            buffer_views: HashMap::new(),
            masked_load: get_operator("tirx.masked_load")?,
            masked_store: get_operator("tirx.masked_store")?,
            access_ptr: get_operator("tirx.tvm_access_ptr")?,
        })
    }

    fn attach_allocations(&self, scope: &impl ObjectRefCore, body: Stmt) -> Result<Stmt> {
        let mut allocations = self.plan.allocations_at(Some(&ObjectIdentity::of(scope)));
        if allocations.is_empty() {
            return Ok(body);
        }
        allocations.push(body);
        Stmt::sequence(allocations)
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
            buffer.with_name_and_type(storage.backing.name.clone(), buffer.type_annotation())?
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
        let last = indices.get(indices.len() - 1)?;
        let last = remap_offset(last, bit_offset, u64::from(buffer.dtype().dtype.bits))?;
        let mut values = indices.iter().collect::<Vec<_>>();
        values[indices.len() - 1] = last;
        indices = Array::new(values);
        Ok(indices)
    }
}

#[tvm_ffi::dispatch(mutate)]
impl StoragePlanRewriter {
    fn mutate_call(&mut self, value: Call, mutator: &mut Mutator) -> Result<Expr> {
        let is_load = value.op.same_as(&self.masked_load);
        if is_load || value.op.same_as(&self.masked_store) {
            let first_index = if is_load { 1 } else { 2 };
            if value.args.len() < first_index + 2 {
                return Err(value_error(
                    "masked access requires a buffer, index, and predicate",
                ));
            }
            let source = BufferVar::try_from(value.args.get(0)?)?;
            let stored: Option<PrimExpr> = if is_load {
                None
            } else {
                Some(mutator.mutate(self, &value.args.get(1)?)?.try_into()?)
            };
            let indices = value
                .args
                .iter()
                .skip(first_index)
                .take(value.args.len() - first_index - 1)
                .map(|index| mutator.mutate(self, &index).and_then(PrimExpr::try_from))
                .collect::<Result<Vec<_>>>()?;
            let (buffer, bit_offset) = self.remap_buffer(&source)?.unwrap_or((source.clone(), 0));
            let indices = self.remap_indices(&source, Array::new(indices), bit_offset)?;
            let ty = if is_load {
                TensorLoad::from_buffer_with_span(
                    buffer.as_var().clone(),
                    indices.iter().map(Into::into).collect(),
                    value.span.as_ref(),
                )?
                .ty
                .clone()
            } else {
                crate::ir::Type::from(PrimType::void())
            };
            let predicate: Expr = mutator
                .mutate(self, &value.args.get(value.args.len() - 1)?)?
                .try_into()?;
            let mut arguments = vec![buffer.into()];
            arguments.extend(stored.map(Expr::from));
            arguments.extend(indices.iter().map(Expr::from));
            arguments.push(predicate);
            return Ok(value
                .copy_with(ty, value.op.clone(), Array::new(arguments))
                .into());
        }
        if value.op.same_as(&self.access_ptr) {
            if value.args.len() != 5 {
                return Err(value_error("tvm_access_ptr requires five arguments"));
            }
            if let Some(variable) = buffer_data_var(&value.args.get(1)?) {
                let root = self.plan.root(&variable);
                if let Some(remap) = self.plan.remaps.get(&root).cloned() {
                    let marker: PrimExpr = value.args.get(0)?.try_into()?;
                    let offset: PrimExpr = mutator.mutate(self, &value.args.get(2)?)?.try_into()?;
                    let extent: PrimExpr = mutator.mutate(self, &value.args.get(3)?)?.try_into()?;
                    let dtype = marker.dtype();
                    let offset = remap_offset(
                        offset,
                        remap.bit_offset,
                        u64::from(dtype.bits) * u64::from(dtype.lanes),
                    )?;
                    return Ok(Call::with_metadata(
                        value.ty.clone(),
                        value.op.clone(),
                        vec![
                            marker.into(),
                            self.plan.storage[remap.storage]
                                .backing
                                .as_var()
                                .clone()
                                .into(),
                            offset.into(),
                            extent.into(),
                            value.args.get(4)?,
                        ],
                        value.attrs.clone(),
                        Vec::new(),
                        value.span.as_ref(),
                    )
                    .into());
                }
            }
        }
        super::utils::mutate_expr_default(self, mutator, value.into())
    }

    fn mutate_loop(&mut self, value: For, mutator: &mut Mutator) -> Result<For> {
        if value.kind == ForKind::kVectorized {
            return Err(value_error("VectorizeLoop must run before StorageRewrite"));
        }
        let mapped: For =
            super::utils::mutate_stmt_default(self, mutator, value.clone().into())?.try_cast()?;
        let body = self.attach_allocations(&value, mapped.body.clone())?;
        if body.same_as(&mapped.body) {
            return Ok(mapped);
        }
        For::with_metadata(
            mapped.loop_var.as_var().clone(),
            mapped.min.clone().into(),
            mapped.extent.clone().into(),
            mapped.kind,
            body,
            mapped.thread_binding.clone(),
            mapped.annotations.clone(),
            mapped.step.as_ref().map(|step| step.clone().into()),
            None,
        )
    }

    fn mutate_attribute(&mut self, value: AttrStmt, mutator: &mut Mutator) -> Result<AttrStmt> {
        let mapped: AttrStmt =
            super::utils::mutate_stmt_default(self, mutator, value.clone().into())?.try_cast()?;
        let body = self.attach_allocations(&value, mapped.body.clone())?;
        if body.same_as(&mapped.body) {
            return Ok(mapped);
        }
        AttrStmt::new(
            mapped.node.clone(),
            mapped.attr_key.as_str(),
            mapped.value.clone(),
            body,
        )
    }

    fn mutate_allocation(&mut self, value: AllocBuffer) -> Result<Stmt> {
        // Every used allocation has been hoisted to its planned attachment scope;
        // unused allocations are intentionally removed as well.
        if let Some((mapped, _)) = self.remap_buffer(&value.buffer)? {
            let (storage, _) = self.plan.remap(&value.buffer).expect("remapped buffer");
            if !value.buffer.same_as(&storage.backing) && !mapped.same_as(&storage.backing) {
                return Ok(DeclBuffer::new(mapped, storage.backing.data()?)?.into());
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
        Ok(value.copy_with(buffer, storage.backing.data()?).into())
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

    fn mutate_variable(&mut self, value: Var) -> Expr {
        let Ok(buffer) = BufferVar::try_from(value.clone()) else {
            return value.into();
        };
        let Some((_, remap)) = self.plan.remap(&buffer) else {
            return value.into();
        };
        if remap.bit_offset != 0 {
            // Preserve the native warning-worthy behavior: an opaque use of an
            // offset view can only refer to the backing allocation itself.
        }
        self.plan.storage[remap.storage]
            .backing
            .as_var()
            .clone()
            .into()
    }

    fn mutate_default(&mut self, value: &MapValue, mutator: &mut Mutator) -> Result<Any> {
        mutate_stmt_expr_default(self, mutator, value)
    }
}

struct StorageEntry {
    allocations: Vec<usize>,
    attach_scope: Option<ObjectIdentity>,
    constant_bits: u64,
    element_type: PrimType,
}

impl StorageEntry {
    fn matches(&self, allocation: &AllocationInfo, allocations: &[AllocationInfo]) -> bool {
        self.attach_scope == allocation.attach_scope
            && allocations[self.allocations[0]].storage_scope == allocation.storage_scope
    }

    fn can_free(&self, allocations: &[AllocationInfo]) -> bool {
        let first = &allocations[self.allocations[0]];
        let ty = first.buffer.type_annotation();
        !first.storage_scope.tag.is_empty()
            || (!first.storage_scope.is_thread_private()
                && !is_scalable_dtype(ty.dtype.dtype.lanes)
                && (self.constant_bits == 0 || self.constant_bits > 32))
    }
}

fn scalar_dtype(dtype: &PrimType) -> tvm_ffi::DLDataType {
    tvm_ffi::DLDataType {
        lanes: 1,
        ..dtype.dtype
    }
}

fn reusable_allocation(allocation: &AllocationInfo, bits: u64) -> bool {
    let ty = allocation.buffer.type_annotation();
    let small = allocation.storage_scope.tag.is_empty()
        && (allocation.storage_scope.is_thread_private() || (bits != 0 && bits <= 32));
    ty.shape.len() == 1 && !is_scalable_dtype(ty.dtype.dtype.lanes) && !small
}

fn find_free_entry(
    free: &mut Vec<usize>,
    entries: &[StorageEntry],
    allocations: &[AllocationInfo],
    allocation: &AllocationInfo,
    bits: u64,
    exact_dtype: bool,
) -> Option<usize> {
    let dtype = allocation.buffer.dtype();
    let matches = |entry: &StorageEntry| entry.matches(allocation, allocations);
    let position = if bits == 0 {
        // Symbolic allocations use the native FIFO policy, with equal scalar types.
        free.iter().position(|&index| {
            let entry = &entries[index];
            entry.constant_bits == 0
                && matches(entry)
                && entry.element_type.dtype == scalar_dtype(dtype)
        })
    } else {
        let candidates = || {
            free.iter().enumerate().filter(|&(_, &index)| {
                let entry = &entries[index];
                entry.constant_bits != 0
                    && matches(entry)
                    && (!exact_dtype || entry.element_type.dtype == dtype.dtype)
            })
        };
        // Try the smallest sufficient block first, then grow the largest smaller
        // block.  Native StorageRewrite limits both searches to a factor of 16.
        candidates()
            .filter(|&(_, &index)| {
                entries[index].constant_bits >= bits
                    && entries[index].constant_bits <= bits.saturating_mul(16)
            })
            .min_by_key(|&(position, &index)| (entries[index].constant_bits, position))
            .or_else(|| {
                candidates()
                    .filter(|&(_, &index)| {
                        entries[index].constant_bits < bits
                            && entries[index].constant_bits >= bits / 16
                            && entries[index].element_type.dtype == scalar_dtype(dtype)
                    })
                    .max_by_key(|&(position, &index)| (entries[index].constant_bits, position))
            })
            .map(|(position, _)| position)
    };
    position.map(|position| free.remove(position))
}

fn expr_deep_equal(lhs: &PrimExpr, rhs: &PrimExpr) -> Result<bool> {
    tvm_ffi::cached_global_func!("tirx.analysis.expr_deep_equal")
        .call_tuple((lhs, rhs))?
        .try_into()
}

fn resized_buffer(buffer: &BufferVar, dtype: PrimType, extent: PrimExpr) -> Result<BufferVar> {
    // A combined allocation has fresh compact metadata, as in native BufferType.
    let ty = BufferType::new(
        buffer.type_annotation().storage_scope.as_str(),
        &dtype.dtype.to_string(),
        vec![extent.into()],
    )?;
    buffer.with_name_and_type(buffer.name.clone(), ty)
}

fn prepare_backing(
    entry: &StorageEntry,
    allocations: &[AllocationInfo],
    analyzer: &Analyzer,
) -> Result<BufferVar> {
    let first = &allocations[entry.allocations[0]].buffer;
    let mut identical = true;
    let mut dtype = first.dtype().clone();
    for &index in entry.allocations.iter().skip(1) {
        let buffer = &allocations[index].buffer;
        if buffer.dtype().dtype.lanes > dtype.dtype.lanes {
            dtype = buffer.dtype().clone();
        }
        let first_ty = first.type_annotation();
        let ty = buffer.type_annotation();
        if ty.dtype.dtype != first_ty.dtype.dtype || ty.shape.len() != first_ty.shape.len() {
            identical = false;
        } else {
            for (lhs, rhs) in first_ty.shape.iter().zip(ty.shape.iter()) {
                identical &= expr_deep_equal(&lhs, &rhs)?;
            }
        }
    }
    if allocations[entry.allocations[0]]
        .storage_scope
        .is_special_tagged()
        && entry.constant_bits == 0
    {
        return Err(value_error(
            "special tagged storage requires a constant allocation size",
        ));
    }
    if identical {
        return Ok(first.clone());
    }

    let mut size = None;
    for &index in &entry.allocations {
        let buffer = &allocations[index].buffer;
        let ty = buffer.type_annotation();
        if ty.shape.len() != 1 {
            return Err(value_error("reused storage requires a flat allocation"));
        }
        let bits = i64::from(ty.dtype.dtype.bits) * i64::from(ty.dtype.dtype.lanes);
        let mut extent = ty.shape.get(0)?;
        if int_value(&extent).is_some_and(|value| value > i64::from(i32::MAX) / bits) {
            extent = IntImm::new("int64", int_value(&extent).expect("constant extent"))?.into();
        }
        let size_bits = binary_op("tirx._OpMul", extent, IntImm::new("int32", bits)?.into())?;
        size = Some(match size {
            Some(previous) => binary_op("tirx._OpMax", previous, size_bits)?,
            None => size_bits,
        });
    }
    let size = size.expect("a storage entry has at least one allocation");
    let bits: PrimExpr = IntImm::new(
        "int32",
        i64::from(dtype.dtype.bits) * i64::from(dtype.dtype.lanes),
    )?
    .into();
    let remainder = binary_op("tirx._OpFloorMod", size.clone(), bits.clone())?;
    let divisible = analyzer.can_prove(&binary_op(
        "tirx._OpEQ",
        remainder,
        IntImm::new("int32", 0)?.into(),
    )?)?;
    let mut extent = binary_op("tirx._OpFloorDiv", size, bits)?;
    if !divisible {
        extent = binary_op("tirx._OpAdd", extent, IntImm::new("int32", 1)?.into())?;
    }
    resized_buffer(first, dtype, analyzer.simplify(&extent)?)
}

fn constant_allocation_bits(buffer: &BufferVar) -> Result<Option<u64>> {
    let ty = buffer.type_annotation();
    if is_scalable_dtype(ty.dtype.dtype.lanes) {
        return Ok(None);
    }
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

fn buffer_data_var(value: &Expr) -> Option<Var> {
    if value.as_node::<VarObj>().is_some() {
        return value.clone().try_cast().ok();
    }
    let call = value.as_node::<CallObj>()?;
    if !call.op.same_as(&get_operator("tirx.buffer_data").ok()?) || call.args.len() != 1 {
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
        for (key, _) in annotation.iter() {
            if key.as_str() == "tirx.volatile" {
                volatile = true;
            }
        }
    }
    if volatile {
        values.push((FfiString::from("tirx.volatile"), Any::from(true)));
    }
    Map::from_iter(values)
}

#[derive(Clone, PartialEq, Eq)]
struct StorageScope {
    base: &'static str,
    tag: String,
}

impl StorageScope {
    fn parse(scope: &str) -> Result<Self> {
        let scope = if scope.is_empty() { "global" } else { scope };
        // Dots in built-in names such as wmma.matrix_a are not storage tags.
        for base in [
            "global",
            "shared",
            "warp",
            "local",
            "wmma.matrix_a",
            "wmma.matrix_b",
            "wmma.accumulator",
            "texture",
            "amx.tmm",
            "m16n8k8.matrixA",
            "m16n8k8.matrixB",
            "m16n8k8.matrixC",
            "metal.simdgroup",
            "metal.cooperative_tensor",
            "trn.sbuf",
            "trn.psum",
        ] {
            if let Some(tag) = scope.strip_prefix(base) {
                return Ok(Self {
                    base,
                    tag: tag.to_owned(),
                });
            }
        }
        Err(value_error(&format!("unknown storage scope {scope}")))
    }

    fn is_thread_private(&self) -> bool {
        !matches!(self.base, "global" | "shared")
    }

    fn is_special_tagged(&self) -> bool {
        !matches!(self.tag.as_str(), "" | ".dyn" | ".workspace" | ".vtcm")
    }
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

fn align_to(value: u64, alignment: u64) -> Result<u64> {
    value
        .div_ceil(alignment)
        .checked_mul(alignment)
        .ok_or_else(|| value_error("aligned storage size overflow"))
}

fn remap_offset(index: PrimExpr, bit_offset: u64, element_bits: u64) -> Result<PrimExpr> {
    if element_bits == 0 || !bit_offset.is_multiple_of(element_bits) {
        return Err(value_error(
            "a merged storage offset is not aligned to the accessed element type",
        ));
    }
    if bit_offset == 0 {
        return Ok(index);
    }
    let offset = IntImm::from_dtype(
        index.dtype(),
        i64::try_from(bit_offset / element_bits)
            .map_err(|_| value_error("a merged storage offset does not fit i64"))?,
    )?;
    binary_op("tirx._OpAdd", offset.into(), index)
}
