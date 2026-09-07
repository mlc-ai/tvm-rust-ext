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
    structural_mutate, structural_visit, structural_walk, Any, Array, MapValue, Mutator,
    ObjectIdentity, ObjectRefCast, ObjectRefCore, Result, String, VisitCallbacks, VisitContext,
    VisitInterrupt, VisitValue, WalkOrder, WalkResult,
};

use super::utils::{
    array_same_as, binary_op, get_operator, int_value as optional_int_value, mutate_expr_default,
    mutate_stmt_expr_default, value_error, visit_stmt_expr_default, with_prim_func_body,
};
use super::{create_prim_func_pass, Pass};
use crate::analysis::{detect_linear_equation, Analyzer};
use crate::ir::prim::Ramp;
use crate::ir::{Call, CallObj, Expr, IntImm, PrimExpr, PrimType, Range, TensorLoad, Var};
use crate::target::Target;
use crate::tirx::{
    AllocBuffer, AttrStmt, BufferStore, BufferType, BufferVar, DeclBuffer, For, IterVar, PrimFunc,
    PrimVar, SeqStmt, Stmt,
};

const THREAD_EXTENT: &str = "thread_extent";
const VIRTUAL_THREAD: &str = "virtual_thread";

/// Lower warp-scoped buffers to thread-local buffers and shuffle operations.
pub fn lower_warp_memory_prim_func(function: PrimFunc) -> Result<PrimFunc> {
    let target = function_target(&function)?;
    let warp_size = target
        .export()?
        .get(&String::from("thread_warp_size"))?
        .map(i64::try_from)
        .transpose()?
        .unwrap_or(1);
    if warp_size == 1 {
        return Ok(function);
    }
    let warp_size = i32::try_from(warp_size).map_err(|_| value_error("warp size exceeds i32"))?;

    let analyzer = Analyzer::new()?;
    bind_variable_bounds(function.body(), &analyzer)?;
    let mut rewriter = WarpMemoryRewriter::new(warp_size, analyzer);
    let body: Stmt = structural_mutate(function.body().clone(), &mut rewriter)?.try_into()?;
    let body = update_pointer_storage_scope(body, rewriter.new_storage_scopes)?;
    Ok(with_prim_func_body(function, body))
}

/// Build TVM's `tirx.LowerWarpMemory` pass in Rust.
pub fn lower_warp_memory() -> Result<Pass> {
    create_prim_func_pass(
        "tirx.LowerWarpMemory",
        0,
        Vec::new(),
        false,
        lower_warp_memory_prim_func,
    )
}

struct BoundBinder {
    analyzer: Analyzer,
    thread_domains: HashMap<ObjectIdentity, Range>,
}

fn bind_variable_bounds(body: &Stmt, analyzer: &Analyzer) -> Result<()> {
    let mut callbacks = VisitCallbacks::new(
        BoundBinder {
            analyzer: analyzer.clone(),
            thread_domains: HashMap::new(),
        },
        (bind_loop, bind_attribute, bind_default),
    );
    structural_visit(body, &mut callbacks).map(|_| ())
}

fn bind_loop(value: For, visitor: &mut VisitContext<'_, BoundBinder>) -> Result<()> {
    visitor.state().analyzer.bind(
        value.loop_var.as_var(),
        &Range::from_min_extent(value.min.clone(), value.extent.clone())?,
    )?;
    visitor.visit_children()?;
    Ok(())
}

fn bind_attribute(value: AttrStmt, visitor: &mut VisitContext<'_, BoundBinder>) -> Result<()> {
    if matches!(value.attr_key.as_str(), THREAD_EXTENT | VIRTUAL_THREAD) {
        let iteration = IterVar::try_from(value.node.clone())?;
        if iteration.thread_tag()?.is_empty() {
            return Err(value_error("thread extent requires a tagged IterVar"));
        }
        let variable = iteration.var()?;
        let identity = ObjectIdentity::of(variable.as_var());
        if !visitor.state().thread_domains.contains_key(&identity) {
            let domain = Range::from_min_extent(IntImm::new("int32", 0)?, value.value.clone())?;
            visitor.state().analyzer.bind(variable.as_var(), &domain)?;
            visitor.state_mut().thread_domains.insert(identity, domain);
        }
    }
    visitor.visit_children()?;
    Ok(())
}

fn bind_default(
    value: &VisitValue,
    visitor: &mut VisitContext<'_, BoundBinder>,
) -> Result<Option<VisitInterrupt>> {
    visit_stmt_expr_default(visitor, value)
}

struct WarpIndexState {
    warp_size: i32,
    warp_index: Option<PrimVar>,
    width: i32,
}

fn find_warp_index(body: &Stmt, warp_size: i32) -> Result<(PrimVar, i32)> {
    let mut callbacks = VisitCallbacks::new(
        WarpIndexState {
            warp_size,
            warp_index: None,
            width: 0,
        },
        (find_warp_attribute, find_warp_default),
    );
    structural_visit(body, &mut callbacks)?;
    let state = callbacks.into_state();
    state
        .warp_index
        .map(|variable| (variable, state.width))
        .ok_or_else(|| value_error("cannot find threadIdx.x within warp-memory scope"))
}

fn find_warp_attribute(
    value: AttrStmt,
    visitor: &mut VisitContext<'_, WarpIndexState>,
) -> Result<()> {
    if value.attr_key.as_str() == THREAD_EXTENT {
        let iteration = IterVar::try_from(value.node.clone())?;
        if iteration.thread_tag()?.as_str() == "threadIdx.x" {
            let width = int_value(&value.value)?;
            let width =
                i32::try_from(width).map_err(|_| value_error("thread extent exceeds i32"))?;
            if width <= 0
                || width > visitor.state().warp_size
                || visitor.state().warp_size % width != 0
            {
                return Err(value_error(
                    "threadIdx.x extent must be a positive factor of the target warp size",
                ));
            }
            let variable = iteration.var()?;
            if let Some(existing) = &visitor.state().warp_index {
                if !existing.same_as(&variable) {
                    return Err(value_error(
                        "multiple threadIdx.x variables occur in one warp-memory scope",
                    ));
                }
            } else {
                visitor.state_mut().width = width;
                visitor.state_mut().warp_index = Some(variable);
            }
        }
    }
    visitor.visit_children()?;
    Ok(())
}

fn find_warp_default(
    value: &VisitValue,
    visitor: &mut VisitContext<'_, WarpIndexState>,
) -> Result<Option<VisitInterrupt>> {
    visit_stmt_expr_default(visitor, value)
}

struct WarpCoeffState {
    buffer: BufferVar,
    warp_index: PrimVar,
    analyzer: Analyzer,
    coefficient: i64,
    mma_fill: Expr,
    ptx_ldmatrix: Expr,
    mma_fill_legacy: Expr,
    buffer_data: Expr,
}

fn find_warp_coefficient(
    body: &Stmt,
    buffer: &BufferVar,
    warp_index: PrimVar,
    analyzer: &Analyzer,
) -> Result<i64> {
    let mut callbacks = VisitCallbacks::new(
        WarpCoeffState {
            buffer: buffer.clone(),
            warp_index,
            analyzer: analyzer.clone(),
            coefficient: 0,
            mma_fill: get_operator("tirx.mma_fill")?,
            ptx_ldmatrix: get_operator("tirx.ptx_legacy.ldmatrix")?,
            mma_fill_legacy: get_operator("tirx.mma_fill_legacy")?,
            buffer_data: get_operator("tirx.buffer_data")?,
        },
        (
            find_coefficient_call,
            find_coefficient_store,
            find_coefficient_default,
        ),
    );
    structural_visit(body, &mut callbacks)?;
    let coefficient = callbacks.into_state().coefficient;
    if coefficient == 0 {
        Err(value_error(
            "could not infer a positive warp-index coefficient from a warp-memory store",
        ))
    } else {
        Ok(coefficient)
    }
}

fn find_coefficient_call(
    value: Call,
    visitor: &mut VisitContext<'_, WarpCoeffState>,
) -> Result<()> {
    let index = if value.op.same_as(&visitor.state().mma_fill) {
        if value.args.len() > 1 && is_warp_buffer(&value.args.get(1)?, visitor.state())? {
            update_coefficient(visitor.state_mut(), int_expr_value(&value.args.get(0)?)?)?;
        }
        None
    } else if value.op.same_as(&visitor.state().ptx_ldmatrix) {
        if value.args.len() > 4 && is_warp_buffer(&value.args.get(3)?, visitor.state())? {
            Some(value.args.get(4)?.try_cast::<PrimExpr>()?)
        } else {
            None
        }
    } else if value.op.same_as(&visitor.state().mma_fill_legacy) {
        if value.args.len() > 1 && is_warp_buffer(&value.args.get(1)?, visitor.state())? {
            update_coefficient(visitor.state_mut(), int_expr_value(&value.args.get(0)?)?)?;
        }
        None
    } else {
        None
    };
    if let Some(index) = index {
        update_coefficient_from_index(visitor.state_mut(), &index)?;
    }
    visitor.visit_children()?;
    Ok(())
}

fn find_coefficient_store(
    value: BufferStore,
    visitor: &mut VisitContext<'_, WarpCoeffState>,
) -> Result<()> {
    if value.buffer.same_as(&visitor.state().buffer) {
        if value.indices.len() != 1 {
            return Err(value_error(
                "warp memory requires a flat one-dimensional buffer access",
            ));
        }
        let mut index = value.indices.get(0)?;
        if value.value.dtype().lanes != 1 {
            let ramp = index
                .clone()
                .try_cast::<Ramp>()
                .map_err(|_| value_error("vector warp store requires a contiguous ramp index"))?;
            if int_value(&ramp.stride)? != 1 {
                return Err(value_error("vector warp store requires unit stride"));
            }
            index = ramp.base.clone();
        }
        update_coefficient_from_index(visitor.state_mut(), &index)?;
    }
    visitor.visit_children()?;
    Ok(())
}

fn find_coefficient_default(
    value: &VisitValue,
    visitor: &mut VisitContext<'_, WarpCoeffState>,
) -> Result<Option<VisitInterrupt>> {
    visit_stmt_expr_default(visitor, value)
}

fn is_warp_buffer(value: &Expr, state: &WarpCoeffState) -> Result<bool> {
    if value.same_as(&state.buffer) {
        return Ok(true);
    }
    if let Some(call) = value.as_node::<CallObj>() {
        if call.op.same_as(&state.buffer_data) && call.args.len() == 1 {
            return Ok(call.args.get(0)?.same_as(&state.buffer));
        }
    }
    Ok(false)
}

fn update_coefficient_from_index(state: &mut WarpCoeffState, index: &PrimExpr) -> Result<()> {
    let terms = detect_linear_equation(index, vec![state.warp_index.clone()])?;
    if terms.len() != 2 {
        return Err(value_error(
            "warp-memory store index is not linear in threadIdx.x",
        ));
    }
    let coefficient = state.analyzer.canonical_simplify(&terms.get(0)?)?;
    update_coefficient(state, int_value(&coefficient)?)
}

fn update_coefficient(state: &mut WarpCoeffState, coefficient: i64) -> Result<()> {
    if coefficient <= 0 {
        return Err(value_error(
            "warp-memory store index requires a positive threadIdx.x coefficient",
        ));
    }
    if state.coefficient != 0 && state.coefficient != coefficient {
        return Err(value_error(
            "warp memory has inconsistent threadIdx.x store coefficients",
        ));
    }
    state.coefficient = coefficient;
    Ok(())
}

struct WarpMemoryRewriter {
    warp_size: i32,
    analyzer: Analyzer,
    new_storage_scopes: HashMap<ObjectIdentity, String>,
}

impl WarpMemoryRewriter {
    fn new(warp_size: i32, analyzer: Analyzer) -> Self {
        Self {
            warp_size,
            analyzer,
            new_storage_scopes: HashMap::new(),
        }
    }
}

#[tvm_ffi::dispatch(mutate)]
impl WarpMemoryRewriter {
    fn mutate_sequence(&mut self, value: SeqStmt, mutator: &mut Mutator) -> Result<Stmt> {
        let mut statements = Vec::new();
        let mut changed = false;
        for (index, statement) in value.seq.iter().enumerate() {
            if let Ok(allocation) = statement.clone().try_cast::<AllocBuffer>() {
                if allocation.buffer.type_annotation().storage_scope.as_str() == "warp" {
                    self.new_storage_scopes.insert(
                        ObjectIdentity::of(allocation.buffer.as_var()),
                        String::from("local"),
                    );
                    let body = Stmt::sequence(value.seq.iter().skip(index + 1).collect())?;
                    let mut rewriter =
                        WarpAccessRewriter::new(self.warp_size, self.analyzer.clone())?;
                    statements.push(rewriter.rewrite(&allocation, body)?);
                    changed = true;
                    break;
                }
            }
            let mapped: Stmt = mutator.mutate(self, &statement)?.try_into()?;
            changed |= !mapped.same_as(&statement);
            statements.push(mapped);
        }
        if changed {
            Stmt::sequence_with_span(statements, value.span.as_ref())
        } else {
            Ok(value.into())
        }
    }

    fn mutate_default(&mut self, value: &MapValue, mutator: &mut Mutator) -> Result<Any> {
        mutate_stmt_expr_default(self, mutator, value)
    }
}

struct WarpAccessRewriter {
    warp_size: i32,
    analyzer: Analyzer,
    old_buffer: Option<BufferVar>,
    new_buffer: Option<BufferVar>,
    warp_index: Option<PrimVar>,
    width: i32,
    warp_coefficient: i64,
    warp_group: i64,
    special_call_indices: Vec<(Expr, Vec<usize>)>,
}

impl WarpAccessRewriter {
    fn new(warp_size: i32, analyzer: Analyzer) -> Result<Self> {
        Ok(Self {
            warp_size,
            analyzer,
            old_buffer: None,
            new_buffer: None,
            warp_index: None,
            width: 0,
            warp_coefficient: 0,
            warp_group: 0,
            special_call_indices: vec![
                (get_operator("tirx.mma_store")?, vec![3]),
                (get_operator("tirx.mma_fill")?, vec![1]),
                (get_operator("tirx.ptx_legacy.mma")?, vec![6, 8, 10]),
                (get_operator("tirx.ptx_legacy.ldmatrix")?, vec![3]),
                (get_operator("tirx.mma_store_legacy")?, vec![3]),
                (get_operator("tirx.mma_fill_legacy")?, vec![1]),
            ],
        })
    }

    fn rewrite(&mut self, allocation: &AllocBuffer, body: Stmt) -> Result<Stmt> {
        let old_type = allocation.buffer.type_annotation();
        let mut allocation_size = 1_i64;
        for dimension in old_type.shape.iter() {
            allocation_size = allocation_size
                .checked_mul(int_value(&dimension)?)
                .ok_or_else(|| value_error("warp allocation size overflows i64"))?;
        }
        allocation_size = allocation_size
            .checked_mul(i64::from(old_type.dtype.dtype.lanes))
            .ok_or_else(|| value_error("warp allocation size overflows i64"))?;
        if allocation_size <= 0 {
            return Err(value_error(
                "warp memory requires a positive constant allocation size",
            ));
        }

        let (warp_index, width) = find_warp_index(&body, self.warp_size)?;
        let coefficient = find_warp_coefficient(
            &body,
            &allocation.buffer,
            warp_index.clone(),
            &self.analyzer,
        )?;
        let factor = i64::from(width)
            .checked_mul(coefficient)
            .ok_or_else(|| value_error("warp-memory factor overflows i64"))?;
        let group = allocation_size
            .checked_add(factor - 1)
            .and_then(|value| value.checked_div(factor))
            .ok_or_else(|| value_error("invalid warp-memory allocation factor"))?;
        let aligned = group
            .checked_mul(factor)
            .ok_or_else(|| value_error("aligned warp allocation size overflows i64"))?;

        let new_type = BufferType::from_complete_fields(
            old_type.span.clone(),
            old_type.dtype.clone(),
            String::from("local"),
            Array::new(vec![
                IntImm::new("int32", aligned / i64::from(width))?.into()
            ]),
            Array::new(Vec::new()),
            IntImm::from_dtype(old_type.elem_offset.dtype(), 0)?.into(),
            old_type.data_alignment,
            old_type.offset_factor,
            old_type.layout.clone(),
            old_type.allocated_addr.clone(),
        );
        let new_buffer = allocation.buffer.with_type(new_type)?;
        self.old_buffer = Some(allocation.buffer.clone());
        self.new_buffer = Some(new_buffer.clone());
        self.warp_index = Some(warp_index);
        self.width = width;
        self.warp_coefficient = coefficient;
        self.warp_group = group;

        let body: Stmt = structural_mutate(body, &mut *self)?.try_into()?;
        Stmt::sequence(vec![allocation.copy_with(new_buffer).into(), body])
    }

    fn old_buffer(&self) -> &BufferVar {
        self.old_buffer
            .as_ref()
            .expect("warp access rewriter must be initialized")
    }

    fn new_buffer(&self) -> BufferVar {
        self.new_buffer
            .as_ref()
            .expect("warp access rewriter must be initialized")
            .clone()
    }

    fn rewrite_special_call(&mut self, value: Call, positions: &[usize]) -> Result<Expr> {
        let mut arguments = value.args.iter().collect::<Vec<_>>();
        for &position in positions {
            if position + 1 >= arguments.len() {
                continue;
            }
            if arguments[position].same_as(self.old_buffer()) {
                let index = arguments[position + 1].clone().try_cast::<PrimExpr>()?;
                let (local_index, _) = self.split_index_by_group(&index)?;
                arguments[position] = self.new_buffer().into();
                arguments[position + 1] = local_index.into();
            }
        }
        Ok(value
            .copy_with(value.ty.clone(), value.op.clone(), Array::new(arguments))
            .into())
    }

    fn split_index_by_group(&self, index: &PrimExpr) -> Result<(PrimExpr, PrimExpr)> {
        let index_type = index.type_annotation();
        if index_type.dtype.lanes != 1 {
            let ramp = index
                .clone()
                .try_cast::<Ramp>()
                .map_err(|_| value_error("vector warp access requires a ramp index"))?;
            if int_value(&ramp.stride)? != 1 {
                return Err(value_error("vector warp access requires unit stride"));
            }
            let (local, group) = self.split_index_by_group(&ramp.base)?;
            let local = Ramp::new(
                local,
                IntImm::from_dtype(ramp.stride.dtype(), 1)?,
                ramp.lanes.clone(),
            )?
            .into();
            return Ok((local, group));
        }
        let coefficient: PrimExpr =
            IntImm::from_dtype(index_type.dtype, self.warp_coefficient)?.into();
        let local_remainder = binary_op("tirx._OpIndexMod", index.clone(), coefficient.clone())?;
        if self.warp_group == 1 {
            return Ok((
                self.analyzer.canonical_simplify(&local_remainder)?,
                self.analyzer.canonical_simplify(&binary_op(
                    "tirx._OpIndexDiv",
                    index.clone(),
                    coefficient,
                )?)?,
            ));
        }

        let width_coefficient: PrimExpr = IntImm::from_dtype(
            index_type.dtype,
            self.warp_coefficient * i64::from(self.width),
        )?
        .into();
        let quotient = binary_op("tirx._OpDiv", index.clone(), width_coefficient.clone())?;
        let local = binary_op(
            "tirx._OpAdd",
            binary_op("tirx._OpMul", quotient, coefficient.clone())?,
            local_remainder,
        )?;
        let group = binary_op(
            "tirx._OpIndexDiv",
            binary_op("tirx._OpIndexMod", index.clone(), width_coefficient)?,
            coefficient,
        )?;
        Ok((
            self.analyzer.canonical_simplify(&local)?,
            self.analyzer.canonical_simplify(&group)?,
        ))
    }
}

#[tvm_ffi::dispatch(mutate)]
impl WarpAccessRewriter {
    fn mutate_call(&mut self, value: Call, mutator: &mut Mutator) -> Result<Expr> {
        if let Some(positions) = self
            .special_call_indices
            .iter()
            .find(|(operator, _)| value.op.same_as(operator))
            .map(|(_, positions)| positions.clone())
        {
            return self.rewrite_special_call(value, &positions);
        }
        mutate_expr_default(self, mutator, value.into())
    }

    fn mutate_variable(&mut self, value: Var) -> Result<Var> {
        if value.same_as(self.old_buffer()) {
            Err(value_error(
                "warp buffer address cannot be accessed directly",
            ))
        } else {
            Ok(value)
        }
    }

    fn mutate_store(&mut self, value: BufferStore, mutator: &mut Mutator) -> Result<BufferStore> {
        let stored: PrimExpr = mutator.mutate(self, &value.value)?.try_into()?;
        let indices: Array<PrimExpr> = mutator.mutate(self, &value.indices)?.try_into()?;
        if !value.buffer.same_as(self.old_buffer()) {
            if stored.same_as(&value.value) && array_same_as(&indices, &value.indices) {
                return Ok(value);
            }
            return Ok(value.copy_with(value.buffer.clone(), stored, indices));
        }
        if indices.len() != 1 {
            return Err(value_error("warp memory requires a flat buffer store"));
        }
        let (local_index, _) = self.split_index_by_group(&indices.get(0)?)?;
        Ok(value.copy_with(self.new_buffer(), stored, Array::new(vec![local_index])))
    }

    fn mutate_load(&mut self, value: TensorLoad, mutator: &mut Mutator) -> Result<Expr> {
        let indices: Array<PrimExpr> = mutator.mutate(self, &value.indices)?.try_into()?;
        if !value.source.same_as(self.old_buffer()) {
            if array_same_as(&indices, &value.indices) {
                return Ok(value.into());
            }
            return Ok(value.copy_with(value.source.clone(), indices).into());
        }
        if indices.len() != 1 {
            return Err(value_error("warp memory requires a flat buffer load"));
        }
        let (local_index, group) = self.split_index_by_group(&indices.get(0)?)?;
        if expression_uses_variable(
            &local_index,
            self.warp_index
                .as_ref()
                .expect("warp access rewriter must be initialized")
                .as_var(),
        )? {
            return Err(value_error(
                "lowered local warp index still depends on threadIdx.x",
            ));
        }
        let load: PrimExpr = TensorLoad::from_buffer_with_span(
            self.new_buffer().as_var().clone(),
            vec![local_index.into()],
            value.span.as_ref(),
        )?
        .into();
        let warp_index: PrimExpr = self
            .warp_index
            .as_ref()
            .expect("warp access rewriter must be initialized")
            .clone()
            .into();
        if self.analyzer.can_prove_equal(&group, &warp_index)? {
            return Ok(load.into());
        }
        let mask = Call::new(
            PrimType::new("uint32")?,
            get_operator("tirx.tvm_warp_activemask")?,
            Vec::new(),
        );
        Ok(Call::new(
            load.type_annotation(),
            get_operator("tirx.tvm_warp_shuffle")?,
            vec![
                mask.into(),
                load.into(),
                group.into(),
                IntImm::new("int32", i64::from(self.width))?.into(),
                IntImm::new("int32", i64::from(self.warp_size))?.into(),
            ],
        )
        .into())
    }

    fn mutate_default(&mut self, value: &MapValue, mutator: &mut Mutator) -> Result<Any> {
        mutate_stmt_expr_default(self, mutator, value)
    }
}

struct PointerScopeUpdater {
    variable_remaps: HashMap<ObjectIdentity, Var>,
}

fn update_pointer_storage_scope(
    body: Stmt,
    scopes: HashMap<ObjectIdentity, String>,
) -> Result<Stmt> {
    let mut updater = PointerScopeUpdater {
        variable_remaps: HashMap::new(),
    };
    collect_scope_variables(&body, &scopes, &mut updater.variable_remaps)?;
    structural_mutate(body, &mut updater)?.try_into()
}

fn collect_scope_variables(
    body: &Stmt,
    scopes: &HashMap<ObjectIdentity, String>,
    remaps: &mut HashMap<ObjectIdentity, Var>,
) -> Result<()> {
    structural_walk(
        body,
        |variable: Var| -> Result<WalkResult> {
            let identity = ObjectIdentity::of(&variable);
            let Some(scope) = scopes.get(&identity) else {
                return Ok(WalkResult::Advance);
            };
            let replacement = if let Ok(buffer) = BufferVar::try_from(&variable) {
                let old_type = buffer.type_annotation();
                buffer
                    .with_type(old_type.copy_with(
                        scope.clone(),
                        old_type.dtype.clone(),
                        old_type.shape.clone(),
                    ))?
                    .as_var()
                    .clone()
            } else {
                let pointer = variable.ty.clone().try_cast::<crate::ir::PointerType>()?;
                variable.copy_with(
                    variable.name.clone(),
                    crate::ir::PointerType::new(pointer.element_type().clone(), scope.as_str())?
                        .into(),
                )
            };
            remaps.insert(identity, replacement);
            Ok(WalkResult::Advance)
        },
        WalkOrder::PreOrder,
    )?;
    Ok(())
}

#[tvm_ffi::dispatch(mutate)]
impl PointerScopeUpdater {
    fn mutate_variable(&mut self, value: Var) -> Var {
        self.variable_remaps
            .get(&ObjectIdentity::of(&value))
            .cloned()
            .unwrap_or(value)
    }

    fn mutate_allocation(&mut self, value: AllocBuffer) -> Result<AllocBuffer> {
        let buffer = self.updated_buffer(&value.buffer)?;
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
        let buffer = self.updated_buffer(&value.buffer)?;
        if data.same_as(&value.data) && buffer.same_as(&value.buffer) {
            return Ok(value);
        }
        Ok(value.copy_with(buffer, data))
    }

    fn mutate_store(&mut self, value: BufferStore, mutator: &mut Mutator) -> Result<BufferStore> {
        let stored: PrimExpr = mutator.mutate(self, &value.value)?.try_into()?;
        let indices: Array<PrimExpr> = mutator.mutate(self, &value.indices)?.try_into()?;
        let buffer = self.updated_buffer(&value.buffer)?;
        if stored.same_as(&value.value)
            && array_same_as(&indices, &value.indices)
            && buffer.same_as(&value.buffer)
        {
            return Ok(value);
        }
        Ok(value.copy_with(buffer, stored, indices))
    }

    fn mutate_load(&mut self, value: TensorLoad, mutator: &mut Mutator) -> Result<TensorLoad> {
        let indices: Array<PrimExpr> = mutator.mutate(self, &value.indices)?.try_into()?;
        let buffer = BufferVar::try_from(&value.source)?;
        let buffer = self.updated_buffer(&buffer)?;
        if array_same_as(&indices, &value.indices) && buffer.same_as(&value.source) {
            return Ok(value);
        }
        TensorLoad::from_buffer_with_span(
            buffer.as_var().clone(),
            indices.iter().map(Into::into).collect(),
            value.span.as_ref(),
        )
    }

    fn mutate_default(&mut self, value: &MapValue, mutator: &mut Mutator) -> Result<Any> {
        mutate_stmt_expr_default(self, mutator, value)
    }
}

impl PointerScopeUpdater {
    fn updated_buffer(&self, buffer: &BufferVar) -> Result<BufferVar> {
        self.variable_remaps
            .get(&ObjectIdentity::of(buffer.as_var()))
            .map(BufferVar::try_from)
            .transpose()
            .map(|value| value.unwrap_or_else(|| buffer.clone()))
    }
}

fn expression_uses_variable(expression: &PrimExpr, variable: &Var) -> Result<bool> {
    Ok(structural_walk(
        expression,
        |value: Var| {
            if value.same_as(variable) {
                WalkResult::Interrupt
            } else {
                WalkResult::Advance
            }
        },
        WalkOrder::PreOrder,
    )?
    .is_some())
}

fn function_target(function: &PrimFunc) -> Result<Target> {
    function
        .attrs
        .dict
        .get(&String::from("target"))?
        .ok_or_else(|| value_error("LowerWarpMemory requires the target attribute"))?
        .try_into()
}

fn int_expr_value(value: &Expr) -> Result<i64> {
    optional_int_value(value).ok_or_else(|| value_error("expected a constant integer expression"))
}

fn int_value(value: &PrimExpr) -> Result<i64> {
    optional_int_value(value).ok_or_else(|| value_error("expected a constant integer expression"))
}
