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
    structural_mutate, structural_visit, Any, Error, Map, Mutator, ObjectCore, ObjectIdentity,
    ObjectRefCast, ObjectRefCore, Result, VisitCallbacks, VisitContext, VisitInterrupt, VisitValue,
    TYPE_ERROR,
};

use super::utils::{
    mutate_expr_default, visit_stmt_expr_default, with_prim_func_body, BufferRemaps,
};
use super::{create_prim_func_pass, Pass};
use crate::ir::prim::{
    Add, And, Cast, Div, FloorDiv, FloorMod, Let, Max, Min, Mod, Mul, Not, Or, Select, StringImm,
    Sub, EQ, GE, GT, LE, LT, NE,
};
use crate::ir::{Call, Expr, FloatImm, IntImm, PrimExpr, TensorLoad, Var};
use crate::te::Reduce;
use crate::tirx::{
    AllocBuffer, AttrStmt, Bind, BufferStore, BufferVar, DeclBuffer, For, IfThenElse, PrimFunc,
    SeqStmt, Stmt, TileLayout, While,
};

/// Eliminate repeated pure arithmetic expressions using the same two-phase
/// plan and rewrite algorithm as TVM's `tirx::CommonSubexprElim`.
pub fn common_subexpr_elim_prim_func(function: PrimFunc) -> Result<PrimFunc> {
    let plan = CsePlanner::plan(function.body())?;
    if plan.insert_before.is_empty() {
        return Ok(function);
    }

    let mut rewriter = CseRewriter::new(plan);
    let body = structural_mutate(function.body().clone(), &mut rewriter)?.try_into()?;
    Ok(with_prim_func_body(function, body))
}

/// Build TVM's `tirx.CommonSubexprElim` PrimFunc pass in Rust.
pub fn common_subexpr_elim() -> Result<Pass> {
    create_prim_func_pass(
        "tirx.CommonSubexprElim",
        0,
        Vec::new(),
        false,
        common_subexpr_elim_prim_func,
    )
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ExprClass {
    Leaf,
    Recordable,
    Let,
    Call,
    TensorLoad,
    Other,
}

fn type_index_of<T: ObjectRefCore>() -> i32 {
    <T::ContainerType as ObjectCore>::type_index()
}

fn expr_class(type_index: i32) -> ExprClass {
    macro_rules! is_one_of {
        ($($node:ty),+ $(,)?) => {
            $(type_index == type_index_of::<$node>())||+
        };
    }

    if is_one_of!(Var, IntImm, FloatImm, StringImm) {
        ExprClass::Leaf
    } else if is_one_of!(
        Add, Sub, Mul, Div, Mod, FloorDiv, FloorMod, Min, Max, EQ, NE, LT, LE, GT, GE, And, Or,
        Not, Cast, Select,
    ) {
        ExprClass::Recordable
    } else if is_one_of!(Let) {
        ExprClass::Let
    } else if is_one_of!(Call) {
        ExprClass::Call
    } else if is_one_of!(TensorLoad) {
        ExprClass::TensorLoad
    } else {
        ExprClass::Other
    }
}

fn structural_hash(expression: &PrimExpr) -> Result<i64> {
    tvm_ffi::cached_global_func!("ffi.StructuralHash")
        .call_tuple((expression, false, false))?
        .try_into()
}

fn structurally_equal(lhs: &PrimExpr, rhs: &PrimExpr) -> Result<bool> {
    tvm_ffi::cached_global_func!("ffi.StructuralEqual")
        .call_tuple((lhs, rhs, false, false))?
        .try_into()
}

#[derive(Clone)]
struct ScopeEntry {
    parent: Option<usize>,
    depth: usize,
    creator: Option<Stmt>,
}

#[derive(Clone)]
struct DagChild {
    entry: usize,
    multiplicity: usize,
}

struct ExprEntry {
    original: PrimExpr,
    repr: PrimExpr,
    count: usize,
    depth: usize,
    lca_scope: usize,
    first_use_scope: usize,
    first_use_stmt: Stmt,
    children: Vec<DagChild>,
    consumed: usize,
}

struct ExprFrame {
    expression: PrimExpr,
    class: ExprClass,
    direct_children: Vec<PrimExpr>,
    contains_forbidden: bool,
}

#[derive(Default)]
struct ExprTable {
    entries: Vec<ExprEntry>,
    buckets: HashMap<i64, Vec<usize>>,
}

struct StructuralExprMap<V> {
    entries: Vec<(PrimExpr, V)>,
    buckets: HashMap<i64, Vec<usize>>,
}

impl<V> Default for StructuralExprMap<V> {
    fn default() -> Self {
        Self {
            entries: Vec::new(),
            buckets: HashMap::new(),
        }
    }
}

impl<V> StructuralExprMap<V> {
    fn get(&self, expression: &PrimExpr) -> Result<Option<&V>> {
        let hash = structural_hash(expression)?;
        let Some(bucket) = self.buckets.get(&hash) else {
            return Ok(None);
        };
        for &index in bucket {
            let (candidate, value) = &self.entries[index];
            if structurally_equal(candidate, expression)? {
                return Ok(Some(value));
            }
        }
        Ok(None)
    }

    fn insert(&mut self, expression: PrimExpr, value: V) -> Result<()> {
        let hash = structural_hash(&expression)?;
        let index = self.entries.len();
        self.entries.push((expression, value));
        self.buckets.entry(hash).or_default().push(index);
        Ok(())
    }
}

impl ExprTable {
    fn find(&self, expression: &PrimExpr) -> Result<Option<usize>> {
        let hash = structural_hash(expression)?;
        let Some(bucket) = self.buckets.get(&hash) else {
            return Ok(None);
        };
        for &index in bucket {
            if structurally_equal(&self.entries[index].original, expression)? {
                return Ok(Some(index));
            }
        }
        Ok(None)
    }

    fn insert(&mut self, hash: i64, entry: ExprEntry) -> usize {
        let index = self.entries.len();
        self.entries.push(entry);
        self.buckets.entry(hash).or_default().push(index);
        index
    }
}

struct CsePlan {
    insert_before: HashMap<ObjectIdentity, Vec<Stmt>>,
    expression_remap: StructuralExprMap<Expr>,
}

struct CsePlanner {
    scopes: Vec<ScopeEntry>,
    current_scope: usize,
    current_stmt: Option<Stmt>,
    let_depth: usize,
    expression_frames: Vec<ExprFrame>,
    table: ExprTable,
}

impl CsePlanner {
    fn plan(body: &Stmt) -> Result<CsePlan> {
        let planner = Self {
            scopes: vec![ScopeEntry {
                parent: None,
                depth: 0,
                creator: None,
            }],
            current_scope: 0,
            current_stmt: None,
            let_depth: 0,
            expression_frames: Vec::new(),
            table: ExprTable::default(),
        };
        let mut visitor = VisitCallbacks::new(planner, visit_cse_value);
        structural_visit(body, &mut visitor)?;
        visitor.into_state().compute_plan()
    }

    fn allocate_scope(&mut self, creator: Stmt) -> usize {
        let scope = self.scopes.len();
        self.scopes.push(ScopeEntry {
            parent: Some(self.current_scope),
            depth: self.scopes[self.current_scope].depth + 1,
            creator: Some(creator),
        });
        scope
    }

    fn lca(&self, mut lhs: usize, mut rhs: usize) -> usize {
        while self.scopes[lhs].depth > self.scopes[rhs].depth {
            lhs = self.scopes[lhs]
                .parent
                .expect("a non-root scope has a parent");
        }
        while self.scopes[rhs].depth > self.scopes[lhs].depth {
            rhs = self.scopes[rhs]
                .parent
                .expect("a non-root scope has a parent");
        }
        while lhs != rhs {
            lhs = self.scopes[lhs]
                .parent
                .expect("distinct scopes have a common root");
            rhs = self.scopes[rhs]
                .parent
                .expect("distinct scopes have a common root");
        }
        lhs
    }

    fn visit_statement(
        value: &VisitValue,
        statement: Stmt,
        visitor: &mut VisitContext<'_, Self>,
    ) -> Result<Option<VisitInterrupt>> {
        visitor.state_mut().current_stmt = Some(statement.clone());

        if let Some(node) = value.cast::<For>() {
            visit_cse_child(visitor, &node.min)?;
            visit_cse_child(visitor, &node.extent)?;
            visit_cse_scope(visitor, statement, &node.body)?;
            return Ok(None);
        }

        if let Some(node) = value.cast::<IfThenElse>() {
            visit_cse_child(visitor, &node.condition)?;
            visit_cse_scope(visitor, statement.clone(), &node.then_case)?;
            if let Some(else_case) = &node.else_case {
                visit_cse_scope(visitor, statement, else_case)?;
            }
            return Ok(None);
        }

        if let Some(node) = value.cast::<AttrStmt>() {
            visit_cse_child(visitor, &node.value)?;
            visit_cse_scope(visitor, statement, &node.body)?;
            return Ok(None);
        }

        if let Some(node) = value.cast::<AllocBuffer>() {
            visit_cse_buffer_definition(visitor, &node.buffer)?;
            return Ok(None);
        }

        if let Some(node) = value.cast::<DeclBuffer>() {
            visit_cse_buffer_definition(visitor, &node.buffer)?;
            return Ok(None);
        }

        if let Some(node) = value.cast::<BufferStore>() {
            visit_cse_child(visitor, &node.value)?;
            for index in node.indices.iter() {
                visit_cse_child(visitor, &index)?;
            }
            return Ok(None);
        }

        if let Some(node) = value.cast::<While>() {
            visit_cse_child(visitor, &node.condition)?;
            visit_cse_scope(visitor, statement, &node.body)?;
            return Ok(None);
        }

        visit_stmt_expr_default(visitor, value)
    }

    fn visit_expression(
        value: &VisitValue,
        expression: PrimExpr,
        visitor: &mut VisitContext<'_, Self>,
    ) -> Result<Option<VisitInterrupt>> {
        if let Some(parent) = visitor.state_mut().expression_frames.last_mut() {
            parent.direct_children.push(expression.clone());
        }

        let class = expr_class(value.type_index());
        visitor.state_mut().expression_frames.push(ExprFrame {
            expression,
            class,
            direct_children: Vec::new(),
            contains_forbidden: false,
        });

        if matches!(class, ExprClass::Call | ExprClass::TensorLoad) {
            for frame in &mut visitor.state_mut().expression_frames {
                frame.contains_forbidden = true;
            }
        }

        let visit_result = (|| -> Result<()> {
            match class {
                ExprClass::Leaf => {}
                ExprClass::Let => {
                    let node = value
                        .cast::<Let>()
                        .expect("ir.prim.Let has already been classified above");
                    visit_cse_child(visitor, &node.value)?;
                    visitor.state_mut().let_depth += 1;
                    let body_result = visit_cse_child(visitor, &node.body);
                    visitor.state_mut().let_depth -= 1;
                    body_result?;
                }
                ExprClass::Call => {
                    let node = value
                        .cast::<Call>()
                        .expect("ir.Call has already been classified above");
                    for argument in node.args.iter() {
                        visit_cse_child(visitor, &argument)?;
                    }
                }
                ExprClass::TensorLoad => {
                    let node = value
                        .cast::<TensorLoad>()
                        .expect("ir.TensorLoad has already been classified above");
                    for index in node.indices.iter() {
                        visit_cse_child(visitor, &index)?;
                    }
                }
                ExprClass::Other => {
                    if let Some(node) = value.cast::<Reduce>() {
                        for axis in node.axis.iter() {
                            if let Some(domain) = axis.dom()? {
                                visit_cse_child(visitor, &domain.min)?;
                                visit_cse_child(visitor, &domain.extent)?;
                            }
                        }
                        for source in node.source.iter() {
                            visit_cse_child(visitor, &source)?;
                        }
                        for init in node.init.iter() {
                            visit_cse_child(visitor, &init)?;
                        }
                        visit_cse_child(visitor, &node.condition)?;
                    } else {
                        reject_cse_interrupt(visit_stmt_expr_default(visitor, value)?)?;
                    }
                }
                ExprClass::Recordable => {
                    reject_cse_interrupt(visit_stmt_expr_default(visitor, value)?)?;
                }
            }
            Ok(())
        })();

        let frame = visitor
            .state_mut()
            .expression_frames
            .pop()
            .expect("the expression frame was pushed above");
        visit_result?;
        if frame.class == ExprClass::Recordable
            && !frame.contains_forbidden
            && visitor.state().let_depth == 0
            && !is_bool(&frame.expression)
        {
            visitor
                .state_mut()
                .record_expression(frame.expression, frame.direct_children)?;
        }
        Ok(None)
    }

    fn record_expression(
        &mut self,
        expression: PrimExpr,
        direct_children: Vec<PrimExpr>,
    ) -> Result<()> {
        if let Some(index) = self.table.find(&expression)? {
            let previous_scope = self.table.entries[index].lca_scope;
            let widened_scope = self.lca(previous_scope, self.current_scope);
            let entry = &mut self.table.entries[index];
            entry.lca_scope = widened_scope;
            entry.count += 1;
            return Ok(());
        }

        let current_stmt = self.current_stmt.clone().ok_or_else(|| {
            Error::new(
                TYPE_ERROR,
                "CSE found an expression outside a containing statement",
                "",
            )
        })?;
        let mut children = Vec::<DagChild>::new();
        let mut maximum_child_depth = 0;
        for child in direct_children {
            let Some(child_index) = self.table.find(&child)? else {
                continue;
            };
            maximum_child_depth = maximum_child_depth.max(self.table.entries[child_index].depth);
            if let Some(existing) = children.iter_mut().find(|entry| entry.entry == child_index) {
                existing.multiplicity += 1;
            } else {
                children.push(DagChild {
                    entry: child_index,
                    multiplicity: 1,
                });
            }
        }

        let hash = structural_hash(&expression)?;
        self.table.insert(
            hash,
            ExprEntry {
                original: expression.clone(),
                repr: expression,
                count: 1,
                depth: maximum_child_depth + 1,
                lca_scope: self.current_scope,
                first_use_scope: self.current_scope,
                first_use_stmt: current_stmt,
                children,
                consumed: 0,
            },
        );
        Ok(())
    }

    fn insertion_statement(&self, entry: &ExprEntry) -> Stmt {
        if entry.first_use_scope == entry.lca_scope {
            return entry.first_use_stmt.clone();
        }

        let mut scope = entry.first_use_scope;
        while self.scopes[scope].parent != Some(entry.lca_scope) {
            scope = self.scopes[scope]
                .parent
                .expect("the LCA is an ancestor of the first-use scope");
        }
        self.scopes[scope]
            .creator
            .clone()
            .expect("every non-root scope has a creator statement")
    }

    fn compute_plan(mut self) -> Result<CsePlan> {
        let mut ordered = (0..self.table.entries.len()).collect::<Vec<_>>();
        ordered.sort_by_key(|&index| self.table.entries[index].depth);

        for &index in &ordered {
            if self.table.entries[index].count < 2 {
                continue;
            }
            let repetitions = self.table.entries[index].count - 1;
            let children = self.table.entries[index].children.clone();
            for child in children {
                self.table.entries[child.entry].consumed += repetitions * child.multiplicity;
            }
        }

        let mut insert_before = HashMap::<ObjectIdentity, Vec<Stmt>>::new();
        let mut expression_remap = StructuralExprMap::default();
        let mut counter = 0;

        for (position, &index) in ordered.iter().enumerate() {
            let entry = &self.table.entries[index];
            if entry.count.saturating_sub(entry.consumed) < 2 {
                continue;
            }

            let insertion = self.insertion_statement(entry);
            let original = entry.original.clone();
            let representation = entry.repr.clone();
            let depth = entry.depth;
            counter += 1;
            let variable = Var::with_type(&format!("cse_v{counter}"), representation.ty.clone());
            let binding: Stmt =
                Bind::new(variable.clone(), Expr::from(representation.clone()))?.into();
            insert_before
                .entry(ObjectIdentity::of(&insertion))
                .or_default()
                .push(binding);

            expression_remap.insert(original, Expr::from(variable.clone()))?;

            let replacement = PrimExpr::try_from(Expr::from(variable))?;
            for &other_index in ordered.iter().skip(position + 1) {
                if self.table.entries[other_index].depth <= depth {
                    continue;
                }
                let current = self.table.entries[other_index].repr.clone();
                self.table.entries[other_index].repr =
                    replace_structural_subexpression(current, &representation, &replacement)?;
            }
        }

        Ok(CsePlan {
            insert_before,
            expression_remap,
        })
    }
}

// CSE runs code both before and after each expression's children and opens
// scopes around branches, so its callback controls recursion through VisitContext.
fn visit_cse_value(
    value: &VisitValue,
    visitor: &mut VisitContext<'_, CsePlanner>,
) -> Result<Option<VisitInterrupt>> {
    if let Some(statement) = value.cast::<Stmt>() {
        return CsePlanner::visit_statement(value, statement, visitor);
    }
    if let Some(expression) = value.cast::<PrimExpr>() {
        return CsePlanner::visit_expression(value, expression, visitor);
    }
    visitor.visit_children()
}

fn visit_cse_child<T>(visitor: &mut VisitContext<'_, CsePlanner>, child: &T) -> Result<()>
where
    for<'a> tvm_ffi::AnyView<'a>: From<&'a T>,
{
    reject_cse_interrupt(visitor.visit(child)?)
}

fn reject_cse_interrupt(interrupt: Option<VisitInterrupt>) -> Result<()> {
    if let Some(interrupt) = interrupt {
        return Err(Error::new(
            TYPE_ERROR,
            &format!(
                "unexpected interrupt while planning CSE: type_index={}",
                interrupt.value.type_index()
            ),
            "",
        ));
    }
    Ok(())
}

fn visit_cse_scope(
    visitor: &mut VisitContext<'_, CsePlanner>,
    creator: Stmt,
    child: &Stmt,
) -> Result<()> {
    let saved_scope = visitor.state().current_scope;
    let child_scope = visitor.state_mut().allocate_scope(creator);
    visitor.state_mut().current_scope = child_scope;
    let result = visit_cse_child(visitor, child);
    visitor.state_mut().current_scope = saved_scope;
    result
}

fn visit_cse_buffer_definition(
    visitor: &mut VisitContext<'_, CsePlanner>,
    buffer: &BufferVar,
) -> Result<()> {
    let buffer_type = buffer.type_annotation();
    for expression in buffer_type.shape.iter() {
        visit_cse_child(visitor, &expression)?;
    }
    for expression in buffer_type.strides.iter() {
        visit_cse_child(visitor, &expression)?;
    }
    visit_cse_child(visitor, &buffer_type.elem_offset)?;
    for expression in buffer_type.allocated_addr.iter() {
        visit_cse_child(visitor, &expression)?;
    }
    if let Some(layout) = &buffer_type.layout {
        if let Ok(layout) = layout.clone().try_cast::<TileLayout>() {
            for iter in layout.shard()?.iter().chain(layout.replica()?.iter()) {
                visit_cse_child(visitor, &iter.extent)?;
                visit_cse_child(visitor, &iter.stride)?;
            }
        }
    }
    Ok(())
}

fn is_bool(expression: &PrimExpr) -> bool {
    expression.dtype().code == tvm_ffi::DLDataTypeCode::kDLBool as u8
}

struct StructuralExprReplacer {
    target: PrimExpr,
    replacement: PrimExpr,
}

#[tvm_ffi::dispatch(mutate)]
impl StructuralExprReplacer {
    fn mutate_expression(&mut self, value: Expr, mutator: &mut Mutator) -> Result<Expr> {
        if let Ok(expression) = PrimExpr::try_from(value.clone()) {
            if structurally_equal(&expression, &self.target)? {
                return Ok(self.replacement.clone().into());
            }
        }
        mutate_expr_default(self, mutator, value)
    }
}

fn replace_structural_subexpression(
    expression: PrimExpr,
    target: &PrimExpr,
    replacement: &PrimExpr,
) -> Result<PrimExpr> {
    let mut replacer = StructuralExprReplacer {
        target: target.clone(),
        replacement: replacement.clone(),
    };
    structural_mutate(expression, &mut replacer)?.try_into()
}

struct CseRewriter {
    insert_before: HashMap<ObjectIdentity, Vec<Stmt>>,
    expression_remap: StructuralExprMap<Expr>,
    materialized: HashSet<ObjectIdentity>,
    buffer_remaps: BufferRemaps,
}

impl CseRewriter {
    fn new(plan: CsePlan) -> Self {
        Self {
            insert_before: plan.insert_before,
            expression_remap: plan.expression_remap,
            materialized: HashSet::new(),
            buffer_remaps: BufferRemaps::default(),
        }
    }

    fn replacement(&self, expression: &PrimExpr) -> Result<Option<Expr>> {
        Ok(self.expression_remap.get(expression)?.cloned())
    }

    fn materialize_insertions(&mut self, original: &Stmt, visited: Stmt) -> Result<Stmt> {
        let identity = ObjectIdentity::of(original);
        let Some(planned) = self.insert_before.get(&identity).cloned() else {
            return Ok(visited);
        };

        let mut statements = Vec::with_capacity(planned.len() + 1);
        if self.materialized.insert(identity) {
            statements.extend(planned);
        } else {
            let mut remap_entries = Vec::<(Var, Expr)>::new();
            for statement in planned {
                let binding = statement.try_cast::<Bind>()?;
                let remap: Map<Var, Expr> = remap_entries.iter().cloned().collect();
                let value: Expr = substitute(&binding.value, &remap)?;
                let fresh = Var::with_type(binding.var.name.as_str(), binding.var.ty.clone());
                remap_entries.push((binding.var.clone(), Expr::from(fresh.clone())));
                statements.push(Bind::new(fresh, value)?.into());
            }
            let remap = Map::from_iter(remap_entries);
            let visited: Stmt = substitute(&visited, &remap)?;
            statements.push(visited);
            return Stmt::sequence(statements);
        }
        statements.push(visited);
        Stmt::sequence(statements)
    }
}

#[tvm_ffi::dispatch(mutate)]
impl CseRewriter {
    fn mutate_expression(&mut self, value: Expr, mutator: &mut Mutator) -> Result<Expr> {
        if let Ok(expression) = PrimExpr::try_from(value.clone()) {
            if let Some(replacement) = self.replacement(&expression)? {
                return Ok(replacement);
            }
        }
        BufferRemaps::mutate_expr(self, mutator, value, |state| &mut state.buffer_remaps)
    }

    fn mutate_statement(&mut self, value: Stmt, mutator: &mut Mutator) -> Result<Stmt> {
        let mut visited = BufferRemaps::mutate_stmt(self, mutator, value.clone(), |state| {
            &mut state.buffer_remaps
        })?;
        if let Ok(sequence) = visited.clone().try_cast::<SeqStmt>() {
            visited = sequence.flatten()?;
        }
        self.materialize_insertions(&value, visited)
    }
}

fn substitute<T, R>(node: &T, replacements: &Map<Var, Expr>) -> Result<R>
where
    T: Clone + Into<Any>,
    R: TryFrom<Any, Error = Error>,
{
    let node: Any = node.clone().into();
    tvm_ffi::cached_global_func!("tirx.Substitute")
        .call_tuple((node, replacements))?
        .try_into()
}
