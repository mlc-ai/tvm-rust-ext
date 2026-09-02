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

//! Shorthand constructors that take only the required fields.
//!
//! The complete semantic constructors in [`crate::ir`] and [`crate::tirx`]
//! (`with_metadata`, `with_span`, ...) accept every field the C++ constructor
//! accepts.  Most call sites never change the trailing metadata: a loop's
//! `span`, `annotations`, `thread_binding`, and `step`, a module's
//! `source_map`, a function's `ret_type`.  The functions in this module drop
//! those parameters and fill them with the same defaults the C++ constructors
//! apply.
//!
//! The second half of the module serves passes rather than fresh
//! construction.  A mutator that rewrites a node's children has to allocate a
//! new node and copy every other field from the old one; `with_children`
//! (and the smaller `with_kind`, `with_buffer`, `with_value`, `with_statements`,
//! `with_body`, `with_attr`, `without_attr`, and `with_functions`) takes only
//! the replaced fields and copies the rest from `self`, exactly as the passes
//! did by hand with `from_complete_fields`.
//!
//! Everything here is pure delegation.  A shorthand never validates,
//! normalizes, or allocates on its own; it forwards to the full constructor,
//! so validation, defaults, and derived fields stay in exactly one place and
//! the full constructor remains reachable whenever a caller does need the
//! metadata.  This module is a handwritten convenience layer, not part of the
//! mechanical stubgen surface described in `BINDING_CONTRACT.md`.

use tvm_ffi::{Any, Array, Map, Result, String};

use crate::ir::{
    BaseFunc, Call, DictAttrs, Expr, GlobalVar, IRModule, PrimExpr, Range, SourceMap, TensorLoad,
    Type, Var,
};
use crate::tirx::{
    AllocBuffer, AssertStmt, AttrStmt, Bind, BufferStore, BufferType, BufferVar, DeclBuffer,
    Evaluate, For, ForKind, IfThenElse, IterVar, IterVarType, Layout, Let, PrimFunc, Reduce,
    Select, SeqStmt, Stmt, StringImm, While,
};

// ---------------------------------------------------------------------------
// Fresh construction from the required fields only.
// ---------------------------------------------------------------------------

impl For {
    /// Construct a loop of the given `kind` from its required fields only.
    ///
    /// Equivalent to [`For::with_metadata`] with no thread binding, empty
    /// annotations, no explicit step, and no span.  Use [`For::serial`] when
    /// the kind is also the default.
    pub fn new<V, M, E, B>(
        loop_var: V,
        minimum: M,
        extent: E,
        kind: ForKind,
        body: B,
    ) -> Result<Self>
    where
        V: Into<Var>,
        M: Into<Expr>,
        E: Into<Expr>,
        B: Into<Stmt>,
    {
        Self::with_metadata(
            loop_var.into(),
            minimum.into(),
            extent.into(),
            kind,
            body.into(),
            None,
            Map::new(),
            None,
            None,
        )
    }

    /// Construct a `kThreadBinding` loop bound to `thread_binding`.
    ///
    /// Equivalent to [`For::with_metadata`] with `ForKind::kThreadBinding`,
    /// `Some(thread_binding)`, empty annotations, no explicit step, and no
    /// span.
    pub fn thread_bound<V, M, E, B>(
        loop_var: V,
        minimum: M,
        extent: E,
        thread_binding: IterVar,
        body: B,
    ) -> Result<Self>
    where
        V: Into<Var>,
        M: Into<Expr>,
        E: Into<Expr>,
        B: Into<Stmt>,
    {
        Self::with_metadata(
            loop_var.into(),
            minimum.into(),
            extent.into(),
            ForKind::kThreadBinding,
            body.into(),
            Some(thread_binding),
            Map::new(),
            None,
            None,
        )
    }
}

impl IterVar {
    /// Construct a `kThreadIndex` iteration variable tagged with `thread_tag`.
    ///
    /// Equivalent to [`IterVar::with_metadata`] with
    /// `IterVarType::kThreadIndex` and no span.  The domain may be absent, as
    /// the native constructor allows for launch-bound thread axes.
    pub fn thread_index<V>(domain: Option<Range>, variable: V, thread_tag: &str) -> Result<Self>
    where
        V: Into<Var>,
    {
        Self::with_metadata(
            domain,
            variable.into(),
            IterVarType::kThreadIndex,
            thread_tag,
            None,
        )
    }
}

impl PrimFunc {
    /// Construct a PrimFunc with function attributes and a derived return type.
    ///
    /// Equivalent to [`PrimFunc::with_metadata`] with a missing `ret_type`
    /// (derived from `body` like [`PrimFunc::new`]) and no span.
    pub fn with_attrs<S, A>(params: Vec<Var>, body: S, attrs: A) -> Result<Self>
    where
        S: Into<Stmt>,
        A: Into<DictAttrs>,
    {
        Self::with_metadata(params, body, Type::missing(), attrs, None)
    }
}

impl IRModule {
    /// Construct a module that holds only `functions`.
    ///
    /// Equivalent to [`IRModule::with_metadata`] with an empty source map,
    /// empty attributes, and no global infos.  Accepts any iterator of
    /// `(GlobalVar, function)` pairs, including `&Map<GlobalVar, BaseFunc>`
    /// and arrays of `(GlobalVar, PrimFunc)`; duplicate global names are
    /// still rejected by the full constructor.
    pub fn from_functions<I, F>(functions: I) -> Result<Self>
    where
        I: IntoIterator<Item = (GlobalVar, F)>,
        F: Into<BaseFunc>,
    {
        Self::with_metadata(
            functions
                .into_iter()
                .map(|(global_var, function)| (global_var, function.into()))
                .collect(),
            SourceMap::new(),
            DictAttrs::empty(),
            Map::new(),
        )
    }
}

// ---------------------------------------------------------------------------
// Rebuilding an existing node with replaced children.
//
// Each `with_children` takes the fields a pass rewrites and copies every other
// field (span, derived type, kind, annotations, attributes, definitions the
// pass leaves alone) from `self`.  A definition-renaming pass such as SSA
// conversion still uses `from_complete_fields` when it replaces a field that
// is not listed here.
// ---------------------------------------------------------------------------

impl For {
    /// Copy this loop with new bounds, body, and step.
    ///
    /// Keeps `span`, `loop_var`, `kind`, `thread_binding`, and `annotations`.
    pub fn with_children(
        &self,
        min: PrimExpr,
        extent: PrimExpr,
        body: Stmt,
        step: Option<PrimExpr>,
    ) -> Self {
        Self::from_complete_fields(
            self.span.clone(),
            self.loop_var.clone(),
            min,
            extent,
            self.kind,
            body,
            self.thread_binding.clone(),
            self.annotations.clone(),
            step,
        )
    }

    /// Copy this loop with a different execution `kind`; every other field is kept.
    pub fn with_kind(&self, kind: ForKind) -> Self {
        Self::from_complete_fields(
            self.span.clone(),
            self.loop_var.clone(),
            self.min.clone(),
            self.extent.clone(),
            kind,
            self.body.clone(),
            self.thread_binding.clone(),
            self.annotations.clone(),
            self.step.clone(),
        )
    }
}

impl AttrStmt {
    /// Copy this attribute statement with a new `value` and `body`.
    ///
    /// Keeps `span`, `node`, and `attr_key`.
    pub fn with_children(&self, value: PrimExpr, body: Stmt) -> Self {
        Self::from_complete_fields(
            self.span.clone(),
            self.node.clone(),
            self.attr_key.clone(),
            value,
            body,
        )
    }
}

impl Bind {
    /// Copy this binding with a new value; keeps `span` and `var`.
    pub fn with_value(&self, value: Expr) -> Self {
        Self::from_complete_fields(self.span.clone(), self.var.clone(), value)
    }
}

impl Evaluate {
    /// Copy this evaluation statement with a new value; keeps `span`.
    pub fn with_value(&self, value: Expr) -> Self {
        Self::from_complete_fields(self.span.clone(), value)
    }
}

impl While {
    /// Copy this while loop with a new condition and body; keeps `span`.
    pub fn with_children(&self, condition: PrimExpr, body: Stmt) -> Self {
        Self::from_complete_fields(self.span.clone(), condition, body)
    }
}

impl SeqStmt {
    /// Copy this sequence with new statements; keeps `span`.
    pub fn with_statements(&self, statements: Array<Stmt>) -> Self {
        Self::from_complete_fields(self.span.clone(), statements)
    }
}

impl IfThenElse {
    /// Copy this conditional with a new condition and branches; keeps `span`.
    pub fn with_children(
        &self,
        condition: PrimExpr,
        then_case: Stmt,
        else_case: Option<Stmt>,
    ) -> Self {
        Self::from_complete_fields(self.span.clone(), condition, then_case, else_case)
    }
}

impl AssertStmt {
    /// Copy this assertion with a new condition and message; keeps `span`.
    pub fn with_children(
        &self,
        condition: PrimExpr,
        error_kind: StringImm,
        message_parts: Array<StringImm>,
    ) -> Self {
        Self::from_complete_fields(self.span.clone(), condition, error_kind, message_parts)
    }
}

impl BufferStore {
    /// Copy this store with a new target buffer, value, and indices; keeps `span`.
    pub fn with_children(
        &self,
        buffer: BufferVar,
        value: PrimExpr,
        indices: Array<PrimExpr>,
    ) -> Self {
        Self::from_complete_fields(self.span.clone(), buffer, value, indices)
    }
}

impl AllocBuffer {
    /// Copy this allocation for a remapped `buffer`; keeps `span` and `annotations`.
    pub fn with_buffer(&self, buffer: BufferVar) -> Self {
        Self::from_complete_fields(self.span.clone(), buffer, self.annotations.clone())
    }
}

impl DeclBuffer {
    /// Copy this declaration with a remapped `buffer` and new `data`; keeps `span`.
    pub fn with_children(&self, buffer: BufferVar, data: Expr) -> Self {
        Self::from_complete_fields(self.span.clone(), buffer, data)
    }
}

impl TensorLoad {
    /// Copy this load with a new `source` and `indices`.
    ///
    /// Keeps `span` and the load's result type, so the new access must read
    /// the same element type as the old one.
    pub fn with_children(&self, source: Expr, indices: Array<PrimExpr>) -> Self {
        Self::from_complete_fields(
            self.span.clone(),
            PrimExpr::from(self).type_annotation(),
            source,
            indices,
        )
    }
}

impl Call {
    /// Copy this call with a new operator and arguments.
    ///
    /// Keeps `span`, the result type, `attrs`, and `ty_args`.
    pub fn with_children(&self, op: Expr, args: Array<Expr>) -> Self {
        Self::from_complete_fields(
            self.span.clone(),
            self.ty.clone(),
            op,
            args,
            self.attrs.clone(),
            self.ty_args.clone(),
        )
    }
}

impl Let {
    /// Copy this let expression with a new bound `value` and `body`.
    ///
    /// Keeps `span` and `var`; the result type follows the new `body`, as in
    /// the C++ constructor.
    pub fn with_children(&self, value: PrimExpr, body: PrimExpr) -> Self {
        Self::from_complete_fields(
            self.span.clone(),
            body.type_annotation(),
            self.var.clone(),
            value,
            body,
        )
    }
}

impl Select {
    /// Copy this select with new operands.
    ///
    /// Keeps `span`; the result type follows the new `true_value`, as in the
    /// C++ constructor.
    pub fn with_children(
        &self,
        condition: PrimExpr,
        true_value: PrimExpr,
        false_value: PrimExpr,
    ) -> Self {
        Self::from_complete_fields(
            self.span.clone(),
            true_value.type_annotation(),
            condition,
            true_value,
            false_value,
        )
    }
}

impl Reduce {
    /// Copy this reduction with new sources, initial values, axes, and condition.
    ///
    /// Keeps `span`, the result type, `combiner`, and `value_index`.
    pub fn with_children(
        &self,
        source: Array<PrimExpr>,
        init: Array<PrimExpr>,
        axis: Array<IterVar>,
        condition: PrimExpr,
    ) -> Self {
        Self::from_complete_fields(
            self.span.clone(),
            PrimExpr::from(self).type_annotation(),
            self.combiner.clone(),
            source,
            init,
            axis,
            condition,
            self.value_index,
        )
    }
}

impl IterVar {
    /// Copy this iteration variable with a new `domain` and `variable`.
    ///
    /// Keeps `iter_type`, `thread_tag`, and `span`.  The native constructor
    /// still validates the pair, hence the `Result`.
    pub fn with_children(&self, domain: Option<Range>, variable: Var) -> Result<Self> {
        Self::with_metadata(
            domain,
            variable,
            self.iter_type()?,
            self.thread_tag()?.as_str(),
            self.span()?.as_ref(),
        )
    }
}

impl BufferType {
    /// Copy this buffer type with new shape, strides, offset, layout, and addresses.
    ///
    /// Keeps `span`, `dtype`, `storage_scope`, `data_alignment`, and
    /// `offset_factor`.  No defaults are re-applied: the inputs are already
    /// normalized fields of an existing buffer type.
    pub fn with_children(
        &self,
        shape: Array<PrimExpr>,
        strides: Array<PrimExpr>,
        elem_offset: PrimExpr,
        layout: Option<Layout>,
        allocated_addr: Array<PrimExpr>,
    ) -> Self {
        Self::from_complete_fields(
            self.span.clone(),
            self.dtype.clone(),
            self.storage_scope.clone(),
            shape,
            strides,
            elem_offset,
            self.data_alignment,
            self.offset_factor,
            layout,
            allocated_addr,
        )
    }
}

impl PrimFunc {
    /// Copy this function with a new `body`; every other field, including
    /// the derived function type, is kept.
    pub fn with_body(&self, body: Stmt) -> Self {
        Self::from_complete_fields(
            self.span.clone(),
            self.ty.clone(),
            self.attrs.clone(),
            self.params.clone(),
            self.ret_type.clone(),
            body,
        )
    }

    /// Copy this function with attribute `key` set to `value`, replacing an
    /// existing entry of the same name (TVM's `WithAttr`).
    pub fn with_attr(&self, key: &str, value: impl Into<Any>) -> Self {
        let key = String::from(key);
        let mut attributes = self
            .attrs
            .dict
            .iter()
            .filter(|(existing, _)| existing.as_str() != key.as_str())
            .collect::<Vec<_>>();
        attributes.push((key, value.into()));
        let attrs = DictAttrs::from_dictionary(Map::from_iter(attributes));
        Self::from_complete_fields(
            self.span.clone(),
            self.ty.clone(),
            attrs,
            self.params.clone(),
            self.ret_type.clone(),
            self.body.clone(),
        )
    }

    /// Copy this function without attribute `key`; every other field is kept.
    pub fn without_attr(&self, key: &str) -> Self {
        let attrs = DictAttrs::from_dictionary(Map::from_iter(
            self.attrs
                .dict
                .iter()
                .filter(|(existing, _)| existing.as_str() != key),
        ));
        Self::from_complete_fields(
            self.span.clone(),
            self.ty.clone(),
            attrs,
            self.params.clone(),
            self.ret_type.clone(),
            self.body.clone(),
        )
    }

    /// Copy this function with new parameters, body, and attributes.
    ///
    /// Keeps `ret_type` and `span`; the function type is derived again from
    /// the new parameters through [`PrimFunc::with_metadata`].
    pub fn with_children(&self, params: Vec<Var>, body: Stmt, attrs: DictAttrs) -> Result<Self> {
        Self::with_metadata(
            params,
            body,
            self.ret_type.clone(),
            attrs,
            self.span.as_ref(),
        )
    }
}

impl IRModule {
    /// Copy this module with a new function table.
    ///
    /// Keeps `source_map`, `attrs`, and `global_infos`; the name index is
    /// rebuilt and duplicate names rejected by [`IRModule::with_metadata`].
    pub fn with_functions<I, F>(&self, functions: I) -> Result<Self>
    where
        I: IntoIterator<Item = (GlobalVar, F)>,
        F: Into<BaseFunc>,
    {
        Self::with_metadata(
            functions
                .into_iter()
                .map(|(global_var, function)| (global_var, function.into()))
                .collect(),
            self.source_map.clone(),
            self.attrs.clone(),
            self.global_infos.clone(),
        )
    }
}
