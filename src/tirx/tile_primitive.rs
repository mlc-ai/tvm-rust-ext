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

//! Tile primitives and dispatch: hand-written semantics for the generated `tirx.ExecScope`,
//! `tirx.ScopeIdDef*`, `tirx.LambdaExpr`, `tirx.DispatchContext`, and `tirx.TilePrimitiveCall`
//! bindings in `tirx.rs`.

use super::*;
use super::{BufferVar, PrimVar};
use crate::ir::{Op, PrimExpr, Range, Span, Var};
use crate::target::Target;
use std::cell::UnsafeCell;
use tvm_ffi::object::ObjectRef as AnyObjectRef;
use tvm_ffi::{Any, Array, Error, Map, Result, String as FfiString, VALUE_ERROR};

/// Map slot that native methods replace even through a shared context handle.
///
/// The generated `DispatchContext` layout stores `callbacks` and
/// `shared_state` in this cell (see the `field` directives in
/// `src/tirx.rs`).  Rust never lends out a reference to the
/// stored map; it only hands out owned snapshots, so a native update cannot
/// invalidate an outstanding Rust borrow.
#[repr(transparent)]
pub struct NativeMutableMap(UnsafeCell<Map<FfiString, AnyObjectRef>>);

impl NativeMutableMap {
    /// Wrap the initial map value.
    pub fn new(map: Map<FfiString, AnyObjectRef>) -> Self {
        Self(UnsafeCell::new(map))
    }

    /// Snapshot the stored map. Later native updates leave this map unchanged.
    pub fn snapshot(&self) -> Map<FfiString, AnyObjectRef> {
        // SAFETY: The owning context cannot be shared across threads, and
        // Map::clone only increments the reference count, without calling user
        // code. The native writer uses Map's copy-on-write after this clone.
        unsafe { (&*self.0.get()).clone() }
    }
}

impl ScopeKind {
    pub fn from_name(name: &str) -> Result<Self> {
        match name {
            "cluster" => Ok(Self::CLUSTER),
            "cta" => Ok(Self::CTA),
            "warpgroup" => Ok(Self::WARPGROUP),
            "warp" => Ok(Self::WARP),
            "thread" => Ok(Self::THREAD),
            _ => Err(value_error(&format!(
                "unknown tile execution scope `{name}`"
            ))),
        }
    }

    /// Return the native name of a scope understood by this TVM build.
    pub fn name(self) -> Result<&'static str> {
        match self.as_raw() {
            2 => Ok("cluster"),
            3 => Ok("cta"),
            4 => Ok("warpgroup"),
            5 => Ok("warp"),
            6 => Ok("thread"),
            _ => Err(value_error(&format!(
                "unknown ScopeKind value {}",
                self.as_raw()
            ))),
        }
    }
}

impl ScopeBinding {
    pub(crate) fn name_pair(self) -> Result<(&'static str, &'static str)> {
        match self.as_raw() {
            0 => Ok(("kernel", "cluster")),
            1 => Ok(("kernel", "cta")),
            2 => Ok(("cluster", "cta")),
            3 => Ok(("cta", "warpgroup")),
            4 => Ok(("cta", "warp")),
            5 => Ok(("warpgroup", "warp")),
            6 => Ok(("warp", "thread")),
            7 => Ok(("cta", "thread")),
            8 => Ok(("warpgroup", "thread")),
            9 => Ok(("cluster", "cta_pair")),
            _ => Err(value_error(&format!(
                "unknown ScopeBinding value {}",
                self.as_raw()
            ))),
        }
    }
}

impl ExecScope {
    pub fn new(kind: ScopeKind) -> Result<Self> {
        kind.name()?;
        Ok(Self::from_complete_fields(kind))
    }

    pub fn from_name(name: &str) -> Result<Self> {
        ScopeKind::from_name(name).map(Self::from_complete_fields)
    }

    pub fn name(&self) -> Result<&'static str> {
        self.kind.name()
    }
}

impl ScopeIdDef {
    pub fn new(
        def_ids: Vec<PrimVar>,
        extents: Option<Vec<PrimExpr>>,
        scope: ScopeBinding,
        preferred_extents: Option<Vec<PrimExpr>>,
    ) -> Result<Self> {
        scope.name_pair()?;
        let def_ids = Array::new(def_ids);
        let extents = extents.map(Array::new);
        let preferred_extents = preferred_extents.map(Array::new);
        match &extents {
            Some(extents) if def_ids.len() != extents.len() => {
                return Err(value_error(&format!(
                    "number of ScopeIdDef variables ({}) does not match its extents ({})",
                    def_ids.len(),
                    extents.len()
                )));
            }
            None if def_ids.len() != 1 => {
                return Err(value_error(
                    "a deferred ScopeIdDef must define exactly one variable",
                ));
            }
            None if preferred_extents.is_some() => {
                return Err(value_error(
                    "a deferred ScopeIdDef cannot carry preferred extents",
                ));
            }
            _ => {}
        }
        Ok(Self::from_complete_fields(
            def_ids,
            extents,
            scope,
            preferred_extents,
        ))
    }

    pub fn is_deferred(&self) -> bool {
        self.extents.is_none()
    }

    pub fn fused_extent(&self) -> Result<PrimExpr> {
        let extents = self
            .extents
            .as_ref()
            .ok_or_else(|| value_error("a deferred ScopeIdDef has no fused extent"))?;
        let mut values = extents.iter();
        let mut result = values
            .next()
            .ok_or_else(|| value_error("cannot fuse an empty ScopeIdDef"))?;
        for extent in values {
            result = crate::ir::prim::Mul::new(result, extent)?.into();
        }
        Ok(result)
    }
}

impl ScopeIdDefStmt {
    pub fn new(def: ScopeIdDef, span: Option<&Span>) -> Self {
        Self::from_complete_fields(span.cloned(), def)
    }
}

impl LambdaExpr {
    pub fn new(vars: Vec<Var>, pred: PrimExpr) -> Self {
        Self::from_complete_fields(Array::new(vars), pred)
    }

    pub fn apply(&self, indices: Vec<PrimExpr>) -> Result<PrimExpr> {
        tvm_ffi::cached_global_func!("tirx.LambdaExprApply")
            .call_tuple((self, Array::new(indices)))?
            .try_into()
    }
}

impl DispatchContext {
    pub fn new(target: Target, exec_scope: ExecScope) -> Self {
        Self::with_metadata(
            target,
            exec_scope,
            Map::new(),
            Map::new(),
            false,
            Map::new(),
            Map::new(),
            Map::new(),
            Map::new(),
            FfiString::from(""),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn with_metadata(
        target: Target,
        exec_scope: ExecScope,
        launch_params: Map<FfiString, super::IterVar>,
        var_range_map: Map<Var, Range>,
        alloc_only: bool,
        callbacks: Map<FfiString, AnyObjectRef>,
        shared_state: Map<FfiString, AnyObjectRef>,
        inter: Map<FfiString, Array<PrimExpr>>,
        intra: Map<FfiString, Array<PrimExpr>>,
        scope_kind: FfiString,
    ) -> Self {
        Self::from_complete_fields(
            target,
            exec_scope,
            launch_params,
            var_range_map,
            alloc_only,
            NativeMutableMap::new(callbacks),
            NativeMutableMap::new(shared_state),
            inter,
            intra,
            scope_kind,
        )
    }

    /// Snapshot the callbacks. Later native updates leave this map unchanged.
    pub fn callbacks(&self) -> Map<FfiString, AnyObjectRef> {
        self.callbacks.snapshot()
    }

    /// Snapshot the shared state. Later native updates leave this map unchanged.
    pub fn shared_state(&self) -> Map<FfiString, AnyObjectRef> {
        self.shared_state.snapshot()
    }

    pub fn add_alloc_buffer(&self, buffer: BufferVar) -> Result<()> {
        tvm_ffi::cached_global_func!("tirx.DispatchContextAddAllocBuffer")
            .call_tuple((self, buffer))?;
        Ok(())
    }

    pub fn add_init_stmt(&self, statement: Stmt, host: bool) -> Result<()> {
        tvm_ffi::cached_global_func!("tirx.DispatchContextAddInitStmt")
            .call_tuple((self, statement, host))?;
        Ok(())
    }

    pub fn add_post_buffer_def_stmt(&self, buffer: BufferVar, statement: Stmt) -> Result<()> {
        tvm_ffi::cached_global_func!("tirx.DispatchContextAddPostBufferDefStmt")
            .call_tuple((self, buffer, statement))?;
        Ok(())
    }

    pub fn set_shared_state(&self, key: &str, value: &Any) -> Result<()> {
        tvm_ffi::cached_global_func!("tirx.DispatchContextSharedStateSet").call_tuple((
            self,
            FfiString::from(key),
            value,
        ))?;
        Ok(())
    }

    pub fn get_shared_state(&self, key: &str) -> Result<Option<Any>> {
        let value = tvm_ffi::cached_global_func!("tirx.DispatchContextSharedStateGet")
            .call_tuple((self, FfiString::from(key)))?;
        // `Option<Any>` would treat every `Any` as `Some`, so decode the ABI's
        // None tag before returning the heterogeneous value.
        if value.type_index() == tvm_ffi::TypeIndex::kTVMFFINone as i32 {
            Ok(None)
        } else {
            Ok(Some(value))
        }
    }
}

impl TilePrimitiveCall {
    fn from_fields(
        op: Op,
        args: Array<Any>,
        workspace: Map<FfiString, BufferVar>,
        config: Map<FfiString, Any>,
        dispatch: Option<FfiString>,
        scope: ExecScope,
        span: Option<Span>,
    ) -> Result<Self> {
        let category: Option<FfiString> = tvm_ffi::cached_global_func!("ir.OpGetAttr")
            .call_tuple((&op, FfiString::from("TIRxOpCategory")))?
            .try_into()?;
        if category.as_ref().map(FfiString::as_str) != Some("tile_primitive") {
            return Err(value_error(
                "only tile primitive ops can be used in tirx.TilePrimitiveCall",
            ));
        }
        Ok(Self::from_complete_fields(
            span,
            op,
            args,
            workspace,
            config,
            dispatch.into(),
            scope,
        ))
    }

    pub fn new(
        op: Op,
        args: Vec<Any>,
        workspace: Map<FfiString, BufferVar>,
        config: Map<FfiString, Any>,
        dispatch: Option<FfiString>,
        scope: ExecScope,
    ) -> Result<Self> {
        Self::with_span(op, args, workspace, config, dispatch, scope, None)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn with_span(
        op: Op,
        args: Vec<Any>,
        workspace: Map<FfiString, BufferVar>,
        config: Map<FfiString, Any>,
        dispatch: Option<FfiString>,
        scope: ExecScope,
        span: Option<&Span>,
    ) -> Result<Self> {
        Self::from_fields(
            op,
            Array::new(args),
            workspace,
            config,
            dispatch,
            scope,
            span.cloned(),
        )
    }

    pub fn copy_with(&self, args: Array<Any>, config: Map<FfiString, Any>) -> Result<Self> {
        Self::from_fields(
            self.op.clone(),
            args,
            self.workspace.clone(),
            config,
            Option::from(self.dispatch.clone()),
            self.scope.clone(),
            self.span.clone(),
        )
    }
}

fn value_error(message: &str) -> Error {
    Error::new(VALUE_ERROR, message, "")
}
