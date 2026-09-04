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

use std::cell::UnsafeCell;

use tvm_ffi::derive::{Object, ObjectRef};
use tvm_ffi::object::ObjectRef as AnyObjectRef;
use tvm_ffi::{
    Any, Array, Error, Map, ObjectArc, Optional, Result, String as FfiString, VALUE_ERROR,
};

use super::{BufferVar, PrimVar, Stmt, StmtObj};
use crate::ir::{Op, PrimExpr, Range, Span, Var};
use crate::target::Target;

#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ScopeKind(i32);

impl ScopeKind {
    pub const CLUSTER: Self = Self(2);
    pub const CTA: Self = Self(3);
    pub const WARPGROUP: Self = Self(4);
    pub const WARP: Self = Self(5);
    pub const THREAD: Self = Self(6);

    /// Preserve a native value not yet known by this Rust binding.
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }

    pub const fn as_raw(self) -> i32 {
        self.0
    }

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
        match self.0 {
            2 => Ok("cluster"),
            3 => Ok("cta"),
            4 => Ok("warpgroup"),
            5 => Ok("warp"),
            6 => Ok("thread"),
            _ => Err(value_error(&format!("unknown ScopeKind value {}", self.0))),
        }
    }
}

impl TryFrom<i64> for ScopeKind {
    type Error = Error;

    fn try_from(value: i64) -> Result<Self> {
        i32::try_from(value)
            .map(Self)
            .map_err(|_| value_error("ScopeKind does not fit its native i32 representation"))
    }
}

#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ScopeBinding(i32);

impl ScopeBinding {
    pub const KERNEL_CLUSTER: Self = Self(0);
    pub const KERNEL_CTA: Self = Self(1);
    pub const CLUSTER_CTA: Self = Self(2);
    pub const CTA_WARPGROUP: Self = Self(3);
    pub const CTA_WARP: Self = Self(4);
    pub const WARPGROUP_WARP: Self = Self(5);
    pub const WARP_THREAD: Self = Self(6);
    pub const CTA_THREAD: Self = Self(7);
    pub const WARPGROUP_THREAD: Self = Self(8);
    pub const CLUSTER_CTA_PAIR: Self = Self(9);

    /// Preserve a native value not yet known by this Rust binding.
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }

    pub const fn as_raw(self) -> i32 {
        self.0
    }

    pub(crate) fn name_pair(self) -> Result<(&'static str, &'static str)> {
        match self.0 {
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
                self.0
            ))),
        }
    }
}

impl TryFrom<i64> for ScopeBinding {
    type Error = Error;

    fn try_from(value: i64) -> Result<Self> {
        i32::try_from(value)
            .map(Self)
            .map_err(|_| value_error("ScopeBinding does not fit its native i32 representation"))
    }
}

/// ABI-complete Rust representation of a native execution scope.
#[repr(C)]
#[derive(Object)]
#[type_key = "tirx.ExecScope"]
pub struct ExecScopeObj {
    base: tvm_ffi::Object,
    pub kind: ScopeKind,
}

#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct ExecScope {
    data: ObjectArc<ExecScopeObj>,
}

impl std::ops::Deref for ExecScope {
    type Target = ExecScopeObj;

    fn deref(&self) -> &Self::Target {
        &self.data
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

    /// Construct an execution scope from its complete physical state.
    pub fn from_complete_fields(kind: ScopeKind) -> Self {
        Self {
            data: ObjectArc::new(ExecScopeObj {
                base: tvm_ffi::Object::new(),
                kind,
            }),
        }
    }

    pub fn name(&self) -> Result<&'static str> {
        self.kind.name()
    }
}

/// ABI-complete Rust representation of a scope-id definition.
#[repr(C)]
#[derive(Object)]
#[type_key = "tirx.ScopeIdDef"]
#[type_final]
pub struct ScopeIdDefObj {
    base: tvm_ffi::Object,
    pub def_ids: Array<PrimVar>,
    pub extents: Option<Array<PrimExpr>>,
    pub scope: ScopeBinding,
    pub preferred_extents: Option<Array<PrimExpr>>,
}

#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct ScopeIdDef {
    data: ObjectArc<ScopeIdDefObj>,
}

impl std::ops::Deref for ScopeIdDef {
    type Target = ScopeIdDefObj;

    fn deref(&self) -> &Self::Target {
        &self.data
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

    /// Construct a scope-id definition from its complete physical state after validation.
    pub fn from_complete_fields(
        def_ids: Array<PrimVar>,
        extents: Option<Array<PrimExpr>>,
        scope: ScopeBinding,
        preferred_extents: Option<Array<PrimExpr>>,
    ) -> Self {
        Self {
            data: ObjectArc::new(ScopeIdDefObj {
                base: tvm_ffi::Object::new(),
                def_ids,
                extents,
                scope,
                preferred_extents,
            }),
        }
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

#[repr(C)]
#[derive(Object)]
#[type_key = "tirx.ScopeIdDefStmt"]
#[type_final]
pub struct ScopeIdDefStmtObj {
    base: StmtObj,
    pub def: ScopeIdDef,
}

#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct ScopeIdDefStmt {
    data: ObjectArc<ScopeIdDefStmtObj>,
}

impl std::ops::Deref for ScopeIdDefStmt {
    type Target = ScopeIdDefStmtObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl std::ops::Deref for ScopeIdDefStmtObj {
    type Target = StmtObj;

    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

impl ScopeIdDefStmt {
    pub fn new(def: ScopeIdDef, span: Option<&Span>) -> Self {
        Self::from_complete_fields(span.cloned(), def)
    }

    /// Construct a scope-id statement from its complete physical state.
    pub fn from_complete_fields(span: Option<Span>, def: ScopeIdDef) -> Self {
        Self {
            data: ObjectArc::new(ScopeIdDefStmtObj {
                base: StmtObj::new(span),
                def,
            }),
        }
    }
}

#[repr(C)]
#[derive(Object)]
#[type_key = "tirx.LambdaExpr"]
#[type_final]
pub struct LambdaExprObj {
    base: tvm_ffi::Object,
    pub vars: Array<Var>,
    pub pred: PrimExpr,
}

#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct LambdaExpr {
    data: ObjectArc<LambdaExprObj>,
}

impl std::ops::Deref for LambdaExpr {
    type Target = LambdaExprObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl LambdaExpr {
    pub fn new(vars: Vec<Var>, pred: PrimExpr) -> Self {
        Self::from_complete_fields(Array::new(vars), pred)
    }

    /// Construct a lambda expression from its complete physical state.
    pub fn from_complete_fields(vars: Array<Var>, pred: PrimExpr) -> Self {
        Self {
            data: ObjectArc::new(LambdaExprObj {
                base: tvm_ffi::Object::new(),
                vars,
                pred,
            }),
        }
    }

    pub fn apply(&self, indices: Vec<PrimExpr>) -> Result<PrimExpr> {
        tvm_ffi::cached_global_func!("tirx.LambdaExprApply")
            .call_tuple((self, Array::new(indices)))?
            .try_into()
    }
}

#[repr(C)]
#[derive(Object)]
#[type_key = "tirx.DispatchContext"]
#[type_final]
pub struct DispatchContextObj {
    base: tvm_ffi::Object,
    pub target: Target,
    pub exec_scope: ExecScope,
    pub launch_params: Map<FfiString, super::IterVar>,
    pub var_range_map: Map<Var, Range>,
    pub alloc_only: bool,
    // Native methods replace these maps even through a shared context handle.
    // Keep their ABI layout, but never lend out a reference to either slot.
    callbacks: UnsafeCell<Map<FfiString, AnyObjectRef>>,
    shared_state: UnsafeCell<Map<FfiString, AnyObjectRef>>,
    pub inter: Map<FfiString, Array<PrimExpr>>,
    pub intra: Map<FfiString, Array<PrimExpr>>,
    pub scope_kind: FfiString,
}

#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct DispatchContext {
    data: ObjectArc<DispatchContextObj>,
}

impl std::ops::Deref for DispatchContext {
    type Target = DispatchContextObj;

    fn deref(&self) -> &Self::Target {
        &self.data
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
            callbacks,
            shared_state,
            inter,
            intra,
            scope_kind,
        )
    }

    /// Construct a dispatch context from its complete physical state.
    #[allow(clippy::too_many_arguments)]
    pub fn from_complete_fields(
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
        Self {
            data: ObjectArc::new(DispatchContextObj {
                base: tvm_ffi::Object::new(),
                target,
                exec_scope,
                launch_params,
                var_range_map,
                alloc_only,
                callbacks: UnsafeCell::new(callbacks),
                shared_state: UnsafeCell::new(shared_state),
                inter,
                intra,
                scope_kind,
            }),
        }
    }

    /// Snapshot the callbacks. Later native updates leave this map unchanged.
    pub fn callbacks(&self) -> Map<FfiString, AnyObjectRef> {
        // SAFETY: This context cannot be shared across threads, and Map::clone
        // only increments the reference count, without calling user code.
        // The native writer uses Map's copy-on-write after this clone.
        unsafe { (&*self.callbacks.get()).clone() }
    }

    /// Snapshot the shared state. Later native updates leave this map unchanged.
    pub fn shared_state(&self) -> Map<FfiString, AnyObjectRef> {
        // SAFETY: As in callbacks(), the borrow ends before any native mutation.
        unsafe { (&*self.shared_state.get()).clone() }
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

#[repr(C)]
#[derive(Object)]
#[type_key = "tirx.TilePrimitiveCall"]
#[type_final]
pub struct TilePrimitiveCallObj {
    base: StmtObj,
    pub op: Op,
    pub args: Array<Any>,
    pub workspace: Map<FfiString, BufferVar>,
    pub config: Map<FfiString, Any>,
    pub dispatch: Optional<FfiString>,
    pub scope: ExecScope,
}

#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct TilePrimitiveCall {
    data: ObjectArc<TilePrimitiveCallObj>,
}

impl std::ops::Deref for TilePrimitiveCall {
    type Target = TilePrimitiveCallObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl std::ops::Deref for TilePrimitiveCallObj {
    type Target = StmtObj;

    fn deref(&self) -> &Self::Target {
        &self.base
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

    /// Construct a tile-primitive call from its complete physical state after validation.
    #[allow(clippy::too_many_arguments)]
    pub fn from_complete_fields(
        span: Option<Span>,
        op: Op,
        args: Array<Any>,
        workspace: Map<FfiString, BufferVar>,
        config: Map<FfiString, Any>,
        dispatch: Optional<FfiString>,
        scope: ExecScope,
    ) -> Self {
        Self {
            data: ObjectArc::new(TilePrimitiveCallObj {
                base: StmtObj::new(span),
                op,
                args,
                workspace,
                config,
                dispatch,
                scope,
            }),
        }
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
