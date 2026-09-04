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

use tvm_ffi::derive::{Object, ObjectRef};
use tvm_ffi::{
    Any, AnyView, Array, Error, FieldGetter, Map, ObjectArc, ObjectCore, ObjectRefCore, Result,
    String as FfiString, VALUE_ERROR,
};

use super::{BufferVar, PrimVar, Stmt, StmtObj};
use crate::ir::{Expr, PrimExpr, Range, Span, Var};
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

    pub const fn name(self) -> &'static str {
        match self.0 {
            2 => "cluster",
            3 => "cta",
            4 => "warpgroup",
            5 => "warp",
            6 => "thread",
            _ => panic!("ScopeKind values are validated when constructed"),
        }
    }
}

impl TryFrom<i64> for ScopeKind {
    type Error = Error;

    fn try_from(value: i64) -> Result<Self> {
        let value = i32::try_from(value)
            .map_err(|_| value_error("ScopeKind does not fit its native i32 representation"))?;
        match value {
            2..=6 => Ok(Self(value)),
            _ => Err(value_error(&format!("unknown ScopeKind value {value}"))),
        }
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

    pub const fn as_raw(self) -> i32 {
        self.0
    }

    pub(crate) const fn name_pair(self) -> (&'static str, &'static str) {
        match self.0 {
            0 => ("kernel", "cluster"),
            1 => ("kernel", "cta"),
            2 => ("cluster", "cta"),
            3 => ("cta", "warpgroup"),
            4 => ("cta", "warp"),
            5 => ("warpgroup", "warp"),
            6 => ("warp", "thread"),
            7 => ("cta", "thread"),
            8 => ("warpgroup", "thread"),
            9 => ("cluster", "cta_pair"),
            _ => panic!("ScopeBinding values are validated when constructed"),
        }
    }
}

impl TryFrom<i64> for ScopeBinding {
    type Error = Error;

    fn try_from(value: i64) -> Result<Self> {
        let value = i32::try_from(value)
            .map_err(|_| value_error("ScopeBinding does not fit its native representation"))?;
        if (0..=9).contains(&value) {
            Ok(Self(value))
        } else {
            Err(value_error(&format!("unknown ScopeBinding value {value}")))
        }
    }
}

/// Native execution scope. Its C++ layout is intentionally opaque.
#[repr(C)]
#[derive(Object)]
#[type_key = "tirx.ExecScope"]
pub struct ExecScopeObj {
    base: tvm_ffi::Object,
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
        Self::from_name(kind.name())
    }

    pub fn from_name(name: &str) -> Result<Self> {
        tvm_ffi::cached_global_func!("tirx.ExecScope")
            .call_tuple((FfiString::from(name),))?
            .try_into()
    }

    pub fn kind(&self) -> Result<ScopeKind> {
        ScopeKind::try_from(field::<i64, _>(self, "kind")?)
    }

    pub fn name(&self) -> Result<&'static str> {
        Ok(self.kind()?.name())
    }
}

/// Native scope-id definition. Its fields are read through reflection.
#[repr(C)]
#[derive(Object)]
#[type_key = "tirx.ScopeIdDef"]
#[type_final]
pub struct ScopeIdDefObj {
    base: tvm_ffi::Object,
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
        let (parent, child) = scope.name_pair();
        tvm_ffi::cached_global_func!("tirx.ScopeIdDef")
            .call_tuple((
                Array::new(def_ids),
                extents.map(Array::new),
                FfiString::from(parent),
                FfiString::from(child),
                preferred_extents.map(Array::new),
            ))?
            .try_into()
    }

    pub fn def_ids(&self) -> Result<Array<PrimVar>> {
        field(self, "def_ids")
    }

    pub fn extents(&self) -> Result<Option<Array<PrimExpr>>> {
        field(self, "extents")
    }

    pub fn scope(&self) -> Result<ScopeBinding> {
        ScopeBinding::try_from(field::<i64, _>(self, "scope")?)
    }

    pub fn preferred_extents(&self) -> Result<Option<Array<PrimExpr>>> {
        field(self, "preferred_extents")
    }

    pub fn is_deferred(&self) -> Result<bool> {
        Ok(self.extents()?.is_none())
    }

    pub fn fused_extent(&self) -> Result<PrimExpr> {
        let extents = self
            .extents()?
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

impl ScopeIdDefStmt {
    pub fn new(definition: ScopeIdDef, span: Option<&Span>) -> Result<Self> {
        tvm_ffi::cached_global_func!("tirx.ScopeIdDefStmt")
            .call_tuple((definition, span.cloned()))?
            .try_into()
    }

    pub fn definition(&self) -> Result<ScopeIdDef> {
        field(self, "def")
    }
}

#[repr(C)]
#[derive(Object)]
#[type_key = "tirx.LambdaExpr"]
#[type_final]
pub struct LambdaExprObj {
    base: tvm_ffi::Object,
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
    pub fn new(vars: Vec<Var>, pred: PrimExpr) -> Result<Self> {
        tvm_ffi::cached_global_func!("tirx.LambdaExpr")
            .call_tuple((Array::new(vars), pred))?
            .try_into()
    }

    pub fn vars(&self) -> Result<Array<Var>> {
        field(self, "vars")
    }

    pub fn pred(&self) -> Result<PrimExpr> {
        field(self, "pred")
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
    pub fn new(target: Target, exec_scope: ExecScope) -> Result<Self> {
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
        callbacks: Map<FfiString, Any>,
        shared_state: Map<FfiString, Any>,
        inter: Map<FfiString, Array<PrimExpr>>,
        intra: Map<FfiString, Array<PrimExpr>>,
        scope_kind: FfiString,
    ) -> Result<Self> {
        tvm_ffi::cached_global_func!("tirx.DispatchContext")
            .call_packed(&[
                AnyView::from(&target),
                AnyView::from(&exec_scope),
                AnyView::from(&launch_params),
                AnyView::from(&var_range_map),
                AnyView::from(&alloc_only),
                AnyView::from(&callbacks),
                AnyView::from(&shared_state),
                AnyView::from(&inter),
                AnyView::from(&intra),
                AnyView::from(&scope_kind),
            ])?
            .try_into()
    }

    pub fn target(&self) -> Result<Target> {
        field(self, "target")
    }

    pub fn exec_scope(&self) -> Result<ExecScope> {
        field(self, "exec_scope")
    }

    pub fn launch_params(&self) -> Result<Map<FfiString, super::IterVar>> {
        field(self, "launch_params")
    }

    pub fn var_range_map(&self) -> Result<Map<Var, Range>> {
        field(self, "var_range_map")
    }

    pub fn alloc_only(&self) -> Result<bool> {
        field(self, "alloc_only")
    }

    pub fn callbacks(&self) -> Result<Map<FfiString, Any>> {
        field(self, "callbacks")
    }

    pub fn shared_state(&self) -> Result<Map<FfiString, Any>> {
        field(self, "shared_state")
    }

    pub fn inter(&self) -> Result<Map<FfiString, Array<PrimExpr>>> {
        field(self, "inter")
    }

    pub fn intra(&self) -> Result<Map<FfiString, Array<PrimExpr>>> {
        field(self, "intra")
    }

    pub fn scope_kind(&self) -> Result<FfiString> {
        field(self, "scope_kind")
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

impl TilePrimitiveCall {
    pub fn new(
        op: Expr,
        args: Vec<Any>,
        workspace: Map<FfiString, BufferVar>,
        config: Map<FfiString, Any>,
        dispatch: Option<FfiString>,
        scope: ExecScope,
    ) -> Result<Self> {
        tvm_ffi::cached_global_func!("tirx.TilePrimitiveCall")
            .call_tuple((op, Array::new(args), workspace, config, dispatch, scope))?
            .try_into()
    }

    pub fn scope(&self) -> Result<ExecScope> {
        field(self, "scope")
    }

    pub fn op(&self) -> Result<Expr> {
        field(self, "op")
    }

    pub fn args(&self) -> Result<Array<Any>> {
        field(self, "args")
    }

    pub fn workspace(&self) -> Result<Map<FfiString, BufferVar>> {
        field(self, "workspace")
    }

    pub fn config(&self) -> Result<Map<FfiString, Any>> {
        field(self, "config")
    }

    pub fn dispatch(&self) -> Result<Option<FfiString>> {
        field(self, "dispatch")
    }
}

fn field<T, O>(object: &O, name: &str) -> Result<T>
where
    T: TryFrom<Any, Error = Error>,
    O: ObjectRefCore,
{
    FieldGetter::new(O::ContainerType::type_index(), name)?.get(&**O::data(object))
}

fn value_error(message: &str) -> Error {
    Error::new(VALUE_ERROR, message, "")
}
