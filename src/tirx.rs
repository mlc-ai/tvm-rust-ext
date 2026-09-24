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

#![allow(dead_code, unused_imports)]

//! TIRx nodes: statements, loops, buffers, layouts, index maps, tile primitives, and PrimFuncs.
//!
//! The blocks are emitted by `tvm-ffi-stubgen --target rust`; the reviewed
//! statement semantics follow in `mod stmt`, and the buffer, function, index-map,
//! iteration-variable, and tile-primitive semantics live in the sibling modules.

mod buffer;
mod index_map;
mod iter_var;
mod tile_primitive;

pub use buffer::BufferVar;
pub use stmt::{PrimFunc, PrimFuncObj, PrimVar};
pub use tile_primitive::NativeMutableMap;

// Every object registered under `tirx` gets its block in this file; `skip` leaves one out.
// tvm-ffi-stubgen(prefix): tirx
// tvm-ffi-stubgen(custom-new): tirx.TensorMapType
// SBlock APIs are outside this crate's TIRx pass scope.
// tvm-ffi-stubgen(skip): tirx.SBlock
// tvm-ffi-stubgen(skip): tirx.SBlockRealize
// `tirx.PrimFuncPass` derives from `transform.Pass` and carries a `transform.PassInfo`;
// this crate binds neither, and passes are created through the registered factories.
// tvm-ffi-stubgen(skip): tirx.PrimFuncPass
// `tirx.PrimFunc.body` is storage a native pass can move out of and fail to refill; no
// directive expresses that (see STUBGEN_FEEDBACK.md), so the binding stays hand-written in
// `mod stmt` and the two `TensorIntrin` fields name it through `field` overrides.
// tvm-ffi-stubgen(skip): tirx.PrimFunc
// `tirx.DispatchContext.target` refers to the hand-written `target` binding.
// tvm-ffi-stubgen(ty-map): target.Target -> crate::target::Target
// Hand-maintained directives; tvm-ffi-stubgen applies them on every run.
// tvm-ffi-stubgen(import-object): crate::ir::PrimExpr
// tvm-ffi-stubgen(nullable): ir.Expr.span
// tvm-ffi-stubgen(nullable): ir.Type.span
// tvm-ffi-stubgen(nullable): tirx.Stmt.span
// tvm-ffi-stubgen(nullable): tirx.IterVar.dom
// tvm-ffi-stubgen(nullable): tirx.IterVar.span
// tvm-ffi-stubgen(enum): tirx.For.kind -> ForKind(i32) { kSerial=0, kParallel=1, kVectorized=2, kUnrolled=3, kThreadBinding=4 }
// tvm-ffi-stubgen(enum): tirx.ExecScope.kind -> ScopeKind(i32) { CLUSTER=2, CTA=3, WARPGROUP=4, WARP=5, THREAD=6 }
// tvm-ffi-stubgen(enum): tirx.ScopeIdDef.scope -> ScopeBinding(i32) { KERNEL_CLUSTER=0, KERNEL_CTA=1, CLUSTER_CTA=2, CTA_WARPGROUP=3, CTA_WARP=4, WARPGROUP_WARP=5, WARP_THREAD=6, CTA_THREAD=7, WARPGROUP_THREAD=8, CLUSTER_CTA_PAIR=9 }
// tvm-ffi-stubgen(enum): tirx.IterVar.iter_type -> IterVarType(i32) { kDataPar=0, kThreadIndex=1, kCommReduce=2, kOrdered=3, kOpaque=4, kUnrolled=5, kVectorized=6, kParallelized=7, kTensorized=8 }
// tvm-ffi-stubgen(field): tirx.IterVar.var -> PrimVar
// tvm-ffi-stubgen(field): tirx.For.loop_var -> PrimVar
// tvm-ffi-stubgen(field): tirx.For.min -> PrimExpr
// tvm-ffi-stubgen(field): tirx.For.extent -> PrimExpr
// tvm-ffi-stubgen(field): tirx.For.step -> Option<PrimExpr>
// tvm-ffi-stubgen(field): tirx.While.condition -> PrimExpr
// tvm-ffi-stubgen(field): tirx.IfThenElse.condition -> PrimExpr
// tvm-ffi-stubgen(field): tirx.AssertStmt.condition -> PrimExpr
// tvm-ffi-stubgen(field): tirx.AttrStmt.value -> Expr
// tvm-ffi-stubgen(field): tirx.BufferStore.buffer -> BufferVar
// tvm-ffi-stubgen(field): tirx.BufferStore.value -> PrimExpr
// tvm-ffi-stubgen(field): tirx.BufferStore.indices -> Array<PrimExpr>
// tvm-ffi-stubgen(field): tirx.DeclBuffer.buffer -> BufferVar
// tvm-ffi-stubgen(field): tirx.AllocBuffer.buffer -> BufferVar
// tvm-ffi-stubgen(field): tirx.BufferType.shape -> Array<PrimExpr>
// tvm-ffi-stubgen(field): tirx.BufferType.strides -> Array<PrimExpr>
// tvm-ffi-stubgen(field): tirx.BufferType.elem_offset -> PrimExpr
// tvm-ffi-stubgen(field): tirx.BufferType.allocated_addr -> Array<PrimExpr>
// tvm-ffi-stubgen(field): tirx.Iter.extent -> PrimExpr
// tvm-ffi-stubgen(field): tirx.Iter.stride -> PrimExpr
// tvm-ffi-stubgen(field): tirx.IndexMap.initial_indices -> Array<PrimVar>
// tvm-ffi-stubgen(field): tirx.IndexMap.final_indices -> Array<PrimExpr>
// tvm-ffi-stubgen(field): tirx.IndexMap.inverse_index_map -> Option<IndexMap>
// tvm-ffi-stubgen(field): tirx.LambdaExpr.pred -> PrimExpr
// tvm-ffi-stubgen(field): tirx.ScopeIdDef.def_ids -> Array<PrimVar>
// tvm-ffi-stubgen(field): tirx.ScopeIdDef.extents -> Option<Array<PrimExpr>>
// tvm-ffi-stubgen(field): tirx.ScopeIdDef.preferred_extents -> Option<Array<PrimExpr>>
// tvm-ffi-stubgen(field): tirx.TilePrimitiveCall.workspace -> Map<String, BufferVar>
// tvm-ffi-stubgen(field): tirx.DispatchContext.callbacks -> NativeMutableMap
// tvm-ffi-stubgen(field): tirx.DispatchContext.shared_state -> NativeMutableMap
// tvm-ffi-stubgen(field): tirx.DispatchContext.inter -> Map<String, Array<PrimExpr>>
// tvm-ffi-stubgen(field): tirx.DispatchContext.intra -> Map<String, Array<PrimExpr>>
// tvm-ffi-stubgen(custom-new): tirx.Bind
// tvm-ffi-stubgen(custom-new): tirx.AttrStmt
// tvm-ffi-stubgen(custom-new): tirx.AssertStmt
// tvm-ffi-stubgen(custom-new): tirx.Evaluate
// tvm-ffi-stubgen(custom-new): tirx.SeqStmt
// tvm-ffi-stubgen(custom-new): tirx.IfThenElse
// tvm-ffi-stubgen(custom-new): tirx.For
// tvm-ffi-stubgen(custom-new): tirx.While
// tvm-ffi-stubgen(custom-new): tirx.Return
// tvm-ffi-stubgen(custom-new): tirx.Break
// tvm-ffi-stubgen(custom-new): tirx.Continue
// tvm-ffi-stubgen(custom-new): tirx.BufferType
// tvm-ffi-stubgen(custom-new): tirx.BufferRegionType
// tvm-ffi-stubgen(custom-new): tirx.BufferStore
// tvm-ffi-stubgen(custom-new): tirx.DeclBuffer
// tvm-ffi-stubgen(custom-new): tirx.AllocBuffer
// tvm-ffi-stubgen(custom-new): tirx.Iter
// tvm-ffi-stubgen(custom-new): tirx.IndexMap
// tvm-ffi-stubgen(custom-new): tirx.ExecScope
// tvm-ffi-stubgen(custom-new): tirx.ScopeIdDef
// tvm-ffi-stubgen(custom-new): tirx.ScopeIdDefStmt
// tvm-ffi-stubgen(custom-new): tirx.LambdaExpr
// tvm-ffi-stubgen(custom-new): tirx.DispatchContext
// tvm-ffi-stubgen(custom-new): tirx.TilePrimitiveCall

// tvm-ffi-stubgen(begin): import-section
use super::ir::Attrs;
use super::ir::AttrsObj;
use super::ir::Expr;
use super::ir::Op;
use super::ir::PrimExprConvertible;
use super::ir::PrimExprConvertibleObj;
use super::ir::PrimType;
use super::ir::Range;
use super::ir::Span;
use super::ir::StringImm;
use super::ir::Type;
use super::ir::TypeObj;
use super::ir::Var;
use crate::ir::PrimExpr;
use crate::target::Target;
use std::ops::Deref;
use tvm_ffi::Any;
use tvm_ffi::Array;
use tvm_ffi::DLDataType;
use tvm_ffi::Error;
use tvm_ffi::FieldGetter;
use tvm_ffi::Map;
use tvm_ffi::Object;
use tvm_ffi::ObjectArc;
use tvm_ffi::ObjectCore;
use tvm_ffi::Optional;
use tvm_ffi::Result;
use tvm_ffi::String;
use tvm_ffi::VALUE_ERROR;
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/tirx.Axis
/// Opaque: bytes [40, 48) of [24, 48) are not accounted for by reflected fields. Fields are read through the C ABI getters.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "tirx.Axis"]
#[type_final]
pub struct AxisObj {
    base: Object,
}

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Axis {
    base: ObjectArc<AxisObj>,
}

impl Deref for Axis {
    type Target = AxisObj;
    fn deref(&self) -> &AxisObj {
        &self.base
    }
}

impl AxisObj {
    pub fn name(&self) -> Result<String> {
        FieldGetter::new(Self::type_index(), "name")?.get(self)
    }
}
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/tirx.BufferRegionType
/// Complete: reflected fields fill [32, 32) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "tirx.BufferRegionType"]
#[type_final]
pub struct BufferRegionTypeObj {
    base: TypeObj,
}

const _: () = {
    assert!(::core::mem::size_of::<BufferRegionTypeObj>() == 32);
    assert!(::core::mem::align_of::<BufferRegionTypeObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct BufferRegionType {
    base: ObjectArc<BufferRegionTypeObj>,
}

impl Deref for BufferRegionType {
    type Target = BufferRegionTypeObj;
    fn deref(&self) -> &BufferRegionTypeObj {
        &self.base
    }
}

impl Deref for BufferRegionTypeObj {
    type Target = TypeObj;
    fn deref(&self) -> &TypeObj {
        &self.base
    }
}

impl BufferRegionTypeObj {
    pub(crate) fn new(span: Option<Span>) -> Self {
        let base = TypeObj::new(span);
        Self { base }
    }
}

impl BufferRegionType {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(span: Option<Span>) -> Self {
        let obj = BufferRegionTypeObj::new(span);
        Self {
            base: ObjectArc::new(obj),
        }
    }
}

tvm_ffi::impl_object_upcast!(BufferRegionType => Type);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/tirx.BufferType
/// Complete: reflected fields fill [32, 104) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "tirx.BufferType"]
#[type_final]
pub struct BufferTypeObj {
    base: TypeObj,
    pub dtype: PrimType,
    pub storage_scope: String,
    pub shape: Array<PrimExpr>,
    pub strides: Array<PrimExpr>,
    pub elem_offset: PrimExpr,
    pub data_alignment: i32,
    pub offset_factor: i32,
    pub layout: Option<Layout>,
    pub allocated_addr: Array<PrimExpr>,
}

const _: () = {
    assert!(::core::mem::size_of::<BufferTypeObj>() == 104);
    assert!(::core::mem::align_of::<BufferTypeObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct BufferType {
    base: ObjectArc<BufferTypeObj>,
}

impl Deref for BufferType {
    type Target = BufferTypeObj;
    fn deref(&self) -> &BufferTypeObj {
        &self.base
    }
}

impl Deref for BufferTypeObj {
    type Target = TypeObj;
    fn deref(&self) -> &TypeObj {
        &self.base
    }
}

impl BufferTypeObj {
    pub(crate) fn new(
        span: Option<Span>,
        dtype: PrimType,
        storage_scope: String,
        shape: Array<PrimExpr>,
        strides: Array<PrimExpr>,
        elem_offset: PrimExpr,
        data_alignment: i32,
        offset_factor: i32,
        layout: Option<Layout>,
        allocated_addr: Array<PrimExpr>,
    ) -> Self {
        let base = TypeObj::new(span);
        Self {
            base,
            dtype,
            storage_scope,
            shape,
            strides,
            elem_offset,
            data_alignment,
            offset_factor,
            layout,
            allocated_addr,
        }
    }
}

impl BufferType {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(
        span: Option<Span>,
        dtype: PrimType,
        storage_scope: String,
        shape: Array<PrimExpr>,
        strides: Array<PrimExpr>,
        elem_offset: PrimExpr,
        data_alignment: i32,
        offset_factor: i32,
        layout: Option<Layout>,
        allocated_addr: Array<PrimExpr>,
    ) -> Self {
        let obj = BufferTypeObj::new(
            span,
            dtype,
            storage_scope,
            shape,
            strides,
            elem_offset,
            data_alignment,
            offset_factor,
            layout,
            allocated_addr,
        );
        Self {
            base: ObjectArc::new(obj),
        }
    }
}

tvm_ffi::impl_object_upcast!(BufferType => Type);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/tirx.DispatchContext
/// Complete: reflected fields fill [24, 112) exactly (alignment padding [57, 64)).
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "tirx.DispatchContext"]
#[type_final]
pub struct DispatchContextObj {
    base: Object,
    pub target: Target,
    pub exec_scope: ExecScope,
    pub launch_params: Map<String, IterVar>,
    pub var_range_map: Map<Var, Range>,
    pub alloc_only: bool,
    pub callbacks: NativeMutableMap,
    pub shared_state: NativeMutableMap,
    pub inter: Map<String, Array<PrimExpr>>,
    pub intra: Map<String, Array<PrimExpr>>,
    pub scope_kind: String,
}

const _: () = {
    assert!(::core::mem::size_of::<DispatchContextObj>() == 112);
    assert!(::core::mem::align_of::<DispatchContextObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct DispatchContext {
    base: ObjectArc<DispatchContextObj>,
}

impl Deref for DispatchContext {
    type Target = DispatchContextObj;
    fn deref(&self) -> &DispatchContextObj {
        &self.base
    }
}

impl DispatchContextObj {
    pub(crate) fn new(
        target: Target,
        exec_scope: ExecScope,
        launch_params: Map<String, IterVar>,
        var_range_map: Map<Var, Range>,
        alloc_only: bool,
        callbacks: NativeMutableMap,
        shared_state: NativeMutableMap,
        inter: Map<String, Array<PrimExpr>>,
        intra: Map<String, Array<PrimExpr>>,
        scope_kind: String,
    ) -> Self {
        let base = Object::new();
        Self {
            base,
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
        }
    }
}

impl DispatchContext {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(
        target: Target,
        exec_scope: ExecScope,
        launch_params: Map<String, IterVar>,
        var_range_map: Map<Var, Range>,
        alloc_only: bool,
        callbacks: NativeMutableMap,
        shared_state: NativeMutableMap,
        inter: Map<String, Array<PrimExpr>>,
        intra: Map<String, Array<PrimExpr>>,
        scope_kind: String,
    ) -> Self {
        let obj = DispatchContextObj::new(
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
        );
        Self {
            base: ObjectArc::new(obj),
        }
    }
}
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/tirx.ExecScope
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct ScopeKind(i32);

#[allow(non_upper_case_globals)]
impl ScopeKind {
    pub const CLUSTER: Self = Self(2);
    pub const CTA: Self = Self(3);
    pub const WARPGROUP: Self = Self(4);
    pub const WARP: Self = Self(5);
    pub const THREAD: Self = Self(6);
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}

impl TryFrom<i64> for ScopeKind {
    type Error = Error;
    fn try_from(value: i64) -> Result<Self> {
        i32::try_from(value).map(Self).map_err(|_| {
            Error::new(
                VALUE_ERROR,
                &format!("ScopeKind value {value} does not fit i32"),
                "",
            )
        })
    }
}

/// Complete: reflected fields fill [24, 32) exactly (alignment padding [28, 32)).
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "tirx.ExecScope"]
pub struct ExecScopeObj {
    base: Object,
    pub kind: ScopeKind,
}

const _: () = {
    assert!(::core::mem::size_of::<ExecScopeObj>() == 32);
    assert!(::core::mem::align_of::<ExecScopeObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct ExecScope {
    base: ObjectArc<ExecScopeObj>,
}

impl Deref for ExecScope {
    type Target = ExecScopeObj;
    fn deref(&self) -> &ExecScopeObj {
        &self.base
    }
}

impl ExecScopeObj {
    pub(crate) fn new(kind: ScopeKind) -> Self {
        let base = Object::new();
        Self { base, kind }
    }
}

impl ExecScope {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(kind: ScopeKind) -> Self {
        let obj = ExecScopeObj::new(kind);
        Self {
            base: ObjectArc::new(obj),
        }
    }
}
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/tirx.IndexMap
/// Complete: reflected fields fill [24, 48) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "tirx.IndexMap"]
#[type_final]
pub struct IndexMapObj {
    base: Object,
    pub initial_indices: Array<PrimVar>,
    pub final_indices: Array<PrimExpr>,
    pub inverse_index_map: Option<IndexMap>,
}

const _: () = {
    assert!(::core::mem::size_of::<IndexMapObj>() == 48);
    assert!(::core::mem::align_of::<IndexMapObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct IndexMap {
    base: ObjectArc<IndexMapObj>,
}

impl Deref for IndexMap {
    type Target = IndexMapObj;
    fn deref(&self) -> &IndexMapObj {
        &self.base
    }
}

impl IndexMapObj {
    pub(crate) fn new(
        initial_indices: Array<PrimVar>,
        final_indices: Array<PrimExpr>,
        inverse_index_map: Option<IndexMap>,
    ) -> Self {
        let base = Object::new();
        Self {
            base,
            initial_indices,
            final_indices,
            inverse_index_map,
        }
    }
}

impl IndexMap {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(
        initial_indices: Array<PrimVar>,
        final_indices: Array<PrimExpr>,
        inverse_index_map: Option<IndexMap>,
    ) -> Self {
        let obj = IndexMapObj::new(initial_indices, final_indices, inverse_index_map);
        Self {
            base: ObjectArc::new(obj),
        }
    }
}
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/tirx.Iter
/// Complete: reflected fields fill [24, 48) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "tirx.Iter"]
#[type_final]
pub struct IterObj {
    base: Object,
    pub extent: PrimExpr,
    pub stride: PrimExpr,
    pub axis: Axis,
}

const _: () = {
    assert!(::core::mem::size_of::<IterObj>() == 48);
    assert!(::core::mem::align_of::<IterObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Iter {
    base: ObjectArc<IterObj>,
}

impl Deref for Iter {
    type Target = IterObj;
    fn deref(&self) -> &IterObj {
        &self.base
    }
}

impl IterObj {
    pub(crate) fn new(extent: PrimExpr, stride: PrimExpr, axis: Axis) -> Self {
        let base = Object::new();
        Self {
            base,
            extent,
            stride,
            axis,
        }
    }
}

impl Iter {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(extent: PrimExpr, stride: PrimExpr, axis: Axis) -> Self {
        let obj = IterObj::new(extent, stride, axis);
        Self {
            base: ObjectArc::new(obj),
        }
    }
}
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/tirx.IterVar
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct IterVarType(i32);

#[allow(non_upper_case_globals)]
impl IterVarType {
    pub const kDataPar: Self = Self(0);
    pub const kThreadIndex: Self = Self(1);
    pub const kCommReduce: Self = Self(2);
    pub const kOrdered: Self = Self(3);
    pub const kOpaque: Self = Self(4);
    pub const kUnrolled: Self = Self(5);
    pub const kVectorized: Self = Self(6);
    pub const kParallelized: Self = Self(7);
    pub const kTensorized: Self = Self(8);
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}

impl TryFrom<i64> for IterVarType {
    type Error = Error;
    fn try_from(value: i64) -> Result<Self> {
        i32::try_from(value).map(Self).map_err(|_| {
            Error::new(
                VALUE_ERROR,
                &format!("IterVarType value {value} does not fit i32"),
                "",
            )
        })
    }
}

/// Opaque: parent 'ir.PrimExprConvertible' is opaque (layout-unknown). Fields are read through the C ABI getters.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "tirx.IterVar"]
#[type_final]
pub struct IterVarObj {
    base: PrimExprConvertibleObj,
}

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct IterVar {
    base: ObjectArc<IterVarObj>,
}

impl Deref for IterVar {
    type Target = IterVarObj;
    fn deref(&self) -> &IterVarObj {
        &self.base
    }
}

impl Deref for IterVarObj {
    type Target = PrimExprConvertibleObj;
    fn deref(&self) -> &PrimExprConvertibleObj {
        &self.base
    }
}

impl IterVarObj {
    pub fn dom(&self) -> Result<Option<Range>> {
        FieldGetter::new(Self::type_index(), "dom")?.get(self)
    }

    pub fn var(&self) -> Result<PrimVar> {
        FieldGetter::new(Self::type_index(), "var")?.get(self)
    }

    pub fn iter_type(&self) -> Result<IterVarType> {
        let raw: i64 = FieldGetter::new(Self::type_index(), "iter_type")?.get(self)?;
        IterVarType::try_from(raw)
    }

    pub fn thread_tag(&self) -> Result<String> {
        FieldGetter::new(Self::type_index(), "thread_tag")?.get(self)
    }

    pub fn span(&self) -> Result<Option<Span>> {
        FieldGetter::new(Self::type_index(), "span")?.get(self)
    }
}

tvm_ffi::impl_object_upcast!(IterVar => PrimExprConvertible);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/tirx.LambdaExpr
/// Complete: reflected fields fill [24, 40) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "tirx.LambdaExpr"]
#[type_final]
pub struct LambdaExprObj {
    base: Object,
    pub vars: Array<Var>,
    pub pred: PrimExpr,
}

const _: () = {
    assert!(::core::mem::size_of::<LambdaExprObj>() == 40);
    assert!(::core::mem::align_of::<LambdaExprObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct LambdaExpr {
    base: ObjectArc<LambdaExprObj>,
}

impl Deref for LambdaExpr {
    type Target = LambdaExprObj;
    fn deref(&self) -> &LambdaExprObj {
        &self.base
    }
}

impl LambdaExprObj {
    pub(crate) fn new(vars: Array<Var>, pred: PrimExpr) -> Self {
        let base = Object::new();
        Self { base, vars, pred }
    }
}

impl LambdaExpr {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(vars: Array<Var>, pred: PrimExpr) -> Self {
        let obj = LambdaExprObj::new(vars, pred);
        Self {
            base: ObjectArc::new(obj),
        }
    }
}
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/tirx.Layout
/// Opaque: no metadata of its own: total_size is unknown. Fields are read through the C ABI getters.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "tirx.Layout"]
pub struct LayoutObj {
    base: Object,
}

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Layout {
    base: ObjectArc<LayoutObj>,
}

impl Deref for Layout {
    type Target = LayoutObj;
    fn deref(&self) -> &LayoutObj {
        &self.base
    }
}
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/tirx.ComposeLayout
/// Opaque: parent 'tirx.Layout' is opaque (layout-unknown). Fields are read through the C ABI getters.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "tirx.ComposeLayout"]
#[type_final]
pub struct ComposeLayoutObj {
    base: LayoutObj,
}

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct ComposeLayout {
    base: ObjectArc<ComposeLayoutObj>,
}

impl Deref for ComposeLayout {
    type Target = ComposeLayoutObj;
    fn deref(&self) -> &ComposeLayoutObj {
        &self.base
    }
}

impl Deref for ComposeLayoutObj {
    type Target = LayoutObj;
    fn deref(&self) -> &LayoutObj {
        &self.base
    }
}

impl ComposeLayoutObj {
    pub fn per_element(&self) -> Result<i64> {
        FieldGetter::new(Self::type_index(), "per_element")?.get(self)
    }

    pub fn swizzle_len(&self) -> Result<i64> {
        FieldGetter::new(Self::type_index(), "swizzle_len")?.get(self)
    }

    pub fn atom_len(&self) -> Result<i64> {
        FieldGetter::new(Self::type_index(), "atom_len")?.get(self)
    }

    pub fn swizzle_inner(&self) -> Result<bool> {
        FieldGetter::new(Self::type_index(), "swizzle_inner")?.get(self)
    }

    pub fn inner_mask(&self) -> Result<i64> {
        FieldGetter::new(Self::type_index(), "inner_mask")?.get(self)
    }

    pub fn outer_mask(&self) -> Result<i64> {
        FieldGetter::new(Self::type_index(), "outer_mask")?.get(self)
    }

    pub fn tile_layout(&self) -> Result<TileLayout> {
        FieldGetter::new(Self::type_index(), "tile_layout")?.get(self)
    }
}

tvm_ffi::impl_object_upcast!(ComposeLayout => Layout);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/tirx.ScopeIdDef
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct ScopeBinding(i32);

#[allow(non_upper_case_globals)]
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
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}

impl TryFrom<i64> for ScopeBinding {
    type Error = Error;
    fn try_from(value: i64) -> Result<Self> {
        i32::try_from(value).map(Self).map_err(|_| {
            Error::new(
                VALUE_ERROR,
                &format!("ScopeBinding value {value} does not fit i32"),
                "",
            )
        })
    }
}

/// Complete: reflected fields fill [24, 56) exactly (alignment padding [44, 48)).
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "tirx.ScopeIdDef"]
#[type_final]
pub struct ScopeIdDefObj {
    base: Object,
    pub def_ids: Array<PrimVar>,
    pub extents: Option<Array<PrimExpr>>,
    pub scope: ScopeBinding,
    pub preferred_extents: Option<Array<PrimExpr>>,
}

const _: () = {
    assert!(::core::mem::size_of::<ScopeIdDefObj>() == 56);
    assert!(::core::mem::align_of::<ScopeIdDefObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct ScopeIdDef {
    base: ObjectArc<ScopeIdDefObj>,
}

impl Deref for ScopeIdDef {
    type Target = ScopeIdDefObj;
    fn deref(&self) -> &ScopeIdDefObj {
        &self.base
    }
}

impl ScopeIdDefObj {
    pub(crate) fn new(
        def_ids: Array<PrimVar>,
        extents: Option<Array<PrimExpr>>,
        scope: ScopeBinding,
        preferred_extents: Option<Array<PrimExpr>>,
    ) -> Self {
        let base = Object::new();
        Self {
            base,
            def_ids,
            extents,
            scope,
            preferred_extents,
        }
    }
}

impl ScopeIdDef {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(
        def_ids: Array<PrimVar>,
        extents: Option<Array<PrimExpr>>,
        scope: ScopeBinding,
        preferred_extents: Option<Array<PrimExpr>>,
    ) -> Self {
        let obj = ScopeIdDefObj::new(def_ids, extents, scope, preferred_extents);
        Self {
            base: ObjectArc::new(obj),
        }
    }
}
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/tirx.Stmt
/// Complete: reflected fields fill [24, 32) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "tirx.Stmt"]
pub struct StmtObj {
    base: Object,
    pub span: Option<Span>,
}

const _: () = {
    assert!(::core::mem::size_of::<StmtObj>() == 32);
    assert!(::core::mem::align_of::<StmtObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Stmt {
    base: ObjectArc<StmtObj>,
}

impl Deref for Stmt {
    type Target = StmtObj;
    fn deref(&self) -> &StmtObj {
        &self.base
    }
}

impl StmtObj {
    pub(crate) fn new(span: Option<Span>) -> Self {
        let base = Object::new();
        Self { base, span }
    }
}

impl Stmt {
    /// Lossless complete-field allocation.
    pub fn new(span: Option<Span>) -> Self {
        let obj = StmtObj::new(span);
        Self {
            base: ObjectArc::new(obj),
        }
    }
}
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/tirx.AllocBuffer
/// Complete: reflected fields fill [32, 48) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "tirx.AllocBuffer"]
#[type_final]
pub struct AllocBufferObj {
    base: StmtObj,
    pub buffer: BufferVar,
    pub annotations: Map<String, Any>,
}

const _: () = {
    assert!(::core::mem::size_of::<AllocBufferObj>() == 48);
    assert!(::core::mem::align_of::<AllocBufferObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct AllocBuffer {
    base: ObjectArc<AllocBufferObj>,
}

impl Deref for AllocBuffer {
    type Target = AllocBufferObj;
    fn deref(&self) -> &AllocBufferObj {
        &self.base
    }
}

impl Deref for AllocBufferObj {
    type Target = StmtObj;
    fn deref(&self) -> &StmtObj {
        &self.base
    }
}

impl AllocBufferObj {
    pub(crate) fn new(
        span: Option<Span>,
        buffer: BufferVar,
        annotations: Map<String, Any>,
    ) -> Self {
        let base = StmtObj::new(span);
        Self {
            base,
            buffer,
            annotations,
        }
    }
}

impl AllocBuffer {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(
        span: Option<Span>,
        buffer: BufferVar,
        annotations: Map<String, Any>,
    ) -> Self {
        let obj = AllocBufferObj::new(span, buffer, annotations);
        Self {
            base: ObjectArc::new(obj),
        }
    }
}

tvm_ffi::impl_object_upcast!(AllocBuffer => Stmt);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/tirx.AssertStmt
/// Complete: reflected fields fill [32, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "tirx.AssertStmt"]
#[type_final]
pub struct AssertStmtObj {
    base: StmtObj,
    pub condition: PrimExpr,
    pub error_kind: StringImm,
    pub message_parts: Array<StringImm>,
}

const _: () = {
    assert!(::core::mem::size_of::<AssertStmtObj>() == 56);
    assert!(::core::mem::align_of::<AssertStmtObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct AssertStmt {
    base: ObjectArc<AssertStmtObj>,
}

impl Deref for AssertStmt {
    type Target = AssertStmtObj;
    fn deref(&self) -> &AssertStmtObj {
        &self.base
    }
}

impl Deref for AssertStmtObj {
    type Target = StmtObj;
    fn deref(&self) -> &StmtObj {
        &self.base
    }
}

impl AssertStmtObj {
    pub(crate) fn new(
        span: Option<Span>,
        condition: PrimExpr,
        error_kind: StringImm,
        message_parts: Array<StringImm>,
    ) -> Self {
        let base = StmtObj::new(span);
        Self {
            base,
            condition,
            error_kind,
            message_parts,
        }
    }
}

impl AssertStmt {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(
        span: Option<Span>,
        condition: PrimExpr,
        error_kind: StringImm,
        message_parts: Array<StringImm>,
    ) -> Self {
        let obj = AssertStmtObj::new(span, condition, error_kind, message_parts);
        Self {
            base: ObjectArc::new(obj),
        }
    }
}

tvm_ffi::impl_object_upcast!(AssertStmt => Stmt);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/tirx.AttrStmt
/// Complete: reflected fields fill [32, 80) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "tirx.AttrStmt"]
#[type_final]
pub struct AttrStmtObj {
    base: StmtObj,
    pub node: Any,
    pub attr_key: String,
    pub value: Expr,
    pub body: Stmt,
}

const _: () = {
    assert!(::core::mem::size_of::<AttrStmtObj>() == 80);
    assert!(::core::mem::align_of::<AttrStmtObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct AttrStmt {
    base: ObjectArc<AttrStmtObj>,
}

impl Deref for AttrStmt {
    type Target = AttrStmtObj;
    fn deref(&self) -> &AttrStmtObj {
        &self.base
    }
}

impl Deref for AttrStmtObj {
    type Target = StmtObj;
    fn deref(&self) -> &StmtObj {
        &self.base
    }
}

impl AttrStmtObj {
    pub(crate) fn new(
        span: Option<Span>,
        node: Any,
        attr_key: String,
        value: Expr,
        body: Stmt,
    ) -> Self {
        let base = StmtObj::new(span);
        Self {
            base,
            node,
            attr_key,
            value,
            body,
        }
    }
}

impl AttrStmt {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(
        span: Option<Span>,
        node: Any,
        attr_key: String,
        value: Expr,
        body: Stmt,
    ) -> Self {
        let obj = AttrStmtObj::new(span, node, attr_key, value, body);
        Self {
            base: ObjectArc::new(obj),
        }
    }
}

tvm_ffi::impl_object_upcast!(AttrStmt => Stmt);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/tirx.Bind
/// Complete: reflected fields fill [32, 48) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "tirx.Bind"]
#[type_final]
pub struct BindObj {
    base: StmtObj,
    pub var: Var,
    pub value: Expr,
}

const _: () = {
    assert!(::core::mem::size_of::<BindObj>() == 48);
    assert!(::core::mem::align_of::<BindObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Bind {
    base: ObjectArc<BindObj>,
}

impl Deref for Bind {
    type Target = BindObj;
    fn deref(&self) -> &BindObj {
        &self.base
    }
}

impl Deref for BindObj {
    type Target = StmtObj;
    fn deref(&self) -> &StmtObj {
        &self.base
    }
}

impl BindObj {
    pub(crate) fn new(span: Option<Span>, var: Var, value: Expr) -> Self {
        let base = StmtObj::new(span);
        Self { base, var, value }
    }
}

impl Bind {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(span: Option<Span>, var: Var, value: Expr) -> Self {
        let obj = BindObj::new(span, var, value);
        Self {
            base: ObjectArc::new(obj),
        }
    }
}

tvm_ffi::impl_object_upcast!(Bind => Stmt);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/tirx.Break
/// Complete: reflected fields fill [32, 32) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "tirx.Break"]
#[type_final]
pub struct BreakObj {
    base: StmtObj,
}

const _: () = {
    assert!(::core::mem::size_of::<BreakObj>() == 32);
    assert!(::core::mem::align_of::<BreakObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Break {
    base: ObjectArc<BreakObj>,
}

impl Deref for Break {
    type Target = BreakObj;
    fn deref(&self) -> &BreakObj {
        &self.base
    }
}

impl Deref for BreakObj {
    type Target = StmtObj;
    fn deref(&self) -> &StmtObj {
        &self.base
    }
}

impl BreakObj {
    pub(crate) fn new(span: Option<Span>) -> Self {
        let base = StmtObj::new(span);
        Self { base }
    }
}

impl Break {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(span: Option<Span>) -> Self {
        let obj = BreakObj::new(span);
        Self {
            base: ObjectArc::new(obj),
        }
    }
}

tvm_ffi::impl_object_upcast!(Break => Stmt);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/tirx.BufferStore
/// Complete: reflected fields fill [32, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "tirx.BufferStore"]
#[type_final]
pub struct BufferStoreObj {
    base: StmtObj,
    pub buffer: BufferVar,
    pub value: PrimExpr,
    pub indices: Array<PrimExpr>,
}

const _: () = {
    assert!(::core::mem::size_of::<BufferStoreObj>() == 56);
    assert!(::core::mem::align_of::<BufferStoreObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct BufferStore {
    base: ObjectArc<BufferStoreObj>,
}

impl Deref for BufferStore {
    type Target = BufferStoreObj;
    fn deref(&self) -> &BufferStoreObj {
        &self.base
    }
}

impl Deref for BufferStoreObj {
    type Target = StmtObj;
    fn deref(&self) -> &StmtObj {
        &self.base
    }
}

impl BufferStoreObj {
    pub(crate) fn new(
        span: Option<Span>,
        buffer: BufferVar,
        value: PrimExpr,
        indices: Array<PrimExpr>,
    ) -> Self {
        let base = StmtObj::new(span);
        Self {
            base,
            buffer,
            value,
            indices,
        }
    }
}

impl BufferStore {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(
        span: Option<Span>,
        buffer: BufferVar,
        value: PrimExpr,
        indices: Array<PrimExpr>,
    ) -> Self {
        let obj = BufferStoreObj::new(span, buffer, value, indices);
        Self {
            base: ObjectArc::new(obj),
        }
    }
}

tvm_ffi::impl_object_upcast!(BufferStore => Stmt);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/tirx.Continue
/// Complete: reflected fields fill [32, 32) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "tirx.Continue"]
#[type_final]
pub struct ContinueObj {
    base: StmtObj,
}

const _: () = {
    assert!(::core::mem::size_of::<ContinueObj>() == 32);
    assert!(::core::mem::align_of::<ContinueObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Continue {
    base: ObjectArc<ContinueObj>,
}

impl Deref for Continue {
    type Target = ContinueObj;
    fn deref(&self) -> &ContinueObj {
        &self.base
    }
}

impl Deref for ContinueObj {
    type Target = StmtObj;
    fn deref(&self) -> &StmtObj {
        &self.base
    }
}

impl ContinueObj {
    pub(crate) fn new(span: Option<Span>) -> Self {
        let base = StmtObj::new(span);
        Self { base }
    }
}

impl Continue {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(span: Option<Span>) -> Self {
        let obj = ContinueObj::new(span);
        Self {
            base: ObjectArc::new(obj),
        }
    }
}

tvm_ffi::impl_object_upcast!(Continue => Stmt);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/tirx.DeclBuffer
/// Complete: reflected fields fill [32, 48) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "tirx.DeclBuffer"]
#[type_final]
pub struct DeclBufferObj {
    base: StmtObj,
    pub buffer: BufferVar,
    pub data: Expr,
}

const _: () = {
    assert!(::core::mem::size_of::<DeclBufferObj>() == 48);
    assert!(::core::mem::align_of::<DeclBufferObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct DeclBuffer {
    base: ObjectArc<DeclBufferObj>,
}

impl Deref for DeclBuffer {
    type Target = DeclBufferObj;
    fn deref(&self) -> &DeclBufferObj {
        &self.base
    }
}

impl Deref for DeclBufferObj {
    type Target = StmtObj;
    fn deref(&self) -> &StmtObj {
        &self.base
    }
}

impl DeclBufferObj {
    pub(crate) fn new(span: Option<Span>, buffer: BufferVar, data: Expr) -> Self {
        let base = StmtObj::new(span);
        Self { base, buffer, data }
    }
}

impl DeclBuffer {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(span: Option<Span>, buffer: BufferVar, data: Expr) -> Self {
        let obj = DeclBufferObj::new(span, buffer, data);
        Self {
            base: ObjectArc::new(obj),
        }
    }
}

tvm_ffi::impl_object_upcast!(DeclBuffer => Stmt);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/tirx.Evaluate
/// Complete: reflected fields fill [32, 40) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "tirx.Evaluate"]
#[type_final]
pub struct EvaluateObj {
    base: StmtObj,
    pub value: Expr,
}

const _: () = {
    assert!(::core::mem::size_of::<EvaluateObj>() == 40);
    assert!(::core::mem::align_of::<EvaluateObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Evaluate {
    base: ObjectArc<EvaluateObj>,
}

impl Deref for Evaluate {
    type Target = EvaluateObj;
    fn deref(&self) -> &EvaluateObj {
        &self.base
    }
}

impl Deref for EvaluateObj {
    type Target = StmtObj;
    fn deref(&self) -> &StmtObj {
        &self.base
    }
}

impl EvaluateObj {
    pub(crate) fn new(span: Option<Span>, value: Expr) -> Self {
        let base = StmtObj::new(span);
        Self { base, value }
    }
}

impl Evaluate {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(span: Option<Span>, value: Expr) -> Self {
        let obj = EvaluateObj::new(span, value);
        Self {
            base: ObjectArc::new(obj),
        }
    }
}

tvm_ffi::impl_object_upcast!(Evaluate => Stmt);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/tirx.For
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct ForKind(i32);

#[allow(non_upper_case_globals)]
impl ForKind {
    pub const kSerial: Self = Self(0);
    pub const kParallel: Self = Self(1);
    pub const kVectorized: Self = Self(2);
    pub const kUnrolled: Self = Self(3);
    pub const kThreadBinding: Self = Self(4);
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}

impl TryFrom<i64> for ForKind {
    type Error = Error;
    fn try_from(value: i64) -> Result<Self> {
        i32::try_from(value).map(Self).map_err(|_| {
            Error::new(
                VALUE_ERROR,
                &format!("ForKind value {value} does not fit i32"),
                "",
            )
        })
    }
}

/// Complete: reflected fields fill [32, 96) exactly (alignment padding [60, 64)).
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "tirx.For"]
#[type_final]
pub struct ForObj {
    base: StmtObj,
    pub loop_var: PrimVar,
    pub min: PrimExpr,
    pub extent: PrimExpr,
    pub kind: ForKind,
    pub body: Stmt,
    pub thread_binding: Option<IterVar>,
    pub annotations: Map<String, Any>,
    pub step: Option<PrimExpr>,
}

const _: () = {
    assert!(::core::mem::size_of::<ForObj>() == 96);
    assert!(::core::mem::align_of::<ForObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct For {
    base: ObjectArc<ForObj>,
}

impl Deref for For {
    type Target = ForObj;
    fn deref(&self) -> &ForObj {
        &self.base
    }
}

impl Deref for ForObj {
    type Target = StmtObj;
    fn deref(&self) -> &StmtObj {
        &self.base
    }
}

impl ForObj {
    pub(crate) fn new(
        span: Option<Span>,
        loop_var: PrimVar,
        min: PrimExpr,
        extent: PrimExpr,
        kind: ForKind,
        body: Stmt,
        thread_binding: Option<IterVar>,
        annotations: Map<String, Any>,
        step: Option<PrimExpr>,
    ) -> Self {
        let base = StmtObj::new(span);
        Self {
            base,
            loop_var,
            min,
            extent,
            kind,
            body,
            thread_binding,
            annotations,
            step,
        }
    }
}

impl For {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(
        span: Option<Span>,
        loop_var: PrimVar,
        min: PrimExpr,
        extent: PrimExpr,
        kind: ForKind,
        body: Stmt,
        thread_binding: Option<IterVar>,
        annotations: Map<String, Any>,
        step: Option<PrimExpr>,
    ) -> Self {
        let obj = ForObj::new(
            span,
            loop_var,
            min,
            extent,
            kind,
            body,
            thread_binding,
            annotations,
            step,
        );
        Self {
            base: ObjectArc::new(obj),
        }
    }
}

tvm_ffi::impl_object_upcast!(For => Stmt);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/tirx.IfThenElse
/// Complete: reflected fields fill [32, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "tirx.IfThenElse"]
#[type_final]
pub struct IfThenElseObj {
    base: StmtObj,
    pub condition: PrimExpr,
    pub then_case: Stmt,
    pub else_case: Option<Stmt>,
}

const _: () = {
    assert!(::core::mem::size_of::<IfThenElseObj>() == 56);
    assert!(::core::mem::align_of::<IfThenElseObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct IfThenElse {
    base: ObjectArc<IfThenElseObj>,
}

impl Deref for IfThenElse {
    type Target = IfThenElseObj;
    fn deref(&self) -> &IfThenElseObj {
        &self.base
    }
}

impl Deref for IfThenElseObj {
    type Target = StmtObj;
    fn deref(&self) -> &StmtObj {
        &self.base
    }
}

impl IfThenElseObj {
    pub(crate) fn new(
        span: Option<Span>,
        condition: PrimExpr,
        then_case: Stmt,
        else_case: Option<Stmt>,
    ) -> Self {
        let base = StmtObj::new(span);
        Self {
            base,
            condition,
            then_case,
            else_case,
        }
    }
}

impl IfThenElse {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(
        span: Option<Span>,
        condition: PrimExpr,
        then_case: Stmt,
        else_case: Option<Stmt>,
    ) -> Self {
        let obj = IfThenElseObj::new(span, condition, then_case, else_case);
        Self {
            base: ObjectArc::new(obj),
        }
    }
}

tvm_ffi::impl_object_upcast!(IfThenElse => Stmt);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/tirx.Return
/// Complete: reflected fields fill [32, 40) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "tirx.Return"]
#[type_final]
pub struct ReturnObj {
    base: StmtObj,
    pub value: Expr,
}

const _: () = {
    assert!(::core::mem::size_of::<ReturnObj>() == 40);
    assert!(::core::mem::align_of::<ReturnObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Return {
    base: ObjectArc<ReturnObj>,
}

impl Deref for Return {
    type Target = ReturnObj;
    fn deref(&self) -> &ReturnObj {
        &self.base
    }
}

impl Deref for ReturnObj {
    type Target = StmtObj;
    fn deref(&self) -> &StmtObj {
        &self.base
    }
}

impl ReturnObj {
    pub(crate) fn new(span: Option<Span>, value: Expr) -> Self {
        let base = StmtObj::new(span);
        Self { base, value }
    }
}

impl Return {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(span: Option<Span>, value: Expr) -> Self {
        let obj = ReturnObj::new(span, value);
        Self {
            base: ObjectArc::new(obj),
        }
    }
}

tvm_ffi::impl_object_upcast!(Return => Stmt);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/tirx.ScopeIdDefStmt
/// Complete: reflected fields fill [32, 40) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "tirx.ScopeIdDefStmt"]
#[type_final]
pub struct ScopeIdDefStmtObj {
    base: StmtObj,
    pub def: ScopeIdDef,
}

const _: () = {
    assert!(::core::mem::size_of::<ScopeIdDefStmtObj>() == 40);
    assert!(::core::mem::align_of::<ScopeIdDefStmtObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct ScopeIdDefStmt {
    base: ObjectArc<ScopeIdDefStmtObj>,
}

impl Deref for ScopeIdDefStmt {
    type Target = ScopeIdDefStmtObj;
    fn deref(&self) -> &ScopeIdDefStmtObj {
        &self.base
    }
}

impl Deref for ScopeIdDefStmtObj {
    type Target = StmtObj;
    fn deref(&self) -> &StmtObj {
        &self.base
    }
}

impl ScopeIdDefStmtObj {
    pub(crate) fn new(span: Option<Span>, def: ScopeIdDef) -> Self {
        let base = StmtObj::new(span);
        Self { base, def }
    }
}

impl ScopeIdDefStmt {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(span: Option<Span>, def: ScopeIdDef) -> Self {
        let obj = ScopeIdDefStmtObj::new(span, def);
        Self {
            base: ObjectArc::new(obj),
        }
    }
}

tvm_ffi::impl_object_upcast!(ScopeIdDefStmt => Stmt);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/tirx.SeqStmt
/// Complete: reflected fields fill [32, 40) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "tirx.SeqStmt"]
#[type_final]
pub struct SeqStmtObj {
    base: StmtObj,
    pub seq: Array<Stmt>,
}

const _: () = {
    assert!(::core::mem::size_of::<SeqStmtObj>() == 40);
    assert!(::core::mem::align_of::<SeqStmtObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct SeqStmt {
    base: ObjectArc<SeqStmtObj>,
}

impl Deref for SeqStmt {
    type Target = SeqStmtObj;
    fn deref(&self) -> &SeqStmtObj {
        &self.base
    }
}

impl Deref for SeqStmtObj {
    type Target = StmtObj;
    fn deref(&self) -> &StmtObj {
        &self.base
    }
}

impl SeqStmtObj {
    pub(crate) fn new(span: Option<Span>, seq: Array<Stmt>) -> Self {
        let base = StmtObj::new(span);
        Self { base, seq }
    }
}

impl SeqStmt {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(span: Option<Span>, seq: Array<Stmt>) -> Self {
        let obj = SeqStmtObj::new(span, seq);
        Self {
            base: ObjectArc::new(obj),
        }
    }
}

tvm_ffi::impl_object_upcast!(SeqStmt => Stmt);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/tirx.TileLayout
/// Opaque: parent 'tirx.Layout' is opaque (layout-unknown). Fields are read through the C ABI getters.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "tirx.TileLayout"]
#[type_final]
pub struct TileLayoutObj {
    base: LayoutObj,
}

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct TileLayout {
    base: ObjectArc<TileLayoutObj>,
}

impl Deref for TileLayout {
    type Target = TileLayoutObj;
    fn deref(&self) -> &TileLayoutObj {
        &self.base
    }
}

impl Deref for TileLayoutObj {
    type Target = LayoutObj;
    fn deref(&self) -> &LayoutObj {
        &self.base
    }
}

impl TileLayoutObj {
    pub fn shard(&self) -> Result<Array<Iter>> {
        FieldGetter::new(Self::type_index(), "shard")?.get(self)
    }

    pub fn replica(&self) -> Result<Array<Iter>> {
        FieldGetter::new(Self::type_index(), "replica")?.get(self)
    }

    pub fn offset(&self) -> Result<Map<Axis, Expr>> {
        FieldGetter::new(Self::type_index(), "offset")?.get(self)
    }
}

tvm_ffi::impl_object_upcast!(TileLayout => Layout);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/tirx.TilePrimitiveCall
/// Complete: reflected fields fill [32, 88) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "tirx.TilePrimitiveCall"]
#[type_final]
pub struct TilePrimitiveCallObj {
    base: StmtObj,
    pub op: Op,
    pub args: Array<Any>,
    pub workspace: Map<String, BufferVar>,
    pub config: Map<String, Any>,
    pub dispatch: Optional<String>,
    pub scope: ExecScope,
}

const _: () = {
    assert!(::core::mem::size_of::<TilePrimitiveCallObj>() == 88);
    assert!(::core::mem::align_of::<TilePrimitiveCallObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct TilePrimitiveCall {
    base: ObjectArc<TilePrimitiveCallObj>,
}

impl Deref for TilePrimitiveCall {
    type Target = TilePrimitiveCallObj;
    fn deref(&self) -> &TilePrimitiveCallObj {
        &self.base
    }
}

impl Deref for TilePrimitiveCallObj {
    type Target = StmtObj;
    fn deref(&self) -> &StmtObj {
        &self.base
    }
}

impl TilePrimitiveCallObj {
    pub(crate) fn new(
        span: Option<Span>,
        op: Op,
        args: Array<Any>,
        workspace: Map<String, BufferVar>,
        config: Map<String, Any>,
        dispatch: Optional<String>,
        scope: ExecScope,
    ) -> Self {
        let base = StmtObj::new(span);
        Self {
            base,
            op,
            args,
            workspace,
            config,
            dispatch,
            scope,
        }
    }
}

impl TilePrimitiveCall {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(
        span: Option<Span>,
        op: Op,
        args: Array<Any>,
        workspace: Map<String, BufferVar>,
        config: Map<String, Any>,
        dispatch: Optional<String>,
        scope: ExecScope,
    ) -> Self {
        let obj = TilePrimitiveCallObj::new(span, op, args, workspace, config, dispatch, scope);
        Self {
            base: ObjectArc::new(obj),
        }
    }
}

tvm_ffi::impl_object_upcast!(TilePrimitiveCall => Stmt);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/tirx.While
/// Complete: reflected fields fill [32, 48) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "tirx.While"]
#[type_final]
pub struct WhileObj {
    base: StmtObj,
    pub condition: PrimExpr,
    pub body: Stmt,
}

const _: () = {
    assert!(::core::mem::size_of::<WhileObj>() == 48);
    assert!(::core::mem::align_of::<WhileObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct While {
    base: ObjectArc<WhileObj>,
}

impl Deref for While {
    type Target = WhileObj;
    fn deref(&self) -> &WhileObj {
        &self.base
    }
}

impl Deref for WhileObj {
    type Target = StmtObj;
    fn deref(&self) -> &StmtObj {
        &self.base
    }
}

impl WhileObj {
    pub(crate) fn new(span: Option<Span>, condition: PrimExpr, body: Stmt) -> Self {
        let base = StmtObj::new(span);
        Self {
            base,
            condition,
            body,
        }
    }
}

impl While {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(span: Option<Span>, condition: PrimExpr, body: Stmt) -> Self {
        let obj = WhileObj::new(span, condition, body);
        Self {
            base: ObjectArc::new(obj),
        }
    }
}

tvm_ffi::impl_object_upcast!(While => Stmt);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/tirx.TensorMapType
/// Complete: reflected fields fill [32, 32) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "tirx.TensorMapType"]
#[type_final]
pub struct TensorMapTypeObj {
    base: TypeObj,
}

const _: () = {
    assert!(::core::mem::size_of::<TensorMapTypeObj>() == 32);
    assert!(::core::mem::align_of::<TensorMapTypeObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct TensorMapType {
    base: ObjectArc<TensorMapTypeObj>,
}

impl Deref for TensorMapType {
    type Target = TensorMapTypeObj;
    fn deref(&self) -> &TensorMapTypeObj {
        &self.base
    }
}

impl Deref for TensorMapTypeObj {
    type Target = TypeObj;
    fn deref(&self) -> &TypeObj {
        &self.base
    }
}

impl TensorMapTypeObj {
    pub(crate) fn new(span: Option<Span>) -> Self {
        let base = TypeObj::new(span);
        Self { base }
    }
}

impl TensorMapType {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(span: Option<Span>) -> Self {
        let obj = TensorMapTypeObj::new(span);
        Self {
            base: ObjectArc::new(obj),
        }
    }
}

tvm_ffi::impl_object_upcast!(TensorMapType => Type);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/tirx.CallFFIKernelAttr
/// Complete: reflected fields fill [24, 32) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "tirx.CallFFIKernelAttr"]
#[type_final]
pub struct CallFFIKernelAttrObj {
    base: AttrsObj,
    pub launch_params: Array<String>,
}

const _: () = {
    assert!(::core::mem::size_of::<CallFFIKernelAttrObj>() == 32);
    assert!(::core::mem::align_of::<CallFFIKernelAttrObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct CallFFIKernelAttr {
    base: ObjectArc<CallFFIKernelAttrObj>,
}

impl Deref for CallFFIKernelAttr {
    type Target = CallFFIKernelAttrObj;
    fn deref(&self) -> &CallFFIKernelAttrObj {
        &self.base
    }
}

impl Deref for CallFFIKernelAttrObj {
    type Target = AttrsObj;
    fn deref(&self) -> &AttrsObj {
        &self.base
    }
}

impl CallFFIKernelAttrObj {
    pub(crate) fn new(launch_params: Array<String>) -> Self {
        let base = AttrsObj::new();
        Self {
            base,
            launch_params,
        }
    }
}

impl CallFFIKernelAttr {
    /// Lossless complete-field allocation.
    pub fn new(launch_params: Array<String>) -> Self {
        let obj = CallFFIKernelAttrObj::new(launch_params);
        Self {
            base: ObjectArc::new(obj),
        }
    }
}

tvm_ffi::impl_object_upcast!(CallFFIKernelAttr => Attrs);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/tirx.TensorMapEncodeTiledAttr
/// Complete: reflected fields fill [24, 80) exactly (alignment padding [28, 32)).
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "tirx.TensorMapEncodeTiledAttr"]
#[type_final]
pub struct TensorMapEncodeTiledAttrObj {
    base: AttrsObj,
    pub descriptor_dtype: DLDataType,
    pub rank: i64,
    pub interleave: i64,
    pub swizzle: i64,
    pub l2_promotion: i64,
    pub oob_fill: i64,
    pub force_cu_dtype: i64,
}

const _: () = {
    assert!(::core::mem::size_of::<TensorMapEncodeTiledAttrObj>() == 80);
    assert!(::core::mem::align_of::<TensorMapEncodeTiledAttrObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct TensorMapEncodeTiledAttr {
    base: ObjectArc<TensorMapEncodeTiledAttrObj>,
}

impl Deref for TensorMapEncodeTiledAttr {
    type Target = TensorMapEncodeTiledAttrObj;
    fn deref(&self) -> &TensorMapEncodeTiledAttrObj {
        &self.base
    }
}

impl Deref for TensorMapEncodeTiledAttrObj {
    type Target = AttrsObj;
    fn deref(&self) -> &AttrsObj {
        &self.base
    }
}

impl TensorMapEncodeTiledAttrObj {
    pub(crate) fn new(
        descriptor_dtype: DLDataType,
        rank: i64,
        interleave: i64,
        swizzle: i64,
        l2_promotion: i64,
        oob_fill: i64,
        force_cu_dtype: i64,
    ) -> Self {
        let base = AttrsObj::new();
        Self {
            base,
            descriptor_dtype,
            rank,
            interleave,
            swizzle,
            l2_promotion,
            oob_fill,
            force_cu_dtype,
        }
    }
}

impl TensorMapEncodeTiledAttr {
    /// Lossless complete-field allocation.
    pub fn new(
        descriptor_dtype: DLDataType,
        rank: i64,
        interleave: i64,
        swizzle: i64,
        l2_promotion: i64,
        oob_fill: i64,
        force_cu_dtype: i64,
    ) -> Self {
        let obj = TensorMapEncodeTiledAttrObj::new(
            descriptor_dtype,
            rank,
            interleave,
            swizzle,
            l2_promotion,
            oob_fill,
            force_cu_dtype,
        );
        Self {
            base: ObjectArc::new(obj),
        }
    }
}

tvm_ffi::impl_object_upcast!(TensorMapEncodeTiledAttr => Attrs);
// tvm-ffi-stubgen(end)

// ---------------------------------------------------------------------------
// Hand-written semantics for the generated bindings above.  Lines outside the
// `tvm-ffi-stubgen(begin)`/`(end)` blocks are kept verbatim by the generator,
// so this section survives regeneration.  It lives in a nested module so its
// imports cannot collide with the regenerated import section.
// ---------------------------------------------------------------------------

#[warn(dead_code, unused_imports)]
mod stmt {
    use super::*;
    use crate::ir::StringImm;
    use crate::ir::{
        BaseFuncObj, DictAttrs, Expr, IntImm, IntImmObj, PointerTypeObj, PrimExpr, PrimType,
        PrimTypeObj, Span, TupleType, TupleTypeObj, Type, TypedVar, Var,
    };
    use crate::prim::primitive_type;
    use tvm_ffi::{
        Any, Array, DLDataType, DLDataTypeCode, DLDataTypeExt, Error, Map, ObjectRefCore, Result,
        String, TYPE_ERROR, VALUE_ERROR,
    };

    /// Checked scalar view over a `Var` whose expression type is `PrimType`.
    pub type PrimVar = TypedVar<PrimType>;

    tvm_ffi::impl_try_from_any!(PrimVar);
    tvm_ffi::impl_arg_into_ref!(PrimVar);
    tvm_ffi::impl_into_arg_holder_default!(PrimVar);
    impl Bind {
        /// Construct a flat variable binding after checking the value type.
        pub fn new<V>(var: Var, value: V) -> Result<Self>
        where
            V: Into<Expr>,
        {
            Self::with_span(var, value, None)
        }

        /// Construct a flat variable binding with optional source metadata.
        pub fn with_span<V>(var: Var, value: V, span: Option<&Span>) -> Result<Self>
        where
            V: Into<Expr>,
        {
            let value = value.into();
            let same_type: bool = tvm_ffi::cached_global_func!("ffi.StructuralEqual")
                .call_tuple((&var.ty, &value.ty, false, false))?
                .try_into()?;
            if !same_type {
                return Err(Error::new(
                    TYPE_ERROR,
                    "Bind value type must match the bound variable type",
                    "",
                ));
            }
            Ok(Self::from_complete_fields(span.cloned(), var, value))
        }

        /// Copy this node with new `var`, `value`; every other field, span
        /// included, is carried over from `self`.
        ///
        /// Takes the same required fields as [`Bind::new`] and, like
        /// [`Bind::from_complete_fields`], runs no validation.
        pub fn copy_with(&self, var: Var, value: Expr) -> Self {
            Self::from_complete_fields(self.span.clone(), var, value)
        }
    }

    impl AttrStmt {
        /// Construct a scoped attribute directly in Rust.
        pub fn new<N, V, B>(node: N, attr_key: &str, value: V, body: B) -> Result<Self>
        where
            N: Into<Any>,
            V: Into<Expr>,
            B: Into<Stmt>,
        {
            Self::with_span(node, attr_key, value, body, None)
        }

        /// Construct a scoped attribute with optional source metadata.
        pub fn with_span<N, V, B>(
            node: N,
            attr_key: &str,
            value: V,
            body: B,
            span: Option<&Span>,
        ) -> Result<Self>
        where
            N: Into<Any>,
            V: Into<Expr>,
            B: Into<Stmt>,
        {
            Ok(Self::from_complete_fields(
                span.cloned(),
                node.into(),
                String::from(attr_key),
                value.into(),
                body.into(),
            ))
        }

        /// Copy this node with new `node`, `attr_key`, `value`, `body`; every other field, span
        /// included, is carried over from `self`.
        ///
        /// Takes the same required fields as [`AttrStmt::new`] and, like
        /// [`AttrStmt::from_complete_fields`], runs no validation.
        pub fn copy_with(
            &self,
            node: Any,
            attr_key: String,
            value: impl Into<Expr>,
            body: Stmt,
        ) -> Self {
            Self::from_complete_fields(self.span.clone(), node, attr_key, value.into(), body)
        }
    }

    impl Stmt {
        /// Normalize statements into TVM's canonical sequence representation.
        ///
        /// Nested sequences are flattened and `Evaluate(0)` nodes are removed.
        /// An empty result becomes `Evaluate(0)`, one remaining statement is
        /// returned directly, and two or more statements become a [`SeqStmt`].
        pub fn sequence(statements: Vec<Stmt>) -> Result<Self> {
            Self::sequence_with_span(statements, None)
        }

        /// Normalize statements while retaining a span on a newly created sequence.
        pub fn sequence_with_span(statements: Vec<Stmt>, span: Option<&Span>) -> Result<Self> {
            let mut flattened = Vec::new();
            for statement in statements {
                flatten_statement(statement, &mut flattened);
            }
            match flattened.len() {
                0 => Ok(Evaluate::from_i64(0)?.into()),
                1 => Ok(flattened.pop().expect("one statement is present")),
                _ => Ok(SeqStmt::from_complete_fields(span.cloned(), Array::new(flattened)).into()),
            }
        }
    }

    impl SeqStmt {
        /// Consume this sequence and return TVM's canonical flattened statement.
        pub fn flatten(self) -> Result<Stmt> {
            if self.seq.iter().all(|statement| {
                statement.as_node::<SeqStmtObj>().is_none() && !is_evaluate_zero(&statement)
            }) {
                return Ok(self.into());
            }
            Stmt::sequence_with_span(self.seq.iter().collect(), self.span.as_ref())
        }

        /// Construct and recursively flatten a sequence directly in Rust.
        pub fn new(statements: Vec<Stmt>) -> Result<Self> {
            Self::with_span(statements, None)
        }

        /// Construct and recursively flatten a sequence with source metadata.
        pub fn with_span(statements: Vec<Stmt>, span: Option<&Span>) -> Result<Self> {
            let requires_flattening = statements
                .iter()
                .any(|statement| statement.as_node::<SeqStmtObj>().is_some());
            let flattened = if requires_flattening {
                let mut flattened = Vec::new();
                for statement in statements {
                    flatten_statement(statement, &mut flattened);
                }
                flattened
            } else {
                statements
            };
            if flattened.len() < 2 {
                return Err(Error::new(
                    VALUE_ERROR,
                    if flattened.is_empty() {
                        "an empty SeqStmt is prohibited; use Evaluate(0) for a no-op"
                    } else {
                        "a SeqStmt of length one is prohibited; use its single statement directly"
                    },
                    "",
                ));
            }
            Ok(Self::from_complete_fields(
                span.cloned(),
                Array::new(flattened),
            ))
        }

        /// Copy this node with new `seq`; every other field, span
        /// included, is carried over from `self`.
        ///
        /// Takes the same required fields as [`SeqStmt::new`] and, like
        /// [`SeqStmt::from_complete_fields`], runs no validation.
        pub fn copy_with(&self, seq: Array<Stmt>) -> Self {
            Self::from_complete_fields(self.span.clone(), seq)
        }
    }

    fn flatten_statement(statement: Stmt, output: &mut Vec<Stmt>) {
        if let Some(sequence) = statement.as_node::<SeqStmtObj>() {
            for child in sequence.seq.iter() {
                flatten_statement(child, output);
            }
        } else if !is_evaluate_zero(&statement) {
            output.push(statement);
        }
    }

    fn is_evaluate_zero(statement: &Stmt) -> bool {
        statement
            .as_node::<EvaluateObj>()
            .and_then(|evaluate| evaluate.value.as_node::<IntImmObj>())
            .is_some_and(|literal| literal.value_i64() == 0)
    }

    impl IfThenElse {
        // customized_new(IfThenElse) begin
        /// Construct a conditional statement with no `else` branch and no source
        /// metadata, matching the C++ constructor defaults.
        ///
        /// Use [`IfThenElse::with_span`] to attach an `else` branch or a span.
        pub fn new<C, T>(condition: C, then_case: T) -> Result<Self>
        where
            C: Into<Expr>,
            T: Into<Stmt>,
        {
            Self::with_span(condition, then_case, None, None)
        }
        // customized_new(IfThenElse) end

        /// Construct a conditional statement with optional source metadata.
        pub fn with_span<C, T>(
            condition: C,
            then_case: T,
            else_case: Option<Stmt>,
            span: Option<&Span>,
        ) -> Result<Self>
        where
            C: Into<Expr>,
            T: Into<Stmt>,
        {
            let condition = condition.into();
            primitive_type(&condition, "IfThenElse condition")?;
            let condition = PrimExpr::try_from(condition)?;
            Ok(Self::from_complete_fields(
                span.cloned(),
                condition,
                then_case.into(),
                else_case,
            ))
        }

        /// Copy this node with new `condition`, `then_case`; every other field, span
        /// included, is carried over from `self`.
        ///
        /// Takes the same required fields as [`IfThenElse::new`] and, like
        /// [`IfThenElse::from_complete_fields`], runs no validation.
        pub fn copy_with(&self, condition: PrimExpr, then_case: Stmt) -> Self {
            Self::from_complete_fields(
                self.span.clone(),
                condition,
                then_case,
                self.else_case.clone(),
            )
        }
    }

    impl For {
        // customized_new(For) begin
        /// Construct a serial loop with no thread binding, annotations, custom
        /// step, or source metadata.
        ///
        /// Every field the C++ constructor defaults, plus the loop kind, is fixed
        /// here; use [`For::with_metadata`] for any other loop.
        pub fn new<V, M, E, B>(loop_var: V, minimum: M, extent: E, body: B) -> Result<Self>
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
                ForKind::kSerial,
                body.into(),
                None,
                Map::new(),
                None,
                None,
            )
        }
        // customized_new(For) end

        /// Construct a loop directly in Rust with TVM's bound normalization and validation.
        #[allow(clippy::too_many_arguments)]
        pub fn with_metadata(
            loop_var: Var,
            minimum: Expr,
            extent: Expr,
            kind: ForKind,
            body: Stmt,
            thread_binding: Option<IterVar>,
            annotations: Map<String, Any>,
            step: Option<Expr>,
            span: Option<&Span>,
        ) -> Result<Self> {
            let loop_expr: Expr = loop_var.clone().into();
            let loop_dtype = require_scalar_integer(&loop_expr, "loop_var")?;
            require_scalar_integer(&minimum, "min")?;
            require_scalar_integer(&extent, "extent")?;
            let minimum = normalize_loop_bound(&minimum, loop_dtype, "min")?;
            let extent = normalize_loop_bound(&extent, loop_dtype, "extent")?;
            let step = step
                .as_ref()
                .map(|step| {
                    require_scalar_integer(step, "step")?;
                    normalize_loop_bound(step, loop_dtype, "step")
                })
                .transpose()?;
            let loop_var = PrimVar::try_from(loop_var)?;
            Ok(Self::from_complete_fields(
                span.cloned(),
                loop_var,
                minimum,
                extent,
                kind,
                body,
                thread_binding,
                annotations,
                step,
            ))
        }

        /// Copy this node with new `loop_var`, `min`, `extent`, `body`; every other field, span
        /// included, is carried over from `self`.
        ///
        /// Takes the same required fields as [`For::new`] and, like
        /// [`For::from_complete_fields`], runs no validation.
        pub fn copy_with(
            &self,
            loop_var: PrimVar,
            min: PrimExpr,
            extent: PrimExpr,
            body: Stmt,
        ) -> Self {
            Self::from_complete_fields(
                self.span.clone(),
                loop_var,
                min,
                extent,
                self.kind,
                body,
                self.thread_binding.clone(),
                self.annotations.clone(),
                self.step.clone(),
            )
        }
    }

    impl While {
        /// Construct a while loop directly in Rust.
        pub fn new<C, B>(condition: C, body: B) -> Result<Self>
        where
            C: Into<Expr>,
            B: Into<Stmt>,
        {
            Self::with_span(condition, body, None)
        }

        /// Construct a while loop with optional source metadata.
        pub fn with_span<C, B>(condition: C, body: B, span: Option<&Span>) -> Result<Self>
        where
            C: Into<Expr>,
            B: Into<Stmt>,
        {
            let condition = condition.into();
            let dtype = primitive_type(&condition, "While condition")?.dtype;
            if dtype.lanes != 1 {
                return Err(Error::new(
                    TYPE_ERROR,
                    "While condition must be a scalar primitive expression",
                    "",
                ));
            }
            Ok(Self::from_complete_fields(
                span.cloned(),
                PrimExpr::try_from(condition)?,
                body.into(),
            ))
        }

        /// Copy this node with new `condition`, `body`; every other field, span
        /// included, is carried over from `self`.
        ///
        /// Takes the same required fields as [`While::new`] and, like
        /// [`While::from_complete_fields`], runs no validation.
        pub fn copy_with(&self, condition: PrimExpr, body: Stmt) -> Self {
            Self::from_complete_fields(self.span.clone(), condition, body)
        }
    }

    impl Return {
        /// Construct a return statement directly in Rust.
        pub fn new<V>(value: V) -> Self
        where
            V: Into<Expr>,
        {
            Self::with_span(value, None)
        }

        /// Construct a return statement with optional source metadata.
        pub fn with_span<V>(value: V, span: Option<&Span>) -> Self
        where
            V: Into<Expr>,
        {
            Self::from_complete_fields(span.cloned(), value.into())
        }

        /// Copy this node with new `value`; every other field, span
        /// included, is carried over from `self`.
        ///
        /// Takes the same required fields as [`Return::new`] and, like
        /// [`Return::from_complete_fields`], runs no validation.
        pub fn copy_with(&self, value: Expr) -> Self {
            Self::from_complete_fields(self.span.clone(), value)
        }
    }

    macro_rules! define_control_flow_leaf {
        ($reference:ident) => {
            impl $reference {
                /// Construct the control-flow statement directly in Rust.
                pub fn new(span: Option<&Span>) -> Self {
                    Self::from_complete_fields(span.cloned())
                }
            }
        };
    }

    define_control_flow_leaf!(Break);
    define_control_flow_leaf!(Continue);

    fn require_scalar_integer(value: &Expr, field: &str) -> Result<DLDataType> {
        let dtype = primitive_type(value, field)?.dtype;
        let is_integer = dtype.code == DLDataTypeCode::kDLInt as u8
            || dtype.code == DLDataTypeCode::kDLUInt as u8;
        if dtype.lanes != 1 || !is_integer {
            return Err(Error::new(
                TYPE_ERROR,
                &format!("TIR For nodes require a scalar integer {field}"),
                "",
            ));
        }
        Ok(dtype)
    }

    fn normalize_loop_bound(value: &Expr, loop_dtype: DLDataType, field: &str) -> Result<PrimExpr> {
        let value_dtype = primitive_type(value, field)?.dtype;
        if value_dtype == loop_dtype {
            return PrimExpr::try_from(value);
        }
        if let Some(literal) = value.as_node::<IntImmObj>() {
            return PrimExpr::try_from(Expr::from(IntImm::from_dtype(
                loop_dtype,
                literal.value_i64(),
            )?));
        }
        if value_dtype.bits > loop_dtype.bits {
            return Err(Error::new(
                TYPE_ERROR,
                &format!("loop variable dtype is narrower than {field}"),
                "",
            ));
        }
        Err(Error::new(
            TYPE_ERROR,
            &format!("loop variable and {field} must have the same dtype"),
            "",
        ))
    }

    impl AssertStmt {
        /// Construct a leaf assertion directly in Rust with one string message part.
        pub fn new<C>(condition: C, error_kind: &str, message: &str) -> Result<Self>
        where
            C: Into<Expr>,
        {
            let error_kind = StringImm::new(error_kind);
            let message = StringImm::new(message);
            Self::with_metadata(condition, error_kind, vec![message], None)
        }

        /// Construct an assertion from all fields accepted by the C++ constructor.
        pub fn with_metadata<C, K>(
            condition: C,
            error_kind: K,
            message_parts: Vec<StringImm>,
            span: Option<&Span>,
        ) -> Result<Self>
        where
            C: Into<Expr>,
            K: Into<StringImm>,
        {
            let condition = condition.into();
            let dtype = primitive_type(&condition, "AssertStmt condition")?.dtype;
            if dtype.code != DLDataTypeCode::kDLBool as u8 {
                return Err(Error::new(
                    TYPE_ERROR,
                    &format!(
                        "AssertStmt condition must have bool type, but received {}",
                        dtype.to_string()
                    ),
                    "",
                ));
            }
            let condition = PrimExpr::try_from(condition)?;
            Ok(Self::from_complete_fields(
                span.cloned(),
                condition,
                error_kind.into(),
                Array::new(message_parts),
            ))
        }

        /// Copy this node with new `condition`, `error_kind`, `message_parts`; every other field, span
        /// included, is carried over from `self`.
        ///
        /// Takes the same required fields as [`AssertStmt::with_metadata`] and, like
        /// [`AssertStmt::from_complete_fields`], runs no validation.
        pub fn copy_with(
            &self,
            condition: PrimExpr,
            error_kind: StringImm,
            message_parts: Array<StringImm>,
        ) -> Self {
            Self::from_complete_fields(self.span.clone(), condition, error_kind, message_parts)
        }
    }

    impl Evaluate {
        /// Construct `Evaluate(value)` directly in Rust.
        pub fn new<E>(value: E) -> Result<Self>
        where
            E: Into<Expr>,
        {
            Self::with_span(value, None)
        }

        /// Construct `Evaluate(value)` with optional source metadata.
        pub fn with_span<E>(value: E, span: Option<&Span>) -> Result<Self>
        where
            E: Into<Expr>,
        {
            let value = value.into();
            if buffer::BufferVar::try_from(&value).is_ok() {
                return Err(Error::new(
                    VALUE_ERROR,
                    "a buffer variable cannot be used as a scalar Evaluate value",
                    "",
                ));
            }
            Ok(Self::from_complete_fields(span.cloned(), value))
        }

        /// Copy this node with new `value`; every other field, span
        /// included, is carried over from `self`.
        ///
        /// Takes the same required fields as [`Evaluate::new`] and, like
        /// [`Evaluate::from_complete_fields`], runs no validation.
        pub fn copy_with(&self, value: Expr) -> Self {
            Self::from_complete_fields(self.span.clone(), value)
        }

        /// Construct `Evaluate(IntImm("int32", value))`.
        pub fn from_i64(value: i64) -> Result<Self> {
            Self::new(IntImm::new("int32", value)?)
        }
    }

    /// ABI-complete Rust representation of TVM's `PrimFuncNode`.
    ///
    /// Hand-written (the block is `skip`ped): `body` is storage a native pass can
    /// move out of and fail to refill, which no generator directive expresses.
    #[repr(C)]
    #[derive(tvm_ffi::derive::Object)]
    #[type_key = "tirx.PrimFunc"]
    #[type_final]
    pub struct PrimFuncObj {
        base: BaseFuncObj,
        pub params: Array<Var>,
        pub ret_type: Type,
        // Native passes can move this field out and throw before replacing it.
        // Keep the moved-from null state valid for Rust's field destructor.
        body: Option<Stmt>,
    }

    const _: () = {
        assert!(::core::mem::size_of::<PrimFuncObj>() == 72);
        assert!(::core::mem::align_of::<PrimFuncObj>() == 8);
    };

    impl PrimFuncObj {
        /// Borrow the body of an initialized function without cloning its handle.
        ///
        /// # Panics
        ///
        /// Panics if native code has left the function without a body.
        #[inline]
        pub fn body(&self) -> &Stmt {
            self.body.as_ref().expect("PrimFunc has no body")
        }
    }

    /// Reference-counted handle to a TIR primitive function.
    #[repr(C)]
    #[derive(tvm_ffi::derive::ObjectRef, Clone)]
    pub struct PrimFunc {
        data: tvm_ffi::ObjectArc<PrimFuncObj>,
    }

    impl std::ops::Deref for PrimFunc {
        type Target = PrimFuncObj;

        fn deref(&self) -> &Self::Target {
            &self.data
        }
    }

    impl std::ops::Deref for PrimFuncObj {
        type Target = BaseFuncObj;

        fn deref(&self) -> &Self::Target {
            &self.base
        }
    }

    tvm_ffi::impl_object_upcast!(PrimFunc => Expr, PrimFunc => crate::ir::BaseFunc);

    impl PrimFunc {
        /// Construct a PrimFunc allocation entirely in Rust from its complete state.
        ///
        /// `ty` is the native function type stored in the inherited `ExprObj::ty`
        /// field. Supplying it explicitly keeps this raw constructor lossless;
        /// [`PrimFunc::new`] derives it before allocation. The body is required
        /// here even though the stored slot can be left empty by native code.
        pub fn from_complete_fields(
            span: Option<Span>,
            ty: Type,
            attrs: DictAttrs,
            params: Array<Var>,
            ret_type: Type,
            body: Stmt,
        ) -> Self {
            Self {
                data: tvm_ffi::ObjectArc::new(PrimFuncObj {
                    base: BaseFuncObj::new(span, ty, attrs),
                    params,
                    ret_type,
                    body: Some(body),
                }),
            }
        }
    }

    impl PrimFunc {
        /// Construct a parameterless PrimFunc around `body`.
        pub fn from_body<S>(body: S) -> Result<Self>
        where
            S: Into<Stmt>,
        {
            Self::new(Vec::new(), body)
        }

        /// Construct a PrimFunc after deriving its complete type metadata in Rust.
        pub fn new<S>(params: Vec<Var>, body: S) -> Result<Self>
        where
            S: Into<Stmt>,
        {
            let attrs = DictAttrs::empty();
            let ret_type = crate::ir::Type::missing();
            Self::with_metadata(params, body, ret_type, attrs, None)
        }

        /// Construct a PrimFunc using Rust control flow and existing analysis services.
        pub fn with_metadata<S, T, A>(
            params: Vec<Var>,
            body: S,
            ret_type: T,
            attrs: A,
            span: Option<&Span>,
        ) -> Result<Self>
        where
            S: Into<Stmt>,
            T: Into<crate::ir::Type>,
            A: Into<DictAttrs>,
        {
            let body: Stmt = body.into();
            let ret_type = ret_type.into();
            let attrs = attrs.into();
            let params = Array::new(params);
            let (ret_type, function_type) = derive_prim_func_types(&params, &body, ret_type)?;
            Ok(Self::from_complete_fields(
                span.cloned(),
                function_type,
                attrs,
                params,
                ret_type,
                body,
            ))
        }

        /// Copy this function with new `params` and `body`; `ret_type`, `attrs`,
        /// and `span` are carried over from `self`.
        ///
        /// Takes the same required fields as [`PrimFunc::new`]. The function type
        /// is derived again from the new parameters through
        /// [`PrimFunc::with_metadata`], which is why this returns `Result`.
        pub fn copy_with(&self, params: Vec<Var>, body: Stmt) -> Result<Self> {
            Self::with_metadata(
                params,
                body,
                self.ret_type.clone(),
                self.attrs.clone(),
                self.span.as_ref(),
            )
        }
    }

    fn derive_prim_func_types(
        params: &Array<Var>,
        body: &Stmt,
        mut ret_type: Type,
    ) -> Result<(Type, Type)> {
        if ret_type.is_missing() {
            ret_type = TupleType::empty().into();
        }

        let mut parameter_types = Vec::with_capacity(params.len());
        for parameter in params.iter() {
            let parameter_type = if let Some(buffer) = parameter.ty.as_node::<BufferTypeObj>() {
                let mut shape = Vec::with_capacity(buffer.shape.len());
                for dimension in buffer.shape.iter() {
                    shape.push(cast_index_to_i64(dimension.into())?);
                }
                let shape = make_native_shape_expr(Array::new(shape))?;
                make_native_tensor_type(shape, buffer.dtype.clone())?
            } else if parameter.ty.as_node::<PointerTypeObj>().is_some() {
                make_native_any_type()?
            } else {
                parameter.ty.clone()
            };
            parameter_types.push(parameter_type);
        }

        let relax_return_type = if ret_type.as_node::<PrimTypeObj>().is_some() {
            ret_type.clone()
        } else if ret_type
            .as_node::<TupleTypeObj>()
            .is_some_and(|tuple| tuple.fields.is_empty())
        {
            TupleType::empty().into()
        } else {
            make_native_any_type()?
        };

        let provisional = PrimFunc::from_complete_fields(
            None,
            Type::missing(),
            DictAttrs::empty(),
            params.clone(),
            ret_type.clone(),
            body.clone(),
        );
        let purity: bool = tvm_ffi::cached_global_func!("s_tir.analysis.is_pure_function")
            .call_tuple((&provisional, false))?
            .try_into()?;
        let function_type =
            make_native_func_type(Array::new(parameter_types), relax_return_type, purity)?;
        Ok((ret_type, function_type))
    }

    // TVM stores a Relax FuncType in PrimFuncNode::ty even when no Relax IR
    // bindings are exposed.  Keep that native implementation detail behind
    // generic Type/Expr handles so the public Rust surface can remain TIR-only.
    fn make_native_func_type(params: Array<Type>, ret: Type, purity: bool) -> Result<Type> {
        tvm_ffi::cached_global_func!("relax.FuncType")
            .call_tuple((params, ret, purity, Option::<Span>::None))?
            .try_into()
    }

    fn make_native_any_type() -> Result<Type> {
        tvm_ffi::cached_global_func!("relax.AnyType")
            .call_tuple((Option::<Span>::None,))?
            .try_into()
    }

    fn make_native_shape_expr(values: Array<Expr>) -> Result<Expr> {
        tvm_ffi::cached_global_func!("relax.ShapeExpr")
            .call_tuple((values, Option::<Span>::None))?
            .try_into()
    }

    fn make_native_tensor_type(shape: Expr, dtype: PrimType) -> Result<Type> {
        tvm_ffi::cached_global_func!("relax.TensorType")
            .call_tuple((Some(shape), Some(dtype), -1_i32, (), Option::<Span>::None))?
            .try_into()
    }

    fn cast_index_to_i64(value: Expr) -> Result<Expr> {
        let target = PrimType::new("int64")?;
        if primitive_type(&value, "buffer shape")?.dtype == target.dtype {
            return Ok(value);
        }
        if let Some(literal) = value.as_node::<IntImmObj>() {
            return Ok(IntImm::from_complete_fields(
                literal.span.clone(),
                target,
                literal.value.clone(),
            )
            .into());
        }
        tvm_ffi::cached_global_func!("prim.Cast")
            .call_tuple((target, value, Option::<Span>::None))?
            .try_into()
    }
}
