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

//! Common IR nodes: expressions, variables, calls, types, spans, functions, and modules.
//!
//! The object layouts, reference wrappers, `Deref` impls, complete-field
//! allocators, and upcasts inside the `tvm-ffi-stubgen` blocks are emitted by
//! `tvm-ffi-stubgen --target rust` (see README, "Generated bindings"); the
//! reviewed semantic constructors and typed views follow in `mod semantic`.

/// Primitive expression nodes shared by TIRx and other IR dialects.
pub mod prim;

/// Native-operation wrappers of the generated `UniqueNameSupply` handle.
mod unique_name_supply;

pub use semantic::{PrimExpr, TypedExpr, TypedVar};

// Every object registered under `ir` gets its block in this file; `skip` leaves one out.
// tvm-ffi-stubgen(prefix): ir
// `ir.VDevice.target` refers to the hand-written `target` binding (its object struct is `TargetObj`).
// tvm-ffi-stubgen(ty-map): target.Target -> crate::target::Target
// Hand-maintained directives; tvm-ffi-stubgen applies them on every run.
// tvm-ffi-stubgen(nullable): ir.Expr.span
// tvm-ffi-stubgen(nullable): ir.Type.span
// tvm-ffi-stubgen(nullable): ir.Range.span
// tvm-ffi-stubgen(nullable): ir.Span.source_name
// tvm-ffi-stubgen(nullable): ir.Call.attrs
// tvm-ffi-stubgen(field): ir.Range.min -> PrimExpr
// tvm-ffi-stubgen(field): ir.Range.extent -> PrimExpr
// tvm-ffi-stubgen(field): ir.TensorLoad.indices -> Array<PrimExpr>
// tvm-ffi-stubgen(field): ir.TensorLoad.ty -> PrimType
// tvm-ffi-stubgen(field): ir.IntImm.ty -> PrimType
// tvm-ffi-stubgen(field): ir.FloatImm.ty -> PrimType
// tvm-ffi-stubgen(upcast): ir.IntImm -> PrimExpr
// tvm-ffi-stubgen(upcast): ir.FloatImm -> PrimExpr
// tvm-ffi-stubgen(upcast): ir.TensorLoad -> PrimExpr
// tvm-ffi-stubgen(custom-new): ir.Span
// tvm-ffi-stubgen(custom-new): ir.SequentialSpan
// tvm-ffi-stubgen(custom-new): ir.SourceMap
// tvm-ffi-stubgen(custom-new): ir.SourceName
// tvm-ffi-stubgen(custom-new): ir.Range
// tvm-ffi-stubgen(custom-new): ir.Tuple
// tvm-ffi-stubgen(custom-new): ir.TupleGetItem
// tvm-ffi-stubgen(custom-new): ir.TensorLoad
// tvm-ffi-stubgen(custom-new): ir.PointerType
// tvm-ffi-stubgen(custom-new): ir.PrimType
// tvm-ffi-stubgen(custom-new): ir.TupleType
// tvm-ffi-stubgen(custom-new): ir.FuncType
// tvm-ffi-stubgen(custom-new): ir.TensorMapType
// tvm-ffi-stubgen(custom-new): ir.IntImm
// tvm-ffi-stubgen(custom-new): ir.FloatImm
// tvm-ffi-stubgen(custom-new): ir.Var
// tvm-ffi-stubgen(custom-new): ir.GlobalVar
// tvm-ffi-stubgen(custom-new): ir.Call
// tvm-ffi-stubgen(custom-new): ir.IRModule
// tvm-ffi-stubgen(custom-new): ir.DictAttrs

// tvm-ffi-stubgen(begin): import-section
use crate::target::Target;
use std::ops::Deref;
use tvm_ffi::Any;
use tvm_ffi::Array;
use tvm_ffi::DLDataType;
use tvm_ffi::FieldGetter;
use tvm_ffi::Function;
use tvm_ffi::Map;
use tvm_ffi::Object;
use tvm_ffi::ObjectArc;
use tvm_ffi::ObjectCore;
use tvm_ffi::Result;
use tvm_ffi::String;
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/ir.ArgumentInfo
/// Complete: reflected fields fill [24, 72) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.ArgumentInfo"]
#[type_final]
pub struct ArgumentInfoObj {
    base: Object,
    pub name: String,
    pub type_info: String,
    pub description: String,
}

const _: () = {
    assert!(::core::mem::size_of::<ArgumentInfoObj>() == 72);
    assert!(::core::mem::align_of::<ArgumentInfoObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct ArgumentInfo {
    base: ObjectArc<ArgumentInfoObj>,
}

impl Deref for ArgumentInfo {
    type Target = ArgumentInfoObj;
    fn deref(&self) -> &ArgumentInfoObj {
        &self.base
    }
}

impl ArgumentInfoObj {
    pub(crate) fn new(name: String, type_info: String, description: String) -> Self {
        let base = Object::new();
        Self {
            base,
            name,
            type_info,
            description,
        }
    }
}

impl ArgumentInfo {
    /// Lossless complete-field allocation.
    pub fn new(name: String, type_info: String, description: String) -> Self {
        let obj = ArgumentInfoObj::new(name, type_info, description);
        Self {
            base: ObjectArc::new(obj),
        }
    }
}
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/ir.Attrs
/// Complete: reflected fields fill [24, 24) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.Attrs"]
pub struct AttrsObj {
    base: Object,
}

const _: () = {
    assert!(::core::mem::size_of::<AttrsObj>() == 24);
    assert!(::core::mem::align_of::<AttrsObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Attrs {
    base: ObjectArc<AttrsObj>,
}

impl Deref for Attrs {
    type Target = AttrsObj;
    fn deref(&self) -> &AttrsObj {
        &self.base
    }
}

impl AttrsObj {
    pub(crate) fn new() -> Self {
        let base = Object::new();
        Self { base }
    }
}

impl Attrs {
    /// Lossless complete-field allocation.
    pub fn new() -> Self {
        let obj = AttrsObj::new();
        Self {
            base: ObjectArc::new(obj),
        }
    }
}
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/ir.DictAttrs
/// Complete: reflected fields fill [24, 32) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.DictAttrs"]
#[type_final]
pub struct DictAttrsObj {
    base: AttrsObj,
    pub dict: Map<String, Any>,
}

const _: () = {
    assert!(::core::mem::size_of::<DictAttrsObj>() == 32);
    assert!(::core::mem::align_of::<DictAttrsObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct DictAttrs {
    base: ObjectArc<DictAttrsObj>,
}

impl Deref for DictAttrs {
    type Target = DictAttrsObj;
    fn deref(&self) -> &DictAttrsObj {
        &self.base
    }
}

impl Deref for DictAttrsObj {
    type Target = AttrsObj;
    fn deref(&self) -> &AttrsObj {
        &self.base
    }
}

impl DictAttrsObj {
    pub(crate) fn new(dict: Map<String, Any>) -> Self {
        let base = AttrsObj::new();
        Self { base, dict }
    }
}

impl DictAttrs {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(dict: Map<String, Any>) -> Self {
        let obj = DictAttrsObj::new(dict);
        Self {
            base: ObjectArc::new(obj),
        }
    }
}

tvm_ffi::impl_object_upcast!(DictAttrs => Attrs);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/ir.EnvFunc
/// Complete: reflected fields fill [24, 48) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.EnvFunc"]
#[type_final]
pub struct EnvFuncObj {
    base: Object,
    pub name: String,
    pub func: Function,
}

const _: () = {
    assert!(::core::mem::size_of::<EnvFuncObj>() == 48);
    assert!(::core::mem::align_of::<EnvFuncObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct EnvFunc {
    base: ObjectArc<EnvFuncObj>,
}

impl Deref for EnvFunc {
    type Target = EnvFuncObj;
    fn deref(&self) -> &EnvFuncObj {
        &self.base
    }
}

impl EnvFuncObj {
    pub(crate) fn new(name: String, func: Function) -> Self {
        let base = Object::new();
        Self { base, name, func }
    }
}

impl EnvFunc {
    /// Lossless complete-field allocation.
    pub fn new(name: String, func: Function) -> Self {
        let obj = EnvFuncObj::new(name, func);
        Self {
            base: ObjectArc::new(obj),
        }
    }
}
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/ir.Expr
/// Complete: reflected fields fill [24, 40) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.Expr"]
pub struct ExprObj {
    base: Object,
    pub span: Option<Span>,
    pub ty: Type,
}

const _: () = {
    assert!(::core::mem::size_of::<ExprObj>() == 40);
    assert!(::core::mem::align_of::<ExprObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Expr {
    base: ObjectArc<ExprObj>,
}

impl Deref for Expr {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl ExprObj {
    pub(crate) fn new(span: Option<Span>, ty: Type) -> Self {
        let base = Object::new();
        Self { base, span, ty }
    }
}

impl Expr {
    /// Lossless complete-field allocation.
    pub fn new(span: Option<Span>, ty: Type) -> Self {
        let obj = ExprObj::new(span, ty);
        Self {
            base: ObjectArc::new(obj),
        }
    }
}
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/ir.BaseFunc
/// Complete: reflected fields fill [40, 48) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.BaseFunc"]
pub struct BaseFuncObj {
    base: ExprObj,
    pub attrs: DictAttrs,
}

const _: () = {
    assert!(::core::mem::size_of::<BaseFuncObj>() == 48);
    assert!(::core::mem::align_of::<BaseFuncObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct BaseFunc {
    base: ObjectArc<BaseFuncObj>,
}

impl Deref for BaseFunc {
    type Target = BaseFuncObj;
    fn deref(&self) -> &BaseFuncObj {
        &self.base
    }
}

impl Deref for BaseFuncObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl BaseFuncObj {
    pub(crate) fn new(span: Option<Span>, ty: Type, attrs: DictAttrs) -> Self {
        let base = ExprObj::new(span, ty);
        Self { base, attrs }
    }
}

impl BaseFunc {
    /// Lossless complete-field allocation.
    pub fn new(span: Option<Span>, ty: Type, attrs: DictAttrs) -> Self {
        let obj = BaseFuncObj::new(span, ty, attrs);
        Self {
            base: ObjectArc::new(obj),
        }
    }
}

tvm_ffi::impl_object_upcast!(BaseFunc => Expr);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/ir.Call
/// Complete: reflected fields fill [40, 72) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.Call"]
#[type_final]
pub struct CallObj {
    base: ExprObj,
    pub op: Expr,
    pub args: Array<Expr>,
    pub attrs: Option<Attrs>,
    pub ty_args: Array<Type>,
}

const _: () = {
    assert!(::core::mem::size_of::<CallObj>() == 72);
    assert!(::core::mem::align_of::<CallObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Call {
    base: ObjectArc<CallObj>,
}

impl Deref for Call {
    type Target = CallObj;
    fn deref(&self) -> &CallObj {
        &self.base
    }
}

impl Deref for CallObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl CallObj {
    pub(crate) fn new(
        span: Option<Span>,
        ty: Type,
        op: Expr,
        args: Array<Expr>,
        attrs: Option<Attrs>,
        ty_args: Array<Type>,
    ) -> Self {
        let base = ExprObj::new(span, ty);
        Self {
            base,
            op,
            args,
            attrs,
            ty_args,
        }
    }
}

impl Call {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(
        span: Option<Span>,
        ty: Type,
        op: Expr,
        args: Array<Expr>,
        attrs: Option<Attrs>,
        ty_args: Array<Type>,
    ) -> Self {
        let obj = CallObj::new(span, ty, op, args, attrs, ty_args);
        Self {
            base: ObjectArc::new(obj),
        }
    }
}

tvm_ffi::impl_object_upcast!(Call => Expr);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/ir.FloatImm
/// Complete: reflected fields fill [40, 48) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.FloatImm"]
#[type_final]
pub struct FloatImmObj {
    base: ExprObj,
    pub value: f64,
}

const _: () = {
    assert!(::core::mem::size_of::<FloatImmObj>() == 48);
    assert!(::core::mem::align_of::<FloatImmObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct FloatImm {
    base: ObjectArc<FloatImmObj>,
}

impl Deref for FloatImm {
    type Target = FloatImmObj;
    fn deref(&self) -> &FloatImmObj {
        &self.base
    }
}

impl Deref for FloatImmObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl FloatImmObj {
    pub(crate) fn new(span: Option<Span>, ty: PrimType, value: f64) -> Self {
        let base = ExprObj::new(span, ty.into());
        Self { base, value }
    }
}

impl FloatImm {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(span: Option<Span>, ty: PrimType, value: f64) -> Self {
        let obj = FloatImmObj::new(span, ty, value);
        Self {
            base: ObjectArc::new(obj),
        }
    }
}

tvm_ffi::impl_object_upcast!(FloatImm => Expr, FloatImm => PrimExpr);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/ir.GlobalInfo
/// Opaque: no metadata of its own: total_size is unknown. Fields are read through the C ABI getters.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.GlobalInfo"]
pub struct GlobalInfoObj {
    base: Object,
}

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct GlobalInfo {
    base: ObjectArc<GlobalInfoObj>,
}

impl Deref for GlobalInfo {
    type Target = GlobalInfoObj;
    fn deref(&self) -> &GlobalInfoObj {
        &self.base
    }
}
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/ir.DummyGlobalInfo
/// Opaque: parent 'ir.GlobalInfo' is opaque (layout-unknown). Fields are read through the C ABI getters.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.DummyGlobalInfo"]
#[type_final]
pub struct DummyGlobalInfoObj {
    base: GlobalInfoObj,
}

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct DummyGlobalInfo {
    base: ObjectArc<DummyGlobalInfoObj>,
}

impl Deref for DummyGlobalInfo {
    type Target = DummyGlobalInfoObj;
    fn deref(&self) -> &DummyGlobalInfoObj {
        &self.base
    }
}

impl Deref for DummyGlobalInfoObj {
    type Target = GlobalInfoObj;
    fn deref(&self) -> &GlobalInfoObj {
        &self.base
    }
}

tvm_ffi::impl_object_upcast!(DummyGlobalInfo => GlobalInfo);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/ir.GlobalVar
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.GlobalVar"]
#[type_final]
pub struct GlobalVarObj {
    base: ExprObj,
    pub name_hint: String,
}

const _: () = {
    assert!(::core::mem::size_of::<GlobalVarObj>() == 56);
    assert!(::core::mem::align_of::<GlobalVarObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct GlobalVar {
    base: ObjectArc<GlobalVarObj>,
}

impl Deref for GlobalVar {
    type Target = GlobalVarObj;
    fn deref(&self) -> &GlobalVarObj {
        &self.base
    }
}

impl Deref for GlobalVarObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl GlobalVarObj {
    pub(crate) fn new(span: Option<Span>, ty: Type, name_hint: String) -> Self {
        let base = ExprObj::new(span, ty);
        Self { base, name_hint }
    }
}

impl GlobalVar {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(span: Option<Span>, ty: Type, name_hint: String) -> Self {
        let obj = GlobalVarObj::new(span, ty, name_hint);
        Self {
            base: ObjectArc::new(obj),
        }
    }
}

tvm_ffi::impl_object_upcast!(GlobalVar => Expr);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/ir.IRModule
/// Complete: reflected fields fill [24, 64) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.IRModule"]
#[type_final]
pub struct IRModuleObj {
    base: Object,
    pub functions: Map<GlobalVar, BaseFunc>,
    pub source_map: SourceMap,
    pub attrs: DictAttrs,
    pub global_infos: Map<String, Array<GlobalInfo>>,
    pub global_var_map: Map<String, GlobalVar>,
}

const _: () = {
    assert!(::core::mem::size_of::<IRModuleObj>() == 64);
    assert!(::core::mem::align_of::<IRModuleObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct IRModule {
    base: ObjectArc<IRModuleObj>,
}

impl Deref for IRModule {
    type Target = IRModuleObj;
    fn deref(&self) -> &IRModuleObj {
        &self.base
    }
}

impl IRModuleObj {
    pub(crate) fn new(
        functions: Map<GlobalVar, BaseFunc>,
        source_map: SourceMap,
        attrs: DictAttrs,
        global_infos: Map<String, Array<GlobalInfo>>,
        global_var_map: Map<String, GlobalVar>,
    ) -> Self {
        let base = Object::new();
        Self {
            base,
            functions,
            source_map,
            attrs,
            global_infos,
            global_var_map,
        }
    }
}

impl IRModule {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(
        functions: Map<GlobalVar, BaseFunc>,
        source_map: SourceMap,
        attrs: DictAttrs,
        global_infos: Map<String, Array<GlobalInfo>>,
        global_var_map: Map<String, GlobalVar>,
    ) -> Self {
        let obj = IRModuleObj::new(functions, source_map, attrs, global_infos, global_var_map);
        Self {
            base: ObjectArc::new(obj),
        }
    }
}
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/ir.IntImm
/// Complete: reflected fields fill [40, 48) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.IntImm"]
#[type_final]
pub struct IntImmObj {
    base: ExprObj,
    pub value: i64,
}

const _: () = {
    assert!(::core::mem::size_of::<IntImmObj>() == 48);
    assert!(::core::mem::align_of::<IntImmObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct IntImm {
    base: ObjectArc<IntImmObj>,
}

impl Deref for IntImm {
    type Target = IntImmObj;
    fn deref(&self) -> &IntImmObj {
        &self.base
    }
}

impl Deref for IntImmObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl IntImmObj {
    pub(crate) fn new(span: Option<Span>, ty: PrimType, value: i64) -> Self {
        let base = ExprObj::new(span, ty.into());
        Self { base, value }
    }
}

impl IntImm {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(span: Option<Span>, ty: PrimType, value: i64) -> Self {
        let obj = IntImmObj::new(span, ty, value);
        Self {
            base: ObjectArc::new(obj),
        }
    }
}

tvm_ffi::impl_object_upcast!(IntImm => Expr, IntImm => PrimExpr);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/ir.IntSet
/// Opaque: no metadata of its own: total_size is unknown. Fields are read through the C ABI getters.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.IntSet"]
pub struct IntSetObj {
    base: Object,
}

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct IntSet {
    base: ObjectArc<IntSetObj>,
}

impl Deref for IntSet {
    type Target = IntSetObj;
    fn deref(&self) -> &IntSetObj {
        &self.base
    }
}
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/ir.Op
/// Opaque: bytes [96, 100) of [40, 112) are not accounted for by reflected fields. Fields are read through the C ABI getters.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.Op"]
#[type_final]
pub struct OpObj {
    base: ExprObj,
}

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Op {
    base: ObjectArc<OpObj>,
}

impl Deref for Op {
    type Target = OpObj;
    fn deref(&self) -> &OpObj {
        &self.base
    }
}

impl Deref for OpObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl OpObj {
    pub fn name(&self) -> Result<String> {
        FieldGetter::new(Self::type_index(), "name")?.get(self)
    }

    pub fn description(&self) -> Result<String> {
        FieldGetter::new(Self::type_index(), "description")?.get(self)
    }

    pub fn arguments(&self) -> Result<Array<ArgumentInfo>> {
        FieldGetter::new(Self::type_index(), "arguments")?.get(self)
    }

    pub fn attrs_type_key(&self) -> Result<String> {
        FieldGetter::new(Self::type_index(), "attrs_type_key")?.get(self)
    }

    pub fn num_inputs(&self) -> Result<i64> {
        FieldGetter::new(Self::type_index(), "num_inputs")?.get(self)
    }

    pub fn support_level(&self) -> Result<i64> {
        FieldGetter::new(Self::type_index(), "support_level")?.get(self)
    }
}

tvm_ffi::impl_object_upcast!(Op => Expr);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/ir.OpaqueExpr
/// Complete: reflected fields fill [40, 40) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.OpaqueExpr"]
pub struct OpaqueExprObj {
    base: ExprObj,
}

const _: () = {
    assert!(::core::mem::size_of::<OpaqueExprObj>() == 40);
    assert!(::core::mem::align_of::<OpaqueExprObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct OpaqueExpr {
    base: ObjectArc<OpaqueExprObj>,
}

impl Deref for OpaqueExpr {
    type Target = OpaqueExprObj;
    fn deref(&self) -> &OpaqueExprObj {
        &self.base
    }
}

impl Deref for OpaqueExprObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl OpaqueExprObj {
    pub(crate) fn new(span: Option<Span>, ty: Type) -> Self {
        let base = ExprObj::new(span, ty);
        Self { base }
    }
}

impl OpaqueExpr {
    /// Lossless complete-field allocation.
    pub fn new(span: Option<Span>, ty: Type) -> Self {
        let obj = OpaqueExprObj::new(span, ty);
        Self {
            base: ObjectArc::new(obj),
        }
    }
}

tvm_ffi::impl_object_upcast!(OpaqueExpr => Expr);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/ir.PrimExprConvertible
/// Opaque: no metadata of its own: total_size is unknown. Fields are read through the C ABI getters.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.PrimExprConvertible"]
pub struct PrimExprConvertibleObj {
    base: Object,
}

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct PrimExprConvertible {
    base: ObjectArc<PrimExprConvertibleObj>,
}

impl Deref for PrimExprConvertible {
    type Target = PrimExprConvertibleObj;
    fn deref(&self) -> &PrimExprConvertibleObj {
        &self.base
    }
}
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/ir.Range
/// Complete: reflected fields fill [24, 48) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.Range"]
#[type_final]
pub struct RangeObj {
    base: Object,
    pub min: PrimExpr,
    pub extent: PrimExpr,
    pub span: Option<Span>,
}

const _: () = {
    assert!(::core::mem::size_of::<RangeObj>() == 48);
    assert!(::core::mem::align_of::<RangeObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Range {
    base: ObjectArc<RangeObj>,
}

impl Deref for Range {
    type Target = RangeObj;
    fn deref(&self) -> &RangeObj {
        &self.base
    }
}

impl RangeObj {
    pub(crate) fn new(min: PrimExpr, extent: PrimExpr, span: Option<Span>) -> Self {
        let base = Object::new();
        Self {
            base,
            min,
            extent,
            span,
        }
    }
}

impl Range {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(min: PrimExpr, extent: PrimExpr, span: Option<Span>) -> Self {
        let obj = RangeObj::new(min, extent, span);
        Self {
            base: ObjectArc::new(obj),
        }
    }
}
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/ir.Source
/// Opaque: bytes [48, 72) of [24, 72) are not accounted for by reflected fields. Fields are read through the C ABI getters.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.Source"]
#[type_final]
pub struct SourceObj {
    base: Object,
}

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Source {
    base: ObjectArc<SourceObj>,
}

impl Deref for Source {
    type Target = SourceObj;
    fn deref(&self) -> &SourceObj {
        &self.base
    }
}

impl SourceObj {
    pub fn source_name(&self) -> Result<SourceName> {
        FieldGetter::new(Self::type_index(), "source_name")?.get(self)
    }

    pub fn source(&self) -> Result<String> {
        FieldGetter::new(Self::type_index(), "source")?.get(self)
    }
}
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/ir.SourceMap
/// Complete: reflected fields fill [24, 32) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.SourceMap"]
#[type_final]
pub struct SourceMapObj {
    base: Object,
    pub source_map: Map<SourceName, Source>,
}

const _: () = {
    assert!(::core::mem::size_of::<SourceMapObj>() == 32);
    assert!(::core::mem::align_of::<SourceMapObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct SourceMap {
    base: ObjectArc<SourceMapObj>,
}

impl Deref for SourceMap {
    type Target = SourceMapObj;
    fn deref(&self) -> &SourceMapObj {
        &self.base
    }
}

impl SourceMapObj {
    pub(crate) fn new(source_map: Map<SourceName, Source>) -> Self {
        let base = Object::new();
        Self { base, source_map }
    }
}

impl SourceMap {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(source_map: Map<SourceName, Source>) -> Self {
        let obj = SourceMapObj::new(source_map);
        Self {
            base: ObjectArc::new(obj),
        }
    }
}
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/ir.SourceName
/// Complete: reflected fields fill [24, 40) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.SourceName"]
#[type_final]
pub struct SourceNameObj {
    base: Object,
    pub name: String,
}

const _: () = {
    assert!(::core::mem::size_of::<SourceNameObj>() == 40);
    assert!(::core::mem::align_of::<SourceNameObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct SourceName {
    base: ObjectArc<SourceNameObj>,
}

impl Deref for SourceName {
    type Target = SourceNameObj;
    fn deref(&self) -> &SourceNameObj {
        &self.base
    }
}

impl SourceNameObj {
    pub(crate) fn new(name: String) -> Self {
        let base = Object::new();
        Self { base, name }
    }
}

impl SourceName {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(name: String) -> Self {
        let obj = SourceNameObj::new(name);
        Self {
            base: ObjectArc::new(obj),
        }
    }
}
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/ir.Span
/// Complete: reflected fields fill [24, 48) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.Span"]
pub struct SpanObj {
    base: Object,
    pub source_name: Option<SourceName>,
    pub line: i32,
    pub column: i32,
    pub end_line: i32,
    pub end_column: i32,
}

const _: () = {
    assert!(::core::mem::size_of::<SpanObj>() == 48);
    assert!(::core::mem::align_of::<SpanObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Span {
    base: ObjectArc<SpanObj>,
}

impl Deref for Span {
    type Target = SpanObj;
    fn deref(&self) -> &SpanObj {
        &self.base
    }
}

impl SpanObj {
    pub(crate) fn new(
        source_name: Option<SourceName>,
        line: i32,
        column: i32,
        end_line: i32,
        end_column: i32,
    ) -> Self {
        let base = Object::new();
        Self {
            base,
            source_name,
            line,
            column,
            end_line,
            end_column,
        }
    }
}

impl Span {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(
        source_name: Option<SourceName>,
        line: i32,
        column: i32,
        end_line: i32,
        end_column: i32,
    ) -> Self {
        let obj = SpanObj::new(source_name, line, column, end_line, end_column);
        Self {
            base: ObjectArc::new(obj),
        }
    }
}
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/ir.SequentialSpan
/// Complete: reflected fields fill [48, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.SequentialSpan"]
#[type_final]
pub struct SequentialSpanObj {
    base: SpanObj,
    pub spans: Array<Span>,
}

const _: () = {
    assert!(::core::mem::size_of::<SequentialSpanObj>() == 56);
    assert!(::core::mem::align_of::<SequentialSpanObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct SequentialSpan {
    base: ObjectArc<SequentialSpanObj>,
}

impl Deref for SequentialSpan {
    type Target = SequentialSpanObj;
    fn deref(&self) -> &SequentialSpanObj {
        &self.base
    }
}

impl Deref for SequentialSpanObj {
    type Target = SpanObj;
    fn deref(&self) -> &SpanObj {
        &self.base
    }
}

impl SequentialSpanObj {
    pub(crate) fn new(
        source_name: Option<SourceName>,
        line: i32,
        column: i32,
        end_line: i32,
        end_column: i32,
        spans: Array<Span>,
    ) -> Self {
        let base = SpanObj::new(source_name, line, column, end_line, end_column);
        Self { base, spans }
    }
}

impl SequentialSpan {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(
        source_name: Option<SourceName>,
        line: i32,
        column: i32,
        end_line: i32,
        end_column: i32,
        spans: Array<Span>,
    ) -> Self {
        let obj = SequentialSpanObj::new(source_name, line, column, end_line, end_column, spans);
        Self {
            base: ObjectArc::new(obj),
        }
    }
}

tvm_ffi::impl_object_upcast!(SequentialSpan => Span);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/ir.TensorLoad
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.TensorLoad"]
#[type_final]
pub struct TensorLoadObj {
    base: ExprObj,
    pub source: Expr,
    pub indices: Array<PrimExpr>,
}

const _: () = {
    assert!(::core::mem::size_of::<TensorLoadObj>() == 56);
    assert!(::core::mem::align_of::<TensorLoadObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct TensorLoad {
    base: ObjectArc<TensorLoadObj>,
}

impl Deref for TensorLoad {
    type Target = TensorLoadObj;
    fn deref(&self) -> &TensorLoadObj {
        &self.base
    }
}

impl Deref for TensorLoadObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl TensorLoadObj {
    pub(crate) fn new(
        span: Option<Span>,
        ty: PrimType,
        source: Expr,
        indices: Array<PrimExpr>,
    ) -> Self {
        let base = ExprObj::new(span, ty.into());
        Self {
            base,
            source,
            indices,
        }
    }
}

impl TensorLoad {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(
        span: Option<Span>,
        ty: PrimType,
        source: Expr,
        indices: Array<PrimExpr>,
    ) -> Self {
        let obj = TensorLoadObj::new(span, ty, source, indices);
        Self {
            base: ObjectArc::new(obj),
        }
    }
}

tvm_ffi::impl_object_upcast!(TensorLoad => Expr, TensorLoad => PrimExpr);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/ir.Tuple
/// Complete: reflected fields fill [40, 48) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.Tuple"]
#[type_final]
pub struct TupleObj {
    base: ExprObj,
    pub fields: Array<Expr>,
}

const _: () = {
    assert!(::core::mem::size_of::<TupleObj>() == 48);
    assert!(::core::mem::align_of::<TupleObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Tuple {
    base: ObjectArc<TupleObj>,
}

impl Deref for Tuple {
    type Target = TupleObj;
    fn deref(&self) -> &TupleObj {
        &self.base
    }
}

impl Deref for TupleObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl TupleObj {
    pub(crate) fn new(span: Option<Span>, ty: Type, fields: Array<Expr>) -> Self {
        let base = ExprObj::new(span, ty);
        Self { base, fields }
    }
}

impl Tuple {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(span: Option<Span>, ty: Type, fields: Array<Expr>) -> Self {
        let obj = TupleObj::new(span, ty, fields);
        Self {
            base: ObjectArc::new(obj),
        }
    }
}

tvm_ffi::impl_object_upcast!(Tuple => Expr);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/ir.TupleGetItem
/// Complete: reflected fields fill [40, 56) exactly (alignment padding [52, 56)).
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.TupleGetItem"]
#[type_final]
pub struct TupleGetItemObj {
    base: ExprObj,
    pub tuple_value: Expr,
    pub index: i32,
}

const _: () = {
    assert!(::core::mem::size_of::<TupleGetItemObj>() == 56);
    assert!(::core::mem::align_of::<TupleGetItemObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct TupleGetItem {
    base: ObjectArc<TupleGetItemObj>,
}

impl Deref for TupleGetItem {
    type Target = TupleGetItemObj;
    fn deref(&self) -> &TupleGetItemObj {
        &self.base
    }
}

impl Deref for TupleGetItemObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl TupleGetItemObj {
    pub(crate) fn new(span: Option<Span>, ty: Type, tuple_value: Expr, index: i32) -> Self {
        let base = ExprObj::new(span, ty);
        Self {
            base,
            tuple_value,
            index,
        }
    }
}

impl TupleGetItem {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(
        span: Option<Span>,
        ty: Type,
        tuple_value: Expr,
        index: i32,
    ) -> Self {
        let obj = TupleGetItemObj::new(span, ty, tuple_value, index);
        Self {
            base: ObjectArc::new(obj),
        }
    }
}

tvm_ffi::impl_object_upcast!(TupleGetItem => Expr);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/ir.Type
/// Complete: reflected fields fill [24, 32) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.Type"]
pub struct TypeObj {
    base: Object,
    pub span: Option<Span>,
}

const _: () = {
    assert!(::core::mem::size_of::<TypeObj>() == 32);
    assert!(::core::mem::align_of::<TypeObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Type {
    base: ObjectArc<TypeObj>,
}

impl Deref for Type {
    type Target = TypeObj;
    fn deref(&self) -> &TypeObj {
        &self.base
    }
}

impl TypeObj {
    pub(crate) fn new(span: Option<Span>) -> Self {
        let base = Object::new();
        Self { base, span }
    }
}

impl Type {
    /// Lossless complete-field allocation.
    pub fn new(span: Option<Span>) -> Self {
        let obj = TypeObj::new(span);
        Self {
            base: ObjectArc::new(obj),
        }
    }
}
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/ir.FuncType
/// Complete: reflected fields fill [32, 48) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.FuncType"]
#[type_final]
pub struct FuncTypeObj {
    base: TypeObj,
    pub arg_types: Array<Type>,
    pub ret_type: Type,
}

const _: () = {
    assert!(::core::mem::size_of::<FuncTypeObj>() == 48);
    assert!(::core::mem::align_of::<FuncTypeObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct FuncType {
    base: ObjectArc<FuncTypeObj>,
}

impl Deref for FuncType {
    type Target = FuncTypeObj;
    fn deref(&self) -> &FuncTypeObj {
        &self.base
    }
}

impl Deref for FuncTypeObj {
    type Target = TypeObj;
    fn deref(&self) -> &TypeObj {
        &self.base
    }
}

impl FuncTypeObj {
    pub(crate) fn new(span: Option<Span>, arg_types: Array<Type>, ret_type: Type) -> Self {
        let base = TypeObj::new(span);
        Self {
            base,
            arg_types,
            ret_type,
        }
    }
}

impl FuncType {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(
        span: Option<Span>,
        arg_types: Array<Type>,
        ret_type: Type,
    ) -> Self {
        let obj = FuncTypeObj::new(span, arg_types, ret_type);
        Self {
            base: ObjectArc::new(obj),
        }
    }
}

tvm_ffi::impl_object_upcast!(FuncType => Type);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/ir.OpaqueType
/// Complete: reflected fields fill [32, 32) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.OpaqueType"]
#[type_final]
pub struct OpaqueTypeObj {
    base: TypeObj,
}

const _: () = {
    assert!(::core::mem::size_of::<OpaqueTypeObj>() == 32);
    assert!(::core::mem::align_of::<OpaqueTypeObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct OpaqueType {
    base: ObjectArc<OpaqueTypeObj>,
}

impl Deref for OpaqueType {
    type Target = OpaqueTypeObj;
    fn deref(&self) -> &OpaqueTypeObj {
        &self.base
    }
}

impl Deref for OpaqueTypeObj {
    type Target = TypeObj;
    fn deref(&self) -> &TypeObj {
        &self.base
    }
}

impl OpaqueTypeObj {
    pub(crate) fn new(span: Option<Span>) -> Self {
        let base = TypeObj::new(span);
        Self { base }
    }
}

impl OpaqueType {
    /// Lossless complete-field allocation.
    pub fn new(span: Option<Span>) -> Self {
        let obj = OpaqueTypeObj::new(span);
        Self {
            base: ObjectArc::new(obj),
        }
    }
}

tvm_ffi::impl_object_upcast!(OpaqueType => Type);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/ir.PointerType
/// Complete: reflected fields fill [32, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.PointerType"]
#[type_final]
pub struct PointerTypeObj {
    base: TypeObj,
    pub element_type: Type,
    pub storage_scope: String,
}

const _: () = {
    assert!(::core::mem::size_of::<PointerTypeObj>() == 56);
    assert!(::core::mem::align_of::<PointerTypeObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct PointerType {
    base: ObjectArc<PointerTypeObj>,
}

impl Deref for PointerType {
    type Target = PointerTypeObj;
    fn deref(&self) -> &PointerTypeObj {
        &self.base
    }
}

impl Deref for PointerTypeObj {
    type Target = TypeObj;
    fn deref(&self) -> &TypeObj {
        &self.base
    }
}

impl PointerTypeObj {
    pub(crate) fn new(span: Option<Span>, element_type: Type, storage_scope: String) -> Self {
        let base = TypeObj::new(span);
        Self {
            base,
            element_type,
            storage_scope,
        }
    }
}

impl PointerType {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(
        span: Option<Span>,
        element_type: Type,
        storage_scope: String,
    ) -> Self {
        let obj = PointerTypeObj::new(span, element_type, storage_scope);
        Self {
            base: ObjectArc::new(obj),
        }
    }
}

tvm_ffi::impl_object_upcast!(PointerType => Type);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/ir.PrimType
/// Complete: reflected fields fill [32, 40) exactly (alignment padding [36, 40)).
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.PrimType"]
#[type_final]
pub struct PrimTypeObj {
    base: TypeObj,
    pub dtype: DLDataType,
}

const _: () = {
    assert!(::core::mem::size_of::<PrimTypeObj>() == 40);
    assert!(::core::mem::align_of::<PrimTypeObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct PrimType {
    base: ObjectArc<PrimTypeObj>,
}

impl Deref for PrimType {
    type Target = PrimTypeObj;
    fn deref(&self) -> &PrimTypeObj {
        &self.base
    }
}

impl Deref for PrimTypeObj {
    type Target = TypeObj;
    fn deref(&self) -> &TypeObj {
        &self.base
    }
}

impl PrimTypeObj {
    pub(crate) fn new(span: Option<Span>, dtype: DLDataType) -> Self {
        let base = TypeObj::new(span);
        Self { base, dtype }
    }
}

impl PrimType {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(span: Option<Span>, dtype: DLDataType) -> Self {
        let obj = PrimTypeObj::new(span, dtype);
        Self {
            base: ObjectArc::new(obj),
        }
    }
}

tvm_ffi::impl_object_upcast!(PrimType => Type);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/ir.TensorMapType
/// Complete: reflected fields fill [32, 32) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.TensorMapType"]
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

// tvm-ffi-stubgen(begin): object/ir.TupleType
/// Complete: reflected fields fill [32, 40) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.TupleType"]
#[type_final]
pub struct TupleTypeObj {
    base: TypeObj,
    pub fields: Array<Type>,
}

const _: () = {
    assert!(::core::mem::size_of::<TupleTypeObj>() == 40);
    assert!(::core::mem::align_of::<TupleTypeObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct TupleType {
    base: ObjectArc<TupleTypeObj>,
}

impl Deref for TupleType {
    type Target = TupleTypeObj;
    fn deref(&self) -> &TupleTypeObj {
        &self.base
    }
}

impl Deref for TupleTypeObj {
    type Target = TypeObj;
    fn deref(&self) -> &TypeObj {
        &self.base
    }
}

impl TupleTypeObj {
    pub(crate) fn new(span: Option<Span>, fields: Array<Type>) -> Self {
        let base = TypeObj::new(span);
        Self { base, fields }
    }
}

impl TupleType {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(span: Option<Span>, fields: Array<Type>) -> Self {
        let obj = TupleTypeObj::new(span, fields);
        Self {
            base: ObjectArc::new(obj),
        }
    }
}

tvm_ffi::impl_object_upcast!(TupleType => Type);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/ir.UniqueNameSupply
/// Opaque: bytes [24, 64) of [24, 64) are not accounted for by reflected fields. Fields are read through the C ABI getters.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.UniqueNameSupply"]
#[type_final]
pub struct UniqueNameSupplyObj {
    base: Object,
}

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct UniqueNameSupply {
    base: ObjectArc<UniqueNameSupplyObj>,
}

impl Deref for UniqueNameSupply {
    type Target = UniqueNameSupplyObj;
    fn deref(&self) -> &UniqueNameSupplyObj {
        &self.base
    }
}
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/ir.VDevice
/// Opaque: parent 'ir.GlobalInfo' is opaque (layout-unknown). Fields are read through the C ABI getters.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.VDevice"]
#[type_final]
pub struct VDeviceObj {
    base: GlobalInfoObj,
}

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct VDevice {
    base: ObjectArc<VDeviceObj>,
}

impl Deref for VDevice {
    type Target = VDeviceObj;
    fn deref(&self) -> &VDeviceObj {
        &self.base
    }
}

impl Deref for VDeviceObj {
    type Target = GlobalInfoObj;
    fn deref(&self) -> &GlobalInfoObj {
        &self.base
    }
}

impl VDeviceObj {
    pub fn target(&self) -> Result<Target> {
        FieldGetter::new(Self::type_index(), "target")?.get(self)
    }

    pub fn vdevice_id(&self) -> Result<i64> {
        FieldGetter::new(Self::type_index(), "vdevice_id")?.get(self)
    }

    pub fn memory_scope(&self) -> Result<String> {
        FieldGetter::new(Self::type_index(), "memory_scope")?.get(self)
    }
}

tvm_ffi::impl_object_upcast!(VDevice => GlobalInfo);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/ir.Var
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.Var"]
pub struct VarObj {
    base: ExprObj,
    pub name: String,
}

const _: () = {
    assert!(::core::mem::size_of::<VarObj>() == 56);
    assert!(::core::mem::align_of::<VarObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Var {
    base: ObjectArc<VarObj>,
}

impl Deref for Var {
    type Target = VarObj;
    fn deref(&self) -> &VarObj {
        &self.base
    }
}

impl Deref for VarObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl VarObj {
    pub(crate) fn new(span: Option<Span>, ty: Type, name: String) -> Self {
        let base = ExprObj::new(span, ty);
        Self { base, name }
    }
}

impl Var {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(span: Option<Span>, ty: Type, name: String) -> Self {
        let obj = VarObj::new(span, ty, name);
        Self {
            base: ObjectArc::new(obj),
        }
    }
}

tvm_ffi::impl_object_upcast!(Var => Expr);
// tvm-ffi-stubgen(end)

// ---------------------------------------------------------------------------
// Hand-written semantics for the generated bindings above.  Lines outside the
// `tvm-ffi-stubgen(begin)`/`(end)` blocks are kept verbatim by the generator,
// so this section survives regeneration.  It lives in a nested module so its
// imports cannot collide with the regenerated import section.
// ---------------------------------------------------------------------------

#[warn(dead_code, unused_imports)]
mod semantic {
    use super::*;
    use tvm_ffi::{
        Any, AnyCompatible, AnyView, Array, DLDataType, DLDataTypeCode, DLDataTypeExt, Error,
        FieldGetter, Map, ObjectArc, ObjectCore, ObjectRefCast, ObjectRefCore, Result, String,
        TVMFFIAny, INDEX_ERROR, TYPE_ERROR, VALUE_ERROR,
    };

    impl Op {
        /// Return the registry singleton named `name`.
        pub fn get(name: &str) -> Result<Self> {
            tvm_ffi::cached_global_func!("ir.GetOp")
                .call_tuple((String::from(name),))?
                .try_into()
        }

        /// Return the operator's registered name.
        pub fn name(&self) -> Result<String> {
            FieldGetter::new(OpObj::type_index(), "name")?.get(&**self)
        }
    }

    /// Checked view over an expression whose result type is `T`.
    ///
    /// This mirrors C++ `TypedExpr<T>`: it retains the same `ExprNode` allocation
    /// and adds a runtime check on the expression's `ty` field.
    #[repr(transparent)]
    pub struct TypedExpr<T> {
        expr: Expr,
        _expected_type: std::marker::PhantomData<T>,
    }

    impl<T> TypedExpr<T> {
        #[inline]
        pub(crate) unsafe fn from_expr_unchecked(expr: Expr) -> Self {
            Self {
                expr,
                _expected_type: std::marker::PhantomData,
            }
        }

        /// Return the general expression view over the same allocation.
        #[inline]
        pub fn as_expr(&self) -> &Expr {
            &self.expr
        }
    }

    impl<T> TypedExpr<T>
    where
        T: ObjectRefCore + AnyCompatible + 'static,
    {
        /// Return the type guaranteed by this checked expression view.
        #[inline]
        pub fn type_annotation(&self) -> T {
            self.expr
                .ty
                .clone()
                .try_cast()
                .expect("TypedExpr type invariant was violated")
        }
    }

    impl<T> Clone for TypedExpr<T> {
        #[inline]
        fn clone(&self) -> Self {
            Self {
                expr: self.expr.clone(),
                _expected_type: std::marker::PhantomData,
            }
        }
    }

    impl<T> std::ops::Deref for TypedExpr<T> {
        type Target = Expr;

        #[inline]
        fn deref(&self) -> &Self::Target {
            &self.expr
        }
    }

    unsafe impl<T: 'static> ObjectRefCore for TypedExpr<T> {
        type ContainerType = ExprObj;

        #[inline]
        fn data(this: &Self) -> &ObjectArc<Self::ContainerType> {
            <Expr as ObjectRefCore>::data(&this.expr)
        }

        #[inline]
        fn into_data(this: Self) -> ObjectArc<Self::ContainerType> {
            <Expr as ObjectRefCore>::into_data(this.expr)
        }

        #[inline]
        unsafe fn from_data(data: ObjectArc<Self::ContainerType>) -> Self {
            // SAFETY: The caller of this unsafe constructor has guaranteed that
            // the expression's `ty` satisfies this typed view's invariant.
            unsafe { Self::from_expr_unchecked(<Expr as ObjectRefCore>::from_data(data)) }
        }
    }

    unsafe impl<T> AnyCompatible for TypedExpr<T>
    where
        T: ObjectRefCore + AnyCompatible + 'static,
    {
        #[inline]
        unsafe fn copy_to_any_view(src: &Self, data: &mut TVMFFIAny) {
            Expr::copy_to_any_view(&src.expr, data);
        }

        #[inline]
        unsafe fn move_to_any(src: Self, data: &mut TVMFFIAny) {
            Expr::move_to_any(src.expr, data);
        }

        #[inline]
        unsafe fn check_any_strict(data: &TVMFFIAny) -> bool {
            if !Expr::check_any_strict(data) {
                return false;
            }
            let expr = &*data.data_union.v_obj.cast::<ExprObj>();
            let mut type_data = TVMFFIAny::new();
            Type::copy_to_any_view(&expr.ty, &mut type_data);
            T::check_any_strict(&type_data)
        }

        #[inline]
        unsafe fn copy_from_any_view_after_check(data: &TVMFFIAny) -> Self {
            Self::from_expr_unchecked(Expr::copy_from_any_view_after_check(data))
        }

        #[inline]
        unsafe fn move_from_any_after_check(data: &mut TVMFFIAny) -> Self {
            Self::from_expr_unchecked(Expr::move_from_any_after_check(data))
        }

        #[inline]
        unsafe fn try_cast_from_any_view(data: &TVMFFIAny) -> std::result::Result<Self, ()> {
            if Self::check_any_strict(data) {
                Ok(Self::copy_from_any_view_after_check(data))
            } else {
                Err(())
            }
        }

        fn type_str() -> std::string::String {
            format!("TypedExpr<{}>", T::type_str())
        }
    }

    impl<T> From<TypedExpr<T>> for Expr {
        #[inline]
        fn from(value: TypedExpr<T>) -> Self {
            value.expr
        }
    }

    impl<T> From<&TypedExpr<T>> for Expr {
        #[inline]
        fn from(value: &TypedExpr<T>) -> Self {
            value.expr.clone()
        }
    }

    impl<T> From<&TypedExpr<T>> for TypedExpr<T> {
        #[inline]
        fn from(value: &TypedExpr<T>) -> Self {
            value.clone()
        }
    }

    impl<T> TryFrom<Expr> for TypedExpr<T>
    where
        T: ObjectRefCore + AnyCompatible + 'static,
    {
        type Error = Error;

        #[inline]
        fn try_from(value: Expr) -> Result<Self> {
            value.try_cast()
        }
    }

    impl<T> TryFrom<&Expr> for TypedExpr<T>
    where
        T: ObjectRefCore + AnyCompatible + 'static,
    {
        type Error = Error;

        #[inline]
        fn try_from(value: &Expr) -> Result<Self> {
            value.clone().try_cast()
        }
    }

    /// Checked view over an expression whose result is a primitive scalar or vector type.
    pub type PrimExpr = TypedExpr<PrimType>;

    tvm_ffi::impl_try_from_any!(PrimExpr);
    tvm_ffi::impl_arg_into_ref!(PrimExpr);
    tvm_ffi::impl_into_arg_holder_default!(PrimExpr);

    /// Checked view over a `Var` whose expression type is `T`.
    ///
    /// C++ uses zero-state views such as `PrimVar` and `BufferVar` over the same
    /// `VarNode`; this type gives generated Rust bindings the same representation.
    #[repr(transparent)]
    pub struct TypedVar<T> {
        var: Var,
        _expected_type: std::marker::PhantomData<T>,
    }

    impl<T> TypedVar<T> {
        #[inline]
        pub(crate) unsafe fn from_var_unchecked(var: Var) -> Self {
            Self {
                var,
                _expected_type: std::marker::PhantomData,
            }
        }

        /// Return the general variable view over the same allocation.
        #[inline]
        pub fn as_var(&self) -> &Var {
            &self.var
        }
    }

    impl<T> TypedVar<T>
    where
        T: ObjectRefCore + AnyCompatible + 'static,
    {
        /// Return the type guaranteed by this checked variable view.
        #[inline]
        pub fn type_annotation(&self) -> T {
            self.var
                .ty
                .clone()
                .try_cast()
                .expect("TypedVar type invariant was violated")
        }
    }

    impl<T> Clone for TypedVar<T> {
        #[inline]
        fn clone(&self) -> Self {
            Self {
                var: self.var.clone(),
                _expected_type: std::marker::PhantomData,
            }
        }
    }

    impl<T> std::ops::Deref for TypedVar<T> {
        type Target = Var;

        #[inline]
        fn deref(&self) -> &Self::Target {
            &self.var
        }
    }

    unsafe impl<T: 'static> ObjectRefCore for TypedVar<T> {
        type ContainerType = VarObj;

        #[inline]
        fn data(this: &Self) -> &ObjectArc<Self::ContainerType> {
            <Var as ObjectRefCore>::data(&this.var)
        }

        #[inline]
        fn into_data(this: Self) -> ObjectArc<Self::ContainerType> {
            <Var as ObjectRefCore>::into_data(this.var)
        }

        #[inline]
        unsafe fn from_data(data: ObjectArc<Self::ContainerType>) -> Self {
            // SAFETY: The caller has guaranteed both the `Var` dynamic type and
            // this typed variable view's `ty` invariant.
            unsafe { Self::from_var_unchecked(<Var as ObjectRefCore>::from_data(data)) }
        }
    }

    unsafe impl<T> AnyCompatible for TypedVar<T>
    where
        T: ObjectRefCore + AnyCompatible + 'static,
    {
        #[inline]
        unsafe fn copy_to_any_view(src: &Self, data: &mut TVMFFIAny) {
            Var::copy_to_any_view(&src.var, data);
        }

        #[inline]
        unsafe fn move_to_any(src: Self, data: &mut TVMFFIAny) {
            Var::move_to_any(src.var, data);
        }

        #[inline]
        unsafe fn check_any_strict(data: &TVMFFIAny) -> bool {
            Var::check_any_strict(data) && TypedExpr::<T>::check_any_strict(data)
        }

        #[inline]
        unsafe fn copy_from_any_view_after_check(data: &TVMFFIAny) -> Self {
            Self::from_var_unchecked(Var::copy_from_any_view_after_check(data))
        }

        #[inline]
        unsafe fn move_from_any_after_check(data: &mut TVMFFIAny) -> Self {
            Self::from_var_unchecked(Var::move_from_any_after_check(data))
        }

        #[inline]
        unsafe fn try_cast_from_any_view(data: &TVMFFIAny) -> std::result::Result<Self, ()> {
            if Self::check_any_strict(data) {
                Ok(Self::copy_from_any_view_after_check(data))
            } else {
                Err(())
            }
        }

        fn type_str() -> std::string::String {
            format!("TypedVar<{}>", T::type_str())
        }
    }

    impl<T> From<TypedVar<T>> for Var {
        #[inline]
        fn from(value: TypedVar<T>) -> Self {
            value.var
        }
    }

    impl<T> From<&TypedVar<T>> for Var {
        #[inline]
        fn from(value: &TypedVar<T>) -> Self {
            value.var.clone()
        }
    }

    impl<T> From<&TypedVar<T>> for TypedVar<T> {
        #[inline]
        fn from(value: &TypedVar<T>) -> Self {
            value.clone()
        }
    }

    impl<T> From<TypedVar<T>> for Expr {
        #[inline]
        fn from(value: TypedVar<T>) -> Self {
            value.var.into()
        }
    }

    impl<T> From<&TypedVar<T>> for Expr {
        #[inline]
        fn from(value: &TypedVar<T>) -> Self {
            value.var.clone().into()
        }
    }

    impl<T> From<TypedVar<T>> for TypedExpr<T>
    where
        T: ObjectRefCore + AnyCompatible + 'static,
    {
        #[inline]
        fn from(value: TypedVar<T>) -> Self {
            unsafe { TypedExpr::from_expr_unchecked(value.var.into()) }
        }
    }

    impl<T> From<&TypedVar<T>> for TypedExpr<T>
    where
        T: ObjectRefCore + AnyCompatible + 'static,
    {
        #[inline]
        fn from(value: &TypedVar<T>) -> Self {
            value.clone().into()
        }
    }

    impl<T> TryFrom<Var> for TypedVar<T>
    where
        T: ObjectRefCore + AnyCompatible + 'static,
    {
        type Error = Error;

        #[inline]
        fn try_from(value: Var) -> Result<Self> {
            value.try_cast()
        }
    }

    impl<T> TryFrom<&Var> for TypedVar<T>
    where
        T: ObjectRefCore + AnyCompatible + 'static,
    {
        type Error = Error;

        #[inline]
        fn try_from(value: &Var) -> Result<Self> {
            value.clone().try_cast()
        }
    }

    impl<T> TryFrom<Expr> for TypedVar<T>
    where
        T: ObjectRefCore + AnyCompatible + 'static,
    {
        type Error = Error;

        #[inline]
        fn try_from(value: Expr) -> Result<Self> {
            value.try_cast()
        }
    }

    impl<T> TryFrom<&Expr> for TypedVar<T>
    where
        T: ObjectRefCore + AnyCompatible + 'static,
    {
        type Error = Error;

        #[inline]
        fn try_from(value: &Expr) -> Result<Self> {
            value.clone().try_cast()
        }
    }

    impl SourceName {
        /// Return the interned native source name for `name`.
        pub fn get(name: &str) -> Result<Self> {
            tvm_ffi::cached_global_func!("ir.SourceName")
                .call_tuple((String::from(name),))?
                .try_into()
        }
    }

    impl Source {
        fn field<T>(&self, name: &str) -> Result<T>
        where
            T: TryFrom<Any, Error = Error>,
        {
            FieldGetter::new(SourceObj::type_index(), name)?.get(&**self)
        }

        /// Return the interned name associated with this native source object.
        pub fn source_name(&self) -> Result<SourceName> {
            self.field("source_name")
        }

        /// Return the native source text.
        pub fn text(&self) -> Result<String> {
            self.field("source")
        }
    }

    impl SourceMap {
        /// Construct an empty source map directly in Rust.
        pub fn new() -> Self {
            Self::from_map(Map::new())
        }

        /// Construct a source map directly from its complete backing map.
        pub fn from_map(source_map: Map<SourceName, Source>) -> Self {
            Self::from_complete_fields(source_map)
        }

        /// Ask TVM to construct and add a native source fragment to this map.
        pub fn add(&mut self, name: &str, content: &str) -> Result<SourceName> {
            let name = String::from(name);
            let content = String::from(content);
            tvm_ffi::cached_global_func!("SourceMapAdd")
                .call_packed(&[
                    AnyView::from(&*self),
                    AnyView::from(&name),
                    AnyView::from(&content),
                ])?
                .try_into()
        }
    }

    impl Span {
        /// Construct source-location metadata in TVM's `(line, end_line, column, end_column)` order.
        pub fn new<S>(
            source_name: S,
            line: i64,
            end_line: i64,
            column: i64,
            end_column: i64,
        ) -> Result<Self>
        where
            S: Into<SourceName>,
        {
            let line = i32::try_from(line).map_err(|_| integer_field_overflow("line", line))?;
            let end_line = i32::try_from(end_line)
                .map_err(|_| integer_field_overflow("end_line", end_line))?;
            let column =
                i32::try_from(column).map_err(|_| integer_field_overflow("column", column))?;
            let end_column = i32::try_from(end_column)
                .map_err(|_| integer_field_overflow("end_column", end_column))?;
            Ok(Self::from_complete_fields(
                Some(source_name.into()),
                line,
                column,
                end_line,
                end_column,
            ))
        }
    }

    impl SequentialSpan {
        /// Construct a sequential span, flattening nested sequential spans like TVM.
        pub fn new(spans: Vec<Span>) -> Self {
            let mut flattened = Vec::new();
            for span in spans {
                if let Some(sequence) = span.as_node::<SequentialSpanObj>() {
                    flattened.extend(sequence.spans.iter());
                } else {
                    flattened.push(span);
                }
            }
            Self::from_complete_fields(None, 0, 0, 0, 0, Array::new(flattened))
        }
    }

    fn integer_field_overflow(field: &str, value: i64) -> Error {
        Error::new(
            VALUE_ERROR,
            &format!("{field} value {value} does not fit TVM's 32-bit integer field"),
            "",
        )
    }

    impl PrimExprConvertible {
        /// Invoke TVM's standard FFI fallback conversion to a primitive expression.
        pub fn to_prim_expr(&self) -> Result<Expr> {
            tvm_ffi::cached_global_func!("tirx.convert")
                .call_tuple((self,))?
                .try_into()
        }
    }

    impl Range {
        /// Construct a range from its minimum and extent.
        pub fn from_min_extent<M, E>(minimum: M, extent: E) -> Result<Self>
        where
            M: Into<Expr>,
            E: Into<Expr>,
        {
            Self::from_min_extent_with_span(minimum, extent, None)
        }

        /// Construct a range from all of its physical fields.
        pub fn from_min_extent_with_span<M, E>(
            minimum: M,
            extent: E,
            span: Option<&Span>,
        ) -> Result<Self>
        where
            M: Into<Expr>,
            E: Into<Expr>,
        {
            let minimum = minimum.into();
            let extent = extent.into();
            let minimum = require_primitive_expr(minimum, "Range minimum")?;
            let extent = require_primitive_expr(extent, "Range extent")?;
            Ok(Self::from_complete_fields(minimum, extent, span.cloned()))
        }

        /// Copy this node with new `min`, `extent`; every other field, span
        /// included, is carried over from `self`.
        ///
        /// Takes the same required fields as [`Range::from_min_extent`] and, like
        /// [`Range::from_complete_fields`], runs no validation.
        pub fn copy_with(&self, min: PrimExpr, extent: PrimExpr) -> Self {
            Self::from_complete_fields(min, extent, self.span.clone())
        }
    }

    fn require_primitive_expr(value: Expr, context: &str) -> Result<PrimExpr> {
        PrimExpr::try_from(value).map_err(|_| {
            Error::new(
                TYPE_ERROR,
                &format!("{context} must have a primitive type"),
                "",
            )
        })
    }

    impl Tuple {
        /// Construct a tuple and derive its type when every field has a known type.
        pub fn new(fields: Vec<Expr>) -> Self {
            Self::with_span(fields, None)
        }

        /// Construct a tuple with optional source metadata.
        pub fn with_span(fields: Vec<Expr>, span: Option<&Span>) -> Self {
            let ty = fields
                .iter()
                .map(|field| (!field.ty.is_missing()).then(|| field.ty.clone()))
                .collect::<Option<Vec<_>>>()
                .map(|fields| Type::from(TupleType::new(fields)))
                .unwrap_or_else(Type::missing);
            Self::from_complete_fields(span.cloned(), ty, Array::new(fields))
        }

        /// Copy this tuple with new fields and preserve its span.
        pub fn copy_with(&self, fields: Array<Expr>) -> Self {
            Self::with_span(fields.iter().collect(), self.span.as_ref())
        }
    }

    impl TupleGetItem {
        /// Construct a tuple field projection after validating its index.
        pub fn new<T>(tuple: T, index: i32) -> Result<Self>
        where
            T: Into<Expr>,
        {
            Self::with_span(tuple, index, None)
        }

        /// Construct a tuple field projection with optional source metadata.
        pub fn with_span<T>(tuple: T, index: i32, span: Option<&Span>) -> Result<Self>
        where
            T: Into<Expr>,
        {
            if index < 0 {
                return Err(Error::new(
                    INDEX_ERROR,
                    "tuple index cannot be negative",
                    "",
                ));
            }
            let tuple = tuple.into();
            let ty = if let Some(tuple_type) = tuple.ty.as_node::<TupleTypeObj>() {
                tuple_type.fields.get(index as usize).map_err(|_| {
                    Error::new(
                        INDEX_ERROR,
                        &format!(
                            "tuple of length {} cannot be accessed at index {index}",
                            tuple_type.fields.len()
                        ),
                        "",
                    )
                })?
            } else {
                Type::missing()
            };
            Ok(Self::from_complete_fields(span.cloned(), ty, tuple, index))
        }

        /// Copy this projection with a new tuple and preserve its index and span.
        pub fn copy_with(&self, tuple: Expr) -> Result<Self> {
            Self::with_span(tuple, self.index, self.span.as_ref())
        }
    }

    impl TensorLoad {
        /// Copy this node with new `source`, `indices`; every other field, span
        /// included, is carried over from `self`.
        ///
        /// Takes the same required fields as [`TensorLoad::from_buffer`] and, like
        /// [`TensorLoad::from_complete_fields`], runs no validation.
        /// The result type is carried over unchanged.
        pub fn copy_with(&self, source: Expr, indices: Array<PrimExpr>) -> Self {
            Self::from_complete_fields(
                self.span.clone(),
                PrimExpr::from(self).type_annotation(),
                source,
                indices,
            )
        }
    }

    impl PointerType {
        /// Construct a pointer type directly in Rust.
        pub fn new<T>(element_type: T, storage_scope: &str) -> Result<Self>
        where
            T: Into<Type>,
        {
            let element_type = element_type.into();
            if element_type.is_missing() {
                return Err(Error::new(
                    TYPE_ERROR,
                    "PointerType element_type cannot be Type::Missing()",
                    "",
                ));
            }
            let storage_scope = if storage_scope.is_empty() {
                "global"
            } else {
                storage_scope
            };
            Ok(Self::from_complete_fields(
                None,
                element_type,
                String::from(storage_scope),
            ))
        }

        /// Borrow the type stored at this pointer.
        pub fn element_type(&self) -> &Type {
            &self.element_type
        }

        /// Borrow the pointer's storage scope.
        pub fn storage_scope(&self) -> &str {
            self.storage_scope.as_str()
        }
    }

    impl TypedExpr<PrimType> {
        /// Return this primitive expression's dtype without cloning its type handle.
        #[inline]
        pub fn dtype(&self) -> DLDataType {
            self.expr
                .ty
                .as_node::<PrimTypeObj>()
                .expect("PrimExpr type invariant was violated")
                .dtype
        }
    }

    impl TypedVar<PrimType> {
        /// Return this primitive variable's dtype without cloning its type handle.
        #[inline]
        pub fn dtype(&self) -> DLDataType {
            self.var
                .ty
                .as_node::<PrimTypeObj>()
                .expect("PrimVar type invariant was violated")
                .dtype
        }
    }

    impl TupleType {
        /// Construct a tuple type directly in Rust.
        pub fn new(fields: Vec<Type>) -> Self {
            Self::with_span(fields, None)
        }

        /// Construct a tuple type with optional source metadata.
        pub fn with_span(fields: Vec<Type>, span: Option<&Span>) -> Self {
            Self::from_complete_fields(span.cloned(), Array::new(fields))
        }

        /// Construct the empty tuple type used as TVM's void type.
        pub fn empty() -> Self {
            Self::new(Vec::new())
        }
    }

    impl FuncType {
        /// Construct a function type directly in Rust.
        pub fn new(arg_types: Vec<Type>, ret_type: Type) -> Self {
            Self::with_span(arg_types, ret_type, None)
        }

        /// Construct a function type with optional source metadata.
        pub fn with_span(arg_types: Vec<Type>, ret_type: Type, span: Option<&Span>) -> Self {
            Self::from_complete_fields(span.cloned(), Array::new(arg_types), ret_type)
        }
    }

    impl TensorMapType {
        /// Construct a tensor-map marker type directly in Rust.
        pub fn new() -> Self {
            Self::with_span(None)
        }

        /// Construct a tensor-map marker type with optional source metadata.
        pub fn with_span(span: Option<&Span>) -> Self {
            Self::from_complete_fields(span.cloned())
        }
    }

    impl Default for TensorMapType {
        fn default() -> Self {
            Self::new()
        }
    }

    impl DummyGlobalInfo {
        /// Construct TVM's fieldless global-info test value through its native
        /// constructor: `ir.GlobalInfo` registers no layout of its own, so the
        /// generated binding keeps this node opaque.
        pub fn new() -> Self {
            tvm_ffi::cached_global_func!("ir.DummyGlobalInfo")
                .call_tuple(())
                .expect("native DummyGlobalInfo constructor failed")
                .try_into()
                .expect("native DummyGlobalInfo constructor returned the wrong type")
        }
    }

    impl Default for DummyGlobalInfo {
        fn default() -> Self {
            Self::new()
        }
    }

    impl IntImm {
        /// Construct an integer literal directly in Rust.
        pub fn new(dtype: &str, value: i64) -> Result<Self> {
            Self::from_dtype(DLDataType::try_from_str(dtype)?, value)
        }

        /// Construct an integer literal from a parsed DLPack dtype.
        pub fn from_dtype(dtype: DLDataType, value: i64) -> Result<Self> {
            Self::from_dtype_with_span(dtype, value, None)
        }

        /// Construct an integer literal directly in Rust with source metadata.
        pub fn from_dtype_with_span(
            dtype: DLDataType,
            value: i64,
            span: Option<&Span>,
        ) -> Result<Self> {
            validate_integer_literal(dtype, value)?;
            let value_type = PrimType::from_dtype(dtype)?;
            Ok(Self::from_complete_fields(span.cloned(), value_type, value))
        }
    }

    impl FloatImm {
        /// Construct a scalar floating-point literal directly in Rust.
        pub fn new(dtype: &str, value: f64) -> Result<Self> {
            Self::from_dtype(DLDataType::try_from_str(dtype)?, value)
        }

        /// Construct a scalar floating-point literal from a parsed DLPack dtype.
        pub fn from_dtype(dtype: DLDataType, value: f64) -> Result<Self> {
            Self::from_dtype_with_span(dtype, value, None)
        }

        /// Construct a scalar floating-point literal with source metadata.
        pub fn from_dtype_with_span(
            dtype: DLDataType,
            value: f64,
            span: Option<&Span>,
        ) -> Result<Self> {
            if dtype.lanes != 1 {
                return Err(Error::new(
                    VALUE_ERROR,
                    "FloatImm can only represent a scalar value",
                    "",
                ));
            }
            let is_floating = matches!(
                dtype.code,
                x if x == DLDataTypeCode::kDLFloat as u8
                    || x == DLDataTypeCode::kDLBfloat as u8
                    || x == DLDataTypeCode::kDLFloat8_e3m4 as u8
                    || x == DLDataTypeCode::kDLFloat8_e4m3 as u8
                    || x == DLDataTypeCode::kDLFloat8_e4m3b11fnuz as u8
                    || x == DLDataTypeCode::kDLFloat8_e4m3fn as u8
                    || x == DLDataTypeCode::kDLFloat8_e4m3fnuz as u8
                    || x == DLDataTypeCode::kDLFloat8_e5m2 as u8
                    || x == DLDataTypeCode::kDLFloat8_e5m2fnuz as u8
                    || x == DLDataTypeCode::kDLFloat8_e8m0fnu as u8
                    || x == DLDataTypeCode::kDLFloat6_e2m3fn as u8
                    || x == DLDataTypeCode::kDLFloat6_e3m2fn as u8
                    || x == DLDataTypeCode::kDLFloat4_e2m1fn as u8
                    || x >= 129
            );
            if !is_floating {
                let dtype_name = dtype.to_string();
                return Err(Error::new(
                    VALUE_ERROR,
                    &format!("FloatImm requires a floating-point dtype, but received {dtype_name}"),
                    "",
                ));
            }
            Ok(Self::from_complete_fields(
                span.cloned(),
                PrimType::from_dtype(dtype)?,
                value,
            ))
        }
    }

    fn validate_integer_literal(dtype: DLDataType, value: i64) -> Result<()> {
        let dtype_text = dtype.to_string();
        if dtype.lanes != 1 {
            return Err(Error::new(
                VALUE_ERROR,
                &format!("IntImm can only take a scalar, but {dtype_text} was supplied"),
                "",
            ));
        }
        let is_int = dtype.code == DLDataTypeCode::kDLInt as u8;
        let is_uint = dtype.code == DLDataTypeCode::kDLUInt as u8;
        let is_bool = dtype.code == DLDataTypeCode::kDLBool as u8;
        if !is_int && !is_uint && !is_bool {
            return Err(Error::new(
                VALUE_ERROR,
                &format!("IntImm supports only int, uint, or bool, but {dtype_text} was supplied"),
                "",
            ));
        }
        let bits = u32::from(dtype.bits);
        if bits == 0 {
            return Err(Error::new(
                VALUE_ERROR,
                &format!("invalid integer bit width in {dtype_text}"),
                "",
            ));
        }
        let in_range = if is_uint {
            value >= 0 && (bits >= 64 || (value as u64) < (1_u64 << bits))
        } else if is_bool || bits == 1 {
            value == 0 || value == 1
        } else if bits >= 64 {
            true
        } else {
            let bound = 1_i64 << (bits - 1);
            value >= -bound && value < bound
        };
        if in_range {
            Ok(())
        } else {
            Err(Error::new(
                VALUE_ERROR,
                &format!("literal value {value} is outside the range of {dtype_text}"),
                "",
            ))
        }
    }

    impl PrimType {
        /// Construct a primitive type from a DLPack dtype string.
        pub fn new(dtype: &str) -> Result<Self> {
            Self::from_dtype(DLDataType::try_from_str(dtype)?)
        }

        /// Construct a primitive type from a parsed DLPack dtype directly in Rust.
        pub fn from_dtype(dtype: DLDataType) -> Result<Self> {
            Self::from_dtype_with_span(dtype, None)
        }

        /// Construct a primitive type from all of its physical fields.
        pub fn from_dtype_with_span(dtype: DLDataType, span: Option<&Span>) -> Result<Self> {
            let is_opaque_handle = dtype.code == DLDataTypeCode::kDLOpaqueHandle as u8;
            let is_void = is_opaque_handle && dtype.bits == 0 && dtype.lanes == 0;
            if is_opaque_handle && !is_void {
                return Err(Error::new(
                    TYPE_ERROR,
                    "PrimType cannot represent an opaque pointer; use a pointer type",
                    "",
                ));
            }
            Ok(Self::from_complete_fields(span.cloned(), dtype))
        }

        /// Construct TVM's void primitive type without parsing a user-supplied dtype string.
        pub fn void() -> Self {
            Self::from_complete_fields(
                None,
                DLDataType {
                    code: DLDataTypeCode::kDLOpaqueHandle as u8,
                    bits: 0,
                    lanes: 0,
                },
            )
        }
    }

    impl Type {
        /// Construct TVM's language-independent sentinel for an unavailable static type.
        pub fn missing() -> Self {
            tvm_ffi::cached_global_func!("ir.TypeMissing")
                .call_tuple(())
                .expect("native missing-type constructor failed")
                .try_into()
                .expect("native missing-type constructor returned the wrong type")
        }

        /// Return whether this value is the exact `ir.Type` missing-type sentinel.
        pub fn is_missing(&self) -> bool {
            AnyView::from(self).type_index() == <TypeObj as tvm_ffi::ObjectCore>::type_index()
        }
    }

    impl Var {
        /// Construct a variable directly in Rust with an explicit primitive type.
        pub fn new(name: &str, dtype: &str) -> Result<Self> {
            Ok(Self::with_type(name, PrimType::new(dtype)?))
        }

        /// Construct a variable with an arbitrary TVM type annotation.
        pub fn with_type<T>(name: &str, ty: T) -> Self
        where
            T: Into<Type>,
        {
            Self::with_type_and_span(name, ty, None)
        }

        /// Construct a variable directly in Rust with its complete base fields.
        pub fn with_type_and_span<T>(name: &str, ty: T, span: Option<&Span>) -> Self
        where
            T: Into<Type>,
        {
            Self::with_optional_type_and_span(name, Some(ty.into()), span)
        }

        /// Construct a variable with the native constructor's optional type annotation.
        pub fn with_optional_type_and_span(
            name: &str,
            ty: Option<Type>,
            span: Option<&Span>,
        ) -> Self {
            Self::from_complete_fields(
                span.cloned(),
                ty.unwrap_or_else(Type::missing),
                String::from(name),
            )
        }

        /// Copy this node with new `name`, `ty`; every other field, span
        /// included, is carried over from `self`.
        ///
        /// Takes the same required fields as [`Var::with_type`] and, like
        /// [`Var::from_complete_fields`], runs no validation.
        pub fn copy_with(&self, name: String, ty: Type) -> Self {
            Self::from_complete_fields(self.span.clone(), ty, name)
        }
    }

    impl GlobalVar {
        /// Construct a module-level symbol directly in Rust.
        pub fn new(name_hint: &str) -> Self {
            Self::with_span(name_hint, None)
        }

        /// Construct a module-level symbol with optional source metadata.
        pub fn with_span(name_hint: &str, span: Option<&Span>) -> Self {
            Self::from_complete_fields(span.cloned(), Type::missing(), String::from(name_hint))
        }
    }

    impl Call {
        /// Construct a call directly in Rust with no attributes or explicit type arguments.
        pub fn new<T, O>(ret_type: T, operator: O, arguments: Vec<Expr>) -> Self
        where
            T: Into<Type>,
            O: Into<Expr>,
        {
            Self::with_metadata(ret_type, operator, arguments, None, Vec::new(), None)
        }

        /// Construct a call with all reflected metadata supplied explicitly.
        pub fn with_metadata<T, O>(
            ret_type: T,
            operator: O,
            arguments: Vec<Expr>,
            attrs: Option<Attrs>,
            type_arguments: Vec<Type>,
            span: Option<&Span>,
        ) -> Self
        where
            T: Into<Type>,
            O: Into<Expr>,
        {
            Self::from_complete_fields(
                span.cloned(),
                ret_type.into(),
                operator.into(),
                Array::new(arguments),
                attrs,
                Array::new(type_arguments),
            )
        }

        /// Copy this node with new `ty`, `op`, `args`; every other field, span
        /// included, is carried over from `self`.
        ///
        /// Takes the same required fields as [`Call::new`] and, like
        /// [`Call::from_complete_fields`], runs no validation.
        pub fn copy_with(&self, ty: Type, op: Expr, args: Array<Expr>) -> Self {
            Self::from_complete_fields(
                self.span.clone(),
                ty,
                op,
                args,
                self.attrs.clone(),
                self.ty_args.clone(),
            )
        }
    }

    impl IRModule {
        /// Wrap a function expression in an IRModule whose entry is `main`.
        pub fn from_expr<E>(expr: E) -> Result<Self>
        where
            E: Into<Expr>,
        {
            let function = expr.into().try_cast::<BaseFunc>()?;
            let global_symbol = function
                .attrs
                .dict
                .get(&String::from("global_symbol"))?
                .map(String::try_from)
                .transpose()?;
            let global_name = global_symbol
                .as_deref()
                .filter(|name| !name.is_empty())
                .unwrap_or("main");
            let global_var = GlobalVar::new(global_name);
            Self::new([(global_var, function)].into_iter().collect())
        }

        // customized_new(IRModule) begin
        /// Construct a module that holds only `functions`.
        ///
        /// The source map, attributes, and global infos take the C++ constructor
        /// defaults (all empty); use [`IRModule::with_metadata`] to supply them.
        pub fn new(functions: Map<GlobalVar, BaseFunc>) -> Result<Self> {
            Self::with_metadata(functions, SourceMap::new(), DictAttrs::empty(), Map::new())
        }
        // customized_new(IRModule) end

        /// Construct a module directly in Rust from all of its stored state.
        ///
        /// The derived name-to-global-variable index is rebuilt and checked here;
        /// callers never supply it independently.
        pub fn with_metadata(
            functions: Map<GlobalVar, BaseFunc>,
            source_map: SourceMap,
            attrs: DictAttrs,
            global_infos: Map<String, Array<GlobalInfo>>,
        ) -> Result<Self> {
            let mut indexed_globals = Vec::with_capacity(functions.len());
            let mut names = std::collections::HashSet::with_capacity(functions.len());
            for (global_var, _) in functions.iter() {
                let name = global_var.name_hint.clone();
                if !names.insert(name.as_str().to_owned()) {
                    return Err(Error::new(
                        VALUE_ERROR,
                        &format!("duplicate global function name {}", name.as_str()),
                        "",
                    ));
                }
                indexed_globals.push((name, global_var));
            }
            Ok(Self::from_complete_fields(
                functions,
                source_map,
                attrs,
                global_infos,
                indexed_globals.into_iter().collect(),
            ))
        }

        /// Return an independently updatable module with one function replaced.
        ///
        /// Rust rebuilds both immutable maps and the module node, so other handles
        /// that share `self` remain unchanged and the derived global-name index
        /// cannot become stale.
        pub fn with_updated_function(
            &self,
            global_var: &GlobalVar,
            function: &BaseFunc,
        ) -> Result<Self> {
            self.copy_for_update()?
                .update_function_owned(global_var, function)
        }

        /// Return an independently updatable module with one global-info group set.
        pub fn with_updated_global_info(
            &self,
            name: &str,
            global_info: Vec<GlobalInfo>,
        ) -> Result<Self> {
            let name = String::from(name);
            let mut global_infos = self
                .global_infos
                .iter()
                .filter(|(existing, _)| existing.as_str() != name.as_str())
                .collect::<Vec<_>>();
            global_infos.push((name, Array::new(global_info)));
            Self::with_metadata(
                self.functions.clone(),
                self.source_map.clone(),
                self.attrs.clone(),
                global_infos.into_iter().collect(),
            )
        }

        pub(crate) fn copy_for_update(&self) -> Result<Self> {
            Self::with_metadata(
                self.functions.clone(),
                self.source_map.clone(),
                self.attrs.clone(),
                self.global_infos.clone(),
            )
        }

        pub(crate) fn update_function_owned(
            self,
            global_var: &GlobalVar,
            function: &BaseFunc,
        ) -> Result<Self> {
            let name = global_var.name_hint.clone();
            let mut functions = Vec::with_capacity(self.functions.len() + 1);
            for (existing_var, existing_function) in self.functions.iter() {
                if existing_var.name_hint.as_str() == name.as_str() {
                    if !existing_var.same_as(global_var) {
                        return Err(Error::new(
                            VALUE_ERROR,
                            &format!("duplicate global function name {}", name.as_str()),
                            "",
                        ));
                    }
                } else {
                    functions.push((existing_var, existing_function));
                }
            }
            functions.push((global_var.clone(), function.clone()));
            Self::with_metadata(
                functions.into_iter().collect(),
                self.source_map.clone(),
                self.attrs.clone(),
                self.global_infos.clone(),
            )
        }
    }

    impl DictAttrs {
        /// Construct a defined, empty DictAttrs object.
        pub fn empty() -> Self {
            Self::from_dictionary(Map::new())
        }

        /// Construct DictAttrs from a heterogeneous string-to-value map.
        pub fn from_dictionary(dictionary: Map<String, Any>) -> Self {
            Self::from_complete_fields(dictionary)
        }
    }

    impl Default for SourceMap {
        fn default() -> Self {
            Self::new()
        }
    }
}
