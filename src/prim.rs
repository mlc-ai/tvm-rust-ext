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

//! Primitive scalar and vector expression nodes (`prim.*`).
//!
//! The blocks are emitted by `tvm-ffi-stubgen --target rust`; the reviewed
//! semantic constructors follow in `mod semantic`.

pub(crate) use semantic::primitive_type;

// Every object registered under `ir.prim` gets its block in this file; `skip` leaves one out.
// tvm-ffi-stubgen(prefix): prim
// Hand-maintained directives; tvm-ffi-stubgen applies them on every run.
// tvm-ffi-stubgen(import-object): crate::ir::PrimExpr
// tvm-ffi-stubgen(import-object): crate::ir::PrimType
// tvm-ffi-stubgen(nullable): ir.Expr.span
// tvm-ffi-stubgen(field): prim.Add.a -> PrimExpr
// tvm-ffi-stubgen(field): prim.Add.b -> PrimExpr
// tvm-ffi-stubgen(field): prim.Sub.a -> PrimExpr
// tvm-ffi-stubgen(field): prim.Sub.b -> PrimExpr
// tvm-ffi-stubgen(field): prim.Mul.a -> PrimExpr
// tvm-ffi-stubgen(field): prim.Mul.b -> PrimExpr
// tvm-ffi-stubgen(field): prim.Div.a -> PrimExpr
// tvm-ffi-stubgen(field): prim.Div.b -> PrimExpr
// tvm-ffi-stubgen(field): prim.Mod.a -> PrimExpr
// tvm-ffi-stubgen(field): prim.Mod.b -> PrimExpr
// tvm-ffi-stubgen(field): prim.FloorDiv.a -> PrimExpr
// tvm-ffi-stubgen(field): prim.FloorDiv.b -> PrimExpr
// tvm-ffi-stubgen(field): prim.FloorMod.a -> PrimExpr
// tvm-ffi-stubgen(field): prim.FloorMod.b -> PrimExpr
// tvm-ffi-stubgen(field): prim.Min.a -> PrimExpr
// tvm-ffi-stubgen(field): prim.Min.b -> PrimExpr
// tvm-ffi-stubgen(field): prim.Max.a -> PrimExpr
// tvm-ffi-stubgen(field): prim.Max.b -> PrimExpr
// tvm-ffi-stubgen(field): prim.EQ.a -> PrimExpr
// tvm-ffi-stubgen(field): prim.EQ.b -> PrimExpr
// tvm-ffi-stubgen(field): prim.NE.a -> PrimExpr
// tvm-ffi-stubgen(field): prim.NE.b -> PrimExpr
// tvm-ffi-stubgen(field): prim.LT.a -> PrimExpr
// tvm-ffi-stubgen(field): prim.LT.b -> PrimExpr
// tvm-ffi-stubgen(field): prim.LE.a -> PrimExpr
// tvm-ffi-stubgen(field): prim.LE.b -> PrimExpr
// tvm-ffi-stubgen(field): prim.GT.a -> PrimExpr
// tvm-ffi-stubgen(field): prim.GT.b -> PrimExpr
// tvm-ffi-stubgen(field): prim.GE.a -> PrimExpr
// tvm-ffi-stubgen(field): prim.GE.b -> PrimExpr
// tvm-ffi-stubgen(field): prim.And.a -> PrimExpr
// tvm-ffi-stubgen(field): prim.And.b -> PrimExpr
// tvm-ffi-stubgen(field): prim.Or.a -> PrimExpr
// tvm-ffi-stubgen(field): prim.Or.b -> PrimExpr
// tvm-ffi-stubgen(field): prim.Not.a -> PrimExpr
// tvm-ffi-stubgen(field): prim.Cast.value -> PrimExpr
// tvm-ffi-stubgen(field): prim.Ramp.base -> PrimExpr
// tvm-ffi-stubgen(field): prim.Ramp.stride -> PrimExpr
// tvm-ffi-stubgen(field): prim.Ramp.lanes -> PrimExpr
// tvm-ffi-stubgen(field): prim.Broadcast.value -> PrimExpr
// tvm-ffi-stubgen(field): prim.Broadcast.lanes -> PrimExpr
// tvm-ffi-stubgen(field): prim.Shuffle.vectors -> Array<PrimExpr>
// tvm-ffi-stubgen(field): prim.Shuffle.indices -> Array<PrimExpr>
// tvm-ffi-stubgen(field): prim.Select.condition -> PrimExpr
// tvm-ffi-stubgen(field): prim.Select.true_value -> PrimExpr
// tvm-ffi-stubgen(field): prim.Select.false_value -> PrimExpr
// tvm-ffi-stubgen(field): prim.Let.value -> PrimExpr
// tvm-ffi-stubgen(field): prim.Let.body -> PrimExpr
// tvm-ffi-stubgen(field): prim.Add.ty -> PrimType
// tvm-ffi-stubgen(field): prim.Sub.ty -> PrimType
// tvm-ffi-stubgen(field): prim.Mul.ty -> PrimType
// tvm-ffi-stubgen(field): prim.Div.ty -> PrimType
// tvm-ffi-stubgen(field): prim.Mod.ty -> PrimType
// tvm-ffi-stubgen(field): prim.FloorDiv.ty -> PrimType
// tvm-ffi-stubgen(field): prim.FloorMod.ty -> PrimType
// tvm-ffi-stubgen(field): prim.Min.ty -> PrimType
// tvm-ffi-stubgen(field): prim.Max.ty -> PrimType
// tvm-ffi-stubgen(field): prim.EQ.ty -> PrimType
// tvm-ffi-stubgen(field): prim.NE.ty -> PrimType
// tvm-ffi-stubgen(field): prim.LT.ty -> PrimType
// tvm-ffi-stubgen(field): prim.LE.ty -> PrimType
// tvm-ffi-stubgen(field): prim.GT.ty -> PrimType
// tvm-ffi-stubgen(field): prim.GE.ty -> PrimType
// tvm-ffi-stubgen(field): prim.And.ty -> PrimType
// tvm-ffi-stubgen(field): prim.Or.ty -> PrimType
// tvm-ffi-stubgen(field): prim.Not.ty -> PrimType
// tvm-ffi-stubgen(field): prim.Cast.ty -> PrimType
// tvm-ffi-stubgen(field): prim.Ramp.ty -> PrimType
// tvm-ffi-stubgen(field): prim.Broadcast.ty -> PrimType
// tvm-ffi-stubgen(field): prim.Shuffle.ty -> PrimType
// tvm-ffi-stubgen(field): prim.Select.ty -> PrimType
// tvm-ffi-stubgen(field): prim.Let.ty -> PrimType
// tvm-ffi-stubgen(upcast): prim.Add -> PrimExpr
// tvm-ffi-stubgen(upcast): prim.Sub -> PrimExpr
// tvm-ffi-stubgen(upcast): prim.Mul -> PrimExpr
// tvm-ffi-stubgen(upcast): prim.Div -> PrimExpr
// tvm-ffi-stubgen(upcast): prim.Mod -> PrimExpr
// tvm-ffi-stubgen(upcast): prim.FloorDiv -> PrimExpr
// tvm-ffi-stubgen(upcast): prim.FloorMod -> PrimExpr
// tvm-ffi-stubgen(upcast): prim.Min -> PrimExpr
// tvm-ffi-stubgen(upcast): prim.Max -> PrimExpr
// tvm-ffi-stubgen(upcast): prim.EQ -> PrimExpr
// tvm-ffi-stubgen(upcast): prim.NE -> PrimExpr
// tvm-ffi-stubgen(upcast): prim.LT -> PrimExpr
// tvm-ffi-stubgen(upcast): prim.LE -> PrimExpr
// tvm-ffi-stubgen(upcast): prim.GT -> PrimExpr
// tvm-ffi-stubgen(upcast): prim.GE -> PrimExpr
// tvm-ffi-stubgen(upcast): prim.And -> PrimExpr
// tvm-ffi-stubgen(upcast): prim.Or -> PrimExpr
// tvm-ffi-stubgen(upcast): prim.Not -> PrimExpr
// tvm-ffi-stubgen(upcast): prim.Cast -> PrimExpr
// tvm-ffi-stubgen(upcast): prim.Ramp -> PrimExpr
// tvm-ffi-stubgen(upcast): prim.Broadcast -> PrimExpr
// tvm-ffi-stubgen(upcast): prim.Shuffle -> PrimExpr
// tvm-ffi-stubgen(upcast): prim.Select -> PrimExpr
// tvm-ffi-stubgen(upcast): prim.Let -> PrimExpr
// tvm-ffi-stubgen(custom-new): prim.Add
// tvm-ffi-stubgen(custom-new): prim.Sub
// tvm-ffi-stubgen(custom-new): prim.Mul
// tvm-ffi-stubgen(custom-new): prim.Div
// tvm-ffi-stubgen(custom-new): prim.Mod
// tvm-ffi-stubgen(custom-new): prim.FloorDiv
// tvm-ffi-stubgen(custom-new): prim.FloorMod
// tvm-ffi-stubgen(custom-new): prim.Min
// tvm-ffi-stubgen(custom-new): prim.Max
// tvm-ffi-stubgen(custom-new): prim.EQ
// tvm-ffi-stubgen(custom-new): prim.NE
// tvm-ffi-stubgen(custom-new): prim.LT
// tvm-ffi-stubgen(custom-new): prim.LE
// tvm-ffi-stubgen(custom-new): prim.GT
// tvm-ffi-stubgen(custom-new): prim.GE
// tvm-ffi-stubgen(custom-new): prim.And
// tvm-ffi-stubgen(custom-new): prim.Or
// tvm-ffi-stubgen(custom-new): prim.Not
// tvm-ffi-stubgen(custom-new): prim.Cast
// tvm-ffi-stubgen(custom-new): prim.Ramp
// tvm-ffi-stubgen(custom-new): prim.Broadcast
// tvm-ffi-stubgen(custom-new): prim.Shuffle
// tvm-ffi-stubgen(custom-new): prim.Select
// tvm-ffi-stubgen(custom-new): prim.Let
// tvm-ffi-stubgen(import-object): tvm_ffi::Array

// tvm-ffi-stubgen(begin): import-section
use super::ir::Expr;
use super::ir::ExprObj;
use super::ir::Span;
use super::ir::Type;
use super::ir::Var;
use crate::ir::PrimExpr;
use crate::ir::PrimType;
use std::ops::Deref;
use tvm_ffi::Array;
use tvm_ffi::ObjectArc;
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/prim.Add
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.Add"]
#[type_final]
pub struct AddObj {
    base: ExprObj,
    pub a: PrimExpr,
    pub b: PrimExpr,
}

const _: () = {
    assert!(::core::mem::size_of::<AddObj>() == 56);
    assert!(::core::mem::align_of::<AddObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Add {
    base: ObjectArc<AddObj>,
}

impl Deref for Add {
    type Target = AddObj;
    fn deref(&self) -> &AddObj {
        &self.base
    }
}

impl Deref for AddObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl AddObj {
    pub(crate) fn new(span: Option<Span>, ty: PrimType, a: PrimExpr, b: PrimExpr) -> Self {
        let base = ExprObj::new(span, ty.into());
        Self { base, a, b }
    }
}

impl Add {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(
        span: Option<Span>,
        ty: PrimType,
        a: PrimExpr,
        b: PrimExpr,
    ) -> Self {
        let obj = AddObj::new(span, ty, a, b);
        Self {
            base: ObjectArc::new(obj),
        }
    }
}

tvm_ffi::impl_object_upcast!(Add => Expr, Add => PrimExpr);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/prim.And
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.And"]
#[type_final]
pub struct AndObj {
    base: ExprObj,
    pub a: PrimExpr,
    pub b: PrimExpr,
}

const _: () = {
    assert!(::core::mem::size_of::<AndObj>() == 56);
    assert!(::core::mem::align_of::<AndObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct And {
    base: ObjectArc<AndObj>,
}

impl Deref for And {
    type Target = AndObj;
    fn deref(&self) -> &AndObj {
        &self.base
    }
}

impl Deref for AndObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl AndObj {
    pub(crate) fn new(span: Option<Span>, ty: PrimType, a: PrimExpr, b: PrimExpr) -> Self {
        let base = ExprObj::new(span, ty.into());
        Self { base, a, b }
    }
}

impl And {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(
        span: Option<Span>,
        ty: PrimType,
        a: PrimExpr,
        b: PrimExpr,
    ) -> Self {
        let obj = AndObj::new(span, ty, a, b);
        Self {
            base: ObjectArc::new(obj),
        }
    }
}

tvm_ffi::impl_object_upcast!(And => Expr, And => PrimExpr);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/prim.Broadcast
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.Broadcast"]
#[type_final]
pub struct BroadcastObj {
    base: ExprObj,
    pub value: PrimExpr,
    pub lanes: PrimExpr,
}

const _: () = {
    assert!(::core::mem::size_of::<BroadcastObj>() == 56);
    assert!(::core::mem::align_of::<BroadcastObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Broadcast {
    base: ObjectArc<BroadcastObj>,
}

impl Deref for Broadcast {
    type Target = BroadcastObj;
    fn deref(&self) -> &BroadcastObj {
        &self.base
    }
}

impl Deref for BroadcastObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl BroadcastObj {
    pub(crate) fn new(span: Option<Span>, ty: PrimType, value: PrimExpr, lanes: PrimExpr) -> Self {
        let base = ExprObj::new(span, ty.into());
        Self { base, value, lanes }
    }
}

impl Broadcast {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(
        span: Option<Span>,
        ty: PrimType,
        value: PrimExpr,
        lanes: PrimExpr,
    ) -> Self {
        let obj = BroadcastObj::new(span, ty, value, lanes);
        Self {
            base: ObjectArc::new(obj),
        }
    }
}

tvm_ffi::impl_object_upcast!(Broadcast => Expr, Broadcast => PrimExpr);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/prim.Cast
/// Complete: reflected fields fill [40, 48) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.Cast"]
#[type_final]
pub struct CastObj {
    base: ExprObj,
    pub value: PrimExpr,
}

const _: () = {
    assert!(::core::mem::size_of::<CastObj>() == 48);
    assert!(::core::mem::align_of::<CastObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Cast {
    base: ObjectArc<CastObj>,
}

impl Deref for Cast {
    type Target = CastObj;
    fn deref(&self) -> &CastObj {
        &self.base
    }
}

impl Deref for CastObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl CastObj {
    pub(crate) fn new(span: Option<Span>, ty: PrimType, value: PrimExpr) -> Self {
        let base = ExprObj::new(span, ty.into());
        Self { base, value }
    }
}

impl Cast {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(span: Option<Span>, ty: PrimType, value: PrimExpr) -> Self {
        let obj = CastObj::new(span, ty, value);
        Self {
            base: ObjectArc::new(obj),
        }
    }
}

tvm_ffi::impl_object_upcast!(Cast => Expr, Cast => PrimExpr);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/prim.Div
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.Div"]
#[type_final]
pub struct DivObj {
    base: ExprObj,
    pub a: PrimExpr,
    pub b: PrimExpr,
}

const _: () = {
    assert!(::core::mem::size_of::<DivObj>() == 56);
    assert!(::core::mem::align_of::<DivObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Div {
    base: ObjectArc<DivObj>,
}

impl Deref for Div {
    type Target = DivObj;
    fn deref(&self) -> &DivObj {
        &self.base
    }
}

impl Deref for DivObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl DivObj {
    pub(crate) fn new(span: Option<Span>, ty: PrimType, a: PrimExpr, b: PrimExpr) -> Self {
        let base = ExprObj::new(span, ty.into());
        Self { base, a, b }
    }
}

impl Div {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(
        span: Option<Span>,
        ty: PrimType,
        a: PrimExpr,
        b: PrimExpr,
    ) -> Self {
        let obj = DivObj::new(span, ty, a, b);
        Self {
            base: ObjectArc::new(obj),
        }
    }
}

tvm_ffi::impl_object_upcast!(Div => Expr, Div => PrimExpr);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/prim.EQ
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.EQ"]
#[type_final]
pub struct EQObj {
    base: ExprObj,
    pub a: PrimExpr,
    pub b: PrimExpr,
}

const _: () = {
    assert!(::core::mem::size_of::<EQObj>() == 56);
    assert!(::core::mem::align_of::<EQObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct EQ {
    base: ObjectArc<EQObj>,
}

impl Deref for EQ {
    type Target = EQObj;
    fn deref(&self) -> &EQObj {
        &self.base
    }
}

impl Deref for EQObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl EQObj {
    pub(crate) fn new(span: Option<Span>, ty: PrimType, a: PrimExpr, b: PrimExpr) -> Self {
        let base = ExprObj::new(span, ty.into());
        Self { base, a, b }
    }
}

impl EQ {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(
        span: Option<Span>,
        ty: PrimType,
        a: PrimExpr,
        b: PrimExpr,
    ) -> Self {
        let obj = EQObj::new(span, ty, a, b);
        Self {
            base: ObjectArc::new(obj),
        }
    }
}

tvm_ffi::impl_object_upcast!(EQ => Expr, EQ => PrimExpr);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/prim.FloorDiv
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.FloorDiv"]
#[type_final]
pub struct FloorDivObj {
    base: ExprObj,
    pub a: PrimExpr,
    pub b: PrimExpr,
}

const _: () = {
    assert!(::core::mem::size_of::<FloorDivObj>() == 56);
    assert!(::core::mem::align_of::<FloorDivObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct FloorDiv {
    base: ObjectArc<FloorDivObj>,
}

impl Deref for FloorDiv {
    type Target = FloorDivObj;
    fn deref(&self) -> &FloorDivObj {
        &self.base
    }
}

impl Deref for FloorDivObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl FloorDivObj {
    pub(crate) fn new(span: Option<Span>, ty: PrimType, a: PrimExpr, b: PrimExpr) -> Self {
        let base = ExprObj::new(span, ty.into());
        Self { base, a, b }
    }
}

impl FloorDiv {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(
        span: Option<Span>,
        ty: PrimType,
        a: PrimExpr,
        b: PrimExpr,
    ) -> Self {
        let obj = FloorDivObj::new(span, ty, a, b);
        Self {
            base: ObjectArc::new(obj),
        }
    }
}

tvm_ffi::impl_object_upcast!(FloorDiv => Expr, FloorDiv => PrimExpr);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/prim.FloorMod
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.FloorMod"]
#[type_final]
pub struct FloorModObj {
    base: ExprObj,
    pub a: PrimExpr,
    pub b: PrimExpr,
}

const _: () = {
    assert!(::core::mem::size_of::<FloorModObj>() == 56);
    assert!(::core::mem::align_of::<FloorModObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct FloorMod {
    base: ObjectArc<FloorModObj>,
}

impl Deref for FloorMod {
    type Target = FloorModObj;
    fn deref(&self) -> &FloorModObj {
        &self.base
    }
}

impl Deref for FloorModObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl FloorModObj {
    pub(crate) fn new(span: Option<Span>, ty: PrimType, a: PrimExpr, b: PrimExpr) -> Self {
        let base = ExprObj::new(span, ty.into());
        Self { base, a, b }
    }
}

impl FloorMod {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(
        span: Option<Span>,
        ty: PrimType,
        a: PrimExpr,
        b: PrimExpr,
    ) -> Self {
        let obj = FloorModObj::new(span, ty, a, b);
        Self {
            base: ObjectArc::new(obj),
        }
    }
}

tvm_ffi::impl_object_upcast!(FloorMod => Expr, FloorMod => PrimExpr);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/prim.GE
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.GE"]
#[type_final]
pub struct GEObj {
    base: ExprObj,
    pub a: PrimExpr,
    pub b: PrimExpr,
}

const _: () = {
    assert!(::core::mem::size_of::<GEObj>() == 56);
    assert!(::core::mem::align_of::<GEObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct GE {
    base: ObjectArc<GEObj>,
}

impl Deref for GE {
    type Target = GEObj;
    fn deref(&self) -> &GEObj {
        &self.base
    }
}

impl Deref for GEObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl GEObj {
    pub(crate) fn new(span: Option<Span>, ty: PrimType, a: PrimExpr, b: PrimExpr) -> Self {
        let base = ExprObj::new(span, ty.into());
        Self { base, a, b }
    }
}

impl GE {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(
        span: Option<Span>,
        ty: PrimType,
        a: PrimExpr,
        b: PrimExpr,
    ) -> Self {
        let obj = GEObj::new(span, ty, a, b);
        Self {
            base: ObjectArc::new(obj),
        }
    }
}

tvm_ffi::impl_object_upcast!(GE => Expr, GE => PrimExpr);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/prim.GT
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.GT"]
#[type_final]
pub struct GTObj {
    base: ExprObj,
    pub a: PrimExpr,
    pub b: PrimExpr,
}

const _: () = {
    assert!(::core::mem::size_of::<GTObj>() == 56);
    assert!(::core::mem::align_of::<GTObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct GT {
    base: ObjectArc<GTObj>,
}

impl Deref for GT {
    type Target = GTObj;
    fn deref(&self) -> &GTObj {
        &self.base
    }
}

impl Deref for GTObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl GTObj {
    pub(crate) fn new(span: Option<Span>, ty: PrimType, a: PrimExpr, b: PrimExpr) -> Self {
        let base = ExprObj::new(span, ty.into());
        Self { base, a, b }
    }
}

impl GT {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(
        span: Option<Span>,
        ty: PrimType,
        a: PrimExpr,
        b: PrimExpr,
    ) -> Self {
        let obj = GTObj::new(span, ty, a, b);
        Self {
            base: ObjectArc::new(obj),
        }
    }
}

tvm_ffi::impl_object_upcast!(GT => Expr, GT => PrimExpr);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/prim.LE
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.LE"]
#[type_final]
pub struct LEObj {
    base: ExprObj,
    pub a: PrimExpr,
    pub b: PrimExpr,
}

const _: () = {
    assert!(::core::mem::size_of::<LEObj>() == 56);
    assert!(::core::mem::align_of::<LEObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct LE {
    base: ObjectArc<LEObj>,
}

impl Deref for LE {
    type Target = LEObj;
    fn deref(&self) -> &LEObj {
        &self.base
    }
}

impl Deref for LEObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl LEObj {
    pub(crate) fn new(span: Option<Span>, ty: PrimType, a: PrimExpr, b: PrimExpr) -> Self {
        let base = ExprObj::new(span, ty.into());
        Self { base, a, b }
    }
}

impl LE {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(
        span: Option<Span>,
        ty: PrimType,
        a: PrimExpr,
        b: PrimExpr,
    ) -> Self {
        let obj = LEObj::new(span, ty, a, b);
        Self {
            base: ObjectArc::new(obj),
        }
    }
}

tvm_ffi::impl_object_upcast!(LE => Expr, LE => PrimExpr);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/prim.LT
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.LT"]
#[type_final]
pub struct LTObj {
    base: ExprObj,
    pub a: PrimExpr,
    pub b: PrimExpr,
}

const _: () = {
    assert!(::core::mem::size_of::<LTObj>() == 56);
    assert!(::core::mem::align_of::<LTObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct LT {
    base: ObjectArc<LTObj>,
}

impl Deref for LT {
    type Target = LTObj;
    fn deref(&self) -> &LTObj {
        &self.base
    }
}

impl Deref for LTObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl LTObj {
    pub(crate) fn new(span: Option<Span>, ty: PrimType, a: PrimExpr, b: PrimExpr) -> Self {
        let base = ExprObj::new(span, ty.into());
        Self { base, a, b }
    }
}

impl LT {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(
        span: Option<Span>,
        ty: PrimType,
        a: PrimExpr,
        b: PrimExpr,
    ) -> Self {
        let obj = LTObj::new(span, ty, a, b);
        Self {
            base: ObjectArc::new(obj),
        }
    }
}

tvm_ffi::impl_object_upcast!(LT => Expr, LT => PrimExpr);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/prim.Let
/// Complete: reflected fields fill [40, 64) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.Let"]
#[type_final]
pub struct LetObj {
    base: ExprObj,
    pub var: Var,
    pub value: PrimExpr,
    pub body: PrimExpr,
}

const _: () = {
    assert!(::core::mem::size_of::<LetObj>() == 64);
    assert!(::core::mem::align_of::<LetObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Let {
    base: ObjectArc<LetObj>,
}

impl Deref for Let {
    type Target = LetObj;
    fn deref(&self) -> &LetObj {
        &self.base
    }
}

impl Deref for LetObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl LetObj {
    pub(crate) fn new(
        span: Option<Span>,
        ty: PrimType,
        var: Var,
        value: PrimExpr,
        body: PrimExpr,
    ) -> Self {
        let base = ExprObj::new(span, ty.into());
        Self {
            base,
            var,
            value,
            body,
        }
    }
}

impl Let {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(
        span: Option<Span>,
        ty: PrimType,
        var: Var,
        value: PrimExpr,
        body: PrimExpr,
    ) -> Self {
        let obj = LetObj::new(span, ty, var, value, body);
        Self {
            base: ObjectArc::new(obj),
        }
    }
}

tvm_ffi::impl_object_upcast!(Let => Expr, Let => PrimExpr);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/prim.Max
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.Max"]
#[type_final]
pub struct MaxObj {
    base: ExprObj,
    pub a: PrimExpr,
    pub b: PrimExpr,
}

const _: () = {
    assert!(::core::mem::size_of::<MaxObj>() == 56);
    assert!(::core::mem::align_of::<MaxObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Max {
    base: ObjectArc<MaxObj>,
}

impl Deref for Max {
    type Target = MaxObj;
    fn deref(&self) -> &MaxObj {
        &self.base
    }
}

impl Deref for MaxObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl MaxObj {
    pub(crate) fn new(span: Option<Span>, ty: PrimType, a: PrimExpr, b: PrimExpr) -> Self {
        let base = ExprObj::new(span, ty.into());
        Self { base, a, b }
    }
}

impl Max {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(
        span: Option<Span>,
        ty: PrimType,
        a: PrimExpr,
        b: PrimExpr,
    ) -> Self {
        let obj = MaxObj::new(span, ty, a, b);
        Self {
            base: ObjectArc::new(obj),
        }
    }
}

tvm_ffi::impl_object_upcast!(Max => Expr, Max => PrimExpr);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/prim.Min
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.Min"]
#[type_final]
pub struct MinObj {
    base: ExprObj,
    pub a: PrimExpr,
    pub b: PrimExpr,
}

const _: () = {
    assert!(::core::mem::size_of::<MinObj>() == 56);
    assert!(::core::mem::align_of::<MinObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Min {
    base: ObjectArc<MinObj>,
}

impl Deref for Min {
    type Target = MinObj;
    fn deref(&self) -> &MinObj {
        &self.base
    }
}

impl Deref for MinObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl MinObj {
    pub(crate) fn new(span: Option<Span>, ty: PrimType, a: PrimExpr, b: PrimExpr) -> Self {
        let base = ExprObj::new(span, ty.into());
        Self { base, a, b }
    }
}

impl Min {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(
        span: Option<Span>,
        ty: PrimType,
        a: PrimExpr,
        b: PrimExpr,
    ) -> Self {
        let obj = MinObj::new(span, ty, a, b);
        Self {
            base: ObjectArc::new(obj),
        }
    }
}

tvm_ffi::impl_object_upcast!(Min => Expr, Min => PrimExpr);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/prim.Mod
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.Mod"]
#[type_final]
pub struct ModObj {
    base: ExprObj,
    pub a: PrimExpr,
    pub b: PrimExpr,
}

const _: () = {
    assert!(::core::mem::size_of::<ModObj>() == 56);
    assert!(::core::mem::align_of::<ModObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Mod {
    base: ObjectArc<ModObj>,
}

impl Deref for Mod {
    type Target = ModObj;
    fn deref(&self) -> &ModObj {
        &self.base
    }
}

impl Deref for ModObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl ModObj {
    pub(crate) fn new(span: Option<Span>, ty: PrimType, a: PrimExpr, b: PrimExpr) -> Self {
        let base = ExprObj::new(span, ty.into());
        Self { base, a, b }
    }
}

impl Mod {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(
        span: Option<Span>,
        ty: PrimType,
        a: PrimExpr,
        b: PrimExpr,
    ) -> Self {
        let obj = ModObj::new(span, ty, a, b);
        Self {
            base: ObjectArc::new(obj),
        }
    }
}

tvm_ffi::impl_object_upcast!(Mod => Expr, Mod => PrimExpr);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/prim.Mul
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.Mul"]
#[type_final]
pub struct MulObj {
    base: ExprObj,
    pub a: PrimExpr,
    pub b: PrimExpr,
}

const _: () = {
    assert!(::core::mem::size_of::<MulObj>() == 56);
    assert!(::core::mem::align_of::<MulObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Mul {
    base: ObjectArc<MulObj>,
}

impl Deref for Mul {
    type Target = MulObj;
    fn deref(&self) -> &MulObj {
        &self.base
    }
}

impl Deref for MulObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl MulObj {
    pub(crate) fn new(span: Option<Span>, ty: PrimType, a: PrimExpr, b: PrimExpr) -> Self {
        let base = ExprObj::new(span, ty.into());
        Self { base, a, b }
    }
}

impl Mul {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(
        span: Option<Span>,
        ty: PrimType,
        a: PrimExpr,
        b: PrimExpr,
    ) -> Self {
        let obj = MulObj::new(span, ty, a, b);
        Self {
            base: ObjectArc::new(obj),
        }
    }
}

tvm_ffi::impl_object_upcast!(Mul => Expr, Mul => PrimExpr);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/prim.NE
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.NE"]
#[type_final]
pub struct NEObj {
    base: ExprObj,
    pub a: PrimExpr,
    pub b: PrimExpr,
}

const _: () = {
    assert!(::core::mem::size_of::<NEObj>() == 56);
    assert!(::core::mem::align_of::<NEObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct NE {
    base: ObjectArc<NEObj>,
}

impl Deref for NE {
    type Target = NEObj;
    fn deref(&self) -> &NEObj {
        &self.base
    }
}

impl Deref for NEObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl NEObj {
    pub(crate) fn new(span: Option<Span>, ty: PrimType, a: PrimExpr, b: PrimExpr) -> Self {
        let base = ExprObj::new(span, ty.into());
        Self { base, a, b }
    }
}

impl NE {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(
        span: Option<Span>,
        ty: PrimType,
        a: PrimExpr,
        b: PrimExpr,
    ) -> Self {
        let obj = NEObj::new(span, ty, a, b);
        Self {
            base: ObjectArc::new(obj),
        }
    }
}

tvm_ffi::impl_object_upcast!(NE => Expr, NE => PrimExpr);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/prim.Not
/// Complete: reflected fields fill [40, 48) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.Not"]
#[type_final]
pub struct NotObj {
    base: ExprObj,
    pub a: PrimExpr,
}

const _: () = {
    assert!(::core::mem::size_of::<NotObj>() == 48);
    assert!(::core::mem::align_of::<NotObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Not {
    base: ObjectArc<NotObj>,
}

impl Deref for Not {
    type Target = NotObj;
    fn deref(&self) -> &NotObj {
        &self.base
    }
}

impl Deref for NotObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl NotObj {
    pub(crate) fn new(span: Option<Span>, ty: PrimType, a: PrimExpr) -> Self {
        let base = ExprObj::new(span, ty.into());
        Self { base, a }
    }
}

impl Not {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(span: Option<Span>, ty: PrimType, a: PrimExpr) -> Self {
        let obj = NotObj::new(span, ty, a);
        Self {
            base: ObjectArc::new(obj),
        }
    }
}

tvm_ffi::impl_object_upcast!(Not => Expr, Not => PrimExpr);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/prim.Or
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.Or"]
#[type_final]
pub struct OrObj {
    base: ExprObj,
    pub a: PrimExpr,
    pub b: PrimExpr,
}

const _: () = {
    assert!(::core::mem::size_of::<OrObj>() == 56);
    assert!(::core::mem::align_of::<OrObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Or {
    base: ObjectArc<OrObj>,
}

impl Deref for Or {
    type Target = OrObj;
    fn deref(&self) -> &OrObj {
        &self.base
    }
}

impl Deref for OrObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl OrObj {
    pub(crate) fn new(span: Option<Span>, ty: PrimType, a: PrimExpr, b: PrimExpr) -> Self {
        let base = ExprObj::new(span, ty.into());
        Self { base, a, b }
    }
}

impl Or {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(
        span: Option<Span>,
        ty: PrimType,
        a: PrimExpr,
        b: PrimExpr,
    ) -> Self {
        let obj = OrObj::new(span, ty, a, b);
        Self {
            base: ObjectArc::new(obj),
        }
    }
}

tvm_ffi::impl_object_upcast!(Or => Expr, Or => PrimExpr);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/prim.Ramp
/// Complete: reflected fields fill [40, 64) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.Ramp"]
#[type_final]
pub struct RampObj {
    base: ExprObj,
    pub base_: PrimExpr,
    pub stride: PrimExpr,
    pub lanes: PrimExpr,
}

const _: () = {
    assert!(::core::mem::size_of::<RampObj>() == 64);
    assert!(::core::mem::align_of::<RampObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Ramp {
    base: ObjectArc<RampObj>,
}

impl Deref for Ramp {
    type Target = RampObj;
    fn deref(&self) -> &RampObj {
        &self.base
    }
}

impl Deref for RampObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl RampObj {
    pub(crate) fn new(
        span: Option<Span>,
        ty: PrimType,
        base_: PrimExpr,
        stride: PrimExpr,
        lanes: PrimExpr,
    ) -> Self {
        let base = ExprObj::new(span, ty.into());
        Self {
            base,
            base_,
            stride,
            lanes,
        }
    }
}

impl Ramp {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(
        span: Option<Span>,
        ty: PrimType,
        base_: PrimExpr,
        stride: PrimExpr,
        lanes: PrimExpr,
    ) -> Self {
        let obj = RampObj::new(span, ty, base_, stride, lanes);
        Self {
            base: ObjectArc::new(obj),
        }
    }
}

tvm_ffi::impl_object_upcast!(Ramp => Expr, Ramp => PrimExpr);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/prim.Select
/// Complete: reflected fields fill [40, 64) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.Select"]
#[type_final]
pub struct SelectObj {
    base: ExprObj,
    pub condition: PrimExpr,
    pub true_value: PrimExpr,
    pub false_value: PrimExpr,
}

const _: () = {
    assert!(::core::mem::size_of::<SelectObj>() == 64);
    assert!(::core::mem::align_of::<SelectObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Select {
    base: ObjectArc<SelectObj>,
}

impl Deref for Select {
    type Target = SelectObj;
    fn deref(&self) -> &SelectObj {
        &self.base
    }
}

impl Deref for SelectObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl SelectObj {
    pub(crate) fn new(
        span: Option<Span>,
        ty: PrimType,
        condition: PrimExpr,
        true_value: PrimExpr,
        false_value: PrimExpr,
    ) -> Self {
        let base = ExprObj::new(span, ty.into());
        Self {
            base,
            condition,
            true_value,
            false_value,
        }
    }
}

impl Select {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(
        span: Option<Span>,
        ty: PrimType,
        condition: PrimExpr,
        true_value: PrimExpr,
        false_value: PrimExpr,
    ) -> Self {
        let obj = SelectObj::new(span, ty, condition, true_value, false_value);
        Self {
            base: ObjectArc::new(obj),
        }
    }
}

tvm_ffi::impl_object_upcast!(Select => Expr, Select => PrimExpr);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/prim.Shuffle
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.Shuffle"]
#[type_final]
pub struct ShuffleObj {
    base: ExprObj,
    pub vectors: Array<PrimExpr>,
    pub indices: Array<PrimExpr>,
}

const _: () = {
    assert!(::core::mem::size_of::<ShuffleObj>() == 56);
    assert!(::core::mem::align_of::<ShuffleObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Shuffle {
    base: ObjectArc<ShuffleObj>,
}

impl Deref for Shuffle {
    type Target = ShuffleObj;
    fn deref(&self) -> &ShuffleObj {
        &self.base
    }
}

impl Deref for ShuffleObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl ShuffleObj {
    pub(crate) fn new(
        span: Option<Span>,
        ty: PrimType,
        vectors: Array<PrimExpr>,
        indices: Array<PrimExpr>,
    ) -> Self {
        let base = ExprObj::new(span, ty.into());
        Self {
            base,
            vectors,
            indices,
        }
    }
}

impl Shuffle {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(
        span: Option<Span>,
        ty: PrimType,
        vectors: Array<PrimExpr>,
        indices: Array<PrimExpr>,
    ) -> Self {
        let obj = ShuffleObj::new(span, ty, vectors, indices);
        Self {
            base: ObjectArc::new(obj),
        }
    }
}

tvm_ffi::impl_object_upcast!(Shuffle => Expr, Shuffle => PrimExpr);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/prim.Sub
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.Sub"]
#[type_final]
pub struct SubObj {
    base: ExprObj,
    pub a: PrimExpr,
    pub b: PrimExpr,
}

const _: () = {
    assert!(::core::mem::size_of::<SubObj>() == 56);
    assert!(::core::mem::align_of::<SubObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Sub {
    base: ObjectArc<SubObj>,
}

impl Deref for Sub {
    type Target = SubObj;
    fn deref(&self) -> &SubObj {
        &self.base
    }
}

impl Deref for SubObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl SubObj {
    pub(crate) fn new(span: Option<Span>, ty: PrimType, a: PrimExpr, b: PrimExpr) -> Self {
        let base = ExprObj::new(span, ty.into());
        Self { base, a, b }
    }
}

impl Sub {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(
        span: Option<Span>,
        ty: PrimType,
        a: PrimExpr,
        b: PrimExpr,
    ) -> Self {
        let obj = SubObj::new(span, ty, a, b);
        Self {
            base: ObjectArc::new(obj),
        }
    }
}

tvm_ffi::impl_object_upcast!(Sub => Expr, Sub => PrimExpr);
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
    use crate::ir::{
        Call, CallObj, Expr, FloatImm, FloatImmObj, IntImm, IntImmObj, Op, PrimExpr, PrimType,
        PrimTypeObj, Span, Var,
    };
    use tvm_ffi::{
        Array, DLDataType, DLDataTypeCode, DLDataTypeExt, Error, ObjectRefCast, ObjectRefCore,
        Result, String, TYPE_ERROR, VALUE_ERROR,
    };

    impl Add {
        /// Construct an addition expression directly in Rust.
        pub fn new<L, R>(lhs: L, rhs: R) -> Result<Self>
        where
            L: Into<Expr>,
            R: Into<Expr>,
        {
            Self::with_span(lhs, rhs, None)
        }

        /// Construct an addition expression with optional source metadata.
        pub fn with_span<L, R>(lhs: L, rhs: R, span: Option<&Span>) -> Result<Self>
        where
            L: Into<Expr>,
            R: Into<Expr>,
        {
            let lhs = lhs.into();
            let rhs = rhs.into();
            let result_type = matching_binary_type(&lhs, &rhs)?;
            let lhs = PrimExpr::try_from(lhs)?;
            let rhs = PrimExpr::try_from(rhs)?;
            Ok(Self::from_complete_fields(
                span.cloned(),
                result_type,
                lhs,
                rhs,
            ))
        }

        /// Copy this node with new `a`, `b`; every other field, span
        /// included, is carried over from `self`.
        ///
        /// Takes the same required fields as [`Add::new`] and, like
        /// [`Add::from_complete_fields`], runs no validation.
        /// The result type follows the new `a`.
        pub fn copy_with(&self, a: PrimExpr, b: PrimExpr) -> Self {
            Self::from_complete_fields(self.span.clone(), a.type_annotation(), a, b)
        }
    }

    pub(crate) fn primitive_type(expr: &Expr, context: &str) -> Result<crate::ir::PrimType> {
        expr.ty
            .clone()
            .try_cast::<crate::ir::PrimType>()
            .map_err(|_| {
                Error::new(
                    TYPE_ERROR,
                    &format!("{context} must have a primitive type"),
                    "",
                )
            })
    }

    fn matching_binary_type(lhs: &Expr, rhs: &Expr) -> Result<PrimType> {
        let lhs_type = primitive_type(lhs, "left binary operand")?;
        let lhs_dtype = lhs_type.dtype;
        let rhs_dtype = primitive_type(rhs, "right binary operand")?.dtype;
        if lhs_dtype != rhs_dtype {
            return Err(Error::new(
                TYPE_ERROR,
                &format!(
                    "mismatched binary operand types: {} vs. {}",
                    lhs_dtype.to_string(),
                    rhs_dtype.to_string()
                ),
                "",
            ));
        }
        Ok(lhs_type)
    }

    macro_rules! define_binary_expression {
        ($reference:ident) => {
            impl $reference {
                /// Construct the binary expression directly in Rust.
                pub fn new<L, R>(lhs: L, rhs: R) -> Result<Self>
                where
                    L: Into<Expr>,
                    R: Into<Expr>,
                {
                    Self::with_span(lhs, rhs, None)
                }

                /// Construct the binary expression with optional source metadata.
                pub fn with_span<L, R>(lhs: L, rhs: R, span: Option<&Span>) -> Result<Self>
                where
                    L: Into<Expr>,
                    R: Into<Expr>,
                {
                    let lhs = lhs.into();
                    let rhs = rhs.into();
                    let result_type = matching_binary_type(&lhs, &rhs)?;
                    let lhs = PrimExpr::try_from(lhs)?;
                    let rhs = PrimExpr::try_from(rhs)?;
                    Ok(Self::from_complete_fields(
                        span.cloned(),
                        result_type,
                        lhs,
                        rhs,
                    ))
                }

                /// Copy this node with new `a`, `b`; every other field, span
                /// included, is carried over from `self`.
                ///
                /// Takes the same required fields as `new` and, like
                /// `from_complete_fields`, runs no validation.
                /// The result type follows the new `a`.
                pub fn copy_with(&self, a: PrimExpr, b: PrimExpr) -> Self {
                    Self::from_complete_fields(self.span.clone(), a.type_annotation(), a, b)
                }
            }
        };
    }

    define_binary_expression!(Sub);
    define_binary_expression!(Mul);
    define_binary_expression!(Div);
    define_binary_expression!(Mod);
    define_binary_expression!(FloorDiv);
    define_binary_expression!(FloorMod);
    define_binary_expression!(Min);
    define_binary_expression!(Max);

    impl EQ {
        /// Construct an equality comparison directly in Rust.
        pub fn new<L, R>(lhs: L, rhs: R) -> Result<Self>
        where
            L: Into<Expr>,
            R: Into<Expr>,
        {
            Self::with_span(lhs, rhs, None)
        }

        /// Construct an equality comparison with optional source metadata.
        pub fn with_span<L, R>(lhs: L, rhs: R, span: Option<&Span>) -> Result<Self>
        where
            L: Into<Expr>,
            R: Into<Expr>,
        {
            let lhs = lhs.into();
            let rhs = rhs.into();
            let operand_type = matching_binary_type(&lhs, &rhs)?;
            let result_type = PrimType::from_dtype(DLDataType {
                code: DLDataTypeCode::kDLBool as u8,
                bits: 8,
                lanes: operand_type.dtype.lanes,
            })?;
            Ok(Self::from_complete_fields(
                span.cloned(),
                result_type,
                PrimExpr::try_from(lhs)?,
                PrimExpr::try_from(rhs)?,
            ))
        }

        /// Copy this node with new `a`, `b`; every other field, span
        /// included, is carried over from `self`.
        ///
        /// Takes the same required fields as [`EQ::new`] and, like
        /// [`EQ::from_complete_fields`], runs no validation.
        /// The result type is carried over unchanged.
        pub fn copy_with(&self, a: PrimExpr, b: PrimExpr) -> Self {
            Self::from_complete_fields(
                self.span.clone(),
                PrimExpr::from(self).type_annotation(),
                a,
                b,
            )
        }
    }

    macro_rules! define_comparison_expression {
        ($reference:ident) => {
            impl $reference {
                /// Construct the comparison directly in Rust.
                pub fn new<L, R>(lhs: L, rhs: R) -> Result<Self>
                where
                    L: Into<Expr>,
                    R: Into<Expr>,
                {
                    Self::with_span(lhs, rhs, None)
                }

                /// Construct the comparison with optional source metadata.
                pub fn with_span<L, R>(lhs: L, rhs: R, span: Option<&Span>) -> Result<Self>
                where
                    L: Into<Expr>,
                    R: Into<Expr>,
                {
                    let lhs = lhs.into();
                    let rhs = rhs.into();
                    let operand_type = matching_binary_type(&lhs, &rhs)?;
                    let result_type = PrimType::from_dtype(DLDataType {
                        code: DLDataTypeCode::kDLBool as u8,
                        bits: 8,
                        lanes: operand_type.dtype.lanes,
                    })?;
                    Ok(Self::from_complete_fields(
                        span.cloned(),
                        result_type,
                        PrimExpr::try_from(lhs)?,
                        PrimExpr::try_from(rhs)?,
                    ))
                }

                /// Copy this node with new `a`, `b`; every other field, span
                /// included, is carried over from `self`.
                ///
                /// Takes the same required fields as `new` and, like
                /// `from_complete_fields`, runs no validation.
                /// The result type is carried over unchanged.
                pub fn copy_with(&self, a: PrimExpr, b: PrimExpr) -> Self {
                    Self::from_complete_fields(
                        self.span.clone(),
                        PrimExpr::from(self).type_annotation(),
                        a,
                        b,
                    )
                }
            }
        };
    }

    define_comparison_expression!(NE);
    define_comparison_expression!(LT);
    define_comparison_expression!(LE);
    define_comparison_expression!(GT);
    define_comparison_expression!(GE);

    impl And {
        /// Construct a logical conjunction directly in Rust.
        pub fn new<L, R>(lhs: L, rhs: R) -> Result<Self>
        where
            L: Into<Expr>,
            R: Into<Expr>,
        {
            Self::with_span(lhs, rhs, None)
        }

        /// Construct a logical conjunction with optional source metadata.
        pub fn with_span<L, R>(lhs: L, rhs: R, span: Option<&Span>) -> Result<Self>
        where
            L: Into<Expr>,
            R: Into<Expr>,
        {
            let lhs = lhs.into();
            let rhs = rhs.into();
            let operand_type = matching_binary_type(&lhs, &rhs)?;
            if operand_type.dtype.code != DLDataTypeCode::kDLBool as u8 {
                return Err(Error::new(
                    TYPE_ERROR,
                    "logical conjunction operands must have bool type",
                    "",
                ));
            }
            let result_type = PrimType::from_dtype(DLDataType {
                code: DLDataTypeCode::kDLBool as u8,
                bits: 8,
                lanes: operand_type.dtype.lanes,
            })?;
            Ok(Self::from_complete_fields(
                span.cloned(),
                result_type,
                PrimExpr::try_from(lhs)?,
                PrimExpr::try_from(rhs)?,
            ))
        }

        /// Copy this node with new `a`, `b`; every other field, span
        /// included, is carried over from `self`.
        ///
        /// Takes the same required fields as [`And::new`] and, like
        /// [`And::from_complete_fields`], runs no validation.
        /// The result type is carried over unchanged.
        pub fn copy_with(&self, a: PrimExpr, b: PrimExpr) -> Self {
            Self::from_complete_fields(
                self.span.clone(),
                PrimExpr::from(self).type_annotation(),
                a,
                b,
            )
        }
    }

    impl Or {
        /// Construct a logical disjunction directly in Rust.
        pub fn new<L, R>(lhs: L, rhs: R) -> Result<Self>
        where
            L: Into<Expr>,
            R: Into<Expr>,
        {
            Self::with_span(lhs, rhs, None)
        }

        /// Construct a logical disjunction with optional source metadata.
        pub fn with_span<L, R>(lhs: L, rhs: R, span: Option<&Span>) -> Result<Self>
        where
            L: Into<Expr>,
            R: Into<Expr>,
        {
            let lhs = lhs.into();
            let rhs = rhs.into();
            let operand_type = matching_binary_type(&lhs, &rhs)?;
            if operand_type.dtype.code != DLDataTypeCode::kDLBool as u8 {
                return Err(Error::new(
                    TYPE_ERROR,
                    "logical disjunction operands must have bool type",
                    "",
                ));
            }
            let result_type = PrimType::from_dtype(DLDataType {
                code: DLDataTypeCode::kDLBool as u8,
                bits: 8,
                lanes: operand_type.dtype.lanes,
            })?;
            Ok(Self::from_complete_fields(
                span.cloned(),
                result_type,
                PrimExpr::try_from(lhs)?,
                PrimExpr::try_from(rhs)?,
            ))
        }

        /// Copy this node with new `a`, `b`; every other field, span
        /// included, is carried over from `self`.
        ///
        /// Takes the same required fields as [`Or::new`] and, like
        /// [`Or::from_complete_fields`], runs no validation.
        /// The result type is carried over unchanged.
        pub fn copy_with(&self, a: PrimExpr, b: PrimExpr) -> Self {
            Self::from_complete_fields(
                self.span.clone(),
                PrimExpr::from(self).type_annotation(),
                a,
                b,
            )
        }
    }

    impl Not {
        /// Construct a logical negation directly in Rust.
        pub fn new<A>(value: A) -> Result<Self>
        where
            A: Into<Expr>,
        {
            Self::with_span(value, None)
        }

        /// Construct a logical negation with optional source metadata.
        pub fn with_span<A>(value: A, span: Option<&Span>) -> Result<Self>
        where
            A: Into<Expr>,
        {
            let value = value.into();
            let value_type = primitive_type(&value, "logical negation operand")?;
            if value_type.dtype.code != DLDataTypeCode::kDLBool as u8 {
                return Err(Error::new(
                    TYPE_ERROR,
                    "logical negation operand must have bool type",
                    "",
                ));
            }
            Ok(Self::from_complete_fields(
                span.cloned(),
                value_type,
                PrimExpr::try_from(value)?,
            ))
        }

        /// Copy this node with new `a`; every other field, span
        /// included, is carried over from `self`.
        ///
        /// Takes the same required fields as [`Not::new`] and, like
        /// [`Not::from_complete_fields`], runs no validation.
        /// The result type is carried over unchanged.
        pub fn copy_with(&self, a: PrimExpr) -> Self {
            Self::from_complete_fields(self.span.clone(), PrimExpr::from(self).type_annotation(), a)
        }
    }

    impl Cast {
        /// Cast a primitive expression while preserving its lane count.
        pub fn new<V>(ty: PrimType, value: V) -> Result<Self>
        where
            V: Into<Expr>,
        {
            Self::with_span(ty, value, None)
        }

        /// Cast a primitive expression with optional source metadata.
        pub fn with_span<V>(ty: PrimType, value: V, span: Option<&Span>) -> Result<Self>
        where
            V: Into<Expr>,
        {
            let value = PrimExpr::try_from(value.into())?;
            if value.dtype().lanes != ty.dtype.lanes {
                return Err(Error::new(
                    TYPE_ERROR,
                    "Cast must preserve the operand lane count",
                    "",
                ));
            }
            Ok(Self::from_complete_fields(span.cloned(), ty, value))
        }

        /// Copy this node with new `ty`, `value`; every other field, span
        /// included, is carried over from `self`.
        ///
        /// Takes the same required fields as [`Cast::new`] and, like
        /// [`Cast::from_complete_fields`], runs no validation.
        pub fn copy_with(&self, ty: PrimType, value: PrimExpr) -> Self {
            Self::from_complete_fields(self.span.clone(), ty, value)
        }
    }

    impl Ramp {
        /// Construct a vector ramp, converting the stride to the base's scalar type.
        pub fn new<B: Into<Expr>, S: Into<Expr>, L: Into<Expr>>(
            base: B,
            stride: S,
            lanes: L,
        ) -> Result<Self> {
            Self::with_span(base, stride, lanes, None)
        }

        /// Construct a vector ramp with optional source metadata.
        pub fn with_span<B: Into<Expr>, S: Into<Expr>, L: Into<Expr>>(
            base: B,
            stride: S,
            lanes: L,
            span: Option<&Span>,
        ) -> Result<Self> {
            let base = PrimExpr::try_from(base.into())?;
            let mut stride = PrimExpr::try_from(stride.into())?;
            if base.dtype().lanes != 1 || stride.dtype().lanes != 1 {
                return Err(Error::new(
                    TYPE_ERROR,
                    "Ramp base and stride must be scalar",
                    "",
                ));
            }
            if stride.dtype() != base.dtype() {
                stride = cast_ramp_stride(base.type_annotation(), stride)?;
            }
            let (ty, lanes) = vector_type_and_lanes(base.dtype(), lanes.into().try_into()?)?;
            Ok(Self::from_complete_fields(
                span.cloned(),
                ty,
                base,
                stride,
                lanes,
            ))
        }
    }

    impl Broadcast {
        /// Repeat one scalar value across fixed or scalable vector lanes.
        pub fn new<V: Into<Expr>, L: Into<Expr>>(value: V, lanes: L) -> Result<Self> {
            Self::with_span(value, lanes, None)
        }

        /// Construct a broadcast with optional source metadata.
        pub fn with_span<V: Into<Expr>, L: Into<Expr>>(
            value: V,
            lanes: L,
            span: Option<&Span>,
        ) -> Result<Self> {
            let value = PrimExpr::try_from(value.into())?;
            if value.dtype().lanes != 1 {
                return Err(Error::new(TYPE_ERROR, "Broadcast value must be scalar", ""));
            }
            let (ty, lanes) = vector_type_and_lanes(value.dtype(), lanes.into().try_into()?)?;
            Ok(Self::from_complete_fields(span.cloned(), ty, value, lanes))
        }
    }

    impl Shuffle {
        /// Construct a shuffle of vectors with the same scalar element type.
        pub fn new(vectors: Array<PrimExpr>, indices: Array<PrimExpr>) -> Result<Self> {
            Self::with_span(vectors, indices, None)
        }

        /// Construct a shuffle with optional source metadata.
        pub fn with_span(
            vectors: Array<PrimExpr>,
            indices: Array<PrimExpr>,
            span: Option<&Span>,
        ) -> Result<Self> {
            if vectors.is_empty() || indices.is_empty() {
                return Err(Error::new(
                    VALUE_ERROR,
                    "Shuffle vectors and indices must be nonempty",
                    "",
                ));
            }
            let dtype = vectors.get(0)?.dtype();
            let mut total_lanes = 0_usize;
            for vector in vectors.iter() {
                let other = vector.dtype();
                if other.code != dtype.code || other.bits != dtype.bits {
                    return Err(Error::new(
                        TYPE_ERROR,
                        "Shuffle element types must match",
                        "",
                    ));
                }
                total_lanes += fixed_vector_lanes(&vector)?;
            }
            if indices.len() > total_lanes {
                return Err(Error::new(
                    VALUE_ERROR,
                    "Shuffle has more indices than input lanes",
                    "",
                ));
            }
            let ty = PrimType::from_dtype(DLDataType {
                lanes: indices.len() as u16,
                ..dtype
            })?;
            Ok(Self::from_complete_fields(
                span.cloned(),
                ty,
                vectors,
                indices,
            ))
        }

        /// Concatenate vectors, reusing the input when there is only one.
        pub fn concat(vectors: Array<PrimExpr>, span: Option<&Span>) -> Result<PrimExpr> {
            if vectors.len() == 1 {
                return vectors.get(0);
            }
            let mut indices = Vec::new();
            for vector in vectors.iter() {
                for _ in 0..fixed_vector_lanes(&vector)? {
                    indices.push(IntImm::new("int32", indices.len() as i64)?.into());
                }
            }
            Ok(Self::with_span(vectors, Array::new(indices), span)?.into())
        }

        /// Extract one vector lane as a scalar expression.
        pub fn extract_element(
            vector: PrimExpr,
            index: i32,
            span: Option<&Span>,
        ) -> Result<PrimExpr> {
            Ok(Self::with_span(
                Array::new(vec![vector]),
                Array::new(vec![IntImm::new("int32", i64::from(index))?.into()]),
                span,
            )?
            .into())
        }
    }

    // Ramp uses TVM's scalar cast semantics: fold literals, otherwise create a Cast.
    fn cast_ramp_stride(ty: PrimType, value: PrimExpr) -> Result<PrimExpr> {
        let integer = value
            .as_node::<IntImmObj>()
            .map(|literal| literal.value_i64());
        let float = value.as_node::<FloatImmObj>().map(|literal| literal.value);
        if integer.is_none() && float.is_none() {
            return Ok(Cast::new(ty, value)?.into());
        }
        let dtype = ty.dtype;
        let span = value.span.as_ref();
        let code = dtype.code;
        if code == DLDataTypeCode::kDLUInt as u8 {
            let unsigned = if let Some(integer) = integer {
                u64::try_from(integer).ok()
            } else {
                float
                    .filter(|value| (0.0..18446744073709551616.0).contains(value))
                    .map(|value| value as u64)
            }
            .ok_or_else(|| Error::new(VALUE_ERROR, "Stride literal is outside uint64 range", ""))?;
            if unsigned > i64::MAX as u64 {
                let word_type = PrimType::new("uint32")?.dtype;
                return Call::with_metadata(
                    ty,
                    Op::get("tirx.large_uint_imm")?,
                    vec![
                        IntImm::from_dtype_with_span(
                            word_type,
                            (unsigned & 0xffff_ffff) as i64,
                            span,
                        )?
                        .into(),
                        IntImm::from_dtype_with_span(word_type, (unsigned >> 32) as i64, span)?
                            .into(),
                    ],
                    None,
                    Vec::new(),
                    span,
                )
                .try_cast();
            }
            return Ok(IntImm::from_dtype_with_span(dtype, unsigned as i64, span)?.into());
        }
        if code == DLDataTypeCode::kDLInt as u8 || code == DLDataTypeCode::kDLBool as u8 {
            let integer = integer
                .or_else(|| {
                    float
                        .map(f64::trunc)
                        .filter(|value| {
                            (-9223372036854775808.0..9223372036854775808.0).contains(value)
                        })
                        .map(|value| value as i64)
                })
                .ok_or_else(|| {
                    Error::new(VALUE_ERROR, "Stride literal is outside int64 range", "")
                })?;
            return Ok(IntImm::from_dtype_with_span(dtype, integer, span)?.into());
        }
        // MakeConstScalar does not construct custom floating-point constants.
        if code >= 129 {
            return Err(Error::new(
                TYPE_ERROR,
                "Cannot cast a stride literal to a custom dtype",
                "",
            ));
        }
        let float = float.unwrap_or_else(|| integer.unwrap() as f64);
        Ok(FloatImm::from_dtype_with_span(dtype, float, span)?.into())
    }

    fn fixed_vector_lanes(value: &PrimExpr) -> Result<usize> {
        usize::try_from(value.dtype().lanes as i16).map_err(|_| {
            Error::new(
                TYPE_ERROR,
                "Shuffle requires fixed-length input vectors",
                "",
            )
        })
    }

    // Select compares fixed lane counts or vscale factors, as its C++ constructor does.
    fn lanes_or_vscale_factor(dtype: DLDataType) -> Result<i32> {
        let lanes = i32::from(dtype.lanes as i16);
        if lanes == -1 {
            return Err(Error::new(TYPE_ERROR, "Invalid vector lane encoding", ""));
        }
        Ok(lanes.abs())
    }

    fn vector_type_and_lanes(dtype: DLDataType, lanes: PrimExpr) -> Result<(PrimType, PrimExpr)> {
        if let Some(literal) = lanes.as_node::<IntImmObj>() {
            let count = literal.value_i64() as i32;
            if count <= 1 {
                return Err(Error::new(
                    VALUE_ERROR,
                    "Vector lane count must be greater than one",
                    "",
                ));
            }
            let ty = PrimType::from_dtype(DLDataType {
                lanes: count as u16,
                ..dtype
            })?;
            return Ok((ty, IntImm::new("int32", i64::from(count))?.into()));
        }
        let vscale = Op::get("prim.vscale")?;
        let factor = lanes.as_node::<MulObj>().and_then(|multiply| {
            [(&multiply.a, &multiply.b), (&multiply.b, &multiply.a)]
                .into_iter()
                .find_map(|(constant, call)| {
                    let constant = constant.as_node::<IntImmObj>()?;
                    let call = call.as_node::<CallObj>()?;
                    call.op
                        .same_as(&vscale)
                        .then_some(constant.value_i64() as i32)
                })
        });
        let factor = factor
            .filter(|factor| (2..32768).contains(factor))
            .ok_or_else(|| {
                Error::new(
                    VALUE_ERROR,
                    "Scalable lanes must be vscale() times a factor in 2..32768",
                    "",
                )
            })?;
        let ty = PrimType::from_dtype(DLDataType {
            lanes: (-factor) as u16,
            ..dtype
        })?;
        let call = Call::new(PrimType::new("int32")?, vscale, Vec::new());
        let lanes = Mul::new(call, IntImm::new("int32", i64::from(factor))?)?.into();
        Ok((ty, lanes))
    }

    impl Select {
        /// Choose between two primitive values using a boolean condition.
        pub fn new<C, T, F>(condition: C, true_value: T, false_value: F) -> Result<Self>
        where
            C: Into<Expr>,
            T: Into<Expr>,
            F: Into<Expr>,
        {
            Self::with_span(condition, true_value, false_value, None)
        }

        /// Construct a select expression with optional source metadata.
        pub fn with_span<C, T, F>(
            condition: C,
            true_value: T,
            false_value: F,
            span: Option<&Span>,
        ) -> Result<Self>
        where
            C: Into<Expr>,
            T: Into<Expr>,
            F: Into<Expr>,
        {
            let condition = condition.into();
            let true_value = true_value.into();
            let false_value = false_value.into();
            let condition_type = primitive_type(&condition, "Select condition")?;
            if condition_type.dtype.code != DLDataTypeCode::kDLBool as u8 {
                return Err(Error::new(
                    TYPE_ERROR,
                    "Select condition must have bool type",
                    "",
                ));
            }
            let result_type = matching_binary_type(&true_value, &false_value)?;
            if lanes_or_vscale_factor(condition_type.dtype)?
                != lanes_or_vscale_factor(result_type.dtype)?
                && condition_type.dtype.lanes != 1
            {
                return Err(Error::new(
                    TYPE_ERROR,
                    "Select condition lanes must match the selected values",
                    "",
                ));
            }
            Ok(Self::from_complete_fields(
                span.cloned(),
                result_type,
                PrimExpr::try_from(condition)?,
                PrimExpr::try_from(true_value)?,
                PrimExpr::try_from(false_value)?,
            ))
        }

        /// Copy this node with new `condition`, `true_value`, `false_value`; every other field, span
        /// included, is carried over from `self`.
        ///
        /// Takes the same required fields as [`Select::new`] and, like
        /// [`Select::from_complete_fields`], runs no validation.
        /// The result type follows the new `true_value`.
        pub fn copy_with(
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

    impl Let {
        /// Bind `var` to `value`, then evaluate `body`.
        pub fn new<V, B>(var: Var, value: V, body: B) -> Result<Self>
        where
            V: Into<Expr>,
            B: Into<Expr>,
        {
            Self::with_span(var, value, body, None)
        }

        /// Construct a let expression with optional source metadata.
        pub fn with_span<V, B>(var: Var, value: V, body: B, span: Option<&Span>) -> Result<Self>
        where
            V: Into<Expr>,
            B: Into<Expr>,
        {
            let value = value.into();
            let body = body.into();
            let variable_dtype = var
                .ty
                .as_node::<PrimTypeObj>()
                .ok_or_else(|| {
                    Error::new(TYPE_ERROR, "Let variable must have a primitive type", "")
                })?
                .dtype;
            let value = PrimExpr::try_from(value)?;
            let body = PrimExpr::try_from(body)?;
            if variable_dtype != value.dtype() {
                return Err(Error::new(
                    TYPE_ERROR,
                    "Let value type must match the bound variable type",
                    "",
                ));
            }
            Ok(Self::from_complete_fields(
                span.cloned(),
                body.type_annotation(),
                var,
                value,
                body,
            ))
        }

        /// Copy this node with new `var`, `value`, `body`; every other field, span
        /// included, is carried over from `self`.
        ///
        /// Takes the same required fields as [`Let::new`] and, like
        /// [`Let::from_complete_fields`], runs no validation.
        /// The result type follows the new `body`.
        pub fn copy_with(&self, var: Var, value: PrimExpr, body: PrimExpr) -> Self {
            Self::from_complete_fields(self.span.clone(), body.type_annotation(), var, value, body)
        }
    }
}
