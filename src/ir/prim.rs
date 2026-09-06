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

//! Primitive scalar and vector expression nodes (`ir.prim.*`).
//!
//! The blocks are emitted by `tvm-ffi-stubgen --target rust`; the reviewed
//! semantic constructors follow in `mod semantic`.

pub(crate) use semantic::primitive_type;

// Every object registered under `ir.prim` gets its block in this file; `skip` leaves one out.
// tvm-ffi-stubgen(prefix): ir.prim
// Hand-maintained directives; tvm-ffi-stubgen applies them on every run.
// tvm-ffi-stubgen(import-object): crate::ir::PrimExpr
// tvm-ffi-stubgen(import-object): super::super::ir::PrimType
// tvm-ffi-stubgen(nullable): ir.Expr.span
// tvm-ffi-stubgen(field): ir.prim.Add.a -> PrimExpr
// tvm-ffi-stubgen(field): ir.prim.Add.b -> PrimExpr
// tvm-ffi-stubgen(field): ir.prim.Sub.a -> PrimExpr
// tvm-ffi-stubgen(field): ir.prim.Sub.b -> PrimExpr
// tvm-ffi-stubgen(field): ir.prim.Mul.a -> PrimExpr
// tvm-ffi-stubgen(field): ir.prim.Mul.b -> PrimExpr
// tvm-ffi-stubgen(field): ir.prim.Div.a -> PrimExpr
// tvm-ffi-stubgen(field): ir.prim.Div.b -> PrimExpr
// tvm-ffi-stubgen(field): ir.prim.Mod.a -> PrimExpr
// tvm-ffi-stubgen(field): ir.prim.Mod.b -> PrimExpr
// tvm-ffi-stubgen(field): ir.prim.FloorDiv.a -> PrimExpr
// tvm-ffi-stubgen(field): ir.prim.FloorDiv.b -> PrimExpr
// tvm-ffi-stubgen(field): ir.prim.FloorMod.a -> PrimExpr
// tvm-ffi-stubgen(field): ir.prim.FloorMod.b -> PrimExpr
// tvm-ffi-stubgen(field): ir.prim.Min.a -> PrimExpr
// tvm-ffi-stubgen(field): ir.prim.Min.b -> PrimExpr
// tvm-ffi-stubgen(field): ir.prim.Max.a -> PrimExpr
// tvm-ffi-stubgen(field): ir.prim.Max.b -> PrimExpr
// tvm-ffi-stubgen(field): ir.prim.EQ.a -> PrimExpr
// tvm-ffi-stubgen(field): ir.prim.EQ.b -> PrimExpr
// tvm-ffi-stubgen(field): ir.prim.NE.a -> PrimExpr
// tvm-ffi-stubgen(field): ir.prim.NE.b -> PrimExpr
// tvm-ffi-stubgen(field): ir.prim.LT.a -> PrimExpr
// tvm-ffi-stubgen(field): ir.prim.LT.b -> PrimExpr
// tvm-ffi-stubgen(field): ir.prim.LE.a -> PrimExpr
// tvm-ffi-stubgen(field): ir.prim.LE.b -> PrimExpr
// tvm-ffi-stubgen(field): ir.prim.GT.a -> PrimExpr
// tvm-ffi-stubgen(field): ir.prim.GT.b -> PrimExpr
// tvm-ffi-stubgen(field): ir.prim.GE.a -> PrimExpr
// tvm-ffi-stubgen(field): ir.prim.GE.b -> PrimExpr
// tvm-ffi-stubgen(field): ir.prim.And.a -> PrimExpr
// tvm-ffi-stubgen(field): ir.prim.And.b -> PrimExpr
// tvm-ffi-stubgen(field): ir.prim.Or.a -> PrimExpr
// tvm-ffi-stubgen(field): ir.prim.Or.b -> PrimExpr
// tvm-ffi-stubgen(field): ir.prim.Not.a -> PrimExpr
// tvm-ffi-stubgen(field): ir.prim.Cast.value -> PrimExpr
// tvm-ffi-stubgen(field): ir.prim.Ramp.base -> PrimExpr
// tvm-ffi-stubgen(field): ir.prim.Ramp.stride -> PrimExpr
// tvm-ffi-stubgen(field): ir.prim.Ramp.lanes -> PrimExpr
// tvm-ffi-stubgen(field): ir.prim.Broadcast.value -> PrimExpr
// tvm-ffi-stubgen(field): ir.prim.Broadcast.lanes -> PrimExpr
// tvm-ffi-stubgen(field): ir.prim.Shuffle.vectors -> Array<PrimExpr>
// tvm-ffi-stubgen(field): ir.prim.Shuffle.indices -> Array<PrimExpr>
// tvm-ffi-stubgen(field): ir.prim.Select.condition -> PrimExpr
// tvm-ffi-stubgen(field): ir.prim.Select.true_value -> PrimExpr
// tvm-ffi-stubgen(field): ir.prim.Select.false_value -> PrimExpr
// tvm-ffi-stubgen(field): ir.prim.Let.value -> PrimExpr
// tvm-ffi-stubgen(field): ir.prim.Let.body -> PrimExpr
// tvm-ffi-stubgen(field): ir.prim.Add.ty -> PrimType
// tvm-ffi-stubgen(field): ir.prim.Sub.ty -> PrimType
// tvm-ffi-stubgen(field): ir.prim.Mul.ty -> PrimType
// tvm-ffi-stubgen(field): ir.prim.Div.ty -> PrimType
// tvm-ffi-stubgen(field): ir.prim.Mod.ty -> PrimType
// tvm-ffi-stubgen(field): ir.prim.FloorDiv.ty -> PrimType
// tvm-ffi-stubgen(field): ir.prim.FloorMod.ty -> PrimType
// tvm-ffi-stubgen(field): ir.prim.Min.ty -> PrimType
// tvm-ffi-stubgen(field): ir.prim.Max.ty -> PrimType
// tvm-ffi-stubgen(field): ir.prim.EQ.ty -> PrimType
// tvm-ffi-stubgen(field): ir.prim.NE.ty -> PrimType
// tvm-ffi-stubgen(field): ir.prim.LT.ty -> PrimType
// tvm-ffi-stubgen(field): ir.prim.LE.ty -> PrimType
// tvm-ffi-stubgen(field): ir.prim.GT.ty -> PrimType
// tvm-ffi-stubgen(field): ir.prim.GE.ty -> PrimType
// tvm-ffi-stubgen(field): ir.prim.And.ty -> PrimType
// tvm-ffi-stubgen(field): ir.prim.Or.ty -> PrimType
// tvm-ffi-stubgen(field): ir.prim.Not.ty -> PrimType
// tvm-ffi-stubgen(field): ir.prim.Cast.ty -> PrimType
// tvm-ffi-stubgen(field): ir.prim.Ramp.ty -> PrimType
// tvm-ffi-stubgen(field): ir.prim.Broadcast.ty -> PrimType
// tvm-ffi-stubgen(field): ir.prim.Shuffle.ty -> PrimType
// tvm-ffi-stubgen(field): ir.prim.Select.ty -> PrimType
// tvm-ffi-stubgen(field): ir.prim.Let.ty -> PrimType
// tvm-ffi-stubgen(field): ir.prim.StringImm.ty -> PrimType
// tvm-ffi-stubgen(upcast): ir.prim.Add -> PrimExpr
// tvm-ffi-stubgen(upcast): ir.prim.Sub -> PrimExpr
// tvm-ffi-stubgen(upcast): ir.prim.Mul -> PrimExpr
// tvm-ffi-stubgen(upcast): ir.prim.Div -> PrimExpr
// tvm-ffi-stubgen(upcast): ir.prim.Mod -> PrimExpr
// tvm-ffi-stubgen(upcast): ir.prim.FloorDiv -> PrimExpr
// tvm-ffi-stubgen(upcast): ir.prim.FloorMod -> PrimExpr
// tvm-ffi-stubgen(upcast): ir.prim.Min -> PrimExpr
// tvm-ffi-stubgen(upcast): ir.prim.Max -> PrimExpr
// tvm-ffi-stubgen(upcast): ir.prim.EQ -> PrimExpr
// tvm-ffi-stubgen(upcast): ir.prim.NE -> PrimExpr
// tvm-ffi-stubgen(upcast): ir.prim.LT -> PrimExpr
// tvm-ffi-stubgen(upcast): ir.prim.LE -> PrimExpr
// tvm-ffi-stubgen(upcast): ir.prim.GT -> PrimExpr
// tvm-ffi-stubgen(upcast): ir.prim.GE -> PrimExpr
// tvm-ffi-stubgen(upcast): ir.prim.And -> PrimExpr
// tvm-ffi-stubgen(upcast): ir.prim.Or -> PrimExpr
// tvm-ffi-stubgen(upcast): ir.prim.Not -> PrimExpr
// tvm-ffi-stubgen(upcast): ir.prim.Cast -> PrimExpr
// tvm-ffi-stubgen(upcast): ir.prim.Ramp -> PrimExpr
// tvm-ffi-stubgen(upcast): ir.prim.Broadcast -> PrimExpr
// tvm-ffi-stubgen(upcast): ir.prim.Shuffle -> PrimExpr
// tvm-ffi-stubgen(upcast): ir.prim.Select -> PrimExpr
// tvm-ffi-stubgen(upcast): ir.prim.Let -> PrimExpr
// tvm-ffi-stubgen(upcast): ir.prim.StringImm -> PrimExpr
// tvm-ffi-stubgen(custom-new): ir.prim.Add
// tvm-ffi-stubgen(custom-new): ir.prim.Sub
// tvm-ffi-stubgen(custom-new): ir.prim.Mul
// tvm-ffi-stubgen(custom-new): ir.prim.Div
// tvm-ffi-stubgen(custom-new): ir.prim.Mod
// tvm-ffi-stubgen(custom-new): ir.prim.FloorDiv
// tvm-ffi-stubgen(custom-new): ir.prim.FloorMod
// tvm-ffi-stubgen(custom-new): ir.prim.Min
// tvm-ffi-stubgen(custom-new): ir.prim.Max
// tvm-ffi-stubgen(custom-new): ir.prim.EQ
// tvm-ffi-stubgen(custom-new): ir.prim.NE
// tvm-ffi-stubgen(custom-new): ir.prim.LT
// tvm-ffi-stubgen(custom-new): ir.prim.LE
// tvm-ffi-stubgen(custom-new): ir.prim.GT
// tvm-ffi-stubgen(custom-new): ir.prim.GE
// tvm-ffi-stubgen(custom-new): ir.prim.And
// tvm-ffi-stubgen(custom-new): ir.prim.Or
// tvm-ffi-stubgen(custom-new): ir.prim.Not
// tvm-ffi-stubgen(custom-new): ir.prim.Cast
// tvm-ffi-stubgen(custom-new): ir.prim.Ramp
// tvm-ffi-stubgen(custom-new): ir.prim.Broadcast
// tvm-ffi-stubgen(custom-new): ir.prim.Shuffle
// tvm-ffi-stubgen(custom-new): ir.prim.Select
// tvm-ffi-stubgen(custom-new): ir.prim.Let
// tvm-ffi-stubgen(custom-new): ir.prim.StringImm
// tvm-ffi-stubgen(import-object): tvm_ffi::Array

// tvm-ffi-stubgen(begin): import-section
use super::super::ir::Expr;
use super::super::ir::ExprObj;
use super::super::ir::PrimType;
use super::super::ir::Span;
use super::super::ir::Type;
use super::super::ir::Var;
use crate::ir::PrimExpr;
use std::ops::Deref;
use tvm_ffi::Array;
use tvm_ffi::ObjectArc;
use tvm_ffi::String;
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/ir.prim.Add
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.prim.Add"]
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

// tvm-ffi-stubgen(begin): object/ir.prim.And
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.prim.And"]
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

// tvm-ffi-stubgen(begin): object/ir.prim.Broadcast
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.prim.Broadcast"]
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

// tvm-ffi-stubgen(begin): object/ir.prim.Cast
/// Complete: reflected fields fill [40, 48) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.prim.Cast"]
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

// tvm-ffi-stubgen(begin): object/ir.prim.Div
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.prim.Div"]
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

// tvm-ffi-stubgen(begin): object/ir.prim.EQ
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.prim.EQ"]
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

// tvm-ffi-stubgen(begin): object/ir.prim.FloorDiv
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.prim.FloorDiv"]
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

// tvm-ffi-stubgen(begin): object/ir.prim.FloorMod
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.prim.FloorMod"]
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

// tvm-ffi-stubgen(begin): object/ir.prim.GE
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.prim.GE"]
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

// tvm-ffi-stubgen(begin): object/ir.prim.GT
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.prim.GT"]
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

// tvm-ffi-stubgen(begin): object/ir.prim.LE
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.prim.LE"]
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

// tvm-ffi-stubgen(begin): object/ir.prim.LT
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.prim.LT"]
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

// tvm-ffi-stubgen(begin): object/ir.prim.Let
/// Complete: reflected fields fill [40, 64) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.prim.Let"]
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

// tvm-ffi-stubgen(begin): object/ir.prim.Max
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.prim.Max"]
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

// tvm-ffi-stubgen(begin): object/ir.prim.Min
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.prim.Min"]
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

// tvm-ffi-stubgen(begin): object/ir.prim.Mod
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.prim.Mod"]
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

// tvm-ffi-stubgen(begin): object/ir.prim.Mul
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.prim.Mul"]
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

// tvm-ffi-stubgen(begin): object/ir.prim.NE
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.prim.NE"]
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

// tvm-ffi-stubgen(begin): object/ir.prim.Not
/// Complete: reflected fields fill [40, 48) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.prim.Not"]
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

// tvm-ffi-stubgen(begin): object/ir.prim.Or
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.prim.Or"]
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

// tvm-ffi-stubgen(begin): object/ir.prim.Ramp
/// Complete: reflected fields fill [40, 64) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.prim.Ramp"]
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

// tvm-ffi-stubgen(begin): object/ir.prim.Select
/// Complete: reflected fields fill [40, 64) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.prim.Select"]
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

// tvm-ffi-stubgen(begin): object/ir.prim.Shuffle
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.prim.Shuffle"]
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

// tvm-ffi-stubgen(begin): object/ir.prim.StringImm
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.prim.StringImm"]
#[type_final]
pub struct StringImmObj {
    base: ExprObj,
    pub value: String,
}

const _: () = {
    assert!(::core::mem::size_of::<StringImmObj>() == 56);
    assert!(::core::mem::align_of::<StringImmObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct StringImm {
    base: ObjectArc<StringImmObj>,
}

impl Deref for StringImm {
    type Target = StringImmObj;
    fn deref(&self) -> &StringImmObj {
        &self.base
    }
}

impl Deref for StringImmObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl StringImmObj {
    pub(crate) fn new(span: Option<Span>, ty: PrimType, value: String) -> Self {
        let base = ExprObj::new(span, ty.into());
        Self { base, value }
    }
}

impl StringImm {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(span: Option<Span>, ty: PrimType, value: String) -> Self {
        let obj = StringImmObj::new(span, ty, value);
        Self {
            base: ObjectArc::new(obj),
        }
    }
}

tvm_ffi::impl_object_upcast!(StringImm => Expr, StringImm => PrimExpr);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/ir.prim.Sub
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.prim.Sub"]
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
    use crate::ir::{Expr, PrimExpr, PrimType, Span, Var};
    use tvm_ffi::{
        DLDataType, DLDataTypeCode, DLDataTypeExt, Error, ObjectRefCast, Result, String, TYPE_ERROR,
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
            if condition_type.dtype.lanes != 1
                && condition_type.dtype.lanes != result_type.dtype.lanes
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
            var.ty.clone().try_cast::<PrimType>()?;
            let value = PrimExpr::try_from(value)?;
            let body = PrimExpr::try_from(body)?;
            let same_type: bool = tvm_ffi::cached_global_func!("ffi.StructuralEqual")
                .call_tuple((&var.ty, &value.ty, false, false))?
                .try_into()?;
            if !same_type {
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

    impl StringImm {
        /// Construct a string literal directly in Rust.
        pub fn new(value: &str) -> Self {
            Self::with_span(value, None)
        }

        /// Construct a string literal with optional source metadata.
        pub fn with_span(value: &str, span: Option<&Span>) -> Self {
            let value_type = crate::ir::PrimType::void();
            Self::from_complete_fields(span.cloned(), value_type, String::from(value))
        }
    }
}
