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
    Array, DLDataType, DLDataTypeCode, DLDataTypeExt, Error, ObjectArc, ObjectRefCast,
    ObjectRefCore, Result, String, TYPE_ERROR, VALUE_ERROR,
};

use crate::ir::{
    Call, CallObj, Expr, ExprObj, FloatImm, FloatImmObj, IntImm, IntImmObj, Op, PrimExpr, PrimType,
    Span, Var,
};

/// ABI-complete Rust representation of TVM's `AddNode`.
#[repr(C)]
#[derive(Object)]
#[type_key = "ir.prim.Add"]
#[type_final]
pub struct AddObj {
    base: ExprObj,
    pub a: PrimExpr,
    pub b: PrimExpr,
}

/// Reference-counted handle to an addition expression.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct Add {
    data: ObjectArc<AddObj>,
}

impl std::ops::Deref for Add {
    type Target = AddObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl std::ops::Deref for AddObj {
    type Target = ExprObj;

    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

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

    /// Construct an addition from every physical field without re-deriving its result type.
    pub fn from_complete_fields(
        span: Option<Span>,
        ty: PrimType,
        a: PrimExpr,
        b: PrimExpr,
    ) -> Self {
        Self {
            data: ObjectArc::new(AddObj {
                base: ExprObj::new(span, ty.into()),
                a,
                b,
            }),
        }
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
    ($object:ident, $reference:ident, $type_key:literal, $description:literal) => {
        #[doc = concat!("ABI-complete Rust representation of TVM's `", $type_key, "` node.")]
        #[repr(C)]
        #[derive(Object)]
        #[type_key = $type_key]
        #[type_final]
        pub struct $object {
            base: ExprObj,
            pub a: PrimExpr,
            pub b: PrimExpr,
        }

        #[doc = concat!("Reference-counted handle to ", $description, ".")]
        #[repr(C)]
        #[derive(ObjectRef, Clone)]
        pub struct $reference {
            data: ObjectArc<$object>,
        }

        impl std::ops::Deref for $reference {
            type Target = $object;

            fn deref(&self) -> &Self::Target {
                &self.data
            }
        }

        impl std::ops::Deref for $object {
            type Target = ExprObj;

            fn deref(&self) -> &Self::Target {
                &self.base
            }
        }

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

            /// Construct the expression from every physical field without re-deriving its type.
            pub fn from_complete_fields(
                span: Option<Span>,
                ty: PrimType,
                a: PrimExpr,
                b: PrimExpr,
            ) -> Self {
                Self {
                    data: ObjectArc::new($object {
                        base: ExprObj::new(span, ty.into()),
                        a,
                        b,
                    }),
                }
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

define_binary_expression!(SubObj, Sub, "ir.prim.Sub", "a subtraction expression");
define_binary_expression!(MulObj, Mul, "ir.prim.Mul", "a multiplication expression");
define_binary_expression!(
    DivObj,
    Div,
    "ir.prim.Div",
    "a truncating-division expression"
);
define_binary_expression!(
    ModObj,
    Mod,
    "ir.prim.Mod",
    "a truncating-remainder expression"
);
define_binary_expression!(
    FloorDivObj,
    FloorDiv,
    "ir.prim.FloorDiv",
    "a floor-division expression"
);
define_binary_expression!(
    FloorModObj,
    FloorMod,
    "ir.prim.FloorMod",
    "a floor-remainder expression"
);
define_binary_expression!(MinObj, Min, "ir.prim.Min", "a minimum expression");
define_binary_expression!(MaxObj, Max, "ir.prim.Max", "a maximum expression");

/// ABI-complete Rust representation of TVM's `ir.prim.EQ` node.
#[repr(C)]
#[derive(Object)]
#[type_key = "ir.prim.EQ"]
#[type_final]
pub struct EQObj {
    base: ExprObj,
    pub a: PrimExpr,
    pub b: PrimExpr,
}

/// Reference-counted handle to an equality comparison.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct EQ {
    data: ObjectArc<EQObj>,
}

impl std::ops::Deref for EQ {
    type Target = EQObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl std::ops::Deref for EQObj {
    type Target = ExprObj;

    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

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

    /// Construct a comparison from every physical field after external validation.
    pub fn from_complete_fields(
        span: Option<Span>,
        ty: PrimType,
        a: PrimExpr,
        b: PrimExpr,
    ) -> Self {
        Self {
            data: ObjectArc::new(EQObj {
                base: ExprObj::new(span, ty.into()),
                a,
                b,
            }),
        }
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
    ($object:ident, $reference:ident, $type_key:literal, $description:literal) => {
        #[doc = concat!("ABI-complete Rust representation of TVM's `", $type_key, "` node.")]
        #[repr(C)]
        #[derive(Object)]
        #[type_key = $type_key]
        #[type_final]
        pub struct $object {
            base: ExprObj,
            pub a: PrimExpr,
            pub b: PrimExpr,
        }

        #[doc = concat!("Reference-counted handle to ", $description, ".")]
        #[repr(C)]
        #[derive(ObjectRef, Clone)]
        pub struct $reference {
            data: ObjectArc<$object>,
        }

        impl std::ops::Deref for $reference {
            type Target = $object;

            fn deref(&self) -> &Self::Target {
                &self.data
            }
        }

        impl std::ops::Deref for $object {
            type Target = ExprObj;

            fn deref(&self) -> &Self::Target {
                &self.base
            }
        }

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

            /// Construct the comparison from every physical field.
            pub fn from_complete_fields(
                span: Option<Span>,
                ty: PrimType,
                a: PrimExpr,
                b: PrimExpr,
            ) -> Self {
                Self {
                    data: ObjectArc::new($object {
                        base: ExprObj::new(span, ty.into()),
                        a,
                        b,
                    }),
                }
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

define_comparison_expression!(NEObj, NE, "ir.prim.NE", "an inequality comparison");
define_comparison_expression!(LTObj, LT, "ir.prim.LT", "a less-than comparison");
define_comparison_expression!(LEObj, LE, "ir.prim.LE", "a less-than-or-equal comparison");
define_comparison_expression!(GTObj, GT, "ir.prim.GT", "a greater-than comparison");
define_comparison_expression!(
    GEObj,
    GE,
    "ir.prim.GE",
    "a greater-than-or-equal comparison"
);

/// ABI-complete Rust representation of TVM's `ir.prim.And` node.
#[repr(C)]
#[derive(Object)]
#[type_key = "ir.prim.And"]
#[type_final]
pub struct AndObj {
    base: ExprObj,
    pub a: PrimExpr,
    pub b: PrimExpr,
}

/// Reference-counted handle to a logical conjunction.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct And {
    data: ObjectArc<AndObj>,
}

impl std::ops::Deref for And {
    type Target = AndObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl std::ops::Deref for AndObj {
    type Target = ExprObj;

    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

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

    /// Construct a conjunction from every physical field after external validation.
    pub fn from_complete_fields(
        span: Option<Span>,
        ty: PrimType,
        a: PrimExpr,
        b: PrimExpr,
    ) -> Self {
        Self {
            data: ObjectArc::new(AndObj {
                base: ExprObj::new(span, ty.into()),
                a,
                b,
            }),
        }
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

/// ABI-complete Rust representation of TVM's `ir.prim.Or` node.
#[repr(C)]
#[derive(Object)]
#[type_key = "ir.prim.Or"]
#[type_final]
pub struct OrObj {
    base: ExprObj,
    pub a: PrimExpr,
    pub b: PrimExpr,
}

/// Reference-counted handle to a logical disjunction.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct Or {
    data: ObjectArc<OrObj>,
}

impl std::ops::Deref for Or {
    type Target = OrObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl std::ops::Deref for OrObj {
    type Target = ExprObj;

    fn deref(&self) -> &Self::Target {
        &self.base
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

    /// Construct a disjunction from every physical field after external validation.
    pub fn from_complete_fields(
        span: Option<Span>,
        ty: PrimType,
        a: PrimExpr,
        b: PrimExpr,
    ) -> Self {
        Self {
            data: ObjectArc::new(OrObj {
                base: ExprObj::new(span, ty.into()),
                a,
                b,
            }),
        }
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

/// ABI-complete Rust representation of TVM's `ir.prim.Not` node.
#[repr(C)]
#[derive(Object)]
#[type_key = "ir.prim.Not"]
#[type_final]
pub struct NotObj {
    base: ExprObj,
    pub a: PrimExpr,
}

/// Reference-counted handle to a logical negation.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct Not {
    data: ObjectArc<NotObj>,
}

impl std::ops::Deref for Not {
    type Target = NotObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl std::ops::Deref for NotObj {
    type Target = ExprObj;

    fn deref(&self) -> &Self::Target {
        &self.base
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

    /// Construct a negation from every physical field after external validation.
    pub fn from_complete_fields(span: Option<Span>, ty: PrimType, a: PrimExpr) -> Self {
        Self {
            data: ObjectArc::new(NotObj {
                base: ExprObj::new(span, ty.into()),
                a,
            }),
        }
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

/// ABI-complete Rust representation of TVM's `ir.prim.Cast` node.
#[repr(C)]
#[derive(Object)]
#[type_key = "ir.prim.Cast"]
#[type_final]
pub struct CastObj {
    base: ExprObj,
    pub value: PrimExpr,
}

/// Reference-counted handle to a primitive cast expression.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct Cast {
    data: ObjectArc<CastObj>,
}

impl std::ops::Deref for Cast {
    type Target = CastObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl std::ops::Deref for CastObj {
    type Target = ExprObj;

    fn deref(&self) -> &Self::Target {
        &self.base
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

    /// Construct a cast from every physical field after external validation.
    pub fn from_complete_fields(span: Option<Span>, ty: PrimType, value: PrimExpr) -> Self {
        Self {
            data: ObjectArc::new(CastObj {
                base: ExprObj::new(span, ty.into()),
                value,
            }),
        }
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

/// ABI-complete Rust representation of TVM's `ir.prim.Ramp` node.
#[repr(C)]
#[derive(Object)]
#[type_key = "ir.prim.Ramp"]
#[type_final]
pub struct RampObj {
    expr: ExprObj,
    pub base: PrimExpr,
    pub stride: PrimExpr,
    pub lanes: PrimExpr,
}

/// Reference-counted handle to a vector ramp expression.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct Ramp {
    data: ObjectArc<RampObj>,
}

impl std::ops::Deref for Ramp {
    type Target = RampObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl std::ops::Deref for RampObj {
    type Target = ExprObj;

    fn deref(&self) -> &Self::Target {
        &self.expr
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

    /// Construct a ramp from every physical field after external validation.
    pub fn from_complete_fields(
        span: Option<Span>,
        ty: PrimType,
        base: PrimExpr,
        stride: PrimExpr,
        lanes: PrimExpr,
    ) -> Self {
        Self {
            data: ObjectArc::new(RampObj {
                expr: ExprObj::new(span, ty.into()),
                base,
                stride,
                lanes,
            }),
        }
    }
}

/// ABI-complete Rust representation of TVM's `ir.prim.Broadcast` node.
#[repr(C)]
#[derive(Object)]
#[type_key = "ir.prim.Broadcast"]
#[type_final]
pub struct BroadcastObj {
    base: ExprObj,
    pub value: PrimExpr,
    pub lanes: PrimExpr,
}

/// Reference-counted handle to a vector broadcast expression.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct Broadcast {
    data: ObjectArc<BroadcastObj>,
}

impl std::ops::Deref for Broadcast {
    type Target = BroadcastObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl std::ops::Deref for BroadcastObj {
    type Target = ExprObj;

    fn deref(&self) -> &Self::Target {
        &self.base
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

    /// Construct a broadcast from every physical field after external validation.
    pub fn from_complete_fields(
        span: Option<Span>,
        ty: PrimType,
        value: PrimExpr,
        lanes: PrimExpr,
    ) -> Self {
        Self {
            data: ObjectArc::new(BroadcastObj {
                base: ExprObj::new(span, ty.into()),
                value,
                lanes,
            }),
        }
    }
}

/// ABI-complete Rust representation of TVM's `ir.prim.Shuffle` node.
#[repr(C)]
#[derive(Object)]
#[type_key = "ir.prim.Shuffle"]
#[type_final]
pub struct ShuffleObj {
    base: ExprObj,
    pub vectors: Array<PrimExpr>,
    pub indices: Array<PrimExpr>,
}

/// Reference-counted handle to a vector shuffle expression.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct Shuffle {
    data: ObjectArc<ShuffleObj>,
}

impl std::ops::Deref for Shuffle {
    type Target = ShuffleObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl std::ops::Deref for ShuffleObj {
    type Target = ExprObj;

    fn deref(&self) -> &Self::Target {
        &self.base
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
    pub fn extract_element(vector: PrimExpr, index: i32, span: Option<&Span>) -> Result<PrimExpr> {
        Ok(Self::with_span(
            Array::new(vec![vector]),
            Array::new(vec![IntImm::new("int32", i64::from(index))?.into()]),
            span,
        )?
        .into())
    }

    /// Construct a shuffle from every physical field after external validation.
    pub fn from_complete_fields(
        span: Option<Span>,
        ty: PrimType,
        vectors: Array<PrimExpr>,
        indices: Array<PrimExpr>,
    ) -> Self {
        Self {
            data: ObjectArc::new(ShuffleObj {
                base: ExprObj::new(span, ty.into()),
                vectors,
                indices,
            }),
        }
    }
}

// Ramp uses TVM's scalar cast semantics: fold literals, otherwise create a Cast.
fn cast_ramp_stride(ty: PrimType, value: PrimExpr) -> Result<PrimExpr> {
    let integer = value.as_node::<IntImmObj>().map(|literal| literal.value);
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
                    IntImm::from_dtype_with_span(word_type, (unsigned & 0xffff_ffff) as i64, span)?
                        .into(),
                    IntImm::from_dtype_with_span(word_type, (unsigned >> 32) as i64, span)?.into(),
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
                    .filter(|value| (-9223372036854775808.0..9223372036854775808.0).contains(value))
                    .map(|value| value as i64)
            })
            .ok_or_else(|| Error::new(VALUE_ERROR, "Stride literal is outside int64 range", ""))?;
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

fn vector_type_and_lanes(dtype: DLDataType, lanes: PrimExpr) -> Result<(PrimType, PrimExpr)> {
    if let Some(literal) = lanes.as_node::<IntImmObj>() {
        let count = literal.value as i32;
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
    let vscale = Op::get("ir.prim.vscale")?;
    let factor = lanes.as_node::<MulObj>().and_then(|multiply| {
        [(&multiply.a, &multiply.b), (&multiply.b, &multiply.a)]
            .into_iter()
            .find_map(|(constant, call)| {
                let constant = constant.as_node::<IntImmObj>()?;
                let call = call.as_node::<CallObj>()?;
                call.op.same_as(&vscale).then_some(constant.value as i32)
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

/// ABI-complete Rust representation of TVM's `SelectNode`.
#[repr(C)]
#[derive(Object)]
#[type_key = "ir.prim.Select"]
#[type_final]
pub struct SelectObj {
    base: ExprObj,
    pub condition: PrimExpr,
    pub true_value: PrimExpr,
    pub false_value: PrimExpr,
}

/// Reference-counted handle to a conditional primitive expression.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct Select {
    data: ObjectArc<SelectObj>,
}

impl std::ops::Deref for Select {
    type Target = SelectObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl std::ops::Deref for SelectObj {
    type Target = ExprObj;

    fn deref(&self) -> &Self::Target {
        &self.base
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
        if condition_type.dtype.lanes != 1 && condition_type.dtype.lanes != result_type.dtype.lanes
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

    /// Construct a select expression from every physical field after external validation.
    pub fn from_complete_fields(
        span: Option<Span>,
        ty: PrimType,
        condition: PrimExpr,
        true_value: PrimExpr,
        false_value: PrimExpr,
    ) -> Self {
        Self {
            data: ObjectArc::new(SelectObj {
                base: ExprObj::new(span, ty.into()),
                condition,
                true_value,
                false_value,
            }),
        }
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

/// ABI-complete Rust representation of TVM's `LetNode`.
#[repr(C)]
#[derive(Object)]
#[type_key = "ir.prim.Let"]
#[type_final]
pub struct LetObj {
    base: ExprObj,
    pub var: Var,
    pub value: PrimExpr,
    pub body: PrimExpr,
}

/// Reference-counted handle to a let expression.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct Let {
    data: ObjectArc<LetObj>,
}

impl std::ops::Deref for Let {
    type Target = LetObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl std::ops::Deref for LetObj {
    type Target = ExprObj;

    fn deref(&self) -> &Self::Target {
        &self.base
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

    /// Construct a let expression from every physical field after external validation.
    pub fn from_complete_fields(
        span: Option<Span>,
        ty: PrimType,
        var: Var,
        value: PrimExpr,
        body: PrimExpr,
    ) -> Self {
        Self {
            data: ObjectArc::new(LetObj {
                base: ExprObj::new(span, ty.into()),
                var,
                value,
                body,
            }),
        }
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

/// ABI-complete Rust representation of TVM's `StringImmNode`.
#[repr(C)]
#[derive(Object)]
#[type_key = "ir.prim.StringImm"]
#[type_final]
pub struct StringImmObj {
    base: ExprObj,
    pub value: String,
}

/// Reference-counted handle to a TIR string literal.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct StringImm {
    data: ObjectArc<StringImmObj>,
}

impl std::ops::Deref for StringImm {
    type Target = StringImmObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl std::ops::Deref for StringImmObj {
    type Target = ExprObj;

    fn deref(&self) -> &Self::Target {
        &self.base
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

    /// Construct a string literal from every physical field without re-deriving its type.
    pub fn from_complete_fields(span: Option<Span>, ty: PrimType, value: String) -> Self {
        Self {
            data: ObjectArc::new(StringImmObj {
                base: ExprObj::new(span, ty.into()),
                value,
            }),
        }
    }
}

tvm_ffi::impl_object_upcast!(
    Add => Expr,
    Add => PrimExpr,
    Sub => Expr,
    Sub => PrimExpr,
    Mul => Expr,
    Mul => PrimExpr,
    Div => Expr,
    Div => PrimExpr,
    Mod => Expr,
    Mod => PrimExpr,
    FloorDiv => Expr,
    FloorDiv => PrimExpr,
    FloorMod => Expr,
    FloorMod => PrimExpr,
    Min => Expr,
    Min => PrimExpr,
    Max => Expr,
    Max => PrimExpr,
    EQ => Expr,
    EQ => PrimExpr,
    NE => Expr,
    NE => PrimExpr,
    LT => Expr,
    LT => PrimExpr,
    LE => Expr,
    LE => PrimExpr,
    GT => Expr,
    GT => PrimExpr,
    GE => Expr,
    GE => PrimExpr,
    And => Expr,
    And => PrimExpr,
    Or => Expr,
    Or => PrimExpr,
    Not => Expr,
    Not => PrimExpr,
    Cast => Expr,
    Cast => PrimExpr,
    Ramp => Expr,
    Ramp => PrimExpr,
    Broadcast => Expr,
    Broadcast => PrimExpr,
    Shuffle => Expr,
    Shuffle => PrimExpr,
    Select => Expr,
    Select => PrimExpr,
    Let => Expr,
    Let => PrimExpr,
    StringImm => Expr,
    StringImm => PrimExpr,
);
