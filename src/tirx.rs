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
    Any, Array, DLDataType, DLDataTypeCode, DLDataTypeExt, Error, Map, ObjectArc, ObjectRefCast,
    Result, String, TYPE_ERROR, VALUE_ERROR,
};

use crate::ir::{
    BaseFuncObj, DictAttrs, Expr, ExprObj, IntImm, PointerType, PrimExpr, PrimType, Span,
    TupleType, Type, TypedVar, Var,
};

mod block;
mod buffer;

pub use block::{
    IterVar, IterVarObj, IterVarType, SBlock, SBlockObj, SBlockRealize, SBlockRealizeObj,
};
pub use buffer::{
    Axis, AxisObj, BufferLoad, BufferLoadObj, BufferRegion, BufferRegionObj, BufferStore,
    BufferStoreObj, BufferType, BufferTypeObj, BufferVar, Iter, IterObj, Layout, LayoutObj,
    MatchBufferRegion, MatchBufferRegionObj, TileLayout, TileLayoutObj,
};

/// Checked scalar view over a `Var` whose expression type is `PrimType`.
pub type PrimVar = TypedVar<PrimType>;

tvm_ffi::impl_try_from_any!(PrimVar);
tvm_ffi::impl_arg_into_ref!(PrimVar);
tvm_ffi::impl_into_arg_holder_default!(PrimVar);
/// ABI-complete Rust representation of TVM's `AddNode`.
#[repr(C)]
#[derive(Object)]
#[type_key = "tirx.Add"]
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
        }
    };
}

define_binary_expression!(SubObj, Sub, "tirx.Sub", "a subtraction expression");
define_binary_expression!(MulObj, Mul, "tirx.Mul", "a multiplication expression");

/// ABI-complete Rust representation of TVM's `tirx.EQ` node.
#[repr(C)]
#[derive(Object)]
#[type_key = "tirx.EQ"]
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
}

/// ABI-complete Rust representation of TVM's `tirx.And` node.
#[repr(C)]
#[derive(Object)]
#[type_key = "tirx.And"]
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
}

/// ABI-complete Rust representation of TVM's `StringImmNode`.
#[repr(C)]
#[derive(Object)]
#[type_key = "tirx.StringImm"]
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

/// ABI-complete Rust representation of TVM's `StmtNode` prefix.
#[repr(C)]
#[derive(Object)]
#[type_key = "tirx.Stmt"]
pub struct StmtObj {
    base: tvm_ffi::Object,
    pub span: Option<Span>,
}

/// Reference-counted handle to any TIR statement.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct Stmt {
    data: ObjectArc<StmtObj>,
}

/// ABI-complete Rust representation of TVM's `AttrStmtNode`.
#[repr(C)]
#[derive(Object)]
#[type_key = "tirx.AttrStmt"]
#[type_final]
pub struct AttrStmtObj {
    base: StmtObj,
    pub node: Any,
    pub attr_key: String,
    pub value: PrimExpr,
    pub body: Stmt,
}

/// Reference-counted handle to a scoped TIR attribute.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct AttrStmt {
    data: ObjectArc<AttrStmtObj>,
}

impl std::ops::Deref for AttrStmt {
    type Target = AttrStmtObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl std::ops::Deref for AttrStmtObj {
    type Target = StmtObj;

    fn deref(&self) -> &Self::Target {
        &self.base
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
            PrimExpr::try_from(value.into())?,
            body.into(),
        ))
    }

    /// Construct an attribute statement from every physical field.
    pub fn from_complete_fields(
        span: Option<Span>,
        node: Any,
        attr_key: String,
        value: PrimExpr,
        body: Stmt,
    ) -> Self {
        Self {
            data: ObjectArc::new(AttrStmtObj {
                base: StmtObj::new(span),
                node,
                attr_key,
                value,
                body,
            }),
        }
    }
}

impl std::ops::Deref for Stmt {
    type Target = StmtObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl StmtObj {
    pub(crate) fn new(span: Option<Span>) -> Self {
        Self {
            base: tvm_ffi::Object::new(),
            span,
        }
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

/// ABI-complete Rust representation of TVM's `AssertStmtNode`.
#[repr(C)]
#[derive(Object)]
#[type_key = "tirx.AssertStmt"]
#[type_final]
pub struct AssertStmtObj {
    base: StmtObj,
    pub condition: PrimExpr,
    pub error_kind: StringImm,
    pub message_parts: Array<StringImm>,
}

/// Reference-counted handle to a TIR assertion.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct AssertStmt {
    data: ObjectArc<AssertStmtObj>,
}

impl std::ops::Deref for AssertStmt {
    type Target = AssertStmtObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl std::ops::Deref for AssertStmtObj {
    type Target = StmtObj;

    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

/// ABI-complete Rust representation of TVM's `EvaluateNode`.
#[repr(C)]
#[derive(Object)]
#[type_key = "tirx.Evaluate"]
#[type_final]
pub struct EvaluateObj {
    base: StmtObj,
    pub value: Expr,
}

/// Reference-counted handle to a TIR evaluate statement.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct Evaluate {
    data: ObjectArc<EvaluateObj>,
}

impl std::ops::Deref for Evaluate {
    type Target = EvaluateObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl std::ops::Deref for EvaluateObj {
    type Target = StmtObj;

    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

/// ABI-complete Rust representation of TVM's `SeqStmtNode`.
#[repr(C)]
#[derive(Object)]
#[type_key = "tirx.SeqStmt"]
#[type_final]
pub struct SeqStmtObj {
    base: StmtObj,
    pub seq: Array<Stmt>,
}

/// Reference-counted handle to a sequence of statements.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct SeqStmt {
    data: ObjectArc<SeqStmtObj>,
}

impl std::ops::Deref for SeqStmt {
    type Target = SeqStmtObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl std::ops::Deref for SeqStmtObj {
    type Target = StmtObj;

    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

impl SeqStmt {
    /// Construct and recursively flatten a sequence directly in Rust.
    pub fn new(statements: Vec<Stmt>) -> Result<Self> {
        Self::with_span(statements, None)
    }

    /// Construct and recursively flatten a sequence with source metadata.
    pub fn with_span(statements: Vec<Stmt>, span: Option<&Span>) -> Result<Self> {
        let requires_flattening = statements
            .iter()
            .any(|statement| statement.clone().try_cast::<SeqStmt>().is_ok());
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

    /// Construct a sequence from its already-normalized physical fields.
    pub fn from_complete_fields(span: Option<Span>, seq: Array<Stmt>) -> Self {
        Self {
            data: ObjectArc::new(SeqStmtObj {
                base: StmtObj::new(span),
                seq,
            }),
        }
    }
}

fn flatten_statement(statement: Stmt, output: &mut Vec<Stmt>) {
    match statement.clone().try_cast::<SeqStmt>() {
        Ok(sequence) => {
            for child in sequence.seq.iter() {
                flatten_statement(child, output);
            }
        }
        Err(_) if !is_evaluate_zero(&statement) => output.push(statement),
        Err(_) => {}
    }
}

fn is_evaluate_zero(statement: &Stmt) -> bool {
    statement
        .clone()
        .try_cast::<Evaluate>()
        .ok()
        .and_then(|evaluate| evaluate.value.clone().try_cast::<IntImm>().ok())
        .is_some_and(|literal| literal.value == 0)
}

/// ABI-complete Rust representation of TVM's `IfThenElseNode`.
#[repr(C)]
#[derive(Object)]
#[type_key = "tirx.IfThenElse"]
#[type_final]
pub struct IfThenElseObj {
    base: StmtObj,
    pub condition: PrimExpr,
    pub then_case: Stmt,
    pub else_case: Option<Stmt>,
}

/// Reference-counted handle to a conditional statement.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct IfThenElse {
    data: ObjectArc<IfThenElseObj>,
}

impl std::ops::Deref for IfThenElse {
    type Target = IfThenElseObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl std::ops::Deref for IfThenElseObj {
    type Target = StmtObj;

    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

impl IfThenElse {
    /// Construct a conditional statement directly in Rust.
    pub fn new<C, T>(condition: C, then_case: T, else_case: Option<Stmt>) -> Result<Self>
    where
        C: Into<Expr>,
        T: Into<Stmt>,
    {
        Self::with_span(condition, then_case, else_case, None)
    }

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

    /// Construct a conditional statement from every physical field after external validation.
    pub fn from_complete_fields(
        span: Option<Span>,
        condition: PrimExpr,
        then_case: Stmt,
        else_case: Option<Stmt>,
    ) -> Self {
        Self {
            data: ObjectArc::new(IfThenElseObj {
                base: StmtObj::new(span),
                condition,
                then_case,
                else_case,
            }),
        }
    }
}

/// Execution policy attached to a TIR `For` loop.
///
/// This is an open integer newtype rather than a Rust enum: reading a newer
/// C++ enumerator through an older generated binding must remain memory-safe.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct ForKind(i32);

#[allow(non_upper_case_globals)]
impl ForKind {
    pub const Serial: Self = Self(0);
    pub const Parallel: Self = Self(1);
    pub const Vectorized: Self = Self(2);
    pub const Unrolled: Self = Self(3);
    pub const ThreadBinding: Self = Self(4);

    /// Preserve an enumerator not yet known by this Rust binding.
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
                &format!("tirx.ForKind value {value} does not fit its native i32 representation"),
                "",
            )
        })
    }
}

/// ABI-complete Rust representation of TVM's `ForNode`.
#[repr(C)]
#[derive(Object)]
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

/// Reference-counted handle to a TIR loop.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct For {
    data: ObjectArc<ForObj>,
}

impl std::ops::Deref for For {
    type Target = ForObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl std::ops::Deref for ForObj {
    type Target = StmtObj;

    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

impl For {
    /// Construct a serial loop with no thread binding, annotations, or custom step.
    pub fn serial<V, M, E, B>(loop_var: V, minimum: M, extent: E, body: B) -> Result<Self>
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
            ForKind::Serial,
            body.into(),
            None,
            Map::new(),
            None,
            None,
        )
    }

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

    /// Construct a loop from every physical field after external validation.
    #[allow(clippy::too_many_arguments)]
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
        Self {
            data: ObjectArc::new(ForObj {
                base: StmtObj::new(span),
                loop_var,
                min,
                extent,
                kind,
                body,
                thread_binding,
                annotations,
                step,
            }),
        }
    }
}

fn require_scalar_integer(value: &Expr, field: &str) -> Result<DLDataType> {
    let dtype = primitive_type(value, field)?.dtype;
    let is_integer =
        dtype.code == DLDataTypeCode::kDLInt as u8 || dtype.code == DLDataTypeCode::kDLUInt as u8;
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
    if let Ok(literal) = value.clone().try_cast::<IntImm>() {
        return PrimExpr::try_from(Expr::from(IntImm::from_dtype(loop_dtype, literal.value)?));
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

/// ABI-complete Rust representation of TVM's `PrimFuncNode`.
#[repr(C)]
#[derive(Object)]
#[type_key = "tirx.PrimFunc"]
#[type_final]
pub struct PrimFuncObj {
    base: BaseFuncObj,
    pub params: Array<Var>,
    pub ret_type: crate::ir::Type,
    pub body: Stmt,
}

/// Reference-counted handle to a TIR primitive function.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct PrimFunc {
    data: ObjectArc<PrimFuncObj>,
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

    /// Construct an assertion from every physical field after external validation.
    pub fn from_complete_fields(
        span: Option<Span>,
        condition: PrimExpr,
        error_kind: StringImm,
        message_parts: Array<StringImm>,
    ) -> Self {
        Self {
            data: ObjectArc::new(AssertStmtObj {
                base: StmtObj::new(span),
                condition,
                error_kind,
                message_parts,
            }),
        }
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
        if value.clone().try_cast::<Var>().is_ok()
            && value.ty.clone().try_cast::<buffer::BufferType>().is_ok()
        {
            return Err(Error::new(
                VALUE_ERROR,
                "a buffer variable cannot be used as a scalar Evaluate value",
                "",
            ));
        }
        Ok(Self::from_complete_fields(span.cloned(), value))
    }

    /// Construct an evaluation statement from every physical field after external validation.
    pub fn from_complete_fields(span: Option<Span>, value: Expr) -> Self {
        Self {
            data: ObjectArc::new(EvaluateObj {
                base: StmtObj::new(span),
                value,
            }),
        }
    }

    /// Construct `Evaluate(IntImm("int32", value))`.
    pub fn from_i64(value: i64) -> Result<Self> {
        Self::new(IntImm::new("int32", value)?)
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

    /// Construct a PrimFunc allocation entirely in Rust from its complete state.
    ///
    /// `function_type` is the native function type stored in the inherited
    /// `ExprObj::ty` field. Supplying it explicitly keeps this raw
    /// constructor lossless; [`PrimFunc::new`] derives it before allocation.
    #[allow(clippy::too_many_arguments)]
    pub fn from_complete_fields(
        span: Option<Span>,
        ty: crate::ir::Type,
        attrs: DictAttrs,
        params: Array<Var>,
        ret_type: crate::ir::Type,
        body: Stmt,
    ) -> Self {
        Self {
            data: ObjectArc::new(PrimFuncObj {
                base: BaseFuncObj::new(span, ty, attrs),
                params,
                ret_type,
                body,
            }),
        }
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
        let parameter_type = if let Ok(buffer) = parameter.ty.clone().try_cast::<BufferType>() {
            let mut shape = Vec::with_capacity(buffer.shape.len());
            for dimension in buffer.shape.iter() {
                shape.push(cast_index_to_i64(dimension.into())?);
            }
            let shape = make_native_shape_expr(Array::new(shape))?;
            make_native_tensor_type(shape, buffer.dtype.clone())?
        } else if parameter.ty.clone().try_cast::<PointerType>().is_ok() {
            make_native_any_type()?
        } else {
            parameter.ty.clone()
        };
        parameter_types.push(parameter_type);
    }

    let relax_return_type = if ret_type.clone().try_cast::<PrimType>().is_ok() {
        ret_type.clone()
    } else if ret_type
        .clone()
        .try_cast::<TupleType>()
        .is_ok_and(|tuple| tuple.fields.is_empty())
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
    if let Ok(literal) = value.clone().try_cast::<IntImm>() {
        return Ok(
            IntImm::from_complete_fields(literal.span.clone(), target, literal.value).into(),
        );
    }
    tvm_ffi::cached_global_func!("tirx.Cast")
        .call_tuple((target, value, Option::<Span>::None))?
        .try_into()
}

tvm_ffi::impl_object_upcast!(
    Add => Expr,
    Add => PrimExpr,
    Sub => Expr,
    Sub => PrimExpr,
    Mul => Expr,
    Mul => PrimExpr,
    EQ => Expr,
    EQ => PrimExpr,
    And => Expr,
    And => PrimExpr,
    StringImm => Expr,
    StringImm => PrimExpr,
    AttrStmt => Stmt,
    For => Stmt,
    AssertStmt => Stmt,
    Evaluate => Stmt,
    SeqStmt => Stmt,
    IfThenElse => Stmt,
    PrimFunc => crate::ir::BaseFunc,
    PrimFunc => Expr,
);
