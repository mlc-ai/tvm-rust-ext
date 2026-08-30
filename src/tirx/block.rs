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
    Any, Array, DLDataTypeCode, Error, FieldGetter, Map, ObjectArc, ObjectCore, ObjectRefCast,
    Result, String, TYPE_ERROR, VALUE_ERROR,
};

use super::{primitive_type, BufferRegion, BufferVar, MatchBufferRegion, PrimVar, Stmt, StmtObj};
use crate::ir::{Expr, PrimExpr, PrimExprConvertible, PrimExprConvertibleObj, Range, Span, Var};

/// Scheduling role attached to a TIR block iteration variable.
///
/// Keep unknown future native enumerators representable instead of creating an
/// invalid Rust enum discriminant.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct IterVarType(i32);

#[allow(non_upper_case_globals)]
impl IterVarType {
    pub const DataParallel: Self = Self(0);
    pub const ThreadIndex: Self = Self(1);
    pub const CommutativeReduction: Self = Self(2);
    pub const Ordered: Self = Self(3);
    pub const Opaque: Self = Self(4);
    pub const Unrolled: Self = Self(5);
    pub const Vectorized: Self = Self(6);
    pub const Parallelized: Self = Self(7);
    pub const Tensorized: Self = Self(8);

    /// Preserve an enumerator not yet known by this Rust binding.
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
                &format!(
                    "tirx.IterVarType value {value} does not fit its native i32 representation"
                ),
                "",
            )
        })
    }
}

/// Opaque Rust representation of a polymorphic TIR block iteration variable.
///
/// The native class carries a C++ vtable, so Rust accesses its reflected fields
/// and asks the native constructor to allocate it instead of reproducing its bytes.
#[repr(C)]
#[derive(Object)]
#[type_key = "tirx.IterVar"]
#[type_final]
pub struct IterVarObj {
    base: PrimExprConvertibleObj,
}

/// Reference-counted handle to a block iteration variable.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct IterVar {
    data: ObjectArc<IterVarObj>,
}

impl std::ops::Deref for IterVar {
    type Target = IterVarObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl std::ops::Deref for IterVarObj {
    type Target = PrimExprConvertibleObj;

    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

impl IterVar {
    fn field<T>(&self, name: &str) -> Result<T>
    where
        T: TryFrom<Any, Error = Error>,
    {
        FieldGetter::new(IterVarObj::type_index(), name)?.get(&**self)
    }

    pub fn dom(&self) -> Result<Option<Range>> {
        self.field("dom")
    }

    pub fn var(&self) -> Result<PrimVar> {
        self.field("var")
    }

    pub fn iter_type(&self) -> Result<IterVarType> {
        let raw: i64 = self.field("iter_type")?;
        IterVarType::try_from(raw)
    }

    pub fn thread_tag(&self) -> Result<String> {
        self.field("thread_tag")
    }

    pub fn span(&self) -> Result<Option<Span>> {
        self.field("span")
    }

    /// Construct an untagged iteration variable through its native constructor.
    pub fn new<D, V>(domain: D, variable: V, iter_type: IterVarType) -> Result<Self>
    where
        D: Into<Range>,
        V: Into<Var>,
    {
        Self::with_metadata(Some(domain.into()), variable.into(), iter_type, "", None)
    }

    /// Validate and construct an iteration variable, allowing a missing domain
    /// for thread axes just like the native constructor.
    pub fn with_metadata(
        domain: Option<Range>,
        variable: Var,
        iter_type: IterVarType,
        thread_tag: &str,
        span: Option<&Span>,
    ) -> Result<Self> {
        validate_iter_var(domain.as_ref(), &variable)?;
        let variable = PrimVar::try_from(variable)?;
        tvm_ffi::cached_global_func!("tirx.IterVar")
            .call_tuple((
                domain,
                variable,
                i64::from(iter_type.as_raw()),
                String::from(thread_tag),
                span.cloned(),
            ))?
            .try_into()
    }
}

fn validate_iter_var(domain: Option<&Range>, variable: &Var) -> Result<()> {
    let variable_type = variable
        .ty
        .clone()
        .try_cast::<crate::ir::PrimType>()
        .map_err(|_| {
            Error::new(
                TYPE_ERROR,
                "IterVar variable must have a primitive type",
                "",
            )
        })?;
    if let Some(domain) = domain {
        let extent_type = primitive_type(&domain.extent, "IterVar domain extent")?;
        if extent_type.dtype.code != DLDataTypeCode::kDLInt as u8 {
            return Err(Error::new(
                TYPE_ERROR,
                "IterVar domain extent must have a signed integer type",
                "",
            ));
        }
        if extent_type.dtype != variable_type.dtype {
            return Err(Error::new(
                TYPE_ERROR,
                "IterVar domain extent type must match its variable type",
                "",
            ));
        }
    }
    Ok(())
}

/// ABI-complete Rust representation of TVM's current TIR block node.
#[repr(C)]
#[derive(Object)]
#[type_key = "tirx.SBlock"]
#[type_final]
pub struct SBlockObj {
    base: StmtObj,
    pub iter_vars: Array<IterVar>,
    pub reads: Array<BufferRegion>,
    pub writes: Array<BufferRegion>,
    pub name_hint: String,
    pub alloc_buffers: Array<BufferVar>,
    pub match_buffers: Array<MatchBufferRegion>,
    pub annotations: Map<String, Any>,
    pub init: Option<Stmt>,
    pub body: Stmt,
}

/// Reference-counted handle to a TIR scheduling block.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct SBlock {
    data: ObjectArc<SBlockObj>,
}

impl std::ops::Deref for SBlock {
    type Target = SBlockObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl std::ops::Deref for SBlockObj {
    type Target = StmtObj;

    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

impl SBlock {
    /// Construct a block with no axes, declared regions, or local buffers.
    pub fn new<B>(name_hint: &str, body: B) -> Self
    where
        B: Into<Stmt>,
    {
        Self::with_metadata(
            Vec::new(),
            Vec::new(),
            Vec::new(),
            name_hint,
            body.into(),
            None,
            Vec::new(),
            Vec::new(),
            Map::new(),
            None,
        )
    }

    /// Construct a scheduling block directly in Rust with all structural fields.
    #[allow(clippy::too_many_arguments)]
    pub fn with_metadata(
        iter_vars: Vec<IterVar>,
        reads: Vec<BufferRegion>,
        writes: Vec<BufferRegion>,
        name_hint: &str,
        body: Stmt,
        init: Option<Stmt>,
        allocated_buffers: Vec<BufferVar>,
        match_buffers: Vec<MatchBufferRegion>,
        annotations: Map<String, Any>,
        span: Option<&Span>,
    ) -> Self {
        Self::from_complete_fields(
            span.cloned(),
            Array::new(iter_vars),
            Array::new(reads),
            Array::new(writes),
            String::from(name_hint),
            Array::new(allocated_buffers),
            Array::new(match_buffers),
            annotations,
            init,
            body,
        )
    }

    /// Construct a scheduling block from every physical field.
    #[allow(clippy::too_many_arguments)]
    pub fn from_complete_fields(
        span: Option<Span>,
        iter_vars: Array<IterVar>,
        reads: Array<BufferRegion>,
        writes: Array<BufferRegion>,
        name_hint: String,
        alloc_buffers: Array<BufferVar>,
        match_buffers: Array<MatchBufferRegion>,
        annotations: Map<String, Any>,
        init: Option<Stmt>,
        body: Stmt,
    ) -> Self {
        Self {
            data: ObjectArc::new(SBlockObj {
                base: StmtObj::new(span),
                iter_vars,
                reads,
                writes,
                name_hint,
                alloc_buffers,
                match_buffers,
                annotations,
                init,
                body,
            }),
        }
    }
}

/// ABI-complete Rust representation of a block realization.
#[repr(C)]
#[derive(Object)]
#[type_key = "tirx.SBlockRealize"]
#[type_final]
pub struct SBlockRealizeObj {
    base: StmtObj,
    pub iter_values: Array<PrimExpr>,
    pub predicate: PrimExpr,
    pub block: SBlock,
}

/// Reference-counted handle to one realized scheduling block.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct SBlockRealize {
    data: ObjectArc<SBlockRealizeObj>,
}

impl std::ops::Deref for SBlockRealize {
    type Target = SBlockRealizeObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl std::ops::Deref for SBlockRealizeObj {
    type Target = StmtObj;

    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

impl SBlockRealize {
    /// Construct one realization of `block` directly in Rust.
    pub fn new<P, B>(iter_values: Vec<Expr>, predicate: P, block: B) -> Result<Self>
    where
        P: Into<Expr>,
        B: Into<SBlock>,
    {
        Self::with_span(iter_values, predicate, block, None)
    }

    /// Construct one realization with optional source metadata.
    pub fn with_span<P, B>(
        iter_values: Vec<Expr>,
        predicate: P,
        block: B,
        span: Option<&Span>,
    ) -> Result<Self>
    where
        P: Into<Expr>,
        B: Into<SBlock>,
    {
        let predicate = predicate.into();
        let block = block.into();
        if block.iter_vars.len() != iter_values.len() {
            return Err(Error::new(
                VALUE_ERROR,
                "SBlockRealize needs the same number of iter_vars and binding values",
                "",
            ));
        }
        for value in &iter_values {
            primitive_type(value, "SBlockRealize binding value")?;
        }
        let predicate_dtype = primitive_type(&predicate, "SBlockRealize predicate")?.dtype;
        if predicate_dtype.code != DLDataTypeCode::kDLBool as u8 {
            return Err(Error::new(
                TYPE_ERROR,
                "SBlockRealize predicate must have bool type",
                "",
            ));
        }
        let iter_values = iter_values
            .into_iter()
            .map(PrimExpr::try_from)
            .collect::<Result<Vec<_>>>()?;
        let predicate = PrimExpr::try_from(predicate)?;
        Ok(Self::from_complete_fields(
            span.cloned(),
            Array::new(iter_values),
            predicate,
            block,
        ))
    }

    /// Construct a block realization from every physical field after external validation.
    pub fn from_complete_fields(
        span: Option<Span>,
        iter_values: Array<PrimExpr>,
        predicate: PrimExpr,
        block: SBlock,
    ) -> Self {
        Self {
            data: ObjectArc::new(SBlockRealizeObj {
                base: StmtObj::new(span),
                iter_values,
                predicate,
                block,
            }),
        }
    }
}

tvm_ffi::impl_object_upcast!(
    IterVar => PrimExprConvertible,
    SBlock => Stmt,
    SBlockRealize => Stmt,
);
