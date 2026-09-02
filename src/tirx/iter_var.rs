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
    Any, DLDataTypeCode, Error, FieldGetter, ObjectArc, ObjectCore, ObjectRefCast, Result, String,
    TYPE_ERROR, VALUE_ERROR,
};

use super::{primitive_type, PrimVar};
use crate::ir::{PrimExprConvertible, PrimExprConvertibleObj, Range, Span, Var};

/// Scheduling role attached to a TIR iteration variable.
///
/// Keep unknown future native enumerators representable instead of creating an
/// invalid Rust enum discriminant.
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

/// Opaque Rust representation of a polymorphic TIR iteration variable.
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

/// Reference-counted handle to an iteration variable.
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

    // customized_new(IterVar) begin
    /// Construct an untagged data-parallel (`kDataPar`) iteration variable
    /// through its native constructor.
    ///
    /// Use [`IterVar::with_metadata`] for another iteration type, a thread
    /// tag, a missing domain, or a span.
    pub fn new<D, V>(domain: D, variable: V) -> Result<Self>
    where
        D: Into<Range>,
        V: Into<Var>,
    {
        Self::with_metadata(
            Some(domain.into()),
            variable.into(),
            IterVarType::kDataPar,
            "",
            None,
        )
    }
    // customized_new(IterVar) end

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

tvm_ffi::impl_object_upcast!(IterVar => PrimExprConvertible);
