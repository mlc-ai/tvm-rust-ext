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

//! Iteration variables: hand-written semantics for the generated `tirx.IterVar` binding.

use super::PrimVar;
use super::*;
use crate::ir::prim::primitive_type;
use crate::ir::{Range, Span, Var};
use tvm_ffi::{DLDataTypeCode, Error, ObjectRefCast, Result, String, TYPE_ERROR};

impl IterVar {
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
