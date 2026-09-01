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

use tvm_ffi::{Any, Map, ObjectRefCast, Result, String};

use crate::analysis::Analyzer;
use crate::ir::{DictAttrs, Expr, IntImm, PrimExpr, PrimType};
use crate::tirx::{PrimFunc, Stmt};

/// Lazily create the native analyzer only when a pass reaches a case that needs it.
#[derive(Default)]
pub(super) struct LazyAnalyzer(Option<Analyzer>);

impl LazyAnalyzer {
    pub(super) fn get(&mut self) -> Result<&Analyzer> {
        if self.0.is_none() {
            self.0 = Some(Analyzer::new()?);
        }
        Ok(self.0.as_ref().expect("analyzer was initialized above"))
    }
}

pub(super) fn int_value(expr: &Expr) -> Option<i64> {
    expr.clone()
        .try_cast::<IntImm>()
        .ok()
        .map(|value| value.value)
}

pub(super) fn with_prim_func_body(function: PrimFunc, body: Stmt) -> PrimFunc {
    PrimFunc::from_complete_fields(
        function.span.clone(),
        function.ty.clone(),
        function.attrs.clone(),
        function.params.clone(),
        function.ret_type.clone(),
        body,
    )
}

pub(super) fn with_prim_func_attr(
    function: PrimFunc,
    key: &str,
    value: impl Into<Any>,
) -> PrimFunc {
    let key = String::from(key);
    let mut attributes = function
        .attrs
        .dict
        .iter()
        .filter(|(existing, _)| existing.as_str() != key.as_str())
        .collect::<Vec<_>>();
    attributes.push((key, value.into()));
    let attrs = DictAttrs::from_dictionary(Map::from_iter(attributes));

    PrimFunc::from_complete_fields(
        function.span.clone(),
        function.ty.clone(),
        attrs,
        function.params.clone(),
        function.ret_type.clone(),
        function.body.clone(),
    )
}

pub(super) fn cast_prim_expr(value: PrimExpr, target: PrimType) -> Result<PrimExpr> {
    tvm_ffi::cached_global_func!("tirx.Cast")
        .call_tuple((target, value, Option::<crate::ir::Span>::None))?
        .try_into()
}
