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

use tvm_ffi::{ObjectRefCast, Result};

use crate::analysis::Analyzer;
use crate::ir::{Expr, IntImm};

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
