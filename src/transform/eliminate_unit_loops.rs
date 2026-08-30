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

use std::collections::HashMap;

use tvm_ffi::{
    structural_mutate, Any, AnyCompatible, DefRegionKind, Map, ObjectIdentity, Result, String,
    StructuralMutator,
};

use super::utils::int_value;
use crate::ir::{Expr, Var};
use crate::tirx::{For as TirFor, PrimFunc, Stmt};

#[derive(Default)]
struct UnitLoopEliminator {
    replacements: HashMap<ObjectIdentity, Expr>,
}

/// Eliminate unannotated unit loops and substitute their variables.
///
/// This is the unit-loop portion of C++ `tirx.transform.LowerTIRxOpaque`.
/// Substitution is based on object identity, so buffer indices and all other
/// uses of the loop variable receive the mapped loop minimum.
pub fn eliminate_unit_loops_prim_func(function: PrimFunc) -> Result<PrimFunc> {
    let mut mutator = UnitLoopEliminator::default();
    structural_mutate(function, &mut mutator)?.try_into()
}

#[tvm_ffi::dispatch(mutate)]
impl UnitLoopEliminator {
    fn mutate_loop(&mut self, value: TirFor, region: DefRegionKind) -> Result<Any> {
        let minimum = Expr::try_from(self.mutate(&value.min, region)?)?;
        let extent = Expr::try_from(self.mutate(&value.extent, region)?)?;
        let annotations = value.annotations.clone();
        let kind = value.kind;
        let should_eliminate = kind != crate::tirx::ForKind::ThreadBinding
            && int_value(&extent) == Some(1)
            && annotations.is_empty();

        if should_eliminate {
            let key = ObjectIdentity::of(&value.loop_var);
            let previous = self.replacements.insert(key.clone(), minimum.clone());
            let body_result = self.mutate(&value.body, region);
            match previous {
                Some(previous) => {
                    self.replacements.insert(key, previous);
                }
                None => {
                    self.replacements.remove(&key);
                }
            }
            return body_result;
        }

        let loop_var = Var::try_from(self.mutate(&value.loop_var, DefRegionKind::Recursive)?)?;
        let body = Stmt::try_from(self.mutate(&value.body, region)?)?;
        let thread_binding =
            Option::<crate::tirx::IterVar>::try_from(self.mutate(&value.thread_binding, region)?)?;
        let annotations = Map::<String, Any>::try_from(self.mutate(&annotations, region)?)?;
        let step = value
            .step
            .as_ref()
            .map(|step| self.mutate(step, region).and_then(Expr::try_from))
            .transpose()?;

        Ok(Any::from(TirFor::with_metadata(
            loop_var,
            minimum,
            extent,
            kind,
            body,
            thread_binding,
            annotations,
            step,
            value.span.as_ref(),
        )?))
    }

    fn mutate_variable(&mut self, value: Var, region: DefRegionKind) -> Result<Any> {
        if let Some(replacement) = self.replacements.get(&ObjectIdentity::of(&value)) {
            return Ok(replacement.to_any());
        }
        self.default_mutate_value(&value, region)
    }
}
