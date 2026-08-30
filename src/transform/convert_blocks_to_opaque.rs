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

use std::collections::{HashMap, HashSet};

use tvm_ffi::{
    structural_mutate, Any, Array, DefRegionKind, Error, ObjectIdentity, ObjectRefCore, Result,
    StructuralMutator, RUNTIME_ERROR,
};

use super::{create_prim_func_pass, Pass};
use crate::ir::{PrimExpr, Var};
use crate::tirx::{
    BufferRegion, BufferVar, IterVar, MatchBufferRegion, PrimFunc, SBlock, SBlockRealize, Stmt,
};

#[derive(Default)]
struct OpaqueBlockConverter {
    substitutions: HashMap<ObjectIdentity, PrimExpr>,
    forbidden_iter_vars: HashSet<ObjectIdentity>,
}

/// Substitute block bindings and remove block iteration variables.
pub fn convert_blocks_to_opaque_prim_func(function: PrimFunc) -> Result<PrimFunc> {
    let mut converter = OpaqueBlockConverter::default();
    structural_mutate(function, &mut converter)?.try_into()
}

#[tvm_ffi::dispatch(mutate)]
impl OpaqueBlockConverter {
    fn mutate_variable(&mut self, value: Var, _region: DefRegionKind) -> Result<Any> {
        let identity = ObjectIdentity::of(&value);
        if self.forbidden_iter_vars.contains(&identity) {
            return Err(Error::new(
                RUNTIME_ERROR,
                "a block iteration variable occurs in its predicate or binding values",
                "",
            ));
        }
        Ok(self.substitutions.get(&identity).map_or_else(
            || Any::from(value),
            |replacement| Any::from(replacement.clone()),
        ))
    }

    fn mutate_block(&mut self, value: SBlock, region: DefRegionKind) -> Result<Any> {
        if value.init.is_some() {
            return Err(Error::new(
                RUNTIME_ERROR,
                "block init is not allowed in ConvertBlocksToOpaque",
                "",
            ));
        }
        // Block variables are definitions.  C++ visits those definitions but
        // does not substitute them with the values bound by SBlockRealize.
        let identities = value
            .iter_vars
            .iter()
            .map(|iter_var| iter_var.var().map(|var| ObjectIdentity::of(&var)))
            .collect::<Result<Vec<_>>>()?;
        let hidden = identities
            .iter()
            .filter_map(|identity| {
                self.substitutions
                    .remove(identity)
                    .map(|replacement| (identity.clone(), replacement))
            })
            .collect::<Vec<_>>();
        let iter_vars_result = self.mutate(&value.iter_vars, region);
        for (identity, replacement) in hidden {
            self.substitutions.insert(identity, replacement);
        }
        let iter_vars = Array::<IterVar>::try_from(iter_vars_result?)?;

        // Match StmtMutator's SBlock field order.  Name, annotations, and span
        // are metadata and are copied without recursive mutation.
        let alloc_buffers =
            Array::<BufferVar>::try_from(self.mutate(&value.alloc_buffers, region)?)?;
        let reads = Array::<BufferRegion>::try_from(self.mutate(&value.reads, region)?)?;
        let writes = Array::<BufferRegion>::try_from(self.mutate(&value.writes, region)?)?;
        let match_buffers =
            Array::<MatchBufferRegion>::try_from(self.mutate(&value.match_buffers, region)?)?;
        let body = Stmt::try_from(self.mutate(&value.body, region)?)?;

        if value.iter_vars.is_empty()
            && iter_vars.same_as(&value.iter_vars)
            && alloc_buffers.same_as(&value.alloc_buffers)
            && reads.same_as(&value.reads)
            && writes.same_as(&value.writes)
            && match_buffers.same_as(&value.match_buffers)
            && body.same_as(&value.body)
        {
            return Ok(Any::from(value));
        }

        Ok(Any::from(SBlock::from_complete_fields(
            value.span.clone(),
            Array::new(Vec::new()),
            reads,
            writes,
            value.name_hint.clone(),
            alloc_buffers,
            match_buffers,
            value.annotations.clone(),
            None,
            body,
        )))
    }

    fn mutate_block_realize(&mut self, value: SBlockRealize, region: DefRegionKind) -> Result<Any> {
        if value.block.init.is_some() {
            return Err(Error::new(
                RUNTIME_ERROR,
                "block init is not allowed in ConvertBlocksToOpaque",
                "",
            ));
        }

        let identities = value
            .block
            .iter_vars
            .iter()
            .map(|iter_var| iter_var.var().map(|var| ObjectIdentity::of(&var)))
            .collect::<Result<Vec<_>>>()?;
        for identity in &identities {
            self.forbidden_iter_vars.insert(identity.clone());
        }
        let initial_result: Result<(PrimExpr, Vec<PrimExpr>)> = (|| {
            let predicate = PrimExpr::try_from(self.mutate(&value.predicate, region)?)?;
            let iter_values = value
                .iter_values
                .iter()
                .map(|item| self.mutate(&item, region).and_then(PrimExpr::try_from))
                .collect::<Result<Vec<_>>>()?;
            Ok((predicate, iter_values))
        })();
        for identity in &identities {
            self.forbidden_iter_vars.remove(identity);
        }
        let (predicate, iter_values) = initial_result?;

        if identities.len() != iter_values.len() {
            return Err(Error::new(
                RUNTIME_ERROR,
                "block iter_vars and binding values must have the same length",
                "",
            ));
        }

        let bindings_result: Result<()> = (|| {
            for (identity, binding) in identities.iter().zip(&iter_values) {
                if self.substitutions.contains_key(identity) {
                    return Err(Error::new(
                        RUNTIME_ERROR,
                        "a block iteration variable is already active in an enclosing block",
                        "",
                    ));
                }
                let binding = PrimExpr::try_from(self.mutate(binding, region)?)?;
                self.substitutions.insert(identity.clone(), binding);
            }
            Ok(())
        })();
        if let Err(error) = bindings_result {
            for identity in &identities {
                self.substitutions.remove(identity);
            }
            return Err(error);
        }
        let block_result = self.mutate(&value.block, region);
        for identity in &identities {
            self.substitutions.remove(identity);
        }
        let block = SBlock::try_from(block_result?)?;

        let predicate_unchanged = predicate.same_as(&value.predicate);
        let bindings_unchanged = iter_values.len() == value.iter_values.len()
            && iter_values
                .iter()
                .zip(value.iter_values.iter())
                .all(|(mapped, original)| mapped.same_as(&original));
        if value.iter_values.is_empty()
            && predicate_unchanged
            && bindings_unchanged
            && block.same_as(&value.block)
        {
            Ok(Any::from(value))
        } else {
            Ok(Any::from(SBlockRealize::from_complete_fields(
                None,
                Array::new(Vec::new()),
                predicate,
                block,
            )))
        }
    }
}

/// Build TVM's `s_tir.ConvertBlocksToOpaque` PrimFunc pass in Rust.
pub fn convert_blocks_to_opaque() -> Result<Pass> {
    create_prim_func_pass(
        "s_tir.ConvertBlocksToOpaque",
        0,
        Vec::new(),
        false,
        |function, _module, _context| convert_blocks_to_opaque_prim_func(function),
    )
}
