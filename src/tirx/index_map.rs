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

//! Index maps: hand-written semantics for the generated `tirx.IndexMap` binding.

use super::PrimVar;
use super::*;
use crate::analysis::Analyzer;
use crate::ir::{Expr, PrimExpr, Range};
use tvm_ffi::{Any, Array, Result, Tensor};

impl IndexMap {
    /// Construct an index map directly in Rust.
    pub fn new(
        initial_indices: Vec<PrimVar>,
        final_indices: Vec<PrimExpr>,
        inverse_index_map: Option<IndexMap>,
    ) -> Self {
        Self::from_complete_fields(
            Array::new(initial_indices),
            Array::new(final_indices),
            inverse_index_map,
        )
    }

    /// Apply this mapping to concrete indices.
    pub fn map_indices(
        &self,
        indices: Vec<PrimExpr>,
        analyzer: Option<&Analyzer>,
    ) -> Result<Array<PrimExpr>> {
        tvm_ffi::cached_global_func!("tirx.IndexMapMapIndices")
            .call_tuple((self, Array::new(indices), analyzer.cloned()))?
            .try_into()
    }

    /// Compute the output shape produced by this mapping.
    pub fn map_shape(
        &self,
        shape: Vec<PrimExpr>,
        analyzer: Option<&Analyzer>,
    ) -> Result<Array<PrimExpr>> {
        tvm_ffi::cached_global_func!("tirx.IndexMapMapShape")
            .call_tuple((self, Array::new(shape), analyzer.cloned()))?
            .try_into()
    }

    /// Derive a bijective inverse over the supplied input ranges.
    pub fn inverse(&self, initial_ranges: Vec<Range>, analyzer: Option<&Analyzer>) -> Result<Self> {
        tvm_ffi::cached_global_func!("tirx.IndexMapInverse")
            .call_tuple((self, Array::new(initial_ranges), analyzer.cloned()))?
            .try_into()
    }

    /// Map a runtime tensor with this index transformation.
    pub fn map_tensor(&self, tensor: Tensor) -> Result<Tensor> {
        tvm_ffi::cached_global_func!("tirx.IndexMapMapTensor")
            .call_tuple((self, tensor))?
            .try_into()
    }

    /// Derive a possibly non-surjective inverse and its padding predicate.
    pub fn non_surjective_inverse(
        &self,
        initial_ranges: Vec<Range>,
        analyzer: Option<&Analyzer>,
    ) -> Result<(Self, PrimExpr)> {
        let result: Array<Any> = tvm_ffi::cached_global_func!("tirx.IndexMapNonSurjectiveInverse")
            .call_tuple((self, Array::new(initial_ranges), analyzer.cloned()))?
            .try_into()?;
        let inverse = IndexMap::try_from(result.get(0)?)?;
        let predicate = PrimExpr::try_from(Expr::try_from(result.get(1)?)?)?;
        Ok((inverse, predicate))
    }
}
