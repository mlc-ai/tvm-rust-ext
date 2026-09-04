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
use tvm_ffi::{Any, Array, ObjectArc, Result, Tensor};

use super::PrimVar;
use crate::analysis::Analyzer;
use crate::ir::{Expr, PrimExpr, Range};

/// ABI-complete Rust representation of an index transformation.
#[repr(C)]
#[derive(Object)]
#[type_key = "tirx.IndexMap"]
#[type_final]
pub struct IndexMapObj {
    base: tvm_ffi::Object,
    pub initial_indices: Array<PrimVar>,
    pub final_indices: Array<PrimExpr>,
    pub inverse_index_map: Option<IndexMap>,
}

/// Reference-counted handle to an index transformation.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct IndexMap {
    data: ObjectArc<IndexMapObj>,
}

impl std::ops::Deref for IndexMap {
    type Target = IndexMapObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

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

    /// Construct an index map from every physical field.
    pub fn from_complete_fields(
        initial_indices: Array<PrimVar>,
        final_indices: Array<PrimExpr>,
        inverse_index_map: Option<IndexMap>,
    ) -> Self {
        Self {
            data: ObjectArc::new(IndexMapObj {
                base: tvm_ffi::Object::new(),
                initial_indices,
                final_indices,
                inverse_index_map,
            }),
        }
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
