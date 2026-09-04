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

//! Minimal TE nodes required by TIRx expression passes.

use tvm_ffi::derive::{Object, ObjectRef};
use tvm_ffi::{Array, ObjectArc};

use crate::ir::{Expr, OpaqueExprObj, PrimExpr, PrimType, Span};
use crate::tirx::{IterVar, PrimVar};

/// ABI-complete Rust representation of TVM's TE `CommReducerNode`.
#[repr(C)]
#[derive(Object)]
#[type_key = "te.CommReducer"]
#[type_final]
pub struct CommReducerObj {
    base: tvm_ffi::Object,
    pub lhs: Array<PrimVar>,
    pub rhs: Array<PrimVar>,
    pub result: Array<PrimExpr>,
    pub identity_element: Array<PrimExpr>,
    pub span: Option<Span>,
}

/// Reference-counted handle to a commutative reducer definition.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct CommReducer {
    data: ObjectArc<CommReducerObj>,
}

impl std::ops::Deref for CommReducer {
    type Target = CommReducerObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl CommReducer {
    /// Construct a reducer from every physical field after external validation.
    pub fn from_complete_fields(
        lhs: Array<PrimVar>,
        rhs: Array<PrimVar>,
        result: Array<PrimExpr>,
        identity_element: Array<PrimExpr>,
        span: Option<Span>,
    ) -> Self {
        Self {
            data: ObjectArc::new(CommReducerObj {
                base: tvm_ffi::Object::new(),
                lhs,
                rhs,
                result,
                identity_element,
                span,
            }),
        }
    }
}

/// ABI-complete Rust representation of TVM's TE `ReduceNode`.
#[repr(C)]
#[derive(Object)]
#[type_key = "te.Reduce"]
#[type_final]
pub struct ReduceObj {
    base: OpaqueExprObj,
    pub combiner: CommReducer,
    pub source: Array<PrimExpr>,
    pub init: Array<PrimExpr>,
    pub axis: Array<IterVar>,
    pub condition: PrimExpr,
    pub value_index: i32,
}

/// Reference-counted handle to a reduction expression.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct Reduce {
    data: ObjectArc<ReduceObj>,
}

impl std::ops::Deref for Reduce {
    type Target = ReduceObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl std::ops::Deref for ReduceObj {
    type Target = OpaqueExprObj;

    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

impl Reduce {
    /// Construct a reduction from all physical fields after semantic validation.
    #[allow(clippy::too_many_arguments)]
    pub fn from_complete_fields(
        span: Option<Span>,
        ty: PrimType,
        combiner: CommReducer,
        source: Array<PrimExpr>,
        init: Array<PrimExpr>,
        axis: Array<IterVar>,
        condition: PrimExpr,
        value_index: i32,
    ) -> Self {
        Self {
            data: ObjectArc::new(ReduceObj {
                base: OpaqueExprObj::new(span, ty.into()),
                combiner,
                source,
                init,
                axis,
                condition,
                value_index,
            }),
        }
    }
}

tvm_ffi::impl_object_upcast!(
    Reduce => Expr,
    Reduce => PrimExpr,
);
