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
use tvm_ffi::{Error, ObjectArc, ObjectRefCore, Result, String, VALUE_ERROR};

use super::{BufferTypeObj, PrimFunc};
use crate::ir::PointerTypeObj;

/// ABI-complete Rust representation of a tensor intrinsic.
#[repr(C)]
#[derive(Object)]
#[type_key = "tirx.TensorIntrin"]
#[type_final]
pub struct TensorIntrinObj {
    base: tvm_ffi::Object,
    pub desc: PrimFunc,
    pub implementation: PrimFunc,
}

/// Reference-counted handle to a tensor intrinsic.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct TensorIntrin {
    data: ObjectArc<TensorIntrinObj>,
}

impl std::ops::Deref for TensorIntrin {
    type Target = TensorIntrinObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl TensorIntrin {
    /// Construct a tensor intrinsic after applying TVM's parameter checks.
    pub fn new(desc: PrimFunc, implementation: PrimFunc) -> Result<Self> {
        if desc.params.len() != implementation.params.len() {
            return Err(Error::new(
                VALUE_ERROR,
                "tensor intrinsic description and implementation must have the same number of parameters",
                "",
            ));
        }
        for (desc_parameter, implementation_parameter) in
            desc.params.iter().zip(implementation.params.iter())
        {
            ensure_handle_parameter(&desc_parameter, "description")?;
            ensure_handle_parameter(&implementation_parameter, "implementation")?;
        }
        Ok(Self::from_complete_fields(desc, implementation))
    }

    /// Construct a tensor intrinsic from every physical field.
    pub fn from_complete_fields(desc: PrimFunc, implementation: PrimFunc) -> Self {
        Self {
            data: ObjectArc::new(TensorIntrinObj {
                base: tvm_ffi::Object::new(),
                desc,
                implementation,
            }),
        }
    }

    /// Register this intrinsic in TVM's process-wide tensor-intrinsic registry.
    pub fn register(&self, name: &str, overwrite: bool) -> Result<()> {
        tvm_ffi::cached_global_func!("tirx.TensorIntrinRegister").call_tuple((
            String::from(name),
            self,
            overwrite,
        ))?;
        Ok(())
    }

    /// Look up a registered tensor intrinsic.
    pub fn get(name: &str) -> Result<Self> {
        tvm_ffi::cached_global_func!("tirx.TensorIntrinGet")
            .call_tuple((String::from(name), false))?
            .try_into()
    }

    /// Look up a registered tensor intrinsic without raising for a missing name.
    pub fn try_get(name: &str) -> Result<Option<Self>> {
        tvm_ffi::cached_global_func!("tirx.TensorIntrinGet")
            .call_tuple((String::from(name), true))?
            .try_into()
    }
}

fn ensure_handle_parameter(parameter: &crate::ir::Var, owner: &str) -> Result<()> {
    if parameter.ty.as_node::<PointerTypeObj>().is_some()
        || parameter.ty.as_node::<BufferTypeObj>().is_some()
    {
        return Ok(());
    }
    Err(Error::new(
        VALUE_ERROR,
        &format!("tensor intrinsic {owner} parameters must be handles"),
        "",
    ))
}
