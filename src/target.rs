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
use tvm_ffi::{Any, Array, FieldGetter, Map, ObjectArc, ObjectCore, Result, String};

/// Opaque handle to TVM's canonical compilation-target object.
#[repr(C)]
#[derive(Object)]
#[type_key = "target.Target"]
#[type_final]
pub struct TargetObj {
    base: tvm_ffi::Object,
}

/// Reference-counted target handle managed by TVM.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct Target {
    data: ObjectArc<TargetObj>,
}

impl std::ops::Deref for Target {
    type Target = TargetObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl Target {
    /// Return the target active in TVM's current target scope.
    pub fn current(allow_none: bool) -> Result<Option<Self>> {
        tvm_ffi::cached_global_func!("target.TargetCurrent")
            .call_tuple((allow_none,))?
            .try_into()
    }

    /// Parse and canonicalize a target specification with TVM.
    pub fn new(specification: &str) -> Result<Self> {
        tvm_ffi::cached_global_func!("target.Target")
            .call_tuple((String::from(specification),))?
            .try_into()
    }

    /// Construct and canonicalize a target configuration with TVM.
    pub fn from_config(config: Map<String, Any>) -> Result<Self> {
        tvm_ffi::cached_global_func!("target.Target")
            .call_tuple((config,))?
            .try_into()
    }

    /// Export the canonical target as its language-independent configuration.
    pub fn export(&self) -> Result<Map<String, Any>> {
        tvm_ffi::cached_global_func!("target.TargetExport")
            .call_tuple((self,))?
            .try_into()
    }

    /// Return the configured target kind, such as `llvm` or `cuda`.
    pub fn kind_name(&self) -> Result<String> {
        self.export()?
            .get(&String::from("kind"))?
            .ok_or_else(|| {
                tvm_ffi::Error::new(tvm_ffi::VALUE_ERROR, "canonical target has no kind", "")
            })?
            .try_into()
    }

    /// Return whether this target advertises `key` in its canonical key list.
    pub fn has_key(&self, key: &str) -> Result<bool> {
        let keys: Array<String> =
            FieldGetter::new(TargetObj::type_index(), "keys")?.get(&**self)?;
        Ok(keys.iter().any(|candidate| candidate.as_str() == key))
    }

    /// Return the host target, if one is attached.
    pub fn host(&self) -> Result<Option<Self>> {
        FieldGetter::new(TargetObj::type_index(), "host")?.get(&**self)
    }

    /// Return the same target with its host component removed.
    pub fn without_host(&self) -> Result<Self> {
        let config = self.export()?;
        Self::from_config(Map::from_iter(
            config.iter().filter(|(key, _)| key.as_str() != "host"),
        ))
    }

    /// Attach `host` using TVM's canonical target operation.
    pub fn with_host(&self, host: &Self) -> Result<Self> {
        tvm_ffi::cached_global_func!("target.WithHost")
            .call_tuple((self, host))?
            .try_into()
    }

    /// Return TVM's runtime device-type number for this target.
    pub fn device_type(&self) -> Result<i32> {
        let raw: i64 = tvm_ffi::cached_global_func!("target.TargetGetDeviceType")
            .call_tuple((self,))?
            .try_into()?;
        i32::try_from(raw).map_err(|_| {
            tvm_ffi::Error::new(
                tvm_ffi::VALUE_ERROR,
                "target device type does not fit i32",
                "",
            )
        })
    }
}
