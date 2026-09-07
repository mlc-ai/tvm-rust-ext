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

//! Thread-confined handle to TVM's native naming service.
//!
//! The naming state is owned by C++ and mutated through the registered
//! methods, so keep every alias of one supply on a single thread.

use std::{marker::PhantomData, rc::Rc};

use tvm_ffi::derive::{Object, ObjectRef};
use tvm_ffi::{ObjectArc, Result, String};

/// Opaque native name supply with private, mutable C++ storage.
#[repr(C)]
#[derive(Object)]
#[type_key = "ir.UniqueNameSupply"]
#[type_final]
pub struct UniqueNameSupplyObj {
    base: tvm_ffi::Object,
    _not_send_sync: PhantomData<Rc<()>>,
}

/// Shared naming state whose aliases must stay on one thread.
///
/// ```compile_fail
/// use tvm::ir::UniqueNameSupply;
/// fn require_send<T: Send>() {}
/// require_send::<UniqueNameSupply>();
/// ```
///
/// ```compile_fail
/// use tvm::ir::UniqueNameSupply;
/// fn require_sync<T: Sync>() {}
/// require_sync::<UniqueNameSupply>();
/// ```
///
/// Generic FFI object handles can erase these restrictions; that conversion
/// does not make the native naming state thread-safe.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct UniqueNameSupply {
    data: ObjectArc<UniqueNameSupplyObj>,
}

impl std::ops::Deref for UniqueNameSupply {
    type Target = UniqueNameSupplyObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl UniqueNameSupply {
    /// Create an empty name supply with an optional naming prefix.
    pub fn new(prefix: &str) -> Result<Self> {
        tvm_ffi::cached_global_func!("ir.UniqueNameSupply")
            .call_tuple((String::from(prefix),))?
            .try_into()
    }

    /// Reserve an existing name before generating new ones.
    pub fn reserve_name(&self, name: &str, add_prefix: bool) -> Result<String> {
        tvm_ffi::cached_global_func!("ir.UniqueNameSupply_ReserveName")
            .call_tuple((self, String::from(name), add_prefix))?
            .try_into()
    }

    /// Normalize the name and add a suffix if it has already been used.
    pub fn fresh_name(&self, name: &str, add_prefix: bool, add_underscore: bool) -> Result<String> {
        tvm_ffi::cached_global_func!("ir.UniqueNameSupply_FreshName")
            .call_tuple((self, String::from(name), add_prefix, add_underscore))?
            .try_into()
    }
}
