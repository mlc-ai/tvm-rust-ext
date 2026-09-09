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

//! Locate and load the TVM shared libraries installed by the `apache-tvm` pip
//! package.
//!
//! This is the Rust counterpart of `python/tvm/libinfo.py`.  The crate never
//! needs a TVM source tree: `build.rs` records where the active Python
//! environment keeps `libtvm_compiler.so`, and this module resolves the
//! library at run time, honouring the same overrides as TVM's Python package:
//!
//! - `TVM_COMPILER_LIBRARY`: explicit path to the compiler library.
//! - `TVM_LIBRARY_PATH`: directory searched before the build-time default.
//!
//! `libtvm_ffi` itself is linked by the `tvm-ffi` crate; loading the compiler
//! library registers TVM's IR, TIR, and pass functions in the shared
//! `tvm-ffi` global registry, which is all the bindings in this crate need.

use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use tvm_ffi::Module;

/// File name of the TVM compiler library on this platform.
pub const COMPILER_LIBRARY: &str = if cfg!(target_os = "windows") {
    "tvm_compiler.dll"
} else if cfg!(target_os = "macos") {
    "libtvm_compiler.dylib"
} else {
    "libtvm_compiler.so"
};

/// File name of the TVM runtime library on this platform.
pub const RUNTIME_LIBRARY: &str = if cfg!(target_os = "windows") {
    "tvm_runtime.dll"
} else if cfg!(target_os = "macos") {
    "libtvm_runtime.dylib"
} else {
    "libtvm_runtime.so"
};

/// Directory holding the TVM libraries that `build.rs` found through the
/// installed `tvm` Python package, if discovery succeeded.
pub fn build_time_library_dir() -> Option<&'static Path> {
    option_env!("TVM_LIBRARY_DIR").map(Path::new)
}

/// Directory holding `libtvm_ffi` as reported by `tvm-ffi-config --libdir` at
/// build time, if discovery succeeded.
pub fn tvm_ffi_library_dir() -> Option<&'static Path> {
    option_env!("TVM_FFI_LIBRARY_DIR").map(Path::new)
}

/// Directories searched for TVM libraries, highest priority first.
pub fn library_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Some(dir) = std::env::var_os("TVM_LIBRARY_PATH") {
        dirs.push(PathBuf::from(dir));
    }
    if let Some(dir) = build_time_library_dir() {
        dirs.push(dir.to_path_buf());
    }
    dirs
}

/// Error raised when a TVM library cannot be located.
#[derive(Debug)]
pub struct LibraryNotFound {
    /// File name that was searched for.
    pub file_name: String,
    /// Directories that were searched, in order.
    pub searched: Vec<PathBuf>,
}

impl fmt::Display for LibraryNotFound {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "cannot find {}", self.file_name)?;
        if self.searched.is_empty() {
            write!(
                f,
                "; no search directory is known (the apache-tvm pip package was not found when \
                 this crate was built, and TVM_LIBRARY_PATH is unset)"
            )
        } else {
            write!(f, "; searched:")?;
            for dir in &self.searched {
                write!(f, "\n  {}", dir.display())?;
            }
            Ok(())
        }
    }
}

impl std::error::Error for LibraryNotFound {}

/// Error raised when the TVM compiler library cannot be located or loaded.
#[derive(Debug)]
pub enum LoadError {
    /// The library file could not be located.
    NotFound(LibraryNotFound),
    /// The library was found but `tvm-ffi` failed to load it.
    Load {
        /// Path that was passed to the loader.
        path: PathBuf,
        /// Error reported by `tvm-ffi`.
        source: tvm_ffi::Error,
    },
}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LoadError::NotFound(error) => error.fmt(f),
            LoadError::Load { path, source } => {
                write!(f, "failed to load {}: {}", path.display(), source)
            }
        }
    }
}

impl std::error::Error for LoadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            LoadError::NotFound(error) => Some(error),
            LoadError::Load { source, .. } => Some(source),
        }
    }
}

impl From<LibraryNotFound> for LoadError {
    fn from(error: LibraryNotFound) -> Self {
        LoadError::NotFound(error)
    }
}

/// Resolve `file_name` against [`library_dirs`].
pub fn find_library(file_name: &str) -> Result<PathBuf, LibraryNotFound> {
    let searched = library_dirs();
    searched
        .iter()
        .map(|dir| dir.join(file_name))
        .find(|path| path.is_file())
        .ok_or_else(|| LibraryNotFound {
            file_name: file_name.to_string(),
            searched,
        })
}

/// Path of the TVM compiler library.
///
/// `TVM_COMPILER_LIBRARY` takes precedence; otherwise [`COMPILER_LIBRARY`] is
/// resolved against [`library_dirs`].
pub fn compiler_library_path() -> Result<PathBuf, LibraryNotFound> {
    if let Some(path) = std::env::var_os("TVM_COMPILER_LIBRARY") {
        let path = PathBuf::from(path);
        if path.is_file() {
            return Ok(path);
        }
        return Err(LibraryNotFound {
            file_name: path.display().to_string(),
            searched: vec![path.parent().map(Path::to_path_buf).unwrap_or_default()],
        });
    }
    find_library(COMPILER_LIBRARY)
}

/// Path of the TVM runtime library, resolved against [`library_dirs`].
pub fn runtime_library_path() -> Result<PathBuf, LibraryNotFound> {
    find_library(RUNTIME_LIBRARY)
}

static COMPILER: OnceLock<()> = OnceLock::new();
static COMPILER_INIT: Mutex<()> = Mutex::new(());

/// Load the TVM compiler library once for the whole process.
///
/// The module is retained for the lifetime of the process without sharing its
/// non-thread-safe handle; its registered global functions are available through
/// `tvm_ffi::Function::get_global` afterwards.  Repeated calls are cheap.
pub fn load_compiler() -> Result<(), LoadError> {
    if COMPILER.get().is_some() {
        return Ok(());
    }
    let _guard = COMPILER_INIT
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if COMPILER.get().is_some() {
        return Ok(());
    }
    let path = compiler_library_path()?;
    let module = Module::load_from_file(path.to_string_lossy())
        .map_err(|source| LoadError::Load { path, source })?;
    COMPILER.get_or_init(|| std::mem::forget(module));
    Ok(())
}
