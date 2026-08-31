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

//! Build script: locate the TVM shared libraries that the `apache-tvm-ffi` and
//! `apache-tvm` pip packages install into the active Python environment.
//!
//! This crate deliberately vendors neither TVM nor tvm-ffi sources.  The
//! `tvm-ffi` Rust crate is a pinned git dependency whose own build script links
//! `libtvm_ffi` from `tvm-ffi-config --libdir`; this script complements it by
//!
//! 1. recording that directory as an rpath, so `cargo test`/`cargo run`
//!    executables resolve `libtvm_ffi.so` without `LD_LIBRARY_PATH`, and
//! 2. recording the directory that holds `libtvm_compiler.so` (found through
//!    the installed `tvm` Python package, mirroring `python/tvm/libinfo.py`)
//!    as the compile-time default used by `tvm::libinfo`.
//!
//! Nothing here is specific to conda.  Executables are resolved the way
//! pyo3-build-config resolves the interpreter: an explicit override variable,
//! then the active `VIRTUAL_ENV` / `CONDA_PREFIX` environment's own `bin/`,
//! then `PATH`.  Any Python environment (venv, uv, conda, system
//! site-packages) works; `tvm-ffi-config` on `PATH` is additionally required
//! by the `tvm-ffi-sys` crate's own build script.  All recorded paths are
//! absolute paths of the environment the crate was built against; the script
//! re-runs when the override variables, `VIRTUAL_ENV`, or `CONDA_PREFIX`
//! change, and on `PATH` changes only when `PATH` resolved an executable.
//!
//! Environment overrides: `TVM_LIBRARY_PATH` (directory containing the TVM
//! libraries), `TVM_PYTHON` (interpreter used to locate the `tvm` package),
//! `TVM_FFI_CONFIG` (path of the `tvm-ffi-config` executable).  Discovery
//! failures are reported as build warnings; the library still compiles and
//! `tvm::libinfo` reports the problem at run time.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn compiler_library_name(target_os: &str) -> &'static str {
    match target_os {
        "windows" => "tvm_compiler.dll",
        "macos" => "libtvm_compiler.dylib",
        _ => "libtvm_compiler.so",
    }
}

/// Executables named `name` inside the active virtualenv / conda environment.
///
/// `VIRTUAL_ENV` is checked before `CONDA_PREFIX` because a virtualenv
/// activated on top of a conda environment is the more specific one.
fn env_executables(name: &str) -> Vec<PathBuf> {
    let mut found = Vec::new();
    for prefix in ["VIRTUAL_ENV", "CONDA_PREFIX"] {
        if let Some(prefix) = env::var_os(prefix) {
            let prefix = PathBuf::from(prefix);
            if cfg!(windows) {
                found.push(prefix.join(format!("{name}.exe")));
                found.push(prefix.join("Scripts").join(format!("{name}.exe")));
            } else {
                found.push(prefix.join("bin").join(name));
            }
        }
    }
    found
}

/// Declare `PATH` as a build-script input (once).
fn track_path() {
    use std::sync::atomic::{AtomicBool, Ordering};
    static TRACKED: AtomicBool = AtomicBool::new(false);
    if !TRACKED.swap(true, Ordering::Relaxed) {
        println!("cargo:rerun-if-env-changed=PATH");
    }
}

/// Candidate executables for `name`: an explicit `override_var`, then the
/// active environment's own copy, then `PATH`.  Like pyo3-build-config, the
/// build script is only re-run on `PATH` changes when `PATH` was actually the
/// thing that resolved the executable.
fn candidate_executables(override_var: &str, name: &str) -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    if let Some(explicit) = env::var_os(override_var) {
        candidates.push(PathBuf::from(explicit));
    }
    let env_candidates = env_executables(name);
    if env_candidates.iter().all(|path| !path.is_file()) {
        track_path();
    }
    candidates.extend(env_candidates);
    candidates.push(PathBuf::from(name));
    candidates
}

/// Run `tvm-ffi-config --libdir` (installed by the `apache-tvm-ffi` package).
fn tvm_ffi_library_dir() -> Option<PathBuf> {
    for exe in candidate_executables("TVM_FFI_CONFIG", "tvm-ffi-config") {
        let Ok(output) = Command::new(&exe).arg("--libdir").output() else {
            continue;
        };
        if !output.status.success() {
            continue;
        }
        let dir = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if !dir.is_empty() {
            return Some(PathBuf::from(dir));
        }
    }
    None
}

/// Ask Python where the `tvm` package lives without importing it (importing
/// would load every TVM runtime library just to answer a path question).
fn tvm_package_dirs() -> Vec<PathBuf> {
    const SCRIPT: &str = "\
import importlib.util, os, sys
spec = importlib.util.find_spec('tvm')
if spec is None:
    sys.exit(1)
dirs = list(spec.submodule_search_locations or [])
if spec.origin:
    dirs.append(os.path.dirname(spec.origin))
sys.stdout.write('\\n'.join(dirs))
";
    let mut interpreters = candidate_executables("TVM_PYTHON", "python");
    interpreters.push(PathBuf::from("python3"));
    for python in interpreters {
        let Ok(output) = Command::new(&python).args(["-c", SCRIPT]).output() else {
            continue;
        };
        if !output.status.success() {
            continue;
        }
        let stdout = String::from_utf8_lossy(&output.stdout);
        let dirs: Vec<PathBuf> = stdout
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .map(PathBuf::from)
            .collect();
        if !dirs.is_empty() {
            return dirs;
        }
    }
    Vec::new()
}

/// Same candidate layouts as `tvm.libinfo.package_lib_paths()`: the wheel
/// layout `<pkg>/lib` and the in-tree development layouts.
fn tvm_library_dir(library: &str) -> Option<PathBuf> {
    let mut candidates = Vec::new();
    if let Some(dir) = env::var_os("TVM_LIBRARY_PATH") {
        candidates.push(PathBuf::from(dir));
    }
    for pkg in tvm_package_dirs() {
        candidates.push(pkg.join("lib"));
        candidates.push(pkg.join("..").join("..").join("build").join("lib"));
        candidates.push(pkg.join("..").join("..").join("lib"));
    }
    candidates
        .into_iter()
        .find(|dir| dir.join(library).is_file())
        .and_then(|dir| dir.canonicalize().ok())
}

fn emit_rpath(dir: &Path, target_os: &str) {
    if target_os != "windows" {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", dir.display());
    }
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    for var in [
        "TVM_LIBRARY_PATH",
        "TVM_PYTHON",
        "TVM_FFI_CONFIG",
        "CONDA_PREFIX",
        "VIRTUAL_ENV",
    ] {
        println!("cargo:rerun-if-env-changed={var}");
    }
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();

    match tvm_ffi_library_dir() {
        Some(dir) => {
            println!("cargo:rustc-env=TVM_FFI_LIBRARY_DIR={}", dir.display());
            emit_rpath(&dir, &target_os);
        }
        None => println!(
            "cargo:warning=tvm-ffi-config not found; install the apache-tvm-ffi pip package \
             in the active Python environment (executables will need LD_LIBRARY_PATH)"
        ),
    }

    let library = compiler_library_name(&target_os);
    match tvm_library_dir(library) {
        Some(dir) => {
            println!("cargo:rustc-env=TVM_LIBRARY_DIR={}", dir.display());
            emit_rpath(&dir, &target_os);
        }
        None => println!(
            "cargo:warning={library} not found through the `tvm` Python package; install the \
             apache-tvm pip package in the active Python environment or set TVM_LIBRARY_PATH"
        ),
    }
}
