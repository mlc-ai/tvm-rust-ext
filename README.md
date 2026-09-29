<!--
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
-->

# TVM Rust extension

Rust bindings for TVM's TIRx, generated using the `tvm-ffi` stubgen tool
(`tvm-ffi-stubgen --target rust`). The generated bindings cover `tirx` and
its shared `ir` and `prim` types. The crate also includes hand-written
semantic helpers and Rust implementations of TIRx passes.

## Build and test

With Rust installed and a Python environment active:

```bash
python -m pip install -r requirements.txt
cargo test
```

TVM and tvm-ffi are consumed as packages. Python dependencies are pinned in
[requirements.txt](requirements.txt), and the Rust `tvm-ffi` dependency is
pinned in [Cargo.toml](Cargo.toml). Keep `python` and `tvm-ffi-config` from
the active environment on `PATH`.

## Regenerate bindings

With the same Python environment active:

```bash
TVM_STUBGEN_LIB=$(python -c "import tvm, os; print(os.path.join(os.path.dirname(tvm.__file__), 'lib', 'libtvm_compiler.so'))")
tvm-ffi-stubgen --target rust --dlls "$TVM_STUBGEN_LIB" src
cargo fmt
```

Generated code lives in marked blocks in [src/ir.rs](src/ir.rs),
[src/prim.rs](src/prim.rs), and [src/tirx.rs](src/tirx.rs).
Hand-written helpers and passes live outside those blocks.

## Python demo

Build the shared library and run [python/demo.py](python/demo.py) to load
it through `tvm_ffi` and use the exported passes:

```bash
cargo build
python python/demo.py
```

See [BINDING_CONTRACT.md](BINDING_CONTRACT.md) for binding requirements and
[STUBGEN_FEEDBACK.md](STUBGEN_FEEDBACK.md) for generator details and remaining gaps.
