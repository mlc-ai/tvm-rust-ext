# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Call the Rust passes of this crate from Python through its shared library.

Build the library first, in a Python environment that has the ``apache-tvm``
and ``apache-tvm-ffi`` packages the crate is built against::

    cargo build            # -> target/debug/libtvm.so
    python python/demo.py

``libtvm.so`` is an ordinary tvm-ffi module: ``tvm_ffi.load_module`` opens it
and every ``__tvm_ffi_<name>`` symbol exported by ``src/exports.rs`` becomes a
callable.  TVM objects cross the boundary unchanged, so a ``tvm.tirx.PrimFunc``
built in Python goes in and the transformed ``PrimFunc`` comes back.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import tvm  # noqa: F401  (loads libtvm_compiler, which the Rust passes call into)
import tvm_ffi
from tvm.script import tirx as T

LIBRARY_NAMES = ("libtvm.so", "libtvm.dylib", "tvm.dll")


def find_library() -> Path:
    """Locate the cdylib built by ``cargo build`` (or ``TVM_RUST_LIBRARY``)."""
    override = os.environ.get("TVM_RUST_LIBRARY")
    if override:
        return Path(override)
    target = Path(__file__).resolve().parent.parent / "target"
    for profile in ("release", "debug"):
        for name in LIBRARY_NAMES:
            candidate = target / profile / name
            if candidate.is_file():
                return candidate
    sys.exit(
        "shared library not found; run `cargo build` in the repository root "
        "or set TVM_RUST_LIBRARY"
    )


# TVMScript folds `x + 0` and `1 + 2` while parsing, so build those nodes
# explicitly to leave the Rust passes something to simplify.
def add(lhs, rhs):
    return tvm.tirx.Add(lhs, rhs)


ZERO = tvm.tirx.IntImm("int32", 0)
ONE = tvm.tirx.IntImm("int32", 1)
TWO = tvm.tirx.IntImm("int32", 2)


@T.prim_func
def before(A: T.Buffer((16,), "int32"), B: T.Buffer((16,), "int32")):
    for i in T.serial(16):
        with T.Assert(i < 16, "index in range"):
            for j in T.serial(1):
                B[i] = add(add(add(A[i], ZERO), add(ONE, TWO)), j)


@T.prim_func
def expected(A: T.Buffer((16,), "int32"), B: T.Buffer((16,), "int32")):
    for i in T.serial(16):
        B[i] = A[i] + 3


def main() -> None:
    library = find_library()
    lib = tvm_ffi.load_module(library)
    print(f"loaded {library}\n")

    print("before:")
    print(before.script())
    print(f"expression nodes: {lib['expr_complexity'](before)}\n")

    # PrimFunc -> PrimFunc passes, applied one after another.
    func = before
    for name in (
        "eliminate_unit_loops",  # for j in range(1): body  ->  body[j := 0]
        "simplify_add_zero",  # A[i] + 0  ->  A[i]
        "fold_integer_constants",  # 1 + 2  ->  3
        "skip_assert",  # drop the assert, keep its body
    ):
        func = lib[name](func)
        print(f"after {name}:")
        print(func.script())
    print(f"expression nodes: {lib['expr_complexity'](func)}\n")
    # Structural equality also compares the function name stored in attrs.
    tvm.ir.assert_structural_equal(func, expected.with_attr("global_symbol", "before"))

    # Pass factories return TVM pass objects that compose with tvm.transform.
    skip_assert = lib["skip_assert_pass"]()
    print(f"pass object: {skip_assert}")
    mod = tvm.IRModule({"main": before})
    mod = skip_assert(mod)
    mod = lib["simplify_add_zero_module"](mod)
    print("module after skip_assert_pass + simplify_add_zero_module:")
    print(mod.script())

    print("OK")


if __name__ == "__main__":
    main()
