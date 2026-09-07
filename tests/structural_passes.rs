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

use tvm::analysis::{loop_nesting, memory_access_statistics, node_statistics};
use tvm::ir::prim::{
    Add, FloorDiv, FloorMod, Let, Mul, Not, Select, StringImm, EQ, GE, GT, LE, LT, NE,
};
use tvm::ir::{
    BaseFunc, Call, DictAttrs, DummyGlobalInfo, Expr, FloatImm, GlobalVar, IRModule, IntImm,
    IntImmObj, OpaqueExpr, PointerType, PrimExpr, PrimExprConvertible, PrimType, Range,
    SequentialSpan, SourceMap, SourceName, Span, TensorLoad, Type, Var,
};
use tvm::tirx::{
    AllocBuffer, AssertStmt, AttrStmt, Axis, Bind, BindObj, BufferRegion, BufferRegionType,
    BufferStore, BufferType, BufferVar, ComposeLayout, DeclBuffer, DispatchContext, Evaluate,
    EvaluateObj, ExecScope, For, ForKind, IfThenElse, IndexMap, Iter, IterVar, IterVarType,
    LambdaExpr, Layout, MatchBufferRegion, PrimFunc, PrimVar, Return, ScopeBinding, ScopeIdDef,
    ScopeIdDefStmt, ScopeKind, SeqStmt, Stmt, TensorIntrin, TileLayout, TilePrimitiveCall, While,
};
use tvm::transform;
use tvm::tvm_ffi::{
    structural_walk, Any, Array, Function, Map, ObjectRefCast, ObjectRefCore, Result, WalkOrder,
    WalkResult,
};

mod common;
use common::{assert_structural_equal, load_tvm_compiler, object_pointer};

fn int_expression(value: i64) -> Expr {
    typed_int_expression("int32", value)
}

fn typed_int_expression(dtype: &str, value: i64) -> Expr {
    IntImm::new(dtype, value).unwrap().into()
}

fn prim_int_expression(value: i64) -> PrimExpr {
    IntImm::new("int32", value).unwrap().into()
}

fn cpp_pass(name: &str) -> transform::Pass {
    Function::get_global(name)
        .unwrap()
        .call_tuple(())
        .unwrap()
        .try_into()
        .unwrap()
}

fn has_nonzero_function_attr(function: &BaseFunc, key: &str) -> bool {
    function
        .attrs
        .dict
        .get(&tvm::tvm_ffi::String::from(key))
        .unwrap()
        .map(i64::try_from)
        .transpose()
        .unwrap()
        .unwrap_or(0)
        != 0
}

fn module_from_named_prim_funcs(functions: Vec<(&str, PrimFunc)>) -> IRModule {
    IRModule::new(
        functions
            .into_iter()
            .map(|(name, function)| (GlobalVar::new(name), BaseFunc::from(function)))
            .collect(),
    )
    .unwrap()
}

fn prim_func_with_global_symbol(value: i64, symbol: Option<&str>) -> PrimFunc {
    let attrs = symbol
        .map(|symbol| {
            DictAttrs::from_dictionary(Map::from_iter([(
                tvm::tvm_ffi::String::from("global_symbol"),
                Any::from(tvm::tvm_ffi::String::from(symbol)),
            )]))
        })
        .unwrap_or_else(DictAttrs::empty);
    PrimFunc::with_metadata(
        Vec::new(),
        Evaluate::from_i64(value).unwrap(),
        Type::missing(),
        attrs,
        None,
    )
    .unwrap()
}

fn prim_func_body_integer(function: &PrimFunc) -> i64 {
    function
        .body()
        .as_node::<EvaluateObj>()
        .expect("test function must contain an Evaluate")
        .value
        .as_node::<IntImmObj>()
        .expect("test evaluation must contain an IntImm")
        .value
}

#[path = "structural_passes/codegen.rs"]
mod codegen;
#[path = "structural_passes/cse_and_intrin.rs"]
mod cse_and_intrin;
#[path = "structural_passes/loops_and_ssa.rs"]
mod loops_and_ssa;
#[path = "structural_passes/module_and_opaque.rs"]
mod module_and_opaque;
#[path = "structural_passes/simplify_and_legalize.rs"]
mod simplify_and_legalize;
#[path = "structural_passes/structural_api.rs"]
mod structural_api;
#[path = "structural_passes/verification.rs"]
mod verification;
