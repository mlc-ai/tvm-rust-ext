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

//! Reflection contracts and constructor checks for the object slice listed below.

use tvm::analysis::CallEffectKind;
use tvm::ir::StringImmObj;
use tvm::ir::TensorRegionObj;
use tvm::ir::{
    AttrsObj, BaseFuncObj, CallObj, DictAttrsObj, Expr, ExprObj, FuncType, FuncTypeObj,
    GlobalInfoObj, GlobalVarObj, IRModuleObj, IntImm, IntImmObj, OpaqueExprObj, OpaqueTypeObj,
    PointerType, PointerTypeObj, PrimExpr, PrimExprConvertibleObj, PrimType, PrimTypeObj, RangeObj,
    SequentialSpanObj, SourceMapObj, SourceNameObj, SourceObj, Span, SpanObj, TensorLoadObj, Tuple,
    TupleGetItem, TupleGetItemObj, TupleObj, TupleType, TupleTypeObj, Type, TypeObj, Var, VarObj,
};
use tvm::prim::{
    AddObj, AndObj, BroadcastObj, CastObj, DivObj, EQObj, FloorDivObj, FloorModObj, GEObj, GTObj,
    LEObj, LTObj, LetObj, MaxObj, MinObj, ModObj, MulObj, NEObj, NotObj, OrObj, RampObj, SelectObj,
    ShuffleObj, SubObj,
};
use tvm::te::{CommReducerObj, ReduceObj};
use tvm::tirx::{
    AllocBufferObj, AssertStmtObj, AttrStmtObj, AxisObj, BindObj, BufferRegionTypeObj,
    BufferStoreObj, BufferType, BufferTypeObj, BufferVar, ComposeLayoutObj, DeclBufferObj,
    DispatchContextObj, EvaluateObj, ExecScopeObj, ForKind, ForObj, IfThenElseObj, IndexMapObj,
    IterObj, IterVarObj, IterVarType, LambdaExprObj, LayoutObj, PrimFuncObj, PrimVar, ScopeBinding,
    ScopeIdDefObj, ScopeIdDefStmtObj, ScopeKind, SeqStmtObj, StmtObj, TileLayoutObj,
    TilePrimitiveCallObj,
};
use tvm::tirx::{TensorMapType, TensorMapTypeObj};
use tvm::tvm_ffi::tvm_ffi_sys::{TVMFFIFieldFlagBitMask, TVMFFISEqHashKind};
use tvm::tvm_ffi::{Array, Function, Object, ObjectCore, ObjectRefCore, String};

mod common;
use common::{assert_structural_equal, direct_fields, load_tvm_compiler, runtime_type_info};

const DEFAULT: i64 = TVMFFIFieldFlagBitMask::kTVMFFIFieldFlagBitMaskHasDefault as i64;
const IGNORE: i64 = TVMFFIFieldFlagBitMask::kTVMFFIFieldFlagBitMaskSEqHashIgnore as i64;
const DEF_RECURSIVE: i64 = TVMFFIFieldFlagBitMask::kTVMFFIFieldFlagBitMaskSEqHashDefPattern as i64;
const DEF_NON_RECURSIVE: i64 =
    TVMFFIFieldFlagBitMask::kTVMFFIFieldFlagBitMaskSEqHashDefSimple as i64;

const SCHEMA_ANY_MAP: &str = r#"{"type":"ffi.Map","args":[{"type":"ffi.String"},{"type":"Any"}]}"#;
const SCHEMA_ANY: &str = r#"{"type":"Any"}"#;
const SCHEMA_ARRAY_ANY: &str = r#"{"type":"ffi.Array","args":[{"type":"Any"}]}"#;
const SCHEMA_ARRAY_EXPR: &str = r#"{"type":"ffi.Array","args":[{"type":"ir.Expr"}]}"#;
const SCHEMA_ARRAY_GLOBAL_INFO_MAP: &str = r#"{"type":"ffi.Map","args":[{"type":"ffi.String"},{"type":"ffi.Array","args":[{"type":"ir.GlobalInfo"}]}]}"#;
const SCHEMA_ARRAY_ITER: &str = r#"{"type":"ffi.Array","args":[{"type":"tirx.Iter"}]}"#;
const SCHEMA_ARRAY_ITER_VAR: &str = r#"{"type":"ffi.Array","args":[{"type":"tirx.IterVar"}]}"#;
const SCHEMA_ARRAY_RANGE: &str = r#"{"type":"ffi.Array","args":[{"type":"ir.Range"}]}"#;
const SCHEMA_ARRAY_SPAN: &str = r#"{"type":"ffi.Array","args":[{"type":"ir.Span"}]}"#;
const SCHEMA_ARRAY_STMT: &str = r#"{"type":"ffi.Array","args":[{"type":"tirx.Stmt"}]}"#;
const SCHEMA_ARRAY_STRING_IMM: &str = r#"{"type":"ffi.Array","args":[{"type":"ir.StringImm"}]}"#;
const SCHEMA_ARRAY_TYPE: &str = r#"{"type":"ffi.Array","args":[{"type":"ir.Type"}]}"#;
const SCHEMA_ARRAY_VAR: &str = r#"{"type":"ffi.Array","args":[{"type":"ir.Var"}]}"#;
const SCHEMA_ATTRS: &str = r#"{"type":"ir.Attrs"}"#;
const SCHEMA_AXIS: &str = r#"{"type":"tirx.Axis"}"#;
const SCHEMA_BOOL: &str = r#"{"type":"bool"}"#;
const SCHEMA_COMM_REDUCER: &str = r#"{"type":"te.CommReducer"}"#;
const SCHEMA_DICT_ATTRS: &str = r#"{"type":"ir.DictAttrs"}"#;
const SCHEMA_DTYPE: &str = r#"{"type":"DataType"}"#;
const SCHEMA_EXPR: &str = r#"{"type":"ir.Expr"}"#;
const SCHEMA_EXEC_SCOPE: &str = r#"{"type":"tirx.ExecScope"}"#;
const SCHEMA_INT: &str = r#"{"type":"int"}"#;
const SCHEMA_LAYOUT_OPTIONAL: &str = r#"{"type":"Optional","args":[{"type":"tirx.Layout"}]}"#;
const SCHEMA_MAP_AXIS_EXPR: &str =
    r#"{"type":"ffi.Map","args":[{"type":"tirx.Axis"},{"type":"ir.Expr"}]}"#;
const SCHEMA_MAP_FUNCTIONS: &str =
    r#"{"type":"ffi.Map","args":[{"type":"ir.GlobalVar"},{"type":"ir.BaseFunc"}]}"#;
const SCHEMA_MAP_GLOBAL_VAR: &str =
    r#"{"type":"ffi.Map","args":[{"type":"ffi.String"},{"type":"ir.GlobalVar"}]}"#;
const SCHEMA_MAP_SOURCE: &str =
    r#"{"type":"ffi.Map","args":[{"type":"ir.SourceName"},{"type":"ir.Source"}]}"#;
const SCHEMA_MAP_STRING_ARRAY_EXPR: &str = r#"{"type":"ffi.Map","args":[{"type":"ffi.String"},{"type":"ffi.Array","args":[{"type":"ir.Expr"}]}]}"#;
const SCHEMA_MAP_STRING_ITER_VAR: &str =
    r#"{"type":"ffi.Map","args":[{"type":"ffi.String"},{"type":"tirx.IterVar"}]}"#;
const SCHEMA_MAP_STRING_OBJECT: &str =
    r#"{"type":"ffi.Map","args":[{"type":"ffi.String"},{"type":"ffi.Object"}]}"#;
const SCHEMA_MAP_STRING_VAR: &str =
    r#"{"type":"ffi.Map","args":[{"type":"ffi.String"},{"type":"ir.Var"}]}"#;
const SCHEMA_MAP_VAR_RANGE: &str =
    r#"{"type":"ffi.Map","args":[{"type":"ir.Var"},{"type":"ir.Range"}]}"#;
const SCHEMA_OP: &str = r#"{"type":"ir.Op"}"#;
const SCHEMA_OPTIONAL_EXPR: &str = r#"{"type":"Optional","args":[{"type":"ir.Expr"}]}"#;
const SCHEMA_OPTIONAL_ITER_VAR: &str = r#"{"type":"Optional","args":[{"type":"tirx.IterVar"}]}"#;
const SCHEMA_OPTIONAL_STRING: &str = r#"{"type":"Optional","args":[{"type":"ffi.String"}]}"#;
const SCHEMA_OPTIONAL_STMT: &str = r#"{"type":"Optional","args":[{"type":"tirx.Stmt"}]}"#;
const SCHEMA_OPTIONAL_OBJECT: &str = r#"{"type":"Optional","args":[{"type":"ffi.Object"}]}"#;
const SCHEMA_OPTIONAL_ARRAY_EXPR: &str =
    r#"{"type":"Optional","args":[{"type":"ffi.Array","args":[{"type":"ir.Expr"}]}]}"#;
const SCHEMA_PRIM_TYPE: &str = r#"{"type":"ir.PrimType"}"#;
const SCHEMA_RANGE: &str = r#"{"type":"ir.Range"}"#;
const SCHEMA_SOURCE_MAP: &str = r#"{"type":"ir.SourceMap"}"#;
const SCHEMA_SOURCE_NAME: &str = r#"{"type":"ir.SourceName"}"#;
const SCHEMA_SPAN: &str = r#"{"type":"ir.Span"}"#;
const SCHEMA_STMT: &str = r#"{"type":"tirx.Stmt"}"#;
const SCHEMA_STRING: &str = r#"{"type":"ffi.String"}"#;
const SCHEMA_STRING_IMM: &str = r#"{"type":"ir.StringImm"}"#;
const SCHEMA_TYPE: &str = r#"{"type":"ir.Type"}"#;
const SCHEMA_VAR: &str = r#"{"type":"ir.Var"}"#;
const SCHEMA_TILE_LAYOUT: &str = r#"{"type":"tirx.TileLayout"}"#;
const SCHEMA_TARGET: &str = r#"{"type":"target.Target"}"#;
const SCHEMA_SCOPE_ID_DEF: &str = r#"{"type":"tirx.ScopeIdDef"}"#;

fn assert_contract<N: ObjectCore, P: ObjectCore>(
    expected_final: bool,
    expected_structural_kind: Option<TVMFFISEqHashKind>,
    expected_fields: &[(&str, i64, &str)],
) {
    let info = runtime_type_info::<N>();
    assert_eq!(
        info.type_index,
        N::type_index(),
        "{} type index",
        N::TYPE_KEY
    );
    assert_eq!(info.type_key.as_str(), N::TYPE_KEY);
    assert_eq!(info.type_depth, N::TYPE_DEPTH, "{} depth", N::TYPE_KEY);
    assert_eq!(
        N::TYPE_DEPTH,
        P::TYPE_DEPTH + 1,
        "{} parent depth",
        N::TYPE_KEY
    );
    assert_eq!(N::TYPE_FINAL, expected_final, "{} finality", N::TYPE_KEY);

    assert!(!info.type_acenstors.is_null());
    let parent = unsafe { *info.type_acenstors.add(P::TYPE_DEPTH as usize) };
    assert!(!parent.is_null());
    assert_eq!(
        unsafe { (*parent).type_index },
        P::type_index(),
        "{} parent",
        N::TYPE_KEY
    );

    let actual_fields = direct_fields::<N>();
    assert_eq!(
        actual_fields.len(),
        expected_fields.len(),
        "{} field count",
        N::TYPE_KEY
    );
    for (field, (expected_name, expected_flags, expected_schema)) in
        actual_fields.iter().zip(expected_fields)
    {
        assert_eq!(
            field.name.as_str(),
            *expected_name,
            "{} field name",
            N::TYPE_KEY
        );
        assert_eq!(
            field.flags,
            *expected_flags,
            "{}.{} flags",
            N::TYPE_KEY,
            expected_name
        );
        if expected_flags & DEFAULT != 0 {
            // Expr.ty uses the exact `ir.Type` missing-type sentinel.  The
            // other defaults in this slice are nullable object metadata and
            // therefore use `None`.
            let expected_type_index = if N::TYPE_KEY == ExprObj::TYPE_KEY && *expected_name == "ty"
            {
                TypeObj::type_index()
            } else {
                tvm::tvm_ffi::tvm_ffi_sys::TVMFFITypeIndex::kTVMFFINone as i32
            };
            assert_eq!(
                field.default_value_or_factory.type_index,
                expected_type_index,
                "{}.{} default value",
                N::TYPE_KEY,
                expected_name
            );
            if expected_type_index == TypeObj::type_index() {
                assert!(
                    !unsafe { field.default_value_or_factory.data_union.v_obj }.is_null(),
                    "{}.{} missing-type default has a null object",
                    N::TYPE_KEY,
                    expected_name
                );
            }
        }
        assert!(
            field.getter.is_some(),
            "reflected field {}.{} has no getter",
            N::TYPE_KEY,
            expected_name
        );
        let expected_metadata = format!(
            "{{\"type_schema\":\"{}\"}}",
            expected_schema.replace('"', "\\\"")
        );
        assert_eq!(
            field.metadata.as_str(),
            expected_metadata,
            "{}.{} type schema",
            N::TYPE_KEY,
            expected_name
        );
    }

    let actual_structural_kind = if info.metadata.is_null() {
        None
    } else {
        Some(unsafe { (*info.metadata).structural_eq_hash_kind })
    };
    assert_eq!(
        actual_structural_kind,
        expected_structural_kind.map(|kind| kind as i32),
        "{} structural kind",
        N::TYPE_KEY
    );
}

#[test]
fn covered_object_schemas_match_runtime_metadata() {
    load_tvm_compiler();
    use TVMFFISEqHashKind::{
        kTVMFFISEqHashKindFreeVar as FreeVar, kTVMFFISEqHashKindTreeNode as Tree,
        kTVMFFISEqHashKindUnsupported as Unsupported,
    };

    assert_contract::<ExprObj, Object>(
        false,
        Some(Tree),
        &[
            ("span", DEFAULT | IGNORE, SCHEMA_SPAN),
            ("ty", DEFAULT, SCHEMA_TYPE),
        ],
    );
    assert_contract::<OpaqueExprObj, ExprObj>(false, Some(Tree), &[]);
    assert_contract::<BaseFuncObj, ExprObj>(false, Some(Tree), &[("attrs", 0, SCHEMA_DICT_ATTRS)]);
    assert_contract::<GlobalVarObj, ExprObj>(
        true,
        Some(FreeVar),
        &[("name_hint", 0, SCHEMA_STRING)],
    );
    assert_contract::<VarObj, ExprObj>(false, Some(FreeVar), &[("name", IGNORE, SCHEMA_STRING)]);
    assert_contract::<SourceNameObj, Object>(true, Some(Tree), &[("name", 0, SCHEMA_STRING)]);
    assert_contract::<SourceObj, Object>(
        true,
        Some(Unsupported),
        &[
            ("source_name", 0, SCHEMA_SOURCE_NAME),
            ("source", 0, SCHEMA_STRING),
        ],
    );
    assert_contract::<SourceMapObj, Object>(
        true,
        Some(Tree),
        &[("source_map", 0, SCHEMA_MAP_SOURCE)],
    );
    assert_contract::<SpanObj, Object>(
        false,
        Some(Tree),
        &[
            ("source_name", 0, SCHEMA_SOURCE_NAME),
            ("line", 0, SCHEMA_INT),
            ("column", 0, SCHEMA_INT),
            ("end_line", 0, SCHEMA_INT),
            ("end_column", 0, SCHEMA_INT),
        ],
    );
    assert_contract::<SequentialSpanObj, SpanObj>(
        true,
        Some(Tree),
        &[("spans", 0, SCHEMA_ARRAY_SPAN)],
    );
    assert_contract::<PrimExprConvertibleObj, Object>(false, None, &[]);
    assert_contract::<RangeObj, Object>(
        true,
        Some(Tree),
        &[
            ("min", 0, SCHEMA_EXPR),
            ("extent", 0, SCHEMA_EXPR),
            ("span", IGNORE, SCHEMA_SPAN),
        ],
    );
    assert_contract::<CallObj, ExprObj>(
        true,
        Some(Tree),
        &[
            ("op", 0, SCHEMA_EXPR),
            ("args", 0, SCHEMA_ARRAY_EXPR),
            ("attrs", 0, SCHEMA_ATTRS),
            ("ty_args", 0, SCHEMA_ARRAY_TYPE),
        ],
    );
    assert_contract::<TupleObj, ExprObj>(true, Some(Tree), &[("fields", 0, SCHEMA_ARRAY_EXPR)]);
    assert_contract::<TupleGetItemObj, ExprObj>(
        true,
        Some(Tree),
        &[("tuple_value", 0, SCHEMA_EXPR), ("index", 0, SCHEMA_INT)],
    );
    assert_contract::<TypeObj, Object>(
        false,
        Some(Tree),
        &[("span", DEFAULT | IGNORE, SCHEMA_SPAN)],
    );
    assert_contract::<OpaqueTypeObj, TypeObj>(true, Some(Tree), &[]);
    assert_contract::<PointerTypeObj, TypeObj>(
        true,
        Some(Tree),
        &[
            ("element_type", 0, SCHEMA_TYPE),
            ("storage_scope", 0, SCHEMA_STRING),
        ],
    );
    assert_contract::<PrimTypeObj, TypeObj>(true, Some(Tree), &[("dtype", 0, SCHEMA_DTYPE)]);
    assert_contract::<TupleTypeObj, TypeObj>(true, Some(Tree), &[("fields", 0, SCHEMA_ARRAY_TYPE)]);
    assert_contract::<FuncTypeObj, TypeObj>(
        true,
        Some(Tree),
        &[
            ("arg_types", 0, SCHEMA_ARRAY_TYPE),
            ("ret_type", 0, SCHEMA_TYPE),
        ],
    );
    assert_contract::<TensorMapTypeObj, TypeObj>(true, Some(Tree), &[]);
    assert_contract::<IntImmObj, tvm::ir::ConstantObj>(
        true,
        Some(Tree),
        &[("value", 0, r#"{"type":"ffi.BigInt"}"#)],
    );
    assert_contract::<AttrsObj, Object>(false, Some(Tree), &[]);
    assert_contract::<DictAttrsObj, AttrsObj>(true, Some(Tree), &[("__dict__", 0, SCHEMA_ANY_MAP)]);
    assert_contract::<GlobalInfoObj, Object>(false, None, &[]);
    assert_contract::<IRModuleObj, Object>(
        true,
        Some(Tree),
        &[
            ("functions", 0, SCHEMA_MAP_FUNCTIONS),
            ("global_var_map_", 0, SCHEMA_MAP_GLOBAL_VAR),
            ("source_map", 0, SCHEMA_SOURCE_MAP),
            ("attrs", 0, SCHEMA_DICT_ATTRS),
            ("global_infos", 0, SCHEMA_ARRAY_GLOBAL_INFO_MAP),
        ],
    );

    assert_contract::<AddObj, ExprObj>(
        true,
        Some(Tree),
        &[("a", 0, SCHEMA_EXPR), ("b", 0, SCHEMA_EXPR)],
    );
    assert_contract::<SubObj, ExprObj>(
        true,
        Some(Tree),
        &[("a", 0, SCHEMA_EXPR), ("b", 0, SCHEMA_EXPR)],
    );
    assert_contract::<MulObj, ExprObj>(
        true,
        Some(Tree),
        &[("a", 0, SCHEMA_EXPR), ("b", 0, SCHEMA_EXPR)],
    );
    assert_contract::<DivObj, ExprObj>(
        true,
        Some(Tree),
        &[("a", 0, SCHEMA_EXPR), ("b", 0, SCHEMA_EXPR)],
    );
    assert_contract::<ModObj, ExprObj>(
        true,
        Some(Tree),
        &[("a", 0, SCHEMA_EXPR), ("b", 0, SCHEMA_EXPR)],
    );
    assert_contract::<FloorDivObj, ExprObj>(
        true,
        Some(Tree),
        &[("a", 0, SCHEMA_EXPR), ("b", 0, SCHEMA_EXPR)],
    );
    assert_contract::<FloorModObj, ExprObj>(
        true,
        Some(Tree),
        &[("a", 0, SCHEMA_EXPR), ("b", 0, SCHEMA_EXPR)],
    );
    assert_contract::<MinObj, ExprObj>(
        true,
        Some(Tree),
        &[("a", 0, SCHEMA_EXPR), ("b", 0, SCHEMA_EXPR)],
    );
    assert_contract::<MaxObj, ExprObj>(
        true,
        Some(Tree),
        &[("a", 0, SCHEMA_EXPR), ("b", 0, SCHEMA_EXPR)],
    );
    assert_contract::<EQObj, ExprObj>(
        true,
        Some(Tree),
        &[("a", 0, SCHEMA_EXPR), ("b", 0, SCHEMA_EXPR)],
    );
    assert_contract::<NEObj, ExprObj>(
        true,
        Some(Tree),
        &[("a", 0, SCHEMA_EXPR), ("b", 0, SCHEMA_EXPR)],
    );
    assert_contract::<LTObj, ExprObj>(
        true,
        Some(Tree),
        &[("a", 0, SCHEMA_EXPR), ("b", 0, SCHEMA_EXPR)],
    );
    assert_contract::<LEObj, ExprObj>(
        true,
        Some(Tree),
        &[("a", 0, SCHEMA_EXPR), ("b", 0, SCHEMA_EXPR)],
    );
    assert_contract::<GTObj, ExprObj>(
        true,
        Some(Tree),
        &[("a", 0, SCHEMA_EXPR), ("b", 0, SCHEMA_EXPR)],
    );
    assert_contract::<GEObj, ExprObj>(
        true,
        Some(Tree),
        &[("a", 0, SCHEMA_EXPR), ("b", 0, SCHEMA_EXPR)],
    );
    assert_contract::<NotObj, ExprObj>(true, Some(Tree), &[("a", 0, SCHEMA_EXPR)]);
    assert_contract::<AndObj, ExprObj>(
        true,
        Some(Tree),
        &[("a", 0, SCHEMA_EXPR), ("b", 0, SCHEMA_EXPR)],
    );
    assert_contract::<OrObj, ExprObj>(
        true,
        Some(Tree),
        &[("a", 0, SCHEMA_EXPR), ("b", 0, SCHEMA_EXPR)],
    );
    assert_contract::<CastObj, ExprObj>(true, Some(Tree), &[("value", 0, SCHEMA_EXPR)]);
    assert_contract::<RampObj, ExprObj>(
        true,
        Some(Tree),
        &[
            ("base", 0, SCHEMA_EXPR),
            ("stride", 0, SCHEMA_EXPR),
            ("lanes", 0, SCHEMA_EXPR),
        ],
    );
    assert_contract::<BroadcastObj, ExprObj>(
        true,
        Some(Tree),
        &[("value", 0, SCHEMA_EXPR), ("lanes", 0, SCHEMA_EXPR)],
    );
    assert_contract::<ShuffleObj, ExprObj>(
        true,
        Some(Tree),
        &[
            ("vectors", 0, SCHEMA_ARRAY_EXPR),
            ("indices", 0, SCHEMA_ARRAY_EXPR),
        ],
    );
    assert_contract::<SelectObj, ExprObj>(
        true,
        Some(Tree),
        &[
            ("condition", 0, SCHEMA_EXPR),
            ("true_value", 0, SCHEMA_EXPR),
            ("false_value", 0, SCHEMA_EXPR),
        ],
    );
    assert_contract::<LetObj, ExprObj>(
        true,
        Some(Tree),
        &[
            ("var", DEF_NON_RECURSIVE, SCHEMA_VAR),
            ("value", 0, SCHEMA_EXPR),
            ("body", 0, SCHEMA_EXPR),
        ],
    );
    assert_contract::<StringImmObj, tvm::ir::ConstantObj>(
        true,
        Some(Tree),
        &[("value", 0, SCHEMA_STRING)],
    );
    assert_contract::<CommReducerObj, Object>(
        true,
        Some(Tree),
        &[
            ("lhs", DEF_RECURSIVE, SCHEMA_ARRAY_VAR),
            ("rhs", DEF_RECURSIVE, SCHEMA_ARRAY_VAR),
            ("result", 0, SCHEMA_ARRAY_EXPR),
            ("identity_element", 0, SCHEMA_ARRAY_EXPR),
            ("span", IGNORE, SCHEMA_SPAN),
        ],
    );
    assert_contract::<ReduceObj, OpaqueExprObj>(
        true,
        Some(Tree),
        &[
            ("combiner", 0, SCHEMA_COMM_REDUCER),
            ("source", 0, SCHEMA_ARRAY_EXPR),
            ("init", 0, SCHEMA_ARRAY_EXPR),
            ("axis", 0, SCHEMA_ARRAY_ITER_VAR),
            ("condition", 0, SCHEMA_EXPR),
            ("value_index", 0, SCHEMA_INT),
        ],
    );
    assert_contract::<StmtObj, Object>(false, Some(Tree), &[("span", IGNORE, SCHEMA_SPAN)]);
    assert_contract::<BindObj, StmtObj>(
        true,
        Some(Tree),
        &[
            ("var", DEF_NON_RECURSIVE, SCHEMA_VAR),
            ("value", 0, SCHEMA_EXPR),
        ],
    );
    assert_contract::<AttrStmtObj, StmtObj>(
        true,
        Some(Tree),
        &[
            ("node", 0, SCHEMA_ANY),
            ("attr_key", 0, SCHEMA_STRING),
            ("value", 0, SCHEMA_EXPR),
            ("body", 0, SCHEMA_STMT),
        ],
    );
    assert_contract::<AssertStmtObj, StmtObj>(
        true,
        Some(Tree),
        &[
            ("condition", 0, SCHEMA_EXPR),
            ("error_kind", 0, SCHEMA_STRING_IMM),
            ("message_parts", 0, SCHEMA_ARRAY_STRING_IMM),
        ],
    );
    assert_contract::<EvaluateObj, StmtObj>(true, Some(Tree), &[("value", 0, SCHEMA_EXPR)]);
    assert_contract::<SeqStmtObj, StmtObj>(true, Some(Tree), &[("seq", 0, SCHEMA_ARRAY_STMT)]);
    assert_contract::<IfThenElseObj, StmtObj>(
        true,
        Some(Tree),
        &[
            ("condition", 0, SCHEMA_EXPR),
            ("then_case", 0, SCHEMA_STMT),
            ("else_case", 0, SCHEMA_OPTIONAL_STMT),
        ],
    );
    assert_contract::<ForObj, StmtObj>(
        true,
        Some(Tree),
        &[
            ("loop_var", DEF_NON_RECURSIVE, SCHEMA_VAR),
            ("min", 0, SCHEMA_EXPR),
            ("extent", 0, SCHEMA_EXPR),
            ("kind", 0, SCHEMA_INT),
            ("body", 0, SCHEMA_STMT),
            ("thread_binding", 0, SCHEMA_OPTIONAL_ITER_VAR),
            ("annotations", 0, SCHEMA_ANY_MAP),
            ("step", 0, SCHEMA_OPTIONAL_EXPR),
        ],
    );
    assert_contract::<PrimFuncObj, BaseFuncObj>(
        true,
        Some(Tree),
        &[
            ("params", DEF_RECURSIVE, SCHEMA_ARRAY_VAR),
            ("ret_type", 0, SCHEMA_TYPE),
            ("body", 0, SCHEMA_STMT),
        ],
    );
    assert_contract::<LayoutObj, Object>(false, None, &[]);
    assert_contract::<AxisObj, Object>(true, Some(Tree), &[("name", 0, SCHEMA_STRING)]);
    assert_contract::<IterObj, Object>(
        true,
        Some(Tree),
        &[
            ("extent", 0, SCHEMA_EXPR),
            ("stride", 0, SCHEMA_EXPR),
            ("axis", 0, SCHEMA_AXIS),
        ],
    );
    assert_contract::<TileLayoutObj, LayoutObj>(
        true,
        Some(Tree),
        &[
            ("shard", 0, SCHEMA_ARRAY_ITER),
            ("replica", 0, SCHEMA_ARRAY_ITER),
            ("offset", 0, SCHEMA_MAP_AXIS_EXPR),
        ],
    );
    assert_contract::<ComposeLayoutObj, LayoutObj>(
        true,
        Some(Tree),
        &[
            ("per_element", 0, SCHEMA_INT),
            ("swizzle_len", 0, SCHEMA_INT),
            ("atom_len", 0, SCHEMA_INT),
            ("swizzle_inner", 0, SCHEMA_BOOL),
            ("inner_mask", 0, SCHEMA_INT),
            ("outer_mask", 0, SCHEMA_INT),
            ("tile_layout", 0, SCHEMA_TILE_LAYOUT),
        ],
    );
    assert_contract::<IndexMapObj, Object>(
        true,
        Some(Tree),
        &[
            ("initial_indices", DEF_RECURSIVE, SCHEMA_ARRAY_VAR),
            ("final_indices", 0, SCHEMA_ARRAY_EXPR),
            ("inverse_index_map", IGNORE, SCHEMA_OPTIONAL_OBJECT),
        ],
    );
    assert_contract::<BufferTypeObj, TypeObj>(
        true,
        Some(Tree),
        &[
            ("dtype", 0, SCHEMA_PRIM_TYPE),
            ("storage_scope", 0, SCHEMA_STRING),
            ("shape", DEF_RECURSIVE, SCHEMA_ARRAY_EXPR),
            ("strides", DEF_RECURSIVE, SCHEMA_ARRAY_EXPR),
            ("elem_offset", DEF_RECURSIVE, SCHEMA_EXPR),
            ("data_alignment", 0, SCHEMA_INT),
            ("offset_factor", 0, SCHEMA_INT),
            ("layout", 0, SCHEMA_LAYOUT_OPTIONAL),
            ("allocated_addr", 0, SCHEMA_ARRAY_EXPR),
        ],
    );
    assert_contract::<TensorLoadObj, ExprObj>(
        true,
        Some(Tree),
        &[
            ("source", 0, SCHEMA_EXPR),
            ("indices", 0, SCHEMA_ARRAY_EXPR),
        ],
    );
    assert_contract::<BufferStoreObj, StmtObj>(
        true,
        Some(Tree),
        &[
            ("buffer", 0, SCHEMA_VAR),
            ("value", 0, SCHEMA_EXPR),
            ("indices", 0, SCHEMA_ARRAY_EXPR),
        ],
    );
    assert_contract::<DeclBufferObj, StmtObj>(
        true,
        Some(Tree),
        &[
            ("buffer", DEF_NON_RECURSIVE, SCHEMA_VAR),
            ("data", 0, SCHEMA_EXPR),
        ],
    );
    assert_contract::<AllocBufferObj, StmtObj>(
        true,
        Some(Tree),
        &[
            ("buffer", DEF_NON_RECURSIVE, SCHEMA_VAR),
            ("annotations", 0, SCHEMA_ANY_MAP),
        ],
    );
    assert_contract::<BufferRegionTypeObj, TypeObj>(true, Some(Tree), &[]);
    assert_contract::<TensorRegionObj, ExprObj>(
        true,
        Some(Tree),
        &[
            ("source", DEF_RECURSIVE, SCHEMA_EXPR),
            ("region", 0, SCHEMA_ARRAY_RANGE),
        ],
    );
    assert_contract::<ExecScopeObj, Object>(false, Some(Tree), &[("kind", 0, SCHEMA_INT)]);
    assert_contract::<ScopeIdDefObj, Object>(
        true,
        Some(Tree),
        &[
            ("def_ids", DEF_NON_RECURSIVE, SCHEMA_ARRAY_VAR),
            ("extents", 0, SCHEMA_OPTIONAL_ARRAY_EXPR),
            ("scope", 0, SCHEMA_INT),
            ("preferred_extents", 0, SCHEMA_OPTIONAL_ARRAY_EXPR),
        ],
    );
    assert_contract::<ScopeIdDefStmtObj, StmtObj>(
        true,
        Some(Tree),
        &[("def", 0, SCHEMA_SCOPE_ID_DEF)],
    );
    assert_contract::<LambdaExprObj, Object>(
        true,
        Some(Tree),
        &[
            ("vars", DEF_RECURSIVE, SCHEMA_ARRAY_VAR),
            ("pred", 0, SCHEMA_EXPR),
        ],
    );
    assert_contract::<DispatchContextObj, Object>(
        true,
        Some(Unsupported),
        &[
            ("target", 0, SCHEMA_TARGET),
            ("exec_scope", 0, SCHEMA_EXEC_SCOPE),
            ("launch_params", 0, SCHEMA_MAP_STRING_ITER_VAR),
            ("var_range_map", 0, SCHEMA_MAP_VAR_RANGE),
            ("alloc_only", 0, SCHEMA_BOOL),
            ("callbacks", 0, SCHEMA_MAP_STRING_OBJECT),
            ("shared_state", 0, SCHEMA_MAP_STRING_OBJECT),
            ("inter", 0, SCHEMA_MAP_STRING_ARRAY_EXPR),
            ("intra", 0, SCHEMA_MAP_STRING_ARRAY_EXPR),
            ("scope_kind", 0, SCHEMA_STRING),
        ],
    );
    assert_contract::<TilePrimitiveCallObj, StmtObj>(
        true,
        Some(Tree),
        &[
            ("op", 0, SCHEMA_OP),
            ("args", 0, SCHEMA_ARRAY_ANY),
            ("workspace", 0, SCHEMA_MAP_STRING_VAR),
            ("config", 0, SCHEMA_ANY_MAP),
            ("dispatch", 0, SCHEMA_OPTIONAL_STRING),
            ("scope", 0, SCHEMA_EXEC_SCOPE),
        ],
    );
    assert_contract::<IterVarObj, PrimExprConvertibleObj>(
        true,
        Some(Tree),
        &[
            ("dom", 0, SCHEMA_RANGE),
            ("var", DEF_NON_RECURSIVE, SCHEMA_VAR),
            ("iter_type", 0, SCHEMA_INT),
            ("thread_tag", 0, SCHEMA_STRING),
            ("span", DEFAULT | IGNORE, SCHEMA_SPAN),
        ],
    );
}

#[test]
fn native_enum_values_preserve_width_and_unknown_variants() {
    assert_eq!(std::mem::size_of::<ForKind>(), std::mem::size_of::<i32>());
    assert_eq!(
        std::mem::size_of::<IterVarType>(),
        std::mem::size_of::<i32>()
    );
    assert_eq!(std::mem::size_of::<ScopeKind>(), std::mem::size_of::<i32>());
    assert_eq!(
        std::mem::size_of::<ScopeBinding>(),
        std::mem::size_of::<i32>()
    );
    assert_eq!(ForKind::from_raw(99).as_raw(), 99);
    assert_eq!(IterVarType::from_raw(99).as_raw(), 99);
    assert_eq!(ForKind::try_from(99_i64).unwrap().as_raw(), 99);
    assert_eq!(IterVarType::try_from(99_i64).unwrap().as_raw(), 99);
    assert!(ForKind::try_from(i64::from(i32::MAX) + 1).is_err());
    assert!(IterVarType::try_from(i64::from(i32::MIN) - 1).is_err());
    let effect = CallEffectKind::from_raw(17);
    assert_eq!(effect.as_raw(), 17);
    assert!(effect.may_update_state());
    assert_eq!(CallEffectKind::try_from(17).unwrap(), effect);
    assert!(CallEffectKind::try_from(i64::MAX).is_err());
}

#[test]
fn typed_expression_views_check_types_and_preserve_identity() {
    load_tvm_compiler();

    let integer: Expr = IntImm::new("int32", 1).unwrap().into();
    let primitive = PrimExpr::try_from(&integer).unwrap();
    assert!(primitive.same_as(&integer));
    assert!(primitive.type_annotation().same_as(&integer.ty));

    let tuple_typed: Expr = Var::with_type("tuple", TupleType::empty()).into();
    assert!(PrimExpr::try_from(&tuple_typed).is_err());

    let scalar_var = Var::new("i", "int32").unwrap();
    let primitive_var = PrimVar::try_from(&scalar_var).unwrap();
    assert!(primitive_var.same_as(&scalar_var));
    assert!(BufferVar::try_from(&scalar_var).is_err());

    let buffer_type = BufferType::new("global", "float32", Vec::new()).unwrap();
    let buffer_var = buffer_type.new_var("buffer");
    assert!(buffer_var.same_as(buffer_var.as_var()));
    assert!(buffer_var.type_annotation().same_as(&buffer_type));
    assert!(PrimVar::try_from(buffer_var.as_var()).is_err());

    let buffer_expr: Expr = buffer_var.clone().into();
    let recovered_buffer: BufferVar = (&buffer_expr).try_into().unwrap();
    assert!(recovered_buffer.same_as(&buffer_var));
    assert!(BufferVar::try_from(&integer).is_err());
}

#[test]
fn tuple_constructors_match_native_type_derivation_and_bounds() {
    load_tvm_compiler();

    let first: Expr = IntImm::new("int32", 1).unwrap().into();
    let second: Expr = IntImm::new("int64", 2).unwrap().into();
    let tuple = Tuple::new(vec![first, second]);
    let tuple_type = tuple.ty.as_node::<TupleTypeObj>().unwrap();
    assert_eq!(tuple_type.fields.len(), 2);
    assert!(tuple_type
        .fields
        .get(0)
        .unwrap()
        .same_as(&tuple.fields.get(0).unwrap().ty));
    assert!(tuple_type
        .fields
        .get(1)
        .unwrap()
        .same_as(&tuple.fields.get(1).unwrap().ty));

    let native_tuple: Tuple = Function::get_global("ir.Tuple")
        .unwrap()
        .call_tuple((tuple.fields.clone(), Option::<Span>::None))
        .unwrap()
        .try_into()
        .unwrap();
    assert_structural_equal(&tuple, &native_tuple);

    let projection = TupleGetItem::new(tuple.clone(), 1).unwrap();
    assert!(projection.ty.same_as(&tuple.fields.get(1).unwrap().ty));
    let native_projection: TupleGetItem = Function::get_global("ir.TupleGetItem")
        .unwrap()
        .call_tuple((Expr::from(tuple), 1_i32, Option::<Span>::None))
        .unwrap()
        .try_into()
        .unwrap();
    assert_structural_equal(&projection, &native_projection);

    assert!(TupleGetItem::new(native_tuple.clone(), -1).is_err());
    assert!(TupleGetItem::new(native_tuple, 2).is_err());

    let argument_type: Type = PrimType::new("int32").unwrap().into();
    let return_type: Type = TupleType::empty().into();
    let function_type = FuncType::new(vec![argument_type.clone()], return_type.clone());
    let native_function_type: FuncType = Function::get_global("ir.FuncType")
        .unwrap()
        .call_tuple((Array::new(vec![argument_type]), return_type))
        .unwrap()
        .try_into()
        .unwrap();
    assert_structural_equal(&function_type, &native_function_type);

    let tensor_map_type = TensorMapType::new();
    let native_tensor_map_type: TensorMapType = Function::get_global("tirx.TensorMapType")
        .unwrap()
        .call_tuple((Option::<Span>::None,))
        .unwrap()
        .try_into()
        .unwrap();
    assert_structural_equal(&tensor_map_type, &native_tensor_map_type);
}

#[test]
fn pointer_type_constructor_matches_native_defaults() {
    load_tvm_compiler();

    let element_type = PrimType::new("float32").unwrap();
    let pointer = PointerType::new(element_type.clone(), "").unwrap();
    assert_eq!(pointer.storage_scope(), "global");
    assert!(pointer.element_type().same_as(&element_type));

    let native: PointerType = Function::get_global("ir.PointerType")
        .unwrap()
        .call_tuple((Type::from(element_type), String::from("")))
        .unwrap()
        .try_into()
        .unwrap();
    assert_structural_equal(&pointer, &native);
    assert!(PointerType::new(Type::missing(), "global").is_err());
}
