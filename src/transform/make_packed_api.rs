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

use std::collections::HashMap;

use tvm_ffi::{
    structural_mutate, Any, Array, DLDataTypeExt, MapValue, Mutator, ObjectIdentity, ObjectRefCast,
    ObjectRefCore, Result, String as FfiString, TypeIndex,
};

use super::utils::{
    cast_prim_expr, get_operator, int_dtype_and_value, int_value, is_buffer_var, is_pointer_type,
    mutate_stmt_expr_default, value_error, variable_name, with_prim_func_body,
};
use super::{create_module_pass, Pass};
use crate::analysis::Analyzer;
use crate::ir::prim::StringImm;
use crate::ir::{
    BaseFunc, Call, DictAttrs, Expr, GlobalVar, IRModule, IntImm, PointerType, PrimExpr, PrimType,
    Type, Var,
};
use crate::target::Target;
use crate::tirx::{
    AttrStmt, Bind, DeclBuffer, Evaluate, For, ForKind, IfThenElse, PrimFunc, Return, Stmt,
};

const CALLING_CONV: &str = "calling_conv";
const GLOBAL_SYMBOL: &str = "global_symbol";
const TARGET: &str = "target";
const COMPUTE_SCOPE: &str = "compute_scope";
const CPACKED_FUNC: i64 = 1;

const FFI_ANY_TYPE_INDEX: i64 = 13;
const FFI_ANY_ZERO_PADDING: i64 = 14;
const FFI_ANY_UNION_VALUE: i64 = 15;
const INT64_ARRAY_ELEMENT: i64 = 17;
const DLTENSOR_DATA: i64 = 1;
const DLTENSOR_SHAPE: i64 = 2;
const DLTENSOR_STRIDES: i64 = 3;
const DLTENSOR_NDIM: i64 = 4;
const DLTENSOR_TYPE_CODE: i64 = 5;
const DLTENSOR_TYPE_BITS: i64 = 6;
const DLTENSOR_TYPE_LANES: i64 = 7;
const DLTENSOR_BYTE_OFFSET: i64 = 8;
const DLTENSOR_DEVICE_ID: i64 = 9;
const DLTENSOR_DEVICE_TYPE: i64 = 10;

/// Lower a module to TVM's C packed-function calling convention in Rust.
pub fn make_packed_api_module(module: IRModule) -> Result<IRModule> {
    let packed_symbols = module
        .functions
        .iter()
        .filter_map(|(global, function)| {
            let function = function.try_cast::<PrimFunc>().ok()?;
            requires_packed_api(&function)
                .transpose()
                .map(|result| result.map(|symbol| (ObjectIdentity::of(&global), symbol)))
        })
        .collect::<Result<HashMap<_, _>>>()?;

    let mut result = module.copy_for_update()?;
    for (global, function) in module.functions.iter() {
        let Ok(mut function) = function.try_cast::<PrimFunc>() else {
            continue;
        };
        function = rewrite_subroutine_calls(function, &packed_symbols)?;
        function = make_packed_api_prim_func(function)?;
        result = result.update_function_owned(&global, &BaseFunc::from(function))?;
    }
    Ok(result)
}

/// Lower one externally visible PrimFunc to TVM's C packed-function ABI.
pub fn make_packed_api_prim_func(function: PrimFunc) -> Result<PrimFunc> {
    let Some(global_symbol) = requires_packed_api(&function)? else {
        return Ok(function);
    };
    let target: Target = function
        .attrs
        .dict
        .get(&FfiString::from(TARGET))?
        .ok_or_else(|| value_error("MakePackedAPI requires the target attribute"))?
        .try_into()?;
    let Some(host_target) = target.host()? else {
        return Ok(function);
    };

    let void_pointer = PointerType::new(PrimType::void(), "")?;
    let self_handle = Var::with_type("self_handle", void_pointer.clone());
    let packed_args = Var::with_type("args", void_pointer.clone());
    let num_args = Var::new("num_args", "int32")?;
    let result = Var::with_type("result", void_pointer);

    let mut binder = PackedAbiBinder::new(
        global_symbol.as_str(),
        &function.params,
        packed_args.clone(),
        num_args.clone(),
        target.device_type()?,
    )?;
    binder.decode_all()?;

    let rewritten = rewrite_returns(function.body.clone(), result.clone())?;
    let mut device_setup = Vec::new();
    let mut set_device = None;
    if binder.device_id_is_bound() {
        let no_op = Evaluate::from_i64(0)?;
        device_setup.push(
            AttrStmt::new(
                FfiString::from("default"),
                "device_id",
                binder.device_id.clone(),
                no_op.clone(),
            )?
            .into(),
        );
        device_setup.push(
            AttrStmt::new(
                FfiString::from("default"),
                "device_type",
                IntImm::new("int32", i64::from(binder.device_type))?,
                no_op,
            )?
            .into(),
        );
        if binder.device_type != 1 {
            set_device = Some(
                Evaluate::new(Call::new(
                    PrimType::new("int32")?,
                    get_operator("tirx.tvm_call_packed")?,
                    vec![
                        StringImm::new("__tvm_set_device").into(),
                        IntImm::new("int32", i64::from(binder.device_type))?.into(),
                        binder.device_id.clone().into(),
                    ],
                ))?
                .into(),
            );
        }
    }
    let compute_scope = AttrStmt::new(
        0_i64,
        COMPUTE_SCOPE,
        StringImm::new(&format!("{}_compute_", global_symbol.as_str())),
        rewritten,
    )?;

    let scoped_body: Stmt = compute_scope.into();
    let scoped_body = if let Some(set_device) = set_device {
        crate::tirx::SeqStmt::new(vec![set_device, scoped_body])?.into()
    } else {
        scoped_body
    };
    let body = crate::tirx::SeqStmt::new(vec![
        scoped_body,
        Return::new(IntImm::new("int32", 0)?).into(),
    ])?
    .into();
    let body = merge_nest(&binder.buffer_declarations, body)?;
    let body = merge_nest(&binder.assertions, body)?;
    let body = merge_nest(&device_setup, body)?;
    let body = merge_nest(&binder.initialization, body)?;

    let attributes = replace_attributes(
        &function.attrs,
        [
            (CALLING_CONV, Any::from(CPACKED_FUNC)),
            (TARGET, Any::from(host_target)),
            (
                GLOBAL_SYMBOL,
                Any::from(FfiString::from(format!(
                    "__tvm_ffi_{}",
                    global_symbol.as_str()
                ))),
            ),
        ],
    );
    PrimFunc::with_metadata(
        vec![self_handle, packed_args, num_args, result],
        body,
        PrimType::new("int32")?,
        attributes,
        function.span.as_ref(),
    )
}

/// Build TVM's `tirx.MakePackedAPI` module pass in Rust.
pub fn make_packed_api() -> Result<Pass> {
    create_module_pass(
        "tirx.MakePackedAPI",
        0,
        Vec::new(),
        false,
        make_packed_api_module,
    )
}

fn requires_packed_api(function: &PrimFunc) -> Result<Option<FfiString>> {
    if function
        .attrs
        .dict
        .get(&FfiString::from(CALLING_CONV))?
        .map(i64::try_from)
        .transpose()?
        .is_some_and(|convention| convention != 0)
    {
        return Ok(None);
    }
    function
        .attrs
        .dict
        .get(&FfiString::from(GLOBAL_SYMBOL))?
        .map(FfiString::try_from)
        .transpose()
}

fn rewrite_subroutine_calls(
    function: PrimFunc,
    packed_symbols: &HashMap<ObjectIdentity, FfiString>,
) -> Result<PrimFunc> {
    let mut rewriter = SubroutineCallRewriter {
        packed_symbols,
        cpacked_operator: get_operator("tirx.tvm_call_cpacked")?,
    };
    let body: Stmt = structural_mutate(function.body.clone(), &mut rewriter)?.try_into()?;
    if body.same_as(&function.body) {
        Ok(function)
    } else {
        Ok(with_prim_func_body(function, body))
    }
}

struct SubroutineCallRewriter<'a> {
    packed_symbols: &'a HashMap<ObjectIdentity, FfiString>,
    cpacked_operator: Expr,
}

#[tvm_ffi::dispatch(mutate)]
impl SubroutineCallRewriter<'_> {
    fn mutate_call(&mut self, _value: Call, mutator: &mut Mutator) -> Result<Expr> {
        let value: Call = mutator.default_mutate(self)?.try_into()?;
        let Ok(global) = value.op.clone().try_cast::<GlobalVar>() else {
            return Ok(value.into());
        };
        let Some(symbol) = self.packed_symbols.get(&ObjectIdentity::of(&global)) else {
            return Ok(value.into());
        };
        let mut arguments = Vec::with_capacity(value.args.len() + 2);
        arguments.push(StringImm::new(symbol.as_str()).into());
        arguments.extend(value.args.iter());
        arguments.push(const_handle(0)?);
        Ok(Call::new(value.ty.clone(), self.cpacked_operator.clone(), arguments).into())
    }

    fn mutate_default(&mut self, value: &MapValue, mutator: &mut Mutator) -> Result<Any> {
        mutate_stmt_expr_default(self, mutator, value)
    }
}

fn rewrite_returns(body: Stmt, result: Var) -> Result<Stmt> {
    structural_mutate(
        body,
        &mut ReturnRewriter {
            result,
            parallel_depth: 0,
            struct_set_operator: get_operator("tirx.tvm_struct_set")?,
        },
    )?
    .try_into()
}

struct ReturnRewriter {
    result: Var,
    parallel_depth: usize,
    struct_set_operator: Expr,
}

#[tvm_ffi::dispatch(mutate)]
impl ReturnRewriter {
    fn mutate_loop(&mut self, value: For, mutator: &mut Mutator) -> Result<Stmt> {
        let is_parallel = value.kind == ForKind::kParallel;
        self.parallel_depth += usize::from(is_parallel);
        let rewritten = mutator.default_mutate(self).and_then(Stmt::try_from);
        self.parallel_depth -= usize::from(is_parallel);
        rewritten
    }

    fn mutate_return(&mut self, value: Return, mutator: &mut Mutator) -> Result<Stmt> {
        if self.parallel_depth != 0 {
            return Err(value_error("Return cannot be used in parallel scope"));
        }
        let value: Expr = mutator.mutate(self, &value.value)?.try_into()?;
        let (type_index, value) = convert_return_value(value)?;
        Stmt::sequence(vec![
            Evaluate::new(Call::new(
                PrimType::new("int32")?,
                self.struct_set_operator.clone(),
                vec![
                    self.result.clone().into(),
                    IntImm::new("int32", 0)?.into(),
                    IntImm::new("int32", FFI_ANY_TYPE_INDEX)?.into(),
                    IntImm::new("int32", type_index)?.into(),
                ],
            ))?
            .into(),
            Evaluate::new(Call::new(
                PrimType::new("int32")?,
                self.struct_set_operator.clone(),
                vec![
                    self.result.clone().into(),
                    IntImm::new("int32", 0)?.into(),
                    IntImm::new("int32", FFI_ANY_ZERO_PADDING)?.into(),
                    IntImm::new("int32", 0)?.into(),
                ],
            ))?
            .into(),
            Evaluate::new(Call::new(
                PrimType::new("int32")?,
                self.struct_set_operator.clone(),
                vec![
                    self.result.clone().into(),
                    IntImm::new("int32", 0)?.into(),
                    IntImm::new("int32", FFI_ANY_UNION_VALUE)?.into(),
                    value,
                ],
            ))?
            .into(),
            Return::new(IntImm::new("int32", 0)?).into(),
        ])
    }

    fn mutate_default(&mut self, value: &MapValue, mutator: &mut Mutator) -> Result<Any> {
        mutate_stmt_expr_default(self, mutator, value)
    }
}

fn convert_return_value(value: Expr) -> Result<(i64, Expr)> {
    if is_pointer_type(&value.ty) {
        return Ok((TypeIndex::kTVMFFIOpaquePtr as i64, value));
    }
    let primitive = value.ty.clone().try_cast::<PrimType>()?;
    let dtype = primitive.dtype;
    let converted = PrimExpr::try_from(value)?;
    if dtype.code == tvm_ffi::DLDataTypeCode::kDLBool as u8 {
        Ok((
            TypeIndex::kTVMFFIBool as i64,
            Expr::from(cast_prim_expr(converted, PrimType::new("int64")?)?),
        ))
    } else if matches!(
        dtype.code,
        code if code == tvm_ffi::DLDataTypeCode::kDLInt as u8
            || code == tvm_ffi::DLDataTypeCode::kDLUInt as u8
    ) {
        Ok((
            TypeIndex::kTVMFFIInt as i64,
            Expr::from(cast_prim_expr(converted, PrimType::new("int64")?)?),
        ))
    } else if dtype.code == tvm_ffi::DLDataTypeCode::kDLFloat as u8 {
        Ok((
            TypeIndex::kTVMFFIFloat as i64,
            Expr::from(cast_prim_expr(converted, PrimType::new("float64")?)?),
        ))
    } else if dtype.code == tvm_ffi::DLDataTypeCode::kDLOpaqueHandle as u8
        && dtype.bits == 0
        && dtype.lanes == 0
    {
        Ok((TypeIndex::kTVMFFINone as i64, converted.into()))
    } else {
        Err(value_error(
            "MakePackedAPI does not support this return type",
        ))
    }
}

struct PackedAbiBinder {
    signature: String,
    signature_imm: StringImm,
    when_calling_imm: StringImm,
    parameters: Array<Var>,
    packed_args: Var,
    num_args: Var,
    device_type: i32,
    device_id: Var,
    initialization: Vec<Stmt>,
    assertions: Vec<Stmt>,
    pending_assertions: Vec<Stmt>,
    buffer_declarations: Vec<Stmt>,
    definitions: HashMap<ObjectIdentity, PrimExpr>,
    buffer_handles: HashMap<ObjectIdentity, Var>,
    struct_get_operator: Expr,
    is_null_operator: Expr,
    reinterpret_operator: Expr,
    if_then_else_operator: Expr,
    handle_add_byte_offset_operator: Expr,
    analyzer: Analyzer,
}

impl PackedAbiBinder {
    fn new(
        function_name: &str,
        parameters: &Array<Var>,
        packed_args: Var,
        num_args: Var,
        device_type: i32,
    ) -> Result<Self> {
        let signature = function_signature(function_name, parameters);
        let signature_imm = StringImm::new(&signature);
        let mut binder = Self {
            signature,
            signature_imm,
            when_calling_imm: StringImm::new(" when calling:\n  `"),
            parameters: parameters.clone(),
            packed_args,
            num_args,
            device_type,
            device_id: Var::new("dev_id", "int32")?,
            initialization: Vec::new(),
            assertions: Vec::new(),
            pending_assertions: Vec::new(),
            buffer_declarations: Vec::new(),
            definitions: HashMap::new(),
            buffer_handles: HashMap::new(),
            struct_get_operator: get_operator("tirx.tvm_struct_get")?,
            is_null_operator: get_operator("tirx.isnullptr")?,
            reinterpret_operator: get_operator("tirx.reinterpret")?,
            if_then_else_operator: get_operator("ir.prim.if_then_else")?,
            handle_add_byte_offset_operator: get_operator("tirx.handle_add_byte_offset")?,
            analyzer: Analyzer::new()?,
        };
        let count = IntImm::new("int32", i64::try_from(parameters.len()).unwrap_or(i64::MAX))?;
        binder.emit_assert(
            equal(binder.num_args.clone(), count)?,
            "TypeError",
            vec![
                "Expected ".to_owned(),
                parameters.len().to_string(),
                " arguments".to_owned(),
                " when calling:\n  `".to_owned(),
                binder.signature.clone(),
                "`".to_owned(),
            ],
        )?;
        if !parameters.is_empty() {
            binder.emit_assert(
                not(is_null(
                    &binder.is_null_operator,
                    binder.packed_args.clone().into(),
                )?)?,
                "TypeError",
                vec![
                    "args pointer is NULL".to_owned(),
                    " when calling:\n  `".to_owned(),
                    binder.signature.clone(),
                    "`".to_owned(),
                ],
            )?;
        }
        Ok(binder)
    }

    fn device_id_is_bound(&self) -> bool {
        self.definitions
            .contains_key(&ObjectIdentity::of(&self.device_id))
    }

    fn decode_all(&mut self) -> Result<()> {
        let parameters = self.parameters.iter().collect::<Vec<_>>();
        for (index, parameter) in parameters.iter().cloned().enumerate() {
            self.decode_parameter(index, parameter)?;
        }
        // Buffer parameter decoding is kept in a separate phase so scalar
        // shape variables are defined before buffer-shape checks use them.
        for (index, parameter) in parameters.into_iter().enumerate() {
            if is_buffer_var(&parameter) {
                self.decode_buffer(index, parameter)?;
            }
        }
        self.assertions.append(&mut self.pending_assertions);
        Ok(())
    }

    fn decode_parameter(&mut self, index: usize, parameter: Var) -> Result<()> {
        let type_index_var = Var::new(&format!("{}.type_index", parameter.name.as_str()), "int32")?;
        let loaded_type = self.struct_get(
            PrimType::new("int32")?.into(),
            self.packed_args.clone().into(),
            index,
            FFI_ANY_TYPE_INDEX,
        )?;
        self.initialization
            .push(Bind::new(type_index_var.clone(), loaded_type)?.into());
        let type_index_expr = PrimExpr::try_from(Expr::from(type_index_var))?;

        if is_buffer_var(&parameter) {
            let handle = Var::with_type(
                &format!("{}.handle", parameter.name.as_str()),
                PointerType::new(PrimType::void(), "")?,
            );
            let value = self.decode_opaque_handle(index, &type_index_expr, "Tensor")?;
            self.initialization
                .push(Bind::new(handle.clone(), value)?.into());
            self.buffer_handles
                .insert(ObjectIdentity::of(&parameter), handle);
            return Ok(());
        }
        if is_pointer_type(&parameter.ty) {
            let value = self.decode_opaque_handle(index, &type_index_expr, "pointer")?;
            let value = Call::new(
                parameter.ty.clone(),
                self.reinterpret_operator.clone(),
                vec![value],
            );
            self.initialization
                .push(Bind::new(parameter, value)?.into());
            return Ok(());
        }

        let primitive = parameter.ty.clone().try_cast::<PrimType>()?;
        let dtype = primitive.dtype;
        let value = if dtype.code == tvm_ffi::DLDataTypeCode::kDLBool as u8 {
            self.emit_type_check(
                index,
                or(
                    equal(type_index_expr.clone(), type_index(TypeIndex::kTVMFFIBool)?)?,
                    equal(type_index_expr.clone(), type_index(TypeIndex::kTVMFFIInt)?)?,
                )?,
                "boolean",
            )?;
            cast_prim_expr(self.load_union_primitive(index, "int64")?, primitive)?
        } else if matches!(
            dtype.code,
            code if code == tvm_ffi::DLDataTypeCode::kDLInt as u8
                || code == tvm_ffi::DLDataTypeCode::kDLUInt as u8
        ) {
            self.emit_type_check(
                index,
                or(
                    equal(type_index_expr.clone(), type_index(TypeIndex::kTVMFFIInt)?)?,
                    equal(type_index_expr.clone(), type_index(TypeIndex::kTVMFFIBool)?)?,
                )?,
                "int",
            )?;
            cast_prim_expr(self.load_union_primitive(index, "int64")?, primitive)?
        } else if dtype.code == tvm_ffi::DLDataTypeCode::kDLFloat as u8 {
            self.emit_type_check(
                index,
                or(
                    equal(
                        type_index_expr.clone(),
                        type_index(TypeIndex::kTVMFFIFloat)?,
                    )?,
                    or(
                        equal(type_index_expr.clone(), type_index(TypeIndex::kTVMFFIInt)?)?,
                        equal(type_index_expr.clone(), type_index(TypeIndex::kTVMFFIBool)?)?,
                    )?,
                )?,
                "float",
            )?;
            let floating = cast_prim_expr(
                self.load_union_primitive(index, "float64")?,
                primitive.clone(),
            )?;
            let integer = cast_prim_expr(self.load_union_primitive(index, "int64")?, primitive)?;
            Call::new(
                floating.ty.clone(),
                self.if_then_else_operator.clone(),
                vec![
                    equal(type_index_expr, type_index(TypeIndex::kTVMFFIFloat)?)?.into(),
                    floating.into(),
                    integer.into(),
                ],
            )
            .try_cast::<PrimExpr>()?
        } else {
            return Err(value_error("unsupported packed parameter type"));
        };
        self.bind_scalar(parameter, value, true)
    }

    fn decode_buffer(&mut self, index: usize, parameter: Var) -> Result<()> {
        let buffer = parameter.try_cast::<crate::tirx::BufferVar>()?;
        let ty = buffer.type_annotation();
        let handle = self
            .buffer_handles
            .get(&ObjectIdentity::of(buffer.as_var()))
            .cloned()
            .ok_or_else(|| value_error("missing decoded DLTensor handle"))?;

        self.emit_type_check(
            index,
            not(is_null(&self.is_null_operator, handle.clone().into())?)?,
            "Tensor",
        )?;
        let ndim = self.struct_get(
            PrimType::new("int32")?.into(),
            handle.clone().into(),
            0,
            DLTENSOR_NDIM,
        )?;
        self.emit_assert(
            equal(
                IntImm::new("int32", i64::try_from(ty.shape.len()).unwrap_or(i64::MAX))?,
                ndim,
            )?,
            "ValueError",
            self.buffer_error_parts(index, &buffer, "ndim", ty.shape.len().to_string()),
        )?;

        if !is_sub_byte_integer(&ty.dtype) {
            let code = self.struct_get(
                PrimType::new("uint8")?.into(),
                handle.clone().into(),
                0,
                DLTENSOR_TYPE_CODE,
            )?;
            let bits = self.struct_get(
                PrimType::new("uint8")?.into(),
                handle.clone().into(),
                0,
                DLTENSOR_TYPE_BITS,
            )?;
            let lanes = self.struct_get(
                PrimType::new("uint16")?.into(),
                handle.clone().into(),
                0,
                DLTENSOR_TYPE_LANES,
            )?;
            let dtype_matches = and(
                and(
                    equal(code, IntImm::new("uint8", i64::from(ty.dtype.dtype.code))?)?,
                    equal(bits, IntImm::new("uint8", i64::from(ty.dtype.dtype.bits))?)?,
                )?,
                equal(
                    lanes,
                    IntImm::new("uint16", i64::from(ty.dtype.dtype.lanes))?,
                )?,
            )?;
            self.emit_assert(
                dtype_matches,
                "TypeError",
                self.buffer_error_parts(
                    index,
                    &buffer,
                    "dtype",
                    ty.dtype.dtype.to_string().as_str().to_owned(),
                ),
            )?;
        }

        let int64_pointer = PointerType::new(PrimType::new("int64")?, "")?;
        let shape_pointer = Var::with_type(
            &format!("{}.{}_shape", self.function_name(), buffer.name.as_str()),
            int64_pointer.clone(),
        );
        let loaded_shape_pointer = self.struct_get(
            int64_pointer.clone().into(),
            handle.clone().into(),
            0,
            DLTENSOR_SHAPE,
        )?;
        self.initialization
            .push(Bind::new(shape_pointer.clone(), loaded_shape_pointer)?.into());
        if !is_sub_byte_integer(&ty.dtype) {
            for (axis, expected) in ty.shape.iter().enumerate() {
                let loaded: PrimExpr = self
                    .struct_get(
                        PrimType::new("int64")?.into(),
                        shape_pointer.clone().into(),
                        axis,
                        INT64_ARRAY_ELEMENT,
                    )?
                    .try_cast()?;
                let loaded = cast_prim_expr(loaded, expected.type_annotation())?;
                self.bind_expected(
                    &expected,
                    loaded,
                    true,
                    index,
                    &format!("{}.shape[{axis}]", buffer.name.as_str()),
                )?;
            }
        }

        let strides_pointer = Var::with_type(
            &format!("{}.{}_strides", self.function_name(), buffer.name.as_str()),
            int64_pointer,
        );
        let loaded_strides_pointer = self.struct_get(
            strides_pointer.ty.clone(),
            handle.clone().into(),
            0,
            DLTENSOR_STRIDES,
        )?;
        self.initialization
            .push(Bind::new(strides_pointer.clone(), loaded_strides_pointer)?.into());
        let strides_are_null = is_null(&self.is_null_operator, strides_pointer.clone().into())?;
        if ty.strides.is_empty() {
            self.bind_compact_strides(index, &buffer, &strides_pointer, strides_are_null)?;
        } else {
            self.bind_regular_strides(
                index,
                &buffer,
                &ty,
                &shape_pointer,
                &strides_pointer,
                strides_are_null,
            )?;
        }

        let byte_offset: PrimExpr = self
            .struct_get(
                PrimType::new("uint64")?.into(),
                handle.clone().into(),
                0,
                DLTENSOR_BYTE_OFFSET,
            )?
            .try_cast()?;
        let data_bytes = storage_bytes(ty.dtype.dtype);
        if let Some(offset) = int_value(&ty.elem_offset) {
            let expected: PrimExpr =
                IntImm::new("uint64", offset.saturating_mul(data_bytes))?.into();
            self.bind_expected(
                &expected,
                byte_offset,
                false,
                index,
                &format!("{}.byte_offset", buffer.name.as_str()),
            )?;
        } else {
            let element_offset = cast_prim_expr(
                divide(byte_offset, IntImm::new("uint64", data_bytes)?)?,
                ty.elem_offset.type_annotation(),
            )?;
            self.bind_expected(
                &ty.elem_offset,
                element_offset,
                true,
                index,
                &format!("{}.byte_offset", buffer.name.as_str()),
            )?;
        }

        let actual_device_type = self.struct_get(
            PrimType::new("int32")?.into(),
            handle.clone().into(),
            0,
            DLTENSOR_DEVICE_TYPE,
        )?;
        self.emit_assert(
            equal(
                actual_device_type,
                IntImm::new("int32", i64::from(self.device_type))?,
            )?,
            "ValueError",
            self.buffer_error_parts(
                index,
                &buffer,
                "device_type",
                device_type_name(self.device_type).to_owned(),
            ),
        )?;
        let actual_device_id: PrimExpr = self
            .struct_get(
                PrimType::new("int32")?.into(),
                handle.clone().into(),
                0,
                DLTENSOR_DEVICE_ID,
            )?
            .try_cast()?;
        self.bind_scalar(self.device_id.clone(), actual_device_id, true)?;

        let raw_data = self.struct_get(
            PointerType::new(PrimType::void(), "")?.into(),
            handle.into(),
            0,
            DLTENSOR_DATA,
        )?;
        let data_pointer = PointerType::new(ty.dtype.clone(), ty.storage_scope.as_str())?;
        let typed_data = Call::new(
            data_pointer,
            self.reinterpret_operator.clone(),
            vec![raw_data],
        );
        let index_type = ty
            .shape
            .get(0)
            .map(|shape| shape.type_annotation())
            .unwrap_or(PrimType::new("int64")?);
        let mut allocation_size: PrimExpr = IntImm::from_dtype(index_type.dtype, 1)?.into();
        for extent in ty.shape.iter() {
            allocation_size = self
                .analyzer
                .simplify(&multiply(allocation_size, extent)?)?;
        }
        let data_non_null = not(is_null(&self.is_null_operator, typed_data.clone().into())?)?;
        self.assertions.push(
            crate::tirx::AssertStmt::with_metadata(
                or(
                    equal(allocation_size, IntImm::from_dtype(index_type.dtype, 0)?)?,
                    data_non_null,
                )?,
                StringImm::new("ValueError"),
                vec![
                    StringImm::new(buffer.name.as_str()),
                    StringImm::new(" data pointer is NULL on argument #"),
                    StringImm::new(&index.to_string()),
                    self.when_calling_imm.clone(),
                    self.signature_imm.clone(),
                    StringImm::new("`,\n  expected non-NULL data pointer"),
                ],
                None,
            )?
            .into(),
        );
        self.initialization.push(
            AttrStmt::new(
                buffer.as_var().clone(),
                "storage_alignment",
                IntImm::new("int32", i64::from(ty.data_alignment))?,
                Evaluate::from_i64(0)?,
            )?
            .into(),
        );
        self.buffer_declarations
            .push(DeclBuffer::new(&buffer, typed_data)?.into());
        Ok(())
    }

    fn decode_opaque_handle(
        &mut self,
        index: usize,
        type_index_value: &PrimExpr,
        expected: &str,
    ) -> Result<Expr> {
        let condition = [
            equal(
                type_index_value.clone(),
                type_index(TypeIndex::kTVMFFINone)?,
            )?,
            equal(
                type_index_value.clone(),
                type_index(TypeIndex::kTVMFFIOpaquePtr)?,
            )?,
            equal(
                type_index_value.clone(),
                type_index(TypeIndex::kTVMFFIDLTensorPtr)?,
            )?,
            greater_equal(
                type_index_value.clone(),
                type_index(TypeIndex::kTVMFFIStaticObjectBegin)?,
            )?,
        ]
        .into_iter()
        .reduce(|left, right| or(left, right).expect("boolean operands have matching types"))
        .expect("packed handles have accepted type indices");
        self.emit_type_check(index, condition, expected)?;
        let void_pointer = PointerType::new(PrimType::void(), "")?;
        let value = self.struct_get(
            void_pointer.into(),
            self.packed_args.clone().into(),
            index,
            FFI_ANY_UNION_VALUE,
        )?;
        let tensor_data = Call::new(
            value.ty.clone(),
            self.handle_add_byte_offset_operator.clone(),
            vec![value.clone(), IntImm::new("int32", 24)?.into()],
        );
        Ok(Call::new(
            value.ty.clone(),
            self.if_then_else_operator.clone(),
            vec![
                equal(
                    type_index_value.clone(),
                    type_index(TypeIndex::kTVMFFITensor)?,
                )?
                .into(),
                tensor_data.into(),
                value,
            ],
        )
        .into())
    }

    fn load_union_primitive(&self, index: usize, dtype: &str) -> Result<PrimExpr> {
        self.struct_get(
            PrimType::new(dtype)?.into(),
            self.packed_args.clone().into(),
            index,
            FFI_ANY_UNION_VALUE,
        )?
        .try_cast()
    }

    fn struct_get(&self, ty: Type, pointer: Expr, index: usize, field: i64) -> Result<Expr> {
        Ok(Call::new(
            ty,
            self.struct_get_operator.clone(),
            vec![
                pointer,
                IntImm::new("int32", i64::try_from(index).unwrap_or(i64::MAX))?.into(),
                IntImm::new("int32", field)?.into(),
            ],
        )
        .into())
    }

    fn bind_scalar(&mut self, variable: Var, value: PrimExpr, emit_bind: bool) -> Result<()> {
        let identity = ObjectIdentity::of(&variable);
        if let Some(previous) = self.definitions.get(&identity) {
            let condition = equal(previous.clone(), value)?;
            self.assertions.push(
                crate::tirx::AssertStmt::new(condition, "ValueError", "Mismatched value")?.into(),
            );
        } else {
            self.definitions.insert(identity, value.clone());
            if emit_bind {
                self.initialization.push(Bind::new(variable, value)?.into());
            }
        }
        Ok(())
    }

    fn bind_expected(
        &mut self,
        expected: &PrimExpr,
        actual: PrimExpr,
        emit_bind: bool,
        index: usize,
        field: &str,
    ) -> Result<()> {
        if let Ok(variable) = expected.clone().try_cast::<Var>() {
            return self.bind_scalar(variable, actual, emit_bind);
        }
        let expected_is_unsigned = expected
            .type_annotation()
            .try_cast::<PrimType>()
            .is_ok_and(|ty| ty.dtype.code == tvm_ffi::DLDataTypeCode::kDLUInt as u8);
        let condition = if expected_is_unsigned {
            equal(expected.clone(), actual)?
        } else {
            equal(actual, expected.clone())?
        };
        self.pending_assertions.push(
            crate::tirx::AssertStmt::with_metadata(
                condition,
                StringImm::new("ValueError"),
                vec![
                    StringImm::new("Invalid "),
                    StringImm::new(field),
                    StringImm::new(" on argument #"),
                    StringImm::new(&index.to_string()),
                    self.when_calling_imm.clone(),
                    self.signature_imm.clone(),
                    StringImm::new("`,\n  expected "),
                    StringImm::new(&render_expected_expression(expected.clone())),
                ],
                None,
            )?
            .into(),
        );
        Ok(())
    }

    fn bind_compact_strides(
        &mut self,
        index: usize,
        buffer: &crate::tirx::BufferVar,
        strides_pointer: &Var,
        strides_are_null: PrimExpr,
    ) -> Result<()> {
        let ty = buffer.type_annotation();
        let index_type = ty
            .shape
            .get(0)
            .map(|shape| shape.type_annotation())
            .unwrap_or(PrimType::new("int64")?);
        let mut expected: PrimExpr = IntImm::from_dtype(index_type.dtype, 1)?.into();
        let mut conditions = Vec::new();
        for axis in (0..ty.shape.len()).rev() {
            let loaded: PrimExpr = self
                .struct_get(
                    PrimType::new("int64")?.into(),
                    strides_pointer.clone().into(),
                    axis,
                    INT64_ARRAY_ELEMENT,
                )?
                .try_cast()?;
            let loaded = cast_prim_expr(loaded, index_type.clone())?;
            let shape = ty.shape.get(axis)?;
            let stride_matches = equal(expected.clone(), loaded)?;
            let condition = if let Some(constant) = int_value(&shape) {
                if constant == 1 {
                    IntImm::new("bool", 1)?.into()
                } else {
                    stride_matches
                }
            } else {
                or(
                    equal(shape.clone(), IntImm::from_dtype(index_type.dtype, 1)?)?,
                    stride_matches,
                )?
            };
            conditions.push(condition);
            expected = self.analyzer.simplify(&multiply(expected, shape)?)?;
        }
        if conditions.is_empty() {
            return Ok(());
        }
        let mut conditions = conditions.into_iter();
        let mut condition = conditions.next().expect("non-empty strides condition");
        for next in conditions {
            condition = and(condition, next)?;
        }
        let assertion = crate::tirx::AssertStmt::with_metadata(
            condition,
            StringImm::new("ValueError"),
            vec![
                StringImm::new("Mismatched "),
                StringImm::new(buffer.name.as_str()),
                StringImm::new(".strides on argument #"),
                StringImm::new(&index.to_string()),
                self.when_calling_imm.clone(),
                self.signature_imm.clone(),
                StringImm::new("`,\n  expected to be compact array"),
            ],
            None,
        )?;
        let check = crate::tirx::IfThenElse::new(not(strides_are_null)?, assertion)?;
        self.assertions.push(
            crate::tirx::SeqStmt::new(vec![check.into(), Evaluate::from_i64(0)?.into()])?.into(),
        );
        Ok(())
    }

    fn bind_regular_strides(
        &mut self,
        index: usize,
        buffer: &crate::tirx::BufferVar,
        ty: &crate::tirx::BufferType,
        shape_pointer: &Var,
        strides_pointer: &Var,
        strides_are_null: PrimExpr,
    ) -> Result<()> {
        let index_type = ty
            .shape
            .get(0)
            .map(|shape| shape.type_annotation())
            .unwrap_or(PrimType::new("int64")?);
        let mut from_shape: PrimExpr = IntImm::from_dtype(index_type.dtype, 1)?.into();
        for axis in (0..ty.strides.len()).rev() {
            let explicit: PrimExpr = self
                .struct_get(
                    PrimType::new("int64")?.into(),
                    strides_pointer.clone().into(),
                    axis,
                    INT64_ARRAY_ELEMENT,
                )?
                .try_cast()?;
            let explicit = cast_prim_expr(explicit, index_type.clone())?;
            let actual: PrimExpr = Call::new(
                index_type.clone(),
                self.if_then_else_operator.clone(),
                vec![
                    strides_are_null.clone().into(),
                    from_shape.clone().into(),
                    explicit.into(),
                ],
            )
            .try_cast()?;
            self.bind_expected(
                &ty.strides.get(axis)?,
                actual,
                true,
                index,
                &format!("{}.strides[{axis}]", buffer.name.as_str()),
            )?;
            let shape: PrimExpr = self
                .struct_get(
                    PrimType::new("int64")?.into(),
                    shape_pointer.clone().into(),
                    axis,
                    INT64_ARRAY_ELEMENT,
                )?
                .try_cast()?;
            from_shape = self.analyzer.simplify(&multiply(
                from_shape,
                cast_prim_expr(shape, index_type.clone())?,
            )?)?;
        }
        Ok(())
    }

    fn function_name(&self) -> &str {
        self.signature
            .split_once('(')
            .map_or(self.signature.as_str(), |(name, _)| name)
    }

    fn buffer_error_parts(
        &self,
        index: usize,
        buffer: &crate::tirx::BufferVar,
        field: &str,
        expected: String,
    ) -> Vec<String> {
        vec![
            "Mismatched ".to_owned(),
            buffer.name.as_str().to_owned(),
            format!(".{field} on argument #"),
            index.to_string(),
            " when calling:\n  `".to_owned(),
            self.signature.clone(),
            "`,\n  expected ".to_owned(),
            expected,
        ]
    }

    fn emit_type_check(&mut self, index: usize, condition: PrimExpr, expected: &str) -> Result<()> {
        self.emit_assert(
            condition,
            "TypeError",
            vec![
                "Mismatched type on argument #".to_owned(),
                index.to_string(),
                " when calling:\n  `".to_owned(),
                self.signature.clone(),
                "`,\n  expected ".to_owned(),
                expected.to_owned(),
            ],
        )
    }

    fn emit_assert(
        &mut self,
        condition: PrimExpr,
        error_kind: &str,
        messages: Vec<String>,
    ) -> Result<()> {
        self.initialization.push(
            crate::tirx::AssertStmt::with_metadata(
                condition,
                StringImm::new(error_kind),
                messages
                    .into_iter()
                    .map(|message| {
                        if message == self.when_calling_imm.value.as_str() {
                            self.when_calling_imm.clone()
                        } else if message == self.signature {
                            self.signature_imm.clone()
                        } else {
                            StringImm::new(&message)
                        }
                    })
                    .collect(),
                None,
            )?
            .into(),
        );
        Ok(())
    }
}

fn function_signature(name: &str, parameters: &Array<Var>) -> String {
    let mut rendered = Vec::with_capacity(parameters.len());
    for parameter in parameters.iter() {
        let value = if let Ok(buffer) = parameter.clone().try_cast::<crate::tirx::BufferVar>() {
            let ty = buffer.type_annotation();
            let shape = ty
                .shape
                .iter()
                .map(render_signature_expression)
                .collect::<Vec<_>>()
                .join(", ");
            format!(
                "{}: Tensor([{}], {})",
                buffer.name.as_str(),
                shape,
                ty.dtype.dtype.to_string().as_str()
            )
        } else if let Ok(primitive) = parameter.ty.clone().try_cast::<PrimType>() {
            format!(
                "{}: {}",
                parameter.name.as_str(),
                primitive.dtype.to_string().as_str()
            )
        } else {
            format!("{}: pointer", parameter.name.as_str())
        };
        rendered.push(value);
    }
    format!("{name}({})", rendered.join(", "))
}

fn render_signature_expression(expression: PrimExpr) -> String {
    if let Some(name) = variable_name(&expression) {
        return if name.is_empty() {
            "v".to_owned()
        } else {
            name.to_owned()
        };
    }
    if let Some((dtype, value)) = int_dtype_and_value(&expression) {
        let dtype = dtype.to_string();
        return if dtype.as_str() == "int32" {
            value.to_string()
        } else {
            format!("T.{}({value})", dtype.as_str())
        };
    }
    "?".to_owned()
}

fn render_expected_expression(expression: PrimExpr) -> String {
    if let Some(value) = int_value(&expression) {
        return value.to_string();
    }
    render_signature_expression(expression)
}

fn replace_attributes<const N: usize>(
    attributes: &DictAttrs,
    replacements: [(&str, Any); N],
) -> DictAttrs {
    let mut entries = attributes.dict.iter().collect::<Vec<_>>();
    for (name, value) in replacements {
        entries.retain(|(key, _)| key.as_str() != name);
        entries.push((FfiString::from(name), value));
    }
    DictAttrs::from_dictionary(entries.into_iter().collect())
}

fn merge_nest(statements: &[Stmt], mut body: Stmt) -> Result<Stmt> {
    for statement in statements.iter().rev() {
        if let Ok(attribute) = statement.clone().try_cast::<AttrStmt>() {
            body = attribute
                .copy_with(
                    attribute.node.clone(),
                    attribute.attr_key.clone(),
                    attribute.value.clone(),
                    body,
                )
                .into();
        } else if statement.clone().try_cast::<Bind>().is_ok()
            || statement
                .clone()
                .try_cast::<crate::tirx::AssertStmt>()
                .is_ok()
            || statement.clone().try_cast::<DeclBuffer>().is_ok()
        {
            body = Stmt::sequence(vec![statement.clone(), body])?;
        } else if let Ok(conditional) = statement.clone().try_cast::<crate::tirx::IfThenElse>() {
            body = IfThenElse::from_complete_fields(
                conditional.span.clone(),
                conditional.condition.clone(),
                body,
                None,
            )
            .into();
        } else if let Ok(sequence) = statement.clone().try_cast::<crate::tirx::SeqStmt>() {
            let mut prefix = sequence.seq.iter().collect::<Vec<_>>();
            if !prefix.is_empty() {
                prefix.pop();
            }
            prefix.push(body);
            body = sequence.copy_with(Array::new(prefix)).into();
        } else {
            return Err(value_error(
                "unsupported statement in packed ABI binding nest",
            ));
        }
    }
    Ok(body)
}

fn const_handle(value: i64) -> Result<Expr> {
    let pointer = PointerType::new(PrimType::void(), "")?;
    Ok(Call::new(
        pointer,
        get_operator("tirx.reinterpret")?,
        vec![IntImm::new("uint64", value)?.into()],
    )
    .into())
}

fn type_index(index: TypeIndex) -> Result<PrimExpr> {
    Ok(IntImm::new("int32", index as i64)?.into())
}

fn is_null(operator: &Expr, value: Expr) -> Result<PrimExpr> {
    Call::new(PrimType::new("bool")?, operator.clone(), vec![value]).try_cast()
}

fn not(value: PrimExpr) -> Result<PrimExpr> {
    Ok(crate::ir::prim::Not::new(value)?.into())
}

fn equal<L, R>(lhs: L, rhs: R) -> Result<PrimExpr>
where
    L: Into<Expr>,
    R: Into<Expr>,
{
    Ok(crate::ir::prim::EQ::new(lhs, rhs)?.into())
}

fn greater_equal<L, R>(lhs: L, rhs: R) -> Result<PrimExpr>
where
    L: Into<Expr>,
    R: Into<Expr>,
{
    Ok(crate::ir::prim::GE::new(lhs, rhs)?.into())
}

fn or<L, R>(lhs: L, rhs: R) -> Result<PrimExpr>
where
    L: Into<Expr>,
    R: Into<Expr>,
{
    Ok(crate::ir::prim::Or::new(lhs, rhs)?.into())
}

fn and<L, R>(lhs: L, rhs: R) -> Result<PrimExpr>
where
    L: Into<Expr>,
    R: Into<Expr>,
{
    Ok(crate::ir::prim::And::new(lhs, rhs)?.into())
}

fn multiply<L, R>(lhs: L, rhs: R) -> Result<PrimExpr>
where
    L: Into<Expr>,
    R: Into<Expr>,
{
    Ok(crate::ir::prim::Mul::new(lhs, rhs)?.into())
}

fn divide<L, R>(lhs: L, rhs: R) -> Result<PrimExpr>
where
    L: Into<Expr>,
    R: Into<Expr>,
{
    Ok(crate::ir::prim::Div::new(lhs, rhs)?.into())
}

fn is_sub_byte_integer(ty: &PrimType) -> bool {
    matches!(
        (ty.dtype.code, ty.dtype.bits),
        (code, 1 | 4)
            if code == tvm_ffi::DLDataTypeCode::kDLInt as u8
                || code == tvm_ffi::DLDataTypeCode::kDLUInt as u8
    )
}

fn storage_bytes(dtype: tvm_ffi::DLDataType) -> i64 {
    let bits = i64::from(dtype.bits).saturating_mul(i64::from(dtype.lanes));
    (bits.saturating_add(7)) / 8
}

fn device_type_name(device_type: i32) -> &'static str {
    match device_type {
        1 => "cpu",
        2 => "cuda",
        4 => "opencl",
        7 => "vulkan",
        8 => "metal",
        10 => "rocm",
        15 => "webgpu",
        _ => "device",
    }
}
