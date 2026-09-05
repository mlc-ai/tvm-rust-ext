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

use tvm_ffi::{
    structural_mutate, Any, Array, DLDataTypeCode, DLDeviceType, MapValue, Mutator, ObjectRefCast,
    ObjectRefCore, Result, String, TypeIndex,
};

use super::utils::{
    binary_op, get_operator, int_value, is_pointer_type, is_string_imm, mutate_expr_default,
    mutate_stmt_default, mutate_stmt_expr_default, value_error, with_prim_func_body,
};
use super::{create_prim_func_pass, Pass};
use crate::ir::prim::{Cast, StringImm};
use crate::ir::{
    Call, CallObj, Expr, IntImm, PointerType, PrimExpr, PrimType, TensorLoad, Type, Var,
};
use crate::target::Target;
use crate::tirx::{
    AllocBuffer, AttrStmt, Bind, BufferStore, BufferType, BufferVar, DeclBuffer, Evaluate, For,
    ForKind, IfThenElse, PrimFunc, Stmt,
};

const DEVICE_ID: &str = "device_id";
const DEVICE_TYPE: &str = "device_type";
const DISABLE_LOWER_BUILTIN: &str = "disable_lower_builtin";
const MAX_STACK_ALLOCA: i64 = 1024;

const DLTENSOR_ADDR: i64 = 0;
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
const TVM_FFI_ANY_TYPE_INDEX: i64 = 13;
const TVM_FFI_ANY_ZERO_PADDING: i64 = 14;
const TVM_FFI_ANY_UNION_VALUE: i64 = 15;

/// Lower host-side TVM builtins to the explicit runtime ABI.
pub fn lower_tvm_builtin_prim_func(function: PrimFunc) -> Result<PrimFunc> {
    if !is_host_function(&function)? {
        return Ok(function);
    }
    let device_type = function_target(&function)?
        .map(|target| target.device_type())
        .transpose()?
        .map(|value| IntImm::new("int32", i64::from(value)).map(Into::into))
        .transpose()?;
    let mut lowerer = BuiltinLower::new(device_type)?;
    let body = lowerer.visit_body_and_realize_alloca(function.body.clone())?;
    Ok(with_prim_func_body(function, body))
}

/// Build TVM's `tirx.LowerTVMBuiltin` pass in Rust.
pub fn lower_tvm_builtin() -> Result<Pass> {
    create_prim_func_pass(
        "tirx.LowerTVMBuiltin",
        0,
        Vec::new(),
        false,
        lower_tvm_builtin_prim_func,
    )
}

#[derive(Clone, Copy)]
struct StackSizes {
    shape: i64,
    array: u64,
    arguments: u64,
}

impl Default for StackSizes {
    fn default() -> Self {
        Self {
            shape: -1,
            array: 0,
            arguments: 0,
        }
    }
}

impl StackSizes {
    fn include(&mut self, other: Self) {
        self.shape = self.shape.max(other.shape);
        self.array = self.array.max(other.array);
        self.arguments = self.arguments.max(other.arguments);
    }

    fn contains(&self, other: Self) -> bool {
        self.shape >= other.shape && self.array >= other.array && self.arguments >= other.arguments
    }
}

struct AllocaScope {
    stack_shape: BufferVar,
    stack_array: Var,
    stack_any: Var,
    maximum: StackSizes,
    current: StackSizes,
}

impl AllocaScope {
    fn new(maximum: StackSizes) -> Result<Self> {
        let void_pointer = PointerType::new(PrimType::new("void")?, "")?;
        Ok(Self {
            stack_shape: stack_shape_buffer(0)?,
            stack_array: Var::with_type("stack_array", void_pointer.clone()),
            stack_any: Var::with_type("stack_ffi_any", void_pointer),
            maximum,
            current: StackSizes::default(),
        })
    }
}

struct BuiltinOperators {
    call_packed: Expr,
    call_cpacked: Expr,
    call_trace_packed: Expr,
    anylist_setitem_packed: Expr,
    anylist_setitem_cpacked: Expr,
    call_packed_lowered: Expr,
    call_cpacked_lowered: Expr,
    call_trace_packed_lowered: Expr,
    stack_make_shape: Expr,
    stack_make_array: Expr,
    context_id: Expr,
    dma_copy: Expr,
    dma_wait: Expr,
    dma_start_group: Expr,
    dma_end_group: Expr,
    nd_mem_alloc_with_scope: Expr,
    anylist_getitem: Expr,
    struct_get: Expr,
}

impl BuiltinOperators {
    fn new() -> Result<Self> {
        Ok(Self {
            call_packed: get_operator("tirx.tvm_call_packed")?,
            call_cpacked: get_operator("tirx.tvm_call_cpacked")?,
            call_trace_packed: get_operator("tirx.tvm_call_trace_packed")?,
            anylist_setitem_packed: get_operator("tirx.anylist_setitem_call_packed")?,
            anylist_setitem_cpacked: get_operator("tirx.anylist_setitem_call_cpacked")?,
            call_packed_lowered: get_operator("tirx.tvm_call_packed_lowered")?,
            call_cpacked_lowered: get_operator("tirx.tvm_call_cpacked_lowered")?,
            call_trace_packed_lowered: get_operator("tirx.tvm_call_trace_packed_lowered")?,
            stack_make_shape: get_operator("tirx.tvm_stack_make_shape")?,
            stack_make_array: get_operator("tirx.tvm_stack_make_array")?,
            context_id: get_operator("tirx.tvm_context_id")?,
            dma_copy: get_operator("tirx.dma_copy")?,
            dma_wait: get_operator("tirx.dma_wait")?,
            dma_start_group: get_operator("tirx.dma_start_group")?,
            dma_end_group: get_operator("tirx.dma_end_group")?,
            nd_mem_alloc_with_scope: get_operator("tirx.nd_mem_alloc_with_scope")?,
            anylist_getitem: get_operator("tirx.anylist_getitem")?,
            struct_get: get_operator("tirx.tvm_struct_get")?,
        })
    }
}

struct BuiltinLower {
    operators: BuiltinOperators,
    device_type: Option<PrimExpr>,
    device_id: Option<PrimExpr>,
    precheck: bool,
    alloca_scopes: Vec<AllocaScope>,
    preparation: Vec<Vec<Stmt>>,
    pending_frees: Vec<Vec<Stmt>>,
}

impl BuiltinLower {
    fn new(device_type: Option<PrimExpr>) -> Result<Self> {
        Ok(Self {
            operators: BuiltinOperators::new()?,
            device_type,
            device_id: None,
            precheck: false,
            alloca_scopes: Vec::new(),
            preparation: Vec::new(),
            pending_frees: Vec::new(),
        })
    }

    fn get_maximum_stack(&self, statement: &Stmt) -> Result<StackSizes> {
        let mut precheck = Self::new(self.device_type.clone())?;
        precheck.device_id = self.device_id.clone();
        precheck.precheck = true;
        precheck
            .alloca_scopes
            .push(AllocaScope::new(StackSizes::default())?);
        precheck.pending_frees.push(Vec::new());
        let _: Stmt = structural_mutate(statement.clone(), &mut precheck)?.try_into()?;
        Ok(precheck
            .alloca_scopes
            .pop()
            .expect("precheck alloca scope must remain balanced")
            .maximum)
    }

    fn visit_body_and_realize_alloca(&mut self, statement: Stmt) -> Result<Stmt> {
        if self.precheck {
            return Ok(statement);
        }
        let maximum = self.get_maximum_stack(&statement)?;
        let mut scope = AllocaScope::new(maximum)?;
        let mut prefix = Vec::new();
        if maximum.arguments != 0 {
            prefix.push(
                Bind::new(
                    scope.stack_any.clone(),
                    stack_alloca(scope.stack_any.ty.clone(), "tvm_ffi_any", maximum.arguments)?,
                )?
                .into(),
            );
        }
        if maximum.array != 0 {
            prefix.push(
                Bind::new(
                    scope.stack_array.clone(),
                    stack_alloca(scope.stack_array.ty.clone(), "array", maximum.array)?,
                )?
                .into(),
            );
        }
        let statement = if maximum.shape != -1 {
            scope.stack_shape = stack_shape_buffer(maximum.shape)?;
            let pointer_type = buffer_pointer_type(&scope.stack_shape)?;
            Stmt::sequence(vec![
                DeclBuffer::new(
                    scope.stack_shape.clone(),
                    stack_alloca(pointer_type.into(), "shape", maximum.shape as u64)?,
                )?
                .into(),
                statement,
            ])?
        } else {
            statement
        };

        self.alloca_scopes.push(scope);
        self.pending_frees.push(Vec::new());
        let result = structural_mutate(statement, &mut *self).and_then(Stmt::try_from);
        let result = result.and_then(|body| self.append_pending_frees(body));
        self.pending_frees.pop();
        self.alloca_scopes.pop();
        let mut body = result?;
        if !prefix.is_empty() {
            prefix.push(body);
            body = Stmt::sequence(prefix)?;
        }
        Ok(body)
    }

    fn append_pending_frees(&mut self, body: Stmt) -> Result<Stmt> {
        let Some(frees) = self.pending_frees.last_mut() else {
            return Ok(body);
        };
        if frees.is_empty() {
            return Ok(body);
        }
        let mut statements = vec![body];
        statements.extend(frees.drain(..).rev());
        Stmt::sequence(statements)
    }

    fn with_free_scope<F>(&mut self, operation: F) -> Result<Stmt>
    where
        F: FnOnce(&mut Self) -> Result<Stmt>,
    {
        self.pending_frees.push(Vec::new());
        let result = operation(self).and_then(|body| self.append_pending_frees(body));
        self.pending_frees.pop();
        result
    }

    fn mutate_statement_inner(&mut self, value: Stmt, mutator: &mut Mutator) -> Result<Stmt> {
        if let Ok(binding) = value.clone().try_cast::<Bind>() {
            if let Ok(call) = binding.value.clone().try_cast::<Call>() {
                if call.op.same_as(&self.operators.nd_mem_alloc_with_scope) {
                    return self.make_nd_memory_allocation(binding, call, mutator);
                }
            }
        }
        if let Ok(allocation) = value.clone().try_cast::<AllocBuffer>() {
            return self.mutate_allocation(allocation);
        }
        if let Ok(attribute) = value.clone().try_cast::<AttrStmt>() {
            return self.mutate_attribute(attribute, mutator);
        }
        if let Ok(loop_node) = value.clone().try_cast::<For>() {
            return self.mutate_loop(loop_node, mutator);
        }
        if let Ok(conditional) = value.clone().try_cast::<IfThenElse>() {
            return self.mutate_conditional(conditional, mutator);
        }
        mutate_stmt_default(self, mutator, value)
    }

    fn mutate_allocation(&mut self, value: AllocBuffer) -> Result<Stmt> {
        if value
            .annotations
            .get(&String::from(DISABLE_LOWER_BUILTIN))?
            .map(IntImm::try_from)
            .transpose()?
            .is_some_and(|flag| flag.value != 0)
        {
            return Ok(value.into());
        }
        let buffer_type = value.buffer.type_annotation();
        if is_scalable_vector(&buffer_type.dtype) {
            return Ok(value.into());
        }
        let element_bytes = storage_bytes(&buffer_type.dtype)?;
        if self.device_type.as_ref().and_then(int_value) == Some(DLDeviceType::kDLCPU as i64)
            && buffer_type.storage_scope.as_str() == "global"
        {
            if let Some(elements) = constant_allocation_size(&buffer_type.shape) {
                if elements > 0 && elements.saturating_mul(element_bytes) < MAX_STACK_ALLOCA {
                    return Ok(value.into());
                }
            }
        }
        let device_type = self
            .device_type
            .clone()
            .ok_or_else(|| value_error("allocation requires a device type"))?;
        let device_id = self
            .device_id
            .clone()
            .ok_or_else(|| value_error("allocation requires a device id"))?;
        let mut total_bytes: PrimExpr = IntImm::new("uint64", element_bytes)?.into();
        for extent in buffer_type.shape.iter() {
            total_bytes = binary_op("tirx._OpMul", total_bytes, extent)?;
        }
        let throw = Evaluate::new(Call::new(
            PrimType::new("int32")?,
            get_operator("tirx.tvm_throw_last_error")?,
            Vec::new(),
        ))?;
        let data = value.buffer.data()?;
        let null_check = IfThenElse::new(
            Call::new(
                PrimType::new("bool")?,
                get_operator("tirx.isnullptr")?,
                vec![data.clone()],
            ),
            throw.clone(),
        )?;
        let free_call = Call::new(
            PrimType::new("int32")?,
            get_operator("tirx.TVMBackendFreeWorkspace")?,
            vec![
                cast("int32", device_type.clone())?.into(),
                cast("int32", device_id.clone())?.into(),
                data,
            ],
        );
        let free_call = PrimExpr::try_from(Expr::from(free_call))?;
        let free = IfThenElse::new(
            crate::ir::prim::NE::new(free_call, IntImm::new("int32", 0)?)?,
            throw,
        )?;
        self.pending_frees
            .last_mut()
            .ok_or_else(|| value_error("allocation has no enclosing lifetime scope"))?
            .push(free.into());

        let pointer_type = buffer_pointer_type(&value.buffer)?;
        let allocation_call = Call::new(
            pointer_type,
            get_operator("tirx.TVMBackendAllocWorkspace")?,
            vec![
                cast("int32", device_type)?.into(),
                cast("int32", device_id)?.into(),
                total_bytes.into(),
                IntImm::new("int32", i64::from(buffer_type.dtype.dtype.code))?.into(),
                IntImm::new("int32", i64::from(buffer_type.dtype.dtype.bits))?.into(),
            ],
        );
        Stmt::sequence(vec![
            DeclBuffer::from_complete_fields(
                value.span.clone(),
                value.buffer.clone(),
                allocation_call.into(),
            )
            .into(),
            null_check.into(),
        ])
    }

    fn mutate_attribute(&mut self, value: AttrStmt, mutator: &mut Mutator) -> Result<Stmt> {
        if value.attr_key.as_str() == DEVICE_ID {
            let saved = self.device_id.replace(value.value.clone());
            let result = self.with_free_scope(|this| mutator.mutate(this, &value.body)?.try_into());
            self.device_id = saved;
            return result;
        }
        if value.attr_key.as_str() == DEVICE_TYPE {
            let saved = self.device_type.replace(value.value.clone());
            let result = self.with_free_scope(|this| mutator.mutate(this, &value.body)?.try_into());
            self.device_type = saved;
            return result;
        }

        self.pending_frees.push(Vec::new());
        let mapped = mutate_stmt_default(self, mutator, value.clone().into());
        let frees = self.pending_frees.pop().unwrap_or_default();
        let mapped = mapped?;
        if frees.is_empty() {
            return Ok(mapped);
        }
        let mapped = mapped.try_cast::<AttrStmt>()?;
        let mut body = vec![mapped.body.clone()];
        body.extend(frees.into_iter().rev());
        Ok(mapped
            .copy_with(
                mapped.node.clone(),
                mapped.attr_key.clone(),
                mapped.value.clone(),
                Stmt::sequence(body)?,
            )
            .into())
    }

    fn mutate_loop(&mut self, value: For, mutator: &mut Mutator) -> Result<Stmt> {
        let minimum: PrimExpr = mutator.mutate(self, &value.min)?.try_into()?;
        let extent: PrimExpr = mutator.mutate(self, &value.extent)?.try_into()?;
        let body = if value.kind == ForKind::kParallel {
            self.visit_body_and_realize_alloca(value.body.clone())?
        } else {
            self.with_free_scope(|this| mutator.mutate(this, &value.body)?.try_into())?
        };
        if minimum.same_as(&value.min) && extent.same_as(&value.extent) && body.same_as(&value.body)
        {
            return Ok(value.into());
        }
        Ok(value
            .copy_with(value.loop_var.clone(), minimum, extent, body)
            .into())
    }

    fn mutate_conditional(&mut self, value: IfThenElse, mutator: &mut Mutator) -> Result<Stmt> {
        let condition: PrimExpr = mutator.mutate(self, &value.condition)?.try_into()?;
        let then_case =
            self.with_free_scope(|this| mutator.mutate(this, &value.then_case)?.try_into())?;
        let else_case = value
            .else_case
            .as_ref()
            .map(|branch| self.with_free_scope(|this| mutator.mutate(this, branch)?.try_into()))
            .transpose()?;
        if condition.same_as(&value.condition)
            && then_case.same_as(&value.then_case)
            && option_same_as(&else_case, &value.else_case)
        {
            return Ok(value.into());
        }
        Ok(
            IfThenElse::from_complete_fields(value.span.clone(), condition, then_case, else_case)
                .into(),
        )
    }

    fn lower_call(&mut self, value: Call, mutator: &mut Mutator) -> Result<Expr> {
        if value.op.same_as(&self.operators.call_packed) {
            let lowered = self.operators.call_packed_lowered.clone();
            return Ok(self
                .make_call_packed_generic(value, 0, lowered, false, mutator)?
                .into());
        }
        if value.op.same_as(&self.operators.call_cpacked) {
            let lowered = self.operators.call_cpacked_lowered.clone();
            return Ok(self
                .make_call_packed_generic(value, 0, lowered, false, mutator)?
                .into());
        }
        if value.op.same_as(&self.operators.call_trace_packed) {
            let lowered = self.operators.call_trace_packed_lowered.clone();
            return Ok(self
                .make_call_packed_generic(value, 0, lowered, true, mutator)?
                .into());
        }
        if value.op.same_as(&self.operators.anylist_setitem_packed) {
            let lowered = self.operators.call_packed_lowered.clone();
            return self.make_anylist_setitem(value, lowered, mutator);
        }
        if value.op.same_as(&self.operators.anylist_setitem_cpacked) {
            let lowered = self.operators.call_cpacked_lowered.clone();
            return self.make_anylist_setitem(value, lowered, mutator);
        }
        if value.op.same_as(&self.operators.stack_make_shape) {
            return self.make_shape(value, mutator);
        }
        if value.op.same_as(&self.operators.stack_make_array) {
            return self.make_array(value, mutator);
        }
        if value.op.same_as(&self.operators.context_id) {
            return Ok(
                IntImm::from_dtype(value.ty.clone().try_cast::<PrimType>()?.dtype, 0)?.into(),
            );
        }
        if value.op.same_as(&self.operators.dma_copy) {
            return self.make_dma_call(value, "dma_copy", mutator);
        }
        if value.op.same_as(&self.operators.dma_wait) {
            return self.make_dma_call(value, "dma_wait", mutator);
        }
        if value.op.same_as(&self.operators.dma_start_group) {
            return self.make_dma_call(value, "dma_start_group", mutator);
        }
        if value.op.same_as(&self.operators.dma_end_group) {
            return self.make_dma_call(value, "dma_end_group", mutator);
        }
        mutate_expr_default(self, mutator, value.into())
    }

    fn make_dma_call(&mut self, value: Call, method: &str, mutator: &mut Mutator) -> Result<Expr> {
        let mut arguments = vec![self.device_method_name(method)?.into()];
        arguments.extend(value.args.iter());
        let call = Call::new(
            PrimType::new("int32")?,
            self.operators.call_packed.clone(),
            arguments,
        );
        self.lower_call(call, mutator)
    }

    fn device_method_name(&self, method: &str) -> Result<StringImm> {
        let device_type = self
            .device_type
            .as_ref()
            .and_then(int_value)
            .ok_or_else(|| value_error("device method requires a constant device type"))?;
        Ok(StringImm::new(&format!(
            "device_api.{}.{}",
            device_type_name(device_type)?,
            method
        )))
    }

    fn make_shape(&mut self, value: Call, mutator: &mut Mutator) -> Result<Expr> {
        let begin = {
            let scope = self.current_alloca_scope_mut()?;
            if scope.current.shape == -1 {
                scope.current.shape = 0;
            }
            let begin = scope.current.shape;
            scope.current.shape += value.args.len() as i64;
            begin
        };
        let arguments: Array<Expr> = mutator.mutate(self, &value.args)?.try_into()?;
        let buffer = self.current_alloca_scope()?.stack_shape.clone();
        for (offset, argument) in arguments.iter().enumerate() {
            let primitive = argument.try_cast::<PrimExpr>()?;
            self.current_preparation_mut()?.push(
                BufferStore::new(
                    buffer.clone(),
                    cast("int64", primitive)?,
                    vec![IntImm::new("int32", begin + offset as i64)?.into()],
                )?
                .into(),
            );
        }
        let load =
            TensorLoad::from_buffer(buffer.clone(), vec![IntImm::new("int32", begin)?.into()])?;
        Ok(Call::new(
            buffer_pointer_type(&buffer)?,
            get_operator("tirx.address_of")?,
            vec![load.into()],
        )
        .into())
    }

    fn make_array(&mut self, value: Call, mutator: &mut Mutator) -> Result<Expr> {
        let index = {
            let scope = self.current_alloca_scope_mut()?;
            let index = scope.current.array;
            scope.current.array += 1;
            index
        };
        let arguments: Array<Expr> = mutator.mutate(self, &value.args)?.try_into()?;
        if arguments.len() != 6 {
            return Err(value_error("tvm_stack_make_array expects six arguments"));
        }
        let stack = self.current_alloca_scope()?.stack_array.clone();
        let mut statements = vec![
            struct_set(&stack, index, DLTENSOR_DATA, arguments.get(0)?)?,
            struct_set(&stack, index, DLTENSOR_SHAPE, arguments.get(1)?)?,
        ];
        let strides = arguments.get(2)?;
        statements.push(struct_set(
            &stack,
            index,
            DLTENSOR_STRIDES,
            if strides
                .clone()
                .try_cast::<PrimExpr>()
                .ok()
                .and_then(|value| int_value(&value))
                == Some(0)
            {
                const_handle(0)?
            } else {
                strides
            },
        )?);
        statements.push(struct_set(&stack, index, DLTENSOR_NDIM, arguments.get(3)?)?);
        let dtype = arguments.get(4)?.try_cast::<PrimExpr>()?.type_annotation();
        statements.extend([
            struct_set(
                &stack,
                index,
                DLTENSOR_TYPE_CODE,
                IntImm::new("uint8", i64::from(dtype.dtype.code))?,
            )?,
            struct_set(
                &stack,
                index,
                DLTENSOR_TYPE_BITS,
                IntImm::new("uint8", i64::from(dtype.dtype.bits))?,
            )?,
            struct_set(
                &stack,
                index,
                DLTENSOR_TYPE_LANES,
                IntImm::new("uint16", i64::from(dtype.dtype.lanes))?,
            )?,
        ]);
        let element_offset = arguments.get(5)?.try_cast::<PrimExpr>()?;
        let byte_offset = if int_value(&element_offset) == Some(0) {
            element_offset
        } else {
            binary_op(
                "tirx._OpMul",
                element_offset.clone(),
                IntImm::from_dtype(element_offset.dtype(), storage_bytes(&dtype)?)?.into(),
            )?
        };
        let device_id = self
            .device_id
            .clone()
            .ok_or_else(|| value_error("tvm_stack_make_array requires a device id"))?;
        let device_type = self
            .device_type
            .clone()
            .ok_or_else(|| value_error("tvm_stack_make_array requires a device type"))?;
        statements.extend([
            struct_set(
                &stack,
                index,
                DLTENSOR_BYTE_OFFSET,
                cast("uint64", byte_offset)?,
            )?,
            struct_set(&stack, index, DLTENSOR_DEVICE_ID, cast("int32", device_id)?)?,
            struct_set(
                &stack,
                index,
                DLTENSOR_DEVICE_TYPE,
                cast("int32", device_type)?,
            )?,
        ]);
        self.current_preparation_mut()?.extend(statements);
        struct_get(
            PointerType::new(PrimType::new("void")?, "")?.into(),
            &stack,
            index,
            DLTENSOR_ADDR,
        )
    }

    fn make_call_packed_generic(
        &mut self,
        value: Call,
        name_offset: usize,
        lowered_operator: Expr,
        traced: bool,
        mutator: &mut Mutator,
    ) -> Result<Call> {
        let (saved, argument_begin) = {
            let scope = self.current_alloca_scope_mut()?;
            let saved = scope.current;
            let argument_begin = scope.current.arguments;
            let first_argument = name_offset + 1;
            if first_argument > value.args.len() {
                return Err(value_error("packed call has no function name"));
            }
            let count = value.args.len() - first_argument;
            scope.current.arguments += count as u64 + 1;
            (saved, argument_begin)
        };
        let arguments: Array<Expr> = mutator.mutate(self, &value.args)?.try_into()?;
        let first_argument = name_offset + 1;
        let count = arguments.len() - first_argument;
        let stack = self.current_alloca_scope()?.stack_any.clone();
        for index in 0..count {
            self.set_packed_argument(
                arguments.get(first_argument + index)?,
                &stack,
                argument_begin + index as u64,
            )?;
        }
        self.current_preparation_mut()?.extend([
            struct_set(
                &stack,
                count as u64,
                TVM_FFI_ANY_TYPE_INDEX,
                IntImm::new("int32", TypeIndex::kTVMFFINone as i64)?,
            )?,
            struct_set(
                &stack,
                count as u64,
                TVM_FFI_ANY_ZERO_PADDING,
                IntImm::new("int32", 0)?,
            )?,
            struct_set(
                &stack,
                count as u64,
                TVM_FFI_ANY_UNION_VALUE,
                IntImm::new("int64", 0)?,
            )?,
        ]);
        {
            let precheck = self.precheck;
            let scope = self.current_alloca_scope_mut()?;
            if precheck {
                scope.maximum.include(scope.current);
            } else if !scope.maximum.contains(scope.current) {
                return Err(value_error(
                    "packed-call stack exceeded its precomputed maximum",
                ));
            }
            scope.current = saved;
        }
        let mut lowered_arguments = vec![
            arguments.get(name_offset)?,
            stack.into(),
            IntImm::new("int32", argument_begin as i64)?.into(),
            IntImm::new("int32", (argument_begin + count as u64) as i64)?.into(),
        ];
        if traced {
            lowered_arguments.push(arguments.get(arguments.len() - 1)?);
        }
        Ok(value.copy_with(
            value.ty.clone(),
            lowered_operator,
            Array::new(lowered_arguments),
        ))
    }

    fn set_packed_argument(&mut self, mut argument: Expr, stack: &Var, offset: u64) -> Result<()> {
        if let Some(call) = argument.as_node::<CallObj>() {
            if call.op.same_as(&self.operators.anylist_getitem) {
                self.current_preparation_mut()?.push(
                    Evaluate::new(Call::new(
                        PrimType::new("int32")?,
                        get_operator("tirx.TVMBackendAnyListSetPackedArg")?,
                        vec![
                            call.args.get(0)?,
                            call.args.get(1)?,
                            stack.clone().into(),
                            IntImm::new("int32", offset as i64)?.into(),
                        ],
                    ))?
                    .into(),
                );
                return Ok(());
            }
        }
        let type_index = if is_string_imm(&argument) {
            argument = reinterpret(PointerType::new(PrimType::new("void")?, "")?, argument)?;
            TypeIndex::kTVMFFIRawStr as i32
        } else if is_pointer_type(&argument.ty) {
            if is_array_handle(&argument, &self.operators.struct_get)? {
                TypeIndex::kTVMFFIDLTensorPtr as i32
            } else {
                TypeIndex::kTVMFFIOpaquePtr as i32
            }
        } else {
            let primitive = argument.clone().try_cast::<PrimExpr>()?;
            let original = primitive.type_annotation();
            let api_type = api_type(&original)?;
            if api_type.dtype != original.dtype {
                argument = Cast::new(api_type.clone(), primitive)?.into();
            }
            if api_type.dtype.code == DLDataTypeCode::kDLBool as u8 {
                TypeIndex::kTVMFFIBool as i32
            } else if matches!(
                api_type.dtype.code,
                value if value == DLDataTypeCode::kDLInt as u8
                    || value == DLDataTypeCode::kDLUInt as u8
            ) {
                TypeIndex::kTVMFFIInt as i32
            } else if api_type.dtype.code == DLDataTypeCode::kDLFloat as u8 {
                TypeIndex::kTVMFFIFloat as i32
            } else {
                return Err(value_error("unsupported packed-call scalar argument type"));
            }
        };
        if type_index == TypeIndex::kTVMFFIOpaquePtr as i32 {
            let condition = Call::new(
                PrimType::new("bool")?,
                get_operator("tirx.isnullptr")?,
                vec![argument.clone()],
            );
            let condition = PrimExpr::try_from(Expr::from(condition))?;
            self.current_preparation_mut()?.push(
                IfThenElse::with_span(
                    condition,
                    struct_set(
                        stack,
                        offset,
                        TVM_FFI_ANY_TYPE_INDEX,
                        IntImm::new("int32", TypeIndex::kTVMFFINone as i64)?,
                    )?,
                    Some(struct_set(
                        stack,
                        offset,
                        TVM_FFI_ANY_TYPE_INDEX,
                        IntImm::new("int32", TypeIndex::kTVMFFIOpaquePtr as i64)?,
                    )?),
                    None,
                )?
                .into(),
            );
        } else {
            self.current_preparation_mut()?.push(struct_set(
                stack,
                offset,
                TVM_FFI_ANY_TYPE_INDEX,
                IntImm::new("int32", i64::from(type_index))?,
            )?);
        }
        self.current_preparation_mut()?.extend([
            struct_set(
                stack,
                offset,
                TVM_FFI_ANY_ZERO_PADDING,
                IntImm::new("int32", 0)?,
            )?,
            struct_set(stack, offset, TVM_FFI_ANY_UNION_VALUE, argument)?,
        ]);
        Ok(())
    }

    fn make_anylist_setitem(
        &mut self,
        value: Call,
        lowered: Expr,
        mutator: &mut Mutator,
    ) -> Result<Expr> {
        if value.args.len() < 3 {
            return Err(value_error(
                "anylist setitem packed call expects at least three arguments",
            ));
        }
        let list = value.args.get(0)?;
        let index = value.args.get(1)?;
        let lowered = self.make_call_packed_generic(value, 2, lowered, false, mutator)?;
        let stack = lowered.args.get(1)?;
        let return_offset = lowered.args.get(3)?;
        self.current_preparation_mut()?
            .push(Evaluate::new(lowered)?.into());
        Ok(Call::new(
            PrimType::new("int32")?,
            get_operator("tirx.TVMBackendAnyListMoveFromPackedReturn")?,
            vec![list, index, stack, return_offset],
        )
        .into())
    }

    fn make_nd_memory_allocation(
        &mut self,
        binding: Bind,
        call: Call,
        mutator: &mut Mutator,
    ) -> Result<Stmt> {
        let device_type = self
            .device_type
            .clone()
            .ok_or_else(|| value_error("nd allocation requires a device type"))?;
        let device_id = self
            .device_id
            .clone()
            .ok_or_else(|| value_error("nd allocation requires a device id"))?;
        let pointer = binding.var.ty.clone().try_cast::<PointerType>()?;
        let dtype = pointer.element_type().clone().try_cast::<PrimType>()?;
        let mut arguments = vec![
            self.device_method_name("alloc_nd")?.into(),
            device_type.clone().into(),
            device_id.clone().into(),
            IntImm::new("int32", i64::from(dtype.dtype.code))?.into(),
            IntImm::new("int32", i64::from(dtype.dtype.bits))?.into(),
        ];
        arguments.extend(call.args.iter());
        let packed = Call::new(
            binding.var.ty.clone(),
            self.operators.call_packed.clone(),
            arguments,
        );
        let packed = self.lower_call(packed, mutator)?;
        let null_check = IfThenElse::new(
            Call::new(
                PrimType::new("bool")?,
                get_operator("tirx.isnullptr")?,
                vec![binding.var.clone().into()],
            ),
            Evaluate::new(Call::new(
                PrimType::new("int32")?,
                get_operator("tirx.tvm_throw_last_error")?,
                Vec::new(),
            ))?,
        )?;
        let storage_scope = call.args.get(0)?;
        let free_call = Call::new(
            PrimType::new("int32")?,
            self.operators.call_packed.clone(),
            vec![
                self.device_method_name("free_nd")?.into(),
                device_type.into(),
                device_id.into(),
                storage_scope,
                binding.var.clone().into(),
            ],
        );
        let free_call: PrimExpr = self.lower_call(free_call, mutator)?.try_into()?;
        let free = IfThenElse::new(
            crate::ir::prim::NE::new(free_call, IntImm::new("int32", 0)?)?,
            Evaluate::new(Call::new(
                PrimType::new("int32")?,
                get_operator("tirx.tvm_throw_last_error")?,
                Vec::new(),
            ))?,
        )?;
        self.pending_frees
            .last_mut()
            .ok_or_else(|| value_error("nd allocation has no enclosing lifetime scope"))?
            .push(free.into());
        Stmt::sequence(vec![
            binding.copy_with(binding.var.clone(), packed).into(),
            null_check.into(),
        ])
    }

    fn current_alloca_scope(&self) -> Result<&AllocaScope> {
        self.alloca_scopes
            .last()
            .ok_or_else(|| value_error("builtin lowering has no active alloca scope"))
    }

    fn current_alloca_scope_mut(&mut self) -> Result<&mut AllocaScope> {
        self.alloca_scopes
            .last_mut()
            .ok_or_else(|| value_error("builtin lowering has no active alloca scope"))
    }

    fn current_preparation_mut(&mut self) -> Result<&mut Vec<Stmt>> {
        self.preparation
            .last_mut()
            .ok_or_else(|| value_error("builtin expression has no enclosing statement"))
    }
}

#[tvm_ffi::dispatch(mutate)]
impl BuiltinLower {
    fn mutate_statement(&mut self, value: Stmt, mutator: &mut Mutator) -> Result<Stmt> {
        self.preparation.push(Vec::new());
        let scope_count = self.alloca_scopes.len();
        let result = self.mutate_statement_inner(value, mutator);
        if self.alloca_scopes.len() != scope_count {
            return Err(value_error(
                "alloca scope became unbalanced while mutating a statement",
            ));
        }
        if let Some(scope) = self.alloca_scopes.last() {
            if scope.current.shape != -1 || scope.current.array != 0 {
                return Err(value_error(
                    "stack shape/array allocations escaped their packed-call scope",
                ));
            }
        }
        let preparation = self.preparation.pop().unwrap_or_default();
        let result = result?;
        if preparation.is_empty() {
            Ok(result)
        } else {
            let mut sequence = preparation;
            sequence.push(result);
            Stmt::sequence(sequence)
        }
    }

    fn mutate_call(&mut self, value: Call, mutator: &mut Mutator) -> Result<Expr> {
        self.lower_call(value, mutator)
    }

    fn mutate_default(&mut self, value: &MapValue, mutator: &mut Mutator) -> Result<Any> {
        mutate_stmt_expr_default(self, mutator, value)
    }
}

fn stack_shape_buffer(size: i64) -> Result<BufferVar> {
    Ok(
        BufferType::new("global", "int64", vec![IntImm::new("int64", size)?.into()])?
            .new_var("stack_shape"),
    )
}

fn stack_alloca(return_type: Type, kind: &str, count: u64) -> Result<Call> {
    Ok(Call::new(
        return_type,
        get_operator("tirx.tvm_stack_alloca")?,
        vec![
            StringImm::new(kind).into(),
            IntImm::new("int32", count as i64)?.into(),
        ],
    ))
}

fn struct_set<V>(handle: &Var, index: u64, kind: i64, value: V) -> Result<Stmt>
where
    V: Into<Expr>,
{
    Ok(Evaluate::new(Call::new(
        PrimType::new("int32")?,
        get_operator("tirx.tvm_struct_set")?,
        vec![
            handle.clone().into(),
            IntImm::new("int32", index as i64)?.into(),
            IntImm::new("int32", kind)?.into(),
            value.into(),
        ],
    ))?
    .into())
}

fn struct_get(result_type: Type, handle: &Var, index: u64, kind: i64) -> Result<Expr> {
    Ok(Call::new(
        result_type,
        get_operator("tirx.tvm_struct_get")?,
        vec![
            handle.clone().into(),
            IntImm::new("int32", index as i64)?.into(),
            IntImm::new("int32", kind)?.into(),
        ],
    )
    .into())
}

fn buffer_pointer_type(buffer: &BufferVar) -> Result<PointerType> {
    let ty = buffer.type_annotation();
    PointerType::new(ty.dtype.clone(), ty.storage_scope.as_str())
}

fn cast(dtype: &str, value: PrimExpr) -> Result<PrimExpr> {
    let target = PrimType::new(dtype)?;
    if value.dtype() == target.dtype {
        Ok(value)
    } else {
        tvm_ffi::cached_global_func!("tirx._cast")
            .call_tuple((target, value, Option::<crate::ir::Span>::None))?
            .try_into()
    }
}

fn reinterpret(pointer_type: PointerType, value: Expr) -> Result<Expr> {
    Ok(Call::new(pointer_type, get_operator("tirx.reinterpret")?, vec![value]).into())
}

fn const_handle(value: i64) -> Result<Expr> {
    reinterpret(
        PointerType::new(PrimType::new("void")?, "")?,
        IntImm::new("int64", value)?.into(),
    )
}

fn api_type(dtype: &PrimType) -> Result<PrimType> {
    if dtype.dtype.lanes != 1 {
        return Err(value_error(
            "packed API does not accept vector scalar arguments",
        ));
    }
    if matches!(
        dtype.dtype.code,
        value if value == DLDataTypeCode::kDLBool as u8
            || value == DLDataTypeCode::kDLInt as u8
            || value == DLDataTypeCode::kDLUInt as u8
    ) {
        PrimType::new("int64")
    } else if dtype.dtype.code == DLDataTypeCode::kDLFloat as u8 {
        PrimType::new("float64")
    } else {
        Err(value_error("unsupported packed API scalar type"))
    }
}

fn storage_bytes(dtype: &PrimType) -> Result<i64> {
    if is_scalable_vector(dtype) {
        return Err(value_error(
            "scalable vector has no compile-time storage size",
        ));
    }
    Ok((i64::from(dtype.dtype.bits) * i64::from(dtype.dtype.lanes) + 7) / 8)
}

fn is_scalable_vector(dtype: &PrimType) -> bool {
    (dtype.dtype.lanes as i16) < 0
}

fn constant_allocation_size(shape: &Array<PrimExpr>) -> Option<i64> {
    shape.iter().try_fold(1_i64, |size, extent| {
        int_value(&extent).and_then(|extent| size.checked_mul(extent))
    })
}

fn is_array_handle(value: &Expr, struct_get_operator: &Expr) -> Result<bool> {
    let Some(call) = value.as_node::<CallObj>() else {
        return Ok(false);
    };
    if !call.op.same_as(struct_get_operator) || call.args.len() < 3 {
        return Ok(false);
    }
    Ok(call
        .args
        .get(2)?
        .try_cast::<IntImm>()
        .is_ok_and(|field| field.value == DLTENSOR_ADDR))
}

fn option_same_as<T: ObjectRefCore>(lhs: &Option<T>, rhs: &Option<T>) -> bool {
    match (lhs, rhs) {
        (Some(lhs), Some(rhs)) => lhs.same_as(rhs),
        (None, None) => true,
        _ => false,
    }
}

fn function_target(function: &PrimFunc) -> Result<Option<Target>> {
    function
        .attrs
        .dict
        .get(&String::from("target"))?
        .map(Target::try_from)
        .transpose()
}

fn is_host_function(function: &PrimFunc) -> Result<bool> {
    if let Some(attribute) = function
        .attrs
        .dict
        .get(&String::from("tirx.is_host_func"))?
    {
        if bool::try_from(attribute.clone()).unwrap_or(false)
            || i64::try_from(attribute).is_ok_and(|value| value != 0)
        {
            return Ok(true);
        }
    }
    function_target(function)?
        .map(|target| target.has_key("cpu"))
        .transpose()
        .map(Option::unwrap_or_default)
}

fn device_type_name(device_type: i64) -> Result<&'static str> {
    match device_type {
        value if value == DLDeviceType::kDLCPU as i64 => Ok("cpu"),
        value if value == DLDeviceType::kDLCUDA as i64 => Ok("cuda"),
        value if value == DLDeviceType::kDLCUDAHost as i64 => Ok("cuda_host"),
        value if value == DLDeviceType::kDLCUDAManaged as i64 => Ok("cuda_managed"),
        value if value == DLDeviceType::kDLOpenCL as i64 => Ok("opencl"),
        value if value == DLDeviceType::kDLVulkan as i64 => Ok("vulkan"),
        value if value == DLDeviceType::kDLMetal as i64 => Ok("metal"),
        value if value == DLDeviceType::kDLVPI as i64 => Ok("vpi"),
        value if value == DLDeviceType::kDLROCM as i64 => Ok("rocm"),
        value if value == DLDeviceType::kDLROCMHost as i64 => Ok("rocm_host"),
        value if value == DLDeviceType::kDLExtDev as i64 => Ok("ext_dev"),
        value if value == DLDeviceType::kDLOneAPI as i64 => Ok("oneapi"),
        value if value == DLDeviceType::kDLWebGPU as i64 => Ok("webgpu"),
        value if value == DLDeviceType::kDLHexagon as i64 => Ok("hexagon"),
        value if value == DLDeviceType::kDLTrn as i64 => Ok("trn"),
        _ => Err(value_error("unknown device type")),
    }
}
