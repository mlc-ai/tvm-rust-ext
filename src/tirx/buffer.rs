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
use tvm_ffi::{
    Any, Array, DLDataType, DLDataTypeCode, DLDataTypeExt, Error, FieldGetter, Map, ObjectArc,
    ObjectCore, ObjectRefCast, Result, String, TYPE_ERROR, VALUE_ERROR,
};

use super::{primitive_type, Stmt, StmtObj};
use crate::analysis::Analyzer;
use crate::ir::{
    Expr, ExprObj, IntImm, PrimExpr, PrimExprConvertible, PrimExprConvertibleObj, PrimType, Range,
    Span, Type, TypeObj, TypedVar, Var,
};

/// Opaque Rust representation of TVM's polymorphic layout base class.
#[repr(C)]
#[derive(Object)]
#[type_key = "tirx.Layout"]
pub struct LayoutObj {
    base: tvm_ffi::Object,
}
/// Reference-counted handle to a TIRx buffer layout.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct Layout {
    data: ObjectArc<LayoutObj>,
}

impl std::ops::Deref for Layout {
    type Target = LayoutObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl Layout {
    /// Check whether this layout can describe the supplied logical shape.
    pub fn compatible_with_shape(&self, shape: &Array<PrimExpr>) -> Result<bool> {
        tvm_ffi::cached_global_func!("tirx.LayoutCompatibleWithShape")
            .call_tuple((self, shape))?
            .try_into()
    }

    /// Validate the concrete layout through TVM's registered operation.
    pub fn verify_well_formed(&self) -> Result<bool> {
        tvm_ffi::cached_global_func!("tirx.LayoutVerifyWellFormed")
            .call_tuple((self,))?
            .try_into()
    }

    /// Return the logical size for all axes or one named axis.
    pub fn get_size(&self, axis_name: Option<&str>) -> Result<PrimExpr> {
        let axis_name = axis_name.map(String::from);
        tvm_ffi::cached_global_func!("tirx.LayoutGetSize")
            .call_tuple((self, axis_name))?
            .try_into()
    }

    /// Return the physical span for all axes or one named axis.
    pub fn get_span(&self, axis_name: Option<&str>) -> Result<PrimExpr> {
        let axis_name = axis_name.map(String::from);
        tvm_ffi::cached_global_func!("tirx.LayoutGetSpan")
            .call_tuple((self, axis_name))?
            .try_into()
    }

    /// Map one structured coordinate through this layout.
    pub fn apply(&self, coord: &Array<PrimExpr>) -> Result<Map<String, PrimExpr>> {
        tvm_ffi::cached_global_func!("tirx.LayoutApply")
            .call_tuple((self, coord))?
            .try_into()
    }

    /// Map one flattened coordinate through this layout.
    pub fn apply_linear(&self, coord: &PrimExpr) -> Result<Map<String, PrimExpr>> {
        tvm_ffi::cached_global_func!("tirx.LayoutApplyLinear")
            .call_tuple((self, coord))?
            .try_into()
    }

    /// Map a coordinate whose dimensions are grouped by `shape`.
    pub fn apply_with_shape(
        &self,
        coord: &Array<PrimExpr>,
        shape: &Array<PrimExpr>,
    ) -> Result<Map<String, PrimExpr>> {
        tvm_ffi::cached_global_func!("tirx.LayoutApplyWithShape")
            .call_tuple((self, coord, shape))?
            .try_into()
    }

    /// Return the canonical form of this layout.
    pub fn canonicalize(&self) -> Result<Layout> {
        tvm_ffi::cached_global_func!("tirx.LayoutCanonicalize")
            .call_tuple((self,))?
            .try_into()
    }

    /// Tile this layout with an outer tile.
    pub fn tile(
        &self,
        outer: &TileLayout,
        outer_shape: &Array<PrimExpr>,
        inner_shape: &Array<PrimExpr>,
    ) -> Result<Layout> {
        tvm_ffi::cached_global_func!("tirx.LayoutTile")
            .call_tuple((self, outer, outer_shape, inner_shape))?
            .try_into()
    }

    /// Restrict this layout to one region.
    pub fn slice(&self, shape: &Array<PrimExpr>, region: &Array<Range>) -> Result<Option<Layout>> {
        tvm_ffi::cached_global_func!("tirx.LayoutSlice")
            .call_tuple((self, shape, region))?
            .try_into()
    }

    /// Form the layout direct sum with a left tile.
    pub fn direct_sum(
        &self,
        left: &TileLayout,
        left_shape: &Array<PrimExpr>,
        right_shape: &Array<PrimExpr>,
    ) -> Result<Layout> {
        tvm_ffi::cached_global_func!("tirx.LayoutDirectSum")
            .call_tuple((self, left, left_shape, right_shape))?
            .try_into()
    }

    /// Recover an outer tile when this layout is the tiled inner component.
    pub fn is_tile_inner(
        &self,
        layout: &Layout,
        tiled_shape: &Array<PrimExpr>,
        inner_shape: &Array<PrimExpr>,
    ) -> Result<Option<TileLayout>> {
        tvm_ffi::cached_global_func!("tirx.LayoutIsTileInner")
            .call_tuple((self, layout, tiled_shape, inner_shape))?
            .try_into()
    }

    /// Recover an inner layout when this layout is the tiled outer component.
    pub fn is_tile_outer(
        &self,
        layout: &Layout,
        tiled_shape: &Array<PrimExpr>,
        outer_shape: &Array<PrimExpr>,
    ) -> Result<Option<Layout>> {
        tvm_ffi::cached_global_func!("tirx.LayoutIsTileOuter")
            .call_tuple((self, layout, tiled_shape, outer_shape))?
            .try_into()
    }

    /// Recover the left direct-sum component when this layout is the right one.
    pub fn is_direct_sum_right(
        &self,
        layout: &Layout,
        interleaved_shape: &Array<PrimExpr>,
        right_shape: &Array<PrimExpr>,
    ) -> Result<Option<TileLayout>> {
        tvm_ffi::cached_global_func!("tirx.LayoutIsDirectSumRight")
            .call_tuple((self, layout, interleaved_shape, right_shape))?
            .try_into()
    }

    /// Recover the right direct-sum component when this layout is the left one.
    pub fn is_direct_sum_left(
        &self,
        layout: &Layout,
        interleaved_shape: &Array<PrimExpr>,
        left_shape: &Array<PrimExpr>,
    ) -> Result<Option<Layout>> {
        tvm_ffi::cached_global_func!("tirx.LayoutIsDirectSumLeft")
            .call_tuple((self, layout, interleaved_shape, left_shape))?
            .try_into()
    }
}

/// Opaque Rust representation of one registered TIRx layout axis.
///
/// Axis objects are native registry singletons.  Rust obtains the registered
/// object instead of constructing a second object with a copied registry index.
#[repr(C)]
#[derive(Object)]
#[type_key = "tirx.Axis"]
#[type_final]
pub struct AxisObj {
    base: tvm_ffi::Object,
}

/// Reference-counted handle to a native axis-registry entry.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct Axis {
    data: ObjectArc<AxisObj>,
}

impl std::ops::Deref for Axis {
    type Target = AxisObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl Axis {
    /// Return the native registry object for `name`.
    pub fn get(name: &str) -> Result<Self> {
        tvm_ffi::cached_global_func!("tirx.AxisGet")
            .call_tuple((String::from(name),))?
            .try_into()
    }

    /// Return the registered axis name through native reflection.
    pub fn name(&self) -> Result<String> {
        FieldGetter::new(AxisObj::type_index(), "name")?.get(&**self)
    }
}

/// ABI-complete Rust representation of one layout extent/stride/axis component.
#[repr(C)]
#[derive(Object)]
#[type_key = "tirx.Iter"]
#[type_final]
pub struct IterObj {
    base: tvm_ffi::Object,
    pub extent: PrimExpr,
    pub stride: PrimExpr,
    pub axis: Axis,
}

/// Reference-counted handle to one layout iterator.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct Iter {
    data: ObjectArc<IterObj>,
}

impl std::ops::Deref for Iter {
    type Target = IterObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl Iter {
    /// Construct one layout iterator directly in Rust.
    pub fn new<E, S, A>(extent: E, stride: S, axis: A) -> Result<Self>
    where
        E: Into<Expr>,
        S: Into<Expr>,
        A: Into<Axis>,
    {
        let extent = extent.into();
        let stride = stride.into();
        primitive_type(&extent, "layout iterator extent")?;
        primitive_type(&stride, "layout iterator stride")?;
        let extent = PrimExpr::try_from(extent)?;
        let stride = PrimExpr::try_from(stride)?;
        Ok(Self::from_complete_fields(extent, stride, axis.into()))
    }

    /// Construct one layout iterator from every physical field after external validation.
    pub fn from_complete_fields(extent: PrimExpr, stride: PrimExpr, axis: Axis) -> Self {
        Self {
            data: ObjectArc::new(IterObj {
                base: tvm_ffi::Object::new(),
                extent,
                stride,
                axis,
            }),
        }
    }
}

/// Opaque Rust representation of TVM's polymorphic tiled layout.
#[repr(C)]
#[derive(Object)]
#[type_key = "tirx.TileLayout"]
#[type_final]
pub struct TileLayoutObj {
    base: LayoutObj,
}

/// Reference-counted handle to a tiled layout.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct TileLayout {
    data: ObjectArc<TileLayoutObj>,
}

impl std::ops::Deref for TileLayout {
    type Target = TileLayoutObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl std::ops::Deref for TileLayoutObj {
    type Target = LayoutObj;

    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

impl TileLayout {
    fn field<T>(&self, name: &str) -> Result<T>
    where
        T: TryFrom<Any, Error = Error>,
    {
        FieldGetter::new(TileLayoutObj::type_index(), name)?.get(&**self)
    }

    pub fn shard(&self) -> Result<Array<Iter>> {
        self.field("shard")
    }

    pub fn replica(&self) -> Result<Array<Iter>> {
        self.field("replica")
    }

    pub fn offset(&self) -> Result<Map<Axis, PrimExpr>> {
        self.field("offset")
    }

    /// Construct a tile layout through its native constructor.
    pub fn new(shard: Vec<Iter>, replica: Vec<Iter>, offset: Map<Axis, PrimExpr>) -> Result<Self> {
        tvm_ffi::cached_global_func!("tirx.TileLayout")
            .call_tuple((Array::new(shard), Array::new(replica), offset))?
            .try_into()
    }
}

/// ABI-complete Rust representation of TVM's immutable buffer access contract.
#[repr(C)]
#[derive(Object)]
#[type_key = "tirx.BufferType"]
#[type_final]
pub struct BufferTypeObj {
    base: TypeObj,
    pub dtype: PrimType,
    pub storage_scope: String,
    pub shape: Array<PrimExpr>,
    pub strides: Array<PrimExpr>,
    pub elem_offset: PrimExpr,
    pub data_alignment: i32,
    pub offset_factor: i32,
    pub layout: Option<Layout>,
    pub allocated_addr: Array<PrimExpr>,
}

/// Checked variable view whose expression type is `BufferType`.
pub type BufferVar = TypedVar<BufferType>;

tvm_ffi::impl_try_from_any!(BufferVar);
tvm_ffi::impl_arg_into_ref!(BufferVar);
tvm_ffi::impl_into_arg_holder_default!(BufferVar);

// These values mirror the build configuration used by this handwritten
// target-code demo. Stubgen should emit them from the native build manifest.
const DEFAULT_INDEX_DTYPE: &str = "int64";
const DEFAULT_ALLOC_ALIGNMENT: i32 = 64;

/// Reference-counted buffer type carried by an ordinary `ir.Var`.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct BufferType {
    data: ObjectArc<BufferTypeObj>,
}

impl std::ops::Deref for BufferType {
    type Target = BufferTypeObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl std::ops::Deref for BufferTypeObj {
    type Target = TypeObj;

    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

impl BufferType {
    /// Construct a compact buffer type directly in Rust.
    pub fn new(storage_scope: &str, dtype: &str, shape: Vec<Expr>) -> Result<Self> {
        let dtype = PrimType::new(dtype)?;
        Self::normalize_and_allocate(
            storage_scope,
            dtype,
            shape,
            Vec::new(),
            None,
            0,
            0,
            None,
            Vec::new(),
            None,
        )
    }

    /// Construct a buffer type directly in Rust with all reflected access metadata.
    #[allow(clippy::too_many_arguments)]
    pub fn with_metadata(
        storage_scope: &str,
        dtype: PrimType,
        shape: Vec<Expr>,
        strides: Vec<Expr>,
        element_offset: Expr,
        data_alignment: i32,
        offset_factor: i32,
        layout: Option<Layout>,
        allocated_addresses: Vec<Expr>,
        span: Option<&Span>,
    ) -> Result<Self> {
        Self::normalize_and_allocate(
            storage_scope,
            dtype,
            shape,
            strides,
            Some(element_offset),
            data_alignment,
            offset_factor,
            layout,
            allocated_addresses,
            span,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn normalize_and_allocate(
        storage_scope: &str,
        dtype: PrimType,
        shape: Vec<Expr>,
        strides: Vec<Expr>,
        element_offset: Option<Expr>,
        data_alignment: i32,
        offset_factor: i32,
        layout: Option<Layout>,
        allocated_addresses: Vec<Expr>,
        span: Option<&Span>,
    ) -> Result<Self> {
        for extent in &shape {
            primitive_type(extent, "buffer shape extent")?;
        }
        for stride in &strides {
            primitive_type(stride, "buffer stride")?;
        }
        if let Some(element_offset) = element_offset.as_ref() {
            primitive_type(element_offset, "buffer element offset")?;
        }
        for address in &allocated_addresses {
            primitive_type(address, "buffer allocated address")?;
        }
        let storage_scope = String::from(if storage_scope.is_empty() {
            "global"
        } else {
            storage_scope
        });
        let element_offset = match element_offset {
            Some(value) => value,
            None => {
                let index_type = if let Some(first) = shape.first() {
                    primitive_type(first, "buffer shape extent")?
                } else {
                    PrimType::new(DEFAULT_INDEX_DTYPE)?
                };
                IntImm::from_complete_fields(None, index_type, 0).into()
            }
        };
        let data_alignment = if data_alignment <= 0 {
            DEFAULT_ALLOC_ALIGNMENT
        } else {
            data_alignment
        };
        let offset_factor = if offset_factor == 0 { 1 } else { offset_factor };
        let shape = Array::new(
            shape
                .into_iter()
                .map(PrimExpr::try_from)
                .collect::<Result<Vec<_>>>()?,
        );
        let strides = Array::new(
            strides
                .into_iter()
                .map(PrimExpr::try_from)
                .collect::<Result<Vec<_>>>()?,
        );
        let element_offset = PrimExpr::try_from(element_offset)?;
        let allocated_addresses = Array::new(
            allocated_addresses
                .into_iter()
                .map(PrimExpr::try_from)
                .collect::<Result<Vec<_>>>()?,
        );
        Ok(Self::from_complete_fields(
            span.cloned(),
            dtype,
            storage_scope,
            shape,
            strides,
            element_offset,
            data_alignment,
            offset_factor,
            layout,
            allocated_addresses,
        ))
    }

    /// Construct a buffer type from every physical field without applying defaults.
    #[allow(clippy::too_many_arguments)]
    pub fn from_complete_fields(
        span: Option<Span>,
        dtype: PrimType,
        storage_scope: String,
        shape: Array<PrimExpr>,
        strides: Array<PrimExpr>,
        elem_offset: PrimExpr,
        data_alignment: i32,
        offset_factor: i32,
        layout: Option<Layout>,
        allocated_addr: Array<PrimExpr>,
    ) -> Self {
        Self {
            data: ObjectArc::new(BufferTypeObj {
                base: TypeObj::new(span),
                dtype,
                storage_scope,
                shape,
                strides,
                elem_offset,
                data_alignment,
                offset_factor,
                layout,
                allocated_addr,
            }),
        }
    }

    /// Construct a buffer variable.  Its runtime identity is an ordinary `ir.Var`.
    pub fn new_var(&self, name: &str) -> BufferVar {
        unsafe { BufferVar::from_var_unchecked(Var::with_type(name, Type::from(self.clone()))) }
    }
}

/// ABI-complete Rust representation of a buffer read.
#[repr(C)]
#[derive(Object)]
#[type_key = "tirx.BufferLoad"]
#[type_final]
pub struct BufferLoadObj {
    base: ExprObj,
    pub buffer: BufferVar,
    pub indices: Array<PrimExpr>,
    pub predicate: Option<PrimExpr>,
}

/// Reference-counted handle to a TIR buffer read.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct BufferLoad {
    data: ObjectArc<BufferLoadObj>,
}

impl std::ops::Deref for BufferLoad {
    type Target = BufferLoadObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl std::ops::Deref for BufferLoadObj {
    type Target = ExprObj;

    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

impl BufferLoad {
    /// Construct a buffer read directly in Rust after validating its access types.
    pub fn new<B>(buffer: B, indices: Vec<Expr>, predicate: Option<Expr>) -> Result<Self>
    where
        B: Into<Var>,
    {
        Self::with_span(buffer.into(), indices, predicate, None)
    }

    /// Construct a validated buffer read with optional source metadata.
    pub fn with_span(
        buffer: Var,
        indices: Vec<Expr>,
        predicate: Option<Expr>,
        span: Option<&Span>,
    ) -> Result<Self> {
        let buffer_type = buffer_type(&buffer)?;
        validate_access_dimensions(&buffer, &buffer_type, &indices)?;
        validate_nonfinal_indices(&indices)?;

        let buffer_dtype = buffer_type.dtype.clone();
        let result_type = if let Some(index) = indices.last() {
            let index_type = primitive_type(index, "buffer load index")?;
            vectorized_buffer_type(&buffer_dtype, &index_type)?
        } else {
            buffer_dtype.clone()
        };
        if let Some(predicate) = predicate.as_ref() {
            validate_load_predicate(&buffer_dtype, indices.last(), predicate)?;
        }

        let buffer = BufferVar::try_from(buffer)?;
        let indices = indices
            .into_iter()
            .map(PrimExpr::try_from)
            .collect::<Result<Vec<_>>>()?;
        let predicate = predicate.map(PrimExpr::try_from).transpose()?;

        Ok(Self::from_complete_fields(
            span.cloned(),
            result_type,
            buffer,
            Array::new(indices),
            predicate,
        ))
    }

    /// Construct a buffer load from every physical field without re-deriving its result type.
    pub fn from_complete_fields(
        span: Option<Span>,
        ty: PrimType,
        buffer: BufferVar,
        indices: Array<PrimExpr>,
        predicate: Option<PrimExpr>,
    ) -> Self {
        Self {
            data: ObjectArc::new(BufferLoadObj {
                base: ExprObj::new(span, ty.into()),
                buffer,
                indices,
                predicate,
            }),
        }
    }
}

/// ABI-complete Rust representation of a buffer write.
#[repr(C)]
#[derive(Object)]
#[type_key = "tirx.BufferStore"]
#[type_final]
pub struct BufferStoreObj {
    base: StmtObj,
    pub buffer: BufferVar,
    pub value: PrimExpr,
    pub indices: Array<PrimExpr>,
    pub predicate: Option<PrimExpr>,
}

/// Reference-counted handle to a TIR buffer write.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct BufferStore {
    data: ObjectArc<BufferStoreObj>,
}

impl std::ops::Deref for BufferStore {
    type Target = BufferStoreObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl std::ops::Deref for BufferStoreObj {
    type Target = StmtObj;

    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

impl BufferStore {
    /// Construct a buffer write directly in Rust after validating its access types.
    pub fn new<B, V>(
        buffer: B,
        value: V,
        indices: Vec<Expr>,
        predicate: Option<Expr>,
    ) -> Result<Self>
    where
        B: Into<Var>,
        V: Into<Expr>,
    {
        Self::with_span(buffer.into(), value.into(), indices, predicate, None)
    }

    /// Construct a validated buffer write with optional source metadata.
    pub fn with_span(
        buffer: Var,
        value: Expr,
        indices: Vec<Expr>,
        predicate: Option<Expr>,
        span: Option<&Span>,
    ) -> Result<Self> {
        let buffer_type = buffer_type(&buffer)?;
        validate_access_dimensions(&buffer, &buffer_type, &indices)?;
        validate_nonfinal_indices(&indices)?;

        let buffer_dtype = buffer_type.dtype.clone();
        let value_dtype = primitive_type(&value, "buffer store value")?;
        let index_dtype = indices
            .last()
            .map(|index| primitive_type(index, "buffer store index"))
            .transpose()?;
        let expected_dtype = match &index_dtype {
            Some(index_dtype) => vectorized_buffer_type(&buffer_dtype, index_dtype)?,
            None => buffer_dtype.clone(),
        };
        if expected_dtype.dtype != value_dtype.dtype {
            return Err(Error::new(
                TYPE_ERROR,
                &format!(
                    "dtype mismatch on BufferStore: expected {}, got {}",
                    expected_dtype.dtype.to_string(),
                    value_dtype.dtype.to_string()
                ),
                "",
            ));
        }
        if let Some(predicate) = predicate.as_ref() {
            validate_store_predicate(&value_dtype, predicate)?;
        }

        let buffer = BufferVar::try_from(buffer)?;
        let value = PrimExpr::try_from(value)?;
        let indices = indices
            .into_iter()
            .map(PrimExpr::try_from)
            .collect::<Result<Vec<_>>>()?;
        let predicate = predicate.map(PrimExpr::try_from).transpose()?;

        Ok(Self::from_complete_fields(
            span.cloned(),
            buffer,
            value,
            Array::new(indices),
            predicate,
        ))
    }

    /// Construct a buffer write from every physical field after external validation.
    pub fn from_complete_fields(
        span: Option<Span>,
        buffer: BufferVar,
        value: PrimExpr,
        indices: Array<PrimExpr>,
        predicate: Option<PrimExpr>,
    ) -> Self {
        Self {
            data: ObjectArc::new(BufferStoreObj {
                base: StmtObj::new(span),
                buffer,
                value,
                indices,
                predicate,
            }),
        }
    }
}

fn buffer_type(buffer: &Var) -> Result<BufferType> {
    buffer.ty.clone().try_cast::<BufferType>().map_err(|_| {
        Error::new(
            TYPE_ERROR,
            "buffer access requires a variable whose type is tirx.BufferType",
            "",
        )
    })
}

fn validate_access_dimensions(
    buffer: &Var,
    buffer_type: &BufferType,
    indices: &[Expr],
) -> Result<()> {
    if buffer_type.shape.len() == indices.len() {
        Ok(())
    } else {
        Err(Error::new(
            VALUE_ERROR,
            &format!(
                "buffer {} is {}-dimensional but received {} indices",
                buffer.name.as_str(),
                buffer_type.shape.len(),
                indices.len()
            ),
            "",
        ))
    }
}

fn validate_nonfinal_indices(indices: &[Expr]) -> Result<()> {
    for index in indices.iter().take(indices.len().saturating_sub(1)) {
        let dtype = primitive_type(index, "buffer index")?.dtype;
        if encoded_lanes(dtype) != 1 {
            return Err(Error::new(
                TYPE_ERROR,
                "only the last index of a buffer access may be a vector type",
                "",
            ));
        }
    }
    Ok(())
}

fn vectorized_buffer_type(buffer: &PrimType, index: &PrimType) -> Result<PrimType> {
    let buffer_lanes = encoded_lanes(buffer.dtype);
    let index_lanes = encoded_lanes(index.dtype);
    let buffer_scalable = buffer_lanes < -1;
    let index_scalable = index_lanes < -1;
    if buffer_scalable && index_scalable {
        return Err(Error::new(
            TYPE_ERROR,
            "index dtype and buffer dtype cannot both be scalable",
            "",
        ));
    }
    let lanes = lane_factor(buffer.dtype)?
        .checked_mul(lane_factor(index.dtype)?)
        .ok_or_else(|| Error::new(VALUE_ERROR, "buffer access lane count overflow", ""))?;
    vector_type_like(buffer.dtype, lanes, buffer_scalable || index_scalable)
}

fn validate_load_predicate(
    buffer: &PrimType,
    index: Option<&Expr>,
    predicate: &Expr,
) -> Result<()> {
    let index = index
        .map(|index| primitive_type(index, "buffer load index"))
        .transpose()?;
    let predicate = primitive_type(predicate, "buffer load predicate")?;
    let index_scalable = index
        .as_ref()
        .is_some_and(|index| encoded_lanes(index.dtype) < -1);
    let predicate_scalable = encoded_lanes(predicate.dtype) < -1;
    if index_scalable != predicate_scalable {
        return Err(Error::new(
            TYPE_ERROR,
            "predicate mask dtype and load indices must both be scalable",
            "",
        ));
    }
    let index_lanes = index
        .as_ref()
        .map(|index| lane_factor(index.dtype))
        .transpose()?
        .unwrap_or(1);
    let expected_lanes = index_lanes
        .checked_mul(lane_factor(buffer.dtype)?)
        .ok_or_else(|| Error::new(VALUE_ERROR, "buffer load lane count overflow", ""))?;
    if lane_factor(predicate.dtype)? != expected_lanes {
        return Err(Error::new(
            TYPE_ERROR,
            "predicate mask lanes must match the loaded value lanes",
            "",
        ));
    }
    validate_predicate_element_type(&predicate)
}

fn validate_store_predicate(value: &PrimType, predicate: &Expr) -> Result<()> {
    let predicate = primitive_type(predicate, "buffer store predicate")?;
    if (encoded_lanes(value.dtype) < -1) != (encoded_lanes(predicate.dtype) < -1) {
        return Err(Error::new(
            TYPE_ERROR,
            "predicate mask dtype and value dtype must both be scalable",
            "",
        ));
    }
    if lane_factor(value.dtype)? != lane_factor(predicate.dtype)? {
        return Err(Error::new(
            TYPE_ERROR,
            "predicate mask lanes must match the stored value lanes",
            "",
        ));
    }
    validate_predicate_element_type(&predicate)
}

fn validate_predicate_element_type(predicate: &PrimType) -> Result<()> {
    let dtype = predicate.dtype;
    let is_boolean = dtype.code == DLDataTypeCode::kDLBool as u8;
    let is_uint1 = dtype.code == DLDataTypeCode::kDLUInt as u8 && dtype.bits == 1;
    if is_boolean || is_uint1 {
        Ok(())
    } else {
        Err(Error::new(
            TYPE_ERROR,
            "predicate mask elements must be boolean values",
            "",
        ))
    }
}

fn vector_type_like(element: DLDataType, lanes: i32, scalable: bool) -> Result<PrimType> {
    if lanes <= 0 || lanes > i32::from(i16::MAX) {
        return Err(Error::new(
            VALUE_ERROR,
            "buffer access lane count does not fit the DLPack encoding",
            "",
        ));
    }
    let encoded = if scalable {
        i16::try_from(-lanes)
    } else {
        i16::try_from(lanes)
    }
    .map_err(|_| Error::new(VALUE_ERROR, "invalid buffer access lane count", ""))?;
    PrimType::from_dtype(DLDataType {
        code: element.code,
        bits: element.bits,
        lanes: encoded as u16,
    })
}

fn lane_factor(dtype: DLDataType) -> Result<i32> {
    let encoded = encoded_lanes(dtype);
    if encoded < -1 {
        Ok(i32::from(-encoded))
    } else if encoded > 0 {
        Ok(i32::from(encoded))
    } else {
        Err(Error::new(TYPE_ERROR, "invalid vector lane encoding", ""))
    }
}

fn encoded_lanes(dtype: DLDataType) -> i16 {
    dtype.lanes as i16
}

/// Opaque Rust representation of one polymorphic declared buffer region.
#[repr(C)]
#[derive(Object)]
#[type_key = "tirx.BufferRegion"]
#[type_final]
pub struct BufferRegionObj {
    base: PrimExprConvertibleObj,
}

/// Reference-counted handle to a multidimensional buffer region.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct BufferRegion {
    data: ObjectArc<BufferRegionObj>,
}

impl std::ops::Deref for BufferRegion {
    type Target = BufferRegionObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl std::ops::Deref for BufferRegionObj {
    type Target = PrimExprConvertibleObj;

    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

impl BufferRegion {
    fn field<T>(&self, name: &str) -> Result<T>
    where
        T: TryFrom<Any, Error = Error>,
    {
        FieldGetter::new(BufferRegionObj::type_index(), name)?.get(&**self)
    }

    pub fn buffer(&self) -> Result<BufferVar> {
        self.field("buffer")
    }

    pub fn region(&self) -> Result<Array<Range>> {
        self.field("region")
    }

    /// Validate and construct a declared region through its native constructor.
    pub fn new<B>(buffer: B, region: Vec<Range>) -> Result<Self>
    where
        B: Into<Var>,
    {
        let buffer = buffer.into();
        let region = Array::new(region);
        let dimensions = buffer_type(&buffer)?.shape.len();
        if dimensions != region.len() {
            return Err(Error::new(
                VALUE_ERROR,
                &format!(
                    "BufferRegion dimension mismatch: buffer has {dimensions}, region has {}",
                    region.len()
                ),
                "",
            ));
        }
        let buffer = BufferVar::try_from(buffer)?;
        tvm_ffi::cached_global_func!("tirx.BufferRegion")
            .call_tuple((buffer, region))?
            .try_into()
    }
}

/// ABI-complete Rust representation of a match-buffer declaration.
#[repr(C)]
#[derive(Object)]
#[type_key = "tirx.MatchBufferRegion"]
#[type_final]
pub struct MatchBufferRegionObj {
    base: tvm_ffi::Object,
    pub buffer: BufferVar,
    pub source: BufferRegion,
}

/// Reference-counted handle to a match-buffer declaration.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct MatchBufferRegion {
    data: ObjectArc<MatchBufferRegionObj>,
}

impl std::ops::Deref for MatchBufferRegion {
    type Target = MatchBufferRegionObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl MatchBufferRegion {
    /// Validate the declaration and allocate the object directly in Rust.
    pub fn new<B, S>(buffer: B, source: S) -> Result<Self>
    where
        B: Into<Var>,
        S: Into<BufferRegion>,
    {
        let buffer = buffer.into();
        let source = source.into();
        validate_match_buffer_region(&buffer, &source)?;
        let buffer = BufferVar::try_from(buffer)?;
        Ok(Self::from_complete_fields(buffer, source))
    }

    /// Allocate a match-buffer declaration directly after its invariants have been validated.
    pub fn from_complete_fields(buffer: BufferVar, source: BufferRegion) -> Self {
        Self {
            data: ObjectArc::new(MatchBufferRegionObj {
                base: tvm_ffi::Object::new(),
                buffer,
                source,
            }),
        }
    }
}

fn validate_match_buffer_region(buffer: &Var, source: &BufferRegion) -> Result<()> {
    let target = buffer_type(buffer)?;
    let source_buffer = source.buffer()?;
    let source_type = buffer_type(&source_buffer)?;
    if target.storage_scope != source_type.storage_scope {
        return Err(Error::new(
            TYPE_ERROR,
            "match-buffer storage scopes differ",
            "",
        ));
    }
    if target.dtype.dtype != source_type.dtype.dtype {
        return Err(Error::new(TYPE_ERROR, "match-buffer data types differ", ""));
    }
    if target.data_alignment == 0
        || source_type.data_alignment.rem_euclid(target.data_alignment) != 0
    {
        return Err(Error::new(
            VALUE_ERROR,
            "source buffer does not satisfy the required alignment",
            "",
        ));
    }

    let region = source.region()?;
    if region.len() < target.shape.len() {
        return Err(Error::new(
            VALUE_ERROR,
            "source region has fewer dimensions than the target buffer",
            "",
        ));
    }
    let analyzer = Analyzer::new()?;
    let offset = region.len() - target.shape.len();
    for range in region.iter().take(offset) {
        let one: PrimExpr = IntImm::from_complete_fields(
            None,
            primitive_type(&range.extent, "source-region extent")?,
            1,
        )
        .into();
        if !analyzer.can_prove_equal(&range.extent, &one)? {
            return Err(Error::new(
                VALUE_ERROR,
                "a leading source-region dimension is not provably one",
                "",
            ));
        }
    }
    for (range, expected) in region.iter().skip(offset).zip(target.shape.iter()) {
        let is_primitive_variable = expected.clone().try_cast::<Var>().is_ok()
            && primitive_type(&expected, "match-buffer shape").is_ok();
        if !is_primitive_variable && !analyzer.can_prove_equal(&range.extent, &expected)? {
            return Err(Error::new(
                VALUE_ERROR,
                "source-region extent does not match the target buffer shape",
                "",
            ));
        }
    }
    Ok(())
}

tvm_ffi::impl_object_upcast!(
    TileLayout => Layout,
    BufferType => Type,
    BufferLoad => Expr,
    BufferLoad => PrimExpr,
    BufferStore => Stmt,
    BufferRegion => PrimExprConvertible,
);
