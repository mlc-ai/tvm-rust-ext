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
    Any, AnyCompatible, AnyView, Array, DLDataType, DLDataTypeCode, DLDataTypeExt, Error,
    FieldGetter, Map, ObjectArc, ObjectCore, ObjectRefCast, ObjectRefCore, Result, String,
    TVMFFIAny, TYPE_ERROR, VALUE_ERROR,
};

/// ABI-complete Rust representation of TVM's `ExprNode` prefix.
#[repr(C)]
#[derive(Object)]
#[type_key = "ir.Expr"]
pub struct ExprObj {
    base: tvm_ffi::Object,
    pub span: Option<Span>,
    pub ty: Type,
}

/// Reference-counted handle to any TVM expression.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct Expr {
    data: ObjectArc<ExprObj>,
}

impl std::ops::Deref for Expr {
    type Target = ExprObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

/// ABI-complete Rust representation of TVM's opaque construction-time expression base.
#[repr(C)]
#[derive(Object)]
#[type_key = "ir.OpaqueExpr"]
pub struct OpaqueExprObj {
    base: ExprObj,
}

/// Reference-counted handle to an opaque construction-time expression.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct OpaqueExpr {
    data: ObjectArc<OpaqueExprObj>,
}

impl std::ops::Deref for OpaqueExpr {
    type Target = OpaqueExprObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl std::ops::Deref for OpaqueExprObj {
    type Target = ExprObj;

    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

/// Checked view over an expression whose result type is `T`.
///
/// This mirrors C++ `TypedExpr<T>`: it retains the same `ExprNode` allocation
/// and adds a runtime check on the expression's `ty` field.
#[repr(transparent)]
pub struct TypedExpr<T> {
    expr: Expr,
    _expected_type: std::marker::PhantomData<T>,
}

impl<T> TypedExpr<T> {
    #[inline]
    pub(crate) unsafe fn from_expr_unchecked(expr: Expr) -> Self {
        Self {
            expr,
            _expected_type: std::marker::PhantomData,
        }
    }

    /// Return the general expression view over the same allocation.
    #[inline]
    pub fn as_expr(&self) -> &Expr {
        &self.expr
    }
}

impl<T> Clone for TypedExpr<T> {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            expr: self.expr.clone(),
            _expected_type: std::marker::PhantomData,
        }
    }
}

impl<T> std::ops::Deref for TypedExpr<T> {
    type Target = Expr;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.expr
    }
}

unsafe impl<T: 'static> ObjectRefCore for TypedExpr<T> {
    type ContainerType = ExprObj;

    #[inline]
    fn data(this: &Self) -> &ObjectArc<Self::ContainerType> {
        <Expr as ObjectRefCore>::data(&this.expr)
    }

    #[inline]
    fn into_data(this: Self) -> ObjectArc<Self::ContainerType> {
        <Expr as ObjectRefCore>::into_data(this.expr)
    }

    #[inline]
    unsafe fn from_data(data: ObjectArc<Self::ContainerType>) -> Self {
        // SAFETY: The caller of this unsafe constructor has guaranteed that
        // the expression's `ty` satisfies this typed view's invariant.
        unsafe { Self::from_expr_unchecked(<Expr as ObjectRefCore>::from_data(data)) }
    }
}

unsafe impl<T> AnyCompatible for TypedExpr<T>
where
    T: ObjectRefCore + AnyCompatible + 'static,
{
    #[inline]
    unsafe fn copy_to_any_view(src: &Self, data: &mut TVMFFIAny) {
        Expr::copy_to_any_view(&src.expr, data);
    }

    #[inline]
    unsafe fn move_to_any(src: Self, data: &mut TVMFFIAny) {
        Expr::move_to_any(src.expr, data);
    }

    #[inline]
    unsafe fn check_any_strict(data: &TVMFFIAny) -> bool {
        if !Expr::check_any_strict(data) {
            return false;
        }
        let expr = &*data.data_union.v_obj.cast::<ExprObj>();
        let mut type_data = TVMFFIAny::new();
        Type::copy_to_any_view(&expr.ty, &mut type_data);
        T::check_any_strict(&type_data)
    }

    #[inline]
    unsafe fn copy_from_any_view_after_check(data: &TVMFFIAny) -> Self {
        Self::from_expr_unchecked(Expr::copy_from_any_view_after_check(data))
    }

    #[inline]
    unsafe fn move_from_any_after_check(data: &mut TVMFFIAny) -> Self {
        Self::from_expr_unchecked(Expr::move_from_any_after_check(data))
    }

    #[inline]
    unsafe fn try_cast_from_any_view(data: &TVMFFIAny) -> std::result::Result<Self, ()> {
        if Self::check_any_strict(data) {
            Ok(Self::copy_from_any_view_after_check(data))
        } else {
            Err(())
        }
    }

    fn type_str() -> std::string::String {
        format!("TypedExpr<{}>", T::type_str())
    }
}

impl<T> From<TypedExpr<T>> for Expr {
    #[inline]
    fn from(value: TypedExpr<T>) -> Self {
        value.expr
    }
}

impl<T> From<&TypedExpr<T>> for Expr {
    #[inline]
    fn from(value: &TypedExpr<T>) -> Self {
        value.expr.clone()
    }
}

impl<T> From<&TypedExpr<T>> for TypedExpr<T> {
    #[inline]
    fn from(value: &TypedExpr<T>) -> Self {
        value.clone()
    }
}

impl<T> TryFrom<Expr> for TypedExpr<T>
where
    T: ObjectRefCore + AnyCompatible + 'static,
{
    type Error = Error;

    #[inline]
    fn try_from(value: Expr) -> Result<Self> {
        value.try_cast()
    }
}

impl<T> TryFrom<&Expr> for TypedExpr<T>
where
    T: ObjectRefCore + AnyCompatible + 'static,
{
    type Error = Error;

    #[inline]
    fn try_from(value: &Expr) -> Result<Self> {
        value.clone().try_cast()
    }
}

/// Checked view over an expression whose result is a primitive scalar or vector type.
pub type PrimExpr = TypedExpr<PrimType>;

tvm_ffi::impl_try_from_any!(PrimExpr);
tvm_ffi::impl_arg_into_ref!(PrimExpr);
tvm_ffi::impl_into_arg_holder_default!(PrimExpr);

impl ExprObj {
    pub(crate) fn new(span: Option<Span>, ty: Type) -> Self {
        Self {
            base: tvm_ffi::Object::new(),
            span,
            ty,
        }
    }
}

/// ABI-complete Rust representation of TVM's `BaseFuncNode` prefix.
#[repr(C)]
#[derive(Object)]
#[type_key = "ir.BaseFunc"]
pub struct BaseFuncObj {
    base: ExprObj,
    pub attrs: DictAttrs,
}

/// Reference-counted handle to any TVM base function.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct BaseFunc {
    data: ObjectArc<BaseFuncObj>,
}

impl std::ops::Deref for BaseFunc {
    type Target = BaseFuncObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl std::ops::Deref for BaseFuncObj {
    type Target = ExprObj;

    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

impl BaseFuncObj {
    pub(crate) fn new(span: Option<Span>, ty: Type, attrs: DictAttrs) -> Self {
        Self {
            base: ExprObj::new(span, ty),
            attrs,
        }
    }
}

/// ABI-complete Rust representation of TVM's `GlobalVarNode`.
#[repr(C)]
#[derive(Object)]
#[type_key = "ir.GlobalVar"]
#[type_final]
pub struct GlobalVarObj {
    base: ExprObj,
    pub name_hint: String,
}

/// Reference-counted handle to a global function name.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct GlobalVar {
    data: ObjectArc<GlobalVarObj>,
}

impl std::ops::Deref for GlobalVar {
    type Target = GlobalVarObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl std::ops::Deref for GlobalVarObj {
    type Target = ExprObj;

    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

/// ABI-complete Rust representation of TVM's `VarNode` prefix.
#[repr(C)]
#[derive(Object)]
#[type_key = "ir.Var"]
pub struct VarObj {
    base: ExprObj,
    pub name: String,
}

/// Reference-counted handle to a TVM variable.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct Var {
    data: ObjectArc<VarObj>,
}

impl std::ops::Deref for Var {
    type Target = VarObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl std::ops::Deref for VarObj {
    type Target = ExprObj;

    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

/// Checked view over a `Var` whose expression type is `T`.
///
/// C++ uses zero-state views such as `PrimVar` and `BufferVar` over the same
/// `VarNode`; this type gives generated Rust bindings the same representation.
#[repr(transparent)]
pub struct TypedVar<T> {
    var: Var,
    _expected_type: std::marker::PhantomData<T>,
}

impl<T> TypedVar<T> {
    #[inline]
    pub(crate) unsafe fn from_var_unchecked(var: Var) -> Self {
        Self {
            var,
            _expected_type: std::marker::PhantomData,
        }
    }

    /// Return the general variable view over the same allocation.
    #[inline]
    pub fn as_var(&self) -> &Var {
        &self.var
    }
}

impl<T> Clone for TypedVar<T> {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            var: self.var.clone(),
            _expected_type: std::marker::PhantomData,
        }
    }
}

impl<T> std::ops::Deref for TypedVar<T> {
    type Target = Var;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.var
    }
}

unsafe impl<T: 'static> ObjectRefCore for TypedVar<T> {
    type ContainerType = VarObj;

    #[inline]
    fn data(this: &Self) -> &ObjectArc<Self::ContainerType> {
        <Var as ObjectRefCore>::data(&this.var)
    }

    #[inline]
    fn into_data(this: Self) -> ObjectArc<Self::ContainerType> {
        <Var as ObjectRefCore>::into_data(this.var)
    }

    #[inline]
    unsafe fn from_data(data: ObjectArc<Self::ContainerType>) -> Self {
        // SAFETY: The caller has guaranteed both the `Var` dynamic type and
        // this typed variable view's `ty` invariant.
        unsafe { Self::from_var_unchecked(<Var as ObjectRefCore>::from_data(data)) }
    }
}

unsafe impl<T> AnyCompatible for TypedVar<T>
where
    T: ObjectRefCore + AnyCompatible + 'static,
{
    #[inline]
    unsafe fn copy_to_any_view(src: &Self, data: &mut TVMFFIAny) {
        Var::copy_to_any_view(&src.var, data);
    }

    #[inline]
    unsafe fn move_to_any(src: Self, data: &mut TVMFFIAny) {
        Var::move_to_any(src.var, data);
    }

    #[inline]
    unsafe fn check_any_strict(data: &TVMFFIAny) -> bool {
        Var::check_any_strict(data) && TypedExpr::<T>::check_any_strict(data)
    }

    #[inline]
    unsafe fn copy_from_any_view_after_check(data: &TVMFFIAny) -> Self {
        Self::from_var_unchecked(Var::copy_from_any_view_after_check(data))
    }

    #[inline]
    unsafe fn move_from_any_after_check(data: &mut TVMFFIAny) -> Self {
        Self::from_var_unchecked(Var::move_from_any_after_check(data))
    }

    #[inline]
    unsafe fn try_cast_from_any_view(data: &TVMFFIAny) -> std::result::Result<Self, ()> {
        if Self::check_any_strict(data) {
            Ok(Self::copy_from_any_view_after_check(data))
        } else {
            Err(())
        }
    }

    fn type_str() -> std::string::String {
        format!("TypedVar<{}>", T::type_str())
    }
}

impl<T> From<TypedVar<T>> for Var {
    #[inline]
    fn from(value: TypedVar<T>) -> Self {
        value.var
    }
}

impl<T> From<&TypedVar<T>> for Var {
    #[inline]
    fn from(value: &TypedVar<T>) -> Self {
        value.var.clone()
    }
}

impl<T> From<&TypedVar<T>> for TypedVar<T> {
    #[inline]
    fn from(value: &TypedVar<T>) -> Self {
        value.clone()
    }
}

impl<T> From<TypedVar<T>> for Expr {
    #[inline]
    fn from(value: TypedVar<T>) -> Self {
        value.var.into()
    }
}

impl<T> From<&TypedVar<T>> for Expr {
    #[inline]
    fn from(value: &TypedVar<T>) -> Self {
        value.var.clone().into()
    }
}

impl<T> From<TypedVar<T>> for TypedExpr<T>
where
    T: ObjectRefCore + AnyCompatible + 'static,
{
    #[inline]
    fn from(value: TypedVar<T>) -> Self {
        unsafe { TypedExpr::from_expr_unchecked(value.var.into()) }
    }
}

impl<T> From<&TypedVar<T>> for TypedExpr<T>
where
    T: ObjectRefCore + AnyCompatible + 'static,
{
    #[inline]
    fn from(value: &TypedVar<T>) -> Self {
        value.clone().into()
    }
}

impl<T> TryFrom<Var> for TypedVar<T>
where
    T: ObjectRefCore + AnyCompatible + 'static,
{
    type Error = Error;

    #[inline]
    fn try_from(value: Var) -> Result<Self> {
        value.try_cast()
    }
}

impl<T> TryFrom<&Var> for TypedVar<T>
where
    T: ObjectRefCore + AnyCompatible + 'static,
{
    type Error = Error;

    #[inline]
    fn try_from(value: &Var) -> Result<Self> {
        value.clone().try_cast()
    }
}

/// Opaque Rust representation of an interned TVM source name.
#[repr(C)]
#[derive(Object)]
#[type_key = "ir.SourceName"]
#[type_final]
pub struct SourceNameObj {
    base: tvm_ffi::Object,
}

/// Reference-counted handle to a source name.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct SourceName {
    data: ObjectArc<SourceNameObj>,
}

impl std::ops::Deref for SourceName {
    type Target = SourceNameObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl SourceName {
    /// Return the interned native source name for `name`.
    pub fn get(name: &str) -> Result<Self> {
        tvm_ffi::cached_global_func!("ir.SourceName")
            .call_tuple((String::from(name),))?
            .try_into()
    }

    /// Return the source name text through native reflection.
    pub fn name(&self) -> Result<String> {
        FieldGetter::new(SourceNameObj::type_index(), "name")?.get(&**self)
    }
}

/// Opaque Rust representation of TVM's `SourceNode`.
///
/// The native node owns a `std::vector` line index, so Rust must not reproduce
/// or directly allocate its physical layout.
#[repr(C)]
#[derive(Object)]
#[type_key = "ir.Source"]
#[type_final]
pub struct SourceObj {
    base: tvm_ffi::Object,
}

/// Reference-counted handle to one program source fragment.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct Source {
    data: ObjectArc<SourceObj>,
}

impl std::ops::Deref for Source {
    type Target = SourceObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl Source {
    fn field<T>(&self, name: &str) -> Result<T>
    where
        T: TryFrom<Any, Error = Error>,
    {
        FieldGetter::new(SourceObj::type_index(), name)?.get(&**self)
    }

    /// Return the interned name associated with this native source object.
    pub fn source_name(&self) -> Result<SourceName> {
        self.field("source_name")
    }

    /// Return the native source text.
    pub fn text(&self) -> Result<String> {
        self.field("source")
    }
}

/// ABI-complete Rust representation of TVM's `SourceMapObj`.
#[repr(C)]
#[derive(Object)]
#[type_key = "ir.SourceMap"]
#[type_final]
pub struct SourceMapObj {
    base: tvm_ffi::Object,
    pub source_map: Map<SourceName, Source>,
}

/// Reference-counted handle to TVM's source-name-to-source mapping.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct SourceMap {
    data: ObjectArc<SourceMapObj>,
}

impl std::ops::Deref for SourceMap {
    type Target = SourceMapObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl SourceMap {
    /// Construct an empty source map directly in Rust.
    pub fn new() -> Self {
        Self::from_map(Map::new())
    }

    /// Construct a source map directly from its complete backing map.
    pub fn from_map(source_map: Map<SourceName, Source>) -> Self {
        Self::from_complete_fields(source_map)
    }

    /// Construct a source map from its complete physical state.
    pub fn from_complete_fields(source_map: Map<SourceName, Source>) -> Self {
        Self {
            data: ObjectArc::new(SourceMapObj {
                base: tvm_ffi::Object::new(),
                source_map,
            }),
        }
    }

    /// Ask TVM to construct and add a native source fragment to this map.
    pub fn add(&mut self, name: &str, content: &str) -> Result<SourceName> {
        let name = String::from(name);
        let content = String::from(content);
        tvm_ffi::cached_global_func!("SourceMapAdd")
            .call_packed(&[
                AnyView::from(&*self),
                AnyView::from(&name),
                AnyView::from(&content),
            ])?
            .try_into()
    }
}

/// ABI-complete Rust representation of TVM's source-span metadata.
#[repr(C)]
#[derive(Object)]
#[type_key = "ir.Span"]
pub struct SpanObj {
    base: tvm_ffi::Object,
    pub source_name: SourceName,
    pub line: i32,
    pub column: i32,
    pub end_line: i32,
    pub end_column: i32,
}

/// Reference-counted handle to source-span metadata.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct Span {
    data: ObjectArc<SpanObj>,
}

impl Span {
    /// Construct source-location metadata in TVM's `(line, end_line, column, end_column)` order.
    pub fn new<S>(
        source_name: S,
        line: i64,
        end_line: i64,
        column: i64,
        end_column: i64,
    ) -> Result<Self>
    where
        S: Into<SourceName>,
    {
        let line = i32::try_from(line).map_err(|_| integer_field_overflow("line", line))?;
        let end_line =
            i32::try_from(end_line).map_err(|_| integer_field_overflow("end_line", end_line))?;
        let column = i32::try_from(column).map_err(|_| integer_field_overflow("column", column))?;
        let end_column = i32::try_from(end_column)
            .map_err(|_| integer_field_overflow("end_column", end_column))?;
        Ok(Self::from_complete_fields(
            source_name.into(),
            line,
            column,
            end_line,
            end_column,
        ))
    }

    /// Construct a span from the exact-width values stored by the native object.
    pub fn from_complete_fields(
        source_name: SourceName,
        line: i32,
        column: i32,
        end_line: i32,
        end_column: i32,
    ) -> Self {
        Self {
            data: ObjectArc::new(SpanObj {
                base: tvm_ffi::Object::new(),
                source_name,
                line,
                column,
                end_line,
                end_column,
            }),
        }
    }
}

fn integer_field_overflow(field: &str, value: i64) -> Error {
    Error::new(
        VALUE_ERROR,
        &format!("{field} value {value} does not fit TVM's 32-bit integer field"),
        "",
    )
}

/// Opaque prefix for TVM objects that can convert to a primitive expression.
#[repr(C)]
#[derive(Object)]
#[type_key = "ir.PrimExprConvertible"]
pub struct PrimExprConvertibleObj {
    base: tvm_ffi::Object,
}
/// Reference-counted handle to a primitive-expression-convertible object.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct PrimExprConvertible {
    data: ObjectArc<PrimExprConvertibleObj>,
}

impl std::ops::Deref for PrimExprConvertible {
    type Target = PrimExprConvertibleObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl PrimExprConvertible {
    /// Invoke TVM's standard FFI fallback conversion to a primitive expression.
    pub fn to_prim_expr(&self) -> Result<Expr> {
        tvm_ffi::cached_global_func!("tirx.convert")
            .call_tuple((self,))?
            .try_into()
    }
}

/// ABI-complete Rust representation of an integer range.
#[repr(C)]
#[derive(Object)]
#[type_key = "ir.Range"]
#[type_final]
pub struct RangeObj {
    base: tvm_ffi::Object,
    pub min: PrimExpr,
    pub extent: PrimExpr,
    pub span: Option<Span>,
}

/// Reference-counted handle to `min .. min + extent`.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct Range {
    data: ObjectArc<RangeObj>,
}

impl std::ops::Deref for Range {
    type Target = RangeObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl Range {
    /// Construct a range from its minimum and extent.
    pub fn from_min_extent<M, E>(minimum: M, extent: E) -> Result<Self>
    where
        M: Into<Expr>,
        E: Into<Expr>,
    {
        Self::from_min_extent_with_span(minimum, extent, None)
    }

    /// Construct a range from all of its physical fields.
    pub fn from_min_extent_with_span<M, E>(
        minimum: M,
        extent: E,
        span: Option<&Span>,
    ) -> Result<Self>
    where
        M: Into<Expr>,
        E: Into<Expr>,
    {
        let minimum = minimum.into();
        let extent = extent.into();
        let minimum = require_primitive_expr(minimum, "Range minimum")?;
        let extent = require_primitive_expr(extent, "Range extent")?;
        Ok(Self::from_complete_fields(minimum, extent, span.cloned()))
    }

    /// Construct a range from every physical field after external validation.
    pub fn from_complete_fields(min: PrimExpr, extent: PrimExpr, span: Option<Span>) -> Self {
        Self {
            data: ObjectArc::new(RangeObj {
                base: tvm_ffi::Object::new(),
                min,
                extent,
                span,
            }),
        }
    }
}

fn require_primitive_expr(value: Expr, context: &str) -> Result<PrimExpr> {
    PrimExpr::try_from(value).map_err(|_| {
        Error::new(
            TYPE_ERROR,
            &format!("{context} must have a primitive type"),
            "",
        )
    })
}

impl std::ops::Deref for Span {
    type Target = SpanObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

/// ABI-complete Rust representation of TVM's `CallNode`.
#[repr(C)]
#[derive(Object)]
#[type_key = "ir.Call"]
#[type_final]
pub struct CallObj {
    base: ExprObj,
    pub op: Expr,
    pub args: Array<Expr>,
    pub attrs: Option<Attrs>,
    pub ty_args: Array<Type>,
}

/// Reference-counted handle to a generic call expression.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct Call {
    data: ObjectArc<CallObj>,
}

impl std::ops::Deref for Call {
    type Target = CallObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl std::ops::Deref for CallObj {
    type Target = ExprObj;

    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

/// ABI-complete Rust representation of TVM's `TypeNode` prefix.
#[repr(C)]
#[derive(Object)]
#[type_key = "ir.Type"]
pub struct TypeObj {
    base: tvm_ffi::Object,
    pub span: Option<Span>,
}

/// Reference-counted handle to any TVM type.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct Type {
    data: ObjectArc<TypeObj>,
}

impl std::ops::Deref for Type {
    type Target = TypeObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

/// ABI-complete Rust representation of TVM's opaque construction-time type marker.
#[repr(C)]
#[derive(Object)]
#[type_key = "ir.OpaqueType"]
#[type_final]
pub struct OpaqueTypeObj {
    base: TypeObj,
}

/// Reference-counted handle to TVM's opaque construction-time type marker.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct OpaqueType {
    data: ObjectArc<OpaqueTypeObj>,
}

impl std::ops::Deref for OpaqueType {
    type Target = OpaqueTypeObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl std::ops::Deref for OpaqueTypeObj {
    type Target = TypeObj;

    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

impl TypeObj {
    pub(crate) fn new(span: Option<Span>) -> Self {
        Self {
            base: tvm_ffi::Object::new(),
            span,
        }
    }
}

/// Checked Rust view of TVM's low-level pointer type.
///
/// The constructor logic only needs to distinguish pointer parameters from
/// ordinary types; stubgen will eventually replace this minimal view with the
/// complete generated binding.
#[repr(C)]
#[derive(Object)]
#[type_key = "ir.PointerType"]
#[type_final]
pub struct PointerTypeObj {
    base: TypeObj,
}

#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct PointerType {
    data: ObjectArc<PointerTypeObj>,
}

impl std::ops::Deref for PointerType {
    type Target = PointerTypeObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl std::ops::Deref for PointerTypeObj {
    type Target = TypeObj;

    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

/// ABI-complete Rust representation of TVM's `PrimTypeNode`.
#[repr(C)]
#[derive(Object)]
#[type_key = "ir.PrimType"]
#[type_final]
pub struct PrimTypeObj {
    base: TypeObj,
    pub dtype: DLDataType,
}

/// Reference-counted handle to a primitive TVM type.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct PrimType {
    data: ObjectArc<PrimTypeObj>,
}

impl std::ops::Deref for PrimType {
    type Target = PrimTypeObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl std::ops::Deref for PrimTypeObj {
    type Target = TypeObj;

    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

/// ABI-complete Rust representation of TVM's `TupleTypeNode`.
#[repr(C)]
#[derive(Object)]
#[type_key = "ir.TupleType"]
#[type_final]
pub struct TupleTypeObj {
    base: TypeObj,
    pub fields: Array<Type>,
}

/// Reference-counted handle to a tuple type.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct TupleType {
    data: ObjectArc<TupleTypeObj>,
}

impl std::ops::Deref for TupleType {
    type Target = TupleTypeObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl std::ops::Deref for TupleTypeObj {
    type Target = TypeObj;

    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

impl TupleType {
    /// Construct a tuple type directly in Rust.
    pub fn new(fields: Vec<Type>) -> Self {
        Self::with_span(fields, None)
    }

    /// Construct a tuple type with optional source metadata.
    pub fn with_span(fields: Vec<Type>, span: Option<&Span>) -> Self {
        Self::from_complete_fields(span.cloned(), Array::new(fields))
    }

    /// Construct a tuple type from every physical field.
    pub fn from_complete_fields(span: Option<Span>, fields: Array<Type>) -> Self {
        Self {
            data: ObjectArc::new(TupleTypeObj {
                base: TypeObj::new(span),
                fields,
            }),
        }
    }

    /// Construct the empty tuple type used as TVM's void type.
    pub fn empty() -> Self {
        Self::new(Vec::new())
    }
}

/// ABI-complete Rust representation of TVM's `IntImmNode`.
#[repr(C)]
#[derive(Object)]
#[type_key = "ir.IntImm"]
#[type_final]
pub struct IntImmObj {
    base: ExprObj,
    pub value: i64,
}

/// Reference-counted handle to an integer literal.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct IntImm {
    data: ObjectArc<IntImmObj>,
}

impl std::ops::Deref for IntImm {
    type Target = IntImmObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl std::ops::Deref for IntImmObj {
    type Target = ExprObj;

    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

/// ABI-complete fieldless base for TVM attribute objects.
#[repr(C)]
#[derive(Object)]
#[type_key = "ir.Attrs"]
pub struct AttrsObj {
    base: tvm_ffi::Object,
}

/// Reference-counted handle to any TVM attributes object.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct Attrs {
    data: ObjectArc<AttrsObj>,
}

impl std::ops::Deref for Attrs {
    type Target = AttrsObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

/// ABI-complete representation of TVM's dictionary-backed attributes.
#[repr(C)]
#[derive(Object)]
#[type_key = "ir.DictAttrs"]
#[type_final]
pub struct DictAttrsObj {
    base: AttrsObj,
    pub dict: Map<String, Any>,
}

/// Reference-counted handle to dictionary-backed TVM attributes.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct DictAttrs {
    data: ObjectArc<DictAttrsObj>,
}

impl std::ops::Deref for DictAttrs {
    type Target = DictAttrsObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl std::ops::Deref for DictAttrsObj {
    type Target = AttrsObj;

    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

/// ABI-complete fieldless base for module-level global metadata.
#[repr(C)]
#[derive(Object)]
#[type_key = "ir.GlobalInfo"]
pub struct GlobalInfoObj {
    base: tvm_ffi::Object,
}

/// Reference-counted handle to module-level global metadata.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct GlobalInfo {
    data: ObjectArc<GlobalInfoObj>,
}

impl std::ops::Deref for GlobalInfo {
    type Target = GlobalInfoObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

/// ABI-complete representation of TVM's fieldless test global-info object.
#[repr(C)]
#[derive(Object)]
#[type_key = "ir.DummyGlobalInfo"]
#[type_final]
pub struct DummyGlobalInfoObj {
    base: GlobalInfoObj,
}

/// Reference-counted handle to a dummy global-info value.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct DummyGlobalInfo {
    data: ObjectArc<DummyGlobalInfoObj>,
}

impl std::ops::Deref for DummyGlobalInfo {
    type Target = DummyGlobalInfoObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl std::ops::Deref for DummyGlobalInfoObj {
    type Target = GlobalInfoObj;

    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

impl DummyGlobalInfo {
    /// Construct TVM's fieldless global-info test value directly in Rust.
    pub fn new() -> Self {
        Self::from_complete_fields()
    }

    /// Construct the fieldless value from its complete physical state.
    pub fn from_complete_fields() -> Self {
        Self {
            data: ObjectArc::new(DummyGlobalInfoObj {
                base: GlobalInfoObj {
                    base: tvm_ffi::Object::new(),
                },
            }),
        }
    }
}

impl Default for DummyGlobalInfo {
    fn default() -> Self {
        Self::new()
    }
}

tvm_ffi::impl_object_upcast!(DummyGlobalInfo => GlobalInfo);

/// ABI-complete Rust representation of TVM's `IRModuleNode`.
#[repr(C)]
#[derive(Object)]
#[type_key = "ir.IRModule"]
#[type_final]
pub struct IRModuleObj {
    base: tvm_ffi::Object,
    pub functions: Map<GlobalVar, BaseFunc>,
    pub source_map: SourceMap,
    pub attrs: DictAttrs,
    pub global_infos: Map<String, Array<GlobalInfo>>,
    pub global_var_map: Map<String, GlobalVar>,
}

/// Reference-counted handle to a TVM IRModule.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct IRModule {
    data: ObjectArc<IRModuleObj>,
}

impl std::ops::Deref for IRModule {
    type Target = IRModuleObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl IntImm {
    /// Construct an integer literal directly in Rust.
    pub fn new(dtype: &str, value: i64) -> Result<Self> {
        Self::from_dtype(DLDataType::try_from_str(dtype)?, value)
    }

    /// Construct an integer literal from a parsed DLPack dtype.
    pub fn from_dtype(dtype: DLDataType, value: i64) -> Result<Self> {
        Self::from_dtype_with_span(dtype, value, None)
    }

    /// Construct an integer literal directly in Rust with source metadata.
    pub fn from_dtype_with_span(
        dtype: DLDataType,
        value: i64,
        span: Option<&Span>,
    ) -> Result<Self> {
        validate_integer_literal(dtype, value)?;
        let value_type = PrimType::from_dtype(dtype)?;
        Ok(Self::from_complete_fields(span.cloned(), value_type, value))
    }

    /// Construct an integer literal from every physical field after external validation.
    pub fn from_complete_fields(span: Option<Span>, ty: PrimType, value: i64) -> Self {
        Self {
            data: ObjectArc::new(IntImmObj {
                base: ExprObj::new(span, ty.into()),
                value,
            }),
        }
    }
}

fn validate_integer_literal(dtype: DLDataType, value: i64) -> Result<()> {
    let dtype_text = dtype.to_string();
    if dtype.lanes != 1 {
        return Err(Error::new(
            VALUE_ERROR,
            &format!("IntImm can only take a scalar, but {dtype_text} was supplied"),
            "",
        ));
    }
    let is_int = dtype.code == DLDataTypeCode::kDLInt as u8;
    let is_uint = dtype.code == DLDataTypeCode::kDLUInt as u8;
    let is_bool = dtype.code == DLDataTypeCode::kDLBool as u8;
    if !is_int && !is_uint && !is_bool {
        return Err(Error::new(
            VALUE_ERROR,
            &format!("IntImm supports only int, uint, or bool, but {dtype_text} was supplied"),
            "",
        ));
    }
    let bits = u32::from(dtype.bits);
    if bits == 0 {
        return Err(Error::new(
            VALUE_ERROR,
            &format!("invalid integer bit width in {dtype_text}"),
            "",
        ));
    }
    let in_range = if is_uint {
        value >= 0 && (bits >= 64 || (value as u64) < (1_u64 << bits))
    } else if is_bool || bits == 1 {
        value == 0 || value == 1
    } else if bits >= 64 {
        true
    } else {
        let bound = 1_i64 << (bits - 1);
        value >= -bound && value < bound
    };
    if in_range {
        Ok(())
    } else {
        Err(Error::new(
            VALUE_ERROR,
            &format!("literal value {value} is outside the range of {dtype_text}"),
            "",
        ))
    }
}

impl PrimType {
    /// Construct a primitive type from a DLPack dtype string.
    pub fn new(dtype: &str) -> Result<Self> {
        Self::from_dtype(DLDataType::try_from_str(dtype)?)
    }

    /// Construct a primitive type from a parsed DLPack dtype directly in Rust.
    pub fn from_dtype(dtype: DLDataType) -> Result<Self> {
        Self::from_dtype_with_span(dtype, None)
    }

    /// Construct a primitive type from all of its physical fields.
    pub fn from_dtype_with_span(dtype: DLDataType, span: Option<&Span>) -> Result<Self> {
        let is_opaque_handle = dtype.code == DLDataTypeCode::kDLOpaqueHandle as u8;
        let is_void = is_opaque_handle && dtype.bits == 0 && dtype.lanes == 0;
        if is_opaque_handle && !is_void {
            return Err(Error::new(
                TYPE_ERROR,
                "PrimType cannot represent an opaque pointer; use a pointer type",
                "",
            ));
        }
        Ok(Self::from_complete_fields(span.cloned(), dtype))
    }

    /// Construct a primitive type from every physical field after external validation.
    pub fn from_complete_fields(span: Option<Span>, dtype: DLDataType) -> Self {
        Self {
            data: ObjectArc::new(PrimTypeObj {
                base: TypeObj::new(span),
                dtype,
            }),
        }
    }

    /// Construct TVM's void primitive type without parsing a user-supplied dtype string.
    pub fn void() -> Self {
        Self::from_complete_fields(
            None,
            DLDataType {
                code: DLDataTypeCode::kDLOpaqueHandle as u8,
                bits: 0,
                lanes: 0,
            },
        )
    }
}

impl Type {
    /// Construct TVM's language-independent sentinel for an unavailable static type.
    pub fn missing() -> Self {
        tvm_ffi::cached_global_func!("ir.TypeMissing")
            .call_tuple(())
            .expect("native missing-type constructor failed")
            .try_into()
            .expect("native missing-type constructor returned the wrong type")
    }

    /// Return whether this value is the exact `ir.Type` missing-type sentinel.
    pub fn is_missing(&self) -> bool {
        AnyView::from(self).type_index() == <TypeObj as tvm_ffi::ObjectCore>::type_index()
    }
}

impl Var {
    /// Construct a variable directly in Rust with an explicit primitive type.
    pub fn new(name: &str, dtype: &str) -> Result<Self> {
        Ok(Self::with_type(name, PrimType::new(dtype)?))
    }

    /// Construct a variable with an arbitrary TVM type annotation.
    pub fn with_type<T>(name: &str, ty: T) -> Self
    where
        T: Into<Type>,
    {
        Self::with_type_and_span(name, ty, None)
    }

    /// Construct a variable directly in Rust with its complete base fields.
    pub fn with_type_and_span<T>(name: &str, ty: T, span: Option<&Span>) -> Self
    where
        T: Into<Type>,
    {
        Self::with_optional_type_and_span(name, Some(ty.into()), span)
    }

    /// Construct a variable with the native constructor's optional type annotation.
    pub fn with_optional_type_and_span(name: &str, ty: Option<Type>, span: Option<&Span>) -> Self {
        Self::from_complete_fields(
            span.cloned(),
            ty.unwrap_or_else(Type::missing),
            String::from(name),
        )
    }

    /// Construct a variable from every physical field.
    pub fn from_complete_fields(span: Option<Span>, ty: Type, name: String) -> Self {
        Self {
            data: ObjectArc::new(VarObj {
                base: ExprObj::new(span, ty),
                name,
            }),
        }
    }
}

impl GlobalVar {
    /// Construct a module-level symbol directly in Rust.
    pub fn new(name_hint: &str) -> Self {
        Self::with_span(name_hint, None)
    }

    /// Construct a module-level symbol with optional source metadata.
    pub fn with_span(name_hint: &str, span: Option<&Span>) -> Self {
        Self::from_complete_fields(span.cloned(), Type::missing(), String::from(name_hint))
    }

    /// Construct a module-level symbol from all physical fields without re-deriving its type.
    pub fn from_complete_fields(span: Option<Span>, ty: Type, name_hint: String) -> Self {
        Self {
            data: ObjectArc::new(GlobalVarObj {
                base: ExprObj::new(span, ty),
                name_hint,
            }),
        }
    }
}

impl Call {
    /// Construct a call directly in Rust with no attributes or explicit type arguments.
    pub fn new<T, O>(ret_type: T, operator: O, arguments: Vec<Expr>) -> Self
    where
        T: Into<Type>,
        O: Into<Expr>,
    {
        Self::with_metadata(ret_type, operator, arguments, None, Vec::new(), None)
    }

    /// Construct a call with all reflected metadata supplied explicitly.
    pub fn with_metadata<T, O>(
        ret_type: T,
        operator: O,
        arguments: Vec<Expr>,
        attrs: Option<Attrs>,
        type_arguments: Vec<Type>,
        span: Option<&Span>,
    ) -> Self
    where
        T: Into<Type>,
        O: Into<Expr>,
    {
        Self::from_complete_fields(
            span.cloned(),
            ret_type.into(),
            operator.into(),
            Array::new(arguments),
            attrs,
            Array::new(type_arguments),
        )
    }

    /// Construct a call from every physical field.
    pub fn from_complete_fields(
        span: Option<Span>,
        ty: Type,
        op: Expr,
        args: Array<Expr>,
        attrs: Option<Attrs>,
        ty_args: Array<Type>,
    ) -> Self {
        Self {
            data: ObjectArc::new(CallObj {
                base: ExprObj::new(span, ty),
                op,
                args,
                attrs,
                ty_args,
            }),
        }
    }
}

tvm_ffi::impl_object_upcast!(
    BaseFunc => Expr,
    IntImm => Expr,
    IntImm => PrimExpr,
    OpaqueExpr => Expr,
    OpaqueType => Type,
    PointerType => Type,
    PrimType => Type,
    TupleType => Type,
    Var => Expr,
    GlobalVar => Expr,
    Call => Expr,
    DictAttrs => Attrs,
);

impl IRModule {
    /// Wrap a function expression in an IRModule whose entry is `main`.
    pub fn from_expr<E>(expr: E) -> Result<Self>
    where
        E: Into<Expr>,
    {
        let function = expr.into().try_cast::<BaseFunc>()?;
        let global_symbol = function
            .attrs
            .dict
            .get(&String::from("global_symbol"))?
            .map(String::try_from)
            .transpose()?;
        let global_name = global_symbol
            .as_deref()
            .filter(|name| !name.is_empty())
            .unwrap_or("main");
        let global_var = GlobalVar::new(global_name);
        Self::with_metadata(
            [(global_var, function)].into_iter().collect(),
            SourceMap::new(),
            DictAttrs::empty(),
            Map::new(),
        )
    }

    /// Construct a module directly in Rust from all of its stored state.
    ///
    /// The derived name-to-global-variable index is rebuilt and checked here;
    /// callers never supply it independently.
    pub fn with_metadata(
        functions: Map<GlobalVar, BaseFunc>,
        source_map: SourceMap,
        attrs: DictAttrs,
        global_infos: Map<String, Array<GlobalInfo>>,
    ) -> Result<Self> {
        let mut indexed_globals = Vec::with_capacity(functions.len());
        let mut names = std::collections::HashSet::with_capacity(functions.len());
        for (global_var, _) in functions.iter() {
            let name = global_var.name_hint.clone();
            if !names.insert(name.as_str().to_owned()) {
                return Err(Error::new(
                    VALUE_ERROR,
                    &format!("duplicate global function name {}", name.as_str()),
                    "",
                ));
            }
            indexed_globals.push((name, global_var));
        }
        Ok(Self::from_complete_fields(
            functions,
            source_map,
            attrs,
            global_infos,
            indexed_globals.into_iter().collect(),
        ))
    }

    /// Allocate a module from every physical field without rebuilding derived indexes.
    pub fn from_complete_fields(
        functions: Map<GlobalVar, BaseFunc>,
        source_map: SourceMap,
        attrs: DictAttrs,
        global_infos: Map<String, Array<GlobalInfo>>,
        global_var_map: Map<String, GlobalVar>,
    ) -> Self {
        Self {
            data: ObjectArc::new(IRModuleObj {
                base: tvm_ffi::Object::new(),
                functions,
                source_map,
                attrs,
                global_infos,
                global_var_map,
            }),
        }
    }

    /// Return an independently updatable module with one function replaced.
    ///
    /// Rust rebuilds both immutable maps and the module node, so other handles
    /// that share `self` remain unchanged and the derived global-name index
    /// cannot become stale.
    pub fn with_updated_function(
        &self,
        global_var: &GlobalVar,
        function: &BaseFunc,
    ) -> Result<Self> {
        self.copy_for_update()?
            .update_function_owned(global_var, function)
    }

    /// Return an independently updatable module with one global-info group set.
    pub fn with_updated_global_info(
        &self,
        name: &str,
        global_info: Vec<GlobalInfo>,
    ) -> Result<Self> {
        let name = String::from(name);
        let mut global_infos = self
            .global_infos
            .iter()
            .filter(|(existing, _)| existing.as_str() != name.as_str())
            .collect::<Vec<_>>();
        global_infos.push((name, Array::new(global_info)));
        Self::with_metadata(
            self.functions.clone(),
            self.source_map.clone(),
            self.attrs.clone(),
            global_infos.into_iter().collect(),
        )
    }

    pub(crate) fn copy_for_update(&self) -> Result<Self> {
        Self::with_metadata(
            self.functions.clone(),
            self.source_map.clone(),
            self.attrs.clone(),
            self.global_infos.clone(),
        )
    }

    pub(crate) fn update_function_owned(
        self,
        global_var: &GlobalVar,
        function: &BaseFunc,
    ) -> Result<Self> {
        let name = global_var.name_hint.clone();
        let mut functions = Vec::with_capacity(self.functions.len() + 1);
        for (existing_var, existing_function) in self.functions.iter() {
            if existing_var.name_hint.as_str() == name.as_str() {
                if !existing_var.same_as(global_var) {
                    return Err(Error::new(
                        VALUE_ERROR,
                        &format!("duplicate global function name {}", name.as_str()),
                        "",
                    ));
                }
            } else {
                functions.push((existing_var, existing_function));
            }
        }
        functions.push((global_var.clone(), function.clone()));
        Self::with_metadata(
            functions.into_iter().collect(),
            self.source_map.clone(),
            self.attrs.clone(),
            self.global_infos.clone(),
        )
    }
}

impl DictAttrs {
    /// Construct a defined, empty DictAttrs object.
    pub fn empty() -> Self {
        Self::from_dictionary(Map::new())
    }

    /// Construct DictAttrs from a heterogeneous string-to-value map.
    pub fn from_dictionary(dictionary: Map<String, Any>) -> Self {
        Self::from_complete_fields(dictionary)
    }

    /// Construct dictionary attributes from their complete physical state.
    pub fn from_complete_fields(dict: Map<String, Any>) -> Self {
        Self {
            data: ObjectArc::new(DictAttrsObj {
                base: AttrsObj {
                    base: tvm_ffi::Object::new(),
                },
                dict,
            }),
        }
    }
}

impl Default for SourceMap {
    fn default() -> Self {
        Self::new()
    }
}
