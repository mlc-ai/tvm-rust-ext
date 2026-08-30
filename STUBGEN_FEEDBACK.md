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

# Feedback for Rust TVM stubgen

This crate handwrites a representative Rust IR surface, then runs real
`structural_walk`, `structural_map`, pass integration, and C++ differential
tests. Its purpose is to establish what a one-command, Rust-native stubgen must
generate, what belongs in `tvm-ffi`, and which C++ declarations must be changed
before they can be safely constructed by Rust.

## Main correction from the experiment

The first prototype treated every IR node as an opaque header and routed every
constructor through a registered packed function. That is unnecessarily slow
and is not the intended object-ABI model.

For an ordinary C++ data node, Rust can use the same memory layout and allocate
the object directly:

```rust
#[repr(C)]
#[derive(Object)]
#[type_key = "tirx.Add"]
#[type_final]
pub struct AddObj {
    base: ExprObj,
    pub a: Expr,
    pub b: Expr,
}

pub fn from_complete_fields(
    span: Option<Span>,
    ty: Type,
    a: Expr,
    b: Expr,
) -> Add {
    Add {
        data: ObjectArc::new(AddObj {
            base: ExprObj::new(span, ty),
            a,
            b,
        }),
    }
}

pub fn new(a: &Expr, b: &Expr) -> Result<Add> {
    // Validate the same dtype rule as C++, then delegate to the owned path.
    let result_type = matching_binary_type(a, b)?;
    Ok(from_complete_fields(
        None,
        result_type,
        a.clone(),
        b.clone(),
    ))
}
```

Stubgen emits this allocator only after authoritative build-time layout input
lets the generator verify the complete type's parent, size, alignment,
finality, and fields. The generated target code then uses `ObjectArc::new` to
install a common `TVMFFIObject` header and a Rust deleter without repeating the
generator's validation as a per-type Rust macro.
C++ sees the normal runtime type index and field offsets, while destruction
returns through the deleter stored in the header. This was verified by invoking
a C++ reflection getter on a Rust-created `Add`, running C++ structural
traversal on it, and comparing it with a C++-created `Add` through C++
structural equality.

## What stubgen should generate

The public construction syntax for an ordinary generated class should be a
single Rust call such as `IntPair::new(a, b, scale)`.  Stubgen should not emit
an `ffi_new().a(a).b(b).build()` chain: the field list and defaults are already
known while generating the binding, so a builder only adds types and call-site
noise.  When a TVM type has a reviewed semantic constructor, that constructor
still owns the public `new(...)` name and delegates to a mechanical
`from_complete_fields(...)` allocator after validation and derivation.  The
second name is needed only because Rust cannot overload the semantic and
physical constructor signatures; it is not a builder or a C++ allocation
fallback.

For a directly constructible node, generate:

- the complete `#[repr(C)]` inheritance prefix;
- all physical fields in exact C++ order, including unreflected fields when
  their layout is part of the supported ABI;
- exact-width scalar types (`int` is `i32`, not `i64`);
- open integer newtypes for native enum fields, with named associated constants
  and raw conversion; a Rust `enum` is unsound when a newer C++ library can
  produce an enumerator unknown to the generated crate; conversions from a
  wider integer check only that the value fits the native width and must not
  reject an otherwise representable unknown enumerator;
- the reference wrapper with `ObjectArc`, read-only `Deref`, casts, and
  upcasts;
- public physical fields, so `node.a` borrows through `Deref` and
  `node.a.clone()` is an explicit ownership decision by the caller;
- no cloning getter boilerplate and no second reflection-backed field path;
- no unconditional `DerefMut`, because two reference wrappers may share one
  allocation; mutation belongs in structural mutation or a checked COW API;
- no Rust allocator for a polymorphic base or concrete subclass: reflected
  methods may provide Rust access, but they do not reproduce a compiler-owned
  vptr or authorize a C++ ABI migration;
- a direct Rust constructor only when its full validation/default logic is
  known and reproduced;
- one lossless complete-field allocation path for every ABI-complete concrete
  node: call it `new(...)` for an ordinary generated class, or
  `from_complete_fields(...)` when the type already has a reviewed semantic
  `new(...)`; names such as `with_span` and `with_metadata` are semantic
  convenience APIs, never substitutes for this uniform generated entry;
- no generated allocator when raw fields encode an external identity invariant;
  for example, `Axis` exposes `get(name)` through TVM's existing registry lookup
  instead of creating a second object with a copied registry index;
- complete-construction parameters use the exact stored Rust field types and
  take them by value (`String`, `Array<T>`, `Option<T>`, and object handles),
  so moving a field performs no hidden clone, conversion, or allocation;
- a complete constructor variant that accepts optional metadata such as
  `Span`, with convenience constructors delegating to it;
- direct `Self` returns for complete-field allocation and other infallible
  Rust-only constructors; use `Result<Self>` only when generated code performs
  parsing, checked conversion, validation, casting, or a fallible ABI call;
- finality and structural metadata used by typed walk/map dispatch.
- direct `ObjectArc::new` allocation only for concrete types that stubgen has
  classified as complete and constructible; opaque wrappers and layout-only
  base prefixes receive no generated allocator.

The generator cannot recover complete physical layout from reflection alone.
It needs a layout source derived from C++ declarations or an explicitly curated
schema. Reflection is still the authority for field schemas, flags, and the
language-independent structural protocol.

## One-invocation input and output contract

One-command generation must consume a merged manifest with three independently
auditable sources.  No source is allowed to fill in another source's missing
facts by guessing:

| Manifest section | Required facts | Authority |
| --- | --- | --- |
| Runtime type | type key, parent chain, reflected field names/schemas/flags, structural attributes | TVM FFI registry |
| Native layout | finality, total size/alignment, ordered physical fields, exact C++ widths/offsets, and explicit vptr/STL/unreflected blockers | C++ build-generated layout data |
| Rust mapping | object/reference names, module path, exact Rust type for every physical field, nullable/container mapping, and upcasts | reviewed mapping rules |

For each type the generator first joins these sections by type key, verifies
that the parent layout and every reflected field agree, and chooses exactly one
outcome:

1. **complete:** emit the `#[repr(C)]` object, reference wrapper, read-only
   `Deref`, casts, and one owned complete-field constructor after validating
   the layout inside stubgen;
2. **complete plus semantic:** additionally emit generated convenience
   constructors whose semantics are fully described by authoritative metadata;
   semantic constructors outside that set remain reviewed handwritten Rust; or
3. **blocked:** emit only a safe opaque reference wrapper and a diagnostic that
   names the missing layout fact or ABI blocker.

The generated complete-field signature is mechanical: flatten the stored
fields from the rootmost supported base to the concrete node and keep each
class's native declaration order. Parameter names and Rust types match the
generated Rust fields after deterministic identifier sanitization (`span, ty,
a, b` for `Add`; native `global_var_map_` becomes Rust `global_var_map`), and
each parameter moves exactly once into the object. It contains no `clone`,
`String::from`, `Array::new`, packed call, validation, or `Result`. Conversions
and semantic argument order belong in the reviewed semantic constructor.

Field nullability belongs to the field/constructor contract, not to the C++
reference wrapper in isolation.  C++ commonly gives an `ObjectRef` type a
default undefined state, while nodes still require a defined value in a field
of that type.  An explicit `ffi::Optional<T>` schema maps to Rust `Option<T>`;
a plain object field stays non-optional and its semantic constructor must
enforce any required-value invariant.

The invocation is successful only if every requested type reaches one of those
three explicit outcomes, generated files are deterministic, and the emitted
coverage manifest records the source and status of every field, constructor,
enum, behavior method, and blocker. An omitted type or silently generated
packed-constructor fallback is a generation error.

The generated crate is tied to the native-layout manifest consumed by its
stubgen invocation and is built and tested against that TVM build. If a future
distribution supports loading arbitrary TVM shared-library versions, it should
add one centralized ABI version gate rather than emit a validator beside every
generated object.

## Integration with the existing stubgen

This should extend `tvm-ffi-stubgen`, not create a second TVM-only generator.
The existing pipeline already:

- loads the requested shared libraries from `--dlls`;
- collects registered type keys and global functions;
- constructs language-neutral `ObjectInfo`/`FuncInfo` values;
- topologically sorts objects by runtime inheritance; and
- delegates rendering to a pluggable language `Generator`.

Today only `PythonGenerator` is registered and the CLI restricts `--target` to
`python`. The Rust implementation should register `RustGenerator`, add `.rs`
marker/scaffolding support, and remove that hard-coded target restriction. The
language-neutral object model must be extended rather than bypassed: current
`ObjectInfo` keeps field schemas but drops offsets, sizes, alignments, flags,
defaults, and total size that are already available on runtime `TypeInfo`.
Rust generation needs those facts plus the independently generated native
layout/finality/blocker section described above.

The intended command shape is therefore one existing stubgen invocation, for
example:

```text
tvm-ffi-stubgen rust/src --dlls build/libtvm_compiler.so --target rust \
  --native-layout build/rust_native_layout.json
```

The exact option names can follow the upstream CLI review, but the ownership is
fixed: runtime registry collection remains shared, native layout is merged
before rendering, and `RustGenerator` only emits code. It must not rediscover
C++ layout or guess constructor semantics while formatting Rust.

## Constructor classification

Direct Rust allocation is the target for ordinary ABI-complete data nodes.
Classification also identifies native semantic blockers that must reuse an
existing TVM operation:

| Class | Examples | Generated behavior |
| --- | --- | --- |
| Plain data node | `Span`, `Range`, `Var`, `IntImm`, `Add`, `Evaluate`, `SBlock` | complete layout and direct Rust allocation |
| Plain node with local validation | integer literals, binary ops, `SeqStmt`, `SBlockRealize` | direct allocation plus equivalent Rust validation |
| Native registry identity | `Axis` | emit an opaque wrapper and call the existing `tirx.AxisGet` singleton lookup |
| Native interned identity | `SourceName` | emit an opaque wrapper and call the existing `ir.SourceName` lookup |
| C++ polymorphic hierarchy | `Layout`, `PrimExprConvertible`, `IterVar`, `BufferRegion` | preserve the virtual ABI, emit opaque Rust wrappers, and allocate concrete objects through existing native constructors |
| Native STL storage | `Source` | keep the node opaque and construct it through the existing `SourceMapAdd` operation |
| Complex semantic constructor | `PrimFunc`, match buffer | use reviewed handwritten Rust analysis/validation, then allocate complete fields in Rust |
| Build-dependent defaults | `BufferType` | use reviewed handwritten Rust defaults and validation, then allocate in Rust |
| Derived mutable state | `IRModule` | rebuild and validate derived indexes in generated Rust code |

`PrimType`, `TupleType`, `For`, `BufferType`, buffer load/store, and `IRModule`
demonstrate that many former exceptions can be Rust-native once
their semantics are made explicit. `Type::Missing` deliberately remains the
native singleton.

## Language-neutral behavior and handwritten constructors

Ordinary ABI-complete constructors do not call packed global constructors or
maintain custom raw vtables. Registry, interning, STL, and polymorphic blockers
reuse existing native operations. Constructor-specific behavior that stubgen
cannot derive is written directly in Rust and checked against the C++ constructor
with differential tests. It may call an existing language-neutral analysis or
runtime service, but it does not add a new constructor-only method or metadata
protocol. Rust still performs final allocation for ABI-complete objects.

Other opaque behavior uses APIs that TVM already registers. `tirx.convert`
performs `PrimExprConvertible` fallback conversion, and the existing
`tirx.Layout*` global functions retain C++ virtual dispatch for native layouts.
The Rust binding must not duplicate either surface in every concrete type's
method table.

The tests exercise both directions: Rust calls existing registered native
operations where needed, while C++ entry points inspect and transform Rust-created
ordinary nodes through the same tvm-ffi object ABI. Identity- and
resource-owning blockers are separately checked to remain native objects.

Calls that execute passes, structural protocols, and container runtime
operations remain cross-language services. They are not hidden implementations
of generated IR constructors.

Adding reflected type methods is not a C++ ABI migration. Existing virtual
interfaces remain intact for native and out-of-tree subclasses; the reflected
entry is an additional language-neutral route for Rust. Stubgen must keep such
types opaque until a separate, explicitly reviewed ABI migration removes their
vptr.

## Runtime support learned from the experiment

### Heterogeneous containers

TVM fields include both heterogeneous arrays and maps, such as schedule values,
`DictAttrs::__dict__`, loop annotations, and block annotations. A blanket
`AnyCompatible for Any` conflicts with Rust's identity conversion rules. The
runtime therefore provides one sealed container-element conversion contract,
shared by `Array<T>` and both sides of `Map<K, V>`. Generated bindings can use
`Array<Any>`, `Map<String, Any>`, or `Map<Any, T>` directly without defining a
TVM-specific container. This support belongs in `tvm-ffi`, not generated IR.

### Object origins and deletion

The same Rust reference wrapper must accept both origins:

- a Rust allocation has a Rust object deleter in its header;
- a C++ allocation has a C++ object deleter;
- refcount operations always dispatch through the common header.

Generated safe direct construction must be disabled for incomplete layouts.
Creating a header-only Rust allocation and labeling it as a larger native type
is unsound even if field access is done through reflection. A native virtual
base and its concrete subclasses remain opaque even when registered functions
make their behavior callable; that does not make the compiler-owned object
layout constructible in Rust.

### Consuming packed arguments

C++ pass APIs use `RValueRef<T>` at some boundaries. `tvm-ffi::RValueRef<T>` is
the reusable owning packed-argument holder for the same ABI representation. A
matching callee steals its object slot without an extra reference-count
increment; an ordinary lvalue remains supported through the C++-compatible
copy path. Generated pass wrappers should use this standard holder.

## Metadata gaps

The current experiment still needs explicit decisions for:

- obtaining authoritative build-time layouts and explicit opaque blockers while
  deciding what Rust code to emit;
- constructor parameter type schemas, nullability/defaults, and validation
  semantics that should eventually be generated rather than handwritten;
- build-configuration values used by constructor defaults;
- enum names and values, including their underlying C++ width; generated
  bindings represent these as open integer newtypes rather than closed Rust
  enums;
- `RValueRef<T>` code generation (the reusable runtime holder already exists).

These gaps require metadata, reviewed handwritten implementations, or an ABI
refactor. They are not permission for stubgen to guess or to hide allocation
behind a packed global.

## Layout evidence

Stubgen must validate physical layouts from authoritative build-time input;
runtime reflection metadata does not certify that every physical field is
reflected. The accepted target code contains only the resulting `#[repr(C)]`
object and allocator, not a second copy of the generator's layout validator.

## Structural-pass lessons

The IR binding and the pass algorithm are separate concerns:

- `structural_walk`/`structural_map` control recursion through pre/post order;
- `structural_visit`/`structural_mutate` let callbacks explicitly recurse;
- maps preserve keys and recursively transform values;
- FreeVar and DAG identity semantics come from registered structural metadata;
- `IRModule` cannot be rebuilt by blindly mapping reflected storage because
  `global_var_map_` is a derived index;
- a Rust pass claiming C++ parity must be checked with C++ structural equality,
  not only a few field assertions.

Three additional pass probes separate traversal support from compiler-semantic
support. Checked integer folding is straightforward with post-order mapping,
but fixed-width overflow, casts, and symbolic reasoning should reuse TVM's
arithmetic analyzer. The prototype therefore binds `arith.Analyzer` as an
opaque FFI object and calls its existing registered operations; stubgen must be
able to distinguish an opaque compiler service like this from an ABI-complete,
Rust-allocated IR node. Control-flow removal can now recognize values
simplified by that analyzer. The Rust side-effect classifier walks the bound IR
and queries the existing `ir.OpGetAttr` service for each operator's
`TCallEffectKind`, allowing Rust to discard pure evaluations while preserving
opaque or state-updating calls. Function reachability can combine a walk-built
call graph, `global_symbol` linkage roots, and an `IRModule` rebuild, but full
dead-code elimination still needs exact callee and effect semantics. These are
reusable compiler services, not facts stubgen can derive from object layout or
structural metadata. Stubgen should generate typed wrappers only after such a
service has a language-neutral entry.

The example Rust passes remain prototype evidence. Stubgen should generate the
IR surface they consume, not generate those transformations.

## Recommended implementation order

1. Add a reliable C++-layout input and generate the minimal complete structs.
2. Generate a lossless complete-field Rust allocator only for ABI-complete
   nodes whose constructor classification permits Rust ownership.
3. Generate mechanical semantic constructors first for `PrimType`, `Var`,
   `IntImm`, `Add`, `Evaluate`, `For`, buffers, tuples, and modules; keep
   non-mechanical constructor logic in reviewed Rust and emit native-operation
   wrappers for `Type::Missing`, `Axis`, `SourceName`, and `Source`.
4. Run `tests/stubgen_acceptance.rs` unchanged, including the C++-getter-on-Rust
   allocation test.
5. Generate heterogeneous fields as `Map<K, Any>` and other container fields
   with the normal typed `Array`/`Map` wrappers.
6. Reuse existing registered operations for opaque behavior and compiler
   services, and reject unrefactored vptr/STL-backed nodes.
7. Generate pass boundaries with the standard `RValueRef<T>` holder.
8. Expand the type slice only after each new layout/constructor pattern has a
   focused test and a named owner.

The goal is not to preserve handwritten bindings. The goal is for one stubgen
invocation to replace them with Rust structs and Rust allocation code while
retaining the same cross-language ABI and pass behavior. Packed calls may still
run compiler services and passes; they must not be the hidden implementation of
ordinary generated IR constructors. The handwritten slice demonstrates the
intended Rust surface and cross-language behavior. It does not yet supply every
one-command generator input listed in the metadata gaps; the next stubgen step
is to emit the same mechanical code from an authoritative native-layout
manifest while retaining reviewed handwritten semantics where needed.
