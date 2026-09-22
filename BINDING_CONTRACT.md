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

# Rust TVM IR binding contract

This document defines when a handwritten or generated Rust IR binding is
correct. The prototype is executable evidence for stubgen; the C++ declaration
and the shared TVM FFI ABI remain the sources of truth.

## Core rule

TVM FFI objects are not inherently C++-owned. A language may allocate an
object when it can reproduce the complete object layout and constructor
semantics. For an ordinary data node, generated Rust should therefore:

1. emit the complete inheritance prefix and every physical field in C++ order;
2. use ABI-equivalent Rust field types under `#[repr(C)]`;
3. let stubgen verify the generated inheritance, size, alignment, finality,
   and field offsets against authoritative build-time layout input before
   emitting direct `ObjectArc::new` allocation;
4. initialize the same defaults and validate the same invariants as C++; and
5. expose immutable data fields through the reference wrapper's `Deref`, so
   reading borrows and callers explicitly `clone()` only when they need an
   owning handle. Fields native code can mutate through shared handles require
   private interior-mutable storage and accessors that return owned snapshots.

This lets Rust construct `AddObj { a, b, ... }` without a packed global call
while C++ reflection, structural traversal, reference counting, and destruction
continue to work.

A packed C++ constructor is not an acceptable final fallback for generated IR
bindings. Stubgen must instead distinguish two generated APIs:

- a uniform lossless complete-field path that allocates every ABI-complete
  concrete object from all of its physical state, named `new(...)` for an
  ordinary class and `from_complete_fields(...)` when a reviewed semantic
  constructor already owns `new(...)`; this path stays internal when its fields
  contain registry identity or another invariant callers cannot validate
  locally; and
- a semantic convenience constructor whose defaults, validation,
  normalization, and derived fields are either generated from authoritative
  metadata or supplied as reviewed handwritten Rust code.

If the physical layout is unavailable, stubgen must omit direct allocation and
report a named blocker. If semantic constructor logic cannot be generated, its
Rust implementation is maintained manually instead of adding a constructor-only
FFI protocol. It must not silently route `new()` through a registered global.
Interning tables and other shared runtime services may still cross the ABI, but
they are runtime capabilities rather than per-node C++ allocation.

## Sources of truth

Use these sources together:

1. C++ declarations establish base layout, field order, and exact scalar widths.
2. C++ constructor bodies establish defaults, validation, normalization,
   interning, and derived state.
3. FFI type/reflection metadata establishes runtime type identity, inheritance,
   field schemas, structural flags, and language-independent traversal.
4. `tvm-ffi` establishes object allocation, ownership, casting, container, and
   packed-call rules.
5. Differential tests establish parity for Rust passes that claim to match a
   C++ pass.

Stubgen must never infer a physical layout from reflected fields alone. A C++
class may contain an unreflected field, an STL member, or a vptr.

## Minimal acceptance slice

The first generated slice remains intentionally small:

| Rust object | Type key | Direct fields | Construction |
| --- | --- | --- | --- |
| `ExprObj` | `ir.Expr` | `span`, `ty` | base prefix |
| `VarObj` | `ir.Var` | `name` | direct Rust allocation |
| `IntImmObj` | `ir.IntImm` | `value` | direct after integer validation |
| `AddObj` | `prim.Add` | `a`, `b` | direct after dtype validation |
| `StmtObj` | `tirx.Stmt` | `span` | base prefix |
| `EvaluateObj` | `tirx.Evaluate` | `value` | direct after value validation |
| `BaseFuncObj` | `ir.BaseFunc` | `attrs` | base prefix |
| `PrimFuncObj` | `tirx.PrimFunc` | `params`, `ret_type`, `body` | Rust type derivation plus direct Rust allocation |

`PrimType` and `TupleType` are constructed directly in Rust. `Type::Missing`
continues to use TVM's existing native singleton because its semantics include
allocation identity, not merely the bytes of `TypeNode`.

## Required checks

A binding is accepted only when every applicable check passes:

| Check | Required evidence |
| --- | --- |
| Runtime identity | exact type key, parent, depth, and finality |
| Physical layout | complete base prefix, field order, size, alignment, and exact scalar widths |
| Generation-time layout compatibility | stubgen accepts the type only when authoritative build-time size, alignment, finality, and field offsets match the generated representation |
| Reflected surface | exact reflected names, schemas, defaults, and structural flags |
| Complete allocator API | exact stored types by value, direct `Self` return, no hidden clone/conversion work, and visibility that preserves external invariants |
| Constructor parity | matching defaults, rejection cases, normalization, and derived state |
| Cross-language ABI | a C++ registered field getter can read a Rust allocation |
| Native behavior boundary | reuse an existing registered FFI operation when one exists; do not add a constructor-only FFI protocol |
| Cross-language semantics | C++ structural equality accepts Rust- and C++-created equivalents |
| Ownership | Rust and C++ may clone/drop the handle and destroy moved-from fields on native errors without leaks, double drops, or dangling fields |
| Walk/map behavior | exact callback selection, order, definition regions, identity remapping, and COW behavior |
| Pass parity | structural equality with the named C++ pass on representative IR |

The focused checks live in `tests/stubgen_acceptance.rs`. It explicitly invokes
a C++ field getter on a Rust-created `Add` and compares that node with a
C++-created `Add` using C++ structural equality. For the ABI-complete object
slice explicitly enumerated there, `tests/binding_contract.rs` checks the exact
reflected schema, flags, and registered default values. Complete-field
allocators are exercised by construction and pass tests; this suite does not
duplicate every constructor signature in a separate compile-only checklist.
Layout completeness and comparison belong to stubgen and its generation tests,
not to runtime reflection registration or each generated Rust object.
Broader pass behavior is in
`tests/structural_passes.rs`.

## Pattern classification

| Pattern | Representative types | Owner/status |
| --- | --- | --- |
| Complete ordinary data layout | `Expr`, `Var`, `IntImm`, `Add`, `Stmt`, `Evaluate`, `Span`, `SequentialSpan`, `FuncType`, `IndexMap`, `TensorIntrin`, `ExecScope`, `ScopeIdDef`, `ScopeIdDefStmt`, `LambdaExpr`, `TilePrimitiveCall` | **GENERATE / verified** |
| Owning object reference and checked casts | all reference wrappers | **GENERATE / verified** |
| Direct scalar/object/optional/array/map fields | `IntImm`, `Call`, `For` | **GENERATE / verified** |
| Heterogeneous `Array<Any>` / `Map<K, Any>` | schedule values, `DictAttrs`, annotations | **RUNTIME / verified via shared container-element support** |
| Direct construction with validation | `IntImm`, binary arithmetic, `SeqStmt` | **GENERATE or reviewed template** |
| Complete layout, build-dependent defaults | `BufferType` | **handwritten Rust semantics + Rust allocation / verified** |
| Native registry identity | `Axis` | **opaque wrapper + existing `tirx.AxisGet` singleton lookup / verified** |
| Native interned identity | `SourceName` | **complete layout + existing `ir.SourceName` lookup, without a Rust allocator / verified** |
| Native polymorphic behavior | `Layout`, `TileLayout`, `ComposeLayout`, `PrimExprConvertible`, `IterVar` | **opaque wrapper + native allocation + reflected Rust access / verified** |
| Typed ordinary expression | `BufferRegion` | **complete `Expr` layout + singleton `BufferRegionType` + Rust allocation / verified** |
| Native STL storage | `Source` | **opaque wrapper + existing `SourceMapAdd` construction / verified** |
| Non-object optional ABI | `TilePrimitiveCall::dispatch` | **complete layout using `tvm_ffi::Optional<String>`; registry category check + Rust allocation / verified** |
| Native mutable fields | `DispatchContext::callbacks`, `DispatchContext::shared_state` | **private ABI-compatible `UnsafeCell` fields + owned snapshots / verified** |
| Complex semantic constructor | `BufferType`, `PrimFunc`, match buffer | **handwritten Rust semantics + complete-field Rust allocation / verified** |
| Derived mutable indexes | `IRModule` construction/update | **GENERATE rebuild logic / verified** |
| Consuming `RValueRef<T>` packed argument | pass boundaries | **RUNTIME / verified without an extra reference-count increment** |
| Pass ports and analyses | `analysis`, `transform/*` | **handwritten consumers; pass ports require C++ differential tests** |

An incomplete type is safe only as a runtime-owned handle. Stubgen must not
expose `ObjectArc::new` for it or pretend that its reflected fields are the
complete physical object. A type blocked by a native vptr remains opaque unless
a separately reviewed C++ ABI migration removes that blocker.

## Important ABI details

- Rust `Option<ObjectRef>` represents a nullable C++ object handle. A field may
  require a defined value at construction yet become null during native mutation:
  `PrimFunc.body` uses private `Option<Stmt>` storage and a borrowing `body()`
  accessor so native error cleanup can destroy its moved-from state. The public
  complete-field constructor still requires `Stmt`, not an incomplete function.
- Do not infer a field's optionality from the referenced C++ `ObjectRef`
  class's `_type_is_nullable` flag alone. Stubgen must combine the declared
  field type with constructor and move/destruction behavior. An explicit
  `ffi::Optional<T>` uses `Option<T>` for object-pointer types and
  `tvm_ffi::Optional<T>` for non-object
  types such as strings and scalars. A plain handle normally stays non-optional.
  A derived constructor may prove an exception: `SequentialSpan` intentionally leaves
  its inherited `SpanNode::source_name` undefined, so the physical Rust base
  field must be `Option<SourceName>` even though ordinary `Span::new` requires
  a source name.
- C++ `int` and `enum class ... : int` use an `i32` representation, not `i64`.
- Native enum fields use a `#[repr(transparent)]` integer newtype with named
  constants that preserve the exact C++ enumerator spelling, not a closed Rust
  `enum`; this keeps unknown values from a newer C++ library representable
  without undefined behavior. A conversion from `i64` checks only native-width
  narrowing; it does not reject an unknown value that fits the underlying
  integer.
- Field order includes inherited physical fields before derived fields.
- Representing inheritance as a nested Rust base is valid only when every
  derived C++ field offset agrees with that `#[repr(C)]` composition. C++ is
  allowed to reuse base tail padding, so matching field types and total size is
  not enough; the native-layout manifest must make any offset mismatch a hard
  blocker.
- Direct field access borrows through `Deref`; it does not change reference
  counts. `node.field.clone()` explicitly clones an owning handle, not the
  object node.
- A lossless complete-field allocator accepts each exact stored field type by
  value, ordered from the rootmost represented base to the concrete node and
  in native declaration order within each class. Its parameter names match the
  generated Rust fields after deterministic identifier sanitization (for
  example, native `global_var_map_` becomes Rust `global_var_map`). It moves
  those values into `ObjectArc::new` without cloning handles, converting
  strings, or rebuilding containers. Convenience constructors may expose a
  semantic argument order, accept borrowed inputs, validate or derive state,
  and explicitly clone while delegating to this owned path.
- An ordinary generated class exposes that allocator as one direct
  `Type::new(a, b, c)` call. It must not require a builder chain. When a
  reviewed semantic constructor already owns `new(...)`, the mechanical path
  is named `from_complete_fields(...)` because Rust has no function
  overloading; both forms still perform the same direct Rust allocation.
- Constructor semantics that are not mechanically derivable remain explicit,
  reviewed Rust code. Differential tests against the C++ constructor detect
  drift in defaults, validation, normalization, and derived fields.
- Physical layout compatibility does not by itself authorize construction.
  Registry identity, interning, sentinels, and
  native resource ownership are constructor semantics. `Axis`, `SourceName`,
  `Source`, and `Type::Missing` therefore use their existing native operations
  and expose no direct Rust allocator. This does not require an opaque layout
  when every physical field is known, as `SourceName` demonstrates.
- Generated object references must not implement unconditional `DerefMut`.
  Handles can share one allocation, so mutation requires structural mutation
  or an explicit uniqueness/COW mechanism.
- A context whose native methods mutate shared fields needs a separate borrowing
  policy. `DispatchContext` stores its two mutable maps in private `UnsafeCell`
  fields, preserving their C++ layout. Snapshot accessors clone the map handles
  without lending references to the mutable slots; native Map COW then preserves
  those snapshots. The context is not `Send` or `Sync`. Its complete constructor
  still accepts native field values and wraps them internally in `UnsafeCell`.
- A polymorphic behavior base remains opaque. Additional reflected methods let
  Rust call behavior without changing the native virtual interface, but do not
  authorize Rust to manufacture a vptr or allocate a concrete subclass.
- Stubgen may emit `ObjectArc::new` only for a concrete type that passed its
  complete-layout and constructor-semantics classification. Layout-only bases,
  polymorphic objects, registry-owned identities, interned objects, and
  STL-backed objects receive no generated allocator even when a subset has a
  complete generated layout.
- A Rust-created node carries a Rust deleter. C++ must release it through the
  header rather than assuming a C++ `delete` expression.
- This target-code demo is generated for the same TVM build that supplies its
  native-layout manifest. If generated crates later support loading arbitrary
  TVM shared-library versions, compatibility should be checked once through a
  centralized ABI version gate rather than beside every generated class.
- A C++-created node keeps its C++ deleter; the same Rust wrapper can reference
  either origin.
- Reflection describes structural fields, not necessarily all physical fields.
- Immutable data fields in an ABI-complete Rust struct are public and directly
  borrowable; stubgen must not emit one cloning getter per field. Native-mutable
  fields use the interior-mutability and snapshot policy above.
- Convenience constructors should delegate to a complete constructor that
  exposes optional metadata such as `Span`; they must not silently make that
  metadata impossible to preserve.
- A complete-field allocator and any convenience constructor that only fills
  Rust fields return the object directly. `Result<Self>` is reserved for a
  real failure source: parsing, checked narrowing, semantic validation, a
  fallible cast, or a compiler-service call. Generated callers must not need
  `?` or `unwrap()` around plain `ObjectArc::new` allocation.
- Existing registered functions take precedence over adding duplicate methods.
  Existing C++ virtual dispatch remains internal to C++ and Rust crosses the
  standard packed-function ABI.
- Opaque does not imply thread-safe. Native mutable services such as `Analyzer`
  and `UniqueNameSupply` must remain `!Send` and `!Sync`; hiding their private
  fields must not accidentally enable Rust's automatic thread-safety traits.
  Both typed handles retain a `PhantomData<Rc<()>>` marker. `UniqueNameSupply`
  is `skip`ped and hand-written until the generator can preserve that restriction.
  tvm-ffi's generic `ObjectRef` can erase those restrictions as well, so
  cross-thread isolation is not enforced through all FFI conversions. This gap
  remains unresolved; all aliases of these services must stay on one thread.

## Stubgen output ownership

- **GENERATE:** ABI-complete ordinary object structs with public physical
  fields, reference wrappers, read-only `Deref`, inheritance, casts, and
  constructor bodies whose semantics are fully available.
- **RUNTIME:** `ObjectArc`, heterogeneous `Array<Any>`/`Map<K, Any>` support,
  `RValueRef<T>` packed argument holders, safe object identity,
  and `Function::from_type_method`.
- **HANDWRITTEN:** reviewed Rust constructor semantics that stubgen cannot yet
  generate, including validation, defaults, normalization, and derived fields.
- **ABI BLOCKER:** native vptrs, unreflected members, or non-ABI-shareable
  storage. They expose no Rust allocator until the shared ABI is made
  constructible.
- **HANDWRITTEN CONSUMERS:** analyses and complete TVM pass ports used to
  evaluate the generated surface. Stubgen generates their bindings, not their
  algorithms; every pass port is checked against the corresponding C++ pass.

## Golden-reference freeze gate

The handwritten output is ready to freeze as stubgen's replacement target only
when all of the following are true:

1. stubgen's layout-input tests verify runtime identity, reflection, size,
   alignment, and field offsets before emitting every complete `Object`;
2. every ABI-complete allocatable node has a direct complete-field allocator
   whose signature follows flattened native field order and whose body performs
   only Rust allocation, takes exact field types by value without hidden
   conversions or clones, and returns `Self`; identity-bound allocators remain
   private behind their validated or registry-backed constructor;
3. semantic constructors contain generated or reviewed handwritten Rust logic
   before delegating to that allocator;
4. no ABI-complete ordinary IR constructor calls a packed global; an explicitly
   opaque blocker may call its existing native constructor, and no generated
   wrapper exposes unconditional `DerefMut`;
5. C++ can inspect, traverse, compare, retain, and release Rust allocations,
   while Rust can consume C++ allocations through the same wrappers;
6. both languages exercise every reflected type method used by the handwritten
   slice;
7. behavior-only bases expose no reflection creator and cannot produce a
   method-less standalone object; and
8. the generated output is built and tested against the exact native-layout
   manifest consumed by stubgen; and
9. the complete C++, Rust, and Python suites pass against one build.

This freezes the expected generated Rust surface, not the handwritten files.
The actual stubgen is complete only when one invocation emits the mechanical
surface from layout input and the remaining handwritten semantic layer composes
with it without changing the acceptance tests.

Since 2026-09-06 the mechanical portion of `ir`, `prim`, and `tirx` is
reproduced from generated code: `tvm-ffi-stubgen --target rust` classifies
every registered type from the reflected size, alignment, finality, and field
offsets of `libtvm_compiler`, emits the complete layouts and complete-field
allocators in place (`src/ir.rs`, `src/prim.rs`, `src/tirx.rs`), and keeps
the handwritten semantic constructors next to the blocks. `SourceName`,
`UniqueNameSupply`, and `PrimFunc` retain hand-written layouts; the acceptance
tests pass unchanged against the `tvm-ffi` revision pinned in `Cargo.toml`.
No generated `new()` invokes `__ffi_init__` or another packed global.

What still needs either generator support or a reviewed handwritten
implementation: enum members (reflection carries no enum metadata, so the
`enum` directive spells them), the `!Send`/`!Sync` marker on opaque native
services, complete fields without a direct allocator (`SourceName`),
storage native code can move out of (`PrimFunc.body`, so
`tirx.PrimFunc` is `skip`ped and hand-written), semantic validation/default
logic (hand-written, marked by `custom-new`), and rustfmt-clean output (the
formatted files fail `tvm-ffi-stubgen --check`). `te`, `target`, `sym`, and
the pass infrastructure are not generated yet.
