# Loader graph initialization

These techniques reduce repeated object-graph construction, transform scripting and callable serialization before input execution. Open this topic when duplicate native wrappers, repeated per-feature modules or dynamic callable capture dominates initialization.

[Catalog index](../../technique_catalog.md) · [Startup](../startup.md)

**`intern_native_singleton_wrappers`: deduplicate wrappers around shared native state**
- Trigger: many Python wrappers refer to one native singleton, and constructing or cleaning them up produces quadratic startup behavior.
- Technique: first profile wrapper creation/teardown to confirm that repeated registrations or identity bookkeeping around the same native object are the pathological work. Route eligible construction through a process-local factory that looks up an existing equivalent wrapper before allocating and registering another one. Key the lookup by native-object lifetime identity, wrapper type, and any configuration that affects behavior; a raw pointer alone is unsafe if addresses can be reused after destruction. Reuse only wrappers whose constructor contract and ownership policy allow it, preserving required initialization side effects. Remove entries when their native lifetime ends, or use weak references where supported and appropriate so interning does not keep dead resources alive. Keep distinct mutable configuration out of the shared wrapper or exclude those instances from interning. This reduces duplicate wrapper bookkeeping, not the cost of initializing genuinely different native objects.
- Guards: preserve constructor behavior, pointer and type identity, compatibility, and ownership/lifetime semantics. Do not merge wrappers that carry distinct mutable configuration. Confirm the pathological work actually comes from duplicate wrapper bookkeeping before applying interning broadly.

**`flat_transform_module_scripting`: flatten repeated transform module trees**
- Trigger: many per-feature module objects must be scripted or compiled before the input path can run.
- Technique: when the existing path already scripts transforms, inventory the repeated operations and express feature-specific differences as script-compatible data: feature offsets, operation tags, parameters, and any persistent state. Replace the duplicated nested module instances with a small set of flat modules that apply those operations to the corresponding slices or compatible feature groups. For example, equivalent per-feature affine transforms may use aligned scale/bias arrays instead of one module object per feature, while preserving each feature's output position and arithmetic semantics. Keep sparse, ragged, or event-specific handling separate where it cannot use the same dense operation. Script the reduced module structure once and reuse it in every context that used the original transforms. Validate outputs and state against the original feature-by-feature implementation; the startup hypothesis is less repeated module traversal/scripting, not adopting a different model compiler or changing preprocessing semantics.
- Guards: verify tensor parity across feature types, sparse/event semantics, and every training/evaluation context. Preserve feature ordering, parameters, and state; do not substitute generic model compilation work for an input-startup hypothesis.

**`pickle_native_loader_graph`: serialize static loader callables without dynamic capture**
- Trigger: dynamic callable serialization dominates loader bootstrap even though much of the graph can be represented by importable functions and explicit state.
- Technique: inventory the functions and captured state in the loader graph, then replace supported dynamic closures with top-level importable functions, partial applications, or callable objects whose serializable fields contain the same configuration. Standard pickle can identify an importable function by module/name rather than serializing its dynamic environment; the receiving process must be able to import the same implementation. Keep large immutable data as backing references where appropriate and create live clients/handles in the child, without dropping required bound state. Check the complete graph for unsupported closures or dynamically generated code and retain the original serializer fallback for those graphs. Scope this selection to loader serialization rather than changing unrelated process payloads. Round-trip the same graph and compare outputs, errors, and worker behavior before timing serialization and first batch separately.
- Example: place the following in an importable module. It illustrates moving one captured parameter into explicit bound state; it does not make arbitrary closures or configuration objects picklable.

  ```python
  from functools import partial

  def select_field(row, *, name):
      return row[name]

  select_label = partial(select_field, name="label")
  ```

- Guards: verify closures, bound state, function identity, errors, and deterministic worker behavior. Never silently drop captured values to make a graph serializable. Scope serializer changes to the loader path; compare identical graphs and payloads before attributing first-batch time.
