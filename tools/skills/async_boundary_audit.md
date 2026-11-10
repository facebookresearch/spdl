# Audit Hidden Async-to-Sync Network Boundaries

At the start of a migration or optimization, before architecture selection or parameter sweeps, inspect the recurring network-fetch call chain. An apparently synchronous lookup can hide a native async client behind several wrappers. Repeated event-loop or client setup/teardown can serialize requests and conceal costs that more workers or cached-model benchmarks will not diagnose.

Prioritize remote metadata lookups, object downloads, RPC, cache misses and network-backed data fetching. Do not limit the audit to local file I/O, nor assume every network call has an async implementation.

## Follow the actual implementation

1. Bind the effective callable to the loaded source/package and configuration. Follow loader stage → lookup/cache helper → decorators and sync facades → client request implementation → standard-library or native event-loop/transport boundary. Read implementations, not just names or signatures; include callable objects, partials, aliases, retries and callbacks. Use bounded code search to find definitions, not a recursive repository scan.
2. Look for per-item `asyncio.run`, `asyncio.Runner` context creation, `run_until_complete`, `new_event_loop`, and framework/library sync-to-async adapters. A sync function can call an async API internally; a single decorator or `__wrapped__` inspection is not complete coverage. Check which runtime branch is actually selected, with a bounded trace or invocation counters where static dispatch is ambiguous.
3. Identify event-loop, executor, client/session and connection-pool lifetimes. Determine whether they are created and closed per request, per worker, per epoch or once per process. Trace through to the underlying runner/transport implementation; if generated/native code is unavailable, record that boundary as unresolved rather than asserting no hidden cost.
4. Record a compact call-chain map: each wrapper and source location, the actual async primitive, the bridge, the process/thread/loop owner, call frequency, lifecycle and remaining gaps. One-time CLI entrypoint `asyncio.run(main())` is different from a bridge for every sample; do not flag all uses as equally harmful.

`asyncio.run` ordinarily creates and closes an event loop for each call and performs shutdown work. Network-client recreation can also lose connection reuse and repeat DNS/TLS/authentication setup. A multi-second delay may include executor shutdown, unfinished background work, retries or the request itself; it is not a universal fixed penalty inherent in `asyncio.run`. A persistent runner or pool object alone does not prove connection reuse or an efficient hot path.

## Attribute the cost before changing it

Measure the recurring synchronous wrapper end to end and the nested components: runner/client construction, actual request/await, retries/backoff, response handling and cleanup. Separate cold first-call setup from repeated steady calls; collect distributions and invocation counts rather than subtracting unrelated measurements. Inspect outstanding work and executor/client shutdown when the request completes quickly but the wrapper returns much later. Keep network delay, event-loop blocking and GIL ownership distinct.

Use existing telemetry and read-only bounded probes. Do not invoke unknown callbacks or production-writing client methods merely to discover their behavior. Record unresolved source or measurement coverage. The audit is required early; it is not permission to replace every suspicious wrapper or infer savings from source alone.

## Use the native async path where supported

For a recurring bridge with measured overhead, prefer passing the supported native coroutine to SPDL and awaiting the client directly:

```python
async def fetch(key):
    return await client.fetch_async(key)

builder.pipe(fetch, concurrency=fetch_concurrency)
```

`client.fetch_async` stands for the workload's verified public async API, not an invented SPDL network method. Construct and close loop-bound clients in their owning worker/process and event loop; do not share sessions across forked workers or incompatible loops. Preserve connection limits, cache identity, sample order, worker RNG, timeouts, retry/backoff, cancellation, exceptions and cleanup. Merely moving `asyncio.run` into a thread still pays its lifecycle cost; a synchronous call inside an async wrapper still blocks that loop.

If the public caller must remain synchronous, use a documented client-native synchronous API or a supported lifecycle-managed bridge only after verifying ownership, concurrency and shutdown semantics. Do not invent a global loop, reuse a closed loop, or drop required cleanup to make timings shorter. A persistent loop may reduce setup cost but does not automatically permit concurrent requests or connection sharing.

During migration, preserve the untouched original loader and its measurements. Reuse its processing functions, and introduce a necessary bridge repair as an explicitly scoped, semantics-preserving change; never silently rewrite the original baseline to erase its cost. Validate data/error/cancellation behavior and matched CPU/throughput/E2E after repair. During optimization, isolate the change against its immediate parent and re-profile. Cached model tests deliberately bypass the fetch path, so they cannot validate this network-path optimization.

## Completion evidence

Report whether a bridge was found, where it is called, how often it runs, which setup/shutdown costs recur, and what runtime evidence supports the finding. A no-bridge result must state the inspected paths and unresolved boundaries; it is not repository-wide absence. Preserve the pre-change baseline and compare matched complete-work results before accepting a remedy.
