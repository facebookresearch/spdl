# Local SPDL Tests and Loader-Only Capacity Curves

For model optimization, use the separate [cached-first model benchmark](cached_model_benchmark.md): short warmed native `CacheDataLoader` replay before matched full-epoch E2E validation. The loader-only tests below remain uncached so they measure the input path being changed; neither method replaces final training/quality evidence.

Use matched local original-loader/SPDL tests as the fast iteration loop during initial migration. Reuse real components and the closest execution architecture. Match data, seed/order/batching, cache state, consumer work, CPU allocation, worker/prefetch settings, warmup and timed work. Keep detailed data/tensor consistency checks out of performance timing.

Measure `(CPU utilization, throughput)` over the same interval and full loader/harness process-tree scope, including workers and native/feeder/pin threads. Normalize CPU by the same allocated cores or report aggregate core equivalents. Initial points should be close; materially more CPU to match throughput is not parity. Record memory metric, units and shared-memory accounting separately.

For a capacity curve run the loader **without model forward/backward/optimizer training load**, with the same bounded consumer work for both implementations. Plot CPU utilization on x and throughput plus memory on clearly labelled y axes or aligned panels. Label architecture/resource settings, data/cache state, warmup, windows, repeats and missing values. Do not compare differently normalized CPU or call summed shared RSS unique memory.

A small matched range can characterize the initial trend; broad production parameter sweeps wait until architecture is finalized. If SPDL reaches at least 2x throughput at comparable CPU/resource use, record capacity headroom and consider a separate Phase 2 reduced-training-node candidate. Validate end-to-end compute/communication/resource costs, batch/schedule semantics and model quality before accepting it; loader-only speed does not prove training scaling.

Local tests do not replace final matched training/accuracy validation. Use the phased workflow and persistent development record; keep each attempt concise with commit/job/analysis links.
