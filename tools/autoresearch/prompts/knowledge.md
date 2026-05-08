# SPDL Pipeline Optimization Knowledge

This knowledge is assembled from shared skill files. The canonical sources are in `spdl/tools/skills/pipeline_perf/`.

## Autoresearch-Specific Notes

### MTP Tier 1/Tier 2 Retry

When implementing MTP, always try Tier 1 (module-level functions with `functools.partial`) first. If the job fails with subprocess-related issues (crashes on startup, silent 0-batch output), retry with Tier 2 (picklable callable classes). The autoresearch loop should attempt both automatically.

### Headspace Analysis in the Loop

The autoresearch loop runs CacheDataLoader analysis automatically as part of the first iteration, alongside other experiments like MTP. CacheDataLoader results must NOT be treated as a real training run — they are excluded from the running best and plateau detection.
