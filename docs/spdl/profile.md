# Profiling SPDL

SPDL integrates [Perfetto](https://perfetto.dev/), which allows to trace the operations heppening inside of its thread pool for performance profiling.

To obtain traces, use [spdl.utils.tracing] context manager.

```python
with spdl.utils.tracing("trace.pftrace"):
    ...
```
