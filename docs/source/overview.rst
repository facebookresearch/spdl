Overview
========

SPDL (Scalable and Performant Data Loading) is a research project to explore the design of fast data loading for ML training in free-threading (a.k.a no-GIL) Python, but makes its benefits available to the existing ML systems with as little changes as possible.

It is composed of three independent components;

1. Asynchronous data loading engine.
2. Utility functions to build asynchronous data loading pipelines.
3. Efficient media processing operations that are thread-safe.

Additionally, it also implements multi-thread version of PyTorch's DataLoader, which one can use as an entrypoint for evaluating adoption of SPDL.

Why AsyncIO?
------------

Async IO has two benefits
- Makes it easy to write complex parallelism
- Integrate smoothly with asynchronous network API calls used for data acquisition.
