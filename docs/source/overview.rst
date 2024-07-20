Overview
========

SPDL (Scalable and Performant Data Loading) is a research project to explore the design of fast data loading for ML training in free-threading (a.k.a no-GIL) Python, but makes its benefits available to the existing ML systems with as little changes as possible.

It is composed of three independent components;

1. Asynchronous data loading engine.
2. Utility functions to build asynchronous data loading pipelines.
3. Efficient media processing operations that are thread-safe.

How to work around GIL?
-----------------------

In Python, GIL (Global Interpreter Lock) practically prevents the use of multi-threading, however extension modules that are written in low-level languages, such as C, C++ and Rust, can release GIL when executing operations that do not interact with Python interpreter.

Many libraries used for dataloading release the GIL. To name a few;

- OpenCV
- Decord
- tiktoken

In addition, SPDL provides its own I/O module for processing multimedia. (audio, video and images) The I/O module was designed from scratch for performance and throughput.

Often, the bottleneck of dataloading is in media decoding and pre-processing. So dataloading can be improved by simply using libraries that release GIL in thread pool.

Why Async IO?
-------------

When training a model with large amount of data, the data are retrieved from remote locations. Network utilities often provide APIs based on Async I/O.

The Async I/O allows to easily build complex data preprocessing pipeline and execute them while automatically parallelizing parts of the pipline, achiving high throughput.

Synchronous operations that release GIL can be converted to async operations easily by running them in a thread pool. So by converting the synchronous preprocessing functions that release GIL into asynchronous operations, the entire data preprocessing pipeline can be executed in async event loop. The event loop handles the scheduling of data processing functions, and execute them concurrently.

SPDL Pipeline
-------------

SPDL provides Pipeline abstraction which facilitates the construction of performant pipeline, while providing the knobs to tune the concurrency.

Profiling is imporant when building high-throughput applications, so the pipeline built with SPDL by default comes with stats counter from which users can easily tell what stage of pipeline is the bottleneck. This information help users guide in optimizing the pipeline.
