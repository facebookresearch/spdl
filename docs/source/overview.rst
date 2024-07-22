Overview
========

What is SPDL?
-------------

SPDL (Scalable and Performant Data Loading) is a research project to explore the design of fast data loading for ML training in free-threading (a.k.a no-GIL) Python, but makes its benefits available to the current ML systems with as little changes as possible.

SPDL implements an abstraction that facilitates building performant data processing pipelines that utilizes multithreading. These pipelines not only achieve high throughput but also provide various insights about its performance, which helps users optimize the its performance.

Oftentimes, the bottleneck of dataloading is in media decoding and pre-processing. So, in addition to the pipline abstraction, SPDL also provides an I/O module for multimedia (audio, video and image) processing. This I/O module was designed from scratch to achieve high performance and high throughput.

There are three main components in SPDL.

1. Asynchronous data loading engine. (The pipeline abstraction)
2. Utility functions to build asynchronous data loading pipelines.
3. Efficient media processing operations that are thread-safe.

How to work around GIL?
-----------------------

In Python, GIL (Global Interpreter Lock) practically prevents the use of multi-threading, however extension modules that are written in low-level languages, such as C, C++ and Rust, can release GIL when executing operations that do not interact with Python interpreter.

Many libraries used for dataloading release the GIL. To name a few;

- Pillow
- OpenCV
- Decord
- tiktoken

Typically, the bottleneck of model training in loading and preprocessing the media data.
So even though there are still parts of pipelines that are constrained by GIL,
by taking advantage of preprocessing functions that release GIL,
we can achieve high throughput.

Why Async IO?
-------------

When training a model with large amount of data, the data are retrieved from remote locations. Network utilities often provide APIs based on Async I/O.

The Async I/O allows to easily build complex data preprocessing pipeline and execute them while automatically parallelizing parts of the pipline, achiving high throughput.

Synchronous operations that release GIL can be converted to async operations easily by running them in a thread pool. So by converting the synchronous preprocessing functions that release GIL into asynchronous operations, the entire data preprocessing pipeline can be executed in async event loop. The event loop handles the scheduling of data processing functions, and execute them concurrently.
