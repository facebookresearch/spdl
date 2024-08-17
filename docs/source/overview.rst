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

What SPDL is NOT
----------------

* SPDL is not a drop-in replacement of existing dataloading solutions.
* SPDL does not guarantee automagical performance improvement.

SPDL is an attempt and an experimental evidence that thread-based parallelism can
achieve higher throughput than common process-based parallelism.

SPDL provides many ways to tune the performance. Some of them are explicit,
such as stage-wise concurrency and the size of thread pool.
Others are implicit and subtle, such as the choice of sync and async functions
and structuring and ordering of the preprocessing.

SPDL makes it easy to diagnose the performance and optimize the performance, 

SPDL does not claim to be the fastest solution out there, nor it aims to be the
fastest. The goal of SPDL is to pave the way for to take advantage of free-threading
Python in Machine Learning, by first solving the bottleneck in data loading,
then moving onto running preprocessings with multi-threading.
