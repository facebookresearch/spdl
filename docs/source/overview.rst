Overview
========

What is SPDL?
-------------

SPDL (Scalable and Performant Data Loading) is a research project to explore
the design of fast data loading for ML training with free-threaded (a.k.a no-GIL) Python,
but brings its benefits to the current ML systems.

SPDL implements an abstraction that facilitates building performant data processing
pipelines that utilizes multi-threading.

Oftentimes, the bottleneck of dataloading is in media decoding and pre-processing.
So, in addition to the pipline abstraction, SPDL also provides an I/O module for
multimedia (audio, video and image) processing.
This I/O module was designed from scratch to achieve high performance and high throughput.

There are three main components in SPDL.

1. Task execution engine. (The pipeline abstraction)
2. Utilities to build the data processing pipelines with the task execution engine.
3. Efficient media processing operations that are thread-safe.

SPDL provides many ways to tune the performance. Some of them are explicit,
such as stage-wise concurrency and the size of thread pool.
Others are implicit and subtle, such as the choice of sync and async functions
and structuring and ordering of the preprocessing.
The pipelines built with SPDL provide various insights about its performance.
This makes it easy to diagnose the performance and optimize the performance, 

What SPDL is NOT
----------------

* SPDL is not a drop-in replacement of existing dataloading solutions.
* SPDL does not guarantee automagical performance improvement.

SPDL is an attempt and an experimental evidence that thread-based parallelism can
achieve higher throughput than common process-based parallelism,
even under the constraint of GIL.

SPDL does not claim to be the fastest solution out there, nor it aims to be the
fastest. The goal of SPDL is to pave the way for to take advantage of free-threaded
Python in Machine Learning, by first solving the bottleneck (media decoding)
in data loading, then later performing other parts of preprocessings within
multi-threading paradigm.

How to use SPDL?
----------------

SPDL is highly flexible. You can use it in variety of ways.

1. As a new end-to-end data loading pipeline.
   The primal goal of SPDL is to build performant data loading solutions for ML.
   The project mostly talk about the performance in end-to-end (from data storage
   to GPUs) context.
   Using SPDL as a replacement for existing data loading solution is what the
   development team intends.
2. As a replacement for media preprocessor.
   SPDL uses multi-threading for fast data processing. It is possible to use it in
   subprocesses. If your current data loading pipeline is elabodated, and it is not
   ideal to replace the whole data loading pipeline, you can start adopting SPDL
   by replacing the media processing part. This should allow reducing the number of
   subprocesses, improving the overall peformance.
3. As a research tool in free-threaded Python and high-performance computing.
   SPDL's task execute engine uses async event loop at its core. Async event loop
   itself is single-threaded. Only the functions passed to the executors are
   executed concurrently. This makes SPDL an ideal test bed for experimenting with
   free-threaded Python and high-performance computing.
   SPDL provides option to enable GIL in its I/O module when building from source.
   The ability to enable/disable GIL helps performing controlled experiments.
   Please check out the `Installation <./installation.html>`_ for how to customize the build.
