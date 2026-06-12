Resolving the Bottleneck
========================

Say we find a bottleneck in a pipeline (or narrow down to a few suspects)
by analyzing the runtime statistics, how can we optimize it?

The following are common approaches we try.

Adjust the concurrency
----------------------

Adjusting the concurrency is effective for preprocessing with
a light to medium load.
With SPDL, typically 4 or 8 is enough.
It should not go beyond the number of CPU cores divided by 8.
When preprocessing is CPU-intensive (like decoding videos),
increasing the concurrency can cause the :ref:`noisy-neighbour` effect,
so care must be taken.

Restructure the stages
----------------------

Data loading is composed of stages bound by different factors.

- Downloading is bound by network bandwidth and API call rate.
- Preprocessing is bound by CPU and memory.
- GPU data transfer is bound by the cable specifications.

When migrating existing pipelines, there were cases where all the logics are
squashed into one function.
Splitting such functions into subroutines gives more granular control
over the pipeline.

Profiling the function
----------------------

There are many useful tools for finding out where in a function is the
bottleneck.

Tracer
~~~~~~

A tracer like
`PyTorch Profiler <https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html>`_
can trace the program execution.
It allows you to see how much time is being spent on each operation.

The following figure shows a function making multiple network calls.

.. image:: ../../_static/data/resolution_tracer.png

The underlying API is defined with ``async`` keyword.
It is a common practice (although it's an anti-pattern) to wrap async
functions with :py:func:`asyncio.run` to provide pseudo synchronous API
for the sake of ease of use.
As a result, users are not aware that an event loop is created and
destroyed for each invocation of API call, which is redundant and
inefficient especially when the series of calls are made in event loop.
This is what is happening in the above figure.

The function is being executed in an asynchronous event loop,
but it is calling a pseudo-synchronous API,
so an event loop is created and destroyed for each call.
A lot of time is consumed in construction and destruction of
redundant event loops.

The PyTorch Profiler also annotates GPU and Tensor metadata, so it is
useful to verify GPU-related activities.
For example, to ensure that the data transfer is not interrupting the
the main training, we can check that the data transfer is happening in
a separate stream.

.. image:: ../../_static/data/parallelism_transfer.png

Sampler
~~~~~~~

Another common approach for profiling a function is sampling.

You can use `perf <https://docs.python.org/3/howto/perf_profiling.html>`_
to collect stats and visualize them with 
`FlameGraph <https://github.com/brendangregg/FlameGraph>`_.

At Meta, we have
`Strobelight <https://engineering.fb.com/2025/01/21/production-engineering/strobelight-a-profiling-service-built-on-open-source-technology/>`_
enabled for production pipelines.

You can look for a particular function that's taking up significant
portion of execution time.

One interesting way to use profiler is to check how often the GIL is held.
The `take_gil() <https://github.com/python/cpython/blob/3.12/Python/ceval_gil.c#L331-L458>`_ function is resposible for acquiring the GIL.
By organizing the stack around ``take_gil`` function, we can see what functions
are competing for the GIL, and identify a potential bottleneck.

The following figure is an example of stacks sampled with Strobelight.
It says that the pipeline spends 13% of the time trying to acquire the GIL.

.. image:: ../../_static/data/strobe_light.png

What stands out is that loading array data from NPZ file holds the GIL longest.
Functions from I/O and preprocessing take up only 0.5% of the runtime for acquiring the GIL.
But loading NPZ holds the GIL for more than 2.7% of the runtime.

This pipeline uses :py:func:`spdl.io.load_npz` to load the NPZ file.
This function is more efficient than the official :py:func:`numpy.load` function,
however, it only partially releases the GIL.

The `PR#849 <https://github.com/facebookresearch/spdl/pull/849>`_ is one of out attempts
to make it fast and efficient.
Another approach to resolve this is to change the file format.
This NPZ file uses compressed format, and loading compressed NPZ file requires additional
memory allocation and compute resource for decompression.
If the storage space permits, re-creating the dataset without compression can help
reducing the GIL contention.
