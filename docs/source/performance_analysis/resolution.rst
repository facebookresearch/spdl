Resolving the Bottleneck
========================

Say we find a bottleneck in a pipeline (or narrow down to a few suspects)
by analyzing the runtime statistics, how can we optimize it?

The following is common approaches we can try.

Adjust the concurrency
----------------------

Adjusting the concurrency is effective for preprocessing with light to medium
load.
With SPDL, typically 4 or 8 is enough. It should not go beyond
``#of CPU cores / 8``.
When the preprocessing is CPU-intensive, (like decoding videos), increasing the
concurrency can cause the `noisy neighbour <./noisy_neighbour.html>`_ effect,
so care must be taken.

Restructure the stages
----------------------

Data loading is composed of stages bound by different factors.

- downloading is bound by network bandwidth and API call rate.
- preprocessing is bound by CPU and memory.
- GPU data transfer is bound by the cable spec.

When migrating existing pipelines, there were cases where all the logics are
squashed into one function.
Splitting such function into subroutines gives more granular control over
the pipeline.

Profiling the function
----------------------

There are many useful tools for finding out where in a function is the
bottleneck.

Tracer
~~~~~~

A tracer like
`PyTorch Profiler <https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html>`_
can trace the program execution and allows to see how much time is being
spent on what operation.

It captures the entire process, so it allows to see what other things
are going on.
Sometimes you can find a background work that should not be happening.

It also annotates GPU and Tensor metadata, so it is useful to verify
that the memory-pinning is working as expected.

.. image:: ../../_static/data/parallelism_transfer.png

Sampler
~~~~~~~

Another common approach for profiling a function is by sampling.

You can use `perf <https://docs.python.org/3/howto/perf_profiling.html>`_
to collect stats and visualize it with 
`FlameGraph <https://github.com/brendangregg/FlameGraph>`_.

At Meta, we have
`Strobelight <https://engineering.fb.com/2025/01/21/production-engineering/strobelight-a-profiling-service-built-on-open-source-technology/>`_
enabled for production pipelines.

You can look for a particular function that's taking up significant
portion of execution time.
