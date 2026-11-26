Paradigm Shift
==============

Concurrency Structure
---------------------

When adopting SPDL, it is important to understand the difference in how SPDL structure
the concurrency, compared against common process-based data loaders.

In process-based data loading, each process runs the entire pipeline.
The pipeline is implemented as DataSet.

.. mermaid::

   flowchart
        subgraph P1[Process 3]
            direction TB
            S10[src] --> S11[Stage 1] --> S12[Stage 2]
        end
        subgraph P2[Process 2]
            direction TB
            S20[src] --> S21[Stage 1] --> S22[Stage 2]
        end
        subgraph P3[Process 1]
            direction TB
            S30[src] --> S31[Stage 1] --> S32[Stage 2]
        end

Whereas SPDL parallelizes the the pipeline stage-by-stage, using different concurrency.
This approach is better fit for achieving higher throughput.

.. mermaid::

   flowchart
        subgraph P1[Stage 1]
            direction TB
            T11[Task 1]
            T12[Task 2]
            T13[Task 3]
        end
        subgraph P2[Stage 2]
            direction TB
            T21[Task 1]
            T22[Task 2]
        end
        subgraph P3[Stage 3]
            direction TB
            T31[Task 1]
        end
        src --> P1 --> P2 --> P3

It is worth noting that in this setup, there is no equivalent of Dataset class.

This paradigm shift makes it difficult to achieve mechanical migration (such as
one-line-change or swap-the-class type of migration) to SPDL.

.. _migration-thread-safety:

Thread Safety and the GIL
-------------------------

In process-based data loading, each process runs its own copy of the pipeline,
so the functions in the pipeline do not run concurrently within a process.
In SPDL, the stage functions run concurrently in a thread pool by default.
This has two consequences for migration.

1. *Functions must be thread-safe.*
   An object that is not thread-safe needs one instance per thread.
   For example, `HuggingFace's Tokenizers <https://github.com/huggingface/tokenizers>`_
   releases the GIL but is not thread-safe
   (see `this issue <https://github.com/huggingface/tokenizers/issues/537#issuecomment-1372231603>`_).
   `Thread local storage <https://docs.python.org/3/library/threading.html#thread-local-data>`_
   is a simple way to create one instance per thread.
2. *Throughput depends on functions releasing the GIL.*
   A function that holds the GIL blocks the other stages while it runs.
   Mixing such functions with those that release the GIL can severely
   degrade performance.
   See :ref:`working-around-the-gil` for which operations release the GIL,
   and what to do when a function does not.
