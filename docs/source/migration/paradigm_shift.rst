Paradigm Shift
==============

Concurrency Structure
---------------------

When using SPDL, it is important to understand the difference in how SPDL structure
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

This paradigm shift makes it difficult to achieve mechanical update (such as
one-line-change or swap-the-class type of update) to SPDL.
