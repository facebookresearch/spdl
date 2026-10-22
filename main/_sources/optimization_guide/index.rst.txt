.. _optimization-guide:

Optimization Guide
==================

.. py:currentmodule:: spdl.pipeline

In this section, we share the methodologies we developed while
optimizing production AI training pipelines.

When optimizing a training pipeline, it is important to determine
if data loading is a bottleneck, and identify which part of the pipeline
is the bottleneck.

One of the goals of SPDL project is to allow users to programmatically
determine where the bottleneck is.
SPDL achieves this by making the :py:class:`Pipeline` observable.
You can export the runtime performance statistics of the ``Pipeline``,
and analyse them to determine the bottleneck.

We walk through the steps we take to analyse the production system,
and examine some cases, and discuss how we approach performance issues.

.. toctree::

   headspace_analysis
   noisy_neighbour
   setup
   analysis
   stats
   fleet_metrics
   straggler
   resolution

Shared-memory arenas
--------------------

Once measurements identify allocation, inter-process copying or CPU-to-GPU
staging as a relevant cost, consider these shared-memory optimizations. Reusable
buffers trade some memory-budget and lifetime discipline for less allocation or
copying work; measure the application's results before selecting a backend.
Read :doc:`shared_memory_arenas` for the core pool/ring workflow and the later
memory-pinning option for GPU transfer. For technical storage-lifetime management,
see :doc:`../notes/shared_memory_ownership` in Development Notes.

.. toctree::
   :caption: Shared-memory arenas
   :maxdepth: 1

   shared_memory_arenas
