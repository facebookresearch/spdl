.. _autoresearch-index:

Autoresearch
============

.. py:currentmodule:: spdl.pipeline

In the :ref:`optimization guide <optimization-guide>`, we discussed a
manual workflow for optimizing data loading pipelines: instrument the
pipeline, run experiments, analyze metrics, form hypotheses, apply
changes, and repeat.

This process is systematic and methodical, which makes it a good
candidate for automation. **Autoresearch** is an engine that automates
this entire optimization loop. It uses a coding agent (Claude or Codex)
to analyze pipeline metrics, identify bottlenecks, propose code changes,
and iteratively improve performance with minimal human intervention.

The following chart shows a real autoresearch run on a video
classification pipeline (R3D-18 on Kinetics-400, 1×8 A100 GPUs).
Over 225 experiments, the engine improved throughput from 170 to
1482 samples/s — a **8.7× speedup** — by discovering a combination
of subprocess isolation, video subclipping, concurrency reduction,
and GC alignment:

.. image:: /_static/data/autoresearch_progress_v3.png
   :alt: Autoresearch progress chart showing throughput, step time, SM utilization, duration, and raw SM utilization across 225 experiments
   :width: 100%

.. toctree::

   autoresearch
   autoresearch_example
   autoresearch_example_v2
   autoresearch_architecture
