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

.. toctree::

   autoresearch
   autoresearch_example
   autoresearch_architecture
