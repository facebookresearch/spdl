Performance Analysis
====================

.. py:currentmodule:: spdl.pipeline

When optimizing an ML model pipeline, it is important to determine
if data loading is a bottleneck, and which part of the pipeline is
the bottleneck.

SPDL facilitates the pipeline performance analysis.
The :py:class:`Pipeline` records runtime performance statistics and
exports them.
This section covers how to log the performance statistics and
analyze the pipeline performance.

.. toctree::

   setup
   logging
   analysis
