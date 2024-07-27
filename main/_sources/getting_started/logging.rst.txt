Collecting Runtime Statistics
=============================

.. seealso::

   Please check out the complete example of :py:mod:`performance_analysis`, which shows
   how to log the runtime performance statistics to TensorBoard.

.. note::

   If you are Meta employee, please refer to `this <https://fburl.com/workplace/goxtxyng>`_.

.. py:currentmodule:: spdl.pipeline

The process of optimizing performance is driven by continuous observation, analysis,
and iterative improvement.
It is utmost important to measure the performance, so that one can find the bottleneck
and improves it.

.. image:: ../../_static/data/optimization_cycle.png
   :alt: Performance optimization involves a cyclical process of observing, analyzing, and refining.

The SPDL is designed in a way that allows to collect runtime statistics and export them
so that one can analyze and determine the bottleneck.

In this section, we explain how you can export the statistics.
(We will go over the detail of how to analyze the statistics in
`Optimization Guide <../optimization_guide/index.html>`_.)

There are two kinds of statistics that :py:class:`Pipeline` collects,
:py:class:`TaskPerfStats` and :py:class:`QueuePerfStats`.

The :py:class:`TaskPerfStats` carries the information about functions passed to
:py:meth:`Pipeline.pipe`, and it is collected by :py:class:`TaskStatsHook`.
The :py:class:`QueuePerfStats` carries the information about the flow of data going
through the pipeline, and it is collected by :py:class:`StatsQueue`.

The following is the steps to export the stats.

#. Subclass :py:class:`StatsQueue` and :py:class:`TaskStatsHook` and
   override ``interval_stats_callback`` method.†
#. In the ``interval_stats_callback`` method, save the fields of ``QueuePerfStats`` to
   a location you can access later. ††
#. For ``StatsQueue``, provide the class object (not an instance) to
   :py:meth:`PipelineBuilder.build` method.
#. For ``TaskStatsHook``, create a factory function that takes a name of the
   stage function and returns a list of :py:class:`TaskHook` s applied to the stage,
   then provide the factory function to :py:meth:`PipelineBuilder.build` method.

.. note::

   .. raw:: html

      <ul style="list-style-type: '† ';">
      <li>When overriding the method, ensure that it does not hold the GIL, as this can degrade pipeline performance.</li>
      </ul>
      <ul style="list-style-type: '†† ';">
      <li>The destination can be anywhere such as a remote database, or a local file.</li>
      </ul>
