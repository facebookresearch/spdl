Measuring the performance
=========================

.. note::

   Meta employees, please refer to `this <https://fburl.com/workplace/goxtxyng>`_.

.. py:currentmodule:: spdl.pipeline

When :py:class:`Pipeline` orchestrates the execution of functions,
it also keeps track of various timing information.

Such information are exposed as :py:class:`TaskPerfStats` and :py:class:`QueuePerfStats`.

You can log such information by following steps.

.. seealso::

   The :py:mod:`performance_analysis` example demonstrates writing to TensorBoard.

For :py:class:`QueuePerfStats`

#. Subclass :py:class:`StatsQueue` and override :py:meth:`StatsQueue.interval_stats_callback`.†
#. In the ``interval_stats_callback`` method, save the fields of ``QueuePerfStats`` to somewhere you can access later. ††
#. Provide the new class to :py:meth:`PipelineBuilder.build` method.

Similarly for :py:class:`TaskPerfStats`

#. Subclass :py:class:`TaskStatsHook` and override :py:meth:`TaskStatsHook.interval_stats_callback`.†
#. In the ``interval_stats_callback`` method, save the fields of ``TaskPerfStats`` to somewhere you can access later. ††
#. Create a factory function that takes a name of the stage functoin and return a list of :py:class:`TaskHook`-s applied to the stage.
#. Provide the factory function to :py:meth:`PipelineBuilder.build` method.

.. note::

   .. raw:: html

      <ul style="list-style-type: '† ';">
      <li>When overriding the method, make sure that the method does not hold the GIL, otherwise the logging can degrade the pipeline performance.</li>
      </ul>
      <ul style="list-style-type: '†† ';">
      <li>The destination can be anywhere such as remote database, or local file.</li>
      </ul>
