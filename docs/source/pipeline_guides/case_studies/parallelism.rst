.. _parallelism-performance:

Parallelism and Performance
===========================

.. admonition:: tl;dr
   :class: tip

   If in-process loader threads slow training despite good component timings,
   compare the same pipeline in a subprocess with
   :py:func:`spdl.pipeline.run_pipeline_in_subprocess`. Count IPC, CPU, and memory
   costs in the comparison.

When we started SPDL, we expected multithreading to remove many data-loading
bottlenecks. In this workload it slowed training, even though the loader
released the GIL during its heavy operations. Moving the threaded pipeline
out of the trainer process improved throughput. This case shows why neither
component timings nor average CPU utilization predicted the result.

.. note::

   To learn about the use of different parallelism with SPDL, please
   refer to the :ref:`pipeline-parallelism`.

Pipeline Speed
--------------

The following plot shows the training QPS (the inverse of step time) of the pipeline.

- The ``baseline`` is the original implementation, which is based on TorchData's
  ``StatefulDataLaoder``.
- The ``headspace`` is the upper limit obtained by using
  :py:class:`~spdl.dataloader.CacheDataLoader`.
  (See :ref:`headspace-analysis` for details.)
- The ``mt`` is the solution based on simple multi-threading.
- The ``mtp`` is the optimized solution, where the pipeline (multi-threaded) is
  executed in a subprocess.

.. raw:: html

   <div id="parallelism_step_time"></div>

The baseline can process approximately 2.4 batches per second.
The headspace analysis suggests the maximum performance we can get is around 4.2.

This suggests that there is enough room for data loading improvement,
so we tried SPDL with multithreading.
To our surprise, the speed dropped to 1.2 batches per second.

Inconsistent Performance
------------------------

What confused us was that the other performance metrics from the model part
were not consistent with the overall performance drop.
The following plots show the average time for the backward path and optimizer step.

.. raw:: html

   <div id="parallelism_backward"></div>

.. raw:: html

   <div id="parallelism_optimizer"></div>

We did not make any changes to the model architecture or loss
when we swapped the data loader.
The backward path and the optimizer step appear to be faster than the
baseline.
However, since the ``mt`` run is overall slower, this suggests that
other model computation parts are slower for the ``mt`` run.
(We do not have breakdown for forward path.)
We can also see that the ``mt`` run shows spikier behavior than the others.

Noisy Neighbour and multiprocessing
-----------------------------------

We suspected :ref:`noisy-neighbour` interference even though average CPU
utilization in the threaded run was just over 20%, versus more than 60% in the
baseline. A low average does not rule out contention on a specific thread or
core during kernel launches. We tested isolating the threaded pipeline from
the trainer by running it in a dedicated subprocess.

The following figure illustrates this.

.. image:: /_static/data/parallelism_architecture.png

It turned out that this multi-threading in subprocess (``mtp``) is
highly effective.
It consumed a little more CPU resources (~26%) than simple ``mt`` (~20%),
but it was faster, and no spiky behavior was observed.

.. include:: ../../plots/parallelism.txt
