.. _analysis:

Analyzing the Performance
=========================

.. currentmodule:: spdl.pipeline

In this section, we examine the performance statistics gathered from
a production training system.

.. note::

   To setup the pipeline to collect runtime performance statistics,
   please refer to the :ref:`logging`
   section and the :py:mod:`performance_analysis` example.

The pipeline consists of four stages: download, preprocess, batch,
and transfer. The following code snippet and diagram illustrate this.

.. code-block:: python

   pipeline = (
       PipelineBuilder()
       .add_source(Dataset())
       .aggregate(8)
       .pipe(download, concurrency=12)
       .disaggregate()
       .pipe(preprocess, concurrency=12)
       .aggregate(4)
       .pipe(collate)
       .pipe(gpu_transfer)
       .add_sink()
       .build(num_threads=12)
   )



.. raw:: html

   <div style="width: 360px">

.. mermaid::

   flowchart TB
       subgraph Download
         direction TB
         t11[Task 1]
         t12[Task 2]
         t13[...]
         t1n[Task 12]
       end
       subgraph Preprocessing
         direction TB
         t21[Task 1]
         t22[Task 2]
         t23[...]
         t2n[12]
       end

       Source
        --> Aggregate1["Aggregate (8)"]
        --> Download
        --> Disaggregate
        --> Preprocessing
        --> Aggregate2["Aggregate(4)"]
        --> Collate
        --> transfer[GPU Transfer]
        --> Sink

.. raw:: html

   </div>


.. _compare-loader-implementations:

When a replacement loader is slower
-----------------------------------

Read the original and replacement implementations side by side before
changing tuning knobs. Trace the same samples from source lookup to the
batch consumed by the trainer, including defaults and work hidden behind
wrappers. Compare:

* **What work runs.** Check sampling, decoding, transforms, batching and
  transfers. Look for work the original skips, reuses or performs once per
  batch, but the replacement repeats per sample or epoch.
* **Where work runs.** Check what stays inside each worker, what crosses a
  thread or process boundary, and where the loader waits, copies or turns
  compact buffers into Python objects. For example, the original may pass
  compact indices into a worker and return a finished batch, while the
  replacement passes expanded rows back before assembling the batch.
* **What resources it uses.** Check worker and native thread pools, cache
  placement and reuse, prefetch limits, and the amount of data held in
  flight. Identical worker counts or a full ready queue do not establish
  equivalent work or cost.

Use timings or traces to test which differences explain the slowdown. Close
one confirmed gap at a time, reusing original components where possible.
Verify that the replacement produces equivalent batches and matches
end-to-end throughput at comparable resource cost. Compare the same data,
model, hardware and measurement windows, accounting for all workers and
checking cold and warm behavior separately. Once the replacement reaches
parity, use it as the baseline for further optimization.

.. _compact-loader-metadata:

Keep loader metadata in compact representations
-----------------------------------------------

Avoid JSON as the default format for large or frequently read dataset
metadata. Keep it for small configuration or interchange files where human
readability matters. In the loader's hot path, a faster JSON parser can
save time scanning bytes while the conversion into Python dictionaries,
lists and numbers still dominates. For tabular metadata, prefer a typed
columnar representation such as Arrow or Parquet. For fixed-width numeric
records, consider a format such as NumPy's ``.npy``. Keep values in those
representations through the consumer instead of immediately expanding them
back into Python objects.

When a numeric list contains more than a few hundred values of the same
type, use a NumPy array or PyTorch tensor with an explicit dtype instead of
a Python list. This is a rule of thumb, not a cutoff: a typed buffer stores
the values together so the interpreter does not have to manage a separate
Python object or list slot for each value. Preserve shape, precision and
missing-value semantics, and measure reading, conversion and consumption
together.

TaskPerfStats
-------------

The :py:class:`TaskPerfStats` class contains information about the stage functions.

The statistics are aggregated (averaged or summed) over the (possibly concurrent)
invocations within the measurement window.

Average Time
~~~~~~~~~~~~

The :py:attr:`TaskPerfStats.ave_time` represents the average time of successful function
invocations. Typically, download time is the largest, followed by decoding/preprocessing,
collation, and transfer.

When decoding is lightweight but the data size is large (such as processing long audio
in WAV format), collation and transfer can take longer than preprocessing.

In the following, we can see that the duration for downloading data changes over the course
of training. We also see occasional spikes.
The performance of downloading is affected by both external factors (i.e. the state of
networking) and internal factors (such as the size of data requested).
As we will see later, the speed of download is fast enough for this case, so the overall
change is not an issue here.

Other stages show the consistent performance throughout the training.

.. raw:: html

   <div id='perf_analysis_ave_time'></div>

The number of tasks executed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :py:attr:`TaskPerfStats.num_tasks` represents the number of function invocation
completed, including both successful and failure completion.
You can obtain the number of successful task counts, by subtracting 
:py:attr:`TaskPerfStats.num_failures` from ``TaskPerfStats.num_tasks``.
In this case, there was no failure during the training, so we omit the plot for that.

.. raw:: html

    <div id='perf_analysis_num_tasks'></div>

QueuePerfStats
--------------

When a pipeline is built, queues are inserted between each stage.
The outputs from stage functions are put in their corresponding queues.
The inputs to the stage functions are retrieved from queues upstream to them.

The following figure illustrates this.
Using queues makes it easy to generalize the architecture of the pipeline,
as it allows for the independent orchestration of stage functions.

.. raw:: html

   <div style="width: 420px">

.. mermaid::

   flowchart LR
        Queue
        subgraph S1[Stage 1]
            direction LR
            T11[Task 1]
            T12[Task 2]
            T13[Task 3]
            T14[Task ...]
    	end
        Queue[Queue 1]
        subgraph S2[Stage 2]
            direction LR
            T21[Task 1]
            T22[Task 2]
            T23[Task ...]
        end
        Queue2[Queue 2]

        T11 -- Put --> Queue
        T12 -- Put --> Queue
        T13 -- Put --> Queue
        T14 -- Put --> Queue
        Queue ~~~ T21 -- Get --> Queue
        Queue ~~~ T22 -- Get --> Queue
        Queue ~~~ T23 -- Get --> Queue
        T21 -- Put --> Queue2
        T22 -- Put --> Queue2
        T23 -- Put --> Queue2

.. raw:: html

   </div>


.. note::

   - In the following plots, a legned ``<STAGE>`` indicates that it is referring
     to the performance of the queue attached at the exit of the ``<STAGE>``.
   - The ``sink`` queue is the last queue of the pipeline and it is consumed by
     the foreground (the training loop).


Throughput
~~~~~~~~~~

Each queue has a fixed-size buffer. So the buffer also serves as
a regulator of back pressure.
For this reason, the throughput is roughly same across stages.
The :py:attr:`QueuePerfStats.qps` represents the number of items that
went through the queue (fetched by the downstream stage) per second.

Note that due to the operations ``aggregate`` and ``disaggregate``,
the QPS values from different stages may live in different value
range, but they should match if normalized.

.. raw:: html

    <div id='perf_analysis_qps'></div>


.. _data-readiness:

Data Readiness
~~~~~~~~~~~~~~

The goal of performance analysis is to simplify the identification
of performance bottlenecks.
Due to the concurrency and elastic nature of the pipeline,
this task is not as straightforward as we want it to be.

The "Data Readiness" is our latest attempt in finding the bottleneck.
It measures the relative time that a queue has the next data available.

We define the data readiness as follow

.. code-block:: text

   data_readiness := 1 - d_empty / d_measure

where ``d_measure`` is the duration of the measurement (or elapsed time),
``d_empty`` is the duration the queue was empty during the measurement.

.. raw:: html

   <div id='perf_analysis_data_readiness'></div>

Data readiness close to 100% means that data is available whenever
the downstream stage (including the foreground training loop) tries to
fetch the next data, indicating that the data processing pipeline is
faster than the demand.
   
On the other hand, the data readiness close to 0% suggests that downstream
stages are starving. They fetch the next data as soon as it is available,
but the upstream is not able to fill it quick enough, indicating that the
data processing pipeline is not fast enough, and the system is
experiencing data starvation.

The following plot shows the data readiness of the same training run we observed.
From the source to up to the collate, the data readiness are close or above 99%.
But at GPU transfer, it drops to 85% and the sink readiness follows.

.. raw:: html

    <div id='perf_analysis_occupancy_rate'></div>

Note that the value of data readiness is meaningful only in the context of
stages before and after the queue. It is not valid to compare the readiness of
different queues.

The drop at preprocess means that the downstream stage (aggregate) is consuming
the results faster than the rate of production.

The recovery of the data readiness in collate means that the rate at which
the downstream stage (gpu_transfer) is consuming the result is not as fast
as the rate of production.

The data readiness of the sink slightly recovers.
This suggests that the rate at which the foreground training loop consume the
data are slightly slower than the rate of transfer.

Queue Get/Put Time
~~~~~~~~~~~~~~~~~~

Other stats related to data readiness are the time stage functions wait for the
completion of queue get and put operations.

If data is always available (i.e. data readiness is close to 100%), then stages
won't be blocked when fetching the next data to process.
Therefore, the wait time for getting the next data is minimal.

Conversely, if the upstream is not producing data fast enough, the
downstream stages will be blocked while fetching the next data.

.. raw:: html

    <div id='perf_analysis_queue_get'></div>

The time for put operation exhibits opposite trend.

If the data readiness is close to 100%, queues are full, and stage functions
cannot put the next result, so they must wait until a spot becomes available.
As a result, the wait time increases.
The wait time increases with higher concurrency, as multiple function invocations
attempt to put results into the queue.

If the production rate is slow, downstream stages wait for the next items.
Consequently, items placed in the queue are fetched immediately,
resulting in a less occupied queue.
This leads to the situation where the put time becomes close to zero.

The following plot shows that download and collate stages wait longer because
it is faster than the rate of the consumption of their downstreams
(preprocess and gpu_transfer).

.. raw:: html

    <div id='perf_analysis_queue_put'></div>

Summary
-------

SPDL Pipeline exposes performance statistics which help identifying the
performance bottleneck.

The most important metrics is the queue get time of sink stage, because the
wait time directly translates to the time the foreground training loop
waits for.

.. include:: ../plots/perf_analysis.txt
