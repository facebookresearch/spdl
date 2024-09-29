Is data loading bottleneck?
===========================

.. py:currentmodule:: spdl.dataloader

When optimizing ML training pipeline, it is important to determine if data loading is
the bottleneck.

The sink stage of :py:class:`Pipeline` is the interface between the data processing pipeline
running in the background thread and the model training running in the foreground thread.

The sink is responsible for fetching data produced by the pipeline upstream stages and
putting them in the buffer, which is accessed by the foreground thread.

If the pipeline upstream is not producing data fast enough, the sink gets blocked on the input
data.
Similarly, if the foreground thread is not consuming the data fast enough, the buffer gets full
and the sink gets gets blocked on the output buffer.

Therefore, by monitoring the time sink gets blocked on each end of the buffer,
we can determine if the training is bound by data loading or model training.

We look at this with some toy examples.

Baseline
--------

The following snippet constructs a Pipeline without any processing.

.. code-block::

   >>> pipeline = (
   ...     PipelineBuilder()
   ...     .add_source(range(100))
   ...     .add_sink(1)
   ...     .build()
   ... )

We execute it as following. There is no bottleneck in the execution.

.. code-block::

   >>> with pipeline.auto_stop():
   ...     for item in pipeline:
   ...         pass

Executing the above code gives a log like the following.

.. code-block::

   [sink]	Processed   100 items in 7.0900 [ms ]. QPS: 14104.37. Average wait time: Upstream: 0.0674 [ms ], Downstream: 0.0007 [ms ].

The ``Upstream`` in the log is the time that sink waited on the production of data, and the ``Downstream`` is the time that sink waited for the buffer space to become available.

In the above case, there is no bottleneck and both are quit fast.

Bottleneck is in training loop
------------------------------

Now, we introduce an artificial delay in the foreground thread to see how the average time changes.
We change the way the pipeline is executed as following.

.. code-block::

   >>> with pipeline.auto_stop():
   ...     for item in pipeline:
   ...         time.sleep(100)

.. code-block::

   [sink]	Processed   100 items in 10.2294 [sec]. QPS: 9.78. Average wait time: Upstream: 0.0157 [ms ], Downstream: 102.2399 [ms ].

We see that the average time that sink waited on the downstream buffer increased from less than 1 millisecond to over 100 milliseconds, which corresponds to the delay we introduced.

Bottleneck is in data loading
-----------------------------

Next, we modify the pipeline as follow to introduce an artificial delay in the data loading pipeline.

.. code-block::

   >>> async def delay(item):
   ...     await asyncio.sleep(0.1)
   ...     return item
   ...
   >>>
   >>> pipeline = (
   ...     PipelineBuilder()
   ...     .add_source(range(100))
   ...     .pipe(delay)
   ...     .add_sink(1)
   ...     .build()
   ... )

We execute the pipeline, like the first time without any delay in the foreground thread.

.. code-block::

   >>> with pipeline.auto_stop():
   ...     for item in pipeline:
   ...         pass

Executing the above, we obtain the following log.

.. code-block::

   [delay]	Completed   100 tasks (  0 failed) in 10.1984 [sec]. QPS: 9.81 (Concurrency:   1). Average task time: 101.3354 [ ms].
   [sink]	Processed   100 items in 10.1938 [sec]. QPS: 9.81. Average wait time: Upstream: 100.8673 [ms ], Downstream: 0.0167 [ms ].

The average upstream wait time is increased to 100 millisecond.

Summary
-------

Using :py:class:`Pipeline`, it becomes easy to determine if the bottleneck is in data loading or not. When the data loading is the bottleneck, the sink stage gets blocked on the input queue. So if the upstream wait time is larger than that of downstream, the data loading is the bottleneck.
