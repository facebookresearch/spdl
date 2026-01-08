.. raw:: html

   <script src='https://cdn.plot.ly/plotly-2.34.0.min.js'></script>
   <script src='../_static/js/plot_perf_stats.js'></script>

.. _stats:

Understanding the performance statistics
========================================


To optimize the pipeline performance, we need to infer how the pipeline is behaving through performance statistics.
In this section, we simulate some pipeline behavior patterns for simple cases to understand what the pipeline performance looks like.

.. seealso::

   :ref:`example-performance-analysis`
      Explains how to collect the pipeline runtime performance statistics.

   :ref:`example-performance-simulation`
      The script used in this section to simulate the pipeline runtime behavior.

Single-stage Pipeline
---------------------

First, we look at pipelines with one (mock) preprocessing stage, and how the performance statistics look
for cases where the data loading is fast enough and otherwise.

.. code-block:: python

   pipeline = (
       PipelineBuilder()
       .add_source(...)
       .pipe(op1)
       # We will change the task time (25 ms and 35 ms) to simulate the cases where
       # the data loading is faster or slower than the foreground task.
       .add_sink(...)
       .build(...)
   )

   with pipeline.auto_stop():
       for batch in pipeline.get_iterator():
           process(batch)  # Assume this takes 30 ms

The following diagram illustrates the structure of a single-stage pipeline used in our performance analysis:

.. mermaid::

   graph LR
       subgraph Pipeline
           direction LR
           Source
           Q0@{ shape: das, label: "Source Queue" }

           subgraph S1[Stage 1]
               O1[Op 1]
               Q1@{ shape: das, label: "Queue 1" }
           end
           style S1 stroke-width:3px

           Sink
           QE@{ shape: das, label: "Sink Queue" }
       end

       Source
       --> |Put| Q0
       --> |Get| O1
       --> |Put| Q1
       --> |Get| Sink
       --> |Put| QE
       --> |Get| Foreground

Suppose that we have a foreground job (can be inference, training or anything else) that takes 30 ms to complete.
If the data loading pipeline can produce one batch within 30 ms, then we can keep the foreground busy.

We construct pipelines with one stage, and simulate its performance by having the stage sleep either for 25 ms or 35 ms.
We run the pipeline and gather the performance statistics.

Task Execution Time and Throughput
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following figure shows the average task time and throughput when the preprocessing stage takes 25 ms to complete.

.. raw:: html

    <div id="single1-1"></div>
    <script>
    document.addEventListener(
        'DOMContentLoaded',
        function() {
            loadAndPlotQueueStats(
            'single1-1',
            '../_static/data/example_simu_single1.json',
            ['task_time', 'qps']);
        },
        false);
    </script>

We can see that the preprocessing stage takes about 25 ms.
The throughput (QPS) is around 33 ms (~ 1000 / 30), which corresponds to the speed that foreground code consumes the data.

When we change the task execution time to 35 ms, we obtain the following.

.. raw:: html

    <div id="single2-1"></div>
    <script>
    document.addEventListener(
        'DOMContentLoaded',
        function() {
            loadAndPlotQueueStats('single2-1',
            '../_static/data/example_simu_single2.json',
            ['task_time', 'qps']);
        },
        false);
    </script>

The throughput is now around 28 ms (~ 1000 / 35).
Because the data loading is now slower than the foreground consumer,
the pipeline throughput is governed by the slowest stage of the data loading pipeline.

Since the SPDL pipeline is composed of series of asynchronously-blocking queues
with fixed buffer size, the throughput of each pipe is affected by its upstream
and downstream stages.
Therefore, the throughput is governed by the slowest stage (including the foreground task)
and its value is the same everywhere(†) in the pipeline.

.. note::

   (†) When using aggregate and disaggregate, the value of throughput reported is multiplied by the
   size of the batch, and it appears different on the surface, but they should produce the same value if
   normalized by the batch size. We discuss this later.

Queue Get/Put Time
~~~~~~~~~~~~~~~~~~

Since multiple tasks are executed concurrently for a pipeline stage,
the task performance does not illustrate the pipeline's overall performance.
To understand the overall performance, we need to look at the performance
statistics of queue inputs and outputs.

First we look at the average time it takes to complete the
:py:meth:`asyncio.Queue.get` and :py:meth:`asyncio.Queue.put` operations.

.. note::

   In the following, the put time refers to the time it takes to put data
   **in the queue**, and the get time refers to the time it takes to get data
   **from the queue**.
   The ``STAGE_queue`` refers to the queue attached at the end of the ``STAGE``.

   So the put time of ``OP_queue`` refers to the time for ``op`` to put a
   result to the ``OP_queue``.
   The get time of ``OP_queue`` refers to the time for **the downstream stage
   (not the ``OP`` referred to by the queue name)** to retrieve the data from the
   ``OP_queue``.

Fast to slow
^^^^^^^^^^^^

When a stage is faster than its downstream,
that stage has to wait for the downstream to consume data from the queue.

.. mermaid::

   graph LR

       s1 e1@==> |Put| q e2@--> |Get| s2
       s1@{ label: "Fast Op" }
       s2@{ label: "Slow Op" }
       q@{ shape: das, label: "Queue" }
       e1@{ animation: fast }
       e2@{ animation: slow }

This translates to the observation that the ``put`` operation gets blocked
more often because the downstream is not consuming the data fast enough
and there is no space in the queue.
Meanwhile the ``get`` operation completes quickly, because the data is
always available.

In the following example, the pipeline is operating at 30 ms per batch.
The pipe stage takes 25 ms to complete the task then put the result to the queue,
so its put wait time is 5 ms (30 - 25).

.. note::

   The source stage has no performance overhead because it has
   no operation to perform.
   Therefore it is always blocked on the put operation, which
   corresponds to the pipeline throughput. (30 ms in this case)

.. raw:: html

    <div id="single1-2"></div>
    <script>
    document.addEventListener(
        'DOMContentLoaded',
        function() {
            loadAndPlotQueueStats(
                'single1-2',
                '../_static/data/example_simu_single1.json',
                ['get_time',  'put_time'],
                hideSourceSink=true);
        },
        false);
    </script>

Slow to fast
^^^^^^^^^^^^

When a stage is slower than its downstream, the downstream consumes all data
before the stage completes.

.. mermaid::

   graph LR

       s1 e1@--> |Put| q e2@==> |Get| s2
       s1@{ label: "Slow Op" }
       s2@{ label: "Fast Op" }
       q@{ shape: das, label: "Queue" }
       e1@{ animation: slow }
       e2@{ animation: fast }

In such cases, we see the queue stats flipped from the fast-to-slow case.
The ``put`` operation completes quickly because the data put in the queue
is fetched immediately by the downstream task, thus there is always space in the queue.
The ``get`` operation is blocked because the upstream stage is not producing
the data fast enough and there is no data to get from the queue.

The following figures show the average queue get/put time when the pipe stage takes 35 ms.

.. raw:: html

    <div id="single2-2"></div>
    <script>
    document.addEventListener(
        'DOMContentLoaded',
        function() {
            loadAndPlotQueueStats(
                'single2-2',
                '../_static/data/example_simu_single2.json',
                 ['get_time',  'put_time'],
                 hideSourceSink=true);
        },
        false);
    </script>


The get time for ``35ms_queue`` (that is, the sink stage's attempt to
get data from the pipe stage) takes about 35 ms, which corresponds to
the pipe task execution time.

The put time for ``35ms_queue`` (that is, the ``35ms`` stage's attempt to
put result data to the queue) takes no time.

Queue Occupancy (Data Readiness)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The queue put/get times reveal whether the pipeline stage is fast enough
regardless of its task execution time and concurrency.
However, it turns out that we cannot judge the performance from the absolute
value of wait time, so we always have to watch these metrics as a pair.
This is inconvenient.
Essentially, what we are interested in is whether the next data to consume
is available or not.
We can approximately measure the data readiness by how often the queue
has data.
We call such metrics Queue Occupancy or Data Readiness.

According to the previous observations, when the pipeline is producing the
data fast enough, the queue has always data.
So the queue occupancy becomes 100%.

.. raw:: html

    <div id="single1-3"></div>
    <script>
    document.addEventListener(
        'DOMContentLoaded',
        function() {
            loadAndPlotQueueStats(
                'single1-3',
                '../_static/data/example_simu_single1.json',
                ['occupancy'],
                hideSourceSink=true);
        },
        false);
    </script>


On contrary, when a stage in the pipeline is not producing the data fast enough,
the subsequent queues do not have data available.
So the queue occupancy becomes 0%.

.. raw:: html

    <div id="single2-3"></div>
    <script>
    document.addEventListener(
        'DOMContentLoaded',
        function() {
            loadAndPlotQueueStats(
                'single2-3',
                '../_static/data/example_simu_single2.json',
                ['occupancy'],
                hideSourceSink=true);
        },
        false);
    </script>

The first clue to see if a pipeline is operating at a speed that is fast enough for
its consumer is to check the data readiness of the sink stage.

Performance Characteristics Summary
------------------------------------

The following table summarizes how queue metrics behave depending on whether the pipeline runs faster or slower than the foreground loop:

.. list-table:: Queue Metrics vs Pipeline/Foreground Speed
   :header-rows: 1
   :widths: 25 25 25 25

   * - Scenario
     - Queue Put Time
     - Queue Get Time
     - Queue Occupancy
   * - ✅ Pipeline is fast enough
     - High (queues full)
     - Low (data always available)
     - High occupancy
   * - ❌ Pipeline is not fast enough
     - Low (queues rarely full)
     - High (data rarely available)
     - Low occupancy



Multi-Stage Pipeline Structure
-------------------------------

Now we understand how the performance statistics look like for a single stage pipeline,
let's look at the multi-stage pipeline.

We consider a pipeline with 3 stages.
We change the task time of each stage to 10, 25 and 40 ms, and run the pipeline.
Then we reversed the order of the stages and run the pipeline again.

The following code and figure illustrate the 3-stage pipeline.

.. code-block:: python

   pipeline = (
       PipelineBuilder()
       .add_source(...)
       .pipe(op1)
       .pipe(op2)
       .pipe(op3)
       # We will change the task time (10, 25, 40 ms and the reversed order) to
       # simulate the cases where bottlenecks come at early or late stages.
       .add_sink(...)
       .build(...)
   )

   with pipeline.auto_stop():
       for batch in pipeline.get_iterator():
           process(batch)  # Assume this takes 30 ms

.. mermaid::

   graph LR
       subgraph Pipeline
           direction LR
           subgraph S1[Stage1]
               O1[Op 1]
               Q1@{ shape: das, label: "Queue 1" }
           end
           subgraph S2[Stage2]
               O2[Op 2]
               Q2@{ shape: das, label: "Queue 2" }
           end
           subgraph S3[Stage3]
               O3[Op 3]
               Q3@{ shape: das, label: "Queue 3" }
           end
           Source:::invisible
           --> |Get| O1
           --> |Put| Q1
           --> |Get| O2
           --> |Put| Q2
           --> |Get| O3
           --> |Put| Q3
           --> |Get| Sink:::invisible
       end

       style S1 stroke-width:3px
       style S2 stroke-width:3px
       style S3 stroke-width:3px

       style Source fill-opacity:0, stroke-opacity:0
       classDef invisible display:none;


Fast to Slow
~~~~~~~~~~~~

We assign the task execution time of 10, 25 and 40 ms to each stage.
The foreground task takes 30 ms to complete.

First, we check the average task execution time and throughput.

.. raw:: html

   <div id="multi1-1"></div>

    <script>
    document.addEventListener(
        'DOMContentLoaded',
        function() {
            loadAndPlotQueueStats(
                'multi1-1',
                '../_static/data/example_simu_multi1.json',
                ['task_time', 'qps']);
        },
        false);
    </script>

The pipeline throughput is governed by the slowest stage, which takes 40 ms
per each invocation.
So the throughput is 25 (~ 1000/40 ms).

Now we look at the queue stats.

.. raw:: html

   <div id="multi1-2"></div>

    <script>
    document.addEventListener(
        'DOMContentLoaded',
        function() {
            loadAndPlotQueueStats(
                'multi1-2',
                '../_static/data/example_simu_multi1.json',
                ['put_time', 'get_time', 'occupancy']);
        },
        false);
    </script>

**Put time**

The pipeline is operating at the speed of 40 ms / iteration, so the source has to
wait for 40 ms to put the next item.

Stages 1, 2 and 3 take 10, 25 and 40 ms, so the queue put time for each stage
corresponds to the subtraction of the task execution time from the
pipeline throughput, that is 30 ms (40 - 10), 15 ms (40 - 25), 0 ms (40 - 40) for
each stage respectively.

**Get time**

Since the source and stages 1 and 2 are operating faster than their downstream stage,
their queue get time is 0.
(The get time is the time the next stage pulls data from the queue.)

The sink has no work to do and it has to wait for stage 3 to complete, so its
get time is 40ms.
The foreground completes its task in 30ms, so it waits for 10 ms until the next
batch becomes available in the sink.

**Occupancy**

The queue occupancy communicates this information in a compact manner.
The queue occupancy of the source, stage 1 and stage 2 are close to 100%,
which indicates that each stage is faster than the previous one.

The occupancy of stage 3 and the sink drop to 0, meaning they are not
fast enough for their downstream consumer.

Slow to fast
~~~~~~~~~~~~

Now we reverse the order of the stages, and run the pipeline again.
The bottleneck is now at the stage 1.

.. raw:: html

   <div id="multi2-1"></div>

    <script>
    document.addEventListener(
        'DOMContentLoaded',
        function() {
            loadAndPlotQueueStats(
                'multi2-1',
                '../_static/data/example_simu_multi2.json',
                ['task_time', 'qps']);
        },
        false);
    </script>

The pipeline throughput is still governed by the slowest stage,
so the throughput is same as before. (25)

.. raw:: html

   <div id="multi2-2"></div>

    <script>
    document.addEventListener(
        'DOMContentLoaded',
        function() {
            loadAndPlotQueueStats(
                'multi2-2',
                '../_static/data/example_simu_multi2.json',
                ['put_time', 'get_time', 'occupancy']);
        },
        false);
    </script>

**Put time**

The queue put time is 0 everywhere in the pipe except source.
This is because the bottleneck stage is at the very beginning of the pipeline,
queues are empty so the entire pipeline is starving for the data.

**Get time**

Since the pipeline is starving everywhere, the queue get time is high.
It corresponds to the ``iteration time - task execution time of the next stage``.
So, it is 0 ms (40 - 40) for the ``src_queue``,
15 ms (40 - 25) for the ``40ms_queue``,
30 ms (40 - 10) for the ``25ms_queue``,
40 ms (40 - 0) for the ``10ms_queue``,
and 10 ms (40 - 30) for the ``sink_queue``.

**Occupancy**

The queue occupancy drops to 0% at the stage 1, and subsequent stages are not able to recover
even though their data processing speed is faster than previous stages.

Introducing the Concurrency
~~~~~~~~~~~~~~~~~~~~~~~~~~~

So far, we have looked at pipelines without concurrency, but in real use cases,
stages are executed with concurrency.

Let's see how adding concurrency changes the statistics.

We use the fast-to-slow example from the previous 3-stage pipeline,
and set the concurrency of the slowest stage to 2.

.. code-block:: python

   pipeline = (
       PipelineBuilder()
       .add_source(...)
       .pipe(op1)
       .pipe(op2)
       .pipe(op3, concurrency=2)
       #          ^^^^^^^^^^^^^^
       .add_sink(...)
       .build(...)
   )

.. mermaid::

   graph LR
       subgraph Pipeline
           direction LR
           subgraph S1[Stage1]
               O1[Op 1]
               Q1@{ shape: das, label: "Queue 1" }
           end
           subgraph S2[Stage2]
               O2[Op 2]
               Q2@{ shape: das, label: "Queue 2" }
           end
           subgraph S3[Stage3]
               O31[Op 3]
               O32[Op 3]
               Q3@{ shape: das, label: "Queue 3" }
           end
           Source:::invisible
           --> |Get| O1
           --> |Put| Q1
           --> |Get| O2
           --> |Put| Q2
           Q2 --> |Get| O31 --> |Put| Q3
           Q2 --> |Get| O32 --> |Put| Q3
           --> |Get| Sink:::invisible
       end

       style S1 stroke-width:3px
       style S2 stroke-width:3px
       style S3 stroke-width:3px

       style Source fill-opacity:0, stroke-opacity:0
       classDef invisible display:none;

.. raw:: html

    <div id="multi3-1"></div>

    <script>
    document.addEventListener(
        'DOMContentLoaded',
        function() {
            loadAndPlotQueueStats(
                'multi3-1',
                '../_static/data/example_simu_multi3.json',
                ['task_time', 'qps']);
        },
        false);
    </script>

The task execution time does not change, but since there are
two tasks of stage 3 executed concurrently, the stage 3 is twice faster.
The stage 3 is now faster than the speed of the foreground consumer.
Therefore, the throughput of the pipeline has been improved,
and now the bottleneck is the foreground consumer.

.. raw:: html

    <div id="multi3-2"></div>

    <script>
    document.addEventListener(
        'DOMContentLoaded',
        function() {
            loadAndPlotQueueStats(
                'multi3-2',
                '../_static/data/example_simu_multi3.json',
                ['put_time', 'get_time', 'occupancy']);
        },
        false);
    </script>

**Put Time**

For the source, the stage 1 and the stage 2, there is no change.
The put time corresponds to ``iteration time - task execution time``.
That is 30 ms (30 - 0) for the source, 20 ms (30 - 10) for the stage 1,
5 ms (30 - 25) for the stage 2.

For the stage 3, it is slightly different from the others.
The stage 2 is producing data at 30 ms / iteration.
The stage 3 can now consume the data at 20 ms / iteration on average.
So each invocation of the stage 3 tasks has to wait for 10 ms on average.
But since now there are two tasks calling ``put`` operation, so the total
wait time becomes 20 ms.

**Get Time**

The queue get time is now all 0 because all the stages on average are
operating faster than the foreground consumer.

**Occupancy**

Now that the bottleneck at stage 3 is resolved, the pipeline is not
data-starved, and the queue occupancy is 100% everywhere.

Concurrency beyond optimal setting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

What happens if we increase the concurrency to a value beyond the optimal
pipeline behavior?

The following is the queue performance statistics of the same pipeline
but the concurrency of stage 3 is set to 3.

.. raw:: html

    <div id="multi4-2"></div>

    <script>
    document.addEventListener(
        'DOMContentLoaded',
        function() {
            loadAndPlotQueueStats(
                'multi4-2',
                '../_static/data/example_simu_multi4.json',
                ['put_time', 'get_time', 'occupancy']);
        },
        false);
    </script>

With 3 concurrent tasks, the stage 3 can process the data at 13 (40/3)
ms on average.
So each task is blocked about 17 ms on average.
In the reported stats, the queue get time is summed up, so now we see
the value of queue put time for the stage 3 is 51 ms.

.. note::

   You may wonder why SPDL does not take the average get/put time when
   the concurrency is more than 1.

   This is just an implementation detail.
   The queue stats measurement is implemented by extending the
   :py:class:`~asyncio.Queue` class.
   And by the nature of async function, it is not aware of the concurrent
   method calls.

   Passing the information of concurrency does not provide a benefit despite
   the increased complexity.

Aggregation
-----------

Now we introduce ``aggregate`` step to the pipeline and
see how it affects the performance stats.

We look at the following pipeline that performs some preprocessing before aggregation.
We change the size of the batch to simulate the case where the
pipeline is faster than the consumer and otherwise.

.. code-block:: python

   pipeline = (
       PipelineBuilder()
       .add_source(...)
       .pipe(op1)
       .aggregate(batch_size)
       #^^^^^^^^^^^^^^^^^^^^^
       .add_sink(...)
       .build(...)
   )

.. mermaid::

   graph LR
       subgraph Pipeline
           direction LR
           subgraph S1[Stage1]
               O1[Op 1]
               Q1@{ shape: das, label: "Queue 1" }
           end
           subgraph S2[Aggregate]
               O2[[aggregate]]
               Q2@{ shape: das, label: "Queue 2" }
           end
           Source:::invisible
           --> |Get| O1
           --> |Put| Q1
           --> |Get| O2
           --> |Put| Q2
           --> |Get| Sink:::invisible
       end

       style S2 stroke-width:3px

       style Source fill-opacity:0, stroke-opacity:0
       classDef invisible display:none;

Optimal
~~~~~~~

Let's look at a case where the aggregation is fast enough.
We set the task average time to 6 ms and the aggregation size to 4.
So it takes the pipeline 24 ms to create a batch, which is
faster than the consumer's speed of 30 ms.

.. raw:: html

    <div id="agg-11"></div>

    <script>
    document.addEventListener(
        'DOMContentLoaded',
        function() {
            loadAndPlotQueueStats(
                'agg-11',
                '../_static/data/example_simu_agg1.json',
                ['task_time', 'qps']);
        },
        false);
    </script>

Same as the previous cases, the pipeline is operating at the speed of the
consumer, so the QPS at the sink and aggregation is 33 (~ 1000 / 30).

When an aggregation is involved, the apparent throughput changes.
This is because SPDL pipeline is not aware of the semantics of the operations
it is executing.
SPDL only counts the number of items emitted from the pipeline stages.
Aggregation step emits the data only when the batch is full.
In this case, the throughput of the aggregation step is
the throughput of the pipeline.
The stages before the aggregation have throughput higher than the aggregation
(multiplied by the batch size), but they essentially mean the same.

.. raw:: html

    <div id="agg-12"></div>

    <script>
    document.addEventListener(
        'DOMContentLoaded',
        function() {
            loadAndPlotQueueStats(
                'agg-12',
                '../_static/data/example_simu_agg1.json',
                ['put_time', 'get_time', 'occupancy']);
        },
        false);
    </script>

**Put Time**

The pipeline is operating at the pace of 30 ms / iteration.
But now each batch contains 4 items.
So before the aggregation stage, on average one sample is processed every 7.5 ms.

We see this in the source queue put value.
The source is putting data to the queue every 7.5 ms.
Then stage 1 takes 6 ms to process each.
So stage 1 waits for 1.5 ms until its queue has space.

The sink stage has no operation overhead, so it spends time
only on putting data to the queue.
So the queue put time for the sink stage is 30 ms, which is same as
the speed of the foreground consumption.

The aggregation stage is a bit different and more elaborate.
The aggregation stage is also producing data every 30 ms.
It starts the put operation when all the underlying data arrives.
This is supposed to take 24 ms (6 * 4), but
the queue itself has 2 slots for buffering and the task that is attempting to
put the next item serves as extra buffering, so when the aggregation
stage completes and starts to collect the next batch, only one task is required.
Since one task of stage 1 takes 6 ms, the aggregation batch can have the
next batch ready in around 6 ms, and it starts the next put effort, then
gets blocked for 24 ms.


**Get Time**

Since the pipeline is operating at the optimal speed, the get time is 0
almost everywhere.

However the get time for ``6ms_queue`` is an exception.
This is the time that the aggregation stage tries to fetch data from stage 1.
As mentioned in the **Put Time** section, the aggregation stage needs to
wait for one complete cycle of stage 1 when fetching one batch (4 items).
So the corresponding queue get time is around 1.5 ms (~ 6 / 4).

**Occupancy**

There is something unique about the occupancy of the aggregation step.
Even though the pipeline is producing data at a speed optimal for the
foreground consumer, stage 1 does not always have the queue full.

This is because the aggregation step drains the queue when it starts
creating the next batch.

In this case, while the aggregation step is putting a batch to its queue,
stage 1 can produce 3 items. 2 of them are put in the queue, and the
last one is held in the blocked attempt to put the next item to the queue.
When the aggregation step completes putting the batch in its queue, it
starts to collect the next batch.
It will quickly pick up the 3 already processed items, which makes the queue
empty.
Then the aggregate will take the 4th item to complete a batch and start
putting it in its queue.
This keeps the stage 1 queue empty for 6 ms.
Stage 1 now processes the next item and puts it in the queue,
which takes another 6 ms.

So in the cycle of 30 ms, there are about 12 ms of window that the queue is
empty.
This is the ~57% occupancy we see in the above figure.


Non-optimal
~~~~~~~~~~~

When we increase the batch size to 6, the pipeline no longer produces the data
at the speed that the foreground consumer needs.
It takes the pipeline 36 ms to create a batch,
which is slower than the rate of the consumption. (30 ms)

.. raw:: html

    <div id="agg-21"></div>

    <script>
    document.addEventListener(
        'DOMContentLoaded',
        function() {
            loadAndPlotQueueStats(
                'agg-21',
                '../_static/data/example_simu_agg2.json',
                ['qps']);
        },
        false);
    </script>

The throughput is a little below 27.7 (~ 1000/36 ms) after the aggregation,
and 6 times that before the aggregation.

.. raw:: html

    <div id="agg-22"></div>

    <script>
    document.addEventListener(
        'DOMContentLoaded',
        function() {
            loadAndPlotQueueStats(
                'agg-22',
                '../_static/data/example_simu_agg2.json',
                ['put_time', 'get_time', 'occupancy']);
        },
        false);
    </script>

Since the pipeline is starving for data everywhere other than the source,
the put time is 0 and the occupancy is 0%.

The get time mostly corresponds to ``Pipeline iteration time - stage task execution time``.

Summary
-------

Throughout this guide, we've explored how fast and slow stages interact and how these interactions
manifest in queue statistics. The key relationships are:

**Fast-to-Slow Stages:**

When a stage is faster than its downstream consumer, data accumulates in the queue between them.
This results in high put time (the fast stage waits for queue space), low get time (data is always available),
and high queue occupancy (typically near 100%).

**Slow-to-Fast Stages:**

When a stage is slower than its downstream consumer, the downstream stage starves for data.
This results in low put time (queue space is always available), high get time (downstream waits for data),
and low queue occupancy (typically near 0%).

The Effect of Concurrency
~~~~~~~~~~~~~~~~~~~~~~~~~~

Increasing concurrency has the effect of virtually reducing the task execution time of a stage.
If a stage takes 40 ms per task but runs with concurrency of 2, it can effectively process data at 20 ms per batch.
This allows you to improve pipeline throughput by parallelizing slow stages without changing the underlying operation.
However, the reported queue statistics (put/get times) are summed across all concurrent tasks, not averaged,
so you need to account for the concurrency level when interpreting these values.

Special Considerations for Aggregation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When using aggregate operations, the queue occupancy of the stage before aggregation may not reach 100%
even when the pipeline is running at optimal speed. This is because the aggregation stage drains the
queue when collecting a batch, creating periods where the queue is temporarily empty while waiting for
the next batch to be assembled. This behavior is normal and does not indicate a performance problem
as long as the overall pipeline throughput meets the foreground consumer's requirements.

Real-World Complexity
~~~~~~~~~~~~~~~~~~~~~

The simulations in this guide assume idealized conditions with consistent task execution times and no resource contention.
Real-world use cases are more complex due to:

- **Limited CPU resources:** Multiple stages competing for CPU cores can affect execution times
- **Network bandwidth constraints:** Data fetching operations may experience variable latency and throughput
- **Noisy neighbor phenomenon:** Other processes on the same machine can interfere with pipeline performance (see :doc:`noisy_neighbour`)
- **Other resource contentions:** Memory pressure, disk I/O, GPU utilization, etc.

These factors mean that the background data loading pipeline and the foreground inference/training loop
are not independent. Resource contention between them can create performance characteristics that differ
from the idealized simulations. When optimizing real pipelines, consider monitoring system resource utilization
alongside the queue statistics to identify bottlenecks caused by resource contention rather than
just computational complexity.
