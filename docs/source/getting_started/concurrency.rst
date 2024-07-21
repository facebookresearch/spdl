Concurrency
===========

.. py:currentmodule:: spdl.dataloader

The pipelines we looked at so far process data sequentially.
Now let's introduce concurrency to the pipeline so that it finishes jobs faster.

There are two parameters that affects the pipeline performance.

1. Stage concurrency
2. Thread pool size

Stage Concurrency
-----------------

The stage concurrency can be configured with ``concurrency`` argument in the :py:meth:`PipelineBuilder.pipe` method.

This argument determines at most how many operations of the stage the event loop would schedule at a given time.

.. important::

   Please note that **scheduling multiple tasks concurrently does not necessarily mean
   all of them are executed concurrently.** The execution of scheduled tasks is subject to
   the availability of resources required for the execution.

   See the :ref:`Thread Pool Size<Thread Pool Size>` for the detail.

For example, let's say we have a pipeline that downloads data and preprocess them, and we implement it like the following.

.. code-block::

   pipeline = (
       PipelineBuilder()
       .add_source(url_generator)
       .pipe(download, concurrency=4)
       .pipe(preprocess, concurrency=2)
       .add_sink(3)
       .build()
   )

The ``download`` stage will schedule 4 tasks, and wait for any of the tasks to complete. When a task is completed, the stage will schedule another task with new input data.

Similarly, ``preprocess`` stage will schedule 2 tasks and when a task is completed, it will schedule another task.

The following diagram illustrates this.

.. mermaid::

   flowchart TD
       A[url_generator]
       subgraph B[download]
           b1[download 1]
           b2[download 2]
           b3[download 3]
           b4[download 4] 
       end
       subgraph C[preprocess]
           c1[preprocess 1]
           c2[preprocess 2]
       end
       subgraph D[sink]
           d1[result 1]
           d2[result 2]
           d3[result 3]
       end
       A --> B
       B --> C
       C --> D

.. note::

   When the stage concurrency is bigger than 1, the results of the operations are,
   by default, put to the output queue in the order of task completion.
   Therefore the order of the processed items can change.

   This behavior can be changed by specifing ``output_order="input"`` in
   :py:meth:`PipelineBuilder.pipe` method, so that the order of the output is same
   as the input.

Thread Pool Size
----------------

Async event loop uses :py:class:`~concurrent.futures.ThreadPoolExecutor` to execute
synchronous functions as async functions.

When executing functions that are not natively asynchronous, the event loop can offload
its execution to the thread pool and wait for its completion.
This is what :py:meth:`~asyncio.loop.run_in_executor` does and it is a primal
way to execute synchronous functions in async event loop.

The majority of operations performed in ML dataloading are synchronous, so we need to
use this mechanism to run them in asyncronous context.
Or in an alternative view, the event loop acts as a surrogate who manages the thread pool
and does all the scheduling and inter/intra op parallelization.

The size of thread pool can be specified with ``num_threads`` argument in the
:py:meth:`PipelineBuilder.build` method.

The size of the thread pool serves as the capacity that pipeline can execute synchronous
fnunctions concurrently. Therefore, if concurrently scheudling multiple tasks of
synchronous operations, the size of thread pool must be bigger than the number of
the concurrency.

The following code snippet illustrates this.

.. code-block::

   def preprocess(data):
       """A hypothetical preprocessing function. (not async)"""
       ...

   async def async_preprocess(data):
       """Run the `preprocess` function asynchronously."""
       loop = asyncio.get_running_loop()
       return await loop.run_in_executor(None, preprocess, data)

   pipeline = (
       PipelineBuilder()
       .add_source(source)
       .pipe(async_preprocess, concurrency=3)
       # Run at most 3 `preprocess` functions concurrently.
       .add_sink(3)
       .build(num_threads=3)
       # Use 3 threads in the thread pool to accomodate 3 async_preprocess
   )

.. note::

   Note that there are cases where the stage concurrency and
   thread pool size are irrelevant.

   For example, some libraries implement thread-based parallelism in
   low-level language like C++. When using such libraries the concurrency
   is constraint by the resource managed by the library.
