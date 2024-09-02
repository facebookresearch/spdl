Implemantation detail of Pipeline
=================================

.. py:currentmodule:: spdl.dataloader

Simply put, the data processing pipeline is an async functions executed in a background thread, and the foreground fetches the processed data from the sink.

When implementing foreground/background components, it turned out that a subtle design choice in one part constraints the design choices of other parts of the system, and there are multiple constraints that must be met at the same time.

This note is a memorandum of the design trade offs and their consequences encountered during the implementation of the :py:class:`Pipeline`.

Async IO
--------

The Pipeline is composed of async functions. This is to make it easy to integrate the network utilities which are often async functions. Also executing the data processing functions in async context makes it easy to realize inter-op and intra-op parallelism.

Queue vs Async Queue as Buffer
------------------------------

The sink of the Pipeline is where the processed data are buffered. Pipeline runs in the background thread, so that the data are written to the sink in the background thread. They are fetched by the foreground thread. Therefore, the access to the sink must be thread-safe. In addition, pipeline is executed in async event loop, so it is ideal that the sink buffer supports async accessor natively.

Python has two types of queues. One is thread-safe :py:class:`queue.Queue` (sync queue) and the other is its async variant :py:class:`asyncio.Queue` (async queue).

The accessors of sync queue, :py:meth:`queue.Queue.get` and :py:meth:`queue.Queue.put`, are thread-safe, and they support blocking operations with timeout.
The accessors of async queue, :py:meth:`asyncio.Queue.get` and :py:meth:`asyncio.Queue.put`, are not thread-safe. They return coroutine which can be awaited. For the foreground thread to actually fetch the values from the queue, these coroutinues must be executed by the same async event loop that's running the pipieline. There are synchronouse variant of these accessors, :py:meth:`asyncio.Queue.get_nowait` and :py:meth:`asyncio.Queue.put_nowait`, which can work without an event loop, but since they are not thread-safe, they can only be used when the pipeline is not running.

If we choose sync queue, reading from the foreground is straightforward because the its accessors are thread-safe, but writing to the queue can block the event loop.
If we choose async queue, wiriting to the queue is straightforward in an event loop, but reading from the foreground is convoluted, because the access must be thread-safe, and if the loop is running and the Pipeline is still writing the queue, then the read access must use async operation as well.

From the perspective of the apparent code simplicity, :py:class:`queue.Queue` requires less code to write, however, having the blocking :py:meth:`queue.Queue.put` call in event loop makes it impossible to cleanly stop the background thread. This is because the synchronous blocking call blocks the event loop, and prevents the loop from processing cancellation request.

For this reason, we use :py:class:`asyncio.Queue` in the :py:class:`Pipeline`. As a result, the implementation of :py:meth:`Pipeline.get_item` becomes a bit convoluted. The next section explains why it is the case.

Thread, loop and task
---------------------

In implementing :py:class:`Pipeline`, there are several object states that need to be carefully managed. They are

- The state of the background thread which runs the event loop.
- The state of the async event loop managed by the background thread.
- The state of the pipeline task, which process data and puts in the sink buffer.

When the foreground thread attempts to fetch data from sink buffer, which is an async queue, it must use the different API (sync vs async accessor) to get the data, depending on the state of the state of the pipeline execution. This is because when the pipeline is running, the pipeline puts data in the async queue, and the event loop controls its execution. To access the async queue in cooperative manner, the foreground has to issue a request to run fetch coroutine (:py:meth:`asyncio.Queue.get`) to the background thread and wait for the result. However if the event loop is not running, then this request to run the fetch coroutine will never be fullfilled. Therefore, if the event loop is not running, the foreground must use sync accessor (:py:meth:`asyncio.Queue.get_nowait`).

Another thing to consider is how to run the event loop. The foreground attempts to fetch data, the fetch request must be made via :py:func:`asyncio.run_coroutine_threadsafe`, so the system needs access to the loop object. In general, however, it is recommended not to manage loop object explicitly i.e. :py:meth:`asyncio.loop.run_forever` or :py:meth:`asyncio.loop.run_until_complete`). Instead it is encouraged to use :py:func:`asyncio.run`. But if we simply pass the pipeline coroutine to the :py:func:`asyncio.run` function, as soon as the task completes, the event loop is stopped and closed. We would like to encapsulate the event loop in the background thread and abstract away from the foreground thread. But this way, the foreground thread cannot know if the loop is running or not.

Following the above considerations, the implementation of the pipeline executions follows the following constraints.

- 1. To make the state management simpler, overlap the lifecycle of the background thread and the event loop.

  - a. When the thread is started, the control flow is not returned to the foreground thread unitl the event loop is initialized.
  - b. The thread is stopped when the event loop is stopped.

- 2. Detach the lifecycle of pipeline task from that of the event loop.

  - a. Keep the event loop alive after the pipeline task is completed.
  - b. Wait for the explicit request to stop the loop.

- 3. The event loop signals the object that manages the background thread that the task is completed.

Following the above constraints, the foreground can decide whether it should use sync or async accessor.

- If the background thread is not started. -> Fail
- If the task is completed. -> Use sync API
- Othewise, the task is running. ->  use async API.

The following sequence diagram summarizes the interaction between the foreground thread, the background thread, the event loop and the pipeline task.

.. mermaid::

   sequenceDiagram
       FG Thread   ->>+ BG Thread: Start BG Thread

       create participant Event Loop
       BG Thread   ->>  Event Loop: Start Event loop
       Event Loop  ->>  BG Thread: Event loop initialized
       BG Thread   ->>- FG Thread: Return

       create participant Task
       Event Loop  ->>  Task: Start Task
       FG Thread  --)+  BG Thread: Q: "Is task started?"
       BG Thread  --)-  FG Thread: A: "Not yet."
       Event Loop -->>  BG Thread: Signal task start
       FG Thread  --)+  BG Thread: Q: "Is task started?"
       BG Thread  --)-  FG Thread: A: "Yes it is started."
       FG Thread  --)+  BG Thread: Q: "Is task completed?"
       BG Thread  --)-  FG Thread: A: "Not yet."

       destroy Task
       Task        ->>  Event Loop: Task completed
       Event Loop -->>  BG Thread: Signal task completion
       FG Thread  --)+  BG Thread: Q: "Is task completed?"
       BG Thread  --)-  FG Thread: A: "Yes it is completed."
       Event Loop  ->>  Event Loop: Keep event loop alive
       FG Thread   ->>+ BG Thread: Request stop event loop
       BG Thread  -->>  Event Loop: Signal Stop
       BG Thread   ->>- FG Thread: Return without waiting for the loop stop

       destroy Event Loop
       Event Loop  ->>  BG Thread: Loop Stopped
       FG Thread   ->>+ BG Thread: Join thread
       BG Thread   ->>- FG Thread: Return

If the foreground thread decides to stop the pipeline before its completion, the
event loop will cancel the pipeline task, (in turn the pipeline task will cancel
tasks correspond to pipeline stages) then the foreground thread will wait for the
background thread to complete the loop and join.


.. mermaid::

   sequenceDiagram
       FG Thread  ->>+ BG Thread: Start BG Thread

       create participant Event Loop
       BG Thread  ->>  Event Loop: Start Event Loop
       Event Loop ->>  BG Thread: Event loop initialized
       BG Thread  ->>- FG Thread: Return

       create participant Task
       Event Loop  ->>  Task: Start Task
       Event Loop -->>  BG Thread: Signal task start
       FG Thread   ->>+ BG Thread: Request stop event loop
       BG Thread  -->>  Event Loop: Signal Stop
       BG Thread   ->>- FG Thread: Return without waiting for the loop stop
       Event Loop -->>  Task: Signal Stop

       destroy Task
       Task ->> Event Loop: Task cancelled

       destroy Event Loop
       Event Loop ->> BG Thread: Loop Stopped
       FG Thread  ->>+ BG Thread: Join thread
       BG Thread  ->>- FG Thread: Return
