Implementation detail of Pipeline
=================================

This note is a memorandum of the design trade offs and their consequences encountered
during the development of the :py:class:`spdl.pipeline.Pipeline`.

Simply put, the data processing pipeline is an async functions executed in a background thread,
and the foreground fetches the processed data from the sink.

When implementing foreground/background components, it turned out that a subtle design choice in
one part constraints the design choices of other parts of the system,
and there are multiple constraints that must be met at the same time.

Architecture Overview
---------------------

The following diagram illustrates the key components of the Pipeline and their interactions:

.. mermaid::

   graph TB
       subgraph FG["Foreground Thread"]
           Client[Client Code]
       end

       subgraph BG["Background Thread"]
           EL[Event Loop]

           subgraph Pipeline["Pipeline"]
               Source[Source Stage]
               SQ1(Queue)
               Stage1[Processing Stage 1]
               SQ2(Queue)
               Stage2[Processing Stage 2]
               SQ3(Queue)
               Sink[Sink Stage]
               SinkQ(Sink Queue)

               Source -->|data| SQ1
               SQ1 --> Stage1
               Stage1 -->|data| SQ2
               SQ2 --> Stage2
               Stage2 -->|data| SQ3
               SQ3 --> Sink
               Sink -->|data| SinkQ
           end
       end

       subgraph TP["Thread Pool"]
           TPQ(Queue)
           Thread1["Thread 1<br/>(Task 1)"]
           Thread2["Thread 2<br/>(Task 2)"]
           Thread3["Thread 3<br/>(Task 3)"]
           ThreadN["Thread N<br/>(Task N)"]

           TPQ --> Thread1
           TPQ --> Thread2
           TPQ --> Thread3
           TPQ --> ThreadN
       end

       Client -.->|request data from sink queue| EL
       EL -.->|return data| Client

       Source -.->|delegate task| TPQ
       Stage1 -.->|delegate task| TPQ
       Stage2 -.->|delegate task| TPQ
       Sink -.->|delegate task| TPQ

       EL -->|manages| Pipeline
       EL -->|controls| TP

       style FG fill:#e1f5ff
       style BG fill:#fff4e1
       style Pipeline fill:#f0f0f0
       style TP fill:#d4edda
       style Client fill:#4a90e2
       style EL fill:#e67e22
       style SinkQ fill:#e74c3c
       style TPQ fill:#ffd700

The diagram shows:

- **Foreground Thread**: Contains the client code that requests data from the Pipeline
- **Background Thread**: Hosts the event loop and executes the Pipeline
- **Event Loop**: Manages the Pipeline execution and controls the thread pool
- **Thread Pool**: Executes tasks delegated from each Pipeline stage
- **Pipeline**: Consists of multiple stages (Source, Processing Stages, Sink) connected by queues
- **Tasks**: Multiple tasks (Task 1, Task 2, Task 3, ... Task N) executed by the thread pool for each stage
- **Data Flow** (solid arrows): Data flows through the Pipeline from Source → Processing Stages → Sink → Sink Queue
- **Task Delegation** (dashed arrows): Each Pipeline stage delegates tasks to the thread pool for execution
- **Data Request** (dashed arrows): Foreground thread requests data from the Sink Queue


Async IO
--------

The Pipeline is composed of async functions for the following reasons:

1. **Orchestration Requirements**: Building a pipeline with different concurrency levels and
   flexible structure requires an orchestrator to schedule tasks.
   Using bare :py:class:`concurrent.futures.ThreadPoolExecutor`
   (or :py:class:`concurrent.futures.ProcessPoolExecutor`) makes it difficult
   to cleanly implement the Pipeline abstraction.
   The async event loop provides the necessary orchestration capabilities.
   For details on the limitations of using bare ``ThreadPoolExecutor``
   for pipeline implementation, see :doc:`../async/index`.

2. **Integration and Parallelism**: The async context makes it easy to integrate
   network utilities, which are often async functions.
   Additionally, executing data processing functions in async context enables
   inter-op and intra-op parallelism.

Queue vs Async Queue as Buffer
------------------------------

The sink of the Pipeline is where the processed data are buffered. Pipeline runs in the background thread, so that the data are written to the sink in the background thread. They are fetched by the foreground thread. Therefore, the access to the sink must be thread-safe. In addition, pipeline is executed in async event loop, so it is ideal that the sink buffer supports async accessor natively.

Python has two types of queues. One is thread-safe :py:class:`queue.Queue` (sync queue) and the other is its async variant :py:class:`asyncio.Queue` (async queue).

The accessors of sync queue, :py:meth:`queue.Queue.get` and :py:meth:`queue.Queue.put`, are thread-safe, and they support blocking operations with timeout.
The accessors of async queue, :py:meth:`asyncio.Queue.get` and :py:meth:`asyncio.Queue.put`, are not thread-safe. They return coroutine which can be awaited. For the foreground thread to actually fetch the values from the queue, these coroutinues must be executed by the same async event loop that's running the pipeline. There are synchronous variant of these accessors, :py:meth:`asyncio.Queue.get_nowait` and :py:meth:`asyncio.Queue.put_nowait`, which can work without an event loop, but since they are not thread-safe, they can only be used when the pipeline is not running.

If we choose sync queue, reading from the foreground is straightforward because the its accessors are thread-safe, but writing to the queue can block the event loop.
If we choose async queue, writing to the queue is straightforward in an event loop, but reading from the foreground is convoluted, because the access must be thread-safe, and if the loop is running and the Pipeline is still writing the queue, then the read access must use async operation as well.

From the perspective of the apparent code simplicity, :py:class:`queue.Queue` requires less code to write, however, having the blocking :py:meth:`queue.Queue.put` call in event loop makes it impossible to cleanly stop the background thread. This is because the synchronous blocking call blocks the event loop, and prevents the loop from processing cancellation request.

For this reason, we use :py:class:`asyncio.Queue` in the :py:class:`spdl.pipeline.Pipeline`. As a result, the implementation of :py:meth:`spdl.pipeline.Pipeline.get_item` becomes a bit convoluted. The next section explains why it is the case.

Thread, loop and task
---------------------

In implementing :py:class:`spdl.pipeline.Pipeline`, there are several object states that need to be carefully managed. They are

- The state of the background thread which runs the event loop.
- The state of the async event loop managed by the background thread.
- The state of the pipeline task, which process data and puts in the sink buffer.

When the foreground thread attempts to fetch data from sink buffer, which is an async queue, it must use the different API (sync vs async accessor) to get the data, depending on the state of the state of the pipeline execution. This is because when the pipeline is running, the pipeline puts data in the async queue, and the event loop controls its execution. To access the async queue in cooperative manner, the foreground has to issue a request to run fetch coroutine (:py:meth:`asyncio.Queue.get`) to the background thread and wait for the result. However if the event loop is not running, then this request to run the fetch coroutine will never be fulfilled. Therefore, if the event loop is not running, the foreground must use sync accessor (:py:meth:`asyncio.Queue.get_nowait`).

Another thing to consider is how to run the event loop. The foreground attempts to fetch data, the fetch request must be made via :py:func:`asyncio.run_coroutine_threadsafe`, so the system needs access to the loop object. In general, however, it is recommended not to manage loop object explicitly i.e. :py:meth:`asyncio.loop.run_forever` or :py:meth:`asyncio.loop.run_until_complete`). Instead it is encouraged to use :py:func:`asyncio.run`. But if we simply pass the pipeline coroutine to the :py:func:`asyncio.run` function, as soon as the task completes, the event loop is stopped and closed. We would like to encapsulate the event loop in the background thread and abstract away from the foreground thread. But this way, the foreground thread cannot know if the loop is running or not.

Following the above considerations, the implementation of the pipeline executions follows the following constraints.

1. To make the state management simpler, overlap the life cycle of the background thread and the event loop.

- When the thread is started, the control flow is not returned to the foreground thread until the event loop is initialized.
- The thread is stopped when the event loop is stopped.

2. Detach the life cycle of pipeline task from that of the event loop.

- Keep the event loop alive after the pipeline task is completed.
- Wait for the explicit request to stop the loop.

3. The event loop signals the object that manages the background thread that the task is completed.

Following the above constraints, the foreground can decide whether it should use sync or async accessor.

- If the background thread is not started. → Fail
- If the task is completed. → Use sync API
- Otherwise, the task is running. →  use async API.

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


Building Pipeline from Configuration
-------------------------------------

The process of converting a :py:class:`~spdl.pipeline.defs.PipelineConfig` into an executable :py:class:`~spdl.pipeline.Pipeline` is handled by :py:func:`spdl.pipeline.build_pipeline`. This section explains the multi-step transformation process.

Overview of the Build Process
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The build process can be summarized as follows:

1. **Config to Node Graph Conversion**: The :py:class:`~spdl.pipeline.defs.PipelineConfig` is converted into a linked list of :py:class:`~spdl.pipeline._components._node._Node` objects, forming a directed acyclic graph (DAG) without branching.

2. **Recursive Node Traversal**: Starting from the sink node, the upstream nodes are recursively traversed.

3. **Coroutine Creation**: Each node is converted into a coroutine that processes a stream of input data. Each coroutine completes when it receives an EOF (End-of-File) token.

4. **Main Coroutine Assembly**: A main coroutine is created to monitor the state of all stage coroutines.

5. **Error Handling and Cleanup**: The main coroutine is responsible for handling failures, performing cleanup (ensuring upstream stages are completed or cancelled when a downstream node completes), and reacting to interrupt requests from the foreground client code.

6. **Event Loop Execution**: The main coroutine is executed by the event loop running in the background thread.

Detailed Build Steps
~~~~~~~~~~~~~~~~~~~~~

Step 1: Configuration to Node Graph
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :py:class:`~spdl.pipeline.defs.PipelineConfig` represents the pipeline structure declaratively. The function :py:func:`~spdl.pipeline._components._node._convert_config` transforms this configuration into a linked list of :py:class:`~spdl.pipeline._components._node._Node` objects.

Each :py:class:`~spdl.pipeline._components._node._Node` is linked with references to its upstream :py:class:`~spdl.pipeline._components._node._Node` objects, forming a directed graph. Each node also has a queue object (:py:class:`~spdl.pipeline.AsyncQueue`) that will be used when building coroutines to buffer data between stages.

The following diagram illustrates how a simple pipeline configuration is converted into a node graph:

.. mermaid::

   graph LR
       subgraph PConfig["PipelineConfig"]
           PC_Src[SourceConfig]
           PC_P1["PipeConfig: Stage1"]
           PC_P2["PipeConfig: Stage2"]
           PC_Sink[SinkConfig]

           PC_Src --> PC_P1
           PC_P1 --> PC_P2
           PC_P2 --> PC_Sink
       end

       subgraph NGraph["Node Graph"]
           N_Src["Node: Source"]
           N_P1["Node: Stage1"]
           N_P2["Node: Stage2"]
           N_Sink["Node: Sink"]

           N_Src -.->|queue| N_P1
           N_P1 -.->|queue| N_P2
           N_P2 -.->|queue| N_Sink

           N_P1 -->|upstream| N_Src
           N_P2 -->|upstream| N_P1
           N_Sink -->|upstream| N_P2
       end

       PC_Src -.->|converts to| N_Src
       PC_P1 -.->|converts to| N_P1
       PC_P2 -.->|converts to| N_P2
       PC_Sink -.->|converts to| N_Sink

       style PC_Src fill:#e1f5ff
       style PC_P1 fill:#e1f5ff
       style PC_P2 fill:#e1f5ff
       style PC_Sink fill:#e1f5ff
       style N_Src fill:#fff4e1
       style N_P1 fill:#fff4e1
       style N_P2 fill:#fff4e1
       style N_Sink fill:#fff4e1

Step 2: Recursive Coroutine Creation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The function :py:func:`~spdl.pipeline._components._node._build_node_recursive` traverses
the node graph recursively starting from the sink node and creates a coroutine for each node.
The traversal follows the upstream references, ensuring that all upstream coroutines are created
before the downstream coroutine.

For each node type, it calls :py:func:`~spdl.pipeline._components._node._build_node` to
create a coroutine for the corresponding config.
Each coroutine processes input data in a loop and completes when it receives an EOF token.

Step 3: Main Coroutine Assembly
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The function :py:func:`~spdl.pipeline._components._node._run_pipeline_coroutines` creates the main coroutine that orchestrates the execution of all stage coroutines. This main coroutine:

1. Creates asyncio Tasks from each node's coroutine
2. Monitors the tasks using :py:func:`asyncio.wait` with ``FIRST_COMPLETED`` strategy
3. When a task completes, cancels orphaned upstream tasks
4. Handles cancellation requests from the foreground thread
5. Gathers errors from failed tasks and raises :py:class:`~spdl.pipeline.PipelineFailure` if any task failed

The following diagram illustrates the execution flow of the main coroutine:

.. mermaid::

   flowchart TD
       Start([Main Coroutine Starts]) --> CreateTasks[Create Tasks from All Nodes]
       CreateTasks --> StartTasks[Start All Tasks]
       StartTasks --> Wait{Wait for<br/>First Completion}

       Wait -->|Task Completed| CancelOrphans[Cancel Orphaned<br/>Upstream Tasks]
       Wait -->|Cancellation Request| CancelAll[Cancel All<br/>Pending Tasks]

       CancelOrphans --> CheckPending{Any Tasks<br/>Pending?}
       CancelAll --> WaitAll[Wait for All<br/>Tasks to Complete]

       CheckPending -->|Yes| Wait
       CheckPending -->|No| GatherErrors[Gather Errors<br/>from Tasks]

       WaitAll --> RaiseCancelled[Raise<br/>CancelledError]

       GatherErrors --> HasErrors{Any Errors?}
       HasErrors -->|Yes| RaiseFailure[Raise<br/>PipelineFailure]
       HasErrors -->|No| Complete([Complete Successfully])

       style Start fill:#90EE90
       style Complete fill:#90EE90
       style RaiseFailure fill:#FFB6C1
       style RaiseCancelled fill:#FFB6C1

Step 4: Task Creation and Monitoring
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When the main coroutine starts, it recursively creates asyncio Tasks for each node
starting from the sink and traversing upstream.
Each task wraps the node's coroutine and begins executing immediately.

The main coroutine then enters a loop monitoring these tasks:

- It uses :py:func:`asyncio.wait` with ``return_when=FIRST_COMPLETED`` to wait for
  any task to complete
- When a task completes, it cancels upstream tasks that would become orphaned producers
  (i.e., tasks that would keep producing data that will never be consumed)
- If the main coroutine receives a cancellation request, it cancels all pending tasks and
  waits for them to complete before re-raising the :py:class:`asyncio.CancelledError`

Step 5: Error Handling and Cleanup
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The main coroutine implements a comprehensive error handling and cleanup strategy:

**Successful Execution**:
When all tasks complete successfully, each stage passes EOF to its output queue
before completing.
The sink filters out EOF and does not pass it to the output queue.
The main coroutine then gathers any errors (should be none) and completes successfully.

**Failures at Some Stage**:
When a task fails:

1. The failed stage should pass EOF to its output queue before exiting (even when failing)
2. The main coroutine detects the completion and cancels all upstream tasks to prevent orphaned producers
3. Downstream tasks continue processing remaining items in their queues and eventually complete
4. The main coroutine waits for all tasks to complete
5. The main coroutine gathers errors and raises :py:class:`~spdl.pipeline.PipelineFailure`

**Cancellation**:
When the foreground thread requests cancellation:

1. The event loop signals cancellation to the main coroutine
2. The main coroutine catches :py:class:`asyncio.CancelledError`
3. The main coroutine cancels all pending tasks
4. The main coroutine waits for all tasks to complete (allowing graceful cleanup)
5. The main coroutine re-raises :py:class:`asyncio.CancelledError`

Step 6: Pipeline Object Creation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Finally, :py:func:`spdl.pipeline.build_pipeline` creates a :py:class:`~spdl.pipeline.Pipeline` object with:

- ``coro``: The main coroutine from :py:func:`~spdl.pipeline._components._node._run_pipeline_coroutines`
- ``queue``: The sink node's output queue (where processed data is buffered)
- ``executor``: A :py:class:`~concurrent.futures.ThreadPoolExecutor` for executing tasks
- ``desc``: A string description of the pipeline

The :py:class:`~spdl.pipeline.Pipeline` object manages the lifecycle of the background thread and
event loop as described in the previous sections.


API reference
-------------

The following functions and classes are the main components of Pipeline's implementation.

They are not public API, but listed here for developers who are interested in learning
how the :py:class:`~spdl.pipeline.Pipeline` is implemented.

.. py:currentmodule:: spdl.pipeline._components._node

.. autoclass:: _Node
   :members:

.. autofunction:: _convert_config

.. autofunction:: _build_node_recursive

.. autofunction:: _build_node

.. autofunction:: spdl.pipeline._components._source._source

.. autofunction:: spdl.pipeline._components._pipe._merge

.. autofunction:: spdl.pipeline._components._pipe._pipe

.. autofunction:: spdl.pipeline._components._pipe._ordered_pipe

.. autofunction:: spdl.pipeline._components._sink._sink

.. autofunction:: _run_pipeline_coroutines
