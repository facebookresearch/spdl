Running Synchronous Functions
=============================

The :py:mod:`asyncio` class has helper functions/class methods that allow to
execute a synchronous function in the async context.

- :py:func:`asyncio.wrap_future` - Wrap a ``concurrent.futures.Future`` object in a ``asyncio.Future`` object.
- :py:meth:`asyncio.loop.run_in_executor` - Schedule the provided function to be executed in an executor, and returns an ``asyncio.Future`` object.
- :py:func:`asyncio.to_thread` - A wrapper around ``run_in_executor``. Propagates the context variable.

This makes it possible to mix the execution of sync/async functions smoothly.
The event loop can be regarded as a manager for dispatching synchronous
execution to thread/process pool executor.

Using ``run_in_executor`` to run synchronous function in Async I/O
------------------------------------------------------------------

SPDL uses :py:meth:`asyncio.loop.run_in_executor` method extensively.

.. seealso::

   - :ref:`pipeline-parallelism` - Explains SPDL's high-level API for switching executor.

The following example shows how to convert a synchronous function into asynchronous function.

.. code-block::

   executor = ThreadPoolExecutor()  # or ProcessPoolExecutor

   async def async_wrapper(input):

       loop = asyncio.get_running_loop()
       return await loop.run_in_executor(executor, sync_func, input)

If using multi-threading, you can also use the default ``ThreadPoolExecutor`` attached to
the event loop †.

.. admonition:: † The maximum concurrency of the default executor.
   :class: Note

   The maximum concurrency of the default executor is as follow

   - ``min(32, os.cpu_count() + 4)`` (before Python 3.13)
   - ``min(32, (os.process_cpu_count() or 1) + 4)`` (since Python 3.13)

   These values are intended for I/O tasks, and it is rather high for data loading,
   which might involve CPU tasks like media processing.

   You can change this by explicitly setting the default executor with
   :py:func:`asyncio.loop.set_default_executor` function.

   **See also**

   - :py:class:`~concurrent.futures.ThreadPoolExecutor`
   - :ref:`noisy-neighbour` explains why it is important to always keep CPU utilization low.

The difference of multi-threading and multi-processing
------------------------------------------------------

There are difference between multi-threading and multi-processing.
The following table summarizes them.

.. list-table::
   :widths: 10 45 45
   :header-rows: 1

   * -
     - ``ThreadPoolExecutor``
     - ``ProcessPoolExecutor``
   * - Pros
     - - Faster launch and lighter data handling compared to ``ProcessPoolExecutor``.
     - - Free from the constraint imposed by the GIL. (Can use any function to achieve concurrency.)
   * - Cons
     - - The functions must not hold the GIL. (or hold the GIL for extremely short amount of time).
       - Potential data race (though samples are usually independent of each other in AI application).
     - - Data passed between processes must be picklable.
       - Data copy between processes. (Using shared memory can improve the performance.)
       - The start up can be slow if there are libraries that perform static initialization.

To achieve high performance in multi-threading, we need to work around the GIL.
SPDL provides :py:mod:`spdl.io` module, which offers efficient media processing while releasing the GIL.
It complements the numerical computation like NumPy and PyTorch, so many AI applications are covered.

Using ``SharedMemory`` for faster inter-process-communication
-------------------------------------------------------------

The multi-processing does not have the GIL constraint, but it comes with the cost of memory copy between processes.
Array formats like NumPy's NDArray and PyTorch's Tensor use shared memory to make this performant.
If you need to pass a large data between processes
(such as a dataset, though we don't recommend passing around dataset)
you can write it to a shared memory in the worker process, then let the main process read it from there.

.. code-block::

   from multiprocessing.shared_memory import SharedMemory

   # In worker process, write the result to a shared memory
   def serialize(obj):
       data = pickle.dumps(obj)
       shmem = SharedMemory(create=True, size=len(data))
       shmem.buf[:] = data
       return shmem

   # pass the name of the shared memory to the main process

   # then the main process load it from the shared memory
   def deserialize(name):
       shmem = SharedMemory(name=name)
       obj = pickle.loads(shmem.buf)
       return obj
