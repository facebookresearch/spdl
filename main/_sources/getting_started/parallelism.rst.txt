Parallelism
===========

.. currentmodule:: spdl.pipeline

The :py:class:`Pipeline` supports multi-threading and multi-processing.
You can also use a ``Pipeline`` objects as source iterator of another ``Pipeline``.
When experimenting, this flexibility makes it easy to switch multi-threading,
multi-processing and mixtures of them.

Specifying an executor
----------------------

The core mechanism to deploy concurrency is :py:meth:`asyncio.loop.run_in_executor`
method. The synchronous function (or generator) provided to ``Pipeline``
is executed asynchronously using the ``run_in_executor`` method.

When you provide a synchronous function (or generator), the ``PipelineBuilder``
internally converts it to an asynchronous equivalent using the
``run_in_executor`` method.

In the following snippet, an ``executor`` argument is provided when constructing
the ``Pipeline``.

.. code-block::

   executor: ThreadPoolExecutor | ProcessPoolExecutor | None = ...

   def my_func(input):
       ...

   pipeline = (
       PipelineBuilder()
       .add_source(...)
       .pipe(my_func, executor=executor)
       .add_sink(...)
       .build(...)
   )

Internally, the ``my_func`` function is converted to an asynchronous equivalent,
meaning it's dispatched to the provided executor (or a default one if the executor
is ``None``) as follow.

.. code-block::

   async asynchronous_my_func(input):

       loop = asyncio.get_running_loop()
       coroutine = loop.run_in_executor(executor, my_func, input)
       return await coroutine

Multi-threading (default)
-------------------------

If you build a pipeline without any customization, it defaults to
multi-threading.

The event loop dispatches the tasks to the default
:py:class:`~concurrent.futures.ThreadPoolExecutor`
created with the maximum concurrency specified in :py:meth:`PipelineBuilder.build`
method.

.. note::

   Before your application can take advantage of free-threaded Python,
   to properly achieve the concurrency, your stage functions must mainly
   consists of functions that release the GIL.

   Libraries such as PyTorch and NumPy release the GIL when manipulating
   arrays, so they are usually fine.

   For loading raw byte strings into array format, SPDL offers efficient
   functions through :py:mod:`spdl.io` module.


Multi-threading (custom)
------------------------

There are cases where you want to use a dedicated thread for certain task.

#. You need to maintain a state across multiple task invocations.
   (caching for faster execution or storing the application context)
#. You want to specify a different number of concurrency.

One notable example that comports with these conditions is transferring a
data to the GPU.
Due to the hardware constraints, only one data transfer can be performed
at a time.
To transfer data without interrupting the model training,
you need to use a stream object dedicated for the transfer, and you want
to keep using the same stream object across multiple function invocations.

To maintain a state, you can either encapsulate it in a callable class
instance, or put it in a
`thread-local storage <https://docs.python.org/3/library/threading.html#thread-local-data>`_.
The following example shows how to initialize and store a CUDA stream
in a thread-local storage.

.. code-block:: python

   import threading


   THREAD_LOCAL = threading.local()

   def _get_threadlocal_stream(index: int) -> tuple[torch.cuda.Stream, torch.device]:
       if not hasattr(THREAD_LOCAL, "stream"):
           device = torch.device(f"cuda:{index}")
           THREAD_LOCAL.stream = torch.cuda.Stream(device)
           THREAD_LOCAL.device = device
       return THREAD_LOCAL.stream, THREAD_LOCAL.device

The following code illustrates a way to transfer data using the same dedicated stream
across function invocations.

.. code-block:: python

   def transfer_data(data: list[Tensor], index: int = 0):
       stream, device = _get_threadlocal_stream(index)
       with torch.cuda.stream(stream):
           data = [
               t.obj.pin_memory().to(device, non_blocking=True)
               for t in data]
       stream.synchronize()
       return data

Now we want to run this function in background, but we want to use only one thread,
and keep using the same thread.
For this purpose we create a ``ThreadPoolExecutor`` with one thread and pass it to
the pipeline.

.. code-block:: python

   transfer_executor = ThreadPoolExecutor(max_workers=1)

   pipeline = (
       PipelineBuilder()
       .add_source(...)
       .pipe(...)
       .pipe(transfer_data, executor=transfer_executor)
       .add_sink(...)
   )

This way, the transfer function is always executed in a dedicated thread, so that
it keeps using the same CUDA stream.

When tracing this pipeline with
`PyTorch Profiler <https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html>`_,
we can see that it is always the one background thread that issues data transfer,
and the transfer overlaps with the stream executing the model training.

.. image:: ../../_static/data/parallelism_transfer.png

Multi-processing (stage)
------------------------

Similar to the custom multi-threading, by providing an instance of
:py:class:`~concurrent.futures.ProcessPoolExecutor`, that stage is executed in a
subprocess.

.. code-block::

   executor = ProcessPoolExecutor(...)

   pipeline = (
       PipelineBuilder()
       .add_source(...)
       .pipe(task_function, executor=executor)
       .add_sink(...)
   )

Note that when you dispatch the stage to subprocess, both the function (callable)
and the argument are sent from the main process to the subprocess.
Then the result obtained by passing the argument to the function is sent back
from the subprocess to the main process.
Therefore, all of the function (callable), the input argument and the output
value must be
`picklable <https://docs.python.org/3/library/pickle.html#pickle-picklable>`_.

If you want to bind extra arguments to a function, you can use
:py:func:`functools.partial`.
If you want to pass around an object that's not picklable by default,
you can define the serialization protocol by providing
:py:meth:`object.__getstate__` and :py:meth:`object.__setstate__`.

Multi-processing (combined)
---------------------------

If you have multiple stages that you want to run in subprocess, it is
inefficient to copy data between processes back and forth.

One workaround is to combine stages and let each process run processes
in a batch.

.. code-block::

   def preprocess(items: list[T]) -> U:
       # performs decode/preprocess and collation
       ...

   executor = ProcessPoolExecutor(...)

   pipeline = (
       PipelineBuilder()
       .add_source(...)
       .aggregate(batch_size)
       .pipe(preprocess, executor=executor, concurrency=...)
       .add_sink(...)
   )

This approach is similar to the conventional DataLoader.
One downside with this approach is less robust in error handling than
the previous approaches. If preprocessing fails for one item, and
if you want to ensure the size of the batch to be consistent,
then all items must be dropped too.
The other approach does not suffer from this.

Multi-threading in subprocess
-----------------------------

The multi-threading in subprocess is a paradigm we found effective when
`multi-threading degrades the performance <../case_studies/parallelism.html>`_.

The :py:func:`spdl.pipeline.run_pipeline_in_subprocess` function moves the given
instance of :py:class:`PipelineBuilder` to a subprocess, build and execute the
:py:class:`Pipeline` and put the results to inter-process queue.
(There is also :py:func:`spdl.pipeline.iterate_in_subprocess` function for
running a generic :py:class:`Iterable` object in subprocess.)

The following example shows how to use the function.

.. code-block:: python

   # Construct a builder
   builder = (
       spdl.pipeline.PipelineBuilder()
       .add_source(...)
       .pipe(...)
       ...
       .add_sink(...)
   )

   # Move it to the subprocess, build the Pipeline
   iterator = run_pipeline_in_subprocess(builder)

   # Iterate
   for item in iterator:
       ...

The MTP mode helps the OS to schedule GPU kernel launches from the main thread
(where the training loop is running) in timely manner, and reduces the
number of Python objects that the Python interpreter in the main process
has to handle.
