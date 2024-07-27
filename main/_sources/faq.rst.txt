Frequently Asked Questions
==========================

How to work around GIL?
-----------------------

In Python, GIL (Global Interpreter Lock) practically prevents the use of multi-threading, however extension modules that are written in low-level languages, such as C, C++ and Rust, can release GIL when executing operations that do not interact with Python interpreter.

Many libraries used for data loading release the GIL. To name a few;

- Pillow
- OpenCV
- Decord
- tiktoken

Typically, the bottleneck of model training in loading and pre-processing the media data.
So even though there are still parts of pipelines that are constrained by GIL,
by taking advantage of pre-processing functions that release GIL,
we can achieve high throughput.

What if a function does not release GIL?
----------------------------------------

In case you need to use a function that takes long time to execute (e.g. network utilities)
but it does not release GIL, you can delegate the stage to sub-process.

:py:meth:`spdl.dataloader.PipelineBuilder.pipe` method takes an optional ``executor`` argument.
The default behavior of the ``Pipeline`` is to use the thread pool shared among all stages.
You can pass an instance of :py:class:`concurrent.futures.ProcessPoolExecutor`,
and that stage will execute the function in the sub-process.

.. code-block::

   executor = ProcessPoolExecutor(max_workers=num_processes)

   pipeline = (
       PipelineBuilder()
       .add_source(src)
       .pipe(stage1, executor=executor, concurrency=num_processes)
       .pipe(stage2, ...)
       .pipe(stage3, ...)
       .add_sink(1)
       .build()
   )

This will build pipeline like the following.

.. include:: ./plots/faq_subprocess_chart.txt

.. note::

   Along with the function arguments and return values, the function itself is also
   serialized and passed to the sub-process. Therefore, the function to be executed
   must be a plain function. Closures and class methods cannot be passed.

.. tip::

   If you need to perform one-time initialization in sub-process, you can use
   ``initializer`` and ``initargs`` arguments.

   The values passed as ``initializer`` and ``initargs`` must be picklable.
   If constructing an object in a process that does not support picke, then
   you can pass constructor arguments instead and store the resulting object
   in global scope. See also https://stackoverflow.com/a/68783184/3670924.

   Example

   .. code-block::

      def init_resource(*args):
          global rsc
          rsc = ResourceClass(*args)

      def process_with_resource(item):
          global rsc

          return rsc.process(item)

      executor = ProcessPoolExecutor(
          max_workers=4,
          mp_context=None,
          initializer=init_resource,
          initargs=(...),
      )

      pipeline = (
          PipelineBuilder()
          .add_source()
          .pipe(
              process_with_resource,
              executor=executor,
              concurrency=4,
          )
          .add_sink(3)
          .build()
      )


Why Async IO?
-------------

When training a model with large amount of data, the data are retrieved from remote locations. Network utilities often provide APIs based on Async I/O.

The Async I/O allows to easily build complex data pre-processing pipeline and execute them while automatically parallelizing parts of the pipeline, achieving high throughput.

Synchronous operations that release GIL can be converted to async operations easily by running them in a thread pool. So by converting the synchronous pre-processing functions that release GIL into asynchronous operations, the entire data pre-processing pipeline can be executed in async event loop. The event loop handles the scheduling of data processing functions, and execute them concurrently.
