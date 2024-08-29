Frequently Asked Questions
==========================

How to work around GIL?
-----------------------

In Python, GIL (Global Interpreter Lock) practically prevents the use of multi-threading, however extension modules that are written in low-level languages, such as C, C++ and Rust, can release GIL when executing operations that do not interact with Python interpreter.

Many libraries used for dataloading release the GIL. To name a few;

- Pillow
- OpenCV
- Decord
- tiktoken

Typically, the bottleneck of model training in loading and preprocessing the media data.
So even though there are still parts of pipelines that are constrained by GIL,
by taking advantage of preprocessing functions that release GIL,
we can achieve high throughput.

What if a function does not release GIL?
----------------------------------------

In case you need to use a function that takes long time to execute (e.g. network utilities)
but it does not release GIL, you can delegate the stage to subprocess.

:py:meth:`spdl.dataloader.PipelineBuilder.pipe` method takes an optional ``executor`` argument.
The default behavior of the ``Pipeline`` is to use the thread pool shared among all stages.
You can pass an instance of :py:class:`concurrent.futures.ProcessPoolExecutor`,
and that stage will execute the function in the subprocess.

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
   serialized and passed to the subprocess. Therefore, the function to be executed
   must be a plain function. Closures and class methods cannot be passed.


Why Async IO?
-------------

When training a model with large amount of data, the data are retrieved from remote locations. Network utilities often provide APIs based on Async I/O.

The Async I/O allows to easily build complex data preprocessing pipeline and execute them while automatically parallelizing parts of the pipline, achiving high throughput.

Synchronous operations that release GIL can be converted to async operations easily by running them in a thread pool. So by converting the synchronous preprocessing functions that release GIL into asynchronous operations, the entire data preprocessing pipeline can be executed in async event loop. The event loop handles the scheduling of data processing functions, and execute them concurrently.
