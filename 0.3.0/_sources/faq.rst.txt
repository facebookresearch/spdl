Frequently Asked Questions
==========================

How to work around GIL?
-----------------------

In Python, the GIL (Global Interpreter Lock) practically prevents the use of multi-threading. However, extension modules written in low-level languages such as C, C++, and Rust can release the GIL when executing operations that do not interact with the Python interpreter.

Many libraries used for data loading release the GIL. To name a few;

- Pillow
- OpenCV
- Decord
- tiktoken

Typically, the bottleneck in model training is loading and pre-processing media data.
Even though some parts of pipelines are constrained by the GIL,
we can achieve high throughput by using pre-processing functions that release the GIL.

What if a function does not release GIL?
----------------------------------------

In case you need to use a function that takes long time to execute (e.g. network utilities)
but it does not release GIL, you can delegate the stage to sub-process.

:py:meth:`spdl.pipeline.PipelineBuilder.pipe` method takes an optional ``executor`` argument.
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
   If constructing an object in a process that does not support pickle, then
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

Which functions hold the GIL?
-----------------------------

The following is the list of functions that we are aware that they hold the GIL.
So it is advised to use them with ``ProcessPoolExecutor`` or avoid using them in SPDL.

* `np.load <https://github.com/numpy/numpy/blob/maintenance/2.1.x/numpy/lib/_npyio_impl.py#L312-L500>`_: Please refer to :ref:`data-formats-case-study` for possible workaround.

Why Async I/O?
--------------

The Async I/O facilitates building complex pipeline.
Please refer to :ref:`intro-async` for more detail.
