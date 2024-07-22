Building and Running Pipeline
=============================

.. py:currentmodule:: spdl.dataloader

First, let's look at how easy it is to build the pipeline in SPDL.

The following snippet demonstrates how one can construct a
:py:class:`Pipeline` object using a :py:class:`PipelineBuilder` object.

.. code-block::

   >>> from spdl.dataloader import PipelineBuilder
   >>>
   >>> def source():
   ...     for i in range(10):
   ...         yield i
   ...
   >>> async def double(i: int):
   ...     return 2 * i
   ...
   >>> async def plus1(i: int):
   ...     return i + 1
   ...
   >>> pipeline = (
   ...     PipelineBuilder()
   ...     .add_source(range(12))
   ...     .pipe(double)
   ...     .pipe(plus1)
   ...     .aggregate(3)
   ...     .add_sink(3)
   ...     .build()
   ... )


The resulting :py:class:`Pipeline` object contains all the logic to
perform the operations in an async event loop in the background thread.

To run the pipeline, call :py:meth:`Pipeline.start`.
Once the pipeline starts executing, you can iterate on the pipeline.
Finally call :py:meth:`Pipeline.stop` to stop the background thread.

.. code-block::

   >>> pipeline.start()
   >>>
   >>> for item in pipeline:
   ...     print(item)
   [1, 3, 5]
   [7, 9, 11]
   [13, 15, 17]
   [19, 21, 23]
   >>> pipeline.stop()

It is important to call :py:meth:`Pipeline.stop`.
Forgetting to do so will leave the background thread running,
leading to the situation where Python interpreter gets stuck at exit.

In practice, there is always a chance that data processing raises an error,
so there is a context manager :py:meth:`Pipeline.auto_stop` to make sure that
pipeline is stopped.

.. code-block::

   >>> with pipeline.auto_stop():
   ...    for item in pipeline:
   ...        print(item)

.. note::

   Once :py:meth:`Pipeline.stop` method is called, the ``Pipeline`` object is unusable.
   To pause and resume the execution, simply keep the reference around until the
   next use.
