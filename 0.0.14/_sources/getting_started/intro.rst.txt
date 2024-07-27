Building and Running Pipeline
=============================

.. py:currentmodule:: spdl.pipeline

First, let's look at how easy it is to build the pipeline in SPDL.

The following snippet demonstrates how one can construct a
:py:class:`Pipeline` object using a :py:class:`PipelineBuilder` object.

.. code-block::

   >>> from spdl.pipeline import PipelineBuilder
   >>>
   >>> pipeline = (
   ...     PipelineBuilder()
   ...     .add_source(range(12))
   ...     .pipe(lambda x: 2 * x)
   ...     .pipe(lambda x: x + 1)
   ...     .aggregate(3)
   ...     .add_sink(3)
   ...     .build(num_threads=1)
   ... )


The resulting :py:class:`Pipeline` object contains all the logic to
perform the operations in an async event loop in a background thread.

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

In practice, there is always a possibility that the application is
interrupted for unexpected reasons.
To make sure that the pipeline is stopped, it is recommended to use
:py:meth:`Pipeline.auto_stop` context manager, which calls
``Pipeline.start`` and ``Pipeline.stop`` automatically.

.. code-block::

   >>> with pipeline.auto_stop():
   ...    for item in pipeline:
   ...        print(item)

.. warning::

   Do not call :py:func:`iter` on the pipeline because :py:meth:`Pipeline.stop`
   might not be called at the right time.

   Say you wrap a ``Pipeline`` to create an class that resembles conventional
   ``DataLoader``.

   .. code-block:: python

      class DataLoader:
          ...

          def __iter__(self):
              with self.pipeline.auto_stop():
                  for item in pipeline:
                      yield item

      dataloader = DataLoader(...)

   Make sure to use this class like the following.
   This way, the context manager properly calls ``Pipeline.stop`` when
   the execution flow goes out of the loop, even
   when the application is exiting with unexpected errors.

   .. code-block:: python

      for item in dataloader:
          ...

   Do not use it like the following. This way, the ``Pipeline.stop``
   does not get called until the garbage collector deletes the object,
   which might cause deadlock.

   .. code-block:: python

      iterator = iter(dataloader)
      item = next(iterator)


.. note::

   Once :py:meth:`Pipeline.stop` method is called, the ``Pipeline`` object is unusable.
   To pause the execution, simply stop consuming the output.
   The ``Pipeline`` will get blocked when the internal buffers are full.
   To resume the execution, resume consuming the data.
