.. _data-formats-case-study:

Data Format and Performance
===========================

.. admonition:: tl;dr
   :class: tip

   If you are using NPZ format as data storage, we recommend to take the following
   changes

   - Avoid using compression if possible.
     (Avoid using :py:func:`numpy.savez_compressed`,
     instead use :py:func:`numpy.save` or :py:func:`numpy.savez`)
   - Only store numerical arrays. Do not store objects, such as metadata dictionary.
     Load numerical archives with ``allow_pickle=False``.
   - Switch to multi-threading and use :py:func:`spdl.io.load_npz` or
     :py:func:`spdl.io.load_npy`.

.. note::

   The benchmark script used in this section is found at :py:mod:`benchmark_numpy`
   example.

We often see teams work on data collection bundle multiple arrays with and metadata in
``NPZ`` format.

.. code-block::

   # Data to save
   data = {
     'image': <ARRAY>,
     'segmentation': <ARRAY>,
     'metadata': <DICTIONARY>,
     'timestamp': <TIMESTAMP>,
   }

   # Save data to remote service as NPZ format
   handle = get_handle_for_remote("my_bucket://my_data.npz")
   np.savez(handle, **data)

Values passed to :py:func:`numpy.savez` are stored as arrays in ``NPY`` †
format. Object arrays, such as an array containing a dictionary, serialize
their Python objects with :py:mod:`pickle`.
Then NPZ format uses `ZIP <https://en.wikipedia.org/wiki/ZIP_(file_format)>`_
to bundle the serialized objects.
As such, the NPZ format supports saving generic Python objects, so it makes it very
easy to save whatever data created in data collection stage.

However, the NPZ format is not a performant solution when bulk-processing them.

.. code-block::

   # Load data from NPZ format
   raw_data = download_from_remote("my_bucket://my_data.npz")
   data = np.load(BytesIO(raw_data))

In this section we look at issues loading NPZ files and discuss alternative solutions.

.. seealso::

   † :py:mod:`numpy.lib.format` for the detail of the NPY format.

The Performance of ``numpy.load`` on NPZ data
---------------------------------------------

When reading entries returned by :py:func:`numpy.load`, Python archive and
object-handling work can contend for the GIL.† In this benchmark, adding
threads reduced throughput.
The following figure shows how the performance of ``numpy.load`` function scale
with multi-processing (``MP``) and multi-threading (``MT``) when loading NPZ files.

.. raw:: html

   <div id='npz_basic'></div>

The multi-processing can improve the throughput to certain degree.
Its peak is at 4 workers.
As we see later, even when we use a function that is significantly
faster, the throughput of multi-processing is similar.
So the bottleneck likely is at the inter-process communication
(see :ref:`ipc-cost`).

With multi-threading, the throughput is highest when there is only one worker.
The performance decreases as more workers are used.
GIL contention in the read path can explain this pattern, but it does not
mean every operation in the reader holds the GIL.

When we used ``numpy.load`` with multi-threading,
it degraded the training speed.
(See the Practical Example section bellow.)

As described in the :ref:`noisy-neighbour` section,
the training speed is governed by whether the CPU can
schedule the GPU kernel launch at timely manner.
When the GIL is held in the background thread of the main process,
the training loop has to wait for the GIL to be released
before it can launch the next GPU kernel.
This slows down the training.

.. admonition:: † The implementation of ``numpy.load`` function
   :class: note

   The NPZ format uses Zip format to bundle multiple items with optional compression.
   The :py:func:`numpy.load` uses the Python's :py:mod:`zipfile` module to parse the
   archive and exposes its entries lazily; accessing an entry reads its array.
   Object dtypes require the :py:mod:`pickle` path. Python work in the archive
   and object-handling steps needs the GIL, but native I/O, decompression and
   array-reading routines need not hold it for the entire operation.

Changing the data format
------------------------

So what can we do to improve the data loading performance?
There are couple of possibilities.

#. Write a faster version of the :py:func:`numpy.load` function.
#. Change the data format to something else that is more performant for loading.

Rewriting the ``numpy.load`` function with 100% feature compatibility requires
high engineering efforts/maintanance cost.
As mentioned previously, the ``numpy.load`` function is implemented with
the combination of ``zipfile`` and ``pickle``.
There are many libraries we can leverage to rewrite the ``zipfile``
as an extension module (so that we can release the GIL), but ``pickle``
by nature, cannot be implemented without interacting the Python Interpreter.
If we can limit the data type to be those types that can be loaded with
``allow_pickle=False``, (such as integers and floats, but not Python's ``object`` type)
then it becomes more feasible.
We explore this option in the next section.

Generally speaking, changing the data format is one of the most versatile
way to improve the data loading performance.
There are many data formats, and they are optimized for different properties,
such as serialization performance, deserialization performance,
storage (transfer).
If you switch to one that is performant for loading,
it improves the training pipeline performance.
You need to analyze the trade off (such as cost of additional conversion job,
cost of extra data storage and network transfer).

When changing the data format, what data format is most suitable?
The answer depends on the requirements of the pipeline, but generally,
you do not want to use a format that is exotic and too specific.
You want to keep using the format that is easy to handle.

Some formats that are as accessible as NPZ in ML domain include
NPY format and PyTorch's serialization format.
The following plot shows how their performance scale.

.. raw:: html

   <div id='npz_2'></div>

What is interesting is that all solutions exhibit similar performance
when using multi-processing.
This suggests that when using multi-processing, the bottleneck is in the data
copying at the boundary of the main process and the worker processes.

If you want to keep bundling multiple data into one file, using PyTorch's
serialization format in multi-threading can give you a boost.
PyTorch's serialization format also uses the ZIP format
to bundle multiple objects, but the part that parses the ZIP format
is implemented in C++ and releases the GIL,
which helped the threaded comparison shown here.

If you want to squeeze the last bit of the performance, switching to
NPY format gives a little bit more performance.
If you were using :py:func:`numpy.savez` as opposed to
:py:func:`numpy.savez_compressed`, the storage space would not change
as much.
You can get rid of the time otherwise spend on processing the ZIP.

Loading data from memory
------------------------

The :py:func:`numpy.load` function expects the input to implement
the file-like object interface.
It does not support directly interpreting a byte string as array.
When a serialized data is in byte string, it must be wrapped by
the :py:class:`io.BytesIO` class.
However, this makes ``numpy.load`` to branch to a slower path††, in which
the data is read incrementally with intermediate copies.†††

To achieve high throughput with multi-threading, it is important to reduce
the number of data copies.
Reducing the memory copy also helps reducing the time the function holds the GIL.

For this purpose, we implemented the :py:func:`spdl.io.load_npy` and
:py:func:`spdl.io.load_npz` functions. For supported NPY payloads and stored
NPZ entries, they can interpret the original bytes as arrays without copying
the array data. Compressed NPZ entries require decompression into new storage.
These functions do not support object arrays and are not full replacements
for ``numpy.load``. See :doc:`../../io/numpy_arrays` for their format and
backing-buffer requirements.

These function are faster than the ``numpy.load`` function,
even though it does not release the GIL entirely.
The following plot shows this.

.. raw:: html

   <div id='npz_all'></div>

.. note::

   The relevant code is found at the following locations.

   †† `isfileobj <https://github.com/numpy/numpy/blob/v2.2.0/numpy/lib/format.py#L999>`_
   function, which returns ``False`` for ``io.BytesIO`` object.

   ††† `The branch point <https://github.com/numpy/numpy/blob/v2.2.0/numpy/lib/format.py#L830-L853>`_
   where the data is copied and processed chunk by chunk.

Practical Example
-----------------

The following plot shows how the choice of data format and IO function affects the
performance of a training pipeline.

.. raw:: html

   <div id='npz_exp'></div>

- The ``NPZ, multi-processing (Baseline)`` is the original implementation,
  which is based on TorchData's ``StatefulDataLoader``.
  It loads NPZ files in subprocesses. We have seen throughput decline as
  training progresses in multiple runs using ``StatefulDataLoader``. Compare
  throughput over time rather than relying only on an early measurement.
- The ``Upperbound`` is the estimated maximum throughput obtained by using
  :py:class:`~spdl.dataloader.CacheDataLoader`.
  (See :ref:`headspace-analysis` for details.)
- The ``NPZ, multi-threading`` uses SPDL and calls ``numpy.load`` in
  a background thread of the main process.
- The ``NPY, multi-threading`` uses SPDL and uses ``spdl.io.load_npy`` function
  in the background.
  The data has been reformatted from NPZ to NPY.

As you can see, loading NPZ files with ``numpy.load`` function in multi-threading
slows down the training speed,
even though when it was faster with multi-threading when we benchmarked
the function itself.

Switching from NPZ to NPY required a one-time conversion job. The resulting
pipeline was faster than the baseline in this case. Because the comparison
changed both the format and the reader, it does not isolate each one's effect.

.. include:: ../../plots/data_format.txt
