Data Format and Performance
===========================

We often see teams work on data collection bundle multiple arrays with and metadata in
`NPZ <https://numpy.org/doc/2.2/reference/generated/numpy.savez.html>`_ format.

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

Arrays passed to the :py:func:`np.savez` function is serialized into byte strings as
`NPY format <https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html>`_.
Other objects are serialized using Python's :py:mod:`pickle` module.
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

.. note::

   The benchmark script used in this section is found at :py:mod:`data_formats`
   example.

The Performnce of ``numpy.load`` on NPZ data
--------------------------------------------

The :py:func:`numpy.load` function holds the GIL.†
So concurrently calling it with multi-threading degrades the performance.
The following figure shows how the performance of ``numpy.load`` function scale
with multi-processing and multi-threading when loading NPZ files.

.. raw:: html

   <div id='npz_basic'></div>

The multi-processing can improve the throughput to certain degree.
Its peak is at 4 workers.
As we see later, even when we use a function that is significantly
faster, the throughput of multi-processing is similar.
So the bottleneck likely is at the inter-process communication.

With multi-threading, the througput is highest when there is only one worker.
The performance decreases as more workers are used.
This is because the ``numpy.load`` holds the GIL.

Even though the performance does not scale with multi-threading,
it appears faster than multi-processing.
This makes us want to conclude that mult-threading is the answer.
However, when we used ``numpy.load`` with multi-threading,
it degraded the training speed.
(See the Practical Example section bellow.)

As described in the `Noisy Neighbour <../optimization_guide/noisy_neighbout.html>`_
section, the training speed is governed by whether the CPU can
schedule the GPU kernel launch at timely manner.
When the GIL is held in the background thread of the main process,
the training loop has to wait for the GIL to be released
before it can launch the next GPU kernel.
This slows down the training.

.. admonition:: † The implementation of ``numpy.load`` function
   :class: note

   The NPZ format uses Zip format to bundle multiple items with optional compression.
   The :py:func:`numpy.load` uses the Python's :py:mod:`zipfile` module to parse the
   raw byte string, then deserialize each item.
   When deserializing, if the object is not NumPy NDArray type, then it resorts to
   the :py:mod:`pickle` module.

   Since the ``zipfile`` and ``pickle`` modules are pure-Python packages, and the
   rest of ``numpy.load`` function is also written in Python without an extension module,
   the entire process of NPZ deserialization (``numpy.load``) holds the GIL.

Changing the data format
------------------------

So what can we do to improve the data loading performance?
There are couple of possibilities.

#. Rewrite the :py:func:`numpy.load` as an extention module (with C/C++/Rust) and make sure that the GIL is released.
#. Change the data format to something else that is more performant for loading.

The first approach comes with non-trivial (actually pretty high) engineering/maintanance
cost, yet it still comes with some limitations.
If the data needs to be deserialized with ``pickle``, there is no way to release the GIL. 
Prohibiting ``pickle`` can help this, but if an existing data collection pipeline relies
on this, that part has to be changed.
If one takes this approach, the benefit must be clear and huge.

The second approach of changing the data format is generally applicable,
but takes some task-specific considerations.
Practically, switching to a data format that is optimized for loading
always work and it is much easier than optimizing the data loader architecture.
So if you are at the initial stage of optimization, we highly recommend
considering this approach.

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
so it is faster than ``numpy.load`` in processing the ZIP.

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
It does not support directly interpretating a byte string as array.
When a serialized data is in byte string, it must be wrapped by
the :py:class:`io.BytesIO` class.
However, this makes ``numpy.load`` to branch to a slower path††, in which 
the data is read incrementally with intermediate copies.†††

To achieve high throughput with multi-threading, it is important to reduce
the number of data copies.
Reducing the memory copy also helps reducing the time the function holds the GIL.

For this purpose, we implemented the :py:func:`spdl.io.load_npy` function
which takes a byte string of serialized Numpy NDArray,
and interpret it as an array object without creating a copy.

This function is much faster than loading NPY files with the
``numpy.load`` function, even though it still holds the GIL.
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
  which is based on TorchData's ``StatefulDataLaoder``.
  It loads NPZ files in subprocesses.
  (The performance of ``StatefulDataLoader`` is known to decay as the training progress.)
- The ``Upperbound`` is the estimated maximum througput obtained by using
  :py:class:`~spdl.dataloader.CacheDataLoader`.
  (See the
  `Headspace Analysis <../optimization_guide/headspace_analysis.html>`_
  for the detail.)
- The ``NPZ, multi-threading`` uses SPDL and calls ``numpy.load`` in
  a background thread of the main process.
- The ``NPY, multi-threading`` uses SPDL and uses ``spdl.io.load_npy`` function
  in the background.
  The data has been reformated from NPZ to NPY.

As you can see, loading NPZ files with ``numpy.load`` function in multi-threading
slows down the training speed,
even though when it was faster with multi-threading when we benchmarked
the function itself.

Switching dataset format from NPZ to NPY takes required some efforts.
It required one-time conversion job, but the resulting pipeline is faster than
the baseline.

.. include:: ../plots/data_format.txt
