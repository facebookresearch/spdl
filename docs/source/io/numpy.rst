NumPy Array
===========

.. seealso::

   - :ref:`data-formats-case-study`: 

Often, data collection teams use NumPy serialization formats,
such as NPY and NPZ (with optional compression),
to store training data.

One can use :py:func:`numpy.load` function to load such data.
However, this function is almost entirely implemented in Python,
so under the constraint of the GIL,
it is difficult to process bulk of them concurrently.
Also when loading an array from a byte string, this function processes the data chunk-by-chunk, and copyies the whole byte string in the end.

The :py:func:`spdl.io.load_npy` function loads an array from byte string by re-interpreting the data, without creating a copy.

The :py:func:`spdl.io.load_npz` function can load NPZ format file.
Unlike the :py:func:`numpy.load` function, it does not support Python object that requires pickling.
Under the hood, it takes a portition of the given byte string (without copying), and pass it to :py:func:`~spdl.io.load_npy`.

Additionally, if the NPZ data is compressed, :py:func:`~spdl.io.load_npy` decompresses the only required part, then use :py:func:`~spdl.io.load_npy` to load array data.
When decompressing the data, it uses `libdeflate <https://github.com/ebiggers/libdeflate?tab=readme-ov-file>`_, so we can expect that the speed of the decompression itself is faster.
