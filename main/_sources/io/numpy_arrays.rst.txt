Loading NumPy Arrays
====================

NumPy serialization formats (NPY, NPZ) are often used in data collection.
The :py:func:`numpy.save`, :py:func:`numpy.savez`, and :py:func:`numpy.savez_compressed`
functions are versatile—they can serialize almost any data, which is why they are a popular choice.
However, loading such data with :py:func:`numpy.load` is not performant, and this is difficult to optimize.

SPDL provides optimized functions for loading NumPy arrays from byte strings (data already in memory).

.. seealso::

   :doc:`../case_studies/data_format`
      Case study comparing different data serialization formats for performance and efficiency.

Overview
--------

NumPy's serialization format stores arrays in two formats:

- **NPY format** (``.npy`` files): Single array per file
- **NPZ format** (``.npz`` files): Multiple arrays in a ZIP archive

SPDL provides functions optimized for loading arrays from byte strings that are already in memory.
These are designed for data downloaded from remote storage, network APIs, or other sources
where the binary data is already loaded into RAM.

.. important::

   These functions work **exclusively with byte strings** (``bytes`` or ``bytearray``).
   They do **not** accept file paths or file-like objects (e.g., ``BytesIO``).
   They are optimized for scenarios where data is already loaded into memory.

.. note::

   To load NumPy arrays with SPDL IO functions, the data must be serialized with ``allow_pickle=False``.
   This is the default behavior for :py:func:`numpy.save`, :py:func:`numpy.savez`, and
   :py:func:`numpy.savez_compressed` when saving numeric arrays.

**Key benefits:**

- **Works with byte strings**: Accepts ``bytes`` or ``bytearray`` objects directly
- **Fast array creation**: Creates NumPy arrays from in-memory data without computation
- **Zero-copy loading**: No intermediate copies for supported formats
- **Memory efficiency**: Direct memory mapping when possible
- **Optimized for in-memory data**: No file I/O overhead, works directly with downloaded data

**Performance characteristics:**

- Since the GIL is not released, performance does not scale in multi-threading
- However, these functions are faster than standard NumPy functions when working with in-memory data
  because they do not perform any computation and work directly with byte data

Loading NPY Files
------------------

:py:func:`spdl.io.load_npy` loads a single NumPy array from a byte string in NPY format.
The input must be ``bytes`` or ``bytearray``, not a file path or file-like object.

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   import spdl.io
   import numpy as np
   from io import BytesIO

   # Create and save an array
   original = np.arange(100).reshape(10, 10)
   buffer = BytesIO()
   np.save(buffer, original)

   # Load using SPDL
   data = buffer.getvalue()
   restored = spdl.io.load_npy(data)

   assert np.array_equal(restored, original)

Comparison with numpy.load
~~~~~~~~~~~~~~~~~~~~~~~~~~~

SPDL's implementation is more efficient than using :py:func:`numpy.load` with :py:class:`io.BytesIO`
when working with data already in memory:

.. code-block:: python

   import spdl.io
   import numpy as np
   from io import BytesIO

   # NumPy's approach (slower for in-memory data)
   data = buffer.getvalue()  # bytes object
   restored_numpy = np.load(BytesIO(data))  # Wraps bytes in file-like object

    # SPDL's approach (faster for in-memory data)
    restored_spdl = spdl.io.load_npy(data)  # Works directly with bytes

**Why SPDL is faster for in-memory data:**

1. **No intermediate BytesIO wrapper**: SPDL works directly with byte strings
2. **Zero-copy when possible**: Avoids unnecessary memory allocation
3. **No computation**: Creates array objects from in-memory data without processing
4. **Designed for bytes in memory**: Optimized for data already downloaded/loaded

**Note:** If you need to load from a file path, use :py:func:`numpy.load` directly.
SPDL functions are designed for byte strings, not file paths.

Zero-Copy Loading
~~~~~~~~~~~~~~~~~

By default, :py:func:`spdl.io.load_npy` returns a view into the original byte data without copying:

.. code-block:: python

   import spdl.io
   import numpy as np

   # Load without copying (default)
   data = bytearray(npy_bytes)
   array = spdl.io.load_npy(data)

   # Modifying the array affects the original data
   array[0] = 999
   # The underlying byte data is also modified

   # Force a copy if needed
   array = spdl.io.load_npy(data, copy=True)
   array[0] = 999
   # Now the original byte data is unchanged

.. warning::

   When using zero-copy mode (``copy=False``), the returned array shares memory with the input data.
   Ensure the input data remains valid for the lifetime of the array.

Supported Data Types
~~~~~~~~~~~~~~~~~~~~

:py:func:`spdl.io.load_npy` supports all numeric NumPy dtypes:

- **Integer types**: ``uint8``, ``int16``, ``uint16``, ``int32``, ``uint32``, ``int64``, ``uint64``
- **Floating point**: ``float16``, ``float32``, ``float64``
- **Boolean**: ``bool``

**Limitations:**

- **No object dtype support**: Arrays with ``dtype=object`` are not supported
- **No Fortran order**: Only C-contiguous arrays are supported

.. code-block:: python

   import spdl.io
   import numpy as np

   # Supported: numeric types
   int_array = np.array([1, 2, 3], dtype=np.int32)
   float_array = np.array([1.0, 2.0, 3.0], dtype=np.float64)

   # Not supported: object dtype
   # obj_array = np.array([{"key": "value"}], dtype=object)  # Will fail

Loading NPZ Files
------------------

:py:func:`spdl.io.load_npz` loads multiple NumPy arrays from a byte string containing NPZ (ZIP archive) data.
The input must be ``bytes`` or ``bytearray``, not a file path or file-like object.

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   import spdl.io
   import numpy as np
   from io import BytesIO

   # Create and save multiple arrays
   x = np.arange(10)
   y = np.sin(x)
   z = np.cos(x)

   buffer = BytesIO()
   np.savez(buffer, x=x, y=y, z=z)

   # Load using SPDL
   data = buffer.getvalue()
   npz_file = spdl.io.load_npz(data)

   # Access arrays by name
   assert np.array_equal(npz_file["x"], x)
   assert np.array_equal(npz_file["y"], y)
   assert np.array_equal(npz_file["z"], z)

NpzFile Interface
~~~~~~~~~~~~~~~~~

:py:func:`spdl.io.load_npz` returns a :py:class:`spdl.io.NpzFile` object that mimics :py:class:`numpy.lib.npyio.NpzFile`.

The :py:class:`~spdl.io.NpzFile` class implements the :py:class:`collections.abc.Mapping` interface:

.. code-block:: python

   import spdl.io
   import numpy as np

   npz_file = spdl.io.load_npz(data)

   # Dictionary-like access
   x = npz_file["x"]

   # Check if key exists
   if "x" in npz_file:
       print("x is in the archive")

   # List all arrays
   print(npz_file.files)  # ['x', 'y', 'z']

   # Iterate over keys
   for key in npz_file:
       print(f"{key}: {npz_file[key].shape}")

   # Get number of arrays
   print(len(npz_file))  # 3

Accessing Arrays
~~~~~~~~~~~~~~~~

Arrays can be accessed with or without the ``.npy`` suffix:

.. code-block:: python

   import spdl.io

   npz_file = spdl.io.load_npz(data)

   # Both work the same
   x1 = npz_file["x"]
   x2 = npz_file["x.npy"]

   assert np.array_equal(x1, x2)

Compressed NPZ Files
~~~~~~~~~~~~~~~~~~~~

:py:func:`spdl.io.load_npz` supports both uncompressed and DEFLATE-compressed archives:

.. code-block:: python

   import spdl.io
   import numpy as np
   from io import BytesIO

   x = np.arange(1000)
   y = np.random.random(1000)

   # Compressed NPZ (savez_compressed)
   buffer = BytesIO()
   np.savez_compressed(buffer, x=x, y=y)

   data = buffer.getvalue()
   npz_file = spdl.io.load_npz(data)

   assert np.array_equal(npz_file["x"], x)
   assert np.array_equal(npz_file["y"], y)

Positional Arguments
~~~~~~~~~~~~~~~~~~~~

Arrays saved without names get automatic ``arr_0``, ``arr_1`` naming:

.. code-block:: python

   import spdl.io
   import numpy as np
   from io import BytesIO

   x = np.arange(10)
   y = np.sin(x)

   # Save with positional arguments (no names)
   buffer = BytesIO()
   np.savez(buffer, x, y)

   data = buffer.getvalue()
   npz_file = spdl.io.load_npz(data)

   # Access using auto-generated names
   assert np.array_equal(npz_file["arr_0"], x)
   assert np.array_equal(npz_file["arr_1"], y)

Use Cases
---------

Loading from Remote Storage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These functions are specifically designed for loading data from remote storage or network APIs,
where the data is first downloaded into memory as bytes:

.. code-block:: python

   import spdl.io
   import numpy as np

   def load_from_s3(bucket: str, key: str) -> np.ndarray:
       # Download bytes from S3 into memory
       data = s3_client.get_object(Bucket=bucket, Key=key)["Body"].read()
       # `data` is now a bytes object in memory

       # Load efficiently with SPDL - works directly with the bytes
       return spdl.io.load_npy(data)

   def load_from_http(url: str) -> np.ndarray:
       # Download from HTTP endpoint
       response = requests.get(url)
       data = response.content  # bytes object

       # Load directly from the downloaded bytes
       return spdl.io.load_npy(data)

   # Use in data pipeline
   for key in data_keys:
       array = load_from_s3("my-bucket", key)
       # Process array...

**Why these functions are ideal for remote storage:**

- Data is already downloaded into memory as bytes
- No need to write to disk and read back
- Efficient conversion from bytes to NumPy arrays
- Optimized for this specific use case

Performance Considerations
---------------------------

GIL Behavior
~~~~~~~~~~~~

Both :py:func:`spdl.io.load_npy` and :py:func:`spdl.io.load_npz` do **not** release the GIL.
They create NumPy array objects from in-memory data without performing any computation.
Since the majority of time is spent on Python object creation, it is not possible to release the GIL.

**Implications:**

- **No multi-threading scalability**: Performance does not scale with multiple threads
- **Still faster than NumPy**: Despite not releasing the GIL, these functions are faster than
  standard NumPy functions because they avoid computation and work directly with byte data
- **Best for single-threaded or I/O-bound pipelines**: Use when data loading is not the bottleneck

Memory Usage
~~~~~~~~~~~~

Zero-copy mode (``copy=False``) is more memory-efficient but requires careful lifetime management:

.. code-block:: python

   import spdl.io

   # Memory-efficient: No copy
   array = spdl.io.load_npy(data, copy=False)
   # 'data' must remain valid while 'array' is in use

   # Memory-safe: Independent copy
   array = spdl.io.load_npy(data, copy=True)
   # 'data' can be deleted, 'array' is independent

When to Use
~~~~~~~~~~~

**Use SPDL's NumPy loaders when:**

- Working with **byte strings** downloaded from remote storage (S3, HTTP, etc.)
- Data is already in memory as ``bytes`` or ``bytearray``
- Loading from network APIs or cloud storage
- Memory efficiency is important (zero-copy loading)
- Performance is critical for in-memory data conversion

**Use standard numpy.load when:**

- Working with **file paths** on disk (``numpy.load('file.npy')``)
- Working with **file-like objects** (e.g., ``BytesIO``, file handles)
- Need support for object dtype
- Need support for Fortran-order arrays
- Working with advanced NumPy features

.. note::

   SPDL functions accept **only byte strings** (``bytes``/``bytearray``),
   not file paths or file-like objects. They are optimized for scenarios
   where data has already been downloaded or loaded into memory.


See Also
--------

- :doc:`basic` - High-level media loading functions
- :doc:`decoding_overview` - Understanding the decoding process
- :py:func:`numpy.save` - Save arrays to NPY format
- :py:func:`numpy.savez` - Save multiple arrays to NPZ format
- :py:func:`numpy.savez_compressed` - Save compressed NPZ files
