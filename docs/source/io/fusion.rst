Fusing IO Calls
===============

.. warning::

   :py:func:`spdl.io.fuse` is experimental. Its behavior and interface may
   change in future releases.

:py:func:`spdl.io.fuse` is a decorator that executes the SPDL IO calls in a
function with fewer calls into the native code, typically one per function
call. It reduces the time that loading functions spend waiting for the GIL,
which can dominate the loading time in a process where other Python threads
are busy, such as the main process of a training job.

Why fuse IO calls
-----------------

Each SPDL IO function releases the GIL while it processes data in the native
code, so that other Python threads can run in the meantime. When it returns,
it has to re-acquire the GIL. If another thread is running Python code at
that moment, the function waits until the interpreter switches threads,
which can take several milliseconds (see :py:func:`sys.getswitchinterval`).

Loading one sample usually takes multiple calls. Even
:py:func:`~spdl.io.load_audio` internally opens a demuxer, demuxes, decodes
and converts the data, and it re-acquires the GIL after each of them.
Batch loading and reading metadata add even more calls. In a busy process,
waiting for the GIL after each of these calls adds up.

:py:func:`~spdl.io.fuse` records the SPDL IO calls in a function and executes
them in one native call, so the GIL is re-acquired only once.

The following table shows the time per call (in milliseconds) of loading
functions with small test media (Python 3.14, x86 server). "idle" is a
single thread, "busy" has another Python thread spinning in the background,
and "8 threads" is the wall time per call when 8 threads load concurrently.

.. list-table::
   :header-rows: 1

   * - Function
     - Condition
     - Eager
     - Fused
   * - Decode 32 PNG images and batch them
     - idle
     - 5.08
     - 4.64
   * -
     - busy
     - 891.31
     - 10.11
   * -
     - 8 threads
     - 2.75
     - 0.87
   * - Demux 4 windows of a video, decode, batch and read PTS
     - idle
     - 7.90
     - 7.93
   * -
     - busy
     - 88.31
     - 13.40
   * - :py:func:`~spdl.io.load_audio` and :py:func:`~spdl.io.to_numpy`
     - idle
     - 1.63
     - 1.71
   * -
     - busy
     - 37.57
     - 7.24

When nothing else competes for the GIL, fusing makes no difference, or adds
a small overhead for recording the calls. The benefit comes only from
re-acquiring the GIL fewer times. With free-threaded Python, which does not
have the GIL, eager calls are not slowed down by other threads, so
:py:func:`~spdl.io.fuse` does nothing there. (See
`Free-threaded Python`_.)

Using ``fuse``
--------------

Decorate the function that loads data with SPDL IO functions.

.. code-block:: python

   import spdl.io

   filter_desc = spdl.io.get_video_filter_desc(scale_width=224, scale_height=224)

   @spdl.io.fuse
   def load_images(srcs: list[bytes]) -> spdl.io.CPUBuffer:
       frames = [
           spdl.io.decode_packets(spdl.io.demux_image(src), filter_desc=filter_desc)
           for src in srcs
       ]
       return spdl.io.convert_frames(frames)

   buffer = load_images([data1, data2, data3])  # One call into the native code

The function runs as usual, but the SPDL IO functions do not execute
immediately. They record operations and return placeholders for their
results (packets, frames, buffers and their metadata). The recorded
operations are executed in one native call when the function returns, and
the placeholders in the returned value are replaced with the actual values.

Since the function itself runs as usual, Python code in it works as before,
including loops, list comprehensions, random sampling and attributes of
objects.

Using metadata in Python
------------------------

Metadata, such as ``packets.width``, ``frames.num_frames`` and
``frames.get_pts()``, is also recorded. It can be returned from the function
as-is.

When Python code needs an actual value, the operations recorded so far are
executed in one native call at that point. This happens, for example, when
branching on a value, computing a filter description from a value, or
operating on the array returned by :py:func:`~spdl.io.to_torch`.

.. code-block:: python

   @spdl.io.fuse
   def load_video(src: bytes) -> tuple[torch.Tensor, list[int]]:
       packets = spdl.io.demux_video(src, timestamp=(0, 2))
       width, height = packets.width, packets.height
       # `width // 2` needs the actual value. The first native call demuxes,
       # and fetches both width and height.
       filter_desc = spdl.io.get_video_filter_desc(
           scale_width=width // 2, scale_height=height // 2,
       )
       frames = spdl.io.decode_packets(packets, filter_desc=filter_desc)
       pts = frames.get_pts()
       # `to_torch` needs the actual buffer. The second native call decodes,
       # fetches the PTS, and converts the frames.
       tensor = spdl.io.to_torch(spdl.io.convert_frames(frames))
       return tensor.permute(0, 3, 1, 2), pts

Pending metadata is fetched together, so reading multiple values in a row
needs only one native call. To keep the number of native calls small, read
metadata in Python only where it is necessary, and return the rest.

What is recorded
----------------

The following functions are recorded.

- :py:class:`~spdl.io.Demuxer` (including demuxing multiple windows from one
  demuxer), :py:func:`~spdl.io.demux_audio`, :py:func:`~spdl.io.demux_video`,
  :py:func:`~spdl.io.demux_image` and :py:func:`~spdl.io.apply_bsf`
- :py:func:`~spdl.io.decode_packets` and :py:func:`~spdl.io.sample_decode_video`
- :py:func:`~spdl.io.convert_frames`, :py:func:`~spdl.io.to_numpy`,
  :py:func:`~spdl.io.to_torch`, :py:func:`~spdl.io.to_numba` and
  :py:func:`~spdl.io.to_jax`
- :py:func:`~spdl.io.load_audio`, :py:func:`~spdl.io.load_video`,
  :py:func:`~spdl.io.load_image`, :py:func:`~spdl.io.load_image_batch` and
  :py:func:`~spdl.io.get_filter_desc`
- The metadata of demuxers, codecs, packets and frames, ``clone()``, and
  indexing video frames with an integer, a slice or a list of integers.

Operations that are not recorded run as usual with the actual values. These
include streaming demuxing and decoding, transfer to CUDA devices, NVDEC
decoding, :py:class:`~spdl.io.BSF`, ``load_image_batch(strict=False)``, and
attributes that are not recorded. The recorded operations are executed
before them.

Fused loading functions
-----------------------

:py:func:`~spdl.io.load_audio`, :py:func:`~spdl.io.load_video`,
:py:func:`~spdl.io.load_image` and :py:func:`~spdl.io.load_image_batch` are
decorated with ``fuse(cache=True, sources=("src",))`` (``("srcs",)`` for
:py:func:`~spdl.io.load_image_batch`). Calling them directly executes their
IO operations in one native call, and calling them in a fused function
records their operations as part of the function. They depend only on their
arguments, so the recorded program is cached for each combination of the
arguments other than the source, e.g. each ``filter_desc``.

``load_image_batch(strict=False)`` decodes the images without fusion, so that
it can skip the images that fail to decode.

The following table shows the time per call of the loading functions called
directly (in milliseconds, Python 3.14, mean and 95% confidence interval of
10 trials of 10 calls; small generated test media). "busy" has another Python
thread spinning in the background, and "8 threads" is the wall time per call
when 8 threads load concurrently. "Without fusion" is the undecorated
function, and ``fuse`` is the function decorated without the cache, for
comparison.

.. list-table::
   :header-rows: 1

   * - Function
     - Condition
     - Without fusion
     - ``fuse``
     - ``fuse(cache=True)``
   * - ``load_audio``
     - idle
     - 1.59 ± 0.01
     - 1.68 ± 0.01
     - 1.59 ± 0.01
   * -
     - busy
     - 39.58 ± 1.51
     - 7.23 ± 0.08
     - 7.09 ± 0.06
   * -
     - 8 threads
     - 0.27 ± 0.01
     - 0.28 ± 0.00
     - 0.26 ± 0.00
   * - ``load_video``
     - idle
     - 2.16 ± 0.01
     - 2.25 ± 0.01
     - 2.16 ± 0.01
   * -
     - busy
     - 35.89 ± 1.00
     - 7.84 ± 0.07
     - 7.77 ± 0.06
   * -
     - 8 threads
     - 0.35 ± 0.00
     - 0.36 ± 0.00
     - 0.37 ± 0.05
   * - ``load_image``, 640x480 JPEG
     - idle
     - 1.12 ± 0.00
     - 1.20 ± 0.01
     - 1.13 ± 0.00
   * -
     - busy
     - 33.28 ± 1.30
     - 6.77 ± 0.08
     - 6.73 ± 0.06
   * -
     - 8 threads
     - 0.18 ± 0.00
     - 0.19 ± 0.00
     - 0.18 ± 0.00
   * - ``load_image``, 64x48 PNG
     - idle
     - 0.16 ± 0.00
     - 0.24 ± 0.01
     - 0.17 ± 0.00
   * -
     - busy
     - 33.12 ± 1.56
     - 5.57 ± 0.05
     - 5.48 ± 0.04
   * -
     - 8 threads
     - 0.08 ± 0.00
     - 0.10 ± 0.00
     - 0.05 ± 0.00
   * - ``load_image_batch``, 32 PNGs
     - idle
     - 13.24 ± 0.05
     - 12.76 ± 0.07
     - 12.22 ± 0.06
   * -
     - busy
     - 885.43 ± 33.83
     - 18.17 ± 0.06
     - 17.77 ± 0.20
   * -
     - 8 threads
     - 2.20 ± 0.03
     - 1.73 ± 0.11
     - 1.62 ± 0.10

With the cache, the loading functions are as fast as without fusion when no
other Python thread is busy, as the cache saves recording the operations in
Python. With another busy Python thread, they save 27 to 870 ms per call.

Guidelines
----------

Errors run the function again
   Recorded operations fail when they are executed, that is, when the
   function returns or when Python needs an actual value, not where the
   SPDL IO function is called. When that happens, :py:func:`~spdl.io.fuse`
   executes the function again without recording, so that the error is
   raised where the SPDL IO function is called, and ``try`` statements and
   logging in the function work as they do without :py:func:`~spdl.io.fuse`.
   Python code in the function runs twice in this case, so avoid side
   effects that must not be repeated, such as consuming an iterator.

Do not decorate generators and coroutines
   The recording ends when the decorated function returns, which, for
   generators and coroutines, happens before their body runs. Decorate a
   regular function that they call instead.

Packets are consumed by decoding
   As with eager calls, :py:func:`~spdl.io.decode_packets` and
   :py:func:`~spdl.io.sample_decode_video` consume the packets. Using the
   packets after that raises an error. Use ``packets.clone()`` to decode the
   same packets multiple times.

Returned values
   The placeholders in the returned value are replaced with the actual
   values if they are in lists, tuples, named tuples and dicts. Placeholders
   in other objects (e.g. dataclasses) are executed, and they behave like
   the actual values, but they are not instances of the actual classes.
   Return the results in such containers when the type matters.

Caching the recorded program
----------------------------

With ``cache=True``, :py:func:`~spdl.io.fuse` saves running the function in
Python as well.

Compared to the gain of fusing itself, the difference that ``cache=True``
makes is marginal. The following table shows the time per call (in
milliseconds) with Python 3.14, as the mean and the 95% confidence interval
of 10 trials. Each trial is 20 calls (240 calls for "8 threads"), and the
three variants are measured in alternating order. The conditions are the same
as in the table above.

.. list-table::
   :header-rows: 1

   * - Function
     - Condition
     - Eager
     - ``fuse``
     - ``fuse(cache=True)``
   * - Decode 32 PNG images and batch them
     - idle
     - 5.08 ± 0.03
     - 4.64 ± 0.03
     - 3.92 ± 0.01
   * -
     - busy
     - 891.31 ± 30.27
     - 10.11 ± 0.09
     - 9.47 ± 0.08
   * -
     - 8 threads
     - 2.75 ± 0.13
     - 0.87 ± 0.10
     - 0.54 ± 0.03
   * - Demux 4 windows of a video, decode, batch and read PTS
     - idle
     - 7.90 ± 0.03
     - 7.93 ± 0.03
     - 7.70 ± 0.01
   * -
     - busy
     - 88.31 ± 1.89
     - 13.40 ± 0.04
     - 13.08 ± 0.02
   * -
     - 8 threads
     - 1.06 ± 0.01
     - 1.12 ± 0.01
     - 1.05 ± 0.01
   * - :py:func:`~spdl.io.load_audio` and :py:func:`~spdl.io.to_numpy`
     - idle
     - 1.63 ± 0.01
     - 1.71 ± 0.01
     - 1.58 ± 0.01
   * -
     - busy
     - 37.57 ± 0.55
     - 7.24 ± 0.05
     - 7.05 ± 0.02
   * -
     - 8 threads
     - 0.25 ± 0.00
     - 0.26 ± 0.00
     - 0.23 ± 0.00
   * - Filter computed from ``packets.width``, check ``num_frames`` (not cached)
     - idle
     - 2.78 ± 0.01
     - 2.94 ± 0.01
     - 2.95 ± 0.01
   * -
     - busy
     - 37.70 ± 0.69
     - 19.09 ± 0.05
     - 19.11 ± 0.04
   * -
     - 8 threads
     - 0.39 ± 0.00
     - 0.44 ± 0.00
     - 0.44 ± 0.01

With another busy Python thread, fusing saves 19 to 880 ms per call, while
the cache saves a further 0.2 to 0.6 ms. Without contention, the cache saves
0.1 to 0.7 ms, the time to run the function in Python. The last function uses
metadata in Python, so it cannot be cached, and ``cache=True`` makes no
difference. Prefer the default unless profiling
shows otherwise.

.. code-block:: python

   @spdl.io.fuse(cache=True)
   def load_images(srcs: list[bytes]) -> spdl.io.CPUBuffer:
       ...

After a call executes all the operations in one native call, the next call
with the same argument structure records the function as a program, with
the source arguments (``str``, ``bytes`` and other buffer objects) replaced
by placeholders. The subsequent calls with the same argument structure run
the program without executing the function. The argument structure is the
nesting of lists, tuples and dicts, and the values of the arguments that are
not sources. The programs for the last 128 argument structures are kept.

This is correct only if the result of the function depends only on its
arguments. Do not use ``cache=True`` if the function uses random numbers,
or the state of objects or global variables.

Arguments are classified as sources by their type, so a ``str`` or a NumPy
array that is not a source is replaced by a placeholder as well. To avoid
this, declare the parameters that receive sources with ``sources``. Then the
arguments of the other parameters, including ``str``, are part of the
argument structure, so each value has its own cached program.

.. code-block:: python

   @spdl.io.fuse(cache=True, sources=("src",))
   def load(src: bytes, filter_desc: str) -> spdl.io.CPUBuffer:
       return spdl.io.load_image(src, filter_desc=filter_desc)

Without ``sources``, if the function uses such an argument in any other way
than passing it to SPDL IO functions (e.g. formatting it into a string,
comparing or indexing it), or does not use it, :py:func:`~spdl.io.fuse`
detects it, and runs the function without the cache for that argument
structure.

Free-threaded Python
--------------------

**When the GIL is disabled, fuse() does nothing**, and the
function runs eagerly. Without the GIL, a native call does not wait for
other threads when it returns, so recording only adds overhead.

The following table shows the same measurement as the one above, with
Python 3.14 and its free-threaded build (3.14t). The fused function was
measured before making :py:func:`~spdl.io.fuse` do nothing without the GIL.

.. list-table::
   :header-rows: 1

   * - Function
     - Condition
     - 3.14, Eager
     - 3.14, Fused
     - 3.14t, Eager
     - 3.14t, Fused
   * - Decode 32 PNG images and batch them
     - busy
     - 796.69
     - 13.91
     - 3.47
     - 3.59
   * -
     - 8 threads
     - 1.22
     - 0.51
     - 0.46
     - 0.56
   * - :py:func:`~spdl.io.load_audio` and :py:func:`~spdl.io.to_numpy`
     - busy
     - 36.07
     - 11.78
     - 1.31
     - 1.40

The check is made on every call, because free-threaded Python enables the
GIL when it imports an extension module that does not support running
without the GIL.
