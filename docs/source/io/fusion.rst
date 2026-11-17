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

The following figure and table show the time per call of loading functions with
small test media as GIL contention grows from 0% to 100% (Python 3.14 and its
free-threaded build 3.14t, mean and 95% confidence interval of 10 trials, in
milliseconds). GIL contention is the share of time another Python thread runs
Python code: it runs a Python loop for that share of each 10 ms period and
sleeps for the rest, like a training loop that runs Python code between other
work. Without fusion, every native call has to re-acquire the GIL while the
other thread may hold it, so the time grows quickly with the contention. With
fusion, a call re-acquires the GIL about once, so its time grows much more
slowly. On the free-threaded build, which has no GIL, the time does not change.
The dotted line in the figure is the time without contention, and its y-axis is
in log scale. (The figure and the table also show ``fuse(cache=True)``,
explained below.)

.. image:: /_static/data/fuse_benchmark.png
   :alt: Time per call of the functions without fusion, with fuse, with
         fuse(cache=True), and without fusion on the free-threaded build,
         as GIL contention grows from 0% to 100%.

===============  ==========  ==============  ============  ==============  ===========
Function         Contention  Eager           ``fuse``      ``cache=True``  3.14t eager
===============  ==========  ==============  ============  ==============  ===========
32 PNG images    0%          5.05 ± 0.03     4.59 ± 0.02   3.92 ± 0.02     5.20 ± 0.06
32 PNG images    10%         5.59 ± 0.03     4.79 ± 0.12   4.03 ± 0.02     5.14 ± 0.02
32 PNG images    25%         6.95 ± 0.06     5.00 ± 0.04   5.00 ± 0.03     5.17 ± 0.06
32 PNG images    50%         10.65 ± 0.12    5.36 ± 0.33   5.01 ± 0.02     5.18 ± 0.06
32 PNG images    75%         22.34 ± 0.53    9.87 ± 0.16   7.98 ± 0.09     5.20 ± 0.05
32 PNG images    90%         63.56 ± 3.25    9.88 ± 0.13   8.51 ± 0.13     5.16 ± 0.06
32 PNG images    100%        839.74 ± 24.37  10.03 ± 0.04  9.32 ± 0.03     5.10 ± 0.02
4 video windows  0%          3.39 ± 0.01     3.46 ± 0.02   3.24 ± 0.02     3.46 ± 0.01
4 video windows  10%         3.64 ± 0.03     3.48 ± 0.02   3.29 ± 0.04     3.45 ± 0.01
4 video windows  25%         4.54 ± 0.07     3.69 ± 0.10   3.34 ± 0.02     3.48 ± 0.03
4 video windows  50%         6.93 ± 0.08     5.05 ± 0.03   5.02 ± 0.02     3.48 ± 0.04
4 video windows  75%         12.44 ± 0.46    7.40 ± 0.11   6.76 ± 0.12     3.44 ± 0.03
4 video windows  90%         29.96 ± 1.71    7.60 ± 0.10   7.12 ± 0.23     3.42 ± 0.03
4 video windows  100%        81.04 ± 4.39    8.91 ± 0.03   8.63 ± 0.05     3.42 ± 0.03
``load_audio``   0%          1.62 ± 0.01     1.72 ± 0.01   1.59 ± 0.01     1.64 ± 0.01
``load_audio``   10%         1.71 ± 0.03     1.75 ± 0.03   1.65 ± 0.01     1.64 ± 0.01
``load_audio``   25%         2.11 ± 0.08     2.05 ± 0.05   2.01 ± 0.03     1.65 ± 0.02
``load_audio``   50%         3.29 ± 0.11     3.40 ± 0.06   2.99 ± 0.18     1.64 ± 0.01
``load_audio``   75%         6.55 ± 0.05     3.92 ± 0.07   3.50 ± 0.12     1.63 ± 0.01
``load_audio``   90%         16.04 ± 1.49    5.03 ± 0.00   5.03 ± 0.00     1.66 ± 0.02
``load_audio``   100%        36.86 ± 1.63    7.15 ± 0.08   6.98 ± 0.06     1.63 ± 0.01
Metadata used    0%          2.45 ± 0.01     2.61 ± 0.01   2.61 ± 0.01     2.48 ± 0.02
Metadata used    10%         2.52 ± 0.03     2.70 ± 0.02   2.71 ± 0.02     2.47 ± 0.04
Metadata used    25%         3.08 ± 0.07     3.36 ± 0.05   3.35 ± 0.05     2.46 ± 0.03
Metadata used    50%         4.91 ± 0.10     5.09 ± 0.08   5.04 ± 0.02     2.46 ± 0.02
Metadata used    75%         9.62 ± 0.22     10.01 ± 0.04  10.04 ± 0.03    2.46 ± 0.02
Metadata used    90%         15.33 ± 2.09    10.07 ± 0.00  10.07 ± 0.00    2.46 ± 0.02
Metadata used    100%        33.72 ± 0.98    18.69 ± 0.09  18.67 ± 0.07    2.47 ± 0.04
===============  ==========  ==============  ============  ==============  ===========

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
- :py:func:`~spdl.io.transfer_buffer`, :py:func:`~spdl.io.decode_packets_nvdec`,
  :py:func:`~spdl.io.nv12_to_rgb` and :py:func:`~spdl.io.nv12_to_bgr`, when their
  input is recorded and the CUDA extension supports fusion.
- The metadata of demuxers, codecs, packets and frames, ``clone()``, and
  indexing video frames with an integer, a slice or a list of integers.

Operations that are not recorded run as usual with the actual values. These
include streaming demuxing and decoding, :py:func:`~spdl.io.decode_image_nvjpeg`,
:py:class:`~spdl.io.BSF`, ``load_image_batch(strict=False)``, and attributes
that are not recorded. The recorded operations are executed before them.

CUDA operations
---------------

Passing ``device_config`` to the loading functions records the CPU-to-CUDA
transfer in the same native call as demuxing, decoding and conversion. The
cached program includes the transfer; ``device_config`` is a non-source
constant, keyed by object identity. The CUDA extension supplies native
operations without making CUDA a dependency of the CPU executor.

:py:func:`~spdl.io.decode_packets_nvdec` records one-off NVDEC decoding,
including the automatic Annex B bitstream filter for H264/HEVC and RGB/BGR
conversion. Explicit :py:func:`~spdl.io.apply_bsf` is supported as well.
Streaming :py:class:`~spdl.io.NvDecDecoder` operations are not recorded.

CUDA buffers can be returned directly or passed to :py:func:`~spdl.io.to_torch`.
Reading their attributes (e.g. ``shape`` or ``device_index``) materializes them,
so these accesses prevent caching just like other unrecorded attributes.
Inputs that are already concrete, or extensions without the native factories,
use the eager path. Custom Python allocators, including the torch caching
allocator, acquire the GIL for their callbacks even during a fused native call.

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
   values if they are in lists, tuples, named tuples and dicts, or in the
   fields of dataclass instances (in place, so the instance stays the same
   object). Placeholders in other objects (e.g. subclasses of ``dict``) are
   executed, and they behave like the actual values, but they are not
   instances of the actual classes. Return the results in such containers
   when the type matters. Returning the same value more than once returns
   the same object at each position.

Caching the recorded program
----------------------------

With ``cache=True``, :py:func:`~spdl.io.fuse` saves running the function in
Python as well.

Compared to the gain of fusing itself, the difference that ``cache=True`` makes
is marginal (see the figure and the table in `Why fuse IO calls`_). At 100% GIL
contention, fusing saves 15 to 830 ms per call, while the cache saves a further
0.2 to 0.7 ms. Without contention, the cache saves 0.1 to 0.7 ms, the time to
run the function in Python. The last function uses metadata in Python, so it
cannot be cached, and ``cache=True`` makes no difference. Prefer the default
unless profiling shows otherwise.

.. code-block:: python

   @spdl.io.fuse(cache=True)
   def load_images(srcs: list[bytes]) -> spdl.io.CPUBuffer:
       ...

After a call executes all the operations in one native call, the next call
with the same argument structure records the function as a program, with
the source arguments (``str``, ``bytes`` and other buffer objects) replaced
by placeholders. The subsequent calls with the same argument structure run
the program without executing the function. The argument structure is the
nesting of lists, tuples, named tuples and dicts (exact types, which are
passed with the same types when tracing), and the values of the arguments
that are not sources. The programs for the last 128 argument structures are kept.

This is correct only if the result of the function depends only on its
arguments. Do not use ``cache=True`` if the function uses random numbers,
or the state of objects or global variables.

Arguments are classified as sources by their type, so a ``str`` or a NumPy
array that is not a source is replaced by a placeholder as well. If the
function uses such an argument in any other way than passing it to SPDL IO
functions (e.g. formatting it into a string, comparing or indexing it), or
does not use it, :py:func:`~spdl.io.fuse` detects it, and runs the function
without the cache for that argument structure.

Free-threaded Python
--------------------

**When the GIL is disabled, fuse() does nothing**, and the
function runs eagerly. Without the GIL, a native call does not wait for
other threads when it returns, so recording only adds overhead.

The free-threaded series in `Why fuse IO calls`_ shows the time without the
GIL: it does not change with GIL contention. The following table compares
both builds in two other conditions (time per call in milliseconds, one
measurement each). "1 busy thread" is the time per call while another Python
thread runs a Python loop all the time, which is 100% GIL contention.
"8 loading threads" is the wall time per call while 8 threads call the
function concurrently, with no other Python thread, so it is the total time
divided by the number of calls of all 8 threads. The fused function was
measured before making :py:func:`~spdl.io.fuse` do nothing without the GIL.

==============  =================  ===========  ===========  ============  ============
Function        Condition          3.14, Eager  3.14, Fused  3.14t, Eager  3.14t, Fused
==============  =================  ===========  ===========  ============  ============
32 PNG images   1 busy thread      796.69       13.91        3.47          3.59
``load_audio``  1 busy thread      36.07        11.78        1.31          1.40
32 PNG images   8 loading threads  1.22         0.51         0.46          0.56
==============  =================  ===========  ===========  ============  ============

The check is made on every call, because free-threaded Python enables the
GIL when it imports an extension module that does not support running
without the GIL.
