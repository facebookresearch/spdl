Video Color Space
=================

By default, ``spdl.io`` decodes video frames to **RGB24** so they can be fed
directly into models that expect RGB tensors.  When this default is not what
you want, ``spdl.io`` can also decode into the codec's native YUV color space
or into other pixel formats.  This is useful when:

- You want to skip the YUV → RGB conversion and let your model (or downstream
  GPU code) consume YUV directly.
- You have a model trained on YUV inputs (e.g. NV12 from a hardware decoder).
- You want to reduce memory traffic.  ``yuv420p`` is roughly half the size of
  ``rgb24`` for the same resolution.

This page covers the supported pixel formats, the resulting tensor shapes,
and how to control which one ``spdl.io`` produces.

.. image:: ../_static/data/io_color_space_overview.png

The ``filter_desc`` argument
----------------------------

Color-space conversion in ``spdl.io`` is controlled by the
``filter_desc`` argument passed to :py:func:`spdl.io.load_video`,
:py:func:`spdl.io.decode_packets`, and related functions.
There are three modes:

1. **Unset (default).** ``spdl.io`` builds a filter graph that converts the
   output to ``rgb24``.  This is the easy path for models that expect RGB.

   .. code-block:: python

      import spdl.io

      # Decoded to rgb24 — shape (num_frames, H, W, 3), dtype uint8.
      buffer = spdl.io.load_video("video.mp4")

2. **Explicit pixel format.** Use :py:func:`spdl.io.get_video_filter_desc` to
   pick a different ``pix_fmt``.

   .. code-block:: python

      filter_desc = spdl.io.get_video_filter_desc(pix_fmt="yuv420p")
      buffer = spdl.io.load_video("video.mp4", filter_desc=filter_desc)

3. **Disabled (``filter_desc=None``).** No filter graph is constructed and no
   pixel-format conversion happens.  The buffer keeps the codec's native pixel
   format — for most codecs this is ``yuv420p``.  This is the cheapest path
   when you do not need RGB.

   .. code-block:: python

      buffer = spdl.io.load_video("video.mp4", filter_desc=None)

Supported pixel formats
-----------------------

The following pixel formats are recognised by ``spdl.io`` when converting
decoded frames into a buffer.  See :py:func:`spdl.io.get_video_filter_desc`
for the parameter reference.

.. list-table::
   :header-rows: 1
   :widths: 18 14 18 30 20

   * - ``pix_fmt``
     - Layout
     - Channels
     - Buffer shape (per video)
     - Notes
   * - ``rgb24``
     - Interleaved
     - 3
     - ``(N, H, W, 3)``
     - Default. NHWC, ``uint8``.
   * - ``rgba``
     - Interleaved
     - 4
     - ``(N, H, W, 4)``
     - NHWC, ``uint8``. With alpha.
   * - ``gray8``
     - Interleaved
     - 1
     - ``(N, H, W, 1)``
     - Single luma channel, ``uint8``.
   * - ``yuv444p`` / ``yuvj444p``
     - Planar
     - 3
     - ``(N, 3, H, W)``
     - NCHW. Three full-resolution planes (Y, U, V).
       ``yuvj`` variants use full range (0–255); ``yuv`` uses limited range
       (16–235 for Y).
   * - ``yuv420p`` / ``yuvj420p``
     - Planar (packed)
     - 1
     - ``(N, 1, H + H/2, W)``
     - Y plane on top (``H × W``), U and V planes (each ``H/2 × W/2``)
       packed underneath.
   * - ``yuv422p`` / ``yuvj422p``
     - Planar (packed)
     - 1
     - ``(N, 1, 2H, W)``
     - Y plane on top, U and V planes (each ``H × W/2``) packed underneath.
   * - ``nv12``
     - Semi-planar (packed)
     - 1
     - ``(N, 1, H + H/2, W)``
     - Y plane on top, U/V interleaved underneath. The native format of many
       hardware decoders (e.g. NVDEC).

In every case, ``N`` is the number of decoded frames, ``H`` and ``W`` are
the frame dimensions, and the dtype is ``uint8``.

Examples
--------

The figures below illustrate each pixel format on the same source frame.  The
left panel is the canonical RGB rendering (decoded with the default
``filter_desc``); the remaining panels are the raw planes that ``spdl.io``
returns, plotted as grayscale images at their actual resolution.  The buffer
shape is shown in the title.

RGB24 (default)
~~~~~~~~~~~~~~~

Three interleaved channels stored in NHWC order — the simplest layout, and
what most PyTorch / NumPy code expects.

.. code-block:: python

   buffer = spdl.io.load_video("video.mp4")
   array = spdl.io.to_numpy(buffer)
   # array.shape == (num_frames, H, W, 3)

.. image:: ../_static/data/io_color_space_rgb24.png

RGBA
~~~~

Same as RGB24 but with an alpha channel.  Useful for sources that carry
transparency.

.. code-block:: python

   filter_desc = spdl.io.get_video_filter_desc(pix_fmt="rgba")
   buffer = spdl.io.load_video("video.mp4", filter_desc=filter_desc)
   # array.shape == (num_frames, H, W, 4)

.. image:: ../_static/data/io_color_space_rgba.png

Grayscale (``gray8``)
~~~~~~~~~~~~~~~~~~~~~

A single luma channel.

.. code-block:: python

   filter_desc = spdl.io.get_video_filter_desc(pix_fmt="gray8")
   buffer = spdl.io.load_video("video.mp4", filter_desc=filter_desc)
   # array.shape == (num_frames, H, W, 1)

.. image:: ../_static/data/io_color_space_gray8.png

YUV 4:4:4 — full chroma resolution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

True planar YUV: three independent planes at full resolution, in NCHW order.

.. code-block:: python

   filter_desc = spdl.io.get_video_filter_desc(pix_fmt="yuv444p")
   buffer = spdl.io.load_video("video.mp4", filter_desc=filter_desc)
   # array.shape == (num_frames, 3, H, W)

.. image:: ../_static/data/io_color_space_yuv444p.png

YUV 4:2:0 — codec-native chroma subsampling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The native pixel format for most H.264 / H.265 / VP9 video.  The chroma
planes are subsampled by 2× in both dimensions and packed below the luma
plane in a single ``(H + H/2) × W`` buffer.  Total size is 1.5× ``H × W``,
half the size of ``rgb24``.

.. code-block:: python

   filter_desc = spdl.io.get_video_filter_desc(pix_fmt="yuv420p")
   buffer = spdl.io.load_video("video.mp4", filter_desc=filter_desc)
   # array.shape == (num_frames, 1, H + H // 2, W)

.. image:: ../_static/data/io_color_space_yuv420p.png

YUV 4:2:2 — chroma subsampled horizontally
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Chroma subsampled by 2× horizontally only.  The packed buffer is
``2H × W`` — Y on top, U and V each ``H × W/2`` packed beneath.

.. code-block:: python

   filter_desc = spdl.io.get_video_filter_desc(pix_fmt="yuv422p")
   buffer = spdl.io.load_video("video.mp4", filter_desc=filter_desc)
   # array.shape == (num_frames, 1, 2 * H, W)

.. image:: ../_static/data/io_color_space_yuv422p.png

NV12 — semi-planar YUV 4:2:0
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NV12 has the same total size and Y layout as ``yuv420p`` but interleaves U
and V samples (``UVUVUV...``) in a single chroma plane.  This is the format
emitted by many hardware decoders (including NVDEC); decoding directly into
NV12 avoids a deinterleave step.

.. code-block:: python

   filter_desc = spdl.io.get_video_filter_desc(pix_fmt="nv12")
   buffer = spdl.io.load_video("video.mp4", filter_desc=filter_desc)
   # array.shape == (num_frames, 1, H + H // 2, W)

.. image:: ../_static/data/io_color_space_nv12.png

Skipping the conversion entirely
--------------------------------

Passing ``filter_desc=None`` disables the filter graph altogether.  The
returned buffer keeps whatever pixel format the decoder produces — for
software H.264 / H.265 / VP9 decoders this is almost always ``yuv420p``.

.. code-block:: python

   # No filtering, no color conversion, no scaling.
   buffer = spdl.io.load_video("video.mp4", filter_desc=None)

Use this mode when:

- You want maximum decoding throughput and do not need RGB.
- You will do color conversion later on the GPU (often cheaper than on the
  CPU).
- You want to be sure ``spdl.io`` is not silently inserting a ``scale`` or
  ``format`` filter.

When ``filter_desc`` is ``None``, other filter-graph features
(``scale_width`` / ``scale_height``, ``crop_width`` / ``crop_height``,
``num_frames`` padding, ``timestamp`` trimming, etc.) are also disabled —
those are all built on top of the filter graph.  If you need any of them,
pass an explicit ``filter_desc`` instead.

See also
--------

- :doc:`filtering` — full guide to constructing filter descriptions.
- :doc:`ffmpeg_cheatsheet` — the ``pix_fmt`` row sits in the
  ``get_video_filter_desc`` table.
- :py:func:`spdl.io.get_video_filter_desc` — Python API reference.
