FFmpeg CLI Cheat Sheet
======================

This page is for users who already know the ``ffmpeg`` command line and want to
find the equivalent SPDL knob quickly.

It assumes you know the basics of ``spdl.io`` — that demuxing is done with
:py:func:`spdl.io.demux_audio` / :py:func:`spdl.io.demux_video` /
:py:func:`spdl.io.demux_image`, decoding with :py:func:`spdl.io.decode_packets`,
and encoding/muxing with :py:class:`spdl.io.Muxer`. Rather than re-explaining the
pipeline, this page maps the **customization surfaces** — the configuration
objects and the filter-description builders — onto the ``ffmpeg`` flags you
already know.

The page is a sequence of per-config tables. In each table:

- **Attribute** shows the config attribute with a representative
  *non-default* value (this page is about non-default behavior).
- **FFmpeg** shows the matching ``ffmpeg`` flag, with the same value.
- **Note** appears only when the mapping is non-obvious.

Each table is followed by a worked example: a partial ``ffmpeg`` command and the
equivalent ``spdl.io`` snippet. All snippets assume ``import spdl.io`` and use
the factory functions directly.


``DemuxConfig`` — ``demux_config(...)``
---------------------------------------

Pass the result as ``demux_config=`` to a ``demux_*`` function.

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Attribute
     - FFmpeg
     - Note
   * - ``format="s16le"``
     - ``-f s16le`` (input)
     - Input format override (e.g. for headerless raw PCM).
   * - ``format_options={"probesize": "1M"}``
     - ``-probesize 1M``
     - Dict of demuxer / ``AVFormatContext`` options.
   * - ``buffer_size=65536``
     - *(no CLI equivalent)*
     - SPDL I/O buffer tuning.

.. admonition:: Example

   .. code-block:: text

      ffmpeg -f s16le -probesize 1M -i sample.raw ...

   .. code-block:: python

      packets = spdl.io.demux_audio(
          "sample.raw",
          demux_config=spdl.io.demux_config(
              format="s16le",
              format_options={"probesize": "1M"},
          ),
      )


``DecodeConfig`` — ``decode_config(...)``
-----------------------------------------

Pass the result as ``decode_config=`` to :py:func:`spdl.io.decode_packets`.

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Attribute
     - FFmpeg
     - Note
   * - ``decoder="libopenh264"``
     - ``-c:v libopenh264``
     - Software decoder name.
   * - ``decoder_options={"threads": "0"}``
     - ``-threads 0``
     - Codec-private options; values must be strings. ``threads=0`` lets FFmpeg
       choose (SPDL defaults to a single thread).

.. admonition:: Example

   .. code-block:: text

      ffmpeg -c:v libopenh264 -threads 0 -i input.mp4 ...

   .. code-block:: python

      packets = spdl.io.demux_video("input.mp4")
      frames = spdl.io.decode_packets(
          packets,
          decode_config=spdl.io.decode_config(
              decoder="libopenh264",
              decoder_options={"threads": "0"},
          )
      )


``VideoEncodeConfig`` — ``video_encode_config(...)``
----------------------------------------------------

Pass the result to
:py:meth:`Muxer.add_encode_stream <spdl.io.Muxer.add_encode_stream>`.

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Attribute
     - FFmpeg
     - Note
   * - ``width=1280, height=720``
     - ``-s 1280x720``
     - Required.
   * - ``pix_fmt="yuv420p"``
     - ``-pix_fmt yuv420p``
     - Pixel format of the frames fed to the encoder.
   * - ``frame_rate=(30, 1)``
     - ``-r 30``
     - ``(numerator, denominator)`` tuple.
   * - ``bit_rate=4_000_000``
     - ``-b:v 4M``
     -
   * - ``qscale=23``
     - ``-q:v 23``
     -
   * - ``compression_level=6``
     - ``-compression_level 6``
     -
   * - ``gop_size=60``
     - ``-g 60``
     -
   * - ``max_b_frames=2``
     - ``-bf 2``
     -
   * - ``colorspace="bt709"``
     - ``-colorspace bt709``
     -
   * - ``color_primaries="bt709"``
     - ``-color_primaries bt709``
     -
   * - ``color_trc="bt709"``
     - ``-color_trc bt709``
     -

.. admonition:: Example

   .. code-block:: text

      ffmpeg ... \
        -s 1280x720 \
        -pix_fmt yuv420p \
        -r 30 \
        -b:v 4M \
        -g 60 \
        -bf 2 \
        -colorspace bt709 \
        -color_primaries bt709 \
        -color_trc bt709 \
        out.mp4

   .. code-block:: python

      cfg = spdl.io.video_encode_config(
          width=1280, height=720,
          pix_fmt="yuv420p",
          frame_rate=(30, 1),
          bit_rate=4_000_000,
          gop_size=60,
          max_b_frames=2,
          colorspace="bt709",
          color_primaries="bt709",
          color_trc="bt709",
      )


``AudioEncodeConfig`` — ``audio_encode_config(...)``
----------------------------------------------------

Pass the result to
:py:meth:`Muxer.add_encode_stream <spdl.io.Muxer.add_encode_stream>`.

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Attribute
     - FFmpeg
     - Note
   * - ``num_channels=2``
     - ``-ac 2``
     - Required.
   * - ``sample_rate=48000``
     - ``-ar 48000``
     - Required.
   * - ``sample_fmt="fltp"``
     - ``-sample_fmt fltp``
     -
   * - ``bit_rate=192_000``
     - ``-b:a 192k``
     -
   * - ``qscale=4``
     - ``-q:a 4``
     -
   * - ``compression_level=10``
     - ``-compression_level 10``
     -

.. admonition:: Example

   .. code-block:: text

      ffmpeg ... -ac 2 -ar 48000 -sample_fmt fltp -b:a 192k out.m4a

   .. code-block:: python

      cfg = spdl.io.audio_encode_config(
          num_channels=2,
          sample_rate=48000,
          sample_fmt="fltp",
          bit_rate=192_000,
      )


Encoder selection — ``Muxer.add_encode_stream(...)``
----------------------------------------------------

The encode config above describes *what* to produce; the encoder and its private
options are chosen on :py:meth:`Muxer.add_encode_stream
<spdl.io.Muxer.add_encode_stream>`.

.. list-table::
   :header-rows: 1
   :widths: 45 30 25

   * - Attribute
     - FFmpeg
     - Note
   * - ``encoder="libx264"``
     - ``-c:v libx264``
     - See ``ffmpeg -encoders``.
   * - ``encoder_config={"preset": "fast", "crf": "23"}``
     - ``-preset fast -crf 23``
     - Encoder-private options; see ``ffmpeg -h encoder=libx264``.

.. admonition:: Example

   .. code-block:: text

      ffmpeg ... -c:v libx264 -preset fast -crf 23 out.mp4

   .. code-block:: python

      muxer.add_encode_stream(
          spdl.io.video_encode_config(
              width=1280,
              height=720,
              pix_fmt="yuv420p",
              frame_rate=(30, 1),
          ),
          encoder="libx264",
          encoder_config={
              "preset": "fast",
              "crf": "23",
          },
      )


Output container — ``Muxer(...)`` / ``Muxer.open(...)``
-------------------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 50 28 22

   * - Attribute
     - FFmpeg
     - Note
   * - ``format="mp4"`` (constructor)
     - ``-f mp4`` (output)
     - Output format / device override.
   * - ``muxer_config={"movflags": "+faststart"}`` (``open()``)
     - ``-movflags +faststart``
     - Dict of muxer options.

.. admonition:: Example

   .. code-block:: text

      ffmpeg ... -f mp4 -movflags +faststart out.mp4

   .. code-block:: python

      muxer = spdl.io.Muxer("out.mp4", format="mp4")
      muxer.add_encode_stream(...)
      with muxer.open(muxer_config={"movflags": "+faststart"}):
          ...


Video filters — ``get_video_filter_desc(...)``
----------------------------------------------

:py:func:`spdl.io.get_video_filter_desc` builds a filter-graph string that is
passed as ``filter_desc=`` (raw libavfilter syntax, ≈ ``-vf``). The **Note**
column shows the filter the attribute is emitted as.

.. list-table::
   :header-rows: 1
   :widths: 38 27 35

   * - Attribute
     - FFmpeg
     - Note (emitted as)
   * - ``scale_width=256, scale_height=256``
     - ``-vf scale=256:256``
     - ``scale=256:256``
   * - ``scale_algo="bicubic"``
     - ``-sws_flags bicubic``
     - ``scale=...:flags=bicubic``
   * - ``scale_mode="pad"``
     - ``-vf ...,pad=...``
     - ``"pad"`` or ``"crop"``.
   * - ``crop_width=224, crop_height=224``
     - ``-vf crop=224:224``
     - Center crop.
   * - ``pix_fmt="rgb24"``
     - ``-pix_fmt rgb24``
     - ``format=pix_fmts=rgb24``
   * - ``frame_rate=(30, 1)``
     - ``-r 30``
     - ``fps=30/1``
   * - ``timestamp=(1.5, 4.0)``
     - ``-ss 1.5 -to 4.0``
     - ``trim=start=1.5:end=4.0``
   * - ``num_frames=16``
     - ``-frames:v 16``
     - ``tpad,trim=end_frame=16``
   * - ``pad_mode="black"``
     - ``pad=...:color=black``
     - Color used when padding to ``num_frames``.
   * - ``filter_desc="hflip"``
     - extra ``-vf hflip``
     - Escape hatch; appended verbatim.

.. admonition:: Example

   .. code-block:: text

      ffmpeg -i in.mp4 \
        -vf "fps=30/1,trim=start=1.5:end=4.0,scale=w=256:h=256:flags=bicubic:force_original_aspect_ratio=decrease,pad=w=256:h=256:x=-1:y=-1:color=black,crop=w=224:h=224,tpad=stop=-1:stop_mode=clone,trim=end_frame=16,format=pix_fmts=rgb24" \
        out.mp4

   .. code-block:: python

      filter_desc = spdl.io.get_video_filter_desc(
          timestamp=(1.5, 4.0),
          scale_width=256,
          scale_height=256,
          scale_algo="bicubic",
          crop_width=224,
          crop_height=224,
          frame_rate=(30, 1),
          pix_fmt="rgb24",
          num_frames=16,
      )
      # creates
      # "fps=30/1,trim=start=1.5:end=4.0,scale=w=256:h=256:flags=bicubic:force_original_aspect_ratio=decrease,pad=w=256:h=256:x=-1:y=-1:color=black,crop=w=224:h=224,tpad=stop=-1:stop_mode=clone,trim=end_frame=16,format=pix_fmts=rgb24"

      # Use it when decoding. Providing timestamp to demuxer reduces the amount of processing:
      packets = spdl.io.demux_video("in.mp4", timestamp=(1.5, 4.0))
      frames = spdl.io.decode_packets(packets, filter_desc=filter_desc)



Audio filters — ``get_audio_filter_desc(...)``
----------------------------------------------

:py:func:`spdl.io.get_audio_filter_desc` builds a filter-graph string passed as
``filter_desc=`` (≈ ``-af``). The filters are emitted in a fixed order —
``num_channels`` → ``sample_rate`` → ``timestamp`` → ``num_frames`` → your
custom ``filter_desc`` → ``sample_fmt`` — so the sample-format conversion is
always applied last.

.. list-table::
   :header-rows: 1
   :widths: 38 27 35

   * - Attribute
     - FFmpeg
     - Note (emitted as)
   * - ``sample_rate=16000``
     - ``-ar 16000``
     - ``aresample=16000``
   * - ``num_channels=1``
     - ``-ac 1``
     - ``aformat=channel_layouts=1c``
   * - ``sample_fmt="fltp"``
     - ``-sample_fmt fltp``
     - ``aformat=sample_fmts=fltp``
   * - ``timestamp=(0.0, 10.0)``
     - ``-ss 0 -to 10``
     - ``atrim=start=0.0:end=10.0``
   * - ``num_frames=16000``
     - ``-frames:a 16000``
     - ``apad,atrim=end_sample=16000``
   * - ``filter_desc="volume=0.5"``
     - extra ``-af volume=0.5``
     - Escape hatch; appended verbatim.

.. admonition:: Example

   .. code-block:: text

      ffmpeg -i in.wav \
        -af "aformat=channel_layouts=1c,aresample=16000,atrim=start=0.0:end=10.0,volume=0.5,aformat=sample_fmts=fltp" \
        out.wav

   .. code-block:: python

      filter_desc = spdl.io.get_audio_filter_desc(
          timestamp=(0.0, 10.0),
          sample_rate=16000,
          num_channels=1,
          sample_fmt="fltp",
          filter_desc="volume=0.5",
      )
      # creates
      # "aformat=channel_layouts=1c,aresample=16000,atrim=start=0.0:end=10.0,volume=0.5,aformat=sample_fmts=fltp"

      # Use it when decoding. Providing timestamp to demuxer reduces the amount of processing:
      packets = spdl.io.demux_audio("in.wav", timestamp=(0.0, 10.0))
      frames = spdl.io.decode_packets(packets, filter_desc=filter_desc)


Complex filter graphs — ``FilterGraph``
----------------------------------------

The builders above produce *linear* chains — one input, one output — passed as
``filter_desc=``. For graphs with **multiple inputs or outputs** (FFmpeg's
``-filter_complex``), use :py:class:`spdl.io.FilterGraph` directly.

The key difference: ``ffmpeg`` adds the source and sink pads implicitly (from
``-i`` and ``-map``). With ``FilterGraph`` you spell them out — ``buffer`` /
``abuffer`` source nodes and ``buffersink`` / ``abuffersink`` sink nodes — built
with :py:func:`spdl.io.get_buffer_desc` / :py:func:`spdl.io.get_abuffer_desc`.

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - FFmpeg ``-filter_complex``
     - spdl.io ``FilterGraph``
   * - ``-filter_complex "<graph>"``
     - ``FilterGraph("<graph>")``
   * - input pads ``[0:v]``, ``[1:v]`` (from ``-i``)
     - explicit ``buffer`` / ``abuffer`` nodes, labeled ``buffer@in0``,
       ``buffer@in1`` (``get_buffer_desc(codec, label="in0")``)
   * - feeding each input
     - ``add_frames(frames, key="buffer@in0")``
   * - link labels ``[a]``, ``[tmp]``, ``[out]``
     - identical ``[a]`` / ``[tmp]`` / ``[out]`` labels inside the string
   * - output pad + ``-map "[out]"``
     - named sink ``buffersink@out0``; read with
       ``get_frames(key="buffersink@out0")``

.. admonition:: Example — side-by-side stack of two inputs

   .. code-block:: text

      ffmpeg -i left.mp4 -i right.mp4 -filter_complex "[0:v][1:v]hstack" out.mp4

   .. code-block:: python

      # Spell out the source/sink nodes that ffmpeg adds implicitly:
      codec = spdl.io.Demuxer("left.mp4").video_codec
      buf0 = spdl.io.get_buffer_desc(codec, label="in0")
      buf1 = spdl.io.get_buffer_desc(codec, label="in1")
      filter_desc = f"{buf0} [in0];{buf1} [in1],[in0] [in1] hstack,buffersink"

      fg = spdl.io.FilterGraph(filter_desc)
      fg.add_frames(left_frames, key="buffer@in0")
      fg.add_frames(right_frames, key="buffer@in1")
      stacked = fg.get_frames()

.. seealso::

   :doc:`advanced_filtering` — the full guide to multi-input / multi-output
   graphs, streaming, and audio-to-video (multimedia) filters.


``CUDAConfig`` — ``cuda_config(...)``
-------------------------------------

Used to select the GPU (and CUDA stream / allocator) for transfer and hardware
decoding.

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Attribute
     - FFmpeg
     - Note
   * - ``device_index=1``
     - ``-hwaccel_device 1``
     - Which GPU.
   * - ``stream=<int>``
     - *(no CLI equivalent)*
     - SPDL-specific (CUDA stream handle).
   * - ``allocator=<callable>``
     - *(no CLI equivalent)*
     - SPDL-specific (custom allocator/deleter).

.. admonition:: Example

   .. code-block:: text

      ffmpeg -hwaccel cuda -hwaccel_device 1 -i in.mp4 ...

   .. code-block:: python

      device_config = spdl.io.cuda_config(device_index=1)


NVDEC / nvJPEG — *not* a 1:1 of ``*_cuvid``
-------------------------------------------

.. warning::

   :py:func:`spdl.io.decode_packets_nvdec`, :py:func:`spdl.io.nvdec_decoder`, and
   :py:func:`spdl.io.decode_image_nvjpeg` are SPDL's own NVDEC / nvJPEG
   integrations — **not** wrappers over FFmpeg's ``h264_cuvid`` / ``hevc_cuvid`` /
   ``mjpeg_cuvid``. The flags below are the closest analogues for *intent* only;
   the color conversion, scaling resampler, and post-processing differ, so output
   will not be byte-exact. These APIs are also experimental.

``decode_packets_nvdec(...)`` / ``nvdec_decoder(...)``:

.. list-table::
   :header-rows: 1
   :widths: 50 30 20

   * - Attribute
     - FFmpeg analogue
     - Note
   * - ``device_config=cuda_config(device_index=0)``
     - ``-hwaccel cuda -hwaccel_device 0``
     -
   * - *(decoder implicit from codec)*
     - ``-c:v h264_cuvid``
     - Not user-selectable.
   * - ``pix_fmt="rgb"``
     - output of internal CSC
     - ``"rgb"`` / ``"bgr"`` / ...
   * - ``scale_width=224, scale_height=224``
     - ``-vf scale_cuda=224:224``
     - Intent only.
   * - ``crop_left=8, crop_top=8, crop_right=8, crop_bottom=8``
     - ``-vf crop=...`` (GPU)
     - Intent only.

.. admonition:: Example

   .. code-block:: text

      # Closest FFmpeg analogue — not byte-exact:
      ffmpeg -hwaccel cuda -hwaccel_device 0 -c:v h264_cuvid -i in.mp4 \
        -vf "crop=iw-16:ih-16:8:8,scale_cuda=224:224,format=rgb24" out.mp4

   .. code-block:: python

      buffer = spdl.io.decode_packets_nvdec(
          packets,
          device_config=spdl.io.cuda_config(device_index=0),
          pix_fmt="rgb",
          scale_width=224, scale_height=224,
          crop_left=8, crop_top=8, crop_right=8, crop_bottom=8,
      )

``decode_image_nvjpeg(...)``:

.. list-table::
   :header-rows: 1
   :widths: 50 30 20

   * - Attribute
     - FFmpeg analogue
     - Note
   * - ``device_config=cuda_config(device_index=0)``
     - ``-hwaccel cuda``
     -
   * - *(codec implicit nvJPEG)*
     - ≈ ``-c:v mjpeg_cuvid``
     - Different implementation.
   * - ``scale_width=256, scale_height=256``
     - ``-vf scale_cuda=256:256``
     -
   * - ``pix_fmt="rgb"``
     - output color format
     -

.. admonition:: Example

   .. code-block:: text

      ffmpeg -hwaccel cuda -c:v mjpeg_cuvid -i in.jpg \
        -vf "scale_cuda=256:256,format=rgb24" out.png

   .. code-block:: python

      buffer = spdl.io.decode_image_nvjpeg(
          "in.jpg",
          device_config=spdl.io.cuda_config(device_index=0),
          scale_width=256, scale_height=256,
          pix_fmt="rgb",
      )


Deliberately not mapped
-----------------------

- ``-vsync`` / ``-async`` / ``-itsoffset`` / ``-shortest`` — timing is handled
  per-stream via filter graphs and the ``pts`` on reference frames.
- Anything beyond demux + decode + filter + encode + mux — ``spdl.io`` is a
  data-loading library, not a full transcoder.
