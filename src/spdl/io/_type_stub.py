# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from __future__ import annotations

from typing import Any, overload

__all__ = [
    "CPUBuffer",
    "CUDABuffer",
    "Packets",
    "AudioPackets",
    "VideoPackets",
    "ImagePackets",
    "Frames",
    "AudioFrames",
    "VideoFrames",
    "ImageFrames",
    "DemuxConfig",
    "DecodeConfig",
    "EncodeConfig",
    "CUDAConfig",
    "CPUStorage",
]


class Packets:
    """Packets()
    Packets object. Returned from demux functions.

    Packets objects represent the result of the demuxing.
    (Internally, it holds a series of FFmpeg's ``AVPacket`` objects.)

    Decode functions receive Packets objects and generate audio samples and
    visual frames.
    The ``Packets`` objects are exposed to public API to allow composing
    demux/decoding functions in ways that they are executed concurrently.

    For example, when decoding multiple clips from a single audio and video file,
    by emitting ``Packets`` objects in between, decoding can be started while the
    demuxer is demuxing the subsequent windows.

    The following code will kick-off the decoding job as soon as the streaming
    demux function yields a ``VideoPackets`` object.

    .. admonition:: Example

       >>> src = "sample.mp4"
       >>> windows = [
       ...     (3, 5),
       ...     (7, 9),
       ...     (13, 15),
       ... ]
       >>>
       >>> tasks = []
       >>> async for packets in spdl.io.async_streaming_demux_video(src, windows):
       >>>     # Kick off decoding job while demux function is demuxing the next window
       >>>     task = asyncio.create_task(decode_packets(packets))
       >>>     task.append(task)
       >>>
       >>> # Wait for all the decoding to be complete
       >>> asyncio.wait(tasks)

    .. important::

       About the Lifetime of Packets Object

       When packets objects are passed to a decode function, its ownership is
       also passed to the function. Therefore, accessing the packets object after
       it is passed to decode function will cause an error.

       .. code-block:: python

          >>> # Demux an image
          >>> packets = spdl.io.demux_image("foo.png").result()
          >>> packets  # this works.
          ImagePackets<src="foo.png", pixel_format="rgb24", bit_rate=0, bits_per_sample=0, codec="png", width=320, height=240>
          >>>
          >>> # Decode the packets
          >>> frames = spdl.io.decode_packets(packets)
          >>> frames
          FFmpegImageFrames<pixel_format="rgb24", num_planes=1, width=320, height=240>
          >>>
          >>> # The packets object is no longer valid.
          >>> packets
          RuntimeWarning: nanobind: attempted to access an uninitialized instance of type 'spdl.lib._spdl_ffmpeg6.ImagePackets'!

          Traceback (most recent call last):
            File "<stdin>", line 1, in <module>
          TypeError: __repr__(): incompatible function arguments. The following argument types are supported:
              1. __repr__(self) -> str

          Invoked with types: spdl.lib._spdl_ffmpeg6.ImagePackets
          >>>

       This design choice was made for two reasons;

       1. Decoding is performed in background thread, potentially long after
          since the job was created due to other decoding jobs. To ensure the
          existance of the packets, the decoding function should take the
          ownership of the packets, instead of a reference.

       2. An alternative approch to 1 is to share the ownership, however, in this
          approach, it is not certain when the Python variable holding the shared
          ownership of the Packets object is deleted. Python might keep the
          reference for a long time, or the garbage collection might kick-in when
          the execution is in critical code path. By passing the ownership to
          decoding function, the ``Packets`` object resouce is also released in
          background.

       To decode packets multiple times, use the ``clone`` method.

       .. code-block:: python

          >>> packets = spdl.io.demux_image("foo.png").result()
          >>> # Decode the cloned packets
          >>> packets2 = packets.clone()
          >>> packets2
          ImagePackets<src="foo.png", pixel_format="rgb24", bit_rate=0, bits_per_sample=0, codec="png", width=320, height=240>
          >>> frames = spdl.io.decode_packets(packets)
          >>>
          >>> # The original packets object is still valid
          >>> packets
          ImagePackets<src="foo.png", pixel_format="rgb24", bit_rate=0, bits_per_sample=0, codec="png", width=320, height=240>

       .. admonition:: Note on ``clone`` method

          The underlying FFmpeg implementation employs reference counting for
          ``AVPacket`` object.

          Therefore, even though the method is called ``clone``, this method
          does not copy the frame data.
    """

    pass


class AudioPackets(Packets):
    """AudioPackets()
    Packets object containing audio samples."""

    @property
    def timestamp(self) -> tuple[float, float] | None:
        """The window this packets covers, denoted by start and end time in second.

        This is the value specified by user when demuxing the stream.
        """
        ...

    def clone(self) -> AudioPackets:
        """Clone the packets, so that data can be decoded multiple times.

        Returns:
            A clone of the packets.
        """
        ...


class VideoPackets(Packets):
    """Packets object containing video frames."""

    @property
    def timestamp(self) -> tuple[float, float] | None:
        """The window this packets covers, denoted by start and end time in second.

        This is the value specified by user when demuxing the stream.
        """
        ...

    @property
    def pix_fmt(self) -> str:
        """The name of the pixel format, such as ``"yuv420p"``."""
        ...

    @property
    def width(self) -> int:
        """The width of video."""
        ...

    @property
    def height(self) -> int:
        """The height of video."""
        ...

    @property
    def frame_rate(self) -> tuple[int, int]:
        """The frame rate of the video in the form of ``(numerator, denominator)``."""
        ...

    def clone(self) -> VideoPackets:
        """Clone the packets, so that data can be decoded multiple times.

        Returns:
            A clone of the packets.
        """
        ...

    def __len__(self) -> int:
        """Returns the number of packets.

        .. note::

           Each packet typically contains one compressed frame, but it is not guaranteed.
        """
        ...


class ImagePackets(Packets):
    """ImagePackets()
    Packets object contain an image frame."""

    @property
    def pix_fmt(self) -> str:
        """The name of the pixel format, such as ``"yuv420p"``."""
        ...

    @property
    def width(self) -> int:
        """The width of image."""
        ...

    @property
    def height(self) -> int:
        """The height of image."""
        ...

    def clone(self) -> ImagePackets:
        """Clone the packets, so that data can be decoded multiple times.

        Returns:
            A clone of the packets.
        """
        ...


class Frames:
    """Frames()
    Frames object. Returned from decode functions.

    Frames objects represent the result of the decoding and filtering.
    Internally, it holds a series of FFmpeg's ``AVFrame`` objects thus thery
    are not a contiguous memory object.
    """


class AudioFrames(Frames):
    """AudioFrames()
    Audio frames."""

    @property
    def num_frames(self) -> int:
        """The number of audio frames. Same as ``__len__`` method.

        .. note::

           In SPDL,
           ``The number of samples`` == ``the number of frames`` x ``the number of channels``
        """
        ...

    @property
    def sample_rate(self) -> int:
        """The sample rate of audio."""
        ...

    @property
    def num_channels(self) -> int:
        """The number of channels."""
        ...

    @property
    def sample_fmt(self) -> str:
        """The name of sample format.

        Possible values are

        - ``"u8"`` for unsigned 8-bit integer.
        - ``"s16"``, ``"s32"``, ``"s64"`` for signed 16-bit, 32-bit and 64-bit integer.
        - ``"flt"``, ``"dbl"`` for 32-bit and 64-bit float.

        If the frame is planar format (separate planes for different channels), the
        name will be suffixed with ``"p"``. When converted to buffer, the buffer's shape
        will be channel-first format ``(channel, num_samples)`` instead of interweaved
        ``(num_samples, channel)``.
        """
        ...

    def __len__(self) -> int:
        """Returns the number of frames. Same as ``num_frames``."""
        ...

    def clone(self) -> AudioFrames:
        """Clone the frames, so that data can be converted to buffer multiple times.

        Returns:
            A clone of the frame.
        """
        ...


class VideoFrames(Frames):
    """VideoFrames()
    Video frames."""

    @property
    def num_frames(self) -> int:
        """The number of video frames. Same as ``__len__`` method."""
        ...

    @property
    def num_planes(self) -> int:
        """The number of planes in the each frame.

        .. note::

           This corresponds to the number of color components, however
           it does not always match with the number of color channels when
           the frame is converted to buffer/array object.

           For example, if a video file is YUV format (which is one of the most
           common formats, and comprised of different plane sizes), and
           color space conversion is disabled during the decoding, then
           the resulting frames are converted to buffer as single channel frame
           where all the Y, U, V components are packed.

           SPDL by default converts the color space to RGB, so this is
           usually not an issue.
        """
        ...

    @property
    def width(self) -> int:
        """The width of video."""
        ...

    @property
    def height(self) -> int:
        """The height of video."""
        ...

    @property
    def pix_fmt(self) -> str:
        """The name of the pixel format."""
        ...

    def __len__(self) -> int:
        """Returns the number of frames. Same as ``num_frames``."""
        ...

    @overload
    def __getitem__(self, key: int) -> ImageFrames: ...
    @overload
    def __getitem__(self, key: slice) -> VideoFrames: ...
    @overload
    def __getitem__(self, key: list[int]) -> VideoFrames: ...

    def __getitem__(self, key: int | slice | list[int]) -> VideoFrames | ImageFrames:
        """Slice frame by key.

        Args:
            key: If the key is int type, a single frame is returned as ``ImageFrames``.
                If the key is slice type, a new ``VideoFrames`` object pointing the
                corresponding frames are returned.

        Returns:
            The sliced frame.
        """
        ...

    def clone(self) -> VideoFrames:
        """Clone the frames, so that data can be converted to buffer multiple times.

        Returns:
            A clone of the frame.
        """
        ...


class ImageFrames(Frames):
    """ImageFrames()
    Image frames."""

    @property
    def num_planes(self) -> int:
        """The number of planes in the each frame.

        See :py:class:`~spdl.io.VideoFrames` for a caveat.
        """
        ...

    @property
    def width(self) -> int:
        """The width of image."""
        ...

    @property
    def height(self) -> int:
        """The height of image."""
        ...

    @property
    def pix_fmt(self) -> str:
        """The name of the pixel format."""
        ...

    @property
    def metadata(self) -> dict[str, str]:
        """Metadata attached to the frame."""
        ...

    def clone(self) -> VideoFrames:
        """Clone the frames, so that data can be converted to buffer multiple times.

        Returns:
            A clone of the frame.
        """
        ...


class CPUBuffer:
    """CPUBuffer()
    Buffer implements array interface.

    To be passed to casting functions like :py:func:`~spdl.io.to_numpy`,
    :py:func:`~spdl.io.to_torch` and :py:func:`~spdl.io.to_numba`.
    """

    @property
    def __array_interface__(self) -> dict[str, Any]:
        """See https://numpy.org/doc/stable/reference/arrays.interface.html."""
        ...


class CUDABuffer:
    """CUDABuffer()
    CUDABuffer implements CUDA array interface.

    To be passed to casting functions like :py:func:`~spdl.io.to_torch` and
    :py:func:`~spdl.io.to_numba`.
    """

    @property
    def device_index(self) -> int:
        """The device index."""
        ...

    @property
    def __cuda_array_interface__(self) -> dict[str, Any]:
        """See https://numba.pydata.org/numba-doc/latest/cuda/cuda_array_interface.html."""
        ...


class DemuxConfig:
    """DemuxConfig()
    Demux configuration.

    See the factory function :py:func:`~spdl.io.demux_config`."""


class DecodeConfig:
    """DecodeConfig()
    Decode configuration.

    See the factory function :py:func:`~spdl.io.decode_config`."""


class EncodeConfig:
    """EncodeConfig()
    Encode configuration.

    See the factory function :py:func:`~spdl.io.encode_config`."""


class CUDAConfig:
    """CUDAConfig()
    Specify the CUDA devie and memory management.

    See the factory function :py:func:`~spdl.io.cuda_config`."""


class CPUStorage:
    """CPUStorage()
    Allocate a block of CPU memory.

    See the factory function :py:func:`~spdl.io.cpu_storage`."""
