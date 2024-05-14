from __future__ import annotations

from typing import Any, Dict, List, Tuple

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
]


class Packets:
    """Packets object. Returned from demux functions.

    Packets objects represent the result of the demuxing.
    Internally, it holds a series of FFmpeg's `AVPacket` objects.

    Decode function recieve Packets objects and generate audio samples and
    visual frames.
    The reason why Packets objects are exposed to public API is to allow
    a composition of demux/decoding functions in a way that demuxing and
    decoding can be performed concurrently.

    For example, when decoding multiple clips from a single audio and video file,
    by emitting Packets objects in between, decoding can be started while the
    demuxer is demuxing the subsequent windows.

    The following code will kick-off the decoding job as soon as the streaming
    demux function yields a `VideoPackets` object.

    !!! note "Example"

        ```python
        src = "sample.mp4"
        windows = [
            (3, 5),
            (7, 9),
            (13, 15),
        ]

        tasks = []
        async for packets in spdl.io.async_stream_demux("video", src, windows):
            # Kick off decoding job while demux function is demuxing the next window
            task = asyncio.create_task(decode_packets(packets))
            task.append(task)

        # Wait for all the decoding to be complete
        asyncio.wait(tasks)
        ```

    !!! warning "Important: Lifetime of Packets object."

        When packets objects are passed to a decode function, its ownership is
        also passed to the function. Therefore, accessing the packets object after
        it is passed to decode function will cause an error.

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
           decoding function, the Packets object resouce is also released in a
           background.

        To decode packets multiple times, use the `clone` method.

        ```python
        >>> # Demux an image
        >>> packets = spdl.io.demux_media("image", "foo.png").result()
        >>> packets  # this works.
        ... ImagePackets<src="foo.png", pixel_format="rgb24", bit_rate=0, bits_per_sample=0, codec="png", width=320, height=240>
        >>>
        >>> # Decode the packets
        >>> frames = spdl.io.decode_packets(packets)
        >>> frames
        ... FFmpegImageFrames<pixel_format="rgb24", num_planes=1, width=320, height=240>
        >>>
        >>> # The packets object is no longer valid.
        >>> packets
        ... RuntimeWarning: nanobind: attempted to access an uninitialized instance of type 'spdl.lib._spdl_ffmpeg6.ImagePackets'!
        ...
        ... Traceback (most recent call last):
        ...   File "<stdin>", line 1, in <module>
        ... TypeError: __repr__(): incompatible function arguments. The following argument types are supported:
        ...     1. __repr__(self) -> str
        ...
        ... Invoked with types: spdl.lib._spdl_ffmpeg6.ImagePackets
        >>>
        >>> packets = spdl.io.demux_media("image", "foo.png").result()
        >>> # Decode the cloned packets
        >>> packets2 = packets.clone()
        >>> packets2
        ... ImagePackets<src="foo.png", pixel_format="rgb24", bit_rate=0, bits_per_sample=0, codec="png", width=320, height=240>
        >>> frames = spdl.io.decode_packets(packets)
        >>>
        >>> # The original packets object is still valid
        >>> packets
        ... ImagePackets<src="foo.png", pixel_format="rgb24", bit_rate=0, bits_per_sample=0, codec="png", width=320, height=240>
        ```

        !!! note

            The memory region that packets occupy is actually a reference countered
            buffer. Therefore, even though the method is called `clone`, the method
            does not copy the frame data.
    """


class AudioPackets(Packets):
    """Packets object containing audio samples."""

    @property
    def timestamp(self) -> Tuple[float, float]:
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
    def timestamp(self) -> Tuple[float, float]:
        """The window this packets covers, denoted by start and end time in second.

        This is the value specified by user when demuxing the stream.

        Returns:
            (Tuple[float, float]): timestamp
        """
        ...

    def clone(self) -> VideoPackets:
        """Clone the packets, so that data can be decoded multiple times.

        Returns:
            A clone of the packets.
        """
        ...


class ImagePackets(Packets):
    """Packets object contain an image frame."""

    def clone(self) -> ImagePackets:
        """Clone the packets, so that data can be decoded multiple times.

        Returns:
            A clone of the packets.
        """
        ...


class Frames:
    """Frames object. Returned from decode functions.

    Frames objects represent the result of the decoding and filtering.
    Internally, it holds a series of FFmpeg's `AVFrame` objects thus thery
    are not a contiguous memory object.
    """


class AudioFrames(Frames):
    """Audio frames."""

    @property
    def num_frames(self) -> int:
        """The number of audio frames. Same as `__len__` method.

        !!! note

            In SPDL,
            `The number of samples == the number of frames x the number of channels`
        """
        ...

    @property
    def sample_rate(self) -> int:
        """The sample rate."""
        ...

    @property
    def num_channels(self) -> int:
        """The number of channels."""
        ...

    @property
    def format(self) -> str:
        """The name of sample format.

        Possible values are

          - `"u8"` for unsigned 8-bit integer.
          - `"s16"`, `"s32"`, `"s64"` for signed 16-bit, 32-bit and 64-bit integer.
          - `"flt"`, `"dbl"` for 32-bit and 64-bit float.

        If the frame is planar format (separate planes for different channels), the
        name will be suffixed with `"p"`. When converted to buffer, the buffer's shape
        will be channel-first format `(channel, num_samples)` instead of interweaved
        `(num_samples, channel)`.
        """
        ...

    def __len__(self) -> int:
        """Returns the number of frames. Same as `num_frames`."""
        ...

    def clone(self) -> AudioFrames:
        """Clone the frames, so that data can be converted to buffer multiple times.

        Returns:
            A clone of the frame.
        """
        ...


class VideoFrames(Frames):
    """Video frames."""

    @property
    def num_frames(self) -> int:
        """The number of video frames. Same as `__len__` method."""
        ...

    @property
    def num_planes(self) -> int:
        """The number of planes in the each frame.

        !!! note

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
    def format(self) -> str:
        """The name of the pixel format."""
        ...

    def __len__(self) -> int:
        """Returns the number of frames. Same as `num_frames`."""
        ...

    def __getitem__(self, key: int | slice) -> ImageFrames | VideoFrames:
        """Slice frame by key.

        Args:
            key: If the key is int type, a single frame is returned as `ImageFrames`.
                If the key is slice type, a new `VideoFrames` object pointing the
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
    """Image frames."""

    @property
    def num_planes(self) -> int:
        """The number of planes in the each frame.

        See [VideoFrames][spdl.io.VideoFrames] for a caveat.
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
    def format(self) -> str:
        """The name of the pixel format."""
        ...

    def clone(self) -> VideoFrames:
        """Clone the frames, so that data can be converted to buffer multiple times.

        Returns:
            A clone of the frame.
        """
        ...


class CPUBuffer:
    """Buffer implements array interface.

    To be passed to casting functions like [spdl.io.to_numpy][],
    [spdl.io.to_torch][] or [spdl.io.to_numba][].
    """

    @property
    def __array_interface__(self) -> Dict[str, Any]:
        """See https://numpy.org/doc/stable/reference/arrays.interface.html."""
        ...


class CUDABuffer:
    """CUDABuffer implements CUDA array interface.

    To be passed to casting functions like [spdl.io.to_numpy][],
    [spdl.io.to_torch][] or [spdl.io.to_numba][].
    """

    @property
    def device_index(self) -> int:
        """The device index."""
        ...

    @property
    def __cuda_array_interface__(self) -> Dict[str, Any]:
        """See https://numba.pydata.org/numba-doc/latest/cuda/cuda_array_interface.html."""
        ...


class DemuxConfig:
    """Demux configuration.

    See the factory function [spdl.io.demux_config][]."""


class DecodeConfig:
    """Decode configuration.

    See the factory function [spdl.io.decode_config][]."""


class EncodeConfig:
    """Encode configuration.

    See the factory function [spdl.io.encode_config][]."""
