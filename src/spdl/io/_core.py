import asyncio
import contextlib
import functools
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from typing import overload, TypeVar

from spdl.io import (
    AudioFrames,
    AudioPackets,
    CPUBuffer,
    CUDABuffer,
    ImageFrames,
    ImagePackets,
    VideoFrames,
    VideoPackets,
)
from spdl.lib import _libspdl

from . import _preprocessing

__all__ = [
    # DEMUXING
    "demux_audio",
    "demux_video",
    "demux_image",
    "async_demux_audio",
    "async_demux_video",
    "async_demux_image",
    "streaming_demux_audio",
    "streaming_demux_video",
    "async_streaming_demux_audio",
    "async_streaming_demux_video",
    # DECODING
    "decode_packets",
    "decode_packets_nvdec",
    "decode_image_nvjpeg",
    "async_decode_packets",
    "async_decode_packets_nvdec",
    "async_decode_image_nvjpeg",
    "async_streaming_decode",
    # FRAME CONVERSION
    "convert_frames",
    "async_convert_frames",
    # DATA TRANSFER
    "transfer_buffer",
    "async_transfer_buffer",
    # ENCODING
    "encode_image",
    "async_encode_image",
]

_LG = logging.getLogger(__name__)
T = TypeVar("T")


def _run_async(func: Callable[..., T], *args, executor=None, **kwargs) -> Awaitable[T]:
    loop = asyncio.get_running_loop()
    _func = functools.partial(func, *args, **kwargs)
    return loop.run_in_executor(executor, _func)  # pyre-ignore: [6]


################################################################################
# Demuxing
################################################################################


def demux_audio(
    src: str | bytes, *, timestamp: tuple[float, float] | None = None, **kwargs
) -> AudioPackets:
    return _libspdl.demux_audio(src, timestamp=timestamp, **kwargs)


def demux_video(
    src: str | bytes, *, timestamp: tuple[float, float] | None = None, **kwargs
) -> VideoPackets:
    return _libspdl.demux_video(src, timestamp=timestamp, **kwargs)


def demux_image(src: str | bytes, **kwargs) -> ImagePackets:
    return _libspdl.demux_image(src, **kwargs)


async def async_demux_audio(
    src: str | bytes, *, timestamp: tuple[float, float] | None = None, **kwargs
) -> AudioPackets:
    """Demux audio from the source.

    Args:
        src: Source identifier. If `str` type, it is interpreted as a source location,
            such as local file path or URL. If `bytes` type, then
            they are interpreted as in-memory data.

        timestamp:
            A time window. If omitted, the entire data are demuxed.

    Other args:
        demux_config (DemuxConfig): Custom I/O config.

    Returns:
        (AudioPackets): object.
    """
    return await _run_async(demux_audio, src, timestamp=timestamp, **kwargs)


async def async_demux_video(
    src: str | bytes, *, timestamp: tuple[float, float] | None = None, **kwargs
) -> VideoPackets:
    """Demux video from the source.

    Args:
        src: Source identifier. If `str` type, it is interpreted as a source location,
            such as local file path or URL. If `bytes` type, then
            they are interpreted as in-memory data.

        timestamp:
            A time window. If omitted, the entire data are demuxed.

    Other args:
        demux_config (DemuxConfig): Custom I/O config.

    Returns:
        (VideoPackets): object.
    """
    return await _run_async(demux_video, src, timestamp=timestamp, **kwargs)


async def async_demux_image(src: str | bytes, **kwargs) -> ImagePackets:
    """Demux image from the source.

    Args:
        src: Source identifier. If `str` type, it is interpreted as a source location,
            such as local file path or URL. If `bytes` type, then
            they are interpreted as in-memory data.


    Other args:
        demux_config (DemuxConfig): Custom I/O config.

    Returns:
        (ImagePackets): object.
    """
    return await _run_async(demux_image, src, **kwargs)


def streaming_demux_audio(
    src: str | bytes, timestamps: list[tuple[float, float]], **kwargs
) -> Iterator[AudioPackets]:
    demuxer = _libspdl._streaming_audio_demuxer(src, **kwargs)
    for window in timestamps:
        demuxer, packets = _libspdl._demux(demuxer, window)
        yield packets


def streaming_demux_video(
    src: str | bytes, timestamps: list[tuple[float, float]], **kwargs
) -> Iterator[VideoPackets]:
    demuxer = _libspdl._streaming_video_demuxer(src, **kwargs)
    for window in timestamps:
        demuxer, packets = _libspdl._demux(demuxer, window)
        yield packets


class _streaming_demuxer_wrpper:
    def __init__(self, demuxer):
        self.demuxer = demuxer

    async def demux(self, window):
        self.demuxer, packets = await _run_async(_libspdl._demux, self.demuxer, window)
        return packets


@contextlib.asynccontextmanager
async def _streaming_demuxer(constructor, src, **kwargs):
    demuxer = await _run_async(constructor, src, **kwargs)
    wrapper = _streaming_demuxer_wrpper(demuxer)
    try:
        yield wrapper
    finally:
        # Move the deallocation to the background thread.
        # (Do not deallocate memory in the main thread)
        await _run_async(_libspdl._drop, wrapper.demuxer)


async def _stream_demux(demuxer, src, timestamps, **kwargs):
    async with _streaming_demuxer(demuxer, src, **kwargs) as _demuxer:
        for window in timestamps:
            yield await _demuxer.demux(window)


async def async_streaming_demux_audio(
    src: str | bytes, timestamps: list[tuple[float, float]], **kwargs
) -> AsyncIterator[AudioPackets]:
    """Demux audio of the given time windows.

    Args:
        src: Source identifier. If `str` type, it is interpreted as a source location,
            such as local file path or URL. If `bytes` type, then
            they are interpreted as in-memory data.
        timestamps: List of timestamps, indicating the start and end time of the window
            in seconds.

    Other args:
        demux_config (DemuxConfig): Custom I/O config.

    Yields:
        (AudioPackets): AudioPackets object, corresponds to the given window.
    """
    demuxer = _libspdl._streaming_audio_demuxer
    async for packets in _stream_demux(demuxer, src, timestamps, **kwargs):
        yield packets


async def async_streaming_demux_video(
    src: str | bytes, timestamps: list[tuple[float, float]], **kwargs
) -> AsyncIterator[VideoPackets]:
    """Demux video of the given time windows.

    Args:
        src: Source identifier. If `str` type, it is interpreted as a source location,
            such as local file path or URL. If `bytes` type, then
            they are interpreted as in-memory data.
        timestamps: List of timestamps, indicating the start and end time of the window
            in seconds.

    Other args:
        demux_config (DemuxConfig): Custom I/O config.

    Returns:
        (VideoPackets): VideoPackets object, corresponds to the given window.
    """
    demuxer = _libspdl._streaming_video_demuxer
    async for packets in _stream_demux(demuxer, src, timestamps, **kwargs):
        yield packets


################################################################################
# Decoding
################################################################################


@overload
def decode_packets(packets: AudioPackets, **kwargs) -> AudioFrames: ...
@overload
def decode_packets(packets: VideoPackets, **kwargs) -> VideoFrames: ...
@overload
def decode_packets(packets: ImagePackets, **kwargs) -> ImageFrames: ...


def decode_packets(packets, **kwargs):
    if "filter_desc" not in kwargs:
        kwargs["filter_desc"] = _preprocessing.get_filter_desc(packets)
    return _libspdl.decode_packets(packets, **kwargs)


@overload
async def async_decode_packets(packets: AudioPackets, **kwargs) -> AudioFrames: ...
@overload
async def async_decode_packets(packets: VideoPackets, **kwargs) -> VideoFrames: ...
@overload
async def async_decode_packets(packets: ImagePackets, **kwargs) -> ImageFrames: ...


async def async_decode_packets(packets, **kwargs):
    """Decode packets.

    Args:
        packets (AudioPackets, VideoPackets or ImagePackets): Packets object.

    Other args:
        decoder_config (DecodeConfig):
            *Optional:* Custom decode config.

        filter_desc (str):
            *Optional:* Custom filter applied after decoding.

    Returns:
        (AudioFrames, VideoFrames or ImageFrames): A Frames object.
            The media type of the returned object corresponds to the input Packets type.
    """
    return await _run_async(decode_packets, packets, **kwargs)


def decode_packets_nvdec(
    packets: VideoPackets | ImagePackets | list[ImagePackets],
    *,
    cuda_device_index: int,
    **kwargs,
) -> CUDABuffer:
    return _libspdl.decode_packets_nvdec(
        packets, cuda_device_index=cuda_device_index, **kwargs
    )


async def async_decode_packets_nvdec(
    packets: VideoPackets | ImagePackets | list[ImagePackets],
    *,
    cuda_device_index: int,
    **kwargs,
) -> CUDABuffer:
    """Decode packets with NVDEC.

    Unlike FFmpeg-based decoding, NVDEC returns GPU buffer directly.

    ``` mermaid
    graph LR
      Source -->|Demux| Packets;
      Packets -->|Decode| Buffer;
      Buffer -->|Cast| Array[Array / Tensor];
    ```

    Args:
        packets: Packets object.

        cuda_device_index: The CUDA device to use for decoding.

    Other args:
        crop_left,crop_top,crop_right,crop_bottom (int):
            *Optional:* Crop the given number of pixels from each side.

        width,height (int): *Optional:* Resize the frame. Resizing is done after
            cropping.

        pix_fmt (str or `None`): *Optional:* Change the format of the pixel.
            Supported value is `"rgba"`. Default: `"rgba"`.

    Returns:
        A CUDABuffer object.
    """
    return await _run_async(
        decode_packets_nvdec, packets, cuda_device_index=cuda_device_index, **kwargs
    )


def decode_image_nvjpeg(
    src: str | bytes, *, cuda_device_index: int, **kwargs
) -> CUDABuffer:
    if isinstance(src, bytes):
        data = src
    else:
        with open(src, "rb") as f:
            data = f.read()
    return _libspdl.decode_image_nvjpeg(
        data, cuda_device_index=cuda_device_index, **kwargs
    )


async def async_decode_image_nvjpeg(
    src: str | bytes, *, cuda_device_index: int, **kwargs
) -> CUDABuffer:
    """Decode image with NVJPEG.

    Unlike FFmpeg-based decoding, nvJPEG returns GPU buffer directly.

    ``` mermaid
    graph LR
      Source -->|Decode| Buffer;
      Buffer -->|Cast| Array[Array / Tensor];
    ```

    Args:
        data: File path to a JPEG image or data in bytes.
        cuda_device_index: The CUDA device to use for decoding.

    Other args:
        pix_fmt (str): *Optional* Output pixel format.
            Supported values are `"RGB"` or `"BGR"`.

        cuda_allocator: See [async_convert_frames][spdl.io.async_convert_frames].

    Returns:
        A CUDABuffer object. Shape is [C==3, H, W].
    """
    return await _run_async(
        decode_image_nvjpeg, src, cuda_device_index=cuda_device_index, **kwargs
    )


class _streaming_decoder_wrpper:
    def __init__(self, decoder):
        self.decoder = decoder

    async def decode(self, num_frames):
        self.decoder, frames = await _run_async(
            _libspdl._decode, self.decoder, num_frames
        )
        return frames


@contextlib.asynccontextmanager
async def _streaming_decoder(constructor, packets, **kwargs):
    decoder = await _run_async(constructor, packets, **kwargs)
    wrapper = _streaming_decoder_wrpper(decoder)
    try:
        yield wrapper
    finally:
        await _run_async(_libspdl._drop, wrapper.decoder)


async def async_streaming_decode(
    packets: VideoPackets, num_frames: int, **kwargs
) -> AsyncIterator[VideoFrames]:
    """Decode the video packets chunk by chunk.

    Args:
        packets: VideoPackets object.
        num_frames: Number of frames to decode at a time.

    Yields:
        VideoFrames object containing at most `num_frames` frames.
    """
    decoder = _libspdl._streaming_decoder
    async with _streaming_decoder(decoder, packets, **kwargs) as _decoder:
        while (item := await _decoder.decode(num_frames)) is not None:
            yield item


################################################################################
# Frame conversion
################################################################################


def convert_frames(
    frames: AudioFrames | VideoFrames | ImageFrames,
    **kwargs,
) -> CPUBuffer:
    return _libspdl.convert_frames(frames, **kwargs)


async def async_convert_frames(
    frames: AudioFrames | VideoFrames | ImageFrames,
    **kwargs,
) -> CPUBuffer:
    """Convert the decoded frames to buffer.

    Args:
        frames: Frames objects.

    Returns:
        (CPUBuffer): A Buffer object.
    """
    return await _run_async(convert_frames, frames, **kwargs)


################################################################################
# Device data transfer
################################################################################
def transfer_buffer(buffer: CPUBuffer, **kwargs) -> CUDABuffer:
    return _libspdl.transfer_to_cuda(buffer, **kwargs)


async def async_transfer_buffer(buffer: CPUBuffer, **kwargs) -> CUDABuffer:
    """Transfer the given buffer to CUDA device.

    Args:
        data (CPUBuffer): Buffer object on host memory.

    Other args:
        cuda_device_index (int):
            *Optional:* When provided, the buffer is moved to CUDA device.

        cuda_stream (int (uintptr_t) ):
            *Optional:* Pointer to a custom CUDA stream. By default, it uses the
            per-thread default stream.

            !!! note

                Host to device buffer transfer is performed in a thread different than
                Python main thread.

                Since the frame data are available only for the duration of the
                background job, the transfer is performed with synchronization.

                It is possible to provide the same stream as the one used in Python's
                main thread, but it might introduce undesired synchronization.

            ??? note "How to retrieve CUDA stream pointer on PyTorch"

                An example to fetch the default stream from PyTorch.

                ```python
                stream = torch.cuda.Stream()
                cuda_stream = stream.cuda_stream
                ```

        cuda_allocator (tuple[Callable[[int, int, int], int], Callable[[int], None]]):
            *Optional:* A pair of custom CUDA memory allcoator and deleter functions.

            The allocator function, takes the following arguments, and
            return the address of the allocated memory.

            - Size: `int`
            - CUDA device index: `int`
            - CUDA stream address: `int` (`uintptr_t`)

            An example of such function is
            [PyTorch's CUDA caching allocator][torch.cuda.caching_allocator_alloc].

            The deleter takes the address of memory allocated
            by the allocator and free the memory.

            An example of such function is
            [PyTorch's CUDA caching allocator][torch.cuda.caching_allocator_delete].

    Returns:
        (CUDABuffer): A Buffer object.
    """
    return await _run_async(transfer_buffer, buffer, **kwargs)


################################################################################
# Encoding
################################################################################

Array = TypeVar("Array")


def encode_image(path: str, data: Array, pix_fmt: str = "rgb24", **kwargs):
    return _libspdl.encode_image(path, data, pix_fmt=pix_fmt, **kwargs)


async def async_encode_image(path: str, data: Array, pix_fmt: str = "rgb24", **kwargs):
    """Save the given image array/tensor to file.

    Args:
        dst: The path to which the data are written.

        data (NumPy NDArray, PyTorch Tensor):
            Image data in array format. The data  must be `uint8` type,
            either on CPU or CUDA device.

            The shape must be one of the following and must match the
            value of `pix_fmt`.

            - `(height, width, channel==3)` when `pix_fmt="rgb24"`
            - `(height, width)` when `pix_fmt=gray8`
            - `(channel==3, height, width)` when `pix_fmt="yuv444p"`

        pix_fmt: See above.

    Other args:
        encode_config (EncodeConfig): Customize the encoding.

        executor (Executor): Thread pool executor to run the operation.
            By default, it uses the default encode thread pool.

    ??? note "Example - Save image as PNG with resizing"

        ```python
        >>> import asyncio
        >>> import numpy as np
        >>> import spdl.io
        >>>
        >>> data = np.random.randint(255, size=(32, 16, 3), dtype=np.uint8)
        >>> coro = spdl.io.async_encode_image(
        ...     "foo.png",
        ...     data,
        ...     pix_fmt="rgb24",
        ...     encode_config=spdl.io.encode_config(width=198, height=96, scale_algo="neighbor"),
        ... )
        >>> asyncio.run(coro)
        >>>
        ```

    ??? note "Example - Directly save CUDA tensor as image"

        ```python
        >>> import torch
        >>>
        >>> data = torch.randint(255, size=(32, 16, 3), dtype=torch.uint8, device="cuda")
        >>> coro = spdl.io.async_encode_image(
        ...     "foo.png",
        ...     data,
        ...     pix_fmt="rgb24",
        ... )
        >>> asyncio.run(coro)
        >>>
        ```

    """
    return await _run_async(encode_image, path, data, pix_fmt, **kwargs)
