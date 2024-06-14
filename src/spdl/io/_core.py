# pyre-unsafe

import contextlib
import logging
from collections.abc import AsyncIterator, Iterator
from typing import overload, TypeVar

from spdl.io import (
    AudioFrames,
    AudioPackets,
    CPUBuffer,
    CUDABuffer,
    CUDAConfig,
    ImageFrames,
    ImagePackets,
    VideoFrames,
    VideoPackets,
)
from spdl.lib import _libspdl
from spdl.utils import run_async  # pyre-ignore: [21]

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
    "async_decode_packets",
    "decode_packets_nvdec",
    "async_decode_packets_nvdec",
    "streaming_decode_packets",
    "async_streaming_decode_packets",
    "decode_image_nvjpeg",
    "async_decode_image_nvjpeg",
    # FRAME CONVERSION
    "convert_frames",
    "async_convert_frames",
    # DATA TRANSFER
    "transfer_buffer",
    "async_transfer_buffer",
    "transfer_buffer_cpu",
    "async_transfer_buffer_cpu",
    # ENCODING
    "encode_image",
    "async_encode_image",
]

_LG = logging.getLogger(__name__)
T = TypeVar("T")


################################################################################
# Demuxing
################################################################################


def demux_audio(
    src: str | bytes, *, timestamp: tuple[float, float] | None = None, **kwargs
) -> AudioPackets:
    """Demux audio from the source.

    Args:
        src: Source identifier.
            If `str` type, it is interpreted as a source location,
            such as local file path or URL. If `bytes` type,
            then they are interpreted as in-memory data.

        timestamp:
            A time window. If omitted, the entire audio are demuxed.

    Other args:
        demux_config (DemuxConfig): Custom I/O config.

    Returns:
        (AudioPackets): object.
    """
    return _libspdl.demux_audio(src, timestamp=timestamp, **kwargs)


async def async_demux_audio(
    src: str | bytes, *, timestamp: tuple[float, float] | None = None, **kwargs
) -> AudioPackets:
    """Async version of [demux_audio][spdl.io.demux_audio]"""
    return await run_async(demux_audio, src, timestamp=timestamp, **kwargs)


def demux_video(
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
    return _libspdl.demux_video(src, timestamp=timestamp, **kwargs)


async def async_demux_video(
    src: str | bytes, *, timestamp: tuple[float, float] | None = None, **kwargs
) -> VideoPackets:
    """Async version of [demux_video][spdl.io.demux_video]"""
    return await run_async(demux_video, src, timestamp=timestamp, **kwargs)


def demux_image(src: str | bytes, **kwargs) -> ImagePackets:
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
    return _libspdl.demux_image(src, **kwargs)


async def async_demux_image(src: str | bytes, **kwargs) -> ImagePackets:
    """Async version of [demux_image][spdl.io.demux_image]"""
    return await run_async(demux_image, src, **kwargs)


################################################################################
# Demuxing
################################################################################


def streaming_demux_audio(
    src: str | bytes, timestamps: list[tuple[float, float]], **kwargs
) -> Iterator[AudioPackets]:
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
        AudioPackets object, corresponds to the given window.
    """
    demuxer = _libspdl._demuxer(src, **kwargs)
    for window in timestamps:
        demuxer, packets = _libspdl._demux_audio(demuxer, window)
        yield packets


def streaming_demux_video(
    src: str | bytes, timestamps: list[tuple[float, float]], **kwargs
) -> Iterator[VideoPackets]:
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
        VideoPackets object, corresponds to the given window.
    """
    demuxer = _libspdl._demuxer(src, **kwargs)
    for window in timestamps:
        demuxer, packets = _libspdl._demux_video(demuxer, window)
        yield packets


class _streaming_demuxer_wrpper:
    def __init__(self, demuxer, media_type):
        self.demuxer = demuxer
        match media_type:
            case "audio":
                self.demux_func = _libspdl._demux_audio
            case "video":
                self.demux_func = _libspdl._demux_video
            case _:
                raise ValueError(f"Unsupported media type: {media_type}")

    async def demux(self, window, **kwargs):
        self.demuxer, packets = await run_async(
            self.demux_func, self.demuxer, window, **kwargs
        )
        return packets


@contextlib.asynccontextmanager
async def _streaming_demuxer(media_type, src, **kwargs):
    demuxer = await run_async(_libspdl._demuxer, src, **kwargs)
    wrapper = _streaming_demuxer_wrpper(demuxer, media_type)
    try:
        yield wrapper
    finally:
        # Move the deallocation to the background thread.
        # (Do not deallocate memory in the main thread)
        await run_async(_libspdl._drop, wrapper.demuxer)


async def _stream_demux(media_type, src, timestamps, bsf=None, **kwargs):
    async with _streaming_demuxer(media_type, src, **kwargs) as _demuxer:
        for window in timestamps:
            yield await _demuxer.demux(window, bsf=bsf)


async def async_streaming_demux_audio(
    src: str | bytes, timestamps: list[tuple[float, float]], **kwargs
) -> AsyncIterator[AudioPackets]:
    """Async version of [streaming_demux_audio][spdl.io.streaming_demux_audio]."""
    async for packets in _stream_demux("audio", src, timestamps, **kwargs):
        yield packets


async def async_streaming_demux_video(
    src: str | bytes, timestamps: list[tuple[float, float]], **kwargs
) -> AsyncIterator[VideoPackets]:
    """Async version of [streaming_demux_video][spdl.io.streaming_demux_video]."""
    async for packets in _stream_demux("video", src, timestamps, **kwargs):
        yield packets


################################################################################
# Decoding
################################################################################


@overload
def decode_packets(
    packets: AudioPackets, filter_desc: str | None = None, **kwargs
) -> AudioFrames: ...
@overload
def decode_packets(
    packets: VideoPackets, filter_desc: str | None = None, **kwargs
) -> VideoFrames: ...
@overload
def decode_packets(
    packets: ImagePackets, filter_desc: str | None = None, **kwargs
) -> ImageFrames: ...


def decode_packets(packets, filter_desc=None, **kwargs):
    """Decode packets.

    Args:
        packets (AudioPackets, VideoPackets or ImagePackets): Packets object.

    Other args:
        decode_config (DecodeConfig):
            *Optional:* Custom decode config.

        filter_desc (str):
            *Optional:* Custom filter applied after decoding.

    Returns:
        (AudioFrames, VideoFrames or ImageFrames): A Frames object.
            The media type of the returned object corresponds to the input Packets type.
    """
    if filter_desc is None:
        filter_desc = _preprocessing.get_filter_desc(packets)
    return _libspdl.decode_packets(packets, filter_desc=filter_desc, **kwargs)


@overload
async def async_decode_packets(packets: AudioPackets, **kwargs) -> AudioFrames: ...
@overload
async def async_decode_packets(packets: VideoPackets, **kwargs) -> VideoFrames: ...
@overload
async def async_decode_packets(packets: ImagePackets, **kwargs) -> ImageFrames: ...


async def async_decode_packets(packets, **kwargs):
    """Async version of [decode_packets][spdl.io.decode_packets]."""
    return await run_async(decode_packets, packets, **kwargs)


def decode_packets_nvdec(
    packets: VideoPackets | ImagePackets | list[ImagePackets],
    *,
    cuda_config: CUDAConfig,
    **kwargs,
) -> CUDABuffer:
    """Decode packets with NVDEC.

    Unlike FFmpeg-based decoding, NVDEC returns GPU buffer directly.

    Args:
        packets: Packets object.

        cuda_config: The CUDA device to use for decoding.

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
    return _libspdl.decode_packets_nvdec(packets, cuda_config=cuda_config, **kwargs)


async def async_decode_packets_nvdec(
    packets: VideoPackets | ImagePackets | list[ImagePackets],
    *,
    cuda_config: CUDAConfig,
    **kwargs,
) -> CUDABuffer:
    """Async version of [decode_packets_nvdec][spdl.io.decode_packets_nvdec]."""
    return await run_async(
        decode_packets_nvdec, packets, cuda_config=cuda_config, **kwargs
    )


def decode_image_nvjpeg(src: str | bytes, *, cuda_config: int, **kwargs) -> CUDABuffer:
    """Decode image with nvJPEG.

    Unlike FFmpeg-based decoding, nvJPEG returns GPU buffer directly.

    Args:
        src: File path to a JPEG image or data in bytes.
        cuda_config: The CUDA device to use for decoding.

    Other args:
        pix_fmt (str): *Optional* Output pixel format.
            Supported values are `"RGB"` or `"BGR"`.

    Returns:
        A CUDABuffer object. Shape is [C==3, H, W].
    """
    if isinstance(src, bytes):
        data = src
    else:
        with open(src, "rb") as f:
            data = f.read()
    return _libspdl.decode_image_nvjpeg(data, cuda_config=cuda_config, **kwargs)


async def async_decode_image_nvjpeg(
    src: str | bytes, *, cuda_config: CUDAConfig, **kwargs
) -> CUDABuffer:
    """Async version of [decode_image_nvjpeg][spdl.io.decode_image_nvjpeg]."""
    return await run_async(decode_image_nvjpeg, src, cuda_config=cuda_config, **kwargs)


def streaming_decode_packets(
    packets: VideoPackets, num_frames: int, **kwargs
) -> Iterator[VideoFrames]:
    """Decode the video packets chunk by chunk.

    Args:
        packets: VideoPackets object.
        num_frames: Number of frames to decode at a time.

    Yields:
        VideoFrames object containing at most `num_frames` frames.
    """
    decoder = _libspdl._streaming_decoder(packets, **kwargs)
    while True:
        decoder, frames = _libspdl._decode(decoder, num_frames)
        if frames is None:
            return
        yield frames


class _streaming_decoder_wrpper:
    def __init__(self, decoder):
        self.decoder = decoder

    async def decode(self, num_frames):
        self.decoder, frames = await run_async(
            _libspdl._decode, self.decoder, num_frames
        )
        return frames


@contextlib.asynccontextmanager
async def _streaming_decoder(constructor, packets, **kwargs):
    decoder = await run_async(constructor, packets, **kwargs)
    wrapper = _streaming_decoder_wrpper(decoder)
    try:
        yield wrapper
    finally:
        await run_async(_libspdl._drop, wrapper.decoder)


async def async_streaming_decode_packets(
    packets: VideoPackets, num_frames: int, **kwargs
) -> AsyncIterator[VideoFrames]:
    """Async version of [streaming_decode_packets][spdl.io.streaming_decode_packets]."""
    decoder = _libspdl._streaming_decoder
    async with _streaming_decoder(decoder, packets, **kwargs) as _decoder:
        while (item := await _decoder.decode(num_frames)) is not None:
            yield item


################################################################################
# Frame conversion
################################################################################


def convert_frames(
    frames: (
        AudioFrames
        | VideoFrames
        | ImageFrames
        | list[AudioFrames]
        | list[VideoFrames]
        | list[ImageFrames]
    ),
    **kwargs,
) -> CPUBuffer:
    """Convert the decoded frames to buffer.

    Args:
        frames: Frames objects.

    Returns:
        (CPUBuffer): A Buffer object.
            The shape of the buffer obejct is

            - AudioFrames -> `[C, H]` or `[N, C]`.
            - VideoFrames -> `[N, C, H, W]` or `[N, H, W, C]`.
            - ImageFrames -> `[C, H, W]`.

            where

            - `C`: channel (color channel or audio channel)
            - `N`: frames
            - `W`: width
            - `H`: height

            If a list of Frames objects are passed, then they are treated as one batch,
            so that the resulting buffer object has batch dimention at the beginning.
    """
    return _libspdl.convert_frames(frames, **kwargs)


async def async_convert_frames(
    frames: AudioFrames | VideoFrames | ImageFrames | list[ImageFrames],
    **kwargs,
) -> CPUBuffer:
    """Async version of [convert_frames][spdl.io.convert_frames]."""
    return await run_async(convert_frames, frames, **kwargs)


################################################################################
# Device data transfer
################################################################################
def transfer_buffer(buffer: CPUBuffer, *, cuda_config: CUDAConfig) -> CUDABuffer:
    """Move the given CPU buffer to GPU.

    Args:
        buffer: Source data

        cuda_config: Target GPU.

    Returns:
        Buffer data on the target GPU.
    """
    return _libspdl.transfer_buffer(buffer, cuda_config=cuda_config)


async def async_transfer_buffer(
    buffer: CPUBuffer, *, cuda_config: CUDAConfig
) -> CUDABuffer:
    """Transfer the given buffer to CUDA device.

    Args:
        buffer: Buffer object on host memory.
        cuda_config: CUDA configuration.

    Returns:
        (CUDABuffer): A Buffer object.
    """
    return await run_async(transfer_buffer, buffer, cuda_config=cuda_config)


def transfer_buffer_cpu(buffer: CUDABuffer) -> CPUBuffer:
    """Copy the given CUDA buffer to CPU.

    Args:
        buffer: Source data

    Returns:
        Buffer data on CPU.
    """
    return _libspdl.transfer_buffer_cpu(buffer)


def async_transfer_buffer_cpu(buffer: CUDABuffer) -> CPUBuffer:
    """Copy the given CUDA buffer to CPU.

    Args:
        buffer: Source data

    Returns:
        Buffer data on CPU.
    """
    return run_async(transfer_buffer_cpu, buffer)


################################################################################
# Encoding
################################################################################

Array = TypeVar("Array")


def encode_image(path: str, data: Array, pix_fmt: str = "rgb24", **kwargs):
    """Save the given image array/tensor to file.

    Args:
        path: The path to which the data are written.

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
    return _libspdl.encode_image(path, data, pix_fmt=pix_fmt, **kwargs)


async def async_encode_image(path: str, data: Array, pix_fmt: str = "rgb24", **kwargs):
    """Async version of [encode_image][spdl.io.encode_image]."""
    return await run_async(encode_image, path, data, pix_fmt, **kwargs)
