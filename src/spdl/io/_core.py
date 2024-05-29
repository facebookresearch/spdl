import asyncio
import functools
import logging
from typing import Awaitable, Callable, List, Tuple, TypeVar

from spdl.io import Packets, VideoPackets
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


def demux_audio(src: str | bytes, **kwargs):
    return _libspdl.demux_audio(src, **kwargs)


def demux_video(src: str | bytes, **kwargs):
    return _libspdl.demux_video(src, **kwargs)


def demux_image(src: str | bytes, **kwargs):
    return _libspdl.demux_image(src, **kwargs)


async def async_demux_audio(src: str | bytes, **kwargs):
    return await _run_async(demux_audio, src, **kwargs)


async def async_demux_video(src: str | bytes, **kwargs):
    return await _run_async(demux_video, src, **kwargs)


async def async_demux_image(src: str | bytes, **kwargs):
    return await _run_async(demux_image, src, **kwargs)


async def async_streaming_demux_audio(
    src: str | bytes, timestamps: List[Tuple[float, float]], **kwargs
):
    demuxer = await _run_async(_libspdl.streaming_audio_demuxer, src, **kwargs)
    for window in timestamps:
        yield await _run_async(demuxer.demux_window, window)


async def async_streaming_demux_video(
    src: str | bytes, timestamps: List[Tuple[float, float]], **kwargs
):
    demuxer = await _run_async(_libspdl.streaming_video_demuxer, src, **kwargs)
    for window in timestamps:
        yield await _run_async(demuxer.demux_window, window)


################################################################################
# Decoding
################################################################################


def decode_packets(packets: Packets, **kwargs):
    if "filter_desc" not in kwargs:
        kwargs["filter_desc"] = _preprocessing.get_filter_desc(packets)
    return _libspdl.decode_packets(packets, **kwargs)


async def async_decode_packets(*args, **kwargs):
    return await _run_async(decode_packets, *args, **kwargs)


def decode_packets_nvdec(packets: VideoPackets, *, cuda_device_index: int, **kwargs):
    return _libspdl.decode_packets_nvdec(
        packets, cuda_device_index=cuda_device_index, **kwargs
    )


async def async_decode_packets_nvdec(*args, **kwargs):
    return await _run_async(decode_packets_nvdec, *args, **kwargs)


def decode_image_nvjpeg(src: str | bytes, *, cuda_device_index: int, **kwargs):
    if isinstance(src, bytes):
        data = src
    else:
        with open(src, "rb") as f:
            data = f.read()
    return _libspdl.decode_image_nvjpeg(data, cuda_device_index, **kwargs)


async def async_decode_image_nvjpeg(*args, **kwargs):
    return await _run_async(decode_image_nvjpeg, *args, **kwargs)


async def async_streaming_decode(packets: Packets, num_frames: int, **kwargs):
    decoder = await _run_async(_libspdl.streaming_decoder, packets, **kwargs)
    while (item := await _run_async(decoder.decode, num_frames)) is not None:
        yield item


################################################################################
# Frame conversion
################################################################################


def convert_frames(
    frames,
    cuda_device_index: int | None = None,
    **kwargs,
):
    if cuda_device_index is not None:
        func = _libspdl.convert_frames_cuda
        kwargs["cuda_device_index"] = cuda_device_index
    else:
        func = _libspdl.convert_frames
    return func(frames, **kwargs)


async def async_convert_frames(*args, **kwargs):
    return await _run_async(convert_frames, *args, **kwargs)


################################################################################
# Encoding
################################################################################

Array = TypeVar("Array")


def encode_image(path: str, data: Array, pix_fmt: str = "rgb24", **kwargs):
    return _libspdl.encode_image(path, data, pix_fmt=pix_fmt, **kwargs)


async def async_encode_image(*args, **kwargs):
    return await _run_async(encode_image, *args, **kwargs)
