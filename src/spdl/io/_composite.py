# pyre-unsafe

import asyncio
import builtins
import logging
from collections.abc import AsyncIterator, Iterator
from typing import Any

from spdl.io import (
    CPUBuffer,
    CUDABuffer,
    CUDAConfig,
    DecodeConfig,
    DemuxConfig,
    ImageFrames,
    ImagePackets,
    VideoPackets,
)

from spdl.lib import _libspdl
from spdl.utils import run_async

from . import _core, _preprocessing

__all__ = [
    "load_audio",
    "load_video",
    "load_image",
    "async_load_audio",
    "async_load_video",
    "async_load_image",
    "streaming_load_audio",
    "streaming_load_video",
    "async_streaming_load_audio",
    "async_streaming_load_video",
    "async_load_image_batch",
    "async_load_image_batch_nvdec",
    "load_image_batch_nvjpeg",
    "async_load_image_batch_nvjpeg",
    "async_sample_decode_video",
]

_LG = logging.getLogger(__name__)

Window = tuple[float, float]


################################################################################
# Load
################################################################################
def _load_packets(
    packets,
    decode_config: DecodeConfig | None = None,
    filter_desc: str | None = None,
    cuda_config: CUDAConfig | None = None,
):
    frames = _core.decode_packets(
        packets, decode_config=decode_config, filter_desc=filter_desc
    )
    buffer = _core.convert_frames(frames)
    if cuda_config is not None:
        buffer = _core.transfer_buffer(buffer, cuda_config=cuda_config)
    return buffer


def load_audio(
    src: str | bytes,
    timestamp: tuple[float, float] | None = None,
    *,
    demux_config: DemuxConfig | None = None,
    decode_config: DecodeConfig | None = None,
    filter_desc: str | None = None,
    cuda_config: CUDAConfig | None = None,
) -> CPUBuffer | CUDABuffer:
    """Load audio from source.

    This function combines [demux_audio][spdl.io.demux_audio],
    [decode_packets][spdl.io.decode_packets], [convert_frames][spdl.io.convert_frames],
    and optionally, [transfer_buffer][spdl.io.transfer_buffer], and produces
    buffer object in one step.

    Args:
        src: See [demux_audio][spdl.io.demux_audio].
        timestamp: See [demux_audio][spdl.io.demux_audio].
        demux_config: See [demux_audio][spdl.io.demux_audio].
        decode_config: See [decode_packets][spdl.io.decode_packets].
        filter_desc: See [decode_packets][spdl.io.decode_packets].
        cuda_config: See [transfer_buffer][spdl.io.transfer_buffer].

    Returns:
        Buffer object.
    """
    packets = _core.demux_audio(src, timestamp=timestamp, demux_config=demux_config)
    return _load_packets(
        packets,
        decode_config=decode_config,
        filter_desc=filter_desc,
        cuda_config=cuda_config,
    )


async def async_load_audio(
    src: str | bytes,
    timestamp: tuple[float, float] | None = None,
    *,
    demux_config: DemuxConfig | None = None,
    decode_config: DecodeConfig | None = None,
    filter_desc: str | None = None,
    cuda_config: CUDAConfig | None = None,
) -> CPUBuffer | CUDABuffer:
    """Async version of [load_audio][spdl.io.load_audio]."""
    return await run_async(
        load_audio,
        src,
        timestamp,
        demux_config=demux_config,
        decode_config=decode_config,
        filter_desc=filter_desc,
        cuda_config=cuda_config,
    )


def load_video(
    src: str | bytes,
    timestamp: tuple[float, float] | None = None,
    *,
    demux_config: DemuxConfig | None = None,
    decode_config: DecodeConfig | None = None,
    filter_desc: str | None = None,
    cuda_config: CUDAConfig | None = None,
) -> CPUBuffer | CUDABuffer:
    """Load video from source.

    This function combines [demux_video][spdl.io.demux_video],
    [decode_packets][spdl.io.decode_packets], [convert_frames][spdl.io.convert_frames],
    and optionally, [transfer_buffer][spdl.io.transfer_buffer], and produces
    buffer object in one step.

    Args:
        src: See [demux_video][spdl.io.demux_video].
        timestamp: See [demux_video][spdl.io.demux_video].
        demux_config: See [demux_video][spdl.io.demux_video].
        decode_config: See [decode_packets][spdl.io.decode_packets].
        filter_desc: See [decode_packets][spdl.io.decode_packets].
        cuda_config: See [transfer_buffer][spdl.io.transfer_buffer].

    Returns:
        Buffer object.
    """
    packets = _core.demux_video(src, timestamp=timestamp, demux_config=demux_config)
    return _load_packets(
        packets,
        decode_config=decode_config,
        filter_desc=filter_desc,
        cuda_config=cuda_config,
    )


async def async_load_video(
    src: str | bytes,
    timestamp: tuple[float, float] | None = None,
    *,
    demux_config: DemuxConfig | None = None,
    decode_config: DecodeConfig | None = None,
    filter_desc: str | None = None,
    cuda_config: CUDAConfig | None = None,
) -> CPUBuffer | CUDABuffer:
    """Async version of [load_video][spdl.io.load_video]."""
    return await run_async(
        load_video,
        src,
        timestamp,
        demux_config=demux_config,
        decode_config=decode_config,
        filter_desc=filter_desc,
        cuda_config=cuda_config,
    )


def load_image(
    src: str | bytes,
    *,
    demux_config: DemuxConfig | None = None,
    decode_config: DecodeConfig | None = None,
    filter_desc: str | None = None,
    cuda_config: CUDAConfig | None = None,
) -> CPUBuffer | CUDABuffer:
    """Load image from source.

    This function combines [demux_image][spdl.io.demux_image],
    [decode_packets][spdl.io.decode_packets], [convert_frames][spdl.io.convert_frames],
    and optionally, [transfer_buffer][spdl.io.transfer_buffer], and produces
    buffer object in one step.

    Args:
        src: See [demux_image][spdl.io.demux_image].
        demux_config: See [demux_image][spdl.io.demux_image].
        decode_config: See [decode_packets][spdl.io.decode_packets].
        filter_desc: See [decode_packets][spdl.io.decode_packets].
        cuda_config: See [transfer_buffer][spdl.io.transfer_buffer].

    Returns:
        Buffer object.
    """
    packets = _core.demux_image(src, demux_config=demux_config)
    return _load_packets(
        packets,
        decode_config=decode_config,
        filter_desc=filter_desc,
        cuda_config=cuda_config,
    )


async def async_load_image(
    src: str | bytes,
    *,
    demux_config: DemuxConfig | None = None,
    decode_config: DecodeConfig | None = None,
    filter_desc: str | None = None,
    cuda_config: CUDAConfig | None = None,
) -> CPUBuffer | CUDABuffer:
    """Async version of [load_image][spdl.io.load_image]."""
    return await run_async(
        load_image,
        src,
        demux_config=demux_config,
        decode_config=decode_config,
        filter_desc=filter_desc,
        cuda_config=cuda_config,
    )


################################################################################
# Streaming load
################################################################################


def _get_src_str(src):
    if isinstance(src, bytes):
        return f"bytes object at {id(src)}"
    return f"'{src}'"


def streaming_load_audio(
    src: str | bytes,
    timestamps: list[tuple[float, float]],
    *,
    demux_config: DemuxConfig | None = None,
    strict: bool = True,
    **kwargs,
) -> Iterator[CPUBuffer] | Iterator[CUDABuffer]:
    demuxer = _core.streaming_demux_audio(src, timestamps, demux_config=demux_config)
    for packets in demuxer:
        try:
            yield _load_packets(packets, **kwargs)
        except Exception:
            if strict:
                raise
            _LG.error(
                "Failed to load audio clip at %s from %s.",
                _get_src_str(src),
                packets.timestamp,
            )


# TODO: Add strict option
def streaming_load_video(
    src: str | bytes,
    timestamps: list[tuple[float, float]],
    *,
    demux_config: DemuxConfig | None = None,
    strict: bool = True,
    **kwargs,
) -> Iterator[CPUBuffer] | Iterator[CUDABuffer]:
    demuxer = _core.streaming_demux_video(src, timestamps, demux_config=demux_config)
    for packets in demuxer:
        try:
            yield _load_packets(packets, **kwargs)
        except Exception:
            if strict:
                raise
            _LG.error(
                "Failed to load video clip at %s from %s.",
                _get_src_str(src),
                packets.timestamp,
                exc_info=True,
            )


################################################################################
# Async streaming load
################################################################################
async def _async_load_packets(*args, **kwargs):
    return await run_async(_load_packets, *args)


# TODO: Add concurrency and strict option
async def _async_streaming_load(
    demuxer, **kwargs
) -> AsyncIterator[asyncio.Task[CPUBuffer | CUDABuffer]]:

    tasks = []
    async for packets in demuxer:
        tasks.append(asyncio.create_task(_async_load_packets(packets, **kwargs)))

        while tasks and tasks[0].done():
            yield tasks.pop(0)

    while tasks:
        await tasks[0]
        yield tasks.pop(0)


async def async_streaming_load_audio(
    src: str | bytes,
    timestamps: list[tuple[float, float]],
    *,
    demux_config: DemuxConfig | None = None,
    strict: bool = True,
    **kwargs,
) -> AsyncIterator[CPUBuffer | CUDABuffer]:
    demuxer = _core.async_streaming_demux_audio(
        src, timestamps, demux_config=demux_config
    )
    i = 0
    async for task in _async_streaming_load(demuxer, **kwargs):
        try:
            yield task.result()
        except Exception:
            if strict:
                raise
            _LG.error(
                "Failed to load audio clip  at %s from %s",
                timestamps[i],
                _get_src_str(src),
                exc_info=True,
            )
        finally:
            i += 1


async def async_streaming_load_video(
    src: str | bytes,
    timestamps: list[tuple[float, float]],
    *,
    demux_config: DemuxConfig | None = None,
    strict: bool = True,
    **kwargs,
) -> AsyncIterator[CPUBuffer | CUDABuffer]:
    demuxer = _core.async_streaming_demux_video(
        src, timestamps, demux_config=demux_config
    )
    i = 0
    async for task in _async_streaming_load(demuxer, **kwargs):
        try:
            yield task.result()
        except Exception:
            if strict:
                raise
            _LG.error(
                "Failed to load video clip at %s from %s",
                timestamps[i],
                _get_src_str(src),
                exc_info=True,
            )
        finally:
            i += 1


################################################################################
#
################################################################################


def _get_err_msg(src, err):
    match type(src):
        case builtins.bytes:
            src_ = f"bytes object at {id(src)}"
        case _:
            src_ = f"'{src}'"
    return f"Failed to decode an image from {src_}: {err}."


def _decode(src, demux_config, decode_config, filter_desc):
    pkts = _core.demux_image(src, demux_config=demux_config)
    return _core.decode_packets(
        pkts, decode_config=decode_config, filter_desc=filter_desc
    )


async def async_load_image_batch(
    srcs: list[str | bytes],
    *,
    width: int | None,
    height: int | None,
    pix_fmt: str | None = "rgb24",
    demux_config: DemuxConfig | None = None,
    decode_config: DecodeConfig | None = None,
    filter_desc: str | None = None,
    cuda_config: CUDAConfig | None = None,
    pin_memory: bool = False,
    strict: bool = True,
):
    """Batch load images.

    Args:
        srcs: List of source identifiers.

        width: *Optional:* Resize the frame.

        height: *Optional:* Resize the frame.

        pix_fmt:
            *Optional:* Change the format of the pixel.

        demux_options (dict[str, Any]):
            *Optional:* Demux options passed to [spdl.io.async_demux_media][].

        decode_options (dict[str, Any]):
            *Optional:* Decode options passed to [spdl.io.async_decode_packets][].

        convert_options (dict[str, Any]):
            *Optional:* Convert options passed to [spdl.io.async_convert_frames][].

        strict:
            *Optional:* If True, raise an error if any of the images failed to load.

    Returns:
        A buffer object.

    ??? note "Example"
        ```python
        >>> srcs = [
        ...     "sample1.jpg",
        ...     "sample2.png",
        ... ]
        >>> coro = async_batch_load_image(
        ...     srcs,
        ...     scale_width=124,
        ...     scale_height=96,
        ...     pix_fmt="rgb24",
        ... )
        >>> buffer = asyncio.run(coro)
        >>> array = spdl.io.to_numpy(buffer)
        >>> # An array with shape HWC==[2, 96, 124, 3]
        >>>
        ```

    """
    if not srcs:
        raise ValueError("`srcs` must not be empty.")

    filter_desc = _preprocessing.get_video_filter_desc(
        scale_width=width,
        scale_height=height,
        pix_fmt=pix_fmt,
        filter_desc=filter_desc,
    )

    tasks = [
        asyncio.create_task(
            run_async(_decode, src, demux_config, decode_config, filter_desc)
        )
        for src in srcs
    ]

    await asyncio.wait(tasks)

    frames: list[ImageFrames] = []
    for src, future in zip(srcs, tasks):
        try:
            frms = future.result()
        except Exception as err:
            _LG.error(_get_err_msg(src, err))
        else:
            frames.append(frms)

    if strict and len(frames) != len(srcs):
        raise RuntimeError("Failed to load some images.")

    if not frames:
        raise RuntimeError("Failed to load all the images.")

    buffer = await _core.async_convert_frames(frames, pin_memory=pin_memory)

    if cuda_config is not None:
        buffer = await _core.async_transfer_buffer(buffer, cuda_config=cuda_config)

    return buffer


async def async_load_image_batch_nvdec(
    srcs: list[str | bytes],
    *,
    cuda_config: CUDAConfig,
    width: int | None,
    height: int | None,
    pix_fmt: str | None = "rgba",
    demux_config: DemuxConfig | None = None,
    decode_options: dict[str, Any] | None = None,
    strict: bool = True,
):
    """Batch load images.

    Args:
        srcs: List of source identifiers.

        cuda_device_index: The CUDA device to use for decoding images.

        width: *Optional:* Resize the frame.

        height: *Optional:* Resize the frame.

        pix_fmt:
            *Optional:* Change the format of the pixel.

        demux_options (dict[str, Any]):
            *Optional:* Demux options passed to [spdl.io.async_demux_media][].

        decode_options (dict[str, Any]):
            *Optional:* Other decode options passed to [spdl.io.async_decode_packets_nvdec][].

        strict:
            *Optional:* If True, raise an error if any of the images failed to load.

    Returns:
        A buffer object.

    ??? note "Example"
        ```python
        >>> srcs = [
        ...     "sample1.jpg",
        ...     "sample2.png",
        ... ]
        >>> coro = async_batch_load_image_nvdec(
        ...     srcs,
        ...     cuda_device_index=0,
        ...     width=124,
        ...     height=96,
        ...     pix_fmt="rgba",
        ... )
        >>> buffer = asyncio.run(coro)
        >>> array = spdl.io.to_torch(buffer)
        >>> # An array with shape NCHW==[2, 4, 96, 124] on CUDA device 0
        >>>
        ```
    """
    if not srcs:
        raise ValueError("`srcs` must not be empty.")

    decode_options = decode_options or {}
    width = -1 if width is None else width
    height = -1 if height is None else height

    tasks = [
        asyncio.create_task(_core.async_demux_image(src, demux_config=demux_config))
        for src in srcs
    ]

    await asyncio.wait(tasks)

    packets = []
    for src, task in zip(srcs, tasks):
        if err := task.exception():
            _LG.error(_get_err_msg(src, err))
            continue
        packets.append(task.result())

    if strict and len(packets) != len(srcs):
        raise RuntimeError("Failed to demux some images.")

    if not packets:
        raise RuntimeError("Failed to demux all images.")

    return await _core.async_decode_packets_nvdec(
        packets,
        cuda_config=cuda_config,
        width=width,
        height=height,
        pix_fmt=pix_fmt,
        strict=strict,
        **decode_options,
    )


def _get_bytes(srcs: list[str | bytes]) -> list[bytes]:
    ret: list[bytes] = []
    for src in srcs:
        if isinstance(src, bytes):
            ret.append(src)
        elif isinstance(src, str):
            with open(src, "rb") as f:
                ret.append(f.read())
        else:
            raise TypeError(
                f"Source must be either `str` (path) or `byets` (data). Found: {type(src)}"
            )
    return ret


async def async_load_image_batch_nvjpeg(
    srcs: list[str | bytes],
    *,
    cuda_config: CUDAConfig,
    width: int | None,
    height: int | None,
    pix_fmt: str | None = "rgb",
    strict: bool = True,
    **kwargs,
):
    srcs_ = _get_bytes(srcs)
    return await run_async(
        _libspdl.decode_image_nvjpeg,
        srcs_,
        scale_width=width,
        scale_height=height,
        cuda_config=cuda_config,
        pix_fmt=pix_fmt,
        **kwargs,
    )


def load_image_batch_nvjpeg(
    srcs: list[str | bytes],
    *,
    cuda_config: CUDAConfig,
    width: int | None,
    height: int | None,
    pix_fmt: str | None = "rgb",
    strict: bool = True,
    **kwargs,
):
    srcs_ = _get_bytes(srcs)
    return _libspdl.decode_image_nvjpeg(
        srcs_,
        scale_width=width,
        scale_height=height,
        cuda_config=cuda_config,
        pix_fmt=pix_fmt,
        **kwargs,
    )


################################################################################
# Sample decode
################################################################################


def _decode_partial(packets, indices, **kwargs):
    """Decode packets but return early when requested frames are decoded."""
    num_frames = max(indices) + 1
    decoder = _core.streaming_decode_packets(packets, num_frames, **kwargs)
    return next(decoder)[indices]


async def async_sample_decode_video(
    packets: VideoPackets, indices: list[int], **kwargs
) -> list[ImagePackets]:
    if not indices:
        raise ValueError("Frame indices must be non-empty.")

    num_packets = len(packets)
    if any(not (0 <= i < num_packets) for i in indices):
        raise IndexError(f"Frame index must be [0, {num_packets}).")
    if sorted(indices) != indices:
        raise ValueError("Frame indices must be sorted in ascending order.")
    if len(set(indices)) != len(indices):
        raise ValueError("Frame indices must be unique.")

    tasks = []
    for split, idxes in _libspdl._extract_packets_at_indices(packets, indices):
        tasks.append(
            asyncio.create_task(run_async(_decode_partial, split, idxes, **kwargs))
        )

    await asyncio.wait(tasks)

    ret = []
    for task in tasks:
        try:
            ret.extend(task.result())
        except Exception as e:
            _LG.error(f"Failed to decode {task.get_name()}. Reason: {e}")
    return ret
