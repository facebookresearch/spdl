import asyncio
import builtins
import logging
from collections.abc import AsyncIterator, Iterator
from typing import Any

from spdl.io import CPUBuffer, CUDABuffer, DecodeConfig, DemuxConfig, TransferConfig

from spdl.lib import _libspdl

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
    "async_sample_decode_video",
]

_LG = logging.getLogger(__name__)

Window = tuple[float, float]


################################################################################
# Load
################################################################################
def _load_packets(
    packets,
    demux_config: DemuxConfig | None = None,
    decode_config: DecodeConfig | None = None,
    filter_desc: str = "",
    transfer_config: TransferConfig | None = None,
):
    frames = _core.decode_packets(
        packets, decode_config=decode_config, filter_desc=filter_desc
    )
    buffer = _core.convert_frames(frames)
    if transfer_config is not None:
        buffer = _core.transfer_buffer(buffer, transfer_config=transfer_config)
    return buffer


def load_audio(
    src: str | bytes,
    timestamp: tuple[float, float] | None = None,
    *,
    demux_config: DemuxConfig | None = None,
    **kwargs,
) -> CPUBuffer | CUDABuffer:
    packets = _core.demux_audio(src, timestamp=timestamp, demux_config=demux_config)
    return _load_packets(packets, **kwargs)


async def async_load_audio(
    *args,
    **kwargs,
) -> CPUBuffer | CUDABuffer:
    return await _core._run_async(load_audio, *args, **kwargs)


def load_video(
    src: str | bytes,
    timestamp: tuple[float, float] | None = None,
    *,
    demux_config: DemuxConfig | None = None,
    **kwargs,
) -> CPUBuffer | CUDABuffer:
    packets = _core.demux_video(src, timestamp=timestamp, demux_config=demux_config)
    return _load_packets(packets, **kwargs)


async def async_load_video(
    *args,
    **kwargs,
) -> CPUBuffer | CUDABuffer:
    return await _core._run_async(load_video, *args, **kwargs)


def load_image(
    src: str | bytes,
    *,
    demux_config: DemuxConfig | None = None,
    **kwargs,
) -> CPUBuffer | CUDABuffer:
    packets = _core.demux_image(src, demux_config=demux_config)
    return _load_packets(packets, **kwargs)


async def async_load_image(
    *args,
    **kwargs,
) -> CPUBuffer | CUDABuffer:
    return await _core._run_async(load_image, *args, **kwargs)


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
) -> Iterator[CPUBuffer | CUDABuffer]:
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
) -> Iterator[CPUBuffer | CUDABuffer]:
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
    return await _core._run_async(_load_packets, *args)


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


async def async_load_image_batch(
    srcs: list[str | bytes],
    *,
    width: int | None,
    height: int | None,
    pix_fmt: str | None = "rgb24",
    demux_options: dict[str, Any] | None = None,
    decode_options: dict[str, Any] | None = None,
    convert_options: dict[str, Any] | None = None,
    transfer_options: dict[str, Any] | None = None,
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

    demux_options = demux_options or {}
    decode_options = decode_options or {}
    convert_options = convert_options or {}

    filter_desc = _preprocessing.get_video_filter_desc(
        scale_width=width,
        scale_height=height,
        pix_fmt=pix_fmt,
    )

    if filter_desc and "filter_desc" in decode_options:
        raise ValueError(
            "`width`, `height` or `pix_fmt` and `filter_desc` in `decode_options` "
            "cannot be present at the same time."
        )
    elif filter_desc:
        decode_options["filter_desc"] = filter_desc

    async def _decode(src):
        pkts = await _core.async_demux_image(src, **demux_options)
        return await _core.async_decode_packets(pkts, **decode_options)

    tasks = [asyncio.create_task(_decode(src)) for src in srcs]

    await asyncio.wait(tasks)
    frames = []
    for src, task in zip(srcs, tasks):
        try:
            frms = task.result()
        except Exception as err:
            _LG.error(_get_err_msg(src, err))
        else:
            frames.append(frms)

    if strict and len(frames) != len(srcs):
        raise RuntimeError("Failed to load some images.")

    if not frames:
        raise RuntimeError("Failed to load all the images.")

    buffer = await _core.async_convert_frames(frames, **convert_options)

    if transfer_options is not None:
        buffer = await _core.async_transfer_buffer(buffer, **transfer_options)

    return buffer


async def async_load_image_batch_nvdec(
    srcs: list[str | bytes],
    *,
    cuda_device_index: int,
    width: int | None,
    height: int | None,
    pix_fmt: str | None = "rgba",
    demux_options: dict[str, Any] | None = None,
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

    demux_options = demux_options or {}
    decode_options = decode_options or {}
    width = -1 if width is None else width
    height = -1 if height is None else height

    tasks = [
        asyncio.create_task(_core.async_demux_image(src, **demux_options))
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
        cuda_device_index=cuda_device_index,
        width=width,
        height=height,
        pix_fmt=pix_fmt,
        strict=strict,
        **decode_options,
    )


async def _decode_partial(packets, indices, **kwargs):
    """Decode packets but return early when requested frames are decoded."""
    num_frames = max(indices) + 1
    async for frames in _core.async_streaming_decode(packets, num_frames, **kwargs):
        return frames[indices]


async def async_sample_decode_video(packets, indices, **kwargs):
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
        tasks.append(asyncio.create_task(_decode_partial(split, idxes, **kwargs)))

    await asyncio.wait(tasks)

    ret = []
    for task in tasks:
        try:
            ret.extend(task.result())
        except Exception as e:
            _LG.error(f"Failed to decode {task.get_name()}. Reason: {e}")
    return ret
