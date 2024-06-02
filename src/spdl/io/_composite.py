import asyncio
import builtins
import logging
from typing import Any

from spdl.lib import _libspdl

from . import _core, _preprocessing

__all__ = [
    "load_audio",
    "load_video",
    "load_image",
    "async_load_audio",
    "async_load_video",
    "async_load_image",
    "async_load_image_batch",
    "async_load_image_batch_nvdec",
    "async_sample_decode_video",
]

_LG = logging.getLogger(__name__)

Window = tuple[float, float]


async def async_load_audio(*args, **kwargs):
    return await _core._run_async(load_audio, *args, **kwargs)


def load_audio(
    src: str | bytes,
    timestamp: Window | list[Window] | None = None,
    *,
    demux_options: dict[str, Any] | None = None,
    decode_options: dict[str, Any] | None = None,
    convert_options: dict[str, Any] | None = None,
    transfer_options: dict[str, Any] | None = None,
):
    demux_options = demux_options or {}
    decode_options = decode_options or {}
    convert_options = convert_options or {}

    if timestamp is None or isinstance(timestamp, tuple):
        packets = _core.demux_audio(src, **demux_options)
        frames = _core.decode_packets(packets, **decode_options)
        buffer = _core.convert_frames(frames, **convert_options)
        if transfer_options is not None:
            buffer = _core.transfer_buffer(buffer, **transfer_options)
        return buffer

    ret = []
    for packets in _core.streaming_demux_audio(src, timestamp):
        frames = _core.decode_packets(packets)
        ret.append(_core.convert_frames(frames, **convert_options))
    return ret


async def async_load_video(*args, **kwargs):
    return await _core._run_async(load_video, *args, **kwargs)


def load_video(
    src: str | bytes,
    timestamp: Window | list[Window] | None = None,
    *,
    demux_options: dict[str, Any] | None = None,
    decode_options: dict[str, Any] | None = None,
    convert_options: dict[str, Any] | None = None,
    use_nvdec: bool = False,
):
    demux_options = demux_options or {}
    decode_options = decode_options or {}
    if use_nvdec and convert_options is not None:
        _LG.warn("`convert_options` is ignored when decoding video with NVDEC.")
    convert_options = convert_options or {}

    if use_nvdec:

        def _decode(packets):
            return _core.decode_packets_nvdec(packets, **decode_options)

    else:

        def _decode(packets):
            frames = _core.decode_packets(packets, **decode_options)
            return _core.convert_frames(frames, **convert_options)

    if timestamp is None or isinstance(timestamp, tuple):
        packets = _core.demux_video(src, **demux_options)
        return _decode(packets)

    ret = []
    for packets in _core.streaming_demux_video(src, timestamp):
        ret.append(_decode(packets))
    return ret


async def async_load_image(*args, **kwargs):
    return await _core._run_async(load_image, *args, **kwargs)


def load_image(
    src: str | bytes,
    *,
    demux_options: dict[str, Any] | None = None,
    decode_options: dict[str, Any] | None = None,
    convert_options: dict[str, Any] | None = None,
    use_nvjpeg: bool = False,
    cuda_device_index: int | None = None,
    _use_nvdec: bool = False,
):
    if use_nvjpeg:
        if cuda_device_index is None:
            raise ValueError("`use_nvjpeg=True` requires `cuda_device_index`.")
        if demux_options is not None:
            _LG.warn("`demux_options` is ignored when decoding image with NVJPEG.")
        if decode_options is not None:
            _LG.warn("`decode_options` is ignored when decoding image with NVJPEG.")
        if convert_options is not None:
            _LG.warn("`convert_options` is ignored when decoding image with NVJPEG.")
        return _core.decode_image_nvjpeg(src, cuda_device_index=cuda_device_index)

    demux_options = demux_options or {}
    decode_options = decode_options or {}
    convert_options = convert_options or {}

    packets = _core.demux_image(src, **demux_options)
    if _use_nvdec:
        return _core.decode_packets_nvdec(packets, **decode_options)
    frames = _core.decode_packets(packets, **decode_options)
    return _core.convert_frames(frames, **convert_options)


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
