# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import asyncio
import builtins
import logging
from typing import Any, overload

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

from . import _core, _preprocessing
from ._core import _FILTER_DESC_DEFAULT, run_async

__all__ = [
    "load_audio",
    "load_video",
    "load_image",
    "async_load_audio",
    "async_load_video",
    "async_load_image",
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
    filter_desc: str | None = _FILTER_DESC_DEFAULT,
    cuda_config: CUDAConfig | None = None,
):
    frames = _core.decode_packets(
        packets, decode_config=decode_config, filter_desc=filter_desc
    )
    buffer = _core.convert_frames(frames)
    if cuda_config is not None:
        buffer = _core.transfer_buffer(buffer, cuda_config=cuda_config)
    return buffer


@overload
def load_audio(
    src: str | bytes,
    timestamp: tuple[float, float] | None = None,
    *,
    demux_config: DemuxConfig | None = None,
    decode_config: DecodeConfig | None = None,
    filter_desc: str | None = _FILTER_DESC_DEFAULT,
    cuda_config: None = None,
) -> CPUBuffer: ...
@overload
def load_audio(
    src: str | bytes,
    timestamp: tuple[float, float] | None = None,
    *,
    demux_config: DemuxConfig | None = None,
    decode_config: DecodeConfig | None = None,
    filter_desc: str | None = _FILTER_DESC_DEFAULT,
    cuda_config: CUDAConfig,
) -> CUDABuffer: ...


def load_audio(
    src,
    timestamp=None,
    *,
    demux_config=None,
    decode_config=None,
    filter_desc=_FILTER_DESC_DEFAULT,
    cuda_config=None,
):
    """Load audio from source into buffer.

    This function combines :py:func:`~spdl.io.demux_audio`,
    :py:func:`~spdl.io.decode_packets`, :py:func:`~spdl.io.convert_frames`,
    and optionally, :py:func:`~spdl.io.transfer_buffer`, to produce
    buffer object from source in one step.

    Args:
        src, timestamp, demux_config: See :py:func:`~spdl.io.demux_audio`.
        decode_config, filter_desc: See :py:func:`~spdl.io.decode_packets`.
        cuda_config: See :py:func:`~spdl.io.transfer_buffer`.
            Providing this argument will move the buffer to CUDA device.

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


@overload
async def async_load_audio(
    src: str | bytes,
    timestamp: tuple[float, float] | None = None,
    *,
    demux_config: DemuxConfig | None = None,
    decode_config: DecodeConfig | None = None,
    filter_desc: str | None = _FILTER_DESC_DEFAULT,
    cuda_config: None = None,
) -> CPUBuffer: ...
@overload
async def async_load_audio(
    src: str | bytes,
    timestamp: tuple[float, float] | None = None,
    *,
    demux_config: DemuxConfig | None = None,
    decode_config: DecodeConfig | None = None,
    filter_desc: str | None = _FILTER_DESC_DEFAULT,
    cuda_config: CUDAConfig,
) -> CUDABuffer: ...


async def async_load_audio(
    src,
    timestamp=None,
    *,
    demux_config=None,
    decode_config=None,
    filter_desc=_FILTER_DESC_DEFAULT,
    cuda_config=None,
):
    """Async version of :py:func:`~spdl.io.load_audio`."""
    return await run_async(
        load_audio,
        src,
        timestamp,
        demux_config=demux_config,
        decode_config=decode_config,
        filter_desc=filter_desc,
        cuda_config=cuda_config,
    )


@overload
def load_video(
    src: str | bytes,
    timestamp: tuple[float, float] | None = None,
    *,
    demux_config: DemuxConfig | None = None,
    decode_config: DecodeConfig | None = None,
    filter_desc: str | None = _FILTER_DESC_DEFAULT,
    cuda_config: None = None,
) -> CPUBuffer: ...
@overload
def load_video(
    src: str | bytes,
    timestamp: tuple[float, float] | None = None,
    *,
    demux_config: DemuxConfig | None = None,
    decode_config: DecodeConfig | None = None,
    filter_desc: str | None = _FILTER_DESC_DEFAULT,
    cuda_config: CUDAConfig,
) -> CUDABuffer: ...


def load_video(
    src: str | bytes,
    timestamp=None,
    *,
    demux_config=None,
    decode_config=None,
    filter_desc=_FILTER_DESC_DEFAULT,
    cuda_config=None,
):
    """Load video from source into buffer.

    This function combines :py:func:`~spdl.io.demux_video`,
    :py:func:`~spdl.io.decode_packets`, :py:func:`~spdl.io.convert_frames`,
    and optionally, :py:func:`~spdl.io.transfer_buffer`, to produce
    buffer object from source in one step.

    Args:
        src, timestamp, demux_config: See :py:func:`~spdl.io.demux_video`.
        decode_config, filter_desc: See :py:func:`~spdl.io.decode_packets`.
        cuda_config: See :py:func:`~spdl.io.transfer_buffer`.
            Providing this argument will move the buffer to CUDA device.

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


@overload
async def async_load_video(
    src: str | bytes,
    timestamp: tuple[float, float] | None = None,
    *,
    demux_config: DemuxConfig | None = None,
    decode_config: DecodeConfig | None = None,
    filter_desc: str | None = _FILTER_DESC_DEFAULT,
    cuda_config: None = None,
) -> CPUBuffer: ...


@overload
async def async_load_video(
    src: str | bytes,
    timestamp: tuple[float, float] | None = None,
    *,
    demux_config: DemuxConfig | None = None,
    decode_config: DecodeConfig | None = None,
    filter_desc: str | None = _FILTER_DESC_DEFAULT,
    cuda_config: CUDAConfig,
) -> CUDABuffer: ...


async def async_load_video(
    src,
    timestamp=None,
    *,
    demux_config=None,
    decode_config=None,
    filter_desc=_FILTER_DESC_DEFAULT,
    cuda_config=None,
):
    """Async version of :py:func:`~spdl.io.load_video`."""
    return await run_async(
        load_video,
        src,
        timestamp,
        demux_config=demux_config,
        decode_config=decode_config,
        filter_desc=filter_desc,
        cuda_config=cuda_config,
    )


@overload
def load_image(
    src: str | bytes,
    *,
    demux_config: DemuxConfig | None = None,
    decode_config: DecodeConfig | None = None,
    filter_desc: str | None = _FILTER_DESC_DEFAULT,
    cuda_config: None = None,
) -> CPUBuffer: ...
@overload
def load_image(
    src: str | bytes,
    *,
    demux_config: DemuxConfig | None = None,
    decode_config: DecodeConfig | None = None,
    filter_desc: str | None = _FILTER_DESC_DEFAULT,
    cuda_config: CUDABuffer,
) -> CUDABuffer: ...


def load_image(
    src,
    *,
    demux_config=None,
    decode_config=None,
    filter_desc=_FILTER_DESC_DEFAULT,
    cuda_config=None,
):
    """Load image from source into buffer.

    This function combines :py:func:`~spdl.io.demux_image`,
    :py:func:`~spdl.io.decode_packets`, :py:func:`~spdl.io.convert_frames`,
    and optionally, :py:func:`~spdl.io.transfer_buffer`, to produce
    buffer object from source in one step.

    Args:
        src, demux_config: See :py:func:`~spdl.io.demux_video`.
        decode_config, filter_desc: See :py:func:`~spdl.io.decode_packets`.
        cuda_config: See :py:func:`~spdl.io.transfer_buffer`.
            Providing this argument will move the buffer to CUDA device.

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


@overload
async def async_load_image(
    src: str | bytes,
    *,
    demux_config: DemuxConfig | None = None,
    decode_config: DecodeConfig | None = None,
    filter_desc: str | None = _FILTER_DESC_DEFAULT,
    cuda_config: None = None,
) -> CPUBuffer: ...
@overload
async def async_load_image(
    src: str | bytes,
    *,
    demux_config: DemuxConfig | None = None,
    decode_config: DecodeConfig | None = None,
    filter_desc: str | None = _FILTER_DESC_DEFAULT,
    cuda_config: CUDABuffer,
) -> CUDABuffer: ...


async def async_load_image(
    src,
    *,
    demux_config=None,
    decode_config=None,
    filter_desc=_FILTER_DESC_DEFAULT,
    cuda_config=None,
):
    """Async version of :py:func:`~spdl.io.load_image`."""
    return await run_async(
        load_image,
        src,
        demux_config=demux_config,
        decode_config=decode_config,
        filter_desc=filter_desc,
        cuda_config=cuda_config,
    )


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


@overload
async def async_load_image_batch(
    srcs: list[str | bytes],
    *,
    width: int | None,
    height: int | None,
    pix_fmt: str | None = "rgb24",
    demux_config: DemuxConfig | None = None,
    decode_config: DecodeConfig | None = None,
    filter_desc: str | None = _FILTER_DESC_DEFAULT,
    cuda_config: None = None,
    pin_memory: bool = False,
    strict: bool = True,
) -> CPUBuffer: ...


@overload
async def async_load_image_batch(
    srcs: list[str | bytes],
    *,
    width: int | None,
    height: int | None,
    pix_fmt: str | None = "rgb24",
    demux_config: DemuxConfig | None = None,
    decode_config: DecodeConfig | None = None,
    filter_desc: str | None = _FILTER_DESC_DEFAULT,
    cuda_config: CUDAConfig,
    pin_memory: bool = False,
    strict: bool = True,
) -> CUDABuffer: ...


async def async_load_image_batch(
    srcs: list[str | bytes],
    *,
    width,
    height,
    pix_fmt="rgb24",
    demux_config=None,
    decode_config=None,
    filter_desc=_FILTER_DESC_DEFAULT,
    cuda_config=None,
    pin_memory=False,
    strict=True,
):
    """Batch load images.

    This function combines :py:func:`~spdl.io.demux_image`,
    :py:func:`~spdl.io.decode_packets`, :py:func:`~spdl.io.convert_frames`,
    and optionally, :py:func:`~spdl.io.transfer_buffer`, to produce
    buffer object from source in one step.

    It concurrently demuxes and decodes the input images, using
    the :py:class:`~concurrent.futures.ThreadPoolExecutor` attached to
    the running async event loop, fetched by :py:func:`~asyncio.get_running_loop`.

    .. mermaid::

       gantt
           title Illustration of asynchronous batch image decoding timeline
           dateFormat X
           axisFormat %s
           section Thread 1
               Demux image 1 :demux1, 0, 3
               Decode/resize image 1 :after demux1, 20
           section Thread 2
               Demux image 2 :demux2, 1, 5
               Decode/resize image 2 :after demux2, 23
           section Thread 3
               Demux image 3 :demux3, 2, 5
               Decode/resize image 3 :after demux3, 24
           section Thread 4
               Demux image 4 :demux4, 3, 8
               Decode/resize image 4 :decode4, after demux4, 25
           section Thread 5
               Batch conversion :batch, after decode4, 30
               Device Transfer :after batch, 33

    Args:
        srcs: List of source identifiers.

        width: *Optional:* Resize the frame.

        height: *Optional:* Resize the frame.

        pix_fmt:
            *Optional:* The output pixel format.

        demux_config:
            *Optional:* Demux configuration passed to
            :py:func:`~spdl.io.async_demux_image`.

        decode_config:
            *Optional:* Decode configuration passed to
            :py:func:`~spdl.io.async_decode_packets`.

        filter_desc:
            *Optional:* Filter description passed to
            :py:func:`~spdl.io.async_decode_packets`.

        cuda_config:
            *Optional:* The CUDA device passed to
            :py:func:`~spdl.io.async_transfer_buffer`.
            Providing this argument will move the resulting buffer to
            the CUDA device.

        strict:
            *Optional:* If True, raise an error if any of the images failed to load.

    Returns:
        A buffer object.

    .. admonition: Example

        >>> srcs = [
        ...     "sample1.jpg",
        ...     "sample2.png",
        ... ]
        >>> coro = async_load_image_batch(
        ...     srcs,
        ...     scale_width=124,
        ...     scale_height=96,
        ...     pix_fmt="rgb24",
        ... )
        >>> buffer = asyncio.run(coro)
        >>> array = spdl.io.to_numpy(buffer)
        >>> # An array with shape HWC==[2, 96, 124, 3]
        >>>
    """
    if not srcs:
        raise ValueError("`srcs` must not be empty.")

    if filter_desc == _FILTER_DESC_DEFAULT:
        filter_desc = _preprocessing.get_video_filter_desc(
            scale_width=width,
            scale_height=height,
            pix_fmt=pix_fmt,
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
) -> CUDABuffer:
    """**[Experimental]** Batch load images with NVDEC.

    This function combines :py:func:`~spdl.io.demux_image` and
    :py:func:`~spdl.io.decode_packets_nvdec` to produce
    buffer object from source in one step.

    It concurrently demuxes the input images, using
    the :py:class:`~concurrent.futures.ThreadPoolExecutor` attached to
    the running async event loop, fetched by :py:func:`~asyncio.get_running_loop`.

    Args:
        srcs: List of source identifiers.

        cuda_device_index: The CUDA device to use for decoding images.

        width: *Optional:* Resize the frame.

        height: *Optional:* Resize the frame.

        pix_fmt:
            *Optional:* Change the format of the pixel.

        demux_config:
            *Optional:* Demux options passed to
            :py:func:`~spdl.io.async_demux_media`.

        decode_options:
            *Optional:* Other decode options passed to
            :py:func:`~spdl.io.async_decode_packets_nvdec`.

        strict:
            *Optional:* If True, raise an error if any of the images failed to load.

    Returns:
        A buffer object.

    .. admonition:: Example

        >>> srcs = [
        ...     "sample1.jpg",
        ...     "sample2.png",
        ... ]
        >>> coro = async_load_image_batch_nvdec(
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
    """**[Experimental]** Async version of :py:func:`~spdl.io.load_image_batch_nvjpeg`.

    Unlike other async batch functions, this function does not employ intra-operation
    parallelism. (Decoding is done sequentially.)
    """
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
) -> CUDABuffer:
    """**[Experimental]** Batch load images with nvJPEG.

    This function decodes images using nvJPEG and resize them with NPP.
    The input images are processed in sequential manner.

    Args:
        srcs: Input images.
        cuda_config: The CUDA device to use for decoding images.
        width: *Optional:* Resize the frame.

        height: *Optional:* Resize the frame.

        pix_fmt:
            *Optional:* Change the format of the pixel.

        strict:
            *Optional:* If True, raise an error if any of the images failed to load.

    Returns:
        A buffer object.
    """
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


def _decode_partial(packets, indices, decode_config, filter_desc):
    """Decode packets but return early when requested frames are decoded."""
    num_frames = max(indices) + 1
    decoder = _core.streaming_decode_packets(
        packets, num_frames, decode_config, filter_desc
    )
    return next(decoder)[indices]


async def async_sample_decode_video(
    packets: VideoPackets,
    indices: list[int],
    decode_config: DecodeConfig | None = None,
    filter_desc: str | None = _FILTER_DESC_DEFAULT,
    strict: bool = True,
) -> list[ImagePackets]:
    """Decode specified frames from the packets.

    This function decodes the input video packets and returns the frames
    specified by ``indices``. Internally, it splits the packets into
    smaller set of packets and decode the minimum number of frames to
    retrieve the specified frames.

    .. mermaid::

       block-beta
         columns 15
           A["Input Packets"]:15

           space:15

           block:B1:3
             columns 3
             P1["1"] P2["2"] P3["3"]
           end
           block:B2:3
             columns 3
             P4["4"] P5["5"] P6["6"]
           end
           block:B3:3
             columns 3
             P7["7"] P8["8"] P9["9"]
           end
           block:B4:3
             columns 3
             P10["10"] P11["11"] P12["12"]
           end
           block:B5:3
             columns 3
             P13["13"] P14["14"] P15["15"]
           end

           space:15

           block:d1:3
             columns 3
             F1["Frame 1"] space:2
           end
           space:3
           block:d2:3
             columns 3
             F7["Frame 7"] F8["Frame 8"] space
           end
           space:3
           block:d3:3
             columns 3
             F13["Frame 13"] F14["Frame 14"] F15["Frame 15"]
           end

           space:15

           space:6
           block:out:3
             columns 3
             O1["Frame 1"] O8["Frame 8"] O15["Frame 15"]
           end
           space:6
           A -- "Split 1" --> B1
           A -- "Split 2" --> B2
           A -- "Split 3" --> B3
           A -- "Split 4" --> B4
           A -- "Split 5" --> B5

           B1 -- "Decode 1" --> d1
           B3 -- "Decode 3" --> d2
           B5 -- "Decode 5" --> d3

           F1 --> O1
           F8 --> O8
           F15 --> O15

    The packet splits are decoded concurrently.
    The following figure illustrates the timeline of the process.

    .. mermaid::

       gantt
           title Illustration of asynchronous sample decode timeline
           dateFormat X
           axisFormat %s
           section Thread 1
               Split Input Packets:split, 0, 3
               Decode Split 1 :decode1, after split, 7
               Gather and return: gather, after decode2, 14
           section Thread 2
               Decode Split 3 :decode2, after split, 10
           section Thread 3
               Decode Split 5 :decode2, after split, 13


    Args:
        packets: The input video packets.
        indices: The list of frame indices.
        decode_config:
            *Optional:* Decode config.
            See :py:func:`~spdl.io.decode_config`.
        filter_desc: *Optional:* Filter description.
            See :py:func:`~spdl.io.decode_packets` for detail.

        strict: *Optional:* If True, raise an error
            if any of the frames failed to decode.

    Returns:
        Decoded frames.
    """
    if not indices:
        raise ValueError("Frame indices must be non-empty.")

    num_packets = len(packets)
    if any(not (0 <= i < num_packets) for i in indices):
        raise IndexError(f"Frame index must be [0, {num_packets}).")
    if sorted(indices) != indices:
        raise ValueError("Frame indices must be sorted in ascending order.")
    if len(set(indices)) != len(indices):
        raise ValueError("Frame indices must be unique.")

    if filter_desc == _FILTER_DESC_DEFAULT:
        filter_desc = _preprocessing.get_video_filter_desc()

    tasks = []
    for split, idxes in _libspdl._extract_packets_at_indices(packets, indices):
        tasks.append(
            asyncio.create_task(
                run_async(_decode_partial, split, idxes, decode_config, filter_desc)
            )
        )

    await asyncio.wait(tasks)

    ret = []
    for task in tasks:
        try:
            ret.extend(task.result())
        except Exception as e:
            _LG.error(f"Failed to decode {task.get_name()}. Reason: {e}")
    if strict and len(ret) != len(indices):
        raise RuntimeError("Failed to decode some frames.")
    return ret
