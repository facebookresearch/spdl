# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import builtins
import logging
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import overload, TYPE_CHECKING

from spdl.io import (
    CPUBuffer,
    CPUStorage,
    CUDABuffer,
    CUDAConfig,
    DecodeConfig,
    DemuxConfig,
    ImagePackets,
    VideoPackets,
)

from . import _config, _core, _preprocessing
from ._core import _FILTER_DESC_DEFAULT, SourceType
from .lib import _libspdl

__all__ = [
    "load_audio",
    "load_video",
    "load_image",
    "save_image",
    "load_image_batch",
    "load_image_batch_nvjpeg",
    "streaming_load_video_nvdec",
    "sample_decode_video",
]

_LG = logging.getLogger(__name__)

Window = tuple[float, float]


if TYPE_CHECKING:
    import numpy as np

    try:
        from numpy.typing import NDArray as Array

    except ImportError:
        Array = np.ndarray
else:
    Array = object


################################################################################
# Load
################################################################################
def _load_packets(
    packets,
    decode_config: DecodeConfig | None = None,
    filter_desc: str | None = _FILTER_DESC_DEFAULT,
    device_config: CUDAConfig | None = None,
    **kwargs,
):
    if device_config is None and "cuda_config" in kwargs:
        warnings.warn(
            "`cuda_config` argument has been renamed to `device_config`.", stacklevel=3
        )
        device_config = kwargs["cuda_config"]

    frames = _core.decode_packets(
        packets, decode_config=decode_config, filter_desc=filter_desc
    )
    buffer = _core.convert_frames(frames)
    if device_config is not None:
        buffer = _core.transfer_buffer(buffer, device_config=device_config)
    return buffer


@overload
def load_audio(
    src: str | bytes,
    timestamp: tuple[float, float] | None = None,
    *,
    demux_config: DemuxConfig | None = None,
    decode_config: DecodeConfig | None = None,
    filter_desc: str | None = _FILTER_DESC_DEFAULT,
    device_config: None = None,
    **kwargs,
) -> CPUBuffer: ...
@overload
def load_audio(
    src: str | bytes,
    timestamp: tuple[float, float] | None = None,
    *,
    demux_config: DemuxConfig | None = None,
    decode_config: DecodeConfig | None = None,
    filter_desc: str | None = _FILTER_DESC_DEFAULT,
    device_config: CUDAConfig,
    **kwargs,
) -> CUDABuffer: ...


def load_audio(
    src,
    timestamp=None,
    *,
    demux_config=None,
    decode_config=None,
    filter_desc=_FILTER_DESC_DEFAULT,
    device_config=None,
    **kwargs,
):
    """Load audio from source into buffer.

    This function combines :py:func:`~spdl.io.demux_audio`,
    :py:func:`~spdl.io.decode_packets`, :py:func:`~spdl.io.convert_frames`,
    and optionally, :py:func:`~spdl.io.transfer_buffer`, to produce
    buffer object from source in one step.

    Args:
        src, timestamp, demux_config: See :py:func:`~spdl.io.demux_audio`.
        decode_config, filter_desc: See :py:func:`~spdl.io.decode_packets`.
        device_config: See :py:func:`~spdl.io.transfer_buffer`.
            Providing this argument will move the buffer to CUDA device.

    Returns:
        Buffer object.
    """
    _core.log_api_usage_once("spdl.io.load_audio")

    packets = _core.demux_audio(src, timestamp=timestamp, demux_config=demux_config)
    return _load_packets(
        packets,
        decode_config=decode_config,
        filter_desc=filter_desc,
        device_config=device_config,
        **kwargs,
    )


@overload
def load_video(
    src: str | bytes,
    timestamp: tuple[float, float] | None = None,
    *,
    demux_config: DemuxConfig | None = None,
    decode_config: DecodeConfig | None = None,
    filter_desc: str | None = _FILTER_DESC_DEFAULT,
    device_config: None = None,
    **kwargs,
) -> CPUBuffer: ...
@overload
def load_video(
    src: str | bytes,
    timestamp: tuple[float, float] | None = None,
    *,
    demux_config: DemuxConfig | None = None,
    decode_config: DecodeConfig | None = None,
    filter_desc: str | None = _FILTER_DESC_DEFAULT,
    device_config: CUDAConfig,
    **kwargs,
) -> CUDABuffer: ...


def load_video(
    src: str | bytes,
    timestamp=None,
    *,
    demux_config=None,
    decode_config=None,
    filter_desc=_FILTER_DESC_DEFAULT,
    device_config=None,
    **kwargs,
):
    """Load video from source into buffer.

    This function combines :py:func:`~spdl.io.demux_video`,
    :py:func:`~spdl.io.decode_packets`, :py:func:`~spdl.io.convert_frames`,
    and optionally, :py:func:`~spdl.io.transfer_buffer`, to produce
    buffer object from source in one step.

    Args:
        src, timestamp, demux_config: See :py:func:`~spdl.io.demux_video`.
        decode_config, filter_desc: See :py:func:`~spdl.io.decode_packets`.
        device_config: See :py:func:`~spdl.io.transfer_buffer`.
            Providing this argument will move the buffer to CUDA device.

    Returns:
        Buffer object.
    """
    _core.log_api_usage_once("spdl.io.load_video")

    packets = _core.demux_video(src, timestamp=timestamp, demux_config=demux_config)
    return _load_packets(
        packets,
        decode_config=decode_config,
        filter_desc=filter_desc,
        device_config=device_config,
        **kwargs,
    )


@overload
def load_image(
    src: str | bytes,
    *,
    demux_config: DemuxConfig | None = None,
    decode_config: DecodeConfig | None = None,
    filter_desc: str | None = _FILTER_DESC_DEFAULT,
    device_config: None = None,
    **kwargs,
) -> CPUBuffer: ...
@overload
def load_image(
    src: str | bytes,
    *,
    demux_config: DemuxConfig | None = None,
    decode_config: DecodeConfig | None = None,
    filter_desc: str | None = _FILTER_DESC_DEFAULT,
    device_config: CUDABuffer,
    **kwargs,
) -> CUDABuffer: ...


def load_image(
    src,
    *,
    demux_config=None,
    decode_config=None,
    filter_desc=_FILTER_DESC_DEFAULT,
    device_config=None,
    **kwargs,
):
    """Load image from source into buffer.

    This function combines :py:func:`~spdl.io.demux_image`,
    :py:func:`~spdl.io.decode_packets`, :py:func:`~spdl.io.convert_frames`,
    and optionally, :py:func:`~spdl.io.transfer_buffer`, to produce
    buffer object from source in one step.

    Args:
        src, demux_config: See :py:func:`~spdl.io.demux_video`.
        decode_config, filter_desc: See :py:func:`~spdl.io.decode_packets`.
        device_config: See :py:func:`~spdl.io.transfer_buffer`.
            Providing this argument will move the buffer to CUDA device.

    Returns:
        Buffer object.
    """
    _core.log_api_usage_once("spdl.io.load_image")

    packets = _core.demux_image(src, demux_config=demux_config)
    return _load_packets(
        packets,
        decode_config=decode_config,
        filter_desc=filter_desc,
        device_config=device_config,
        **kwargs,
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
def load_image_batch(
    srcs: list[str | bytes],
    *,
    width: int | None,
    height: int | None,
    pix_fmt: str | None = "rgb24",
    demux_config: DemuxConfig | None = None,
    decode_config: DecodeConfig | None = None,
    filter_desc: str | None = _FILTER_DESC_DEFAULT,
    device_config: None = None,
    storage: CPUStorage | None = None,
    strict: bool = True,
    **kwargs,
) -> CPUBuffer: ...


@overload
def load_image_batch(
    srcs: list[str | bytes],
    *,
    width: int | None,
    height: int | None,
    pix_fmt: str | None = "rgb24",
    demux_config: DemuxConfig | None = None,
    decode_config: DecodeConfig | None = None,
    filter_desc: str | None = _FILTER_DESC_DEFAULT,
    device_config: CUDAConfig,
    storage: CPUStorage | None = None,
    strict: bool = True,
    **kwargs,
) -> CUDABuffer: ...


def load_image_batch(
    srcs: list[str | bytes],
    *,
    width,
    height,
    pix_fmt="rgb24",
    demux_config=None,
    decode_config=None,
    filter_desc=_FILTER_DESC_DEFAULT,
    device_config=None,
    storage: CPUStorage | None = None,
    strict=True,
    **kwargs,
):
    """Batch load images.

    This function combines :py:func:`~spdl.io.demux_image`,
    :py:func:`~spdl.io.decode_packets`, :py:func:`~spdl.io.convert_frames`,
    and optionally, :py:func:`~spdl.io.transfer_buffer`, to produce
    buffer object from source in one step.

    Args:
        srcs: List of source identifiers.

        width: *Optional:* Resize the frame.

        height: *Optional:* Resize the frame.

        pix_fmt:
            *Optional:* The output pixel format.

        demux_config:
            *Optional:* Demux configuration passed to
            :py:func:`~spdl.io.demux_image`.

        decode_config:
            *Optional:* Decode configuration passed to
            :py:func:`~spdl.io.decode_packets`.

        filter_desc:
            *Optional:* Filter description passed to
            :py:func:`~spdl.io.decode_packets`.

        device_config:
            *Optional:* The CUDA device passed to
            :py:func:`~spdl.io.transfer_buffer`.
            Providing this argument will move the resulting buffer to
            the CUDA device.

        storage:
            *Optional:* The storage object passed to
            :py:func:`~spdl.io.convert_frames`.

        strict:
            *Optional:* If True, raise an error if any of the images failed to load.

    Returns:
        A buffer object.

    .. admonition: Example

        >>> srcs = [
        ...     "sample1.jpg",
        ...     "sample2.png",
        ... ]
        >>> buffer = load_image_batch(
        ...     srcs,
        ...     scale_width=124,
        ...     scale_height=96,
        ...     pix_fmt="rgb24",
        ... )
        >>> array = spdl.io.to_numpy(buffer)
        >>> # An array with shape HWC==[2, 96, 124, 3]
        >>>
    """
    _core.log_api_usage_once("spdl.io.load_image_batch")

    if not srcs:
        raise ValueError("`srcs` must not be empty.")

    if "pin_memory" in kwargs:
        warnings.warn(
            "`pin_memory` argument has been removed. Use `storage` instead.",
            stacklevel=2,
        )
        kwargs.pop("pin_memory")

    if device_config is None and "cuda_config" in kwargs:
        warnings.warn(
            "The `cuda_config` argument has ben renamed to `device_config`.",
            stacklevel=2,
        )
        device_config = kwargs["cuda_config"]

    if filter_desc == _FILTER_DESC_DEFAULT:
        filter_desc = _preprocessing.get_video_filter_desc(
            scale_width=width,
            scale_height=height,
            pix_fmt=pix_fmt,
        )

    frames = []
    for src in srcs:
        try:
            frame = _decode(src, demux_config, decode_config, filter_desc)
        except Exception as err:
            _LG.error(_get_err_msg(src, err))
        else:
            frames.append(frame)

    if strict and len(frames) != len(srcs):
        raise RuntimeError("Failed to load some images.")

    if not frames:
        raise RuntimeError("Failed to load all the images.")

    buffer = _core.convert_frames(frames, storage=storage)

    if device_config is not None:
        buffer = _core.transfer_buffer(buffer, device_config=device_config)

    return buffer


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
                f"Source must be either `str` (path) or `bytes` (data). Found: {type(src)}"
            )
    return ret


def load_image_batch_nvjpeg(
    srcs: list[str | bytes],
    *,
    device_config: CUDAConfig,
    width: int | None,
    height: int | None,
    pix_fmt: str | None = "rgb",
    **kwargs,
) -> CUDABuffer:
    """**[Experimental]** Batch load images with nvJPEG.

    This function decodes images using nvJPEG and resize them with NPP.
    The input images are processed in sequential manner.

    Args:
        srcs: Input images.
        device_config: The CUDA device to use for decoding images.
        width: *Optional:* Resize the frame.

        height: *Optional:* Resize the frame.

        pix_fmt:
            *Optional:* Change the format of the pixel.

        strict:
            *Optional:* If True, raise an error if any of the images failed to load.

    Returns:
        A buffer object.
    """
    _core.log_api_usage_once("spdl.io.load_image_batch_nvjpeg")

    srcs_ = _get_bytes(srcs)
    return _core.decode_image_nvjpeg(
        srcs_,
        scale_width=width,
        scale_height=height,
        device_config=device_config,
        pix_fmt=pix_fmt,
        **kwargs,
    )


def streaming_load_video_nvdec(
    src: SourceType,
    device_config: CUDAConfig,
    *,
    num_frames: int,
    post_processing_params: dict[str, int] | None = None,
) -> Iterator[list[CUDABuffer]]:
    """Load video from source chunk by chunk using NVDEC.

    .. seealso::

       - :py:mod:`streaming_nvdec_decoding`: Demonstrates how to
         decode a long video using NVDEC.

    Args:
        src: The source URI. Passed to :py:class:`Demuxer`.

        device_config: The CUDA device config.
            See :py:class:`NvDecDecoder` for details.

        num_frames: The maximum number of frames yielded at a time.

        post_processing_params: The post processing parameters
            passed to :py:class:`NvDecDecoder.init`.

    Yields:
        List of at most ``num_frames`` CUDA buffers.
        Each CUDA buffer contain a frame in NV12 format.
        The shape is ``(height + height // 2, width)``.
        The bottom one-third contains interleaved UV plane,
        and the top part contains the Y plane.

        Use :py:func:`nv12_to_rgb` or :py:func:`nv12_to_bgr`
        to convert to RGB frames.
    """
    _core.log_api_usage_once("spdl.io.streaming_load_video_nvdec")

    demuxer = _core.Demuxer(src)
    codec = demuxer.video_codec
    bsf = None
    match codec.name:
        case "h264":
            bsf = _core.BSF(codec, "h264_mp4toannexb")
        case "hevc":
            bsf = _core.BSF(codec, "hevc_mp4toannexb")
        case _:
            bsf = None

    decoder = _core.nvdec_decoder()
    decoder.init(device_config, codec, **(post_processing_params or {}))
    buffers = []
    for packets in demuxer.streaming_demux_video(num_frames):
        if bsf is not None:
            packets = bsf.filter(packets)

        buffers += decoder.decode(packets)

        if len(buffers) >= num_frames:
            tmp, buffers = buffers[:num_frames], buffers[num_frames:]
            yield tmp

    if bsf is not None:
        packets = bsf.flush()
        if len(packets):
            buffers += decoder.decode(packets)

    buffers += decoder.flush()

    while len(buffers) >= num_frames:
        tmp, buffers = buffers[:num_frames], buffers[num_frames:]
        yield tmp

    if buffers:
        yield buffers


################################################################################
# Sample decode
################################################################################


def _decode_partial(packets, indices, decode_config, filter_desc):
    """Decode packets but return early when requested frames are decoded."""
    num_frames = max(indices) + 1
    frames = _core.decode_packets(
        packets,
        decode_config=decode_config,
        filter_desc=filter_desc,
        num_frames=num_frames,
    )
    return frames[indices]


def sample_decode_video(
    packets: VideoPackets,
    indices: list[int],
    *,
    decode_config: DecodeConfig | None = None,
    filter_desc: str | None = _FILTER_DESC_DEFAULT,
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
    _core.log_api_usage_once("spdl.io.sample_decode_video")

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

    ret = []
    for split, idxes in _libspdl._extract_packets_at_indices(packets, indices):
        ret.extend(_decode_partial(split, idxes, decode_config, filter_desc))
    return ret


def save_image(path: str | Path, data: Array, pix_fmt: str = "rgb24", **kwargs):
    """Save the given image tensor.

    .. note::

       The current implementation only supports writing data without changing
       the pixel format. If you need to apply preprocessing including pixel
       format conversion or rescaling, then construct such pipeline using
       :py:class:`FilterGraph` and :py:class:`Muxer`.

    Args:
        path: The output path
        data: The data to write.
            The shape must be one of the following and must match the
            value of ``pix_fmt``.

            - ``"rgb24"``, ``"bgr24"``: ``(height, width, channel==3)``
            - ``"gray"``, ``"gray16be"``: ``(height, width)``
            - ``"yuv444p"``: ``(channel==3, height, width)``

            If writing ``"gray16be"``, the dtype must be 16-bit signed integer.
            Otherwise it must be 8-bit unsigned integer.

        pix_fmt: The pixel format used to write data.
            Supported values are, ``"rgb24"``, ``"bgr24"``, ``"gray"``,
            ``"gray16be"``, ``"gray16le"``, ``"gray"``, ``"yuv444p"``.

        **kwargs: Extra arguments passed to :py:func:`video_encode_config`.
    """
    match pix_fmt:
        case "rgb24" | "bgr24":
            if data.ndim != 3 or data.shape[2] != 3:
                raise ValueError(
                    f"The input shape must be [H, W, C==3]. Found: {data.shape}"
                )
            if data.dtype.itemsize != 1:
                raise ValueError(f"The input dtype must be 8bit. Found: {data.dtype}")

            height, width = data.shape[:2]

        case "gray":
            if data.ndim != 2:
                raise ValueError(f"The input shape must be [H, W]. Found: {data.shape}")
            if data.dtype.itemsize != 1:
                raise ValueError(f"The input dtype must be 8bit. Found: {data.dtype}")

            height, width = data.shape[:2]

        case "gray16" | "gray16be" | "gray16le":
            if data.ndim != 2:
                raise ValueError(f"The input shape must be [H, W]. Found: {data.shape}")
            if data.dtype.itemsize != 2:
                raise ValueError(f"The input dtype must be 16-bit. Found: {data.dtype}")

            height, width = data.shape[:2]

        case "yuv444p":
            if data.ndim != 3 or data.shape[0] != 3:
                raise ValueError(
                    f"The input shape must be [C==3, H, W]. Found: {data.shape}"
                )
            if data.dtype.itemsize != 1:
                raise ValueError(f"The input dtype must be 8bit. Found: {data.dtype}")

            height, width = data.shape[1:]

        case _:
            raise ValueError(f"Unsupported pix_fmt: {pix_fmt}")

    muxer = _core.Muxer(path)
    encoder = muxer.add_encode_stream(
        config=_config.video_encode_config(
            height=height,
            width=width,
            frame_rate=(1, 1),
            pix_fmt=pix_fmt,
            **kwargs,
        )
    )

    with muxer.open() as muxer:
        frames = _core.create_reference_video_frame(
            data[None, ...], pix_fmt=pix_fmt, frame_rate=(1, 1), pts=0
        )

        if (packets := encoder.encode(frames)) is not None:
            muxer.write(0, packets)
        if (packets := encoder.flush()) is not None:
            muxer.write(0, packets)
