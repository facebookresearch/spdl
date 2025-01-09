# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import asyncio
import contextlib
import functools
import logging
import warnings
from collections.abc import AsyncIterator, Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import overload, TYPE_CHECKING, TypeVar

# We import NumPy only when type-checkng.
# The functions of this module do not need NumPy itself to run.
# This is for experimenting with FT (no-GIL) Python.
# Once NumPy supports FT Python, we can import normally.
if TYPE_CHECKING:
    import numpy as np

    try:
        from numpy.typing import NDArray

        UintArray = NDArray[np.uint8]
    except ImportError:
        UintArray = np.ndarray
else:
    UintArray = object


from spdl.io import (
    AudioFrames,
    AudioPackets,
    CPUBuffer,
    CPUStorage,
    CUDABuffer,
    CUDAConfig,
    DecodeConfig,
    ImageFrames,
    ImagePackets,
    VideoFrames,
    VideoPackets,
)
from spdl.lib import _libspdl

from . import _preprocessing

__all__ = [
    # DEMUXING
    "Demuxer",
    "demux_audio",
    "demux_video",
    "demux_image",
    # DECODING
    "decode_packets",
    "decode_packets_nvdec",
    "streaming_decode_packets",
    "decode_image_nvjpeg",
    # FRAME CONVERSION
    "convert_array",
    "convert_frames",
    # DATA TRANSFER
    "transfer_buffer",
    "transfer_buffer_cpu",
    # ENCODING
    "encode_image",
]

_LG = logging.getLogger(__name__)
T = TypeVar("T")

_FILTER_DESC_DEFAULT = "__PLACEHOLDER__"


async def run_async(
    func: Callable[..., T],
    *args,
    _executor: ThreadPoolExecutor | None = None,
    **kwargs,
) -> T:
    """Run the given synchronous function asynchronously (in a thread).

    .. note::

       To achieve the true concurrency, the function must be thread-safe and must
       release the GIL.

    Args:
        func: The function to run.
        args: Positional arguments to the ``func``.
        _executor: Custom executor.
            If ``None`` the default executor of the current event loop is used.
        kwargs: Keyword arguments to the ``func``.
    """
    loop = asyncio.get_running_loop()
    _func = functools.partial(func, *args, **kwargs)
    return await loop.run_in_executor(_executor, _func)  # pyre-ignore: [6]


################################################################################
# Demuxing
################################################################################


class Demuxer:
    """Demuxer can demux audio, video and image from the soure.

    Args:
        src: Source identifier.
            If `str` type, it is interpreted as a source location,
            such as local file path or URL. If `bytes` type,
            then they are interpreted as in-memory data.
            If array type (objects implement buffer protocol,
            such as NumPy NDArray and PyTorch Tensor), then they must be
            1 dimentional uint8 array, which contains the raw bytes of the
            source.

        demux_config (DemuxConfig): Custom I/O config.
    """

    def __init__(self, src: str | Path | bytes | UintArray, **kwargs):
        if isinstance(src, Path):
            src = str(src)
        self._demuxer = _libspdl._demuxer(src, **kwargs)

    def demux_audio(
        self, window: tuple[float, float] | None = None, **kwargs
    ) -> AudioPackets:
        """Demux audio from the source.

        Args:
            timestamp:
                A time window. If omitted, the entire audio are demuxed.

        Returns:
            Demuxed audio packets.
        """
        _libspdl.log_api_usage("spdl.io.demux_audio")
        return self._demuxer.demux_audio(window=window, **kwargs)

    def demux_video(
        self, window: tuple[float, float] | None = None, **kwargs
    ) -> VideoPackets:
        """Demux video from the source.

        Args:
            timestamp:
                A time window. If omitted, the entire audio are demuxed.

        Returns:
            Demuxed video packets.
        """
        _libspdl.log_api_usage("spdl.io.demux_video")
        return self._demuxer.demux_video(window=window, **kwargs)

    def demux_image(self, **kwargs) -> ImagePackets:
        """Demux image from the source.

        Returns:
            Demuxed image packets.
        """
        _libspdl.log_api_usage("spdl.io.demux_image")
        return self._demuxer.demux_image(**kwargs)

    def has_audio(self) -> bool:
        """Returns true if the source has audio stream."""
        return self._demuxer.has_audio()

    def __enter__(self) -> "Demuxer":
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        self._demuxer._drop()

    async def __aenter__(self) -> "Demuxer":
        return self

    async def __aexit__(self, exc_type, exc_value, exc_traceback) -> None:
        await run_async(self._demuxer._drop)


def demux_audio(
    src: str | bytes | UintArray,
    *,
    timestamp: tuple[float, float] | None = None,
    **kwargs,
) -> AudioPackets:
    """Demux audio from the source.

    Args:
        src: See :py:class:`~spdl.io.Demuxer`.
        timestamp: See :py:meth:`spdl.io.Demuxer.demux_audio`.
        demux_config (DemuxConfig): See :py:class:`~spdl.io.Demuxer`.

    Returns:
        Demuxed audio packets.
    """
    with Demuxer(src, **kwargs) as demuxer:
        return demuxer.demux_audio(window=timestamp)


async def async_demux_audio(
    src: str | bytes | UintArray,
    *,
    timestamp: tuple[float, float] | None = None,
    **kwargs,
) -> AudioPackets:
    """Async version of :py:func:`~spdl.io.demux_audio`."""
    return await run_async(demux_audio, src, timestamp=timestamp, **kwargs)


def demux_video(
    src: str | bytes | UintArray,
    *,
    timestamp: tuple[float, float] | None = None,
    **kwargs,
) -> VideoPackets:
    """Demux video from the source.

    Args:
        src: See :py:class:`~spdl.io.Demuxer`.
        timestamp: See :py:meth:`spdl.io.Demuxer.demux_video`.
        demux_config (DemuxConfig): See :py:class:`~spdl.io.Demuxer`.


    Returns:
        Demuxed video packets.
    """
    with Demuxer(src, **kwargs) as demuxer:
        return demuxer.demux_video(window=timestamp)


async def async_demux_video(
    src: str | bytes | UintArray,
    *,
    timestamp: tuple[float, float] | None = None,
    **kwargs,
) -> VideoPackets:
    """Async version of :py:func:`~spdl.io.demux_video`."""
    return await run_async(demux_video, src, timestamp=timestamp, **kwargs)


def demux_image(src: str | bytes | UintArray, **kwargs) -> ImagePackets:
    """Demux image from the source.

    Args:
        src: See :py:class:`~spdl.io.Demuxer`.
        demux_config (DemuxConfig): See :py:class:`~spdl.io.Demuxer`.

    Returns:
        Demuxed image packets.
    """
    with Demuxer(src, **kwargs) as demuxer:
        return demuxer.demux_image()


async def async_demux_image(src: str | bytes, **kwargs) -> ImagePackets:
    """Async version of :py:func:`~spdl.io.demux_image`."""
    return await run_async(demux_image, src, **kwargs)


################################################################################
# Decoding
################################################################################


@overload
def decode_packets(
    packets: AudioPackets, filter_desc: str | None = _FILTER_DESC_DEFAULT, **kwargs
) -> AudioFrames: ...
@overload
def decode_packets(
    packets: VideoPackets, filter_desc: str | None = _FILTER_DESC_DEFAULT, **kwargs
) -> VideoFrames: ...
@overload
def decode_packets(
    packets: ImagePackets, filter_desc: str | None = _FILTER_DESC_DEFAULT, **kwargs
) -> ImageFrames: ...


def decode_packets(packets, filter_desc=_FILTER_DESC_DEFAULT, **kwargs):
    """Decode packets.

    Args:
        packets (AudioPackets, VideoPackets or ImagePackets): Packets object.

        filter_desc:
            *Optional:* Custom filter applied after decoding.
            To generate a description for common media processing operations,
            use :py:func:`~spdl.io.get_filter_desc` (if you have a packets object
            that has the timestamp set),
            :py:func:`~spdl.io.get_audio_filter_desc`, or
            :py:func:`~spdl.io.get_video_filter_desc`.
            If ``None`` is provided, then filtering is disabled.

            .. note::

               When decoding image/video packets, by default color space conversion
               is applied so that the output pixel format is rgb24.
               If you want to obtain frames without color conversion, disable filter by
               providing ``filter_desc=None``, or specify ``pix_fmt=None`` in
               the filter desc factory function.

        decode_config (DecodeConfig):
            *Optional:* Custom decode config.
            See :py:func:`~spdl.io.decode_config`,

    Returns:
        Frames object.
    """
    if filter_desc == _FILTER_DESC_DEFAULT:
        filter_desc = _preprocessing.get_filter_desc(packets)
    return _libspdl.decode_packets(packets, filter_desc=filter_desc, **kwargs)


@overload
async def async_decode_packets(packets: AudioPackets, **kwargs) -> AudioFrames: ...
@overload
async def async_decode_packets(packets: VideoPackets, **kwargs) -> VideoFrames: ...
@overload
async def async_decode_packets(packets: ImagePackets, **kwargs) -> ImageFrames: ...


async def async_decode_packets(packets, **kwargs):
    """Async version of :py:func:`~spdl.io.decode_packets`."""
    return await run_async(decode_packets, packets, **kwargs)


def decode_packets_nvdec(
    packets: VideoPackets | ImagePackets | list[ImagePackets],
    *,
    device_config: CUDAConfig | None = None,
    **kwargs,
) -> CUDABuffer:
    """**[Experimental]** Decode packets with NVDEC.

    .. warning::

       This API is exmperimental. The performance is not probed, and the specification
       might change.

    .. note::

       Unlike FFmpeg-based decoding, nvJPEG returns GPU buffer directly.

    .. note::

       For image, only baseline (non-progressive) JPEG formats are supported.

    Args:
        packets: Packets object.

        device_config: The device to use for decoding. See :py:func:`spdl.io.cuda_config`.

        crop_left, crop_top, crop_right, crop_bottom (int):
            *Optional:* Crop the given number of pixels from each side.

        width, height (int): *Optional:* Resize the frame. Resizing is done after
            cropping.

        pix_fmt (str or `None`): *Optional:* Change the format of the pixel.
            Supported value is ``"rgba"``. Default: ``"rgba"``.

    Returns:
        A CUDABuffer object.
    """
    if device_config is None:
        if "cuda_config" not in kwargs:
            raise ValueError("device_config must be provided.")
        if "cuda_config" in kwargs:
            warnings.warn(
                "`cuda_config` argument has been renamed to `device_config`.",
                stacklevel=2,
            )
            device_config = kwargs["cuda_config"]

    return _libspdl.decode_packets_nvdec(packets, device_config=device_config, **kwargs)


async def async_decode_packets_nvdec(
    packets: VideoPackets | ImagePackets | list[ImagePackets],
    *,
    device_config: CUDAConfig | None = None,
    **kwargs,
) -> CUDABuffer:
    """**[Experimental]** Async version of :py:func:`~spdl.io.decode_packets_nvdec`."""
    return await run_async(
        decode_packets_nvdec, packets, device_config=device_config, **kwargs
    )


def decode_image_nvjpeg(
    src: str | bytes, *, device_config: CUDAConfig | None = None, **kwargs
) -> CUDABuffer:
    """**[Experimental]** Decode image with nvJPEG.

    .. warning::

       This API is exmperimental. The performance is not probed, and the specification
       might change.

    .. note::

       Unlike FFmpeg-based decoding, nvJPEG returns GPU buffer directly.

    Args:
        src: File path to a JPEG image or data in bytes.
        device_config: The CUDA device to use for decoding.

        scale_width, scale_height (int): Resize image.
        pix_fmt (str): *Optional* Output pixel format.
            Supported values are ``"RGB"`` or ``"BGR"``.

    Returns:
        A CUDABuffer object. Shape is ``[C==3, H, W]``.
    """
    if device_config is None:
        if "cuda_config" not in kwargs:
            raise ValueError("device_config must be provided.")
        if "cuda_config" in kwargs:
            warnings.warn(
                "`cuda_config` argument has been renamed to `device_config`.",
                stacklevel=2,
            )
            device_config = kwargs["cuda_config"]

    if isinstance(src, bytes):
        data = src
    else:
        with open(src, "rb") as f:
            data = f.read()
    return _libspdl.decode_image_nvjpeg(data, device_config=device_config, **kwargs)


async def async_decode_image_nvjpeg(
    src: str | bytes, *, device_config: CUDAConfig | None = None, **kwargs
) -> CUDABuffer:
    """**[Experimental]** Async version of :py:func:`~spdl.io.decode_image_nvjpeg`."""
    return await run_async(
        decode_image_nvjpeg, src, device_config=device_config, **kwargs
    )


def streaming_decode_packets(
    packets: VideoPackets,
    num_frames: int,
    decode_config: DecodeConfig | None = None,
    filter_desc: str | None = _FILTER_DESC_DEFAULT,
) -> Iterator[VideoFrames]:
    """Decode the video packets chunk by chunk.

    Args:
        packets: Input packets.
        num_frames: Number of frames to decode at a time.
        decode_config: *Optional:* Custom decoding config.
            *Optional:* Custom decode config.
            See :py:func:`~spdl.io.decode_config`,
        filter_desc: *Optional:* Custom filter description.
            See :py:func:`~spdl.io.decode_packets` for the detail.

    Yields:
        VideoFrames object containing at most ``num_frames`` frames.
    """
    if filter_desc == _FILTER_DESC_DEFAULT:
        filter_desc = _preprocessing.get_filter_desc(packets)
    decoder = _libspdl._streaming_decoder(
        packets, decode_config=decode_config, filter_desc=filter_desc
    )
    while (frames := decoder.decode(num_frames)) is not None:
        yield frames


class _streaming_decoder_wrpper:
    def __init__(self, decoder):
        self.decoder = decoder

    async def decode(self, num_frames):
        return await run_async(self.decoder.decode, num_frames)


@contextlib.asynccontextmanager
async def _streaming_decoder(packets, **kwargs):
    if "filter_desc" not in kwargs:
        kwargs["filter_desc"] = _preprocessing.get_filter_desc(packets)
    decoder = await run_async(_libspdl._streaming_decoder, packets, **kwargs)
    wrapper = _streaming_decoder_wrpper(decoder)
    try:
        yield wrapper
    finally:
        await run_async(_libspdl._drop, wrapper.decoder)


async def async_streaming_decode_packets(
    packets: VideoPackets, num_frames: int, **kwargs
) -> AsyncIterator[VideoFrames]:
    """Async version of :py:func:`~spdl.io.streaming_decode_packets`."""
    async with _streaming_decoder(packets, **kwargs) as decoder:
        while (item := await decoder.decode(num_frames)) is not None:
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
    storage: CPUStorage | None = None,
    **kwargs,
) -> CPUBuffer:
    """Convert the decoded frames to buffer.

    Args:
        frames: Frames objects.
        storage (spdl.io.CPUStorage): Storage object. See :py:func:`spdl.io.cpu_storage`.

    Returns:
        A Buffer object.
            The shape of the buffer obejct is

            - ``AudioFrames`` -> ``[C, H]`` or ``[N, C]``.
            - ``VideoFrames`` -> ``[N, C, H, W]`` or ``[N, H, W, C]``.
            - ``ImageFrames`` -> ``[C, H, W]``.
            - ``list[AudioFrames]`` -> ``[B, C, H]`` or ``[B, N, C]``.
            - ``list[VideoFrames]`` -> ``[B, N, C, H, W]`` or ``[B, N, H, W, C]``.
            - ``list[ImageFrames]`` -> ``[B, C, H, W]``.

            where

            - ``B``: batch
            - ``C``: channel (color channel or audio channel)
            - ``N``: frames
            - ``W``: width
            - ``H``: height
    """
    if "pin_memory" in kwargs:
        warnings.warn(
            "`pin_memory` argument has been removed. Use `storage` instead.",
            stacklevel=2,
        )
        kwargs.pop("pin_memory")
    return _libspdl.convert_frames(frames, storage=storage, **kwargs)


async def async_convert_frames(
    frames: (
        AudioFrames
        | VideoFrames
        | ImageFrames
        | list[AudioFrames]
        | list[VideoFrames]
        | list[ImageFrames]
    ),
    storage: CPUStorage | None = None,
    **kwargs,
) -> CPUBuffer:
    """Async version of :py:func:`~spdl.io.convert_frames`."""
    if "pin_memory" in kwargs:
        warnings.warn(
            "`pin_memory` argument has been removed. Use `storage` instead.",
            stacklevel=2,
        )
        kwargs.pop("pin_memory")
    return await run_async(convert_frames, frames, storage=storage, **kwargs)


def convert_array(vals, storage: CPUStorage | None = None) -> CPUBuffer:
    """Convert the given array to buffer.

    This function is intended to be used when sending class labels (which is
    generated from list of integer) to GPU while overlapping the transfer with
    kenrel execution. See :py:func:`spdl.io.cpu_storage` for the detail.

    Args:
        vals: NumPy array with int64 dtype..
        storage (spdl.io.CPUStorage): Storage object. See :py:func:`spdl.io.cpu_storage`.

    Returns:
        A Buffer object.
    """
    return _libspdl.convert_array(vals, storage=storage)


async def async_convert_array(vals, storage: CPUStorage | None = None) -> CPUBuffer:
    """Async version of :py:func:`~spdl.io.convert_array`."""
    return await run_async(convert_array, vals, storage=storage)


################################################################################
# Device data transfer
################################################################################
def transfer_buffer(
    buffer: CPUBuffer, *, device_config: CUDAConfig | None = None, **kwargs
) -> CUDABuffer:
    """Move the given CPU buffer to CUDA device.

    Args:
        buffer: Source data.
        device_config: Target CUDA device configuration.

    Returns:
        Buffer data on the target GPU device.
    """
    if device_config is None:
        if "cuda_config" not in kwargs:
            raise ValueError("device_config must be provided.")
        if "cuda_config" in kwargs:
            warnings.warn(
                "`cuda_config` argument has been renamed to `device_config`.",
                stacklevel=2,
            )
            device_config = kwargs["cuda_config"]

    return _libspdl.transfer_buffer(buffer, device_config=device_config)


async def async_transfer_buffer(
    buffer: CPUBuffer, *, device_config: CUDAConfig | None = None, **kwargs
) -> CUDABuffer:
    """Async version of :py:func:`~spdl.io.transfer_buffer`."""
    return await run_async(
        transfer_buffer, buffer, device_config=device_config, **kwargs
    )


def transfer_buffer_cpu(buffer: CUDABuffer) -> CPUBuffer:
    """Move the given CUDA buffer to CPU.

    Args:
        buffer: Source data

    Returns:
        Buffer data on CPU.
    """
    return _libspdl.transfer_buffer_cpu(buffer)


async def async_transfer_buffer_cpu(buffer: CUDABuffer) -> CPUBuffer:
    """Async version of :py:func:`~spdl.io.transfer_buffer_cpu`."""
    return await run_async(transfer_buffer_cpu, buffer)


################################################################################
# Encoding
################################################################################

Array = TypeVar("Array")


def encode_image(path: str, data: Array, pix_fmt: str = "rgb24", **kwargs):
    """Save the given image array/tensor to file.

    Args:
        path: The path to which the data are written.

        data (NumPy NDArray, PyTorch Tensor):
            Image data in array format. The data  must be ``uint8`` type,
            either on CPU or CUDA device.

            The shape must be one of the following and must match the
            value of ``pix_fmt``.

            - ``(height, width, channel==3)`` when ``pix_fmt="rgb24"``
            - ``(height, width)`` when ``pix_fmt=gray8``
            - ``(channel==3, height, width)`` when ``pix_fmt="yuv444p"``

        pix_fmt: See above.

        encode_config (EncodeConfig): Customize the encoding.

    Example - Save image as PNG with resizing

        >>> import asyncio
        >>> import numpy as np
        >>> import spdl.io
        >>>
        >>> data = np.random.randint(255, size=(32, 16, 3), dtype=np.uint8)
        >>> img = spdl.io.encode_image(
        ...     "foo.png",
        ...     data,
        ...     pix_fmt="rgb24",
        ...     encode_config=spdl.io.encode_config(
        ...         width=198,
        ...         height=96,
        ...         scale_algo="neighbor",
        ...     ),
        ... )
        >>>

    Example - Save CUDA tensor as image

        >>> import torch
        >>>
        >>> data = torch.randint(255, size=(32, 16, 3), dtype=torch.uint8, device="cuda")
        >>>
        >>> def encode(data):
        ...     buffer = spdl.io.transfer_buffer_cpu(data)
        ...     return spdl.io.encode_image(
        ...         "foo.png",
        ...         buffer,
        ...         pix_fmt="rgb24",
        ...     )
        ...
        >>> encode(data)
        >>>
    """
    return _libspdl.encode_image(path, data, pix_fmt=pix_fmt, **kwargs)


async def async_encode_image(path: str, data: Array, pix_fmt: str = "rgb24", **kwargs):
    """Async version of :py:func:`~spdl.io.encode_image`."""
    return await run_async(encode_image, path, data, pix_fmt, **kwargs)
