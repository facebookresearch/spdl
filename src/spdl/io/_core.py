# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import logging
import threading
import warnings
from collections.abc import Iterator
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

    try:
        import torch

        Tensor = torch.Tensor
    except ImportError:
        Tensor = object
else:
    UintArray = object
    Tensor = object


from spdl.io import (
    AudioCodec,
    AudioFrames,
    AudioPackets,
    CPUBuffer,
    CPUStorage,
    CUDABuffer,
    CUDAConfig,
    DecodeConfig,
    ImageCodec,
    ImageFrames,
    ImagePackets,
    VideoCodec,
    VideoFrames,
    VideoPackets,
)
from spdl.io._internal.import_utils import lazy_import

from . import _preprocessing
from .lib import _libspdl, _libspdl_cuda

__all__ = [
    # DEMUXING
    "Demuxer",
    "demux_audio",
    "demux_video",
    "demux_image",
    "apply_bsf",
    # DECODING
    "decode_packets",
    "decode_packets_nvdec",
    "streaming_decode_packets",
    "decode_image_nvjpeg",
    "NvDecDecoder",
    # FRAME CONVERSION
    "convert_array",
    "convert_frames",
    # DATA TRANSFER
    "transfer_buffer",
    "transfer_buffer_cpu",
    # COLOR CONVERSION
    "nv12_to_rgb",
    "nv12_to_bgr",
    # ENCODING
    "encode_image",
]

torch = lazy_import("torch")  # pyre-ignore: [31]

_LG = logging.getLogger(__name__)
T = TypeVar("T")

_FILTER_DESC_DEFAULT = "__PLACEHOLDER__"


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

    def __init__(self, src: str | Path | bytes | UintArray | Tensor, **kwargs):
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
        # NOTE:
        # DO NOT REFACTOR (EXTRACT) FUNCTION CALL. IT WILL BREAK
        # THE META INTERNAL API LOGGING.
        # It's no-op for OSS.
        try:
            torch._C._log_api_usage_once("spdl.io.demux_audio")
        except Exception:
            pass

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
        # NOTE:
        # DO NOT REFACTOR (EXTRACT) FUNCTION CALL. IT WILL BREAK
        # THE META INTERNAL API LOGGING.
        # It's no-op for OSS.
        try:
            torch._C._log_api_usage_once("spdl.io.demux_video")
        except Exception:
            pass

        return self._demuxer.demux_video(window=window, **kwargs)

    def demux_image(self, **kwargs) -> ImagePackets:
        """Demux image from the source.

        Returns:
            Demuxed image packets.
        """
        # NOTE:
        # DO NOT REFACTOR (EXTRACT) FUNCTION CALL. IT WILL BREAK
        # THE META INTERNAL API LOGGING.
        # It's no-op for OSS.
        try:
            torch._C._log_api_usage_once("spdl.io.demux_image")
        except Exception:
            pass

        return self._demuxer.demux_image(**kwargs)

    def streaming_demux_video(
        self,
        num_packets: int,
        bsf: str | None = None,
    ) -> Iterator[VideoPackets]:
        """Demux video frames in streaming fashion.

        Args:
            num_packets: The number of packets to return at a time.
        """
        ite = self._demuxer.streaming_demux_video(num_packets, bsf)
        return _StreamingVideoDemuxer(ite, self)

    def has_audio(self) -> bool:
        """Returns true if the source has audio stream."""
        return self._demuxer.has_audio()

    @property
    def audio_codec(self) -> AudioCodec:
        """The codec metadata of the default audio stream."""
        return self._demuxer.audio_codec

    @property
    def video_codec(self) -> VideoCodec:
        """The codec metadata of the default video stream."""
        return self._demuxer.video_codec

    @property
    def image_codec(self) -> ImageCodec:
        """The codec metadata  of the default image stream."""
        return self._demuxer.image_codec

    def __enter__(self) -> "Demuxer":
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        self._demuxer._drop()


class _StreamingVideoDemuxer:
    def __init__(self, ite, demuxer: Demuxer) -> None:
        self._demuxer = demuxer  # For keeping the reference.
        self._ite = ite

    def __iter__(self):
        return self

    def __next__(self):
        if self._ite.done():
            raise StopIteration
        return self._ite.next()


def demux_audio(
    src: str | bytes | UintArray | Tensor,
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


def demux_video(
    src: str | bytes | UintArray | Tensor,
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


def demux_image(src: str | bytes | UintArray | Tensor, **kwargs) -> ImagePackets:
    """Demux image from the source.

    Args:
        src: See :py:class:`~spdl.io.Demuxer`.
        demux_config (DemuxConfig): See :py:class:`~spdl.io.Demuxer`.

    Returns:
        Demuxed image packets.
    """
    with Demuxer(src, **kwargs) as demuxer:
        return demuxer.demux_image()


@overload
def apply_bsf(packets: AudioPackets, bsf: str) -> AudioPackets: ...
@overload
def apply_bsf(packets: VideoPackets, bsf: str) -> VideoPackets: ...
@overload
def apply_bsf(packets: ImagePackets, bsf: str) -> ImagePackets: ...


def apply_bsf(packets, bsf):
    """Apply bit stream filter to packets.

    Args:
        packets: Packets (audio/video/image) object
        bsf: A bitstream filter description.

    .. seealso::

       - https://ffmpeg.org/ffmpeg-bitstream-filters.html The list of available
         bit stream filters.
    """
    return _libspdl.apply_bsf(packets, bsf)


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


def decode_packets_nvdec(
    packets: VideoPackets,
    *,
    device_config: CUDAConfig | None = None,
    **kwargs,
) -> CUDABuffer:
    """**[Experimental]** Decode packets with NVDEC.

    .. warning::

       This API is exmperimental. The performance is not probed, and the specification
       might change.

    .. note::

       Unlike FFmpeg-based decoding, NVDEC returns GPU buffer directly.

    Args:
        packets: Packets object.

        device_config: The device to use for decoding. See :py:func:`spdl.io.cuda_config`.

        crop_left, crop_top, crop_right, crop_bottom (int):
            *Optional:* Crop the given number of pixels from each side.

        width, height (int): *Optional:* Resize the frame. Resizing is done after
            cropping.

        pix_fmt (str or `None`): *Optional:* Change the format of the pixel.
            Supported value is ``"rgb"`` and ``"bgr"``. Default: ``"rgb"``.

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

    # Note
    # FFmpeg's implementation applies BSF to all H264/HEVC formats,
    #
    # https://github.com/FFmpeg/FFmpeg/blob/5e2b0862eb1d408625232b37b7a2420403cd498f/libavcodec/cuviddec.c#L1185-L1191
    #
    # while NVidia SDK samples exclude those with the following substrings in
    # long_name attribute
    #
    #  "QuickTime / MOV", "FLV (Flash Video)", "Matroska / WebM"
    match packets.codec.name:
        case "h264":
            packets = apply_bsf(packets, "h264_mp4toannexb")
        case "hevc":
            packets = apply_bsf(packets, "hevc_mp4toannexb")
        case _:
            pass

    decoder = NvDecDecoder()
    return decoder.decode(packets, device_config=device_config, flush=True, **kwargs)


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
    return _libspdl_cuda.decode_image_nvjpeg(
        data, device_config=device_config, **kwargs
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


class _DecoderCache:
    def __init__(self, decoder) -> None:
        self.decoder = decoder
        self.decoding = False


_THREAD_LOCAL = threading.local()


def _get_decoder_cache() -> _DecoderCache:
    if not hasattr(_THREAD_LOCAL, "_cache"):
        _THREAD_LOCAL._cache = _DecoderCache(_libspdl_cuda._nvdec_decoder())
    return _THREAD_LOCAL._cache


class NvDecDecoder:
    def __init__(self) -> None:
        self._cache: _DecoderCache = _get_decoder_cache()
        if self._cache.decoding:
            self._cache.decoder.reset()
            self._cache.decoding = False
        self._cache.decoder.set_init_flag()

    def decode(
        self,
        packets: VideoPackets,
        device_config: CUDAConfig | None = None,
        flush: bool = False,
        **kwargs,
    ) -> VideoFrames:
        self._cache.decoding = True
        ret = self._cache.decoder.decode(
            packets, device_config=device_config, flush=flush, **kwargs
        )
        self._cache.decoding = False
        return ret


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

    return _libspdl_cuda.transfer_buffer(buffer, device_config=device_config)


def transfer_buffer_cpu(buffer: CUDABuffer) -> CPUBuffer:
    """Move the given CUDA buffer to CPU.

    Args:
        buffer: Source data

    Returns:
        Buffer data on CPU.
    """
    return _libspdl_cuda.transfer_buffer_cpu(buffer)


################################################################################
# Color conversion
################################################################################
def nv12_to_rgb(
    buffers: list[CUDABuffer],
    *,
    device_config: CUDAConfig,
    coeff: int = 1,
) -> CUDABuffer:
    """Given CUDA buffers of NV12 images, batch and convert them to (interleaved) RGB.

    The pixel values are converted with the following formula;

    .. code-block::

       ┌ R ┐   ┌     ┐   ┌ Y - 16  ┐
       │ G │ = │  M  │ * │ U - 128 │
       └ B ┘   └     ┘   └ V - 128 ┘

    The value of 3x3 matrix ``M`` can be changed with the argument ``coeff``.
    By default, it uses BT709 conversion.

    Args:
        buffers: A list of buffers. The size of images must be same.
            Since it's NV12 format, the expected input size is ``(H + H/2, W)``.
        device_config: Secifies the target CUDA device, and stream to use.
        coeff: Select the matrix coefficient used for color conversion.
            The following values are supported.

            - ``1``: BT709 (default)
            - ``4``: FCC
            - ``5``: BT470
            - ``6``: BT601
            - ``7``: SMPTE240M
            - ``8``: YCgCo
            - ``9``: BT2020
            - ``10``: BT2020C

            If other values are provided, they are silently mapped to ``1``.

    Returns:
        A CUDA buffer object with the shape ``(batch, height, width, color=3)``.
    """
    return _libspdl_cuda.nv12_to_planar_rgb(
        buffers, device_config=device_config, matrix_coeff=coeff
    )


def nv12_to_bgr(
    buffers: list[CUDABuffer],
    *,
    device_config: CUDAConfig,
    coeff: int = 1,
) -> CUDABuffer:
    """Same as :py:func:`nv12_to_rgb`, but the order of the color channel is BGR."""
    return _libspdl_cuda.nv12_to_planar_bgr(
        buffers, device_config=device_config, matrix_coeff=coeff
    )


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
