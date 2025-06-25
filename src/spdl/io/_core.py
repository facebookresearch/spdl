# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import logging
import threading
import warnings
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Generic, overload, TYPE_CHECKING, TypeVar

# We import NumPy only when type-checking.
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


try:
    from spdl._internal import log_api_usage_once
except ImportError:

    def log_api_usage_once(_: str) -> None:
        pass


from spdl.io import (
    AudioCodec,
    AudioDecoder,
    AudioEncodeConfig,
    AudioEncoder,
    AudioFrames,
    AudioPackets,
    CPUBuffer,
    CPUStorage,
    CUDABuffer,
    CUDAConfig,
    DecodeConfig,
    ImageCodec,
    ImageDecoder,
    ImageFrames,
    ImagePackets,
    VideoCodec,
    VideoDecoder,
    VideoEncodeConfig,
    VideoEncoder,
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
    # BIT STREAM FILTERING
    "BSF",
    "apply_bsf",
    # DECODING
    "Decoder",
    "decode_packets",
    "decode_packets_nvdec",
    "decode_image_nvjpeg",
    "NvDecDecoder",
    "nvdec_decoder",
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
    "Muxer",
    "encode_image",
    "create_reference_audio_frame",
    "create_reference_video_frame",
]

torch = lazy_import("torch")  # pyre-ignore: [31]

_LG = logging.getLogger(__name__)
T = TypeVar("T")

_FILTER_DESC_DEFAULT = "__PLACEHOLDER__"


SourceType = str | Path | bytes | UintArray | Tensor

################################################################################
# Demuxing
################################################################################


class Demuxer:
    """Demuxer can demux audio, video and image from the source.

    Args:
        src: Source identifier.
            If `str` type, it is interpreted as a source location,
            such as local file path or URL. If `bytes` type,
            then they are interpreted as in-memory data.
            If array type (objects implement buffer protocol,
            such as NumPy NDArray and PyTorch Tensor), then they must be
            1 dimensional uint8 array, which contains the raw bytes of the
            source.

        demux_config (DemuxConfig): Custom I/O config.
    """

    def __init__(self, src: SourceType, **kwargs):
        if isinstance(src, Path):
            src = str(src)
        self._demuxer = _libspdl._demuxer(src, **kwargs)

    def __getattr__(self, name: str):
        if name == "streaming_demux_video":
            warnings.warn(
                "`streaming_demux_video` method has been deprecated. "
                "Please use `streaming_demux` method.",
                stacklevel=2,
            )
            return self._streaming_demux_video

        raise AttributeError(
            f"{self.__class__.__name__} object has no attribute {name!r}"
        )

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
        log_api_usage_once("spdl.io.Demuxer.demux_audio")

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
        log_api_usage_once("spdl.io.Demuxer.demux_video")

        return self._demuxer.demux_video(window=window, **kwargs)

    def demux_image(self, **kwargs) -> ImagePackets:
        """Demux image from the source.

        Returns:
            Demuxed image packets.
        """
        log_api_usage_once("spdl.io.Demuxer.demux_image")

        return self._demuxer.demux_image(**kwargs)

    def _streaming_demux_video(
        self,
        num_packets: int,
        bsf: str | None = None,
    ) -> Iterator[VideoPackets]:
        """Demux video frames in streaming fashion.

        Args:
            num_packets: The number of packets to return at a time.
            bsf: An optional bitstream filter.
        """
        bsf_ = None if bsf is None else BSF(self.video_codec, bsf)

        ite = self.streaming_demux(self.video_stream_index, num_packets=num_packets)
        for packets in ite:
            if bsf_ is not None:
                packets = bsf_.filter(packets)
            yield packets

        if bsf_ is not None and len(packets := bsf_.flush()):
            yield packets

    @overload
    def streaming_demux(
        self,
        indices: None = None,
        *,
        duration: float = -1,
        num_packets: int = -1,
    ) -> Iterator[VideoPackets | AudioPackets]: ...

    @overload
    def streaming_demux(
        self,
        indices: int,
        *,
        duration: float = -1,
        num_packets: int = -1,
    ) -> Iterator[VideoPackets | AudioPackets]: ...

    @overload
    def streaming_demux(
        self,
        indices: Sequence[int] | set[int],
        *,
        duration: float = -1,
        num_packets: int = -1,
    ) -> Iterator[dict[int, VideoPackets | AudioPackets]]: ...

    def streaming_demux(
        self,
        indices=None,
        *,
        duration=-1,
        num_packets=-1,
    ):
        """Stream demux packets from the source.

        .. admonition:: Example - Streaming decoding audio

           src = "foo.mp4"
           with spdl.io.Demuxer(src) as demuxer:
               index = demuxer.audio_stream_index
               audio_decoder = spdl.io.Decoder(demuxer.audio_codec)
               packet_stream = demuxer.streaming_demux([index], duration=3)
               for packets in packet_stream:
                   if index in packets:
                       frames = decoder.decode(packets[index])
                       buffer = spd.io.convert_frames(frames)

        """
        log_api_usage_once("spdl.io.Demuxer.streaming_demux")

        if duration <= 0 and num_packets <= 0:
            raise ValueError("Either `duration` or `num_packets` must be specified.")
        if duration > 0 and num_packets > 0:
            raise ValueError(
                "Only one of `duration` or `num_packets` can be specified. ",
                f"Found: {duration=}, {num_packets=}.",
            )

        if indices is None:
            idxs = {0}
        else:
            idxs = set([indices] if isinstance(indices, int) else indices)

        ite = self._demuxer.streaming_demux(
            idxs, duration=duration, num_packets=num_packets
        )
        return _StreamingDemuxer(
            ite, self, unwrap=(indices is None or isinstance(indices, int))
        )

    def has_audio(self) -> bool:
        """Returns true if the source has audio stream."""
        return self._demuxer.has_audio()

    @property
    def video_stream_index(self) -> int:
        """The index of default video stream."""
        return self._demuxer.video_stream_index

    @property
    def audio_stream_index(self) -> int:
        """The index of default audio stream."""
        return self._demuxer.audio_stream_index

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


class _StreamingDemuxer:
    def __init__(self, ite, demuxer: Demuxer, unwrap: bool) -> None:
        self._demuxer = demuxer  # For keeping the reference.
        self._ite = ite
        self._unwrap = unwrap

    def __iter__(self):
        return self

    def __next__(self):
        if self._ite.done():
            raise StopIteration
        item = self._ite.next()
        if self._unwrap:
            assert len(item) == 1
            return next(iter(item.values()))
        return item


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


################################################################################
# Bit stream filtering
################################################################################

TCodec = TypeVar("TCodec")
TPackets = TypeVar("TPackets")


class BSF(Generic[TCodec, TPackets]):
    """Apply bitstream filtering on packets object.

    The primal usecase of BFS in SPDL is to convert the H264 video packets to
    Annex B for video decoding.

    .. seealso::

       - https://ffmpeg.org/ffmpeg-bitstream-filters.html: The list of available
         bitstream filters and their usage.

       - :py:class:`NvDecDecoder`: NVDEC decoding requires the input video packets
         to be Annex B.

       - :py:func:`apply_bsf`: Applies bitstream filtering to the packet as one-off
         operation.

    .. admonition:: Example

       src = "foo.mp4"
       demuxer = spdl.io.Demuxer(src)
       # Note: When demuxing in streaming fashion, the packets does not have codec information.
       # To initialize BSF, the codec must be retrieved from Demuxer class.
       bsf = spdl.io.BSF(demuxer.video_codec)

       for packets in demuxer.streaming_demux_video(...):
           packets = bsf.filter(packets)

           ...

        packets = bsf.flush()
        ...

    Args:
        codec: The input codec.
    """

    def __init__(self, codec: TCodec, bsf: str) -> None:
        self._bsf = _libspdl._make_bsf(codec, bsf)

    def filter(self, packets: TPackets, flush: bool = False) -> TPackets | None:
        """Apply the filter to the input packets

        Args:
            packets: The input packets.
            flush: If ``True``, then notify the filter that this is the end
                of the stream and let it flush the internally buffered packets.

        Returns:
            Filtered packet object or ``None`` if the internal filtering mechanism
            holds the packets and does not return any packet.
        """
        packets = self._bsf.filter(packets, flush=flush)
        # TODO: push down the optional to C++ core signature.
        if len(packets):
            return packets
        return None

    def flush(self) -> TPackets | None:
        # TODO: push down the optional to C++ core signature.
        packets = self._bsf.flush()
        if len(packets):
            return packets
        return None


@overload
def apply_bsf(packets: AudioPackets, bsf: str) -> AudioPackets: ...
@overload
def apply_bsf(packets: VideoPackets, bsf: str) -> VideoPackets: ...
@overload
def apply_bsf(packets: ImagePackets, bsf: str) -> ImagePackets: ...


def apply_bsf(packets, bsf):
    """Apply bit stream filter to packets.

    The primal usecase of BFS in SPDL is to convert the H264 video packets to
    Annex B for video decoding.

    .. admonition:: Example - One-off demuxing

       src = "foo.mp4"
       packets = spdl.io.demux_video(src)
       packets = spdl.io.apply_bsf(packets)

    Args:
        packets: Packets (audio/video/image) object
        bsf: A bitstream filter description.

    .. seealso::

       - https://ffmpeg.org/ffmpeg-bitstream-filters.html: The list of available
         bitstream filters and their usage.

       - :py:class:`NvDecDecoder`: NVDEC decoding requires the input video packets
         to be Annex B.

       - :py:class:`BSF`: Same operation but for streaming processing.
    """
    if packets.codec is None:
        raise ValueError("The packets object does not have codec.")
    return BSF(packets.codec, bsf).filter(packets, flush=True)


################################################################################
# Decoding
################################################################################


def _resolve_filter_graph(
    filter_desc: str, codec, timestamp: tuple[float, float] | None = None
) -> str:
    match codec:
        case _libspdl.AudioCodec():
            if filter_desc == _FILTER_DESC_DEFAULT:
                filter_desc = _preprocessing.get_audio_filter_desc(timestamp=timestamp)
            src = _preprocessing.get_abuffer_desc(codec)
            sink = "abuffersink"
        case _libspdl.VideoCodec():
            if filter_desc == _FILTER_DESC_DEFAULT:
                filter_desc = _preprocessing.get_video_filter_desc(timestamp=timestamp)
            src = _preprocessing.get_buffer_desc(codec)
            sink = "buffersink"
        case _libspdl.ImageCodec():
            if filter_desc == _FILTER_DESC_DEFAULT:
                filter_desc = _preprocessing.get_video_filter_desc()
            src = _preprocessing.get_buffer_desc(codec)
            sink = "buffersink"
        case _:
            raise ValueError(f"Unexpected codec type: {type(codec)}")

    return f"{src},{filter_desc},{sink}"


@overload
def Decoder(
    codec: AudioCodec,
    *,
    filter_desc: str | None = _FILTER_DESC_DEFAULT,
    decode_config: DecodeConfig | None = None,
) -> AudioDecoder: ...


@overload
def Decoder(
    codec: VideoCodec,
    *,
    filter_desc: str | None = _FILTER_DESC_DEFAULT,
    decode_config: DecodeConfig | None = None,
) -> VideoDecoder: ...


@overload
def Decoder(
    codec: ImageCodec,
    *,
    filter_desc: str | None = _FILTER_DESC_DEFAULT,
    decode_config: DecodeConfig | None = None,
) -> ImageDecoder: ...


class _Decoder:
    def __init__(self, decoder) -> None:
        self._decoder = decoder

    def decode(self, packets):
        # TODO: push down the optional to C++
        frames = self._decoder.decode(packets)
        if len(frames):
            return frames
        return None

    def flush(self):
        # TODO: push down the optional to C++
        frames = self._decoder.flush()
        if len(frames):
            return frames
        return None


def Decoder(codec, *, filter_desc=_FILTER_DESC_DEFAULT, decode_config=None):
    """Initialize a decoder object that can incrementally decode packets of the same stream.

    .. admonition:: Example

       src = "foo.mp4"

       demuxer = spdl.io.Demuxer(src)
       decoder = spdl.io.Decoder(demuxer.video_codec)
       for packets in demuxer.streaming_demux_video(num_frames):
           frames: VideoFrames | None = decoder.decode(packets)
           ...

        frames: VideoFrames | None = decoder.flush()

    Args:
        codec (AudioCodec, VideoCodec or ImageCodec):
            The codec of the incoming packets.
        filter_desc (str): *Optional:* See :py:func:`decode_packets`.
        decode_config (DecodeConfig): *Optional:* See :py:func:`decode_packets`.

    Returns:
        Decoder instance.

    """
    log_api_usage_once("spdl.io.Decoder")

    if filter_desc is not None:
        filter_desc = _resolve_filter_graph(filter_desc, codec)

    decoder = _libspdl._make_decoder(
        codec, filter_desc=filter_desc, decode_config=decode_config
    )
    return _Decoder(decoder)


@overload
def decode_packets(
    packets: AudioPackets,
    filter_desc: str | None = _FILTER_DESC_DEFAULT,
    decode_config: DecodeConfig | None = None,
    **kwargs,
) -> AudioFrames: ...
@overload
def decode_packets(
    packets: VideoPackets,
    filter_desc: str | None = _FILTER_DESC_DEFAULT,
    decode_config: DecodeConfig | None = None,
    **kwargs,
) -> VideoFrames: ...
@overload
def decode_packets(
    packets: ImagePackets,
    filter_desc: str | None = _FILTER_DESC_DEFAULT,
    decode_config: DecodeConfig | None = None,
    **kwargs,
) -> ImageFrames: ...


def decode_packets(
    packets,
    filter_desc=_FILTER_DESC_DEFAULT,
    decode_config=None,
    **kwargs,
):
    """Decode packets.

    Args:
        packets (AudioPackets, VideoPackets or ImagePackets): Packets object.

        filter_desc (str):
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

    if filter_desc is not None:
        filter_desc = _resolve_filter_graph(
            filter_desc, packets.codec, getattr(packets, "timestamp", None)
        )

    return _libspdl.decode_packets(
        packets, filter_desc=filter_desc, decode_config=decode_config, **kwargs
    )


def decode_packets_nvdec(
    packets: VideoPackets,
    *,
    device_config: CUDAConfig,
    pix_fmt: str = "rgb",
    **kwargs,
) -> CUDABuffer:
    """**[Experimental]** Decode packets with NVDEC.

    .. warning::

       This API is exmperimental. The performance is not probed, and the specification
       might change.

    .. versionchanged:: 0.0.10

       - The alpha channel was removed, and the supported format values were changed
         from ``"rgba"`` and ``"bgra"`` to ``"rgb"`` and ``"bgr"``.

       - ``width`` and ``height`` options were renamed to ``scale_width`` and
         ``scale_height``.

    .. note::

       Unlike FFmpeg-based decoding, NVDEC returns GPU buffer directly.

    .. seealso::

       :py:class:`NvDecDecoder`: The underlying decoder implementation, which supports
       incremental decoding.

    Args:
        packets: Packets object.

        device_config: The device to use for decoding. See :py:func:`spdl.io.cuda_config`.

        crop_left, crop_top, crop_right, crop_bottom (int):
            *Optional:* Crop the given number of pixels from each side.

        scale_width, scale_height (int): *Optional:* Resize the frame. Resizing is done after
            cropping.

        pix_fmt (str or `None`): *Optional:* Change the format of the pixel.
            Supported value is ``"rgb"`` and ``"bgr"``. Default: ``"rgb"``.

    Returns:
        A CUDABuffer object.
    """
    log_api_usage_once("spdl.io.decode_packets_nvdec")

    for k in ("width", "height"):
        if k in kwargs:
            warnings.warn(
                f"The argument '{k}' has been renamed to 'scale_{k}'. "
                "Please update the function call.",
                stacklevel=2,
            )
            kwargs[f"scale_{k}"] = kwargs.pop(k)

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

    decoder = nvdec_decoder()
    decoder.init(device_config, packets.codec, **kwargs)
    buffer = decoder.decode(packets)
    buffer += decoder.flush()

    match pix_fmt:
        case "rgb":
            buffer = _libspdl_cuda.nv12_to_planar_rgb(
                buffer, device_config=device_config
            )
        case "bgr":
            buffer = _libspdl_cuda.nv12_to_planar_bgr(
                buffer, device_config=device_config
            )

    return buffer


def decode_image_nvjpeg(
    src: str | bytes | Sequence[bytes],
    *,
    device_config: CUDAConfig | None = None,
    **kwargs,
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
    log_api_usage_once("spdl.io.decode_image_nvjpeg")

    if device_config is None:
        if "cuda_config" not in kwargs:
            raise ValueError("device_config must be provided.")
        if "cuda_config" in kwargs:
            warnings.warn(
                "`cuda_config` argument has been renamed to `device_config`.",
                stacklevel=2,
            )
            device_config = kwargs["cuda_config"]

    if isinstance(src, str):
        with open(src, "rb") as f:
            data = f.read()
    else:
        data = src
    return _libspdl_cuda.decode_image_nvjpeg(
        data, device_config=device_config, **kwargs
    )


class NvDecDecoder:
    """NvDecDecoder()
    Decods video packets using NVDEC hardware acceleration.

    Use :py:func:`nvdec_decoder` to instantiate.

    To decode videos with NVDEC, first you initialize the
    decoder, then feed video packets. Finally, call flush to
    let the decoder know that it reached the end of the video
    stream, so that the decoder flushes its internally buffered
    frames.

    .. note::

       To decode H264 and HEVC videos, the packets must be Annex B
       format. You can convert video packets to Annex B format by
       applying bit stream filter while demuxing or after demuxing.
       See the examples bellow.

    .. seealso::

       - :py:func:`decode_packets_nvdec`: Decode video packets using
         NVDEC.
       - :py:func:`streaming_load_video_nvdec`: Decode video frames
         from source in streaming fashion.
       - :py:mod:`streaming_nvdec_decoding`: Demonstrates how to
         decode a long video using NVDEC.

    .. admonition:: Example - decoding the whole video

       .. code-block::

          cuda_config = spdl.io.cuda_config(device_index=0)

          packets = spdl.io.demux_video(src)
          # Convert to Annex B format
          if (c := packets.codec.name) in ("h264", "hevc"):
              packets = spdl.io.apply_bsf(f"{c}_mp4toannexb")

          # Initialize the decoder
          decoder = nvdec_decoder()
          decoder.init(cuda_config, packets.codec, ...)

          # Decode packets
          frames = decoder.decode(packets)

          # Done
          frames += decoder.flush()

          # Convert (and batch) the NV12 frames into RGB
          frames = spdl.io.nv12_to_rgb(frames)

    .. admonition:: Example - incremental decoding

       .. code-block::

          cuda_config = spdl.io.cuda_config(device_index=0)

          demuxer = spdl.io.Demuxer(src)
          codec = demuxer.video_codec

          match codec.name:
              case "h264" | "hevc":
                  bsf = f"{codec.name}_mp4toannexb"
              case _:
                  bsf = None

          decoder = nvdec_decoder()
          decoder.init(cuda_config, codec, ...)

          for packets in demuxer.streaming_demux_video(10, bsf=bsf):
              buffer = decoder.decode(packets)
              buffer = spdl.io.nv12_to_rgb(buffer)
              # Process buffer here

          buffer = decoder.flush()
          buffer = spdl.io.nv12_to_rgb(buffer)
    """

    def __init__(self, decoder) -> None:
        log_api_usage_once("spdl.io.NvDecDecoder")

        self._decoder = decoder

    def init(
        self,
        cuda_config: CUDAConfig,
        codec: VideoCodec,
        *,
        crop_left: int = 0,
        crop_top: int = 0,
        crop_right: int = 0,
        crop_bottom: int = 0,
        scale_width: int = -1,
        scale_height: int = -1,
    ) -> None:
        """Initialize the decoder.

        This function must be called before decoding can happen.

        .. note::

           Creation of underlying decoder object is expensive.
           Typically, it takes about 300ms or more.

           To mitigate this the implementation tries to reuse the decoder.
           This works if the new video uses the same codecs as
           the previous one, and the difference is limited to the
           resolution of the video.

           If you are processing videos of different codecs, then the
           decoder has to be re-created.

        Args:
            cuda_config: The device configuration. Specifies the GPU of which
                video decoder chip is used, the CUDA memory allocator and
                CUDA stream used to fetch the result from the decoder engine.

            codec: The information of the source video.

            crop_left, crop_top, crop_right, crop_bottom (int):
                *Optional:* Crop the given number of pixels from each side.

            scale_width, scale_height (int): *Optional:* Resize the frame.
                Resizing is applied after cropping.
        """
        self._decoder.init(
            cuda_config,
            codec,
            crop_left=crop_left,
            crop_top=crop_top,
            crop_right=crop_right,
            crop_bottom=crop_bottom,
            scale_width=scale_width,
            scale_height=scale_height,
        )

    def decode(self, packets: VideoPackets) -> list[CUDABuffer]:
        """Decode video frames from the give packets.

        .. note::

           Due to how video codec works, the number of returned frames
           do not necessarily match the number of packets provided.

           The method can return less number of frames or more number of
           frames.

        Args:
            packets: Video packets.

        Returns:
            The decoded frames.
        """
        return self._decoder.decode(packets)

    def flush(self) -> list[CUDABuffer]:
        """Notify the decoder the end of video stream, and fetch buffered frames.

        Returns:
            The decoded frames. (can be empty)
        """
        return self._decoder.flush()


_THREAD_LOCAL = threading.local()


def _get_decoder():
    if not hasattr(_THREAD_LOCAL, "_decoder"):
        _THREAD_LOCAL._decoder = _libspdl_cuda._nvdec_decoder()
    return _THREAD_LOCAL._decoder


def nvdec_decoder(use_cache: bool = True) -> NvDecDecoder:
    """Instantiate an :py:class:`NvDecDecoder` object.

    Args:
        use_cache: If ``True`` (default), the decoder instance cached in thread
            local storage is used. Otherwise a new decoder instance is created.
    """
    if use_cache:
        decoder = _get_decoder()
        decoder.reset()
    else:
        decoder = _libspdl_cuda._nvdec_decoder()

    return NvDecDecoder(decoder)


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
            The shape of the buffer object is

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
    kernel execution. See :py:func:`spdl.io.cpu_storage` for the detail.

    Args:
        vals: NumPy array with int64 dtype..
        storage (spdl.io.CPUStorage): Storage object. See :py:func:`spdl.io.cpu_storage`.

    Returns:
        A Buffer object.
    """
    return _libspdl.convert_array(vals, storage=storage)


def create_reference_audio_frame(
    array, sample_fmt: str, sample_rate: int, pts: int
) -> AudioFrames:
    """Create an AudioFrame object which refers to the given array/tensor.

    This function should be used when the media data processed in Python should
    be further processed by filter graph, and/or encoded.

    .. attention::

       The resulting frame object references the memory region owned by the input
       array, but it does not own the reference to the original array.

       Make sure that the array object is alive until the frame object is consumed.

    Args:
        array: 2D array or tensor.
            The dtype and channel layout must match what is provided to
            ``sample_fmt``.

        sample_fmt: The format of sample. The valid values and corresponding data type is as follow.

            - ``"u8"``, ``"u8p"`` : 8-bit unsigned integer.
            - ``"s16"``, ``"s16p"`` : 16-bit signed integer.
            - ``"s32"``, ``"s32p"`` : 32-bit signed integer.
            - ``"s64"``, ``"s64p"`` : 64-bit signed integer.
            - ``"flt"``, ``"fltp"`` : 32-bit floating-point.
            - ``"dbl"``, ``"dblp"`` : 64-bit floating-point.

            The suffix ``"p"`` means planar format (channel-first), the input array is
            interpreted as ``(num_channels, num_frames)``.
            Otherwise it is interpreted as packed format (channel-last), i.e.
            ``(num_frames, num_channels)``.

        sample_rate: The sample rate of the audio

        pts: The time of the first sample, in the discrete time unit of sample rate.
            Usually it is the number of samples previously processed.

    Returns:
       Frames object that references the memory region of the input data.
    """
    return _libspdl.create_reference_audio_frame(
        array=array,
        sample_fmt=sample_fmt,
        sample_rate=sample_rate,
        pts=pts,
    )


def create_reference_video_frame(
    array, pix_fmt: str, frame_rate: tuple[int, int], pts: int
) -> VideoFrames:
    """Create an VideoFrame object which refers to the given array/tensor.

    This function should be used when the media data processed in Python should
    be further processed by filter graph, and/or encoded.

    .. attention::

       The resulting frame object references the memory region owned by the input
       array, but it does not own the reference to the original array.

       Make sure that the array object is alive until the frame object is consumed.

    Args:
        array: 3D or 4D array or tensor.
            The dtype and channel layout must match what is provided to
            ``sample_fmt``.

        pix_fmt: The image format. The valid values and corresponding shape is as follow.

            - ``"rgb24"``, ``"bgr24"``: Interleaved RGB/BGR in shape of ``(N, H, W, C==3)``.
            - ``"gray8"``, ``"gray16"``: Grayscale image of 8bit unsigned integer or
              16 bit signed integer in shape of ``(N, H, W)``.
            - ``"yuv444p"``: Planar YUV format in shape of ``(N, C==3, H, W)``.

        frame_rate: The frame rate of the video expressed asfraction.
            ``(numerator, denominator)``.

        pts: The time of the first video frame, in the discrete time unit of frame rate.
            Usually it is the number of frames previously processed.

    Returns:
       Frames object that references the memory region of the input data.
    """
    return _libspdl.create_reference_video_frame(
        array=array,
        pix_fmt=pix_fmt,
        frame_rate=frame_rate,
        pts=pts,
    )


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
    sync: bool = False,
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

        sync: If True, the function waits for the completion after launching the kernel.

    Returns:
        A CUDA buffer object with the shape ``(batch, height, width, color=3)``.
    """
    ret = _libspdl_cuda.nv12_to_planar_rgb(
        buffers,
        device_config=device_config,
        matrix_coeff=coeff,
    )
    if sync:
        _libspdl_cuda.synchronize_stream(device_config)
    return ret


def nv12_to_bgr(
    buffers: list[CUDABuffer],
    *,
    device_config: CUDAConfig,
    coeff: int = 1,
    sync: bool = False,
) -> CUDABuffer:
    """Same as :py:func:`nv12_to_rgb`, but the order of the color channel is BGR."""
    ret = _libspdl_cuda.nv12_to_planar_bgr(
        buffers,
        device_config=device_config,
        matrix_coeff=coeff,
    )
    if sync:
        _libspdl_cuda.synchronize_stream(device_config)

    return ret


################################################################################
# Encoding
################################################################################
class _Encoder:
    def __init__(self, encoder) -> None:
        self._encoder = encoder

    def encode(self, frames):
        # TODO: push down this logic to C++
        packets = self._encoder.encode(frames)
        if len(packets) == 0:
            return None
        return packets

    def flush(self):
        # TODO: push down this logic to C++
        packets = self._encoder.flush()
        if len(packets) == 0:
            return None
        return packets

    def __getattr__(self, name: str):
        return getattr(self._encoder, name)


class Muxer:
    """Multiplexer that convines multiple packet streams. e.g. create a video

    Args:
        dst: The destination such as file path, pipe, URL (such as RTMP, UDP).
        format: *Optional* Override the output format, or specify the output media device.
            This argument serves two different use cases.

            1) Override the output format.
               This is useful when writing raw data or in a format different from the extension.

            2) Specify the output device.
               This allows to output media streams to hardware devices,
               such as speaker and video screen.

            .. note::

               This option roughly corresponds to ``-f`` option of ``ffmpeg`` command.
               Please refer to the ffmpeg documentations for possible values.

               https://ffmpeg.org/ffmpeg-formats.html#Muxers

               For device access, the available values vary based on hardware (AV device) and
               software configuration (ffmpeg build).
               Please refer to the ffmpeg documentations for possible values.

               https://ffmpeg.org/ffmpeg-devices.html#Output-Devices
    """

    def __init__(self, dst: str | Path, /, *, format: str | None = None) -> None:
        self._muxer = _libspdl.muxer(str(dst), format=format)
        self._open = False

    @overload
    def add_encode_stream(
        self,
        config: AudioEncodeConfig,
        *,
        encoder: str | None = None,
        encoder_config: dict[str, str] | None = None,
    ) -> AudioEncoder: ...

    @overload
    def add_encode_stream(
        self,
        config: VideoEncodeConfig,
        *,
        encoder: str | None = None,
        encoder_config: dict[str, str] | None = None,
    ) -> VideoEncoder: ...

    def add_encode_stream(self, config, *, encoder=None, encoder_config=None):
        """Add an output stream with encoding.

        Use this method when you want to create a media from tensor/array.

        Args:
            config: Encoding (codec) configuration.
                See the corresponding factory functions for the detail.
                (:py:func:`audio_encode_config` and :py:func:`video_encode_config`)
            encoder: Specify or override the encoder to use.
                Use `ffmpeg -encoders` to list the available encoders.
            encoder_config: Encoder-specific options.
                Use `ffmpeg -h encoder=<ENCODER>` to list the available options.

        Returns:
            Encoder object which can be used to encode frames object into packets
                object.
        """
        encoder = self._muxer.add_encode_stream(
            config=config, encoder=encoder, encoder_config=encoder_config
        )
        return _Encoder(encoder)

    @overload
    def add_remux_stream(self, codec: AudioCodec) -> None: ...

    @overload
    def add_remux_stream(self, codec: VideoCodec) -> None: ...

    def add_remux_stream(self, codec) -> None:
        """Add an output stream without encoding.

        Use this method when you want to pass demuxed packets to output stream
        without decoding.

        Args:
            codec: Codec parameters from the source.

        .. admonition:: Example

           demuxer = spdl.io.Demuxer("source_video.mp4")

           muxer = spdl.io.Muxer("stripped_audio.aac")
           muxer.add_remux_stream(demuxer.audio_codec)

           with muxer.open():
            for packets in demuxer.streaming_demux(num_packets=5):
                muxer.write(0, packets)
        """
        self._muxer.add_remux_stream(codec)

    def open(self, muxer_config: dict[str, str] | None = None) -> "Muxer":
        """Open the muxer (output file) for writing.

        Args:
            Options spefici to devices and muxers.


        .. admonition:: Example - Protocol option

           muxer = spdl.io.Muxer("rtmp://localhost:1234/live/app", format="flv")
           muxer.add_encode_stream(...)
           # Passing protocol option `listen=1` makes Muxer act as RTMP server.
           with muxer.open(muxer_config={"listen": "1"}):
               muxer.write(0, video_packet)

        .. admonition:: Example - Device option

           muxer = spdl.io.Muxer("-", format="sdl")
           muxer.add_encode_stream(...)
           # Open SDL video player with fullscreen
           with muxer.open(muxer_config={"window_fullscreen": "1"}):
               muxer.write(0, video_packet)

        """
        self._muxer.open(muxer_config)
        self._open = True
        return self

    def write(self, stream_index: int, packets: AudioPackets | VideoPackets) -> None:
        """Write packets to muxer.

        Args:
            stream_index: The stream to write to.
            packets: Audio/video data.
        """
        self._muxer.write(stream_index, packets)

    def flush(self) -> None:
        """Notify the muxer that all the streams are written.

        This is automatically called when using `Muxer` as context manager.
        """
        self._muxer.flush()

    def close(self) -> None:
        """Close the resource.

        This is automatically called when using `Muxer` as context manager.
        """
        self._muxer.close()

    def __enter__(self) -> "Muxer":
        """Context manager to automatically clean up the resources.

        .. admobition::

           muxer = spdl.io.Muxer("foo.mp4")

           # ... configure the output stream

           with muxer.open():

                # ... write data
        """
        return self

    def __exit__(self, *_) -> None:
        """Flush the internally buffered packets and close the open resource.

        .. seealso::

           - :py:meth:`~Muxer.__enter__`

        """
        self.flush()
        self.close()


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
