import logging
from concurrent.futures import Future
from typing import Any, Dict, List, Sequence, Tuple, Type

import spdl.utils

from spdl.io import CPUBuffer, CUDABuffer, Frames, Packets

from . import _common, _preprocessing

__all__ = [
    "convert_frames",
    "decode_packets",
    "decode_packets_nvdec",
    "demux_media",
    "streaming_demux",
    "load_media",
    "batch_load_image",
    "batch_load_image_nvdec",
]

_LG = logging.getLogger(__name__)


def streaming_demux(
    media_type: str,
    src: str | bytes,
    timestamps: List[Tuple[float, float]],
    **kwargs,
) -> List[Future[Type[Packets]]]:
    """Demux the media of given time windows.

    The signature of this function is same as [spdl.io.async_streaming_demux][]
    except the return type.
    This function returns a list of `concurrent.futures.Future`s which in turn return
    Packets objects when fullfilled.

    Returns:
        List of Future object that returns Packets when fullfilled.
    """
    if not timestamps:
        raise ValueError("`timestamps` cannot be an empty list.")
    func = _common._get_stream_demux_func(media_type, src)
    return _common._futurize_generator(func, len(timestamps), src, timestamps, **kwargs)


def demux_media(
    media_type: str,
    src: str | bytes,
    timestamp: Tuple[float, float] | None = None,
    **kwargs,
) -> Future[Type[Packets]]:
    """Demux image or one chunk of audio/video region from the source.

    The signature of this function is same as [spdl.io.async_demux_media][]
    except the return type.
    This function returns `concurrent.futures.Future` which in turn returns a
    Packets object when fullfilled.

    Returns:
        Future object that returns Packets when fullfilled.
    """
    func = _common._get_demux_func(media_type, src)
    if media_type != "image" and timestamp is not None:
        kwargs["timestamp"] = timestamp
    return _common._futurize_task(func, src, **kwargs)


def decode_packets(packets: Type[Packets], **kwargs) -> Future[Type[Frames]]:
    """Decode packets.

    The signature of this function is same as [spdl.io.async_decode_packets][]
    except the return type.
    This function returns `concurrent.futures.Future` which in turn returns a
    Frames object when fullfilled.

    Returns:
        Future object that returns Frames when fullfilled.
    """
    func = _common._get_decoding_func(packets)
    if "filter_desc" not in kwargs:
        kwargs["filter_desc"] = _preprocessing.get_filter_desc(packets)
    return _common._futurize_task(func, packets, **kwargs)


def decode_packets_nvdec(
    packets: Type[Packets], cuda_device_index: int, **kwargs
) -> Future[CUDABuffer]:
    """Decode packets with NVDEC.

    The signature of this function is same as [spdl.io.async_decode_packets_nvdec][]
    except the return type.
    This function returns `concurrent.futures.Future` which in turn returns a
    Frames object when fullfilled.

    Returns:
        Future object that returns Buffer when fullfilled.
    """
    func = _common._get_nvdec_decoding_func(packets)
    return _common._futurize_task(
        func, packets, cuda_device_index=cuda_device_index, **kwargs
    )


def decode_media(
    media_type: str,
    src: str | bytes,
    **kwargs,
) -> Future[Type[Frames]]:
    """Demux and decode media from source.

    The signature of this function is same as [spdl.io.async_decode_media][]
    except the return type.
    This function returns `concurrent.futures.Future` which in turn returns a
    Frames object when fullfilled.

    Returns:
        Future object that returns Frames when fullfilled.
    """
    func = _common._get_decode_from_source_func(media_type, src)
    return _common._futurize_task(func, src, **kwargs)


def convert_frames(
    frames: Type[Frames] | Sequence[Frames],
    cuda_device_index: int | None = None,
    **kwargs,
) -> Future[CPUBuffer | CUDABuffer]:
    """Convert the decoded frames to buffer.

    The signature of this function is same as [spdl.io.async_convert_frames][]
    except the return type.
    This function returns `concurrent.futures.Future` which in turn returns a
    Buffer object when fullfilled.

    Returns:
        Future object that returns Buffer when fullfilled.
    """
    func = _common._get_conversion_func(frames, cuda_device_index)
    if cuda_device_index is not None:
        kwargs["cuda_device_index"] = cuda_device_index
    return _common._futurize_task(func, frames, **kwargs)


################################################################################
# High-level APIs
################################################################################


def load_media(
    media_type: str,
    src: str | bytes,
    *,
    demux_options: Dict[str, Any] | None = None,
    decode_options: Dict[str, Any] | None = None,
    convert_options: Dict[str, Any] | None = None,
    use_nvdec: bool = False,
) -> Future[CPUBuffer | CUDABuffer]:
    """Load media from source.

    The signature of this function is same as [spdl.io.async_load_media][]
    except the return type.
    This function returns `concurrent.futures.Future` which in turn returns a
    Buffer object when fullfilled.

    Returns:
        Future object that returns Buffer when fullfilled.
    """
    demux_options = demux_options or {}
    decode_options = decode_options or {}
    if use_nvdec and convert_options is not None:
        raise ValueError("NVDEC cannot be used with `convert_options`.")
    convert_options = convert_options or {}

    @spdl.utils.chain_futures
    def f():
        packets = yield demux_media(media_type, src, **demux_options)
        if use_nvdec:
            yield decode_packets_nvdec(packets, **decode_options)
        else:
            frames = yield decode_packets(packets, **decode_options)
            yield convert_frames(frames, **convert_options)

    return f()


def batch_load_image(
    srcs: List[str | bytes],
    *,
    width: int | None,
    height: int | None,
    pix_fmt: str | None = "rgb24",
    demux_options: Dict[str, Any] | None = None,
    decode_options: Dict[str, Any] | None = None,
    convert_options: Dict[str, Any] | None = None,
    strict: bool = True,
) -> Future[CPUBuffer | CUDABuffer]:
    """Load media from source.

    The signature of this function is same as [spdl.io.async_batch_load_image][]
    except the return type.
    This function returns `concurrent.futures.Future` which in turn returns a
    Buffer object when fullfilled.

    ??? note "Example"
        ```python
        >>> srcs = [
        ...     "sample1.jpg",
        ...     "sample2.png",
        ... ]
        >>> future = batch_load_image(
        ...     srcs,
        ...     width=124,
        ...     height=96,
        ...     pix_fmt="rgb24",
        ... )
        >>> buffer = future.result()  # blocking wait
        >>> array = spdl.io.to_numpy(buffer)
        >>> # An array with shape HWC==[2, 96, 124, 3]
        >>>
        ```

    Returns:
        Future object that returns Buffer when fullfilled.
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
            "`width`, `height` or `pix_fmt` and `filter_desc` in `decode_options` cannot be present at the same time."
        )
    elif filter_desc:
        decode_options["filter_desc"] = filter_desc

    @spdl.utils.chain_futures
    def _decode(src):
        packets = yield demux_media("image", src, **demux_options)
        yield decode_packets(packets, **decode_options)

    @spdl.utils.chain_futures
    def _convert(frames_futures):
        frames = yield spdl.utils.wait_futures(frames_futures, strict=strict)
        yield spdl.io.convert_frames(frames, **convert_options)

    return _convert([_decode(src) for src in srcs])


def batch_load_image_nvdec(
    srcs: List[str | bytes],
    *,
    cuda_device_index: int,
    width: int | None,
    height: int | None,
    pix_fmt: str | None = "rgba",
    demux_options: Dict[str, Any] | None = None,
    decode_options: Dict[str, Any] | None = None,
    strict: bool = True,
) -> Future[CUDABuffer]:
    if not srcs:
        raise ValueError("`srcs` must not be empty.")

    demux_options = demux_options or {}
    decode_options = decode_options or {}
    width = -1 if width is None else width
    height = -1 if height is None else height

    @spdl.utils.chain_futures
    def f():
        packets = yield spdl.utils.wait_futures(
            [demux_media("image", src, **demux_options) for src in srcs], strict=strict
        )
        yield spdl.io.decode_packets_nvdec(
            packets,
            cuda_device_index=cuda_device_index,
            width=width,
            height=height,
            pix_fmt=pix_fmt,
            strict=strict,
            **decode_options,
        )

    return f()
