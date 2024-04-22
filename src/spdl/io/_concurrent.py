import logging
from concurrent.futures import Future
from typing import Any, Dict, List, Optional, Tuple, Union

import spdl.utils

from . import _common, preprocessing

__all__ = [
    "convert_frames",
    "decode_packets",
    "decode_packets_nvdec",
    "demux_media",
    "streaming_demux",
    "load_media",
    "batch_load_image",
]

_LG = logging.getLogger(__name__)


def streaming_demux(
    media_type: str,
    src: Union[str, bytes, memoryview],
    timestamps: List[Tuple[float, float]],
    **kwargs,
) -> List[Future]:
    """Demux the media of given time windows.

    The signature of this function is same as [spdl.io.async_streaming_demux][]
    except the return type.
    This function returns a list of `concurrent.futures.Future`s which in turn return
    Packets objects when fullfilled.
    """
    func = _common._get_demux_func(media_type, src)
    return _common._futurize_generator(func, len(timestamps), src, timestamps, **kwargs)


def demux_media(
    media_type: str,
    src: Union[str, bytes, memoryview],
    timestamp: Optional[Tuple[float, float]] = None,
    **kwargs,
) -> Future:
    """Demux image or one chunk of audio/video region from the source.

    The signature of this function is same as [spdl.io.async_demux_media][]
    except the return type.
    This function returns `concurrent.futures.Future` which in turn returns a
    Packets object when fullfilled.
    """
    if media_type == "image":
        func = _common._get_demux_func(media_type, src)
        return _common._futurize_task(func, src, **kwargs)

    timestamps = [(0.0, float("inf")) if timestamp is None else timestamp]
    return streaming_demux(media_type, src, timestamps, **kwargs)[0]


def decode_packets(packets, **kwargs) -> Future:
    """Decode packets.

    The signature of this function is same as [spdl.io.async_decode_packets][]
    except the return type.
    This function returns `concurrent.futures.Future` which in turn returns a
    Frames object when fullfilled.
    """
    func = _common._get_decoding_func(packets)
    if "filter_desc" not in kwargs:
        kwargs["filter_desc"] = preprocessing.get_filter_desc(packets)
    return _common._futurize_task(func, packets, **kwargs)


def decode_packets_nvdec(packets, cuda_device_index, **kwargs) -> Future:
    """Decode packets with NVDEC.

    The signature of this function is same as [spdl.io.async_decode_packets_nvdec][]
    except the return type.
    This function returns `concurrent.futures.Future` which in turn returns a
    Frames object when fullfilled.
    """
    func = _common._get_nvdec_decoding_func(packets)
    return _common._futurize_task(
        func, packets, cuda_device_index=cuda_device_index, **kwargs
    )


def decode_media(
    media_type: str,
    src: Union[str, bytes, memoryview],
    **kwargs,
):
    """Demux and decode media from source.

    The signature of this function is same as [spdl.io.async_decode_media][]
    except the return type.
    This function returns `concurrent.futures.Future` which in turn returns a
    Frames object when fullfilled.
    """
    func = _common._get_decode_from_source_func(media_type, src)
    return _common._futurize_task(func, src, **kwargs)


def convert_frames(frames, **kwargs) -> Future:
    """Convert the decoded frames to buffer.

    The signature of this function is same as [spdl.io.async_convert_frames][]
    except the return type.
    This function returns `concurrent.futures.Future` which in turn returns a
    Buffer object when fullfilled.
    """
    func = _common._get_conversion_func(frames)
    return _common._futurize_task(func, frames, **kwargs)


################################################################################
# High-level APIs
################################################################################


@spdl.utils.chain_futures
def load_media(
    media_type: str,
    src: Union[str, bytes, memoryview],
    *,
    demux_options: Optional[Dict[str, Any]] = None,
    decode_options: Optional[Dict[str, Any]] = None,
    convert_options: Optional[Dict[str, Any]] = None,
    use_nvdec: bool = False,
) -> Future:
    """Load media from source.

    The signature of this function is same as [spdl.io.async_load_media][]
    except the return type.
    This function returns `concurrent.futures.Future` which in turn returns a
    Buffer object when fullfilled.
    """
    demux_options = demux_options or {}
    decode_options = decode_options or {}
    convert_options = convert_options or {}
    packets = yield demux_media(media_type, src, **demux_options)
    if use_nvdec:
        frames = yield decode_packets_nvdec(packets, **decode_options)
    else:
        frames = yield decode_packets(packets, **decode_options)
    yield convert_frames(frames, **convert_options)


def batch_load_image(
    srcs: List[Union[str, bytes]],
    *,
    width: int | None,
    height: int | None,
    pix_fmt: str | None = "rgb24",
    demux_options: Optional[Dict[str, Any]] = None,
    decode_options: Optional[Dict[str, Any]] = None,
    convert_options: Optional[Dict[str, Any]] = None,
    strict: bool = True,
):
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
    """
    if not srcs:
        raise ValueError("`srcs` must not be empty.")

    demux_options = demux_options or {}
    decode_options = decode_options or {}
    convert_options = convert_options or {}

    filter_desc = preprocessing.get_video_filter_desc(
        width=width,
        height=height,
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
