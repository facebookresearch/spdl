import asyncio
import builtins
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import spdl.io
from spdl.lib import _libspdl

from . import _common, _preprocessing


__all__ = [
    "_async_sleep",
    "_async_sleep_multi",
    "async_convert_frames",
    "async_decode_packets",
    "async_decode_packets_nvdec",
    "async_demux_media",
    "async_streaming_demux",
    "async_load_media",
    "async_batch_load_image",
]

_LG = logging.getLogger(__name__)


def _async_sleep(time: int):
    """Sleep for a given duration (milliseconds)."""
    future = _common._futurize_task(_libspdl.async_sleep, time)
    return _handle_future(future), future.__spdl_future


def _async_sleep_multi(time: int, count: int):
    """Sleep for a given duration (milliseconds), for given times."""
    assert count > 0
    futures = _common._futurize_generator(
        _libspdl.async_sleep_multi, count + 1, time, count
    )
    return _handle_futures(futures), futures[0].__spdl_future


# The time it waits before rethrowing the async exception.
#
# When the background async op fails, the front end Python code tries to
# fetch and propagate the exception by letting the backend code throw it.
#
# However, Python might reach to the rethrowing part before the background
# C++ execution sets the exception, and in this case, instead of the
# original exception, the background code throws Folly's FutureNotReady
# exception. This practically hides the actual exception that caused the
# background async code to fail.
#
# So we wait a bit before rethrowing the exception. It is not guaranteed
# that this will ensure the C++ exception to be ready.
# This will delay the completion of async code only if it fails.
# This does not affect the performance of success cases.
# It should not affect the overall throughput too.
_EXCEPTION_BACKOFF = 1.00


async def _handle_future(future):
    try:
        return await asyncio.futures.wrap_future(future)
    # Handle the case where unexpected/external thing happens
    except asyncio.CancelledError as e:
        future.__spdl_future.cancel()
        try:
            # Wait till the cancel request is fullfilled or job finishes
            await asyncio.futures.wrap_future(future)
        except (asyncio.CancelledError, spdl.io.AsyncIOFailure):
            pass
        except Exception:
            _LG.exception(
                "An exception was raised while waiting for the task to be cancelled."
            )
        # Propagate the error.
        raise e


async def _async_task(func, *args, **kwargs):
    future = _common._futurize_task(func, *args, **kwargs)
    return await _handle_future(future)


async def _handle_futures(futures):
    try:
        for future in futures:
            yield await asyncio.futures.wrap_future(future)
    except asyncio.CancelledError as ce:
        future.__spdl_future.cancel()
        try:
            # Wait till the cancel request is fullfilled or job finishes
            await asyncio.futures.wrap_future(futures[-1])
        except (asyncio.CancelledError, spdl.io.AsyncIOFailure):
            pass
        except Exception:
            _LG.exception(
                "An exception was raised while waiting for the task to be cancelled."
            )
        # Propagate the error.
        raise ce


async def _async_gen(func, num_items, *args, **kwargs):
    futures = _common._futurize_generator(func, num_items, *args, **kwargs)
    async for item in _handle_futures(futures):
        yield item


def async_streaming_demux(
    media_type: str,
    src: Union[str, bytes, memoryview],
    timestamps: List[Tuple[float, float]],
    **kwargs,
):
    """Demux the media of given time windows.

    Args:
        media_type: `"audio"` or `"video"`.
        src: Source identifier. If `str` type, it is interpreted as a source location,
            such as local file path or URL. If `bytes` or `memoryview` type, then
            they are interpreted as in-memory data.
        timestamps: List of timestamps.

    Other args:
        io_config (IOConfig): Custom I/O config.

    Returns:
        (AsyncGenerator[Packets]): Audio or video Packets generator.
    """
    func = _common._get_demux_func(media_type, src)
    return _async_gen(func, len(timestamps), src, timestamps, **kwargs)


async def _fetch_one(gen):
    async for packets in gen:
        return packets


def async_demux_media(
    media_type: str,
    src: Union[str, bytes, memoryview],
    timestamp: Optional[Tuple[float, float]] = None,
    **kwargs,
):
    """Demux image or one chunk of audio/video region from the source.

    Args:
        media_type: `"audio"`, `"video"` or `"image"`.
        src: Source identifier. If `str` type, it is interpreted as a source location,
            such as local file path or URL. If `bytes` or `memoryview` type, then
            they are interpreted as in-memory data.
        timestamp (Tuple[float, float]): *Audio/video only* Demux the given time window.
            If omitted, the entire data are demuxed.

    Other args:
        io_config (IOConfig): Custom I/O config.

    Returns:
        (Awaitable[Packets]): Awaitable which returns an audio/video/image Packets object.
    """
    if media_type == "image":
        func = _common._get_demux_func(media_type, src)
        return _async_task(func, src, **kwargs)

    timestamps = [(-float("inf"), float("inf")) if timestamp is None else timestamp]
    return _fetch_one(async_streaming_demux(media_type, src, timestamps, **kwargs))


def async_decode_packets(packets, **kwargs):
    """Decode packets.

    Args:
        packets (Packets): Packets object.

    Other args:
        decoder_config (DecodeConfig):
            *Optional:* Custom decode config.

        filter_desc (str):
            *Optional:* Custom filter applied after decoding.

    Returns:
        (Awaitable[FFmpegFrames]): Awaitable which returns a Frames object.
            The type of the returned object corresponds to the input Packets type.

            - `AudioPackets` -> `AudioFFmpegFrames`

            - `VideoPackets` -> `VideoFFmpegFrames`

            - `ImagePackets` -> `ImageFFmpegFrames`
    """
    func = _common._get_decoding_func(packets)
    if "filter_desc" not in kwargs:
        kwargs["filter_desc"] = _preprocessing.get_filter_desc(packets)
    return _async_task(func, packets, **kwargs)


def async_decode_packets_nvdec(packets, cuda_device_index, **kwargs):
    """Decode packets with NVDEC.

    Args:
        packets (Packet): Packets object.
        cuda_device_index (int): The CUDA device to use for decoding.

    Other args:
        crop_left,crop_top,crop_right,crop_bottom (int):
            *Optional:* Crop the given number of pixels from each side.

        width,height (int): *Optional:* Resize the frame. Resizing is done after
            cropping.

        pix_fmt (str or `None`): *Optional:* Change the format of the pixel.
            Supported value is `"rgba"`. Default: `"rgba"`.

    Returns:
        (Awaitable[NvDecFrames]): Awaitable which returns a Frame object.
            The type of the returned object corresponds to the input Packets type.

            - `VideoPackets` -> `VideoNvDecFrames`

            - `ImagePackets` -> `ImageNvDecFrames`
    """
    func = _common._get_nvdec_decoding_func(packets)
    return _async_task(func, packets, cuda_device_index=cuda_device_index, **kwargs)


def async_decode_media(
    media_type: str,
    src: Union[str, bytes, memoryview],
    **kwargs,
):
    """Perform demuxing and decoding as one background job.

    Args:
        media_type: `"audio"` or `"video"`.
        src: Source identifier. If `str` type, it is interpreted as a source location,
            such as local file path or URL. If `bytes` or `memoryview` type, then
            they are interpreted as in-memory data.
    """
    func = _common._get_decode_from_source_func(media_type, src)
    return _async_task(func, src, **kwargs)


def async_convert_frames(frames, **kwargs):
    """Convert the decoded frames to buffer.

    Args:
        frames (Frames): Frames object.

    Other args:
        cuda_device_index (int):
            *Optional:* When provided, the buffer is moved to CUDA device.

        cuda_stream (int (uintptr_t) ):
            *Optional:* Pointer to a custom CUDA stream. By default, it uses the
            per-thread default stream.

            !!! note

                Host to device buffer transfer is performed in a thread different than
                Python main thread.

                Since the frame data are available only for the duration of the
                background job, the transfer is performed with synchronization.

                It is possible to provide the same stream as the one used in Python's
                main thread, but it might introduce undesired synchronization.

            ??? note "How to retrieve CUDA stream pointer on PyTorch"

                An example to fetch the default stream from PyTorch.

                ```python
                stream = torch.cuda.Stream()
                cuda_stream = stream.cuda_stream
                ```

        cuda_allocator (Callable):
            *Optional:* Custom CUDA memory allcoator, which takes the following arguments
            and return the address of the allocated memory.

            - Size: `int`
            - CUDA device index: `int`
            - CUDA stream address: `int` (`uintptr_t`)

            An example of such function is
            [PyTorch's CUDA caching allocator][torch.cuda.caching_allocator_alloc].

        cuda_deleter (Callable):
            *Optional:* Custom CUDA memory deleter, which takes the address of memory allocated
            by the `cuda_allocator`.

            An example of such function is
            [PyTorch's CUDA caching allocator][torch.cuda.caching_allocator_delete].

    Returns:
        (Awaitable[Buffer]): Awaitable which returns a Buffer object.

            The buffer will be created on the device where the frame data are.

            - `FFmpegAudioFrames` -> `CPUBuffer` or `CUDABuffer`

            - `FFmpegVideoFrames` -> `CPUBuffer` or `CUDABuffer`

            - `FFmpegImageFrames` -> `CPUBuffer` or `CUDABuffer`

            - `List[FFmpegImageFrames]` -> `CPUBuffer` or `CUDABuffer`

            - `NvDecVideoFrames` -> `CUDABuffer`

            - `NvDecImageFrames` -> `CUDABuffer`

            - `List[NvDecImageFrames]` -> `CUDABuffer`
    """
    func = _common._get_conversion_func(frames)
    return _async_task(func, frames, **kwargs)


################################################################################
# High-level APIs
################################################################################


async def async_load_media(
    media_type: str,
    src: Union[str, bytes, memoryview],
    *,
    demux_options: Optional[Dict[str, Any]] = None,
    decode_options: Optional[Dict[str, Any]] = None,
    convert_options: Optional[Dict[str, Any]] = None,
    use_nvdec: bool = False,
):
    """Load the given media into buffer.

    This function combines `async_demux_media`, `async_decode_packets` (or
    `async_decode_packets_nvdec`) and `async_convert_frames` and load media
    into buffer.

    Args:
        media_type: `"audio"`, `"video"` or `"image"`.

        src: Source identifier. If `str` type, it is interpreted as a source location,
            such as local file path or URL. If `bytes` or `memoryview` type, then
            they are interpreted as in-memory data.

        demux_options (Dict[str, Any]):
            *Optional:* Demux options passed to [spdl.io.async_demux_media][].

        decode_options (Dict[str, Any]):
            *Optional:* Decode options passed to [spdl.io.async_decode_packets][] or
            [spdl.io.async_decode_packets_nvdec][].

        convert_options (Dict[str, Any]):
            *Optional:* Convert options passed to [spdl.io.async_convert_frames][].

        use_nvdec:
            *Optional:* If True, use NVDEC to decode the media.

    Returns:
        (Buffer): An object implements buffer protocol.

    ??? note "Example: Load an image frame, resize and convert to RGB HWC format."
        ```python
        >>> buffer = asyncio.run(
        ...     async_load_media(
        ...         "image",
        ...         "sample.jpg",
        ...         decode_options={
        ...             "filter_desc": spdl.io.get_video_filter_desc(
        ...                 width=124,
        ...                 height=96,
        ...                 pix_fmt="rgb24",
        ...             ),
        ...         }
        ...     )
        ... )
        >>> array = spdl.io.to_numpy(buffer)  # NumPy array with shape [96, 124, 3]
        >>>
        ```

    ??? note "Example: Load audio frames from the given windows, resample and trim the samples."
        ```python
        >>> coro = async_load_media(
        ...     "audio",
        ...     "sample.wav",
        ...     demux_options={
        ...         "timestamp": (0, 3),
        ...     },
        ...     decode_options={
        ...         "filter_desc": spdl.io.preprocessing.get_audio_filter_desc(
        ...             timestamp=(0, 3),
        ...             sample_rate=16000,
        ...             num_frames=24000,
        ...         ),
        ...     },
        ... )
        >>> buffer = asyncio.run(coro)
        >>> array = spdl.io.to_numpy(buffer)  # NumPy array with shape [24000, 2]
        >>>
        ```

    ??? note "Example: Load video frames from the given window using NVDEC and resize."
        ```python
        >>> import torch
        >>> torch.cuda.set_device(7)
        >>> coro = async_load_media(
        ...     "video",
        ...     "sample.mp4",
        ...     demux_options={
        ...         "timestamp": (10, 14),
        ...     },
        ...     decode_options={
        ...         "cuda_device_index": 7,
        ...         "width": 124,
        ...         "height": 96,
        ...         "pix_fmt": "rgba",
        ...     },
        ...     use_nvdec=True,
        ... )
        >>> buffer = asyncio.run(coro)
        >>> tensor = spdl.io.to_torch(buffer)  # PyTorch CUDA tensor
        >>>
        ```
    """
    demux_options = demux_options or {}
    decode_options = decode_options or {}
    convert_options = convert_options or {}
    packets = await async_demux_media(media_type, src, **demux_options)
    if use_nvdec:
        frames = await async_decode_packets_nvdec(packets, **decode_options)
    else:
        frames = await async_decode_packets(packets, **decode_options)
    buffer = await async_convert_frames(frames, **convert_options)
    return buffer


async def _decode(media_type, src, demux_options, decode_options):
    packets = await async_demux_media(media_type, src, **demux_options)
    return await async_decode_packets(packets, **decode_options)


def _get_err_msg(src, err):
    match type(src):
        case builtins.bytes:
            src_ = f"bytes object at {id(src)}"
        case builtins.memoryview:
            src_ = f"memoryview object at {id(src)}"
        case _:
            src_ = f"'{src}'"
    return f"Failed to decode an image from {src_}: {err}."


async def async_batch_load_image(
    srcs: List[Union[str, bytes, memoryview]],
    *,
    width: int | None,
    height: int | None,
    pix_fmt: str | None = "rgb24",
    demux_options: Optional[Dict[str, Any]] = None,
    decode_options: Optional[Dict[str, Any]] = None,
    convert_options: Optional[Dict[str, Any]] = None,
    strict: bool = True,
):
    """Batch load images.

    Args:
        srcs: List of source identifiers.

        width: *Optional:* Resize the frame.

        height: *Optional:* Resize the frame.

        pix_fmt:
            *Optional:* Change the format of the pixel.

        demux_options (Dict[str, Any]):
            *Optional:* Demux options passed to [spdl.io.async_demux_media][].

        decode_options (Dict[str, Any]):
            *Optional:* Decode options passed to [spdl.io.async_decode_packets][].

        convert_options (Dict[str, Any]):
            *Optional:* Convert options passed to [spdl.io.async_convert_frames][].

        strict:
            *Optional:* If True, raise an error if any of the images failed to load.

    Returns:
        (Buffer): An object implements buffer protocol.
            To be passed to casting functions like [spdl.io.to_numpy][],
            [spdl.io.to_torch][] or [spdl.io.to_numba][].

    ??? note "Example"
        ```python
        >>> srcs = [
        ...     "sample1.jpg",
        ...     "sample2.png",
        ... ]
        >>> coro = async_batch_load_image(
        ...     srcs,
        ...     width=124,
        ...     height=96,
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

    decoding = []
    for src in srcs:
        coro = async_decode_media("image", src, **demux_options, **decode_options)
        decoding.append(asyncio.create_task(coro))

    await asyncio.wait(decoding)

    frames = []
    for src, task in zip(srcs, decoding):
        if err := task.exception():
            _LG.error(_get_err_msg(src, err))
            continue
        frames.append(task.result())

    if len(frames) != len(srcs) and strict:
        raise RuntimeError("Failed to load some images.")

    if not frames:
        raise RuntimeError("Failed to load all the images.")

    return await async_convert_frames(frames, **convert_options)
