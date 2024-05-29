import asyncio
import builtins
import logging
from concurrent.futures import Future

# pyre-strict
from typing import Any, AsyncIterator, Dict, List, overload, Sequence, Tuple, TypeVar

import spdl.io
from spdl.io import (
    AudioFrames,
    AudioPackets,
    CPUBuffer,
    CUDABuffer,
    Frames,
    ImageFrames,
    ImagePackets,
    VideoFrames,
    VideoPackets,
)
from spdl.lib import _libspdl

from . import _common, _preprocessing


__all__ = [
    "_async_sleep",
    "_async_sleep_multi",
    "async_convert_frames",
    "async_decode_packets",
    "async_decode_packets_nvdec",
    "async_decode_image_nvjpeg",
    "async_demux_media",
    "async_demux_audio",
    "async_demux_video",
    "async_demux_image",
    "async_streaming_demux",
    "async_streaming_demux_audio",
    "async_streaming_demux_video",
    "async_streaming_decode",
    "async_sample_decode_video",
    "async_load_media",
    "async_load_audio",
    "async_load_video",
    "async_load_image",
    "async_load_image_batch",
    "async_load_image_batch_nvdec",
]

_LG = logging.getLogger(__name__)

T = TypeVar("T")


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


async def _handle_future(future: Future[T]) -> T:
    try:
        return await asyncio.futures.wrap_future(future)
    # Handle the case where unexpected/external thing happens
    except asyncio.CancelledError as e:
        future.__spdl_future.cancel()  # pyre-ignore[16]
        try:
            # Wait till the cancel request is fullfilled or job finishes
            await asyncio.futures.wrap_future(future)
        except (asyncio.CancelledError, spdl.io.SPDLBackgroundTaskFailure):
            pass
        except Exception:
            _LG.exception(
                "An exception was raised while waiting for the task to be cancelled."
            )
        # Propagate the error.
        raise e


def _async_task(func, *args, **kwargs):
    future = _common._futurize_task(func, *args, **kwargs)
    return _handle_future(future)


async def _handle_futures(futures):
    for fut in futures:
        yield await _handle_future(fut)


def _async_gen(func, num_items, *args, **kwargs):
    futures = _common._futurize_generator(func, num_items, *args, **kwargs)
    return _handle_futures(futures)


async def async_streaming_demux(
    media_type: str,
    src: str | bytes,
    timestamps: List[Tuple[float, float]],
    **kwargs,
):
    """Demux the media of given time windows.

    Args:
        media_type: `"audio"` or `"video"`.
        src: Source identifier. If `str` type, it is interpreted as a source location,
            such as local file path or URL. If `bytes` type, then
            they are interpreted as in-memory data.
        timestamps: List of timestamps.

    Other args:
        demux_config (DemuxConfig): Custom I/O config.

    Returns:
        [AudioPackets][spdl.io.AudioPackets] or [VideoPackets][spdl.io.VideoPackets] generator.
    """
    if not timestamps:
        raise ValueError("`timestamps` cannot be an empty list.")
    func = _common._get_stream_demux_func(media_type, src)
    async for packets in _async_gen(func, len(timestamps), src, timestamps, **kwargs):
        yield packets


async def async_streaming_demux_audio(
    src: str | bytes,
    timestamps: List[Tuple[float, float]],
    **kwargs,
) -> AsyncIterator[AudioPackets]:
    async for packets in async_streaming_demux("audio", src, timestamps, **kwargs):
        yield packets


async def async_streaming_demux_video(
    src: str | bytes,
    timestamps: List[Tuple[float, float]],
    **kwargs,
) -> AsyncIterator[VideoPackets]:
    async for packets in async_streaming_demux("video", src, timestamps, **kwargs):
        yield packets


async def async_demux_media(
    media_type: str,
    src: str | bytes,
    timestamp: Tuple[float, float] | None = None,
    **kwargs,
):
    """Demux image or one chunk of audio/video region from the source.

    Args:
        media_type: `"audio"`, `"video"` or `"image"`.
        src: Source identifier. If `str` type, it is interpreted as a source location,
            such as local file path or URL. If `bytes` type, then
            they are interpreted as in-memory data.
        timestamp (Tuple[float, float]): *Audio/video only* Demux the given time window.
            If omitted, the entire data are demuxed.

    Other args:
        demux_config (DemuxConfig): Custom I/O config.

    Returns:
        AudioPackets/VideoPackets/ImagePackets object.
    """
    func = _common._get_demux_func(media_type, src)
    if media_type != "image" and timestamp is not None:
        kwargs["timestamp"] = timestamp
    return await _async_task(func, src, **kwargs)


async def async_demux_audio(
    src: str | bytes, timestamp: Tuple[float, float] | None = None, **kwargs
) -> AudioPackets:
    return await async_demux_media("audio", src, timestamp=timestamp, **kwargs)


async def async_demux_video(
    src: str | bytes, timestamp: Tuple[float, float] | None = None, **kwargs
) -> VideoPackets:
    return await async_demux_media("video", src, timestamp=timestamp, **kwargs)


async def async_demux_image(src: str | bytes, **kwargs) -> ImagePackets:
    return await async_demux_media("image", src, **kwargs)


@overload
async def async_decode_packets(packets: AudioPackets, **kwargs) -> AudioFrames: ...


@overload
async def async_decode_packets(packets: VideoPackets, **kwargs) -> VideoFrames: ...


@overload
async def async_decode_packets(packets: ImagePackets, **kwargs) -> ImageFrames: ...


async def async_decode_packets(packets, **kwargs):
    """Decode packets.

    Args:
        packets (AudioPackets | VideoPackets | ImagePackets): Packets object.

    Other args:
        decoder_config (DecodeConfig):
            *Optional:* Custom decode config.

        filter_desc (str):
            *Optional:* Custom filter applied after decoding.

    Returns:
        (AudioFrames | VideoFrames | ImageFrames): A Frames object.
            The media type of the returned object corresponds to the input Packets type.
    """
    func = _common._get_decoding_func(packets)
    if "filter_desc" not in kwargs:
        kwargs["filter_desc"] = _preprocessing.get_filter_desc(packets)
    return await _async_task(func, packets, **kwargs)


async def async_streaming_decode(
    packets: VideoPackets,
    num_frames: int,
    **kwargs,
) -> AsyncIterator[VideoFrames]:
    match t := type(packets):
        case _libspdl.VideoPackets:
            constructor = _libspdl.async_streaming_video_decoder
            decode_fn = _libspdl.async_decode_video_frames
        case _:
            raise RuntimeError(f"{t} is not supported.")

    executor = kwargs.get("executor")

    decoder = await _async_task(constructor, packets, **kwargs)
    while True:
        frames = await _async_task(decode_fn, decoder, num_frames, executor=executor)
        if frames is None:
            return
        yield frames


async def async_decode_packets_nvdec(
    packets: ImagePackets | VideoPackets | List[ImagePackets],
    cuda_device_index: int,
    **kwargs,
) -> CUDABuffer:
    """Decode packets with NVDEC.

    Unlike FFmpeg-based decoding, NVDEC returns GPU buffer directly.

    ``` mermaid
    graph LR
      Source -->|Demux| Packets;
      Packets -->|Decode| Buffer;
      Buffer -->|Cast| Array[Array / Tensor];
    ```

    Args:
        packets: Packets object.
            Either `VideoPackets`, `ImagePackets` or a list of `ImagePackets`.

        cuda_device_index (int): The CUDA device to use for decoding.

    Other args:
        crop_left,crop_top,crop_right,crop_bottom (int):
            *Optional:* Crop the given number of pixels from each side.

        width,height (int): *Optional:* Resize the frame. Resizing is done after
            cropping.

        pix_fmt (str or `None`): *Optional:* Change the format of the pixel.
            Supported value is `"rgba"`. Default: `"rgba"`.

    Returns:
        A CUDABuffer object.
    """
    func = _common._get_nvdec_decoding_func(packets)
    kwargs["cuda_device_index"] = cuda_device_index
    return await _async_task(func, packets, **kwargs)


async def async_decode_media(
    media_type: str,
    src: str | bytes,
    **kwargs,
) -> Frames:
    """Perform demuxing and decoding as one background job.

    Args:
        media_type: `"audio"` or `"video"`.
        src: Source identifier. If `str` type, it is interpreted as a source location,
            such as local file path or URL. If `bytes` type, then
            they are interpreted as in-memory data.

    Returns:
        Awaitable which returns Frames object.
    """
    func = _common._get_decode_from_source_func(media_type, src)
    return await _async_task(func, src, **kwargs)


async def async_convert_frames(
    frames: Frames | Sequence[Frames],
    cuda_device_index: int | None = None,
    **kwargs,
) -> CPUBuffer | CUDABuffer:
    """Convert the decoded frames to buffer.

    Args:
        frames: Frames objects.

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

        cuda_allocator (Tuple[Callable[[int, int, int], int], Callable[[int], None]]):
            *Optional:* A pair of custom CUDA memory allcoator and deleter functions.

            The allocator function, takes the following arguments, and
            return the address of the allocated memory.

            - Size: `int`
            - CUDA device index: `int`
            - CUDA stream address: `int` (`uintptr_t`)

            An example of such function is
            [PyTorch's CUDA caching allocator][torch.cuda.caching_allocator_alloc].

            The deleter takes the address of memory allocated
            by the allocator and free the memory.

            An example of such function is
            [PyTorch's CUDA caching allocator][torch.cuda.caching_allocator_delete].

    Returns:
        A Buffer object.
    """
    func = _common._get_conversion_func(frames, cuda_device_index)
    if cuda_device_index is not None:
        kwargs["cuda_device_index"] = cuda_device_index
    return await _async_task(func, frames, **kwargs)


async def async_decode_image_nvjpeg(
    data: bytes, cuda_device_index: int, **kwargs
) -> CUDABuffer:
    """Decode (JPEG) image with nvJPEG.

    Unlike FFmpeg-based decoding, nvJPEG returns GPU buffer directly.

    ``` mermaid
    graph LR
      Source -->|Decode| Buffer;
      Buffer -->|Cast| Array[Array / Tensor];
    ```

    Args:
        data: JPEG image data in bytes.
        cuda_device_index: CUDA device index.

    Other args:
        pix_fmt (str): *Optional* Output pixel format.
            Supported values are `"RGB"` or `"BGR"`.

        cuda_allocator (Callable):
            See [async_convert_frames][spdl.io.async_convert_frames].

    Returns:
        A CUDABuffer object. Shape is [C==3, H, W].
    """
    func = _libspdl.async_decode_image_nvjpeg
    return await _async_task(func, data, cuda_device_index, **kwargs)


async def _decode_partial(
    packets: VideoPackets, indices: List[int], **kwargs
) -> List[ImageFrames]:
    """Decode packets but return early when requested frames are decoded."""
    num_frames = max(indices) + 1
    async for frames in async_streaming_decode(packets, num_frames, **kwargs):
        return frames[indices]


async def async_sample_decode_video(
    packets: VideoPackets, indices: List[int], **kwargs
) -> List[VideoFrames]:
    """Selectively decode frames from the given video packets.

    This function retuns the result similar to decoding all the packets then
    selecting the frames. i.e.

    ```python
    frames = await spdl.io.async_decode_packets(packets)
    frames = frames[indices]
    ```

    But this function decodes the minimum number of packets, so it is faster
    when sampling a few frame sparsely from long video.

    Args:
        packets (VideoPackets): VideoPackets object to sample from.

        indices (List[int]): List of indices for sampling.
            The list must be sorted in ascending order and without duplicates.

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

    i = start = 0
    coros = []
    for split in packets._split_at_keyframes():
        end = start + len(split)
        idx = []
        while i < len(indices) and start <= indices[i] < end:
            idx.append(indices[i] - start)
            i += 1
        if idx:
            coros.append(_decode_partial(split, idx, **kwargs))
        start = end

    tasks = [asyncio.create_task(coro) for coro in coros]
    await asyncio.wait(tasks)
    ret = []
    for task in tasks:
        try:
            ret.extend(task.result())
        except Exception as e:
            _LG.error(f"Failed to decode {task.get_name()}. Reason: {e}")
    return ret


################################################################################
# High-level APIs
################################################################################


async def async_load_media(
    media_type: str,
    src: str | bytes,
    *,
    demux_options: Dict[str, Any] | None = None,
    decode_options: Dict[str, Any] | None = None,
    convert_options: Dict[str, Any] | None = None,
    use_nvdec: bool = False,
) -> CPUBuffer | CUDABuffer:
    """Load the given media into buffer.

    This function combines `async_demux_media`, `async_decode_packets` (or
    `async_decode_packets_nvdec`) and `async_convert_frames` and load media
    into buffer.

    Args:
        media_type: `"audio"`, `"video"` or `"image"`.

        src: Source identifier. If `str` type, it is interpreted as a source location,
            such as local file path or URL. If `bytes` type, then
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
        Resulting buffer object.

    ??? note "Example: Load an image frame, resize and convert to RGB HWC format."
        ```python
        >>> buffer = asyncio.run(
        ...     async_load_media(
        ...         "image",
        ...         "sample.jpg",
        ...         decode_options={
        ...             "filter_desc": spdl.io.get_video_filter_desc(
        ...                 scale_width=124,
        ...                 scale_height=96,
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
    if use_nvdec and convert_options is not None:
        raise ValueError("NVDEC cannot be used with `convert_options`.")
    convert_options = convert_options or {}
    packets = await async_demux_media(media_type, src, **demux_options)
    if use_nvdec:
        return await async_decode_packets_nvdec(packets, **decode_options)

    frames = await async_decode_packets(packets, **decode_options)
    return await async_convert_frames(frames, **convert_options)


async def async_load_audio(*args, **kwargs):
    return await async_load_media("audio", *args, **kwargs)


async def async_load_video(*args, **kwargs):
    return await async_load_media("video", *args, **kwargs)


async def async_load_image(*args, _use_nvdec = None, **kwargs):
    if _use_nvdec is not None:
        kwargs["use_nvdec"] = _use_nvdec
    return await async_load_media("image", *args, **kwargs)


async def _decode(media_type, src, demux_options, decode_options):
    packets = await async_demux_media(media_type, src, **demux_options)
    return await async_decode_packets(packets, **decode_options)


def _get_err_msg(src, err):
    match type(src):
        case builtins.bytes:
            src_ = f"bytes object at {id(src)}"
        case _:
            src_ = f"'{src}'"
    return f"Failed to decode an image from {src_}: {err}."


async def async_load_image_batch(
    srcs: List[str | bytes],
    *,
    width: int | None,
    height: int | None,
    pix_fmt: str | None = "rgb24",
    demux_options: Dict[str, Any] | None = None,
    decode_options: Dict[str, Any] | None = None,
    convert_options: Dict[str, Any] | None = None,
    strict: bool = True,
) -> CPUBuffer | CUDABuffer:
    """Batch load images.

    Args:
        srcs: List of source identifiers.

        width: *Optional:* Resize the frame.

        height: *Optional:* Resize the frame.

        pix_fmt:
            *Optional:* Change the format of the pixel.

        demux_options (Dict[str, Any]):
            *Optional:* Demux options passed to [spdl.io.async_demux_image][].

        decode_options (Dict[str, Any]):
            *Optional:* Decode options passed to [spdl.io.async_decode_packets][].

        convert_options (Dict[str, Any]):
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

    decoding = []
    for src in srcs:
        coro = async_decode_media("image", src, **demux_options, **decode_options)
        decoding.append(asyncio.create_task(coro))

    await asyncio.wait(decoding)

    frames: List[ImageFrames] = []
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


async def async_load_image_batch_nvdec(
    srcs: List[str | bytes],
    *,
    cuda_device_index: int,
    width: int | None,
    height: int | None,
    pix_fmt: str | None = "rgba",
    demux_options: Dict[str, Any] | None = None,
    decode_options: Dict[str, Any] | None = None,
    strict: bool = True,
) -> CUDABuffer:
    """Batch load images.

    Args:
        srcs: List of source identifiers.

        cuda_device_index: The CUDA device to use for decoding images.

        width: *Optional:* Resize the frame.

        height: *Optional:* Resize the frame.

        pix_fmt:
            *Optional:* Change the format of the pixel.

        demux_options (Dict[str, Any]):
            *Optional:* Demux options passed to [spdl.io.async_demux_media][].

        decode_options (Dict[str, Any]):
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
        ```

    """
    if not srcs:
        raise ValueError("`srcs` must not be empty.")

    demux_options = demux_options or {}
    decode_options = decode_options or {}
    width = -1 if width is None else width
    height = -1 if height is None else height

    demuxing = []
    for src in srcs:
        coro = async_demux_media("image", src, **demux_options)
        demuxing.append(asyncio.create_task(coro))

    await asyncio.wait(demuxing)

    packets = []
    for src, task in zip(srcs, demuxing):
        if err := task.exception():
            _LG.error(_get_err_msg(src, err))
            continue
        packets.append(task.result())

    if len(packets) != len(srcs) and strict:
        raise RuntimeError("Failed to demux some images.")

    return await async_decode_packets_nvdec(
        packets,
        cuda_device_index=cuda_device_index,
        width=width,
        height=height,
        pix_fmt=pix_fmt,
        strict=strict,
        **decode_options,
    )
