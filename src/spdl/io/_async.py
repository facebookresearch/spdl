import asyncio
import concurrent.futures
from typing import List, Optional, Tuple, Union

from spdl.lib import _libspdl

from . import _common


__all__ = [
    "_async_sleep",
    "async_convert_frames_cpu",
    "async_convert_frames",
    "async_decode_packets",
    "async_decode_packets_nvdec",
    "async_streaming_demux",
    "async_demux",
]


def _async_sleep(time: int):
    return _async_task(_libspdl.async_sleep, time)


# Exception class used to signal the failure of C++ op to Python.
# Not exposed to user code.
class _AsyncOpFailed(RuntimeError):
    pass


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


async def _async_task(func, *args, **kwargs):
    future = concurrent.futures.Future()
    assert future.set_running_or_notify_cancel()

    def nofify_exception(msg: str):
        future.set_exception(_AsyncOpFailed(msg))

    sf = func(future.set_result, nofify_exception, *args, **kwargs)

    try:
        return await asyncio.futures.wrap_future(future)
    # Handle the case where the async op failed
    except _AsyncOpFailed:
        pass
    # Handle the case where unexpected/external thing happens
    except (asyncio.CancelledError, Exception) as e:
        sf.cancel()
        # Wait till the cancellation is completed
        try:
            await asyncio.futures.wrap_future(future)
        except _AsyncOpFailed:
            pass
        # Propagate the error.
        raise e

    await asyncio.sleep(_EXCEPTION_BACKOFF)
    sf.rethrow()


async def _async_gen(func, num_items, *args, **kwargs):
    futures = [concurrent.futures.Future()]

    def set_result(val):
        futures[-1].set_result(val)

        nonlocal num_items
        if num_items := num_items - 1:
            future = concurrent.futures.Future()
            future.set_running_or_notify_cancel()
            futures.append(future)

    def notify_exception(msg: str):
        futures[-1].set_exception(_AsyncOpFailed(msg))

    futures[0].set_running_or_notify_cancel()
    sf = func(set_result, notify_exception, *args, **kwargs)
    while futures:
        try:
            val = await asyncio.futures.wrap_future(futures[0])
        # Handle the case where the async op failed
        except _AsyncOpFailed:
            break
        # Handle the case where unexpected/external thing happens
        except (asyncio.CancelledError, Exception) as e:
            sf.cancel()
            # Wait till the cancellation is completed
            try:
                await asyncio.futures.wrap_future(futures[0])
            except _AsyncOpFailed:
                pass
            # Propagate the error.
            raise e
        else:
            if val is None:
                return
            yield val
            futures.pop(0)
        finally:
            await asyncio.sleep(0)

    await asyncio.sleep(_EXCEPTION_BACKOFF)
    sf.rethrow()


def async_streaming_demux(
    media_type: str,
    src: Union[str, bytes],
    timestamps: List[Tuple[float, float]],
    **kwargs,
):
    """Demux the given time windows from the source.

    Args:
        media_type: ``"audio"`` or ``"video"``.
        src (str or bytes): Source identifier, such as path or URL.
        timestamps: List of timestamps.

    Other args:
        format (str): *Optional:* The format detection. Optional.
            Can be used to demux headerless format.
        format_options (Dict[str, str]): *Optional:* Format options.
        buffer_size (int, optional): *Optional:* Change the internal buffer size used to process
            the data at a time.
        adaptor (SourceAdaptor, optional): *Optional:* Adaptor to apply to the `src`.
        executor (ThreadPoolExecutor, optional):
            *Optional:* Custom executor to in which the task is performed.
            By default the task is peformed in demuxer thread pool.

    Returns:
        (AsyncGenerator[Packets]): Audio or video Packets generator.
    """
    func = _common._get_demux_func(media_type, src)
    return _async_gen(func, len(timestamps), src, timestamps, **kwargs)


async def _fetch_one(gen):
    async for packets in gen:
        return packets


def async_demux(
    media_type: str,
    src: Union[str, bytes],
    timestamp: Optional[Tuple[float, float]] = None,
    **kwargs,
):
    """Demux image or one chunk of audio/video region from the source.

    Args:
        media_type: ``"audio"``, ``"video"`` or ``"image"``.
        src (str or bytes): Source identifier, such as path or URL.
        timestamp (Tuple[float, float]): *Audio/video only* Demux the given time window.

    Other args:
        format (str): *Optional:* Overwrite the format detection.
            Can be used to demux headerless format.
        format_options (Dict[str, str]): *Optional:* Format options.
        buffer_size (int): *Optional:* Buffer size in bytes.
        adaptor (SourceAdaptor): *Optional:* Adaptor to apply to the `src`.
        executor (ThreadPoolExecutor): *Optional:* Executor to perform the job.
            By default the job is peformed in demuxer thread pool.

    Returns:
        (Awaitable[ImagePackets]): Awaitable which returns an ImagePackets object.
    """
    if media_type == "image":
        func = _common._get_demux_func(media_type, src)
        return _async_task(func, src, **kwargs)

    if timestamp is None:
        timestamps = [(0.0, float("inf"))]
    else:
        timestamps = [timestamp]
    return _fetch_one(async_streaming_demux(media_type, src, timestamps, **kwargs))


def async_decode_packets(packets, **kwargs):
    """Decode the packets to frames.

    Args:
        packets (Packets): Packets object.

    Other args:
        decoder (str): *Optional:* Overwrite the decoder.
        decoder_options (Dict[str, str]): *Optional:* Decoder options.
        sample_rate (int): *Optional, audio only:* Change the sample rate.
        num_channels (int): *Optional, audio only:* Change the number of channels.
        sample_fmt (str): *Optional, audio only:* Change the format of sample.
            Valid values are (``"u8"``, ``"u8p"``, ``s16``, ``s16p``,
            ``"s32"``, ``"s32p"``, ``"flt"``, ``"fltp"``, ``"s64"``,
            ``"s64p"``, ``"dbl"``, ``"dblp"``).
        frame_rate (int): *Optional, video only:* Change the frame rate.
        width,height (int): *Optional, video/image only:* Change the resolution of the frame.
        pix_fmt (str): *Optional, video/image only:* Change the pixel format.
            Valid values are ().
        num_frames (int): *Optional, audio/video only:* Fix the number of output frames by
            dropping the exceeding frames or padding.
            For audio, silence is added. For video, by default the last frame is
            repeated.
        pad_mode (str): *Optional, video only:* Change the padding frames to the given color.
        executor (ThreadPoolExecutor): *Optional:* Executor to perform the job.
            By default the job is peformed in decode thread pool.

    Returns:
        (Awaitable[FFmpegFrames]): Awaitable which returns a Frames object.
            The type of the returned object corresponds to the input Packets type.

            - ``AudioPackets`` -> ``AudioFFmpegFrames``

            - ``VideoPackets`` -> ``VideoFFmpegFrames``

            - ``ImagePackets`` -> ``ImageFFmpegFrames``
    """
    func = _common._get_decoding_func(packets)
    return _async_task(func, packets, **kwargs)


def async_decode_packets_nvdec(packets, cuda_device_index, **kwargs):
    """Decode the packets to frames with NVDEC.

    Args:
        packets (Packet): Packets object.
        cuda_device_index (int): The CUDA device to use for decoding.

    Other args:
        crop_left,crop_top,crop_right,crop_bottom (int):
            *Optional:* Crop the given number of pixels from each side.
        width,height (int): *Optional:* Resize the frame. Resizing is done after
            cropping.
        pix_fmt (str or ``None``): *Optional:* Change the format of the pixel.
            Supported value is ``"rgba"``. Default: ``"rgba"``.
        executor (ThreadPoolExecutor): *Optional:* Executor to perform the job.
            By default the job is peformed in decode thread pool.

    Returns:
        (Awaitable[NvDecFrames]): Awaitable which returns a Frame object.
            The type of the returned object corresponds to the input Packets type.

            - ``VideoPackets`` -> ``VideoNvDecFrames``

            - ``ImagePackets`` -> ``ImageNvDecFrames``
    """
    func = _common._get_nvdec_decoding_func(packets)
    return _async_task(func, packets, cuda_device_index=cuda_device_index, **kwargs)


def async_convert_frames_cpu(frames, executor=None):
    """Convert the frames to buffer.

    Args:
        frames (CPUFrames): Frames object.
            If the frame data are not CPU, then the conversion will fail.

    Other args:
        executor (ThreadPoolExecutor):
            *Optional:* Executor to run the conversion.
            By default, the conversion is performed on
            demuxer thread pool with higher priority than demuxing.

    Returns:
        (Awaitable[Buffer]): Awaitable which returns a Buffer object.
            The type of the returned object corresponds to the input Packets type.

            - ``FFmpegAudioFrames`` -> ``CPUBuffer``

            - ``FFmpegVideoFrames`` -> ``CPUBuffer``

            - ``FFmpegImageFrames`` -> ``CPUBuffer``

            - ``List[FFmpegImageFrames]`` -> ``CPUBuffer``


    """
    func = _common._get_cpu_conversion_func(frames)
    return _async_task(func, frames, index=None, executor=executor)


def async_convert_frames(frames, executor=None):
    """Convert the frames to buffer.

    Args:
        frames (Frames): Frames object.

    Other args:
        executor (ThreadPoolExecutor):
            *Optional:* Executor to run the conversion. By default, the conversion is performed on
            demuxer thread pool with higher priority than demuxing.

    Returns:
        (Awaitable[Buffer]): Awaitable which returns a Buffer object.

            The buffer will be created on the device where the frame data are.

            - ``FFmpegAudioFrames`` -> ``CPUBuffer``

            - ``FFmpegVideoFrames`` -> ``CPUBuffer`` or ``CUDABuffer``

            - ``FFmpegImageFrames`` -> ``CPUBuffer`` or ``CUDABuffer``

            - ``NvDecVideoFrames`` -> ``CUDABuffer``

            - ``NvDecImageFrames`` -> ``CUDABuffer``

            - ``List[FFmpegImageFrames]`` -> ``CPUBuffer``

            - ``List[NvDecImageFrames]`` -> ``CUDABuffer``
    """
    func = _common._get_conversion_name(frames)
    return _async_task(func, frames, index=None, executor=executor)
