import asyncio
import concurrent.futures
import functools
from typing import Any, List, Tuple

from spdl.lib import _libspdl

__all__ = [
    "_async_sleep",
    # TODO: Merge async_apply_bsf with async_decode_nvdec(video)
    "async_apply_bsf",
    "async_convert_frames_cpu",
    "async_convert_frames",
    "async_decode_packets",
    "async_decode_packets_nvdec",
    "async_demux_audio",
    "async_demux_video",
    "async_demux_image",
]

_debug_task = [
    "async_sleep",
]


def _async_sleep(time: int):
    return _async_task(_libspdl.async_sleep, time)


# Exception class used to signal the failure of C++ op to Python.
# Not exposed to user code.
class _AsyncOpFailed(Exception):
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

    def nofify_exception():
        future.set_exception(_AsyncOpFailed())

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


async def _async_gen(func, *args, **kwargs):
    future = concurrent.futures.Future()
    assert future.set_running_or_notify_cancel()

    futures = [future]

    def set_result(val):
        futures[-1].set_result(val)

        if val is not None:
            future = concurrent.futures.Future()
            assert future.set_running_or_notify_cancel()
            futures.append(future)

    def notify_exception():
        futures[-1].set_exception(_AsyncOpFailed())

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


def async_demux_audio(src, timestamps: List[Tuple[float, float]], **kwargs):
    """Demux the audio stream.

    Args:
        src: Source identifier, such as path or URL.
        timestamps (List[Tuple[float, float]]): List of timestamps.
        adaptor (Optional[spdl.SourceAdaptor]): Adaptor to apply to the `src`.
        format (str): Overwrite the format detection.
            Can be used to demux headerless format.
        format_options (Dict[str, str]): Format options.
        buffer_size (int): Buffer size in bytes.
        executor (Optional[spdl.ThreadPoolExecutor]):
            Executor to run the conversion. By default, the conversion is performed on
            demuxer thread pool.

    Returns:
        AsyncGenerator[AudioPackets]: Generator of AudioPackets.
    """
    func = getattr(
        _libspdl,
        "async_demux_audio_bytes" if isinstance(src, bytes) else "async_demux_audio",
    )
    return _async_gen(func, src, timestamps, **kwargs)


def async_demux_video(src, timestamps: List[Tuple[float, float]], **kwargs):
    """Demux the video stream.

    Args:
        src: Source identifier, such as path or URL.
        timestamps (List[Tuple[float, float]]): List of timestamps.
        adaptor (Optional[spdl.SourceAdaptor]): Adaptor to apply to the `src`.
        format (str): Overwrite the format detection.
            Can be used to demux headerless format.
        format_options (Dict[str, str]): Format options.
        buffer_size (int): Buffer size in bytes.
        executor (Optional[spdl.ThreadPoolExecutor]):
            Executor to run the conversion. By default, the conversion is performed on
            demuxer thread pool.

    Returns:
        AsyncGenerator[VideoPackets]: Generator of VideoPackets.
    """
    func = getattr(
        _libspdl,
        "async_demux_video_bytes" if isinstance(src, bytes) else "async_demux_video",
    )
    return _async_gen(func, src, timestamps, **kwargs)


def async_demux_image(src, *args, **kwargs):
    """Demux the image stream.

    Args:
        src: Source identifier, such as path or URL.
        adaptor (Optional[spdl.SourceAdaptor]): Adaptor to apply to the `src`.
        format (str): Overwrite the format detection.
            Can be used to demux headerless format.
        format_options (Dict[str, str]): Format options.
        buffer_size (int): Buffer size in bytes.
        executor (Optional[spdl.ThreadPoolExecutor]):
            Executor to run the conversion. By default, the conversion is performed on
            demuxer thread pool.

    Returns:
        Awaitable: Awaitable which returns an ImagePackets object.
    """
    func = getattr(
        _libspdl,
        "async_demux_image_bytes" if isinstance(src, bytes) else "async_demux_image",
    )
    return _async_task(func, src, **kwargs)


# TODO: Merge this with async_decode_nvdec
def async_apply_bsf(packets, *args, **kwargs):
    """Apply the bitstream filters.

    Args:
        packets (Packet): Packets object.
        executor (Optional[spdl.ThreadPoolExecutor]):
            Executor to run the conversion. By default, the conversion is performed on
            demuxer thread pool.

    Returns:
        Awaitable: Awaitable which returns the filtered Packets object.
    """
    return _async_task(_libspdl.async_apply_bsf, packets, *args, **kwargs)


def _get_decoding_name(packets):
    match t := type(packets):
        case _libspdl.AudioPackets:
            return "async_decode_audio"
        case _libspdl.VideoPackets:
            return "async_decode_video"
        case _libspdl.ImagePackets:
            return "async_decode_image"
        case _:
            raise TypeError(f"Unexpected type: {t}.")


def async_decode_packets(packets, *args, **kwargs):
    """Decode the packets to frames.

    Args:
        packets (Packet): Packets object.

    Returns:
        Awaitable: Awaitable which returns a Frame object.
    """
    func = getattr(_libspdl, _get_decoding_name(packets))
    return _async_task(func, packets, *args, **kwargs)


def _get_nvdec_decoding_name(packets):
    match t := type(packets):
        case _libspdl.VideoPackets:
            return "async_decode_video_nvdec"
        case _libspdl.ImagePackets:
            return "async_decode_image_nvdec"
        case _:
            raise TypeError(f"Unexpected type: {t}.")


def async_decode_packets_nvdec(packets, *args, **kwargs):
    """Decode the packets to frames with NVDEC.

    Args:
        packets (Packet): Packets object.

    Returns:
        Awaitable: Awaitable which returns a Frame object.
    """
    func = getattr(_libspdl, _get_nvdec_decoding_name(packets))
    return _async_task(func, packets, *args, **kwargs)


def _get_cpu_conversion_name(frames):
    match t := type(frames):
        case _libspdl.FFmpegAudioFrames:
            return "async_convert_audio_cpu"
        case _libspdl.FFmpegVideoFrames:
            return "async_convert_video_cpu"
        case _libspdl.FFmpegImageFrames:
            return "async_convert_image_cpu"
        case _:
            if isinstance(frames, list):
                if all(isinstance(f, _libspdl.FFmpegImageFrames) for f in frames):
                    return "async_convert_batch_image_cpu"
            raise TypeError(f"Unexpected type: {t}.")


def async_convert_frames_cpu(frames, executor=None):
    """Convert the frames to buffer.

    Args:
        frames : Frames object. The following types are supported.
            - ``FFmpegAudioFrames``
            - ``FFmpegVideoFrames``
            - ``FFmpegImageFrames``
            - ``List[FFmpegImageFrames]``

            If the frame data are not CPU, then the conversion will fail.

        executor (Optional[spdl.ThreadPoolExecutor]):
            Executor to run the conversion. By default, the conversion is performed on
            demuxer thread pool with higher priority than demuxing.

    Returns:
        Awaitable: Awaitable which returns a Buffer object.
    """
    func = getattr(_libspdl, _get_cpu_conversion_name(frames))
    return _async_task(func, frames, index=None, executor=executor)


def _get_conversion_name(frames):
    match t := type(frames):
        case _libspdl.FFmpegAudioFrames:
            return "async_convert_audio"
        case _libspdl.FFmpegVideoFrames:
            return "async_convert_video"
        case _libspdl.FFmpegImageFrames:
            return "async_convert_image"
        case _libspdl.NvDecVideoFrames:
            return "async_convert_video_nvdec"
        case _libspdl.NvDecImageFrames:
            return "async_convert_image_nvdec"
        case _:
            if isinstance(frames, list):
                if all(isinstance(f, _libspdl.FFmpegImageFrames) for f in frames):
                    return "async_convert_batch_image"
                if all(isinstance(f, _libspdl.NvDecImageFrames) for f in frames):
                    return "async_convert_batch_image_nvdec"
            raise TypeError(f"Unexpected type: {t}.")


def async_convert_frames(frames, executor=None):
    """Convert the frames to buffer.

    Args:
        frames : Frames object.
            - ``FFmpegAudioFrames``
            - ``FFmpegVideoFrames``
            - ``FFmpegImageFrames``
            - ``NvDecVideoFrames``
            - ``NvDecImageFrames``
            - ``List[FFmpegImageFrames]``
            - ``List[NvDecImageFrames]``

            If the buffer will be created on the device where the frame data are.

        executor (Optional[spdl.ThreadPoolExecutor]):
            Executor to run the conversion. By default, the conversion is performed on
            demuxer thread pool with higher priority than demuxing.

    Returns:
        Awaitable: Awaitable which returns a Buffer object.
    """
    func = getattr(_libspdl, _get_conversion_name(frames))
    return _async_task(func, frames, index=None, executor=executor)
