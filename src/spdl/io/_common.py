import builtins
from concurrent.futures import CancelledError, Future

import spdl.io
from spdl.lib import _libspdl

__all__ = [
    "AsyncIOFailure",
]


def _get_demux_func(media_type, src):
    if media_type not in ["audio", "video", "image"]:
        raise ValueError(f"Unexpected media type: {media_type}.")

    match type(src):
        case builtins.bytes:
            name = f"async_demux_{media_type}_bytes"
        case builtins.memoryview:
            name = f"async_demux_{media_type}_buffer"
        case _:
            name = f"async_demux_{media_type}"
    return getattr(_libspdl, name)


def _get_decoding_func(packets):
    match t := type(packets):
        case _libspdl.AudioPackets:
            name = "async_decode_audio"
        case _libspdl.VideoPackets:
            name = "async_decode_video"
        case _libspdl.ImagePackets:
            name = "async_decode_image"
        case _:
            raise TypeError(f"Unexpected type: {t}.")
    return getattr(_libspdl, name)


def _get_nvdec_decoding_func(packets):
    match t := type(packets):
        case _libspdl.VideoPackets:
            name = "async_decode_video_nvdec"
        case _libspdl.ImagePackets:
            name = "async_decode_image_nvdec"
        case _:
            raise TypeError(f"Unexpected type: {t}.")
    return getattr(_libspdl, name)


def _get_conversion_func(frames):
    match t := type(frames):
        case _libspdl.FFmpegAudioFrames:
            name = "async_convert_audio"
        case _libspdl.FFmpegVideoFrames:
            name = "async_convert_video"
        case _libspdl.FFmpegImageFrames:
            name = "async_convert_image"
        case _libspdl.NvDecVideoFrames:
            name = "async_convert_video_nvdec"
        case _libspdl.NvDecImageFrames:
            name = "async_convert_image_nvdec"
        case _:
            if not isinstance(frames, list):
                raise TypeError(f"Unexpected type: {t}.")
            if all(isinstance(f, _libspdl.FFmpegImageFrames) for f in frames):
                name = "async_convert_batch_image"
            elif all(isinstance(f, _libspdl.NvDecImageFrames) for f in frames):
                name = "async_convert_batch_image_nvdec"
            else:
                raise TypeError(
                    f"Unexpected type: {t}. When the container type is list, all frames must be either FFmpegImageFrames or NvDecImageFrames."
                )
    return getattr(_libspdl, name)


# Exception class used to signal the failure of C++ op to Python.
# Not exposed to user code.
class AsyncIOFailure(RuntimeError):
    """Exception type used to pass the error message from libspdl."""

    pass


def _futurize_task(func, *args, **kwargs):
    future = Future()

    def nofify_exception(msg: str, cancelled: bool):
        err = CancelledError() if cancelled else spdl.io.AsyncIOFailure(msg)
        future.set_exception(err)

    future.set_running_or_notify_cancel()
    sf = func(future.set_result, nofify_exception, *args, **kwargs)
    future.__spdl_future = sf
    return future


def _futurize_generator(func, num_items, *args, **kwargs):
    futures = [Future() for _ in range(num_items)]

    index = 0

    def set_result(val):
        nonlocal index
        futures[index].set_result(val)
        index += 1

        if index < num_items:
            futures[index].set_running_or_notify_cancel()

    def nofify_exception(msg: str, cancelled: bool):
        err = CancelledError() if cancelled else spdl.io.AsyncIOFailure(msg)

        nonlocal index
        futures[index].set_exception(err)
        index += 1

        for fut in futures[index:]:
            fut.cancel()

    futures[index].set_running_or_notify_cancel()
    sf = func(set_result, nofify_exception, *args, **kwargs)

    for future in futures:
        future.__spdl_future = sf

    return futures
