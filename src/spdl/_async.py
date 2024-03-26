import asyncio
import concurrent.futures
import functools
from typing import Any

import spdl.libspdl

_task = [
    "async_apply_bsf",
    "async_convert_audio",
    "async_convert_image",
    "async_convert_video",
    "async_convert_audio_cpu",
    "async_convert_image_cpu",
    "async_convert_video_cpu",
    "async_convert_image_nvdec",
    "async_convert_video_nvdec",
    "async_convert_batch_image",
    "async_convert_batch_image_nvdec",
    "async_decode",
    "async_decode_nvdec",
    "async_demux_image",
    "async_sleep",
]

_generator = [
    "async_demux_audio",
    "async_demux_video",
]

__all__ = _task + _generator


def __getattr__(name: str) -> Any:
    if name in __all__:
        func = getattr(spdl.libspdl, name)
        if name in _task:
            return _to_async_task(func)
        if name in _generator:
            return _to_async_generator(func)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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


def _to_async_task(func):
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
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

    return async_wrapper


def _to_async_generator(func):
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
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

    return async_wrapper
