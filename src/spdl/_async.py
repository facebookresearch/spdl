import asyncio
import concurrent.futures
import functools
from typing import Any

import spdl.libspdl

_task = [
    "apply_bsf",
    "convert_to_cpu_buffer_async",
    "decode_audio_async",
    "decode_video_async",
    "decode_image_async",
    "decode_video_nvdec_async",
    "decode_image_nvdec_async",
    "demux_image_async",
]

_generator = [
    "demux_video_async",
    "demux_audio_async",
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


def _to_async_task(func):
    @functools.wraps(func)
    async def async_wrapper(*args, _exception_backoff=0, **kwargs):
        # Implementation note
        #
        # This implementation is essentially simplified version of
        # `loop.run_in_executor`.
        # https://docs.python.org/3/library/asyncio-eventloop.html#asyncio.loop.run_in_executor
        #
        # `loop.run_in_executor` turns synchronous function into asynchronous
        # function by running the function in a thread and monitoring the
        # future object.
        #
        # Since our function already runs in threads constructed in C++
        # realm, we mimic what is done at the outer surface of
        # `ThreadPoolExecutor.submit()`.
        #
        # The relevant implementations are found;
        #
        # * `loop.run_in_executor`
        #   https://github.com/python/cpython/blob/bb66600558cb8d5dd9a56f562bd9531eb1e1685f/Lib/asyncio/base_events.py#L887-L888
        #
        # * `ThreadPoolExecutor.submit`
        #    https://github.com/python/cpython/blob/bb66600558cb8d5dd9a56f562bd9531eb1e1685f/Lib/concurrent/futures/thread.py#L164-L180
        #
        # * `_WorkItem.run`
        #    https://github.com/python/cpython/blob/bb66600558cb8d5dd9a56f562bd9531eb1e1685f/Lib/concurrent/futures/thread.py#L53-L64
        #
        # As you can see from `_WorkItem.run`, the given synchronous function is
        # ran normally, and the result is notified by assigining result/exception
        # to `concurrent.futures.Future` object.
        #
        # In to reflect this in `asyncio`, one needs to wrap the
        # `concurrent.futures.Future` with `asyncio.futures.Future` and await on
        # it.
        #
        # We just need to perform the result assignment in the coroutine, which is
        # executed in backgroud thread pool.
        # For this, we cannot pass `Future` Python object to the C++ code as-is.
        # (I mean, I tried, but could not make it work).
        # Because the call back will be performed in a thread disconnected from
        # Python interpreter, so it causes some synchronization issue around GIL,
        # and fails with segfault or bus error.
        #
        # Interestingly, PyBind11 does the right thing and apply some special
        # handling to the call back function if it is received as C++ function.
        #
        # https://stackoverflow.com/a/72933328/3670924
        #
        # For assining exception, since it is not straightforward to manually
        # translate C++ error to Python exception and assign it to future,
        # we use `notify_exception` helper function which signals Future
        # of the failure by assining the prefixed RuntimeError.
        # Then we raise the underlying exception by poking the
        # `folly::SemiFuture` class.
        # This way, the error happened on C++ are raised in the same way as
        # synchronous functions that PyBind11 will translate it for us.
        #
        future = concurrent.futures.Future()
        assert future.set_running_or_notify_cancel()

        def nofify_exception():
            future.set_exception(RuntimeError("Async operation failed."))

        sf = func(future.set_result, nofify_exception, *args, **kwargs)

        try:
            return await asyncio.futures.wrap_future(future)
        except RuntimeError:
            pass

        await asyncio.sleep(_exception_backoff)
        sf.rethrow()

    return async_wrapper


def _to_async_generator(func):
    @functools.wraps(func)
    async def async_wrapper(*args, _exception_backoff=0, **kwargs):
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
            futures[-1].set_exception(RuntimeError("Async operation failed."))

        sf = func(set_result, notify_exception, *args, **kwargs)
        while futures:
            try:
                val = await asyncio.futures.wrap_future(futures[0])
            except RuntimeError:
                # The error here is the one from notify_exception function, so
                # we absorb it. The actual error will be thrown from the underlying
                # `folly::SemiFuture` in the next phase.
                break
            else:
                if val is None:
                    return
                yield val
                futures.pop(0)
            finally:
                await asyncio.sleep(0)

        await asyncio.sleep(_exception_backoff)
        sf.rethrow()

    return async_wrapper
