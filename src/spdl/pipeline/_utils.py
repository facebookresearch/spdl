# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import asyncio
import logging
import multiprocessing as mp
import queue
import sys
import time
import traceback
from asyncio import Task
from collections.abc import Callable, Coroutine, Generator, Iterable, Iterator
from dataclasses import dataclass
from enum import Enum
from typing import Any, Generic, TypeVar

__all__ = [
    "create_task",
    "iterate_in_subprocess",
    "cache_iterator",
]

_LG: logging.Logger = logging.getLogger(__name__)

T = TypeVar("T")


# Note:
# This function is intentionally made in a way it cannot be directly attached to
# task callback. Instead it should be wrapped in lambda as follow, even though
# there are some opposition on assigning lambda:
#
# `task.add_done_callback(lambda t: _log_exception(t, stacklevel=2))`
#
# This way, the filename and line number of the log points to the location where
# task was created.
# Otherwise the log will point to the location somewhere deep in `asyncio` module
# which is not very helpful.
def _log_exception(task: Task, stacklevel: int, log_cancelled: bool) -> None:
    try:
        task.result()
    except asyncio.exceptions.CancelledError:
        if log_cancelled:
            _LG.warning(
                "Task [%s] was cancelled.", task.get_name(), stacklevel=stacklevel
            )
    except (TimeoutError, asyncio.exceptions.TimeoutError):
        # Timeout does not contain any message
        _LG.error("Task [%s] timeout.", task.get_name(), stacklevel=stacklevel)
    except Exception as err:
        _, _, exc_tb = sys.exc_info()
        f = traceback.extract_tb(exc_tb, limit=-1)[-1]

        _LG.error(
            "Task [%s]: %s: %s (%s:%d:%s)",
            task.get_name(),
            type(err).__name__,
            err,
            f.filename,
            f.lineno,
            f.name,
            stacklevel=stacklevel,
        )


def create_task(
    coro: Coroutine[Any, Any, T] | Generator[Any, None, T],  # pyre-ignore: [2]
    name: str | None = None,
    log_cancelled: bool = False,
) -> Task[T]:
    """Wrapper around :py:func:`asyncio.create_task`. Add logging callback."""
    task = asyncio.create_task(coro, name=name)
    task.add_done_callback(
        lambda t: _log_exception(t, stacklevel=3, log_cancelled=log_cancelled)
    )
    return task


################################################################################
# iterate_in_subprocess
################################################################################


# Command from parent to worker
class _Cmd(Enum):
    ABORT = 0


# Final status of the iterator
class _Status(Enum):
    UNEXPECTED_CMD_RECIEVED = 0
    INITIALIZATION_FAILED = 1
    ITERATOR_FAILED = 2

    ITERATOR_SUCCESS = 3
    ITERATION_FINISHED = 4
    # Iteration finished normally or
    # terminating early par the request from the parent


# Message from worker to the parent
@dataclass
class _Msg(Generic[T]):
    status: _Status
    # additional data
    # String in case of failure
    message: T | str = ""


def _execute_iterator(
    cmd_queue: mp.Queue,
    data_queue: mp.Queue,
    fn: Callable[[], Iterator[T]],
    initializer: Callable[[], None],
) -> None:
    if initializer is not None:
        try:
            initializer()
        except Exception as e:
            data_queue.put(
                _Msg(
                    _Status.INITIALIZATION_FAILED,
                    message=f"Initializer failed: {e}",
                )
            )
            return

    try:
        iterable = fn()
    except Exception as e:
        data_queue.put(
            _Msg(
                _Status.INITIALIZATION_FAILED,
                message=f"Failed to create the iterable: {e}",
            )
        )
        return

    try:
        gen = iter(iterable)
    except Exception as e:
        data_queue.put(
            _Msg(
                _Status.ITERATOR_FAILED,
                message=f"Failed to create the iterator: {e}",
            )
        )
        return

    while True:
        try:
            cmd = cmd_queue.get_nowait()
        except queue.Empty:
            pass
        else:
            match cmd:
                case _Cmd.ABORT:
                    pass
                case _:
                    data_queue.put(_Msg(_Status.UNEXPECTED_CMD_RECIEVED, str(cmd)))
            return

        try:
            item = next(gen)
            data_queue.put(_Msg(_Status.ITERATOR_SUCCESS, message=item))
        except StopIteration:
            data_queue.put(_Msg(_Status.ITERATION_FINISHED))
            return
        except Exception as e:
            data_queue.put(
                _Msg(
                    _Status.ITERATOR_FAILED,
                    message=f"Failed to fetch the next item: {e}",
                )
            )
            return


def _drain(queue: mp.Queue) -> None:
    while not queue.empty():
        queue.get_nowait()


def _shutdown(process: mp.Process) -> None:
    process.join(3)

    if process.exitcode is None:
        _LG.warning("Terminaging the worker process.")
        process.terminate()
        process.join(10)

    if process.exitcode is None:
        _LG.warning("Killing the worker process.")
        process.kill()
        process.join(10)

    if process.exitcode is None:
        _LG.warning("Failed to kill the worker process.")


def _iterate(data_queue: mp.Queue, timeout: float) -> Iterator[object]:
    wait = min(0.1, timeout)

    t0 = time.monotonic()
    while True:
        try:
            item: _Msg[object] = data_queue.get(timeout=wait)
            t0 = time.monotonic()
        except queue.Empty:
            if (elapsed := time.monotonic() - t0) > timeout:
                raise RuntimeError(
                    "The worker process did not produce any data for "
                    f"{elapsed:.2f} seconds."
                ) from None
            continue
        else:
            match item.status:
                case _Status.ITERATOR_SUCCESS:
                    yield item.message
                case _Status.ITERATION_FINISHED:
                    return
                case _Status.INITIALIZATION_FAILED:
                    raise RuntimeError(f"The worker process quit. {item.message}")
                case _Status.ITERATOR_FAILED:
                    raise RuntimeError(f"The worker process quit. {item.message}")
                case _Status.UNEXPECTED_CMD_RECIEVED:
                    raise RuntimeError(
                        "[INTERNAL ERROR] The worker received unexpected command: "
                        f"{item.message}"
                    )
                case _:
                    raise RuntimeError(
                        f"[INTERNAL ERROR] Unexpected return value: {item}"
                    )


def iterate_in_subprocess(
    fn: Callable[[], Iterable[T]],
    *,
    buffer_size: int = 3,
    initializer: Callable[[], None] | None = None,
    mp_context: str | None = None,
    timeout: float | None = None,
    daemon: bool = False,
) -> Iterator[T]:
    """Run an iterator in a separate process, and yield the results one by one.

    Args:
        fn: Function that returns an iterator. Use :py:func:`functools.partial` to
            pass arguments to the function.
        buffer_size: Maximum number of items to buffer in the queue.
        initializer: A function executed in the subprocess before iteration starts.
        mp_context: Context to use for multiprocessing.
            If not specified, a default method is used.
        timeout: Timeout for inactivity. If the generator function does not yield
            any item for this amount of time, the process is terminated.
        daemon: Whether to run the process as a daemon. Use it only for debugging.

    Returns:
        Iterator over the results of the generator function.

    .. note::

       The function and the values yielded by the iterator of generator must be picklable.

    .. seealso::

       - :py:func:`run_pipeline_in_subprocess` for runinng a :py:class:`Pipeline` in
         a subprocess
       - :ref:`parallelism-performance` for the context in which this function was created.

    """
    ctx = mp.get_context(mp_context)
    cmd_queue = ctx.Queue()
    data_queue: mp.Queue = ctx.Queue(maxsize=buffer_size)

    process = ctx.Process(
        target=_execute_iterator,
        args=(cmd_queue, data_queue, fn, initializer),
        daemon=daemon,
    )
    process.start()
    timeout_ = float("inf") if timeout is None else timeout
    try:
        yield from _iterate(data_queue, timeout_)  # pyre-ignore
    except (Exception, KeyboardInterrupt):
        cmd_queue.put(_Cmd.ABORT)
        _drain(data_queue)
        raise
    finally:
        _shutdown(process)


def cache_iterator(
    src: Iterable[T],
    num_caches: int,
    *,
    return_caches_after: int | None = None,
    stop_after: int | None = None,
    delete_src: bool = True,
) -> Iterator[T]:
    """Caches values from the iterator and returns caches after the given iteration.

    The function is intended for estimating the maximum performance gain achieved
    by optimizing the data loader.

    You can wrap your data loader with this function, and run it in the training
    pipeline, and compare the performance to see if the training pipeline is
    bottlenecked with data loading.

    Args:
        src: Source iterator. Expected to be a data loader object.

        num_caches: The number of items (batches) to cache.

        return_caches_after: The number of iterations to use the original
            iterator. By default, it uses the same value as ``num_caches``.

        stop_after: If provided, the iteration stops after the given number
            of iteration is completed (including before and after cached values
            are returned). If not provided, the iterator keeps yielding
            the cached values forever.

        delete_src: When this iterator starts returning the cached value,
            call ``del`` on the original data loader so that resources are
            released.

    Returns:
        The wrapper iterator.
    """

    # Note - Design choice
    # When these optional values are provided, we could choose to not validate.
    # But the purpose of this function is to make sure you are using cache,
    # so we raise an error if these parameters do not make logical sense.
    if return_caches_after is not None:
        if return_caches_after < num_caches:
            raise ValueError(
                "When provided, `return_caches_after` must be greater than or "
                "equal to `num_caches`. "
                f"{num_caches=}, {return_caches_after=}"
            )

    if stop_after is not None:
        if stop_after < num_caches:
            raise ValueError(
                "When provided, `stop_after` must be greater than or equal to "
                "`num_caches`. "
                f"{num_caches=}, {stop_after=}"
            )
        if return_caches_after is not None and stop_after < return_caches_after:
            raise ValueError("")

    cache: list[T] = []

    run_for = num_caches if return_caches_after is None else return_caches_after
    max_ite = stop_after or float("inf")

    num_ite = 0
    for data in src:
        yield data
        num_ite += 1

        if len(cache) < num_caches:
            cache.append(data)

        if num_ite >= max_ite:
            return

        if num_ite >= run_for:
            break

    if delete_src:
        del src

    while True:
        for v in cache:
            yield v
            num_ite += 1

            if num_ite >= max_ite:
                return
