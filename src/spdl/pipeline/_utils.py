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
from typing import Any, TypeVar

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
def _log_exception(task: Task, stacklevel: int, ignore_cancelled: bool) -> None:
    try:
        task.result()
    except asyncio.exceptions.CancelledError:
        if not ignore_cancelled:
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
            "Task [%s] failed: %s %s (%s:%d:%s)",
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
    ignore_cancelled: bool = True,
) -> Task[T]:
    """Wrapper around :py:func:`asyncio.create_task`. Add logging callback."""
    task = asyncio.create_task(coro, name=name)
    task.add_done_callback(
        lambda t: _log_exception(t, stacklevel=3, ignore_cancelled=ignore_cancelled)
    )
    return task


################################################################################
# iterate_in_subprocess
################################################################################

# Message from parent to worker
_MSG_PARENT_REQUEST_STOP = "PARENT_REQUEST_STOP"

# Message from worker to the parent
_MSG_INITIALIZER_FAILED = "INITIALIZER_FAILED"
_MSG_GENERATOR_FAILED = "GENERATOR_FAILED_TO_INITIALIZE"
_MSG_ITERATION_FINISHED = "ITERATION_FINISHED"
_MSG_DATA_QUEUE_FAILED = "DATA_QUEUE_FAILED"


def _execute_iterator(
    msg_queue: mp.Queue,
    data_queue: mp.Queue,
    fn: Callable[[], Iterator[T]],
    initializer: Callable[[], None],
) -> None:
    if initializer is not None:
        try:
            initializer()
        except Exception:
            msg_queue.put(_MSG_INITIALIZER_FAILED)
            raise

    try:
        gen = iter(fn())
    except Exception:
        msg_queue.put(_MSG_GENERATOR_FAILED)
        raise

    while True:
        try:
            msg = msg_queue.get_nowait()
        except queue.Empty:
            pass
        else:
            if msg == _MSG_PARENT_REQUEST_STOP:
                return
            raise ValueError(f"[INTERNAL ERROR] Unexpected message received: {msg}")

        try:
            item = next(gen)
        except StopIteration:
            msg_queue.put(_MSG_ITERATION_FINISHED)
            return
        except Exception:
            msg_queue.put(_MSG_GENERATOR_FAILED)
            return

        try:
            data_queue.put(item)
        except Exception:
            msg_queue.put(_MSG_DATA_QUEUE_FAILED)
            return


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
    """
    ctx = mp.get_context(mp_context)
    msg_q = ctx.Queue()
    data_q: mp.Queue = ctx.Queue(maxsize=buffer_size)

    def _drain() -> Iterator[T]:
        while not data_q.empty():
            yield data_q.get_nowait()

    process = ctx.Process(
        target=_execute_iterator,
        args=(msg_q, data_q, fn, initializer),
        daemon=daemon,
    )
    process.start()
    t0 = time.monotonic()
    try:
        while True:
            try:
                msg = msg_q.get_nowait()
            except queue.Empty:
                pass
            else:
                # When a message is found, the child process stopped putting data.
                yield from _drain()

                if msg == _MSG_ITERATION_FINISHED:
                    return
                if msg == _MSG_INITIALIZER_FAILED:
                    raise RuntimeError(
                        "The worker process quit because the initializer failed."
                    )
                if msg == _MSG_GENERATOR_FAILED:
                    raise RuntimeError(
                        "The worker process quit because the generator failed."
                    )
                if msg == _MSG_DATA_QUEUE_FAILED:
                    raise RuntimeError(
                        "The worker process quit because it failed at passing the data."
                    )

                raise ValueError(f"[INTERNAL ERROR] Unexpected message received: {msg}")

            try:
                yield data_q.get(timeout=1)
            except queue.Empty:
                pass
            else:
                t0 = time.monotonic()

            if timeout is not None:
                if (elapsed := time.monotonic() - t0) > timeout:
                    raise RuntimeError(
                        "The worker process did not produce any data for "
                        f"{elapsed:.2f} seconds."
                    )

    except (Exception, KeyboardInterrupt):
        msg_q.put(_MSG_PARENT_REQUEST_STOP)
        raise
    finally:
        yield from _drain()
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


def cache_iterator(
    src: Iterable[T],
    num_caches: int,
    *,
    return_caches_after: int | None = None,
    delete_src: bool = True,
) -> Iterator[T]:
    """Iterate over the source but returns cached values after the given iteration.

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
        delete_src: When this iterator starts returning the cached value,
            call ``del`` on the original data loader so that resources are
            released.

    Returns:
        The new iterator.
    """
    cache = []

    run_for = num_caches if return_caches_after is None else return_caches_after

    for i, data in enumerate(src, start=1):
        yield data

        if len(cache) < num_caches:
            cache.append(data)

        if i >= run_for:
            break

    if delete_src:
        del src

    while True:
        yield from cache
