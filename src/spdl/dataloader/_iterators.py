# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import multiprocessing as mp
import queue
import time
from collections.abc import Callable, Iterator
from typing import Any, TypeVar

__all__ = ["run_in_subprocess"]

# Message from parent to worker
_MSG_PARENT_REQUEST_STOP = "PARENT_REQUEST_STOP"

# Message from worker to the parent
_MSG_GENERATOR_FAILED = "GENERATOR_FAILED_TO_INITIALIZE"
_MSG_ITERATION_FINISHED = "ITERATION_FINISHED"
_MSG_DATA_QUEUE_FULL = "DATA_QUEUE_FULL"

T = TypeVar("T")

_LG: logging.Logger = logging.getLogger(__name__)

# pyre-unsafe


def _execute_iterator(
    msg_queue: mp.Queue,
    data_queue: mp.Queue,
    fn: Callable[[Any, ...], Iterator[Any]],
    args: tuple[Any, ...] | None,
    kwargs: dict[str, Any] | None,
) -> None:
    try:
        gen = fn(*(args or ()), **(kwargs or {}))
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
            raise ValueError(f"Unexpected message redeived: {msg}")

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
        except queue.Full:
            msg_queue.put(_MSG_DATA_QUEUE_FULL)
            return


def run_in_subprocess(
    fn: Callable[[Any, ...], Iterator[Any]],
    args: tuple[Any, ...] | None = None,
    kwargs: dict[str, Any] | None = None,
    queue_size: int = 64,
    mp_context: str = "forkserver",
    timeout: float | None = None,
    daemon: bool = False,
) -> Iterator[Any]:
    """Run an iterator in a separate process, and yield the results one by one.

    Args:
        fn: Generator function.
        args: Arguments to pass to the generator function.
        kwargs: Keyword arguments to pass to the generator function.
        queue_size: Maximum number of items to buffer in the queue.
        mp_context: Context to use for multiprocessing.
        timeout: Timeout for inactivity. If the generator function does not yield
            any item for this amount of time, the process is terminated.
        daemnon: Whether to run the process as a daemon.

    Returns:
        Iterator over the results of the generator function.

    .. note::

       The generator function, its arguments and the result of generator must be picklable.
    """
    ctx = mp.get_context(mp_context)
    msg_q = ctx.Queue()
    data_q = ctx.Queue(maxsize=queue_size)

    process = ctx.Process(
        target=_execute_iterator,
        args=(msg_q, data_q, fn, args, kwargs),
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
                if msg == _MSG_ITERATION_FINISHED:
                    return
                if msg == _MSG_GENERATOR_FAILED:
                    raise RuntimeError(
                        "The worker process quit because the generator failed."
                    )
                if msg == _MSG_DATA_QUEUE_FULL:
                    raise RuntimeError(
                        "The worker process quit because the data queue is full for too long."
                    )

                raise ValueError(f"Unexpected message received: {msg}")

            try:
                yield data_q.get(timeout=1)
            except queue.Empty:
                pass
            else:
                t0 = time.monotonic()

            if timeout is not None:
                if (elapsed := time.monotonic() - t0) > timeout:
                    raise RuntimeError(
                        f"The worker process did not produce any data for {elapsed:.2f} seconds."
                    )

    except Exception:
        msg_q.put(_MSG_PARENT_REQUEST_STOP)
        raise
    finally:
        while not data_q.empty():
            data_q.get_nowait()
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
