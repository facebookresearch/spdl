# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Subprocess-based iteration support.

This module provides functionality to run iterables in separate processes
using Python's multiprocessing module.
"""

import logging
import multiprocessing as mp
import queue
import weakref
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

from spdl.pipeline._iter_utils._common import (
    _Cmd,
    _drain,
    _drain_until_finished,
    _enter_iteration_mode,
    _execute_iterable,
    _execute_iterable_continuous,
    _iterate_results,
    _Msg,
    _wait_for_init,
)

__all__ = [
    "iterate_in_subprocess",
]

_LG: logging.Logger = logging.getLogger(__name__)

T = TypeVar("T")


def _join(process: mp.Process) -> None:
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


@dataclass
class _ipc(Generic[T]):
    process: mp.Process
    cmd_q: queue.Queue[_Cmd]
    data_q: queue.Queue[_Msg[T]]
    timeout: float

    def terminate(self) -> None:
        self.cmd_q.put(_Cmd.ABORT)
        _drain(self.data_q)
        _join(self.process)


class _SubprocessIterable(Iterable[T]):
    """An Iterable interface that manipulates the iterable in worker process
    and fetch the results.

    This object supports multiple iterations. Each call to ``__iter__()``
    instructs the worker subprocess to create a fresh iterator from the
    underlying iterable (via ``iter(iterable)``), without spawning a new
    process. The subprocess is reused across iterations.
    """

    def __init__(self, interface: _ipc[T]) -> None:
        self._interface: _ipc[T] | None = interface
        self._finalizer = weakref.finalize(self, interface.terminate)

    def __iter__(self) -> Iterator[T]:
        """Instruct the worker process to enter iteration mode and iterate on the results."""
        if (if_ := self._interface) is None:
            raise RuntimeError("The worker process is shutdown. Cannot iterate again.")

        try:
            # pyre-ignore[6]
            _enter_iteration_mode(if_.cmd_q, if_.data_q, if_.timeout, "subprocess")
            yield from _iterate_results(if_.data_q, if_.timeout, "subprocess")
        except BaseException:
            self._shutdown()
            raise

    def _shutdown(self) -> None:
        if (interface := self._interface) is not None:
            interface.terminate()
            self._finalizer.detach()
            self._interface = None


class _ContinuousSubprocessIterable(Iterable[T]):
    """Iterable backed by a continuously-iterating subprocess.

    The worker produces items non-stop, separated by ``ITERATION_FINISHED``
    markers at epoch boundaries. Each ``__iter__()`` call consumes one epoch
    of items (up to the next ``ITERATION_FINISHED``). Items for the next
    epoch may already be buffered in ``data_q``.

    If the previous iteration was abandoned early (parent broke out of the
    ``for`` loop), residual items are drained from ``data_q`` until the next
    ``ITERATION_FINISHED`` before yielding items from the new epoch.
    """

    def __init__(self, interface: _ipc[T]) -> None:
        self._interface: _ipc[T] | None = interface
        self._finalizer = weakref.finalize(self, interface.terminate)
        self._partially_consumed: bool = False

    def __iter__(self) -> Iterator[T]:
        if (if_ := self._interface) is None:
            raise RuntimeError("The worker process is shutdown. Cannot iterate again.")

        try:
            # If previous iteration was abandoned early, drain residual items
            if self._partially_consumed:
                _drain_until_finished(if_.data_q, if_.timeout, "subprocess")

            self._partially_consumed = True
            yield from _iterate_results(if_.data_q, if_.timeout, "subprocess")
            self._partially_consumed = False
        except (Exception, KeyboardInterrupt):
            self._shutdown()
            raise

    def _shutdown(self) -> None:
        if (interface := self._interface) is not None:
            interface.terminate()
            self._finalizer.detach()
            self._interface = None


def iterate_in_subprocess(
    fn: Callable[[], Iterable[T]],
    *,
    buffer_size: int = 3,
    initializer: Callable[[], None] | Sequence[Callable[[], None]] | None = None,
    mp_context: str | None = None,
    timeout: float | None = None,
    daemon: bool = False,
    continuous: bool = False,
) -> Iterable[T]:
    """**[Experimental]** Run the given ``iterable`` in a subprocess.

    The subprocess is created once and reused across iterations.
    The returned :py:class:`Iterable` supports multiple iterations —
    each call to ``iter()`` (or ``for ... in``) instructs the worker to
    create a fresh iterator from the underlying iterable without spawning
    a new process. Because process creation involves overhead (fork/spawn,
    initializer execution, and pickling), reusing the same worker is more
    efficient than calling this function repeatedly.

    .. note::

       ``fn()`` is called once in the subprocess to create the iterable.
       Each subsequent ``iter()`` call creates a fresh iterator by calling
       ``iter(iterable)`` on the same object. If ``fn()`` returns a proper
       ``Iterable`` (a class with ``__iter__`` that creates a new iterator
       each time), re-iteration works as expected.

       However, if ``fn()`` returns a **generator** (or any single-use
       iterator), re-iteration will silently yield no items. This is
       because a generator is its own iterator — ``iter(generator)``
       returns ``self`` — so once exhausted, calling ``iter()`` again
       returns the same exhausted object. The first iteration will work
       correctly, but all subsequent iterations will appear empty.

    Args:
        fn: Function that returns an iterator. Use :py:func:`functools.partial` to
            pass arguments to the function.
        buffer_size: Maximum number of items to buffer in the queue.
        initializer: Functions executed in the subprocess before iteration starts.
        mp_context: Context to use for multiprocessing.
            If not specified, a default method is used.
        timeout: Timeout for inactivity. If the generator function does not yield
            any item for this amount of time, the process is terminated.
        daemon: Whether to run the process as a daemon. Use it only for debugging.
        continuous: If ``True``, the worker continuously iterates without
            waiting for commands. After each ``StopIteration``, it immediately
            starts the next ``iter(iterable)``. ``data_q`` backpressure
            provides flow control. This eliminates idle time at epoch
            boundaries in multi-epoch training.

    Returns:
        Iterator over the results of the generator function.

    .. note::

       The function and the values yielded by the iterator of generator must be picklable.

    .. seealso::

       - :py:func:`run_pipeline_in_subprocess` for runinng a :py:class:`Pipeline` in
         a subprocess
       - :ref:`parallelism-performance` for the context in which this function was created.
       - :doc:`../notes/remote_iterable_protocol` for implementation details
    """
    initializers = (
        None
        if initializer is None
        else ([initializer] if not isinstance(initializer, Sequence) else initializer)
    )

    target = _execute_iterable_continuous if continuous else _execute_iterable

    ctx = mp.get_context(mp_context)
    cmd_q: queue.Queue[_Cmd] = ctx.Queue()
    data_q: queue.Queue[_Msg[T]] = ctx.Queue(maxsize=buffer_size)
    process = ctx.Process(
        target=target,
        args=(cmd_q, data_q, fn, initializers),
        daemon=daemon,
    )

    if_ = _ipc(process, cmd_q, data_q, float("inf") if timeout is None else timeout)

    process.start()

    _wait_for_init(if_.data_q, if_.timeout, "subprocess")

    if continuous:
        return _ContinuousSubprocessIterable(if_)
    return _SubprocessIterable(if_)
