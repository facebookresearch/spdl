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
from typing import cast, Generic, TypeVar

from spdl.pipeline._arena import _Arena, ArenaProtocol
from spdl.pipeline._iter_utils._common import (
    _Cmd,
    _drain,
    _enter_iteration_mode,
    _execute_iterable,
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
    arena: ArenaProtocol | None = None

    def terminate(self) -> None:
        self.cmd_q.put(_Cmd.ABORT)
        _drain(self.data_q)
        _join(self.process)
        # Unlink the shared-memory arena only after the worker is confirmed
        # dead, so nothing touches the segment afterwards. ``unlink`` runs in
        # ``finally`` so a failing ``close`` never leaves the OS-level shm
        # segment behind — teardown is the only place that calls ``unlink``.
        if (arena := self.arena) is not None:
            try:
                arena.close()
            finally:
                arena.unlink()


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
        # First step in the parent: restore arena-offloaded fields, if an arena
        # is in use.
        self._arena: _Arena | None = (
            _Arena(interface.arena) if interface.arena is not None else None
        )

    def __iter__(self) -> Iterator[T]:
        """Instruct the worker process to enter iteration mode and iterate on the results."""
        if (if_ := self._interface) is None:
            raise RuntimeError("The worker process is shutdown. Cannot iterate again.")

        try:
            # pyre-ignore[6]
            _enter_iteration_mode(if_.cmd_q, if_.data_q, if_.timeout, "subprocess")
            if (arena := self._arena) is None:
                yield from _iterate_results(if_.data_q, if_.timeout, "subprocess")
            else:
                # Reset the reader's per-iteration cursors (the worker has reset
                # its side and is quiescent until we start consuming).
                arena.reader.reset()
                for blob in _iterate_results(if_.data_q, if_.timeout, "subprocess"):
                    yield cast(T, arena.restore(cast(bytes, blob)))
        except GeneratorExit:
            return
        except BaseException:
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
    arena: ArenaProtocol | None = None,
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
        arena: Optional shared-memory arena, e.g.
            :py:class:`~spdl.pipeline.SharedMemoryRingBuffer` or
            :py:class:`~spdl.pipeline.SharedMemorySegmentPool`. When provided, large
            binary fields (large ``bytes``, NumPy arrays, Torch tensors) of each
            yielded item are written into this pre-allocated shared-memory arena.
            PyTorch and NumPy already move such payloads through shared memory for
            inter-process transfer by default, but allocate a fresh segment per
            object; the arena reuses one pre-allocated buffer instead. Ownership
            transfers to the returned iterable, which closes and unlinks the arena
            at teardown, so do not reuse the arena afterwards.

    Returns:
        Iterator over the results of the generator function.

    .. versionadded:: 0.5.0
       The ``arena`` argument.

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

    ctx = mp.get_context(mp_context)
    # pyrefly: ignore [bad-assignment]
    cmd_q: queue.Queue[_Cmd] = ctx.Queue()
    # pyrefly: ignore [bad-assignment]
    data_q: queue.Queue[_Msg[T]] = ctx.Queue(maxsize=buffer_size)
    # pyrefly: ignore [missing-attribute]
    process = ctx.Process(
        target=_execute_iterable,
        args=(cmd_q, data_q, fn, initializers, arena),
        daemon=daemon,
    )

    if_ = _ipc(
        process,
        cmd_q,
        data_q,
        float("inf") if timeout is None else timeout,
        arena,
    )

    process.start()

    _wait_for_init(if_.data_q, if_.timeout, "subprocess")

    return _SubprocessIterable(if_)
