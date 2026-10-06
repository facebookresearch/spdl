# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""Subprocess-based iteration support.

This module provides functionality to run iterables in separate processes
using Python's multiprocessing module.
"""

import logging
import multiprocessing as mp
import queue
import threading
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field
from multiprocessing.util import Finalize
from typing import Any, cast, Generic, TypeVar

from spdl.pipeline._arena import _Arena, ArenaProtocol
from spdl.pipeline._iter_utils._common import (
    _Cmd,
    _drain,
    _enter_iteration_mode,
    _execute_iterable,
    _iterate_results,
    _Msg,
    _Status,
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
        _LG.warning("Terminating the worker process.")
        process.terminate()
        process.join(10)

    if process.exitcode is None:
        _LG.warning("Killing the worker process.")
        process.kill()
        process.join(10)

    if process.exitcode is None:
        _LG.warning("Failed to kill the worker process.")


def _close_queue(q: Any, *, abandon: bool = False) -> None:
    """Close a main-process multiprocessing queue and its feeder thread."""
    if abandon:
        try:
            q.cancel_join_thread()
        except Exception:
            _LG.debug("Failed to abandon subprocess queue data", exc_info=True)
    try:
        q.close()
        if not abandon:
            q.join_thread()
    except Exception:
        # A concurrent/earlier cleanup can already have closed the queue. Queue
        # teardown is best-effort and must not mask the pipeline's real result.
        _LG.debug("Failed to close subprocess queue cleanly", exc_info=True)


@dataclass
class _ipc(Generic[T]):
    process: mp.Process
    cmd_q: queue.Queue[_Cmd] | None
    data_q: queue.Queue[_Msg[T]] | None
    timeout: float
    arena: ArenaProtocol | None = None
    closed: threading.Event = field(default_factory=threading.Event)

    def terminate(self, *, force: bool = False) -> None:
        if self.closed.is_set():
            return
        self.closed.set()
        cmd_q = self.cmd_q
        data_q = self.data_q
        assert cmd_q is not None
        assert data_q is not None
        # ``Process.start()`` can fail before assigning a PID (for example when
        # spawn cannot pickle an argument). In that state ``join()`` raises, but
        # the queues and optional arena still belong to this setup attempt and
        # must be released.
        process_started = self.process.pid is not None
        arena = self.arena
        try:
            if process_started and self.process.is_alive():
                if force:
                    # During initialization the worker cannot observe ABORT until the
                    # initializer returns, so terminate it immediately on setup failure.
                    self.process.terminate()
                else:
                    try:
                        cmd_q.put_nowait(_Cmd.ABORT)
                    except (EOFError, OSError, ValueError, queue.Full):
                        # A closed or broken command queue cannot request graceful
                        # shutdown. Continue with arena wakeup, draining, and bounded
                        # process reaping instead of skipping the rest of teardown.
                        _LG.debug(
                            "Could not request graceful subprocess shutdown.",
                            exc_info=True,
                        )
            # Wake any worker that is currently blocked in ``write_binary`` /
            # ``begin_unit`` on the arena's space condition variable. Without this,
            # a producer waiting for a (never-coming) consumer reclaim would stay
            # blocked through ``_join`` and hang teardown.
            if arena is not None:
                shutdown = getattr(arena, "shutdown_arena", None)
                if shutdown is not None:
                    try:
                        shutdown()
                    except Exception:
                        # Arena wakeup is best effort. Continue draining and
                        # reaping so this cleanup error cannot mask the failure
                        # that initiated teardown.
                        _LG.warning(
                            "Failed to wake the subprocess arena during teardown.",
                            exc_info=True,
                        )
            try:
                _drain(data_q)
            except (EOFError, OSError, RuntimeError):
                # Queue cleanup discards unread values, but multiprocessing.Queue
                # still unpickles them. Tensor payloads rebuild storage through the
                # producer's resource_sharer socket, which may disappear as soon as
                # ABORT lets the producer exit. The value is already being
                # discarded, so do not let its deserialization skip worker and IPC
                # cleanup.
                _LG.debug(
                    "Ignoring an unread subprocess payload during teardown.",
                    exc_info=True,
                )
        finally:
            try:
                if process_started:
                    _join(self.process)
                # A consumer can be blocked in ``data_q.get(timeout=...)`` while
                # another thread invokes the finalizer. Process exit alone does
                # not wake that reader because this process still owns the queue's
                # read descriptor. Wake it before closing the queue.
                try:
                    _drain(data_q)
                    data_q.put_nowait(_Msg(_Status.ITERATION_FINISHED))
                except (EOFError, OSError, RuntimeError, ValueError, queue.Full):
                    _LG.debug(
                        "Failed to wake subprocess consumer during teardown",
                        exc_info=True,
                    )
            finally:
                # Unlink the shared-memory arena only after reaping was attempted,
                # so nothing should touch the segment afterwards. ``unlink`` runs in
                # ``finally`` so a failing ``close`` never leaves the OS-level shm
                # segment behind — teardown is the only place that calls ``unlink``.
                try:
                    if arena is not None:
                        try:
                            arena.close()
                        finally:
                            arena.unlink()
                finally:
                    # Buffered commands/results cannot be used after teardown.
                    # Deliberately abandon feeder data before closing descriptors so
                    # cleanup cannot hang while flushing toward a dead worker.
                    try:
                        with ExitStack() as cleanup:
                            for q in reversed((cmd_q, data_q)):
                                cleanup.callback(_close_queue, q, abandon=True)
                    finally:
                        self.cmd_q = None
                        self.data_q = None


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
        self._finalizer = Finalize(self, interface.terminate, exitpriority=10)
        # First step in the parent: restore arena-offloaded fields, if an arena
        # is in use.
        self._arena: _Arena | None = (
            _Arena(interface.arena) if interface.arena is not None else None
        )

    def __iter__(self) -> Iterator[T]:
        """Instruct the worker process to enter iteration mode and iterate on the results."""
        if (if_ := self._interface) is None:
            raise RuntimeError("The worker process is shutdown. Cannot iterate again.")
        if if_.closed.is_set():
            return

        try:
            arena = self._arena
            cmd_q = if_.cmd_q
            data_q = if_.data_q
            if cmd_q is None or data_q is None:
                return
            # pyre-ignore[6]
            _enter_iteration_mode(
                cmd_q,
                data_q,
                if_.timeout,
                "subprocess",
                None if arena is None else arena.discard,
                if_.process.is_alive,
            )
            if arena is None:
                yield from _iterate_results(
                    data_q,
                    if_.timeout,
                    "subprocess",
                    if_.process.is_alive,
                )
            else:
                # Let the backend prepare for the next iteration after the
                # worker has prepared its side.
                arena.reader.reset()
                for blob in _iterate_results(
                    data_q,
                    if_.timeout,
                    "subprocess",
                    if_.process.is_alive,
                ):
                    yield cast(T, arena.restore(cast(bytes, blob)))
        except GeneratorExit:
            return
        except BaseException:
            self._shutdown()
            raise

    def _shutdown(self) -> None:
        if self._interface is not None:
            self._finalizer()
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

    try:
        process.start()
        _wait_for_init(
            data_q,
            if_.timeout,
            "subprocess",
            if_.process.is_alive,
        )
    except BaseException:
        # No iterable/finalizer has been returned yet, so setup owns cleanup.
        # Force termination because a blocked initializer cannot consume ABORT.
        try:
            if_.terminate(force=True)
        except Exception:
            _LG.warning(
                "Failed to clean up subprocess after initialization.", exc_info=True
            )
        raise

    return _SubprocessIterable(if_)
