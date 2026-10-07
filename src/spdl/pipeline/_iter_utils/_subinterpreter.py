# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""Subinterpreter-based iteration support.

This module provides functionality to run iterables in Python subinterpreters
using Python 3.14's concurrent.interpreters module.
"""

import logging
import sys
import threading
import weakref
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Generic, TypeVar

from spdl.pipeline._iter_utils._common import (
    _Cmd,
    _drain,
    _enter_iteration_mode,
    _execute_iterable,
    _iterate_results,
    _wait_for_init,
)

__all__ = [
    "iterate_in_subinterpreter",
]

_LG: logging.Logger = logging.getLogger(__name__)

T = TypeVar("T")

_DEFERRED_JOIN_POLL_INTERVAL = 60.0


if sys.version_info < (3, 14):

    def _impl(
        fn: Callable[[], Iterable[T]],  # noqa: ARG001
        *,
        buffer_size: int = 3,  # noqa: ARG001
        initializer: Callable[[], None] | Sequence[Callable[[], None]] | None = None,  # noqa: ARG001
        timeout: float | None = None,  # noqa: ARG001
    ) -> Iterable[T]:
        raise RuntimeError(
            f"iterate_in_subinterpreter requires Python 3.14 or later. "
            f"Current version: {sys.version_info.major}.{sys.version_info.minor}"
        )

else:
    import concurrent.interpreters

    # short for inter-interpreter communication.
    # (analogous to inter-process communication)
    @dataclass
    class _iic(Generic[T]):
        thread: threading.Thread
        interpreter: "concurrent.interpreters.Interpreter | None"
        cmd_q: "concurrent.interpreters.Queue"
        data_q: "concurrent.interpreters.Queue"
        timeout: float
        _cleanup_scheduled: bool = field(default=False, init=False, repr=False)
        _terminate_owner: object | None = field(default=None, init=False, repr=False)
        _cleanup_lock: threading.Lock = field(
            default_factory=threading.Lock, init=False, repr=False
        )

        def _close_interpreter(self) -> bool:
            if (interpreter := self.interpreter) is None:
                return True
            try:
                interpreter.close()
            except Exception:
                _LG.warning("Failed to close subinterpreter worker.", exc_info=True)
                return False
            self.interpreter = None
            return True

        def _close_after_worker_exit(self) -> None:
            joined = False
            warned_pending = False
            try:
                try:
                    while not joined:
                        self.thread.join(timeout=_DEFERRED_JOIN_POLL_INTERVAL)
                        joined = not self.thread.is_alive()
                        if not joined and not warned_pending:
                            _LG.warning(
                                "Subinterpreter worker is still running; deferred "
                                "cleanup remains pending."
                            )
                            warned_pending = True
                except Exception:
                    _LG.warning(
                        "Failed to join subinterpreter worker during deferred cleanup.",
                        exc_info=True,
                    )
                if joined:
                    try:
                        self._close_interpreter()
                    except BaseException:
                        _LG.warning(
                            "Failed to close subinterpreter worker during deferred "
                            "cleanup.",
                            exc_info=True,
                        )
                        raise
            finally:
                with self._cleanup_lock:
                    # A failed join or an exceptional close must not suppress a
                    # later terminate() retry permanently.
                    self._cleanup_scheduled = False

        def _schedule_deferred_cleanup(self, owner: object) -> None:
            with self._cleanup_lock:
                # Keep helper construction/start and ownership publication atomic.
                # A concurrent terminate() waits here, then either observes the
                # scheduled reaper or takes over after a startup failure.
                try:
                    cleanup_thread = threading.Thread(
                        target=self._close_after_worker_exit,
                        name="spdl-subinterpreter-cleanup",
                        daemon=True,
                    )
                    cleanup_thread.start()
                except BaseException:
                    if self._terminate_owner is owner:
                        self._terminate_owner = None
                    raise
                self._cleanup_scheduled = True
                if self._terminate_owner is owner:
                    self._terminate_owner = None

        def terminate(self) -> None:
            owner = object()
            with self._cleanup_lock:
                if self.interpreter is None:
                    return
                if self._cleanup_scheduled:
                    _LG.debug("Deferred subinterpreter cleanup is already scheduled.")
                    return
                if self._terminate_owner is not None:
                    _LG.debug("Subinterpreter cleanup is already in progress.")
                    return
                self._terminate_owner = owner

            try:
                if self.thread.is_alive():
                    self.cmd_q.put(_Cmd.ABORT)
                _drain(self.data_q)
                self.thread.join(timeout=3)
                if self.thread.is_alive():
                    # Python cannot safely close a running subinterpreter. A
                    # background reaper retains it until the worker exits and then
                    # closes it. If the worker never exits, that interpreter leak is
                    # unavoidable.
                    _LG.warning(
                        "Thread did not terminate gracefully; deferring "
                        "subinterpreter cleanup."
                    )
                    try:
                        self._schedule_deferred_cleanup(owner)
                    except Exception:
                        # Scheduling resets its state for every BaseException, but
                        # control-flow failures still propagate to the caller.
                        _LG.warning(
                            "Failed to schedule subinterpreter cleanup.", exc_info=True
                        )
                    return
                self._close_interpreter()
            finally:
                with self._cleanup_lock:
                    # Scheduling may have atomically transferred or released this
                    # ownership so another caller can retry. Do not clear that
                    # caller's newer ownership from this older finally block.
                    if self._terminate_owner is owner:
                        self._terminate_owner = None

    class _SubinterpreterIterable(Iterable[T]):
        """An Iterable interface that manipulates the iterable in a subinterpreter
        and fetches the results.

        This object supports multiple iterations. Each call to ``__iter__()``
        instructs the worker subinterpreter to create a fresh iterator from the
        underlying iterable (via ``iter(iterable)``), without creating a new
        subinterpreter. The subinterpreter is reused across iterations.
        """

        def __init__(self, interface: _iic[T]) -> None:
            self._if: _iic[T] | None = interface
            self._finalizer = weakref.finalize(self, interface.terminate)

        def __iter__(self) -> Iterator[T]:
            """Enter iteration mode and yield the subinterpreter results."""
            if (if_ := self._if) is None:
                raise RuntimeError(
                    "The subinterpreter is shutdown. Cannot iterate again."
                )

            try:
                _enter_iteration_mode(
                    if_.cmd_q,
                    if_.data_q,
                    if_.timeout,
                    "subinterpreter",
                    is_alive=if_.thread.is_alive,
                )
                yield from _iterate_results(
                    if_.data_q,
                    if_.timeout,
                    "subinterpreter",
                    if_.thread.is_alive,
                )
            except (Exception, KeyboardInterrupt):
                self._terminate()
                raise

        def _terminate(self) -> None:
            if (if_ := self._if) is not None:
                if_.terminate()
                self._finalizer.detach()
                self._if = None

    def _impl(
        fn: Callable[[], Iterable[T]],
        *,
        buffer_size: int = 3,
        initializer: Callable[[], None] | Sequence[Callable[[], None]] | None = None,
        timeout: float | None = None,
    ) -> Iterable[T]:
        initializers = (
            None
            if initializer is None
            else (
                [initializer] if not isinstance(initializer, Sequence) else initializer
            )
        )

        cmd_q = concurrent.interpreters.create_queue()
        data_q = concurrent.interpreters.create_queue(maxsize=buffer_size)
        interp = concurrent.interpreters.create()

        try:
            thread = interp.call_in_thread(
                _execute_iterable, cmd_q, data_q, fn, initializers
            )
        except BaseException as error:
            # No interface/finalizer exists yet, so this scope still owns the
            # interpreter created immediately above. Preserve the primary failure,
            # but retain any cleanup BaseException in its notes and the log.
            try:
                interp.close()
            except BaseException as cleanup_error:  # noqa: B036
                error.add_note(
                    "Closing the orphaned subinterpreter also failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
                _LG.warning("Failed to close orphaned subinterpreter.", exc_info=True)
                raise error from None
            raise

        timeout_ = float("inf") if timeout is None else timeout
        interface = _iic(thread, interp, cmd_q, data_q, timeout_)

        try:
            _wait_for_init(
                interface.data_q,
                interface.timeout,
                "subinterpreter",
                interface.thread.is_alive,
            )
        except BaseException as error:
            # No iterable/finalizer has been returned yet, so setup owns cleanup.
            # Preserve the setup failure, but retain a cleanup BaseException in its
            # notes and the log.
            try:
                interface.terminate()
            except BaseException as cleanup_error:  # noqa: B036
                error.add_note(
                    "Cleaning up the failed subinterpreter initialization also "
                    f"failed: {type(cleanup_error).__name__}: {cleanup_error}"
                )
                _LG.warning(
                    "Failed to clean up subinterpreter after initialization.",
                    exc_info=True,
                )
                raise error from None
            raise

        return _SubinterpreterIterable(interface)


def iterate_in_subinterpreter(
    fn: Callable[[], Iterable[T]],
    *,
    buffer_size: int = 3,
    initializer: Callable[[], None] | Sequence[Callable[[], None]] | None = None,
    timeout: float | None = None,
) -> Iterable[T]:
    """**[Experimental]** Run the given ``iterable`` in a subinterpreter.

    This function behaves similarly to :py:func:`iterate_in_subprocess`, but uses
    Python 3.14's :py:mod:`concurrent.interpreters` module instead of multiprocessing.
    Subinterpreters provide isolation while sharing the same process, which can be
    more lightweight than spawning a separate process.

    The subinterpreter is created once and reused across iterations.
    The returned :py:class:`Iterable` supports multiple iterations —
    each call to ``iter()`` (or ``for ... in``) instructs the worker to
    create a fresh iterator from the underlying iterable without creating
    a new subinterpreter. Reusing the same worker avoids the overhead of
    repeated subinterpreter creation and initializer execution.

    .. note::

       ``fn()`` is called once in the subinterpreter to create the iterable.
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
        initializer: Functions executed in the subinterpreter before iteration starts.
        timeout: Maximum time the caller waits during initialization or for a new
            item. On timeout, cooperative termination is requested. Python cannot
            forcibly interrupt code actively running in a subinterpreter.

    Returns:
        Iterator over the results of the generator function.

    Note:
        - This function requires Python 3.14 or later.
        - The function and the values yielded by the iterator must be
          shareable between interpreters.

    See Also:
        :py:func:`iterate_in_subprocess` for running in a subprocess instead.

    Raises:
        RuntimeError: If Python version is less than 3.14.
    """
    return _impl(fn, buffer_size=buffer_size, initializer=initializer, timeout=timeout)
