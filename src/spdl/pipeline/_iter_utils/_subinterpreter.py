# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Subinterpreter-based iteration support.

This module provides functionality to run iterables in Python subinterpreters
using Python 3.14's concurrent.interpreters module.
"""

import logging
import sys
import threading
import weakref
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
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
        interpreter: "concurrent.interpreters.Interpreter"
        cmd_q: "concurrent.interpreters.Queue"
        data_q: "concurrent.interpreters.Queue"
        timeout: float

        def terminate(self) -> None:
            self.cmd_q.put(_Cmd.ABORT)
            _drain(self.data_q)
            self.thread.join(timeout=3)
            if self.thread.is_alive():
                _LG.warning("Thread did not terminate gracefully")

    class _SubinterpreterIterable(Iterable[T]):
        """An Iterable interface that manipulates the iterable in a subinterpreter
        and fetches the results."""

        def __init__(self, interface: _iic[T]) -> None:
            self._if: _iic[T] | None = interface
            self._finalizer = weakref.finalize(self, interface.terminate)

        def __iter__(self) -> Iterator[T]:
            """Instruct the subinterpreter to enter iteration mode and iterate on the results."""
            if (if_ := self._if) is None:
                raise RuntimeError(
                    "The subinterpreter is shutdown. Cannot iterate again."
                )

            try:
                _enter_iteration_mode(
                    if_.cmd_q, if_.data_q, if_.timeout, "subinterpreter"
                )
                yield from _iterate_results(if_.data_q, if_.timeout, "subinterpreter")
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

        thread = interp.call_in_thread(
            _execute_iterable, cmd_q, data_q, fn, initializers
        )

        timeout_ = float("inf") if timeout is None else timeout
        interface = _iic(thread, interp, cmd_q, data_q, timeout_)

        _wait_for_init(interface.data_q, interface.timeout, "subinterpreter")

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
    Python 3.14's ``concurrent.interpreters`` module instead of multiprocessing.
    Subinterpreters provide isolation while sharing the same process, which can be
    more lightweight than spawning a separate process.

    Args:
        fn: Function that returns an iterator. Use :py:func:`functools.partial` to
            pass arguments to the function.
        buffer_size: Maximum number of items to buffer in the queue.
        initializer: Functions executed in the subinterpreter before iteration starts.
        timeout: Timeout for inactivity. If the generator function does not yield
            any item for this amount of time, the subinterpreter is terminated.

    Returns:
        Iterator over the results of the generator function.

    Note:
        This function requires Python 3.14 or later. The function and the values
        yielded by the iterator must be shareable between interpreters.

    See Also:
        :py:func:`iterate_in_subprocess` for running in a subprocess instead.

    Raises:
        RuntimeError: If Python version is less than 3.14.
    """
    return _impl(fn, buffer_size=buffer_size, initializer=initializer, timeout=timeout)
