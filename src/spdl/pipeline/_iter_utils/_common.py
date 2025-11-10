# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Common utilities for subprocess and subinterpreter-based iteration.

This module contains shared functionality used by both iterate_in_subprocess
and iterate_in_subinterpreter implementations.
"""

import enum
import multiprocessing as mp
import queue
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar

__all__ = [
    "_Cmd",
    "_Status",
    "_Msg",
    "_drain",
    "_wait_for_init",
    "_enter_iteration_mode",
    "_iterate_results",
    "_execute_iterable",
]

T = TypeVar("T")


class _Queue(Protocol[T]):
    """Protocol for queue-like objects used in subprocess and subinterpreter communication.

    This protocol defines the common interface for both multiprocessing.Queue
    and concurrent.interpreters.Queue, allowing code reuse between subprocess
    and subinterpreter implementations.
    """

    def put(self, item: T, block: bool = True, timeout: float | None = None) -> None:
        """Put an item into the queue."""
        ...

    def get(self, block: bool = True, timeout: float | None = None) -> T:
        """Remove and return an item from the queue."""
        ...

    def get_nowait(self) -> T:
        """Remove and return an item if one is immediately available, else raise Empty."""
        ...

    def empty(self) -> bool:
        """Return True if the queue is empty, False otherwise."""
        ...


# Command from parent to worker
class _Cmd(enum.IntEnum):
    """Command issued to the worker."""

    ABORT = enum.auto()
    """Instruct the worker to abort and exit.

    :meta hide-value:
    """

    START_ITERATION = enum.auto()
    """Instruct the worker to start the iteration.

    :meta hide-value:
    """

    STOP_ITERATION = enum.auto()
    """Instruct the worker to stop the ongoing iteration,
    and go back to the stand-by mode.

    If the worker receive this command in stand-by mode,
    it is sliently ignored.
    (This allows the parent process to be sure that the worker is
    in the stand-by mode or failure mode, and not in iteration mode.)

    :meta hide-value:
    """


# Final status of the iterator
class _Status(enum.IntEnum):
    """Status reported by the worker."""

    UNEXPECTED_CMD_RECEIVED = enum.auto()
    """Received an unexpected command.

    :meta hide-value:
    """

    INITIALIZATION_FAILED = enum.auto()
    """Initialization (global, or creation of iterable) failed.

    :meta hide-value:
    """

    INITIALIZATION_SUCCEEDED = enum.auto()
    """Initialization succeeded.
    The worker process is transitioning to stand-by mode.

    :meta hide-value:
    """

    ITERATION_STARTED = enum.auto()
    """The worker is transitioning to iteration mode.

    :meta hide-value:
    """

    ITERATION_FINISHED = enum.auto()
    """The worker finished an iteration, transitioning back to the stand-by mode.

    Note that this will be sent in both cases where the the iterator exhausted
    or the parent process issued ``STOP_ITERATION`` command.

    :meta hide-value:
    """

    ITERATOR_SUCCESS = enum.auto()
    """One step of iteration has been succeeded.

    :meta hide-value:
    """
    ITERATOR_FAILED = enum.auto()
    """There was an error in an iteration step.

    :meta hide-value:
    """


# Message from worker to the parent
@dataclass
class _Msg(Generic[T]):
    status: _Status
    # additional data
    # String in case of failure
    message: T | str = ""


def _drain(q: _Queue[Any]) -> None:
    """Drain a queue by removing all items.

    Works with both :py:class:`multiprocessing.Queue`
    and :py:class:`concurrent.interpreters.Queue`.
    """
    while True:
        try:
            q.get_nowait()
        except queue.Empty:
            break


def _execute_iterable(
    cmd_q: mp.Queue,
    data_q: mp.Queue,
    fn: Callable[[], Iterable[T]],
    initializers: Sequence[Callable[[], None]] | None,
) -> None:
    """Worker implementation for :py:func:`~spdl.pipeline.iterate_in_subprocess`
    and :py:func:`~spdl.pipeline.iterate_in_subinterpreter`.

    The following diagram illustrates the state transition with more details.

    .. mermaid::

       stateDiagram-v2
           state Initialization {
               init0: Call Initializer
               init1: Create Iterable
               init0 --> init1
           }
           state stand_by {
               i2: Wait for a command (block)
               state i3 <<choice>>
           }
           stand_by: Stand By

           init_fail: Push INITIALIZATION_FAILED
           init_success: Push INITIALIZATION_SUCCESS
           i4: Push ITERATION_START
           i5: Iteration
           i6: Push UNEXPECTED_COMMAND
           init_fail-->Done

           [*] --> Initialization
           Initialization --> init_success: Success
           Initialization --> init_fail: Failed
           init_success --> i2

           i2 --> i3: Command received
           i3 --> i4: START_ITERATION
           i3 --> i2: STOP_ITERATION
           i3 --> i6: Other commands
           i3 --> Done: ABORT
           i4 --> j0
           i6 --> Done
           Done --> [*]

           state i5 {
               j0: Create Iterator
               j1: Check command queue (non-blocking)
               j2: Get Next Item
               state j3 <<choice>>
               j4: Push ITERATE_SUCCESS (block)
               j5: Push ITERATION_FINISHED
               j6: Push ITERATE_FAILED

               j0 --> j1 : Success
               j0 --> j6 : Fail
               j1 --> j2 : Command Not Found
               j1 --> j3: Command Found
               j3 --> j5 : STOP_ITERATION

               j2 --> j4 : Success
               j2 --> j5 : EOF
               j2 --> j6 : Fail
               j4 --> j1
               j5 --> [*] : Iteration completed without and error (go back to Stand By)
           }
           j3 --> i6 : START_ITERATION or other commands
           j3 --> Done : ABORT
           j6 --> Done
           i5 --> i2: Iteration completed without an error
    """
    if initializers is not None:
        try:
            for initializer in initializers:
                initializer()
        except Exception as e:
            data_q.put(
                _Msg(
                    _Status.INITIALIZATION_FAILED,
                    message=f"Initializer failed: {e}",
                )
            )
            return

    try:
        iterable = fn()
    except Exception as e:
        data_q.put(
            _Msg(
                _Status.INITIALIZATION_FAILED,
                message=f"Failed to create the iterable: {e}",
            )
        )
        return

    data_q.put(_Msg(_Status.INITIALIZATION_SUCCEEDED))

    while True:
        # Stan-by: Waiting for a command from parent
        try:
            cmd = cmd_q.get(timeout=1)
        except queue.Empty:
            continue
        else:
            match cmd:
                case _Cmd.START_ITERATION:
                    data_q.put(_Msg(_Status.ITERATION_STARTED))
                case _Cmd.STOP_ITERATION:
                    continue
                case _Cmd.ABORT:
                    return
                case _:
                    data_q.put(_Msg(_Status.UNEXPECTED_CMD_RECEIVED, str(cmd)))
                    return

        # One iteration
        try:
            gen = iter(iterable)
        except Exception as e:
            data_q.put(
                _Msg(
                    _Status.ITERATOR_FAILED,
                    message=f"Failed to create the iterator: {e}",
                )
            )
            return

        # Iterate until: Finish, an error, or abort
        while True:
            try:
                cmd = cmd_q.get_nowait()
            except queue.Empty:
                pass
            else:
                match cmd:
                    case _Cmd.STOP_ITERATION:
                        data_q.put(_Msg(_Status.ITERATION_FINISHED))
                        break
                    case _Cmd.ABORT:
                        return
                    case _:
                        data_q.put(_Msg(_Status.UNEXPECTED_CMD_RECEIVED, str(cmd)))
                        return

            try:
                item = next(gen)
                data_q.put(_Msg(_Status.ITERATOR_SUCCESS, message=item))
            except StopIteration:
                data_q.put(_Msg(_Status.ITERATION_FINISHED))
                break
            except Exception as e:
                data_q.put(
                    _Msg(
                        _Status.ITERATOR_FAILED,
                        message=f"Failed to fetch the next item: {e}",
                    )
                )
                return


def _wait_for_init(data_q: _Queue[_Msg[T]], timeout: float, worker_type: str) -> None:
    """Wait for initialization to complete.

    Works with both multiprocessing.Queue and concurrent.interpreters.Queue.

    Args:
        data_q: Queue to receive initialization status messages
        timeout: Maximum time to wait for initialization
        worker_name: Name of the worker (for error messages)
    """
    wtype = f"worker {worker_type}"
    wait = min(0.1, timeout)
    t0 = time.monotonic()
    while True:
        try:
            item = data_q.get(timeout=wait)
        except queue.Empty:
            if (elapsed := time.monotonic() - t0) > timeout:
                raise RuntimeError(
                    f"The {wtype} did not initialize after {elapsed:.2f} seconds."
                ) from None
            continue
        else:
            match item.status:
                case _Status.INITIALIZATION_SUCCEEDED:
                    return
                case _Status.INITIALIZATION_FAILED:
                    raise RuntimeError(f"The {wtype} quit. {item.message}")
                case _:
                    raise RuntimeError(
                        f"[INTERNAL ERROR] The {wtype} is in an unexpected state: {item}"
                    )


def _enter_iteration_mode(
    cmd_q: _Queue[_Cmd],
    data_q: _Queue[_Msg[Any]],
    timeout: float,
    worker_type: str,
) -> None:
    """Instruct the worker to enter iteration mode and wait for the acknowledgement.

    Works with both :py:class:`multiprocessing.Queue` and
    :py:class:`concurrent.interpreters.Queue.`

    Args:
        cmd_q: Queue to send commands to the worker
        data_q: Queue to receive status messages from the worker
        timeout: Maximum time to wait for acknowledgement
        worker_type: Type of worker (for error messages)
    """
    wtype = f"worker {worker_type}"
    cmd_q.put(_Cmd.STOP_ITERATION)
    cmd_q.put(_Cmd.START_ITERATION)

    wait = min(0.1, timeout)
    t0 = time.monotonic()
    while True:
        try:
            item = data_q.get(timeout=wait)
            t0 = time.monotonic()
        except queue.Empty:
            if (elapsed := time.monotonic() - t0) > timeout:
                raise RuntimeError(
                    f"The {wtype} did not produce any data for {elapsed:.2f} seconds."
                ) from None
            continue
        else:
            match item.status:
                case _Status.ITERATION_STARTED:
                    # the worker is properly transitioning to the iteration mode
                    return
                case _Status.ITERATION_FINISHED | _Status.ITERATOR_SUCCESS:
                    # residual from previous iteration. Could be iteration was abandoned, or
                    # the iteration had been completed when parent commanded STOP_ITERATION.
                    continue
                case _Status.UNEXPECTED_CMD_RECEIVED:
                    # the worker was in the invalid state (iteration mode)
                    raise RuntimeError(
                        f"The {wtype} was not in the stand-by mode. Please make sure "
                        "that the previous iterator was exhausted before iterating again. "
                        f"{item.message}"
                    )
                case _:
                    raise RuntimeError(f"The {wtype} is in an unexpected state. {item}")


def _iterate_results(
    data_q: _Queue[_Msg[T]], timeout: float, worker_type: str
) -> Iterable[T]:
    """Watch the result queue and iterate on the results.

    Works with both multiprocessing.Queue and concurrent.interpreters.Queue.

    Args:
        data_q: Queue to receive iteration results
        timeout: Maximum time to wait between results
        worker_name: Name of the worker (for error messages)

    Yields:
        Items from the iterator
    """
    wtype = f"worker {worker_type}"
    wait = min(0.1, timeout)
    t0 = time.monotonic()
    while True:
        try:
            item = data_q.get(timeout=wait)
            t0 = time.monotonic()
        except queue.Empty:
            if (elapsed := time.monotonic() - t0) > timeout:
                raise RuntimeError(
                    f"The {wtype} did not produce any data for {elapsed:.2f} seconds."
                ) from None
            continue
        else:
            match item.status:
                case _Status.ITERATOR_SUCCESS:
                    yield item.message  # pyre-ignore: [7]
                case _Status.ITERATION_FINISHED:
                    return
                case _Status.ITERATOR_FAILED:
                    raise RuntimeError(f"The {wtype} quit. {item.message}")
                case _Status.UNEXPECTED_CMD_RECEIVED:
                    raise RuntimeError(
                        f"[INTERNAL ERROR] The {wtype} received unexpected command: "
                        f"{item.message}"
                    )
                case _:
                    raise RuntimeError(
                        f"[INTERNAL ERROR] Unexpected return value: {item}"
                    )
