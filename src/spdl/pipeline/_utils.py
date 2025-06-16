# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import asyncio
import atexit
import enum
import logging
import multiprocessing as mp
import queue
import sys
import time
import traceback
from asyncio import Task
from collections import defaultdict
from collections.abc import Callable, Coroutine, Generator, Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

__all__ = [
    "create_task",
    "iterate_in_subprocess",
    "cache_iterator",
]

_LG: logging.Logger = logging.getLogger(__name__)

# Dictionary to track exception counts by file and line number
_exception_counts: dict[tuple[str, int], int] = defaultdict(int)

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
def _log_exception(
    task: Task,
    stacklevel: int,
    log_cancelled: bool,
    suppress_repeated_logs: bool = True,
    suppression_threshold: int = 2,
    suppression_warning_interval: int = 100,
) -> None:
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

        if suppress_repeated_logs and f.filename is not None and f.lineno is not None:
            exception_key = (f.filename, f.lineno)
            _exception_counts[exception_key] += 1
            count = _exception_counts[exception_key]
            if count == suppression_threshold:
                _LG.warning(
                    "Errors are repeated at %s:%d:%s, holding on logging.",
                    f.filename,
                    f.lineno,
                    f.name,
                )
                return
            elif count > suppression_threshold:
                if count % suppression_warning_interval == 0:
                    _LG.warning(
                        "%d errors were logged at %s:%d:%s.",
                        count,
                        f.filename,
                        f.lineno,
                        f.name,
                    )
                    return
                else:
                    return

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
    suppress_repeated_logs: bool = True,
    suppression_threshold: int = 2,
    suppression_warning_interval: int = 100,
) -> Task[T]:
    """Wrapper around :py:func:`asyncio.create_task`. Add logging callback."""
    task = asyncio.create_task(coro, name=name)
    task.add_done_callback(
        lambda t: _log_exception(
            t,
            stacklevel=3,
            log_cancelled=log_cancelled,
            suppress_repeated_logs=suppress_repeated_logs,
            suppression_threshold=suppression_threshold,
            suppression_warning_interval=suppression_warning_interval,
        )
    )
    return task


################################################################################
# iterate_in_subprocess
################################################################################


# Command from parent to worker
class _Cmd(enum.IntEnum):
    """_Cmd()
    Command issued from the parent process to the worker process in
    :py:func:`iterate_in_subprocess`.
    """

    ABORT = enum.auto()
    """Instruct the worker process to abort and exit.

    :meta hide-value:
    """

    START_ITERATION = enum.auto()
    """Instruct the worker process to start the iteration.

    :meta hide-value:
    """

    STOP_ITERATION = enum.auto()
    """Instruct the worker process to stop the ongoing iteration,
    and go back to the stand-by mode.

    If the worker process receive this command in stand-by mode,
    it is sliently ignored.
    (This allows the parent process to be sure that the worker process is
    in the stand-by mode or failure mode, and not in iteration mode.)

    :meta hide-value:
    """


# Final status of the iterator
class _Status(enum.IntEnum):
    """_Status()
    Status reported by the worker process in :py:func:`iterate_in_subprocess`.
    """

    UNEXPECTED_CMD_RECIEVED = enum.auto()
    """Received a command unexpected.

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


def _drain(q: queue.Queue[T]) -> None:
    while not q.empty():
        q.get_nowait()


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


def _execute_iterable(
    cmd_q: mp.Queue,
    data_q: mp.Queue,
    fn: Callable[[], Iterable[T]],
    initializers: Sequence[Callable[[], None]] | None,
) -> None:
    """Worker implementation for :py:func:`iterate_in_subprocess`.

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

           i2 --> i3: Command recieved
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
                    data_q.put(_Msg(_Status.UNEXPECTED_CMD_RECIEVED, str(cmd)))
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
                        data_q.put(_Msg(_Status.UNEXPECTED_CMD_RECIEVED, str(cmd)))
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


def _wait_for_init(interface: _ipc[T]) -> None:
    """_wait_for_init()"""
    wait = min(0.1, interface.timeout)
    t0 = time.monotonic()
    while True:
        try:
            item = interface.data_q.get(timeout=wait)
        except queue.Empty:
            if (elapsed := time.monotonic() - t0) > interface.timeout:
                raise RuntimeError(
                    f"The worker process did not initialize after {elapsed:.2f} seconds."
                ) from None
            continue
        else:
            match item.status:
                case _Status.INITIALIZATION_SUCCEEDED:
                    return
                case _Status.INITIALIZATION_FAILED:
                    raise RuntimeError(f"The worker process quit. {item.message}")
                case _:
                    raise RuntimeError(
                        f"[INTERNAL ERROR] The worker is in an unexpected state: {item}"
                    )


def _enter_iteration_mode(interface: _ipc[T]) -> None:
    """Instruct the worker process to enter iteration mode and wait for the acknowledgement."""

    interface.cmd_q.put(_Cmd.STOP_ITERATION)
    interface.cmd_q.put(_Cmd.START_ITERATION)

    wait = min(0.1, interface.timeout)
    t0 = time.monotonic()
    while True:
        try:
            item = interface.data_q.get(timeout=wait)
            t0 = time.monotonic()
        except queue.Empty:
            if (elapsed := time.monotonic() - t0) > interface.timeout:
                raise RuntimeError(
                    "The worker process did not produce any data for "
                    f"{elapsed:.2f} seconds."
                ) from None
            continue
        else:
            match item.status:
                case _Status.ITERATION_STARTED:
                    # the worker process is properly transitioning to the iteration mode
                    return
                case _Status.ITERATION_FINISHED | _Status.ITERATOR_SUCCESS:
                    # residual from previous iteration. Could be iteration was abandoned, or
                    # the iteration had been completed when parent commanded STOP_ITERATION.
                    continue
                case _Status.UNEXPECTED_CMD_RECIEVED:
                    # the worker was in the invalid state (iteration mode)
                    raise RuntimeError(
                        "The worker process was not in the stand-by mode. Please make sure "
                        "that the previous iterator was exhausted before iterating again. "
                        f"{item.message}"
                    )
                case _:
                    raise RuntimeError(
                        f"The worker process is in an unexpected state. {item}"
                    )


def _iterate_results(interface: _ipc[T]) -> Iterator[T]:
    """Watch the result queue and iterate on the results."""
    wait = min(0.1, interface.timeout)
    t0 = time.monotonic()
    while True:
        try:
            item = interface.data_q.get(timeout=wait)
            t0 = time.monotonic()
        except queue.Empty:
            if (elapsed := time.monotonic() - t0) > interface.timeout:
                raise RuntimeError(
                    "The worker process did not produce any data for "
                    f"{elapsed:.2f} seconds."
                ) from None
            continue
        else:
            match item.status:
                case _Status.ITERATOR_SUCCESS:
                    yield item.message  # pyre-ignore: [7]
                case _Status.ITERATION_FINISHED:
                    return
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


class _SubprocessIterable(Iterable[T]):
    """An Iterable interface that manipulates the iterable in worker process
    and fetch the results."""

    def __init__(self, interface: _ipc[T]) -> None:
        self._interface: _ipc[T] | None = interface

    def __iter__(self) -> Iterator[T]:
        """Instruct the worker process to enter iteration mode and iterate on the results."""
        if (interface := self._interface) is None:
            raise RuntimeError("The worker process is shutdown. Cannot iterate again.")

        try:
            _enter_iteration_mode(interface)
            yield from _iterate_results(interface)
        except (Exception, KeyboardInterrupt):
            self._shutdown()
            raise

    def _shutdown(self) -> None:
        if (interface := self._interface) is not None:
            interface.terminate()
            atexit.unregister(interface.terminate)
            self._interface = None

    def __del__(self) -> None:
        self._shutdown()


def iterate_in_subprocess(
    fn: Callable[[], Iterable[T]],
    *,
    buffer_size: int = 3,
    initializer: Callable[[], None] | Sequence[Callable[[], None]] | None = None,
    mp_context: str | None = None,
    timeout: float | None = None,
    daemon: bool = False,
) -> Iterable[T]:
    """**[Experimental]** Run the given ``iterable`` in a subprocess.

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

    Returns:
        Iterator over the results of the generator function.

    .. note::

       The function and the values yielded by the iterator of generator must be picklable.

    .. seealso::

       - :py:func:`run_pipeline_in_subprocess` for runinng a :py:class:`Pipeline` in
         a subprocess
       - :ref:`parallelism-performance` for the context in which this function was created.


    Implementation Detail
    ---------------------

    Manipulting an iterable object in a subprocess requires somewhat elaborated state
    control.
    The following section go over the implementation detail.

    **Wroker State**

    The iterable object is manipulated in the worker process.
    The worker process has three states, "Initialization", "Stand By" and "Iteration".
    The Initialization state performs global initialization and create the iterable object.
    When the Initialization completes, the worker transition to Stand By mode, where
    it waits for a command from the parent process. The command can be "START_ITERATION"
    or "ABORT".
    When the "START_ITERATION" is received, the worker process transition to the
    Iteration mode. In the Iteration mode, the worker creates an iterator object from
    the iterable, then executes it.
    The resulting data are put in the queue, which the parent process is watching.

    The following diagram illustrates worker's state transition in simplified manner.
    Detailed diagram alongside the actual implementation is found in
    :py:func:`~spdl.pipeline._execute_iterable`.

    .. mermaid::

       stateDiagram-v2
        state Parent {
            p1: Start Iteration
            p2: Iterate on the result
            state pf <<fork>>
            state pj <<join>>

            [*] --> p1
            p1 --> pf
            pf --> pj: Wait for worker process
            pj -->  p2
            p2 --> [*]
        }

        state Worker {
            state wf <<fork>>
            w0: Initialization
            w1: Stand By
            w2: Iteration

            [*]--> w0
            w0 --> w1: Success
            w0 --> [*]: Fail
            w1 --> wf: Iteration started
            wf --> w2
            w2 --> w1: Iteration completed

            w1 --> [*]: Abort
            w2 --> [*]: Fail / Abort
        }
        pf --> w1: Issue START_ITERATION command
        wf --> pj: Notify ITERATION_STARTED
        w2 --> p2: Results passed via queue

    Helper functions and data structures
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    The follosing functions and data structures are used to implement the
    :py:func:`iterate_in_subprocess` function.
    They are not public interface, but the logic is sufficiently elaborated,
    and it is helpful to have them in the documentation, so they are listed here.

    .. autoclass:: _Cmd
       :noindex:
       :members:

    .. autoclass:: _Status
       :noindex:
       :members:

    .. autofunction:: _execute_iterable()
       :noindex:

    .. autofunction:: _enter_iteration_mode()
       :noindex:

    .. autofunction:: _iterate_results()
       :noindex:

    .. autoclass:: _SubprocessIterable()
       :noindex:
       :members: __iter__
    """
    initializers = (
        None
        if initializer is None
        else ([initializer] if not isinstance(initializer, Sequence) else initializer)
    )

    ctx = mp.get_context(mp_context)
    cmd_q: queue.Queue[_Cmd] = ctx.Queue()
    data_q: queue.Queue[_Msg[T]] = ctx.Queue(maxsize=buffer_size)
    process = ctx.Process(
        target=_execute_iterable,
        args=(cmd_q, data_q, fn, initializers),
        daemon=daemon,
    )

    interface = _ipc(
        process, cmd_q, data_q, float("inf") if timeout is None else timeout
    )

    # Register an exit callback in case Python tries to exit while the subprocess
    # is blocked on the data_q.put.
    atexit.register(interface.terminate)

    process.start()

    _wait_for_init(interface)  # Block until the initialization is completed

    return _SubprocessIterable(interface)


#######################################################################################
# cache_iterator
#######################################################################################


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
