# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Implements meta-transformations on iterables/iterators."""

__all__ = ["iterate_in_subprocess", "MergeIterator", "repeat_source"]


import logging
import multiprocessing as mp
import queue
import random
import sys
import time
from collections.abc import (
    Callable,
    Iterable,
    Iterator,
    Sequence,
)
from typing import Any, TypeVar

from ._type import IterableWithShuffle

T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")

_LG: logging.Logger = logging.getLogger(__name__)

# pyre-strict

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


################################################################################
# MergeIterator
################################################################################

_FIRST_EXHAUSTION = -1


def _ordered_iter(iterators: list[Iterator[T]], stop_after: float) -> Iterable[T]:
    num_items = 0
    while iterators:
        remove = []

        for i, iterator in enumerate(iterators):
            try:
                yield next(iterator)
            except StopIteration:
                if stop_after == _FIRST_EXHAUSTION:
                    return
                # Insert in reversed order beacause we use this for popping from list
                remove.insert(0, i)
                continue

            num_items += 1
            if stop_after > 0 and num_items >= stop_after:
                return

        if remove:
            for i in remove:
                iterators.pop(i)


def _stocastic_iter(
    iterators: list[Iterator[T]],
    weights: Sequence[float],
    stop_after: float,
    seed: int,
) -> Iterable[T]:
    # These are all checked in MergeIterator constructor
    assert len(iterators) == len(weights)
    assert all(w >= sys.float_info.epsilon for w in weights)

    population = list(range(len(iterators)))
    rng = random.Random(seed)
    num_items = 0

    not_exhausted = [True for _ in range(len(iterators))]
    while any(not_exhausted):
        for i in rng.choices(population, weights, k=100):
            try:
                yield next(iterators[i])
            except StopIteration:
                not_exhausted[i] = False
                if stop_after == _FIRST_EXHAUSTION:
                    return
                continue

            num_items += 1
            if stop_after > 0 and num_items >= stop_after:
                return


class MergeIterator(Iterable[T]):
    """Iterate over given iterables and yield one item from each iterator.


    Args:
        iterables: The source iterables
        weights: The sampling weight used to choose the next iterable.
            If not provided, the given iterables are visited in the given order
            repeatedly.
        stop_after: Determines the stop criteria or the behavior when one of
            the input iterables gets exhausted,
            Available values are;

            - ``0``: The iteration continues until all the input iterables are
              exhausted. (default)
            - ``n > 0``: The iteration stops when the specified number of items
              are yielded or all the input iterables are exhausted before yielding
              ``n`` items.
            - ``-1``: The iteration stops when one of the iterator is exhausted.
        seed: Used to seed the random generator when ``weights`` is provided.

    Example:

        >>> iterables = [
        ...     [0, 1, 2],
        ...     [10, 11, 12],
        ...     [20, 21, 22],
        ... ]
        >>>
        >>> print(list(MergeIterator(iterables)))
        [0, 10, 20, 1, 11, 21, 2, 12, 22]
        >>>
        >>> # By default, it stops after one iterable gets exhausted.
        >>> iterables = [
        ...     [0, 1, 2],
        ...     [10, 11],
        ...     [20, 21, 22],
        ... ]
        >>>
        >>> print(list(MergeIterator(iterables)))
        [0, 10, 20, 1, 11, 21, 2]  # 22 is not included
        >>>
        >>> # Stop after yielding the given number of items
        >>> print(list(MergeIterator(iterables, stop_after=5)))
        [0, 10, 20, 1, 11]
        >>>
        >>> # stop_after>1 ignores the exhaustion.
        >>> print(list(MergeIterator(iterables, stop_after=9)))
        [0, 10, 20, 1, 11, 21, 2, 22]
        >>>
        >>> # Providing weights will pick up the iterable stocastically.
        >>> print(list(MergeIterator(iterables, stop_after=9, weights=[1, 1, 1])))
        [0, 1, 10, 11, 20, 2, 21, 22]
    """

    def __init__(
        self,
        iterables: Sequence[Iterable[T]],
        *,
        weights: Sequence[float] | None = None,
        stop_after: int = 0,
        seed: int = 0,
    ) -> None:
        if not iterables:
            raise ValueError("iterables cannot be empty.")

        if weights is not None:
            if len(weights) != len(iterables):
                raise ValueError(
                    f"The number of probabilities ({len(weights)}) and "
                    f"iterables ({len(iterables)}) must match."
                )

            # If any of them is 0 or negative, then there is something wrong with
            # user logic, so we raise an exception.
            if any(w < sys.float_info.epsilon for w in weights):
                raise ValueError("Weights must be non-zero and positive.")

        if not stop_after >= -1:
            msg = (
                f"`stop_after` must be greater than or equal to -1. Found: {stop_after}"
            )
            raise ValueError(msg)

        self.iterables = iterables
        self.weights = weights
        self.stop_after = stop_after
        self.seed = seed

    def __iter__(self) -> Iterator[T]:
        iterators = [iter(ite) for ite in self.iterables]

        if self.weights is None:
            yield from _ordered_iter(iterators, self.stop_after)
        else:
            yield from _stocastic_iter(
                iterators, self.weights, self.stop_after, self.seed
            )


################################################################################
# repeat_source
################################################################################


def _repeat(src: Iterable[T] | IterableWithShuffle[T], epoch: int) -> Iterator[T]:
    while True:
        _LG.info("Starting source epoch %d.", epoch)
        t0 = time.monotonic()
        if hasattr(src, "shuffle"):
            src.shuffle(seed=epoch)  # pyre-ignore: [16]
            if (elapsed := time.monotonic() - t0) > 3:
                _LG.warning("Shuffling took %.2f sec.", elapsed)
        num_rows = 0
        for batch in src:
            num_rows += 1
            yield batch
        elapsed = time.monotonic() - t0
        qps = num_rows / elapsed
        _LG.info(
            "Finished source epoch %d. (Yielded %d rows in %.2f sec. QPS: %.2f)",
            epoch,
            num_rows,
            elapsed,
            qps,
        )
        epoch += 1


class _RepeatIterator(Iterator[T]):
    def __init__(
        self,
        src: Iterable[T] | IterableWithShuffle[T],
        epoch: int = 0,
    ) -> None:
        self.src = src
        self.epoch = epoch
        self._iter: Iterator[T] | None = None

    def __iter__(self) -> Iterator[T]:
        return self

    def __getstate__(self) -> dict[str, Any]:  # pyre-ignore: [11]
        if self._iter is not None:
            raise ValueError("Cannot pickle after iteration is started.")
        return self.__dict__

    def __next__(self) -> T:
        if self._iter is None:
            self._iter = _repeat(self.src, self.epoch)
        return next(self._iter)


def repeat_source(
    src: Iterable[T] | IterableWithShuffle[T],
    epoch: int = 0,
) -> Iterator[T]:
    """Convert an iterable into an infinite iterator with optional shuffling.

    Roughly equivalent to the following code snippet.

    .. code-block::

       while True:
           if hasattr(src, "shuffle"):
               src.shuffle(seed=epoch)
           yield from src
           epoch += 1

    Args:
        src: The source to repeat.
        epoch: The epoch number to start with.
    """
    # Returning object so that it can be passed to a subprocess.
    return _RepeatIterator(src, epoch)
