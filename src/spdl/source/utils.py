# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Implements meta-transformations on iterables/iterators."""

__all__ = ["MergeIterator", "repeat_source", "embed_shuffle"]


import logging
import math
import random
import time
from bisect import bisect_right
from collections.abc import Iterable, Iterator, Sequence, Sized
from itertools import accumulate
from typing import overload, TypeVar

from ._type import IterableWithShuffle, SizedIterable, SizedIterableWithShuffle

T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")

_LG: logging.Logger = logging.getLogger(__name__)


################################################################################
# MergeIterator
################################################################################

_FIRST_EXHAUSTION = -1
_MAX_RANDRANGE_BITS = 256


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
                # Insert in reversed order because we use this for popping from list
                remove.insert(0, i)
                continue

            num_items += 1
            if stop_after > 0 and num_items >= stop_after:
                return

        if remove:
            for i in remove:
                iterators.pop(i)


def _exact_integer_weights(weights: Sequence[float]) -> list[int]:
    """Build proportional integers from validated binary-float weights."""
    # The constructor converts runtime numeric values to float, matching the
    # annotated API. Their denominators are powers of two, so the largest
    # denominator is a common denominator for every binary-float ratio.
    ratios = [weight.as_integer_ratio() for weight in weights]
    common_denominator = max(denominator for _, denominator in ratios)
    integer_weights = [
        numerator * (common_denominator // denominator)
        for numerator, denominator in ratios
    ]
    common_factor = math.gcd(*integer_weights)
    return [weight // common_factor for weight in integer_weights]


def _bernoulli_ratio(rng: random.Random, numerator: int, denominator: int) -> bool:
    """Draw an exact Bernoulli ratio without a denominator-sized random int."""
    remainder = numerator
    while True:
        remainder *= 2
        threshold_bit = remainder >= denominator
        if threshold_bit:
            remainder -= denominator
        random_bit = rng.getrandbits(1)
        if random_bit != threshold_bit:
            return random_bit < threshold_bit


def _weighted_index(
    rng: random.Random,
    weights: Sequence[int],
    cumulative_weights: Sequence[int],
) -> int:
    """Choose an index exactly while bounding typical per-draw RNG work."""
    total_weight = cumulative_weights[-1]
    if total_weight.bit_length() <= _MAX_RANDRANGE_BITS:
        return bisect_right(cumulative_weights, rng.randrange(total_weight))

    # Extremely different float exponents can produce thousand-bit totals.
    # Compare a uniform binary fraction lazily in that case. Each Bernoulli
    # comparison consumes two random bits on average, independent of the
    # integer weights' bit width.
    remaining_weight = total_weight
    for i, weight in enumerate(weights[:-1]):
        if _bernoulli_ratio(rng, weight, remaining_weight):
            return i
        remaining_weight -= weight
    return len(weights) - 1


def _stochastic_iter(
    iterators: list[Iterator[T]],
    weights: Sequence[float],
    stop_after: float,
    seed: int,
) -> Iterable[T]:
    # These are all checked in MergeIterator constructor
    assert len(iterators) == len(weights)
    assert all(math.isfinite(w) and w > 0 for w in weights)

    # Convert each validated binary float to proportional exact integers once.
    # Removing an exhausted source preserves every surviving weight ratio, so
    # there is no need to repeat the potentially expensive bigint conversion.
    integer_weights = _exact_integer_weights(weights)
    rng = random.Random(seed)
    num_items = 0

    while iterators:
        if len(iterators) == 1:
            # Selection is deterministic once only one source remains. Avoid
            # bigint RNG work and leave the caller's RNG sequence untouched.
            for item in iterators[0]:
                yield item
                num_items += 1
                if stop_after > 0 and num_items >= stop_after:
                    return
            return

        cumulative_weights = list(accumulate(integer_weights))
        while True:
            i = _weighted_index(rng, integer_weights, cumulative_weights)
            try:
                yield next(iterators[i])
            except StopIteration:
                if stop_after == _FIRST_EXHAUSTION:
                    return
                iterators.pop(i)
                integer_weights.pop(i)
                break

            num_items += 1
            if stop_after > 0 and num_items >= stop_after:
                return


class MergeIterator(Iterable[T]):
    """Iterate over given iterables and yield one item from each iterator.


    Args:
        iterables: The source iterables
        weights: The sampling weight used to choose the next iterable.
            Values are interpreted with binary-float semantics, and selection
            remains exact with respect to those converted values.
            Sources with zero weight are skipped.
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
        >>> iterables = [
        ...     [0, 1, 2],
        ...     [10, 11, 12],
        ...     [20, 21, 22],
        ... ]
        >>> print(sorted(MergeIterator(iterables, stop_after=9, weights=[1, 1, 1])))
        [0, 1, 2, 10, 11, 12, 20, 21, 22]

    .. versionchanged:: 0.7.0
       Exhausted weighted sources are removed from future draws, and invalid
       weight totals now raise :class:`ValueError`. Weighted selection preserves
       the exact ratios of the converted binary-float weights.
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

        if not stop_after >= -1:
            msg = (
                f"`stop_after` must be greater than or equal to -1. Found: {stop_after}"
            )
            raise ValueError(msg)

        # Skip iterables with zero weight
        self.iterables = iterables
        self.weights = weights

        if self.weights is not None:
            if len(self.weights) != len(iterables):
                raise ValueError(
                    f"The number of probabilities ({len(self.weights)}) and "
                    f"iterables ({len(iterables)}) must match."
                )
            try:
                float_weights = [float(weight) for weight in self.weights]
            except (OverflowError, TypeError, ValueError) as exc:
                raise ValueError(
                    "Weights must be finite and non-negative; NaN, infinity, "
                    "and negative values are not supported."
                ) from exc
            if any(not math.isfinite(w) or w < 0 for w in float_weights):
                raise ValueError(
                    "Weights must be finite and non-negative; NaN, infinity, "
                    "and negative values are not supported."
                )

            total_weight = sum(float_weights)
            if not math.isfinite(total_weight) or total_weight <= 0:
                raise ValueError("The sum of weights must be positive and finite.")

            nnz_indices = [i for i, w in enumerate(float_weights) if w != 0]
            self.iterables = [iterables[i] for i in nnz_indices]
            self.weights = [float_weights[i] for i in nnz_indices]

        self.stop_after = stop_after
        self.seed = seed

    def __iter__(self) -> Iterator[T]:
        iterators = [iter(ite) for ite in self.iterables]

        if self.weights is None:
            yield from _ordered_iter(iterators, self.stop_after)
        else:
            yield from _stochastic_iter(
                iterators, self.weights, self.stop_after, self.seed
            )


################################################################################
# embed_shuffle
################################################################################


class _ShuffleAndIterate(Iterable[T]):
    def __init__(
        self,
        src: IterableWithShuffle[T] | SizedIterableWithShuffle[T],
        *,
        epoch: int,
        shuffle_last: bool,
    ) -> None:
        self.src = src
        self._epoch = epoch
        self._shuffle_first: bool = not shuffle_last

    def _shuffle(self) -> None:
        t0 = time.monotonic()
        self.src.shuffle(seed=self._epoch)
        if (elapsed := time.monotonic() - t0) > 3:
            _LG.warning("Shuffling took %.2f sec.", elapsed)
        self._epoch += 1

    def __iter__(self) -> Iterator[T]:
        if self._shuffle_first:
            self._shuffle()
            yield from self.src
        else:
            try:
                yield from self.src
            finally:
                # in case the iteration is stopped in the middle.
                # shuffle is called when the iterator is deleted.
                self._shuffle()

    def __len__(self) -> int:
        if isinstance(self.src, Sized):
            return len(self.src)
        else:
            raise TypeError(
                f"Source iterator of type {type(self.src)} does not support length"
            )


@overload
def embed_shuffle(
    src: SizedIterableWithShuffle[T], /, *, shuffle_last: bool = False, epoch: int = 0
) -> SizedIterable[T]: ...


@overload
def embed_shuffle(
    src: IterableWithShuffle[T], /, *, shuffle_last: bool = False, epoch: int = 0
) -> Iterable[T]: ...


def embed_shuffle(
    src: IterableWithShuffle[T] | SizedIterableWithShuffle[T],
    /,
    *,
    shuffle_last: bool = False,
    epoch: int = 0,
) -> Iterable[T] | SizedIterable[T]:
    """**[Experimental]** Convert :py:class:`~spdl.source.IterableWithShuffle` to
    :py:class:`Iterable` by embedding the :py:meth:`~spdl.source.IterableWithShuffle.shuffle`
    call into :py:meth:`~Iterable.__iter__`.


    Roughly equivalent to the following code snippet.

    .. code-block::

       while True:
            if not shuffle_last:
                src.shuffle(seed=epoch)

            yield from src
            epoch += 1

            if shuffle_last:
                src.shuffle(seed=epoch)

    Args:
        src: The original iterable with ``shuffle`` method.
        shuffle_last: If ``False`` (default), then ``shuffle`` is called
            before the iteration. Other wise ``shuffle`` is called
            at the end of iteration.
        epoch: The initial seed value passed to
            :py:meth:`~spdl.source.IterableWithShuffle.shuffle`.

    """
    return _ShuffleAndIterate(src, epoch=epoch, shuffle_last=shuffle_last)


################################################################################
# repeat_source
################################################################################


def _repeat(src: Iterable[T] | IterableWithShuffle[T], epoch: int) -> Iterator[T]:
    while True:
        _LG.info("Starting source epoch %d.", epoch)
        t0 = time.monotonic()
        if isinstance(src, IterableWithShuffle):
            src.shuffle(seed=epoch)
        num_rows = 0
        for batch in src:
            num_rows += 1
            yield batch
        elapsed = time.monotonic() - t0
        qps = num_rows / elapsed if elapsed > 0 else float("nan")
        _LG.info(
            "Finished source epoch %d. (Yielded %d rows in %.2f sec. QPS: %.2f)",
            epoch,
            num_rows,
            elapsed,
            qps,
        )
        epoch += 1


class _RepeatIterator(Iterator[T]):
    def __init__(self, src: Iterable[T], epoch: int) -> None:
        self.src = src
        self.epoch = epoch
        self._iter: Iterator[T] | None = None

    def __iter__(self) -> Iterator[T]:
        return self

    def __getstate__(self) -> dict[str, object]:
        if self._iter is not None:
            raise ValueError("Cannot pickle after iteration is started.")
        return self.__dict__

    def __next__(self) -> T:
        if self._iter is None:
            self._iter = _repeat(self.src, self.epoch)
        return next(self._iter)


def repeat_source(
    src: Iterable[T] | IterableWithShuffle[T],
    /,
    epoch: int = 0,
) -> Iterator[T]:
    """Convert an iterable into an infinite iterator with optional shuffling.

    Roughly equivalent to the following code snippet.

    .. code-block::

       while True:
           if isinstance(src, IterableWithShuffle):
               src.shuffle(seed=epoch)
           yield from src
           epoch += 1

    Args:
        src: The source to repeat.
        epoch: The epoch number to start with.

    .. versionchanged:: 0.7.0
       ``epoch`` now also controls the first shuffle seed.
    """
    # Returning object so that it can be passed to a subprocess.
    return _RepeatIterator(src, epoch)
