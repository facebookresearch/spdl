# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

__all__ = [
    "DistributedRandomSampler",
    "DistributedDeterministicSampler",
]

import logging
import warnings
from collections.abc import Iterator
from typing import Any, TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    # pyrefly: ignore [bad-specialization]
    FloatArray: TypeAlias = NDArray[np.floating[Any]]


else:
    from spdl._internal.import_utils import lazy_import

    np = lazy_import("numpy")
    FloatArray = object

_LG: logging.Logger = logging.getLogger(__name__)


def _validate_args_det(
    n: int,
    rank: int,
    world_size: int,
    ddp_drop_last_distributed_round: bool,
) -> None:
    if n <= 0:
        raise ValueError(f"The size of dataset must be larger than 0. Found: {n}")

    if rank >= world_size:
        raise ValueError(f"rank ({rank}) must be with the range of [0, {world_size=}).")

    if ddp_drop_last_distributed_round and n < world_size:
        raise ValueError(
            f"The size of dataset ({n}) must be larger than or equal to "
            f"the world size ({world_size}) when ddp_drop_last_distributed_round=True"
        )

    if ddp_drop_last_distributed_round and n % world_size:
        warnings.warn(
            f"The size of dataset ({n}) is not divisible by "
            f"the world size ({world_size}). Some samples are never visited.",
            stacklevel=3,
        )


class DistributedDeterministicSampler:
    """Sampler for distributed training that splits indices across multiple ranks.

    This sampler ensures that each rank in a distributed training setup gets a disjoint
    subset of the data indices. When distributed training is not initialized, it returns
    all indices.

    The iteration order is deterministic and always the same: indices are assigned
    in a round-robin fashion (``range(rank, N, world_size)``). Every iteration
    produces the identical sequence. If you need a different order each epoch,
    use :py:class:`DistributedRandomSampler` instead.

    When the dataset size is not divisible by ``world_size``, the final round is
    incomplete. The ``ddp_drop_last_distributed_round`` argument controls how this
    leftover is handled: when ``True`` (default), the incomplete final round is
    dropped so every rank receives the same number of indices; when ``False``,
    every sample is covered, so some ranks receive one more index than others.

    .. code-block:: text

       Dataset indices, N = 11, world_size = 4
       ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐
       │ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │ 8 │ 9 │10 │
       └───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘

       ddp_drop_last_distributed_round=True (default)
       rank 0: 0, 4
       rank 1: 1, 5
       rank 2: 2, 6
       rank 3: 3, 7       (8, 9, 10 are dropped)

       ddp_drop_last_distributed_round=False
       rank 0: 0, 4, 8
       rank 1: 1, 5, 9
       rank 2: 2, 6, 10
       rank 3: 3, 7       (all indices are covered)

    Args:
        n: The size of the dataset.

        rank: The rank in the distributed communication config.
            You can fetch the values with :py:func:`torch.distributed.get_rank`.

        world_size: The number of ranks in the distributed communication config.
            You can fetch the values with :py:func:`torch.distributed.get_world_size`.

        ddp_drop_last_distributed_round: If ``True`` (default), drop the final
            incomplete distributed round so every rank receives the same number
            of indices. If ``False``, cover every sample exactly once across
            ranks; rank lengths may differ by at most one.

    .. versionadded:: 0.5.0
       The ``ddp_drop_last_distributed_round`` argument.
    """

    def __init__(
        self,
        n: int,
        /,
        *,
        rank: int,
        world_size: int,
        ddp_drop_last_distributed_round: bool = True,
    ) -> None:
        _validate_args_det(n, rank, world_size, ddp_drop_last_distributed_round)

        self._rank = rank
        self._world_size = world_size

        if ddp_drop_last_distributed_round:
            self._len: int = n // world_size
            self._total: int = self._len * world_size
        else:
            self._len = n // world_size + int(rank < n % world_size)
            self._total = n

    def __len__(self) -> int:
        """The number of indices returned by this sampler."""
        return self._len

    def __iter__(self) -> Iterator[int]:
        """Iterate over the indices assigned to the current rank.

        Yields:
            Individual indices assigned to the current rank.
        """
        yield from range(self._rank, self._total, self._world_size)


def _weighted_choice(
    population_size: int,
    num_draws: int,
    weights: FloatArray | None,
    replace: bool = True,
    seed: int = 0,
) -> FloatArray:
    """Generate indices using weighted random sampling.

    Args:
        population_size: The number of elements to choose from (0 to size-1).
        num_draws: The number of samples to generate.
        weights: Optional weights for each element. If None, uniform sampling is used.
        replace: Whether to sample with replacement. Defaults to True.
        seed: Random seed for reproducible results.

    Returns:
        Array of num_draws sampled indices.
    """
    rng = np.random.default_rng(seed)
    return rng.choice(
        population_size,
        size=num_draws,
        replace=replace,
        p=weights,
    )


def _validate_args_random(
    n: int,
    rank: int,
    world_size: int,
    num_draws: int | None,
    weights: list[float] | None,
    ddp_drop_last_distributed_round: bool,
) -> FloatArray | None:
    if n <= 0:
        raise ValueError(f"The size of dataset must be larger than 0. Found: {n}")

    if rank >= world_size:
        raise ValueError(f"rank ({rank}) must be with the range of [0, {world_size=}).")

    if ddp_drop_last_distributed_round and n < world_size:
        raise ValueError(
            f"The size of dataset ({n}) must be larger than or equal to "
            f"the world size ({world_size}) when ddp_drop_last_distributed_round=True"
        )

    if num_draws is not None:
        if num_draws <= 0:
            raise ValueError(
                f"`num_draws` must be greater than zero. Found: {num_draws}"
            )

        if ddp_drop_last_distributed_round and num_draws < world_size:
            raise ValueError(
                "`num_draws` must be greater than or equal to `world_size` "
                f"when ddp_drop_last_distributed_round=True. Found: {num_draws=}, {world_size=}"
            )

    if weights is None:
        return None
    else:
        if (s := len(weights)) != n:
            raise ValueError(
                f"When provided, the number of elements in `weights` ({s})"
                f" must be same as the size of dataset ({n})"
            )

        w = np.array(weights, dtype=np.float64)

        if np.any(w < 0) or not np.all(np.isfinite(w)):
            raise ValueError(
                "Failed to normalize the sample weight. "
                "Some elements are negative or not finite."
            )

        w /= np.sum(w)

        return w


class DistributedRandomSampler:
    """Sample dataset indices for the given distributed node while applying randomness.

    This sampler ensures that each rank in a distributed training setup gets a disjoint
    subset of the data indices. When distributed training is not initialized, it returns
    all indices.

    .. admonition:: Example - Distributed sampling
       :class: note

       The samplers from all ranks together cover the entire dataset.

       >>> N = 9
       >>> sampler = DistributedRandomSampler(N, rank=0, world_size=3)
       >>> list(sampler)
       [3, 2, 7]
       >>> sampler = DistributedRandomSampler(N, rank=1, world_size=3)
       >>> list(sampler)
       [6, 1, 4]
       >>> sampler = DistributedRandomSampler(N, rank=2, world_size=3)
       >>> list(sampler)
       [5, 8, 0]

    Without calling :py:meth:`shuffle`, the sampler produces the **same** sequence
    on every iteration. To get a different order each epoch, call
    ``sampler.shuffle(seed=epoch)`` before iterating:

    .. admonition:: Example
       :class: note

       >>> sampler = DistributedRandomSampler(5, rank=0, world_size=1)
       >>> list(sampler)
       [4, 2, 0, 1, 3]
       >>> # If not shuffling, the second iteratoin generates the same sequence
       >>> list(sampler)
       [4, 2, 0, 1, 3]
       >>> sampler.shuffle(seed=1)
       >>> list(sampler)
       [3, 2, 4, 1, 5]

    You can use :py:func:`~spdl.source.utils.embed_shuffle` to shuffle automatically
    at each iteration.

    .. admonition:: Example - Auto-shuffle
       :class: note

       >>> sampler = embed_shuffle(DistributedRandomSampler(5, rank=0, world_size=1))
       >>> list(sampler)
       [4, 2, 0, 1, 3]
       >>> list(sampler)
       [3, 2, 4, 1, 5]
       >>> list(sampler)
       [2, 1, 4, 3, 5]

    This is especially useful when the sampler is iterated in a subprocess via
    :py:func:`~spdl.pipeline.iterate_in_subprocess`, where calling
    :py:meth:`shuffle` manually from the main process has no effect on the subprocess
    copy.

    .. admonition:: Example - Running in a subprocess
       :class: note

       When iterating the sampler in a subprocess, wrap it with
       :py:func:`~spdl.source.utils.embed_shuffle` so that each epoch is automatically
       reshuffled inside the subprocess:

       .. code-block:: python

          from functools import partial
          from spdl.pipeline import iterate_in_subprocess
          from spdl.source import DistributedRandomSampler
          from spdl.source.utils import embed_shuffle

          sampler = DistributedRandomSampler(N, rank=rank, world_size=world_size)
          src = iterate_in_subprocess(embed_shuffle(sampler))

          # Each epoch, ranks generate different disjoint set of indices
          for epoch in range(num_epochs):
              for idx in src:
                  ...

    This sampler supports wieghted sampling.

    .. admonition:: Example - Weighted sampling
       :class: note

       By providing sampling weights, indices are drawn to follow the sampling weights.
       In this case, indices are sampled with replacement, thus they do not necessarily
       cover the entire dataset.

       >>> N = 5
       >>> weights = [0, 0, 1, 1, 1]
       >>> sampler = DistributedRandomSampler(5, rank=0, world_size=1, weights=weights)
       >>> list(sampler)
       [2, 4, 3, 3, 2]

       With weighted sampling, you can sample indices more than the size of the dataset.

       >>> sampler = DistributedRandomSampler(
       ...     5, rank=0, world_size=1, weights=weights, num_draws=10)
       >>> list(sampler)
       [2, 4, 3, 3, 2, 4, 3, 2, 4, 2]

    The draws are distributed across ranks in a round-robin fashion. When the
    total number of draws is not divisible by ``world_size``, the final round is
    incomplete. The ``ddp_drop_last_distributed_round`` argument controls how this
    leftover is handled.

    .. admonition:: Example - Dropping the last distributed round
       :class: note

       When ``ddp_drop_last_distributed_round=True`` (default), the incomplete
       final round is dropped so every rank draws the same number of indices.
       When ``False``, every draw position is used, so some ranks draw one more
       index than others.

       .. code-block:: text

          Draw positions, num_draws = 11, world_size = 4
          ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐
          │ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │ 8 │ 9 │10 │
          └───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘

          ddp_drop_last_distributed_round=True (default)
          rank 0 draws positions: 0, 4
          rank 1 draws positions: 1, 5
          rank 2 draws positions: 2, 6
          rank 3 draws positions: 3, 7       (8, 9, 10 are not drawn)

          ddp_drop_last_distributed_round=False
          rank 0 draws positions: 0, 4, 8
          rank 1 draws positions: 1, 5, 9
          rank 2 draws positions: 2, 6, 10
          rank 3 draws positions: 3, 7       (all draw positions are used)

    Args:
        n: The size of the dataset.

        rank: The rank in the distributed communication config.
            You can fetch the values with :py:func:`torch.distributed.get_rank`.

        world_size: The number of ranks in the distributed communication config.
            You can fetch the values with :py:func:`torch.distributed.get_world_size`.

        num_draws: *Oprional* The number of samples to draw at each iteration.
            When performing weighted sampling (``weights`` is provided),
            this can be greater than the size of the dataset. Otherwise,
            it must be smaller than or equal to the size of the dataset.

        weights: *Optional* The sampling weight of each sample in the dataset.
            When provided, the length of the sequence must match the size of
            the dataset. (``size``).
            Indices are drawn with replacement when weights are provided.

        seed: The seed value for generating the sequence.

        ddp_drop_last_distributed_round: If ``True`` (default), draw only a
            world-size multiple of samples so every rank receives the same
            number of indices. If ``False``, draw exactly ``num_draws`` or
            ``n`` samples; rank lengths may differ by at most one.

    .. versionadded:: 0.5.0
       The ``ddp_drop_last_distributed_round`` argument.
    """

    def __init__(
        self,
        n: int,
        /,
        *,
        rank: int,
        world_size: int,
        num_draws: int | None = None,
        weights: list[float] | None = None,
        seed: int = 0,
        ddp_drop_last_distributed_round: bool = True,
    ) -> None:
        w = _validate_args_random(
            n, rank, world_size, num_draws, weights, ddp_drop_last_distributed_round
        )

        total_draws = num_draws or n
        self._n = n
        self._rank = rank
        self._world_size = world_size
        if ddp_drop_last_distributed_round:
            self._len: int = total_draws // world_size
            self._num_draws: int = self._len * world_size
        else:
            self._len = total_draws // world_size + int(rank < total_draws % world_size)
            self._num_draws = total_draws
        self._weights: FloatArray | None = w
        self._seed = seed

    def __len__(self) -> int:
        """The number of indices returned by this sampler."""
        return self._len

    def __iter__(self) -> Iterator[int]:
        """Iterate over the indices assigned to the current rank.

        Yields:
            Individual indices assigned to the current rank.
        """
        indices = _weighted_choice(
            population_size=self._n,
            num_draws=self._num_draws,
            weights=self._weights,
            replace=self._weights is not None,
            seed=self._seed,
        )
        yield from indices[self._rank :: self._world_size]

    def shuffle(self, seed: int) -> None:
        """Set the random seed for future iterations.

        The resulting sequence depends only on the given ``seed`` value and is not
        affected by any prior iterations or previous calls to :py:meth:`shuffle`.
        Calling ``shuffle(seed=K)`` always produces the same sequence for a given
        sampler configuration, regardless of history.
        """
        self._seed = seed
