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

    FloatArray: TypeAlias = NDArray[np.floating[Any]]


else:
    from spdl._internal.import_utils import lazy_import

    np = lazy_import("numpy")
    FloatArray = object

_LG: logging.Logger = logging.getLogger(__name__)


def _validate_args_det(n: int, rank: int, world_size: int) -> None:
    if n <= 0:
        raise ValueError(f"The size of dataset must be larger than 0. Found: {n}")

    if rank >= world_size:
        raise ValueError(f"rank ({rank}) must be with the range of [0, {world_size=}).")

    if n < world_size:
        raise ValueError(
            f"The size of dataset ({n}) must be larger than or equal to "
            f"the world size ({world_size})"
        )

    if n % world_size:
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

    Args:
        n: The size of the dataset.

        rank: The rank in the distributed communication config.
            You can fetch the values with :py:func:`torch.distributed.get_rank`.

        world_size: The number of ranks in the distributed communication config.
            You can fetch the values with :py:func:`torch.distributed.get_world_size`.
    """

    def __init__(
        self,
        n: int,
        /,
        *,
        rank: int,
        world_size: int,
    ) -> None:
        _validate_args_det(n, rank, world_size)

        self._rank = rank
        self._world_size = world_size

        self._len: int = n // world_size

    def __len__(self) -> int:
        """The number of indices returned by this sampler."""
        return self._len

    def __iter__(self) -> Iterator[int]:
        """Iterate over the indices assigned to the current rank.

        Yields:
            Individual indices assigned to the current rank.
        """
        max_ = self._len * self._world_size
        yield from range(self._rank, max_, self._world_size)


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
) -> FloatArray | None:
    if n <= 0:
        raise ValueError(f"The size of dataset must be larger than 0. Found: {n}")

    if rank >= world_size:
        raise ValueError(f"rank ({rank}) must be with the range of [0, {world_size=}).")

    if n < world_size:
        raise ValueError(
            f"The size of dataset ({n}) must be larger than or equal to "
            f"the world size ({world_size})"
        )

    if num_draws is not None:
        if num_draws <= 0:
            raise ValueError(
                f"`num_draws` must be greater than zero. Found: {num_draws}"
            )

        if num_draws < world_size:
            raise ValueError(
                "``num_draws` must be greater than `world_size`. "
                f"Found: {num_draws=}, {world_size=}"
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

    This sampler can apply two randomness; shuffling and wieghted sampling.

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

       >>> sampler = embed_shuffle(DistributedRandomSampler(5, rank=0, world_size=1))
       >>> list(sampler)
       [4, 2, 0, 1, 3]
       >>> list(sampler)
       [3, 2, 4, 1, 5]
       >>> list(sampler)
       [2, 1, 4, 3, 5]

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

    Args:
        n: The size of the dataset.

        rank: The rank in the distributed communication config.
            You can fetch the values with :py:func:`torch.distributed.get_rank`.

        world_size: The number of ranks in the distributed communication config.
            You can fetch the values with :py:func:`torch.distributed.get_world_size`.

        num_draws: The number of samples to draw at each iteration.
            If peforming weighted sampling, (``deterministic=False`` and
            ``weights`` is provided)  then it can be greater than the
            size of dataset. Otherwise, it must be smaller than or equal
            to the size of dataset.

        weights: *Optional* The sampling weight of each sample in the dataset.
            When provided, the length of the sequence must match the size of the dataset.
            (``size``).
            This option is ignored if ``deterministic=True``.

        seed: The seed value for generating the sequence.
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
    ) -> None:
        w = _validate_args_random(n, rank, world_size, num_draws, weights)

        self._n = n
        self._rank = rank
        self._world_size = world_size
        self._len: int = (num_draws or n) // world_size
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
            num_draws=self._len * self._world_size,
            weights=self._weights,
            replace=self._weights is not None,
            seed=self._seed,
        )
        yield from indices[self._rank :: self._world_size]

    def shuffle(self, seed: int) -> None:
        """Set the random seed for future iterations."""
        self._seed = seed
