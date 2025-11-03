# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Iterator caching utilities.

This module provides utilities for caching iterator values, useful for
benchmarking and performance testing.
"""

from collections.abc import Iterable, Iterator
from typing import TypeVar

__all__ = [
    "cache_iterator",
]

T = TypeVar("T")


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
