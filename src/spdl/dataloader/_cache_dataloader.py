# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Iterable, Iterator
from typing import Any, Generic, TypeVar

from spdl._internal import log_api_usage_once
from spdl.pipeline import cache_iterator

T = TypeVar("T")


class CacheDataLoader(Generic[T]):
    """Caches values from the given data loader and returns caches after the given iteration.

    The class is a simple wrapper around generic data loader instance.
    It is intended for estimating the maximum performance gain achieved
    by optimizing the data loader.

    You can wrap your data loader with this class, and run it in the training
    pipeline, and compare the performance to see if the training pipeline is
    bottlenecked with data loading.

    Args:
        dl: Source iterator. Expected to be a data loader object.
        num_caches,return_caches_after,stop_after: See :py:func:`spdl.pipeline.cache_iterator`.

    Returns:
        The new iterator.

    .. seealso::

       - :py:func:`spdl.pipeline.cache_iterator`: The helper function that
         implements the caching logic.
    """

    def __init__(
        self,
        dl: Iterable[T],
        num_caches: int,
        return_caches_after: int,
        stop_after: int | None = None,
    ) -> None:
        log_api_usage_once("spdl.dataloader.CacheDataLoader")

        self.dl = dl

        self.num_caches = num_caches
        self.return_caches_after = return_caches_after
        self.stop_after = stop_after

    def __iter__(self) -> Iterator[T]:
        """See :py:func:`spdl.pipeline.cache_iterator` for the detail."""
        return cache_iterator(
            self.dl,
            num_caches=self.num_caches,
            return_caches_after=self.return_caches_after,
            stop_after=self.stop_after,
        )

    def __len__(self) -> int:
        """Returns the length of the original data loader if defined."""
        return len(self.dl)  # pyre-ignore: [6]

    def __getattr__(self, name: str) -> Any:
        return getattr(self.dl, name)
