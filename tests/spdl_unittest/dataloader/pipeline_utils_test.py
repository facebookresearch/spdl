# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from spdl.pipeline import cache_iterator


def test_cache_iterator():
    """cache_iterator returns the cached values"""

    ite = iter(cache_iterator(range(5), 3))

    assert next(ite) == 0
    assert next(ite) == 1
    assert next(ite) == 2

    assert next(ite) == 0
    assert next(ite) == 1
    assert next(ite) == 2

    assert next(ite) == 0
    assert next(ite) == 1
    assert next(ite) == 2


def test_cache_iterator_cache_after():
    """cache_iterator returns the cached values"""

    ite = iter(cache_iterator(range(7), 3, return_caches_after=5))

    assert next(ite) == 0
    assert next(ite) == 1
    assert next(ite) == 2
    assert next(ite) == 3
    assert next(ite) == 4

    assert next(ite) == 0
    assert next(ite) == 1
    assert next(ite) == 2

    assert next(ite) == 0
    assert next(ite) == 1
    assert next(ite) == 2
