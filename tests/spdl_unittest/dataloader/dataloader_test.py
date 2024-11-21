# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import time

import pytest
from spdl.dataloader import DataLoader, MapIterator, MergeIterator


def get_dl(*args, timeout=3, num_threads=2, **kwargs):
    # on default values
    # timeout     -> so that test would fail rather stack
    # num_threads -> keep it minimum but have more than 1
    return DataLoader(*args, **kwargs, num_threads=num_threads, timeout=timeout)


def test_dataloader_iterable():
    src = list(range(10))

    dl = get_dl(src)

    assert list(dl) == src


def test_dataloader_stateful_iterable():
    class src:
        def __init__(self, num_items: int = 10):
            self.num_iter = 0
            self.num_items = num_items

        def __iter__(self):
            for i in range(self.num_items):
                yield (self.num_iter, i)
            self.num_iter += 1

    dl = get_dl(src())

    assert list(dl) == [(0, i) for i in range(10)]
    assert list(dl) == [(1, i) for i in range(10)]
    assert list(dl) == [(2, i) for i in range(10)]


def test_dataloader_preprocess():
    """preprocessor process the value of the source"""
    src = list(range(10))

    def double(x):
        time.sleep(0.05 * x)  # to reduce flakiness from multi-threading
        return 2 * x

    dl = get_dl(src, preprocessor=double)

    assert list(dl) == [i * 2 for i in range(10)]


def test_dataloader_preprocess_in_order():
    """When output_order='input', the order must be preserved."""
    src = list(range(10, -1, -1))

    def delay(x):
        time.sleep(0.1 * x)
        return x

    dl = get_dl(src, preprocessor=delay, output_order="input")

    assert list(dl) == src

    dl = get_dl(src, preprocessor=delay, output_order="completion")

    assert list(dl) != src


def test_dataloader_buffer_size():
    """Bigger buffer_size allows the BG to proceed while FG is not fetching the data"""
    src = list(range(12))

    def delay(x):
        time.sleep(0.05)
        return x

    def test(dl):
        # Kick off the background thread
        dli = iter(dl)
        assert next(dli) == 0

        # Wait: (simulate foreground load)
        time.sleep(1)

        # Iterate the rest
        t0 = time.monotonic()
        result = list(dli)
        elapsed = time.monotonic() - t0
        print(elapsed)
        assert result == src[1:]
        return elapsed

    # With buffer_size == 1, then  the background thread cannot proceed
    # while foreground thread does not fetch any.
    dl = get_dl(src, preprocessor=delay, num_threads=1, buffer_size=1)
    elapsed = test(dl)
    assert elapsed > 0.3

    # With bigger buffer_size, the background thread proceed
    # while foreground thread does not fetch any.
    dl = get_dl(src, preprocessor=delay, num_threads=1, buffer_size=len(src))
    elapsed = test(dl)
    assert elapsed < 0.1


def test_dataloader_num_threads():
    """Increasing the num_threads reduces the overall time."""
    src = list(range(10))

    def delay(x):
        time.sleep(0.1)
        return x

    def test(dl):
        t0 = time.monotonic()
        result = list(dl)
        elapsed = time.monotonic() - t0
        print(elapsed)
        assert sorted(result) == src
        return elapsed

    dl = get_dl(src, preprocessor=delay, num_threads=1, buffer_size=1)
    assert test(dl) > 0.8

    dl = get_dl(src, preprocessor=delay, num_threads=len(src), buffer_size=1)
    assert test(dl) < 0.4


def test_dataloader_batch():
    """batching works with or without dropping"""
    src = list(range(10))

    dl = get_dl(src, batch_size=3, drop_last=False)

    assert list(dl) == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]

    dl = get_dl(src, batch_size=3, drop_last=True)

    assert list(dl) == [[0, 1, 2], [3, 4, 5], [6, 7, 8]]


def test_dataloader_aggregate():
    """Aggregator processes the batched input"""
    src = list(range(10))

    def agg(vals: list[int]) -> tuple[int, int, int, int]:
        return len(vals), min(vals), max(vals), sum(vals)

    dl = get_dl(src, batch_size=3, drop_last=False, aggregator=agg)

    expected = [
        (3, 0, 2, 3),
        (3, 3, 5, 12),
        (3, 6, 8, 21),
        (1, 9, 9, 9),
    ]

    assert list(dl) == expected


def test_mapiterator():
    """MapIterator iterates the mapped values"""

    mapping = {0: "a", 1: "b", 2: "c", 3: "d", 4: "e"}

    result = list(MapIterator(mapping))
    assert result == list(mapping.values())


def test_mapiterator_sampler():
    """MapIterator iterates the mapped values picked by sampler"""

    mapping = {0: "a", 1: "b", 2: "c", 3: "d", 4: "e"}
    sampler = [4, 2, 0]

    result = list(MapIterator(mapping, sampler))
    assert result == ["e", "c", "a"]


def test_mergeiterator_ordered():
    """MergeIterator iterates multiple iterators"""

    iterables = [
        [0, 1, 2],
        [10, 11, 12],
        [20, 21, 22],
    ]

    result = list(MergeIterator(iterables))
    assert result == [0, 10, 20, 1, 11, 21, 2, 12, 22]


def test_mergeiterator_ordered_stop_after_first_exhaustion():
    """MergeIterator stops after the first exhaustion"""

    iterables = [
        [0],
        [10, 11, 12],
        [20, 21, 22],
    ]

    result = list(MergeIterator(iterables, stop_after=-1))
    assert result == [0, 10, 20]

    iterables = [
        [0, 1, 2],
        [10],
        [20, 21, 22],
    ]

    result = list(MergeIterator(iterables, stop_after=-1))
    assert result == [0, 10, 20, 1]

    iterables = [
        [0, 1, 2],
        [10, 11],
        [20],
    ]

    result = list(MergeIterator(iterables, stop_after=-1))
    assert result == [0, 10, 20, 1, 11]


def test_mergeiterator_ordered_stop_after_N():
    """MergeIterator stops after N items are yielded"""

    iterables = [
        [0, 1, 2],
        [10, 11, 12],
        [20, 21, 22],
    ]

    result = list(MergeIterator(iterables, stop_after=1))
    assert result == [0]

    result = list(MergeIterator(iterables, stop_after=5))
    assert result == [0, 10, 20, 1, 11]

    result = list(MergeIterator(iterables, stop_after=7))
    assert result == [0, 10, 20, 1, 11, 21, 2]


def test_mergeiterator_ordered_stop_after_minus1():
    """MergeIterator stops after all the iterables are exhausted"""

    iterables = [
        [0, 1, 2],
        [10, 11, 12],
        [20, 21, 22],
    ]

    result = list(MergeIterator(iterables))
    assert result == [0, 10, 20, 1, 11, 21, 2, 12, 22]

    iterables = [
        [0, 1, 2],
        [10],
        [20, 21, 22],
    ]

    result = list(MergeIterator(iterables))
    assert result == [0, 10, 20, 1, 21, 2, 22]

    iterables = [
        [0, 1, 2],
        [10, 11, 12],
        [20],
    ]

    result = list(MergeIterator(iterables))
    assert result == [0, 10, 20, 1, 11, 2, 12]


def test_mergeiterator_ordered_n():
    """with stop_after=N, MergeIterator continues iterating after encountering an exhaustion."""
    iterables = [
        [0, 1, 2],
        [10],
        [20, 21, 22],
    ]

    result = list(MergeIterator(iterables, stop_after=5))
    assert result == [0, 10, 20, 1, 21]

    result = list(MergeIterator(iterables, stop_after=7))
    assert result == [0, 10, 20, 1, 21, 2, 22]

    result = list(MergeIterator(iterables, stop_after=8))
    assert result == [0, 10, 20, 1, 21, 2, 22]


def test_mergeiterator_stochastic_smoke_test():
    """MergeIterator with probabilitiies do not get stuck."""

    iterables = [
        [0, 1, 2],
        [10, 11, 12],
        [20, 21, 22],
    ]

    weights = [1, 1, 1]

    result = list(MergeIterator(iterables, weights=weights))
    assert set(result) == {0, 1, 2, 10, 11, 12, 20, 21, 22}


def test_mergeiterator_stochastic_rejects_zero():
    """weight=0 is rejected."""
    weights = [1, 0]

    with pytest.raises(ValueError):
        MergeIterator([[1]], weights=weights)

    weights = [1, 0.0]

    with pytest.raises(ValueError):
        MergeIterator([[1]], weights=weights)


def test_mergeiterator_stochastic_stop_after_N():
    """Values are taken from iterables with higher weights"""
    weights = [1000000, 1]

    iterables = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    ]

    result = list(MergeIterator(iterables, weights=weights, stop_after=3))
    assert result == [0, 1, 2]


def test_mergeiterator_stochastic_stop_after_first_exhaustion():
    """Values are taken from iterables with higher weights"""
    weights = [1000000, 1]

    iterables = [
        [0, 1, 2, 3],
        [10, 11, 12, 13],
    ]

    result = list(MergeIterator(iterables, weights=weights, stop_after=-1))
    assert result == [0, 1, 2, 3]
