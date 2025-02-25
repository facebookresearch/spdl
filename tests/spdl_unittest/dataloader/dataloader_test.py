# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import os
import platform
import time

import pytest
from spdl.dataloader import DataLoader


def get_dl(*args, timeout=3, num_threads=2, **kwargs):
    # on default values
    # timeout     -> so that test would fail rather stack
    # num_threads -> keep it minimum but have more than 1
    return DataLoader(*args, **kwargs, num_threads=num_threads, timeout=timeout)


def test_dataloader_iterable():
    src = list(range(10))

    dl = get_dl(src)

    assert sorted(dl) == src


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

    assert sorted(dl) == [(0, i) for i in range(10)]
    assert sorted(dl) == [(1, i) for i in range(10)]
    assert sorted(dl) == [(2, i) for i in range(10)]


def test_dataloader_preprocess():
    """preprocessor process the value of the source"""
    src = list(range(10))

    def double(x):
        time.sleep(0.05 * x)  # to reduce flakiness from multi-threading
        return 2 * x

    dl = get_dl(src, preprocessor=double)

    assert sorted(dl) == [i * 2 for i in range(10)]


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


@pytest.mark.skipif(
    platform.system() == "Darwin" and "CI" in os.environ,
    reason="GitHub macOS CI is not timely enough.",
)
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
