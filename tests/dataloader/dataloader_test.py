# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import threading
import time
import unittest

from spdl.dataloader import DataLoader


def get_dl(*args, timeout=3, num_threads=2, **kwargs):
    # on default values
    # timeout     -> so that test would fail rather stack
    # num_threads -> keep it minimum but have more than 1
    return DataLoader(*args, **kwargs, num_threads=num_threads, timeout=timeout)


class TestDataLoader(unittest.TestCase):
    def test_dataloader_iterable(self) -> None:
        src = list(range(10))

        dl = get_dl(src)

        self.assertEqual(sorted(dl), src)

    def test_dataloader_stateful_iterable(self) -> None:
        class src:
            def __init__(self, num_items: int = 10):
                self.num_iter = 0
                self.num_items = num_items

            def __iter__(self):
                for i in range(self.num_items):
                    yield (self.num_iter, i)
                self.num_iter += 1

        dl = get_dl(src())

        self.assertEqual(sorted(dl), [(0, i) for i in range(10)])
        self.assertEqual(sorted(dl), [(1, i) for i in range(10)])
        self.assertEqual(sorted(dl), [(2, i) for i in range(10)])

    def test_dataloader_preprocess(self) -> None:
        """preprocessor process the value of the source"""
        src = list(range(10))

        def double(x):
            time.sleep(0.05 * x)  # to reduce flakiness from multi-threading
            return 2 * x

        dl = get_dl(src, preprocessor=double)

        self.assertEqual(sorted(dl), [i * 2 for i in range(10)])

    def test_dataloader_preprocess_in_order(self) -> None:
        """When output_order='input', the order must be preserved."""
        src = list(range(10, -1, -1))

        def delay(x):
            time.sleep(0.1 * x)
            return x

        dl = get_dl(src, preprocessor=delay, output_order="input")

        self.assertEqual(list(dl), src)

        dl = get_dl(src, preprocessor=delay, output_order="completion")

        self.assertNotEqual(list(dl), src)

    def test_dataloader_buffer_size(self) -> None:
        """A larger buffer lets background work finish while foreground is paused."""
        src = list(range(64))

        def finishes_in_background(buffer_size: int) -> bool:
            last_processed = threading.Event()

            def track(item: int) -> int:
                if item == src[-1]:
                    last_processed.set()
                return item

            iterator = iter(
                get_dl(
                    src,
                    preprocessor=track,
                    num_threads=1,
                    buffer_size=buffer_size,
                )
            )
            self.assertEqual(next(iterator), 0)

            reached_last = last_processed.wait(timeout=3)
            self.assertEqual(list(iterator), src[1:])
            self.assertTrue(last_processed.is_set())
            return reached_last

        self.assertFalse(finishes_in_background(1))
        self.assertTrue(finishes_in_background(len(src)))

    def test_dataloader_num_threads(self) -> None:
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
            self.assertEqual(sorted(result), src)
            return elapsed

        dl = get_dl(src, preprocessor=delay, num_threads=1, buffer_size=1)
        self.assertGreater(test(dl), 0.8)

        dl = get_dl(src, preprocessor=delay, num_threads=len(src), buffer_size=1)
        self.assertLess(test(dl), 0.6)

    def test_dataloader_batch(self) -> None:
        """batching works with or without dropping"""
        src = list(range(10))

        dl = get_dl(src, batch_size=3, drop_last=False)

        self.assertEqual(list(dl), [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]])

        dl = get_dl(src, batch_size=3, drop_last=True)

        self.assertEqual(list(dl), [[0, 1, 2], [3, 4, 5], [6, 7, 8]])

    def test_dataloader_aggregate(self) -> None:
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

        self.assertEqual(list(dl), expected)
