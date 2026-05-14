# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""Regression test for D101554675: breaking out of a subprocess iterable
must not kill the worker, so subsequent iterations still work."""

import unittest
from collections.abc import Iterable, Iterator
from functools import partial

from spdl.pipeline import iterate_in_subprocess


class SourceIterable:
    def __init__(self, n: int) -> None:
        self.n = n

    def __iter__(self) -> Iterator[int]:
        yield from range(self.n)


class TestSubprocessBreakAndReiterate(unittest.TestCase):
    def test_break_then_reiterate(self) -> None:
        """Breaking out of a subprocess iterable must not prevent re-iteration.

        This is a regression test for the BaseException widening in D101554675.
        When a consumer `break`s out of `for ... in iterable`, Python sends
        GeneratorExit into the generator. Prior to D101554675, only
        (Exception, KeyboardInterrupt) triggered _shutdown(). After D101554675,
        BaseException (which includes GeneratorExit) triggers _shutdown(),
        making subsequent iter() calls raise RuntimeError.
        """
        src = iterate_in_subprocess(partial(SourceIterable, 10), timeout=10)

        # First iteration: consume only 3 items, then break
        count = 0
        for item in src:
            count += 1
            if count >= 3:
                break

        # Second iteration: must succeed (worker should still be alive)
        result = list(src)
        self.assertEqual(result, list(range(10)))

    def test_break_then_reiterate_multiple_times(self) -> None:
        """Multiple break-then-reiterate cycles must all succeed."""
        src = iterate_in_subprocess(partial(SourceIterable, 5), timeout=10)

        for cycle in range(3):
            # Break after 2 items
            count = 0
            for item in src:
                count += 1
                if count >= 2:
                    break

            # Full iteration must still work
            result = list(src)
            self.assertEqual(result, list(range(5)), f"cycle {cycle}")

    def test_partial_iteration_via_zip(self) -> None:
        """Partial iteration via zip() (which breaks implicitly) must not kill worker."""
        src = iterate_in_subprocess(partial(SourceIterable, 100), timeout=10)

        # zip stops when the shorter iterable is exhausted, causing
        # GeneratorExit on the longer one
        partial_result = list(zip(range(3), src))
        self.assertEqual(partial_result, [(0, 0), (1, 1), (2, 2)])

        # Subsequent full iteration must work
        result = list(src)
        self.assertEqual(result, list(range(100)))


if __name__ == "__main__":
    unittest.main()
