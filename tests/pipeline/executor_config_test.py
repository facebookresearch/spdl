# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

from spdl.pipeline.defs import (
    InterpreterPoolExecutorConfig,
    MAIN_PROCESS,
    PlacementConfig,
    ProcessPoolExecutorConfig,
)

# NOTE: The behavior of these config types (how a region's worker pool is built and run from a
# spec), including that a ``PipelineConfig`` accepts region markers in its ``pipes``, is covered
# end-to-end by ``marked_region_fuse_test`` and ``builder_to_test``. The tests here cover only what
# those cannot: the hand-written ``__repr__`` methods and the deliberate structural contract of
# ``InterpreterPoolExecutorConfig`` (that it rejects an ``mp_context``). Plain-dataclass mechanics
# (default values, field storage, ``frozen``, auto-generated ``__eq__``) are intentionally not
# re-tested.


class TestInterpreterPoolExecutorConfig(unittest.TestCase):
    """Verify the InterpreterPoolExecutorConfig spec."""

    def test_rejects_mp_context(self) -> None:
        """InterpreterPoolExecutorConfig rejects ``mp_context``; ProcessPoolExecutorConfig honors it.

        A subinterpreter shares the process, so there is no multiprocessing start method to choose.
        Passing ``mp_context`` must raise rather than be silently accepted and ignored -- unlike
        :py:class:`ProcessPoolExecutorConfig`, which starts real processes and stores the chosen
        start method.
        """
        self.assertEqual(
            ProcessPoolExecutorConfig(mp_context="spawn").mp_context, "spawn"
        )
        with self.assertRaises(TypeError):
            InterpreterPoolExecutorConfig(mp_context="spawn")  # pyre-ignore[28]


class TestMainProcess(unittest.TestCase):
    """Verify the MAIN_PROCESS sentinel target."""

    def test_repr(self) -> None:
        """The sentinel's custom ``__repr__`` renders its public name for readable configs."""
        self.assertEqual(repr(MAIN_PROCESS), "MAIN_PROCESS")


class TestPlacementConfig(unittest.TestCase):
    """Verify the PlacementConfig region marker."""

    def test_repr_includes_target(self) -> None:
        """The marker's custom ``__repr__`` surfaces its target for readable pipeline dumps."""
        self.assertEqual(
            repr(PlacementConfig(target=MAIN_PROCESS)), "placement(MAIN_PROCESS)"
        )


if __name__ == "__main__":
    unittest.main()
