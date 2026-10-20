# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import asyncio
import unittest
from collections.abc import Iterator
from contextlib import contextmanager

from spdl.pipeline import PipelineBuilder, profile_pipeline, ProfileHook


class TimedOperation:
    def __init__(self) -> None:
        self.active = 0
        self.peak = 0
        self.completed = 0

    async def __call__(self, item: int) -> int:
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            await asyncio.sleep(0.02)
            self.completed += 1
            return item * 2
        finally:
            self.active -= 1


class ObservedConcurrency(ProfileHook):
    def __init__(self, operation: TimedOperation) -> None:
        self.operation = operation
        self.observed: dict[int, tuple[int, int]] = {}

    @contextmanager
    def stage_profile_hook(self, stage: str, concurrency: int) -> Iterator[None]:
        self.operation.peak = self.operation.completed = 0
        yield
        self.observed[concurrency] = (self.operation.peak, self.operation.completed)

    @contextmanager
    def pipeline_profile_hook(self) -> Iterator[None]:
        yield


class ProfileConcurrencyTest(unittest.TestCase):
    def test_reported_concurrency_matches_active_async_operations(self) -> None:
        """Each profiler label must describe the stage's actual admission limit."""
        operation = TimedOperation()
        hook = ObservedConcurrency(operation)
        config = (
            PipelineBuilder()
            .add_source(range(64))
            .pipe(operation)
            .add_sink(1)
            .get_config()
        )
        results = profile_pipeline(config, num_inputs=64, hook=hook)
        levels = [stat.concurrency for stat in results[0].stats]
        self.assertEqual(levels, [32, 16, 8, 4, 1])
        self.assertEqual(hook.observed, {level: (level, 64) for level in levels})
