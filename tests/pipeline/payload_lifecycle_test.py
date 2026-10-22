# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import gc
import threading
import unittest
import weakref
from collections.abc import AsyncIterator, Iterator

from spdl.pipeline import PipelineBuilder
from spdl.pipeline.defs import MAIN_PROCESS, ProcessPoolExecutorConfig


class _Payload:
    pass


def _make_payload(value: object) -> _Payload:
    return _Payload()


class _PausedSource:
    def __init__(self, *, processes: bool = False) -> None:
        self.processes = processes
        self.waiting = threading.Event()
        self.release = threading.Event()

    def _item(self) -> _Payload | int:
        return 0 if self.processes else _Payload()


class _PausedSyncSource(_PausedSource):
    def __iter__(self) -> Iterator[_Payload | int]:
        yield self._item()
        self.waiting.set()
        self.release.wait()


class _PausedAsyncSource(_PausedSource):
    async def __aiter__(self) -> AsyncIterator[_Payload | int]:
        yield self._item()
        self.waiting.set()
        await asyncio.to_thread(self.release.wait)


class _Capture:
    def __init__(self) -> None:
        self.references: list[weakref.ReferenceType[_Payload]] = []
        self.released = threading.Event()

    def _on_release(self, reference: weakref.ReferenceType[_Payload]) -> None:
        self.released.set()

    def __call__(self, payload: _Payload) -> int:
        self.references.append(weakref.ref(payload, self._on_release))
        return 0


class PayloadLifecycleTest(unittest.TestCase):
    def _check_release(
        self,
        source: _PausedSyncSource | _PausedAsyncSource,
        *,
        continuous: bool,
        ordered: bool = False,
    ) -> None:
        capture = _Capture()
        builder = PipelineBuilder().add_source(source, continuous=continuous)
        if source.processes:
            builder = (
                builder.to(ProcessPoolExecutorConfig(max_workers=2, mp_context="spawn"))
                .pipe(_make_payload)
                .to(MAIN_PROCESS)
            )
        pipeline = (
            builder.pipe(capture, output_order="input" if ordered else "completion")
            .add_sink(1)
            .build(num_threads=2, max_failures=0)
        )
        pipeline.start()
        try:
            self.assertEqual(pipeline.get_item(timeout=15), 0)
            self.assertTrue(source.waiting.wait(5))
            self.assertEqual(len(capture.references), 1)
            gc.collect()
            self.assertTrue(
                capture.released.wait(5),
                "A consumed payload was retained while the source waited for more input.",
            )
            self.assertIsNone(capture.references[0]())
        finally:
            source.release.set()
            pipeline.stop(timeout=15)

    def test_consumed_source_payload_is_released(self) -> None:
        """Sync and async stages release consumed inputs before fetching another item."""
        for source_type in (_PausedSyncSource, _PausedAsyncSource):
            for continuous in (False, True):
                for ordered in (False, True):
                    with self.subTest(
                        source=source_type.__name__,
                        continuous=continuous,
                        ordered=ordered,
                    ):
                        self._check_release(
                            source_type(), continuous=continuous, ordered=ordered
                        )

    def test_consumed_process_result_is_released(self) -> None:
        """Both process collectors release results before waiting for another message."""
        for continuous in (False, True):
            with self.subTest(continuous=continuous):
                self._check_release(
                    _PausedAsyncSource(processes=True), continuous=continuous
                )

    def test_consumed_sink_payload_is_released(self) -> None:
        """The sink releases a consumed payload while its upstream source is idle."""
        for continuous in (False, True):
            with self.subTest(continuous=continuous):
                source = _PausedAsyncSource()
                capture = _Capture()
                pipeline = (
                    PipelineBuilder()
                    .add_source(source, continuous=continuous)
                    .add_sink(1)
                    .build(num_threads=2)
                )
                pipeline.start()
                try:
                    item = pipeline.get_item(timeout=15)
                    capture(item)
                    del item
                    self.assertTrue(source.waiting.wait(5))
                    gc.collect()
                    self.assertTrue(capture.released.wait(5))
                    self.assertIsNone(capture.references[0]())
                finally:
                    source.release.set()
                    pipeline.stop(timeout=15)
