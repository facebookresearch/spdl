# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Empty owner-loop polls use control values rather than exception Futures."""

import threading
import unittest
from typing import Any
from unittest.mock import patch

from parameterized import parameterized
from spdl.pipeline import AsyncQueue, PipelineBuilder
from spdl.pipeline._pipeline import _ReadTimedOut

_ENTERED = threading.Event()
_RELEASE = threading.Event()


def _wait_for_release(item: int) -> int:
    _ENTERED.set()
    if not _RELEASE.wait(30):
        raise TimeoutError("Test did not release the operation")
    return item


class _PlainAsyncSink(AsyncQueue):
    """A custom asyncio sink, which is not wrapped by the foreground mirror."""


_SINKS = [
    ("thread", {}),
    ("mirrored_async", {"use_thread_output_queue": False}),
    ("plain_async", {"use_thread_output_queue": False, "queue_class": _PlainAsyncSink}),
]


class EmptyPollLifetimeTest(unittest.TestCase):
    def _start(self, build_kwargs: dict[str, Any]) -> None:
        _ENTERED.clear()
        _RELEASE.clear()
        self.pipeline = (
            PipelineBuilder()
            .add_source([1])
            .pipe(_wait_for_release)
            .add_sink(1)
            .build(num_threads=2, **build_kwargs)
        )
        self.pipeline.start()
        self.addCleanup(self._stop_pipeline)
        self.assertTrue(_ENTERED.wait(10))

    def _stop_pipeline(self) -> None:
        _RELEASE.set()
        self.pipeline.stop(timeout=30)

    @parameterized.expand(_SINKS)
    def test_empty_poll_has_no_exception_traceback(
        self, _: str, build_kwargs: dict[str, Any]
    ) -> None:
        """An expected empty poll cannot store a frame-retaining exception in its Future."""
        self._start(build_kwargs)
        impl = self.pipeline._impl
        future = impl._event_loop.run_coroutine_threadsafe(impl._get_nowait_on_loop())
        self.assertIsInstance(future.result(timeout=10), _ReadTimedOut)
        self.assertIsNone(future.exception())
        self.assertIsNone(impl._pending_output_read)

    @parameterized.expand(_SINKS)
    def test_real_poll_error_preserves_its_cause(
        self, _: str, build_kwargs: dict[str, Any]
    ) -> None:
        """A non-control sink error remains the original Future exception."""
        self._start(build_kwargs)
        impl = self.pipeline._impl
        root = RuntimeError("original root cause")
        error = ValueError("deliberate real poll failure")
        error.__cause__ = root
        with patch.object(impl._output_queue, "get_nowait", side_effect=error):
            future = impl._event_loop.run_coroutine_threadsafe(
                impl._get_nowait_on_loop()
            )
            with self.assertRaisesRegex(ValueError, "deliberate real poll failure"):
                future.result(timeout=10)
            self.assertIs(future.exception(), error)
            self.assertIs(error.__cause__, root)
