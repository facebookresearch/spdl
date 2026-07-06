# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import gc
import multiprocessing as mp
import threading
import unittest
import unittest.mock
import weakref
from typing import Any, Callable

from spdl.pipeline import _pipeline as _pipeline_mod, Pipeline, PipelineBuilder
from spdl.pipeline._components._subprocess_pipe import _put


def _add_one(x: int) -> int:
    return x + 1


def _build_pipeline() -> Pipeline[int]:
    """Build a simple thread-only pipeline (not started)."""
    return (
        PipelineBuilder()
        .add_source(range(4))
        .pipe(_add_one)
        .add_sink(2)
        .build(num_threads=1)
    )


class _CaptureRegisteredHooks:
    """Capture callbacks the pipeline registers via ``threading._register_atexit``.

    Spies on the real private API (so a stdlib rename/removal makes this fail loudly) but does
    not actually register, keeping the process-wide atexit list clean across tests. This tests
    the contract -- that ``start()`` hands a working stop callback to
    ``threading._register_atexit`` -- without depending on the stdlib's internal storage, which
    differs across Python versions.
    """

    def __init__(self) -> None:
        self.hooks: list[Callable[[], Any]] = []

    def __enter__(self) -> "_CaptureRegisteredHooks":
        self._patch = unittest.mock.patch.object(
            threading, "_register_atexit", side_effect=self.hooks.append
        )
        self._patch.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._patch.stop()


class ShutdownHookRegistrationTest(unittest.TestCase):
    """Registration of the interpreter-exit stop hook on Pipeline.start()."""

    def test_register_atexit_api_available(self) -> None:
        """The private ``threading._register_atexit`` the fix relies on still exists."""
        self.assertTrue(hasattr(threading, "_register_atexit"))

    def test_start_registers_exactly_one_hook(self) -> None:
        """Starting a pipeline registers exactly one interpreter-exit stop hook."""
        with _CaptureRegisteredHooks() as cap:
            p = _build_pipeline()
            p.start()
        try:
            self.assertEqual(len(cap.hooks), 1)
        finally:
            p.stop()

    def test_build_without_start_registers_no_hook(self) -> None:
        """Building (without starting) a pipeline registers no hook."""
        with _CaptureRegisteredHooks() as cap:
            p = _build_pipeline()
        try:
            self.assertEqual(cap.hooks, [])
        finally:
            p.stop()

    def test_each_started_pipeline_registers_its_own_hook(self) -> None:
        """Each started pipeline registers its own hook (one per instance)."""
        with _CaptureRegisteredHooks() as cap:
            p1 = _build_pipeline()
            p1.start()
            p2 = _build_pipeline()
            p2.start()
        try:
            self.assertEqual(len(cap.hooks), 2)
        finally:
            p1.stop()
            p2.stop()


class ShutdownHookBehaviorTest(unittest.TestCase):
    """Behavior of the registered interpreter-exit stop hook."""

    def test_hook_stops_the_pipeline(self) -> None:
        """Invoking the exit hook stops the (still-running) pipeline."""
        with _CaptureRegisteredHooks() as cap:
            p = _build_pipeline()
            p.start()
        try:
            (hook,) = cap.hooks
            self.assertLess(
                p._impl._event_loop_state, _pipeline_mod._EventLoopState.STOPPED
            )
            hook()
            self.assertEqual(
                p._impl._event_loop_state, _pipeline_mod._EventLoopState.STOPPED
            )
        finally:
            p.stop()

    def test_hook_does_not_pin_the_pipeline(self) -> None:
        """The hook holds only a weak ref, so the pipeline stays collectible; then it no-ops."""
        with _CaptureRegisteredHooks() as cap:
            p = _build_pipeline()
            p.start()
        (hook,) = cap.hooks
        wref = weakref.ref(p._impl)
        # Drop the only strong reference; the GC finalizer stops the impl and releases it.
        del p
        gc.collect()
        self.assertIsNone(wref())
        hook()  # dead weakref -> safe no-op, must not raise

    def test_hook_is_noop_after_pipeline_collected(self) -> None:
        """Once the pipeline is cleaned up, its exit hook is a safe no-op (dead weakref).

        Simulates the stdlib firing the interpreter-exit hooks after the pipeline object has
        already been stopped and garbage-collected: the callback must observe a dead weakref and
        return without raising.
        """
        with _CaptureRegisteredHooks() as cap:
            p = _build_pipeline()
            p.start()
        (hook,) = cap.hooks
        wref = weakref.ref(p._impl)
        p.stop()
        del p
        gc.collect()
        self.assertIsNone(wref())
        hook()  # must not raise

    def test_hook_tolerates_raising_stop(self) -> None:
        """The hook swallows an exception raised by stop() so shutdown proceeds."""
        with _CaptureRegisteredHooks() as cap:
            p = _build_pipeline()
            p.start()
        try:
            (hook,) = cap.hooks
            with unittest.mock.patch.object(
                p._impl, "stop", side_effect=RuntimeError("boom")
            ):
                hook()  # must not propagate
        finally:
            p.stop()


# --------------------------------------------------------------------------------------------
# End-to-end interpreter-exit regression across pipeline topologies.
#
# Each scenario builds a pipeline, keeps a strong reference to it (mimicking a framework
# holding the dataloader), iterates a few items, and returns WITHOUT stopping -- so the
# reference survives to interpreter shutdown. Run in a child process: with the fix the child
# exits cleanly; without it the child hangs at exit and the join below times out.
# --------------------------------------------------------------------------------------------

# Strong reference that outlives the scenario function inside the child process.
_HELD: dict[str, object] = {}


def _double(x: int) -> int:
    return x * 2


def _scenario_plain(mp_ctx: str) -> None:
    """A plain thread-only pipeline (non-daemon event-loop thread)."""
    p = (
        PipelineBuilder()
        .add_source(range(1000), continuous=True)
        .aggregate(8)
        .pipe(_double)
        .add_sink(4)
        .build(num_threads=2)
    )
    _HELD["dl"] = p
    for i, _ in enumerate(p.get_iterator(timeout=60)):
        if i >= 10:
            break


def _child_exit_code(
    scenario: Callable[[str], None], mp_ctx: str, timeout: float = 120.0
) -> int | None:
    """Run ``scenario`` in a child process; return its exit code, or None if it hung."""
    ctx: Any = mp.get_context(mp_ctx)
    proc = ctx.Process(target=scenario, args=(mp_ctx,))
    proc.start()
    proc.join(timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join(10)
        return None
    return proc.exitcode


class InterpreterExitTopologyTest(unittest.TestCase):
    """Held, unstopped pipelines exit cleanly at interpreter exit, across topologies."""

    @unittest.skipUnless(
        "forkserver" in mp.get_all_start_methods(),
        "forkserver start method is unavailable on this platform (e.g. Windows)",
    )
    def test_plain_forkserver(self) -> None:
        """Plain thread-only pipeline exits cleanly (forkserver)."""
        exit_code = _child_exit_code(_scenario_plain, "forkserver")
        self.assertIsNotNone(
            exit_code,
            "Child process hung (did not exit within timeout); the pipeline's "
            "non-daemon event-loop thread likely blocked interpreter shutdown.",
        )
        self.assertEqual(exit_code, 0)

    def test_plain_spawn(self) -> None:
        """Plain thread-only pipeline exits cleanly (spawn)."""
        exit_code = _child_exit_code(_scenario_plain, "spawn")
        self.assertIsNotNone(
            exit_code,
            "Child process hung (did not exit within timeout); the pipeline's "
            "non-daemon event-loop thread likely blocked interpreter shutdown.",
        )
        self.assertEqual(exit_code, 0)


# Note: the `.to(ProcessPoolExecutorConfig)` region topology's interpreter-exit teardown is
# covered alongside the region API tests (which is where `.to()` is defined). The nested
# ``run_pipeline_in_subprocess`` topologies (MTP, or a subprocess-run pipeline whose config also
# has a `.to(...)` region) are covered end-to-end by the standalone repro and the SPDL Hive loader
# validation, not here: nesting a subprocess inside another child of this torch-laden test binary
# makes ``forkserver``/``spawn`` re-import too slow, and ``fork`` unsafe (torch threads).


class SubprocessPutInterruptibleTest(unittest.TestCase):
    """The bridge-stage _put releases parked threads on teardown."""

    def test_put_returns_when_stopped_on_full_queue(self) -> None:
        """_put parked on a full queue exits once its stop event is set."""
        q = mp.get_context("spawn").Queue(maxsize=1)
        try:
            q.put((0, "x"))  # fill the single slot
            stop = threading.Event()
            t = threading.Thread(target=_put, args=(q, (0, "y"), stop))
            t.start()
            # Blocked on the full queue: it must not have returned yet.
            t.join(timeout=1.0)
            self.assertTrue(t.is_alive())
            stop.set()
            t.join(timeout=5.0)
            self.assertFalse(t.is_alive())
        finally:
            q.cancel_join_thread()
            q.close()
