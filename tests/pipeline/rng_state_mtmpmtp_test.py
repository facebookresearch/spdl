# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Global-RNG / sampler consistency across MT, MP and MTP execution modes.

A recurring source of train/eval discrepancies is that switching the data
pipeline between multi-threading (MT) and sub-processing (MTP) changes the
behavior of the *global* random number generators (``random``, ``numpy``,
``torch``) used inside preprocessing functions:

* **MT** runs preprocessing in main-process threads sharing one global RNG.
* **MTP** runs the whole (threaded) pipeline in a single subprocess. Without
  coordination, the subprocess either continues the parent's RNG stream
  (``fork``) or seeds from OS entropy (``spawn``), so its draws are not
  reproducible run-to-run.
* **MP** moves the source/sampler into one subprocess and fans preprocessing out
  to a pool of worker processes.

These tests pin the behavior we actually want:

* SPDL's distributed samplers produce the **same** index sequence in every mode,
  including MP (the sampler lives in the single subprocess, never duplicated),
  across single / multi iteration and continuous / non-continuous sources.
* The global-RNG draws made in preprocessing are reproducible run-to-run in MT
  and MTP.

The random state of the **MP pool worker processes** is intentionally out of
scope: those workers run only the preprocessing fan-out, and per-worker random
state is left to the application.
"""

import multiprocessing as mp
import random
import sys
import unittest
import warnings
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from functools import partial
from typing import Any

import numpy as np
import torch
from parameterized import parameterized
from spdl.pipeline import (
    build_pipeline,
    iterate_in_subprocess,
    run_pipeline_in_subprocess,
)
from spdl.pipeline._random_seed import _capture_rng_initializers
from spdl.pipeline.defs import Pipe, PipelineConfig, SinkConfig, SourceConfig
from spdl.source import DistributedRandomSampler
from spdl.source.utils import embed_shuffle

MT = "mt"
MTP = "mtp"
MP = "mp"

_TIMEOUT = 60.0


# ---------------------------------------------------------------------------
# Top-level (picklable) pipeline ops and source factories.
# ---------------------------------------------------------------------------
def _identity(i: int) -> int:
    return i


def _augment(i: int) -> tuple[int, float, float, float]:
    """Return the index alongside one draw from each global RNG.

    Running this in the pipeline exercises exactly the global state the modes are
    supposed to agree on.
    """
    return (
        i,
        float(np.random.rand()),
        random.random(),
        float(torch.rand(1).item()),
    )


def _rng_triples(epoch_items: Iterable[tuple[Any, ...]]) -> list[tuple[Any, ...]]:
    """Drop the index, keep only the (numpy, random, torch) draw of each item."""
    return [item[1:] for item in epoch_items]


@contextmanager
def _quiet_subprocess() -> Iterator[None]:
    """Silence the expected fork-related warnings emitted when a subprocess (or a
    hoisted ``ProcessPoolExecutor``) is started while pipeline threads are alive."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=(
                r"This process \(pid=\d+\) is multi-threaded, use of "
                r"fork\(\) may lead to deadlocks in the child"
            ),
            category=DeprecationWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r"Hoisting a ProcessPoolExecutor.*",
            category=RuntimeWarning,
        )
        yield


def _config(
    source: Iterable[Any],
    op: Callable[..., Any],
    *,
    executor: ProcessPoolExecutor | None = None,
    concurrency: int = 1,
    continuous: bool = False,
) -> PipelineConfig[Any]:
    return PipelineConfig(
        src=SourceConfig(source, continuous=continuous),
        pipes=[Pipe(op, concurrency=concurrency, executor=executor)],
        sink=SinkConfig(3),
    )


def _run(
    mode: str,
    *,
    source: Iterable[Any],
    op: Callable[..., Any],
    workers: int,
    continuous: bool,
    num_epochs: int,
    mp_context: str | None = None,
) -> list[list[Any]]:
    """Run ``op`` over ``source`` for ``num_epochs`` epochs in the given mode.

    Returns one list of outputs per epoch.
    """
    if mode == MT:
        out: list[list[Any]] = []
        if continuous:
            # A continuous pipeline is built once and re-iterated per epoch.
            config = _config(source, op, concurrency=workers, continuous=True)
            pipeline = build_pipeline(config, num_threads=max(workers, 1))
            with pipeline.auto_stop(timeout=_TIMEOUT):
                for _ in range(num_epochs):
                    out.append(list(pipeline.get_iterator(timeout=_TIMEOUT)))
        else:
            # A non-continuous in-process pipeline is single-pass, so rebuild it
            # per epoch (mirroring how the subprocess path rebuilds per iter()).
            # The shared ``source`` object advances its own epoch state across
            # rebuilds, exactly like the reference.
            for _ in range(num_epochs):
                config = _config(source, op, concurrency=workers, continuous=False)
                pipeline = build_pipeline(config, num_threads=max(workers, 1))
                with pipeline.auto_stop(timeout=_TIMEOUT):
                    out.append(list(pipeline.get_iterator(timeout=_TIMEOUT)))
        return out

    if mode == MTP:
        # Threaded pipeline, moved wholesale into one subprocess (no process pool).
        config = _config(source, op, concurrency=workers, continuous=continuous)
    elif mode == MP:
        # Source/sampler in one subprocess; preprocessing fanned out to a pool of
        # worker processes (hoisted into the main process).
        config = _config(
            source,
            op,
            executor=ProcessPoolExecutor(max_workers=workers),
            concurrency=workers,
            continuous=continuous,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    with _quiet_subprocess():
        src = run_pipeline_in_subprocess(
            config,
            num_threads=max(workers, 1),
            mp_context=mp_context,
            timeout=_TIMEOUT,
        )
        try:
            return [list(src) for _ in range(num_epochs)]
        finally:
            # Deterministically tear the worker subprocess (and any hoisted pools) down.
            finalizer = getattr(src, "_finalizer", None)
            if finalizer is not None:
                finalizer()


def _seed_main(seed: int) -> None:
    random.seed(seed)
    # numpy's stub types ``seed`` as a sequence/array; pass a single-element seq.
    np.random.seed([seed])
    torch.manual_seed(seed)


def _available_start_methods() -> list[str]:
    return [m for m in ("fork", "spawn") if m in mp.get_all_start_methods()]


# ---------------------------------------------------------------------------
# 1. The distributed sampler must produce the same sequence in every mode.
# ---------------------------------------------------------------------------
class TestSamplerConsistency(unittest.TestCase):
    N = 24
    SEED = 7

    def _reference(self, *, reshuffle: bool, num_epochs: int) -> list[list[int]]:
        sampler = DistributedRandomSampler(self.N, rank=0, world_size=1, seed=self.SEED)
        src = embed_shuffle(sampler) if reshuffle else sampler
        return [list(src) for _ in range(num_epochs)]

    def _source(self, *, reshuffle: bool) -> Iterable[int]:
        sampler = DistributedRandomSampler(self.N, rank=0, world_size=1, seed=self.SEED)
        return embed_shuffle(sampler) if reshuffle else sampler

    @parameterized.expand(
        [
            (continuous, reshuffle, num_epochs)
            for continuous in (False, True)
            for reshuffle, num_epochs in ((False, 1), (False, 3), (True, 3))
        ]
    )
    def test_sampler_sequence_matches_across_modes(
        self, continuous: bool, reshuffle: bool, num_epochs: int
    ) -> None:
        """Each mode yields the identical per-epoch index sequence (concurrency=1
        preserves order, so we can compare sequences exactly)."""
        reference = self._reference(reshuffle=reshuffle, num_epochs=num_epochs)

        for mode in (MT, MTP, MP):
            got = _run(
                mode,
                source=self._source(reshuffle=reshuffle),
                op=_identity,
                workers=1,
                continuous=continuous,
                num_epochs=num_epochs,
            )
            self.assertEqual(
                got,
                reference,
                msg=f"mode={mode} continuous={continuous} reshuffle={reshuffle}",
            )


# ---------------------------------------------------------------------------
# 2. Global-RNG draws must be reproducible run-to-run (MT and MTP).
# ---------------------------------------------------------------------------
class TestReproducibleAcrossRuns(unittest.TestCase):
    N = 32
    SEED = 2024

    def _collect(
        self, mode: str, *, start_method: str, continuous: bool
    ) -> list[tuple[Any, ...]]:
        _seed_main(self.SEED)
        # workers=1 keeps a single deterministic stream, so the sequence of draws
        # is well-defined regardless of task scheduling.
        epochs = _run(
            mode,
            source=range(self.N),
            op=_augment,
            workers=1,
            continuous=continuous,
            num_epochs=1,
            mp_context=start_method,
        )
        return _rng_triples(epochs[0])

    @parameterized.expand(
        [
            (mode, start_method, continuous)
            for mode in (MT, MTP)
            for start_method in _available_start_methods()
            for continuous in (False, True)
        ]
    )
    def test_same_seed_same_draws(self, mode, start_method, continuous) -> None:
        """Re-seeding the main process and re-running yields identical draws.

        Pre-fix, MTP with ``spawn`` seeds from OS entropy, so the two runs differ;
        with ``fork`` it continues the parent stream, so the value depends on
        whatever the main process drew beforehand."""
        if mode == MT and start_method != _available_start_methods()[0]:
            # MT runs in-process; the subprocess start method is irrelevant.
            self.skipTest("MT is start-method independent")

        first = self._collect(mode, start_method=start_method, continuous=continuous)
        second = self._collect(mode, start_method=start_method, continuous=continuous)
        self.assertEqual(len(first), self.N)
        self.assertEqual(
            first,
            second,
            msg=f"mode={mode} start_method={start_method} continuous={continuous}",
        )


# ---------------------------------------------------------------------------
# 3. The subprocess continues the main process' *exact current* RNG stream.
#
# This is the distinguishing property of copying the live state (vs. merely
# re-deriving a stream from a seed): if the main process advances its global
# RNGs before the pipeline is built, the subprocess must continue from that
# advanced position, not from the original seed.
# ---------------------------------------------------------------------------
def _draw_global_triples(n: int) -> list[tuple[float, float, float]]:
    """Draw ``n`` triples, one from each global RNG. Top-level + returns a list so
    it is picklable and re-iterable across the subprocess boundary."""
    return [
        (random.random(), float(np.random.rand()), float(torch.rand(1).item()))
        for _ in range(n)
    ]


class TestGlobalRngStateCopiedToSubprocess(unittest.TestCase):
    N = 8

    @parameterized.expand([(m,) for m in _available_start_methods()])
    def test_subprocess_continues_advanced_main_stream(self, start_method: str) -> None:
        """The subprocess resumes the main process' global RNG stream from its
        current (already-advanced) position, identically for ``fork`` and
        ``spawn`` -- proving the live state is copied, not just re-seeded."""
        random.seed(20240101)
        np.random.seed([20240101])
        torch.manual_seed(20240101)
        # Advance all three streams so the captured state is distinct from the
        # seed's starting state.
        _draw_global_triples(5)

        # Snapshot the advanced state via the same mechanism the pipeline uses.
        initializers = _capture_rng_initializers()

        # In-process reference: restore the snapshot, then draw.
        for init in initializers:
            init()
        reference = _draw_global_triples(self.N)

        # Subprocess: restore the same snapshot there, then draw. Must match.
        with _quiet_subprocess():
            src = iterate_in_subprocess(
                partial(_draw_global_triples, self.N),
                initializer=initializers,
                mp_context=start_method,
                timeout=_TIMEOUT,
            )
            try:
                got = list(src)
            finally:
                finalizer = getattr(src, "_finalizer", None)
                if finalizer is not None:
                    finalizer()

        self.assertEqual(len(got), self.N)
        self.assertEqual(got, reference, msg=f"start_method={start_method}")


# ---------------------------------------------------------------------------
# 4. The subinterpreter continues the main interpreter's *exact current* stdlib
#    ``random`` stream.
#
# Subinterpreters cannot import numpy / torch and cannot receive their (non
# -shareable) RNG state objects, so only the stdlib ``random`` state crosses the
# boundary. This pins that the stdlib stream is copied (not re-seeded) and, just
# as importantly, that wiring it through ``run_pipeline_in_subinterpreter`` does
# not raise ``NotShareableError``.
# ---------------------------------------------------------------------------
@unittest.skipUnless(
    sys.version_info >= (3, 14), "subinterpreters require Python 3.14+"
)
class TestStdlibRandomStateCopiedToSubinterpreter(unittest.TestCase):
    N = 8

    def test_subinterpreter_continues_advanced_random_stream(self) -> None:
        """The subinterpreter resumes the main interpreter's stdlib ``random``
        stream from its current (already-advanced) position -- proving the live
        state is copied across the interpreter boundary, not re-seeded.

        The op lives in a stdlib-only helper module: a subinterpreter imports the
        op's defining module to resolve it, and this test module's top-level
        ``import numpy`` / ``import torch`` cannot be imported there."""
        from spdl.pipeline import run_pipeline_in_subinterpreter

        from ._subinterp_rng_ops import draw_random

        random.seed(20240101)
        # Advance the stream so the captured state differs from the seed start.
        [random.random() for _ in range(5)]

        # In-process reference: snapshot the advanced state, then draw N values.
        snapshot = random.getstate()
        reference = [random.random() for _ in range(self.N)]

        # The subinterpreter restores the same snapshot (shipped as a shareable
        # initializer) and must produce the identical sequence.
        random.setstate(snapshot)
        config = _config(range(self.N), draw_random, concurrency=1)
        src = run_pipeline_in_subinterpreter(config, num_threads=1, timeout=_TIMEOUT)
        try:
            got = list(src)
        finally:
            finalizer = getattr(src, "_finalizer", None)
            if finalizer is not None:
                finalizer()

        self.assertEqual(len(got), self.N)
        self.assertEqual(got, reference)
