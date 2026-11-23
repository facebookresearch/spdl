# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

r"""Compare ordinary IPC with explicit caller-owned output arenas.

This benchmark shows how using a reusable shared-memory arena at a process
region's output boundary affects throughput, CPU time, and memory use. It runs
with ordinary IPC (``ipc``) or arena-backed transport (``pool`` or ``ring``).
Both arms use the same private CPU collation with ``torch.cat`` or
``torch.stack``, so only output transport changes. See
:ref:`transparent-region-arenas` for usage, ownership, and cleanup.

Run one CPU trial, for example::

    python benchmark_region_arena.py --transport ipc --size-mib 8 \
        --source-commit local --output ipc

Repeat in a fresh process with ``--transport pool --output pool``, keeping the
other settings identical. For the separate normalized pinned workflow, add
``--pin-memory --h2d`` to both arms on a CUDA-eligible consumer.

.. _region-arena-benchmark-results:

Measured output-transport comparisons
-------------------------------------

The table compares ordinary IPC and a shared segment pool on the output
boundary with the same float32 stack. It does not measure ring transport,
direct collation, or the effect of enabling pinning by itself.

.. image:: /_static/data/region_arena_object_results.png
   :alt: IPC and pool throughput, total process CPU and process-group PSS for
      separate CPU and normalized H2D workflows

.. csv-table:: Caller-owned output transport, three paired process repetitions
   :header-rows: 1

   Workflow,MiB,IPC items/s,SD,Pool items/s,SD,Pool/IPC,IPC CPU ms/batch,Pool CPU ms/batch,IPC group PSS MiB,Pool group PSS MiB,IPC pin prep ms,Pool pin prep ms,IPC H2D ms,Pool H2D ms
   cpu,1,961.593,10.727,1391.059,8.429,1.447,2.790,1.936,2072.906,2101.502,N/A,N/A,N/A,N/A
   cpu,2,663.004,9.193,1024.858,5.221,1.546,4.045,2.591,2105.427,2101.209,N/A,N/A,N/A,N/A
   cpu,4,400.170,8.946,635.073,1.442,1.587,6.736,3.879,2150.338,2163.401,N/A,N/A,N/A,N/A
   cpu,8,212.718,1.906,364.349,2.749,1.713,12.283,6.386,2142.099,2152.003,N/A,N/A,N/A,N/A
   cpu,16,111.125,0.808,187.592,2.670,1.688,24.012,12.249,2280.533,2289.528,N/A,N/A,N/A,N/A
   cpu,32,53.261,1.987,91.577,0.408,1.719,48.480,24.236,2472.044,2483.854,N/A,N/A,N/A,N/A
   cpu,64,25.215,0.389,40.452,0.645,1.604,99.839,53.847,2958.302,2990.772,N/A,N/A,N/A,N/A
   cpu,128,11.668,0.242,19.229,0.231,1.648,208.674,113.906,3869.001,3981.182,N/A,N/A,N/A,N/A
   cpu,256,6.233,0.047,9.381,0.209,1.505,392.927,233.120,5399.178,5954.711,N/A,N/A,N/A,N/A
   cpu,512,2.988,0.253,4.201,0.384,1.406,809.378,528.691,8539.411,9989.385,N/A,N/A,N/A,N/A
   cpu,1024,1.569,0.043,2.191,0.021,1.397,1619.297,1072.624,14844.179,17390.719,N/A,N/A,N/A,N/A
   h2d,1,581.856,16.500,782.544,24.966,1.345,4.031,2.870,2345.479,2331.289,0.234,0.057,0.192,0.206
   h2d,2,382.912,24.700,540.463,18.199,1.411,5.692,3.863,2347.831,2361.354,0.326,0.047,0.179,0.206
   h2d,4,231.340,2.916,314.113,2.522,1.358,9.288,6.282,2399.073,2401.553,0.682,0.044,0.199,0.237
   h2d,8,124.785,1.434,176.602,5.010,1.415,16.657,10.542,2410.577,2385.316,1.416,0.051,0.281,0.287
   h2d,16,60.925,4.263,91.750,2.350,1.506,33.667,19.959,2582.278,2564.002,3.103,0.060,0.472,0.453
   h2d,32,28.745,0.231,42.934,0.218,1.494,69.134,40.918,2760.592,2767.637,6.783,0.070,0.831,0.805
   h2d,64,14.497,0.134,21.020,0.081,1.450,134.353,82.306,3172.829,3309.278,13.974,0.078,1.456,1.426
   h2d,128,7.187,0.083,10.350,0.117,1.440,277.917,173.878,4150.282,4386.985,29.136,0.087,2.728,2.718
   h2d,256,3.611,0.041,5.036,0.061,1.394,560.158,368.678,5879.460,6666.470,58.206,0.090,5.199,5.193
   h2d,512,1.804,0.009,2.514,0.005,1.394,1130.811,755.133,9597.625,10449.500,115.737,0.100,10.112,10.102
   h2d,1024,0.903,0.011,1.294,0.013,1.433,2290.033,1585.558,17403.425,18696.734,231.780,0.122,19.920,19.915

**Column legend.** Items/s divides verified batches by summed timed epoch
durations, including generation, collation, transport, restoration, and full
value checks, but not setup, warmup, or teardown. SD is the sample standard
deviation across three fresh paired trials, not a confidence interval;
Pool/IPC is the ratio of mean throughput.

CPU rows use unpinned transport with host-to-device transfer off. In the
separate H2D workflow the IPC consumer normalizes input with ``pin_memory()``,
which may allocate a timed host copy; the pool uses registered shared backing.
Only output is pinned. CPU producers remain CUDA-uninitialized. Pin prep and
H2D are mean helper wall times per batch. H2D includes traversal, checks, and
synchronization, not pure DMA; full CPU and GPU-return value checks are in the
epoch time, outside the transfer helper. Overlapping phases are not additive.
``N/A`` means pinning or transfer was not run.

Group CPU sums the main process and both reported workers over the enclosing
interval, including possible prefetch and between-epoch work, then divides by
verified batches. Helper processes are excluded. Group PSS is the mean peak
of simultaneous process-group samples taken every 100 ms, not a sum of
independent peaks or an atomic peak. Cgroup memory is separate and may include
other work. Configured payload capacity excludes control and header mappings
and is not residency. Unavailable telemetry is not reported as zero.

What these examples show
~~~~~~~~~~~~~~~~~~~~~~~~

- In every one of the 22 measured cells, each pool trial was faster than each
  IPC trial. CPU-transport throughput is 1.397 to 1.719 times IPC (highest at
  8–32 MiB) with 0.500 to 0.694 times the CPU per batch. In the normalized
  pinned/H2D workflow throughput is 1.345 to 1.506 times IPC with 0.592 to 0.712
  times the CPU per batch.
- The cost is memory at large batches. Up to 64 MiB, CPU-transport group PSS
  differs by at most 1.4%; normalized H2D stays within about 1% through 32 MiB
  and is 4.3% higher at 64 MiB. Pool is then higher: for CPU transport 17.0% at
  512 MiB and 17.2% at 1024 MiB (17390.719 versus 14844.179 MiB), and for H2D
  13.4% at 256 MiB.
- With normalized H2D at 1 GiB, IPC pin preparation is 231.780 ms/batch versus
  0.122 ms for pool, while the transfer helper is 19.920 and 19.915 ms. This is
  a staging-policy workflow comparison, not a pure device-transfer gain.
- Three repetitions on one host do not establish statistical significance, an
  all-workload speedup or a training/fleet-cost conclusion; retain the sample SD
  and memory tradeoff. The 1 GiB cells ran only three to five epochs.

Measurement configuration
~~~~~~~~~~~~~~~~~~~~~~~~~

The 132 measured trials used float32 stack, two workers, one-item boundary
buffering, two pool slots per worker, four parts and sixteen rows. Each trial
checked one excluded warmup and measured at least three epochs of eight batches
for a minimum 15-second enclosing interval. Sizes were powers of two from 1 to
1024 MiB. Arm order rotated across three fresh process pairs on the same host,
using one of the job's two reserved GPUs. Four initial 8 MiB gate trials are
excluded from the table. CPU and normalized-H2D populations contain 279,336 and
154,080 fully verified measured batches respectively. All 264 reported worker
PIDs were confirmed absent after their trials exited.

The synthetic payloads use no data-storage service. Full values are checked,
not serialized in the timing receipts. These results describe the measured
revision; later edits to the example are not new measurements. Absolute
throughput is not directly comparable to the direct-collation example, which
has a different input-generation and verification workload.


"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import multiprocessing as mp
import os
import platform
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
from spdl.io import transfer_tensor_h2d
from spdl.pipeline import (
    build_pipeline,
    PipelineBuilder,
    SharedMemoryRingBuffer,
    SharedMemorySegmentPool,
)
from spdl.pipeline.defs import MAIN_PROCESS, ProcessPoolExecutorConfig

__all__ = ["make_parts", "worker", "verify_batch", "run_epoch", "run_benchmark", "main"]


@dataclass(frozen=True)
class _Batches:
    count: int

    def __iter__(self) -> Iterator[int]:
        yield from range(self.count)


def make_parts(
    index: int, *, parts: int, rows: int, columns: int, dtype: str
) -> list[torch.Tensor]:
    """Create fresh deterministic CPU inputs for an ordinary collation stage.

    Args:
        index: Batch identifier, used in each part's repeated value.
        parts: Positive number of input tensors.
        rows: Positive rows per part.
        columns: Positive columns per part.
        dtype: ``float16``, ``float32``, ``float64`` or ``uint8``.

    Returns:
        Fresh tensors, with part ``i`` filled by ``(index + i) % 251``.
        For example, ``make_parts(7, parts=2, rows=2, columns=3, dtype="uint8")``
        creates two 2-by-3 inputs without caching another batch's data.
    """
    if min(parts, rows, columns) <= 0 or dtype not in (
        "float16",
        "float32",
        "float64",
        "uint8",
    ):
        raise ValueError("Positive dimensions and a supported CPU dtype are required")
    return [
        torch.full((rows, columns), (index + part) % 251, dtype=getattr(torch, dtype))
        for part in range(parts)
    ]


def worker(
    index: int, *, parts: int, rows: int, columns: int, dtype: str, op: str
) -> dict[str, Any]:
    """Collate a fresh private CPU batch, identically for every transport arm.

    Args:
        index: Unique batch identifier.
        parts: Input tensor count.
        rows: Rows in each part.
        columns: Columns in each part.
        dtype: Supported CPU dtype name.
        op: ``cat`` or ``stack``, using the ordinary allocating Torch operation.

    Returns:
        Identifier, normal collated tensor, actual producer PID and observed
        CPU-only producer state. For example, ``worker(7, parts=2, rows=2,
        columns=3, dtype="float32", op="cat")`` creates a 4-by-3 tensor.
        No destination reservation, ``out=`` or allocation ticket is involved.
    """
    inputs = make_parts(index, parts=parts, rows=rows, columns=columns, dtype=dtype)
    if op == "cat":
        value = torch.cat(inputs)
    elif op == "stack":
        value = torch.stack(inputs)
    else:
        raise ValueError("Operation must be cat or stack")
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU region producer unexpectedly initialized CUDA")
    return {
        "index": index,
        "value": value,
        "producer_pid": os.getpid(),
        "producer_cuda_initialized": False,
    }


def verify_batch(
    batch: dict[str, Any], args: argparse.Namespace, value: torch.Tensor | None = None
) -> torch.Tensor:
    """Reconstruct and verify the entire received CPU batch, shape, dtype and proof.

    Args:
        batch: Ordinary result dictionary from :py:func:`worker`.
        args: Parts, rows, columns, dtype and operation dimensions.
        value: Optional prepared pinned tensor; otherwise validate the received tensor.

    Returns:
        The independently reconstructed expected tensor, for optional full-value
        GPU verification. ``verify_batch(worker(...), args)`` verifies a complete
        batch; this work remains inside the measured epoch.

    Raises:
        RuntimeError: Any value, shape, dtype or CPU producer evidence differs.
    """
    if (
        batch.get("producer_cuda_initialized") is not False
        or type(batch.get("producer_pid")) is not int
    ):
        raise RuntimeError("Region result lacks actual CPU producer evidence")
    inputs = make_parts(
        batch["index"],
        parts=args.parts,
        rows=args.rows,
        columns=args.columns,
        dtype=args.dtype,
    )
    expected = torch.cat(inputs) if args.op == "cat" else torch.stack(inputs)
    actual = batch["value"] if value is None else value
    if (
        actual.dtype != expected.dtype
        or actual.shape != expected.shape
        or not torch.equal(actual, expected)
    ):
        raise RuntimeError("Region collation content/shape/dtype mismatch")
    return expected


def _prepare(
    value: torch.Tensor, args: argparse.Namespace, warmup: bool
) -> tuple[torch.Tensor, float, bool]:
    if not args.pin_memory:
        return value, 0.0, False
    start = time.perf_counter()
    original = value.data_ptr()
    if args.transport == "ipc":
        value = value.pin_memory()
    elif not value.is_pinned():
        raise RuntimeError("Requested pinned arena restored unpinned storage")
    if not value.is_pinned():
        raise RuntimeError("Public pin normalization did not produce pinned storage")
    elapsed = time.perf_counter() - start
    copied = value.data_ptr() != original
    if warmup:
        alias = value.pin_memory()
        if alias.data_ptr() != value.data_ptr():
            raise RuntimeError("Prepared pinned storage was not reused by pin_memory")
    return value, elapsed, copied


def run_epoch(
    pipeline: Any, args: argparse.Namespace, *, warmup: bool = False
) -> dict[str, Any]:
    """Consume and fully verify one epoch with explicitly scoped public preparation.

    Args:
        pipeline: Started public pipeline producing ordinary collated batches.
        args: Count, dimensions, transport, pinning and optional H2D controls.
        warmup: Check pin alias identity in the excluded warmup only.

    Returns:
        Epoch/first-result wall time, actual checked batch and producer PID counts,
        public pin-preparation time, observed normalization-copy count and optional
        H2D helper time. Use ``run_epoch(pipeline, args)`` for a timed epoch.
        CPU reconstruction and GPU return-value verification are included in epoch
        time; H2D helper time includes public completion, not pure DMA bandwidth.
    """
    start, first = time.perf_counter(), None
    seen: set[int] = set()
    producers: set[int] = set()
    pin_s, h2d_s, copied = 0.0, 0.0, 0
    for batch in pipeline.get_iterator(timeout=120):
        if first is None:
            first = time.perf_counter()
        index = batch["index"]
        if index in seen:
            raise RuntimeError("Region benchmark duplicate batch")
        value, preparation, normalized_copy = _prepare(batch["value"], args, warmup)
        expected = verify_batch(batch, args, value)
        seen.add(index)
        producers.add(batch["producer_pid"])
        pin_s += preparation
        copied += int(normalized_copy)
        if args.h2d:
            before = time.perf_counter()
            device = transfer_tensor_h2d(value, num_caches=0)
            torch.cuda.synchronize()
            h2d_s += time.perf_counter() - before
            if not torch.equal(device.cpu(), expected):
                raise RuntimeError("Public H2D whole-value mismatch")
            del device
        del batch, value, expected
    if seen != set(range(args.batches)) or first is None:
        raise RuntimeError("Region benchmark missing or extra batches")
    return {
        "elapsed_s": time.perf_counter() - start,
        "first_result_s": first - start,
        "items": len(seen),
        "producer_pids": sorted(producers),
        "pin_prepare_s": pin_s if args.pin_memory else None,
        "pin_checked_batches": len(seen) if args.pin_memory else 0,
        "normalization_copied_batches": copied,
        "h2d_helper_s": h2d_s if args.h2d else None,
    }


def _initializer(pids: Any) -> None:
    torch.set_num_threads(1)
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU producer initializer unexpectedly initialized CUDA")
    pids.put(os.getpid())


def _cpu(pid: int) -> tuple[int, float] | None:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return int(fields[19]), (int(fields[11]) + int(fields[12])) / os.sysconf(
            "SC_CLK_TCK"
        )
    except (OSError, ValueError, IndexError):
        return None


def _worker_cpu(
    before: list[tuple[int, float] | None], after: list[tuple[int, float] | None]
) -> float | None:
    if any(value is None for value in before + after):
        return None
    total = 0.0
    for old, new in zip(before, after, strict=True):
        assert old is not None and new is not None
        if old[0] != new[0]:
            raise RuntimeError(
                "Reported worker PID identity changed during measurement"
            )
        total += new[1] - old[1]
    return total


def _pss(pid: int) -> int | None:
    try:
        for line in Path(f"/proc/{pid}/smaps_rollup").read_text().splitlines():
            if line.startswith("Pss:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        return None
    return None


def _cgroup() -> Path | None:
    try:
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            if line.startswith("0::"):
                path = (
                    Path("/sys/fs/cgroup")
                    / line.split("::", 1)[1].lstrip("/")
                    / "memory.current"
                )
                if path.is_file():
                    return path
    except OSError:
        return None
    return None


class _MemorySampler:
    def __init__(self, pids: list[int]) -> None:
        self.pids = pids
        self.group_pss_peak_bytes: int | None = None
        self.cgroup_peak_bytes: int | None = None
        self.samples = 0
        self._path = _cgroup()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self) -> None:
        values = [_pss(pid) for pid in self.pids]
        if all(value is not None for value in values):
            self.group_pss_peak_bytes = max(
                self.group_pss_peak_bytes or 0,
                sum(value for value in values if value is not None),
            )
        if self._path is not None:
            try:
                value = int(self._path.read_text())
                self.cgroup_peak_bytes = max(self.cgroup_peak_bytes or 0, value)
            except (OSError, ValueError):
                pass
        self.samples += 1

    def _run(self) -> None:
        while not self._stop.wait(0.1):
            self._sample()

    def __enter__(self) -> _MemorySampler:
        self._sample()
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        self._thread.join()
        self._sample()


def _retire_caller_arenas(
    arenas: tuple[object, ...], primary: BaseException | None = None
) -> None:
    failure = primary
    for arena in arenas:
        assert isinstance(arena, (SharedMemorySegmentPool, SharedMemoryRingBuffer))
        try:
            arena.close()
            arena.unlink()
        except Exception as error:
            if failure is None:
                failure = error
            else:
                failure.add_note(f"Caller arena cleanup also failed: {error!r}")
    if primary is None and failure is not None:
        raise failure


def _arena_objects(args: argparse.Namespace) -> tuple[object, ...]:
    if args.transport == "ipc":
        return ()
    unit = (args.size_mib << 20) * args.buffer_size + 4096
    objects: list[object] = []
    try:
        for _ in range(args.workers):
            arena = (
                SharedMemorySegmentPool(
                    unit, args.slots, acquire_timeout=30, pin_memory=args.pin_memory
                )
                if args.transport == "pool"
                else SharedMemoryRingBuffer(
                    unit * args.slots, acquire_timeout=30, pin_memory=args.pin_memory
                )
            )
            objects.append(arena)
        return tuple(objects)
    except BaseException as error:
        _retire_caller_arenas(tuple(objects), error)
        raise


def _close_caller_arenas(arenas: tuple[object, ...], pids: list[int]) -> None:
    if not arenas:
        return
    if os.name != "posix" or len(set(pids)) != len(arenas) or len(pids) != len(arenas):
        raise RuntimeError("Worker reap cannot be confirmed; caller arenas retained")
    for pid in pids:
        if type(pid) is not int or pid <= 0:
            raise RuntimeError("Invalid worker PID; caller arenas retained")
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            continue
        except OSError as error:
            raise RuntimeError(
                f"Worker PID {pid} reap cannot be confirmed; caller arenas retained"
            ) from error
        raise RuntimeError(f"Worker PID {pid} still exists; caller arenas retained")
    _retire_caller_arenas(arenas)


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    """Run one borrowed output-arena trial with matched ordinary CPU collation.

    Args:
        args: CLI dimensions, public IPC/pool/ring choice, repetition/time bounds
            and optional pinned H2D workflow. ``runs`` counts timed epochs, not
            independent trials; obtain an equivalent Namespace through :py:func:`main`.

    Returns:
        Fully checked timing, main/all-reported-worker CPU, sampled group PSS and
        separate cgroup memory. One warmup is excluded. For comparison, repeat
        fresh invocations with identical dimensions/resources and rotate arm order.
        This entrypoint never launches infrastructure jobs or performance sweeps.
    """
    if args.transport != "ipc" and os.name != "posix":
        raise ValueError("Caller-owned arena trials require POSIX worker PID checks")
    if (
        args.parts * args.rows * args.columns * getattr(torch, args.dtype).itemsize
        != args.size_mib << 20
    ):
        raise ValueError("Dimensions do not match configured output byte size")
    context = mp.get_context("spawn")
    reports = context.Queue(args.workers)
    arenas: tuple[object, ...] = ()
    pids: list[int] = []
    pipeline = None
    build_attempted = False
    primary: BaseException | None = None
    try:
        arenas = _arena_objects(args)
        config = (
            PipelineBuilder()
            .add_source(_Batches(args.batches), continuous=True)
            .to(
                ProcessPoolExecutorConfig(
                    max_workers=args.workers,
                    mp_context="spawn",
                    initializer=_initializer,
                    initargs=(reports,),
                ),
                buffer_size=args.buffer_size,
            )
            .pipe(
                partial(
                    worker,
                    parts=args.parts,
                    rows=args.rows,
                    columns=args.columns,
                    dtype=args.dtype,
                    op=args.op,
                )
            )
            .to(
                MAIN_PROCESS,
                buffer_size=args.buffer_size,
                arena=(arenas[0] if len(arenas) == 1 else arenas) if arenas else None,
            )
            .add_sink(2)
            .get_config()
        )
        # Building a process region can start workers before it returns a Pipeline.
        build_attempted = True
        pipeline = build_pipeline(config, num_threads=3, max_failures=0)
        with pipeline.auto_stop():
            pids = [int(reports.get(timeout=120)) for _ in range(args.workers)]
            if len(set(pids)) != args.workers or os.getpid() in pids:
                raise RuntimeError("Worker initializer PID reports are not distinct")
            run_epoch(pipeline, args, warmup=True)
            before = [_cpu(pid) for pid in pids]
            main_start, wall_start = time.process_time(), time.perf_counter()
            epochs = []
            with _MemorySampler([os.getpid(), *pids]) as memory:
                while (
                    len(epochs) < args.runs
                    or time.perf_counter() - wall_start < args.duration_sec
                ):
                    epoch = run_epoch(pipeline, args)
                    if not set(epoch["producer_pids"]).issubset(pids):
                        raise RuntimeError(
                            "Stage PID does not match initializer reports"
                        )
                    epochs.append(epoch)
                live_cpu = _worker_cpu(before, [_cpu(pid) for pid in pids])
                main_cpu = time.process_time() - main_start
        seconds = sum(epoch["elapsed_s"] for epoch in epochs)
        items = args.batches * len(epochs)
        capacity = (
            0
            if args.transport == "ipc"
            else args.workers
            * args.slots
            * ((args.size_mib << 20) * args.buffer_size + 4096)
        )
        return {
            "source_commit": args.source_commit,
            "source_commit_scope": (
                "caller label; supplied metadata is not runtime authentication"
            ),
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "numpy": str(np.__version__),
            "workflow": "public-transparent-output-arena"
            if args.transport != "ipc"
            else "public-ordinary-ipc",
            "transport": args.transport,
            "op": args.op,
            "dtype": args.dtype,
            "workers": args.workers,
            "buffer_size": args.buffer_size,
            "slots": args.slots,
            "parts": args.parts,
            "rows": args.rows,
            "columns": args.columns,
            "actual_output_bytes": args.size_mib << 20,
            "epochs": len(epochs),
            "items": items,
            "verified_output_batches": items,
            "elapsed_s": seconds,
            "items_per_s": items / seconds,
            "output_bytes_per_s": items * (args.size_mib << 20) / seconds,
            "pipeline_epoch_elapsed_s": [epoch["elapsed_s"] for epoch in epochs],
            "first_result_s": [epoch["first_result_s"] for epoch in epochs],
            "main_cpu_s": main_cpu,
            "live_worker_cpu_s": live_cpu,
            "reported_worker_count": len(pids),
            "initializer_reported_worker_pids": pids,
            "group_pss_peak_bytes": memory.group_pss_peak_bytes,
            "cgroup_memory_peak_bytes": memory.cgroup_peak_bytes,
            "memory_samples": memory.samples,
            "configured_arena_payload_capacity_bytes": capacity,
            "pin_memory": args.pin_memory,
            "consumer_pin_prepare_s": sum(epoch["pin_prepare_s"] for epoch in epochs)
            if args.pin_memory
            else None,
            "consumer_pin_checked_batches": items if args.pin_memory else 0,
            "ipc_pin_normalization_copied_batches": sum(
                epoch["normalization_copied_batches"] for epoch in epochs
            ),
            "h2d_helper_s": sum(epoch["h2d_helper_s"] for epoch in epochs)
            if args.h2d
            else None,
            "h2d_verified_batches": items if args.h2d else 0,
            "all_producers_cuda_uninitialized": True,
            "timing_scope": (
                "fresh CPU parts/cat-or-stack, output transport, public pin preparation "
                "and whole-value verification; warmup excluded; H2D and GPU return "
                "verification included only when requested"
            ),
            "cpu_scope": (
                "main plus every initializer-reported live region worker; enclosing "
                "snapshots may include prefetch/between-epoch work; helper/"
                "resource-tracker processes excluded"
            ),
            "memory_scope": (
                "100ms sampled sum-of-process PSS peak for main/reported workers; "
                "separate cgroup view is not additive and may include other work"
            ),
            "capacity_scope": (
                "configured payload storage only; excludes control/header mappings; "
                "not physical residency"
            ),
            "pin_scope": (
                "IPC pin_memory normalization may allocate a timed host copy; arena "
                "pinning uses existing backend consumer behavior, not the same "
                "physical source allocation as IPC"
            ),
            "h2d_scope": (
                "public helper plus synchronize wall time, not pure DMA; CPU "
                "reconstruction and GPU-to-CPU whole-value verification are in epoch "
                "time, outside the helper field"
            ),
            "missing_telemetry": [
                "private transport copy/wait/fence counters",
                "exact occupancy peaks",
                "external registration byte counters",
            ],
        }
    except BaseException as error:
        primary = error
        raise
    finally:
        # The benchmark owns its public arena objects, not the pipeline. A
        # stopped continuous pipeline still owns prefetched restored batches, so
        # release it before retiring the storage those batches alias.
        pipeline = None
        gc.collect()
        try:
            if not build_attempted:
                _retire_caller_arenas(arenas, primary)
            elif len(pids) == args.workers:
                _close_caller_arenas(arenas, pids)
            # Once build was attempted, missing PID reports are not proof of reap.
        except Exception as failure:
            if primary is None:
                primary = failure
                raise
            primary.add_note(f"Caller arena cleanup also failed: {failure!r}")
        finally:
            try:
                reports.close()
                reports.join_thread()
            except Exception as failure:
                if primary is None:
                    raise
                primary.add_note(
                    f"Worker report queue cleanup also failed: {failure!r}"
                )


def _write(row: dict[str, Any], prefix: str) -> None:
    path = Path(prefix)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_name(f"{path.name}.json").write_text(
        json.dumps(row, indent=2, sort_keys=True) + "\n"
    )
    flat = {
        key: json.dumps(value, sort_keys=True)
        if isinstance(value, (dict, list, tuple))
        else value
        for key, value in row.items()
    }
    with path.with_name(f"{path.name}.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat))
        writer.writeheader()
        writer.writerow(flat)
    print(json.dumps(row, sort_keys=True), flush=True)


def main(build_identity: dict[str, Any] | None = None) -> None:
    """Run one public IPC or transparent output-arena invocation and write receipts.

    Args:
        build_identity: Optional caller-supplied metadata with ``code_commit``
            source label. Recorded, not interpreted as hidden pipeline configuration.

    Returns:
        None. For example, ``python benchmark_region_arena.py --transport pool
        --size-mib 8 --source-commit local --output region`` compares ordinary
        collation via the public factory interface. Independent paired repeats
        and infrastructure allocation/launches remain the caller's responsibility.
    """
    parser = argparse.ArgumentParser(
        description="Public ordinary-collation IPC versus transparent output arenas"
    )
    parser.add_argument("--transport", choices=("ipc", "pool", "ring"), default="ipc")
    parser.add_argument("--op", choices=("cat", "stack"), default="stack")
    parser.add_argument(
        "--dtype", choices=("float16", "float32", "float64", "uint8"), default="float32"
    )
    for name, default in (
        ("size-mib", 8),
        ("workers", 2),
        ("buffer-size", 1),
        ("slots", 2),
        ("parts", 4),
        ("rows", 16),
        ("batches", 8),
        ("runs", 3),
    ):
        parser.add_argument(f"--{name}", type=int, default=default)
    parser.add_argument("--duration-sec", type=float, default=15)
    parser.add_argument(
        "--pin-memory",
        action="store_true",
        help=(
            "public consumer pinning; IPC normalization copy is timed; "
            "requires CUDA eligibility"
        ),
    )
    parser.add_argument(
        "--h2d",
        action="store_true",
        help=(
            "include public H2D helper/completion and whole GPU value verification; "
            "requires --pin-memory"
        ),
    )
    parser.add_argument(
        "--source-commit",
        required=build_identity is None,
        default=build_identity["code_commit"] if build_identity else None,
    )
    parser.add_argument("--output", default="region-results")
    args = parser.parse_args()
    if (
        min(
            args.size_mib,
            args.workers,
            args.buffer_size,
            args.slots,
            args.parts,
            args.rows,
            args.batches,
            args.runs,
        )
        <= 0
        or not math.isfinite(args.duration_sec)
        or args.duration_sec < 0
    ):
        parser.error("counts/sizes must be positive; duration finite and non-negative")
    denominator = args.parts * args.rows * getattr(torch, args.dtype).itemsize
    size = args.size_mib << 20
    if size % denominator or size < denominator:
        parser.error("output size must form whole columns for parts/rows/dtype")
    args.columns = size // denominator
    if args.h2d and not args.pin_memory:
        parser.error(
            "--h2d requires --pin-memory for the separate normalized pinned workflow"
        )
    if args.pin_memory and not torch.cuda.is_available():
        parser.error("--pin-memory requires a CUDA-eligible consumer")
    if (
        build_identity is not None
        and args.source_commit != build_identity["code_commit"]
    ):
        parser.error("--source-commit differs from the supplied build label")
    torch.set_num_threads(1)
    row = run_benchmark(args)
    if build_identity is not None:
        row["build_identity"] = build_identity
    _write(row, args.output)


if __name__ == "__main__":
    main()
