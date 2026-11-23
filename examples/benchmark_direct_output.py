# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

r"""Compare private collation plus copy with direct public arena allocation.

Both arms generate the same fresh CPU inputs and use an ordinary public
:py:class:`spdl.pipeline.defs.ProcessPoolExecutorConfig` region, parent-created
:py:class:`spdl.pipeline.SharedMemorySegmentPool` instances passed via public
initializer/initargs, ordinary IPC of opaque allocation tickets,
and main-process ``pool.recover(ticket)`` contexts. The ``copy`` control
collates into a private tensor, then uses ``output.copy_(batch)``; ``direct`` uses
``torch.cat/stack(parts, out=output)`` on a caller-created Torch view over
``unit.allocate_bytes`` storage. Shape/dtype metadata travel beside the ticket;
recovery yields bytes that this caller interprets. There is
no private executor subclass, decoder, queue wrapper or SPDL import.

Run the matched public workflow
--------------------------------

In a CUDA-capable environment with SPDL IO, NumPy and Torch installed::

    python examples/benchmark_direct_output.py --sizes 8 --dtypes float32 \
        --ops stack --workers 2 --parts 4 --rows 16 --slots 2 --batches 8 \
        --runs 3 --repeats 3 --duration-sec 15 --source-commit local \
        --output direct-allocation-results

Both performance arms use ``pin_memory=True`` and the public synchronized
:py:func:`spdl.io.transfer_tensor_h2d` helper. Each matched repetition creates
fresh consumer/worker processes; order rotates. The CLI launches local processes
only, not infrastructure jobs. ``--self-test --source-commit local`` selects
bounded CPU/unpinned/H2D-off checks, explicitly **not performance evidence**.
Size requests round up to complete input shapes; actual delivered bytes are
reported. ``source_commit`` and optional build metadata are caller labels, not
runtime source authentication. Exact-source packaging and infrastructure
scheduling are caller-owned, not actions performed by this example.

``make_parts``, ``collate_private``, ``collate_direct``, ``run_epoch``,
``run_benchmark`` and ``run_repetitions`` are executable public entrypoints.
Worker phase timers cover actual source generation, cat/stack, explicit copy,
reservation acquisition, output allocation and seal. Main IPC latency is receipt
minus post-seal monotonic time, including queueing/scheduling/backpressure;
recovery times the public recover call plus context entry. H2D host/completion
includes the public helper's traversal, pin checks and synchronization, not pure DMA. Full-value
CPU/GPU verification is excluded from those phase timers but included in complete
checked epochs and CPU telemetry. Do not add overlapping phase durations.

The consumer and producer CPU measurements cover the same enclosing interval,
which can include prefetch and between-epoch work.
RSS/PSS peaks are sampled sums across main and all workers within one sample,
not sums of individual peaks; sequential read spans and missing observations
are reported. RSS double-counts shared mappings; PSS is proportional. Setup,
one checked warmup epoch and teardown are excluded. The pinned warmup additionally
verifies public ``pin_memory()`` aliases both tensor and underlying storage pointers;
this check is outside measured phases. Public ``pool.close()`` terminal cancellation
runs in cleanup's finally path before public stop/join; physical retirement is
retried after complete initializer receipts and OS-confirmed worker reap. Only
then are shared names unlinked and initializer queues closed.
Failed cleanup retains these owners durably in a process-local lifecycle registry,
independent of tracebacks or GC; ``_drain_failed_cleanups()`` explicitly retries
only after actual worker reap. Throughput uses the sum of
completed epoch times, not the requested enclosing duration. No unqualified
speedup, bandwidth-cause, training-throughput or no-staging claim follows.

Tickets are opaque, with only the read-only ``arena_id`` routing property exposed.
Their implementation details stay private: this example imports no allocation
type and inspects no descriptor fields. Ordinary stdlib pickle has no ownership
effects. Producer PIDs come from ordinary initializer/envelope ``os.getpid()``
receipts, not ticket metadata. Route by ``pools[ticket.arena_id]``; recover once,
or ``pool.discard(ticket)`` once if never read. Retained tensor/storage aliases postpone reuse independently of wrapper
lifetime. See :doc:`../optimization_guide/direct_allocations` for the public
recipe, sizing, lifetime contract and measurement caveats.

.. _direct-allocation-benchmark-results:

Measured caller-managed allocation comparisons
----------------------------------------------

Both arms use pinned shared storage, ordinary allocation-ticket IPC and the
same public synchronized host-to-device (H2D) helper. This is a comparison of
collation paths, not of pinning policies or isolated device-transfer rates.

.. image:: /_static/data/direct_allocation_results.png
   :alt: COPY and DIRECT throughput, CPU time and sampled group memory by size

.. image:: /_static/data/direct_allocation_relative.png
   :alt: DIRECT throughput improvement and CPU time saved relative to COPY

The relative plots compare the arm means. Zero means no change; positive
values mean higher throughput or less CPU time per batch with DIRECT.

.. image:: /_static/data/direct_allocation_phases.png
   :alt: COPY and DIRECT collation, explicit copy and H2D helper wall times

.. csv-table:: Caller-managed pinned allocations, three paired repetitions
   :header-rows: 1

   Output MiB,Arm,Trials,Batches/s,Sample SD (batches/s),DIRECT/COPY,Main CPU ms/batch,Worker CPU ms/batch,Total CPU ms/batch,Group peak PSS MiB,Generation ms/batch,Collation ms/batch,Explicit copy ms/batch,IPC latency ms/batch,Recovery ms/batch,H2D helper ms/batch,Verified batches
   1,COPY,3,261.5505,3.6851,—,5.575,2.477,8.053,2088.3,1.280,0.089,0.127,11.252,0.103,0.160,11752
   1,DIRECT,3,260.4937,6.2483,0.9960,5.653,2.410,8.063,2092.0,1.309,0.106,0.000,11.321,0.102,0.164,11712
   2,COPY,3,148.9848,0.9338,—,9.771,3.980,13.751,2101.8,2.489,0.187,0.241,19.822,0.093,0.176,6696
   2,DIRECT,3,149.0847,1.2409,1.0007,9.722,3.720,13.442,2099.0,2.487,0.183,0.000,19.870,0.092,0.174,6704
   4,COPY,3,77.0393,0.4951,—,18.762,7.299,26.061,2154.8,5.064,0.383,0.591,38.314,0.099,0.214,3472
   4,DIRECT,3,76.0346,0.3213,0.9870,18.987,6.833,25.820,2111.7,5.119,0.426,0.000,38.980,0.095,0.221,3416
   8,COPY,3,34.5428,0.2848,—,38.252,14.579,52.831,2214.0,11.736,0.621,1.496,85.400,0.126,0.347,1568
   8,DIRECT,3,34.7208,0.1111,1.0052,37.927,13.033,50.960,2192.7,11.587,0.741,0.000,85.663,0.124,0.342,1576
   16,COPY,3,16.7211,0.0952,—,80.358,28.789,109.147,2225.0,23.830,1.937,2.871,176.744,0.147,0.532,768
   16,DIRECT,3,16.7361,0.1729,1.0009,80.438,25.741,106.179,2228.4,23.706,1.645,0.000,177.606,0.153,0.538,760
   32,COPY,3,7.9251,0.0228,—,171.316,59.222,230.538,2464.6,50.122,3.436,5.776,373.201,0.159,0.855,360
   32,DIRECT,3,7.9892,0.0211,1.0081,170.103,53.583,223.686,2402.9,50.422,3.220,0.000,372.589,0.158,0.854,360
   64,COPY,3,3.8512,0.0764,—,362.773,129.271,492.043,2915.0,110.136,6.593,12.000,767.451,0.158,1.483,192
   64,DIRECT,3,3.7795,0.0260,0.9814,368.387,117.448,485.835,2780.5,110.315,6.355,0.000,787.262,0.159,1.484,192
   128,COPY,3,1.7997,0.0091,—,797.154,273.333,1070.487,4007.0,228.936,15.564,24.410,1637.853,0.161,2.699,96
   128,DIRECT,3,1.7807,0.0035,0.9894,806.621,247.188,1053.808,3742.8,232.548,12.265,0.000,1666.829,0.157,2.710,96
   256,COPY,3,0.8567,0.0043,—,1733.994,563.194,2297.189,6002.9,464.809,42.625,48.784,3451.479,0.154,5.165,72
   256,DIRECT,3,0.8619,0.0078,1.0061,1723.528,503.889,2227.417,5437.5,470.862,24.522,0.000,3447.512,0.153,5.166,72
   512,COPY,3,0.4300,0.0005,—,3625.334,1181.528,4806.862,10384.1,929.550,94.572,97.788,6886.527,0.153,10.088,72
   512,DIRECT,3,0.4301,0.0024,1.0003,3611.587,1046.944,4658.532,9106.4,941.033,52.471,0.000,6929.197,0.151,10.083,72
   1024,COPY,3,0.2103,0.0030,—,7858.060,2344.028,10202.088,18490.6,1920.150,147.406,195.594,14055.836,0.152,19.870,72
   1024,DIRECT,3,0.2073,0.0004,0.9859,7992.325,2106.806,10099.131,16410.5,1931.294,97.060,0.000,14311.654,0.153,19.891,72

**Column legend.** Each row is the mean of three fresh trials. Batches/s is
verified batches divided by summed measured epoch time; setup, warmup and
teardown are excluded. Sample SD describes the three trial rates, not a
confidence interval. DIRECT/COPY divides the two arm means. Verified batches
is a total. Total CPU time per batch adds CPU time spent by the main consumer
process and both producer workers, then divides by verified batches for each
trial. It is not elapsed time; the measured interval can include prefetch and
between-epoch work. Group PSS
is the mean of each trial's peak sampled group sum, not a sum of independent
peaks, an atomic sample or arena capacity.

Phase columns average per-batch wall times within trials, then across trials.
IPC includes queueing and backpressure; recovery includes context entry. H2D
includes traversal, pin checks and synchronization; it is not pure DMA.
Full-value verification contributes to throughput and CPU but is outside the
named helper timers. Overlapping phases cannot be added as a critical path.
DIRECT's zero explicit-copy time does not imply a copy-free workflow.

What these examples show
~~~~~~~~~~~~~~~~~~~~~~~~

- DIRECT removes the explicit copy, but the mean throughput ratio ranges from
  0.9814 to 1.0081 across sizes. DIRECT is slower at 1, 4, 64, 128 and 1024 MiB.
  The widest relative spread is at 1 MiB for DIRECT (sample SD 2.4% of its mean)
  and at 64 MiB for COPY (2.0%). These are small, workload-specific differences.
- At 512 MiB, total CPU per batch falls from 4806.9 to 4658.5 ms, and sampled
  group peak PSS falls from 10384.1 to 9106.4 MiB. At 1024 MiB, PSS falls from
  18490.6 to 16410.5 MiB, while throughput is about 1.4% lower.
- At 1024 MiB, COPY spends 147.4 ms/batch collating and 195.6 ms explicitly
  copying; DIRECT spends 97.1 ms collating without the explicit copy. The
  H2D helper is nearly unchanged (19.87 vs 19.89 ms/batch). Less collation
  and copy work does not by itself imply faster end-to-end delivery.

Measurement configuration and teardown
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One attempt on one host and one selected GPU completed the 66 measured
trials: 11 sizes from 1 to 1024 MiB, three fresh COPY/DIRECT pairs per size,
with arm order rotated. Float32 stack used two workers, four parts, sixteen
rows, two slots per worker and eight batches per epoch. Each trial ran at
least three epochs and had a 15-second minimum enclosing interval. The
measurements include 50,152 fully verified batches. They exclude 528 warmup
batches and four CPU/CUDA correctness gates. These are synthetic component
measurements, not training results or evidence from independent hosts.

**Teardown caveat.** Measured full-value, pinning and H2D checks passed, and
all 140 reported worker PIDs across measured trials and gates were absent
at the post-exit checks. Nevertheless, all 70 trial stderr logs contain pool
closing or shutdown tracebacks and wrapped pipeline errors. The measured
source completes the checked epochs before creator close, stop and reap;
the error signatures are consistent with cancellation of prefetched work
in cleanup. The logs do not timestamp each error against the measured window,
so this is not evidence of exception-free shutdown or general quiescence.
Unfiltered logs and exact measured source identity are retained separately.

"""  # noqa: B950 - RST CSV rows preserve one wide, auditable metrics table.

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import platform
import statistics
import threading
import time
from _thread import RLock
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from itertools import product
from multiprocessing.queues import Queue
from multiprocessing.sharedctypes import Synchronized
from pathlib import Path
from queue import Empty
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import torch
from spdl.io import transfer_tensor_h2d
from spdl.pipeline import (
    build_pipeline,
    Pipeline,
    PipelineBuilder,
    SharedMemorySegmentPool,
)
from spdl.pipeline.defs import MAIN_PROCESS, ProcessPoolExecutorConfig

__all__ = [
    "make_parts",
    "collate_private",
    "collate_direct",
    "verify_batch",
    "run_epoch",
    "run_benchmark",
    "run_isolated",
    "run_repetitions",
    "parse_args",
    "main",
]

_ARMS: tuple[str, ...] = ("copy", "direct")
_DTYPES: tuple[str, ...] = ("float16", "float32", "float64", "uint8")
_SCHEMA: str = "public-direct-allocations-v2"
_DEFAULT_TRIAL_TIMEOUT = 1800.0
_WORKER_POOL: SharedMemorySegmentPool | None = None


@dataclass(frozen=True)
class _Batches:
    batches: int

    def __iter__(self) -> Iterator[int]:
        yield from range(self.batches)


def make_parts(
    sequence: int, *, parts: int, rows: int, columns: int, dtype: str = "float32"
) -> tuple[int, list[torch.Tensor]]:
    """Create fresh deterministic CPU inputs shared by both collation arms.

    Args:
        sequence: Nonnegative batch identity, incorporated into every input value.
        parts: Positive number of inputs.
        rows: Positive rows per input.
        columns: Positive columns per input.
        dtype: float16, float32, float64 or uint8.

    Returns:
        Identity and independently allocated CPU tensors. Generation includes
        NumPy pattern construction and conversion, not just tensor allocation.

    Example:
        ``make_parts(7, parts=2, rows=2, columns=3)`` creates two 2-by-3 inputs.
    """
    if sequence < 0 or min(parts, rows, columns) <= 0 or dtype not in _DTYPES:
        raise ValueError("nonnegative identity, positive dimensions and dtype required")
    # Vary interior values too, so checking only endpoints cannot validate a batch.
    pattern = np.arange(rows * columns, dtype=np.int64).reshape(rows, columns) % 127
    return sequence, [
        torch.from_numpy(((pattern + sequence * 17 + i * 29) % 251).astype(dtype))
        for i in range(parts)
    ]


def _operation(op: str) -> Callable[..., torch.Tensor]:
    if op == "stack":
        return torch.stack
    if op == "cat":
        return torch.cat
    raise ValueError(f"Unknown collation operation: {op}")


def _collate(
    batch: tuple[int, list[torch.Tensor]],
    op: str,
    pool: SharedMemorySegmentPool,
    direct: bool,
) -> dict[str, Any]:
    sequence, parts = batch
    operation = _operation(op)
    if not parts or any(t.device.type != "cpu" for t in parts):
        raise ValueError("nonempty CPU inputs required")
    shape = (
        (len(parts), *parts[0].shape)
        if op == "stack"
        else (sum(t.shape[0] for t in parts), *parts[0].shape[1:])
    )
    phases = {"collation_s": 0.0, "copy_s": 0.0}
    private = None
    if not direct:
        start = time.perf_counter()
        private = operation(parts)
        phases["collation_s"] = time.perf_counter() - start
    start = time.perf_counter()
    with pool.reserve() as unit:
        phases["reserve_s"] = time.perf_counter() - start
        start = time.perf_counter()
        data = unit.allocate_bytes(math.prod(shape) * parts[0].element_size())
        out = torch.frombuffer(data.obj, dtype=parts[0].dtype).reshape(shape)
        phases["allocation_s"] = time.perf_counter() - start
        start = time.perf_counter()
        if direct:
            operation(parts, out=out)
            phases["collation_s"] = time.perf_counter() - start
        else:
            assert private is not None
            out.copy_(private)
            phases["copy_s"] = time.perf_counter() - start
        if torch.cuda.is_initialized():
            raise RuntimeError("CPU producer unexpectedly initialized CUDA")
        start = time.perf_counter()
        ticket = unit.seal()
        sealed_at = time.perf_counter()
        phases["seal_s"] = sealed_at - start
    return {
        "ticket": ticket,
        "metadata": {
            "shape": shape,
            "dtype": parts[0].dtype,
            "sequence": sequence,
            "producer_cuda_initialized": False,
        },
        "sealed_at_s": sealed_at,
        "phases": phases,
        "producer_pid": os.getpid(),
    }


def collate_private(
    batch: tuple[int, list[torch.Tensor]], *, op: str, pool: SharedMemorySegmentPool
) -> dict[str, Any]:
    """Perform ordinary private collation, then copy into a public reservation.

    Args:
        batch: Identity and CPU inputs from ``make_parts``.
        op: cat or stack along dimension zero.
        pool: This producer's parent-created public segment pool.

    Returns:
        Opaque ticket, post-seal monotonic timestamp, ordinary producer PID
        envelope and wall-phase receipts.
        Ordinary IPC transports this metadata; it does not transport the tensor.

    Example:
        ``row = collate_private(inputs, op='stack', pool=pool)``; consume once
        with ``pool.recover(row['ticket'])``, or discard once if unread.
    """
    return _collate(batch, op, pool, False)


def collate_direct(
    batch: tuple[int, list[torch.Tensor]], *, op: str, pool: SharedMemorySegmentPool
) -> dict[str, Any]:
    """Collate directly into a public reservation using Torch's out argument.

    Args:
        batch: The same identity and CPU inputs as the copy control.
        op: cat or stack along dimension zero.
        pool: This producer's parent-created public segment pool.

    Returns:
        The same metadata envelope as ``collate_private``. ``copy_s=0`` means
        this arm omits the explicit extra copy, not that all downstream copies
        disappear. Required cat/stack writes remain in ``collation_s``.

    Example:
        ``row = collate_direct(inputs, op='cat', pool=pool)`` followed by
        ``with pool.recover(row['ticket']) as actual_batch: ...``.
    """
    return _collate(batch, op, pool, True)


def _init_worker(
    pools: Sequence[SharedMemorySegmentPool], counter: Synchronized, ready: Queue
) -> None:
    global _WORKER_POOL
    torch.set_num_threads(1)
    with counter.get_lock():
        index = counter.value
        counter.value += 1
    if not 0 <= index < len(pools):
        raise RuntimeError("initializer exhausted the one-pool-per-worker assignment")
    # Assignment need not match array order; consumers route by stable public ID.
    pool = pools[len(pools) - 1 - index]
    _WORKER_POOL = pool
    ready.put((os.getpid(), pool.arena_id))


def _produce(sequence: int, *, arm: str, config: dict[str, Any]) -> dict[str, Any]:
    pool = _WORKER_POOL
    if pool is None:
        raise RuntimeError("worker initializer did not assign a pool")
    start = time.perf_counter()
    inputs = make_parts(
        sequence, **{key: config[key] for key in ("parts", "rows", "columns", "dtype")}
    )
    generation_s = time.perf_counter() - start
    collate = collate_direct if arm == "direct" else collate_private
    row = collate(inputs, op=config["op"], pool=pool)
    row["phases"]["generation_s"] = generation_s
    return row


def verify_batch(
    batch: Mapping[str, Any], config: Mapping[str, Any], *, device: str = "cpu"
) -> None:
    """Check identity, producer evidence, device, shape, dtype and every value.

    Args:
        batch: Recovered dictionary containing value, sequence and CPU evidence.
        config: parts, rows, columns, dtype, op and batches.
        device: Required tensor device type, cpu or cuda.

    Returns:
        None; raises RuntimeError for any mismatch. This full reconstruction is
        excluded from phase timers but included in enclosing epoch wall/CPU time.

    Example:
        ``verify_batch(actual_batch, config)`` inside the recovery context.
    """
    sequence = batch.get("sequence")
    if type(sequence) is not int or not 0 <= sequence < config["batches"]:
        raise RuntimeError("batch identity failure")
    if batch.get("producer_cuda_initialized") is not False:
        raise RuntimeError("CPU-only producer proof failure")
    _, parts = make_parts(
        sequence, **{key: config[key] for key in ("parts", "rows", "columns", "dtype")}
    )
    expected = _operation(config["op"])(parts)
    value = batch.get("value")
    if (
        not isinstance(value, torch.Tensor)
        or value.device.type != device
        or value.shape != expected.shape
        or value.dtype != expected.dtype
        or not torch.equal(value.cpu(), expected)
    ):
        raise RuntimeError("entire-value content/shape/dtype/device failure")


def _consume(
    row: dict[str, Any],
    pools: Mapping[str, SharedMemorySegmentPool],
    config: dict[str, Any],
    received_at: float,
    check_pin_alias: bool,
) -> dict[str, Any]:
    ticket = row["ticket"]
    producer_pid = row["producer_pid"]
    if type(producer_pid) is not int or producer_pid <= 0:
        raise RuntimeError("ordinary producer PID envelope is invalid")
    latency = received_at - row["sealed_at_s"]
    if latency < 0:
        raise RuntimeError("invalid same-host monotonic IPC timestamp")
    start = time.perf_counter()
    with pools[ticket.arena_id].recover(ticket) as data:
        recover_s = time.perf_counter() - start
        metadata = row["metadata"]
        actual_batch = {
            **metadata,
            "value": torch.frombuffer(data.obj, dtype=metadata["dtype"]).reshape(
                metadata["shape"]
            ),
        }
        pinned = actual_batch["value"].is_pinned()
        if config["pin_memory"] and not pinned:
            raise RuntimeError("recovered output is not pinned")
        pin_alias_verified = None
        if check_pin_alias:
            value = actual_batch["value"]
            alias = value.pin_memory()
            if (
                not alias.is_pinned()
                or alias.data_ptr() != value.data_ptr()
                or alias.untyped_storage().data_ptr()
                != value.untyped_storage().data_ptr()
            ):
                raise RuntimeError("warmup pin_memory allocated different storage")
            pin_alias_verified = True
            del alias, value
        host_h2d_completion_s = None
        if config["device"] == "cuda":
            start = time.perf_counter()
            gpu = transfer_tensor_h2d(actual_batch["value"], num_caches=0)
            host_h2d_completion_s = time.perf_counter() - start
            verify_batch({**actual_batch, "value": gpu}, config, device="cuda")
            del gpu
        verify_batch(actual_batch, config)
        return {
            "sequence": actual_batch["sequence"],
            "producer_pid": producer_pid,
            "arena_id": ticket.arena_id,
            "source_is_pinned": pinned,
            "pin_memory_storage_alias_verified": pin_alias_verified,
            **row["phases"],
            "ordinary_ipc_latency_s": latency,
            "main_recover_s": recover_s,
            "public_h2d_host_completion_s": host_h2d_completion_s,
        }


class _TrialTimeout(TimeoutError):
    pass


def _operation_timeout(timeout: float, deadline: float | None) -> float:
    if deadline is None:
        return timeout
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise _TrialTimeout("Isolated benchmark trial deadline expired")
    return min(timeout, remaining)


def _epoch_outputs(
    pipeline: Pipeline, timeout: float, deadline: float | None
) -> Iterator[dict[str, Any]]:
    while True:
        wait = _operation_timeout(timeout, deadline)
        try:
            row = pipeline.get_item(timeout=wait)
        except EOFError:
            return
        except TimeoutError:
            _operation_timeout(timeout, deadline)
            raise
        yield row


def run_epoch(
    pipeline: Pipeline,
    pools: Mapping[str, SharedMemorySegmentPool],
    config: dict[str, Any],
    *,
    check_pin_alias: bool = False,
    deadline: float | None = None,
) -> dict[str, Any]:
    """Receive and fully verify one epoch of ordinary-IPC public tickets.

    Args:
        pipeline: Started public process-region pipeline.
        pools: Parent pools indexed by their stable public arena_id.
        config: Payload settings, batches, pin_memory, device and timeout.
        check_pin_alias: Check public pin_memory tensor/storage pointer identity;
            select only for excluded pinned warmup, never measured phases.
        deadline: Optional monotonic trial deadline checked before each read.

    Returns:
        Epoch elapsed time, first-result delay and scalar per-batch phase rows.
        Full CPU/GPU verification is enclosed but not included in phase timers.
        IPC latency is main receipt minus post-seal time, including queueing,
        scheduling and consumer backpressure; it is not pure serialization time.

    Example:
        ``epoch = run_epoch(pipeline, pools, config)`` after setup.
    """
    if check_pin_alias and not config["pin_memory"]:
        raise ValueError(
            "pin-memory alias verification requires an excluded pinned warmup"
        )
    start = time.perf_counter()
    first = None
    rows = []
    seen: set[int] = set()
    for row in _epoch_outputs(pipeline, config["timeout"], deadline):
        received_at = time.perf_counter()
        if first is None:
            first = received_at - start
        result = _consume(row, pools, config, received_at, check_pin_alias)
        if result["sequence"] in seen:
            raise RuntimeError("duplicate batch identity")
        seen.add(result["sequence"])
        rows.append(result)
    if seen != set(range(config["batches"])) or first is None:
        raise RuntimeError("epoch batch population failure")
    return {
        "elapsed_s": time.perf_counter() - start,
        "first_result_s": first,
        "batches": rows,
    }


def _read_cpu(pid: int) -> float | None:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")
    except (OSError, ValueError, IndexError):
        return None


def _read_memory(pid: int) -> dict[str, int | None]:
    result: dict[str, int | None] = {"rss_bytes": None, "pss_bytes": None}
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                result["rss_bytes"] = int(line.split()[1]) * 1024
        for line in Path(f"/proc/{pid}/smaps_rollup").read_text().splitlines():
            if line.startswith("Pss:"):
                result["pss_bytes"] = int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass  # Unavailable samples remain null, never estimated as zero.
    return result


class _MemorySampler:
    def __init__(self, pids: Sequence[int], interval: float) -> None:
        self.pids = pids
        self.interval = interval
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.samples: list[dict[str, Any]] = []

    def _sample(self) -> None:
        start = time.perf_counter()
        values = [_read_memory(pid) for pid in self.pids]
        total: dict[str, int | None] = {}
        for key in ("rss_bytes", "pss_bytes"):
            numbers = [value[key] for value in values]
            total[key] = (
                sum(n for n in numbers if n is not None)
                if all(n is not None for n in numbers)
                else None
            )
        self.samples.append({"start_s": start, "end_s": time.perf_counter(), **total})

    def _run(self) -> None:
        while not self.stop.wait(self.interval):
            self._sample()

    def __enter__(self) -> _MemorySampler:
        self._sample()
        self.thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop.set()
        self.thread.join()
        self._sample()

    def receipt(self) -> dict[str, Any]:
        peaks = {}
        for key in ("rss_bytes", "pss_bytes"):
            values = [sample[key] for sample in self.samples if sample[key] is not None]
            peaks[f"group_peak_{key}"] = max(values) if values else None
            peaks[f"complete_{key}_samples"] = len(values)
        return {
            **peaks,
            "sample_count": len(self.samples),
            "sample_interval_s": self.interval,
            "max_sample_span_s": max(s["end_s"] - s["start_s"] for s in self.samples),
        }


def _validate_config(arm: str, config: Mapping[str, Any]) -> None:
    if os.name != "posix":
        raise ValueError("benchmark requires safe POSIX worker-PID absence checks")
    if arm not in _ARMS or config["op"] not in ("cat", "stack"):
        raise ValueError("supported arm and operation required")
    if config["dtype"] not in _DTYPES:
        raise ValueError("unsupported dtype")
    if (
        min(
            config[k]
            for k in ("workers", "parts", "rows", "columns", "batches", "runs", "slots")
        )
        <= 0
    ):
        raise ValueError("counts and dimensions must be positive")
    if not math.isfinite(config["duration_sec"]) or config["duration_sec"] < 0:
        raise ValueError("duration must be finite and nonnegative")
    if any(
        not math.isfinite(config[k]) or config[k] <= 0
        for k in ("timeout", "sample_interval")
    ):
        raise ValueError("timeout and sample interval must be finite and positive")
    if config["device"] not in ("cpu", "cuda") or config["local_rank"] < 0:
        raise ValueError("invalid device or local rank")
    if config["device"] == "cpu" and config["pin_memory"]:
        raise ValueError("CPU correctness mode requires pin_memory=False and H2D off")
    if config["device"] == "cuda" and not config["pin_memory"]:
        raise ValueError("performance comparison requires both arms pinned")


def _legend() -> dict[str, str]:
    return {
        "generation": (
            "Same fresh NumPy pattern/conversion source in both arms; worker wall timer."
        ),
        "phases": (
            "Per-delivered-batch worker generation, private/direct cat/stack, explicit "
            "copy, reserve acquisition, output allocation and seal wall timers; "
            "no full verification. Concurrent phases are not additive critical-path time."
        ),
        "ipc": (
            "Same-host perf_counter: main receives minus post-seal timestamp; ordinary "
            "IPC, queueing, scheduling and consumer backpressure, not pure serialization."
        ),
        "recover": (
            "Public recover call/context entry only; excludes verification, "
            "context exit and any remaining alias lifetime."
        ),
        "h2d": (
            "Public synchronized transfer_tensor_h2d host/completion wall time including "
            "traversal, pin checks and synchronization, not pure DMA; null when disabled. "
            "No separate asynchronous submission/device timer is exposed."
        ),
        "epoch": (
            "Fresh generation, collation, explicit copy if selected, ordinary IPC, "
            "recovery, full-value CPU/GPU verification and sampling included. Setup, "
            "one checked warmup epoch and teardown excluded. Throughput denominator "
            "sums completed epochs, not requested enclosing duration."
        ),
        "cpu": (
            "Main process CPU and sum of ALL initializer-reported worker /proc "
            "utime+stime deltas with aligned before/after snapshots around the enclosing "
            "measured window. Sequential boundary-read spans are reported; /proc CPU "
            "is quantized to OS ticks. Includes between-epoch overhead and possible "
            "prefetch. Missing workers yield null, not zero; no process-tree scan."
        ),
        "memory": (
            "Sampled simultaneous main+ALL worker RSS/PSS sum peak over enclosing "
            "interval, not sum of individual peaks. Reads are sequential within each "
            "reported sample span, not atomic. RSS double-counts shared pages; PSS is "
            "proportional. arena_capacity_bytes counts data mappings only, excluding "
            "the separate small control backing; not residency. No cgroup/registration "
            "residency claim."
        ),
        "repetitions": (
            "Fresh spawned consumer and workers for every matched arm/repetition; "
            "rotate arm order. Sample SD is null for one repetition. Trials in one "
            "allocation are not independent hosts."
        ),
        "copy": (
            "copy arm: private cat/stack then reservation output.copy_(batch). direct "
            "arm: cat/stack(out=output). Both use identical public region, initializer, "
            "pools, tickets, recovery and pinning; zero explicit copy does not mean "
            "zero downstream copies."
        ),
        "source": (
            "source_commit/build_identity are caller-provided labels, not installed-source "
            "authentication. Exact-source packaging/matrix evidence is separate."
        ),
        "cpu_only": (
            "CPU unpinned/H2D-off results are bounded correctness checks, "
            "not performance evidence for pinned GPU consumers."
        ),
    }


def _measured_epochs(
    pipeline: Pipeline,
    pools: Mapping[str, SharedMemorySegmentPool],
    config: dict[str, Any],
    workers: Sequence[int],
    deadline: float | None,
) -> dict[str, Any]:
    cpu_start, wall_start = time.process_time(), time.perf_counter()
    before = [_read_cpu(pid) for pid in workers]
    before_span_s = time.perf_counter() - wall_start
    sampler = _MemorySampler([os.getpid(), *workers], config["sample_interval"])
    epochs = []
    with sampler:
        while (
            len(epochs) < config["runs"]
            or time.perf_counter() - wall_start < config["duration_sec"]
        ):
            epochs.append(run_epoch(pipeline, pools, config, deadline=deadline))
    after_start = time.perf_counter()
    after = [_read_cpu(pid) for pid in workers]
    after_span_s = time.perf_counter() - after_start
    enclosing_s = time.perf_counter() - wall_start
    main_cpu_s = time.process_time() - cpu_start
    deltas = [
        b - a if a is not None and b is not None else None
        for a, b in zip(before, after, strict=True)
    ]
    workers_cpu_s = (
        sum(delta for delta in deltas if delta is not None)
        if all(delta is not None for delta in deltas)
        else None
    )
    return {
        "epochs": epochs,
        "enclosing_elapsed_s": enclosing_s,
        "main_cpu_s": main_cpu_s,
        "workers_cpu_s": workers_cpu_s,
        "cpu_boundary_read_spans_s": [before_span_s, after_span_s],
        "worker_cpu_deltas_s": dict(zip(workers, deltas, strict=True)),
        "total_cpu_s": main_cpu_s + workers_cpu_s
        if workers_cpu_s is not None
        else None,
        "memory": sampler.receipt(),
    }


def _assert_workers_reaped(assignments: Sequence[tuple[int, str]]) -> None:
    if os.name != "posix":
        raise RuntimeError("safe PID absence proof requires POSIX; owners retained")
    for pid, _ in assignments:
        if type(pid) is not int or pid <= 0:
            raise RuntimeError("initializer reported an invalid worker PID")
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            continue
        except OSError as error:
            raise RuntimeError(
                "worker reap cannot be confirmed; owners retained"
            ) from error
        raise RuntimeError(f"worker PID {pid} still exists; owners retained")


class _BenchmarkLifecycle:
    def __init__(self, workers: int, timeout: float) -> None:
        self.pid = os.getpid()
        self.timeout = timeout
        self.lock = RLock()
        self.arenas: list[SharedMemorySegmentPool | None] = [None] * workers
        self.assignments: list[tuple[int, str] | None] = [None] * workers
        self.counter: Synchronized | None = None
        self.ready: Queue | None = None
        self.definition: object = None
        self.pipeline: Pipeline | None = None
        self.may_have_workers = False
        self.failed = False
        self.complete = False
        self.resources_closed = False

    def collect_assignments(
        self, *, block: bool, deadline: float | None = None
    ) -> list[tuple[int, str]]:
        ready = self.ready
        if ready is None:
            raise RuntimeError("initializer queue unavailable; cleanup owners retained")
        for index, assignment in enumerate(self.assignments):
            if assignment is None:
                try:
                    row = (
                        ready.get(timeout=_operation_timeout(self.timeout, deadline))
                        if block
                        else ready.get_nowait()
                    )
                except Empty:
                    if block:
                        _operation_timeout(self.timeout, deadline)
                    break
                self.assignments[index] = row
        if any(row is None for row in self.assignments):
            raise RuntimeError(
                "not all initializer PIDs are known; cleanup owners retained"
            )
        assignments = [row for row in self.assignments if row is not None]
        arenas = [arena for arena in self.arenas if arena is not None]
        if (
            any(type(row) is not tuple or len(row) != 2 for row in assignments)
            or len({pid for pid, _ in assignments}) != len(self.assignments)
            or {arena_id for _, arena_id in assignments}
            != {arena.arena_id for arena in arenas}
        ):
            raise RuntimeError(
                "initializer ownership population invalid; owners retained"
            )
        return assignments

    def _cancel_and_reap_workers(self) -> None:
        failure: BaseException | None = None
        for arena in self.arenas:
            if arena is not None:
                try:
                    arena.close()
                except BaseException as error:  # noqa: B036 - Delayed raise.
                    if failure is None:
                        failure = error
                    else:
                        _add_cleanup_note(failure, error)
        if self.may_have_workers:
            try:
                if self.pipeline is None:
                    raise RuntimeError(
                        "pipeline construction incomplete; owners retained"
                    )
                self.pipeline.stop(timeout=self.timeout)
            except BaseException as error:  # noqa: B036 - Delayed raise.
                if failure is None:
                    failure = error
                else:
                    _add_cleanup_note(failure, error)
            try:
                _assert_workers_reaped(self.collect_assignments(block=False))
            except BaseException as error:  # noqa: B036 - Delayed raise.
                if failure is None:
                    failure = error
                else:
                    _add_cleanup_note(failure, error)
        if failure is not None:
            raise failure

    def cleanup(self) -> None:
        if self.pid != os.getpid():
            raise RuntimeError("cleanup owner belongs to another process; retained")
        with self.lock:
            if self.complete:
                return
            if not self.resources_closed:
                self._cancel_and_reap_workers()
                for arena in self.arenas:
                    if arena is not None:
                        arena.close()
                        arena.unlink()
                if self.ready is not None:
                    self.ready.close()
                    self.ready.join_thread()
                self.resources_closed = True
            _remove_cleanup_owner(self)
            for index in range(len(self.arenas)):
                self.arenas[index] = None
            self.ready = None
            self.counter = None
            self.pipeline = None
            self.definition = None
            self.failed = False
            self.complete = True


_CLEANUP_OWNERS: dict[_BenchmarkLifecycle, None] = {}
_CLEANUP_LOCK_STATE: dict[str, RLock] = {}


def _cleanup_owners_lock() -> RLock:
    lock = _CLEANUP_LOCK_STATE.get("lock")
    if lock is None:
        # Concurrent first callers must all use the same installed lock.
        lock = _CLEANUP_LOCK_STATE.setdefault("lock", RLock())
    return lock


def _register_cleanup_owner(owner: _BenchmarkLifecycle) -> None:
    with _cleanup_owners_lock():
        _CLEANUP_OWNERS[owner] = None


def _remove_cleanup_owner(owner: _BenchmarkLifecycle) -> None:
    with _cleanup_owners_lock():
        _CLEANUP_OWNERS.pop(owner, None)


def _add_cleanup_note(primary: BaseException, cleanup: BaseException) -> None:
    try:
        primary.add_note(
            f"Benchmark cleanup retained durable process-local ownership: {cleanup!r}. "
            "Retry _drain_failed_cleanups() after confirmed worker reap."
        )
    except BaseException:  # noqa: B036 - Primary wins.
        pass  # Even note-allocation failure must not replace the primary exception.


def _drain_failed_cleanups() -> int:
    """Explicitly retry failed benchmark cleanup without relying on tracebacks or GC.

    Returns:
        Number of failed lifecycle owners drained. Each owner was registered
        before any benchmark resource or worker was created and retains its
        pools, initializer controls, PID receipts and public pipeline context.
        Cleanup uses public terminal close before stop/join, then requires every
        initializer PID known and OS-confirmed absent before retrying close and
        unlink. No control state is reopened or reset.
        Live, missing or uncertain PIDs retain the owner indefinitely and raise.
        No timer, capacity limit or eviction releases these resources.

    Raises:
        BaseException: The first unsuccessful cleanup; other failures become
            notes and all unsuccessful owners remain durably discoverable.

    Example:
        After a failed run and actual worker termination/reap, call
        ``_drain_failed_cleanups()`` in the owning process. Repeated successful
        calls return zero; never infer safe cleanup from clearing an exception.
    """
    with _cleanup_owners_lock():
        owners = tuple(owner for owner in _CLEANUP_OWNERS if owner.failed)
    failure: BaseException | None = None
    drained = 0
    for owner in owners:
        try:
            owner.cleanup()
            drained += 1
        except BaseException as error:  # noqa: B036 - Delayed raise.
            if failure is None:
                failure = error
            else:
                _add_cleanup_note(failure, error)
    if failure is not None:
        raise failure
    return drained


def run_benchmark(arm: str, config: dict[str, Any]) -> dict[str, Any]:
    """Run one matched public collation arm with persistent workers across epochs.

    Args:
        arm: copy (private collation plus copy) or direct (out collation).
        config: Payload dimensions/dtype/op, workers, slots, batches, runs,
            duration_sec, device, pin_memory, local_rank, timeout, sample_interval
            and source_commit. CPU unpinned mode is correctness-only.

    Returns:
        Scalar timing, all batch receipts and aggregate worker/main OS telemetry.
        One complete checked warmup epoch is excluded. Parent creates one pool
        per worker and passes them via ordinary public initializer/initargs; no
        executor subclass, region decoder or custom queue is involved. Failed
        cleanup stays durably registered independently of exception lifetimes;
        retry ``_drain_failed_cleanups()`` only after actual worker reap.

    Example:
        ``run_benchmark('direct', config)`` inside a fresh consumer process.
    """
    return _run_benchmark(arm, config, deadline=None)


def _run_benchmark(
    arm: str, config: dict[str, Any], *, deadline: float | None
) -> dict[str, Any]:
    _validate_config(arm, config)
    _operation_timeout(config["timeout"], deadline)
    owner = _BenchmarkLifecycle(config["workers"], config["timeout"])
    _register_cleanup_owner(owner)
    primary: BaseException | None = None
    try:
        with owner.lock:
            result = _run_owned_benchmark(arm, config, owner, deadline)
            _operation_timeout(config["timeout"], deadline)
    except BaseException as error:
        primary = error
        owner.failed = True
        raise
    finally:
        # Terminal close precedes public stop/join; retirement retries follow reap.
        try:
            owner.cleanup()
        except BaseException as cleanup:  # noqa: B036 - Primary wins.
            owner.failed = True
            if primary is None:
                raise
            _add_cleanup_note(primary, cleanup)
    result["workers_reaped_os_verified"] = True
    return result


def _verify_producer_receipts(
    epochs: Sequence[dict[str, Any]], assignments: Sequence[tuple[int, str]]
) -> None:
    initialized = dict(assignments)
    for epoch in epochs:
        for row in epoch["batches"]:
            if initialized.get(row["producer_pid"]) != row["arena_id"]:
                raise RuntimeError(
                    "ordinary producer envelope disagrees with initializer PID/pool"
                )


def _run_owned_benchmark(
    arm: str,
    config: dict[str, Any],
    owner: _BenchmarkLifecycle,
    deadline: float | None,
) -> dict[str, Any]:
    torch.set_num_threads(1)
    if config["device"] == "cuda":
        os.environ["LOCAL_RANK"] = str(config["local_rank"])
        torch.cuda.set_device(config["local_rank"])
    context = mp.get_context("spawn")
    owner.counter = context.Value("i", 0)
    owner.ready = context.Queue()
    nbytes = (
        config["parts"]
        * config["rows"]
        * config["columns"]
        * getattr(torch, config["dtype"]).itemsize
    )
    with owner.lock:
        for index in range(config["workers"]):
            _operation_timeout(config["timeout"], deadline)
            owner.arenas[index] = SharedMemorySegmentPool(
                segment_size=nbytes + 4096,
                count=config["slots"],
                acquire_timeout=config["timeout"],
                pin_memory=config["pin_memory"],
            )
        arenas = [arena for arena in owner.arenas if arena is not None]
        pools = {pool.arena_id: pool for pool in arenas}
        executor = ProcessPoolExecutorConfig(
            max_workers=config["workers"],
            mp_context="spawn",
            initializer=_init_worker,
            initargs=(tuple(arenas), owner.counter, owner.ready),
        )
        definition = (
            PipelineBuilder()
            .add_source(_Batches(config["batches"]), continuous=True)
            .to(executor)
            .pipe(partial(_produce, arm=arm, config=config))
            .to(MAIN_PROCESS)
            .add_sink(2)
            .get_config()
        )
        owner.definition = definition
        _operation_timeout(config["timeout"], deadline)
        owner.may_have_workers = True
        owner.pipeline = build_pipeline(definition, num_threads=3, max_failures=0)
        pipeline = owner.pipeline
        try:
            pipeline.start(timeout=_operation_timeout(config["timeout"], deadline))
        except TimeoutError:
            _operation_timeout(config["timeout"], deadline)
            raise
        assignments = owner.collect_assignments(block=True, deadline=deadline)
        warmup = run_epoch(
            pipeline,
            pools,
            config,
            check_pin_alias=config["pin_memory"],
            deadline=deadline,
        )
        measured = _measured_epochs(
            pipeline, pools, config, [pid for pid, _ in assignments], deadline
        )
        _verify_producer_receipts([warmup, *measured["epochs"]], assignments)
        seconds = sum(epoch["elapsed_s"] for epoch in measured["epochs"])
        items = config["batches"] * len(measured["epochs"])
        return {
            "schema": _SCHEMA,
            "arm": arm,
            "config": dict(config),
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "numpy": np.__version__,
            "transport": "ordinary-ipc-public-ticket",
            "performance_evidence": config["device"] == "cuda",
            "worker_assignments": assignments,
            "warmup_verified_batches": len(warmup["batches"]),
            "warmup_pin_memory_storage_alias_verified": (
                all(
                    row["pin_memory_storage_alias_verified"] is True
                    for row in warmup["batches"]
                )
                if config["pin_memory"]
                else None
            ),
            "actual_output_bytes": nbytes,
            "arena_capacity_bytes": (nbytes + 4096) * config["slots"] * len(arenas),
            "verified_output_batches": items,
            "elapsed_s": seconds,
            "items_per_s": items / seconds,
            "output_bytes_per_s": items * nbytes / seconds,
            "timed_h2d": config["device"] == "cuda",
            **measured,
        }


def _benchmark_to_file(
    arm: str, config: dict[str, Any], path: str, deadline: float
) -> None:
    try:
        report = _run_benchmark(arm, config, deadline=deadline)
    except _TrialTimeout as error:
        report = {"trial_timeout": str(error), "notes": getattr(error, "__notes__", [])}
    Path(path).write_text(json.dumps(report, allow_nan=False))


def _trial_timeout(config: Mapping[str, Any]) -> float:
    timeout = float(config.get("trial_timeout", _DEFAULT_TRIAL_TIMEOUT))
    if not math.isfinite(timeout) or timeout <= config["duration_sec"]:
        raise ValueError("trial timeout must be finite and exceed the timed duration")
    return timeout


def run_isolated(arm: str, config: dict[str, Any]) -> dict[str, Any]:
    """Run one arm in a fresh spawned consumer, without launching infrastructure.

    Args:
        arm: copy or direct.
        config: Settings for run_benchmark. ``trial_timeout`` is the cooperative
            deadline in seconds for startup, warmup and measurement (default 1800).
            ``timeout`` still bounds individual operations. Cleanup runs in the child
            after a deadline, without killing the process that owns live workers.
            A blocked native call may delay cancellation and cleanup.

    Returns:
        Checked JSON-compatible receipt from the fresh process. A temporary file
        carries only the final report; tickets use ordinary public region IPC.

    Example:
        ``row = run_isolated('copy', config)`` for a fresh-process control.
    """
    _validate_config(arm, config)
    timeout = _trial_timeout(config)
    with TemporaryDirectory(prefix="spdl-direct-") as directory:
        path = str(Path(directory) / "receipt.json")
        deadline = time.monotonic() + timeout
        process = mp.get_context("spawn").Process(
            target=_benchmark_to_file, args=(arm, config, path, deadline)
        )
        process.start()
        try:
            process.join()
            if process.exitcode != 0:
                raise RuntimeError(
                    f"isolated public benchmark failed: exit {process.exitcode}"
                )
            report = json.loads(Path(path).read_text())
            if "trial_timeout" in report:
                error = TimeoutError(report["trial_timeout"])
                for note in report["notes"]:
                    error.add_note(note)
                raise error
            return report
        finally:
            if not process.is_alive():
                process.close()


def run_repetitions(
    configs: Sequence[dict[str, Any]], *, repeats: int
) -> dict[str, Any]:
    """Run both matched arms for each configuration with rotating process order.

    Args:
        configs: Explicit size/dtype/op configurations accepted by run_benchmark.
        repeats: Positive fresh-process pairs per configuration.

    Returns:
        Every raw receipt and each arm's throughput mean/sample SD. No causal
        bandwidth, general speedup or independent-host inference is made.

    Example:
        ``report = run_repetitions([config], repeats=3)``; infrastructure
        scheduling remains caller-owned.
    """
    return _run_repetitions(configs, repeats=repeats)


def _checkpoint_trial(
    path: Path, record: dict[str, Any], *, create: bool = False
) -> None:
    line = json.dumps(record, allow_nan=False) + "\n"
    with path.open("x" if create else "a", encoding="utf-8") as stream:
        stream.write(line)
        stream.flush()
        os.fsync(stream.fileno())


def _run_repetitions(
    configs: Sequence[dict[str, Any]], *, repeats: int, checkpoint: Path | None = None
) -> dict[str, Any]:
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    progress = {
        "status": "partial",
        "expected_trials": len(configs) * repeats * len(_ARMS),
    }
    if checkpoint is not None:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        _checkpoint_trial(checkpoint, {**progress, "completed_trials": 0}, create=True)
    results, summary = [], []
    for config in configs:
        paired = {arm: [] for arm in _ARMS}
        for repeat in range(repeats):
            order = _ARMS if repeat % 2 == 0 else tuple(reversed(_ARMS))
            for arm in order:
                row = run_isolated(arm, {**config, "repeat": repeat})
                if checkpoint is not None:
                    _checkpoint_trial(
                        checkpoint,
                        {
                            **progress,
                            "completed_trials": len(results) + 1,
                            "result": row,
                        },
                    )
                results.append(row)
                paired[arm].append(row["items_per_s"])
        for arm, values in paired.items():
            summary.append(
                {
                    "arm": arm,
                    "config": dict(config),
                    "repeats": repeats,
                    "items_per_s_mean": statistics.mean(values),
                    "items_per_s_sample_sd": statistics.stdev(values)
                    if repeats > 1
                    else None,
                }
            )
    return {
        "schema": _SCHEMA,
        "legend": _legend(),
        "results": results,
        "summary": summary,
    }


def parse_args(
    argv: Sequence[str] | None = None,
    *,
    build_identity: Mapping[str, Any] | None = None,
) -> argparse.Namespace:
    """Parse matched-arm CLI settings without running jobs or changing source.

    Args:
        argv: CLI tokens; None reads sys.argv.
        build_identity: Optional caller-provided code_commit/build labels.

    Returns:
        Validated namespace. Default CUDA mode pins both arms. --self-test selects
        bounded CPU/unpinned/H2D-off correctness only, not performance evidence.

    Example:
        ``parse_args(['--self-test', '--source-commit', 'local'])``.
    """
    parser = argparse.ArgumentParser(
        description="Matched public direct-allocation benchmark",
        fromfile_prefix_chars="@",
    )
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--sizes", nargs="+", type=int, help="output MiB; rounds to complete parts/rows"
    )
    parser.add_argument("--dtypes", nargs="+", choices=_DTYPES, default=["float32"])
    parser.add_argument("--ops", nargs="+", choices=("cat", "stack"), default=["stack"])
    for name, default in (
        ("workers", 2),
        ("parts", 4),
        ("rows", 16),
        ("columns", 4096),
        ("batches", 8),
        ("runs", 3),
        ("slots", 2),
        ("repeats", 3),
    ):
        parser.add_argument(f"--{name}", type=int, default=default)
    parser.add_argument("--duration-sec", type=float, default=15)
    parser.add_argument(
        "--timeout", type=float, default=600, help="per-operation seconds"
    )
    parser.add_argument(
        "--trial-timeout",
        type=float,
        default=_DEFAULT_TRIAL_TIMEOUT,
        help="cooperative deadline in seconds for startup, warmup and measurement",
    )
    parser.add_argument("--sample-interval", type=float, default=0.05)
    parser.add_argument(
        "--local-rank", type=int, default=int(os.environ.get("LOCAL_RANK", "0"))
    )
    parser.add_argument(
        "--source-commit",
        required=build_identity is None,
        default=build_identity["code_commit"] if build_identity else None,
    )
    parser.add_argument("--output", default="direct-allocation-results")
    args = parser.parse_args(argv)
    if args.self_test:
        args.device, args.sizes = "cpu", None
        args.workers, args.parts, args.rows, args.columns = 2, 2, 2, 3
        args.batches, args.runs, args.slots, args.repeats, args.duration_sec = (
            4,
            1,
            2,
            1,
            0,
        )
        args.dtypes, args.ops = list(_DTYPES), ["cat", "stack"]
    if (
        min(
            args.workers,
            args.parts,
            args.rows,
            args.columns,
            args.batches,
            args.runs,
            args.slots,
            args.repeats,
        )
        <= 0
    ):
        parser.error("counts and dimensions must be positive")
    if args.sizes is not None and min(args.sizes) <= 0:
        parser.error("sizes must be positive")
    if (
        build_identity is not None
        and args.source_commit != build_identity["code_commit"]
    ):
        parser.error("--source-commit must match provided build label")
    try:
        for config in _configs(args):
            _validate_config("copy", config)
            _trial_timeout(config)
    except ValueError as error:
        parser.error(str(error))
    return args


def _configs(args: argparse.Namespace) -> list[dict[str, Any]]:
    result = []
    for dtype, op, size in product(args.dtypes, args.ops, args.sizes or [None]):
        config = {
            key: getattr(args, key)
            for key in (
                "workers",
                "parts",
                "rows",
                "columns",
                "batches",
                "runs",
                "slots",
                "duration_sec",
                "timeout",
                "trial_timeout",
                "sample_interval",
                "local_rank",
                "device",
                "source_commit",
            )
        }
        if size is not None:
            divisor = args.parts * args.rows * getattr(torch, dtype).itemsize
            config["columns"] = max(1, math.ceil((size << 20) / divisor))
        result.append(
            {
                **config,
                "dtype": dtype,
                "op": op,
                "pin_memory": args.device == "cuda",
                "requested_output_mib": size,
            }
        )
    return result


def _write(report: dict[str, Any], prefix: str) -> None:
    path = Path(prefix)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_name(f"{path.name}.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    rows = [
        {
            key: json.dumps(value) if isinstance(value, (dict, list, tuple)) else value
            for key, value in row.items()
        }
        for row in report["results"]
    ]
    with path.with_name(f"{path.name}.csv").open("w", newline="") as stream:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report, allow_nan=False), flush=True)


def main(
    build_identity: dict[str, Any] | None = None, *, argv: Sequence[str] | None = None
) -> None:
    """Run a matched local CLI matrix and write raw JSON/CSV measurement receipts.

    Args:
        build_identity: Optional caller-provided code_commit/build metadata,
            preserved as labels only; no injected callbacks or private API.
        argv: CLI tokens; None reads sys.argv for the existing adapter entrypoint.

    Returns:
        None. Both arms are always compared; this launches only local spawned
        processes, never infrastructure jobs. Infrastructure scheduling and
        exact-source package validation are caller-owned.

    Example:
        ``main(argv=['--self-test', '--source-commit', 'local'])`` checks ordinary
        public ticket IPC without H2D. CPU results are correctness-only.
    """
    args = parse_args(argv, build_identity=build_identity)
    output = Path(args.output)
    checkpoint = output.with_name(f"{output.name}.partial.jsonl")
    report = _run_repetitions(
        _configs(args), repeats=args.repeats, checkpoint=checkpoint
    )
    if build_identity is not None:
        report["build_identity"] = build_identity
    _write(report, args.output)
    checkpoint.unlink()


if __name__ == "__main__":
    main()
