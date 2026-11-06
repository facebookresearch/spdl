#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

r"""A benchmark for pinned shared arenas
======================================

Shared arenas reuse host buffers when large batches cross a process boundary.
This example asks a second question: **does making the receiving storage pinned
avoid enough CPU-to-GPU (host-to-device, or H2D) staging work to improve the
complete workflow?** It compares a reusable segment pool and a first-in-first-out
(FIFO) byte ring, with pageable and pinned receiving memory,
plus ordinary inter-process transfer controls.

The pool restores supported views directly from a reusable shared segment. Its
pinned mode registers whole consumer mappings once. The ring must copy data out
before reusing its FIFO space; its pinned mode copies directly into an independent
pinned allocation. Neither mode removes the producer-to-arena copy. See the
:doc:`../optimization_guide/shared_memory_arenas` overview for ordinary CPU
pool/ring usage and the later memory-pinning option, and
:doc:`../notes/shared_memory_ownership` for technical storage-lifetime details.

Run the public comparison
---------------------------

With SPDL IO, PyTorch, NumPy and a CUDA-capable environment installed::

    python examples/benchmark_pinned_arena.py --sizes 8 --kinds torch \
        --dtypes fp32 --workers 1 --slots 4 --items 256 --repeats 3 \
        --output results.json

The default runtime compares **four matched public arms**: ``pool-pageable``,
``pool-pinned``, ``ring-pageable`` and ``ring-pinned``. The only storage-policy
change within each pair is the public constructor's ``pin_memory=False/True``.
All use the same fresh deterministic CPU source, public
:py:func:`spdl.pipeline.iterate_in_subprocess` transport/restoration, public
:py:func:`spdl.io.transfer_tensor_h2d` and entire-value verification. No private
SPDL reader, registry, serializer, transfer stream or cache hooks are used.

Each arm/repetition starts a fresh consumer and CPU producers, one arena per
producer/boundary. The consumer creates each public pool/ring and hands ownership
to the public iterable, which handles transport and teardown. A top-level,
pickle-safe factory returns a reiterable source: full epochs have fresh values
and distinct sequence identities. Producers assert CUDA remains uninitialized.
One Torch CPU thread is selected per process during setup, not module import.
The CUDA consumer uses ``LOCAL_RANK`` and creates no distributed process group.

**New measurement protocol/schema:** ``public-iterate-v1`` is not the historical
endpoint protocol. Setup, complete checked warmup epochs (``--warmup-epochs``)
and teardown are outside the measured epoch. Measured wall time includes epoch
commands/completion, fresh generation, public transport/restoration, tensor-tree
preparation, synchronized H2D if selected, and full CPU/GPU verification.
``next_output_s`` combines waiting/transport/restoration, not internal copy time.
Helper wall time includes its pin/traversal/synchronization, not pure DMA.
Internal registration, acquisition, copy, seam/occupancy and helper-stream/profiler
phases are not measured and are omitted from runtime results. Helper wall/CPU
time is null only in CPU mode, where H2D is not run. Consumer CPU is measured-epoch
process time; producer CPU receipts measure generation only. Before/after PSS/tmpfs
observations are not peaks or producer-plus-consumer totals. Timing phases overlap;
do not add them together. Per-leaf ``is_pinned`` is not a no-staging verdict;
small inline leaves may remain pageable.

``--device cpu`` selects only the pageable public pool/ring workflows.
``--self-test`` performs bounded checked CPU epochs with normal NumPy/Torch
imports. ``--source-receipt`` hashes only this benchmark
source; exact installed-library/native receipts belong to the Meta adapter.
Retired ``ring-original``, ``ipc``, ``torch-pinned`` and internal
probe/trace/threshold/seam/retention flags
fail clearly instead of silently selecting a different protocol. Implementation
lifecycle tests remain in the existing library tests, not this benchmark.

**No new performance measurements accompany this API migration.** Neither source
hashes nor AST normalization establish unchanged performance after the protocol
change. The seven-arm fixed-8-MiB figure/table and their legend below are preserved
original-source historical evidence, not results from this four-arm runtime.

Understand the historical seven arms
--------------------------------------

The following arm descriptions refer only to the original measured protocol.
``ring-original`` is no longer a runtime choice.

* **Reusable pool, pageable** (``pool-pageable``): share reusable segments and
  restore views; the H2D helper may need pinned CPU staging.
* **Reusable pool, pinned** (``pool-pinned``): register full consumer mappings,
  restore views from them, then transfer without another host pinning buffer.
* **Ring, original copy-out** (``ring-original``): reproduce the older
  bytes/concatenation/bytearray receiving path as a historical copy-out control.
* **Ring, direct pageable copy-out** (``ring-pageable``): copy source runs
  directly into an owning bytearray using the installed reader.
* **Ring, pinned copy-out** (``ring-pinned``): make that required receiving copy
  directly into Torch-allocator pinned storage using the installed reader.
* **Torch IPC, pre-pinned transfer** (``torch-pinned``): receive through ordinary
  IPC, explicitly pin the output, then call the public H2D helper. Prepin work
  is outside helper timing but included in overall workflow throughput.
* **Ordinary IPC** (``ipc``): ordinary pickle/IPC transport plus the same helper.

The Torch control is not a pure bandwidth ceiling, and the ring comparison also
changes allocator/bytearray initialization behavior. A gain is not attributable
to page locking alone. Native-backed arena views require optional
``spdl.io.lib._buffer``; ordinary pipeline use and unpinned ring copy-out remain
IO-free. The helper is loaded only for actual native storage paths.

What the reference results show
---------------------------------

For the historical fixed **8-MiB FP32 Torch** workload below, pinned pool and
ring modes improved this checked workflow. Ratios of mean throughput were
**1.343×** for pinned/pageable pool, **1.436×** for pinned/direct-pageable ring,
and **1.486×** for pinned/original ring. These are observations, not a promise
of faster training or statistical significance.

The seven arms ran three fresh producer/consumer repetitions **inside one
resource-matched GPU allocation**, not three independently allocated machines.
Each repetition checked all 256 newly generated GPU payloads, with one producer
and one GPU rank, four slots, a 4-KiB offload threshold, an initial ring seam,
no retained outputs or consumer delay, and one Torch CPU thread. That is
**5,376 fully checked warm payloads** across 21 repetitions. The figure and table
retain that measured source population: later implementation/doc changes do not
remeasure it or replace its original provenance.

.. image:: /_static/data/pinned_arena_8mib.png
   :alt: Throughput, CPU-to-GPU helper time and CPU cost for 8 MiB batches

**Read the plot:** mean ± sample SD across the three fresh-process repetitions
per arm, not confidence intervals and not a size sweep. Throughput includes
generation, transport, the public synchronized H2D helper and full GPU-value
verification. Helper wall time includes pinning/traversal and synchronization;
it is not pure DMA. CPU is total producer-plus-consumer process time. Cold probes
and separate four-copy traces are excluded. Ring pinned output still pays
copy-out; the Torch IPC control's prepin work is counted in overall throughput.
The full table immediately below retains all resource and trace measurement
scopes. Exact measured-source/environment receipts are retained separately from
this reader-facing summary, not substituted for these observations.

Measured workflow and separate trace observations
---------------------------------------------------

.. list-table:: Source- and measurement-scoped observations
   :header-rows: 1

   * - Scope
     - Arm
     - Bytes
     - W/R
     - Cold A ms
     - Cold Reg+V ms
     - CopyIn/W ms
     - Acq ms
     - Read/O ms
     - Restore CPU ms
     - Prep CPU ms
     - Cold pin probe CPU ms
     - Helper pin CPU ms
     - Control prepin ms
     - H2D host ms
     - CUDA interval ms
     - Pure DMA ms
     - items/s ± SD
     - GiB/s ± SD
     - CPU P/C s
     - PSS P/C MiB
     - tmpfs Rsv MiB
     - Reg MiB
     - Alloc active HWM MiB
     - Alloc held after MiB
     - CG current HWM GiB
     - CG lifetime peak GiB
     - Pool HWM units
     - Ring HWM MiB
     - Retain HWM MiB
     - Warm seams
     - Pinned/alias
     - No-staging verdict
     - CPU _pin/copy events
   * - perf (3×256)
     - pool-pageable
     - 8388608
     - 1/1
     - 103.945
     - --
     - 0.848
     - 4.182
     - 0.020
     - 0.193
     - 0.004
     - 5.426
     - --
     - --
     - 2.052
     - 2.074
     - --
     - 137.09 ± 0.39
     - 1.071 ± 0.003
     - 0.667/2.443
     - 565.6/707.4
     - 32.002
     - 0.000
     - 8.000
     - 8.000
     - 3.570
     - 3.607
     - 4
     - --
     - 8.000
     - --
     - no/no
     - --
     - --
   * - perf (3×256)
     - pool-pinned
     - 8388608
     - 1/1
     - 97.547
     - 16.698
     - 0.778
     - 2.593
     - 0.021
     - 0.180
     - 0.006
     - 0.047
     - --
     - --
     - 0.321
     - 0.330
     - --
     - 184.14 ± 5.73
     - 1.439 ± 0.045
     - 0.603/1.807
     - 566.2/713.3
     - 32.002
     - 32.002
     - 0.000
     - 0.000
     - 3.587
     - 3.624
     - 4
     - --
     - 8.000
     - --
     - yes/yes
     - --
     - --
   * - perf (3×256)
     - ring-original
     - 8388608
     - 1/1
     - 90.870
     - --
     - 6.964
     - 0.016
     - 2.181
     - 3.103
     - 0.006
     - 5.123
     - --
     - --
     - 2.146
     - 2.171
     - --
     - 100.57 ± 0.73
     - 0.786 ± 0.006
     - 0.720/3.350
     - 568.6/724.0
     - 32.002
     - 0.000
     - 8.000
     - 8.000
     - 3.601
     - 3.624
     - --
     - 32.000
     - 8.000
     - 64
     - no/no
     - --
     - --
   * - perf (3×256)
     - ring-pageable
     - 8388608
     - 1/1
     - 99.734
     - --
     - 6.817
     - 0.015
     - 1.832
     - 2.651
     - 0.007
     - 5.982
     - --
     - --
     - 2.149
     - 2.174
     - --
     - 104.06 ± 5.96
     - 0.813 ± 0.047
     - 0.690/3.227
     - 570.8/724.2
     - 32.002
     - 0.000
     - 8.000
     - 8.000
     - 3.600
     - 3.624
     - --
     - 32.000
     - 8.000
     - 64
     - no/no
     - --
     - --
   * - perf (3×256)
     - ring-pinned
     - 8388608
     - 1/1
     - 93.946
     - --
     - 4.517
     - 0.014
     - 1.031
     - 1.558
     - 0.005
     - 0.014
     - --
     - --
     - 0.343
     - 0.354
     - --
     - 149.42 ± 5.83
     - 1.167 ± 0.046
     - 0.637/2.213
     - 569.6/714.2
     - 32.002
     - 0.000
     - 8.000
     - 8.000
     - 3.591
     - 3.624
     - --
     - 32.000
     - 8.000
     - 64
     - yes/yes
     - --
     - --
   * - perf (3×256)
     - torch-pinned
     - 8388608
     - 1/1
     - 0.324
     - --
     - --
     - --
     - --
     - 2.997
     - 0.006
     - 0.008
     - --
     - 1.877
     - 0.378
     - 0.391
     - --
     - 69.93 ± 3.26
     - 0.546 ± 0.025
     - 1.913/4.750
     - 572.8/724.7
     - 0.000
     - 0.000
     - 8.000
     - 8.000
     - 3.616
     - 3.629
     - --
     - --
     - 8.000
     - --
     - yes/yes
     - --
     - --
   * - perf (3×256)
     - ipc
     - 8388608
     - 1/1
     - 0.303
     - --
     - --
     - --
     - --
     - 2.951
     - 0.006
     - 5.521
     - --
     - --
     - 2.223
     - 2.250
     - --
     - 69.36 ± 2.89
     - 0.542 ± 0.023
     - 1.883/4.747
     - 575.5/739.3
     - 0.000
     - 0.000
     - 8.000
     - 8.000
     - 3.625
     - 3.633
     - --
     - --
     - 8.000
     - --
     - no/no
     - --
     - --
   * - trace (1×4)
     - pool-pageable
     - 8388608
     - 1/1
     - --
     - --
     - --
     - --
     - --
     - --
     - --
     - --
     - 2.144
     - --
     - --
     - --
     - 0.164057
     - --
     - --
     - --
     - --
     - --
     - --
     - --
     - --
     - --
     - --
     - --
     - --
     - --
     - --
     - --
     - CPU pin stage present
     - 4/12
   * - trace (1×4)
     - pool-pinned
     - 8388608
     - 1/1
     - --
     - --
     - --
     - --
     - --
     - --
     - --
     - --
     - --
     - --
     - --
     - --
     - 0.155713
     - --
     - --
     - --
     - --
     - --
     - --
     - --
     - --
     - --
     - --
     - --
     - --
     - --
     - --
     - --
     - no CPU pin stage observed
     - 0/8
   * - trace (1×4)
     - ring-pageable
     - 8388608
     - 1/1
     - --
     - --
     - --
     - --
     - --
     - --
     - --
     - --
     - 1.648
     - --
     - --
     - --
     - 0.165001
     - --
     - --
     - --
     - --
     - --
     - --
     - --
     - --
     - --
     - --
     - --
     - --
     - --
     - --
     - --
     - CPU pin stage present
     - 4/12
   * - trace (1×4)
     - ring-pinned
     - 8388608
     - 1/1
     - --
     - --
     - --
     - --
     - --
     - --
     - --
     - --
     - --
     - --
     - --
     - --
     - 0.155593
     - --
     - --
     - --
     - --
     - --
     - --
     - --
     - --
     - --
     - --
     - --
     - --
     - --
     - --
     - --
     - no CPU pin stage observed
     - 0/8

Column legend: ``--`` means unavailable or not applicable, not measured zero.
W/R is CPU producers/GPU ranks. Performance rows: except throughput mean ± sample SD, values
are means of three fresh-process repetitions; HWM columns are observed maxima
within each repetition (pool/ring/retention use the largest of those maxima).
Cold A is arena setup, not CUDA initialization or package fetch. Cold Reg+V
includes registration and creating the first exported view; registration-only
time is unavailable. CopyIn/W includes producer ring backpressure; Acq is pool
reservation wait/bookkeeping (ring acquisition is only bookkeeping). Read/O
is allocation/copy-out for rings, but only exported-view creation for pools.
Restore and preparation CPU timers are process-wide per-item measurements;
background-thread work can overlap them. Cold pin probe is an explicit
untimed-before-warm call, not the helper's unexposed warm pin-copy CPU phase.
Control prepin is only the Torch-owned pinned IPC control. H2D host is
synchronized helper completion. CUDA intervals bracket host pin/traversal
stalls too; they are not pure DMA. P/C separates producer/consumer live-process
CPU totals over 256 items and means of sampled per-run process PSS peaks.
Consumer CPU includes full GPU-value verification and resource sampling.
Rsv/Reg are logical tmpfs reservation and external registration bytes.
Alloc active HWM and held-after are Torch active peak and allocated_bytes.current
(including active allocations), not inactive-cache size. Pool registration is
outside those allocator counters. CG current HWM is a sampled current-memory
peak; CG lifetime peak is memory.peak, possibly including prior/unrelated work.
PSS and cgroup memory are not additive. Retention includes the current batch;
ring retained outputs are independent of source-ring capacity. Seam count is
measured warm wraps. Pinned/alias means all warm sources report pinned and all
cold pin_memory probes share their storage. Trace rows are separate single-case
four-copy profiler observations, not timing-repetition means. Their Helper pin
CPU is profiler host-operator elapsed duration (not process CPU consumption):
summed aten::_pin_memory duration divided by four. Absent operators have
no reported duration, not an estimated timing zero. CPU _pin/copy are raw
operator counts. Only trace rows supply pure HtoD DMA and observed CPU-stage
verdicts; do not substitute them for performance-row CUDA intervals.

When to use these measurements
--------------------------------

Compare matched pageable and pinned arms if H2D staging consumes noticeable host
work. Also inspect CPU, pinned-memory budget and actual end-to-end throughput:
generation or model compute can remain the bottleneck even when staging improves.
Keep defaults off when host memory is scarce, payloads stay on CPU, or later
transforms allocate new pageable buffers.

Separate four-copy traces showed no additional CPU pin-staging operator in the
observed pinned pool/ring inputs, unlike their pageable controls. This supports
the staging explanation for those traces; a device-copy label alone is not
proof, because a pageable control can be copied into a pinned staging buffer
first. Traces are a different population from warm timings and do not remove
producer copies, ring copy-out or unsupported/inline leaves.

"""

from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import math
import multiprocessing as mp
import os
import platform
import statistics
import sys
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from functools import partial
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import torch
from spdl.io import transfer_tensor_h2d
from spdl.pipeline import (
    iterate_in_subprocess,
    SharedMemoryRingBuffer,
    SharedMemorySegmentPool,
)

__all__ = [
    "configure_process",
    "make_payload",
    "produce_payloads",
    "verify_payload",
    "restore_payload",
    "transfer_payload",
    "process_item",
    "run_pass",
    "run_benchmark",
    "run_isolated",
    "main",
]

_ARMS = (
    "pool-pageable",
    "pool-pinned",
    "ring-pageable",
    "ring-pinned",
)
_CPU_ARMS = ("pool-pageable", "ring-pageable")
_DTYPES = {"fp32": ("float32", 4), "fp16": ("float16", 2), "uint8": ("uint8", 1)}
_SCHEMA = "public-iterate-v1"
_RETIRED_FLAGS = (
    "--runtime-checks",
    "--trace-dir",
    "--threshold",
    "--seam",
    "--retain-items",
    "--consumer-delay",
    "--sample-interval",
)


def configure_process() -> None:
    """Select one Torch CPU thread in each process before benchmark setup/timing.

    Use this as the public subprocess initializer, not at module import time.
    It does not initialize CUDA or change another process's thread count.

    Example:
        >>> configure_process()
        >>> torch.get_num_threads()
        1
    """
    torch.set_num_threads(1)


def _pattern(count: int, sequence: int, leaf: int) -> bytes:
    shift = (sequence * 17 + leaf * 29) % 251
    cycle = bytes((i + shift) % 251 for i in range(251))
    return (cycle * ((count + 250) // 251))[:count]


def _leaf_sizes(size: int, kind: str, dtype: str) -> list[int]:
    elements = max(1, size // _DTYPES[dtype][1])
    if kind != "nested":
        return [elements]
    first = max(1, elements // 2)
    return [first, max(1, elements - first), 3]


def make_payload(size: int, kind: str, dtype: str, sequence: int) -> Any:
    """Create fresh deterministic CPU data, including for independent verification.

    Args:
        size: Positive requested byte budget. Leaf lengths are dtype-rounded;
            a nested payload also includes a three-element leaf.
        kind: ``torch``, ``numpy`` or a ``nested`` dictionary/list/tuple tensor tree.
        dtype: ``fp32``, ``fp16`` or ``uint8``.
        sequence: Nonnegative item identity incorporated into every leaf.

    Returns:
        A CPU tensor, NumPy array or nested CPU tensor tree; no CUDA storage.

    Example:
        >>> make_payload(8192, 'torch', 'fp32', sequence=1).numel()
        2048
    """
    if size <= 0 or sequence < 0 or kind not in ("torch", "numpy", "nested"):
        raise ValueError("invalid payload size, sequence or kind")
    if dtype not in _DTYPES:
        raise ValueError("unsupported dtype")
    arrays = [
        np.frombuffer(_pattern(n, sequence, i), dtype=np.uint8)
        .astype(_DTYPES[dtype][0])
        .reshape(-1)
        for i, n in enumerate(_leaf_sizes(size, kind, dtype))
    ]
    leaves = arrays if kind == "numpy" else [torch.from_numpy(a) for a in arrays]
    if kind == "nested":
        return {"main": leaves[0], "nested": [leaves[1], (leaves[2],)]}
    return leaves[0]


def _map_leaves(batch: Any, fn: Callable[[Any], Any]) -> Any:
    if isinstance(batch, dict):
        return {key: _map_leaves(value, fn) for key, value in batch.items()}
    if isinstance(batch, list):
        return [_map_leaves(value, fn) for value in batch]
    if isinstance(batch, tuple):
        return tuple(_map_leaves(value, fn) for value in batch)
    return fn(batch)


def _leaves(batch: Any) -> Iterator[Any]:
    if isinstance(batch, Mapping):
        for value in batch.values():
            yield from _leaves(value)
    elif isinstance(batch, (list, tuple)):
        for value in batch:
            yield from _leaves(value)
    else:
        yield batch


class _Payloads:
    def __init__(self, config: dict[str, Any], worker: int) -> None:
        self.config = config
        self.worker = worker
        self.epoch = 0

    def __iter__(self) -> Iterator[dict[str, Any]]:
        config, epoch = self.config, self.epoch
        self.epoch += 1
        for index in range(config["items"]):
            if torch.cuda.is_initialized():
                raise AssertionError("producer initialized CUDA")
            sequence = (epoch * config["workers"] + self.worker) * config[
                "items"
            ] + index
            start, cpu_start = time.perf_counter(), time.process_time()
            payload = make_payload(
                config["size"], config["kind"], config["dtype"], sequence
            )
            yield {
                "sequence": sequence,
                "worker": self.worker,
                "epoch": epoch,
                "payload": payload,
                "producer": {
                    "pid": os.getpid(),
                    "generation_s": time.perf_counter() - start,
                    "generation_cpu_s": time.process_time() - cpu_start,
                    "cuda_initialized": torch.cuda.is_initialized(),
                    "cpu_threads": torch.get_num_threads(),
                },
            }
            del payload
        if torch.cuda.is_initialized():
            raise AssertionError("producer initialized CUDA during production")


def produce_payloads(
    config: dict[str, Any], worker: int = 0
) -> Iterable[dict[str, Any]]:
    """Return a pickle-safe, reiterable CPU source for a public subprocess boundary.

    Args:
        config: ``size``, ``kind``, ``dtype``, ``items`` per epoch and ``workers``.
        worker: Producer index in ``[0, workers)``.

    Returns:
        A reiterable source, not a single-use generator. Each epoch produces fresh
        values and globally distinct sequence identifiers. Rows include generation
        wall/CPU time and the producer's CUDA-uninitialized/thread-count receipt.
        Pass ``configure_process`` as the boundary's initializer.

    Example:
        >>> cfg = dict(size=8192, kind='torch', dtype='fp32', items=2, workers=1)
        >>> source = produce_payloads(cfg)
        >>> [row['sequence'] for row in source]
        [0, 1]
        >>> [row['sequence'] for row in source]
        [2, 3]
    """
    if config["items"] <= 0 or not 0 <= worker < config["workers"]:
        raise ValueError("invalid producer item count or index")
    return _Payloads(config, worker)


def restore_payload(batch: Any) -> tuple[Any, dict[str, float]]:
    """Prepare already-restored public boundary outputs as CPU tensor leaves.

    No serialized blobs, registry or reader endpoints are accepted here. Arena
    restoration belongs entirely to ``iterate_in_subprocess``.

    Args:
        batch: Tensor/array/tree already yielded by the public iterable.

    Returns:
        ``(tensor_batch, phases)``; phases measure only NumPy-to-Torch/tree
        preparation, not the preceding transport or arena restoration.

    Example:
        >>> batch, phases = restore_payload(make_payload(16, 'numpy', 'fp32', 0))
        >>> batch.numel()
        4
    """
    start, cpu_start = time.perf_counter(), time.process_time()
    result = _map_leaves(
        batch,
        lambda leaf: torch.from_numpy(leaf) if isinstance(leaf, np.ndarray) else leaf,
    )
    return result, {
        "preparation_s": time.perf_counter() - start,
        "preparation_cpu_s": time.process_time() - cpu_start,
    }


def _verify_tree(actual: Any, expected: Any, device: str, sequence: int) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise AssertionError("payload structure mismatch")
        for key in expected:
            _verify_tree(actual[key], expected[key], device, sequence)
    elif isinstance(expected, (list, tuple)):
        if type(actual) is not type(expected) or len(actual) != len(expected):
            raise AssertionError("payload structure mismatch")
        for value, target in zip(actual, expected, strict=True):
            _verify_tree(value, target, device, sequence)
    else:
        if not isinstance(actual, torch.Tensor) or actual.device.type != device:
            raise AssertionError("payload device/type mismatch")
        if actual.dtype != expected.dtype or actual.shape != expected.shape:
            raise AssertionError("payload shape/dtype mismatch")
        if not torch.equal(actual.cpu(), expected):
            raise AssertionError(
                f"entire payload value mismatch for sequence {sequence}"
            )


def verify_payload(
    batch: Any, size: int, kind: str, dtype: str, sequence: int, *, device: str = "cpu"
) -> None:
    """Check structure, dtype, shape, device and every value against fresh CPU data.

    Args:
        batch: Prepared tensor/tree to check, on CPU or CUDA.
        size: Requested payload byte budget.
        kind: Original ``torch``, ``numpy`` or ``nested`` payload kind.
        dtype: Original ``fp32``, ``fp16`` or ``uint8`` dtype.
        sequence: Expected producer item identity.
        device: Required leaf device type, ``cpu`` or ``cuda``.

    Raises:
        AssertionError: Structure, layout, device or any value differs.

    Example:
        >>> batch = make_payload(8192, 'torch', 'fp32', 1)
        >>> verify_payload(batch, 8192, 'torch', 'fp32', 1)
    """
    expected, _ = restore_payload(make_payload(size, kind, dtype, sequence))
    _verify_tree(batch, expected, device, sequence)


def transfer_payload(batch: Any) -> tuple[Any, dict[str, Any]]:
    """Use only the public synchronized H2D helper, with its cache disabled.

    Args:
        batch: CPU tensor or supported tensor tree.

    Returns:
        ``(gpu_batch, phases)`` with measured helper completion wall/process-CPU
        time only. Unmeasured internal pin/copy/stream phases are omitted.

    Example:
        On a CUDA consumer, call ``gpu, phases = transfer_payload(cpu_batch)``.
    """
    start, cpu_start = time.perf_counter(), time.process_time()
    result = transfer_tensor_h2d(batch, num_caches=0)
    return result, {
        "host_h2d_completion_s": time.perf_counter() - start,
        "helper_cpu_s": time.process_time() - cpu_start,
    }


def process_item(item: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Prepare, optionally transfer, and fully verify one public output.

    Args:
        item: Producer row already restored by the public subprocess iterable.
        config: Payload settings and ``device`` (``cpu`` or ``cuda``).

    Returns:
        Scalar/list receipts only, with no retained tensors. Every arm uses this
        identical consumer workflow. Source pinning is reported per leaf,
        including small inline/pageable leaves.

    Raises:
        AssertionError: Producer CUDA/thread receipt or full value check fails.

    Example:
        Use this with rows from ``iterate_in_subprocess(produce_factory)``.
    """
    producer = item["producer"]
    if producer["cuda_initialized"] or producer["cpu_threads"] != 1:
        raise AssertionError("producer must use one CPU thread without CUDA")
    batch, phases = restore_payload(item["payload"])
    nbytes = sum(leaf.numel() * leaf.element_size() for leaf in _leaves(batch))
    pinned = [leaf.is_pinned() for leaf in _leaves(batch)]
    h2d: dict[str, Any] = {
        "host_h2d_completion_s": None,
        "helper_cpu_s": None,
    }
    checked = batch
    if config["device"] == "cuda":
        checked, h2d = transfer_payload(batch)
    start = time.perf_counter()
    verify_payload(
        checked,
        config["size"],
        config["kind"],
        config["dtype"],
        item["sequence"],
        device=config["device"],
    )
    return {
        "sequence": item["sequence"],
        "worker": item["worker"],
        "epoch": item["epoch"],
        "producer": producer,
        "payload_bytes": nbytes,
        **phases,
        **h2d,
        "verify_s": time.perf_counter() - start,
        "source_is_pinned": pinned,
    }


def run_pass(
    sources: Iterable[Iterable[dict[str, Any]]], config: dict[str, Any]
) -> dict[str, Any]:
    """Run one whole checked epoch, consuming producers in round-robin order.

    Args:
        sources: One public subprocess iterable per producer, already initialized.
        config: Payload, per-producer ``items``, worker count and device settings.

    Returns:
        Epoch wall/consumer CPU time and per-item receipts. Wall includes epoch
        command dispatch, generation, public transport/restore, preparation,
        synchronized H2D if selected, full verification and final epoch completion.
        ``next_output_s`` combines waiting, transport and restoration; it is not
        an internal read/copy/registration timer. No payloads are retained.

    Example:
        Call once for warmup and again on the same sources for measured work.
    """
    iterators = [iter(source) for source in sources]
    if len(iterators) != config["workers"]:
        raise ValueError("one public boundary is required per producer")
    rows = []
    start, cpu_start = time.perf_counter(), time.process_time()
    for _ in range(config["items"]):
        for worker, iterator in enumerate(iterators):
            fetch_start = time.perf_counter()
            item = next(iterator)
            fetch_s = time.perf_counter() - fetch_start
            if item["worker"] != worker:
                raise AssertionError("producer identity mismatch")
            row = process_item(item, config)
            row["next_output_s"] = fetch_s
            rows.append(row)
            del item
    sentinel = object()
    for iterator in iterators:
        if next(iterator, sentinel) is not sentinel:
            raise AssertionError("producer yielded too many items")
    return {
        "wall_s": time.perf_counter() - start,
        "consumer_cpu_s": time.process_time() - cpu_start,
        "items": rows,
    }


def _memory_sample() -> dict[str, Any]:
    result: dict[str, Any] = {"consumer_pss_bytes": None, "tmpfs_available_bytes": None}
    try:
        for line in Path("/proc/self/smaps_rollup").read_text().splitlines():
            if line.startswith("Pss:"):
                result["consumer_pss_bytes"] = int(line.split()[1]) * 1024
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        pass
    try:
        fs = os.statvfs("/dev/shm")
        result["tmpfs_available_bytes"] = fs.f_bavail * fs.f_frsize
    except (FileNotFoundError, PermissionError):
        pass
    return result


def _source_receipt() -> dict[str, Any]:
    module = sys.modules[__name__]
    try:
        source = inspect.getsource(module).encode("utf-8")
    except (OSError, TypeError):
        source = None
    return {
        "benchmark_sha256": hashlib.sha256(source).hexdigest()
        if source is not None
        else None,
        "hash_scope": "benchmark_source_utf8",
        "cuda_initialized": torch.cuda.is_initialized(),
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "schema": _SCHEMA,
    }


def _make_arena(
    arm: str, config: dict[str, Any]
) -> SharedMemorySegmentPool | SharedMemoryRingBuffer:
    unit = (
        sum(_leaf_sizes(config["size"], config["kind"], config["dtype"]))
        * _DTYPES[config["dtype"]][1]
        + 512
    )
    pinned = arm.endswith("-pinned")
    if arm.startswith("pool-"):
        return SharedMemorySegmentPool(
            segment_size=unit,
            count=config["slots"],
            acquire_timeout=config["timeout"],
            pin_memory=pinned,
        )
    return SharedMemoryRingBuffer(
        capacity=unit * config["slots"],
        acquire_timeout=config["timeout"],
        pin_memory=pinned,
    )


def _validate_config(arm: str, config: dict[str, Any]) -> None:
    if arm not in _ARMS:
        raise ValueError(f"unsupported or retired runtime arm: {arm}")
    if config["device"] not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    if config["device"] == "cpu" and arm not in _CPU_ARMS:
        raise ValueError("pinned runtime arms require a CUDA consumer")
    if min(config[key] for key in ("size", "items", "slots", "workers")) <= 0:
        raise ValueError("size, items, slots and workers must be positive")
    if config["warmup_epochs"] < 1 or config["local_rank"] < 0:
        raise ValueError("warmup_epochs must be positive and local_rank nonnegative")
    if not math.isfinite(config["timeout"]) or config["timeout"] <= 0:
        raise ValueError("timeout must be finite and positive")
    if (
        config["kind"] not in ("torch", "numpy", "nested")
        or config["dtype"] not in _DTYPES
    ):
        raise ValueError("unsupported payload kind/dtype")
    retired = set(config).intersection(
        ("threshold", "seam", "trace_dir", "retain_items", "consumer_delay")
    )
    if retired:
        raise ValueError(f"retired internal configuration fields: {sorted(retired)}")


def run_benchmark(arm: str, config: dict[str, Any]) -> dict[str, Any]:
    """Run an ordinary public CPU/GPU comparison in the current consumer process.

    Args:
        arm: ``pool-pageable``, ``pool-pinned``, ``ring-pageable`` or ``ring-pinned``.
        config: ``size``, ``kind``, ``dtype``, ``items``, ``workers``, ``slots``,
            ``warmup_epochs``, ``timeout``, ``local_rank`` and ``device``.

    Returns:
        ``public-iterate-v1`` receipts: setup, excluded checked warmup epochs,
        one measured epoch and throughput using only measured public metrics.
        The consumer creates one arena per producer and transfers ownership to
        ``iterate_in_subprocess``. That public iterable handles transport and
        teardown. Thread/CUDA setup, warmup and finalizer cleanup are untimed.

    Example:
        >>> cfg = dict(size=8192, kind='torch', dtype='fp32', items=2, workers=1,
        ...            slots=2, warmup_epochs=1, timeout=60., local_rank=0, device='cpu')
        >>> result = run_benchmark('pool-pageable', cfg)
        >>> len(result['measured']['items'])
        2
    """
    _validate_config(arm, config)
    configure_process()
    source = _source_receipt()
    if config["device"] == "cuda":
        os.environ["LOCAL_RANK"] = str(config["local_rank"])
        torch.cuda.set_device(config["local_rank"])
    sources: list[Any] = []
    setup_start = time.perf_counter()
    try:
        for worker in range(config["workers"]):
            sources.append(
                iterate_in_subprocess(
                    partial(produce_payloads, config, worker),
                    initializer=configure_process,
                    mp_context="spawn",
                    buffer_size=config["slots"],
                    timeout=config["timeout"],
                    arena=_make_arena(arm, config),
                )
            )
        setup_s = time.perf_counter() - setup_start
        warmup = [run_pass(sources, config) for _ in range(config["warmup_epochs"])]
        memory_before = _memory_sample()
        measured = run_pass(sources, config)
        return {
            "schema": _SCHEMA,
            "arm": arm,
            "config": dict(config),
            "source": source,
            "setup_s": setup_s,
            "warmup": warmup,
            "measured": measured,
            "items_per_s": len(measured["items"]) / measured["wall_s"],
            "payload_gib_per_s": sum(row["payload_bytes"] for row in measured["items"])
            / (1 << 30)
            / measured["wall_s"],
            "memory_before": memory_before,
            "memory_after": _memory_sample(),
        }
    finally:
        sources.clear()
        gc.collect()


def _benchmark_to_file(arm: str, config: dict[str, Any], path: str) -> None:
    Path(path).write_text(json.dumps(run_benchmark(arm, config)), encoding="utf-8")


def run_isolated(arm: str, config: dict[str, Any]) -> dict[str, Any]:
    """Run one arm in a fresh spawned consumer, with fresh public producer boundaries.

    Args:
        arm: Supported runtime arm.
        config: Settings accepted by ``run_benchmark``; ``timeout`` bounds the
            complete isolated invocation, including setup/warmup/teardown.

    Returns:
        The consumer's JSON-compatible benchmark receipt. A temporary result file
        conveys only the final report, never payloads or arena endpoint messages.

    Raises:
        RuntimeError: The consumer failed or did not write its final report.
        TimeoutError: The invocation exceeded its configured total timeout.

    Example:
        Call ``run_isolated('pool-pageable', cfg)`` for fresh-process repetitions.
    """
    _validate_config(arm, config)
    with TemporaryDirectory(prefix="spdl-public-pin-") as directory:
        path = str(Path(directory) / "result.json")
        process = mp.get_context("spawn").Process(
            target=_benchmark_to_file, args=(arm, config, path)
        )
        process.start()
        try:
            process.join(config["timeout"])
            if process.is_alive():
                raise TimeoutError("isolated benchmark consumer timed out")
            if process.exitcode != 0 or not Path(path).is_file():
                raise RuntimeError(
                    f"isolated benchmark consumer failed: exit {process.exitcode}"
                )
            return json.loads(Path(path).read_text(encoding="utf-8"))
        finally:
            if process.is_alive():
                process.terminate()
                process.join(10)
                if process.is_alive():
                    process.kill()
                    process.join(10)
            process.close()


def _legend() -> dict[str, str]:
    return {
        "protocol": (
            "public-iterate-v1: public iterate_in_subprocess owns transport, "
            "restoration and teardown; no private endpoint or transfer hooks."
        ),
        "historical": (
            "Seven-arm fixed-8-MiB table/figure retain original-source measurements. "
            "ring-original is retired; the changed protocol is not remeasured."
        ),
        "timing": (
            "Setup and checked warmup epochs excluded. Measured epoch includes "
            "generation, epoch dispatch/end, public transport/restore, preparation, "
            "synchronized public H2D if selected and entire-value verification. "
            "next_output_s combines waiting/transport/restore. Do not sum phases."
        ),
        "producer_cpu": (
            "generation_cpu_s measures generation only, not total producer CPU. "
            "consumer_cpu_s is measured-epoch process CPU, including verification."
        ),
        "pinning": (
            "Per-leaf public is_pinned status is not proof of no staging. Small "
            "inline leaves may stay pageable. Pinning is selected only through "
            "the public arena constructor's pin_memory option."
        ),
        "memory": (
            "Before/after consumer /proc PSS and host-wide /dev/shm free space; "
            "not peaks, producer memory or registration/reservation totals."
        ),
        "repeats": (
            "Fresh consumer/producers per arm/repetition, one arena per producer "
            "and one CPU thread per process. Arms rotate per repeat. Sample SD "
            "is null for one repeat; these are not independent GPU allocations."
        ),
    }


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description="Public subprocess pinned-arena comparison"
    )
    retired = [
        flag
        for flag in _RETIRED_FLAGS
        if any(value.split("=", 1)[0] == flag for value in argv)
    ]
    if retired:
        parser.error(
            f"retired internal benchmark flags: {', '.join(retired)}; "
            "lifecycle tests remain in the library test suite"
        )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="bounded checked public CPU workflows, not a performance sweep",
    )
    parser.add_argument(
        "--source-receipt",
        action="store_true",
        help="benchmark source identity only; no private library/package probes",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--arms", nargs="+", choices=_ARMS)
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[8],
        help="requested MiB; overridden by --sizes-bytes",
    )
    parser.add_argument("--sizes-bytes", nargs="+", type=int)
    parser.add_argument(
        "--kinds", nargs="+", choices=("torch", "numpy", "nested"), default=["torch"]
    )
    parser.add_argument("--dtypes", nargs="+", choices=tuple(_DTYPES), default=["fp32"])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--slots", type=int, default=4)
    parser.add_argument("--items", type=int, default=8, help="per producer per epoch")
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=1,
        help="complete checked epochs excluded from measurement",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument(
        "--local-rank", type=int, default=int(os.environ.get("LOCAL_RANK", "0"))
    )
    parser.add_argument("--output", help="JSON report file, otherwise stdout")
    args = parser.parse_args(argv)
    if args.self_test:
        args.device, args.arms = "cpu", list(_CPU_ARMS)
        args.sizes_bytes, args.kinds, args.dtypes = (
            [8192],
            ["torch", "numpy", "nested"],
            ["fp32"],
        )
        args.workers, args.slots, args.items, args.repeats, args.warmup_epochs = (
            1,
            2,
            2,
            1,
            1,
        )
    if args.arms is None:
        args.arms = list(_CPU_ARMS if args.device == "cpu" else _ARMS)
    if min(args.workers, args.slots, args.items, args.warmup_epochs, args.repeats) <= 0:
        parser.error(
            "workers, slots, items, warmup-epochs and repeats must be positive"
        )
    if not math.isfinite(args.timeout) or args.timeout <= 0 or args.local_rank < 0:
        parser.error("timeout must be finite/positive; local-rank must be nonnegative")
    if any(size <= 0 for size in (args.sizes_bytes or args.sizes)):
        parser.error("sizes must be positive")
    if args.device == "cpu" and any(arm not in _CPU_ARMS for arm in args.arms):
        parser.error("pinned runtime arms require --device cuda")
    return args


def _summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[float]] = {}
    for row in rows:
        config = row["config"]
        key = (
            row["arm"],
            config["size"],
            config["kind"],
            config["dtype"],
            config["workers"],
            config["device"],
        )
        groups.setdefault(key, []).append(row["items_per_s"])
    return [
        {
            "arm": key[0],
            "size": key[1],
            "kind": key[2],
            "dtype": key[3],
            "workers": key[4],
            "device": key[5],
            "items_per_s_mean": statistics.mean(values),
            "items_per_s_stdev": statistics.stdev(values) if len(values) > 1 else None,
            "fresh_process_repeats": len(values),
        }
        for key, values in groups.items()
    ]


def main(argv: list[str] | None = None) -> None:
    """Run the four-arm public comparison CLI or bounded CPU/source checks.

    Args:
        argv: CLI arguments, or ``None`` to read ``sys.argv``.

    Returns:
        None; writes structured JSON to ``--output`` or stdout. Reports include
        raw per-item receipts, grouped throughput means/sample SD and scope legend.
        ``--device cpu`` selects only unpinned public workflows. ``--self-test``
        checks bounded CPU epochs without CUDA; ``--source-receipt`` hashes only
        this benchmark, not private library modules or installed native packages.

    Raises:
        SystemExit: Unsupported/retired CLI flags or invalid configuration.
        RuntimeError: A public consumer/producer workflow failed.

    Example:
        >>> main(['--device', 'cpu', '--arms', 'pool-pageable', '--sizes-bytes', '8192',
        ...       '--items', '2', '--repeats', '1', '--output', 'cpu-results.json'])
    """
    args = _parse_args(argv)
    if args.source_receipt:
        report = {"schema": _SCHEMA, "source_receipt": _source_receipt()}
    else:
        rows = []
        sizes = args.sizes_bytes or [size << 20 for size in args.sizes]
        for size, kind, dtype, repeat in product(
            sizes, args.kinds, args.dtypes, range(args.repeats)
        ):
            config = {
                "size": size,
                "kind": kind,
                "dtype": dtype,
                "repeat": repeat,
                "workers": args.workers,
                "slots": args.slots,
                "items": args.items,
                "warmup_epochs": args.warmup_epochs,
                "timeout": args.timeout,
                "local_rank": args.local_rank,
                "device": args.device,
            }
            order = (
                args.arms[repeat % len(args.arms) :]
                + args.arms[: repeat % len(args.arms)]
            )
            rows.extend(run_isolated(arm, config) for arm in order)
        report = {
            "schema": _SCHEMA,
            "legend": _legend(),
            "results": rows,
            "summary": _summaries(rows),
        }
    text = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
