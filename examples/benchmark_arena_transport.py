#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Targeted benchmark: arena transport throughput (regular IPC vs ring vs pool).

Measures how the shared-memory arena affects the cost of shipping large payloads
from a backend :py:class:`~spdl.pipeline.Pipeline` running in a subprocess to the
main process, via :py:func:`spdl.pipeline.run_pipeline_in_subprocess`.

A backend pipeline produces a fixed dataset of ``{"data": payload}`` items; the
main process receives them through one of three transports and does nothing with
them (a no-op stand-in for a training loop), so the measured throughput isolates
**transfer + restore** with no decode or other per-item work in the way:

- **no-arena** — payloads are pickled over the multiprocessing queue (the default).
- **ring** — :py:class:`~spdl.pipeline.SharedMemoryRingBuffer`; the reader copies
  each payload out of shared memory (so it cannot hand out live views).
- **pool** — :py:class:`~spdl.pipeline.SharedMemorySegmentPool`; the reader
  restores each payload as a zero-copy view directly over shared memory.

It sweeps payload sizes x four payload types — ``bytes``, NumPy arrays, Torch
tensors, and ``spdl.io`` ``VideoPackets`` — each of which the arena offloads as a
large binary.

**Results**

From ``--isolate`` at 32 MiB on a CPU-only host (``--num-items 16 --runs 10``):
each ``(kind, transport)`` runs in its own freshly-spawned process, one at a time,
so the ``CPU s`` and ``peak RSS`` columns are attributable to that one config
(``getrusage`` over the timed passes / from a pre-arena baseline; a single shared
process would accumulate both across configs and could not separate them). The
arena raises throughput *and* drops the CPU of moving a payload to near zero —
the pool restores a zero-copy view, so it does essentially no per-item work, while
plain IPC spends several CPU-seconds pickling and copying. ``bytes`` / NumPy gain
the most. Torch transfers tensors through shared memory itself (its
multiprocessing reducer), so plain IPC already avoids a bulk copy — its throughput
barely moves and plain IPC is already its leanest memory; the ring even adds CPU
(it copies out), yet the pool still drops its CPU to ~0. Packets gain less
throughput (restoring rebuilds the ``AVPacket`` structures) but the pool slashes
their CPU and memory too. ``peak RSS`` includes a fixed per-process baseline, so
compare it within a row; for ``bytes`` / NumPy / packets the no-arena pickle
buffers are the heaviest, and the pool the lightest.

.. code-block:: text

   kind     transport   recv MB/s   speedup   CPU s   peak RSS MB
   ------------------------------------------------------------
   bytes    no-arena          719     1.00x     7.1          1872
   bytes    ring             2684     3.73x     1.8          1015
   bytes    pool             3760     5.23x     0.0           673
   numpy    no-arena          647     1.00x     7.7          1865
   numpy    ring             3202     4.95x     1.1          1009
   numpy    pool             3767     5.82x     0.0           669
   torch    no-arena         2899     1.00x     0.6           603
   torch    ring             3106     1.07x     1.2          1262
   torch    pool             3116     1.08x     0.0           834
   packets  no-arena          555     1.00x     8.6          1885
   packets  ring             1359     2.45x     3.3          1236
   packets  pool             1622     2.92x     0.1           825
   (recv MB/s with speedup vs no-arena; CPU s and peak RSS are per-config,
   lower is better. Throughput speedup grows with payload size — at 2 / 8 MiB
   it is smaller; see the size sweep below.)

In the plot below, each line is one (kind, transport) across payload sizes (a
separate throughput sweep): colour and marker encode the payload type and line
style the transport (dotted = no-arena, dashed = ring, solid = pool); the shaded
band is the ~95% confidence interval of the mean:

.. image:: ../../_static/data/example_benchmark_arena_transport.png

**Example**

.. code-block:: shell

   $ python benchmark_arena_transport.py --sizes 2 8 32 --output results.csv
   $ python benchmark_arena_transport_plot.py --input results.csv --output plot.png
"""

from __future__ import annotations

__all__ = [
    "Row",
    "create_video_data",
    "main",
    "read_csv",
    "run_transport",
    "write_csv",
]

import argparse
import csv
import gc
import multiprocessing as mp
import resource
import statistics
import tempfile
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, fields
from itertools import product
from typing import Any

import numpy as np
import spdl.io
import torch
from spdl.pipeline import (
    PipelineBuilder,
    run_pipeline_in_subprocess,
    SharedMemoryRingBuffer,
    SharedMemorySegmentPool,
)

_KINDS = ("bytes", "numpy", "torch", "packets")
_TRANSPORTS = ("no-arena", "ring", "pool")
# 4K frames: at the benchmark's payload sizes the encoder fills the bitrate with
# real frame data instead of mostly CBR filler, so the packets resemble a real
# high-resolution decode workload.
_VIDEO_SIZE = "3840x2160"


@dataclass(frozen=True)
class Row:
    """Row()

    One benchmark measurement: throughput for a (size, kind, transport)."""

    size_mb: int
    """Requested payload size in MiB (the sweep knob)."""

    kind: str
    """Payload kind: ``"bytes"``, ``"numpy"``, ``"torch"``, or ``"packets"``."""

    transport: str
    """Transport used: ``"no-arena"``, ``"ring"``, or ``"pool"``."""

    items_per_s: float
    """Mean items received per second over the timed passes."""

    mb_per_s: float
    """Mean payload throughput in MB/s (``items_per_s`` x payload bytes)."""

    mb_per_s_lo: float
    """Lower bound of the ~95% confidence interval of ``mb_per_s`` (normal
    approximation over the timed passes)."""

    mb_per_s_hi: float
    """Upper bound of the ~95% confidence interval of ``mb_per_s``."""

    speedup: float
    """``items_per_s`` relative to the no-arena baseline for this (size, kind)."""

    peak_rss_mb: float = 0.0
    """Peak resident memory of the config's process tree (consumer + producer), in
    MB. Only populated by ``--isolate``; ``0`` otherwise. Shared arena pages may be
    counted in both processes, so treat it as an upper-bound proxy and compare
    within a (size, kind) row across transports."""

    cpu_sec: float = 0.0
    """Total CPU seconds (user + system, consumer + producer) the config consumed.
    Only populated by ``--isolate``; ``0`` otherwise. Lower is better — moving a
    payload via the arena should cost less CPU than the pickle + copy of plain IPC."""


def create_video_data(target_bytes: int, duration_seconds: float = 2.0) -> bytes:
    """Generate an H.264 MP4 whose stream is approximately ``target_bytes``.

    Encodes low-entropy frames at a constant target bitrate (x264 CBR, which pads
    with filler to hold the rate), so the serialized ``VideoPackets`` payload
    scales with the requested size rather than collapsing to the content's natural
    compressed size. Uses ``spdl.io``'s in-process encoder rather than the
    ``ffmpeg`` CLI, so it also works on minimal container images that do not ship
    the CLI.

    Args:
        target_bytes: Desired size of the encoded video stream, in bytes.
        duration_seconds: Clip duration; the bitrate is derived from it.

    Returns:
        The encoded MP4 file contents as raw bytes.
    """
    width, height = (int(x) for x in _VIDEO_SIZE.split("x"))
    frame_rate = (30, 1)
    num_frames = max(1, round(frame_rate[0] / frame_rate[1] * duration_seconds))
    bit_rate = max(64_000, int(target_bytes * 8 / duration_seconds))
    kbps = bit_rate // 1000
    batch_size = 4  # bound peak memory: 4K yuv444p frames are ~25 MiB each
    with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp_file:
        muxer = spdl.io.Muxer(tmp_file.name)
        encoder = muxer.add_encode_stream(
            config=spdl.io.video_encode_config(
                height=height,
                width=width,
                pix_fmt="yuv444p",
                frame_rate=frame_rate,
                bit_rate=bit_rate,
            ),
            encoder="libx264",
            # CBR with a tight VBV so the encoder pads low-entropy frames with
            # filler up to the target rate (mirrors the ffmpeg nal-hrd=cbr path).
            encoder_config={
                "preset": "ultrafast",
                "x264-params": f"nal-hrd=cbr:vbv-maxrate={kbps}:vbv-bufsize={kbps}",
            },
        )
        with muxer.open():
            for start in range(0, num_frames, batch_size):
                n = min(batch_size, num_frames - start)
                array = np.zeros((n, 3, height, width), dtype=np.uint8)
                frames = spdl.io.create_reference_video_frame(
                    array=array,
                    pix_fmt="yuv444p",
                    frame_rate=frame_rate,
                    pts=start,
                )
                if (packets := encoder.encode(frames)) is not None:
                    muxer.write(0, packets)
            if (packets := encoder.flush()) is not None:
                muxer.write(0, packets)
        with open(tmp_file.name, "rb") as f:
            return f.read()


def _make_payload(kind: str, size: int, video_bytes: bytes | None) -> object:
    """Build one payload of ``kind`` whose serialized size is roughly ``size``."""
    if kind == "bytes":
        return bytes(size)
    if kind == "numpy":
        return np.zeros(max(1, size // 4), dtype=np.float32)
    if kind == "torch":
        return torch.zeros(max(1, size // 4), dtype=torch.float32)
    if kind == "packets":
        assert video_bytes is not None
        return spdl.io.demux_video(video_bytes)
    raise ValueError(f"unknown kind: {kind}")


def _payload_nbytes(kind: str, size: int, video_bytes: bytes | None) -> int:
    """The on-the-wire size of one payload, used to size the arena."""
    payload: Any = _make_payload(kind, size, video_bytes)
    if kind == "packets":
        return len(payload.__getstate__())
    if kind == "numpy":
        return int(payload.nbytes)
    if kind == "torch":
        return int(payload.element_size() * payload.nelement())
    return len(payload)


@dataclass(frozen=True)
class _Dataset:
    """A picklable finite dataset: yields ``num_items`` payload items.

    Only the small spec is pickled into the subprocess; the payload itself is
    built lazily in ``__iter__`` (i.e. in the worker), so shipping the dataset
    across the process boundary stays cheap regardless of payload size.

    Most kinds reuse one payload object so the measured per-item cost is transfer
    + restore rather than construction. Torch is the exception: it moves a CPU
    tensor's storage into shared memory in place on the first IPC send and reuses
    it afterwards, so reusing one tensor would hide the real cost (a fresh
    shared-memory segment per distinct tensor). It therefore yields a distinct
    tensor per item, matching how a real pipeline produces tensors.
    """

    kind: str
    size: int
    num_items: int
    video_bytes: bytes | None = None

    def __iter__(self) -> Iterator[dict[str, object]]:
        # Torch moves a CPU tensor's storage into shared memory in place on the
        # first cross-process send and reuses that same segment for later sends of
        # the *same* tensor (torch.multiprocessing's reducer + its storage cache).
        # So if the benchmark reused one tensor, every send after the first would
        # be nearly free and hide the real per-item cost (a fresh shm segment per
        # distinct tensor) — build a distinct tensor per item for torch instead.
        reused = (
            None
            if self.kind == "torch"
            else _make_payload(self.kind, self.size, self.video_bytes)
        )
        for _ in range(self.num_items):
            payload = (
                reused
                if reused is not None
                else _make_payload(self.kind, self.size, self.video_bytes)
            )
            yield {"data": payload}


def _make_arena(
    transport: str, payload_nbytes: int, buffer_size: int, num_items: int
) -> SharedMemoryRingBuffer | SharedMemorySegmentPool | None:
    """Construct the arena for a transport mode, sized to hold a whole iteration.

    The arena does not block the writer when full — it raises "shared-memory
    arena full" and relies on the pipeline's queue backpressure to bound the
    in-flight units. That bound only holds when the reader keeps up; when restore
    is much slower than offload (e.g. rebuilding ``VideoPackets`` on a busy host),
    the writer can get a whole iteration ahead. So size the arena for the
    worst case — every item of one iteration in flight at once — which never
    overruns regardless of the producer/consumer speed gap.
    """
    if transport == "no-arena":
        return None
    slots = max(2 * buffer_size + 6, num_items + 2)
    unit = max(1 << 20, payload_nbytes * 2)  # headroom for envelope + alignment
    if transport == "ring":
        return SharedMemoryRingBuffer(capacity=unit * slots)
    if transport == "pool":
        return SharedMemorySegmentPool(segment_size=unit, count=slots)
    raise ValueError(f"unknown transport: {transport}")


def run_transport(
    dataset: _Dataset,
    transport: str,
    *,
    payload_nbytes: int,
    num_items: int,
    buffer_size: int,
    runs: int,
    duration_sec: float = 0.0,
    usage: list[float] | None = None,
) -> list[float]:
    """Run the backend pipeline once per timed pass; return per-pass items/second.

    Builds a one-stage backend pipeline (source -> sink) that runs in a
    subprocess via :py:func:`run_pipeline_in_subprocess`, ships its items to the
    main process over ``transport``, and consumes them in a no-op loop. One warmup
    pass is discarded; the per-pass throughput of each timed pass is returned so
    the caller can summarize it (mean + confidence interval). The subprocess and
    arena are reused across passes and torn down at the end.

    Args:
        dataset: The (picklable) backend data source.
        transport: One of ``"no-arena"``, ``"ring"``, ``"pool"``.
        payload_nbytes: On-the-wire size of one payload, used to size the arena.
        num_items: Number of items per pass (must match ``dataset.num_items``).
        buffer_size: Pipeline sink buffer size / arena in-flight unit count.
        runs: Number of timed passes (used when ``duration_sec`` is not positive).
        duration_sec: If positive, keep running timed passes until this many
            seconds have elapsed instead of stopping after ``runs`` passes — used
            for sustained, fixed-duration host-stat sampling by an external sampler.
        usage: If given, one element is appended: the CPU seconds (user + system,
            this process + the producer subprocess) spent across the timed passes.
            The snapshot is taken *after* the warmup pass, so process-startup and
            import costs are excluded and only the transport's per-item work counts.

    Returns:
        One throughput sample (items per second) per timed pass.
    """
    arena = _make_arena(transport, payload_nbytes, buffer_size, num_items)
    config = (
        PipelineBuilder()
        .add_source(dataset)
        .add_sink(buffer_size=buffer_size)
        .get_config()
    )
    src = run_pipeline_in_subprocess(
        config, num_threads=1, arena=arena, buffer_size=buffer_size
    )
    try:
        for item in src:  # warmup pass (subprocess spawn + first-touch costs)
            del item
        cpu_base = _cpu_now()  # after warmup: imports + arena warm-up excluded
        samples: list[float] = []
        deadline = time.perf_counter() + duration_sec
        while True:
            n = 0
            t0 = time.perf_counter()
            for item in src:
                n += 1
                del item  # release the view so the pool can recycle the segment
            samples.append(num_items / (time.perf_counter() - t0))
            assert n == num_items, f"{transport}: expected {num_items}, got {n}"
            done_by_count = duration_sec <= 0 and len(samples) >= runs
            done_by_time = duration_sec > 0 and time.perf_counter() >= deadline
            if done_by_count or done_by_time:
                break
        if usage is not None:
            usage.append(_cpu_now() - cpu_base)
        return samples
    finally:
        # Drop the iterable so its finalizer closes + unlinks the arena before the
        # next config reuses the shared-memory namespace.
        del src
        gc.collect()


def _confidence_interval(samples: list[float]) -> tuple[float, float]:
    """~95% confidence interval of the mean (normal approximation).

    Degenerate cases (a single pass) return ``(mean, mean)``.
    """
    mean = statistics.mean(samples)
    if len(samples) < 2:
        return mean, mean
    half = 1.96 * statistics.stdev(samples) / (len(samples) ** 0.5)
    return mean - half, mean + half


def _run_config(size_mb: int, kind: str, args: argparse.Namespace) -> list[Row]:
    """Benchmark one (size, kind) across the selected transports; one Row each."""
    size = size_mb << 20
    # For packets, synthesize a clip whose serialized payload tracks ``size`` (the
    # other kinds build a payload of exactly ``size`` bytes directly).
    vb = create_video_data(size) if kind == "packets" else None
    payload_nbytes = _payload_nbytes(kind, size, vb)
    dataset = _Dataset(kind, size, args.num_items, vb)
    factor = payload_nbytes / 1e6  # items/s -> MB/s
    rows: list[Row] = []
    baseline = 0.0
    for i, transport in enumerate(args.transports):
        if i and args.gap_sec:
            time.sleep(args.gap_sec)  # idle valley between per-transport windows
        if args.duration_sec:
            # Timestamped window markers so externally-sampled host stats (e.g. an
            # external once-per-minute CPU/memory sampler) can be attributed to each config.
            print(
                f"### begin {kind} {size_mb}M {transport} ts={time.time():.0f}",
                flush=True,
            )
        samples = run_transport(
            dataset,
            transport,
            payload_nbytes=payload_nbytes,
            num_items=args.num_items,
            buffer_size=args.buffer_size,
            runs=args.runs,
            duration_sec=args.duration_sec,
        )
        if args.duration_sec:
            print(
                f"### end   {kind} {size_mb}M {transport} ts={time.time():.0f}",
                flush=True,
            )
        mean = statistics.mean(samples)
        lo, hi = _confidence_interval(samples)
        if transport == "no-arena":
            baseline = mean
        rows.append(
            Row(
                size_mb=size_mb,
                kind=kind,
                transport=transport,
                items_per_s=mean,
                mb_per_s=mean * factor,
                mb_per_s_lo=lo * factor,
                mb_per_s_hi=hi * factor,
                speedup=mean / baseline if baseline else 1.0,
            )
        )
    return rows


def _cpu_now() -> float:
    """CPU seconds (user + system) for this process plus its reaped children."""
    s = resource.getrusage(resource.RUSAGE_SELF)
    c = resource.getrusage(resource.RUSAGE_CHILDREN)
    return s.ru_utime + s.ru_stime + c.ru_utime + c.ru_stime


def _peak_rss_mb() -> float:
    """Peak RSS (self + reaped children) in MB (``ru_maxrss`` is KiB on Linux)."""
    s = resource.getrusage(resource.RUSAGE_SELF)
    c = resource.getrusage(resource.RUSAGE_CHILDREN)
    return (s.ru_maxrss + c.ru_maxrss) / 1024


def _isolated_worker(
    kind: str,
    size: int,
    transport: str,
    payload_nbytes: int,
    num_items: int,
    buffer_size: int,
    runs: int,
    vb: bytes | None,
    q: "mp.Queue[tuple[str, object, float, float]]",
) -> None:
    """Run one config in this fresh process; report samples + CPU + peak RSS.

    Run as the target of a freshly-spawned process. Because the process starts
    clean and exits after one config, the measurements attribute resources to
    *this* config alone — a single long-lived process accumulates them across
    configs and cannot separate them, which is the whole point of ``--isolate``.
    CPU is measured over the timed passes only (excludes this worker's and the
    producer's import/startup cost); peak RSS is the growth from a baseline taken
    before the arena and payloads are built. The video bytes are encoded in the
    parent and passed in, so the encoder's footprint does not land in this peak.
    """
    try:
        rss_base = _peak_rss_mb()
        usage: list[float] = []
        dataset = _Dataset(kind, size, num_items, vb)
        samples = run_transport(
            dataset,
            transport,
            payload_nbytes=payload_nbytes,
            num_items=num_items,
            buffer_size=buffer_size,
            runs=runs,
            usage=usage,
        )
        cpu = usage[0] if usage else 0.0
        q.put(("ok", samples, cpu, _peak_rss_mb() - rss_base))
    except Exception as e:  # surface failure instead of hanging the parent
        q.put(("err", f"{type(e).__name__}: {e}", 0.0, 0.0))


def _run_isolated(
    mp_ctx: "mp.context.BaseContext",
    kind: str,
    size: int,
    transport: str,
    payload_nbytes: int,
    num_items: int,
    buffer_size: int,
    runs: int,
    vb: bytes | None,
) -> tuple[list[float], float, float]:
    """Run ``_isolated_worker`` in a fresh process; return ``(samples, cpu, rss)``."""
    q: "mp.Queue[tuple[str, object, float, float]]" = mp_ctx.Queue()
    p = mp_ctx.Process(  # pyre-ignore[16]
        target=_isolated_worker,
        args=(
            kind,
            size,
            transport,
            payload_nbytes,
            num_items,
            buffer_size,
            runs,
            vb,
            q,
        ),
    )
    p.start()
    status, payload, cpu, rss = q.get()
    p.join()
    if status != "ok":
        raise RuntimeError(str(payload))
    return payload, cpu, rss  # pyre-ignore[7]


def _run_config_isolated(
    size_mb: int, kind: str, args: argparse.Namespace, mp_ctx: "mp.context.BaseContext"
) -> list[Row]:
    """Benchmark one (size, kind), each transport in its own fresh process.

    Per-process isolation is what makes the CPU-time and peak-RSS columns
    meaningful: each transport's footprint is measured from a clean slate, so the
    arena's shared-memory cost and the no-arena pickle/copy cost are attributable
    rather than piled into one accumulating process.
    """
    size = size_mb << 20
    # Encode in the parent so the (large) encoder footprint stays out of the
    # per-config worker's peak RSS.
    vb = create_video_data(size) if kind == "packets" else None
    payload_nbytes = _payload_nbytes(kind, size, vb)
    factor = payload_nbytes / 1e6  # items/s -> MB/s
    rows: list[Row] = []
    baseline = 0.0
    for transport in args.transports:
        samples, cpu, rss = _run_isolated(
            mp_ctx,
            kind,
            size,
            transport,
            payload_nbytes,
            args.num_items,
            args.buffer_size,
            args.runs,
            vb,
        )
        mean = statistics.mean(samples)
        lo, hi = _confidence_interval(samples)
        if transport == "no-arena":
            baseline = mean
        rows.append(
            Row(
                size_mb=size_mb,
                kind=kind,
                transport=transport,
                items_per_s=mean,
                mb_per_s=mean * factor,
                mb_per_s_lo=lo * factor,
                mb_per_s_hi=hi * factor,
                speedup=mean / baseline if baseline else 1.0,
                peak_rss_mb=rss,
                cpu_sec=cpu,
            )
        )
    return rows


def _table_header() -> str:
    """Header for the pivoted table (one transport column group per row)."""
    return f"{'kind':<8} {'size':>5} {'no-arena':>9} {'ring':>15} {'pool':>15}"


def _cell(
    idx: dict[tuple[int, str, str], Row], size_mb: int, kind: str, transport: str
) -> str:
    """MB/s for one (size, kind, transport), with speedup in (x); ``-`` if not run."""
    r = idx.get((size_mb, kind, transport))
    if r is None:
        return "-"
    if transport == "no-arena":
        return f"{r.mb_per_s:.0f}"
    return f"{r.mb_per_s:.0f} ({r.speedup:.2f}x)"


def _table_row(idx: dict[tuple[int, str, str], Row], size_mb: int, kind: str) -> str:
    """One pivoted row: no-arena/ring/pool MB/s, with ring/pool speedup in (x)."""
    na = _cell(idx, size_mb, kind, "no-arena")
    ring = _cell(idx, size_mb, kind, "ring")
    pool = _cell(idx, size_mb, kind, "pool")
    return f"{kind:<8} {size_mb:>4}M {na:>9} {ring:>15} {pool:>15}"


def _print_table(rows: list[Row]) -> None:
    """Print one row per (size, kind); transports are columns (MB/s, speedup)."""
    idx = {(r.size_mb, r.kind, r.transport): r for r in rows}
    keys = sorted(
        {(r.size_mb, r.kind) for r in rows},
        key=lambda k: (_KINDS.index(k[1]), k[0]),
    )
    header = _table_header()
    print(header)
    print("-" * len(header))
    for size_mb, kind in keys:
        print(_table_row(idx, size_mb, kind))
    print("(throughput in MB/s; (Nx) = speedup over the no-arena baseline)")


def _print_isolate_table(rows: list[Row]) -> None:
    """One row per (size, kind, transport): throughput, CPU time, peak RSS.

    Used for ``--isolate`` runs, where CPU and RSS are per-config (each transport
    ran in its own process).
    """
    rows = sorted(
        rows,
        key=lambda r: (_KINDS.index(r.kind), r.size_mb, _TRANSPORTS.index(r.transport)),
    )
    header = (
        f"{'kind':<8} {'size':>5} {'transport':<9} {'recv MB/s':>10} "
        f"{'speedup':>8} {'CPU s':>7} {'peak RSS MB':>12}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r.kind:<8} {r.size_mb:>4}M {r.transport:<9} {r.mb_per_s:>10.0f} "
            f"{r.speedup:>7.2f}x {r.cpu_sec:>7.1f} {r.peak_rss_mb:>12.0f}"
        )
    print(
        "(recv MB/s with speedup vs no-arena; CPU s and peak RSS are per-config, "
        "lower is better)"
    )


def write_csv(rows: list[Row], path: str) -> None:
    """Write benchmark rows to ``path`` as CSV (one column per :class:`Row` field)."""
    names = [f.name for f in fields(Row)]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=names)
        writer.writeheader()
        writer.writerows(asdict(r) for r in rows)
    print(f"wrote {len(rows)} rows to {path}")


def read_csv(path: str) -> list[Row]:
    """Read benchmark rows written by :py:func:`write_csv`."""
    with open(path, newline="") as f:
        return [
            Row(
                size_mb=int(d["size_mb"]),
                kind=d["kind"],
                transport=d["transport"],
                items_per_s=float(d["items_per_s"]),
                mb_per_s=float(d["mb_per_s"]),
                mb_per_s_lo=float(d["mb_per_s_lo"]),
                mb_per_s_hi=float(d["mb_per_s_hi"]),
                speedup=float(d["speedup"]),
                peak_rss_mb=float(d.get("peak_rss_mb") or 0.0),
                cpu_sec=float(d.get("cpu_sec") or 0.0),
            )
            for d in csv.DictReader(f)
        ]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Arena transport throughput: regular IPC vs ring vs pool"
    )
    parser.add_argument("--num-items", type=int, default=300)
    parser.add_argument("--buffer-size", type=int, default=3)
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=[2, 8, 32], help="payload sizes (MiB)"
    )
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument(
        "--kinds",
        nargs="+",
        choices=_KINDS,
        default=list(_KINDS),
        help="payload types to benchmark",
    )
    parser.add_argument(
        "--transports",
        nargs="+",
        choices=_TRANSPORTS,
        default=list(_TRANSPORTS),
        help="transports to benchmark (use one at a time to isolate host stats)",
    )
    parser.add_argument(
        "--duration-sec",
        type=float,
        default=0.0,
        help="if >0, run each (transport, kind, size) for this long instead of "
        "--runs passes — for sustained host-stat sampling by an external sampler",
    )
    parser.add_argument(
        "--gap-sec",
        type=float,
        default=0.0,
        help="idle seconds inserted between transports, to separate per-transport "
        "windows for once-per-minute host-stat sampling",
    )
    parser.add_argument(
        "--isolate",
        action="store_true",
        help="run each (size, kind, transport) in its own fresh process and report "
        "per-config CPU time and peak RSS (clean memory/CPU attribution, no "
        "carryover between configs)",
    )
    parser.add_argument("--output", help="optional path to write results as CSV")
    return parser.parse_args()


def main() -> None:
    """Sweep payload sizes x types x transports; print a table and optionally a CSV."""
    args = _parse_args()
    # spawn (not fork) so each isolated worker starts from a clean interpreter,
    # giving comparable per-config peak RSS rather than inheriting the parent's.
    mp_ctx = mp.get_context("spawn") if args.isolate else None
    rows: list[Row] = []
    for size_mb, kind in product(args.sizes, args.kinds):
        try:
            if mp_ctx is not None:
                rows.extend(_run_config_isolated(size_mb, kind, args, mp_ctx))
            else:
                rows.extend(_run_config(size_mb, kind, args))
        except (OSError, RuntimeError) as e:
            # Building a kind's payload (e.g. encoding the packets clip) can fail
            # on some hosts; skip that kind rather than abort the whole sweep.
            print(f"### skip {kind} {size_mb}M: {type(e).__name__}: {e}", flush=True)
    if args.isolate:
        _print_isolate_table(rows)
    else:
        _print_table(rows)
    if args.output:
        write_csv(rows, args.output)


if __name__ == "__main__":
    main()
