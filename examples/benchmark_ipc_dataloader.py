#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Targeted benchmark: the cost of shipping a dataset to DataLoader workers.

Motivating observation: constructing TorchVision's ImageNet dataset object takes
under a second, yet the first time a :py:class:`torch.utils.data.DataLoader` with
workers is iterated there is a multi-second stall before any batch appears. That
stall is **inter-process communication (IPC)**: the dataset — roughly 1.2 million
``(path, label)`` entries — is pickled and copied to every worker process. This
benchmark reproduces that effect in isolation and shows how it scales.

A :py:class:`ByteStringDataset` holds a list of ``num_strings`` byte strings whose
total size is the sweep knob (mirroring the many small path strings in an
``ImageFolder``-style dataset). We measure two things per configuration:

- **build** — time to *instantiate* the dataset object (build the Python list).
  This is cheap and roughly flat: no data crosses a process boundary.
- **startup** — time from creating the DataLoader iterator (which spawns the
  workers and ships the dataset to each) to receiving the first batch. This is
  the IPC cost, and it grows with both the payload size and the worker count.

``__getitem__`` returns a single ``int`` so the per-item transfer *back* from the
workers is negligible; the measured startup time is dominated by pickling and
copying the dataset *out* to the workers.

.. important::

   The effect is only visible under the ``spawn`` (or ``forkserver``) start
   method, where the dataset is pickled and streamed to each worker through a
   pipe. Under ``fork`` (the Linux default) the workers inherit the parent's
   memory copy-on-write and the dataset is *not* re-serialized, so startup stays
   flat — the benchmark defaults to ``spawn`` to expose the IPC cost, which is
   also the start method used with CUDA. See the :ref:`ipc-cost` case study.

**Results**

From ``--sizes 16 32 64 128 --workers 1 2 4 8 --runs 3`` on a CPU-only host
(``spawn``, 64-byte strings). Building the dataset stays well under a second and
barely moves with size, while startup climbs with both the payload size and the
worker count — every worker gets its own serialized copy:

.. code-block:: text

   payload   workers   build s   startup s
   ----------------------------------------
    16 MiB         1      0.09         2.9
   128 MiB         1      0.65         3.8
    16 MiB         8      0.08        22.4
   128 MiB         8      0.65        28.3
   (startup = create iterator -> first batch; grows with size x workers, while
   build = dataset instantiation stays flat. See the plot for the full sweep.)

.. image:: ../../_static/data/example_benchmark_ipc_dataloader.png

**Example**

.. code-block:: shell

   $ python benchmark_ipc_dataloader.py --sizes 16 32 64 128 --output ipc.csv
   $ python benchmark_ipc_dataloader_plot.py --input ipc.csv --output ipc.png
"""

from __future__ import annotations

__all__ = [
    "ByteStringDataset",
    "Row",
    "main",
    "measure_startup",
    "read_csv",
    "write_csv",
]

import argparse
import csv
import gc
import multiprocessing as mp
import statistics
import time
from dataclasses import asdict, dataclass, fields
from itertools import product

from torch.utils.data import DataLoader, Dataset

_MiB = 1 << 20


class ByteStringDataset(Dataset):
    """A dataset of ``num_strings`` distinct byte strings of ``string_bytes`` each.

    The strings are path-like and vary by index, so they are genuinely distinct
    objects (as real file paths are) and pickle cannot dedupe them — every one is
    traversed and copied when the dataset is shipped to a worker. The total
    payload is ``num_strings * string_bytes``.

    ``__getitem__`` returns only the length of an entry, not the entry itself, so
    the cost measured by the benchmark is the *outbound* transfer of the dataset
    to the workers rather than any per-item work.
    """

    def __init__(self, num_strings: int, string_bytes: int) -> None:
        self.data: list[bytes] = [
            _make_entry(i, string_bytes) for i in range(num_strings)
        ]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> int:
        return len(self.data[index])


def _make_entry(index: int, string_bytes: int) -> bytes:
    """A distinct, path-like byte string padded / trimmed to ``string_bytes``."""
    s = b"/data/imagenet/train/img_%012d.JPEG" % index
    if len(s) < string_bytes:
        return s + b"\0" * (string_bytes - len(s))
    return s[:string_bytes]


@dataclass(frozen=True)
class Row:
    """Row()

    One measurement: dataset build + DataLoader startup for a (size, workers)."""

    total_mb: float
    """Total payload size shipped to each worker, in MiB (the sweep knob)."""

    num_strings: int
    """Number of byte strings in the dataset (``total_mb`` / ``string_bytes``)."""

    string_bytes: int
    """Size of each byte string, in bytes."""

    num_workers: int
    """Number of DataLoader worker processes."""

    start_method: str
    """Multiprocessing start method (``"spawn"``, ``"fork"``, ``"forkserver"``)."""

    build_sec: float
    """Time to instantiate the dataset object (build the list). Cheap and flat."""

    startup_sec: float
    """Mean time from creating the iterator to the first batch — the IPC cost."""

    startup_sec_lo: float
    """Lower bound of the ~95% confidence interval of ``startup_sec``."""

    startup_sec_hi: float
    """Upper bound of the ~95% confidence interval of ``startup_sec``."""


def measure_startup(
    dataset: ByteStringDataset,
    num_workers: int,
    *,
    mp_ctx: "mp.context.BaseContext",
    batch_size: int,
    runs: int,
) -> list[float]:
    """Time creating the DataLoader iterator through the first batch, ``runs`` times.

    Creating the iterator spawns the workers and ships the dataset to each; the
    first :py:func:`next` blocks until a worker has received the dataset and
    produced a batch, so the interval captures the outbound transfer. One warmup
    pass is discarded. The iterator is fully drained and dropped between passes so
    the workers are torn down and re-spawned each time.

    Returns:
        One startup sample (seconds) per timed pass.
    """
    samples: list[float] = []
    for pass_i in range(runs + 1):  # one warmup, then ``runs`` timed passes
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            multiprocessing_context=mp_ctx,
            persistent_workers=False,
        )
        t0 = time.perf_counter()
        it = iter(loader)  # spawn workers + ship the dataset to each
        next(it)  # block until the first batch crosses back
        elapsed = time.perf_counter() - t0
        for _ in it:  # drain so the workers exit cleanly
            pass
        del it, loader
        gc.collect()
        if pass_i:  # skip the warmup pass
            samples.append(elapsed)
    return samples


def _confidence_interval(samples: list[float]) -> tuple[float, float]:
    """~95% confidence interval of the mean (normal approximation).

    Degenerate cases (a single pass) return ``(mean, mean)``.
    """
    mean = statistics.mean(samples)
    if len(samples) < 2:
        return mean, mean
    half = 1.96 * statistics.stdev(samples) / (len(samples) ** 0.5)
    return mean - half, mean + half


def _run_config(
    size_mb: int,
    num_workers: int,
    args: argparse.Namespace,
    mp_ctx: "mp.context.BaseContext",
) -> Row:
    """Benchmark one (size, workers): time the build once and the startup ``runs`` times."""
    num_strings = (size_mb * _MiB) // args.string_bytes
    t0 = time.perf_counter()
    dataset = ByteStringDataset(num_strings, args.string_bytes)
    build_sec = time.perf_counter() - t0
    samples = measure_startup(
        dataset,
        num_workers,
        mp_ctx=mp_ctx,
        batch_size=args.batch_size,
        runs=args.runs,
    )
    lo, hi = _confidence_interval(samples)
    return Row(
        total_mb=num_strings * args.string_bytes / _MiB,
        num_strings=num_strings,
        string_bytes=args.string_bytes,
        num_workers=num_workers,
        start_method=args.start_method,
        build_sec=build_sec,
        startup_sec=statistics.mean(samples),
        startup_sec_lo=lo,
        startup_sec_hi=hi,
    )


def _print_table(rows: list[Row]) -> None:
    """Print one row per (size, workers): dataset build vs DataLoader startup."""
    header = (
        f"{'size':>7} {'strings':>10} {'workers':>7} {'build s':>8} {'startup s':>10}"
    )
    print(header)
    print("-" * len(header))
    for r in sorted(rows, key=lambda r: (r.num_workers, r.total_mb)):
        print(
            f"{r.total_mb:>6.0f}M {r.num_strings:>10} {r.num_workers:>7} "
            f"{r.build_sec:>8.2f} {r.startup_sec:>10.2f}"
        )
    print(
        "(build = dataset instantiation, flat; startup = iterator -> first batch, "
        "the IPC cost, which grows with size and worker count)"
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
                total_mb=float(d["total_mb"]),
                num_strings=int(d["num_strings"]),
                string_bytes=int(d["string_bytes"]),
                num_workers=int(d["num_workers"]),
                start_method=d["start_method"],
                build_sec=float(d["build_sec"]),
                startup_sec=float(d["startup_sec"]),
                startup_sec_lo=float(d["startup_sec_lo"]),
                startup_sec_hi=float(d["startup_sec_hi"]),
            )
            for d in csv.DictReader(f)
        ]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cost of shipping a dataset to DataLoader workers over IPC"
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[16, 32, 64, 128],
        help="payload sizes (MiB)",
    )
    parser.add_argument(
        "--workers", type=int, nargs="+", default=[1, 2, 4, 8], help="worker counts"
    )
    parser.add_argument(
        "--string-bytes", type=int, default=64, help="size of each byte string"
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument(
        "--start-method",
        default="spawn",
        choices=["spawn", "fork", "forkserver"],
        help="start method; fork inherits memory (no IPC copy) so shows the flat baseline",
    )
    parser.add_argument("--output", help="optional path to write results as CSV")
    return parser.parse_args()


def main() -> None:
    """Sweep payload sizes x worker counts; print a table and optionally a CSV."""
    args = _parse_args()
    mp_ctx = mp.get_context(args.start_method)
    rows = [
        _run_config(size_mb, num_workers, args, mp_ctx)
        for size_mb, num_workers in product(args.sizes, args.workers)
    ]
    _print_table(rows)
    if args.output:
        write_csv(rows, args.output)


if __name__ == "__main__":
    main()
