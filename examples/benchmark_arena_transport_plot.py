#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Plot results from :py:mod:`benchmark_arena_transport`.

Reads the CSV written by ``benchmark_arena_transport.py --output`` and renders a
grid of grouped bar charts: one **row per payload type** (bytes / NumPy / Torch /
packets) and three **columns** — recv throughput (MB/s), CPU seconds per config,
and peak RSS (MB). Within each panel the bars are grouped by payload size, with
one bar per transport (no-arena / ring / pool) so the three transports can be
compared directly. The CPU and memory columns are populated by ``--isolate`` runs
(each config measured in its own process); without ``--isolate`` they are zero.

**Example**

.. code-block:: shell

   $ python benchmark_arena_transport.py --isolate --output /tmp/arena.csv
   $ python benchmark_arena_transport_plot.py --input /tmp/arena.csv --output /tmp/arena.png
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

try:
    from examples.benchmark_arena_transport import read_csv, Row  # pyre-ignore[21]
except ImportError:
    from spdl.examples.benchmark_arena_transport import read_csv, Row

_TRANSPORTS = ("no-arena", "ring", "pool")
_KIND_ORDER = ("bytes", "numpy", "torch", "packets")
# Colour encodes the transport (the comparison of interest): no-arena is the grey
# baseline, ring/pool the two arena backends.
_T_COLORS = {"no-arena": "0.6", "ring": "C0", "pool": "C2"}
# (attr, axis label, short title) for the three metric columns.
_METRICS = (
    ("mb_per_s", "recv MB/s", "throughput (higher better)"),
    ("cpu_sec", "CPU s", "CPU per config (lower better)"),
    ("peak_rss_mb", "peak RSS (MB)", "peak memory (lower better)"),
)


def _index(rows: list[Row]) -> dict[tuple[int, str, str], Row]:
    """Index rows by ``(size_mb, kind, transport)`` for O(1), loop-free lookup."""
    return {(r.size_mb, r.kind, r.transport): r for r in rows}


def _bar_panel(
    ax: Axes,
    idx: dict[tuple[int, str, str], Row],
    kind: str,
    sizes: list[int],
    attr: str,
    ylabel: str,
) -> None:
    """Grouped bars for one (kind, metric): payload size on x, one bar per transport.

    Bars are grouped by payload size; within a group the three transports
    (no-arena / ring / pool) sit side by side so they can be read off directly.
    """
    width = 0.26
    x = list(range(len(sizes)))
    for i, transport in enumerate(_TRANSPORTS):
        offset = (i - 1) * width  # centre the 3-bar group on each tick
        vals = [getattr(idx[(s, kind, transport)], attr) for s in sizes]
        ax.bar(
            [xi + offset for xi in x],
            vals,
            width,
            label=transport,
            color=_T_COLORS[transport],
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s} MiB" for s in sizes])
    ax.set_ylabel(ylabel)
    ax.margins(y=0.12)


def plot_results(input_csv: str, output_path: str) -> None:
    """Render a (kind row) x (metric column) grid of grouped per-transport bars."""
    rows = read_csv(input_csv)
    idx = _index(rows)
    sizes = sorted({r.size_mb for r in rows})
    kinds = [k for k in _KIND_ORDER if any(r.kind == k for r in rows)]

    nrows, ncols = len(kinds), len(_METRICS)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.7 * ncols, 2.7 * nrows), squeeze=False
    )
    for r, kind in enumerate(kinds):
        for c, (attr, ylabel, title) in enumerate(_METRICS):
            ax = axes[r][c]
            _bar_panel(ax, idx, kind, sizes, attr, ylabel)
            ax.set_title(f"{kind} — {title}" if r == 0 else kind, fontsize=9)
    axes[0][0].legend(fontsize=8, title="transport")
    fig.suptitle("Arena transport per payload type: regular IPC vs ring vs pool")
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    print(f"saved plot to {output_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot arena transport benchmark results"
    )
    parser.add_argument(
        "--input", required=True, help="CSV from benchmark_arena_transport"
    )
    parser.add_argument("--output", default="arena_transport.png")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    plot_results(args.input, args.output)


if __name__ == "__main__":
    main()
