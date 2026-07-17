#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Plot results from :py:mod:`benchmark_ipc_dataloader`.

Reads the CSV written by ``benchmark_ipc_dataloader.py --output`` and renders a
single line chart: DataLoader **startup time** (creating the iterator through the
first batch) against payload size, with **one line per worker count** and a shaded
~95% confidence band. A dashed grey line shows the dataset **build** time (object
instantiation) for contrast — it stays low and flat because no data crosses a
process boundary, while startup climbs with both size and worker count.

**Example**

.. code-block:: shell

   $ python benchmark_ipc_dataloader.py --output /tmp/ipc.csv
   $ python benchmark_ipc_dataloader_plot.py --input /tmp/ipc.csv --output /tmp/ipc.png
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt

try:
    from examples.benchmark_ipc_dataloader import read_csv  # pyre-ignore[21]
except ImportError:
    from spdl.examples.benchmark_ipc_dataloader import read_csv


def plot_results(input_csv: str, output_path: str) -> None:
    """Render startup-time-vs-size, one line per worker count."""
    rows = read_csv(input_csv)
    sizes = sorted({r.total_mb for r in rows})
    workers = sorted({r.num_workers for r in rows})
    start_method = rows[0].start_method if rows else "spawn"

    fig, ax = plt.subplots(figsize=(8, 5))
    for w in workers:
        series = sorted(
            (r for r in rows if r.num_workers == w), key=lambda r: r.total_mb
        )
        xs = [r.total_mb for r in series]
        ys = [r.startup_sec for r in series]
        (line,) = ax.plot(xs, ys, marker="o", label=f"{w} workers")
        ax.fill_between(
            xs,
            [r.startup_sec_lo for r in series],
            [r.startup_sec_hi for r in series],
            color=line.get_color(),
            alpha=0.15,
        )

    # Dataset instantiation, for contrast: worker-independent, so average per size.
    build_by_size = [_mean(r.build_sec for r in rows if r.total_mb == s) for s in sizes]
    ax.plot(
        sizes,
        build_by_size,
        linestyle="--",
        color="0.5",
        marker="x",
        label="dataset build (instantiation)",
    )

    ax.set_xlabel("dataset payload size (MiB)")
    ax.set_ylabel("time to first batch (s)")
    ax.set_title(f"Cost of shipping a dataset to DataLoader workers ({start_method})")
    ax.margins(y=0.1)
    ax.grid(True, alpha=0.3)
    ax.legend(title="DataLoader")
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    print(f"saved plot to {output_path}")


def _mean(values: "object") -> float:
    xs = list(values)  # pyre-ignore[6]
    return sum(xs) / len(xs) if xs else 0.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot IPC DataLoader benchmark results"
    )
    parser.add_argument(
        "--input", required=True, help="CSV from benchmark_ipc_dataloader"
    )
    parser.add_argument("--output", default="ipc_dataloader.png")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    plot_results(args.input, args.output)


if __name__ == "__main__":
    main()
