# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Plot autoresearch optimization progress in Karpathy's style.

Usage:
    buck run //spdl/tools/autoresearch:plot_progress -- <master_table.tsv> [output.png]

Reads the master_table.tsv produced by autoresearch and generates a two-subplot
chart:
  1. Job duration (primary metric, lower is better) — Karpathy-style scatter
  2. SM utilization (secondary metric) — scatter for context
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

__all__ = [
    "_edge_label",
    "_load_tsv",
    "_plot_hypothesis_tree",
    "_plot_progress",
    "_tree_font_sizes",
    "main",
]


def _unescape(value: str) -> str:
    return value.replace("\\n", "\n").replace("\\t", "\t").replace("\\\\", "\\")


def _load_tsv(tsv_path: Path) -> list[dict]:
    experiments = []
    with open(tsv_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            sm_raw = row.get("sm_util_pct", "")
            try:
                sm = float(sm_raw)
            except (ValueError, TypeError):
                sm = 0.0

            dur_raw = row.get("duration_s", "")
            try:
                duration = float(dur_raw)
            except (ValueError, TypeError):
                duration = None

            status = row.get("status", "")
            if status == "failed" or sm <= 0:
                exp_status = "CRASH"
            else:
                exp_status = "VALID"

            name = row.get("name", row.get("run_id", ""))

            run_id = row.get("run_id", "")
            is_headspace = run_id == "000h" or name == "headspace_cache"

            step_raw = row.get("step_time_ms", "")
            try:
                step_time = float(step_raw)
            except (ValueError, TypeError):
                step_time = None

            steady_step_raw = row.get("steady_step_time_ms", "")
            try:
                steady_step = float(steady_step_raw)
            except (ValueError, TypeError):
                steady_step = None

            steady_sm_raw = row.get("steady_sm_util_pct", "")
            try:
                steady_sm = float(steady_sm_raw)
            except (ValueError, TypeError):
                steady_sm = None

            experiments.append(
                {
                    "run_id": run_id,
                    "name": name,
                    "sm_util": sm,
                    "steady_sm_util": steady_sm,
                    "step_time_ms": step_time,
                    "steady_step_time_ms": steady_step,
                    "duration_s": duration,
                    "status": "HEADSPACE" if is_headspace else exp_status,
                    "description": _unescape(name).replace("_", " "),
                    "change_summary": _unescape(row.get("change_summary", "")),
                }
            )
    return experiments


def _format_label(exp: dict) -> str:
    name = exp["description"]
    parts = []
    dur = exp.get("duration_s")
    sm = exp.get("sm_util")
    if dur:
        parts.append(f"{dur / 60:.1f}m")
    if sm and sm > 0:
        parts.append(f"SM {sm:.0f}%")
    suffix = f" ({', '.join(parts)})" if parts else ""
    label = name + suffix
    if len(label) > 55:
        label = name[:35] + "..." + suffix
    return label


def _mark_kept(experiments: list[dict]) -> None:
    """Mark experiments that set a new best (lowest) duration.
    Headspace runs are excluded — they are not real training runs."""
    best_dur = float("inf")
    for exp in experiments:
        if exp["status"] not in ("VALID",):
            exp["kept"] = False
            continue
        dur = exp.get("duration_s")
        if dur is not None and dur < best_dur:
            exp["kept"] = True
            best_dur = dur
        else:
            exp["kept"] = False


def _running_best(values: list[float], lower_is_better: bool) -> list[float]:
    result = []
    best = float("inf") if lower_is_better else 0.0
    cmp = min if lower_is_better else max
    for y in values:
        best = cmp(best, y)
        result.append(best)
    return result


def _plot_kept_with_line(
    ax: plt.Axes,
    kept: list[dict],
    y_key: str,
    lower_is_better: bool,
    last_idx: int,
    show_annotations: bool,
) -> None:
    kept_x = [e["idx"] for e in kept]
    kept_y = [e[y_key] for e in kept]
    ax.scatter(
        kept_x,
        kept_y,
        c="#2ecc71",
        s=50,
        zorder=4,
        label="Kept",
        edgecolors="black",
        linewidths=0.5,
    )
    running = _running_best(kept_y, lower_is_better)
    ax.step(
        kept_x + [last_idx],
        running + [running[-1]],
        where="post",
        color="#27ae60",
        linewidth=2,
        alpha=0.7,
        zorder=3,
        label="Running best",
    )
    if show_annotations:
        rotation = 30 if lower_is_better else -30
        xytext = (6, 6) if lower_is_better else (6, -6)
        va = "bottom" if lower_is_better else "top"
        for exp in kept:
            ax.annotate(
                _format_label(exp),
                (exp["idx"], exp[y_key]),
                textcoords="offset points",
                xytext=xytext,
                fontsize=7.5,
                color="#1a7a3a",
                alpha=0.9,
                rotation=rotation,
                ha="left",
                va=va,
            )


def _set_y_limits(
    ax: plt.Axes,
    values: list[float],
    lower_is_better: bool,
    has_crashes: bool,
) -> None:
    if not values:
        return
    lo, hi = min(values), max(values)
    margin = max((hi - lo) * 0.15, 1)
    if lower_is_better:
        ax.set_ylim(lo - margin, hi + margin)
    else:
        y_lo = -5 if has_crashes else max(0, lo - margin)
        ax.set_ylim(y_lo, hi + margin)


def _plot_headspace_line(ax: plt.Axes, experiments: list[dict], y_key: str) -> None:
    """Draw a horizontal dashed line at the headspace (CacheDataLoader) value."""
    for exp in experiments:
        if exp["status"] == "HEADSPACE" and exp.get(y_key):
            ax.axhline(
                y=exp[y_key],
                color="#3498db",
                linestyle="--",
                linewidth=1.5,
                alpha=0.7,
                zorder=1,
                label=f"Headspace floor ({exp[y_key]:.0f}s)",
            )
            break


def _plot_crashes(ax, crashes, valid, y_key):
    crash_y = max(e[y_key] for e in valid) * 1.1 if valid else 0
    ax.scatter(
        [e["idx"] for e in crashes],
        [crash_y] * len(crashes),
        c="#ff6b6b",
        s=30,
        alpha=0.6,
        zorder=2,
        label="Failed",
        marker="x",
        linewidths=1.5,
    )


def _plot_discarded(ax, discarded, y_key):
    ax.scatter(
        [e["idx"] for e in discarded],
        [e[y_key] for e in discarded],
        c="#cccccc",
        s=12,
        alpha=0.5,
        zorder=2,
        label="Discarded",
    )


def _partition_experiments(experiments, y_key):
    """Split experiments into valid, crashes, kept, discarded by status."""
    valid = [e for e in experiments if e["status"] == "VALID" and e.get(y_key)]
    crashes = [e for e in experiments if e["status"] == "CRASH"]
    kept = [e for e in experiments if e.get("kept") and e.get(y_key)]
    discarded = [e for e in valid if not e.get("kept")]
    return valid, crashes, kept, discarded


def _collect_y_values(experiments, valid, y_key):
    """Gather all y values including headspace for axis limits."""
    all_y = [e[y_key] for e in valid]
    for e in experiments:
        if e["status"] == "HEADSPACE" and e.get(y_key):
            all_y.append(e[y_key])
            break
    return all_y


def _plot_metric(
    ax: plt.Axes,
    experiments: list[dict],
    y_key: str,
    ylabel: str,
    lower_is_better: bool,
    show_annotations: bool,
    show_legend: bool,
) -> None:
    valid, crashes, kept, discarded = _partition_experiments(experiments, y_key)

    if crashes:
        _plot_crashes(ax, crashes, valid, y_key)
    if discarded:
        _plot_discarded(ax, discarded, y_key)
    if kept:
        _plot_kept_with_line(
            ax,
            kept,
            y_key,
            lower_is_better,
            experiments[-1]["idx"],
            show_annotations,
        )
    if lower_is_better:
        _plot_headspace_line(ax, experiments, y_key)

    ax.set_ylabel(ylabel, fontsize=12)
    if show_legend:
        loc = "upper right" if lower_is_better else "lower right"
        ax.legend(loc=loc, fontsize=9)
    ax.grid(True, alpha=0.2)

    all_y = _collect_y_values(experiments, valid, y_key)
    _set_y_limits(ax, all_y, lower_is_better, bool(crashes))


def _plot_progress(
    experiments: list[dict],
    output_path: str,
    title_prefix: str = "",
) -> None:
    for i, exp in enumerate(experiments):
        exp["idx"] = i
    _mark_kept(experiments)

    n_total = len(experiments)
    n_kept = sum(1 for e in experiments if e.get("kept"))
    has_duration = any(
        e["duration_s"] is not None and e["status"] == "VALID" for e in experiments
    )

    has_step_time = any(
        e.get("step_time_ms") is not None and e["status"] == "VALID"
        for e in experiments
    )
    has_steady_step = any(
        e.get("steady_step_time_ms") is not None and e["status"] == "VALID"
        for e in experiments
    )
    has_steady_sm = any(
        e.get("steady_sm_util") is not None and e["status"] == "VALID"
        for e in experiments
    )

    # Determine subplot count
    plots = []
    if has_duration:
        plots.append(("duration_s", "Duration (seconds, lower is better)", True))
    if has_steady_step:
        plots.append(
            ("steady_step_time_ms", "Steady Step Time (ms, lower is better)", True)
        )
    elif has_step_time:
        plots.append(("step_time_ms", "Step Time (ms, lower is better)", True))
    if has_steady_sm:
        plots.append(("steady_sm_util", "Steady-State SM Utilization %", False))
    plots.append(("sm_util", "SM Utilization % (raw avg)", False))

    n_plots = len(plots)
    fig, axes = plt.subplots(n_plots, 1, figsize=(16, 5 * n_plots), sharex=True)
    if n_plots == 1:
        axes = [axes]

    for i, (y_key, ylabel, lower) in enumerate(plots):
        _plot_metric(
            axes[i],
            experiments,
            y_key=y_key,
            ylabel=ylabel,
            lower_is_better=lower,
            show_annotations=(i == 0),
            show_legend=(i == 0),
        )
    axes[-1].set_xlabel("Experiment #", fontsize=12)

    fig.suptitle(
        f"{title_prefix}Autoresearch Progress: {n_total} Experiments, "
        f"{n_kept} Kept Improvements",
        fontsize=14,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {output_path}")


_STATUS_COLORS = {
    "completed": "#2ecc71",
    "failed": "#e74c3c",
    "running": "#3498db",
    "analyzing": "#3498db",
    "preparing": "#f39c12",
    "queued": "#ecf0f1",
}


def _tree_bfs_order(nodes: dict[str, dict]) -> tuple[list[str], dict[str, int]]:
    """BFS traversal. Returns (ordered node IDs, {node_id: level})."""
    roots = [
        nid
        for nid, n in nodes.items()
        if not n.get("parent_id") or n["parent_id"] not in nodes
    ]
    if not roots:
        roots = [next(iter(nodes))]

    levels: dict[str, int] = {}
    order: list[str] = []
    queue = [(r, 0) for r in roots]
    while queue:
        nid, level = queue.pop(0)
        if nid in levels:
            continue
        levels[nid] = level
        order.append(nid)
        for child_id in nodes[nid].get("children", []):
            if child_id in nodes:
                queue.append((child_id, level + 1))
    return order, levels


def _compute_positions(
    order: list[str], levels: dict[str, int]
) -> tuple[dict[str, float], dict[str, float], dict[int, int]]:
    """Compute centered x/y positions from BFS order. Returns (x_pos, y_pos, level_counts)."""
    level_counts: dict[int, int] = {}
    x_pos: dict[str, float] = {}
    for nid in order:
        lv = levels[nid]
        idx = level_counts.get(lv, 0)
        level_counts[lv] = idx + 1
        x_pos[nid] = idx

    max_level = max(levels.values()) if levels else 0
    for lv in range(max_level + 1):
        count = level_counts.get(lv, 1)
        nids_at_level = [n for n in order if levels[n] == lv]
        for i, nid in enumerate(nids_at_level):
            x_pos[nid] = (i - (count - 1) / 2.0) * 3.0

    y_pos = {nid: -levels[nid] * 2.5 for nid in order}
    return x_pos, y_pos, level_counts


def _find_best_path(nodes: dict[str, dict]) -> tuple[str | None, set[tuple[str, str]]]:
    """Find the best node and the edge set from root to it."""
    best_nid = None
    best_dur = float("inf")
    for nid, n in nodes.items():
        dur = n.get("duration")
        if dur is not None and 0 < dur < best_dur:
            best_dur = dur
            best_nid = nid

    edges: set[tuple[str, str]] = set()
    cur = best_nid
    while cur and cur in nodes:
        parent = nodes[cur].get("parent_id")
        if parent and parent in nodes:
            edges.add((parent, cur))
        cur = parent
    return best_nid, edges


def _shorten(text: str, limit: int) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _edge_label(parent_node: dict, child_node: dict) -> str:
    spec = child_node.get("spec") or {}
    value = spec.get("change_summary")
    if value:
        return _short_change_label(str(value))
    retry_attempt = spec.get("_startup_retry_attempt")
    if retry_attempt:
        return f"startup repair #{retry_attempt}"
    tags = spec.get("best_practices_tags")
    if isinstance(tags, list) and tags:
        return _short_change_label(str(tags[0]))
    return _short_change_label(
        str(child_node.get("name", child_node.get("node_id", "")))
    )


def _short_change_label(text: str) -> str:
    words = [word.strip(".,;:()[]{}") for word in text.replace("_", " ").split()]
    words = [word for word in words if word]
    if not words:
        return "experiment"
    label = " ".join(words[:5])
    return _shorten(label, 34)


def _tree_font_sizes(node_count: int, max_level: int) -> dict[str, float]:
    scale = min(2.0, max(1.0, (node_count / 24) ** 0.35, (max_level + 1) / 6))
    return {
        "node": 8 * scale,
        "edge": 6.5 * scale,
        "legend": 9 * scale,
        "title": 14 * scale,
    }


def _draw_edge_label(ax, label, x1, y1, x2, y2, fontsize):
    ax.text(
        (x1 + x2) / 2.0,
        (y1 + y2) / 2.0,
        label,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#586069",
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": "white",
            "edgecolor": "#d0d7de",
            "alpha": 0.85,
            "linewidth": 0.4,
        },
        zorder=4,
    )


def _draw_tree_edges(
    ax, nodes, x_pos, y_pos, best_path_edges, show_edge_labels, edge_fontsize
):
    for nid, n in nodes.items():
        for child_id in n.get("children", []):
            if child_id not in x_pos:
                continue
            is_best = (nid, child_id) in best_path_edges
            ax.annotate(
                "",
                xy=(x_pos[child_id], y_pos[child_id] + 0.6),
                xytext=(x_pos[nid], y_pos[nid] - 0.6),
                arrowprops={
                    "arrowstyle": "->",
                    "color": "#27ae60" if is_best else "#bdc3c7",
                    "linewidth": 3 if is_best else 1.5,
                    "connectionstyle": "arc3,rad=0.1",
                },
            )
            if show_edge_labels:
                _draw_edge_label(
                    ax,
                    _edge_label(n, nodes[child_id]),
                    x_pos[nid],
                    y_pos[nid] - 0.6,
                    x_pos[child_id],
                    y_pos[child_id] + 0.6,
                    edge_fontsize,
                )


def _draw_tree_node(ax, nid, n, x, y, best_nid, fontsize):
    box_w, box_h = 2.4, 1.0
    status = n.get("status", "queued")
    color = "#27ae60" if nid == best_nid else _STATUS_COLORS.get(status, "#ecf0f1")

    ax.add_patch(
        plt.Rectangle(
            (x - box_w / 2, y - box_h / 2),
            box_w,
            box_h,
            facecolor=color,
            edgecolor="black" if status != "queued" else "#bdc3c7",
            linewidth=1.5 if nid == best_nid else 0.8,
            linestyle="-" if status != "queued" else "--",
            alpha=0.85,
            zorder=5,
        )
    )

    name = n.get("name", nid)
    if len(name) > 20:
        name = name[:17] + "..."
    dur = n.get("duration")
    metric_str = f"\n{dur:.0f}s" if dur and dur > 0 else ""
    text_color = (
        "white" if status in ("completed", "failed") or nid == best_nid else "black"
    )
    ax.text(
        x,
        y,
        f"{name}{metric_str}",
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold" if nid == best_nid else "normal",
        color=text_color,
        zorder=6,
    )


def _plot_hypothesis_tree(
    tree_data: list[dict],
    output_path: str,
    title: str = "",
) -> None:
    """Render the hypothesis tree as a top-down directed graph."""
    if not tree_data:
        return

    nodes = {n["node_id"]: n for n in tree_data}
    order, levels = _tree_bfs_order(nodes)
    x_pos, y_pos, level_counts = _compute_positions(order, levels)
    if not order:
        return

    best_nid, best_path_edges = _find_best_path(nodes)
    max_level = max(levels.values()) if levels else 0
    max_nodes_per_level = max(level_counts.values()) if level_counts else 1
    fig_w = max(12, max_nodes_per_level * 3.5)
    fig_h = max(8, (max_level + 1) * 3)
    fonts = _tree_font_sizes(len(order), max_level)
    dpi = 150 if len(order) < 50 else 200 if len(order) < 200 else 300
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    _draw_tree_edges(
        ax,
        nodes,
        x_pos,
        y_pos,
        best_path_edges,
        show_edge_labels=len(order) <= 80,
        edge_fontsize=fonts["edge"],
    )
    for nid in order:
        _draw_tree_node(
            ax,
            nid,
            nodes[nid],
            x_pos[nid],
            y_pos[nid],
            best_nid,
            fonts["node"],
        )

    legend_items = [
        plt.Rectangle((0, 0), 1, 1, fc="#2ecc71", ec="black", label="Completed"),
        plt.Rectangle((0, 0), 1, 1, fc="#e74c3c", ec="black", label="Failed"),
        plt.Rectangle((0, 0), 1, 1, fc="#3498db", ec="black", label="Running"),
        plt.Rectangle(
            (0, 0), 1, 1, fc="#ecf0f1", ec="#bdc3c7", ls="--", label="Queued"
        ),
        plt.Rectangle((0, 0), 1, 1, fc="#27ae60", ec="black", label="Best"),
    ]
    ax.legend(handles=legend_items, loc="upper right", fontsize=fonts["legend"])

    ax.set_xlim(min(x_pos.values()) - 2, max(x_pos.values()) + 2)
    ax.set_ylim(min(y_pos.values()) - 1.5, max(y_pos.values()) + 1.5)
    ax.set_aspect("equal")
    ax.axis("off")

    title_text = f"{title}Hypothesis Tree" if title else "Hypothesis Tree"
    n_completed = sum(1 for n in nodes.values() if n.get("status") == "completed")
    ax.set_title(
        f"{title_text}: {len(nodes)} nodes, {n_completed} completed",
        fontsize=fonts["title"],
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Hypothesis tree saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tsv", help="Path to master_table.tsv")
    parser.add_argument(
        "output",
        nargs="?",
        default=None,
        help="Output PNG path (default: progress.png next to the TSV)",
    )
    parser.add_argument("--title", default="", help="Title prefix")
    args = parser.parse_args()

    tsv_path = Path(args.tsv).resolve()
    if not tsv_path.exists():
        print(f"Error: {tsv_path} not found", file=sys.stderr)
        sys.exit(1)

    output = args.output or str(tsv_path.parent / "progress.png")
    experiments = _load_tsv(tsv_path)
    _plot_progress(experiments, output, args.title)


if __name__ == "__main__":
    main()
