# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Generate the animation for ``docs/source/migration/paradigm_shift.rst``.

The animation compares a conventional data loader (top) with an SPDL
pipeline (bottom).

- In the conventional data loader, each worker process receives the samples
  of a whole batch, runs every step on one sample at a time, then collates
  the samples into a batch. At most one sample per worker is processed at any
  time, so the concurrency of every step equals the number of workers.
- In the SPDL pipeline, each stage processes one sample at a time per task,
  with the concurrency configured stage by stage, and the samples are
  aggregated into a batch at the end.

In both, the source only hands out the next sample, which costs nothing.
The GPU boxes of both panels are aligned and flash when a batch arrives, so
the batch rates can be compared directly.

The top panel is periodic by construction. The bottom panel is a small
time-stepped simulation: each stage has a number of slots and a bounded
queue in front of it, a finished sample moves downstream when there is room,
and a free slot picks up the next sample from its queue.

Run with::

    buck2 run fbcode//spdl/docs/scripts:generate_paradigm_shift_animation -- \\
        "$(pwd)/fbcode/spdl/docs/source/_static/data/paradigm_shift.gif"
"""

from __future__ import annotations

import argparse
import io
import logging
import math
from dataclasses import dataclass, field

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch  # noqa: E402
from PIL import Image  # noqa: E402

logger: logging.Logger = logging.getLogger(__name__)

Point = tuple[float, float]

# Canvas and timing
WIDTH, HEIGHT = 1400, 1080
OUT_WIDTH, OUT_HEIGHT = 1000, 771
FRAME_MS = 60  # GIF frame delays are stored in 10 ms units
SEC_PER_UNIT = 0.75  # Wall-clock seconds per simulated time unit
LOOP_UNITS = 12.0  # Loop length; a multiple of both panels' periods
WARMUP_UNITS = 36.0  # Simulated time before the loop starts (steady state)
STEP = 0.01  # Simulation step
MOVE = 0.25  # Time a sample spends travelling between boxes
FLASH = 0.5  # Time the GPU box stays highlighted after a batch arrives
NUM_COLORS = 96

# Style (matches the feedback-loop figures in the overview)
INK = "#444444"
IDLE = "#c8c8c8"
TEXT = "#333333"
RAW = "#9a9a9a"
FONT = {"family": "DejaVu Sans", "color": TEXT}

# Pipeline definition, shared by both panels
STAGES = ("src", "s1", "s2")
NAMES = {"src": "Source", "s1": "Stage 1", "s2": "Stage 2"}
# Per sample. The source only hands out the next sample, so it costs nothing.
DURATION = {"src": 0.0, "s1": 2.0, "s2": 1.0}
FILL = {"src": "#a1c4e8", "s1": "#78a6b0", "s2": "#f9ca9c"}
DONE = {"src": RAW, "s1": "#3f7f8c", "s2": "#e8964a"}  # Sample color after a stage
BATCH_SIZE = 4
SAMPLE_TIME = sum(DURATION.values())

# Shared horizontal layout
X = {"src": 230, "s1": 450, "s2": 670, "batch": 880}
GPU_X = 1180
TRAY = (70, 70)

# Top panel (conventional data loader)
NUM_WORKERS = 4
T_ROWS = (160, 278, 396, 514)
T_BOX = (110, 54)
T_GPU = (GPU_X, 337)

# Bottom panel (SPDL)
B_CONCURRENCY = {"src": 1, "s1": 6, "s2": 3}
B_Y = 885
B_GAP = 47
B_BOX = (104, 36)
B_GPU = (GPU_X, B_Y)
QUEUE_SIZE = 2


# --------------------------------------------------------------------- utils
def _lerp(a: Point, b: Point, f: float) -> Point:
    return (a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f)


def _ease(f: float) -> float:
    f = min(max(f, 0.0), 1.0)
    return 0.5 - 0.5 * math.cos(math.pi * f)


def _route(a: Point, b: Point, via: Point | None, f: float) -> Point:
    """Position along ``a -> via -> b`` at eased fraction ``f``."""
    f = _ease(f)
    if via is None:
        return _lerp(a, b, f)
    d1, d2 = math.dist(a, via), math.dist(via, b)
    k = f * (d1 + d2)
    if k <= d1:
        return _lerp(a, via, k / d1 if d1 else 1.0)
    return _lerp(via, b, (k - d1) / d2 if d2 else 1.0)


def _box(ax: Axes, center: Point, size: tuple[float, float], fill: str | None) -> None:
    (cx, cy), (w, h) = center, size
    ax.add_patch(
        FancyBboxPatch(
            (cx - w / 2, cy - h / 2),
            w,
            h,
            boxstyle="round,pad=0,rounding_size=9",
            fc=(fill + "55") if fill else "white",
            ec=fill or IDLE,
            lw=2.0,
        )
    )


def _frame(
    ax: Axes, x0: float, y0: float, w: float, h: float, fill: str = "none"
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x0, y0),
            w,
            h,
            boxstyle="round,pad=0,rounding_size=14",
            fc=fill,
            ec=INK,
            lw=1.4,
            ls="--",
        )
    )


def _arrow(ax: Axes, p0: Point, p1: Point, color: str = INK) -> None:
    ax.add_patch(
        FancyArrowPatch(
            p0, p1, arrowstyle="-|>", mutation_scale=13, color=color, lw=1.5
        )
    )


def _dot(ax: Axes, center: Point, color: str) -> None:
    ax.add_patch(Circle(center, 7.0, fc=color, ec="white", lw=1.0, zorder=5))


def _tray_slot(center: Point, k: int) -> Point:
    dx = -11 if k % 2 == 0 else 11
    dy = -11 if k < 2 else 11
    return (center[0] + dx, center[1] + dy)


def _batch(ax: Axes, center: Point, color: str) -> None:
    for k in range(BATCH_SIZE):
        _dot(ax, _tray_slot(center, k), color)


def _tray(ax: Axes, center: Point, label_y: float) -> None:
    _box(ax, center, TRAY, None)
    ax.text(center[0], label_y, "Batch", ha="center", va="center", fontsize=13, **FONT)
    for k in range(BATCH_SIZE):
        slot = Circle(_tray_slot(center, k), 7.0, fc="none", ec=IDLE, lw=1.0, ls=":")
        ax.add_patch(slot)


def _gpu(ax: Axes, center: Point, height: float, flash: float) -> None:
    """Draw the GPU box; ``flash`` (0 to 1) highlights a batch arrival."""
    alpha = int(0x66 * max(0.0, 1.0 - flash)) if flash < 1.0 else 0
    fill = f"{DONE['s2']}{alpha:02x}" if alpha else "none"
    _frame(ax, center[0] - 45, center[1] - height / 2, 90, height, fill)
    label_y = center[1] - height / 2 + 26
    ax.text(center[0], label_y, "GPU", ha="center", va="center", fontsize=16, **FONT)


def _caption(ax: Axes, y: float, title: str, subtitle: str) -> None:
    ax.text(40, y, title, ha="left", va="center", fontsize=22, weight="bold", **FONT)
    ax.text(40, y + 36, subtitle, ha="left", va="center", fontsize=14, **FONT)


# --------------------------------------------- top: conventional data loader
@dataclass
class _WorkerState:
    stage: str  # Stage the current sample is in
    elapsed: float  # Time spent in that stage
    collated: int  # Samples already in the batch tray
    cycle: float  # Time since the worker started the current batch


def _worker_state(t: float, worker: int) -> _WorkerState:
    period = BATCH_SIZE * SAMPLE_TIME
    cycle = (t + worker * period / NUM_WORKERS) % period
    sample, within = divmod(cycle, SAMPLE_TIME)
    start = 0.0
    for stage in STAGES:
        if DURATION[stage] > 0 and within < start + DURATION[stage]:
            return _WorkerState(stage, within - start, int(sample), cycle)
        start += DURATION[stage]
    return _WorkerState(STAGES[-1], DURATION[STAGES[-1]], int(sample), cycle)


def _in_box(x: float, y: float, f: float) -> Point:
    inset = T_BOX[0] / 2 - 18
    return _lerp((x - inset, y), (x + inset, y), f)


def _top_sample(state: _WorkerState, y: float) -> tuple[Point, str]:
    idx = STAGES.index(state.stage)
    color = DONE[STAGES[idx - 1]]
    if state.elapsed < MOVE:
        prev = _in_box(X[STAGES[idx - 1]], y, 1.0)
        f = _ease(state.elapsed / MOVE)
        return _lerp(prev, _in_box(X[state.stage], y, 0.0), f), color
    f = (state.elapsed - MOVE) / (DURATION[state.stage] - MOVE)
    return _in_box(X[state.stage], y, f), color


def _draw_worker(ax: Axes, worker: int, y: float, t: float) -> None:
    _frame(ax, X["src"] - 90, y - 52, X["batch"] - X["src"] + 160, 104)
    label = f" Worker process {worker + 1} "
    ax.text(
        X["src"] - 72,
        y - 52,
        label,
        ha="left",
        va="center",
        fontsize=13,
        bbox={"fc": "white", "ec": "none", "pad": 1},
        **FONT,
    )
    state = _worker_state(t, worker)
    stops = [X[s] - T_BOX[0] / 2 for s in STAGES[1:]] + [X["batch"] - TRAY[0] / 2]
    for s, stop in zip(STAGES, stops):
        _box(ax, (X[s], y + 6), T_BOX, FILL[s] if s == state.stage else None)
        ax.text(X[s], y - 31, NAMES[s], ha="center", va="center", fontsize=13, **FONT)
        _arrow(ax, (X[s] + T_BOX[0] / 2 + 3, y + 6), (stop - 3, y + 6))
    tray = (X["batch"], y + 6)
    _tray(ax, tray, y - 31)
    gpu_in = (T_GPU[0] - 48, T_GPU[1] + (y - T_GPU[1]) * 0.3)
    _arrow(ax, (X["batch"] + TRAY[0] / 2 + 3, y + 6), gpu_in, IDLE)
    for k in range(state.collated):
        _dot(ax, _tray_slot(tray, k), DONE["s2"])
    for k in range(BATCH_SIZE - state.collated - 1):  # Samples not started yet
        _dot(ax, _in_box(X["src"], y + 6, 1.0 - k * 0.33), RAW)
    pos, color = _top_sample(state, y + 6)
    _dot(ax, pos, color)
    if state.cycle < 2 * MOVE:  # The previous batch is sent to the GPU
        _batch(ax, _lerp(tray, T_GPU, _ease(state.cycle / (2 * MOVE))), DONE["s2"])


def _top_flash(t: float) -> float:
    """Time since the latest batch arrived at the GPU, in units of ``FLASH``."""
    since = min(
        _worker_state(t, w).cycle - 2 * MOVE
        for w in range(NUM_WORKERS)
        if _worker_state(t, w).cycle >= 2 * MOVE
    )
    return since / FLASH


def _draw_top(ax: Axes, t: float) -> None:
    _caption(
        ax,
        40,
        "Conventional data loader",
        "Each worker runs every step on one sample at a time, then batches",
    )
    for worker, y in enumerate(T_ROWS):
        _draw_worker(ax, worker, y, t)
    _gpu(ax, T_GPU, 450, _top_flash(t))


# ---------------------------------------------------------- bottom: SPDL
def _slot_y(stage: str, i: int) -> float:
    return B_Y + (i - (B_CONCURRENCY[stage] - 1) / 2) * B_GAP


def _slot_point(stage: str, i: int, f: float) -> Point:
    inset = B_BOX[0] / 2 - 18
    y = _slot_y(stage, i)
    return _lerp((X[stage] - inset, y), (X[stage] + inset, y), f)


def _entry(stage: str) -> Point:
    """Where samples enter a stage's column (the end of the incoming arrow)."""
    return (X[stage] - B_BOX[0] / 2 - 14, B_Y)


def _exit(stage: str) -> Point:
    return (X[stage] + B_BOX[0] / 2 + 14, B_Y)


def _queue_position(stage: str, k: int) -> Point:
    (xa, _), (xb, _) = _exit(STAGES[STAGES.index(stage) - 1]), _entry(stage)
    return ((xa + xb) / 2 + 9 - k * 18, B_Y)


B_TRAY = (X["batch"], B_Y)


@dataclass
class _Sample:
    color: str
    pos: Point
    prev: Point
    via: Point | None = None
    t_move: float = -1e9

    def move(self, pos: Point, t: float, via: Point | None = None) -> None:
        self.prev = self.at(t)
        self.pos, self.via, self.t_move = pos, via, t

    def at(self, t: float, target: Point | None = None) -> Point:
        return _route(self.prev, target or self.pos, self.via, (t - self.t_move) / MOVE)


@dataclass
class _Slot:
    sample: _Sample
    start: float


@dataclass
class _Pipeline:
    slots: dict[str, list[_Slot | None]] = field(
        default_factory=lambda: {s: [None] * B_CONCURRENCY[s] for s in STAGES}
    )
    queues: dict[str, list[_Sample]] = field(
        default_factory=lambda: {"s1": [], "s2": []}
    )
    tray: list[_Sample] = field(default_factory=list)
    flights: list[float] = field(default_factory=list)  # Start times
    last_arrival: float = -1e9

    def step(self, t: float) -> None:
        self._finish(t)
        self._pull(t)
        self._generate(t)
        self._ship(t)

    def _finish(self, t: float) -> None:
        for idx, stage in enumerate(STAGES):
            for i, slot in enumerate(self.slots[stage]):
                if slot is None or t - slot.start < DURATION[stage] - 1e-9:
                    continue
                if stage == STAGES[-1]:
                    if len(self.tray) >= BATCH_SIZE:
                        continue  # The batch tray is full (backpressure)
                    target = _tray_slot(B_TRAY, len(self.tray))
                    self.tray.append(slot.sample)
                else:
                    queue = self.queues[STAGES[idx + 1]]
                    if len(queue) >= QUEUE_SIZE:
                        continue  # The queue is full (backpressure)
                    target = _queue_position(STAGES[idx + 1], len(queue))
                    queue.append(slot.sample)
                slot.sample.color = DONE[stage]
                slot.sample.prev = _slot_point(stage, i, 1.0)
                slot.sample.move(target, t, via=_exit(stage))
                self.slots[stage][i] = None

    def _pull(self, t: float) -> None:
        for stage, queue in self.queues.items():
            for i, slot in enumerate(self.slots[stage]):
                if slot is None and queue:
                    sample = queue.pop(0)
                    sample.move(_slot_point(stage, i, 0.0), t, via=_entry(stage))
                    self.slots[stage][i] = _Slot(sample, t)
            for k, sample in enumerate(queue):
                if sample.pos != _queue_position(stage, k):
                    sample.move(_queue_position(stage, k), t)

    def _generate(self, t: float) -> None:
        for i, slot in enumerate(self.slots["src"]):
            if slot is None:
                start = _slot_point("src", i, 1.0)
                self.slots["src"][i] = _Slot(_Sample(RAW, start, start), t)

    def _ship(self, t: float) -> None:
        arrived = all(t - s.t_move >= MOVE for s in self.tray)
        if len(self.tray) == BATCH_SIZE and arrived:
            self.tray = []
            self.flights.append(t)
        for start in self.flights:
            if t - start >= 2 * MOVE:
                self.last_arrival = start + 2 * MOVE
        self.flights = [s for s in self.flights if t - s < 2 * MOVE]

    def samples(self, t: float) -> list[tuple[Point, str]]:
        out = []
        for stage in STAGES:
            for i, slot in enumerate(self.slots[stage]):
                if slot is None:
                    continue
                dur = DURATION[stage]
                f = min((t - slot.start) / dur, 1.0) if dur else 1.0
                pos = _slot_point(stage, i, f)
                if t - slot.sample.t_move < MOVE:
                    pos = slot.sample.at(t, target=pos)
                out.append((pos, DONE[stage] if f >= 1.0 else slot.sample.color))
        for sample in [*self.queues["s1"], *self.queues["s2"], *self.tray]:
            out.append((sample.at(t), sample.color))
        return out


def _draw_stage_column(ax: Axes, pipeline: _Pipeline, stage: str) -> None:
    n = B_CONCURRENCY[stage]
    top = _slot_y(stage, 0) - B_BOX[1] / 2 - 36
    bottom = _slot_y(stage, n - 1) + B_BOX[1] / 2 + 12
    _frame(ax, X[stage] - B_BOX[0] / 2 - 13, top, B_BOX[0] + 26, bottom - top)
    label = f"{NAMES[stage]} ×{n}"
    ax.text(X[stage], top + 18, label, ha="center", va="center", fontsize=13, **FONT)
    for i, slot in enumerate(pipeline.slots[stage]):
        working = slot is not None and DURATION[stage] > 0
        fill = FILL[stage] if working else None
        _box(ax, (X[stage], _slot_y(stage, i)), B_BOX, fill)


def _draw_bottom(ax: Axes, pipeline: _Pipeline, t: float) -> None:
    _caption(
        ax,
        645,
        "SPDL pipeline",
        "Each stage runs one sample per task, with its own concurrency; batching at the end",
    )
    for stage in STAGES:
        _draw_stage_column(ax, pipeline, stage)
    for a, b in zip(STAGES, STAGES[1:]):
        _arrow(ax, _exit(a), _entry(b))
    _arrow(ax, _exit(STAGES[-1]), (B_TRAY[0] - TRAY[0] / 2 - 3, B_Y))
    _tray(ax, B_TRAY, B_Y - TRAY[1] / 2 - 18)
    _arrow(ax, (B_TRAY[0] + TRAY[0] / 2 + 3, B_Y), (B_GPU[0] - 48, B_Y), IDLE)
    _gpu(ax, B_GPU, 340, (t - pipeline.last_arrival) / FLASH)
    for pos, color in pipeline.samples(t):
        _dot(ax, pos, color)
    for start in pipeline.flights:
        f = _ease((t - start) / (2 * MOVE))
        _batch(ax, _lerp(B_TRAY, B_GPU, f), DONE["s2"])


# ----------------------------------------------------------------- output
def _render_frame(t: float, pipeline: _Pipeline) -> Image.Image:
    fig = plt.figure(figsize=(WIDTH / 100, HEIGHT / 100), dpi=100)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(HEIGHT, 0)
    ax.axis("off")
    ax.plot([30, WIDTH - 30], [600, 600], color="#e2e2e2", lw=1.2)
    _draw_top(ax, t)
    _draw_bottom(ax, pipeline, t)
    buf = io.BytesIO()
    fig.savefig(buf, dpi=100, facecolor="white", format="png")
    plt.close(fig)
    buf.seek(0)
    image = Image.open(buf).convert("RGB")
    return image.resize((OUT_WIDTH, OUT_HEIGHT), Image.Resampling.LANCZOS)


def _render_frames() -> list[Image.Image]:
    num_frames = round(LOOP_UNITS * SEC_PER_UNIT * 1000 / FRAME_MS)
    frame_times = [
        WARMUP_UNITS + k * LOOP_UNITS / num_frames for k in range(num_frames)
    ]
    pipeline = _Pipeline()
    frames = []
    t = 0.0
    for frame_time in frame_times:
        while t < frame_time - STEP / 2:
            pipeline.step(t)
            t += STEP
        frames.append(_render_frame(t, pipeline))
    return frames


def _save_gif(frames: list[Image.Image], path: str) -> None:
    # Build one palette from a sample of frames so every color appears in it.
    sample = frames[:: max(1, len(frames) // 8)]
    sheet = Image.new("RGB", (OUT_WIDTH, OUT_HEIGHT * len(sample)))
    for i, frame in enumerate(sample):
        sheet.paste(frame, (0, OUT_HEIGHT * i))
    palette = sheet.quantize(colors=NUM_COLORS, method=Image.Quantize.FASTOCTREE)
    quantized = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    quantized[0].save(
        path,
        save_all=True,
        append_images=quantized[1:],
        duration=FRAME_MS,
        loop=0,
        optimize=True,
        disposal=1,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", help="Path of the GIF file to write.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    frames = _render_frames()
    _save_gif(frames, args.output)
    logger.info("Wrote %d frames to %s", len(frames), args.output)


if __name__ == "__main__":
    main()
