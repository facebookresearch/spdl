# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Generate illustrations for ``docs/source/io/color_space.rst``.

For each supported pixel format, decode one frame from an ffmpeg ``lavfi``
test source (SMPTE HD bars) and render a side-by-side figure showing the
canonical RGB rendering alongside the raw tensor planes that ``spdl.io``
returns. Output PNGs are written to
``docs/source/_static/data/io_color_space_*.png``.

Run from any working directory::

    python3 fbcode/spdl/docs/scripts/generate_color_space_assets.py
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from tempfile import NamedTemporaryFile

import matplotlib.pyplot as plt
import numpy as np
import spdl.io

PIX_FMTS = [
    "gray8",
    "rgb24",
    "rgba",
    "yuv444p",
    "yuv420p",
    "yuv422p",
    "nv12",
]

WIDTH, HEIGHT = 320, 240

# FFmpeg's built-in lavfi source used to generate the test video. SMPTE HD bars
# give clean primary-colored regions that make YUV / chroma-subsampling
# differences obvious at the bar boundaries.
LAVFI_SOURCE = f"smptehdbars=size={WIDTH}x{HEIGHT}"


def _decode(path: str, pix_fmt: str | None) -> np.ndarray:
    if pix_fmt is None:
        filter_desc = None
    else:
        filter_desc = spdl.io.get_video_filter_desc(pix_fmt=pix_fmt)
    buf = spdl.io.load_video(path, filter_desc=filter_desc)
    return spdl.io.to_numpy(buf)


def _planes_from_buffer(arr: np.ndarray, pix_fmt: str) -> list[tuple[str, np.ndarray]]:
    """Slice the raw buffer into named planes for visualization.

    The returned arrays mirror exactly what ``spdl.io`` puts in memory — no
    upsampling, no color conversion. This is the point: readers should see what
    the bytes look like.
    """
    a = arr[0]
    if pix_fmt == "gray8":
        if a.ndim == 3 and a.shape[-1] == 1:
            a = a[..., 0]
        return [("Y", a)]
    if pix_fmt == "rgb24":
        return [("R", a[..., 0]), ("G", a[..., 1]), ("B", a[..., 2])]
    if pix_fmt == "rgba":
        return [
            ("R", a[..., 0]),
            ("G", a[..., 1]),
            ("B", a[..., 2]),
            ("A", a[..., 3]),
        ]
    if pix_fmt == "yuv444p":
        return [("Y", a[0]), ("U", a[1]), ("V", a[2])]
    if pix_fmt in {"yuv420p", "yuv422p", "nv12"}:
        # Single packed plane of shape (H + H/2, W) — Y on top, chroma below.
        return [("Y + chroma (packed)", a[0])]
    raise ValueError(pix_fmt)


def _shape_label(arr: np.ndarray, pix_fmt: str) -> str:
    return f"shape={tuple(arr.shape)}  dtype={arr.dtype}  pix_fmt={pix_fmt}"


def _save_figure(
    out_path: Path,
    rgb_reference: np.ndarray,
    raw: np.ndarray,
    pix_fmt: str,
) -> None:
    planes = _planes_from_buffer(raw, pix_fmt)
    n = 1 + len(planes)
    fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 3.0))
    axes[0].imshow(rgb_reference)
    axes[0].set_title("RGB reference", fontsize=10)
    axes[0].axis("off")
    for ax, (name, plane) in zip(axes[1:], planes):
        ax.imshow(plane, cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"{name}  {plane.shape}", fontsize=10)
        ax.axis("off")
    fig.suptitle(_shape_label(raw, pix_fmt), fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _save_overview(out_path: Path, rgb_reference: np.ndarray) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    ax.imshow(rgb_reference)
    ax.set_title("Test pattern used for color-space illustrations", fontsize=10)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    docs_root = Path(__file__).resolve().parent.parent
    out_dir = docs_root / "source" / "_static" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    with NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        sample_path = f.name
    try:
        # Use ffmpeg's built-in lavfi source to generate a 1-frame H.264 mp4
        # encoded as yuv420p (the codec's native pix_fmt). Using a stock
        # lavfi pattern keeps this script free of any hand-coded test image.
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                LAVFI_SOURCE,
                "-frames:v",
                "1",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                sample_path,
            ],
            check=True,
        )

        rgb_ref = _decode(sample_path, "rgb24")[0]
        _save_overview(out_dir / "io_color_space_overview.png", rgb_ref)

        for pix_fmt in PIX_FMTS:
            raw = _decode(sample_path, pix_fmt)
            out = out_dir / f"io_color_space_{pix_fmt}.png"
            _save_figure(out, rgb_ref, raw, pix_fmt)
            print(f"wrote {out}  raw.shape={raw.shape}")
    finally:
        os.unlink(sample_path)


if __name__ == "__main__":
    main()
