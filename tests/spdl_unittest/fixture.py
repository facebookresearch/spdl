# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from numpy.typing import DTypeLike, NDArray

__all__ = [
    "get_sample",
    "get_samples",
    "SrcInfo",
    "load_ref_video",
    "load_ref_image",
]


def _run_in_tmpdir(cmd: str, tmp_dir: Path) -> None:
    print(f"\n{'-' * 40}", flush=True, file=sys.stderr)
    print(f"- Executing `{cmd}`", flush=True, file=sys.stderr)
    print(f"{'-' * 40}\n", flush=True, file=sys.stderr)
    subprocess.run(cmd, cwd=tmp_dir, shell=True, check=True, capture_output=False)


@dataclass
class SrcInfo:
    path: str
    _tmp_dir: TemporaryDirectory[str]


def get_sample(cmd: str) -> SrcInfo:
    samples = get_samples(cmd)
    assert len(samples) == 1
    return samples[0]


def get_samples(cmd: str) -> list[SrcInfo]:
    tmp_dir = TemporaryDirectory()
    tmp_path = Path(tmp_dir.name)

    _run_in_tmpdir(cmd, tmp_path)
    return [SrcInfo(str(f), tmp_dir) for f in tmp_path.glob("**/*") if f.is_file()]


def load_ref_data(
    cmd: list[str],
    shape: tuple[int, ...],
    *,
    dtype: DTypeLike = np.uint8,
):
    print(f"\n{'-' * 40}", flush=True, file=sys.stderr)
    print(f"- Executing `{' '.join(cmd)}`", flush=True, file=sys.stderr)
    print(f"{'-' * 40}\n", flush=True, file=sys.stderr)
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10**8)
    buffer = process.stdout.read()  # pyre-ignore
    return np.frombuffer(buffer, dtype).reshape(*shape)


def _get_video_ref_cmd(path: str, filter_graph: str | None) -> list[str]:
    # fmt: off
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "debug",
        "-y",
        "-i", path,
        "-v", "verbose"
    ]
    # fmt: on
    if filter_graph is not None:
        command.extend(["-vf", filter_graph])
    command.extend(["-f", "rawvideo", "pipe:"])
    return command


def load_ref_video(
    path: str,
    shape: tuple[int, ...],
    *,
    filter_graph: str | None = "format=pix_fmts=rgb24",
    dtype: DTypeLike = np.uint8,
) -> NDArray[np.uint8]:
    cmd = _get_video_ref_cmd(path, filter_graph)
    return load_ref_data(cmd, shape, dtype=dtype)


def load_ref_image(
    path: str,
    shape: tuple[int, ...],
    *,
    filter_graph: str | None = "format=pix_fmts=rgb24",
    dtype: DTypeLike = np.uint8,
) -> NDArray[np.uint8]:
    return load_ref_video(path, shape, dtype=dtype, filter_graph=filter_graph)
