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

FFMPEG_CLI: str = "ffmpeg"


__all__ = [
    "get_sample",
    "get_samples",
    "SrcInfo",
    "load_ref_video",
    "load_ref_image",
    "FFMPEG_CLI",
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
) -> NDArray[np.uint8]:
    print(f"\n{'-' * 40}", flush=True, file=sys.stderr)
    print(f"- Executing `{' '.join(cmd)}`", flush=True, file=sys.stderr)
    print(f"{'-' * 40}\n", flush=True, file=sys.stderr)
    process = subprocess.run(cmd, stdout=subprocess.PIPE, bufsize=10**8, check=True)
    return np.frombuffer(process.stdout, dtype).reshape(*shape)


def load_ref_audio(
    path: str,
    shape: tuple[int, ...],
    *,
    filter_desc: str | None = "aformat=sample_fmts=fltp",
    format: str = "f32le",
    dtype: DTypeLike = np.float32,
) -> NDArray[np.uint8]:
    # fmt: off
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "debug",
        "-y",
        "-i", path,
        "-v", "verbose",
    ]
    if filter_desc:
        cmd.extend(["-af", filter_desc])
    # fmt: on
    cmd.extend(["-f", format, "pipe:"])
    return load_ref_data(cmd, shape, dtype=dtype)


def _get_video_ref_cmd(
    path: str,
    filter_desc: str | None,
    filter_complex: str | None = None,
    raw: dict[str, str] | None = None,
) -> list[str]:
    # fmt: off
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "debug",
        "-v", "verbose",
        "-y",
    ]
    # fmt: on
    if raw is not None:
        command.extend(
            [
                "-f",
                "rawvideo",
                "-pixel_format",
                raw["pix_fmt"],
                "-video_size",
                f"{raw['width']}x{raw['height']}",
            ]
        )

    command.extend(["-i", path])

    if filter_desc is not None:
        command.extend(["-vf", filter_desc])
    if filter_complex is not None:
        command.extend(["-filter_complex", filter_complex])
    command.extend(["-f", "rawvideo", "pipe:"])
    return command


def load_ref_video(
    path: str,
    shape: tuple[int, ...],
    *,
    filter_desc: str | None = "format=pix_fmts=rgb24",
    filter_complex: str | None = None,
    dtype: DTypeLike = np.uint8,
    raw: dict[str, str] | None = None,
) -> NDArray[np.uint8]:
    cmd = _get_video_ref_cmd(path, filter_desc, filter_complex, raw)
    return load_ref_data(cmd, shape, dtype=dtype)


def load_ref_image(
    path: str,
    shape: tuple[int, ...],
    *,
    filter_desc: str | None = "format=pix_fmts=rgb24",
    dtype: DTypeLike = np.uint8,
) -> NDArray[np.uint8]:
    return load_ref_video(path, shape, dtype=dtype, filter_desc=filter_desc)
