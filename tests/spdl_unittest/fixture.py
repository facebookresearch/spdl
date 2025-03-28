# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import subprocess
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

__all__ = [
    "get_sample",
    "get_samples",
    "SrcInfo",
]


def _run_in_tmpdir(cmd: str, tmp_dir: Path) -> None:
    print(f"Executing `{cmd}`")
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
