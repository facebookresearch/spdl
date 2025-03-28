# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import subprocess
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory


def run_in_tmpdir(cmd, tmp_dir):
    print(f"Executing `{cmd}`")
    subprocess.run(cmd, cwd=tmp_dir, shell=True, check=True, capture_output=False)


@dataclass
class SrcInfo:
    path: str | list[str]
    _tmp_dir: TemporaryDirectory


def get_sample(cmd):
    tmp_dir = TemporaryDirectory()
    tmp_path = Path(tmp_dir.name)

    run_in_tmpdir(cmd, tmp_path)
    path = [f for f in tmp_path.glob("*") if f.is_file()]
    assert len(path) == 1
    return SrcInfo(path[0], tmp_dir)


def get_samples(cmd):
    tmp_dir = TemporaryDirectory()
    tmp_path = Path(tmp_dir.name)

    run_in_tmpdir(cmd, tmp_path)
    flist = [str(f) for f in tmp_path.glob("**/*") if f.is_file()]
    return SrcInfo(flist, tmp_dir)
