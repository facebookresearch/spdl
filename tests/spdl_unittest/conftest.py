# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import subprocess
from dataclasses import dataclass

import pytest


@pytest.fixture
def run_in_tmpdir(tmp_path):
    def _run_in_tmpdir(cmd, shell=True, **kwargs):
        print(f"Executing `{cmd}`")
        subprocess.run(
            cmd, cwd=tmp_path, shell=shell, check=True, capture_output=False, **kwargs
        )
        return tmp_path

    return _run_in_tmpdir


@dataclass
class SrcInfo:
    path: str
    width: int = -1
    height: int = -1
    num_channels: int = -1


@pytest.fixture
def get_sample(run_in_tmpdir):
    def _get_sample(cmd, output_file=None, **kwargs):
        tmpdir = run_in_tmpdir(cmd)
        output_file = output_file or cmd.split()[-1]
        path = str(tmpdir / output_file)
        return SrcInfo(path, **kwargs)

    return _get_sample


@pytest.fixture
def get_samples(run_in_tmpdir):
    def _get_samples(cmd):
        tmpdir = run_in_tmpdir(cmd)
        return [str(f) for f in tmpdir.glob("**/*") if f.is_file()]

    return _get_samples
