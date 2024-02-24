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
    width: int
    height: int


@pytest.fixture
def get_sample(run_in_tmpdir):
    def _get_sample(cmd, width, height, output_file=None):
        tmpdir = run_in_tmpdir(cmd)
        output_file = output_file or cmd.split()[-1]
        path = str(tmpdir / output_file)
        return SrcInfo(path, width, height)

    return _get_sample
