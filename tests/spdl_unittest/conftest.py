import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def run_in_tmpdir(tmp_path):
    def run_in_tmpdir_(cmd, shell=True, **kwargs):
        print(f"Executing `{cmd}`")
        subprocess.run(
            cmd, cwd=tmp_path, shell=shell, check=True, capture_output=False, **kwargs
        )
        return tmp_path

    return run_in_tmpdir_
