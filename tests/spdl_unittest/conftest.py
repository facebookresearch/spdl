import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def run_in_tmpdir(tmpdir):
    cwd = Path(tmpdir)

    def run_in_tmpdir_(cmd, shell=True, **kwargs):
        print(f"Executing `{cmd}`")
        subprocess.run(
            cmd, cwd=cwd, shell=shell, check=True, capture_output=False, **kwargs
        )
        return cwd

    return run_in_tmpdir_
