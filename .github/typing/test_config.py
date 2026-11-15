# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Check that the CI profiles enforce types and retain Buck ignore semantics."""

import json
import shutil
import subprocess
import sysconfig
import tempfile
import unittest
from pathlib import Path

ROOT: Path = Path(__file__).resolve().parents[2]
PROFILES: list[Path] = [ROOT / "pyrefly.toml", ROOT / ".github/typing/pyrefly312.toml"]
VALID: str = """\
from typing_contract_api import make_count

def unannotated():
    return "not checked" + 1

def inferred(value: int):
    return value

good: int = inferred(make_count(1))
# pyre-fixme[9]: Verify compatibility with the existing source suppressions.
legacy: int = "suppressed"
"""
INVALID: str = """\
def bad_return() -> int:
    return "wrong return"

bad_argument: int = make_count("wrong argument")
bad_assignment: str = make_count(1)
import typing_contract_missing_dependency
"""


class PyreflyConfigTest(unittest.TestCase):
    def test_profiles_preserve_compatibility_and_reject_errors(self) -> None:
        executable = shutil.which("pyrefly")
        self.assertIsNotNone(
            executable, "Install .github/typing/requirements.txt first"
        )
        assert executable is not None
        with tempfile.TemporaryDirectory(prefix="spdl-type-check-") as directory:
            root = Path(directory)
            source = root / "probe.py"
            output = root / "errors.json"
            (root / "typing_contract_api.pyi").write_text(
                "def make_count(value: int) -> int: ...\n"
            )
            for profile in PROFILES:
                for invalid in [False, True]:
                    with self.subTest(profile=profile.name, invalid=invalid):
                        source.write_text(VALID + (INVALID if invalid else ""))
                        result = subprocess.run(
                            [
                                executable,
                                "check",
                                "--config",
                                str(profile),
                                "--search-path",
                                str(root),
                                "--skip-interpreter-query",
                                "--site-packages-path",
                                sysconfig.get_paths()["purelib"],
                                "--output-format",
                                "json",
                                "-o",
                                str(output),
                                str(source),
                            ],
                            capture_output=True,
                            text=True,
                            timeout=60,
                        )
                        self.assertEqual(result.returncode, int(invalid), result.stderr)
                        errors = json.loads(output.read_text())["errors"]
                        self.assertEqual(
                            {error["name"] for error in errors},
                            {
                                "bad-return",
                                "bad-argument-type",
                                "bad-assignment",
                                "missing-import",
                            }
                            if invalid
                            else set(),
                            errors,
                        )


if __name__ == "__main__":
    unittest.main()
