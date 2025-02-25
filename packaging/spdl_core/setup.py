# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from pathlib import Path
import subprocess
import sys
import sysconfig

from setuptools import find_packages, setup

THIS_DIR = Path(__file__).parent


def main():
    with open(THIS_DIR / "VERSION", 'r') as f:
        version = f.read().strip()

    packages = find_packages(where="src", exclude=["spdl.io*"])

    setup(
        name="spdl_core",
        version=version,
        packages=packages,
        package_dir={"": "src"},
        license_files=('LICENSE', ),
    )


if __name__ == "__main__":
    main()
