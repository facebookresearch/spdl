#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re
import sysconfig
from pathlib import Path
import warnings


def get_dir() -> Path:
    # Find all matching 'nvidia/cu*/lib' paths

    root_dir = Path(sysconfig.get_paths()["purelib"])
    matches: dict[int, Path] = {}
    for path in (root_dir / "nvidia").rglob("lib"):
        for parent in path.parents:
            if parent.match("cu*"):
                if version_match := re.search(r"\d+", parent.name):
                    version_num = int(version_match.group())
                    # Store path by version number to avoid duplicates
                    matches[version_num] = path
                break

    if not matches:
        raise RuntimeError("No cuda lib dir was found.")

    # Check for multiple versions and trigger a warning
    if len(matches) > 1:
        found_versions = sorted(list(matches.keys()))
        latest_version = max(found_versions)
        warnings.warn(
            f"Multiple CUDA versions found: {found_versions}. "
            f"Automatically selecting the latest version: cu{latest_version}.",
            UserWarning,
        )

    # Return the path of the highest version
    latest_version = max(matches.keys())
    return matches[latest_version]


if __name__ == "__main__":
    print(get_dir())
