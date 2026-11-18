# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from pathlib import Path

from setuptools import find_packages, setup
from setuptools_rust import Binding, RustExtension

THIS_DIR = Path(__file__).parent


def _get_rust_extensions():
    # Opt out explicitly on hosts without Rust; never hide a failed native build.
    value = os.environ.get("SPDL_BUILD_PIPELINE_EXTENSION", "1").lower()
    if value in {"0", "false", "off", "no"}:
        return []
    if value not in {"1", "true", "on", "yes"}:
        raise ValueError(f"Invalid SPDL_BUILD_PIPELINE_EXTENSION={value!r}")
    return [
        RustExtension(
            "spdl.pipeline.lib._native",
            path=str(THIS_DIR / "Cargo.toml"),
            binding=Binding.NoBinding,
            cargo_manifest_args=["--locked"],
            py_limited_api=False,
        )
    ]


def main():
    with open(THIS_DIR / "VERSION", "r") as f:
        version = f.read().strip()

    packages = find_packages(
        where="src",
        exclude=["spdl.io*", "*tests*", "*tests.*"],
    )

    setup(
        name="spdl_core",
        version=version,
        packages=packages,
        package_dir={"": "src"},
        rust_extensions=_get_rust_extensions(),
    )


if __name__ == "__main__":
    main()
