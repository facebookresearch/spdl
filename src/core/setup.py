# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import find_packages, setup


def main():
    print(find_packages(include="spdl*"))
    setup(
        name="spdl_core",
        version="0.0.7",
        author="Moto Hira",
        description="SPDL: Scalable and Performant Data Loading.",
        long_description="Fast multimedia data loading and processing.",
        packages=find_packages(include="spdl*"),
        python_requires=">=3.10",
    )


if __name__ == "__main__":
    main()
