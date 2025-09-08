#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Generate interface files for extension modules."""

import re
from argparse import ArgumentParser
from pathlib import Path

from nanobind.stubgen import StubGen
from spdl.io.lib import _archive, _import_libspdl, _import_libspdl_cuda


def _parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", "-o", required=True, type=Path)
    return parser.parse_args()


def _generate(mod):
    sg = StubGen(mod)
    sg.put(mod)
    return sg.get()


def _main():
    args = _parse_args()
    args.output_dir.mkdir(exist_ok=True, parents=True)

    with open(args.output_dir / "_archive.pyi", "w") as f:
        stub = _generate(_archive)
        f.write(stub)

    with open(args.output_dir / "_libspdl.pyi", "w") as f:
        stub = _generate(_import_libspdl())
        f.write(stub)

    with open(args.output_dir / "_libspdl_cuda.pyi", "w") as f:
        stub = _generate(_import_libspdl_cuda())
        


if __name__ == "__main__":
    _main()
