#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Check an installed core wheel, without IO or other runtime dependencies."""

import argparse
import importlib.machinery
import importlib.metadata
import subprocess
import sys
import sysconfig
from pathlib import Path

MODULE = "spdl.pipeline.lib._native"


def _probe(native: bool) -> None:
    # A fresh, isolated process prevents imports from the checkout or cached
    # native modules from turning an absent-binary check into a false positive.
    code = f"""
import importlib
import sys
import sysconfig

def gil_enabled():
    return getattr(sys, '_is_gil_enabled', lambda: True)()

before = gil_enabled()
if sysconfig.get_config_var('Py_GIL_DISABLED'):
    assert not before, 'Run free-threaded checks without forcing the GIL on'

import spdl.pipeline
assert {MODULE!r} not in sys.modules, 'Pipeline import eagerly loaded native code'
assert not any(name == 'spdl.io' or name.startswith('spdl.io.') for name in sys.modules)
assert not any(name in sys.modules for name in ('numpy', 'torch', 'jax'))

pipeline = (spdl.pipeline.PipelineBuilder().add_source(range(3))
            .add_sink(3).build(num_threads=1))
with pipeline.auto_stop():
    assert list(pipeline) == [0, 1, 2]
assert {MODULE!r} not in sys.modules

facade = importlib.import_module('spdl.pipeline.lib._libspdl_pipeline')
buffer = importlib.import_module('spdl.pipeline.lib._buffer')
assert {MODULE!r} not in sys.modules, 'Python interfaces eagerly loaded native code'

if {native!r}:
    assert buffer._require_backend().__name__ == {MODULE!r}, 'Rust backend was not selected'
    released = []
    view = facade._export(memoryview(bytearray(b'owned')), lambda: released.append(True))
    alias = view[1:]
    view.release()
    del view
    assert bytes(alias) == b'wned' and not released
    alias.release()
    del alias
    assert released == [True], released
else:
    try:
        facade._export(memoryview(bytearray(b'owned')), lambda: None)
    except RuntimeError as exc:
        assert isinstance(exc.__cause__, ModuleNotFoundError), exc
        assert exc.__cause__.name == {MODULE!r}, exc
    else:
        raise AssertionError('Missing binary was replaced with a fabricated module')
assert gil_enabled() == before, 'Import changed the interpreter GIL state'
print('installed pipeline probe OK; native=' + str({native!r}) + '; GIL=' + str(before))
"""
    subprocess.run(
        [sys.executable, "-I", "-Werror", "-c", code], check=True, timeout=60
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-native", action="store_true")
    args = parser.parse_args()

    dist = importlib.metadata.distribution("spdl_core")
    assert not dist.requires, "spdl_core must have no runtime dependency"
    wheel = dist.read_text("WHEEL")
    assert wheel is not None
    binaries = [
        Path(dist.locate_file(file))
        for file in dist.files or []
        if file.parent.as_posix() == "spdl/pipeline/lib"
        and file.name.startswith("_native")
        and any(
            file.name.endswith(suffix)
            for suffix in importlib.machinery.EXTENSION_SUFFIXES
        )
    ]
    if args.no_native:
        assert not binaries, binaries
        assert "Root-Is-Purelib: true" in wheel, wheel
        assert "Tag: py3-none-any" in wheel, wheel
        _probe(False)
        return

    assert len(binaries) == 1, binaries
    assert "Root-Is-Purelib: false" in wheel, wheel
    cp = f"cp{sys.version_info.major}{sys.version_info.minor}"
    abi = cp + ("t" if sysconfig.get_config_var("Py_GIL_DISABLED") else "")
    assert f"Tag: {cp}-{abi}-" in wheel, wheel
    _probe(True)
    binary = binaries[0]
    hidden = binary.with_name(binary.name + ".disabled")
    assert not hidden.exists(), hidden
    binary.rename(hidden)
    try:
        _probe(False)
    finally:
        hidden.rename(binary)


if __name__ == "__main__":
    main()
