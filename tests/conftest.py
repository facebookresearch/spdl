# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# OSS-side test bootstrap.
#
# 1. Make ``spdl.io.tests`` importable by registering a synthetic namespace
#    package whose ``__path__`` points at the symlinked ``tests/io/`` shim.
#    Tests do ``from spdl.io.tests.fixture import ...``; the wheel ships
#    ``spdl`` and ``spdl.io`` but excludes the ``tests`` subpackage, so we
#    splice it in at pytest startup.
#
# 2. Some pipeline tests pickle local helper classes (e.g. ``_Wrap``) and
#    unpickle them inside a fresh subinterpreter. A new subinterpreter is
#    initialized from scratch and only sees ``PYTHONPATH`` — pytest's runtime
#    ``sys.path`` additions are not inherited. Prepend ``tests/pipeline/`` to
#    ``PYTHONPATH`` so the subinterpreter can resolve the originating test
#    module when unpickling.
#
# This file is unused in fbcode (Buck wires up ``spdl.io.tests`` directly via
# the ``//spdl/io/tests:fixture`` library).

import importlib
import os
import sys
import types

_HERE = os.path.dirname(os.path.abspath(__file__))
_TESTS_IO = os.path.join(_HERE, "io")
_TESTS_PIPELINE = os.path.join(_HERE, "pipeline")

# Splice ``spdl.io.tests`` to point at our symlink shim. ``spdl.io`` itself is
# loaded from the installed wheel; we attach a synthetic ``tests`` submodule.
_spdl_io = importlib.import_module("spdl.io")
_tests_pkg = types.ModuleType("spdl.io.tests")
_tests_pkg.__path__ = [_TESTS_IO]
sys.modules["spdl.io.tests"] = _tests_pkg
_spdl_io.tests = _tests_pkg  # type: ignore[attr-defined]

# Subinterpreter helper: ensure pickled test classes resolve in fresh interps.
_existing = os.environ.get("PYTHONPATH", "")
os.environ["PYTHONPATH"] = (
    os.pathsep.join([_TESTS_PIPELINE, _existing]) if _existing else _TESTS_PIPELINE
)
