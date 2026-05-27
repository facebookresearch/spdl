#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Entry point for the bundled ``autoresearch_with_pipeline_opt`` binary.

Identical to the bare ``autoresearch`` binary except this shim injects
``--workflow spdl.autoresearch.pipeline_optimization:create_workflow``
when the user has not supplied a ``--workflow`` of their own. The bare
binary keeps the framework workflow-agnostic so OSS users (or other
SPDL workflows) can supply their own factory; this bundled binary is
the convenient out-of-the-box default at Meta.
"""

from __future__ import annotations

import sys

# Force the pipeline_optimization package to be linked into this binary so
# the runtime importlib resolution of `_DEFAULT_WORKFLOW` succeeds. The
# `noqa: F401` is required because the import is consumed at runtime via
# `importlib.import_module`, not at parse time.
import spdl.autoresearch.pipeline_optimization  # noqa: F401
from spdl.autoresearch._app._main import main as framework_main

_DEFAULT_WORKFLOW: str = "spdl.autoresearch.pipeline_optimization:create_workflow"


def main() -> None:
    """Inject the default workflow if absent, then dispatch."""
    args = list(sys.argv[1:])
    sys.argv = [sys.argv[0]] + _inject_default_workflow(args)
    framework_main()


def _inject_default_workflow(args: list[str]) -> list[str]:
    if any(a == "--workflow" or a.startswith("--workflow=") for a in args):
        return args
    if args and args[0] == "engine":
        return ["engine", "--workflow", _DEFAULT_WORKFLOW] + args[1:]
    return ["--workflow", _DEFAULT_WORKFLOW] + args


if __name__ == "__main__":
    main()
