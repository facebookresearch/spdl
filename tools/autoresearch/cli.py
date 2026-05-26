#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Entry point for ``spdl autoresearch``.

Thin wrapper that dispatches to either the framework dispatcher
(``spdl.autoresearch._app._main``) or the legacy pipeline-optimization
supervisor depending on argv shape:

- ``autoresearch engine ...`` — engine subcommand handled by the
  framework dispatcher. Used by the supervisor when it execs the engine
  subprocess and by developers running the engine directly.
- ``autoresearch ... --workflow ...`` — framework dispatcher path.
- otherwise — legacy ``pipeline_optimization.main`` supervisor path.
  Preserves byte-for-byte behavior for the ``examples/*/fb/autoresearch.sh``
  scripts and the bundled ``autoresearch_with_pipeline_opt`` binary's
  default invocation.

Step 6 of the refactor promotes ``--workflow`` to a real argument on the
bare ``autoresearch`` binary and migrates the example scripts to the new
flag shape; step 8 retires the legacy path entirely.
"""

from __future__ import annotations

import sys


def main() -> None:
    """Dispatch to the framework engine, framework supervisor, or legacy CLI."""
    args = sys.argv[1:]
    if args and args[0] == "engine":
        from spdl.autoresearch._app._main import main as framework_main

        framework_main()
        return
    if any(a == "--workflow" or a.startswith("--workflow=") for a in args):
        from spdl.autoresearch._app._main import main as framework_main

        framework_main()
        return
    from spdl.autoresearch.pipeline_optimization import main as legacy_main

    legacy_main()


if __name__ == "__main__":
    main()
