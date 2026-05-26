# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Top-level entry point for the autoresearch framework binary.

The single ``spdl-autoresearch`` binary supports two modes, dispatched by
the first positional argument:

- ``engine`` — run the orchestrator directly. Used by the supervisor
  when it execs the engine subprocess, and by developers running the
  engine without the agent wrapper.
- (default) — run the interactive supervisor wrapper, which builds the
  engine command and ``execvp``\\s a coding agent.
"""

from __future__ import annotations

import sys

__all__ = ["main"]


def main(argv: list[str] | None = None) -> None:
    """Dispatch to the supervisor or engine entry point.

    Args:
        argv: Argv list (excluding ``argv[0]``). Defaults to ``sys.argv[1:]``.
    """
    from ._engine import _run_engine

    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "engine":
        _run_engine(args[1:])
        return
    # Supervisor path is wired up in step 5; for now defer to the engine.
    _run_engine(args)
