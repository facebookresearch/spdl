#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Autoresearch CLI tools for SPDL pipeline optimization.

Individual commands for setting up and inspecting autoresearch experiments.
For running the optimization engine, use _run.py instead.

Usage:
    python _cmd.py init <workdir> [options]
    python _cmd.py assess <workdir> --baseline-job <JOB>
    python _cmd.py status <workdir>
    python _cmd.py queue <workdir> list|remove|priority ...
    python _cmd.py report <workdir>
"""

from __future__ import annotations

import argparse
from enum import StrEnum

__all__ = ["main"]


class _CMD(StrEnum):
    INIT = "init"
    ASSESS = "assess"
    STATUS = "status"
    QUEUE = "queue"
    REPORT = "report"


def _parse_args(
    args: list[str] | None = None,
) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,
    )
    parser.add_argument(
        "command",
        choices=list(_CMD),  # pyre-ignore[6]
        nargs="?",
    )
    ns, others = parser.parse_known_args(args)
    if ns.command is None:
        parser.print_help()
        raise SystemExit(0)
    return ns, others


def main() -> None:
    args, others = _parse_args()

    match args.command:
        case _CMD.INIT:
            from spdl.autoresearch.pipeline_optimization._commands._init import _run

            _run(others)

        case _CMD.ASSESS:
            from spdl.autoresearch.pipeline_optimization._commands._assess import _run

            _run(others)

        case _CMD.STATUS:
            from spdl.autoresearch.pipeline_optimization._commands._status import _run

            _run(others)

        case _CMD.QUEUE:
            from spdl.autoresearch.pipeline_optimization._commands._queue import _run

            _run(others)

        case _CMD.REPORT:
            from spdl.autoresearch.pipeline_optimization._commands._report import _run

            _run(others)


if __name__ == "__main__":
    main()
