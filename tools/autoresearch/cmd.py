#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Autoresearch CLI tools for SPDL pipeline optimization.

Individual commands for setting up and inspecting autoresearch experiments.
For running the optimization engine, use run.py instead.

Usage:
    python cmd.py init <workdir> [options]
    python cmd.py assess <workdir> --baseline-job <JOB>
    python cmd.py status <workdir>
    python cmd.py report <workdir>
"""

from __future__ import annotations

import argparse
from enum import StrEnum


class _CMD(StrEnum):
    INIT = "init"
    ASSESS = "assess"
    STATUS = "status"
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
            from utils.cmd_init import run

            run(others)

        case _CMD.ASSESS:
            from utils.cmd_assess import run

            run(others)

        case _CMD.STATUS:
            from utils.cmd_status import run

            run(others)

        case _CMD.REPORT:
            from utils.cmd_report import run

            run(others)


if __name__ == "__main__":
    main()
