# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Utility CLI for SPDL data processing.

.. versionadded:: 0.5.0
"""

# For quick startup, do not add non-standard import at module level.

import argparse
import logging
import sys


if sys.version_info >= (3, 11):
    from enum import StrEnum

    class _CMD(StrEnum):
        AUTORESEARCH = "autoresearch"

else:
    from enum import Enum

    class _CMD(str, Enum):
        AUTORESEARCH = "autoresearch"


def _parse_args(args: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False,
    )
    parser.add_argument(
        "command",
        choices=list(_CMD),  # pyre-ignore[6]
        nargs="?",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
    )
    ns, others = parser.parse_known_args(args)
    if ns.command is None:
        parser.print_help()
        raise SystemExit(0)
    return ns, others


def main() -> None:
    args, others = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname).1s] %(message)s",
    )

    match args.command:
        case _CMD.AUTORESEARCH:
            from spdl.autoresearch._app._main import main

            main(others)


if __name__ == "__main__":
    main()
