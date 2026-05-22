# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
from pathlib import Path

_configured = False

__all__ = ["setup_logging"]


def setup_logging(workdir: Path) -> None:
    """Configure logging to stderr and workdir/logs/autoresearch.log."""
    global _configured
    if _configured:
        return
    _configured = True

    log_dir = workdir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "autoresearch.log"

    fmt = "%(asctime)s [%(levelname).1s] %(name)s: %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=fmt, handlers=[])

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    fh = logging.FileHandler(log_file, mode="a")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(fmt))
    root.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter(fmt))
    root.addHandler(sh)
