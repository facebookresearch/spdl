"""Utility functions."""

import sys
from contextlib import contextmanager
from typing import Optional

from spdl import libspdl


@contextmanager
def tracing(output: int | str, process_name: Optional[str] = None, enable: bool = True):
    """Trace the decoding operations."""
    if not enable:
        yield
        return

    session = libspdl.init_tracing()
    session.init()
    with open(output, "wb") as f:
        try:
            session.start(f.fileno())
            session.config(sys.argv[0] if process_name is None else process_name)
            yield
        finally:
            session.stop()
