"""Utility functions."""

import sys
from contextlib import contextmanager
from typing import Any, Optional

from spdl import libspdl

__all__ = [  # noqa: F822
    "tracing",
    "trace_counter",
    "trace_event",
]


def __getattr__(name: str) -> Any:
    """Get the attribute from the library."""
    if name in __all__:
        return getattr(libspdl, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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


@contextmanager
def trace_event(name: str):
    """Trace the decoding operations."""
    libspdl.trace_event_begin(name)
    yield
    libspdl.trace_event_end()
