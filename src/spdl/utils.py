"""Utility functions."""

import sys
from contextlib import contextmanager
from typing import Any, List, Optional

from spdl.lib import _libspdl

_funcs = [
    "clear_ffmpeg_cuda_context_cache",
    "create_cuda_context",
    "get_cuda_device_index",
    "get_ffmpeg_log_level",
    "set_ffmpeg_log_level",
    "trace_counter",
    "trace_default_decode_executor_queue_size",
    "trace_default_demux_executor_queue_size",
    "trace_event_begin",
    "trace_event_end",
]

__all__ = [  # noqa: F822
    "init_folly",
    "tracing",
    "trace_counter",
    "trace_event",
] + _funcs


def __getattr__(name: str) -> Any:
    """Get the attribute from the library."""
    if name in _funcs:
        return getattr(_libspdl, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def init_folly(args: List[str]) -> List[str]:
    """Initialize folly internal mechanisms like singletons and logging."""
    return _libspdl.init_folly(sys.argv[0], args)[1:]


@contextmanager
def tracing(output: int | str, process_name: Optional[str] = None, enable: bool = True):
    """Trace the decoding operations."""
    if not enable:
        yield
        return

    session = _libspdl.init_tracing()
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
    _libspdl.trace_event_begin(name)
    yield
    _libspdl.trace_event_end()
