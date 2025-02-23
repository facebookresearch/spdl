# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import sys
from contextlib import contextmanager

from spdl.io.lib import _libspdl

__all__ = [
    "trace_counter",
    "trace_event",
    "tracing",
    "trace_gc",
]


@contextmanager
def tracing(
    output: int | str,
    buffer_size: int = 4096,
    process_name: str | None = None,
    enable: bool = True,
):
    """Enable tracing.

    Args:
        output: The path to the trace file or file descriptor.
        buffer_size: The size of the trace file write buffer in kilo-byte.
        process_name: The name of the tracing process.
        enable: Whether to enable tracing.

    .. admonition:: Example

       >>> with tracing("/tmp/trace.pftrace"):
       >>>     do_operations()
    """
    if not enable:
        yield
        return

    session = _libspdl.init_tracing()
    session.init()
    with open(output, "wb") as f:
        try:
            session.start(f.fileno(), buffer_size)
            session.config(sys.argv[0] if process_name is None else process_name)
            yield
        finally:
            session.stop()


@contextmanager
def trace_event(name: str):
    """Trace an operation with custom name.

    Args:
        name: The name of the tracing slice.

    .. admonition:: Example

       >>> with tracing():
       >>>     # Tracce `my_operation()` with the name "my_operation".
       >>>     with trace_event("my_operation"):
       >>>         my_operation()
    """
    _libspdl.trace_event_begin(name)
    yield
    _libspdl.trace_event_end()


def trace_counter(i: int, val: int | float):
    """Trace a counter value.

    Note:
        Due to the compile-time restriction, there can be
        a fixed number of counter traces.

    Args:
        i: The index of the counter. Must be [0, 7].
        val: The value of the counter.

    .. admonition:: Example

       >>> with tracing():
       >>>     trace_counter(0, 2.5);
       >>>     trace_counter(0, 3);
    """
    _libspdl.trace_counter(i, val)


def trace_gc():
    """Attach tracer to garbage collection."""
    import gc

    def _func(phase, _info):
        if phase == "start":
            _libspdl.trace_event_begin("gc")
        else:
            _libspdl.trace_event_end()

    gc.callbacks.append(_func)
