import sys
from contextlib import contextmanager
from typing import Optional, Union

from spdl.lib import _libspdl


__all__ = [
    "trace_counter",
    "trace_default_decode_executor_queue_size",
    "trace_default_demux_executor_queue_size",
    "trace_event",
    "tracing",
]


@contextmanager
def tracing(
    output: int | str,
    buffer_size: int = 4096,
    process_name: Optional[str] = None,
    enable: bool = True,
):
    """Enable tracing.

    Args:
        output: The path to the trace file or file descriptor.
        buffer_size: The size of the trace file write buffer in kilo-byte.
        process_name: The name of the tracing process.
        enable: Whether to enable tracing.

    ??? note "Example"
        ```python
        with tracing("/tmp/trace.pftrace"):
            do_operations()
        ```
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

    ??? note "Example"
        ```python
        with tracing():
            # Tracce `my_operation()` with the name "my_operation".
            with trace_event("my_operation"):
                my_operation()
        ```
    """
    _libspdl.trace_event_begin(name)
    yield
    _libspdl.trace_event_end()


def trace_counter(i: int, val: Union[int, float]):
    """Trace a counter value.

    Note:
        Due to the compile-time restriction, there can be
        a fixed number of counter traces.

    Args:
        i: The index of the counter. Must be [0, 7].
        val: The value of the counter.

    ??? note "Example"
        ```python
        with tracing():
            trace_counter(0, 2.5);
            trace_counter(0, 3);
        ```
    """
    _libspdl.trace_counter(i, val)


def trace_default_decode_executor_queue_size():
    """Trace the number of queued items in the default decode executor queue.

    ??? note "Example"
        ```python
        with tracing():
            trace_default_decode_executor_queue_size()
            # ... do other stuff
            trace_default_decode_executor_queue_size()
        ```
    """
    _libspdl.trace_default_decode_executor_queue_size()


def trace_default_demux_executor_queue_size():
    """Trace the number of queued items in the default demux executor queue.

    ??? note "Example"
        ```python
        with tracing():
            trace_default_demux_executor_queue_size()
            # ... do other stuff
            trace_default_demux_executor_queue_size()
        ```
    """
    _libspdl.trace_default_demux_executor_queue_size()
