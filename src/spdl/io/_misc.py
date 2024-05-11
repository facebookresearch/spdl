from spdl.lib import _libspdl

__all__ = [
    "SPDLBackgroundTaskFailure",
    "Executor",
]


# Exception class used to signal the failure of C++ op to Python.
# Not exposed to user code.
class SPDLBackgroundTaskFailure(RuntimeError):
    """Exception type used to pass the error message from libspdl."""

    pass


def Executor(num_threads: int, thread_name_prefix: str):
    """Custom thread pool executor.

    Args:
        num_threads: The size of the executor thread pool.
        thread_name_prefix: The prefix of the thread names in the thread pool.

    ??? note "Example: Specifying custom thread pool"
        ```python
        # Use a thread pool different from default one
        exec = Executor(num_threads=10, thread_name_prefix="custom_exec")

        packets = await spdl.io.async_demux_media("video", src, executor=exec)
        frames = await spdl.io.async_decode_packets(packets, executor=exec)
        buffer = await spdl.io.async_convert_frames(frames, executor=exec)
        ```
    """
    return _libspdl.ThreadPoolExecutor(num_threads, thread_name_prefix)
