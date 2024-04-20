from spdl.lib import _libspdl

__all__ = [
    "AsyncIOFailure",
    "IOConfig",
    "DecodeConfig",
    "ThreadPoolExecutor",
]

try:
    _IOConfig = _libspdl.IOConfig
    _DecodeConfig = _libspdl.DecodeConfig
    _ThreadPoolExecutor = _libspdl.ThreadPoolExecutor
except Exception:
    _IOConfig = object
    _DecodeConfig = object
    _ThreadPoolExecutor = object


# Exception class used to signal the failure of C++ op to Python.
# Not exposed to user code.
class AsyncIOFailure(RuntimeError):
    """Exception type used to pass the error message from libspdl."""

    pass


class IOConfig(_IOConfig):
    """Custom IO config.

    Other Args:
        format (str):
            *Optional* Overwrite format. Can be used if the source file does not have
            a header.

        format_options (Dict[str, str]):
            *Optional* Provide demuxer options

        buffer_size (int):
            *Opitonal* Override the size of internal buffer used for demuxing.
    """

    pass


class DecodeConfig(_DecodeConfig):
    """Custom decode config.

    Other Args:
        decoder (str):
            *Optional* Override decoder.

        decoder_options (Dict[str, str]):
            *Optional* Provide decoder options
    """

    pass


class ThreadPoolExecutor(_ThreadPoolExecutor):
    """Custom thread pool executor to perform tasks.

    Note:
        This is mainly for testing.

    Args:
        num_threads (int): The number of threads.
        thread_name_prefix (str): The prefix of the thread name.
    """

    pass
