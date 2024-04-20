from spdl.lib import _libspdl

__all__ = [
    "AsyncIOFailure",
    "IOConfig",
]

try:
    _IOConfig = _libspdl.IOConfig
except AttributeError:
    _IOConfig = object


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
