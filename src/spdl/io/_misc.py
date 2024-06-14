# pyre-unsafe

__all__ = ["SPDLBackgroundTaskFailure"]


# Exception class used to signal the failure of C++ op to Python.
# Not exposed to user code.
class SPDLBackgroundTaskFailure(RuntimeError):
    """Exception type used to pass the error message from libspdl."""

    pass
