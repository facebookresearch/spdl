from spdl.lib import _libspdl

__all__ = [
    "is_cuda_available",
    "is_nvcodec_available",
]


def is_cuda_available() -> bool:
    """Check if SPDL is compiled with CUDA support.

    Returns:
        (bool): True if SPDL is compiled with CUDA support.
    """
    return _libspdl.is_cuda_available()


def is_nvcodec_available() -> bool:
    """Check if SPDL is compiled with NVCODEC support.

    Returns:
        (bool): True if SPDL is compiled with NVCODEC support.
    """
    return _libspdl.is_nvcodec_available()
