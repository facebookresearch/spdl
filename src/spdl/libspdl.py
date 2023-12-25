"""Thin wrapper around the libspdl extension."""

import sys
from typing import Any, List, Optional

import numpy as np
from numpy.typing import NDArray

from spdl.lib import libspdl as _libspdl


__all__ = [  # noqa: F822
    "init_folly",
    "Engine",
    "to_numpy",
]


def __dir__() -> List[str]:
    return sorted(__all__)


def __getattr__(name: str) -> Any:
    if name == "Engine":
        return _libspdl.Engine
    return getattr(_libspdl, name)


def init_folly(args: List[str]) -> List[str]:
    """Initialize folly internal mechanisms like singletons and logging."""
    return _libspdl.init_folly(sys.argv[0], args)[1:]


def to_numpy(buffer, format: Optional[str] = "NCHW") -> NDArray:
    """Convert to numpy array.

    Args:
        buffer (VideoBuffer): Raw buffer.

        format (str or None): Channel order.
            Valid values are ``"NCHW"``, ``"NHWC"`` or ``None``.
            If ``None`` no conversion is performed and native format is returned.
    """
    array = np.array(buffer, copy=False)
    match format:
        case "NCHW":
            if buffer.channel_last:
                array = np.moveaxis(array, -1, -3)
        case "NHWC":
            if not buffer.channel_last:
                array = np.moveaxis(array, -3, -1)
        case None:
            pass
        case _:
            raise ValueError(
                "Expected format value to be one of ['NCHW', 'NHWC', None]"
                f", but received: {format}"
            )
    return array
