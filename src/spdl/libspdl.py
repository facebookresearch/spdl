"""Thin wrapper around the libspdl extension."""

import sys
from typing import Any, List

from spdl.lib import libspdl as _libspdl


__all__ = [  # noqa: F822
    "init_folly",
    "Engine",
]


def __dir__() -> List[str]:
    return sorted(__all__)


def __getattr__(name: str) -> Any:
    if name == "Engine":
        return _libspdl.Engine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def init_folly(args: List[str]) -> List[str]:
    """Initialize folly internal mechanisms like singletons and logging."""
    return _libspdl.init_folly(sys.argv[0], args)[1:]
