import sys

from spdl.lib import _libspdl

__all__ = ["init_folly"]


def init_folly(args: list[str]) -> list[str]:
    """Initialize folly internal mechanisms like singletons and logging."""
    return _libspdl.init_folly(sys.argv[0], args)[1:]
