import sys
from typing import List

from spdl.lib import _libspdl

__all__ = ["init_folly"]


def init_folly(args: List[str]) -> List[str]:
    """Initialize folly internal mechanisms like singletons and logging."""
    return _libspdl.init_folly(sys.argv[0], args)[1:]
