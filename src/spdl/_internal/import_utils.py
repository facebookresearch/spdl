# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import importlib
from types import ModuleType

__all__ = [
    "lazy_import",
]


class _LazilyImportedModule(ModuleType):
    """Delay module import until its attribute is accessed."""

    def __init__(self, name, import_func):
        super().__init__(name)
        self.import_func = import_func
        self.module = None

    # Note:
    # Python caches what was retrieved with `__getattr__`, so this method will not be
    # called again for the same item.
    def __getattr__(self, item):
        self._import_once()
        return getattr(self.module, item)

    def __repr__(self):
        if self.module is None:
            return f"<module '{self.__module__}.{self.__class__.__name__}(\"{self.__name__}\")'>"
        return repr(self.module)

    def __dir__(self):
        self._import_once()
        return dir(self.module)

    def _import_once(self):
        if self.module is None:
            self.module = self.import_func()
            # Note:
            # By attaching the module attributes to self,
            # module attributes are directly accessible.
            # This allows to avoid calling __getattr__ for every attribute access.
            self.__dict__.update(self.module.__dict__)


def lazy_import(name: str) -> ModuleType:
    """Import module lazily.

    Example

        >>> np = lazy_import("numpy")  # not imported yet
        >>> np.__version__  # Now "numpy" is imported.
        1.26.2
    """

    def _import():
        return importlib.import_module(name)

    return _LazilyImportedModule(name, _import)
