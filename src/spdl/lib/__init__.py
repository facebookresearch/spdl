"""Implement mechanism to load extension module `libspdl`.

It abstracts away the selection of FFmpeg version, and provide
lazy access to the module so that the module won't be loaded until
it's used by user code.
"""

import atexit
import importlib
import importlib.resources
import logging
from types import ModuleType
from typing import Any, List

_LG = logging.getLogger(__name__)

__all__ = [
    "libspdl",
]


def __dir__() -> List[str]:
    return sorted(__all__)


def __getattr__(name: str) -> Any:
    if name == "libspdl":
        return _LazilyImportedModule(name, _import_libspdl)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
            return f"<module '{self.__module__}.{self.__class__.__name__}(\"{self.name}\")'>"
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


def _import_libspdl():
    libs = [
        f"{__package__}.{t.name.split('.')[0]}"
        for t in importlib.resources.files(__package__).iterdir()
        if t.name.startswith("_spdl_ffmpeg")
    ]
    # Newer FFmpeg first
    libs.sort(reverse=True)
    for lib in libs:
        _LG.debug("Importing %s", lib)
        try:
            ext = importlib.import_module(lib)
        except Exception:
            _LG.debug("Failed to import %s.", lib, exc_info=True)
            continue

        if hasattr(ext, "clear_ffmpeg_cuda_context_cache"):
            atexit.register(ext.clear_ffmpeg_cuda_context_cache)

        return ext

    raise RuntimeError(
        f"Failed to import libspdl. Tried {libs}. "
        "Enable DEBUG logging to see details about the failure."
    )
