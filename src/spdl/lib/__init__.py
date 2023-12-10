import importlib
import logging
import sys
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
        return _LazyImporter(name, _import_libspdl)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class _LazyImporter(ModuleType):
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
    versions = ["6", "5", "4", ""]
    for ver in versions:
        _LG.debug("Importing libspdl with FFmpeg %s", ver)
        try:
            return _import_libspdl_ver(ver)
        except Exception:
            _LG.debug("Failed to import libspdl with FFmpeg %s.", ver, exc_info=True)
            continue
    raise RuntimeError(
        f"Failed to import libspdl. Tried with FFmpeg versions {versions}. "
        "Enable DEBUG logging to see more details about the error."
    )


def _import_libspdl_ver(ver):
    ext = f"spdl.lib._spdl_ffmpeg{ver}"

    if not importlib.util.find_spec(ext):
        raise RuntimeError(f"Extension is not available: {ext}.")

    try:
        ext = importlib.import_module(ext)
    except Exception:
        raise RuntimeError(f"Failed to load extension: {ext}.")

    return ext
