"""Implement mechanism to load extension module `libspdl`.

It abstracts away the selection of FFmpeg version, and provide
lazy access to the module so that the module won't be loaded until
it's used by user code.
"""

import atexit
import importlib
import importlib.resources
import logging
from typing import Any, List

_LG = logging.getLogger(__name__)

__all__ = [
    "_libspdl",
]


def __dir__() -> List[str]:
    return sorted(__all__)


def __getattr__(name: str) -> Any:
    if name == "_libspdl":
        from spdl._internal.import_utils import _LazilyImportedModule

        return _LazilyImportedModule(name, _import_libspdl)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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

        try:
            ext.register_avdevices()
        except Exception:
            _LG.debug("Failed to register avdevices.", exc_info=True)

        return ext

    raise RuntimeError(
        f"Failed to import libspdl. Tried {libs}. "
        "Enable DEBUG logging to see details about the failure."
    )
