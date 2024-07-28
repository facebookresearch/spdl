# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Implement mechanism to load extension module `libspdl`.

It abstracts away the selection of FFmpeg version, and provide
lazy access to the module so that the module won't be loaded until
it's used by user code.
"""

# pyre-unsafe

import importlib
import importlib.resources
import logging
import sys
from types import ModuleType
from typing import Any

_LG = logging.getLogger(__name__)

__all__ = [
    "_libspdl",
]


def __dir__() -> list[str]:
    return sorted(__all__)


def __getattr__(name: str) -> Any:
    if name == "_libspdl":
        from spdl._internal.import_utils import _LazilyImportedModule

        return _LazilyImportedModule(name, _import_libspdl)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _import_libspdl() -> ModuleType:
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

        try:
            ext.log_api_usage("spdl")
        except Exception:
            _LG.debug("Failed to log API usage.", exc_info=True)

        try:
            ext.init_glog(sys.argv[0])
        except Exception:
            _LG.debug("Faile to initialize Google logging.", exc_info=True)

        try:
            ext.set_ffmpeg_log_level(8)
        except Exception:
            _LG.debug("Failed to set FFmpeg log level.", exc_info=True)

        try:
            ext.register_avdevices()
        except Exception:
            _LG.debug("Failed to register avdevices.", exc_info=True)

        return ext

    raise RuntimeError(
        f"Failed to import libspdl. Tried {libs}. "
        "Enable DEBUG logging to see details about the failure."
    )
