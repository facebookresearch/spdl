# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Implements the core I/O functionalities."""

# NOTE
# 1. When exposing new Python functions/classes, simply add them in the `__all__`
#    attributes of submodules then update `__init__.pyi`.
# 2. If exposing new C++ class at the top-level, update the list of APIs
#    in `__getattr__` function.

# pyre-strict

from . import (
    _array,
    _composite,
    _config,
    _convert,
    _core,
    _preprocessing,
    _tar,
    _transfer,
    _wav,
)

_mods = [
    _array,
    _composite,
    _config,
    _convert,
    _core,
    _preprocessing,
    _tar,
    _transfer,
    _wav,
]


__all__ = sorted(item for mod in _mods for item in mod.__all__)


def __dir__():
    return __all__


def __getattr__(name: str) -> object:
    """Lazily import C++ extension modules and their contents."""
    if name == "__version__":
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version("spdl.io")
        except PackageNotFoundError:
            return "unknown"

    # Lazy loading of C++ extension classes from _libspdl
    _libspdl_items = {
        "CPUStorage",
        "CPUBuffer",
        "AudioCodec",
        "VideoCodec",
        "ImageCodec",
        "AudioPackets",
        "VideoPackets",
        "ImagePackets",
        "AudioFrames",
        "VideoFrames",
        "ImageFrames",
        "AudioEncoder",
        "VideoEncoder",
        "AudioDecoder",
        "VideoDecoder",
        "ImageDecoder",
        "FilterGraph",
        "DemuxConfig",
        "DecodeConfig",
        "VideoEncodeConfig",
        "AudioEncodeConfig",
    }

    if name in _libspdl_items:
        from . import lib

        disabled = lib._LG.disabled
        lib._LG.disabled = True
        try:
            attr = getattr(lib._libspdl, name)
            return attr
        except RuntimeError:

            class _placeholder:
                def __init__(self, *_args: object, **_kwargs: object) -> None:
                    raise RuntimeError(
                        f"Failed to load `_libspdl.{name}`. " "Is FFmpeg available?"
                    )

            return _placeholder

        finally:
            lib._LG.disabled = disabled

    # Lazy loading of C++ extension classes from _libspdl_cuda
    _libspdl_cuda_items = {
        "CUDABuffer",
        "CUDAConfig",
    }

    if name in _libspdl_cuda_items:
        from . import lib

        disabled = lib._LG.disabled
        lib._LG.disabled = True
        try:
            from .lib import _libspdl_cuda

            attr = getattr(_libspdl_cuda, name)
            return attr
        except RuntimeError:

            class _placeholder:
                def __init__(self, *_args: object, **_kwargs: object) -> None:
                    raise RuntimeError(
                        f"Failed to load `_libspdl_cuda.{name}`. "
                        "Is CUDA runtime available?"
                    )

            return _placeholder

        finally:
            lib._LG.disabled = disabled

    for mod in _mods:
        if name in mod.__all__:
            return getattr(mod, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
