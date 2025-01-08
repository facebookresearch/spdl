# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Implements the core I/O functionalities."""

# pyre-unsafe

from typing import Any

# This has to happen before other sub modules are imporeted.
# Otherwise circular import would occur.
#
# I know, I should not use `*`. I don't want to either, but
# for creating annotation for types from C++ code, which might not be
# available at the runtime, while simultaneously pleasing all the linters
# (black, flake8 and pyre) and documentation tools, this seems like
# the simplest solution.
# This import is just for annotation, so please overlook this one.
from ._type_stub import *  # noqa: F403  # isort: skip

from . import _composite, _config, _convert, _core, _preprocessing, _type_stub, _zip

_mods = [
    _composite,
    _config,
    _convert,
    _core,
    _preprocessing,
    _zip,
]


__all__ = sorted(item for mod in [*_mods, _type_stub] for item in mod.__all__)


def __dir__():
    return __all__


_deprecated_core = {
    "async_demux_audio",
    "async_demux_video",
    "async_demux_image",
    "async_decode_packets",
    "async_decode_packets_nvdec",
    "async_streaming_decode_packets",
    "async_decode_image_nvjpeg",
    "async_convert_array",
    "async_convert_frames",
    "async_transfer_buffer",
    "async_transfer_buffer_cpu",
    "async_encode_image",
    "run_async",
}
_deprecated_composite = {
    "async_load_audio",
    "async_load_video",
    "async_load_image",
    "async_load_image_batch",
    "async_load_image_batch_nvdec",
    "async_load_image_batch_nvjpeg",
    "async_sample_decode_video",
}


def __getattr__(name: str) -> Any:
    if name in _deprecated_core or name in _deprecated_composite:
        import warnings

        warnings.warn(
            f"`{name}` has been deprecated. Please use synchronous variant.",
            category=FutureWarning,
            stacklevel=2,
        )

        if name in _deprecated_core:
            return getattr(_core, name)
        return getattr(_composite, name)

    for mod in _mods:
        if name in mod.__all__:
            return getattr(mod, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
