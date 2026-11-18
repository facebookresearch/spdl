# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Native bindings used by the pipeline, loaded only when requested.

.. versionadded:: 0.9.0
"""

from types import ModuleType
from typing import TYPE_CHECKING

from spdl._internal.import_utils import lazy_import

if TYPE_CHECKING:
    from . import _buffer as _buffer, _libspdl_pipeline as _libspdl_pipeline

__all__ = ["_buffer", "_libspdl_pipeline"]


def __getattr__(name: str) -> ModuleType:
    if name in ("_buffer", "_libspdl_pipeline"):
        return lazy_import(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
