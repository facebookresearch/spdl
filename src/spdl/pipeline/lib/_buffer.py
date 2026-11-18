# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Importable buffer interface backed by the optional Rust extension."""

import importlib
from collections.abc import Callable
from functools import cache
from types import ModuleType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._native import _ReleaseTicket

__all__ = [
    "_export",
    "_export_owned",
    "_is_exporter",
    "_load_rust",
    "_release_ticket",
    "_require_backend",
]

_NATIVE = f"{__package__}._native"


@cache
def _load_rust() -> ModuleType:
    return importlib.import_module(_NATIVE)


def _require_backend() -> ModuleType:
    try:
        return _load_rust()
    except ModuleNotFoundError as error:
        if error.name != _NATIVE:
            raise
        raise RuntimeError(
            "Native arena storage requires the SPDL core Rust binding; "
            "build spdl_core with its native extension."
        ) from error


def _release_ticket(release: Callable[[], None]) -> "_ReleaseTicket":
    return _require_backend()._release_ticket(release)


def _export(
    source: memoryview, release: "Callable[[], None] | _ReleaseTicket"
) -> memoryview:
    return _require_backend()._export(source, release)


def _export_owned(address: int, nbytes: int, owner: object) -> memoryview:
    return _require_backend()._export_owned(address, nbytes, owner)


def _is_exporter(owner: object) -> bool:
    """Recognize native storage without importing Rust for foreign buffers."""
    if type(owner).__module__ != _NATIVE:
        return False
    return _load_rust()._is_exporter(owner)
