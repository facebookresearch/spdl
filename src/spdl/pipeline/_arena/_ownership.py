# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Optional IO buffer exports loaded only by shared-storage data paths."""

from __future__ import annotations

from collections.abc import Callable
from types import ModuleType
from typing import cast

__all__ = ["_ExportGuard", "_export_buffer"]


def _load_buffer_helper() -> ModuleType:
    import importlib

    try:
        return importlib.import_module("spdl.io.lib._buffer")
    except ModuleNotFoundError as error:
        if error.name not in (
            "spdl.io",
            "spdl.io.lib",
            "spdl.io.lib._buffer",
        ):
            raise
        raise RuntimeError(
            "Native arena storage requires optional SPDL IO buffer support; "
            "install spdl.io with its native buffer helper."
        ) from error


class _ExportGuard:
    """One hold: setup may roll back; only native storage releases after adoption."""

    def __init__(self, release: Callable[[], None]) -> None:
        self._ticket = _load_buffer_helper()._release_ticket(release)

    def activate(self) -> None:
        self._ticket._activate()

    def cancel(self) -> None:
        self._ticket._cancel()

    def rollback(self) -> None:
        self._ticket._rollback()


def _export_buffer(
    view: "memoryview[bytes]", release: Callable[[], None] | _ExportGuard
) -> "memoryview[bytes]":
    """Export storage through optional SPDL IO, retaining native alias lifetime.

    Native ticket adoption precedes returning the view, including fallible Python
    post-processing. A failed return/traceback cannot retire native-owned storage.
    The immutable exporter releases backing storage before its callback.
    """
    buffer = _load_buffer_helper()
    ticket = release._ticket if isinstance(release, _ExportGuard) else release
    return cast("memoryview[bytes]", buffer._export(view, ticket))
