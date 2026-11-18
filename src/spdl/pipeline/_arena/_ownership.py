# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Storage exports backed by the optional native pipeline binding."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

from spdl.pipeline.lib import _buffer

__all__ = ["_ExportGuard", "_export_buffer"]


class _ExportGuard:
    """One hold: setup may roll back; only native storage releases after adoption."""

    def __init__(self, release: Callable[[], None]) -> None:
        self._ticket = _buffer._release_ticket(release)

    def activate(self) -> None:
        self._ticket._activate()

    def cancel(self) -> None:
        self._ticket._cancel()

    def rollback(self) -> None:
        self._ticket._rollback()


def _export_buffer(
    view: "memoryview[int]", release: Callable[[], None] | _ExportGuard
) -> "memoryview[int]":
    """Export storage through the pipeline binding, retaining native alias lifetime.

    Native ticket adoption precedes returning the view, including fallible Python
    post-processing. A failed return/traceback cannot retire native-owned storage.
    The immutable exporter releases backing storage before its callback.
    """
    ticket = release._ticket if isinstance(release, _ExportGuard) else release
    return cast("memoryview[int]", _buffer._export(view, ticket))
