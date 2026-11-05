# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable

__all__ = ["_export", "_export_owned", "_release_ticket"]

class _ReleaseTicket:
    def _activate(self) -> None: ...
    def _cancel(self) -> None: ...
    def _rollback(self) -> None: ...

def _release_ticket(release: Callable[[], None]) -> _ReleaseTicket: ...
def _export(
    source: memoryview[bytes], release: Callable[[], None] | _ReleaseTicket
) -> memoryview[bytes]: ...
def _export_owned(address: int, nbytes: int, owner: object) -> memoryview: ...
