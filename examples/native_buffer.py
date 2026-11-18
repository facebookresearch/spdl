# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Demonstrate native buffer ownership through the final memoryview alias."""

from spdl.pipeline.lib import _buffer


def main() -> None:
    released = []
    source = bytearray(b"buffer")
    view = _buffer._export(memoryview(source), lambda: released.append(True))
    alias = view[1:]
    view.release()
    del view
    assert not released
    assert bytes(alias) == b"uffer"
    print(f"Retained alias: {bytes(alias)!r}")
    alias.release()
    del alias
    assert released == [True]
    print("Storage released after the final alias.")


if __name__ == "__main__":
    main()
