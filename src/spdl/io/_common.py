# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import os

__all__ = [
    "_resolve_src",
]


def _resolve_src(obj: object) -> "str | memoryview[bytes]":
    if hasattr(obj, "numpy"):  # torch tensor
        obj = obj.numpy()  # pyre-ignore: [16]

    match obj:
        case str() | memoryview():
            # pyrefly: ignore [bad-return]
            return obj
        case _ if hasattr(obj, "__fspath__"):
            # pyrefly: ignore [no-matching-overload]
            return os.fspath(obj)
        case _:
            try:
                # pyrefly: ignore [bad-argument-type, bad-return]
                return memoryview(obj)
            except TypeError as e:
                raise TypeError(
                    "The source must be either path-like object or buffer-like object. "
                    f"Found: {type(obj)=}, {dir(obj)=}"
                ) from e
