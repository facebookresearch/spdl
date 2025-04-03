# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

__all__ = ["log_api_usage_once"]

try:
    from .fb import log_api_usage_once
except ImportError:

    def log_api_usage_once(_: str) -> None:
        pass
