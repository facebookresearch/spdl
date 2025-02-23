# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


try:
    from .fb import _log_api_usage_once
except ImportError:

    def _log_api_usage_once(key: str) -> None:
        pass
