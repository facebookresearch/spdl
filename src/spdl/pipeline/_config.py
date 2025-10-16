# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

from ._hook import get_default_hook_class, set_default_hook_class
from ._profile import (
    get_default_profile_callback,
    get_default_profile_hook,
    set_default_profile_callback,
    set_default_profile_hook,
)
from ._queue import get_default_queue_class, set_default_queue_class

__all__ = [
    "_diagnostic_mode_enabled",
    "_diagnostic_mode_num_sources",
    "set_default_hook_class",
    "get_default_hook_class",
    "set_default_queue_class",
    "get_default_queue_class",
    "set_default_profile_hook",
    "get_default_profile_hook",
    "set_default_profile_callback",
    "get_default_profile_callback",
]

_LG: logging.Logger = logging.getLogger(__name__)


def _env(name, default=False):
    if name not in os.environ:
        return default

    val = os.environ.get(name, "0")
    trues = ["1", "true", "TRUE", "on", "ON", "yes", "YES"]
    falses = ["0", "false", "FALSE", "off", "OFF", "no", "NO"]
    if val in trues:
        return True
    if val not in falses:
        _LG.warning(
            f"Unexpected environment variable value `{name}={val}`. "
            f"Expected one of {trues + falses}",
            stacklevel=2,
        )
    return False


def _diagnostic_mode_enabled() -> bool:
    return _env("SPDL_PIPELINE_DIAGNOSTIC_MODE")


def _diagnostic_mode_num_sources() -> int:
    return int(os.environ.get("SPDL_PIPELINE_DIAGNOSTIC_MODE_NUM_ITEMS", 1000))
