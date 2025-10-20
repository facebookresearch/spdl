# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Module for runtime configuration."""

# IMPLEMENTATION NOTE
# Implement the configuration system where it is used.
# This module should be just exposing them to external consumers.

from spdl.pipeline._components import (
    get_default_hook_class,
    get_default_queue_class,
    set_default_hook_class,
    set_default_queue_class,
)

from ._profile import (
    _diagnostic_mode_num_sources,
    _is_diagnostic_mode_enabled,
    get_default_profile_callback,
    get_default_profile_hook,
    set_default_profile_callback,
    set_default_profile_hook,
)

__all__ = [
    "_is_diagnostic_mode_enabled",
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
