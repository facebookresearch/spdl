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

from ._build import (
    get_default_build_callback,
    set_default_build_callback,
)
from ._profile import (
    diagnostic_mode_num_sources,
    get_default_profile_callback,
    get_default_profile_hook,
    is_diagnostic_mode_enabled,
    set_default_profile_callback,
    set_default_profile_hook,
)

__all__ = [
    "is_diagnostic_mode_enabled",
    "diagnostic_mode_num_sources",
    "set_default_hook_class",
    "get_default_hook_class",
    "set_default_queue_class",
    "get_default_queue_class",
    "set_default_profile_hook",
    "get_default_profile_hook",
    "set_default_profile_callback",
    "get_default_profile_callback",
    "get_default_build_callback",
    "set_default_build_callback",
]
