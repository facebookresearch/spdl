# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Pipeline ops for the subinterpreter RNG test.

These live in their own **stdlib-only** module because a subinterpreter imports
the op's defining module to resolve it. The main test module imports NumPy /
PyTorch (directly and via ``spdl.source``), none of which can be imported in a
subinterpreter, so the op cannot live there.
"""

import random

__all__ = ["draw_random"]


def draw_random(_: int) -> float:
    """Draw one stdlib ``random`` value (stdlib only -- subinterpreter-safe)."""
    return random.random()
