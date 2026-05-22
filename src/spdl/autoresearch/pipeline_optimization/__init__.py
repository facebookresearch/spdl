# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""SPDL pipeline optimization implementation of autoresearch.

Provides the complete workflow for automatically optimizing SPDL data loading
pipelines: supervisor CLI, engine runner, experiment adapter, platform
providers, and CLI commands.
"""

from ._cli import main

__all__ = ["main"]
