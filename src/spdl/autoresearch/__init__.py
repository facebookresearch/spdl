# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Automated experiment engine for optimizing SPDL data loading pipelines.

Provides the core building blocks for autoresearch: an async work scheduler,
domain types for tracking experiments and failures, and prompt template loading
for LLM-driven workflow agents.
"""

from ._prompts import load_knowledge, load_prompt, load_prompt_directory

__all__ = [
    "load_knowledge",
    "load_prompt",
    "load_prompt_directory",
]
