# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Prompt template and shared knowledge loading for autoresearch agents.

Prompt directories are semantic bundles, not platform switches. Callers load a
category such as ``knowledge`` or ``platform``; extension content is included by
packaging files under that category, for example ``knowledge/fb``. Generic code
must not explicitly request extension subdirectories.

+----------------+-------------------------+----------------------------------+------------+
| Category       | Loaded by               | Loaded when                      | Recursive? |
+================+=========================+==================================+============+
| ``supervisor`` | CLI                     | Before launching supervisor      | yes        |
+----------------+-------------------------+----------------------------------+------------+
| ``knowledge``  | Workflow agent adapter  | When agent context is assembled  | yes        |
+----------------+-------------------------+----------------------------------+------------+
| ``platform``   | CLI/platform/workflow   | When execution guidance is used  | yes        |
+----------------+-------------------------+----------------------------------+------------+
| phase prompts  | Engine workflow         | Only for the active phase        | no         |
+----------------+-------------------------+----------------------------------+------------+
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent.parent
_PROMPTS_DIR = _SCRIPT_DIR / "prompts"

__all__ = [
    "_load_knowledge",
    "_load_prompt_directory",
    "_load_prompt",
]


def _load_prompt(name: str, **kwargs: str) -> str:
    path = _PROMPTS_DIR / f"{name}.md"
    if not path.exists():
        print(f"Error: prompt template not found: {path}", file=sys.stderr)
        sys.exit(1)
    template = path.read_text()
    for key, value in kwargs.items():
        template = template.replace(f"__{key.upper()}__", str(value))
    return template


def _load_knowledge() -> str:
    """Load workflow-agent context assembled from semantic prompt categories."""

    return "\n\n".join(
        part
        for part in (
            _load_prompt_directory("knowledge"),
            _load_prompt_directory("platform"),
        )
        if part
    )


def _load_prompt_directory(relative_dir: str) -> str:
    """Load all Markdown prompt fragments under a prompt subdirectory.

    Prompt categories are directory based so extensions can add knowledge
    without updating Python file lists. Files are loaded recursively in sorted
    path order to keep prompt assembly deterministic.
    """

    root = _PROMPTS_DIR / relative_dir
    if not root.is_dir():
        return ""
    return "\n\n".join(
        path.read_text() for path in sorted(root.rglob("*.md")) if path.is_file()
    )
