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
from importlib.resources import files
from importlib.resources.abc import Traversable

_PROMPTS_ROOT = files(__package__) / "prompts"

__all__ = [
    "load_knowledge",
    "load_prompt",
    "load_prompt_directory",
]


def load_prompt(name: str, **kwargs: str) -> str:
    """Load a named prompt template and substitute placeholders.

    Reads the Markdown file at ``prompts/{name}.md`` and performs placeholder
    substitution.  Placeholders are uppercase tokens delimited by double
    underscores (e.g. ``__KNOWLEDGE__``, ``__PIPELINE_CODE__``).  Each keyword
    argument replaces the matching placeholder — the key is uppercased
    automatically.

    Example::

        >>> prompt = load_prompt(
        ...     "headspace",
        ...     KNOWLEDGE="SPDL optimization guide...",
        ...     PIPELINE_SCRIPT="train.py",
        ...     PIPELINE_CODE="def main(): ...",
        ... )
        >>> prompt[:60]
        'You are adding CacheDataLoader instrumentation to a data lo'

    Available phase prompts: ``analyze``, ``apply_changes``,
    ``apply_startup_repair``, ``assess``, ``headspace``, ``instrument``,
    ``plan_next``.

    Args:
        name: Prompt template name (without ``.md`` extension), e.g.
            ``"analyze"`` loads ``prompts/analyze.md``.
        **kwargs: Placeholder values.  A keyword ``KNOWLEDGE="..."`` replaces
            every occurrence of ``__KNOWLEDGE__`` in the template.  Keys are
            case-insensitive (uppercased internally).

    Returns:
        The fully rendered prompt string with all placeholders replaced.

    Raises:
        SystemExit: If the template file does not exist.
    """
    path = _PROMPTS_ROOT / f"{name}.md"
    if not path.is_file():
        print(f"Error: prompt template not found: {path}", file=sys.stderr)
        sys.exit(1)
    template = path.read_text(encoding="utf-8")
    for key, value in kwargs.items():
        template = template.replace(f"__{key.upper()}__", str(value))
    return template


def load_knowledge() -> str:
    """Load combined domain knowledge for workflow agents.

    Assembles a single knowledge string from two prompt directories:

    - ``knowledge/`` — SPDL optimization techniques (MTP, concurrency tuning,
      headspace analysis, etc.) and autoresearch-specific guidance.
    - ``platform/`` — execution environment guidance (e.g. MAST job launch
      patterns under ``platform/fb/``).

    Files within each directory are loaded recursively in sorted order.
    Meta-specific extensions (e.g. ``knowledge/fb/knowledge.md``) are
    included automatically when present.

    Example::

        >>> text = load_knowledge()
        >>> print(text[:200])
        # SPDL Pipeline Optimization Knowledge
        <BLANKLINE>
        This knowledge is assembled from shared skill files. ...

    Returns:
        The concatenated knowledge text with sections separated by double
        newlines, or an empty string when no files exist.
    """

    return "\n\n".join(
        part
        for part in (
            load_prompt_directory("knowledge"),
            load_prompt_directory("platform"),
        )
        if part
    )


def load_prompt_directory(relative_dir: str) -> str:
    """Load and concatenate all Markdown files under a prompt subdirectory.

    Scans ``prompts/{relative_dir}/`` recursively for ``.md`` files, sorts
    them by path, and joins their contents with double newlines.  This
    directory-based approach lets extensions (e.g. ``knowledge/fb/``) add
    content without modifying Python source.

    Example::

        >>> text = load_prompt_directory("supervisor")
        >>> print(text[:120])
        # Autoresearch: Automated SPDL Pipeline Optimization
        <BLANKLINE>
        You are helping the user launch and monitor the autoresearch...

        >>> load_prompt_directory("nonexistent")
        ''

    Available directories: ``"knowledge"``, ``"platform"``, ``"supervisor"``.

    Args:
        relative_dir: Subdirectory name relative to the prompts root,
            e.g. ``"knowledge"`` loads all ``.md`` files under
            ``prompts/knowledge/`` (including ``prompts/knowledge/fb/``).

    Returns:
        The concatenated content of all ``.md`` files found, separated by
        double newlines.  Returns an empty string when the directory does
        not exist.
    """

    root = _PROMPTS_ROOT / relative_dir
    if not root.is_dir():
        return ""

    def _walk(
        node: Traversable, parts: tuple[str, ...]
    ) -> list[tuple[tuple[str, ...], Traversable]]:
        collected: list[tuple[tuple[str, ...], Traversable]] = []
        for child in node.iterdir():
            child_parts = parts + (child.name,)
            if child.is_dir():
                collected.extend(_walk(child, child_parts))
            elif child.is_file() and child.name.endswith(".md"):
                collected.append((child_parts, child))
        return collected

    entries = _walk(root, ())
    entries.sort(key=lambda item: item[0])
    return "\n\n".join(resource.read_text(encoding="utf-8") for _, resource in entries)
