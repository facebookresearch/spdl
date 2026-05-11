# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Source control abstraction for autoresearch.

Detects whether the source directory uses Sapling (sl) or Git, and
provides commit/goto/_current_commit operations. The autoresearch system
uses this to commit code changes with descriptive messages and to revert
to earlier experiment commits when branching out.

An "anchor commit" recorded at init time is the boundary — the system
must never go back beyond it.
"""

from __future__ import annotations

import logging
import shutil
import subprocess

_LG: logging.Logger = logging.getLogger(__name__)

__all__ = [
    "_commit",
    "_current_commit",
    "_detect_scm",
    "_goto",
    "_has_pending_changes",
]


def _run(cmd: list[str], cwd: str) -> subprocess.CompletedProcess:
    _LG.debug("scm: %s (cwd=%s)", cmd, cwd)
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    _LG.debug("scm: rc=%d stdout=%s", result.returncode, result.stdout.strip()[:200])
    if result.returncode != 0:
        _LG.debug("scm: stderr=%s", result.stderr.strip()[:500])
    return result


def _detect_scm(source_dir: str) -> str:
    """Detect which SCM is available. Returns 'sl' or 'git'."""
    if shutil.which("sl"):
        result = _run(["sl", "root"], source_dir)
        if result.returncode == 0:
            _LG.info("Detected SCM: sl (Sapling)")
            return "sl"
    if shutil.which("git"):
        result = _run(["git", "rev-parse", "--git-dir"], source_dir)
        if result.returncode == 0:
            _LG.info("Detected SCM: git")
            return "git"
    raise RuntimeError(
        f"No supported SCM found in {source_dir}. Install sl (Sapling) or git."
    )


def _current_commit(scm: str, source_dir: str) -> str:
    """Return the current commit hash."""
    if scm == "sl":
        result = _run(["sl", "log", "-r", ".", "-T", "{node}"], source_dir)
    else:
        result = _run(["git", "rev-parse", "HEAD"], source_dir)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to get current commit: {result.stderr}")
    return result.stdout.strip()


def _commit(scm: str, source_dir: str, message: str) -> str:
    """Commit all pending changes with the given message. Returns commit hash."""
    if scm == "sl":
        result = _run(["sl", "commit", "-m", message], source_dir)
    else:
        _run(["git", "add", "-A"], source_dir)
        result = _run(["git", "commit", "-m", message], source_dir)

    if result.returncode != 0:
        if "nothing changed" in result.stdout or "nothing to commit" in result.stdout:
            _LG.info("Nothing to commit")
            return _current_commit(scm, source_dir)
        raise RuntimeError(f"Commit failed: {result.stderr}")

    new_hash = _current_commit(scm, source_dir)
    _LG.info("Committed %s: %s", new_hash[:12], message[:80])
    return new_hash


def _goto(scm: str, source_dir: str, target: str, anchor: str) -> None:
    """Go to a target commit. Refuses to go before the anchor commit."""
    if not _is_descendant(scm, source_dir, target, anchor):
        raise ValueError(
            f"Refusing to go to {target[:12]}: it is not a descendant of "
            f"anchor commit {anchor[:12]}."
        )

    _LG.info("Going to commit %s", target[:12])
    if scm == "sl":
        result = _run(["sl", "goto", target], source_dir)
    else:
        result = _run(["git", "checkout", target], source_dir)

    if result.returncode != 0:
        raise RuntimeError(f"Failed to go to {target[:12]}: {result.stderr}")


def _has_pending_changes(scm: str, source_dir: str) -> bool:
    """Check if there are uncommitted changes."""
    if scm == "sl":
        result = _run(["sl", "status"], source_dir)
    else:
        result = _run(["git", "status", "--porcelain"], source_dir)
    return bool(result.stdout.strip())


def _is_descendant(scm: str, source_dir: str, commit_hash: str, ancestor: str) -> bool:
    """Check if commit_hash is a descendant of (or equal to) ancestor."""
    if commit_hash == ancestor:
        return True
    if scm == "sl":
        result = _run(
            ["sl", "log", "-r", f"{ancestor}::{commit_hash}", "-T", "{node}\n"],
            source_dir,
        )
        return ancestor in result.stdout
    else:
        result = _run(
            ["git", "merge-base", "--is-ancestor", ancestor, commit_hash],
            source_dir,
        )
        return result.returncode == 0
