# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Data loading and model resolution helpers (OSS version, local filesystem)."""

__all__ = [
    "format_prompt",
    "load_data",
    "report_progress",
    "resolve_model_path",
]

# pyre-strict

import builtins
import json
import logging
from collections.abc import Sequence

_LG: logging.Logger = logging.getLogger(__name__)

try:
    from .fb.helpers import open_fn, report_progress, resolve_model_path
except ImportError:
    open_fn = builtins.open

    def resolve_model_path(model_path: str) -> str:
        """Resolve a model path. In OSS mode, returns the path as-is (local only)."""
        return model_path

    def report_progress(step: int, total_steps: int) -> None:
        """Report training progress. In OSS mode, does nothing."""
        pass


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_jsonl(path: str) -> list[dict[str, str]]:
    """Load a single JSONL file."""
    samples = []
    with open_fn(path, "r") as file:
        for line in file:
            if line := line.strip():
                samples.append(json.loads(line))
    return samples


def load_data(paths: Sequence[str]) -> list[dict[str, str]]:
    """Load and concatenate data from one or more JSONL files."""
    samples: list[dict[str, str]] = []
    for path in paths:
        _LG.info("Loading data from %s", path)
        smpls = _load_jsonl(path)
        _LG.info("Loaded %d samples from %s", len(smpls), path)
        samples.extend(smpls)
    _LG.info("Total: %d samples from %d file(s)", len(samples), len(paths))
    return samples


def format_prompt(sample: dict[str, str]) -> str:
    """Format an Alpaca-style sample into a single prompt string."""
    instruction = sample["instruction"]
    inp = sample.get("input", "")
    output = sample["output"]
    if inp:
        return (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{inp}\n\n"
            f"### Response:\n{output}"
        )
    return f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
