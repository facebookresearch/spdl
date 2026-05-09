# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""OSS video dataset implementation.

Reads videos from a local directory organized as
``<root>/<split>/<class_name>/<video>.ext``.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

_LG: logging.Logger = logging.getLogger(__name__)

_VIDEO_EXTENSIONS: set[str] = {".avi", ".mp4", ".mkv", ".webm", ".mov"}


class LocalVideoDataset:
    """Video dataset backed by local files.

    Each ``__getitem__`` call reads a video file from disk and returns
    a single-element list of ``{"video_bytes": ..., "label": ...}``.
    Returning a list aligns the interface with variants that fetch
    data from a remote source in bulk.

    Expected directory layout::

        root/<split>/<class_name>/<video>.{avi,mp4,mkv,webm,mov}
    """

    def __init__(self, root: str, split: str = "train") -> None:
        split_dir = Path(root) / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        self.samples: list[tuple[str, str]] = []
        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            label = class_dir.name
            for video_file in sorted(class_dir.iterdir()):
                if video_file.suffix.lower() in _VIDEO_EXTENSIONS:
                    self.samples.append((str(video_file), label))

        if not self.samples:
            raise RuntimeError(
                f"No video files found in {split_dir}. "
                f"Expected layout: {split_dir}/<class_name>/<video>.ext"
            )
        _LG.info("Found %d videos in %s", len(self.samples), split_dir)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> list[dict[str, object]]:
        path, label = self.samples[index]
        with open(path, "rb") as f:
            video_bytes = f.read()
        return [{"video_bytes": video_bytes, "label": label}]


def create_dataset(args: argparse.Namespace) -> LocalVideoDataset:
    return LocalVideoDataset(args.data_dir, args.split)


def get_label_to_index(args: argparse.Namespace) -> dict[str, int]:
    """Derive label-to-index mapping from directory structure."""
    split_dir = Path(args.data_dir) / args.split
    labels = sorted(d.name for d in split_dir.iterdir() if d.is_dir())
    return {label: idx for idx, label in enumerate(labels)}


def add_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Root directory with video files organized as <split>/<class>/<video>.ext",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split subdirectory name",
    )


def report_progress(step: int, total_steps: int | None = None) -> None:
    pass
