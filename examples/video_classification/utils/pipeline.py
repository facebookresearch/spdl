# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""SPDL pipeline construction for video classification.

Provides the pipeline callables (Decode, collate) and ``build_pipeline``
which assembles the full SPDL pipeline.  The pipeline is dataset-agnostic:
any dataset whose ``__getitem__`` returns
``list[{"video_bytes": bytes, "label": str}]`` can be used.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from typing import Protocol

import spdl.io
import spdl.source.utils
import torch
from spdl.io import transfer_tensor
from spdl.pipeline import PipelineBuilder
from spdl.source import DistributedRandomSampler

type _TBatch = dict[str, torch.Tensor]


class VideoDataset(Protocol):
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> list[dict[str, object]]: ...


class Decode:
    """Decode video bytes into a CPU tensor of sampled frames (FFmpeg)."""

    def __init__(
        self,
        num_frames: int,
        filter_desc: str | None,
        label_to_index: dict[str, int],
    ) -> None:
        self.num_frames = num_frames
        self.filter_desc = filter_desc
        self.label_to_index = label_to_index

    def __call__(self, item: dict[str, object]) -> dict[str, torch.Tensor] | None:
        video_bytes = item["video_bytes"]
        label_str = item["label"]
        assert isinstance(video_bytes, (bytes, memoryview))
        assert isinstance(label_str, str)

        try:
            packets = spdl.io.demux_video(video_bytes)
            frames = spdl.io.decode_packets(packets, filter_desc=self.filter_desc)
            buffer = spdl.io.convert_frames(frames)
            tensor = spdl.io.to_torch(buffer)  # [N, H, W, C]
        except RuntimeError:
            return None

        total = tensor.shape[0]
        if total >= self.num_frames:
            indices = torch.linspace(0, total - 1, self.num_frames).long()
            tensor = tensor[indices]
        else:
            pad = tensor[-1:].expand(self.num_frames - total, -1, -1, -1)
            tensor = torch.cat([tensor, pad], dim=0)

        # [T, H, W, C] -> [C, T, H, W] for video models
        tensor = tensor.permute(3, 0, 1, 2).float() / 255.0

        label_index = self.label_to_index[label_str]
        return {
            "video": tensor,
            "label": torch.tensor(label_index),
        }


def collate(
    items: list[dict[str, torch.Tensor]],
) -> _TBatch:
    return {
        "video": torch.stack([it["video"] for it in items]),
        "label": torch.stack([it["label"] for it in items]),
    }


def build_pipeline(
    *,
    args: argparse.Namespace,
    dataset: VideoDataset,
    label_to_index: dict[str, int],
    rank: int,
    world_size: int,
) -> Iterator[_TBatch]:
    """Build a multithreaded SPDL pipeline for video classification.

    Constructs the following pipeline stages::

        sample → fetch → disaggregate → decode → aggregate → collate → transfer

    1. **Sample**: ``DistributedRandomSampler`` produces shuffled indices,
       partitioned across ranks.  The source is continuous (auto-resets
       each epoch).
    2. **Fetch** (``num_fetch_threads``): ``dataset.__getitem__`` returns
       a list of ``{"video_bytes": bytes, "label": str}`` dicts.
       For local files this is a single-element list; for remote bulk
       storage each call may return many items.
    3. **Disaggregate**: Splits each list into individual items.
    4. **Decode** (``num_decode_threads``): Decodes video bytes with FFmpeg,
       samples ``num_frames`` evenly, and produces ``[C, T, H, W]`` tensors.
    5. **Aggregate / collate**: Groups into batches and stacks tensors.
    6. **Transfer**: Async copy to GPU via ``spdl.io.transfer_tensor``.

    Args:
        args: CLI args providing ``num_fetch_threads``, ``num_decode_threads``,
            ``num_frames``, ``frame_width``, ``frame_height``, ``batch_size``.
        dataset: Any object implementing the ``VideoDataset`` protocol.
        label_to_index: Mapping from label string to class index.
        rank: Current distributed rank.
        world_size: Total number of distributed workers.

    Returns:
        An iterator yielding ``{"video": Tensor, "label": Tensor}`` batches.
    """
    num_fetch_threads: int = args.num_fetch_threads
    num_decode_threads: int = args.num_decode_threads

    filter_desc: str | None = spdl.io.get_video_filter_desc(
        scale_width=args.frame_width,
        scale_height=args.frame_height,
        pix_fmt="rgb24",
    )

    source = spdl.source.utils.embed_shuffle(
        DistributedRandomSampler(
            len(dataset),
            rank=rank,
            world_size=world_size,
        )
    )

    pipeline = (
        PipelineBuilder()
        .add_source(source, continuous=True)
        .pipe(dataset.__getitem__, concurrency=num_fetch_threads)
        .disaggregate()
        .pipe(
            Decode(args.num_frames, filter_desc, label_to_index),
            concurrency=num_decode_threads,
        )
        .aggregate(args.batch_size, drop_last=True)
        .pipe(collate)
        .pipe(transfer_tensor)
        .add_sink(buffer_size=3)
        .build(num_threads=num_fetch_threads + num_decode_threads)
    )
    return pipeline.get_iterator(timeout=300)
