# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""SPDL pipeline construction for video classification.

Provides ``build_pipeline`` which assembles a GPU NVDEC-accelerated SPDL
pipeline using Multi-Threading in subprocess (MTP)::

    Backend (subprocess): sample → fetch → disaggregate → demux (CPU)
    Frontend (main process): GPU decode (NVDEC) → aggregate → collate

Demuxed packets are serialized across the process boundary, isolating
CPU-intensive demux work from CUDA kernel scheduling in the training
process.

The pipeline is dataset-agnostic: any dataset whose ``__getitem__`` returns
``list[{"video_bytes": bytes, "label": str}]`` can be used.
"""

from __future__ import annotations

import argparse
import logging
import os
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Protocol, TypedDict

import spdl.io
import spdl.pipeline
import spdl.source.utils
import torch
from spdl.pipeline import PipelineBuilder
from spdl.source import DistributedRandomSampler

_LG: logging.Logger = logging.getLogger(__name__)

type _TBatch = dict[str, torch.Tensor]


class _RawSample(TypedDict):
    video_bytes: bytes
    label: str


class _DemuxedSample(TypedDict):
    packets: "spdl.io.VideoPackets"
    label: int


class VideoDataset(Protocol):
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> list[_RawSample]: ...


# ---------------------------------------------------------------------------
# Backend (subprocess) stage functions — must be picklable.
# Using module-level functions + functools.partial.
# ---------------------------------------------------------------------------


def _fetch_sample(index: int, *, dataset: object) -> list[_RawSample]:
    return dataset[index]  # pyre-ignore[16]


def _demux_sample(
    sample: _RawSample,
    *,
    label_to_index: dict[str, int],
    subclip_duration: float | None = None,
) -> _DemuxedSample | None:
    video_bytes = sample["video_bytes"]
    try:
        timestamp = (0.0, subclip_duration) if subclip_duration else None
        packets = spdl.io.demux_video(video_bytes, timestamp=timestamp)
    except RuntimeError as e:
        _LG.warning("Demux failed: %s", e)
        return None
    label = label_to_index[sample["label"]]
    return {"packets": packets, "label": label}


# ---------------------------------------------------------------------------
# Frontend (main process) — GPU decode, runs in the training process.
# ---------------------------------------------------------------------------


class NvdecDecode:
    """Decode pre-demuxed video packets using GPU NVDEC hardware decoder."""

    def __init__(
        self,
        num_frames: int,
        cuda_cfg: "spdl.io.CUDAConfig",
        width: int,
        height: int,
        device: torch.device,
    ) -> None:
        self.num_frames = num_frames
        self.cuda_cfg = cuda_cfg
        self.width = width
        self.height = height
        self.device = device

    def __call__(self, sample: _DemuxedSample | None) -> dict[str, torch.Tensor] | None:
        if sample is None:
            return None

        packets = sample["packets"]
        try:
            buffer = spdl.io.decode_packets_nvdec(
                packets,
                device_config=self.cuda_cfg,
                scale_width=self.width,
                scale_height=self.height,
                pix_fmt="rgb",
            )
            tensor = spdl.io.to_torch(buffer)  # [T, C, H, W], already on GPU
        except RuntimeError as e:
            _LG.warning("NVDEC decode failed: %s", e)
            return None

        # Frame sampling
        if len(tensor) >= self.num_frames:
            indices = torch.linspace(0, len(tensor) - 1, self.num_frames).long()
            tensor = tensor[indices]
        else:
            padding = tensor[-1:].expand(self.num_frames - len(tensor), -1, -1, -1)
            tensor = torch.cat([tensor, padding], dim=0)

        # [T, C, H, W] -> [C, T, H, W], normalize
        tensor = tensor.permute(1, 0, 2, 3).float() / 255.0

        label = sample["label"]
        return {
            "video": tensor,
            "label": torch.tensor(label, dtype=torch.long, device=self.device),
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
    """Build an MTP pipeline with subprocess demux and GPU NVDEC decode.

    Splits the pipeline into a backend subprocess (CPU-only: fetch,
    disaggregate, demux) and a frontend main process (GPU NVDEC decode,
    aggregate, collate).  Demuxed packets are serialized across the
    process boundary, isolating CPU work from CUDA kernel scheduling.

    Pipeline stages::

        Backend (subprocess): sample → fetch → disaggregate → demux
        Frontend (main process): NVDEC decode → aggregate → collate

    Args:
        args: CLI args providing ``num_fetch_threads``, ``num_decode_threads``,
            ``num_demux_threads``, ``num_frames``, ``frame_width``,
            ``frame_height``, ``batch_size``, and ``subclip_duration``.
        dataset: Any object implementing the ``VideoDataset`` protocol.
        label_to_index: Mapping from label string to class index.
        rank: Current distributed rank.
        world_size: Total number of distributed workers.

    Returns:
        An iterator yielding ``{"video": Tensor, "label": Tensor}`` batches.
    """
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")

    cuda_cfg = spdl.io.cuda_config(
        device_index=local_rank,
        allocator=(
            torch.cuda.caching_allocator_alloc,
            torch.cuda.caching_allocator_delete,
        ),
    )

    num_fetch_threads = args.num_fetch_threads
    num_decode_threads = args.num_decode_threads
    num_demux_threads = args.num_demux_threads

    source = spdl.source.utils.embed_shuffle(
        DistributedRandomSampler(
            len(dataset),
            rank=rank,
            world_size=world_size,  # pyre-ignore[6]
        )
    )

    # Backend (subprocess) — CPU-only stages: fetch → disaggregate → demux
    backend = (
        PipelineBuilder()
        .add_source(source, continuous=True)
        .pipe(
            partial(_fetch_sample, dataset=dataset),
            concurrency=num_fetch_threads,
        )
        .disaggregate()
        .pipe(
            partial(
                _demux_sample,
                label_to_index=label_to_index,
                subclip_duration=args.subclip_duration,
            ),
            concurrency=num_demux_threads,
        )
        .add_sink(buffer_size=3)
    )

    source2 = spdl.pipeline.run_pipeline_in_subprocess(
        backend.get_config(),
        num_threads=max(num_fetch_threads, num_demux_threads),
        mp_context="forkserver",
    )

    # Frontend (main process) — GPU NVDEC decode → aggregate → collate
    decode_executor = ThreadPoolExecutor(max_workers=num_decode_threads)
    nvdec_decode = NvdecDecode(
        num_frames=args.num_frames,
        cuda_cfg=cuda_cfg,
        width=args.frame_width,
        height=args.frame_height,
        device=device,
    )

    frontend = (
        PipelineBuilder()
        .add_source(source2, continuous=True)
        .pipe(nvdec_decode, concurrency=num_decode_threads, executor=decode_executor)
        .aggregate(args.batch_size, drop_last=True)
        .pipe(collate)
        .add_sink(buffer_size=5)
    )
    pipeline = frontend.build(num_threads=2)
    return pipeline.get_iterator(timeout=300)
