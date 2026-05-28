# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Video classification training with SPDL data pipeline.

Trains a video classification model (R3D-18) on video data using SPDL
PipelineBuilder for high-performance concurrent video decoding and GPU
transfer.

SPDL Data Pipeline
^^^^^^^^^^^^^^^^^^

Uses GPU NVDEC hardware decoding in a multithreaded pipeline:

  Sampling → fetch → demux → GPU decode (NVDEC) → aggregate → collate.

The data source is pluggable: local video files (OSS) or WarmStorage
via BulkDataset (Meta-internal), selected automatically via the
compatibility layer in ``utils/__init__.py``.

Usage
^^^^^

::

    torchrun \\
      --nproc_per_node 8 \\
      -m spdl.examples.video_classification.video_classification \\
      --data-dir /path/to/kinetics-400 \\
      --num-classes 400
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import time
from collections.abc import Callable, Iterator
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.models.video import r3d_18

try:
    from examples.video_classification.utils import (  # pyre-ignore[21]
        add_dataset_args,
        build_pipeline,
        create_dataset,
        get_label_to_index,
        report_progress,
    )
except ImportError:
    from spdl.examples.video_classification.utils import (
        add_dataset_args,
        build_pipeline,
        create_dataset,
        get_label_to_index,
        report_progress,
    )

_LG: logging.Logger = logging.getLogger(__name__)

type _TBatch = dict[str, torch.Tensor]


def train(
    *,
    dataloader: Iterator[_TBatch],
    num_classes: int,
    batch_size: int,
    num_epochs: int,
    lr: float,
    weight_decay: float,
    max_grad_norm: float,
    log_interval: int,
    max_steps: int | None = None,
    progress_fn: Callable[[int, int | None], None] | None = None,
) -> None:
    """Main training function, called per-rank."""
    rank: int = dist.get_rank()
    world_size: int = dist.get_world_size()
    local_rank: int = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    gc.disable()

    _LG.info("Rank %d/%d on device %s", rank, world_size, device)

    # --- Model ---
    _LG.info("Building R3D-18 model with %d classes", num_classes)
    model = r3d_18(num_classes=num_classes).to(device=device, dtype=torch.float32)
    ddp_model = DDP(model, device_ids=[local_rank])

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        ddp_model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        fused=True,
    )

    if rank == 0 and progress_fn is not None:
        progress_fn(0, None)

    # --- Training loop ---
    global_step = 0
    ddp_model.train()
    for epoch in range(num_epochs):
        _LG.info("Epoch %d/%d", epoch + 1, num_epochs)

        t0 = time.monotonic()
        epoch_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            if max_steps is not None and num_batches >= max_steps:
                break

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = ddp_model(batch["video"])
                loss = torch.nn.functional.cross_entropy(outputs, batch["label"])

            loss.backward()
            torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

            if global_step % 50 == 0:
                gc.collect()

            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1

            if rank == 0:
                if progress_fn is not None:
                    progress_fn(global_step, None)
                if global_step % log_interval == 0:
                    avg_loss = epoch_loss / num_batches
                    elapsed = time.monotonic() - t0
                    _LG.info(
                        "Step %d | loss=%.4f | %.1f samples/s",
                        global_step,
                        avg_loss,
                        num_batches * batch_size * world_size / elapsed,
                    )

        elapsed = time.monotonic() - t0
        if rank == 0:
            avg_loss = epoch_loss / max(num_batches, 1)
            _LG.info(
                "Epoch %d complete | avg_loss=%.4f | %.1fs | %.1f samples/s",
                epoch + 1,
                avg_loss,
                elapsed,
                num_batches * batch_size * world_size / elapsed,
            )

    gc.enable()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    # Model
    parser.add_argument(
        "--num-classes", type=int, default=400, help="Number of classification classes"
    )
    # Video
    parser.add_argument(
        "--num-frames",
        type=int,
        default=16,
        help="Number of frames to sample per video",
    )
    parser.add_argument("--frame-height", type=int, default=112)
    parser.add_argument("--frame-width", type=int, default=112)
    parser.add_argument(
        "--subclip-duration",
        type=float,
        default=None,
        help="Temporal subclip duration in seconds. If set, only the first N seconds of each video are demuxed and decoded.",
    )
    # Training
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Max training steps per epoch (for benchmarking)",
    )
    # Pipeline
    parser.add_argument(
        "--num-fetch-threads",
        type=int,
        default=8,
        help="Concurrent data fetch threads",
    )
    parser.add_argument(
        "--num-decode-threads",
        type=int,
        default=16,
        help="Concurrent video decode threads",
    )
    # Dataset-specific args (OSS or FB, depending on which module is available)
    add_dataset_args(parser)
    return parser.parse_args()


def init_logging() -> None:
    rank = os.environ.get("RANK", "?")
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [%(levelname).1s] [Rank{rank}] %(name)s: %(message)s",
        force=True,
    )


def main(args: argparse.Namespace) -> None:
    _LG.info("Building label-to-index mapping ...")
    label_to_index = get_label_to_index(args)
    _LG.info("Found %d classes", len(label_to_index))

    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=30))
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        _LG.info("Creating dataset ...")
        dataset = create_dataset(args)

        _LG.info("Building data pipeline ...")
        dataloader = build_pipeline(
            args=args,
            dataset=dataset,
            label_to_index=label_to_index,
            rank=rank,
            world_size=world_size,
        )

        train(
            dataloader=dataloader,
            num_classes=args.num_classes,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            max_grad_norm=args.max_grad_norm,
            log_interval=args.log_interval,
            max_steps=args.max_steps,
            progress_fn=report_progress,
        )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    init_logging()
    main(parse_args())
