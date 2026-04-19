# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""PyTorch DataLoader for LLM fine-tuning."""

from __future__ import annotations

__all__ = ["build_pytorch_dataloader"]

# pyre-strict
from collections.abc import Iterator
from typing import TYPE_CHECKING

import torch
import torch.utils.data

from .utils import _collate, _tokenize_sample

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


class _InstructDataset(torch.utils.data.Dataset):
    """Map-style dataset that tokenizes Alpaca-style samples on the fly."""

    def __init__(
        self,
        samples: list[dict[str, str]],
        tokenizer: PreTrainedTokenizerBase,
        max_seq_len: int,
    ) -> None:
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return _tokenize_sample(self.samples[idx], self.tokenizer, self.max_seq_len)


class _DeviceDataLoader:
    """Wraps a DataLoader to transfer each batch to a CUDA device on iteration.

    Automatically calls ``DistributedSampler.set_epoch`` on each iteration,
    ensuring proper reshuffling across epochs. Build once and reuse across
    epochs—no manual ``set_epoch`` call required.
    """

    def __init__(
        self,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device | None = None,
    ) -> None:
        self.dataloader = dataloader
        self.device = device
        self._epoch: int = 0

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        sampler = self.dataloader.sampler
        if isinstance(sampler, torch.utils.data.DistributedSampler):
            sampler.set_epoch(self._epoch)
        self._epoch += 1
        if self.device is not None:
            for batch in self.dataloader:
                yield {
                    k: v.to(self.device, non_blocking=True) for k, v in batch.items()
                }
        else:
            yield from self.dataloader


def build_pytorch_dataloader(
    samples: list[dict[str, str]],
    tokenizer: PreTrainedTokenizerBase,
    max_seq_len: int,
    batch_size: int,
    rank: int,
    world_size: int,
    num_workers: int,
    mp_context: str = "forkserver",
    device: torch.device | None = None,
) -> _DeviceDataLoader:
    """Build a reusable PyTorch DataLoader for distributed LLM fine-tuning.

    Build once before the training loop and reuse across epochs.
    The returned wrapper automatically calls ``DistributedSampler.set_epoch``
    on each iteration, so no manual ``set_epoch`` call is needed.
    """
    dataset = _InstructDataset(samples, tokenizer, max_seq_len)
    sampler = torch.utils.data.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )
    dl = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=_collate,
        pin_memory=True,
        prefetch_factor=3 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
        multiprocessing_context=mp_context if num_workers > 0 else None,
    )
    return _DeviceDataLoader(dl, device)
