# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""SPDL pipeline builder for LLM fine-tuning."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import spdl.pipeline
import spdl.source.utils
from spdl.io import transfer_tensor
from spdl.pipeline import PipelineBuilder
from spdl.source import DistributedRandomSampler

from .utils import _collate, _TDataLoader, _tokenize_sample, _TSample

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

__all__ = [
    "build_spdl_dataloader",
]


class _Lookup:
    """Picklable callable that looks up a sample by index."""

    def __init__(self, samples: list[dict[str, str]]) -> None:
        self.samples = samples

    def __call__(self, idx: int) -> dict[str, str]:
        return self.samples[idx]


class _ThreadLocalTokenizer(threading.local):
    """Thread-local tokenizer that deepcopies from a source tokenizer.

    Each thread gets its own copy since the HuggingFace fast tokenizer's
    Rust backend is not thread-safe.
    """

    def __init__(self, source: "PreTrainedTokenizerBase") -> None:
        self._source = source
        self._tokenizer: "PreTrainedTokenizerBase | None" = None

    @property
    def tokenizer(self) -> "PreTrainedTokenizerBase":
        if self._tokenizer is None:
            import copy

            self._tokenizer = copy.deepcopy(self._source)
        return self._tokenizer


class _Tokenize:
    """Picklable callable that tokenizes a sample using a thread-local tokenizer.

    The tokenizer object is pickled to the subprocess (avoiding re-import
    of ``transformers`` which can fail in PAR/XAR subprocess environments).
    Each thread gets its own copy via ``_ThreadLocalTokenizer``.
    """

    def __init__(self, tokenizer: "PreTrainedTokenizerBase", max_seq_len: int) -> None:
        self._tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self._tlt = _ThreadLocalTokenizer(tokenizer)

    def __getstate__(self) -> dict[str, object]:
        # threading.local is not picklable, but the tokenizer is.
        return {"tokenizer": self._tokenizer, "max_seq_len": self.max_seq_len}

    def __setstate__(self, state: dict[str, object]) -> None:
        self._tokenizer = state["tokenizer"]  # pyre-ignore[8]
        self.max_seq_len = state["max_seq_len"]  # pyre-ignore[8]
        self._tlt = _ThreadLocalTokenizer(self._tokenizer)

    def __call__(self, sample: dict[str, str]) -> _TSample:
        return _tokenize_sample(sample, self._tlt.tokenizer, self.max_seq_len)


def build_spdl_dataloader(
    samples: list[dict[str, str]],
    tokenizer: PreTrainedTokenizerBase,
    max_seq_len: int,
    batch_size: int,
    rank: int,
    world_size: int,
    num_threads: int,
    mp_context: str = "forkserver",
) -> _TDataLoader:
    """Build a reusable SPDL data loader with nested pipeline architecture.

    Creates two nested pipelines to separate CPU-bound data loading from
    GPU transfer, reducing the noisy-neighbour effect where data loading
    threads in the main process compete with the training loop for CPU
    time, delaying GPU kernel launches.

    **Inner pipeline** (runs in a subprocess):
      Sampling → lookup → tokenize (concurrent) → aggregate → collate.
      All CPU work runs in a dedicated subprocess with its own thread pool,
      completely isolating it from the training process. The subprocess is
      created once and reused across epochs — each ``for ... in`` call
      rebuilds the pipeline inside the same subprocess.

    **Outer pipeline** (runs in the main process):
      Receives CPU batches from the subprocess via IPC queue and transfers
      them to GPU using ``transfer_tensor`` with a dedicated single-thread
      executor. This ensures GPU transfer uses a consistent CUDA stream
      and overlaps with training computation.

    Build once before the training loop and iterate each epoch::

        dataloader = build_spdl_dataloader(samples, tokenizer, ...)
        for epoch in range(num_epochs):
            for batch in dataloader:
                train(batch)
    """
    source = spdl.source.utils.embed_shuffle(
        DistributedRandomSampler(
            len(samples),
            rank=rank,
            world_size=world_size,
        )
    )

    backend = (
        PipelineBuilder()
        .add_source(source, continuous=True)
        .pipe(_Lookup(samples))
        .pipe(_Tokenize(tokenizer, max_seq_len), concurrency=num_threads)
        .aggregate(batch_size, drop_last=True)
        .pipe(_collate)
        .add_sink(buffer_size=3)
    )

    source2 = spdl.pipeline.run_pipeline_in_subprocess(
        backend.get_config(),
        num_threads=num_threads,
        mp_context=mp_context,
    )

    frontend = (
        PipelineBuilder()
        .add_source(source2, continuous=True)
        .pipe(transfer_tensor)
        .add_sink(buffer_size=3)
    )
    return frontend.build(num_threads=1)
