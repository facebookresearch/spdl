# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""SPDL pipeline builder for LLM fine-tuning."""

from __future__ import annotations

import threading
from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING, TypedDict

import spdl.pipeline
import spdl.source.utils
from spdl.io import transfer_tensor
from spdl.pipeline import PipelineBuilder
from spdl.pipeline.defs import PipelineConfig
from spdl.source import DistributedRandomSampler
from torch import Tensor

from .utils import _collate, _tokenize_sample

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


# subclass threading.local for better type inference
class ThreadLocalTokenizer(threading.local):
    def __init__(self, model_path: str) -> None:
        self._model_path = model_path
        self._tokenizer: "PreTrainedTokenizerBase | None" = None

    @property
    def tokenizer(self) -> "PreTrainedTokenizerBase":
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(self._model_path)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            self._tokenizer = tok
        assert self._tokenizer is not None
        return self._tokenizer


class _Tokenize:
    """Picklable callable that tokenizes a sample using a thread-local tokenizer.

    Uses thread-local storage so each SPDL worker thread gets its own
    tokenizer copy, since the HuggingFace fast tokenizer's Rust backend
    is not thread-safe. Implements ``__getstate__``/``__setstate__`` so
    the callable can be pickled for ``run_pipeline_in_subprocess``.
    """

    def __init__(self, model_path: str, max_seq_len: int) -> None:
        self.model_path = model_path
        self.max_seq_len = max_seq_len
        self._tlt = ThreadLocalTokenizer(self.model_path)

    class _State(TypedDict):
        model_path: str
        max_seq_len: int

    def __getstate__(self) -> _State:
        # threading.local is not picklable; reconstruct it in the subprocess.
        return {"model_path": self.model_path, "max_seq_len": self.max_seq_len}

    def __setstate__(self, state: _State) -> None:
        self.model_path = state["model_path"]
        self.max_seq_len = state["max_seq_len"]
        self._tlt = ThreadLocalTokenizer(self.model_path)

    def __call__(self, sample: dict[str, str]) -> dict[str, Tensor]:
        return _tokenize_sample(sample, self._tlt.tokenizer, self.max_seq_len)


class _DataLoaderWrapper(Iterable[dict[str, Tensor]]):
    """Wraps a SPDL pipeline into an iterable."""

    def __init__(self, config: PipelineConfig, num_threads: int) -> None:
        self.config = config
        self.num_threads = num_threads

    def __iter__(self) -> Iterator[dict[str, Tensor]]:
        pipeline = spdl.pipeline.build_pipeline(
            self.config, num_threads=self.num_threads
        )

        with pipeline.auto_stop():
            yield from pipeline.get_iterator(timeout=120)


def build_spdl_dataloader(
    samples: list[dict[str, str]],
    tokenizer: PreTrainedTokenizerBase,
    max_seq_len: int,
    batch_size: int,
    rank: int,
    world_size: int,
    num_threads: int,
    mp_context: str = "forkserver",
) -> Iterable[dict[str, Tensor]]:
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
    sampler = DistributedRandomSampler(
        len(samples),
        rank=rank,
        world_size=world_size,
    )
    source = spdl.source.utils.embed_shuffle(sampler)

    model_path: str = tokenizer.name_or_path

    backend = (
        PipelineBuilder()
        .add_source(source)
        .pipe(_Lookup(samples))
        .pipe(_Tokenize(model_path, max_seq_len), concurrency=num_threads)
        .aggregate(batch_size)
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
        .add_source(source2)
        .pipe(transfer_tensor)
        .add_sink(buffer_size=3)
    )

    return _DataLoaderWrapper(frontend.get_config(), num_threads=1)
