# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import multiprocessing as mp
import os
import pickle
import time
import unittest
from collections.abc import Iterator
from multiprocessing.shared_memory import SharedMemory
from unittest.mock import patch

from spdl.dataloader import get_pytorch_dataloader, PyTorchDataLoader
from spdl.dataloader._pytorch_dataloader import _serialize_dataset
from torch.utils.data import Dataset, Sampler


class _RangeDataset(Dataset[int]):
    def __init__(self, size: int) -> None:
        self._size = size

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, index: int) -> int:
        return index


class _KeyedDataset(Dataset[str]):
    def __getitem__(self, index: str) -> str:
        return index.upper()


class _KeySampler(Sampler[str]):
    def __iter__(self) -> Iterator[str]:
        return iter(("a", "b", "c"))

    def __len__(self) -> int:
        return 3


class _BatchedKeyedDataset(_KeyedDataset):
    def __getitems__(self, indices: list[str]) -> list[str]:
        return [f"batch:{index}" for index in indices]


def _collate_strings(values: list[str]) -> tuple[str, ...]:
    return tuple(values)


def _convert_string(value: str) -> str:
    return f"item:{value}"


class _PageRoundedSharedMemory(SharedMemory):
    """Mimic platforms (e.g. macOS) that round the block up to the page size."""

    def __init__(
        self, name: str | None = None, create: bool = False, size: int = 0
    ) -> None:
        if create:
            size += 4096
        super().__init__(name=name, create=create, size=size)


class _SlowUnpickleDataset(Dataset[int]):
    def __init__(self, state, lock, process_ids, size: int = 16) -> None:
        self._state = state
        self._lock = lock
        self._process_ids = process_ids
        self._size = size

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state) -> None:
        self.__dict__.update(state)
        with self._lock:
            self._state["active"] += 1
            self._state["peak"] = max(self._state["peak"], self._state["active"])
            self._process_ids.append(os.getpid())
        try:
            time.sleep(0.25)
        finally:
            with self._lock:
                self._state["active"] -= 1

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, index: int) -> int:
        return index


class PyTorchDataLoaderTest(unittest.TestCase):
    def test_generic_keys_and_collation_in_spawned_workers(self) -> None:
        """Custom keys and collated output types survive the worker boundary."""
        loader: PyTorchDataLoader[tuple[str, ...]] = get_pytorch_dataloader(
            _KeyedDataset(),
            sampler=_KeySampler(),
            batch_size=2,
            collate_fn=_collate_strings,
            multiprocessing_context="spawn",
            timeout=30,
            in_order=True,
        )
        self.addCleanup(loader._shmem.unlink)
        self.addCleanup(loader._shmem.close)
        self.assertEqual(list(loader), [("A", "B"), ("C",)])

    def test_batched_dataset_fetch_in_spawned_workers(self) -> None:
        """Batch fetching uses the dataset's optional getitems implementation."""
        loader: PyTorchDataLoader[tuple[str, ...]] = get_pytorch_dataloader(
            _BatchedKeyedDataset(),
            sampler=_KeySampler(),
            batch_size=2,
            collate_fn=_collate_strings,
            multiprocessing_context="spawn",
            timeout=30,
            in_order=True,
        )
        self.addCleanup(loader._shmem.unlink)
        self.addCleanup(loader._shmem.close)
        self.assertEqual(list(loader), [("batch:a", "batch:b"), ("batch:c",)])

    def test_scalar_collation_in_spawned_workers(self) -> None:
        """Disabling batching passes each sample to the conversion callback."""
        loader: PyTorchDataLoader[str] = get_pytorch_dataloader(
            _KeyedDataset(),
            sampler=_KeySampler(),
            batch_size=None,
            collate_fn=_convert_string,
            multiprocessing_context="spawn",
            timeout=30,
            in_order=True,
        )
        self.addCleanup(loader._shmem.unlink)
        self.addCleanup(loader._shmem.close)
        self.assertEqual(list(loader), ["item:A", "item:B", "item:C"])

    def test_serialize_dataset_with_larger_shared_memory_block(self) -> None:
        """The dataset is serialized when the block is larger than requested."""
        dataset = _RangeDataset(3)
        with patch(
            "spdl.dataloader._pytorch_dataloader.SharedMemory",
            _PageRoundedSharedMemory,
        ):
            shmem = _serialize_dataset(dataset)
        self.addCleanup(shmem.unlink)
        self.addCleanup(shmem.close)

        self.assertGreater(shmem.size, len(pickle.dumps(dataset)))
        buf = shmem.buf
        assert buf is not None
        restored = pickle.loads(buf)
        self.assertEqual([restored[i] for i in range(len(restored))], [0, 1, 2])

    def test_zero_length_custom_sampler_is_preserved(self) -> None:
        """A falsy custom sampler must not be replaced by the default sampler."""

        class _EmptySampler(Sampler[int]):
            def __iter__(self) -> Iterator[int]:
                return iter(())

            def __len__(self) -> int:
                return 0

        dataset = _SlowUnpickleDataset(None, None, None, size=3)
        sampler = _EmptySampler()

        loader: PyTorchDataLoader[object] = get_pytorch_dataloader(
            dataset,
            batch_size=1,
            sampler=sampler,
            num_workers=0,
        )

        self.addCleanup(loader._shmem.unlink)
        self.addCleanup(loader._shmem.close)
        self.assertEqual(list(loader), [])

    def test_rejects_nonpositive_worker_init_concurrency(self) -> None:
        """Worker initialization concurrency must be positive when set."""
        with self.assertRaisesRegex(
            ValueError, "`worker_init_concurrency` must be greater than 0"
        ):
            get_pytorch_dataloader(
                _SlowUnpickleDataset(None, None, None, size=1),
                worker_init_concurrency=0,
            )

    def test_bounds_spawned_worker_initialization_concurrency(self) -> None:
        """Spawned workers deserialize the dataset within the configured bound."""
        context = mp.get_context("spawn")
        with context.Manager() as manager:
            state = manager.dict(active=0, peak=0)
            lock = manager.Lock()
            process_ids = manager.list()
            dataset = _SlowUnpickleDataset(state, lock, process_ids)
            loader: PyTorchDataLoader[object] = get_pytorch_dataloader(
                dataset,
                batch_size=1,
                num_workers=4,
                timeout=60,
                multiprocessing_context=context,
                worker_init_concurrency=2,
            )

            self.addCleanup(loader._shmem.unlink)
            self.addCleanup(loader._shmem.close)
            self.assertEqual(len(list(loader)), len(dataset))
            self.assertEqual(len(set(process_ids)), 4)
            self.assertLessEqual(state["peak"], 2)
