# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import multiprocessing as mp
import os
import time
import unittest

from spdl.dataloader import get_pytorch_dataloader
from torch.utils.data import Dataset


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
            loader = get_pytorch_dataloader(
                dataset,
                batch_size=1,
                num_workers=4,
                timeout=60,
                multiprocessing_context=context,
                worker_init_concurrency=2,
            )

            self.assertEqual(len(list(loader)), len(dataset))
            self.assertEqual(len(set(process_ids)), 4)
            self.assertLessEqual(state["peak"], 2)
