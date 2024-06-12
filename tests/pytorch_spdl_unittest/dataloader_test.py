from collections.abc import Iterator
from typing import TypeVar

import pytest

import torch
from pytorch_spdl.dataloader import DataLoader
from torch.utils.data import DataLoader as PyTorchDataLoader, IterableDataset


T = TypeVar("T")


def test_dataloader_invalid_args():
    """DataLoader rejects invalid combinations of args"""
    dummy = object()

    with pytest.raises(ValueError):
        DataLoader(dummy, sampler=dummy, batch_sampler=dummy)

    with pytest.raises(ValueError):
        DataLoader(dummy, num_workers=-1)

    with pytest.raises(ValueError):
        DataLoader(dummy, prefetch_factor=0)


async def passthrough(vals: T) -> T:
    return vals


class _SizedIterable(IterableDataset):
    def __init__(self, n):
        super().__init__()
        self.n = n

    def __iter__(self) -> Iterator[int]:
        yield from iter(range(self.n))

    def __len__(self) -> int:
        return self.n


def test_iterable_dataset():
    """Simple smoke test for IterableDataset."""

    dataloader = iter(
        DataLoader(_SizedIterable(10), batch_size=3, collate_fn=passthrough)
    )

    val = next(dataloader)
    assert val == [0, 1, 2]
    val = next(dataloader)
    assert val == [3, 4, 5]
    val = next(dataloader)
    assert val == [6, 7, 8]
    val = next(dataloader)
    assert val == [9]

    with pytest.raises(StopIteration):
        next(dataloader)


def test_iterable_len():
    ref = PyTorchDataLoader(_SizedIterable(10), batch_size=3)
    assert len(ref) == 4

    dataloader = DataLoader(_SizedIterable(10), batch_size=3)
    assert len(dataloader) == 4


def test_iterable_reference():
    ref = iter(PyTorchDataLoader(_SizedIterable(10), batch_size=3))

    dataloader = iter(DataLoader(_SizedIterable(10), batch_size=3))
    for _ in range(4):
        v = next(dataloader)
        ref_v = next(ref)
        assert torch.all(v == ref_v)


class _SizedMap:
    def __init__(self, n):
        super().__init__()
        self.n = n

    def __getitem__(self, i) -> int:
        return i

    def __len__(self) -> int:
        return self.n


def test_map_style_dataset():
    """Simple smoke test for map style dataset."""
    dataset = _SizedMap(10)
    dataloader = iter(DataLoader(dataset, batch_size=3, collate_fn=passthrough))

    val = next(dataloader)
    assert val == [0, 1, 2]
    val = next(dataloader)
    assert val == [3, 4, 5]
    val = next(dataloader)
    assert val == [6, 7, 8]
    val = next(dataloader)
    assert val == [9]

    with pytest.raises(StopIteration):
        next(dataloader)


def test_map_len():
    ref = PyTorchDataLoader(_SizedMap(10), batch_size=3)
    assert len(ref) == 4

    dataloader = DataLoader(_SizedMap(10), batch_size=3)
    assert len(dataloader) == 4


def test_map_reference():
    ref = iter(PyTorchDataLoader(_SizedMap(10), batch_size=3))

    dataloader = iter(DataLoader(_SizedMap(10), batch_size=3))
    for i in range(4):
        v = next(dataloader)
        ref_v = next(ref)
        assert torch.all(v == ref_v)
