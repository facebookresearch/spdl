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


def passthrough(vals: T) -> T:
    print(f"passthrough: {vals=}")
    return vals


class _Iterable(IterableDataset):
    def __init__(self, n):
        super().__init__()
        self.n = n

    def __iter__(self) -> Iterator[int]:
        offset = 10
        for val in range(offset, offset + self.n):
            print(f"_Iterable: {val=}")
            yield val


class _SizedIterable(_Iterable):
    def __init__(self, n):
        super().__init__(n)

    def __len__(self) -> int:
        return self.n


class _MapStyle:
    def __init__(self, n):
        self.n = n

    def __getitem__(self, i) -> int:
        return 10 + i

    def __len__(self) -> int:
        return self.n


def test_iterable_dataset():
    """Simple smoke test for IterableDataset."""

    dataloader = iter(
        DataLoader(_SizedIterable(10), batch_size=3, collate_fn=passthrough)
    )

    val = next(dataloader)
    assert val == [10, 11, 12]
    val = next(dataloader)
    assert val == [13, 14, 15]
    val = next(dataloader)
    assert val == [16, 17, 18]
    val = next(dataloader)
    assert val == [19]

    with pytest.raises(StopIteration):
        next(dataloader)


def test_map_style_dataset():
    """Simple smoke test for map style dataset."""
    dataset = _MapStyle(10)
    dataloader = iter(DataLoader(dataset, batch_size=3, collate_fn=passthrough))

    val = next(dataloader)
    assert val == [10, 11, 12]
    val = next(dataloader)
    assert val == [13, 14, 15]
    val = next(dataloader)
    assert val == [16, 17, 18]
    val = next(dataloader)
    assert val == [19]

    with pytest.raises(StopIteration):
        next(dataloader)


@pytest.mark.parametrize(
    "sized,drop_last",
    [
        (False, False),
        (False, True),
        (True, False),
        (True, True),
    ],
)
def test_compatibility_iterable_batched(sized, drop_last):
    batch_size = 3
    num_samples = 10
    num_batches = 3 if drop_last else 4
    ds = _SizedIterable(num_samples) if sized else _Iterable(num_samples)

    ref = PyTorchDataLoader(
        ds, batch_size=batch_size, drop_last=drop_last, num_workers=0
    )
    hyp = DataLoader(ds, batch_size=batch_size, drop_last=drop_last)

    if sized:
        assert len(ref) == len(hyp) == num_batches

    num_iter = 0
    for ref_val, hyp_val in zip(ref, hyp, strict=True):
        num_iter += 1
        assert torch.all(ref_val == hyp_val)
    assert num_iter == num_batches


@pytest.mark.parametrize("sized", (True, False))
def test_compatibility_iterable_unbatched(sized):
    num_items = 10
    ds = _SizedIterable(num_items) if sized else _Iterable(num_items)

    ref = PyTorchDataLoader(ds, batch_size=None, num_workers=0)
    hyp = DataLoader(ds, batch_size=None)

    if sized:
        assert len(ref) == len(hyp) == num_items

    num_iter = 0
    for ref_val, hyp_val in zip(ref, hyp, strict=True):
        num_iter += 1
        assert torch.all(ref_val == hyp_val)
    assert num_iter == num_items


@pytest.mark.parametrize("drop_last", (False, True))
def test_compatibility_map_batched(drop_last):
    batch_size = 3
    num_samples = 10
    num_batches = 3 if drop_last else 4
    ds = _MapStyle(num_samples)

    ref = PyTorchDataLoader(
        ds,
        batch_size=batch_size,
        drop_last=drop_last,
        num_workers=0,
    )

    hyp = DataLoader(
        ds,
        batch_size=batch_size,
        drop_last=drop_last,
    )

    assert len(ref) == len(hyp) == num_batches

    num_iter = 0
    ref_vals, hyp_vals = [], []
    for ref_val, hyp_val in zip(ref, hyp, strict=True):
        num_iter += 1
        print(ref_val, hyp_val)
        ref_vals.append(ref_val)
        hyp_vals.append(hyp_val)
        assert torch.all(ref_val == hyp_val)
    assert num_iter == num_batches


@pytest.mark.parametrize("drop_last", (False, True))
def test_compatibility_map_sampler(drop_last):
    batch_size = 3
    num_samples = 10
    num_batches = 3 if drop_last else 4
    ds = _MapStyle(num_samples)

    sampler = range(num_samples, -1, -1)

    ref = PyTorchDataLoader(
        ds,
        batch_size=batch_size,
        drop_last=drop_last,
        sampler=sampler,
        num_workers=0,
    )

    hyp = DataLoader(
        ds,
        batch_size=batch_size,
        drop_last=drop_last,
        sampler=sampler,
    )

    assert len(ref) == len(hyp) == num_batches

    num_iter = 0
    ref_vals, hyp_vals = [], []
    for ref_val, hyp_val in zip(ref, hyp, strict=True):
        num_iter += 1
        print(ref_val, hyp_val)
        ref_vals.append(ref_val)
        hyp_vals.append(hyp_val)
        assert torch.all(ref_val == hyp_val)
    assert num_iter == num_batches


@pytest.mark.parametrize(
    "batch_sampler",
    [
        ([0, 1, 2], [3, 4], [5, 6, 7], [8], [9]),
        ([0, 1, 2], [3, 4], [5, 6, 7], [8]),
    ],
)
def test_compatibility_map_batch_sampler(batch_sampler):
    num_samples = 10
    num_batches = len(batch_sampler)
    ds = _MapStyle(num_samples)

    ref = PyTorchDataLoader(
        ds,
        batch_sampler=batch_sampler,
        num_workers=0,
    )

    hyp = DataLoader(
        ds,
        batch_sampler=batch_sampler,
    )

    assert len(ref) == len(hyp) == num_batches

    num_iter = 0
    ref_vals, hyp_vals = [], []
    for ref_val, hyp_val in zip(ref, hyp, strict=True):
        num_iter += 1
        print(ref_val, hyp_val)
        ref_vals.append(ref_val)
        hyp_vals.append(hyp_val)
        assert torch.all(ref_val == hyp_val)
    assert num_iter == num_batches
