from collections.abc import Iterator
from typing import TypeVar

import pytest
from pytorch_spdl.dataloader import DataLoader

from torch.utils.data import IterableDataset


T = TypeVar("T")


async def passthrough(vals: T) -> T:
    return vals


def test_iterable_dataset():
    """Simple smoke test for IterableDataset."""

    class _DataSet(IterableDataset):
        def __init__(self, n):
            super().__init__()
            self.n = n

        def __iter__(self) -> Iterator[int]:
            yield from iter(range(self.n))

    dataset = _DataSet(10)
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


def test_map_style_dataset():
    """Simple smoke test for map style dataset."""

    class _DataSet:
        def __init__(self, n):
            super().__init__()
            self.n = n

        def __getitem__(self, i) -> Iterator[int]:
            return i

        def __len__(self) -> int:
            return self.n

    dataset = _DataSet(10)
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


def test_dataloader_invalid_args():
    """DataLoader rejects invalid combinations of args"""
    dummy = object()

    with pytest.raises(ValueError):
        DataLoader(dummy, sampler=dummy, batch_sampler=dummy)

    with pytest.raises(ValueError):
        DataLoader(dummy, sampler=dummy, shuffle=True)

    with pytest.raises(ValueError):
        DataLoader(dummy, batch_sampler=dummy, shuffle=True)

    with pytest.raises(ValueError):
        DataLoader(dummy, batch_sampler=dummy, batch_size=2)

    with pytest.raises(ValueError):
        DataLoader(dummy, num_workers=-1)

    with pytest.raises(ValueError):
        DataLoader(dummy, prefetch_factor=0)
