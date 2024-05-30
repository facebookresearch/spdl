import asyncio
import time

import pytest

import spdl.dataloader


def test_simple():
    """bg generator can process simple async generator"""

    async def _agen():
        for i in range(10):
            yield i

    bgg = spdl.dataloader.BackgroundGenerator(
        _agen(),
        timeout=None,
    )
    dataloader = iter(bgg)
    for i in range(10):
        assert i == next(dataloader)
    with pytest.raises(StopIteration):
        next(dataloader)


def test_timeout():
    """`timeout` parameter stops the bg generator cleanly"""

    async def _agen():
        await asyncio.sleep(1)
        yield 0
        await asyncio.sleep(3)
        yield 1

    # Iterate all.
    bgg = spdl.dataloader.BackgroundGenerator(
        _agen(),
        timeout=5,
    )
    dataloader = iter(bgg)
    assert next(dataloader) == 0
    assert next(dataloader) == 1
    with pytest.raises(StopIteration):
        next(dataloader)

    # Iterate all.
    bgg = spdl.dataloader.BackgroundGenerator(
        _agen(),
        timeout=None,
    )
    dataloader = iter(bgg)
    assert next(dataloader) == 0
    assert next(dataloader) == 1
    with pytest.raises(StopIteration):
        next(dataloader)

    # Iterate none.
    bgg = spdl.dataloader.BackgroundGenerator(
        _agen(),
        timeout=0.1,
    )
    dataloader = iter(bgg)
    with pytest.raises(TimeoutError):
        next(iter(bgg))

    # Iterate one. foreground timeout before background
    # This could race.
    # bgg = spdl.dataloader.BackgroundGenerator(
    #     _agen(),
    #     timeout=2.0,
    # )
    # dataloader = iter(bgg)

    # assert next(dataloader) == 0
    # with pytest.raises(TimeoutError):
    #     next(dataloader)

    # Iterate one, background timeout before foreground.
    bgg = spdl.dataloader.BackgroundGenerator(
        _agen(),
        timeout=2.0,
    )
    dataloader = iter(bgg)

    assert next(dataloader) == 0
    time.sleep(4)
    with pytest.raises(RuntimeError):
        next(dataloader)
