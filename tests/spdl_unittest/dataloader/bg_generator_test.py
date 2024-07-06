import asyncio

import pytest

import spdl.dataloader
from spdl.dataloader import AsyncPipeline  # pyre-ignore: [16]


async def adouble(i):
    print(f"adouble({i})")
    return 2 * i


async def aplus1(i):
    print(f"aplus1({i})")
    return 1 + i


def test_simple():
    """bg generator can process simple pipeline"""

    apl = AsyncPipeline().add_source(range(10)).pipe(adouble).pipe(aplus1)

    bgg = spdl.dataloader.BackgroundGenerator(apl, timeout=None)
    dataloader = iter(bgg)
    for i in range(10):
        assert 2 * i + 1 == next(dataloader)
    with pytest.raises(StopIteration):
        next(dataloader)


def _get_apl(n=10):
    async def pass_with_sleep(i):
        await asyncio.sleep(0.1 * i)
        return i

    apl = AsyncPipeline().add_source(range(n)).pipe(pass_with_sleep)
    return apl


def test_timeout_none():
    # Iterate all.
    bgg = spdl.dataloader.BackgroundGenerator(
        _get_apl(),
        timeout=None,
    )
    results = list(bgg)
    assert results == list(range(10))


def test_timeout_enough():
    # Iterate all.
    bgg = spdl.dataloader.BackgroundGenerator(
        _get_apl(),
        timeout=1.3,
    )
    results = list(bgg)
    assert results == list(range(10))


def test_timeout_not_enough():
    """Timeout shut down the background generator cleanly."""
    # Iterate none.
    bgg = spdl.dataloader.BackgroundGenerator(
        _get_apl(),
        timeout=0.01,
    )
    with pytest.raises(TimeoutError):
        for _ in bgg:
            pass


def test_run_partial():
    """bg generator can run pipeline partially and repeatedly"""

    class Generator:
        def __iter__(self):
            for i in range(10):
                print(f"Generating: {i}")
                yield i

    apl = AsyncPipeline().add_source(iter(Generator())).pipe(adouble).pipe(aplus1)

    bgg = spdl.dataloader.BackgroundGenerator(apl, timeout=None)

    dataloader = iter(bgg.run(3))
    for i in range(3):
        assert 2 * i + 1 == next(dataloader)

    with pytest.raises(StopIteration):
        next(dataloader)

    dataloader = iter(bgg.run(3))
    for i in range(3, 6):
        assert 2 * i + 1 == next(dataloader)

    with pytest.raises(StopIteration):
        next(dataloader)

    dataloader = iter(bgg.run())
    for i in range(6, 10):
        assert 2 * i + 1 == next(dataloader)

    with pytest.raises(StopIteration):
        next(dataloader)
