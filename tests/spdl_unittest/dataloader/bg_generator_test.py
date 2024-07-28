# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio

import pytest

from spdl.dataloader._bg_generator import BackgroundGenerator
from spdl.dataloader._legacy_pipeline import AsyncPipeline  # pyre-ignore: [16]


async def adouble(i):
    print(f"adouble({i})")
    return 2 * i


async def aplus1(i):
    print(f"aplus1({i})")
    return 1 + i


def test_bgg_simple():
    """bg generator can process simple pipeline"""

    apl = AsyncPipeline().add_source(range(10)).pipe(adouble).pipe(aplus1)

    bgg = BackgroundGenerator(apl, timeout=None)
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


def test_bgg_timeout_none():
    # Iterate all.
    bgg = BackgroundGenerator(
        _get_apl(),
        timeout=None,
    )
    results = list(bgg)
    assert results == list(range(10))


def test_bgg_timeout_enough():
    # Iterate all.
    bgg = BackgroundGenerator(
        _get_apl(),
        timeout=1.3,
    )
    results = list(bgg)
    assert results == list(range(10))


def test_bgg_timeout_not_enough():
    """Timeout shut down the background generator cleanly."""
    # Iterate none.
    bgg = BackgroundGenerator(
        _get_apl(),
        timeout=0.01,
    )
    with pytest.raises(TimeoutError):
        for _ in bgg:
            pass


def test_bgg_run_partial():
    """bg generator can run pipeline partially and repeatedly"""

    class Generator:
        def __iter__(self):
            for i in range(10):
                print(f"Generating: {i}")
                yield i

    apl = AsyncPipeline().add_source(Generator()).pipe(adouble).pipe(aplus1)

    bgg = BackgroundGenerator(apl, timeout=5)

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


def test_bgg_resume_from_cancelled():
    """When bg thread was stopped by foreground, it can be resumed cleanly"""

    def src(i=-1):
        while True:
            yield (i := i + 1)

    apl = (
        AsyncPipeline()
        .add_source(src())
        .pipe(adouble, buffer_size=1)
        .pipe(aplus1, buffer_size=1)
    )

    bgg = BackgroundGenerator(apl, timeout=None)

    async def _test():
        for i, item in enumerate(bgg):
            assert (i * 2 + 1) == item
            if i == 3:
                break

        for i, item in enumerate(bgg, start=4):
            assert (i * 2 + 1) == item
            if i == 6:
                break

    asyncio.run(_test())
