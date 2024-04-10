import asyncio

import pytest
import spdl.io


def test_async_sleep_throw():
    """Verify the premise that the test function `async_sleep` throws an
    exception at the end if it was not cancelled.
    """

    async def _test():
        # Check that the test function `async_sleep` throws an
        # exception if it was not cancelled.
        #
        # This is the basis of all the other tests.
        coro, sf = spdl.io._async_sleep(500)
        task = asyncio.create_task(coro)
        with pytest.raises(spdl.io.AsyncIOFailure):
            await task
        assert not sf.cancelled()

    asyncio.run(_test())


def test_cancel_task():
    """Task cancellation is propagated to SPDL's future object"""

    async def _test():
        coro, sf = spdl.io._async_sleep(500)
        task = asyncio.create_task(coro)
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert sf.cancelled()

    asyncio.run(_test())


def test_wait_for_cancel_task():
    """When `wait_for` timeout, the cancellation is propagated to SPDL's future object"""

    async def _test():
        coro, sf = spdl.io._async_sleep(1000)
        task = asyncio.create_task(coro)
        with pytest.raises(asyncio.exceptions.TimeoutError):
            await asyncio.wait_for(task, timeout=0.1)
        assert task.cancelled()
        assert sf.cancelled()

    asyncio.run(_test())


def test_async_sleep_multi_throws():
    """Verify the premise that the test function `async_sleep_multi` throws an
    exception at the end if it was not cancelled.
    """

    async def _test():
        coro, sf = spdl.io._async_sleep_multi(100, 10)
        assert not sf.cancelled()

        async def _sleep():
            async for i in coro:
                print(i, flush=True)

        with pytest.raises(spdl.io.AsyncIOFailure):
            await _sleep()
        assert not sf.cancelled()

    asyncio.run(_test())


def test_cancel_generator():
    """Async generator can be cancelled"""

    async def _test():
        coro, sf = spdl.io._async_sleep_multi(1000, 3)

        async def _sleep():
            async for i in coro:
                print(i, flush=True)

        task = asyncio.create_task(_sleep())
        await asyncio.sleep(0.5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert sf.cancelled()

    asyncio.run(_test())


def test_wait_for_cancel_generator():
    """When `wait_for` timeout, the cancellation is propagated to SPDL's future object"""

    async def _test():
        coro, sf = spdl.io._async_sleep_multi(1000, 3)

        async def _sleep():
            async for i in coro:
                print(i, flush=True)

        task = asyncio.create_task(_sleep())
        with pytest.raises(asyncio.exceptions.TimeoutError):
            await asyncio.wait_for(task, timeout=0.1)
        assert task.cancelled()
        assert sf.cancelled()

    asyncio.run(_test())
