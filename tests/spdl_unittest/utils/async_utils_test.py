import asyncio
import time

import pytest

import spdl.utils


_SENTINEL = spdl.utils._async._SENTINEL


def _put_queue(queue, vals, sentinel=_SENTINEL):
    for val in vals:
        queue.put_nowait(val)
    queue.put_nowait(sentinel)


def _flush_queue(queue, sentinel=_SENTINEL):
    ret = []
    while (item := queue.get_nowait()) is not sentinel:
        ret.append(item)
    return ret


async def no_op(val):
    return val


################################################################################
# async_generate
################################################################################


def test_async_generate_empty():
    """async_generate should put the values in the queue."""
    src = []

    queue = asyncio.Queue()

    coro = spdl.utils.async_generate(src, no_op, queue)

    asyncio.run(coro)

    vals = _flush_queue(queue)

    assert vals == src


def test_async_generate_simple():
    """async_generate should put the values in the queue."""
    src = list(range(6))

    queue = asyncio.Queue()

    coro = spdl.utils.async_generate(src, no_op, queue)

    asyncio.run(coro)

    vals = _flush_queue(queue)

    assert vals == src


def test_async_generate_task_failure():
    """async_generate should be robust against the error originated from the coroutinue."""
    src = list(range(6))

    async def no_3(val):
        if val == 3:
            raise ValueError(f"{val=} is not allowed")
        return val

    queue = asyncio.Queue()

    coro = spdl.utils.async_generate(src, no_3, queue)

    asyncio.run(coro)

    vals = _flush_queue(queue)

    assert vals == [i for i in range(6) if i != 3]


def test_async_generate_concurrency():
    """async_generate should execute coroutines concurrently."""
    src = [4, 5, 6, 7]

    async def _delay(val):
        await asyncio.sleep(val / 10)
        return val

    queue = asyncio.Queue()

    def run(concurrency):
        coro = spdl.utils.async_generate(src, _delay, queue, concurrency=concurrency)

        t0 = time.monotonic()
        asyncio.run(coro)
        elapsed = time.monotonic() - t0

        vals = _flush_queue(queue)

        assert vals == src
        return elapsed

    elapsed1 = run(concurrency=1)
    elapsed4 = run(concurrency=4)

    assert elapsed1 > 2
    assert elapsed4 < 1


def test_async_generate_timeout_afunc():
    """`async_generate` raises when `afunc` does not return without the given timeout."""

    src = list(range(3))

    async def _long_sleep(val):
        await asyncio.sleep(10)
        return val

    queue = asyncio.Queue()

    coro = spdl.utils.async_generate(src, _long_sleep, queue, timeout=2)

    with pytest.raises((TimeoutError, asyncio.exceptions.TimeoutError)):
        asyncio.run(coro)


def test_async_generate_timeout_queue():
    """TimeoutError is raised when `afunc` is stuck."""
    queue = asyncio.Queue(1)
    queue.put_nowait(0)

    coro = spdl.utils.async_generate(range(3), no_op, queue, timeout=2)

    with pytest.raises((TimeoutError, asyncio.exceptions.TimeoutError)):
        asyncio.run(coro)


def test_async_generate_iterator_failure():
    """When `iterator` fails, the exception is propagated."""

    def src():
        yield 0
        raise RuntimeError("Failing the iterator.")

    coro = spdl.utils.async_generate(src(), no_op, asyncio.Queue())

    with pytest.raises(RuntimeError):
        asyncio.run(coro)


# def test_async_generate_timeout_func():

# Generator timeout raise


################################################################################
# async_iterate
################################################################################


def test_async_iterate():
    """async_iterate should iterate the values in the queue."""
    queue = asyncio.Queue()

    async def test():
        refs = list(range(3))
        _put_queue(queue, refs)

        results = []
        async for val in spdl.utils.async_iterate(queue):
            results.append(val)

        assert results == refs

    asyncio.run(test())


def test_async_iterate_timeout():
    """async_iterate should timeout if there is no value put."""
    queue = asyncio.Queue()

    async def test():
        async for val in spdl.utils.async_iterate(queue, timeout=1):
            pass

    with pytest.raises((TimeoutError, asyncio.exceptions.TimeoutError)):
        asyncio.run(test())


################################################################################
# async_pipe
################################################################################
def test_async_pipe():
    """async_pipe processes the data in input queue and pass it to output queue."""
    input_queue = asyncio.Queue()
    output_queue = asyncio.Queue()

    async def double(val: int):
        return 2 * val

    async def test():
        ref = list(range(6))
        _put_queue(input_queue, ref)

        await spdl.utils.async_pipe(input_queue, double, output_queue, timeout=0.1)

        result = _flush_queue(output_queue)

        assert result == [v * 2 for v in ref]

    asyncio.run(test())


def test_async_pipe_concurrency():
    """async_pipe processes the data in input queue and pass it to output queue."""

    async def delay(val):
        await asyncio.sleep(val / 10)
        return val

    async def test(concurrency):
        input_queue = asyncio.Queue()
        output_queue = asyncio.Queue()

        ref = [4, 5, 6, 7]
        _put_queue(input_queue, ref)

        t0 = time.monotonic()
        await spdl.utils.async_pipe(
            input_queue,
            delay,
            output_queue,
            concurrency=concurrency,
        )
        elapsed = time.monotonic() - t0

        result = _flush_queue(output_queue)

        assert result == ref

        return elapsed

    elapsed1 = asyncio.run(test(1))
    elapsed4 = asyncio.run(test(4))

    assert elapsed1 > 2
    assert elapsed4 < 1


def test_async_pipe_timeout_input_queue():
    """async_pipe should timeout if there is no value in input queue."""
    input_queue = asyncio.Queue()
    output_queue = asyncio.Queue()

    async def test():
        await spdl.utils.async_pipe(input_queue, no_op, output_queue, timeout=1)

    with pytest.raises((TimeoutError, asyncio.exceptions.TimeoutError)):
        asyncio.run(test())


def test_async_pipe_timeout_output_queue():
    """async_pipe should timeout if output queue is full."""
    input_queue = asyncio.Queue()
    output_queue = asyncio.Queue(1)

    async def test():
        _put_queue(input_queue, list(range(3)))

        output_queue.put_nowait(0)  # dummy value to make output_queue full

        await spdl.utils.async_pipe(input_queue, no_op, output_queue, timeout=1)

    with pytest.raises((TimeoutError, asyncio.exceptions.TimeoutError)):
        asyncio.run(test())


################################################################################
# async_genereate + async_iterate
################################################################################

################################################################################
# async_genereate + async_pipe + async_iterate
################################################################################


async def double(val: int):
    return val * 2


async def plus1(val: int):
    return val + 1


def test_3combo():
    """async_genereate + async_pipe + async_iterate"""
    src = list(range(3))

    input_queue = asyncio.Queue()
    output_queue = asyncio.Queue()

    async def test():
        generate = spdl.utils.async_generate(src, double, input_queue)
        pipe = spdl.utils.async_pipe(input_queue, plus1, output_queue)

        gen_task = asyncio.create_task(generate)
        pipe_task = asyncio.create_task(pipe)

        results = []
        async for val in spdl.utils.async_iterate(output_queue):
            results.append(val)

        assert results == [2 * v + 1 for v in src]

        await gen_task
        await pipe_task

    asyncio.run(test())


def test_3combo_src_fail():
    """When src iterator fails, sentinel is propagated properly and other parts works."""

    def src():
        yield from range(3)
        raise RuntimeError("src failed.")

    input_queue = asyncio.Queue()
    output_queue = asyncio.Queue()

    async def test():
        generate = spdl.utils.async_generate(src(), double, input_queue)
        pipe = spdl.utils.async_pipe(input_queue, plus1, output_queue)

        gen_task = asyncio.create_task(generate)
        pipe_task = asyncio.create_task(pipe)

        results = []
        async for val in spdl.utils.async_iterate(output_queue):
            results.append(val)

        assert results == [2 * v + 1 for v in range(3)]

        with pytest.raises(RuntimeError):
            await gen_task

        await pipe_task

    asyncio.run(test())


def test_3combo_src_timeout():
    """When async_generate timeouts, sentinel is propagated properly and other parts works."""

    src = list(range(3))

    async def _delay(val):
        await asyncio.sleep(10)
        return val

    input_queue = asyncio.Queue()
    output_queue = asyncio.Queue()

    async def test():
        generate = spdl.utils.async_generate(src, _delay, input_queue, timeout=2)
        pipe = spdl.utils.async_pipe(input_queue, plus1, output_queue)

        gen_task = asyncio.create_task(generate)
        pipe_task = asyncio.create_task(pipe)

        results = []
        async for val in spdl.utils.async_iterate(output_queue):
            results.append(val)

        assert results == []

        with pytest.raises((TimeoutError, asyncio.exceptions.TimeoutError)):
            await gen_task

        await pipe_task

    asyncio.run(test())


def test_3combo_clogged_input_queue():
    """When input_wueue is clogged, each part timeouts individually."""

    src = list(range(3))

    input_queue = asyncio.Queue(1)
    input_queue.put_nowait(0)  # dummy value to make input queue full

    async def _lazy(val):
        await asyncio.sleep(100)
        return val

    output_queue = asyncio.Queue()

    async def test():
        generate = spdl.utils.async_generate(src, double, input_queue, timeout=2)
        pipe = spdl.utils.async_pipe(input_queue, _lazy, output_queue, timeout=2)

        gen_task = asyncio.create_task(generate)
        pipe_task = asyncio.create_task(pipe)

        with pytest.raises((TimeoutError, asyncio.exceptions.TimeoutError)):
            async for _ in spdl.utils.async_iterate(output_queue, timeout=2):
                pass

        with pytest.raises((TimeoutError, asyncio.exceptions.TimeoutError)):
            await gen_task

        with pytest.raises((TimeoutError, asyncio.exceptions.TimeoutError)):
            await pipe_task

    asyncio.run(test())


def test_3combo_clogged_output_queue():
    """When output_wueue is clogged, each part timeouts individually."""

    src = list(range(5))

    input_queue = asyncio.Queue(1)

    async def _lazy(val):
        await asyncio.sleep(100)
        return val

    output_queue = asyncio.Queue(1)
    output_queue.put_nowait(0)  # dummy value to make input queue full

    async def test():
        generate = spdl.utils.async_generate(src, double, input_queue, timeout=2)
        pipe = spdl.utils.async_pipe(input_queue, _lazy, output_queue, timeout=2)

        gen_task = asyncio.create_task(generate)
        pipe_task = asyncio.create_task(pipe)

        with pytest.raises((TimeoutError, asyncio.exceptions.TimeoutError)):
            async for _ in spdl.utils.async_iterate(output_queue, timeout=2):
                pass

        with pytest.raises((TimeoutError, asyncio.exceptions.TimeoutError)):
            await gen_task

        with pytest.raises((TimeoutError, asyncio.exceptions.TimeoutError)):
            await pipe_task

    asyncio.run(test())


def test_3combo_pipe_fail():
    """When pipe afunc fails, the whole system works fine."""

    src = list(range(3))

    input_queue = asyncio.Queue()

    async def not2(val):
        if val == 2:
            raise ValueError(f"{val=} is not allowed.")
        return val

    output_queue = asyncio.Queue()

    async def test():
        generate = spdl.utils.async_generate(src, no_op, input_queue)
        pipe = spdl.utils.async_pipe(input_queue, not2, output_queue)

        gen_task = asyncio.create_task(generate)
        pipe_task = asyncio.create_task(pipe)

        results = []
        async for val in spdl.utils.async_iterate(output_queue):
            results.append(val)

        assert results == [v for v in range(3) if v != 2]

        await gen_task
        await pipe_task

    asyncio.run(test())


def test_3combo_pipe_timeout():
    """When pipe afunc timeouts, the whole system should timeout."""

    src = list(range(5))

    input_queue = asyncio.Queue(1)

    async def _lazy(val):
        await asyncio.sleep(10)
        return val

    output_queue = asyncio.Queue()

    async def test():
        generate = spdl.utils.async_generate(src, no_op, input_queue, timeout=2)
        pipe = spdl.utils.async_pipe(
            input_queue, _lazy, output_queue, timeout=2, concurrency=1
        )

        gen_task = asyncio.create_task(generate)
        pipe_task = asyncio.create_task(pipe)

        with pytest.raises((TimeoutError, asyncio.exceptions.TimeoutError)):
            async for _ in spdl.utils.async_iterate(output_queue, timeout=2):
                pass

        with pytest.raises((TimeoutError, asyncio.exceptions.TimeoutError)):
            await gen_task

        with pytest.raises((TimeoutError, asyncio.exceptions.TimeoutError)):
            await pipe_task

    asyncio.run(test())


def test_3combo_iter_timeout():
    """When pipe afunc timeouts, the whole system should timeout."""

    src = list(range(5))

    async def _slow(val):
        await asyncio.sleep(1)
        return val

    input_queue = asyncio.Queue(1)

    output_queue = asyncio.Queue(1)

    async def test():
        generate = spdl.utils.async_generate(src, _slow, input_queue, timeout=1)
        pipe = spdl.utils.async_pipe(input_queue, _slow, output_queue, timeout=1)

        gen_task = asyncio.create_task(generate)
        pipe_task = asyncio.create_task(pipe)

        with pytest.raises((TimeoutError, asyncio.exceptions.TimeoutError)):
            async for _ in spdl.utils.async_iterate(output_queue, timeout=1):
                pass

        with pytest.raises((TimeoutError, asyncio.exceptions.TimeoutError)):
            await gen_task

        with pytest.raises((TimeoutError, asyncio.exceptions.TimeoutError)):
            await pipe_task

    asyncio.run(test())
