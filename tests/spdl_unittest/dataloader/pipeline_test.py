import asyncio
import time
from queue import Queue

import pytest

from spdl.dataloader import AsyncPipeline, EOF_SENTINEL as EOF, PipelineFailure
from spdl.dataloader._pipeline import _dequeue, _enqueue, _pipe


def _put_aqueue(queue, vals, *, eof):
    for val in vals:
        queue.put_nowait(val)
    if eof:
        queue.put_nowait(EOF)


def _flush_queue(queue):
    ret = []
    while not queue.empty():
        ret.append(queue.get())
    return ret


def _flush_aqueue(queue):
    ret = []
    while not queue.empty():
        ret.append(queue.get_nowait())
    return ret


async def no_op(val):
    return val


################################################################################
# _async_enqueue
################################################################################


def test_async_enqueue_empty():
    """_async_enqueue can handle empty iterator"""
    queue = asyncio.Queue()
    coro = _enqueue([], queue)
    asyncio.run(coro)
    assert _flush_aqueue(queue) == [EOF]


def test_async_enqueue_simple():
    """_async_enqueue should put the values in the queue."""
    src = list(range(6))
    queue = asyncio.Queue()
    coro = _enqueue(src, queue)
    asyncio.run(coro)
    vals = _flush_aqueue(queue)
    assert vals == src + [EOF]


def test_async_enqueue_iterator_failure():
    """When `iterator` fails, the exception is not propagated."""

    def src():
        yield 0
        raise RuntimeError("Failing the iterator.")

    coro = _enqueue(src(), asyncio.Queue())

    asyncio.run(coro)  # Not raising


def test_async_enqueue_cancel():
    """_async_enqueue is cancellable."""

    async def _test():
        queue = asyncio.Queue(1)

        src = list(range(3))

        coro = _enqueue(src, queue)
        task = asyncio.create_task(coro)

        await asyncio.sleep(0.1)

        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(_test())


################################################################################
# _async_dequeue
################################################################################


@pytest.mark.parametrize("empty", [False, True])
def test_async_dequeue_simple(empty: bool):
    """_async_dequeue pass the contents from input_queue to output_queue"""
    input_queue = asyncio.Queue()
    output_queue = Queue()

    data = [] if empty else list(range(3))
    _put_aqueue(input_queue, data, eof=True)

    coro = _dequeue(input_queue, output_queue)

    asyncio.run(coro)
    results = _flush_queue(output_queue)

    assert results == data


def test_async_dequeue_cancel():
    """_async_dequeue is cancellable."""

    async def _test():
        input_queue = asyncio.Queue()
        output_queue = Queue()

        coro = _dequeue(input_queue, output_queue)
        task = asyncio.create_task(coro)

        await asyncio.sleep(0.1)

        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(_test())


################################################################################
# async_pipe
################################################################################
async def adouble(val: int):
    return 2 * val


async def aplus1(val: int):
    return val + 1


def test_async_pipe():
    """async_pipe processes the data in input queue and pass it to output queue."""
    input_queue = asyncio.Queue()
    output_queue = asyncio.Queue()

    async def test():
        ref = list(range(6))
        _put_aqueue(input_queue, ref, eof=True)

        await _pipe(input_queue, adouble, output_queue)

        result = _flush_aqueue(output_queue)

        assert result == [v * 2 for v in ref] + [EOF]

    asyncio.run(test())


# TOOD: test same for AsyncPipeline
def test_async_pipe_wrong_task_signature():
    """async_pipe fails immediately if user provided incompatible iterator/afunc."""
    input_queue = asyncio.Queue()
    output_queue = asyncio.Queue()

    async def _2args(val: int, _):
        return val

    async def test():
        ref = list(range(6))
        _put_aqueue(input_queue, ref, eof=False)

        with pytest.raises(TypeError):
            await _pipe(input_queue, _2args, output_queue, concurrency=3)

        remaining = _flush_aqueue(input_queue)
        assert remaining == ref[1:]

        result = _flush_aqueue(output_queue)
        assert result == []

    asyncio.run(test())


@pytest.mark.parametrize("full", [False, True])
def test_async_pipe_cancel(full):
    """async_pipeis cancellable."""
    input_queue = asyncio.Queue()
    output_queue = asyncio.Queue(1)

    _put_aqueue(input_queue, list(range(3)), eof=False)

    if full:
        output_queue.put_nowait(None)

    cancelled = False

    async def astuck(i):
        try:
            await asyncio.sleep(10)
            return i
        except asyncio.CancelledError:
            nonlocal cancelled
            cancelled = True
            raise

    async def test():
        coro = _pipe(input_queue, astuck, output_queue)
        task = asyncio.create_task(coro)

        await asyncio.sleep(0.5)

        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

    assert not cancelled
    asyncio.run(test())
    assert cancelled


def test_async_pipe_concurrency():
    """Changing concurrency changes the number of items fetched and processed."""

    async def delay(val):
        await asyncio.sleep(0.5)
        return val

    async def test(concurrency):
        input_queue = asyncio.Queue()
        output_queue = asyncio.Queue()

        ref = [1, 2, 3, 4]
        _put_aqueue(input_queue, ref, eof=False)

        coro = _pipe(
            input_queue,
            delay,
            output_queue,
            concurrency=concurrency,
        )

        task = asyncio.create_task(coro)
        await asyncio.sleep(0.8)
        task.cancel()

        return _flush_aqueue(input_queue), _flush_aqueue(output_queue)

    # With concurrency==1, there should be
    # 1 in output_queue, 2 is in flight, 3 and 4 remain in input_queue
    remain, output = asyncio.run(test(1))
    assert remain == [3, 4]
    assert output == [1]

    # With concurrency==4, there should be
    # 1, 2, 3 and 4 in output_queue.
    remain, output = asyncio.run(test(4))
    assert remain == []
    assert output == [1, 2, 3, 4]


def test_async_pipe_concurrency_throughput():
    """increasing concurrency improves the throughput."""

    async def delay(val):
        await asyncio.sleep(0.5)
        return val

    async def test(concurrency):
        input_queue = asyncio.Queue()
        output_queue = asyncio.Queue()

        ref = [4, 5, 6, 7, EOF]
        _put_aqueue(input_queue, ref, eof=False)

        t0 = time.monotonic()
        await _pipe(
            input_queue,
            delay,
            output_queue,
            concurrency=concurrency,
        )
        elapsed = time.monotonic() - t0

        result = _flush_aqueue(output_queue)

        assert result == ref

        return elapsed

    elapsed1 = asyncio.run(test(1))
    elapsed4 = asyncio.run(test(4))

    assert elapsed1 > 2
    assert elapsed4 < 1


################################################################################
# AsyncPipeline
################################################################################


def test_async_pipeline_simple():
    result_queue = Queue()

    pipeline = (
        AsyncPipeline()
        .add_source(range(10))
        .pipe(adouble)
        .pipe(aplus1)
        .add_sink(result_queue)
    )

    async def _test():
        await pipeline.run()
        results = _flush_queue(result_queue)

        assert 10 == len(results)
        assert results == [1 + 2 * i for i in range(10)]

    asyncio.run(_test())


def test_async_pipeline_noop():
    """AsyncPipeline functions without pipe"""
    result_queue = Queue()

    src = list(range(10))
    pipeline = AsyncPipeline().add_source(src).add_sink(result_queue)

    async def _test():
        await pipeline.run()
        results = _flush_queue(result_queue)

        assert 10 == len(results)
        assert results == src

    asyncio.run(_test())


def test_async_pipeline_source_failure():
    """AsyncPipeline continues when source fails."""
    result_queue = Queue()

    def failing_range(i):
        yield from range(i)
        raise ValueError("Iterator failed")

    pipeline = (
        AsyncPipeline()
        .add_source(failing_range(10))
        .pipe(adouble)
        .pipe(aplus1)
        .add_sink(result_queue)
    )

    async def _test():
        await pipeline.run()

        results = _flush_queue(result_queue)

        assert 10 == len(results)
        assert results == [1 + 2 * i for i in range(10)]

    asyncio.run(_test())


def test_async_pipeline_type_error():
    """AsyncPipeline immediately fails if pipe function has wrong signature"""
    result_queue = Queue()

    async def wrong_sig(i, _):
        return i

    pipeline = (
        AsyncPipeline().add_source(range(10)).pipe(wrong_sig).add_sink(result_queue)
    )

    with pytest.raises(PipelineFailure) as einfo:
        asyncio.run(pipeline.run())

    err = einfo.value
    print(err._errs.keys())
    assert isinstance(err._errs["AsyncPipeline::1_wrong_sig"], TypeError)


def test_async_pipeline_task_failure():
    """AsyncPipeline is robust against task-level failure."""
    result_queue = Queue()

    async def areject_m3(i):
        if i % 3 == 0:
            raise ValueError(f"Multiple of 3 is prohibited: {i}")
        return i

    pipeline = (
        AsyncPipeline()
        .add_source(range(10))
        .pipe(areject_m3)
        .pipe(adouble)
        .pipe(aplus1)
        .add_sink(result_queue)
    )

    async def _test():
        await pipeline.run()
        results = _flush_queue(result_queue)

        assert 6 == len(results)
        assert results == [1 + 2 * i for i in range(10) if i % 3]

    asyncio.run(_test())


def test_async_pipeline_cancel():
    """AsyncPipeline is cancellable."""
    result_queue = Queue()

    cancelled = False

    async def astuck(i):
        try:
            await asyncio.sleep(3)
            return i
        except asyncio.CancelledError:
            nonlocal cancelled
            cancelled = True
            raise

    pipeline = (
        AsyncPipeline()
        .add_source(range(10))
        .pipe(astuck, concurrency=1)
        .add_sink(result_queue)
    )

    async def _test():
        task = asyncio.create_task(pipeline.run())

        await asyncio.sleep(1)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

    assert not cancelled
    asyncio.run(_test())
    assert cancelled
