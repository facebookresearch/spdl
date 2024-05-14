import functools
import logging

from concurrent.futures import Future
from typing import Any, Callable, Generator, List, TypeVar

__all__ = [
    "chain_futures",
    "wait_futures",
    "create_future",
]

_LG = logging.getLogger(__name__)

T = TypeVar("T")


def _chain_futures(generator: Generator[Future[Any], Any, None]) -> Future[Any]:
    # The Future object that client code handles
    f = Future()
    f.set_running_or_notify_cancel()

    # The Future object for the background task
    _future = None

    def _chain(result, cb):
        nonlocal _future
        try:
            _future = generator.send(result)
        except StopIteration:
            f.set_result(result)
        else:
            _future.add_done_callback(cb)

    def _cb(fut):
        try:
            _chain(fut.result(), _cb)
        except Exception as e:
            f.set_exception(e)

    _chain(None, _cb)

    # pyre-ignore: [16]
    f.__spdl_future = _future
    return f


def chain_futures(
    func: Callable[..., Generator[Future[Any], Any, None]]
) -> Callable[..., Future[Any]]:
    """Chain call multiple ``concurrent.futures.Future``s object sequentially.

    Args:
        func: A generator function which yields Future objects.
            The result of each future is sent back to generator, and
            generetor will use the result to launch the next future.

    ??? note "Example"
        ```python

        @chain_futures
        def load_image(src):
            '''demux, decode and convert a single image from src'''

            # The objects at right hand side of ``yield`` expressions
            # are all Future objects.
            # ``chain_futures`` will recieve them and add callbacks,
            # which fetches the next Future from the generator and
            # pass the current result to it.
            packets = yield spdl.io.demux_media("image", src)
            frames = yield spdl.io.decode_packets(packets)
            yield spdl.io.convert_buffer(frames)

        # Chain the futures so that we only have one Future to track
        future = load_image("foo.jpg")
        # Blocking wait
        buffer = future.result()
        ```
    """

    @functools.wraps(func)
    def _func(*args, **kwargs) -> Future[Any]:
        return _chain_futures(func(*args, **kwargs))

    return _func


def wait_futures(futures: List[Future], strict: bool = True) -> Future:
    """Wait for all the given ``Futrue``s to fullfill.

    Args:
        futures: List of futures to wait for.

        strict: If True and if any of the future did not complete, then
            raise an error.

    ??? note "Example"
        ```python
        # Start batch demuxing
        futures = [spdl.io.demux_media("image", s) for s in srcs]
        fut = wait_futures(futures)
        # Wait for all the demuxing to complete
        packets = fut.result()
        ```
    """
    if not futures:
        raise ValueError("No future is provided.")

    f = Future()
    f.set_running_or_notify_cancel()

    num_futures = len(futures)
    sentinel = object()
    results = [sentinel for _ in range(num_futures)]

    def _cb(future):
        nonlocal num_futures

        try:
            result = future.result()
        except Exception as e:
            _LG.error("%s", e)
        else:
            results[futures.index(future)] = result
        finally:
            if (num_futures := num_futures - 1) == 0:
                rs = [r for r in results if r is not sentinel]
                if not rs:
                    f.set_exception(RuntimeError("All the futures have failed."))
                if len(rs) != len(results) and strict:
                    f.set_exception(RuntimeError("Some of the futures have failed."))
                else:
                    f.set_result(rs)

    for future in futures:
        future.add_done_callback(_cb)

    # pyre-ignore: [16]
    f.__spdl_futures = futures

    return f


def create_future(val: T) -> Future[T]:
    """Create a Future object with given value.

    Args:
        val: The initial value of the future.

    Returns:
        A Future object with given value.
    """
    f = Future()
    f.set_running_or_notify_cancel()
    f.set_result(val)
    return f
