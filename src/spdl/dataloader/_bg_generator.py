# pyre-unsafe

import asyncio
import concurrent.futures
import logging
import warnings
from asyncio import AbstractEventLoop as EventLoop
from collections.abc import Iterator
from contextlib import contextmanager
from threading import Thread
from typing import Generic, TypeVar

from ._legacy_pipeline import AsyncPipeline
from ._utils import _get_loop

_LG = logging.getLogger(__name__)

__all__ = [
    "BackgroundGenerator",
]

T = TypeVar("T")


@contextmanager
def _run_in_thread(loop: EventLoop, timeout: int | float | None):
    """Run the given event loop in a thread."""
    if loop.is_running():
        raise RuntimeError("Loop must not be running.")

    thread = Thread(target=loop.run_forever)
    thread.start()
    try:
        yield
    finally:
        _LG.info("Stopping event loop and joining the event loop thread.")
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout)
        if thread.is_alive():
            _LG.warning("The background thread did not join after %f seconds.", timeout)
        else:
            _LG.info("The event loop thread successfully joined.")


class BackgroundGenerator(Generic[T]):
    """**[Deprecated]** Use :py:class:`spdl.dataloader.PipelineBuilder`.

    Run generator in background and iterate the items.

    Args:
        pipeline: Pipeline to run in the background.

        num_workers: The number of worker threads to be attached to the event loop.
            If ``loop`` is provided, this argument is ignored.

        queue_size: The size of the queue that is used to pass the
            generated items from the background thread to the main thread.
            If the queue is full, the background thread will be blocked.

        timeout: The maximum time to wait for the generator to yield an item.
            If the generator does not yield an item within this time, ``TimeoutError``
            is raised.

            This parameter is intended for breaking unforseen situations where
            the background generator is stuck for some reasons.

            It is also used for timeout when waiting for the background thread to join.

        loop: If provided, use this event loop to execute the generator.
            Otherwise, a new event loop will be created. When providing a loop,
            ``num_workers`` is ignored, so the executor must be configured by
            client code.

    .. admonition:: Example

       >>> apl = (
       >>>     spdl.dataloader.AsyncPipeline()
       >>>     .add_source(iter(range(10)))
       >>> )
       >>>
       >>> processor = BackgroundGenerator(apl)
       >>> for item in processor.run(3):
       >>>     # Do something with the item.
    """

    def __init__(
        self,
        pipeline: AsyncPipeline,
        *,
        num_workers: int | None = 3,
        queue_size: int = 10,
        timeout: int | float | None = 300,
        loop: EventLoop | None = None,
    ):
        warnings.warn(
            "`BackgroundGenerator` has been deprecated. Please use `PipelineBuilder`.",
            category=FutureWarning,
            stacklevel=2,
        )

        self.pipeline = pipeline
        self.queue_size = queue_size
        self.loop = _get_loop(num_workers) if loop is None else loop
        self.timeout = timeout

        self.pipeline.add_sink(queue_size)

        try:
            from spdl.lib import _libspdl

            _libspdl.log_api_usage("spdl.dataloader.BackgroundGenerator")
        except Exception:
            pass  # ignore if not supported.

    def __iter__(self) -> Iterator[T]:
        """Run the generator in background thread and iterate the result.

        Yields:
            Items generated by the provided generator.
        """
        warnings.warn(
            "`BackgroundGenerator.__iter__` has been deprecated. "
            "Please use `BackgroundGenerator.run()`.",
            category=FutureWarning,
            stacklevel=2,
        )
        return self.run()

    def run(
        self, num_items: int | None = None, *, fail_on_error: bool = False
    ) -> Iterator[T]:
        """Run the generator in background thread and iterate the result.

        Args:
            num_items: The number of items to yield. If omitted, the generator
                will be iterated until the end.

            fail_on_error: When ``True``, raise the exception if pipeline fails. Otherwise,
                log the error and continue.

        Yields:
            Items generated by the provided generator.
        """
        sentinel = object()

        coros = self.pipeline._build(num_items=num_items)

        async def _run():
            try:
                await coros
            finally:
                await self.pipeline.output_queue.put(sentinel)

        task = self.loop.create_task(_run())

        def _get_item():
            return asyncio.run_coroutine_threadsafe(
                self.pipeline.output_queue.get(), self.loop
            )

        with _run_in_thread(self.loop, self.timeout):
            while True:
                try:
                    item = _get_item().result(timeout=self.timeout)
                except concurrent.futures.TimeoutError:
                    # The pipeline is too slow or stuck.
                    raise TimeoutError(
                        f"The pipeline did not yield an item within {self.timeout} seconds."
                    ) from None
                else:
                    if item is sentinel:
                        break
                    yield item

            if (err := task.exception()) is not None:
                if fail_on_error:
                    raise err
                _LG.exception("Pipeline encountered an error.")
