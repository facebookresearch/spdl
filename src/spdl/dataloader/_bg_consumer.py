import asyncio
import logging
from queue import Empty, Queue
from threading import Event, Thread

from ._utils import _get_loop

__all__ = ["BackgroundConsumer"]

_LG = logging.getLogger(__name__)


def _async_executor(loop: asyncio.AbstractEventLoop, queue: Queue, stopped: Event):
    tasks = set()

    def _cb(task):
        try:
            task.result()
        except asyncio.exceptions.CancelledError:
            _LG.warning("Task [%s] was cancelled.", task.get_name())
        except Exception as err:
            _LG.error("Task [%s] failed: %s", task.get_name(), err)
        finally:
            queue.task_done()
            tasks.discard(task)

    def _process_queue():
        while not queue.empty():
            coro = queue.get_nowait()
            task = asyncio.create_task(coro)
            tasks.add(task)
            task.add_done_callback(_cb)

    async def _main_loop(sleep_interval: float = 0.1):
        while not (stopped.is_set()):
            try:
                _process_queue()
            except Empty:
                pass
            finally:
                await asyncio.sleep(sleep_interval)

        _LG.info("Exiting.")

    try:
        loop.run_until_complete(_main_loop())
    finally:
        stopped.set()


class _QueueWrapper(Queue):
    """Queue with simple `close` mechanism."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._closed = False

    def put(self, item):
        if self._closed:
            raise RuntimeError("Queue is closed.")
        super().put(item)


class BackgroundConsumer:
    """Run tasks in background thread."""

    def __init__(
        self,
        num_workers: int | None = 1,
        loop: asyncio.AbstractEventLoop | None = None,
    ):
        self.loop = _get_loop(num_workers) if loop is None else loop

        self.thread: Thread = None  # pyre-ignore: [6]
        self.queue: Queue = None  # pyre-ignore: [6]
        self.stopped: Event = None  # pyre-ignore: [6]

    def __enter__(self) -> Queue:
        self.queue = _QueueWrapper()
        self.stopped = Event()
        self.thread = Thread(
            target=_async_executor, args=(self.loop, self.queue, self.stopped)
        )
        self.thread.start()
        return self.queue

    def __exit__(self, exc_type, exc_value, traceback):
        # Prevent a new job to be submitted
        self.queue._closed = True

        # If stopped is not set, the background thread is still running.
        # Let it process all the queued jobs before exiting.
        # Otherwise the loop is unexpectedly dead.
        if not self.stopped.is_set():
            # Wait for all the works to be completed.
            _LG.info("Waiting for the background thread to exit.")
            self.queue.join()
            self.stopped.set()

        self.thread.join()
