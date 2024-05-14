import asyncio
import logging
from queue import Empty, Queue
from threading import Event, Thread

_LG = logging.getLogger(__name__)

__all__ = ["BackgroundTaskExecutor"]


def _async_executor(queue: Queue, stopped: Event):
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

    async def _loop(sleep_interval: float = 0.1):
        while not (stopped.is_set()):
            try:
                _process_queue()
            except Empty:
                pass
            finally:
                await asyncio.sleep(sleep_interval)

        _LG.info("Exiting.")

    try:
        asyncio.set_event_loop(asyncio.new_event_loop())
        asyncio.run(_loop())
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


class BackgroundTaskExecutor:
    """Run tasks in background thread."""

    def __init__(self):
        self.thread: Thread = None
        self.queue: Queue = None
        self.stopped: Event = None

    def __enter__(self) -> Queue:
        self.queue = _QueueWrapper()
        self.stopped = Event()
        self.thread = Thread(target=_async_executor, args=(self.queue, self.stopped))
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
