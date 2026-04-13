# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Background task abstractions for the pipeline engine."""

from __future__ import annotations

from collections.abc import Callable

__all__ = [
    "BackgroundTask",
    "BackgroundTaskFactory",
    "get_default_background_tasks",
    "set_default_background_tasks",
]


class BackgroundTask:
    """A background task that runs alongside pipeline stages.

    Subclass this and override :py:meth:`run` to implement custom logic.
    The task is started when the pipeline starts and cancelled when the
    pipeline completes. Errors are logged but do not cause the pipeline
    to fail.

    Example::

        class MyMonitor(BackgroundTask):
            async def run(self) -> None:
                while True:
                    collect_metrics()
                    await asyncio.sleep(60)

        pipeline = build_pipeline(cfg, num_threads=4,
                                  background_tasks=[MyMonitor])
    """

    async def run(self) -> None:
        """Override this to implement the background task logic.

        This coroutine runs in the pipeline's event loop. It will be
        cancelled when the pipeline completes, so use ``try/except
        asyncio.CancelledError`` if cleanup is needed.
        """
        raise NotImplementedError


BackgroundTaskFactory = Callable[[], BackgroundTask]

_DEFAULT_BACKGROUND_TASKS: list[BackgroundTaskFactory] | None = None


def get_default_background_tasks() -> list[BackgroundTaskFactory] | None:
    """Get the default background task factories.

    Returns:
        The default background task factories or None if not set.
    """
    return _DEFAULT_BACKGROUND_TASKS


def set_default_background_tasks(
    tasks: list[BackgroundTaskFactory] | None,
) -> None:
    """Set the default background task factories.

    Each factory is called to create a :py:class:`BackgroundTask` instance
    whose :py:meth:`~BackgroundTask.run` coroutine runs alongside the pipeline
    stages. Tasks are cancelled when the pipeline completes. Their errors are
    logged but do not cause the pipeline to fail.

    Args:
        tasks: A list of background task factories, or None to unset.
    """
    global _DEFAULT_BACKGROUND_TASKS
    _DEFAULT_BACKGROUND_TASKS = tasks
