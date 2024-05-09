"""`spdl.dataloader` module implements utilities to run I/O operations efficiently.

"""

from ._task_runner import apply_async, apply_concurrent, BackgroundGenerator

__all__ = [
    "BackgroundGenerator",
    "apply_async",
    "apply_concurrent",
]
