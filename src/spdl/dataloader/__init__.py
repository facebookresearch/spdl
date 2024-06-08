"""Utilities to run I/O operations efficiently."""

from ._bg_consumer import BackgroundConsumer
from ._bg_generator import BackgroundGenerator

__all__ = [
    "BackgroundConsumer",
    "BackgroundGenerator",
]


def __getattr__(name: str):
    if name == "PyTorchStyleDataLoader":
        from . import _torch

        return _torch.PyTorchStyleDataLoader

    raise AttributeError(f"module {__name__} has no attribute {name}")
