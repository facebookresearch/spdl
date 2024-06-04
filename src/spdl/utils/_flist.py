import logging
from collections.abc import Iterator
from pathlib import Path
from typing import TypeVar


_LG = logging.getLogger(__name__)

T = TypeVar("T")

__all__ = [
    "iter_batch",
    "iter_flist",
]


def _iter_file(path, prefix):
    with open(path, "r") as f:
        for line in f:
            if path := line.strip():
                if prefix:
                    path = prefix + path
                yield path


def _iter_sample_every_n(gen, offset=0, every_n=1, max=None):
    offset = offset % every_n

    num = 0
    for i, item in enumerate(gen):
        if i % every_n == offset:
            yield item
            num += 1

            if max is not None and num >= max:
                return


def iter_batch(
    gen: Iterator[T],
    batch_size: int,
    drop_last: bool = False,
) -> Iterator[list[T]]:
    """Batchify the given generator."""
    batch = []
    for item in gen:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch and not drop_last:
        yield batch


def iter_flist(
    path: str | Path,
    *,
    prefix: str | None = None,
    batch_size: int | None = None,
    n: int = 0,
    N: int = 1,
    max: int | None = None,
    drop_last: bool = False,
    suppress_error: bool = True,
) -> Iterator[str] | Iterator[list[str]]:
    """Given a file that contains a list of files, iterate over the file names."""
    gen = _iter_sample_every_n(_iter_file(path, prefix), n, N, max)
    if batch_size is not None:
        gen = iter_batch(gen, batch_size, drop_last=drop_last)

    try:
        yield from gen
    except Exception:
        if not suppress_error:
            raise
        # Because this utility is intended to be used in background thread,
        # we supress the error and exit
        _LG.exception("Error while iterating over flist %s", path)
        return
