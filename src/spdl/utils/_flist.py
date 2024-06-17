# pyre-unsafe

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import overload, TypeVar


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
    """Batchify the given generator.

    Args:
        gen: An iterator.
        batch_size: The number of items in one batch.
        drop_last: Drop the last batch if it does not contain
            the requested number of items.

    Yields:
        List of items.
    """
    batch = []
    for item in gen:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch and not drop_last:
        yield batch


@overload
def iter_flist(
    path: str | Path,
    *,
    prefix: str | None = None,
    batch_size: None = None,
    n: int = 0,
    N: int = 1,
    max: int | None = None,
    drop_last: bool = False,
    suppress_error: bool = True,
) -> Iterator[str]: ...


@overload
def iter_flist(
    path: str | Path,
    *,
    prefix: str | None = None,
    batch_size: int,
    n: int = 0,
    N: int = 1,
    max: int | None = None,
    drop_last: bool = False,
    suppress_error: bool = True,
) -> Iterator[list[str]]: ...


def iter_flist(
    path,
    *,
    prefix=None,
    batch_size=None,
    n=0,
    N=1,
    max=None,
    drop_last=False,
    suppress_error=True,
):
    """Iterate over the given text file.

    Args:
        path: The input path.
        prefix: Prefix each line.
        batch_size: If not ``None``, batch the return lines.
        n, N: Retrieve n-th item out of every N items.
        max: The maximum number of items to return.
        drop_last: If ``batch_size`` is given, if the last batch
            is shorter than this ``batch_size``, then drop it.
        suppress_error: If ``False``, propagate the error.
            Default: ``True``.

    Yields:
        Single line of batch of lines.
    """
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
