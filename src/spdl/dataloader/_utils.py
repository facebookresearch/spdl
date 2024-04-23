import logging
from pathlib import Path
from typing import Optional, Union

_LG = logging.getLogger(__name__)


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


def _iter_batch(gen, batch_size, drop_last=False):
    batch = []
    for item in gen:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch and not drop_last:
        yield batch


def _iter_flist(
    path: Union[str, Path],
    *,
    prefix: Optional[str] = None,
    batch_size: int = 1,
    n: int = 0,
    N: int = 1,
    max: Optional[int] = None,
    drop_last: bool = False,
):
    gen = _iter_batch(
        _iter_sample_every_n(_iter_file(path, prefix), n, N, max),
        batch_size,
        drop_last=drop_last,
    )
    try:
        yield from gen
    except Exception:
        # Because this utility is intended to be used in background thread,
        # we supress the error and exit
        _LG.exception("Error while iterating over flist %s", path)
        return
