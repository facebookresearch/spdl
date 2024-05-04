import logging
import os
from pathlib import Path

_LG = logging.getLogger(__name__)

################################################################################
# Dataset catalog acquisition
################################################################################

_URL = "https://github.com/mthrok/dataset-catalogs/releases/download/{tag}/{filename}"


def _get_addr(tag, filename):
    return _URL.format(tag=tag, filename=filename)


def _get_dir(tag):
    base_dir = (
        Path(os.environ["XDG_CACHE_HOME"])
        if "XDG_CACHE_HOME" in os.environ
        else Path.home() / ".cache"
    )
    dir = base_dir / "spdl" / "dataset" / tag
    dir.mkdir(parents=True, exist_ok=True)
    return dir


def _download(src, dst, binary):
    import requests

    resp = requests.get(src)
    resp.raise_for_status()

    mode = "wb" if binary else "w"
    with open(dst, mode, encoding=resp.encoding) as out:
        content = resp.text if mode == "w" else resp.content
        out.write(content)


def fetch(tag: str, filename: str, binary: bool = False):
    """Download the dataset catalog asset."""
    dst = _get_dir(tag) / filename

    if not dst.exists():
        src = _get_addr(tag, filename)
        _LG.info("Downloading %s to %s", src, dst)
        _download(src, dst, binary)

    return dst


################################################################################
# Dataset catalog traversal
################################################################################


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
    path: str | Path,
    *,
    prefix: str | None = None,
    batch_size: int | None = None,
    n: int = 0,
    N: int = 1,
    max: int | None = None,
    drop_last: bool = False,
):
    gen = _iter_sample_every_n(_iter_file(path, prefix), n, N, max)
    if batch_size is not None:
        gen = _iter_batch(gen, batch_size, drop_last=drop_last)

    try:
        yield from gen
    except Exception:
        # Because this utility is intended to be used in background thread,
        # we supress the error and exit
        _LG.exception("Error while iterating over flist %s", path)
        return
