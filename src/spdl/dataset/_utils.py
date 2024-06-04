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
