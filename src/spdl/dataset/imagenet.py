"""Utility tools for traversing ImageNet dataset."""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from . import _sql, _utils
from ._dataset import DataSet, ImageData

__all__ = ["ImageNet", "get_flist", "get_mappings", "parse_wnid"]

_LG = logging.getLogger(__name__)


def get_flist(split: str) -> Path:
    """Download the file that contains the list of paths of ImageNet dataset.

    Args:
        split: `"train"`, `"test"` or `"val"`

    Returns:
        (Path): Path to the downloaded file.

    Preview of the content

    * "train"

        ```
        train/n01440764/n01440764_10042.JPEG
        train/n01440764/n01440764_10027.JPEG
        train/n01440764/n01440764_10293.JPEG
        ...
        ```

    * "test"

        ```
        test/ILSVRC2012_test_00000018.JPEG
        test/ILSVRC2012_test_00000006.JPEG
        test/ILSVRC2012_test_00000194.JPEG
        ...
        ```

    * "val"

        ```
        val/n01440764/ILSVRC2012_val_00006697.JPEG
        val/n01440764/ILSVRC2012_val_00010306.JPEG
        val/n01440764/ILSVRC2012_val_00009346.JPEG
        ...
        ```

    """
    vals = ["train", "test", "val"]

    if split not in vals:
        raise ValueError(f"`split` must be one of {vals}")

    return _utils.fetch("imagenet", f"imagenet.{split}.tsv")


def get_mappings():
    """Get the mapping from WordNet ID to class and label.

    1000 IDs from ILSVRC2012 is used. The class indices are the index of
    sorted WordNet ID, which corresponds to most models publicly available.

    Returns:
        (dict[str, int]): Mapping from WordNet ID to class index.
        (dict[str, Tuple[str]]): Mapping from WordNet ID to list of labels.

    ??? note "Example"
        ```python
        >>> class_mapping, label_mapping = get_mappings()
        >>> print(class_mapping["n03709823"])
        636
        >>> print(label_mapping[636])
        ('mailbag', 'postbag')

        ```
    """
    class_mapping = {}
    label_mapping = {}

    path = _utils.fetch("imagenet", "categories.tsv")

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line := line.strip():
                class_, wnid, labels = line.split("\t")[:3]
                class_ = int(class_)
                class_mapping[wnid] = class_
                label_mapping[class_] = tuple(labels.split(","))
    return class_mapping, label_mapping


def parse_wnid(s: str):
    """Parse a WordNet ID (nXXXXXXXX) from string.

    Args:
        s (str): String to parse

    Returns:
        (str): Wordnet ID if found otherwise an exception is raised.
            If the string contain multiple WordNet IDs, the first one is returned.
    """
    if match := re.search(r"n\d{8}", s):
        return match.group(0)
    raise ValueError(f"The given string does not contain WNID: {s}")


def _create_table(con, cur, split: str, table: str, idx_col):
    flist = get_flist(split)

    cols = ",".join([idx_col, "src"])

    tmp_table = f"{table}_tmp"
    cur.execute(f"DROP TABLE IF EXISTS {tmp_table}")
    cur.execute(f"CREATE TABLE {tmp_table}({cols})")

    n = -1
    for paths in _utils._iter_flist(flist, batch_size=1024):
        data = [(n := n + 1, p) for p in paths]
        cur.executemany(f"INSERT INTO {tmp_table}({cols}) VALUES(?, ?)", data)
        con.commit()
    _LG.info(f"{n} entries populated.")

    _LG.debug("Renaming the table.")
    cur.execute(f"ALTER TABLE {tmp_table} RENAME TO {table}")
    con.commit()


def _load_dataset(path: str, split: str, table: str, idx_col: str):
    import sqlite3

    _LG.info(f"Connecting to {path=}")
    con = sqlite3.connect(path)
    cur = con.cursor()

    res = cur.execute(f"SELECT name FROM sqlite_master WHERE name='{table}'")
    if res.fetchone() is None:
        _LG.info(f"{table=} does not exist at {path=}, creating...")
        _create_table(con, cur, split, table, idx_col)
        _LG.debug("Done")
    return con


def ImageNet(path: str, *, split: str) -> DataSet[ImageData]:
    """Load or create ImageNet dataset.

    Args:
        path: Path where the dataset object is searched or newly created.
            Passing `':memory:'` creates database object in-memory.

        split: Passed to [spdl.dataset.librispeech.get_flist][].
            !!! note
                Using `"train"` split will create database with 1.2 million
                records which amounts up to ~80MB of RAM or disk space.

    Returns:
        (DataSet[ImageData]): Dataset object that handles image data.
    """
    table = f"imagenet_{split}".replace("-", "_")
    idx_col = "_index"
    con = _load_dataset(path, split, table=table, idx_col=idx_col)

    class_mapping, _ = get_mappings()

    @dataclass
    class _ImageData(ImageData):
        _index: int = field(repr=False)

        cls: int = field(init=False)

        def __post_init__(self):
            self.cls = class_mapping[parse_wnid(self.src)]

    return _sql.make_dataset(con, _ImageData, table=table, _idx_col=idx_col)
