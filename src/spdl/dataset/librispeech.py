"""Utility tools for traversing LibriSpeech dataset."""

import logging
from dataclasses import dataclass, field
from pathlib import Path

from . import _sql, _utils
from ._dataset import AudioData, DataSet

__all__ = ["LibriSpeech", "get_flist"]

_LG = logging.getLogger(__name__)


def get_flist(split: str) -> Path:
    """Download the file that contains the list of paths of LibriSpeech dataset.

    Args:
        split: Split of the dataset.
            Valid values are `"test-clean"`, `"test-other"`.

    Returns:
        (Path): Path to the downloaded file.

    Preview of the content. They are tab-separated, and the second column is the
    number of samples.

    * "test-clean"

        ```
        LibriSpeech/test-clean/672/122797/672-122797-0033.flac\t20560
        LibriSpeech/test-clean/2094/142345/2094-142345-0041.flac\t22720
        LibriSpeech/test-clean/2830/3980/2830-3980-0026.flac\t23760
        ...
        ```

    * "test-other"

        ```
        LibriSpeech/test-other/2414/128291/2414-128291-0020.flac\t20000
        LibriSpeech/test-other/7902/96592/7902-96592-0020.flac\t21040
        LibriSpeech/test-other/8188/269290/8188-269290-0057.flac\t23520
        ...
        ```
    """
    vals = ["test-clean", "test-other"]
    if split not in vals:
        raise ValueError(f"`split` must be one of {vals}")

    return _utils.fetch("librispeech", f"librispeech.{split}.tsv")


def _create_table(con, cur, split: str, table: str, idx_col: str):
    flist = get_flist(split)

    cols = ",".join([idx_col, "src", "num_frames"])

    tmp_table = f"{table}_tmp"
    cur.execute(f"DROP TABLE IF EXISTS {tmp_table}")
    cur.execute(f"CREATE TABLE {tmp_table}({cols})")

    n = -1
    for rows in _utils._iter_flist(flist, batch_size=1024):
        data = []
        for row in rows:
            path, num_frames = row.split("\t")
            data.append((n := n + 1, path, int(num_frames)))
        cur.executemany(
            f"INSERT INTO {tmp_table}({cols}) VALUES(?, ?, ?)",
            data,
        )
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


@dataclass
class _AudioData(AudioData):
    _index: int = field(repr=False)

    sample_rate: int = field(default=16_000, init=False)


def LibriSpeech(path: str, *, split: str) -> DataSet[AudioData]:
    """Load or create LibriSpeech dataset.

    Args:
        path: Path where the dataset object is searched or newly created.
            Passing `':memory:'` creates database object in-memory.

        split: Passed to [spdl.dataset.librispeech.get_flist][].

    Returns:
        Dataset object.
    """
    table = f"librispeech_{split}".replace("-", "_")
    idx_col = "_index"
    con = _load_dataset(path, split, table=table, idx_col=idx_col)
    return _sql.make_dataset(con, _AudioData, table=table, _idx_col=idx_col)
