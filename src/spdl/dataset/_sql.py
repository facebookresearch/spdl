import dataclasses
import logging
from typing import List

from . import _utils

from ._dataset import DataSet

_LG = logging.getLogger(__name__)


def _get_rows_repr(table, num_items, cols, heads, tails):
    types = (
        [type(getattr(heads[0], c)) for c in cols] if heads else [None for _ in cols]
    )
    heads = [[str(getattr(r, c)) for c in cols] for r in heads]
    tails = [[str(getattr(r, c)) for c in cols] for r in tails]
    all = heads + tails

    # Figure out minimum width
    widths = []
    for i, c in enumerate(cols):
        if all:
            widths.append(max(len(c), max(len(r[i]) for r in all)))
        else:
            widths.append(len(c))
    align = ["<" if t == str else ">" for t in types]

    lines = []

    # Metadata
    sep = "═══"
    wrap = lambda s: f"╒═{s}═╕"
    lines.append(wrap(sep.join("═" * w for w in widths)))

    width = len(lines[0])
    wrap_line = lambda s: f"│{s:{width-2}}│"
    lines.append(wrap_line(f"Dataset: {table}"))
    lines.append(wrap_line(f"The number of records: {num_items}"))

    sep = "═╤═"
    wrap = lambda s: f"╞═{s}═╡"
    lines.append(wrap(sep.join("═" * w for w in widths)))

    sep = " │ "
    wrap = lambda s: f"│ {s} │"
    # column name
    lines.append(wrap(sep.join(f"{c:{a}{w}}" for c, a, w in zip(cols, align, widths))))

    sep = "─┼─"
    wrap = lambda s: f"├─{s}─┤"
    lines.append(wrap(sep.join("─" * w for w in widths)))

    sep = " │ "
    wrap = lambda s: f"│ {s} │"
    # head rows
    for row in heads:
        lines.append(
            wrap(sep.join(f"{c:{a}{w}}" for c, a, w in zip(row, align, widths)))
        )
    # ...
    lines.append(wrap(sep.join(f"{'...':^{w}}" for w in widths)))
    # tail rows
    for row in tails:
        lines.append(
            wrap(sep.join(f"{c:{a}{w}}" for c, a, w in zip(row, align, widths)))
        )

    sep = "═╧═"
    wrap = lambda s: f"╘═{s}═╛"
    lines.append(wrap(sep.join("═" * w for w in widths)))

    return lines


def _get_limit(start: int, stop: int | None, step: int, _idx_col: str):
    if step != 1:
        rem = start % step
        if stop is not None:  # [start:stop:step]
            return (
                f"WHERE {_idx_col} >= {start}"
                f" AND {_idx_col} < {stop}"
                f" AND mod({_idx_col}, {step}) = {rem}"
                f" ORDER BY {_idx_col}"
            )
        else:  # [start::step]
            return (
                f"WHERE {_idx_col} >= {start}"
                f" AND mod({_idx_col}, {step}) = {rem}"
                f" ORDER BY {_idx_col}"
            )
    if stop is None:  # [start::]
        return f"WHERE {_idx_col} >= {start} ORDER BY {_idx_col}"
    else:  # [start:stop:]
        return (
            f"WHERE {_idx_col} >= {start} AND {_idx_col} < {stop} ORDER BY {_idx_col}"
        )


class _DataSet:
    """A DataSet implementation backed by sqlite (in-memory or file) database.

    Wrap a database connection object and provide iterable/mapping interface

    Args:
        con (sqlite3.Connection): An interface to the database

        record_class: A dataclass type to represent the data fetched from the database.

        table: The name of the table in the database.

        _idx_col:
            The name of the column stores integer value os [0, num_records),
            which is used for indexing and slicing.
            It will be updated in-place by shuffle and sort
    """

    def __init__(self, con, record_class, table: str, _idx_col: str):
        self.con = con
        self.record_class = record_class
        self.table = table
        self._idx_col = _idx_col

        self.con.execute(
            f"CREATE INDEX IF NOT EXISTS index_idx ON {table} ({self._idx_col});"
        )

        self._attributes = [
            f.name for f in dataclasses.fields(self.record_class) if f.repr
        ]

        # Columns to be retrieved from database.
        # If an attribute of dataclass is marked as `init=False`, then it is assumed
        # that its values will be initialized some other way, (such as it's constant
        # across the dataset, or dynamically derived from some other value retirieved from
        # the database), so it will be excluded.
        cols = [f.name for f in dataclasses.fields(record_class) if f.init]
        self._base_query = f"SELECT {','.join(cols)} FROM {table}"

        def _row_factory(cursor, row):
            return record_class(**{k: v for k, v in zip(cols, row, strict=True)})

        self._row_factory = _row_factory

    @property
    def attributes(self):
        return self._attributes

    def __len__(self) -> int:
        """Return the number of entries in the table."""
        cur = self._get_cursor()
        res = cur.execute(f"SELECT count(*) FROM {self.table}")
        return res.fetchone()[0]

    def __repr__(self) -> str:
        num_items = len(self)
        heads = self[:3]
        tails = self[num_items - 3 :] if num_items > 3 else []
        rows = _get_rows_repr(self.table, num_items, self.attributes, heads, tails)
        body = "\n".join(rows)
        return body

    ###########################################################################
    # Iterable interface
    ###########################################################################
    def _get_query(self, limit=""):
        query = f"{self._base_query} {limit}"
        _LG.debug(query)
        return query

    def _get_cursor(self, *, set_factory=False):
        cur = self.con.cursor()
        if set_factory:
            cur.row_factory = self._row_factory
        return cur

    def iterate(self, batch_size: int, drop_last: bool, max_batch: int | None):
        limit = _get_limit(start=0, stop=None, step=1, _idx_col=self._idx_col)
        if max_batch is not None:
            if max_batch <= 0:
                raise ValueError(
                    f"`max_batch` must be a positive integer. Found: {max_batch}"
                )
            limit = f"{limit} LIMIT {max_batch * batch_size}"

        query = self._get_query(limit=limit)
        cur = self._get_cursor(set_factory=True)
        return _utils._iter_batch(cur.execute(query), batch_size, drop_last)

    ###########################################################################
    # Map interface: (requires `_idx_col` attribute)
    ###########################################################################
    def __getitem__(self, key: int | slice):
        """Fetch the samples at the given key."""
        if isinstance(key, int):
            return self._get_one(key)
        if isinstance(key, slice):
            return self._get_slice(key.start, key.stop, key.step)
        raise TypeError(f"Expected integer or slice as key. Found: {type(key)}.")

    def _get_one(self, i):
        if i < 0:
            raise IndexError("Negative index is not supported.")

        query = self._get_query(f"WHERE {self._idx_col} = {i}")
        ret = self._get_cursor(set_factory=True).execute(query).fetchone()
        if ret is None:
            raise IndexError("Index out of range.")
        return ret

    def _get_slice(self, start, stop, step):
        start = 0 if start is None else start
        step = 1 if step is None else step

        if start < 0 or (stop is not None and stop < 0) or step < 0:
            raise IndexError("Negative index is not supported.")

        query = self._get_query(_get_limit(start, stop, step, _idx_col=self._idx_col))
        return list(self._get_cursor(set_factory=True).execute(query))

    ################################################################################
    # sort and shuffle
    ################################################################################
    def _sort(self, order_by: str):
        temp_table = "temp.t"
        queries = [
            f"DROP TABLE IF EXISTS {temp_table};",
            (
                f"CREATE TABLE {temp_table} AS"
                f" SELECT {self._idx_col}"
                f" FROM {self.table}"
                f" {order_by};"
            ),
            (
                f"UPDATE {self.table}"
                f" SET {self._idx_col} = tmp.rowid - 1"
                f" FROM {temp_table} tmp"
                f" WHERE {self.table}.{self._idx_col} = tmp.{self._idx_col};"
            ),
            f"DROP TABLE {temp_table};",
        ]
        cur = self._get_cursor()
        for query in queries:
            _LG.debug(query)
            cur.execute(query)
        self.con.commit()

    def shuffle(self):
        """Shuffle the dataset in-place."""
        self._sort(order_by="ORDER BY RANDOM()")

    def sort(self, attribute, desc=True):
        """Sort the dataset by the given attribute."""
        self._sort(order_by=f"ORDER BY {attribute} {'DESC' if desc else 'ASC'}")

    def _split(self, paths: str) -> List[DataSet]:
        return _split(self, paths)


def make_dataset(*args, **kwargs) -> DataSet:
    """Make dataset from the given sqlite database connection."""
    return DataSet(_DataSet(*args, **kwargs))


def _split_table(con, src_table, tgt_table, n, i, idx_col):
    con.execute(f"DROP TABLE IF EXISTS {tgt_table}")
    con.execute(
        f"CREATE TABLE {tgt_table} AS"
        f" SELECT * FROM {src_table}"
        f" WHERE mod(rowid - 1, {n}) = {i};"
    )
    con.execute(f"UPDATE {tgt_table} SET {idx_col} = rowid - 1;")


def _split(src: _DataSet, paths: List[DataSet]) -> List[DataSet]:
    """Split the dataset and create new datasets.

    Args:
        paths: The paths to which new datasets are written.

    !!! note

            If the target database and table exists, the existing table is
            dropped.

    Returns:
        (List[DataSet]): Split DataSet object.
    """
    import sqlite3

    n = len(paths)
    ret = []
    tmp_schema = "split"
    tgt_table = f"{tmp_schema}.{src.table}"
    for i, path in enumerate(paths):
        src.con.execute(f'ATTACH DATABASE "{path}" as {tmp_schema};')
        try:
            _split_table(
                src.con, f"main.{src.table}", f"{tgt_table}", n, i, src._idx_col
            )
            src.con.commit()
        except Exception:
            src.con.rollback()
        finally:
            src.con.execute(f"DETACH {tmp_schema};")

        con = sqlite3.connect(path)
        ret.append(
            make_dataset(con, src.record_class, src.table, _idx_col=src._idx_col)
        )
    return ret
