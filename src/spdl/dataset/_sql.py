import dataclasses
import logging
from typing import List

from ._dataset import DataSet

_LG = logging.getLogger(__name__)


def _get_rows_repr(table, num_items, cols, heads, tails):
    types = [type(getattr(heads[0], c)) for c in cols]
    heads = [[str(getattr(r, c)) for c in cols] for r in heads]
    tails = [[str(getattr(r, c)) for c in cols] for r in tails]
    all = heads + tails

    # Figure out minimum width
    widths = []
    for i, c in enumerate(cols):
        widths.append(max(len(c), max(len(r[i]) for r in all)))
    align = ['<' if t == str else '>' for t in types]

    lines = []

    # Metadata
    sep = '═══'
    wrap = lambda s: f"╒═{s}═╕"
    lines.append(wrap(sep.join("═" * w for w in widths)))

    width = len(lines[0])
    wrap_line = lambda s: f"│{s:{width-2}}│"
    lines.append(wrap_line(f"Dataset: {table}"))
    lines.append(wrap_line(f"The number of records: {num_items}"))

    sep = '═╤═'
    wrap = lambda s: f"╞═{s}═╡"
    lines.append(wrap(sep.join("═" * w for w in widths)))

    sep = ' │ '
    wrap = lambda s: f"│ {s} │"
    # column name
    lines.append(wrap(sep.join(f"{c:{a}{w}}" for c, a, w in zip(cols, align, widths))))

    sep = '─┼─'
    wrap = lambda s: f"├─{s}─┤"
    lines.append(wrap(sep.join("─" * w for w in widths)))

    sep = ' │ '
    wrap = lambda s: f"│ {s} │"
    # head rows
    for row in heads:
        lines.append(wrap(sep.join(f"{c:{a}{w}}" for c, a, w in zip(row, align, widths))))
    # ...
    lines.append(wrap(sep.join(f"{'...':^{w}}" for w in widths)))
    # tail rows
    for row in tails:
        lines.append(wrap(sep.join(f"{c:{a}{w}}" for c, a, w in zip(row, align, widths))))

    sep = '═╧═'
    wrap = lambda s: f"╘═{s}═╛"
    lines.append(wrap(sep.join("═" * w for w in widths)))    

    return lines


def _get_limit(start: int, stop: int | None, step: int, _idx_col: str ="rowid"):
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
        return f"WHERE {_idx_col} >= {start} AND {_idx_col} < {stop} ORDER BY {_idx_col}"


class _DataSet:
    """A DataSet implementation backed by sqlite (in-memory or file) database.

    Wrap a database connection object and provide iterable/mapping interface

    Args:
        con (sqlite3.Connection): An interface to the database

        record_class: A dataclass type to represent the data fetched from the database.

        table: The name of the table in the database.

        row_factory (Callable[[splite3.Cursor, Tuple[Any]], record_class] | None):
            *Optional* Custom row_factory attached to [sqlite3.Cursor][] object
            when selecting records from the database for client request.
            If not provided, the selected values are passed to the constructor
            of `record_class`.
    """
    def __init__(
            self,
            con,
            record_class,
            table: str = "dataset",
            **kwargs):
        self.con = con
        self.record_class = record_class
        self.table = table
        self._cols = [f.name for f in dataclasses.fields(record_class) if f.init]
        # The name of the column used for indexing and slicing.
        # This column must contains integer `[0, num_records)`.
        # It will be updated in-place by shuffle and sort
        self._idx_col = kwargs.get("_idx_col", "_index")

    @property
    def attributes(self):
        return [c for c in self._cols if not c.startswith('_')]

    def __len__(self) -> int:
        """Returns the number of entries in the table."""
        cur = self._get_cursor()
        res = cur.execute(f"SELECT count(*) FROM {self.table}")
        return res.fetchone()[0]

    def __repr__(self) -> str:
        cols = [f.name for f in dataclasses.fields(self.record_class) if f.repr]
        num_items = len(self)
        heads = self[:3]
        tails = self[num_items-3:]
        rows = _get_rows_repr(self.table, num_items, cols, heads, tails)
        body = '\n'.join(rows)
        return body

    ###########################################################################
    # Iterable interface
    ###########################################################################
    def _get_query(self, limit = ""):
        cols = ','.join(self._cols)
        query = f"SELECT {cols} FROM {self.table} {limit}"
        # _LG.debug(query)
        return query

    def _factory(self, cursor, row):
        return self.record_class(**{k: v for k, v in zip(self._cols, row)})

    def _get_cursor(self, *, set_factory = False):
        cur = self.con.cursor()
        if set_factory:
            cur.row_factory = self._factory
        return cur

    def __iter__(self):
        return self._get_cursor(set_factory=True).execute(self._get_query())

    ###########################################################################
    # Map interface: (requires `_idx_col` attribute)
    ###########################################################################
    def __getitem__(self, i):
        if isinstance(i, int):
            return self._get_one(i)
        if isinstance(i, slice):
            return self._get_slice(i.start, i.stop, i.step)
        raise TypeError("Indices must be integers or slices")

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
        return [item for item in self._get_cursor(set_factory=True).execute(query)]

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
                f" ORDER BY {order_by};"
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
        self._sort(order_by="RANDOM()")

    def sort(self, column, desc=True):
        self._sort(order_by=f"{column} {'DESC' if desc else 'ASC'}")


def make_dataset(*args, **kwargs):
    return DataSet(_DataSet(*args, **kwargs))
