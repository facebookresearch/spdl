#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""SQLite-based stats logging backend for SPDL pipeline.

This module provides utilities for storing and querying pipeline performance
statistics using SQLite. The design separates concerns into three components:

1. **Data structures**: TaskStatsLogEntry and QueueStatsLogEntry wrap the
   performance statistics with metadata needed for storage.

2. **Writer mechanism**: SQLiteStatsWriter manages a background thread that
   periodically flushes statistics from a shared buffer to SQLite. This
   allows hooks and queues to remain database-agnostic.

3. **Query functions**: Simple standalone functions (query_task_stats,
   query_queue_stats) for reading statistics from the database.

Example usage:

    >>> # Create shared buffer for collecting stats
    >>> from queue import Queue
    >>> buffer: Queue[TaskStatsLogEntry | QueueStatsLogEntry] = Queue()
    >>>
    >>> # Start writer thread to flush buffer to SQLite
    >>> writer = SQLiteStatsWriter("stats.db", buffer)
    >>> writer.start()
    >>>
    >>> # Hooks write stats to buffer (database-agnostic)
    >>> hook = TaskStatsHookWithLogging("task1", buffer, interval=5.0)
    >>> queue = StatsQueueWithLogging("queue1", buffer, interval=5.0)
    >>>
    >>> # Query stats from database (read-only, no writer needed)
    >>> task_stats = query_task_stats("stats.db")
    >>> queue_stats = query_queue_stats("stats.db")
"""

import atexit
import logging
import sqlite3
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from typing import Any, TypedDict

from spdl.pipeline import QueuePerfStats, TaskPerfStats

__all__ = [
    "TaskStatsLogEntry",
    "QueueStatsLogEntry",
    "EventLogEntry",
    "TaskStatsQueryResult",
    "QueueStatsQueryResult",
    "SQLiteStatsWriter",
    "query_task_stats",
    "query_queue_stats",
    "query_event_stats",
    "log_stats_summary",
]

_LG: logging.Logger = logging.getLogger(__name__)


class TaskStatsQueryResult(TypedDict):
    """Query result for task statistics."""

    timestamp: float
    name: str
    num_tasks: int
    num_failures: int
    ave_time: float


class QueueStatsQueryResult(TypedDict):
    """Query result for queue statistics."""

    timestamp: float
    name: str
    elapsed: float
    num_items: int
    ave_put_time: float
    ave_get_time: float
    occupancy_rate: float


# Database schema constants
TASK_STATS_TABLE = """
CREATE TABLE IF NOT EXISTS task_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    name TEXT NOT NULL,
    num_tasks INTEGER NOT NULL,
    num_failures INTEGER NOT NULL,
    ave_time REAL NOT NULL
)
"""

QUEUE_STATS_TABLE = """
CREATE TABLE IF NOT EXISTS queue_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    name TEXT NOT NULL,
    elapsed REAL NOT NULL,
    num_items INTEGER NOT NULL,
    ave_put_time REAL NOT NULL,
    ave_get_time REAL NOT NULL,
    occupancy_rate REAL NOT NULL
)
"""

EVENT_STATS_TABLE = """
CREATE TABLE IF NOT EXISTS event_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    event_name TEXT NOT NULL
)
"""

CREATE_TASK_STATS_INDEX = """
CREATE INDEX IF NOT EXISTS idx_task_stats_timestamp ON task_stats(timestamp)
"""

CREATE_QUEUE_STATS_INDEX = """
CREATE INDEX IF NOT EXISTS idx_queue_stats_timestamp ON queue_stats(timestamp)
"""

CREATE_EVENT_STATS_INDEX = """
CREATE INDEX IF NOT EXISTS idx_event_stats_timestamp ON event_stats(timestamp)
"""


@dataclass
class TaskStatsLogEntry:
    """Log entry for task statistics.

    Wraps TaskPerfStats with additional metadata needed for logging.
    """

    timestamp: float
    name: str
    stats: TaskPerfStats


@dataclass
class QueueStatsLogEntry:
    """Log entry for queue statistics.

    Wraps QueuePerfStats with additional metadata needed for logging.
    """

    timestamp: float
    name: str
    stats: QueuePerfStats


@dataclass
class EventLogEntry:
    """Log entry for generic events (e.g., garbage collection).

    Records timestamp and event name for tracking system events.
    """

    timestamp: float
    event_name: str


# Union type for log entries
_LogEntry = TaskStatsLogEntry | QueueStatsLogEntry | EventLogEntry


def _insert_task_stats(
    cursor: sqlite3.Cursor, entries: Sequence[TaskStatsLogEntry]
) -> None:
    """Insert multiple task statistics entries into the database.

    Args:
        cursor: SQLite database cursor.
        entries: Sequence of task statistics entries to insert.
    """
    if not entries:
        return

    cursor.executemany(
        """
        INSERT INTO task_stats
        (timestamp, name, num_tasks, num_failures, ave_time)
        VALUES (?, ?, ?, ?, ?)
        """,
        [
            (
                entry.timestamp,
                entry.name,
                entry.stats.num_tasks,
                entry.stats.num_failures,
                entry.stats.ave_time,
            )
            for entry in entries
        ],
    )


def _insert_queue_stats(
    cursor: sqlite3.Cursor, entries: Sequence[QueueStatsLogEntry]
) -> None:
    """Insert multiple queue statistics entries into the database.

    Args:
        cursor: SQLite database cursor.
        entries: Sequence of queue statistics entries to insert.
    """
    if not entries:
        return

    cursor.executemany(
        """
        INSERT INTO queue_stats
        (timestamp, name, elapsed, num_items, ave_put_time,
         ave_get_time, occupancy_rate)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                entry.timestamp,
                entry.name,
                entry.stats.elapsed,
                entry.stats.num_items,
                entry.stats.ave_put_time,
                entry.stats.ave_get_time,
                entry.stats.occupancy_rate,
            )
            for entry in entries
        ],
    )


def _insert_event_stats(
    cursor: sqlite3.Cursor, entries: Sequence[EventLogEntry]
) -> None:
    """Insert multiple event entries into the database.

    Args:
        cursor: SQLite database cursor.
        entries: Sequence of event entries to insert.
    """
    if not entries:
        return

    cursor.executemany(
        """
        INSERT INTO event_stats
        (timestamp, event_name)
        VALUES (?, ?)
        """,
        [(entry.timestamp, entry.event_name) for entry in entries],
    )


def _init_database(db_path: Path) -> None:
    """Initialize the database schema if it doesn't exist.

    Args:
        db_path: Path to the SQLite database file.
    """
    # Ensure directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    try:
        cursor = conn.cursor()
        cursor.execute(TASK_STATS_TABLE)
        cursor.execute(QUEUE_STATS_TABLE)
        cursor.execute(EVENT_STATS_TABLE)
        cursor.execute(CREATE_TASK_STATS_INDEX)
        cursor.execute(CREATE_QUEUE_STATS_INDEX)
        cursor.execute(CREATE_EVENT_STATS_INDEX)
        conn.commit()
        _LG.info(f"Initialized SQLite database at {db_path}")
    finally:
        conn.close()


def query_event_stats(
    db_path: str,
    event_name: str | None = None,
    start_time: float | None = None,
    end_time: float | None = None,
) -> Sequence[EventLogEntry]:
    """Query event statistics from the SQLite database.

    This is a simple read-only function that doesn't require a writer instance.
    It's designed for read-only scenarios like plotting and analysis scripts.

    Args:
        db_path: Path to the SQLite database file.
        event_name: Filter by event name (optional). If None, returns stats for all events.
        start_time: Filter by start timestamp (Unix timestamp, optional).
            Only returns records with timestamp >= start_time.
        end_time: Filter by end timestamp (Unix timestamp, optional).
            Only returns records with timestamp <= end_time.

    Returns:
        List of event statistics records as EventLogEntry instances,
        ordered by timestamp in ascending order (oldest first).

    Example:
        >>> # Get all event stats
        >>> all_events = query_event_stats("stats.db")
        >>>
        >>> # Get GC events only
        >>> gc_events = query_event_stats("stats.db", event_name="gc_start")
        >>>
        >>> # Get events within a time range
        >>> recent_events = query_event_stats(
        ...     "stats.db",
        ...     start_time=time.time() - 3600  # Last hour
        ... )
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    try:
        cursor = conn.cursor()
        query = "SELECT * FROM event_stats WHERE 1=1"
        params: list[Any] = []

        if event_name is not None:
            query += " AND event_name = ?"
            params.append(event_name)
        if start_time is not None:
            query += " AND timestamp >= ?"
            params.append(start_time)
        if end_time is not None:
            query += " AND timestamp <= ?"
            params.append(end_time)

        query += " ORDER BY timestamp ASC"

        cursor.execute(query, params)

        # Convert database rows to EventLogEntry instances
        results = []
        for row in cursor.fetchall():
            entry = EventLogEntry(
                timestamp=row["timestamp"],
                event_name=row["event_name"],
            )
            results.append(entry)

        return results

    finally:
        conn.close()


class SQLiteStatsWriter:
    """Background writer thread for flushing statistics to SQLite.

    This class manages a background thread that periodically flushes statistics
    from a shared queue to a SQLite database. It's designed to be
    decoupled from the hooks and queues that populate the buffer, allowing
    flexibility in changing storage backends.

    The writer uses a periodic flushing strategy:
    - Wakes up at regular intervals (configured by `flush_interval`)
    - Drains all entries from the queue
    - Flushes to database if any entries were collected
    - Repeats until shutdown

    Thread Safety:
        Uses queue.Queue which provides built-in thread safety for
        producer-consumer patterns.

    Args:
        db_path: Path to the SQLite database file. The database schema will
            be initialized if it doesn't exist.
        buffer: Shared queue that serves as the buffer for log entries. This
            buffer is populated by hooks/queues and consumed by this writer.
        flush_interval: Duration in seconds to sleep between queue drain cycles.
            Lower values increase CPU usage but reduce latency.

    Example:
        >>> from queue import Queue
        >>> buffer: Queue[TaskStatsLogEntry | QueueStatsLogEntry] = Queue()
        >>> writer = SQLiteStatsWriter("stats.db", buffer)
        >>> writer.start()  # Start background thread
        >>> # ... hooks populate buffer ...
        >>> writer.shutdown()  # Flush remaining entries and stop thread
    """

    def __init__(
        self,
        db_path: str,
        buffer: Queue[_LogEntry],
        flush_interval: float,
    ) -> None:
        self.db_path = Path(db_path)
        self.buffer = buffer
        self.flush_interval = flush_interval

        self._writer_thread: threading.Thread | None = None
        self._shutdown_event = threading.Event()

        # Initialize database schema
        _init_database(self.db_path)

        # Register cleanup on exit
        atexit.register(self.shutdown)

    def start(self) -> None:
        """Start the background writer thread."""
        if self._writer_thread is not None:
            raise RuntimeError("Writer thread already started")

        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            daemon=True,
            name="SQLiteStatsWriter",
        )
        self._writer_thread.start()
        _LG.debug("Started SQLite writer thread")

    def _drain_and_flush(self, conn: sqlite3.Connection) -> None:
        """Drain queue and flush entries to database if any were collected.

        Args:
            conn: SQLite database connection.
        """
        batch: list[_LogEntry] = []
        while not self.buffer.empty():
            try:
                batch.append(self.buffer.get_nowait())
            except Exception:
                break

        if batch:
            self._flush_batch(conn, batch)

    def _writer_loop(self) -> None:
        """Background thread that writes buffered entries to the database."""
        conn = sqlite3.connect(str(self.db_path))

        try:
            while not self._shutdown_event.is_set():
                self._drain_and_flush(conn)
                time.sleep(self.flush_interval)

            # Final flush on shutdown
            self._drain_and_flush(conn)

        finally:
            conn.close()
            _LG.debug("SQLite writer thread stopped")

    def _flush_batch(self, conn: sqlite3.Connection, batch: list[_LogEntry]) -> None:
        """Flush a batch of log entries to the database.

        Args:
            conn: SQLite database connection.
            batch: List of log entries to flush.
        """
        if not batch:
            return

        try:
            cursor = conn.cursor()
            if entries := [e for e in batch if isinstance(e, TaskStatsLogEntry)]:
                _insert_task_stats(cursor, entries)
            if entries := [e for e in batch if isinstance(e, QueueStatsLogEntry)]:
                _insert_queue_stats(cursor, entries)
            if entries := [e for e in batch if isinstance(e, EventLogEntry)]:
                _insert_event_stats(cursor, entries)
            conn.commit()
            _LG.debug(f"Flushed {len(batch)} entries to SQLite")

        except Exception as e:
            _LG.error(f"Error flushing batch to SQLite: {e}")
            conn.rollback()

    def shutdown(self) -> None:
        """Shutdown the writer and flush all pending entries."""
        if self._shutdown_event.is_set():
            return

        _LG.info("Shutting down SQLite stats writer")
        self._shutdown_event.set()

        # Wait for writer thread to finish
        if self._writer_thread and self._writer_thread.is_alive():
            self._writer_thread.join(timeout=10.0)


def query_task_stats(
    db_path: str,
    name: str | None = None,
    start_time: float | None = None,
    end_time: float | None = None,
) -> Sequence[TaskStatsQueryResult]:
    """Query task statistics from the SQLite database.

    This is a simple read-only function that doesn't require a writer instance.
    It's designed for read-only scenarios like plotting and analysis scripts.

    Args:
        db_path: Path to the SQLite database file.
        name: Filter by task name (optional). If None, returns stats for all tasks.
        start_time: Filter by start timestamp (Unix timestamp, optional).
            Only returns records with timestamp >= start_time.
        end_time: Filter by end timestamp (Unix timestamp, optional).
            Only returns records with timestamp <= end_time.

    Returns:
        List of task statistics records as dictionaries (TaskStatsQueryResult),
        ordered by timestamp in descending order (most recent first).

    Example:
        >>> # Get all task stats
        >>> all_stats = query_task_stats("stats.db")
        >>>
        >>> # Get stats for a specific task
        >>> decode_stats = query_task_stats("stats.db", name="decode")
        >>>
        >>> # Get stats within a time range
        >>> recent_stats = query_task_stats(
        ...     "stats.db",
        ...     start_time=time.time() - 3600  # Last hour
        ... )
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    try:
        cursor = conn.cursor()
        query = "SELECT * FROM task_stats WHERE 1=1"
        params: list[Any] = []

        if name is not None:
            query += " AND name = ?"
            params.append(name)
        if start_time is not None:
            query += " AND timestamp >= ?"
            params.append(start_time)
        if end_time is not None:
            query += " AND timestamp <= ?"
            params.append(end_time)

        query += " ORDER BY timestamp DESC"

        cursor.execute(query, params)

        # Convert database rows to dictionaries
        results: list[TaskStatsQueryResult] = []
        for row in cursor.fetchall():
            entry: TaskStatsQueryResult = {
                "timestamp": row["timestamp"],
                "name": row["name"],
                "num_tasks": row["num_tasks"],
                "num_failures": row["num_failures"],
                "ave_time": row["ave_time"],
            }
            results.append(entry)

        return results

    finally:
        conn.close()


def query_queue_stats(
    db_path: str,
    name: str | None = None,
    start_time: float | None = None,
    end_time: float | None = None,
) -> Sequence[QueueStatsQueryResult]:
    """Query queue statistics from the SQLite database.

    This is a simple read-only function that doesn't require a writer instance.
    It's designed for read-only scenarios like plotting and analysis scripts.

    Args:
        db_path: Path to the SQLite database file.
        name: Filter by queue name (optional). If None, returns stats for all queues.
        start_time: Filter by start timestamp (Unix timestamp, optional).
            Only returns records with timestamp >= start_time.
        end_time: Filter by end timestamp (Unix timestamp, optional).
            Only returns records with timestamp <= end_time.

    Returns:
        List of queue statistics records as dictionaries (QueueStatsQueryResult),
        ordered by timestamp in descending order (most recent first).

    Example:
        >>> # Get all queue stats
        >>> all_stats = query_queue_stats("stats.db")
        >>>
        >>> # Get stats for a specific queue
        >>> queue0_stats = query_queue_stats("stats.db", name="queue_0")
        >>>
        >>> # Get stats within a time range
        >>> recent_stats = query_queue_stats(
        ...     "stats.db",
        ...     start_time=time.time() - 3600  # Last hour
        ... )
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    try:
        cursor = conn.cursor()
        query = "SELECT * FROM queue_stats WHERE 1=1"
        params: list[Any] = []

        if name is not None:
            query += " AND name = ?"
            params.append(name)
        if start_time is not None:
            query += " AND timestamp >= ?"
            params.append(start_time)
        if end_time is not None:
            query += " AND timestamp <= ?"
            params.append(end_time)

        query += " ORDER BY timestamp DESC"

        cursor.execute(query, params)

        # Convert database rows to dictionaries
        results: list[QueueStatsQueryResult] = []
        for row in cursor.fetchall():
            entry: QueueStatsQueryResult = {
                "timestamp": row["timestamp"],
                "name": row["name"],
                "elapsed": row["elapsed"],
                "num_items": row["num_items"],
                "ave_put_time": row["ave_put_time"],
                "ave_get_time": row["ave_get_time"],
                "occupancy_rate": row["occupancy_rate"],
            }
            results.append(entry)

        return results

    finally:
        conn.close()


def log_stats_summary(db_path: str | Path) -> None:
    """Log a summary of the collected statistics using the logger.

    This function aggregates all log entries to compute totals and averages
    across the entire pipeline execution.

    Args:
        db_path: Path to the SQLite database.
    """
    db_path_str = str(db_path)

    _LG.info("=" * 80)
    _LG.info("PIPELINE PERFORMANCE SUMMARY")
    _LG.info("=" * 80)

    # Task stats
    task_stats = query_task_stats(db_path_str)
    if task_stats:
        _LG.info("📊 Task Statistics:")
        _LG.info("-" * 80)

        # Group by name
        task_names: set[str] = {stat["name"] for stat in task_stats}
        for name in sorted(task_names):
            name_stats = [s for s in task_stats if s["name"] == name]
            if name_stats:
                # Calculate totals across all log entries
                total_tasks = sum(s["num_tasks"] for s in name_stats)
                total_failures = sum(s["num_failures"] for s in name_stats)

                # Calculate weighted average time (weighted by num_tasks)
                weighted_sum = sum(s["ave_time"] * s["num_tasks"] for s in name_stats)
                avg_time = weighted_sum / total_tasks if total_tasks > 0 else 0.0

                success_rate = (
                    (total_tasks - total_failures) / max(1, total_tasks) * 100
                )

                _LG.info("")
                _LG.info("  Stage: %s", name)
                _LG.info("    Total tasks processed: %d", total_tasks)
                _LG.info("    Total failures: %d", total_failures)
                _LG.info("    Average task time: %.4fs", avg_time)
                _LG.info("    Success rate: %.1f%%", success_rate)
                _LG.info("    Number of log entries: %d", len(name_stats))

    # Queue stats
    queue_stats = query_queue_stats(db_path_str)
    if queue_stats:
        _LG.info("")
        _LG.info("📈 Queue Statistics:")
        _LG.info("-" * 80)

        # Group by name
        queue_names: set[str] = {stat["name"] for stat in queue_stats}
        for name in sorted(queue_names):
            name_stats = [s for s in queue_stats if s["name"] == name]
            if name_stats:
                # Calculate totals across all log entries
                total_items = sum(s["num_items"] for s in name_stats)
                total_elapsed: float = sum(s["elapsed"] for s in name_stats)

                # Calculate weighted averages (weighted by num_items)
                weighted_put_time = sum(
                    s["ave_put_time"] * s["num_items"] for s in name_stats
                )
                weighted_get_time = sum(
                    s["ave_get_time"] * s["num_items"] for s in name_stats
                )
                weighted_occupancy = sum(
                    s["occupancy_rate"] * s["num_items"] for s in name_stats
                )

                avg_put_time = (
                    weighted_put_time / total_items if total_items > 0 else 0.0
                )
                avg_get_time = (
                    weighted_get_time / total_items if total_items > 0 else 0.0
                )
                avg_occupancy = (
                    weighted_occupancy / total_items if total_items > 0 else 0.0
                )

                # Calculate average QPS
                avg_qps = (
                    float(total_items) / total_elapsed if total_elapsed > 0 else 0.0
                )

                _LG.info("")
                _LG.info("  Queue: %s", name)
                _LG.info("    Total items processed: %d", total_items)
                _LG.info("    Average QPS: %.2f", avg_qps)
                _LG.info("    Average put time: %.4fs", avg_put_time)
                _LG.info("    Average get time: %.4fs", avg_get_time)
                _LG.info("    Average occupancy rate: %.1f%%", avg_occupancy * 100)
                _LG.info("    Number of log entries: %d", len(name_stats))

    _LG.info("")
    _LG.info("=" * 80)
    _LG.info("Database: %s", db_path)
    _LG.info("=" * 80)
