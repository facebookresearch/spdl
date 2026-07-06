# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Process-group resource monitoring (Linux-only).

Collects CPU, memory (RSS, PSS, private), disk IO, and network bytes
across all processes sharing a PGID by reading ``/proc``.
"""

from __future__ import annotations

import asyncio
import logging
import multiprocessing
import multiprocessing.context
import os
import signal
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from spdl.pipeline._bg_task import BackgroundTask

__all__ = [
    "ProcessGroupResourceUsage",
    "ProcessGroupStatsMonitor",
]

_LG: logging.Logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_warned: set[str] = set()


def _warn_once(key: str, msg: str, *args: object) -> None:
    """Log a warning at most once per key."""
    if key not in _warned:
        _warned.add(key)
        _LG.warning(msg, *args)


_PGRP_INTERVAL_SEC: float = 60.0

_SC_CLK_TCK: int | None = None
_PAGE_SIZE: int | None = None


def _get_sc_clk_tck() -> int:
    global _SC_CLK_TCK
    if _SC_CLK_TCK is None:
        _SC_CLK_TCK = os.sysconf("SC_CLK_TCK")
    return _SC_CLK_TCK


def _get_page_size() -> int:
    global _PAGE_SIZE
    if _PAGE_SIZE is None:
        _PAGE_SIZE = os.sysconf("SC_PAGE_SIZE")
    return _PAGE_SIZE


# ---------------------------------------------------------------------------
# /proc parsers
# ---------------------------------------------------------------------------


def _read_file(path: str) -> str | None:
    """Read a file, returning None if it doesn't exist or can't be read."""
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


@dataclass
class _SmapsRollup:
    pss: int
    private_clean: int
    private_dirty: int


def _parse_smaps_rollup(content: str) -> _SmapsRollup:
    """Parse ``/proc/[pid]/smaps_rollup`` for PSS and private memory.

    Example ``/proc/[pid]/smaps_rollup`` content (partial)::

        00400000-ffffffff ---p 00000000 00:00 0          [rollup]
        Rss:              123456 kB
        Pss:               98765 kB
        Private_Clean:     50000 kB
        Private_Dirty:     30000 kB
        ...

    Values are in kB.  We return bytes.

    Raises:
        RuntimeError: If the content is malformed.
    """
    pss = 0
    private_clean = 0
    private_dirty = 0
    try:
        for line in content.splitlines():
            parts = line.split()
            if len(parts) < 3:
                continue
            if parts[0] == "Pss:":
                pss = int(parts[1]) * 1024
            elif parts[0] == "Private_Clean:":
                private_clean = int(parts[1]) * 1024
            elif parts[0] == "Private_Dirty:":
                private_dirty = int(parts[1]) * 1024
    except (ValueError, IndexError) as e:
        raise RuntimeError(
            f"Failed to parse /proc/[pid]/smaps_rollup: {content[:120]}"
        ) from e
    return _SmapsRollup(
        pss=pss, private_clean=private_clean, private_dirty=private_dirty
    )


@dataclass
class _ProcStat:
    pgrp: int
    utime: int
    stime: int
    rss: int


def _parse_proc_stat(content: str) -> _ProcStat:
    """Parse ``/proc/[pid]/stat``.

    Example ``/proc/[pid]/stat`` content::

        12345 (python3) S 100 200 200 0 -1 4194304 0 0 0 0 500 300 0 0 20 0 1 0 12345 0 4096 ...

    Format: ``pid (comm) state ppid pgrp session ...``
    The comm field can contain spaces and parentheses, so we find the last
    ``)`` and split everything after it.  Fields after comm (0-indexed):
      [0] state, [1] ppid, [2] pgrp, [11] utime, [12] stime, [21] rss, ...

    Raises:
        RuntimeError: If the content is malformed or has too few fields.
    """
    close_paren = content.rfind(")")
    if close_paren < 0:
        raise RuntimeError(
            f"Unexpected /proc/[pid]/stat format: missing closing paren: {content[:80]}"
        )
    fields = content[close_paren + 2 :].split()
    if len(fields) < 22:
        raise RuntimeError(
            f"Unexpected /proc/[pid]/stat format: "
            f"expected >=22 fields after comm, got {len(fields)}"
        )
    try:
        return _ProcStat(
            pgrp=int(fields[2]),
            utime=int(fields[11]),
            stime=int(fields[12]),
            rss=int(fields[21]),
        )
    except (ValueError, IndexError) as e:
        raise RuntimeError(f"Failed to parse /proc/[pid]/stat: {content[:80]}") from e


@dataclass
class _ProcIO:
    read_bytes: int
    write_bytes: int


def _parse_proc_io(content: str) -> _ProcIO:
    """Parse ``/proc/[pid]/io``.

    Example ``/proc/[pid]/io`` content::

        rchar: 123456
        wchar: 654321
        syscr: 100
        syscw: 200
        read_bytes: 4096
        write_bytes: 8192
        cancelled_write_bytes: 0

    ``read_bytes`` / ``write_bytes`` are actual storage IO bytes (not
    page-cache reads/writes like ``rchar`` / ``wchar``).

    Raises:
        RuntimeError: If the content contains non-numeric values.
    """
    read_bytes = 0
    write_bytes = 0
    try:
        for line in content.splitlines():
            parts = line.split()
            if len(parts) < 2:
                continue
            if parts[0] == "read_bytes:":
                read_bytes = int(parts[1])
            elif parts[0] == "write_bytes:":
                write_bytes = int(parts[1])
    except (ValueError, IndexError) as e:
        raise RuntimeError(f"Failed to parse /proc/[pid]/io: {content[:80]}") from e
    return _ProcIO(read_bytes=read_bytes, write_bytes=write_bytes)


@dataclass
class _NetworkBytes:
    rx_bytes: int
    tx_bytes: int


def _read_network_bytes() -> _NetworkBytes:
    """Read network RX/TX bytes from ``/proc/net/dev``.

    Example ``/proc/net/dev`` content::

        Inter-|   Receive                                                |  Transmit
         face |bytes packets errs drop fifo frame compressed multicast|bytes packets ...
            lo: 1234    10    0    0    0     0          0         0     5678    20 ...
          eth0: 9999    50    0    0    0     0          0         0     8888    30 ...

    Each line after the two header lines has the interface name followed by
    16 counters.  Column 1 (0-indexed) is RX bytes, column 9 is TX bytes.
    Loopback (``lo``) is excluded.

    Raises:
        RuntimeError: If the file content is malformed.
    """
    rx_bytes = 0
    tx_bytes = 0
    content = _read_file("/proc/net/dev")
    if content:
        for line in content.splitlines()[2:]:  # skip header lines
            parts = line.split()
            if len(parts) < 10:
                continue
            iface = parts[0].rstrip(":")
            if iface == "lo":
                continue
            try:
                rx_bytes += int(parts[1])
                tx_bytes += int(parts[9])
            except (ValueError, IndexError) as e:
                raise RuntimeError(
                    f"Failed to parse /proc/net/dev line: {line.strip()[:80]}"
                ) from e
    return _NetworkBytes(rx_bytes=rx_bytes, tx_bytes=tx_bytes)


# ---------------------------------------------------------------------------
# Aggregation across the process group
# ---------------------------------------------------------------------------


@dataclass
class _PgrpStats:
    cpu_usec: int
    rss_bytes: int
    pss_bytes: int | None
    private_bytes: int | None
    disk_read_bytes: int
    disk_write_bytes: int
    num_procs: int


def _read_pgrp_stats() -> _PgrpStats:
    """Sum CPU, RSS, PSS, private memory, and disk IO across all PIDs in this process group.

    Raises:
        RuntimeError: If /proc cannot be scanned.
    """
    pgid = os.getpgrp()
    total_utime = 0
    total_stime = 0
    total_rss = 0
    total_pss = 0
    total_private = 0
    has_smaps = False
    total_read_bytes = 0
    total_write_bytes = 0
    num_procs = 0

    try:
        for entry in os.scandir("/proc"):
            if not entry.name.isdigit():
                continue
            content = _read_file(f"/proc/{entry.name}/stat")
            if content is None:
                continue
            try:
                stat = _parse_proc_stat(content)
            except RuntimeError as e:
                _warn_once("proc_stat", "%s", e)
                continue
            if stat.pgrp != pgid:
                continue
            total_utime += stat.utime
            total_stime += stat.stime
            total_rss += stat.rss
            num_procs += 1

            smaps_content = _read_file(f"/proc/{entry.name}/smaps_rollup")
            if smaps_content is not None:
                try:
                    smaps = _parse_smaps_rollup(smaps_content)
                except RuntimeError as e:
                    _warn_once("smaps_rollup", "%s", e)
                else:
                    has_smaps = True
                    total_pss += smaps.pss
                    total_private += smaps.private_clean + smaps.private_dirty

            io_content = _read_file(f"/proc/{entry.name}/io")
            if io_content is not None:
                try:
                    io = _parse_proc_io(io_content)
                except RuntimeError as e:
                    _warn_once("proc_io", "%s", e)
                else:
                    total_read_bytes += io.read_bytes
                    total_write_bytes += io.write_bytes
    except OSError as e:
        raise RuntimeError(f"Failed to scan /proc: {e}") from e

    cpu_usec = (total_utime + total_stime) * 1_000_000 // _get_sc_clk_tck()

    return _PgrpStats(
        cpu_usec=cpu_usec,
        rss_bytes=total_rss * _get_page_size(),
        pss_bytes=total_pss if has_smaps else None,
        private_bytes=total_private if has_smaps else None,
        disk_read_bytes=total_read_bytes,
        disk_write_bytes=total_write_bytes,
        num_procs=num_procs,
    )


@dataclass
class ProcessGroupResourceUsage:
    """Snapshot of resource usage across all processes in the process group.

    Collected periodically by :class:`ProcessGroupStatsMonitor` and passed
    to the user-provided callback.

    **Memory metrics** — three complementary views are provided:

    * **RSS** (Resident Set Size, from ``/proc/[pid]/stat``):
      Total physical pages mapped by each process. Shared pages (shared
      libraries, CUDA context, etc.) are counted once *per process* that
      maps them, so summing RSS across a process group **overcounts**
      actual physical memory when pages are shared.

    * **PSS** (Proportional Set Size, from ``/proc/[pid]/smaps_rollup``):
      Each shared page is divided equally among all processes that map it.
      Summing PSS across a process group gives the most accurate estimate
      of actual physical memory consumption without double-counting.

    * **Private bytes** (``Private_Clean + Private_Dirty`` from
      ``/proc/[pid]/smaps_rollup``):
      Only pages exclusive to each process — memory that would be freed
      if the process exited. This **undercounts** total usage because it
      excludes shared memory entirely, but isolates per-process
      allocations (model weights, activations, buffers).

    The difference ``RSS − Private`` approximates each process's shared
    memory contribution.  Reading ``smaps_rollup`` is more expensive than
    ``stat`` (the kernel walks page tables), but it is a single-file read
    per process so the overhead is modest.

    **Which metric to use:**

    * Use **PSS** as the primary metric for comparing memory across
      configuration changes — it reflects the true physical memory cost
      of the process group without double-counting.
    * Use **Private** to isolate per-process allocations (model weights,
      activations, buffers) from shared overhead.
    * Use **RSS** as an upper-bound sanity check.  When ``num_procs == 1``,
      RSS equals PSS.
    * ``PSS − Private`` can be derived in queries to see how much shared
      memory is attributed to this group.
    """

    pid: int
    """PID of the monitoring process."""

    pgid: int
    """Process group ID being monitored."""

    cpu_percent: float | None = None
    """CPU utilization as a percentage of a single core over the last interval.

    Computed as ``delta_cpu_usec / delta_wall_usec * 100``.  A value of
    200.0 means two cores were fully utilized.  ``None`` on the first
    snapshot (no previous value to diff against).
    """

    rss_bytes: int | None = None
    """Total resident set size in bytes across the process group.

    Overcounts physical memory when pages are shared across processes.
    See class docstring for details.
    """

    pss_bytes: int | None = None
    """Total proportional set size in bytes across the process group.

    Shared pages are divided by the number of sharers, giving the most
    accurate view of actual physical memory cost.  ``None`` when
    ``smaps_rollup`` is unavailable.
    """

    private_bytes: int | None = None
    """Total private memory (Private_Clean + Private_Dirty) in bytes.

    Only pages exclusive to each process.  ``None`` when
    ``smaps_rollup`` is unavailable.
    """

    disk_read_bytes: int | None = None
    """Total bytes read from storage across the process group."""

    disk_write_bytes: int | None = None
    """Total bytes written to storage across the process group."""

    num_procs: int | None = None
    """Number of processes in the process group."""

    net_rx_bytes: int | None = None
    """Network bytes received since the previous snapshot (excluding loopback).

    ``None`` on the first snapshot (no previous value to diff against).
    Note: this is host-wide (per network namespace), not process-group-scoped.
    """

    net_tx_bytes: int | None = None
    """Network bytes transmitted since the previous snapshot (excluding loopback).

    ``None`` on the first snapshot (no previous value to diff against).
    Note: this is host-wide (per network namespace), not process-group-scoped.
    """


def _collect_pgrp_stats(
    prev_cpu_usec: int | None = None,
    prev_time_usec: int | None = None,
    prev_net_rx_bytes: int | None = None,
    prev_net_tx_bytes: int | None = None,
) -> tuple[ProcessGroupResourceUsage, int | None, int, int | None, int | None]:
    """Collect all process-group stats.

    Reads CPU, memory (RSS, PSS, private), disk IO from
    ``/proc/[pid]/stat``, ``/proc/[pid]/smaps_rollup``, and
    ``/proc/[pid]/io``, and network bytes from ``/proc/net/dev``
    for all processes in the current process group.

    Args:
        prev_cpu_usec: Cumulative CPU µs from the previous snapshot
            (used to compute ``cpu_percent``).  ``None`` on the first call.
        prev_time_usec: Wall-clock µs (``time.monotonic()`` × 1e6) of the
            previous snapshot.
        prev_net_rx_bytes: Cumulative network RX bytes from the previous
            snapshot.  ``None`` on the first call.
        prev_net_tx_bytes: Cumulative network TX bytes from the previous
            snapshot.  ``None`` on the first call.

    Returns:
        A tuple of
        ``(usage, current_cpu_usec, current_time_usec, current_net_rx_bytes, current_net_tx_bytes)``.
        ``current_cpu_usec`` is ``None`` when /proc could not be read.
        ``current_net_*_bytes`` are ``None`` when /proc/net/dev could not be read.
    """
    import time as _time

    now_usec = int(_time.monotonic() * 1_000_000)
    result = ProcessGroupResourceUsage(pid=os.getpid(), pgid=os.getpgrp())
    current_cpu_usec: int | None = None
    current_net_rx_bytes: int | None = None
    current_net_tx_bytes: int | None = None

    try:
        pgrp = _read_pgrp_stats()
        current_cpu_usec = pgrp.cpu_usec
        if (
            prev_cpu_usec is not None
            and prev_time_usec is not None
            and now_usec > prev_time_usec
        ):
            delta_cpu = pgrp.cpu_usec - prev_cpu_usec
            delta_wall = now_usec - prev_time_usec
            result.cpu_percent = delta_cpu / delta_wall * 100.0
        result.rss_bytes = pgrp.rss_bytes
        result.pss_bytes = pgrp.pss_bytes
        result.private_bytes = pgrp.private_bytes
        result.disk_read_bytes = pgrp.disk_read_bytes
        result.disk_write_bytes = pgrp.disk_write_bytes
        result.num_procs = pgrp.num_procs
    except RuntimeError as e:
        _warn_once("pgrp_stats", "%s", e)

    try:
        net = _read_network_bytes()
        current_net_rx_bytes = net.rx_bytes
        current_net_tx_bytes = net.tx_bytes
        if prev_net_rx_bytes is not None and prev_net_tx_bytes is not None:
            result.net_rx_bytes = net.rx_bytes - prev_net_rx_bytes
            result.net_tx_bytes = net.tx_bytes - prev_net_tx_bytes
    except RuntimeError as e:
        _warn_once("net_dev", "%s", e)

    return (
        result,
        current_cpu_usec,
        now_usec,
        current_net_rx_bytes,
        current_net_tx_bytes,
    )


# ---------------------------------------------------------------------------
# Subprocess entry point and BackgroundTask wrapper
# ---------------------------------------------------------------------------


def _pgrp_monitor_subprocess(
    interval: float,
    callback: Callable[[ProcessGroupResourceUsage], Awaitable[None]],
) -> None:
    """Entry point for the monitoring subprocess.

    Runs a standalone asyncio loop that collects process-group stats and
    dispatches them to *callback*. Both SIGINT (Ctrl-C) and SIGTERM
    (terminate) set a flag to exit the loop gracefully.
    """
    running: bool = True

    def _handle_term(signum: int, frame: object) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, _handle_term)
    signal.signal(signal.SIGTERM, _handle_term)

    async def _loop() -> None:
        loop = asyncio.get_running_loop()
        prev_cpu_usec: int | None = None
        prev_time_usec: int | None = None
        prev_net_rx_bytes: int | None = None
        prev_net_tx_bytes: int | None = None
        while running:
            try:
                (
                    usage,
                    prev_cpu_usec,
                    prev_time_usec,
                    prev_net_rx_bytes,
                    prev_net_tx_bytes,
                ) = await loop.run_in_executor(
                    None,
                    _collect_pgrp_stats,
                    prev_cpu_usec,
                    prev_time_usec,
                    prev_net_rx_bytes,
                    prev_net_tx_bytes,
                )
                await callback(usage)
            except Exception:
                pass  # best-effort, avoid crashing the subprocess
            await asyncio.sleep(interval)

    asyncio.run(_loop())


class ProcessGroupStatsMonitor(BackgroundTask):
    """Background task that spawns a subprocess to collect per-process-group stats.

    In multi-rank training, multiple ranks on the same host often share a
    single cgroup.  External monitoring systems (e.g., dynolog) report
    metrics per cgroup, so they cannot attribute resource usage to
    individual ranks.  This monitor solves that by collecting stats per
    process group — each rank typically runs as its own PGID, giving
    per-rank granularity even when ranks share a cgroup.

    Sums CPU, memory (RSS, PSS, private), disk IO, and network bytes
    across all processes sharing this rank's PGID.

    Runs in a separate subprocess so the main Python runtime is not
    affected by GC pauses or CPU overhead from /proc scanning.

    Args:
        callback: An async callable that receives a
            :class:`ProcessGroupResourceUsage` snapshot each interval.
        interval: Collection interval in seconds (default 60).
        mp_context: A :mod:`multiprocessing` context (e.g. from
            ``multiprocessing.get_context("forkserver")``) used to spawn the
            monitor subprocess.  ``None`` (default) uses the default context.
    """

    def __init__(
        self,
        callback: Callable[[ProcessGroupResourceUsage], Awaitable[None]],
        interval: float = _PGRP_INTERVAL_SEC,
        mp_context: multiprocessing.context.BaseContext | None = None,
    ) -> None:
        super().__init__()
        self._callback = callback
        self._interval = interval
        self._mp_context = mp_context

    async def run(self) -> None:
        """Spawn a daemon subprocess to collect process-group stats and wait for it.

        The subprocess runs its own asyncio event loop that periodically
        reads ``/proc`` to gather CPU, memory, disk IO, and network stats
        for all processes in this process group, then invokes the
        user-provided callback with a :class:`ProcessGroupResourceUsage`
        snapshot.

        This method polls the subprocess every 5 seconds until it exits
        or this coroutine is cancelled.  On cancellation the subprocess
        is terminated (then killed if it does not exit within 5 seconds)
        and ``CancelledError`` is re-raised.

        A daemon process cannot spawn children, so the monitor subprocess cannot
        be started from one (e.g. a worker-pool process running a nested pipeline,
        which runs as a daemon). In that case this skips with a warning rather
        than raising: a main-process monitor already covers the whole process
        group, so a per-worker monitor here would be redundant anyway.
        """
        if multiprocessing.current_process().daemon:
            _LG.warning(
                "Skipping process-group stats monitor: a daemon process cannot "
                "spawn the monitor subprocess (pid=%d).",
                os.getpid(),
            )
            return
        ctx = self._mp_context or multiprocessing.get_context()
        # pyrefly: ignore [missing-attribute]
        proc = ctx.Process(
            target=_pgrp_monitor_subprocess,
            args=(self._interval, self._callback),
            daemon=True,
        )
        proc.start()
        _LG.info(
            "Process-group stats monitor subprocess started "
            "(main_pid=%d, monitor_pid=%d, pgid=%d)",
            os.getpid(),
            proc.pid,
            os.getpgrp(),
        )
        try:
            while proc.is_alive():
                await asyncio.sleep(5)
            _LG.warning(
                "Process-group stats monitor subprocess exited unexpectedly "
                "(exit code %s)",
                proc.exitcode,
            )
        except asyncio.CancelledError:
            proc.terminate()
            proc.join(timeout=5)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=1)
            _LG.info(
                "Process-group stats monitor subprocess stopped (pid=%d)",
                proc.pid,
            )
            raise
