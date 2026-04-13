# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import asyncio
import multiprocessing
import os
import sys
import tempfile
import unittest
from collections.abc import Awaitable, Callable
from unittest.mock import AsyncMock, MagicMock, patch

from spdl.pipeline._bg_task import BackgroundTask
from spdl.pipeline._pgrp_stats import (
    _collect_pgrp_stats,
    _parse_proc_io,
    _parse_proc_stat,
    _parse_smaps_rollup,
    _pgrp_monitor_subprocess,
    _read_file,
    _read_network_bytes,
    _read_pgrp_stats,
    _warned,
    ProcessGroupResourceUsage,
    ProcessGroupStatsMonitor,
)

_MODULE = "spdl.pipeline._pgrp_stats"


@unittest.skipUnless(sys.platform == "linux", "Requires Linux /proc filesystem")
class LiveProcMonitorTest(unittest.TestCase):
    """Integration test that reads real /proc data without mocking."""

    def test_read_pgrp_stats_returns_valid_data(self) -> None:
        """_read_pgrp_stats should find at least this process."""
        _warned.discard("proc_stat")
        _warned.discard("proc_io")
        _warned.discard("smaps_rollup")

        result = _read_pgrp_stats()
        self.assertGreaterEqual(result.num_procs, 1)
        self.assertGreaterEqual(result.cpu_usec, 0)
        self.assertGreaterEqual(result.rss_bytes, 0)
        self.assertGreaterEqual(result.disk_read_bytes, 0)
        self.assertGreaterEqual(result.disk_write_bytes, 0)
        # smaps_rollup should be available on modern Linux
        if result.pss_bytes is not None:
            self.assertGreater(result.pss_bytes, 0)
            self.assertIsNotNone(result.private_bytes)
            self.assertGreater(result.private_bytes, 0)

    def test_collect_pgrp_stats_returns_complete_snapshot(self) -> None:
        """_collect_pgrp_stats should return a fully populated snapshot."""
        result, cpu_usec, time_usec = _collect_pgrp_stats()
        self.assertIsInstance(result, ProcessGroupResourceUsage)
        self.assertEqual(result.pid, os.getpid())
        self.assertEqual(result.pgid, os.getpgrp())

        # First call: cpu_percent should be None (no previous value).
        self.assertIsNone(result.cpu_percent)
        self.assertIsNotNone(cpu_usec)
        self.assertIsNotNone(result.rss_bytes)
        self.assertIsNotNone(result.num_procs)
        self.assertIsNotNone(result.net_rx_bytes)
        self.assertIsNotNone(result.net_tx_bytes)

        # Second call with prev values: cpu_percent should be set.
        result2, _, _ = _collect_pgrp_stats(cpu_usec, time_usec)
        self.assertIsNotNone(result2.cpu_percent)
        self.assertGreaterEqual(result2.cpu_percent, 0.0)

        # Sanity: at least one process (this one) should be counted.
        assert result.num_procs is not None
        self.assertGreaterEqual(result.num_procs, 1)
        # RSS must be positive for a running process.
        assert result.rss_bytes is not None
        self.assertGreater(result.rss_bytes, 0)


class ReadFileTest(unittest.TestCase):
    def test_read_existing_file(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("hello\n")
            f.flush()
            result = _read_file(f.name)
        self.assertEqual(result, "hello")
        os.unlink(f.name)

    def test_read_nonexistent_file(self) -> None:
        result = _read_file("/nonexistent/path/file.txt")
        self.assertIsNone(result)


class ReadNetworkTest(unittest.TestCase):
    @patch(f"{_MODULE}._read_file")
    def test_network_bytes(self, mock_read: MagicMock) -> None:
        mock_read.return_value = (
            "Inter-|   Receive                                                |  Transmit\n"  # noqa: B950
            " face |bytes    packets errs drop fifo frame compressed multicast|bytes    packets errs drop fifo colls carrier compressed\n"  # noqa: B950
            "    lo: 1000       10    0    0    0     0          0         0     2000      20    0    0    0     0       0          0\n"  # noqa: B950
            "  eth0: 5000       50    0    0    0     0          0         0     3000      30    0    0    0     0       0          0\n"  # noqa: B950
            "  eth1: 7000       70    0    0    0     0          0         0     4000      40    0    0    0     0       0          0"  # noqa: B950
        )
        result = _read_network_bytes()
        # lo is excluded; eth0 + eth1
        self.assertEqual(result.rx_bytes, 12000)
        self.assertEqual(result.tx_bytes, 7000)

    @patch(f"{_MODULE}._read_file")
    def test_network_file_missing(self, mock_read: MagicMock) -> None:
        mock_read.return_value = None
        result = _read_network_bytes()
        self.assertEqual(result.rx_bytes, 0)
        self.assertEqual(result.tx_bytes, 0)

    @patch(f"{_MODULE}._read_file")
    def test_network_malformed_line_raises(self, mock_read: MagicMock) -> None:
        mock_read.return_value = (
            "Inter-|   Receive\n"
            " face |bytes\n"
            "  eth0: bad 0 0 0 0 0 0 0 also_bad 0 0 0 0 0 0 0\n"
        )
        with self.assertRaises(RuntimeError, msg="Failed to parse /proc/net/dev"):
            _read_network_bytes()


class ParseProcStatTest(unittest.TestCase):
    def test_normal_comm(self) -> None:
        content = (
            "12345 (python3) S 100 200 200 0 -1 0 0 0 0 0 500 300 0 0 20 0 1 0 0 0 4096"
        )
        stat = _parse_proc_stat(content)
        self.assertEqual(stat.pgrp, 200)
        self.assertEqual(stat.utime, 500)
        self.assertEqual(stat.stime, 300)
        self.assertEqual(stat.rss, 4096)

    def test_comm_with_spaces_and_parens(self) -> None:
        content = "12345 (my (weird) app) S 100 200 200 0 -1 0 0 0 0 0 500 300 0 0 20 0 1 0 0 0 4096"  # noqa: B950
        stat = _parse_proc_stat(content)
        self.assertEqual(stat.pgrp, 200)

    def test_malformed_no_parens_raises(self) -> None:
        with self.assertRaises(RuntimeError, msg="missing closing paren"):
            _parse_proc_stat("no parens here")

    def test_too_few_fields_raises(self) -> None:
        with self.assertRaises(RuntimeError, msg="expected >=22 fields"):
            _parse_proc_stat("12345 (python3) S 100 200")

    def test_non_numeric_field_raises(self) -> None:
        content = (
            "12345 (python3) S 100 abc 200 0 -1 0 0 0 0 0 500 300 0 0 20 0 1 0 0 0 4096"  # noqa: B950
        )
        with self.assertRaises(RuntimeError, msg="Failed to parse"):
            _parse_proc_stat(content)


class ParseProcIoTest(unittest.TestCase):
    def test_normal_io(self) -> None:
        content = (
            "rchar: 123456\n"
            "wchar: 654321\n"
            "syscr: 100\n"
            "syscw: 200\n"
            "read_bytes: 4096\n"
            "write_bytes: 8192\n"
            "cancelled_write_bytes: 0"
        )
        io = _parse_proc_io(content)
        self.assertEqual(io.read_bytes, 4096)
        self.assertEqual(io.write_bytes, 8192)

    def test_empty_content(self) -> None:
        io = _parse_proc_io("")
        self.assertEqual(io.read_bytes, 0)
        self.assertEqual(io.write_bytes, 0)

    def test_malformed_value_raises(self) -> None:
        content = "read_bytes: not_a_number\n"
        with self.assertRaises(RuntimeError, msg="Failed to parse /proc/[pid]/io"):
            _parse_proc_io(content)


class ParseSmapsRollupTest(unittest.TestCase):
    def test_normal_smaps_rollup(self) -> None:
        content = (
            "00400000-ffffffff ---p 00000000 00:00 0          [rollup]\n"
            "Rss:              123456 kB\n"
            "Pss:               98765 kB\n"
            "Pss_Dirty:         40000 kB\n"
            "Private_Clean:     50000 kB\n"
            "Private_Dirty:     30000 kB\n"
            "Shared_Clean:      20000 kB\n"
            "Shared_Dirty:      10000 kB\n"
        )
        result = _parse_smaps_rollup(content)
        self.assertEqual(result.pss, 98765 * 1024)
        self.assertEqual(result.private_clean, 50000 * 1024)
        self.assertEqual(result.private_dirty, 30000 * 1024)

    def test_empty_content(self) -> None:
        result = _parse_smaps_rollup("")
        self.assertEqual(result.pss, 0)
        self.assertEqual(result.private_clean, 0)
        self.assertEqual(result.private_dirty, 0)

    def test_malformed_value_raises(self) -> None:
        content = "Pss: not_a_number kB\n"
        with self.assertRaises(RuntimeError, msg="Failed to parse"):
            _parse_smaps_rollup(content)


@unittest.skipUnless(sys.platform == "linux", "Requires Linux /proc filesystem")
class ReadPgrpStatsTest(unittest.TestCase):
    def setUp(self) -> None:
        _warned.discard("proc_stat")
        _warned.discard("proc_io")

    @patch(f"{_MODULE}.os.scandir")
    @patch(f"{_MODULE}._read_file")
    @patch(f"{_MODULE}.os.getpgrp")
    def test_sums_processes_in_same_pgrp(
        self,
        mock_getpgrp: MagicMock,
        mock_read: MagicMock,
        mock_scandir: MagicMock,
    ) -> None:
        mock_getpgrp.return_value = 1000

        # Two processes in pgrp 1000, one in pgrp 9999
        entries = []
        for name in ["101", "102", "103", "not_a_pid"]:
            entry = MagicMock()
            entry.name = name
            entry.is_dir.return_value = True
            entries.append(entry)
        mock_scandir.return_value = entries

        def read_side_effect(path: str) -> str | None:
            if path == "/proc/101/stat":
                return "101 (python3) S 1 1000 1000 0 -1 0 0 0 0 0 100 50 0 0 20 0 1 0 0 0 2000"  # noqa: B950
            if path == "/proc/101/smaps_rollup":
                return "00400000-ffffffff ---p 00000000 00:00 0          [rollup]\nRss: 8000 kB\nPss: 6000 kB\nPrivate_Clean: 3000 kB\nPrivate_Dirty: 2000 kB\n"  # noqa: B950
            if path == "/proc/101/io":
                return "rchar: 1000\nwchar: 2000\nsyscr: 10\nsyscw: 20\nread_bytes: 4096\nwrite_bytes: 8192\ncancelled_write_bytes: 0"  # noqa: B950
            if path == "/proc/102/stat":
                return "102 (worker) S 1 1000 1000 0 -1 0 0 0 0 0 200 75 0 0 20 0 1 0 0 0 3000"  # noqa: B950
            if path == "/proc/102/smaps_rollup":
                return "00400000-ffffffff ---p 00000000 00:00 0          [rollup]\nRss: 12000 kB\nPss: 9000 kB\nPrivate_Clean: 5000 kB\nPrivate_Dirty: 3000 kB\n"  # noqa: B950
            if path == "/proc/102/io":
                return "rchar: 3000\nwchar: 4000\nsyscr: 30\nsyscw: 40\nread_bytes: 1024\nwrite_bytes: 2048\ncancelled_write_bytes: 0"  # noqa: B950
            if path == "/proc/103/stat":
                # Different pgrp
                return "103 (other) S 1 9999 9999 0 -1 0 0 0 0 0 999 999 0 0 20 0 1 0 0 0 9999"  # noqa: B950
            return None

        mock_read.side_effect = read_side_effect

        result = _read_pgrp_stats()

        # utime: 100+200=300, stime: 50+75=125, total_ticks=425
        from spdl.pipeline._pgrp_stats import _get_sc_clk_tck

        expected_cpu_usec = 425 * 1_000_000 // _get_sc_clk_tck()
        self.assertEqual(result.cpu_usec, expected_cpu_usec)

        # rss: 2000+3000=5000 pages
        from spdl.pipeline._pgrp_stats import _get_page_size

        expected_rss = 5000 * _get_page_size()
        self.assertEqual(result.rss_bytes, expected_rss)

        # pss: 6000+9000=15000 kB
        self.assertEqual(result.pss_bytes, 15000 * 1024)

        # private: (3000+2000)+(5000+3000)=13000 kB
        self.assertEqual(result.private_bytes, 13000 * 1024)

        # disk IO: 4096+1024=5120 read, 8192+2048=10240 write
        self.assertEqual(result.disk_read_bytes, 5120)
        self.assertEqual(result.disk_write_bytes, 10240)

        self.assertEqual(result.num_procs, 2)

    @patch(f"{_MODULE}.os.scandir")
    @patch(f"{_MODULE}.os.getpgrp")
    def test_scandir_failure_raises(
        self,
        mock_getpgrp: MagicMock,
        mock_scandir: MagicMock,
    ) -> None:
        mock_getpgrp.return_value = 1000
        mock_scandir.side_effect = OSError("permission denied")

        with self.assertRaises(RuntimeError, msg="Failed to scan /proc"):
            _read_pgrp_stats()

    @patch(f"{_MODULE}.os.scandir")
    @patch(f"{_MODULE}._read_file")
    @patch(f"{_MODULE}.os.getpgrp")
    def test_missing_io_and_smaps_file(
        self,
        mock_getpgrp: MagicMock,
        mock_read: MagicMock,
        mock_scandir: MagicMock,
    ) -> None:
        """Disk IO is 0 and PSS/private are None when files are unreadable."""
        mock_getpgrp.return_value = 1000

        entry = MagicMock()
        entry.name = "101"
        mock_scandir.return_value = [entry]

        def read_side_effect(path: str) -> str | None:
            if path == "/proc/101/stat":
                return "101 (python3) S 1 1000 1000 0 -1 0 0 0 0 0 100 50 0 0 20 0 1 0 0 0 2000"  # noqa: B950
            # /proc/101/io and /proc/101/smaps_rollup return None
            return None

        mock_read.side_effect = read_side_effect

        result = _read_pgrp_stats()
        self.assertEqual(result.disk_read_bytes, 0)
        self.assertEqual(result.disk_write_bytes, 0)
        self.assertIsNone(result.pss_bytes)
        self.assertIsNone(result.private_bytes)
        self.assertEqual(result.num_procs, 1)

    @patch(f"{_MODULE}.os.scandir")
    @patch(f"{_MODULE}._read_file")
    @patch(f"{_MODULE}.os.getpgrp")
    def test_malformed_stat_skips_with_warning(
        self,
        mock_getpgrp: MagicMock,
        mock_read: MagicMock,
        mock_scandir: MagicMock,
    ) -> None:
        """A process with malformed stat is skipped and warns once."""
        mock_getpgrp.return_value = 1000

        entries = []
        for name in ["101", "102"]:
            entry = MagicMock()
            entry.name = name
            entries.append(entry)
        mock_scandir.return_value = entries

        def read_side_effect(path: str) -> str | None:
            if path == "/proc/101/stat":
                return "malformed content"  # no parens
            if path == "/proc/102/stat":
                return "102 (python3) S 1 1000 1000 0 -1 0 0 0 0 0 200 75 0 0 20 0 1 0 0 0 3000"  # noqa: B950
            return None

        mock_read.side_effect = read_side_effect

        with self.assertLogs(_MODULE, level="WARNING") as cm:
            result = _read_pgrp_stats()

        # Only pid 102 was counted
        self.assertEqual(result.num_procs, 1)
        self.assertTrue(any("missing closing paren" in m for m in cm.output))


class ProcessGroupStatsMonitorClassTest(unittest.TestCase):
    def test_is_background_task(self) -> None:
        monitor = ProcessGroupStatsMonitor(callback=AsyncMock())
        self.assertIsInstance(monitor, BackgroundTask)


@unittest.skipUnless(sys.platform == "linux", "Requires Linux /proc filesystem")
class ProcessGroupStatsMonitorSubprocessTest(unittest.TestCase):
    def test_monitor_spawns_and_cancels_subprocess(self) -> None:
        """Verify the monitor spawns a subprocess and terminates it on cancel."""
        mock_proc = MagicMock(spec=multiprocessing.Process)
        mock_proc.pid = 12345
        mock_proc.is_alive.side_effect = [True, True, False]

        with patch(f"{_MODULE}.multiprocessing.Process") as mock_cls:
            mock_cls.return_value = mock_proc

            async def run_monitor() -> None:
                monitor = ProcessGroupStatsMonitor(callback=AsyncMock())
                task = asyncio.create_task(monitor.run())
                await asyncio.sleep(0.01)
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            asyncio.run(run_monitor())

        mock_proc.start.assert_called_once()
        mock_proc.terminate.assert_called_once()
        mock_proc.join.assert_called()

    def test_monitor_warns_on_unexpected_exit(self) -> None:
        """Verify warning is logged when the subprocess exits unexpectedly."""
        mock_proc = MagicMock(spec=multiprocessing.Process)
        mock_proc.pid = 12345
        mock_proc.exitcode = 1
        mock_proc.is_alive.return_value = False

        with patch(f"{_MODULE}.multiprocessing.Process") as mock_cls:
            mock_cls.return_value = mock_proc

            async def run_monitor() -> None:
                monitor = ProcessGroupStatsMonitor(callback=AsyncMock())
                await monitor.run()

            with self.assertLogs(_MODULE, level="WARNING") as cm:
                asyncio.run(run_monitor())

        exit_warnings = [m for m in cm.output if "exited unexpectedly" in m]
        self.assertEqual(len(exit_warnings), 1)
        self.assertIn("exit code 1", exit_warnings[0])


@unittest.skipUnless(sys.platform == "linux", "Requires Linux /proc filesystem")
class PgrpMonitorSubprocessFunctionTest(unittest.TestCase):
    @patch(f"{_MODULE}._collect_pgrp_stats")
    def test_subprocess_function_collects_and_calls_callback(
        self,
        mock_collect: MagicMock,
    ) -> None:
        """Test the subprocess entry point invokes the callback."""
        usage = ProcessGroupResourceUsage(
            pid=os.getpid(),
            pgid=os.getpgrp(),
            cpu_percent=50.0,
            rss_bytes=1048576,
            pss_bytes=800000,
            private_bytes=600000,
            disk_read_bytes=4096,
            disk_write_bytes=8192,
            num_procs=3,
            net_rx_bytes=100,
            net_tx_bytes=200,
        )
        mock_collect.return_value = (usage, 500000, 1000000)

        mock_callback = AsyncMock()

        call_count = 0
        original_sleep: Callable[[float], Awaitable[None]] = asyncio.sleep

        async def counting_sleep(delay: float) -> None:
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                raise KeyboardInterrupt
            await original_sleep(0)

        with (
            patch("asyncio.sleep", counting_sleep),
            patch(f"{_MODULE}.signal.signal"),
        ):
            try:
                _pgrp_monitor_subprocess(0.01, mock_callback)
            except KeyboardInterrupt:
                pass

        mock_collect.assert_called()
        mock_callback.assert_called()
        received = mock_callback.call_args[0][0]
        self.assertIsInstance(received, ProcessGroupResourceUsage)
        self.assertEqual(received.cpu_percent, 50.0)
        self.assertEqual(received.rss_bytes, 1048576)
        self.assertEqual(received.pss_bytes, 800000)
        self.assertEqual(received.private_bytes, 600000)
        self.assertEqual(received.disk_read_bytes, 4096)
        self.assertEqual(received.disk_write_bytes, 8192)
        self.assertEqual(received.num_procs, 3)
        self.assertEqual(received.net_rx_bytes, 100)
        self.assertEqual(received.net_tx_bytes, 200)
