# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Public CPU payload/epoch/CLI coverage for the portable PIN benchmark."""

from __future__ import annotations

import ast
import inspect
import io
import json
import os
import unittest
from contextlib import redirect_stderr, redirect_stdout
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

import numpy as np
import spdl.examples.benchmark_pinned_arena as benchmark
import torch
from parameterized import parameterized
from spdl.pipeline import (
    iterate_in_subprocess,
    SharedMemoryRingBuffer,
    SharedMemorySegmentPool,
)


def _config(**overrides: Any) -> dict[str, Any]:
    return {
        "size": 8192,
        "kind": "torch",
        "dtype": "fp32",
        "items": 3,
        "workers": 2,
        "slots": 2,
        "warmup_epochs": 1,
        "timeout": 60.0,
        "local_rank": 0,
        "device": "cpu",
        **overrides,
    }


class PinnedArenaBenchmarkTest(unittest.TestCase):
    def setUp(self) -> None:
        threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, threads)
        benchmark.configure_process()

    @parameterized.expand(
        [
            (kind, dtype)
            for kind in ("torch", "numpy", "nested")
            for dtype in ("fp32", "fp16", "uint8")
        ]
    )
    def test_normal_import_payload_full_values(self, kind: str, dtype: str) -> None:
        """Normal imports create fresh CPU payloads with every deterministic value checked."""
        self.assertIs(benchmark.np, np)
        self.assertIs(benchmark.torch, torch)
        first, _ = benchmark.restore_payload(
            benchmark.make_payload(8193, kind, dtype, 7)
        )
        second, _ = benchmark.restore_payload(
            benchmark.make_payload(8193, kind, dtype, 8)
        )
        benchmark.verify_payload(first, 8193, kind, dtype, 7)
        leaf = first["main"] if kind == "nested" else first
        other = second["main"] if kind == "nested" else second
        self.assertFalse(torch.equal(leaf, other))
        self.assertEqual(leaf.device.type, "cpu")
        self.assertFalse(torch.cuda.is_initialized())

    @parameterized.expand([(kind,) for kind in ("torch", "numpy", "nested")])
    def test_whole_value_and_structure_corruption(self, kind: str) -> None:
        """A middle-value or nested-structure corruption fails full verification."""
        value, _ = benchmark.restore_payload(
            benchmark.make_payload(8192, kind, "fp32", 3)
        )
        leaf = value["nested"][1][0] if kind == "nested" else value
        leaf[len(leaf) // 2] += 1
        with self.assertRaisesRegex(AssertionError, "entire payload value mismatch"):
            benchmark.verify_payload(value, 8192, kind, "fp32", 3)
        if kind == "nested":
            value.pop("nested")
            with self.assertRaisesRegex(AssertionError, "structure mismatch"):
                benchmark.verify_payload(value, 8192, kind, "fp32", 3)
        self.assertFalse(torch.cuda.is_initialized())

    def test_reiterable_factory_epoch_identity(self) -> None:
        """The top-level factory creates fresh non-overlapping producer/epoch identities."""
        config = _config(items=2)
        source = benchmark.produce_payloads(config, worker=1)
        first, second = list(source), list(source)
        self.assertEqual([item["sequence"] for item in first], [2, 3])
        self.assertEqual([item["sequence"] for item in second], [6, 7])
        self.assertEqual([item["epoch"] for item in second], [1, 1])
        self.assertIsNot(first[0]["payload"], second[0]["payload"])
        self.assertFalse(first[0]["producer"]["cuda_initialized"])
        self.assertEqual(first[0]["producer"]["cpu_threads"], 1)

    @parameterized.expand(
        [
            (arm, kind)
            for arm in ("pool-pageable", "ring-pageable")
            for kind in ("torch", "numpy", "nested")
        ]
    )
    def test_public_cpu_workflows(self, arm: str, kind: str) -> None:
        """Real public boundaries verify two producers over warmup and measured CPU epochs."""
        config = _config(kind=kind)
        result = benchmark.run_benchmark(arm, config)
        warm, measured = result["warmup"][0]["items"], result["measured"]["items"]
        self.assertEqual(result["schema"], "public-iterate-v1")
        self.assertEqual(len(warm), 6)
        self.assertEqual(len(measured), 6)
        self.assertEqual({row["sequence"] for row in warm}, set(range(6)))
        self.assertEqual({row["sequence"] for row in measured}, set(range(6, 12)))
        self.assertEqual({row["epoch"] for row in measured}, {1})
        producer_pids = {row["producer"]["pid"] for row in measured}
        self.assertEqual(len(producer_pids), 2)
        self.assertNotIn(os.getpid(), producer_pids)
        self.assertGreater(result["items_per_s"], 0)
        self.assertGreater(result["measured"]["wall_s"], 0)
        for key in (
            "private_telemetry",
            "acquire_s",
            "copy_in_s",
            "copy_out_s",
            "restore_cpu_s",
            "registration_bytes",
            "registration_s",
            "pool_highwater_units",
            "ring_highwater_bytes",
            "seams",
            "no_staging_verdict",
        ):
            self.assertNotIn(key, result)
        for row in measured:
            self.assertFalse(row["producer"]["cuda_initialized"])
            self.assertEqual(row["producer"]["cpu_threads"], 1)
            self.assertIsNone(row["host_h2d_completion_s"])
            self.assertIsNone(row["helper_cpu_s"])
            for key in (
                "helper_pin_copy_cpu_s",
                "cuda_event_interval_ms",
                "pure_dma_ms",
            ):
                self.assertNotIn(key, row)
            self.assertGreaterEqual(row["producer"]["generation_cpu_s"], 0)
            self.assertGreaterEqual(row["next_output_s"], 0)
            self.assertGreater(row["payload_bytes"], 0)
        self.assertFalse(torch.cuda.is_initialized())

    @parameterized.expand([("pool",), ("ring",), ("ipc",)])
    def test_public_iterate_reiteration(self, storage: str) -> None:
        """Public CPU arrays survive three epochs through ordinary iterate boundaries."""
        config = _config(items=2, workers=1, kind="numpy")
        arena = None
        if storage == "pool":
            arena = SharedMemorySegmentPool(16384, 2, pin_memory=False)
        elif storage == "ring":
            arena = SharedMemoryRingBuffer(32768, pin_memory=False)
        source = iterate_in_subprocess(
            partial(benchmark.produce_payloads, config),
            initializer=benchmark.configure_process,
            mp_context="spawn",
            timeout=60.0,
            arena=arena,
        )
        try:
            for epoch in range(3):
                count = 0
                for item in source:
                    row = benchmark.process_item(item, config)
                    self.assertEqual(row["epoch"], epoch)
                    self.assertEqual(row["sequence"], epoch * 2 + count)
                    self.assertFalse(row["producer"]["cuda_initialized"])
                    count += 1
                    del item
                self.assertEqual(count, 2)
        finally:
            del source
        self.assertFalse(torch.cuda.is_initialized())

    def test_public_h2d_helper_measured_phases(self) -> None:
        """Only public H2D runs and only measured transfer fields are emitted."""
        batch = benchmark.make_payload(64, "torch", "fp32", 0)
        output = object()
        with patch.object(
            benchmark, "transfer_tensor_h2d", return_value=output
        ) as transfer:
            actual, phases = benchmark.transfer_payload(batch)
        self.assertIs(actual, output)
        transfer.assert_called_once_with(batch, num_caches=0)
        self.assertGreaterEqual(phases["host_h2d_completion_s"], 0)
        self.assertEqual(set(phases), {"host_h2d_completion_s", "helper_cpu_s"})
        for key in ("helper_pin_copy_cpu_s", "cuda_event_interval_ms", "pure_dma_ms"):
            self.assertNotIn(key, phases)
        self.assertFalse(torch.cuda.is_initialized())

    def test_isolated_consumer_cpu_report(self) -> None:
        """A fresh consumer returns a JSON receipt from a public pageable ring boundary."""
        result = benchmark.run_isolated(
            "ring-pageable", _config(workers=1, items=2, kind="numpy")
        )
        self.assertEqual(len(result["measured"]["items"]), 2)
        self.assertEqual(result["source"]["schema"], "public-iterate-v1")
        self.assertFalse(result["source"]["cuda_initialized"])
        self.assertFalse(torch.cuda.is_initialized())

    @parameterized.expand(
        [
            (["--arms", "ring-original"], "invalid choice"),
            (["--arms", "ipc"], "invalid choice"),
            (["--arms", "torch-pinned"], "invalid choice"),
            (["--runtime-checks"], "retired internal benchmark flags"),
            (["--trace-dir=traces"], "retired internal benchmark flags"),
            (["--threshold", "4096"], "retired internal benchmark flags"),
            (["--seam"], "retired internal benchmark flags"),
            (["--retain-items", "1"], "retired internal benchmark flags"),
            (["--consumer-delay", "0.1"], "retired internal benchmark flags"),
            (["--sample-interval", "0.1"], "retired internal benchmark flags"),
            (["--device", "cpu", "--arms", "pool-pinned"], "require --device cuda"),
            (["--workers", "0"], "must be positive"),
            (["--warmup-epochs", "0"], "must be positive"),
            (["--sizes-bytes", "0"], "sizes must be positive"),
            (["--timeout", "nan"], "finite/positive"),
        ]
    )
    def test_cli_rejects_retired_and_invalid_options(
        self, argv: list[str], message: str
    ) -> None:
        """Retired protocols and invalid CPU/config choices fail clearly before work starts."""
        error = io.StringIO()
        with redirect_stderr(error), self.assertRaises(SystemExit) as failure:
            benchmark.main(argv)
        self.assertEqual(failure.exception.code, 2)
        self.assertIn(message, error.getvalue())
        self.assertFalse(torch.cuda.is_initialized())

    def test_programmatic_configuration_rejected(self) -> None:
        """Public run rejects retired internal config fields without constructing a boundary."""
        with self.assertRaisesRegex(ValueError, "retired internal configuration"):
            benchmark.run_benchmark("pool-pageable", _config(threshold=4096))
        with self.assertRaisesRegex(ValueError, "unsupported or retired runtime arm"):
            benchmark.run_benchmark("ring-original", _config())
        with self.assertRaisesRegex(ValueError, "require a CUDA consumer"):
            benchmark.run_benchmark("pool-pinned", _config())
        self.assertFalse(torch.cuda.is_initialized())

    def test_cli_default_four_arms_and_rotation(self) -> None:
        """The normal CLI defaults to four matched arms and rotates them across repetitions."""
        calls = []

        def run(arm: str, config: dict[str, Any]) -> dict[str, Any]:
            calls.append((arm, config))
            return {"arm": arm, "config": config, "items_per_s": 1.0}

        output = io.StringIO()
        with (
            patch.object(benchmark, "run_isolated", side_effect=run),
            redirect_stdout(output),
        ):
            benchmark.main(["--repeats", "2", "--items", "2"])
        report = json.loads(output.getvalue())
        self.assertEqual(
            [call[0] for call in calls],
            [
                "pool-pageable",
                "pool-pinned",
                "ring-pageable",
                "ring-pinned",
                "pool-pinned",
                "ring-pageable",
                "ring-pinned",
                "pool-pageable",
            ],
        )
        self.assertEqual(len(report["summary"]), 4)
        self.assertEqual(report["summary"][0]["fresh_process_repeats"], 2)
        self.assertIn("historical", report["legend"])
        self.assertFalse(torch.cuda.is_initialized())

    def test_cli_cpu_output_and_single_repeat_sd(self) -> None:
        """A real bounded CPU CLI writes a report with explicit null single-repeat SD."""
        with TemporaryDirectory() as directory:
            path = str(Path(directory) / "cpu.json")
            benchmark.main(
                [
                    "--device",
                    "cpu",
                    "--arms",
                    "pool-pageable",
                    "--sizes-bytes",
                    "8192",
                    "--items",
                    "2",
                    "--repeats",
                    "1",
                    "--output",
                    path,
                ]
            )
            report = json.loads(Path(path).read_text())
        self.assertEqual(len(report["results"]), 1)
        self.assertIsNone(report["summary"][0]["items_per_s_stdev"])
        self.assertEqual(report["results"][0]["config"]["device"], "cpu")
        self.assertFalse(torch.cuda.is_initialized())

    def test_source_receipt_benchmark_only(self) -> None:
        """Source mode hashes the benchmark, without CUDA or library/package probes."""
        output = io.StringIO()
        with redirect_stdout(output):
            benchmark.main(["--source-receipt"])
        report = json.loads(output.getvalue())["source_receipt"]
        self.assertEqual(report["hash_scope"], "benchmark_source_utf8")
        self.assertEqual(len(report["benchmark_sha256"]), 64)
        self.assertFalse(report["cuda_initialized"])
        self.assertNotIn("library", report)

    def test_public_members_and_no_private_spdl_access(self) -> None:
        """Portable exports are public definitions, with no private SPDL access."""
        source = inspect.getsource(benchmark)
        tree = ast.parse(source)
        public_defs = {
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))
            and not node.name.startswith("_")
        }
        self.assertEqual(set(benchmark.__all__), public_defs)
        spdl_imports = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.startswith("spdl")
        }
        self.assertEqual(spdl_imports, {"spdl.pipeline", "spdl.io"})
        self.assertFalse(
            any(
                isinstance(node, ast.Attribute)
                and node.attr.startswith("_")
                and node.attr not in ("__version__",)
                for node in ast.walk(tree)
            )
        )
        self.assertNotIn("importlib", source)
        self.assertTrue(
            all(inspect.getdoc(getattr(benchmark, name)) for name in benchmark.__all__)
        )
