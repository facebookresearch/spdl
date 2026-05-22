# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from spdl.autoresearch._common._state import (
    _append_master_row,
    MASTER_TABLE_HEADERS,
    write_state,
)
from spdl.autoresearch._common._visualization import _load_tsv
from spdl.autoresearch.core import (
    AnalysisResult,
    FailureKind,
    FailurePhase,
    HypothesisNode,
    TaskSpec,
)
from spdl.autoresearch.pipeline_optimization._ops import (
    _WorkflowStateStore,
    PipelineOptimizationWorkflow,
)
from spdl.autoresearch.pipeline_optimization._ops._analysis_ops import (
    _update_on_complete,
)
from spdl.autoresearch.pipeline_optimization._ops._failures import (
    _classify_terminal_job_failure,
    _FAILURE_POLICIES,
    _make_failure,
)
from spdl.autoresearch.pipeline_optimization._ops._policy import (
    _build_change_set,
    _change_summary_for_spec,
    _compare_metric_value,
    _extract_default_executor_concurrency,
    _extract_param_changes,
    _extract_total_threads,
    _is_duplicate_spec,
    _node_from_spec,
    _retry_policy_for_failure,
    _select_planning_node,
    _spec_from_node,
    _startup_retry_spec,
    _validate_thread_budget,
)
from spdl.autoresearch.pipeline_optimization._ops._source_ops import _build_apply_prompt
from spdl.autoresearch.pipeline_optimization._ops._store import _write_text_atomic
from spdl.autoresearch.pipeline_optimization._platform import (
    _MetricsEvidence,
    create_platform,
)
from spdl.autoresearch.pipeline_optimization._platform._agents import _MockAgent
from spdl.autoresearch.pipeline_optimization._platform._local import _summarize_error
from spdl.tools.autoresearch.utils.commands.report import _read_failures
from spdl.tools.autoresearch.utils.commands.status import _failure_summary

__all__: list[str] = []


def _config() -> dict:
    return {
        "pipeline_script": "",
        "source_dir": "",
        "scm": "",
        "build_command": "",
        "base_launch_command": "torchx run example --num-fetch-threads 8",
        "stopping_criteria": {
            "max_iterations": 20,
            "patience": 5,
        },
        "max_concurrency": 4,
        "job_timeout_s": 600,
        "poll_interval": 0,
    }


def _state() -> dict:
    return {
        "iteration": 0,
        "status": "looping",
        "baseline_job": None,
        "current_best": None,
        "best_metric": None,
        "plateau_count": 0,
        "best_practices_tried": [],
        "anchor_commit": "",
        "history": [],
    }


class _AutoresearchWorkflowTest(unittest.TestCase):
    def test_fresh_load_creates_initial_must_run_specs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            adapter = self._adapter(Path(tmp))

            specs = adapter.load()

            self.assertEqual(
                ["000_baseline", "000_headspace", "001_mtp"],
                [spec.id for spec in specs],
            )
            self.assertEqual([-1000, -999, -998], [spec.priority for spec in specs])

    def test_checkpoint_writes_compatibility_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            adapter = self._adapter(workdir)
            specs = adapter.load()

            adapter.checkpoint(queued=specs[1:], running=specs[:1], status="running")

            engine_state = json.loads(
                (workdir / "engine" / "engine_state.json").read_text()
            )
            queue = json.loads((workdir / "engine" / "queue.json").read_text())
            active = json.loads((workdir / "engine" / "active.json").read_text())
            baseline_status = (
                workdir / "engine" / "nodes" / "000_baseline" / "status.txt"
            ).read_text()

            self.assertEqual("running", engine_state["status"])
            self.assertEqual(2, engine_state["queued"])
            self.assertEqual(1, engine_state["running"])
            self.assertEqual(
                ["000_headspace", "001_mtp"], [q["node_id"] for q in queue]
            )
            self.assertEqual([], active)
            self.assertEqual("queued\n", baseline_status)

    def test_checkpoint_round_trips_specs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            adapter = self._adapter(workdir)
            specs = adapter.load()
            adapter.checkpoint(queued=specs[1:], running=specs[:1], status="running")

            loaded = self._adapter(workdir).load()

            self.assertEqual(
                ["000_headspace", "001_mtp", "000_baseline"],
                [spec.id for spec in loaded],
            )

    def test_planning_is_blocked_until_must_run_experiments_finish(self) -> None:
        baseline = HypothesisNode(
            node_id="000_baseline",
            name="baseline",
            status="completed",
        )
        headspace = HypothesisNode(
            node_id="000_headspace",
            name="headspace_cache",
            status="queued",
        )
        mtp = HypothesisNode(
            node_id="001_mtp",
            name="mtp",
            status="completed",
        )

        selected = _select_planning_node(
            baseline,
            {
                baseline.node_id: baseline,
                headspace.node_id: headspace,
                mtp.node_id: mtp,
            },
        )

        self.assertIsNone(selected)

    def _load_node(self, spec: TaskSpec) -> HypothesisNode:
        """Extract node from a TaskSpec with a safe dict cast for Pyre."""
        node_data = spec.payload["node"]
        assert isinstance(node_data, dict)
        return HypothesisNode.from_dict(node_data)

    def test_child_spec_parents_under_baseline_when_goto_is_null(self) -> None:
        """Experiments that start from anchor (goto=null) should be parented
        under the baseline node, not whichever node triggered planning."""
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            adapter = self._adapter(workdir)
            specs = adapter.load()
            # Simulate baseline completed with a commit.
            baseline_node = self._load_node(specs[0])
            baseline_node.status = "completed"
            baseline_node.commit = "baseline_commit_abc"
            adapter._store.upsert_node(baseline_node)

            # Simulate MTP completed with a different commit.
            mtp_node = self._load_node(specs[2])
            mtp_node.status = "completed"
            mtp_node.commit = "mtp_commit_xyz"
            adapter._store.upsert_node(mtp_node)

            # Create a child with goto=None (should parent under baseline).
            child_spec = adapter._create_child_spec(
                mtp_node,
                {"name": "nvdec_decode", "goto": None},
            )
            child_node = self._load_node(child_spec)

            self.assertEqual("000_baseline", child_node.parent_id)
            self.assertEqual("baseline_commit_abc", child_node.commit)

    def test_child_spec_parents_under_goto_commit_owner(self) -> None:
        """Experiments with a goto commit should be parented under the node
        that produced that commit."""
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            adapter = self._adapter(workdir)
            specs = adapter.load()

            baseline_node = self._load_node(specs[0])
            baseline_node.status = "completed"
            baseline_node.commit = "baseline_commit_abc"
            adapter._store.upsert_node(baseline_node)

            mtp_node = self._load_node(specs[2])
            mtp_node.status = "completed"
            mtp_node.commit = "mtp_commit_xyz"
            adapter._store.upsert_node(mtp_node)

            # Create a child with goto pointing to MTP's commit.
            child_spec = adapter._create_child_spec(
                baseline_node,  # default parent is baseline
                {"name": "batch_on_mtp", "goto": "mtp_commit_xyz"},
            )
            child_node = self._load_node(child_spec)

            self.assertEqual(mtp_node.node_id, child_node.parent_id)
            self.assertEqual("mtp_commit_xyz", child_node.commit)

    def test_child_spec_parents_under_baseline_without_goto(self) -> None:
        """When goto is absent, the spec defaults to anchor (baseline)."""
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            adapter = self._adapter(workdir)
            specs = adapter.load()

            mtp_node = self._load_node(specs[2])
            mtp_node.status = "completed"
            mtp_node.commit = "mtp_commit_xyz"
            adapter._store.upsert_node(mtp_node)

            # Ensure baseline exists so _resolve_parent can find it.
            baseline_node = self._load_node(specs[0])
            baseline_node.status = "completed"
            adapter._store.upsert_node(baseline_node)

            # No goto key at all — .get("goto") returns None, which means
            # "start from anchor".  Parent should be baseline.
            child_spec = adapter._create_child_spec(
                mtp_node,
                {"name": "some_exp"},
            )
            child_node = self._load_node(child_spec)
            self.assertEqual("000_baseline", child_node.parent_id)

    def test_headspace_completion_selects_mtp_after_must_run_finish(self) -> None:
        baseline = HypothesisNode(
            node_id="000_baseline",
            name="baseline",
            status="completed",
        )
        headspace = HypothesisNode(
            node_id="000_headspace",
            name="headspace_cache",
            status="completed",
            spec={"_is_headspace": True},
        )
        mtp = HypothesisNode(
            node_id="001_mtp",
            name="mtp",
            status="completed",
        )

        selected = _select_planning_node(
            headspace,
            {
                baseline.node_id: baseline,
                headspace.node_id: headspace,
                mtp.node_id: mtp,
            },
        )

        self.assertEqual(mtp, selected)

    def test_policy_helpers_cover_metric_and_thread_decisions(self) -> None:
        self.assertEqual(
            ("step_ms", 12.5),
            _compare_metric_value({"steady_step_time_ms": 12.5, "duration_s": 100}),
        )
        # Only --num-threads → use it directly.
        self.assertEqual(
            12,
            _extract_total_threads("--num-threads 12"),
        )
        # Stage concurrency flags are not additive thread budgets.
        self.assertIsNone(
            _extract_total_threads("--num-fetch-threads 8 --num-decode-threads 16"),
        )
        self.assertEqual(
            16,
            _extract_default_executor_concurrency(
                "--num-fetch-threads 8 --num-decode-threads 16"
            ),
        )
        self.assertEqual(
            8,
            _extract_default_executor_concurrency("--num-fetch-threads 8"),
        )
        self.assertEqual(
            16,
            _extract_default_executor_concurrency("--num-decode-threads 16"),
        )
        # No num_threads flag at all → None (unknown budget).
        self.assertIsNone(
            _extract_total_threads("--num_workers 8 --num_epochs 3"),
        )
        self.assertIsNone(
            _extract_total_threads(""),
        )
        self.assertEqual(
            ["ok", "fits_concurrency"],
            [
                spec["name"]
                for spec in _validate_thread_budget(
                    [
                        {"name": "ok", "launch_command": "--num-threads 8"},
                        {"name": "too_many", "launch_command": "--num-threads 64"},
                        {
                            "name": "too_small",
                            "launch_command": (
                                "--num-threads 8 --num-decode-threads 16"
                            ),
                        },
                        {
                            "name": "fits_concurrency",
                            "launch_command": (
                                "--num-threads 16 --num-decode-threads 16"
                            ),
                        },
                    ],
                    16,
                )
            ],
        )
        # Commands with no thread flags pass validation (unknown budget).
        self.assertEqual(
            ["pass_through"],
            [
                spec["name"]
                for spec in _validate_thread_budget(
                    [
                        {"name": "pass_through", "launch_command": "--num_workers 8"},
                    ],
                    16,
                )
            ],
        )

    # -- _extract_param_changes / _build_change_set / _is_duplicate_spec ------

    def test_extract_param_changes_detects_flag_diffs(self) -> None:
        base = "torchx run app --image $IMAGE --num_workers 8 --num_epochs 3"

        # New flag added.
        self.assertEqual(
            ["batch_size=48"],
            _extract_param_changes(base + " --batch_size 48", base),
        )

        # Flag value changed.
        self.assertEqual(
            ["num_workers=16"],
            _extract_param_changes(
                "torchx run app --image $IMAGE --num_workers 16 --num_epochs 3",
                base,
            ),
        )

        # No diff when identical.
        self.assertEqual([], _extract_param_changes(base, base))

        # Empty commands return nothing.
        self.assertEqual([], _extract_param_changes("", base))
        self.assertEqual([], _extract_param_changes(base, ""))

    def test_extract_param_changes_normalizes_dashes(self) -> None:
        base = "torchx run app --num-fetch-threads 8"
        exp = "torchx run app --num-fetch-threads 16"
        changes = _extract_param_changes(exp, base)
        self.assertEqual(["num_fetch_threads=16"], changes)

    def test_extract_param_changes_handles_negative_values(self) -> None:
        base = "torchx run app --max_steps -1"
        exp = "torchx run app --max_steps -2"
        changes = _extract_param_changes(exp, base)
        self.assertEqual(["max_steps=-2"], changes)

    def test_build_change_set_merges_explicit_and_param_changes(self) -> None:
        base = "torchx run --image $IMAGE --num_workers 8"
        spec = {
            "changes": ["torch_compile"],
            "launch_command": base + " --batch_size 48",
        }
        result = _build_change_set(spec, base)
        self.assertEqual(frozenset({"torch_compile", "batch_size=48"}), result)

    def test_build_change_set_empty_for_baseline(self) -> None:
        base = "torchx run --image $IMAGE --num_workers 8"
        spec = {"changes": [], "launch_command": base}
        self.assertEqual(frozenset(), _build_change_set(spec, base))

    def test_build_change_set_normalizes_case(self) -> None:
        spec = {"changes": ["Torch_Compile", " FUSED_ADAMW "]}
        result = _build_change_set(spec, "")
        self.assertEqual(frozenset({"torch_compile", "fused_adamw"}), result)

    def test_duplicate_requires_matching_change_sets(self) -> None:
        base = "torchx run --image $IMAGE --num_workers 8"
        baseline = HypothesisNode(
            node_id="000_baseline",
            name="baseline",
            status="completed",
            spec={"changes": [], "launch_command": base},
        )
        mtp = HypothesisNode(
            node_id="001_mtp",
            name="mtp",
            status="completed",
            spec={"changes": ["mtp"], "launch_command": base},
        )
        nodes = [baseline, mtp]

        # torch_compile: different code changes, same launch → NOT duplicate.
        self.assertFalse(
            _is_duplicate_spec(
                {"changes": ["torch_compile"], "launch_command": base},
                nodes,
                base,
            )
        )

        # fused_adamw: different code changes, same launch → NOT duplicate.
        self.assertFalse(
            _is_duplicate_spec(
                {"changes": ["fused_adamw"], "launch_command": base},
                nodes,
                base,
            )
        )

        # Exact same change set as baseline → IS duplicate.
        self.assertTrue(
            _is_duplicate_spec(
                {"changes": [], "launch_command": base},
                nodes,
                base,
            )
        )

        # Exact same change set as MTP → IS duplicate.
        self.assertTrue(
            _is_duplicate_spec(
                {"changes": ["mtp"], "launch_command": base},
                nodes,
                base,
            )
        )

    def test_duplicate_distinguishes_param_only_experiments(self) -> None:
        base = "torchx run --image $IMAGE --num_workers 8"
        batch_48 = HypothesisNode(
            node_id="002_batch48",
            name="batch_size_48",
            status="completed",
            spec={
                "changes": [],
                "launch_command": base + " --batch_size 48",
            },
        )
        nodes = [batch_48]

        # batch_size=64 has different param → NOT duplicate.
        self.assertFalse(
            _is_duplicate_spec(
                {"changes": [], "launch_command": base + " --batch_size 64"},
                nodes,
                base,
            )
        )

        # batch_size=48 again → IS duplicate.
        self.assertTrue(
            _is_duplicate_spec(
                {"changes": [], "launch_command": base + " --batch_size 48"},
                nodes,
                base,
            )
        )

    def test_duplicate_skips_failed_nodes(self) -> None:
        base = "torchx run --image $IMAGE"
        failed = HypothesisNode(
            node_id="003_oom",
            name="batch_64",
            status="failed",
            spec={
                "changes": [],
                "launch_command": base + " --batch_size 64",
            },
        )
        # Exact match of a failed node → NOT duplicate (allow retry).
        self.assertFalse(
            _is_duplicate_spec(
                {"changes": [], "launch_command": base + " --batch_size 64"},
                [failed],
                base,
            )
        )

    def test_duplicate_combination_vs_individual(self) -> None:
        base = "torchx run --image $IMAGE"
        mtp_only = HypothesisNode(
            node_id="001_mtp",
            name="mtp",
            status="completed",
            spec={"changes": ["mtp"], "launch_command": base},
        )
        # Combination is not a dup of individual.
        self.assertFalse(
            _is_duplicate_spec(
                {
                    "changes": ["mtp", "torch_compile"],
                    "launch_command": base,
                },
                [mtp_only],
                base,
            )
        )

    def test_duplicate_backward_compat_no_changes_field(self) -> None:
        base = "torchx run --image $IMAGE --num_workers 8"
        # Old spec without changes field — change set is derived purely from
        # launch command diffs.
        old_node = HypothesisNode(
            node_id="002_batch48",
            name="batch_size_48",
            status="completed",
            spec={"launch_command": base + " --batch_size 48"},
        )

        # Same param diff → duplicate.
        self.assertTrue(
            _is_duplicate_spec(
                {"launch_command": base + " --batch_size 48"},
                [old_node],
                base,
            )
        )

        # Different param → not duplicate.
        self.assertFalse(
            _is_duplicate_spec(
                {"launch_command": base + " --batch_size 64"},
                [old_node],
                base,
            )
        )

    def test_startup_retry_inherits_changes(self) -> None:
        node = HypothesisNode(
            node_id="001_mtp",
            name="mtp",
            status="failed",
            spec={
                "name": "mtp",
                "changes": ["mtp"],
                "description": "try MTP",
                "best_practices_tags": ["mtp"],
            },
            failure=_make_failure(
                FailureKind.JOB_STARTUP_FAILED,
                FailurePhase.JOB,
                "Tokenizer cannot pickle",
            ),
        )
        retry = _startup_retry_spec(node, _config())
        self.assertIsNotNone(retry)
        assert retry is not None
        self.assertEqual(["mtp"], retry["changes"])

    def test_initial_nodes_have_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            adapter = self._adapter(Path(tmp))
            specs = adapter.load()
            nodes = []
            for spec in specs:
                node_data = spec.payload["node"]
                assert isinstance(node_data, dict)
                nodes.append(HypothesisNode.from_dict(node_data))

            by_name = {node.name: node for node in nodes}
            self.assertEqual([], by_name["baseline"].spec["changes"])
            self.assertEqual(
                ["cache_dataloader"], by_name["headspace_cache"].spec["changes"]
            )
            self.assertEqual(["mtp"], by_name["mtp"].spec["changes"])

    def test_store_update_spec_refreshes_checkpoint_and_active_view(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            self._write_base_files(workdir)
            store = _WorkflowStateStore(workdir, _state())
            node = HypothesisNode(
                node_id="000_baseline",
                name="baseline",
                status="queued",
            )
            spec = _spec_from_node(node)
            store.save_scheduler_state(queued=[], running=[spec], status="running")

            node.status = "running"
            node.job_id = "remote_job"
            store.update_spec(spec, node)

            checkpoint = json.loads(
                (workdir / "engine" / "checkpoint.json").read_text()
            )
            active = json.loads((workdir / "engine" / "active.json").read_text())
            running_node = _node_from_spec(TaskSpec.from_dict(checkpoint["running"][0]))

            self.assertEqual("remote_job", running_node.job_id)
            self.assertEqual("000_baseline", active[0]["node_id"])
            self.assertEqual("remote_job", active[0]["job_id"])
            self.assertIn("launched_at_iso", active[0])

    def test_queue_view_includes_retry_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            self._write_base_files(workdir)
            store = _WorkflowStateStore(workdir, _state())
            node = HypothesisNode(
                node_id="002_retry",
                name="retry",
                status="queued",
                spec={
                    "_startup_retry_of": "001_mtp",
                    "_startup_retry_attempt": 1,
                },
            )

            store.save_scheduler_state(
                queued=[_spec_from_node(node)],
                running=[],
                status="running",
            )

            queue = json.loads((workdir / "engine" / "queue.json").read_text())
            self.assertEqual("001_mtp", queue[0]["retry_of"])
            self.assertEqual(1, queue[0]["retry_attempt"])

    def test_store_rejects_malformed_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            self._write_base_files(workdir)
            engine_dir = workdir / "engine"
            engine_dir.mkdir()
            (engine_dir / "checkpoint.json").write_text(
                json.dumps({"queued": [{"id": "bad", "payload": {}}]}) + "\n"
            )
            store = _WorkflowStateStore(workdir, _state())

            with self.assertRaisesRegex(ValueError, "payload.node"):
                store.load_checkpoint()

    def test_store_persists_failure_view(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            self._write_base_files(workdir)
            store = _WorkflowStateStore(workdir, _state())
            node = HypothesisNode(
                node_id="001_failed",
                name="failed",
                status="failed",
                failure=_make_failure(
                    FailureKind.BUILD_FAILED,
                    FailurePhase.BUILD,
                    "Build failed",
                ),
            )

            store.upsert_node(node)
            store.write_all()

            failure = json.loads(
                (
                    workdir / "engine" / "nodes" / "001_failed" / "failure.json"
                ).read_text()
            )
            engine_state = json.loads(
                (workdir / "engine" / "engine_state.json").read_text()
            )
            self.assertEqual("build_failed", failure["kind"])
            self.assertEqual({"build_failed": 1}, engine_state["failed_by_kind"])
            self.assertIn("build_failed", _failure_summary(workdir))
            self.assertIn("build_failed", _read_failures(workdir))

    def test_adapter_records_structured_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            adapter = self._adapter(workdir)
            node = HypothesisNode(node_id="001_failed", name="failed")
            spec = _spec_from_node(node)
            failure = _make_failure(
                FailureKind.LAUNCH_FAILED,
                FailurePhase.LAUNCH,
                "No launch command configured",
            )

            asyncio.run(adapter._record_failure(spec, node, failure))

            stored = json.loads(
                (
                    workdir / "engine" / "nodes" / "001_failed" / "failure.json"
                ).read_text()
            )
            master_table = (workdir / "master_table.tsv").read_text()
            self.assertEqual("launch_failed", stored["kind"])
            self.assertIn("launch_failed: No launch command configured", master_table)

    def test_completed_failure_history_uses_structured_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            self._write_base_files(workdir)
            state = _state()
            node = HypothesisNode(
                node_id="001_failed",
                name="failed",
                status="failed",
                spec={"name": "failed"},
            )
            result = AnalysisResult(
                structured={"metrics": {}, "findings": []},
                failure=_make_failure(
                    FailureKind.JOB_FAILED,
                    FailurePhase.JOB,
                    "Job failed",
                ),
            )

            _update_on_complete(workdir, _config(), state, node, result)

            entry = state["history"][0]
            self.assertNotIn("failure", entry)
            self.assertEqual("job_failed", entry["structured"]["failure"]["kind"])

    def test_job_failure_classifier_splits_startup_runtime_and_unknown(self) -> None:
        startup = _classify_terminal_job_failure(
            _MetricsEvidence(
                system_metrics="",
                pipeline_stats_log="TypeError: cannot pickle local function for MTP",
                metrics_summary="",
            ),
            job_id="job_startup",
            progress_seen=False,
        )
        runtime = _classify_terminal_job_failure(
            _MetricsEvidence(
                system_metrics="",
                pipeline_stats_log="[autoresearch] step=12\nCUDA out of memory",
                metrics_summary="steady_step_time_ms: 12",
            ),
            job_id="job_runtime",
            progress_seen=True,
        )
        unknown = _classify_terminal_job_failure(
            _MetricsEvidence(
                system_metrics="", pipeline_stats_log="", metrics_summary=""
            ),
            job_id="job_unknown",
            progress_seen=False,
        )

        self.assertEqual(FailureKind.JOB_STARTUP_FAILED, startup.kind)
        self.assertEqual(FailureKind.JOB_RUNTIME_FAILED, runtime.kind)
        self.assertEqual(FailureKind.JOB_FAILED, unknown.kind)

    def test_structured_metrics_evidence_guides_failure_classification(self) -> None:
        startup = _classify_terminal_job_failure(
            _MetricsEvidence(
                system_metrics="",
                pipeline_stats_log="",
                metrics_summary="",
                error_summary="TypeError: can't pickle tokenizer",
                log_paths=["stderr.log"],
            ),
            job_id="job_startup",
            progress_seen=False,
        )
        runtime = _classify_terminal_job_failure(
            _MetricsEvidence(
                system_metrics="",
                pipeline_stats_log="",
                metrics_summary="",
                progress_seen=True,
                exit_code=1,
            ),
            job_id="job_runtime",
            progress_seen=False,
        )

        self.assertEqual(FailureKind.JOB_STARTUP_FAILED, startup.kind)
        self.assertEqual(["stderr.log"], startup.details["log_paths"])
        self.assertEqual(FailureKind.JOB_RUNTIME_FAILED, runtime.kind)
        self.assertEqual(1, runtime.details["exit_code"])

    def test_startup_failure_retry_is_bounded_for_mtp(self) -> None:
        node = HypothesisNode(
            node_id="001_mtp",
            name="mtp",
            status="failed",
            spec={
                "name": "mtp",
                "description": "try MTP",
                "best_practices_tags": ["mtp"],
            },
            failure=_make_failure(
                FailureKind.JOB_STARTUP_FAILED,
                FailurePhase.JOB,
                "Tokenizer cannot pickle",
            ),
        )

        retry = _startup_retry_spec(node, _config())
        self.assertIsNotNone(retry)
        assert retry is not None
        self.assertEqual("mtp_startup_retry_1", retry["name"])
        self.assertEqual(1, retry["_startup_retry_attempt"])
        self.assertIn("pickling", retry["hypothesis"])

        node.spec["_startup_retry_attempt"] = 2
        self.assertIsNone(_startup_retry_spec(node, _config()))

    def test_retry_policy_is_kind_specific(self) -> None:
        node = HypothesisNode(
            node_id="001_mtp",
            name="mtp",
            spec={"best_practices_tags": ["mtp"]},
            failure=_make_failure(
                FailureKind.JOB_STARTUP_FAILED,
                FailurePhase.JOB,
                "startup",
            ),
        )
        policy = _retry_policy_for_failure(node, _config())
        self.assertIsNotNone(policy)
        assert policy is not None
        self.assertEqual(2, policy["max_attempts"])

        node.failure = _make_failure(
            FailureKind.BUILD_FAILED,
            FailurePhase.BUILD,
            "build",
        )
        self.assertIsNone(_retry_policy_for_failure(node, _config()))

    def test_planning_prefers_non_startup_failed_retry(self) -> None:
        baseline = HypothesisNode(
            node_id="000_baseline",
            name="baseline",
            status="completed",
        )
        headspace = HypothesisNode(
            node_id="000_headspace",
            name="headspace_cache",
            status="completed",
            spec={"_is_headspace": True},
        )
        mtp = HypothesisNode(
            node_id="001_mtp",
            name="mtp",
            status="failed",
            failure=_make_failure(
                FailureKind.JOB_STARTUP_FAILED,
                FailurePhase.JOB,
                "startup",
            ),
        )
        retry = HypothesisNode(
            node_id="002_mtp_startup_retry_1",
            name="mtp_startup_retry_1",
            status="failed",
            spec={"_startup_retry_of": "001_mtp"},
            failure=_make_failure(
                FailureKind.JOB_STARTUP_FAILED,
                FailurePhase.JOB,
                "startup",
            ),
        )
        better_candidate = HypothesisNode(
            node_id="003_threads",
            name="threads",
            status="completed",
        )

        selected = _select_planning_node(
            retry,
            {
                node.node_id: node
                for node in [baseline, headspace, mtp, retry, better_candidate]
            },
        )

        self.assertEqual(better_candidate, selected)

    def test_startup_retry_uses_repair_prompt(self) -> None:
        platform = create_platform({"platform": "local", "agent": "mock"})
        assert isinstance(platform.agent, _MockAgent)
        platform.agent.responses["prompt:apply_startup_repair"] = (
            "failed during job startup __STARTUP_FAILURE_JSON__"
        )
        prompt = _build_apply_prompt(
            platform,
            {
                "name": "mtp_startup_retry_1",
                "description": "repair",
                "hypothesis": "fix startup",
                "_startup_retry_attempt": 1,
                "_startup_failure": {"kind": "job_startup_failed"},
            },
            "002_mtp_startup_retry_1",
            "knowledge",
            "/tmp/pipeline.py",
            "def main():\n    pass\n",
        )

        self.assertIn("failed during job startup", prompt)
        self.assertIn("job_startup_failed", prompt)

    def test_headspace_node_uses_dedicated_prompt_with_knowledge(self) -> None:
        platform = create_platform({"platform": "local", "agent": "mock"})
        assert isinstance(platform.agent, _MockAgent)
        platform.agent.responses["prompt:headspace"] = (
            "headspace prompt __KNOWLEDGE__ __PIPELINE_CODE__"
        )

        prompt = _build_apply_prompt(
            platform,
            {
                "name": "headspace_cache",
                "description": "Wrap with CacheDataLoader for headspace analysis",
                "_is_headspace": True,
            },
            "000_headspace",
            "knowledge",
            "/tmp/pipeline.py",
            "def main():\n    pass\n",
        )

        self.assertIn("headspace prompt", prompt)
        self.assertIn("knowledge", prompt)
        self.assertIn("def main", prompt)

    def test_change_summary_is_concise_and_persisted_in_master_table(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "master_table.tsv").write_text(
                "\t".join(MASTER_TABLE_HEADERS) + "\n"
            )

            summary = _change_summary_for_spec(
                {
                    "description": (
                        "Increase decode thread count while preserving the rest "
                        "of the pipeline"
                    )
                }
            )
            _append_master_row(
                workdir,
                {
                    "run_id": "001_threads",
                    "name": "threads",
                    "status": "completed",
                    "change_summary": summary,
                    "sm_util_pct": "50",
                },
            )

            rows = _load_tsv(workdir / "master_table.tsv")

            self.assertEqual("Increase decode thread count while", summary)
            self.assertEqual(summary, rows[0]["change_summary"])

    def test_every_failure_kind_has_policy(self) -> None:
        self.assertEqual(set(FailureKind.__members__.values()), set(_FAILURE_POLICIES))
        self.assertTrue(_FAILURE_POLICIES[FailureKind.JOB_STARTUP_FAILED].retryable)

    def test_error_summary_prefers_traceback_block(self) -> None:
        summary = _summarize_error(
            "before\n"
            "Traceback (most recent call last):\n"
            '  File "x.py", line 1, in <module>\n'
            "TypeError: cannot pickle tokenizer\n"
            "after\n"
        )

        self.assertIsNotNone(summary)
        assert summary is not None
        self.assertIn("Traceback", summary)
        self.assertIn("cannot pickle", summary)

    def test_atomic_write_replaces_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.json"

            _write_text_atomic(path, '{"a": 1}\n')
            _write_text_atomic(path, '{"a": 2}\n')

            self.assertEqual({"a": 2}, json.loads(path.read_text()))

    def _adapter(self, workdir: Path) -> PipelineOptimizationWorkflow:
        self._write_base_files(workdir)
        return PipelineOptimizationWorkflow(
            workdir=workdir,
            config=_config(),
            state=_state(),
            platform=create_platform({"platform": "auto", "agent": "mock"}, workdir),
        )

    def _write_base_files(self, workdir: Path) -> None:
        workdir.mkdir(parents=True, exist_ok=True)
        (workdir / "config.json").write_text(json.dumps(_config()) + "\n")
        write_state(workdir, _state())
        (workdir / "master_table.tsv").write_text(
            "\t".join(MASTER_TABLE_HEADERS) + "\n"
        )
