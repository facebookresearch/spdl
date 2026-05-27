# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import importlib
import sys
import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path
from typing import get_origin

from spdl.autoresearch._app._engine import _parse_engine_args
from spdl.autoresearch._app._spec import (
    _read_workflow_factory,
    _record_workflow_factory,
    _resolve_workflow,
)
from spdl.autoresearch._app._supervisor import (
    _build_engine_command,
    _parse_supervisor_args,
)

__all__: list[str] = []


def _identity_factory(argv: list[str], workdir: Path | None) -> object:
    """A trivial factory used as a resolution target by the tests below."""
    return (argv, workdir)


class _ResolveWorkflowTest(unittest.TestCase):
    def test_module_factory_form(self) -> None:
        """_resolve_workflow imports module.path:factory_name and returns the callable."""
        factory = _resolve_workflow(f"{__name__}:_identity_factory")
        self.assertIs(factory, _identity_factory)

    def test_empty_specifier_raises(self) -> None:
        """An empty string is rejected with ValueError, not silently importing."""
        with self.assertRaises(ValueError):
            _resolve_workflow("")

    def test_malformed_specifier_raises(self) -> None:
        """A specifier with a colon but missing one half is rejected."""
        for bad in (":factory", "module.path:", ":"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    _resolve_workflow(bad)

    def test_unknown_module_raises_import_error(self) -> None:
        """A non-existent module surfaces ModuleNotFoundError to the caller."""
        with self.assertRaises(ModuleNotFoundError):
            _resolve_workflow("definitely.not.a.real.module:create")

    def test_missing_attribute_raises(self) -> None:
        """An existing module with a missing attribute raises AttributeError."""
        with self.assertRaises(AttributeError):
            _resolve_workflow(f"{__name__}:does_not_exist")

    def test_short_name_lookup_misses_cleanly(self) -> None:
        """Short-name lookup raises LookupError when no entry point matches."""
        with self.assertRaises(LookupError):
            _resolve_workflow("not_registered_workflow_xyz")


class _WorkflowFactoryRecordTest(unittest.TestCase):
    def test_round_trip(self) -> None:
        """_record_workflow_factory followed by _read_workflow_factory returns the spec."""
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            _record_workflow_factory(workdir, "pkg.mod:factory")
            self.assertEqual(_read_workflow_factory(workdir), "pkg.mod:factory")

    def test_read_returns_none_when_missing(self) -> None:
        """_read_workflow_factory on a fresh workdir returns None instead of raising."""
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(_read_workflow_factory(Path(tmp)))

    def test_read_rejects_malformed_record(self) -> None:
        """Reading a malformed record file raises ValueError."""
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "workflow_factory.json").write_text("[]\n")
            with self.assertRaises(ValueError):
                _read_workflow_factory(workdir)

    def test_record_creates_workdir(self) -> None:
        """_record_workflow_factory creates the workdir if it does not yet exist."""
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp) / "nested"
            _record_workflow_factory(workdir, "pkg.mod:factory")
            self.assertEqual(_read_workflow_factory(workdir), "pkg.mod:factory")


class _ArgvSplitTest(unittest.TestCase):
    def test_supervisor_splits_at_double_dash(self) -> None:
        """Tokens after '--' are forwarded as the workflow tail, not parsed by the framework."""
        ns, tail = _parse_supervisor_args(
            [
                "/tmp/workdir",
                "--workflow",
                "pkg.mod:factory",
                "--max-concurrency",
                "5",
                "--",
                "--pipeline-script",
                "x.py",
            ]
        )
        self.assertEqual(ns.workdir, "/tmp/workdir")
        self.assertEqual(ns.workflow, "pkg.mod:factory")
        self.assertEqual(ns.max_concurrency, 5)
        self.assertEqual(tail, ["--pipeline-script", "x.py"])

    def test_supervisor_workdir_optional(self) -> None:
        """The supervisor accepts no workdir during initial config gathering."""
        ns, tail = _parse_supervisor_args(["--workflow", "pkg.mod:factory"])
        self.assertIsNone(ns.workdir)
        self.assertEqual(tail, [])

    def test_engine_requires_workflow_and_workdir(self) -> None:
        """The engine refuses to start without --workflow and --workdir."""
        with self.assertRaises(SystemExit):
            _parse_engine_args(["--workdir", "/tmp/wd"])
        with self.assertRaises(SystemExit):
            _parse_engine_args(["--workflow", "pkg.mod:factory"])

    def test_engine_passes_tail_through(self) -> None:
        """The engine surfaces the workflow tail unchanged to the caller."""
        ns, tail = _parse_engine_args(
            [
                "--workflow",
                "pkg.mod:factory",
                "--workdir",
                "/tmp/wd",
                "--",
                "--build-command",
                "make",
            ]
        )
        self.assertEqual(ns.workflow, "pkg.mod:factory")
        self.assertEqual(ns.workdir, "/tmp/wd")
        self.assertEqual(tail, ["--build-command", "make"])

    def test_engine_max_concurrency_defaults_to_none(self) -> None:
        """Omitting --max-concurrency leaves ns.max_concurrency as None.

        The engine treats this as "use the workflow-supplied default" so
        the WorkflowSpec.max_concurrency value is honored when the user
        does not override it on the CLI.
        """
        ns, _ = _parse_engine_args(
            ["--workflow", "pkg.mod:factory", "--workdir", "/tmp/wd"]
        )
        self.assertIsNone(ns.max_concurrency)

    def test_engine_max_concurrency_accepts_explicit_value(self) -> None:
        """An explicit --max-concurrency value is preserved as an int."""
        ns, _ = _parse_engine_args(
            [
                "--workflow",
                "pkg.mod:factory",
                "--workdir",
                "/tmp/wd",
                "--max-concurrency",
                "7",
            ]
        )
        self.assertEqual(ns.max_concurrency, 7)


class _EngineCommandTest(unittest.TestCase):
    def test_default_uses_argv0_engine(self) -> None:
        """Without an override, the engine prefix is 'sys.argv[0] engine'."""
        cmd = _build_engine_command(
            engine_command_override=None,
            workflow_spec="pkg.mod:factory",
            workdir=Path("/tmp/wd"),
            framework_flags=["--max-concurrency", "3"],
            workflow_argv_tail=["--build-command", "make"],
        )
        # argv[0] varies under buck test; only assert structural shape.
        self.assertEqual(cmd[1], "engine")
        self.assertEqual(
            cmd[2:],
            [
                "--workflow",
                "pkg.mod:factory",
                "--workdir",
                "/tmp/wd",
                "--max-concurrency",
                "3",
                "--",
                "--build-command",
                "make",
            ],
        )

    def test_override_replaces_prefix(self) -> None:
        """An --engine-command override replaces the default argv[0] prefix."""
        cmd = _build_engine_command(
            engine_command_override="buck run //x:engine --",
            workflow_spec="pkg.mod:factory",
            workdir=Path("/tmp/wd"),
            framework_flags=[],
            workflow_argv_tail=[],
        )
        self.assertEqual(
            cmd,
            [
                "buck",
                "run",
                "//x:engine",
                "--",
                "--workflow",
                "pkg.mod:factory",
                "--workdir",
                "/tmp/wd",
            ],
        )

    def test_no_tail_omits_double_dash(self) -> None:
        """An empty workflow tail does not append a stray '--'."""
        cmd = _build_engine_command(
            engine_command_override=None,
            workflow_spec="pkg.mod:factory",
            workdir=Path("/tmp/wd"),
            framework_flags=[],
            workflow_argv_tail=[],
        )
        self.assertNotIn("--", cmd[2:])


class _CoreWorkflowExportTest(unittest.TestCase):
    def test_workflow_spec_is_protocol(self) -> None:
        """``WorkflowSpec`` re-exported from core is a ``Protocol`` subclass.

        ``Protocol`` subclasses are flagged with ``_is_protocol = True`` by
        the typing machinery; this guards against accidentally weakening
        ``WorkflowSpec`` to a regular class (which would silently change
        the runtime semantics for workflow authors).
        """
        from spdl.autoresearch.core import WorkflowSpec

        self.assertTrue(getattr(WorkflowSpec, "_is_protocol", False))

    def test_workflow_factory_is_callable_alias(self) -> None:
        """``WorkflowFactory`` re-exported from core is a ``Callable`` alias."""
        from spdl.autoresearch.core import WorkflowFactory

        self.assertIs(get_origin(WorkflowFactory), Callable)


class _MainImportTest(unittest.TestCase):
    def test_main_import_does_not_load_app(self) -> None:
        """Importing spdl.autoresearch.__main__ as a module is a no-op.

        The framework dispatcher (under spdl.autoresearch._app) must
        NOT be transitively loaded by ``import
        spdl.autoresearch.__main__``. _app is reachable only when
        __main__.py runs as a script (i.e. via ``python -m
        spdl.autoresearch``), at which point ``__name__ ==
        "__main__"`` and the lazy import inside the guard fires.
        """
        removed = {}
        for mod_name in [
            name
            for name in list(sys.modules)
            if name == "spdl.autoresearch.__main__"
            or name.startswith("spdl.autoresearch._app")
        ]:
            removed[mod_name] = sys.modules.pop(mod_name)
        self.addCleanup(sys.modules.update, removed)

        importlib.import_module("spdl.autoresearch.__main__")

        self.assertNotIn("spdl.autoresearch._app", sys.modules)
        self.assertNotIn("spdl.autoresearch._app._main", sys.modules)
