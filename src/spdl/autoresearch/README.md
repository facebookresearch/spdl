# spdl.autoresearch

Pluggable framework for automated, LLM-driven experiment workflows.

## Code Structure

The package is organized into four layers:

- **`core/`** — Public submodule (`spdl.autoresearch.core`). Domain-neutral async scheduling engine with checkpoint/resume, the `WorkflowProtocol` and `WorkflowSpec` contracts that pluggable workflows implement, default JSON-backed engine-state persistence helpers (`write_engine_state`, `read_engine_state`, `load_or_initial`), and shared domain types (failure records, hypothesis nodes, analysis results).

- **`_app/`** — Private framework dispatcher. Resolves `--workflow module.path:factory` (or short names registered under the `spdl.autoresearch.workflows` Python entry-points group), parses framework-level CLI arguments, and either launches an interactive supervisor agent or drives the engine directly. The `spdl-autoresearch` Buck binary delegates here.

- **`_common/`** — Shared utility modules used across the package. No `__init__.py` — import leaf modules directly (e.g. `from spdl.autoresearch._common._state import read_state`).

- **`pipeline_optimization/`** — Public submodule (`spdl.autoresearch.pipeline_optimization`). One concrete workflow that ships with SPDL: SPDL data-loading pipeline optimization. Exposes `create_workflow` as its `WorkflowFactory` entry point. Organized as:
  - `_ops/` — Workflow operations: experiment lifecycle, analysis, planning policy, source mutations, failure handling, and persistent state.
  - `_platform/` — Platform abstraction: coding agent implementations, local/remote execution, and provider discovery.
  - `prompts/` — Markdown prompt templates for LLM workflow agents, organized by category (phase prompts, knowledge, platform, supervisor).

## Adding a new autoresearch workflow

Implement a `WorkflowFactory` (`Callable[[list[str], Path | None], WorkflowSpec]`). The factory parses workflow-specific CLI flags, returns a `WorkflowSpec` with the supervisor- and engine-phase methods filled in, and optionally registers itself under the `spdl.autoresearch.workflows` entry-points group for short-name lookup. The framework dispatcher (`_app/`) and the `core/` contracts handle everything else — workflows do not need to depend on `pipeline_optimization`.
