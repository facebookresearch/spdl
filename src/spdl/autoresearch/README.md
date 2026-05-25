# spdl.autoresearch

Automated experiment engine for optimizing SPDL data loading pipelines.

## Code Structure

The package is organized into three layers:

- **`core/`** — Public submodule (`spdl.autoresearch.core`). Domain-neutral async scheduling engine with checkpoint/resume, and shared domain types (failure records, hypothesis nodes, analysis results).

- **`_common/`** — Shared utility modules used across the package. No `__init__.py` — import leaf modules directly (e.g. `from spdl.autoresearch._common._state import read_state`).

- **`pipeline_optimization/`** — Public submodule (`spdl.autoresearch.pipeline_optimization`). The concrete SPDL pipeline optimization workflow, organized as:
  - `_ops/` — Workflow operations: experiment lifecycle, analysis, planning policy, source mutations, failure handling, and persistent state.
  - `_platform/` — Platform abstraction: coding agent implementations, local/remote execution, and provider discovery.
  - `prompts/` — Markdown prompt templates for LLM workflow agents, organized by category (phase prompts, knowledge, platform, supervisor).

When adding a new autoresearch use case (e.g. model architecture search, hyperparameter tuning), it should be added as a new public submodule under `spdl.autoresearch`, reusing `core/` and `_common/`, but not `pipeline_optimization`.
