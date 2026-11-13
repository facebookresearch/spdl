.. _optimization-knowledge-base:

Knowledge Base
==============

The knowledge base helps you build, migrate and optimize SPDL input
pipelines. It brings together practical guidance and techniques learned
from training workloads, including approaches that did not help.

Use the :ref:`optimization guide <optimization-guide>` to understand
pipeline performance statistics. Use the knowledge base to choose what
to investigate next and plan a comparison for your workload.

Where to start
--------------

- **Moving an existing loader to SPDL?** Start with
  :doc:`Migration guidance <_generated/migration/_categories/migration-guidance>`.
  Preserve sampling, transforms and batch contents, then check correctness
  and performance before optimizing.
- **Building a new pipeline?** Read
  :doc:`Pipeline authoring <_generated/authoring/_categories/pipeline-authoring>`
  for stage composition and execution patterns.
- **Investigating slow training?** Use
  :doc:`Optimization guidance <_generated/optimization/_categories/optimization-guidance>`
  to interpret stage statistics, identify the exposed cost and compare
  candidate changes.
- **Looking for a technique that fits a measured bottleneck?** Browse the
  :doc:`Technique catalog <_generated/optimization/_categories/technique-catalog>`.
  Its entries are indexed by symptoms such as cache pressure, expensive
  collation, transfer delays and uneven data readiness across ranks.
- **Exploring a particular part of the input path?** Open
  :doc:`Techniques by area <_generated/optimization/techniques/_categories/techniques-by-area>`
  for caching, execution, batch assembly, transfer and preprocessing
  techniques, alongside negative results.
- **Checking recurring fetch overhead?** The
  :doc:`Startup audit <_generated/_categories/startup-audit>` follows fetch
  calls through their wrappers to reveal repeated setup or blocking work.

The categories in the sidebar open topic lists. You can follow those
links or use documentation search to find a particular technique.

Reading a technique
-------------------

Each catalog entry explains:

- **Trigger:** the observation that makes the technique worth testing.
- **Technique:** what to change and why it may reduce the measured cost.
- **Guards:** correctness requirements, resource costs and known failure
  modes to check before keeping the change.
- **ID:** a stable name for recording the experiment and its outcome.
- **SPDL reference, when available:** an API or example you can compare
  with your existing implementation. It is not a required replacement.

The catalog covers storage and caching, process and thread execution,
batch assembly, GPU transfer and preprocessing, and training boundaries.
It also records negative results so you can understand when an otherwise
plausible approach failed.

Applying a technique
--------------------

Start from a working baseline and choose a candidate whose trigger
matches your measurements. Compare completed training work, not just an
isolated stage or GPU utilization. Keep sampling, batch semantics and
model-quality checks consistent between the baseline and the candidate.

Measure again after each change: removing one bottleneck can expose
another. A faster implementation is useful only if it preserves the data
and produces a model that meets your quality requirements.

These techniques are starting points, not promised speedups. Hardware,
input representation and storage conditions matter. Source links are
available separately as optional provenance; the entries explain the
mechanism without requiring you to read a past experiment. Some techniques
trade memory or CPU for speed; others help one workload and slow another.
Check the guards and validate the complete pipeline on your own workload.

Using the guidance with an agent
--------------------------------

You can use these pages yourself or give the relevant guidance to a
coding agent. The source Markdown is in ``tools/skills/`` in the SPDL
repository; the catalog starts at
``tools/skills/optimization/technique_catalog.md``. Coding agents can
consume these files as-is, along with the neighboring analysis, migration
and authoring guides, without extracting the guidance from the website.

Provide the pipeline configuration, baseline measurements
and your correctness and quality requirements. Ask the agent to explain
which observations support its proposed changes and to retain the
experiment results, including unsuccessful approaches.
