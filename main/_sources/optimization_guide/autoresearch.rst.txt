.. _autoresearch:

Autoresearch
============

.. py:currentmodule:: spdl.pipeline

In the previous sections, we discussed a manual workflow for optimizing
data loading pipelines: instrument the pipeline, run experiments, analyze
metrics, form hypotheses, apply changes, and repeat.

This process is systematic and methodical, which makes it a good candidate
for automation. **Autoresearch** is an engine that automates this entire
optimization loop. It uses Claude to analyze pipeline metrics, identify
bottlenecks, propose code changes, and iteratively improve performance
with minimal human intervention.

Overview
--------

Autoresearch operates in three phases:

1. **Initialization** -- Instrument the pipeline with metrics logging,
   establish a baseline, and measure headspace.
2. **Fixed experiments** -- Run structural optimizations that are known
   to be high-impact (subprocess pipeline, batch size tuning).
3. **Iterative optimization** -- Claude analyzes results, proposes
   follow-up experiments, and the engine executes them concurrently
   until metrics plateau or all queued work is exhausted.

The following diagram illustrates the overall flow.

.. mermaid::

   flowchart TB
       Init["Initialize workdir\n& instrument pipeline"]
       Baseline["Run baseline job"]
       Headspace["Run headspace analysis\n(CacheDataLoader)"]
       MTP["Run subprocess pipeline\n(MTP)"]
       Analyze["Analyze completed job"]
       Plan["Claude plans\n2-3 follow-up experiments"]
       Launch["Launch experiments\n(up to N concurrent)"]
       Stop{"Plateau\nreached?"}
       Report["Generate report"]

       Init --> Baseline
       Init --> Headspace
       Init --> MTP
       Baseline --> Launch
       Headspace --> Launch
       MTP --> Launch
       Launch --> Analyze
       Analyze --> Plan
       Plan --> Stop
       Stop -- No --> Launch
       Stop -- Yes --> Report


Getting Started
---------------

Running autoresearch requires the following inputs:

- A **pipeline script** -- the Python file containing the SPDL pipeline
  to optimize.
- A **source directory** -- the directory containing the pipeline code.
  The engine modifies files in this directory during experiments.
- A **build command** -- how to build the job image (e.g., ``fbpkg build``
  or ``docker build``).
- A **launch command template** -- the command to launch a training job.
  Use ``$IMAGE`` as a placeholder for the image name.

The engine is invoked with:

.. code-block:: bash

   python run.py <workdir> \
     --pipeline-script <path/to/pipeline.py> \
     --source-dir <path/to/source/> \
     --build-command "<build command>" \
     --base-launch-command "<launch command with \$IMAGE>" \
     --notes "<description of the experiment>" \
     --max-iterations 10 \
     --patience 3 \
     --max-concurrency 3 \
     --job-timeout 1800

Configuration is persisted to ``<workdir>/config.json`` on the first run.
To resume after an interruption, simply re-run with the workdir alone:

.. code-block:: bash

   python run.py <workdir>


How It Works
------------

Instrumentation
~~~~~~~~~~~~~~~

On the first run, the engine automatically instruments the pipeline
script with TTFB (time to first batch) and per-step timing. This is
done by sending the pipeline source to Claude with instructions to add
lightweight logging that records when each training step starts and ends.

The instrumented code is committed to source control (Sapling or Git),
creating a clean baseline for subsequent experiments to branch from.

Fixed Experiments
~~~~~~~~~~~~~~~~~

Autoresearch schedules three fixed experiments before entering the
iterative loop. Each addresses a known high-impact optimization area.

**Baseline**

The unmodified pipeline is run to establish baseline metrics: step time,
GPU SM utilization, data readiness, and throughput. All subsequent
experiments are compared against this baseline.

**Headspace analysis**

As described in :ref:`headspace-analysis`, the pipeline is wrapped with
:py:class:`~spdl.dataloader.CacheDataLoader` to measure the upper
bound of improvement achievable by optimizing data loading. If the
headspace is near zero, the bottleneck is model compute, not data
loading. The engine uses this information to decide which optimizations
to prioritize.

**Subprocess pipeline (MTP)**

The pipeline is moved to a subprocess to eliminate GIL contention
between the data loading threads and the training loop. This is often
the single highest-impact optimization, as discussed in
:ref:`resolution`. By running the pipeline in a separate process, the
data loading threads no longer compete with PyTorch for the GIL.

Iterative Optimization
~~~~~~~~~~~~~~~~~~~~~~

After the fixed experiments, autoresearch enters an iterative loop:

1. **Analyze** -- When a job completes, the workflow collects system
   metrics (GPU SM utilization, CPU utilization) and SPDL pipeline
   statistics (per-stage execution time, queue occupancy, throughput).
   These are sent to Claude, which produces a structured analysis
   identifying the bottleneck and evaluating the experiment's hypothesis.

2. **Plan** -- Claude receives the full experiment history, the current
   best metrics, and the pipeline source code. Based on this context,
   it proposes 2-3 follow-up experiments. Each proposal includes a
   hypothesis, the specific changes to make, and whether the image
   needs rebuilding.

3. **Execute** -- The workflow applies code changes (if any), builds the
   image, and launches jobs. Up to ``max_concurrency`` jobs run
   simultaneously. Each job is monitored for completion and timeout.

4. **Repeat** -- The loop continues until the stopping conditions are
   met: metrics have not improved for ``patience`` consecutive planning
   sessions and all known best practices have been tried.

.. mermaid::

   sequenceDiagram
       participant Runner
       participant Workflow
       participant Claude
       participant Cluster

       Runner->>Workflow: Start WorkSpec coroutine
       Workflow->>Cluster: Launch job
       Cluster-->>Workflow: Job completed
       Workflow->>Workflow: Collect metrics
       Workflow->>Claude: Analyze (metrics + pipeline code)
       Claude-->>Workflow: Bottleneck analysis + structured metrics
       Workflow->>Claude: Plan (history + analysis + code)
       Claude-->>Workflow: 2-3 experiment proposals
       Workflow-->>Runner: Return child WorkSpecs
       Runner->>Workflow: Start next WorkSpec coroutines


Runner Architecture
-------------------

The implementation is split into a small generic runner and an
autoresearch-specific workflow adapter. The split is intentional:
the runner only schedules coroutine work, while the workflow owns
all experiment semantics.

Generic Runner
~~~~~~~~~~~~~~

The generic runner (``utils/runner.py``) knows nothing about SPDL,
training jobs, source control, metrics, Claude, or hypothesis planning.
It operates on serializable ``WorkSpec`` objects and is responsible only
for:

- Maintaining a priority queue of pending ``WorkSpec`` objects.
- Starting up to ``max_concurrency`` coroutines.
- Waiting for the first coroutine to complete.
- Passing completed results back to the adapter.
- Checkpointing queued and running specs.
- Persisting an ``interrupted`` checkpoint on local cancellation.

The runner does not inspect experiment payloads. If a future change
needs to know how a remote job is launched, how code is modified, how
metrics are interpreted, or how follow-up experiments are generated,
that change belongs in the autoresearch workflow layer rather than in
the runner.

Autoresearch Workflow Adapter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The autoresearch workflow (``utils/autoresearch_workflow.py``) is the
domain side of the boundary. It turns an experiment ``WorkSpec`` into a
coroutine that performs the full experiment lifecycle:

- Restore or prepare the source tree.
- Apply code changes when the experiment requires a rebuild.
- Build the image.
- Launch or resume the remote job.
- Poll for completion and detect stalled jobs.
- Collect metrics and run Claude analysis.
- Record state, master-table rows, findings, and plots.
- Ask Claude for follow-up experiments and return them as child
  ``WorkSpec`` objects.

The workflow also writes the user-facing compatibility files under
``<workdir>/engine`` and ``<workdir>/runs``. This keeps monitoring and
reporting stable while the runner remains simple.

Stateless Claude Invocations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each Claude call is fully stateless. The workflow constructs a
self-contained prompt that includes everything Claude needs: the
SPDL optimization knowledge base, the full experiment history,
collected metrics, and the pipeline source code. There is no
persistent conversation or session state.

This design makes the system robust to interruptions. After ``Ctrl+C``,
the runner can resume from the last persisted ``engine/checkpoint.json``
without relying on a Claude session.

Hypothesis Tree
~~~~~~~~~~~~~~~

Experiments are organized in a tree structure. The initial experiments
(baseline, headspace, MTP) are root nodes. Follow-up experiments
proposed by Claude become children of the node that triggered the
planning.

.. code-block:: text

   baseline
   headspace
   subprocess_mtp
   ├── batch_size_8
   │   ├── batch_size_16
   │   └── batch_size_6
   ├── concurrency_4
   └── torch_compile
       ├── compile_fused_bs6
       └── compile_fused_bs12

Each node tracks its status (queued, preparing, running, analyzing,
completed, failed), the source control commit it was built from, and
the analysis results. The tree is owned by the workflow and visualized
as ``hypothesis_tree.png`` after each experiment completes.


Monitoring
----------

All experiment state lives in the workdir. The following files are
useful for monitoring progress.

``engine/engine_state.json``
   Engine status (``running``, ``interrupted``, or ``stopped``) and
   experiment counts (queued, running, completed, failed).

``engine/checkpoint.json``
   Runner checkpoint containing the serialized queued and running
   ``WorkSpec`` objects. This is the source of truth for resume.

``engine/queue.json``
   Compatibility view of pending experiments in priority order.

``engine/active.json``
   Compatibility view of currently running remote jobs.

``summary.md``
   Human-readable progress summary updated after each job completion.

``master_table.tsv``
   Tab-separated table of all experiments with key metrics: step time,
   SM utilization, data readiness, and duration.

``progress.png``
   Scatter plot showing job duration and SM utilization over time.
   Green dots indicate improvements over the previous best.

``hypothesis_tree.png``
   Tree visualization of the experiment hierarchy. Nodes are
   color-coded: green for improved, gray for no improvement, red
   for failed, blue for running, and dashed white for queued.

``runs/<run_id>/analysis.md``
   Claude's detailed analysis for each completed experiment,
   including per-stage pipeline metrics and bottleneck identification.

Generating a Report
~~~~~~~~~~~~~~~~~~~

After the engine finishes, generate a final summary report:

.. code-block:: bash

   python cmd.py report <workdir>

This collects all per-run analyses and the master table, sends them
to Claude for synthesis, and writes the output to ``report.md``.


Stopping and Resuming
---------------------

To stop autoresearch gracefully, send ``SIGINT`` (Ctrl+C) to the
process. The runner cancels local coroutines and persists queued and
running specs to ``engine/checkpoint.json`` with status
``interrupted``. The workflow also keeps the monitoring files under
``engine/`` up to date.

.. warning::

   Do not send ``SIGKILL`` (``kill -9``) to the engine process.
   This prevents state persistence and you may lose the queue and
   in-progress analysis.

Running jobs on the cluster are **not** cancelled when autoresearch
stops. They continue independently. When autoresearch resumes from
``engine/checkpoint.json``, the workflow re-checks their status and
collects results.

To resume, simply re-run with the workdir:

.. code-block:: bash

   python run.py <workdir>

Modifying the Queue
~~~~~~~~~~~~~~~~~~~

To manually adjust the experiment queue, stop the engine and edit
``engine/checkpoint.json``. The ``queued`` list contains serialized
``WorkSpec`` objects; change their ``priority`` values or remove specs
as needed. Lower values run first. ``engine/queue.json`` is a
compatibility view for monitoring and should not be treated as the
resume source of truth.

.. code-block:: json

   {
     "status": "interrupted",
     "queued": [
       {
         "id": "010_compile_fused_bs6",
         "priority": -100,
         "kind": "experiment",
         "payload": {"node": {"node_id": "010_compile_fused_bs6"}}
       }
     ],
     "running": []
   }


Workdir Structure
-----------------

.. code-block:: text

   <workdir>/
   ├── config.json                 # Experiment configuration
   ├── state.json                  # Persistent state (history, best metrics)
   ├── master_table.tsv            # All experiments and their metrics
   ├── summary.md                  # Human-readable progress summary
   ├── progress.png                # Duration/SM scatter plot
   ├── hypothesis_tree.png         # Experiment tree visualization
   ├── engine/
   │   ├── checkpoint.json         # Runner checkpoint for resume
   │   ├── engine_state.json       # Engine status and counts
   │   ├── tree.json               # Full hypothesis tree
   │   ├── queue.json              # Pending experiments (compat view)
   │   ├── active.json             # Currently running jobs (compat view)
   │   └── nodes/
   │       └── <node_id>/
   │           ├── spec.json       # Experiment specification
   │           ├── status.txt      # Current status
   │           └── result.json     # Analysis results
   ├── runs/
   │   └── <run_id>/
   │       ├── analysis.md         # Claude's analysis
   │       └── metrics/            # Raw metrics data
   └── logs/
       ├── autoresearch.log        # Full execution log
       ├── *_prompt.md             # Prompts sent to Claude
       ├── *_output.md             # Claude's responses
       └── *_raw.json              # Raw response JSON with cost


Example
-------

The following is a typical autoresearch session optimizing an LLM
fine-tuning pipeline. The engine found that the baseline had 0%
headspace (data loading was not the bottleneck), but subprocess
pipeline mode (MTP) still improved step time by 15% by eliminating
GIL contention. Subsequent experiments explored batch size tuning
and ``torch.compile``, achieving a 64% throughput improvement.

.. code-block:: text

   $ python run.py /tmp/llm_finetune_opt \
       --pipeline-script pipeline.py \
       --source-dir ./src \
       --build-command "docker build -t my_image ." \
       --base-launch-command "torchx run ... --image \$IMAGE" \
       --max-iterations 20 --patience 3 --max-concurrency 3

   Initialized experiment at /tmp/llm_finetune_opt
   Instrumenting pipeline with TTFB/step-time logging...
   Starting autoresearch engine

   [baseline]     step_time=444ms  SM=65%  data_readiness=94%
   [headspace]    step_time=444ms  SM=62%  headspace=0%
   [mtp]          step_time=376ms  SM=73%  data_readiness=100%  (+15%)
   [batch_size_8] step_time=102ms  SM=60%  throughput=100 samp/s
   [torch_compile] step_time=240ms SM=86%  throughput=133 samp/s (+64%)
   ...

   Engine stopped: plateau reached (3/3), all best practices tried.
   Best result: torch_compile (step_time=240ms, 133 samples/s/rank)
