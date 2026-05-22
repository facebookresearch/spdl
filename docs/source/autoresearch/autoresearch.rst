
.. _autoresearch:

How It Works
============

Overview
--------

Autoresearch operates in three phases:

1. **Initialization** -- Instrument the pipeline with metrics logging,
   establish a baseline, and measure headspace.
2. **Seed experiments** -- Launch structural optimizations that are known
   to be high-impact (baseline measurement, headspace analysis,
   subprocess pipeline).
3. **Iterative optimization** -- The coding agent analyzes results,
   proposes follow-up experiments, and the engine executes them
   concurrently until metrics plateau.

The system is built as two tiers: an interactive **supervisor agent**
that gathers configuration and monitors progress, and a non-interactive
**async engine** that runs the experiment loop. The supervisor starts
the engine in the background and can inspect workdir state while
experiments run.

The following screenshot shows the supervisor agent reporting on an
active autoresearch run — experiment table, running jobs, findings,
and diagnostics are all visible in one status query:

.. image:: /_static/data/autoresearch_supervisor_2.png
   :alt: Supervisor agent reporting status after the initialization
   :width: 100%


.. image:: /_static/data/autoresearch_supervisor_3.png
   :alt: Supervisor agent reporting status responding to user inquery
   :width: 100%

.. image:: /_static/data/autoresearch_supervisor_4.png
   :alt: Supervisor agent reporting status at the end.
   :width: 100%


Getting Started
---------------

The recommended entry point is ``cli.py``, which launches an
interactive supervisor agent that gathers missing configuration,
starts the engine, and monitors progress:

.. code-block:: bash

   python spdl/tools/autoresearch/cli.py <workdir> \
     --pipeline-script <path/to/pipeline.py> \
     --source-dir <path/to/source/> \
     --build-command "<build command>" \
     --base-launch-command "<launch command with \$IMAGE>"

The supervisor asks for any missing values interactively. Once
configuration is complete, it launches the engine in the background
and reports progress as experiments complete.

To run the engine directly without a supervisor (non-interactive mode):

.. code-block:: bash

   python run.py <workdir> \
     --pipeline-script <path/to/pipeline.py> \
     --source-dir <path/to/source/> \
     --build-command "<build command>" \
     --base-launch-command "<launch command with \$IMAGE>" \
     --max-iterations 10 \
     --patience 3 \
     --max-concurrency 3 \
     --job-timeout 1800

The engine requires the following inputs:

- A **pipeline script** -- the Python file containing the SPDL pipeline
  to optimize.
- A **source directory** -- the directory containing the pipeline code.
  The engine modifies files in this directory during experiments.
- A **build command** -- how to build the job image (e.g.,
  ``docker build``).
- A **launch command template** -- the command to launch a training job.
  Use ``$IMAGE`` as a placeholder for the image name.

Configuration is persisted to ``<workdir>/config.json`` on the first run.
To resume after an interruption, simply re-run with the workdir alone:

.. code-block:: bash

   python spdl/tools/autoresearch/cli.py <workdir>

   # or directly:
   python spdl/tools/autoresearch/run.py <workdir>


How It Works
------------

Instrumentation
~~~~~~~~~~~~~~~

On the first run, the engine automatically instruments the pipeline
script with TTFB (time to first batch) and per-step timing. The
pipeline source is sent to the coding agent with instructions to add
lightweight logging that records when each training step starts and ends.

The instrumented code is committed to source control (Sapling or Git),
creating a clean baseline for subsequent experiments to branch from.

Seed Experiments
~~~~~~~~~~~~~~~~

The engine schedules three seed experiments before entering the
iterative loop. Each addresses a known high-impact area and forms a
root node in the hypothesis tree.

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
a high-impact optimization, as discussed in :ref:`resolution`. By
running the pipeline in a separate process, the data loading threads
no longer compete with PyTorch for the GIL.

Iterative Optimization
~~~~~~~~~~~~~~~~~~~~~~

After the seed experiments, autoresearch enters an iterative loop:

1. **Analyze** -- When a job completes, the workflow collects system
   metrics (GPU SM utilization, CPU utilization) and SPDL pipeline
   statistics (per-stage execution time, queue occupancy, throughput).
   These are sent to the coding agent, which produces a structured
   analysis identifying the bottleneck and evaluating the experiment's
   hypothesis.

2. **Plan** -- The coding agent receives the full experiment history,
   the current best metrics, and the pipeline source code. Based on
   this context, it proposes follow-up experiments. Each proposal
   includes a hypothesis, the specific changes to make, and whether the
   image needs rebuilding.

3. **Execute** -- The workflow applies code changes (if any), builds the
   image, and launches jobs. Up to ``max_concurrency`` jobs run
   simultaneously. Each job is monitored for completion and timeout.

4. **Repeat** -- The loop continues until the stopping conditions are
   met: metrics have not improved for ``patience`` consecutive planning
   sessions and all known best practices have been tried.

.. mermaid::

   sequenceDiagram
       participant Engine as Orchestrator
       participant Adapter as AutoresearchAdapter
       participant Agent as Coding Agent
       participant Platform as Platform

       Engine->>Adapter: Start WorkSpec coroutine
       Adapter->>Platform: Launch job
       Platform-->>Adapter: Job completed
       Adapter->>Adapter: Collect metrics
       Adapter->>Agent: Analyze (metrics + pipeline code)
       Agent-->>Adapter: Bottleneck analysis + structured metrics
       Adapter->>Agent: Plan (history + analysis + code)
       Agent-->>Adapter: Experiment proposals
       Adapter-->>Engine: Return child WorkSpecs
       Engine->>Adapter: Start next WorkSpec coroutines


Monitoring
----------

All experiment state lives in the workdir. The following files are
useful for monitoring progress.

``engine/engine_state.json``
   Engine status (``running``, ``interrupted``, or ``stopped``) and
   experiment counts (queued, running, completed, failed).

``engine/checkpoint.json``
   Runner checkpoint containing the serialized queued and running
   ``TaskSpec`` objects. This is the source of truth for resume.

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
   The coding agent's detailed analysis for each completed experiment,
   including per-stage pipeline metrics and bottleneck identification.

Generating a Report
~~~~~~~~~~~~~~~~~~~

After the engine finishes, generate a final summary report:

.. code-block:: bash

   python cmd.py report <workdir>

This collects all per-run analyses and the master table, sends them
to the coding agent for synthesis, and writes the output to
``final_report.md``.


Stopping and Resuming
---------------------

To stop autoresearch gracefully, send ``SIGINT`` (Ctrl+C) to the
process. The engine cancels local coroutines and persists queued and
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

   python spdl/tools/autoresearch/cli.py <workdir>

   # or directly:
   python spdl/tools/autoresearch/run.py <workdir>

Modifying the Queue
~~~~~~~~~~~~~~~~~~~

To manually adjust the experiment queue, stop the engine and edit
``engine/checkpoint.json``. The ``queued`` list contains serialized
``TaskSpec`` objects; change their ``priority`` values or remove specs
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
   │       ├── analysis.md         # Coding agent's analysis
   │       └── metrics/            # Raw metrics data
   └── logs/
       ├── autoresearch.log        # Full execution log
       ├── *_prompt.md             # Prompts sent to the coding agent
       ├── *_output.md             # Coding agent's responses
       └── *_raw.json              # Raw response JSON with cost


Example
-------

For a detailed walkthrough of an autoresearch run on a real pipeline,
see :ref:`autoresearch-example`.
