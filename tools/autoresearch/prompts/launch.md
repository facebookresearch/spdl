# Autoresearch: Automated SPDL Pipeline Optimization

You are helping the user launch and monitor the autoresearch engine, which automatically optimizes SPDL data loading pipeline performance for AI training jobs.

## Step 1: Gather Configuration

Ask the user for these required inputs:

1. **Workdir** — where to store experiment state (e.g., `/tmp/autoresearch_exp`)
2. **Pipeline script** — the Python file containing the SPDL pipeline to optimize (e.g., `path/to/pipeline.py`)
3. **Source directory** — the directory containing the pipeline code (for in-place modifications)
4. **Build command** — how to build the job image (e.g., `docker build -t my_image .`)
5. **Launch command template** — the command to launch a training job, using `$IMAGE` as placeholder (e.g., `torchx run -s local entrypoint.py:main --image $IMAGE --num_workers 8`)

Optional inputs (have sensible defaults):

- **Notes** — free-form description of the experiment context
- **Max iterations** — default 10
- **Patience** — stop after N non-improving iterations, default 3
- **Max concurrency** — concurrent training jobs, default 3
- **Job timeout** — wall-clock timeout per job in seconds, default 1800 (30 min)
- **Poll interval** — seconds between status checks, default 120
- **Platform** — execution platform, default `auto`; use `local` for local subprocess execution
- **Agent** — coding agent, default `claude`; use `codex` when that CLI is configured

## Step 2: Start the Engine

Run the engine with the gathered config. The script is at `spdl/tools/autoresearch/run.py`.

```bash
python run.py <workdir> \
  --pipeline-script <path> \
  --source-dir <path> \
  --build-command "<command>" \
  --base-launch-command "<command template>" \
  --notes "<description>" \
  --dangerously-skip-permissions \
  --max-iterations 10 \
  --patience 3 \
  --max-concurrency 3 \
  --job-timeout 1800 \
  --platform auto \
  --agent claude
```

Run this command in the background so you can monitor it. The engine handles everything: initialization, pipeline instrumentation, baseline job, headspace analysis, MTP experiment, and iterative optimization.

**On resume** (after Ctrl+C or crash), just re-run with the workdir only:

```bash
python run.py <workdir>
```

Config is persisted in `<workdir>/config.json` from the first run.

## Step 3: Generate Final Report

After the engine finishes:

```bash
python cmd.py report <workdir>
```

## What the Engine Does

The engine runs these fixed initial experiments (skipping any already done):

1. **Baseline** — unmodified pipeline, establishes baseline metrics
2. **Headspace** — wraps pipeline with CacheDataLoader to measure data loading overhead ceiling
3. **MTP** — runs pipeline in subprocess to test GIL contention elimination

After the fixed experiments, the coding agent proposes follow-up experiments based on analysis results. The engine:

- Runs up to `max_concurrency` jobs simultaneously
- Analyzes each job immediately on completion (no waiting for the batch)
- Plans follow-up experiments as children in a hypothesis tree
- Re-prioritizes the queue based on which approaches show promise
- Updates `summary.md`, `progress.png`, and `hypothesis_tree.png` after each completion
- Stops when all best practices are tried and metrics plateau

---

## Monitoring a Running Engine

### Key Files to Watch

All files are in `<workdir>/`. Read them periodically to report progress to the user.

**`engine/engine_state.json`** — Engine status and counts. Check this first.

```json
{
  "status": "running",
  "timestamp": "...",
  "total_nodes": 8,
  "queued": 2,
  "running": 3,
  "completed": 2,
  "failed": 1
}
```

- `"running"` — engine is active
- `"interrupted"` — stopped by Ctrl+C, can resume
- `"stopped"` — finished (stopping conditions met or all work exhausted)

**`engine/checkpoint.json`** — Runner checkpoint containing serialized queued and running `WorkSpec` objects. This is the source of truth for resume after Ctrl+C or a crash.

**`summary.md`** — Human-readable progress summary. Updated after each job completion. Show this to the user when they ask about progress.

**`progress.png`** — Karpathy-style scatter plot showing job duration and SM utilization over time. Green dots are improvements, gray are discarded, red X are failures.

**`hypothesis_tree.png`** — Tree visualization of the experiment hierarchy. Color-coded: green=completed/improved, gray=completed/no improvement, red=failed, blue=running, white dashed=queued. The best path is highlighted with thick green edges.

**`engine/active.json`** — Currently running jobs with node_id, job_id, and launch timestamp. This is a monitoring view.

**`engine/queue.json`** — Pending experiments in priority order. This is a monitoring view; do not treat it as the resume source of truth.

**`engine/nodes/<node_id>/`** — Per-experiment details: `spec.json` (what it does), `status.txt` (current state), `result.json` (analysis results after completion).

**`master_table.tsv`** — Tab-separated table of all experiments with metrics.

**`runs/<run_id>/analysis.md`** — Coding-agent detailed analysis for each completed experiment.

### Reporting Progress

When the user asks for status, read `engine/engine_state.json` and `summary.md`, then provide a concise update:

1. How many experiments completed / running / queued
2. Current best result and by how much it improved over baseline
3. What's running right now
4. Any failures or issues

If they want more detail, show `progress.png` or read specific `runs/*/analysis.md` files.

---

## Troubleshooting

### Stopping the Engine

To stop the engine gracefully, send SIGINT to the process. The engine needs time to persist its state (tree, queue, active jobs) to disk before exiting.

1. Send `kill -INT <pid>` to the engine process
2. **Wait at least 15 seconds** for the engine to persist state and exit
3. Check if the process is still alive with `ps -p <pid>`
4. If still alive after 15 seconds, send `kill -TERM <pid>` and wait another 10 seconds
5. Verify `engine/engine_state.json` shows `"status": "interrupted"`

Do NOT send SIGKILL — it prevents state persistence and can lose queued or running work that has not reached `engine/checkpoint.json`.

Running jobs on the cluster are NOT cancelled by stopping the engine. They continue independently. The engine re-checks their status on resume from `engine/checkpoint.json`.

### Engine Status: "interrupted"

Normal — the engine was stopped. Resume by re-running the engine with just the workdir.

### Build Failures

Check `<workdir>/logs/last_build_error.txt`. Common causes: dependency issues from code changes or syntax errors in agent modifications. To intervene: stop the engine (Ctrl+C), fix the code in the source directory, then resume.

### All Jobs Failing

Read `engine/nodes/*/status.txt` to find which nodes failed, then read their `spec.json`. Check:

1. Is the launch command correct? Read `<workdir>/config.json` → `base_launch_command`.
2. Are jobs actually running? Check job status using your infrastructure's CLI.
3. Is the image building correctly? Check `<workdir>/logs/last_build_error.txt`.

### Jobs Timing Out

Jobs exceeding `job_timeout_s` (default 30 min) are killed automatically. If legitimate jobs need more time: stop the engine, edit `<workdir>/config.json` → increase `job_timeout_s`, then resume.

### No Progress / Plateau

Check `hypothesis_tree.png` — are all branches exhausted? Read the latest `runs/*/analysis.md`. Consider adding domain-specific notes: stop the engine, edit `<workdir>/config.json` → `notes` field, then resume.

### Modifying the Queue

The engine reads `engine/checkpoint.json` on resume. To remove or reorder experiments, stop the engine and edit the `queued` list in `engine/checkpoint.json` only if manual queue surgery is required. Remove specs or change their `priority` values; lower priority values run first. `engine/queue.json` is only a monitoring view and is regenerated by the workflow.

### Log Files

- `<workdir>/logs/autoresearch.log` — full execution log
- `<workdir>/logs/*_prompt.md` — prompts sent to the coding agent
- `<workdir>/logs/*_output.md` — Coding-agent responses
- `<workdir>/logs/*_raw.json` — raw JSON including cost and duration
