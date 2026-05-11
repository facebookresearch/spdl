# Autoresearch: Automated SPDL Pipeline Optimization

Automated experiment engine for optimizing SPDL data loading pipeline performance in AI training jobs. Uses a coding agent to analyze metrics, identify bottlenecks, propose parameter and code changes, and iterate to minimize steady-state step time.

## How It Works

```
┌──────────────────────────────────────────────────────────────────┐
│  run.py (single entry point)                                     │
│                                                                  │
│  1. Initialize workdir + instrument pipeline (first run only)    │
│  2. Seed initial experiments: baseline, headspace, MTP           │
│  3. Start async engine ───────────────────────────────────────┐  │
│                                                               │  │
│     ┌─────────────────────────────────────────────────────┐   │  │
│     │  Engine event loop (asyncio)                        │   │  │
│     │                                                     │   │  │
│     │  ┌─ Fill slots from priority queue (up to N)        │   │  │
│     │  │                                                  │   │  │
│     │  ├─ prepare (code change + build, serialized)       │   │  │
│     │  ├─ launch job                                      │   │  │
│     │  ├─ poll until complete or timeout ──┐              │   │  │
│     │  │                                   │ concurrent   │   │  │
│     │  │  asyncio.wait(FIRST_COMPLETED) ◄──┘              │   │  │
│     │  │                                                  │   │  │
│     │  ├─ analyze (agent) ──► update plots + summary      │   │  │
│     │  ├─ plan follow-ups (agent) ──► enqueue children    │   │  │
│     │  ├─ re-prioritize queue                             │   │  │
│     │  └─ loop ◄──────────────────────────────────────────┘   │  │
│     │                                                         │  │
│     │  Stop when: plateau reached + all best practices tried  │  │
│     │  Ctrl+C: persist state, resume with same command        │  │
│     └─────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

Each coding-agent invocation is **stateless**. The engine carries all state and assembles self-contained prompts from:
- `prompts/knowledge.md` — compiled SPDL optimization knowledge
- Experiment state — master table, history, previous analyses
- Fetched metrics — system metrics, pipeline performance stats

## Design Choices

Autoresearch is built around a small runner and explicit persisted state. The runner only schedules coroutine work and checkpoints what is queued or running; the workflow owns experiment semantics such as applying code changes, launching jobs, analyzing results, and proposing follow-ups.

This keeps runs inspectable and resumable. Coding-agent calls are stateless, deterministic workflow decisions are separated into testable helpers, and `engine/checkpoint.json` is the source of truth for recovery after interruption.

The implementation is unusual in one important way: platform behavior is grouped behind explicit capability objects instead of injected as a flat callback bag. `AutoresearchPlatform` owns `workspace`, `artifacts`, `execution`, `evidence`, and `agent`, so the same workflow can run locally, through a remote executor, or with different coding agents without changing the runner.

## Architecture

```
Claude Code (interactive front-end via launch.sh)
  │ gathers config from user, starts engine, monitors progress
  ▼
run.py ──► runner.py (generic async scheduler)
  │              │
  │  adapter     │  manages: priority queue, concurrent coroutine slots,
  │              │  checkpointing, and cancellation persistence
  ▼              │
workflow/      ▼
adapter.py  asyncio.wait(FIRST_COMPLETED)
  │             → immediate analysis on completion
  │             → follow-up experiments returned as child WorkSpecs
  ├─ workflow/store.py       durable state + monitoring views
  ├─ workflow/policy.py      deterministic, testable decisions
  ├─ workflow/*_ops.py       source/build/analysis/planning operations
  └─ platform/               workspace/artifacts/execution/evidence/agent
```

**Two ways to run:**

1. **Claude Code wrapper** (`launch.sh`) — interactive. Claude Code gathers config from the user, starts the engine, monitors progress, and intervenes on errors.
2. **Direct** (`run.py`) — non-interactive. Pass all config as CLI flags.

### Engine Design

The runner (`utils/runner.py`) is generic and domain-agnostic. It knows nothing about SPDL, coding agents, source control, local subprocesses, training jobs, metrics, or hypothesis planning. It only runs serializable `WorkSpec` objects as coroutines, checkpoints queued/running work, and persists an `interrupted` checkpoint when cancellation reaches the process.

Autoresearch behavior belongs in the workflow side of the boundary:

- `utils/workflow/adapter.py` orchestrates each experiment lifecycle: prepare source, build, launch, poll, analyze, update state, and return child `WorkSpec` objects.
- `utils/workflow/store.py` owns durable workdir files such as `engine/checkpoint.json`, `tree.json`, per-node status, and monitoring views.
- `utils/workflow/policy.py` owns deterministic decisions such as status normalization, duplicate filtering, planning gates, and stall policy. Keep predictable logic here so it can be unit-tested without a coding agent or infrastructure.
- `utils/platform/` owns swappable capabilities for workspace, artifacts, execution, evidence collection, and coding-agent calls.

Each experiment node runs as an independent `asyncio.Task` through prepare → launch → poll → analyze. The runner uses `asyncio.wait(return_when=FIRST_COMPLETED)` so whichever job finishes first is immediately analyzed and its follow-ups are enqueued.

Key features:
- **Configurable concurrency** — up to N jobs run simultaneously
- **Tree-structured hypotheses** — experiments branch from parent commits, enabling exploration of multiple optimization paths
- **Wall-clock stuck detection** — jobs exceeding `job_timeout_s` are killed immediately (no stall-counting delay)
- **Serialized state mutation** — workflow state updates are guarded so concurrent completions do not corrupt shared state
- **Graceful shutdown** — SIGINT persists queued and running specs to disk; re-running resumes from where it left off

### Engine Disk Layout

All state is persisted to `{workdir}/engine/` for crash recovery and external observability:

```
engine/
├── checkpoint.json        # Serialized queued/running WorkSpecs; resume source of truth
├── engine_state.json     # {status, timestamp, node counts}
├── tree.json             # Full hypothesis tree
├── queue.json            # Priority-ordered pending node IDs (monitoring view)
├── active.json           # Currently running jobs (monitoring view)
└── nodes/                # Per-node directories
    ├── 000_baseline/
    │   ├── spec.json     # Experiment specification
    │   ├── status.txt    # queued|preparing|running|analyzing|completed|failed
    │   └── result.json   # Analysis result (after completion)
    └── ...
```

### Live Progress

After each completed experiment, the engine updates:
- `summary.md` — current metrics, recent results
- `progress.png` — Karpathy-style progress chart (duration, step time, SM util)
- `hypothesis_tree.png` — tree visualization showing how experiments branch and evolve

### Stop and Resume

Press **Ctrl+C** to stop the engine gracefully. The runner persists queued and running `WorkSpec` objects to `engine/checkpoint.json` before exiting, and the workflow keeps the monitoring views under `engine/` up to date. Running jobs on the cluster continue independently.

Re-run the same command to resume — the engine re-checks job status and picks up where it left off.

## Quick Start

### Option 1: Claude Code (interactive)

```bash
./launch.sh
```

Claude Code asks for your pipeline script, build command, launch command, etc., then starts and monitors the engine.

For a pre-configured use case:
```bash
./launch.sh "Run autoresearch on pipeline at /path/to/pipeline.py ..."
```

### Option 2: Direct CLI

```bash
# First run — provide all config
python run.py /tmp/my_experiment \
  --pipeline-script path/to/pipeline.py \
  --source-dir path/to/source \
  --build-command "docker build -t my_image ." \
  --base-launch-command "torchx run -s local entry.py:main --image \$IMAGE --num_workers 8" \
  --notes "Description of the experiment" \
  --dangerously-skip-permissions \
  --max-iterations 10 \
  --patience 3 \
  --max-concurrency 3 \
  --job-timeout 1800

# Resume after Ctrl+C (config is persisted)
python run.py /tmp/my_experiment

# Check progress
python cmd.py status /tmp/my_experiment

# Generate final report
python cmd.py report /tmp/my_experiment
```

### CLI Options

| Flag | Description | Default |
|---|---|---|
| `--pipeline-script` | Pipeline source file to optimize | — |
| `--source-dir` | Source directory for in-place code modifications | — |
| `--build-command` | Command to build the job image | — |
| `--base-launch-command` | Job launch command; use `$IMAGE` as placeholder | — |
| `--notes` | Free-form context included in agent prompts | — |
| `--max-iterations` | Maximum planning sessions (each produces 2-3 experiments) | 10 |
| `--patience` | Stop after N non-improving planning sessions | 3 |
| `--max-concurrency` | Maximum concurrent training jobs | 3 |
| `--job-timeout` | Wall-clock timeout per job in seconds | 1800 |
| `--poll-interval` | Seconds between job status polls | 120 |
| `--platform` | Execution platform: `auto`, `remote`, or `local` | auto |
| `--agent` | Coding agent: `claude` or `codex` | claude |
| `--local-execution-mode` | Local launch mode: `full`, `dataloader_only`, or `dry_run` | full |
| `--dangerously-skip-permissions` | Skip Claude permission prompts | off |
| `--skip-instrument` | Skip TTFB/step-time auto-instrumentation | off |

## What Gets Optimized

The primary objective is to **minimize steady-state step time** — the median step time after discarding warmup steps. This directly predicts production training throughput.

### Metric Hierarchy

1. **Steady-state step time** (`steady_step_time_ms`) — most reliable. Median step time after warmup (first ~20 steps).
2. **Steady-state SM utilization** (`steady_sm_utilization_pct`) — proxy when step time is unavailable. Uses p50/p75 from system metrics (not the raw average, which is diluted by initialization).
3. **Raw job duration** — least reliable for short experiments due to variable init times.

Raw average SM utilization is **not used** for optimization — it's heavily affected by the zero-utilization initialization phase in short experiments.

### Tracked Metrics

- **Step time** — duration of one training iteration
- **SM Utilization** — GPU streaming multiprocessor activity (steady-state p50/p75)
- **TTFB** — time to first batch
- **Data readiness** — fraction of time the sink queue has data available
- **CPU utilization** — must stay ≤40% to avoid noisy-neighbor effects
- **Job duration** — total wall clock time

Typical optimizations include MTP (subprocess pipeline), concurrency tuning, batch size tuning, `torch.compile`, stage splitting, and GC management.

### Stopping Criteria

The engine stops when **both** conditions are met:

1. **All best practices tried** — baseline, headspace (CacheDataLoader), MTP (subprocess pipeline), batch size tuning, and concurrency tuning must each be attempted.

2. **Plateau detected** — steady-state metrics have not improved for `--patience` consecutive planning sessions.

An "iteration" counts each time the coding agent is asked to plan, not each completed job. With `--max-iterations 40`, the engine runs up to 40 planning sessions, each producing 2-3 experiments.

## Source Control Integration

When `--source-dir` is provided, autoresearch integrates with source control (`sl` or `git`, auto-detected):

- **Anchor commit** recorded at init — the system never reverts beyond this point
- **Code changes are committed** automatically with descriptive messages prefixed `[autoresearch]`
- **Tree-structured commits** — experiments branch from parent commits, enabling multiple optimization paths. When the coding agent specifies `revert_to`, the engine goes to that commit before applying new changes.
- **Instrumentation commit** — TTFB/step-time logging is committed separately during init

## Workdir Structure

```
workdir/
├── config.json              # Experiment configuration
├── state.json               # Engine state (iteration, history, best, plateau)
├── master_table.tsv         # Summary of all runs with metrics
├── summary.md               # Live progress summary (updated after each job)
├── progress.png             # Karpathy-style progress chart (updated live)
├── hypothesis_tree.png      # Tree visualization (updated live)
├── engine/                  # Runner checkpoint + workflow monitoring files
│   ├── checkpoint.json      # Resume source of truth
│   ├── engine_state.json    # Engine status and counts
│   ├── tree.json            # Full hypothesis tree
│   ├── queue.json           # Pending experiments (monitoring view)
│   ├── active.json          # Running jobs (monitoring view)
│   └── nodes/               # Per-node spec/status/result files
├── runs/
│   ├── 000_baseline/
│   │   ├── meta.json           # Experiment specification
│   │   ├── job_id.txt          # Cluster job ID
│   │   ├── system_metrics.txt  # GPU/CPU metrics
│   │   ├── metrics/            # Pipeline performance stats (TSV files)
│   │   └── analysis.md         # Coding-agent analysis
│   └── ...
├── logs/
│   ├── autoresearch.log        # Full execution log
│   ├── *_prompt.md             # Prompts sent to the coding agent
│   ├── *_output.md             # Coding-agent responses
│   └── *_raw.json              # Raw JSON (cost, duration, usage)
└── report.md                   # Final report (from cmd.py report)
```

## Agent Decision Framework

In each planning session, the coding agent chooses one of three actions:

| Action | When | What Happens |
|---|---|---|
| `launch` | Parameter or code changes to test | Engine applies changes, builds, launches, monitors, analyzes |
| `manual` | Change requires human judgment | Engine pauses; user intervenes |
| `stop` | Metrics plateaued AND all best practices tried | Engine finishes |

## Project Structure

```
autoresearch/
├── run.py                         # Single entry point (init + instrument + engine)
├── cmd.py                         # CLI tools (init, assess, status, report)
├── launch.sh                      # Interactive coding-agent front-end
├── plot_progress.py               # Progress chart + hypothesis tree visualization
├── BUCK                           # Build targets
├── utils/
│   ├── runner.py                  # Generic async WorkSpec scheduler
│   ├── workflow/                  # Experiment lifecycle, store, and policy
│   ├── platform/                  # Workspace/artifacts/execution/evidence/agent
│   ├── commands/                  # init, assess, status, report commands
│   ├── infra/                     # Low-level generic helper functions
│   ├── claude.py                  # Claude implementation for CodingAgent
│   ├── state.py                   # Experiment state, config, master table
│   ├── log.py                     # Logging setup
├── prompts/
│   ├── launch.md                  # Interactive launch prompt
│   ├── knowledge.md               # SPDL optimization knowledge
│   ├── how_to_interpret_pipeline_stats.md  # Pipeline metrics guide
│   ├── instrument.md              # Auto-instrumentation prompt
│   ├── headspace.md               # CacheDataLoader headspace prompt
│   ├── assess.md                  # Baseline assessment prompt
│   ├── plan_next.md               # Planning prompt
│   ├── analyze.md                 # Post-job analysis prompt
│   ├── apply_changes.md           # Code modification prompt
│   └── fb/                        # Infrastructure-specific prompts
│       ├── knowledge.md
│       └── launch.md              # Platform-specific launch instructions
└── README.md
```

### Headspace Analysis

The engine automatically launches a **headspace job** that wraps the data pipeline with `spdl.dataloader.CacheDataLoader`. This measures the upper bound of improvement from data loading optimization. The headspace percentage tells you how much of the training step is data loading overhead — if headspace is low, further pipeline optimization won't help much.

Headspace results are excluded from the running best and plateau detection.

## Customization

- **`config.json`**: edit agent-specific flags to pass additional CLI options
- **`prompts/`**: edit templates to adjust agent behavior or add domain-specific knowledge
- **`prompts/knowledge.md`**: add project-specific optimization context
