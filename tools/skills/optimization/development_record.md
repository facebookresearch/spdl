# Persistent SPDL Development Record

Keep one durable campaign record and update it throughout migration and optimization. Reuse the task's existing record when one exists. Link artifacts and exact evidence instead of pasting large logs. Preserve failed/rejected experiments and superseded decisions. Each result must state when it was measured/verified, its scope and whether it is provisional.

## Preferred Google Docs tab layout

Use the following structure for a new campaign record or an authorized reorganization of an existing record:

- **Tab 1 — Index:** stable links to every phase record, one-pager, supplemental note and external evidence. Keep this tab navigational; link the summaries rather than embedding long reports.
- **Tab 2 — Phase 1 record:** minimal migration, data/resource/performance parity, local comparisons and production-launch analyses.
- **Tab 3 — Phase 1 one-pager summary:** objective/control, parity results, key decisions, blockers and the Phase 2 handoff.
- **Tab 4 — Phase 2 record:** architecture and analysis-guided attempts, GIL/event-loop attribution, hygiene decisions, model-performance work once model-bound behavior is established, accuracy results and the validated stack.
- **Tab 5 — Phase 2 one-pager summary:** accepted architecture/stack, speed/resource/quality outcome, rejected directions, blockers and the Phase 3 handoff.
- **Tab 6 — Phase 3 record:** finalized-architecture parameter sweeps, per-launch analysis, comparisons and final-setting decisions.
- **Tab 7 — Phase 3 one-pager summary:** accepted settings, cumulative results/quality, reproducible recipe and remaining work.
- **Other tabs — Supplemental notes:** focused investigations such as inefficiencies in the original implementation, actual GIL holders, asyncio-event-loop blocking, source/cache/IPC details or other evidence. Index every note in Tab 1 and link it from the relevant phase entry.

Each phase record contains its concise discoveries, discussions/decisions, attempt inventory and run/analysis links. Keep the original loader/model reference and measurement protocol linked consistently across phases; do not duplicate long baseline tables or troubleshooting histories in every tab. Preserve completed-phase summaries as phase outcomes while updating the active phase's one-pager.

Reuse the task's existing live record when available. Do not reorder or overwrite unrelated tabs/comments merely to impose this preferred template without an explicit reorganization task. For Markdown or another document system, use the same index-first sequence of phase-record/one-pager sections plus linked supplemental notes. Do not create an unindexed document for every iteration.

## Per-phase one-pager summaries — Tabs 3, 5 and 7

- Objective and owning pipeline.
- Phase: 1 migration / 2 architecture / 3 sweep.
- Original loader/model baseline and accepted minimal SPDL control.
- Current validated architecture/commit stack, its latest validated top, and provisional candidates, with speed, CPU/memory and accuracy status clearly separated. New experiments branch from this validated top; speed uses the immediate parent and quality uses the original model baseline.
- Main discoveries, decisions, blockers and next actions.
- Measurement scopes and references; missing data remains `unavailable` / `pending`, not zero or success.

Keep each summary scannable as a **one-page executive summary**, not another attempt log. Refresh the active phase's summary after each material result/decision, link its supporting phase record/notes, and preserve completed-phase outcomes. Do not turn a speed-only screen into an accepted model-quality claim.

## Discoveries

For each discovery record:

- Date/verified time, observation and affected code revision/job/attempt.
- Data/source, cache, resource and measurement window.
- Hypothesis versus confirmed cause; confidence and unresolved alternatives.
- Process/thread/task scope. For GIL/event-loop issues distinguish actual holder from victim, trainer from isolated workers, and startup from recurring work.
- Evidence links (stacks, queue/stage telemetry, local reproduction, matched comparison).
- Smallest repair or experiment suggested, and the outcome/next question.

## Discussions and decisions

Record question, alternatives, evidence, discussion outcome, chosen direction, rationale, consequences and revisit condition. Link the implementing change/experiment. Keep a superseded decision visible with its replacement; do not overwrite history so it appears that the outcome was known in advance.

## Optimization inventory

Use one inventory with these fields:

- ID, phase, hypothesis/approach, dedicated subagent owner, isolated worktree/checkout, component/architecture and exact immediate parent.
- State: `todo`, `ongoing`, `speed_validated`, `accuracy_pending`, `validated`, `rejected` or `blocked`.
- Source/diff/configuration and any resource/topology/runtime/semantic changes.
- Local data checks and paired CPU-utilization/throughput/memory evidence.
- Production run/attempt and required SPDL queue/stage analysis.
- Speed comparison, original-model accuracy comparison, uncertainty and missing evidence.
- Decision (keep/revert/continue), reason, next action and documentation links.

Only `validated` means the required correctness, accuracy and reliability gates pass together with a repeatable speed win **or explicitly verified performance-neutral hygiene**. Record which acceptance reason applies; neutral hygiene is not a speed-gain claim. Mark unfinished quality checks explicitly even when an implementation is fast.

## Attempt entries: concise, with linked evidence

Each attempt gets a short entry, not a narrative report:

- **What:** one sentence describing the approach/change.
- **Post-analysis:** one sentence stating the measured bottleneck/reason it was slow or fast, or that attribution is inconclusive; include only the decisive result/scope.
- **Links:** exact commit/diff, job/attempt and detailed analysis/accuracy evidence.

Store shared baseline/configuration/measurement protocol once and link it. Put stacks, logs, full metric tables, chronological troubleshooting and long rationales in linked evidence, not the attempt inventory. Keep the executive summary current and the inventory scannable.

## Run ledger — every production launch

Record the launch before treating it as an experiment result: returned job/attempt identity, exact source/package, effective configuration and overrides, baseline/parent, hardware/resources, phase, data/seed/batch/schedule semantics and intended measurement windows.

Attach an analysis to every launch when telemetry appears and finalize it on completion:

- Stage/source/sink QPS, service time, queue put/get waits, occupancy, failures and worst-rank evidence.
- Minimum sink occupancy across all ranks and low/empty-window fractions.
- Realized end-to-end throughput/complete step time, CPU and memory from matched intervals/scopes.
- Cold/startup versus warm/E2E results, coverage and comparison caveats.
- Hypothesis result, discovered bottleneck, quality status and keep/reject/next decision.

Failed, preempted, retried and stopped attempts remain in the ledger; do not silently merge their windows or count partial epochs as complete. Missing queue data is a gap to resolve, not permission to skip analysis.

## Update cadence and handoff

The main agent maintains the shared index, phase one-pagers, decisions and inventory. Approach subagents keep detailed implementation/post-analysis context in isolated worktrees and return concise evidence-linked entry drafts; do not have concurrent children overwrite the shared campaign document.

Update the record after discoveries, discussions, decisions, implementations, local tests, launches, analyses, accuracy results and sweeps. Keep chronological history plus the current inventory/summary/index. The final handoff must identify the accepted recipe, exact baseline and sources, validated gains/resource costs, model-quality evidence, rejected alternatives and remaining work.
