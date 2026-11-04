# Approach Ownership and Job Orchestration

Read [workflow.md](workflow.md) for phase/evidence/acceptance gates. This file owns delegation, not optimization recipes.

## Main agent

- Give each optimization approach a **dedicated persistent subagent and isolated worktree/checkout** based on the exact validated top. Separate branches in one writable tree are not isolation; use separate nodes/checkouts when required. Bound concurrency by resources, not CPU count.
- Send a focused brief: phase, approach/hypothesis, parent/source, evidence or explicit architecture-exploration classification, allowed files, measurement/quality gates and relevant references. Do not copy the main thread's entire troubleshooting history.
- Focus on job launch/monitoring, return identities/provenance, post-analysis assignments, next-step planning, record maintenance and validated-stack integration. Children own detailed implementation/local checks/build preparation and analysis; do not duplicate that work.
- For each production attempt, issue a **post-analysis request to its owning subagent** with job/version/attempt, source/package/configuration, actual resources, measured windows and log/SPDL-stat/trace links. Reuse that same child for follow-up diagnoses/repairs so its context stays intact.
- Consume concise evidence-linked verdicts, not full logs/source digressions. Pending results remain pending; never invent a child's measurements or attribution.
- Integrate accepted commits deliberately and provide the new validated top to later approaches. Rebase old-parent candidates and revalidate the combination, rather than assuming gains add. Do not retroactively alter running-job configuration or source receipts.
- Apply shared campaign index/one-pager/inventory/decision updates centrally from compact child drafts; do not let concurrent document writers overwrite each other. Preserve failed/rejected evidence.

## Approach subagent

- Investigate and implement only its assigned approach in its isolated tree. Reuse baseline functions/classes and preserve processing/data semantics except for explicitly recorded hypotheses. Do not write the main checkout or another approach's tree.
- For non-architectural optimization, identify the bottleneck using [analysis.md](analysis.md), then load only the matching remedy topic. Architecture exploration is the empirical exception in workflow; still analyze its actual result.
- Perform repository-required checks, matched local data/CPU/throughput tests and exact-source build preparation. Startup failures get an attributed repair without silently switching the approach or swallowing errors.
- Analyze each assigned production attempt using SPDL queue/stat evidence and model traces/timings as applicable. Separate holders from named victims, main process from workers, startup from recurring overhead; unavailable attribution stays unresolved.
- Return a compact handoff: **approach; exact parent/commit; decisive speed/CPU/memory result and scope; post-analysis cause or uncertainty; accuracy/reliability/state; next action; commit/job/analysis links**. Put detailed logs/traces/tables behind links.
- Return proposed shared-record updates for the main agent to apply. Do not change production defaults, land, install persistent engines or bypass security/resource approvals.

## Follow-up brief

Each follow-up should name either the measured bottleneck and its proposed remedy, an explicit architecture comparison, a measurement to resolve missing evidence, or a workflow-defined hygiene candidate. The response must explain why that classification applies. A fixed pre-existing technique checklist is not a valid planning brief.
