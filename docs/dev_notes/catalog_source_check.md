# Checking catalog entries against their sources

The SPDL optimization catalog (`tools/skills/optimization/`) is written for
people and coding agents who will not open the original material. Each entry
condenses a pull request, benchmark, issue, report or post into a Trigger, a
Technique and Guards. That condensing can drift from what the source showed.
This note describes the sanity check we use to catch that drift, why we chose
it, what the first full run found, and how to run it for a new or edited
entry.

## Why a source check instead of fixture tests

An earlier pilot gave coding agents small mock repositories and asked whether
they recognized an entry's trigger. It worked, but every fixture had to be
written and maintained, and there was no way to be sure the agent under test
had no prior knowledge of the case. The source check asks a simpler question
that needs no fixtures: given only the entry and the sources it cites, would
an independent reviewer agree that the entry says what the sources show?

## The check

1. **Collect the entry and its sources.** Take the entry block (from its
   ``**`id`: title**`` line through its last field) and its file path, plus
   every source the entry cites as evidence: a `Reference:` link, or the
   pull requests, issues, benchmark write-ups or papers you relied on.
2. **Snapshot each source as a local file.** Write one file per source under
   `sources/`, starting with `SOURCE: <url>` and a blank line, then the
   source text as fetched. Use the same fetch method for every source, and do
   not edit snapshots by hand; re-fetch instead. If a reviewer could only
   confirm a claim by reading code that is not in the snapshot, reword or
   drop the claim.
3. **Build one isolated workspace per entry**, containing only `entry.md`
   (first line `File: <repo path>`, then a blank line and the entry block)
   and `sources/`. Leave out the change itself, its rationale, earlier
   verdicts and other reviewers' answers.
4. **Run at least two independent judges** from different model families,
   one fresh, read-only, non-interactive session per entry, using the prompt
   and schema below. Run each judge in its own fresh directory, ideally on a
   machine that holds nothing else, so it cannot read another judge's answer
   or an earlier verdict. Delete the workspaces after each round. For
   example:
   - Claude Code: `claude -p "$(cat prompt.md)" --output-format json
     --allowedTools Read,Glob,Grep,LS --permission-mode default`, run in the
     workspace; the answer is the `result` field.
   - Codex: `codex exec --skip-git-repo-check -s read-only -C <workspace>
     --output-schema schema.json -o <workspace>/answer.json
     "$(cat prompt.md)"`.
5. **Gate and fix.** The target is `faithful` from every judge, and never
   `major_issues` from any. Fix an entry by matching it to its sources:
   describe what was done, or only proposed, or bundled with other changes,
   and keep the regressions, quality results and validation limits the
   sources report. Re-judge every revision in new sessions with newly built
   workspaces.
6. **Report it.** In the change description, list the entries checked, the
   judges, each judge's final verdict, the number of sources and any recorded
   disputes.

Rules for fixing:

- Do not delete useful guidance just to pass. Generic advice the sources did
  not test may stay if the entry labels it as untested.
- Never add text addressed to reviewers, or claims such as "verified", to an
  entry.
- If a judge misreads a source, make the wording unambiguous instead of
  distorting the entry, and record the dispute. When judges disagree in
  opposite directions, read the source and follow what it says.
- If the cited source does not contain a claim, find the source that does and
  cite it; if none exists, drop the claim or the entry.

## What the first full run found

The first run judged every catalog entry at the time (261 entries citing
about 550 sources) with Claude Code and Codex:

| | Claude Code | Codex |
|---|---|---|
| faithful | 15 | 52 |
| minor issues | 221 | 174 |
| major issues | 25 | 34 |
| unverifiable | 0 | 1 |

The judges agreed on the overall verdict for 195 entries. Sixteen entries
were rated major by both and 27 by one. Most minor issues were a guard that
left out a caveat from the source, which is expected for condensed entries,
so those were not acted on wholesale. The recurring major problems were:

- **The wrong source.** Several negative results cited a broad report that
  did not contain their experiments. The real evidence existed but was not
  linked to the entry.
- **Another system's mechanism presented as an SPDL technique**, or a
  different scheduler, client or service named than the one the source used.
- **Outcomes reversed or misclassified**: an adopted trade-off described as a
  failure, an adopted fix as a rejected idea, or a decision not to test as a
  measured regression.
- **A bundle of changes credited to one mechanism**, and **missing
  regressions or quality results**.
- **Invented specifics**, such as configuration recipes, diagnostics or API
  claims that no source contains.

After fixing, each of the 44 revised entries was rated `faithful` by every
judge on its exact final text. One entry moved to a different technique file,
where its corrected description belonged.

Two lessons shaped the procedure above. First, a revision that passed an
early re-check was rated major again once its sources were re-fetched
independently instead of reusing the fixer's copies, so every round now
fetches its own sources. Second, judges differ in strictness, so using
several judges and requiring all of them reduces noise from any single one.

## Limits

- The check covers faithfulness to sources only. It does not test whether an
  agent recognizes the trigger in a real repository, or whether the technique
  still works on current code.
- Judges are models, not human reviewers. Their verdicts are evidence for a
  reviewer, not a substitute for one.
- Sources that cannot be fetched, or claims visible only in code that is not
  in a snapshot, cannot be judged.

## Judge prompt

```text
You are auditing one entry of an optimization knowledge base against the sources it cites.

Read `entry.md` in the current directory and every file under `sources/`. Each source file starts with `SOURCE: <url>`; a file containing `FETCH_FAILED` could not be retrieved. Judge only from these files. Do not use outside knowledge about the cited cases; if the sources do not cover a point, say so rather than guessing.

Positive entries have a Trigger (when it applies), a Technique (what to change), and Guards (correctness, resource and quality caveats). Counterexamples have a Trigger, an Attempt and its Outcome, or a Not tested field when the proposed change was not run at the stated scope, and Guards and when to revisit. Judge an Attempt or Not tested field under `technique`; include the accuracy of any Outcome in that assessment. Entries may also include a Reference or Source line. Entries are written for readers who will not open the sources: they add generic explanation (definitions, why a mechanism costs time, small illustrative examples) and deliberately omit workload-specific gain numbers. Do not penalize correct generic explanation or omitted historical numbers. Several entries may cite the same broad source; judge whether the part relevant to this entry supports it.

Check:
1. Trigger: does at least one source describe this situation or symptom? Is the trigger broader or narrower than the sources support?
2. Technique: does it match what the sources actually did or proposed (the mechanism, not the wording)? Note if the sources' change was different, bundled with other changes, or only proposed.
3. Guards: are important caveats from the sources missing (failures, regressions, quality problems, prerequisites, partial validation)? Does any guard contradict a source?
4. Overreach: claims of benefit, generality or validation stronger than the sources support.
5. Placement: the entry's file path is given at the top of `entry.md`. Entries in the shared catalog are meant to be usable without organization-internal infrastructure; entries in an organization's private extension directory may depend on it. Say if the placement looks wrong.

Verdict scale for trigger/technique/guards: `supported` (sources back it), `partial` (some support, but broader, narrower or missing context), `unsupported` (sources do not address it), `contradicted` (sources say otherwise), `unverifiable` (sources failed to load or are unrelated).

`overall`: `faithful` (no material issue), `minor_issues` (wording or scope tweaks), `major_issues` (a reader would apply it wrongly, or it misstates what the sources show), `unverifiable` (sources insufficient to judge).

Respond with only a JSON object matching this shape (no prose, no code fence):

{"entry_id": "...", "sources_used": ["<url>", ...], "trigger": {"verdict": "...", "notes": "..."}, "technique": {"verdict": "...", "notes": "..."}, "guards": {"verdict": "...", "missing_caveats": ["..."], "notes": "..."}, "overreach": ["..."], "placement": {"verdict": "ok|should_be_private|should_be_shared|unclear", "notes": "..."}, "overall": "...", "summary": "<two sentences at most>"}
```

## Answer schema

```json
{
  "type": "object",
  "additionalProperties": false,
  "required": ["entry_id", "sources_used", "trigger", "technique", "guards", "overreach", "placement", "overall", "summary"],
  "properties": {
    "entry_id": {"type": "string"},
    "sources_used": {"type": "array", "items": {"type": "string"}},
    "trigger": {"type": "object", "additionalProperties": false, "required": ["verdict", "notes"], "properties": {"verdict": {"type": "string", "enum": ["supported", "partial", "unsupported", "contradicted", "unverifiable"]}, "notes": {"type": "string"}}},
    "technique": {"type": "object", "additionalProperties": false, "required": ["verdict", "notes"], "properties": {"verdict": {"type": "string", "enum": ["supported", "partial", "unsupported", "contradicted", "unverifiable"]}, "notes": {"type": "string"}}},
    "guards": {"type": "object", "additionalProperties": false, "required": ["verdict", "missing_caveats", "notes"], "properties": {"verdict": {"type": "string", "enum": ["supported", "partial", "unsupported", "contradicted", "unverifiable"]}, "missing_caveats": {"type": "array", "items": {"type": "string"}}, "notes": {"type": "string"}}},
    "overreach": {"type": "array", "items": {"type": "string"}},
    "placement": {"type": "object", "additionalProperties": false, "required": ["verdict", "notes"], "properties": {"verdict": {"type": "string", "enum": ["ok", "should_be_private", "should_be_shared", "unclear"]}, "notes": {"type": "string"}}},
    "overall": {"type": "string", "enum": ["faithful", "minor_issues", "major_issues", "unverifiable"]},
    "summary": {"type": "string"}
  }
}
```
