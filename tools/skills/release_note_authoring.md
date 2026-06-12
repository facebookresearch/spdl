# SPDL Release Notes Generator

Generate release notes for a new SPDL release, given either a list of GitHub PRs merged since the last release or a baseline ref (tag/commit) to diff against `HEAD`.

## Input

Either of:

- **List of PRs**: GitHub PR URLs or numbers from `facebookresearch/spdl`, or
- **A baseline ref**: the previous release's tag or commit (the new version is
  everything between that ref and `HEAD`). This is the common case when working
  in a local checkout without internet access.

## Workflow

### Step 1 — Gather Change Data

**Preferred when no internet access is available (use git history):** If you were
given a baseline ref, derive the change set directly from the local repository —
do not rely on the GitHub API:

```bash
# All commits since the baseline
git log --pretty=format:"%h %s" <baseline_ref>..HEAD
# Per-PR detail: title/body carry the "#NNNN" and "Pull Request resolved:" trailer
git show -s --format="%s%n%b" <commit>
# What actually changed (focus on src/ for public-API analysis)
git diff --stat <baseline_ref>..HEAD -- src/
git diff <baseline_ref>..HEAD -- <path>
```

Recover PR numbers from the `#NNNN` suffix in commit subjects and the
`Pull Request resolved: https://github.com/.../pull/NNNN` trailer in commit
bodies. Note in the final **Notes** section that links were reconstructed from
git rather than verified against GitHub.

**When PR numbers are given and the network is reachable:** Create and run a temporary bash script to fetch PR metadata and diffs from the GitHub API.

The environment may not have direct access to `api.github.com`. Use the proxy:

```bash
curl $(fwdproxy-config curl) -s "https://api.github.com/repos/facebookresearch/spdl/pulls/<PR_NUMBER>" \
    -H "Accept: application/vnd.github.v3+json"
```

For each PR, fetch:
- **Metadata**: `https://api.github.com/repos/facebookresearch/spdl/pulls/<N>` — contains `title`, `body`, `labels`
- **Changed files**: `https://api.github.com/repos/facebookresearch/spdl/pulls/<N>/files` — contains `filename`, `patch`
- **Full diff**: `https://github.com/facebookresearch/spdl/pull/<N>.diff`

Save outputs to a temporary directory (e.g., `./spdl_prs/`).

If the proxy also fails (403 from fwdproxy), generate a standalone script for the user to run manually:

```bash
#!/bin/bash
OUTDIR="${1:-./spdl_prs}"
REPO="facebookresearch/spdl"
PRS=(<space-separated PR numbers>)
mkdir -p "$OUTDIR"
for pr in "${PRS[@]}"; do
    echo "Fetching PR #${pr}..."
    curl $(fwdproxy-config curl) -s "https://api.github.com/repos/${REPO}/pulls/${pr}" \
        -H "Accept: application/vnd.github.v3+json" \
        -o "${OUTDIR}/${pr}_meta.json"
    curl $(fwdproxy-config curl) -s "https://api.github.com/repos/${REPO}/pulls/${pr}/files" \
        -H "Accept: application/vnd.github.v3+json" \
        -o "${OUTDIR}/${pr}_files.json"
    curl $(fwdproxy-config curl) -s -L "https://github.com/${REPO}/pull/${pr}.diff" \
        -o "${OUTDIR}/${pr}.diff"
done
# Print summary
for pr in "${PRS[@]}"; do
    title=$(python3 -c "import json; d=json.load(open('${OUTDIR}/${pr}_meta.json')); print(d.get('title',''))" 2>/dev/null || echo "(failed)")
    echo "PR #${pr}: ${title}"
done
```

Ask the user to run it and resume once files are available.

### Step 2 — Parse PR Metadata

For each PR, extract from `{pr}_meta.json`:
- `title` — PR title
- `body` — PR description/summary
- `user.login` — author

Parse the title for `[BC-Breaking]` or `[BC-breaking]` tags.

### Step 3 — Classify PRs

Classify each PR into one of these categories:

1. **BC-Breaking Changes** — PRs with `[BC-Breaking]` in the title, plus any PR where the diff analysis (Step 4) reveals BC-breaking changes
2. **New Features** — PRs adding new public APIs, functionality, or capabilities
3. **Bug Fixes** — PRs fixing incorrect behavior
4. **Other Changes** — Minor features, refactoring, internal improvements
5. **Documentation** — Docstring fixes, doc updates
6. **Dropped** — CI-only changes, typo fixes, token updates (not included in release notes)

### Step 4 — Check for Unlabeled BC-Breaking Changes

For every PR NOT already tagged `[BC-Breaking]`, examine the diff to check for BC-breaking changes. A change is BC-breaking if it modifies the **public API** in an incompatible way.

#### What constitutes public API

**Python (`src/spdl/`):**
- Functions, classes, and constants listed in `__init__.py` `__all__` exports
- Method signatures of public classes (adding required parameters, changing return types, removing methods)
- Removing symbols from `__all__`

**C++ (`src/libspdl/`):**
- Functions and classes declared in non-`detail` headers (e.g., `src/libspdl/core/decoder.h`)
- Changes to function signatures, removed functions, changed return types

#### What is NOT BC-breaking

- Changes to internal/private APIs (prefixed with `_`, inside `detail/` directories)
- Adding new optional parameters with defaults
- Adding new public functions/classes (purely additive)
- Changes to tests, examples, CI, docs only

#### Common patterns to watch for

- Removing a symbol from `__init__.py` `__all__` (e.g., removing `ImageDecoder`)
- Changing method signatures of public abstract classes (e.g., `TaskHook.task_hook()`)
- Changing return types (e.g., `Optional[Frames]` → `Iterator[Frames]`)
- Renaming/removing required parameters in public dataclasses
- Removing public C++ functions from non-detail headers

If a BC-breaking change is found without a label, flag it in the release notes and inform the user.

### Step 5 — Generate Release Notes

Produce release notes in this format:

```markdown
## SPDL vX.Y.Z Release Notes

### Highlights

Write 2-4 short paragraphs for the most significant changes (major new features
and BC-breaking changes). Each highlight should have a bold title, a 2-3 sentence
description, and link to the relevant PR(s).

### BC-Breaking Changes

- **Short description** ([#NNN](url)): What changed, what the migration path is.
  Always include a Before/After code snippet for API changes:

  ```python
  # Before (vX.Y-1.Z)
  old_api(...)

  # After (vX.Y.Z)
  new_api(...)
  ```

### New Features

- **Short description** ([#NNN](url)): What the feature enables, followed by a
  short code snippet illustrating typical usage:

  ```python
  from spdl.<module> import NewThing

  NewThing(...)  # minimal, runnable-looking example
  ```

### Bug Fixes

- Description ([#NNN](url))

### Other Changes

- Description ([#NNN](url))

### Documentation

- Description ([#NNN](url))
```

#### Writing guidelines

- **Highlights** should cover BC-breaking changes and major new features — things users need to know about when upgrading. Major new features in Highlights should also carry a short usage snippet.
- **BC-Breaking Changes** must include migration guidance with a Before/After code snippet for every public API change — this is required, not optional.
- **New Features** should briefly explain what the feature enables **and include a code snippet showing typical usage**. Verify the snippet against the real API before including it: confirm the import path against the module's `__init__.py` `__all__`, and the call signature against the function/class definition (or, for C++ bindings, the `.pyi` stub). Do not invent argument names.
- **Bug Fixes** should describe what was wrong and what the fix does
- Group related PRs together (e.g., if PR A adds a feature and PR B refines its API, mention both together)
- Omit CI-only PRs, typo fixes, and token updates entirely
- Use PR title as a starting point but rewrite for clarity — the audience is library users, not contributors
- Link all PRs with `[#NNN](https://github.com/facebookresearch/spdl/pull/NNN)`

### Step 6 — Verify `.. versionadded::` Directives Exist

Adding the `.. versionadded:: X.Y.Z` (and `.. versionchanged::` / `.. deprecated::`)
directives is a **code-authoring-time** responsibility, not a release-note task —
they should already be in the source when the feature lands (see the SPDL
`code-authoring` skill, "Document API changes across versions").

As a final check, confirm every entry in **New Features** and **BC-Breaking
Changes** has the corresponding directive in its source docstring. If any are
missing, flag them in the **Notes** section (Step 7) so the author can add them at
the source — for C++ binding APIs that means the docstring in `src/spdl/io/lib/core/`,
not the autogenerated `.pyi` stub.

### Step 7 — Flag Concerns

After generating the notes, add a **Notes** section calling out:
- Any PRs where BC-breaking was suspected but not labeled
- Any PRs whose categorization is ambiguous
- Related PRs that were grouped together (explain the relationship)
- Any follow-up fix PRs that clean up after a prior PR (e.g., #1292 fixing symbol resolution after #1280 removed `ImageDecoder`)

## Cleanup

If you used the GitHub API path, remind the user to delete the temporary fetch script and `spdl_prs/` directory when done. (Nothing to clean up when working purely from git history.)
