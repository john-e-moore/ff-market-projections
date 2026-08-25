# Repository Agent Guide

## Authority and scope

- Read this file before making changes.
- `SPEC.md` is the implementation contract. `ROADMAP.md` defines phase scope, order, tests, and exit gates.
- Implement only the phase or review task explicitly requested by the user.
- Do not begin a later phase, merge a pull request, or make unrelated improvements without explicit approval.
- Prefer speed and simplicity, but never bypass a validation gate to produce output.

## Model and effort selection

The task's root model is normally selected by the user in Codex Desktop before the task starts. An agent must not claim it switched the root model when it cannot. At the beginning of a phase:

1. Determine the recommended model and effort below.
2. Compare them with the active task configuration when visible.
3. If the active configuration is below the recommendation, pause before editing and tell the user which setting to select in a new task. A stronger configuration may proceed.
4. Do not escalate merely because a task is long. Escalate only after identifying a concrete reasoning or correctness problem.

Interpret Desktop's `Light` setting as the low-cost/low-reasoning tier. Use `Ultra` only as a last resort; it is not a default for any phase.

| Work | Recommended model | Effort |
|---|---|---|
| Phase 0 — Foundation and contracts | Terra | Medium |
| Phase 1 — DAG and fresh collection | Terra | Medium |
| Phase 2 — Historical outcomes and leakage-free baselines | Sol | High |
| Phase 3 — Raw market validation and normalization | Terra | Medium |
| Phase 4 — Player identity reconciliation | Terra | High |
| Phase 5 — Pricing and no-vig conversion | Terra | High |
| Phase 6A — Historical distribution calibration | Sol | XHigh |
| Phase 6B — Kalshi update and mean inversion | Sol | XHigh |
| Phase 7 — Source consensus | Luna | Medium |
| Phase 8 — Fantasy scoring | Luna | Medium |
| Phase 9 — Excel workbook | Terra | Medium |
| Phase 10 — End-to-end hardening and release | Sol | High |
| Routine test/lint repair with an obvious cause | Luna | Light or Medium |
| Mechanical documentation update | Luna | Light |
| Drafting a PR description | Luna | Light |
| Ordinary code review | Terra | Medium |
| Identity or pricing review | Terra | High |
| Statistical-method review | Sol | XHigh |
| Final release audit | Sol | High |

Escalation ladder:

1. Terra/Medium for normal engineering.
2. Terra/High for nuanced edge cases.
3. Sol/High for architecture, difficult debugging, or integration.
4. Sol/XHigh for statistical calibration.
5. Sol/Ultra only after XHigh fails on a concrete, well-defined problem or the user explicitly requests a highest-cost audit.

Luna is appropriate only for bounded work with strong deterministic tests. Do not use Luna to choose statistical assumptions, resolve ambiguous settlement semantics, or approve identity matching rules.

## Phase isolation

- Use one fresh Codex task/chat per phase so context and token use remain bounded.
- Phase 6 is the exception: implement it as two independently reviewed PRs, 6A and 6B.
- Start from the latest successful `main` commit after the previous phase is merged.
- Prefer a Codex-managed Worktree based on `main` for implementation. Local checkout is acceptable when the user intentionally chooses it and no parallel work will conflict.
- Use branch name `codex/phase-{N}-{slug}`; examples:
  - `codex/phase-0-foundation`
  - `codex/phase-2-historical-baselines`
  - `codex/phase-6a-historical-calibration`
  - `codex/phase-6b-market-inversion`
- Never implement a phase directly on `main`.
- Before editing, inspect Git status, current branch/HEAD, applicable instructions, and the relevant roadmap section. Preserve unrelated user changes.

## Phase execution workflow

1. Read `SPEC.md`, this file, and only the requested phase of `ROADMAP.md` plus directly relevant source files.
2. Restate the requested phase, its exit gate, active model/effort, branch, and any discovered blocker.
3. Create or confirm the phase branch before the first code change.
4. Implement the smallest complete vertical slice satisfying that phase.
5. Add tests alongside the behavior they protect.
6. Run targeted tests during development. Run the cumulative deterministic offline suite once before PR handoff.
7. Inspect generated validation artifacts, not only command exit codes.
8. Commit at meaningful green checkpoints.
9. Compare the result against every roadmap exit criterion.
10. Push the branch and create a draft PR when GitHub authentication and tooling are already available. Otherwise, provide a PR-ready title/body and exact push/PR commands; do not install tools or begin authentication without approval.
11. Pause. Do not merge, start the next phase, or continue polishing outside the requested scope.

## Commit guidance

- Commit logical, independently understandable milestones; do not commit every edit.
- A typical phase may use these chunks, but do not force unnecessary commits:
  1. contracts, fixtures, and failing tests
  2. implementation
  3. validation and integration tests
  4. configuration/documentation updates
- Every commit must leave the repository internally coherent and should pass its relevant test subset.
- Do not rewrite or squash published phase history unless the user requests it.
- Never include secrets, local caches, live run outputs, credentials, or unrelated dirty files.

## Testing and token economy

- Use compact deterministic fixtures for routine work; do not repeatedly load the large checked-in JSON snapshots unless the phase requires a real-data validation.
- Run the smallest relevant tests after local edits rather than the full suite after every change.
- Run the full deterministic offline suite once before draft PR handoff and again before an approved merge.
- Network-backed collector tests must be opt-in and must not be required for ordinary CI.
- Do not spend tokens re-explaining `SPEC.md`; cite the relevant section and focus commentary on decisions, failures, and evidence.
- Avoid proactive subagents unless the user explicitly asks for parallel agent work.

## Draft PR contract

The draft PR must contain:

- phase number and scope
- concise outcome summary
- logical commit list
- important design decisions and assumptions
- files/artifacts created or changed
- tests and exact commands run
- results mapped to every phase exit criterion
- data-quality or numerical validation evidence where applicable
- known limitations, warnings, and unresolved questions
- confirmation that later phases were not implemented
- recommended review model/effort from the table above

Use a title such as:

```text
Phase 2: add leakage-free historical player baselines
```

Create the PR as a draft and stop for user review.

## Review and merge workflow

- Use a separate Codex task for Phase 6 statistical review. Separate review is optional for other phases unless the user requests it.
- Review findings must lead with correctness problems, regressions, missing tests, or unmet exit criteria. Do not modify the branch during a read-only review unless explicitly asked.
- After the user explicitly approves merging:
  1. fetch the remote and confirm the PR/branch still represents the reviewed commit
  2. update from current `main` and resolve conflicts without discarding user work
  3. rerun targeted tests and the deterministic offline suite
  4. confirm required checks pass
  5. merge using the repository/user-selected strategy
  6. delete the phase branch when safe
  7. confirm local `main` is synchronized
  8. stop; the next phase begins in a new task
- If review changes are required, implement them on the same phase branch, add logical commits, update the draft PR, and pause again.

## Copy-paste prompts for Codex Desktop

Implementation task:

```text
Implement Phase {N} from ROADMAP.md. Follow AGENTS.md and SPEC.md. Work only on this phase in a Worktree based on main, create the prescribed codex/phase-{N}-{slug} branch, make logical green commits, run the required tests, create a draft PR, and stop for my review. Before editing, confirm whether the active model and effort meet AGENTS.md's recommendation.
```

Phase 6 implementation tasks:

```text
Implement Phase 6A (historical distribution calibration) only. Follow AGENTS.md and SPEC.md. Do not implement the Kalshi update or market mean inversion. Create a draft PR and stop.
```

```text
Implement Phase 6B (Kalshi update and source-specific mean inversion) only, starting from the merged Phase 6A main. Follow AGENTS.md and SPEC.md. Create a draft PR and stop.
```

Review task:

```text
Review draft PR {PR_NUMBER} against SPEC.md, AGENTS.md, and the Phase {N} ROADMAP.md exit gate. Do not edit or merge. Report correctness issues, missing tests, numerical risks, and whether each exit criterion is satisfied.
```

Merge task:

```text
I approve draft PR {PR_NUMBER}. Follow the AGENTS.md merge workflow: update it from main, resolve conflicts safely, rerun required tests, merge only if all checks pass, clean up the branch, synchronize main, and stop. Do not begin the next phase.
```

## Safety and quality boundaries

- Never hide failed checks, silently loosen thresholds, or replace missing projections with zero.
- Never use current or future historical fields in a preseason baseline.
- Never average raw odds or market lines across sources.
- Never auto-merge ambiguous player identities merely to improve coverage.
- Never treat historical calibration data as a fourth consensus projection source.
- Stop and surface a blocker when proceeding would require a materially new assumption, external credential, destructive action, or scope expansion.
