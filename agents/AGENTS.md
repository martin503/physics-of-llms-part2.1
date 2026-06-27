# Agent Guidelines

This project uses two specialized agents with a strict division of labor:

| Agent | File | Owns |
|-------|------|------|
| Coding | `CODING_AGENT.md` | `src/` directory |
| Testing | `TESTING_AGENT.md` | `tests/` directory |

**Neither agent ever touches the other's files.**

## ROUNDs

A **ROUND** is the atomic unit of work: one feature or fix, taken from request through sign-off. ROUNDs are **strictly sequential** — finish the current ROUND before starting the next one.

The user controls whose turn it is. At any stage the user may also give direct instructions, tips, or corrections — agents follow them immediately without waiting for a turn signal.

### Step 1 — Coding agent receives a feature request

The user describes a feature or bug fix directly to the coding agent.

The coding agent:
- Implements the change in `src/`
- Prepends a CODING/todo entry to `CHANGELOG.md` (see entry format below)
- Does NOT run tests — the testing agent is the sole quality gate
- Does NOT touch `PROJECT.md` yet

### Step 2 — Testing agent reads CHANGELOG

The user says **`CHANGELOG`** to the testing agent. This means: the coding agent has finished and updated `CHANGELOG.md` — read it and act.

The testing agent:
- Reads the topmost CODING/todo entry in `CHANGELOG.md`
- Reads the actual code changes in `src/` to understand the implementation (not just the CHANGELOG description) — this is essential for writing targeted tests
- Adopts an adversarial mindset: the goal is to find holes, unhandled inputs, boundary conditions, and logic errors — not to rubber-stamp the implementation
- Writes new tests covering: the happy path, every edge case listed in "Edge cases to test", and additional adversarial scenarios the testing agent identifies from reviewing the code
- Runs `make test-full`

If all tests pass:
- Marks the CODING entry `status = done` (edit in place)
- Prepends a TESTING/done entry to `CHANGELOG.md`
- Updates `PROJECT.md` Section 3 (test scenarios, edge cases)

If any tests fail:
- Prepends a TESTING/todo entry listing every bug with `status: open`
- Lists everything that passed under "Confirmed working"
- Does NOT mark the CODING entry done — it stays `todo` until fully clean

### Step 3 — Coding agent reads CHANGELOG

The user says **`CHANGELOG`** to the coding agent. This means: the testing agent has finished and updated `CHANGELOG.md` — read it and act.

The coding agent reads the topmost entries and determines the state:

**If the CODING entry is `status = done`** — the testing agent signed off:
- Updates `PROJECT.md` Section 2 (features, bug fixes)
- Prepends a ROUND COMPLETE entry to `CHANGELOG.md`
- The ROUND is finished

**If a TESTING/todo entry has open bugs** — bugs were found:
- Fixes every bug marked `status: open`
- Marks the TESTING/todo entry `status = done` (edit in place)
- Does NOT prepend a new CODING entry — the original entry is the record
- The user will then signal the testing agent to re-test (back to Step 2)

**If nothing is actionable** (e.g., everything is already ROUND COMPLETE):
- Reports the current state and waits for user instructions

### Step 4 — Testing agent re-tests (bug-fix loop)

Same as Step 2, but triggered after the coding agent fixed bugs. The testing agent:
- Re-runs `make test-full`
- Does NOT add a new TESTING entry — updates the existing TESTING/todo entry in place

If all tests pass:
- Marks the original CODING entry `status = done`
- Marks the TESTING/todo entry `status = done`, all bugs `status = fixed`
- Updates `PROJECT.md` Section 3

If some bugs are fixed but some remain:
- Marks fixed bugs `status = fixed`
- Leaves remaining bugs `status = open`
- Loop continues back to Step 3

If new bugs appeared:
- Adds them to the existing TESTING/todo entry as new items with `status: open`

## Communication via CHANGELOG.md

`CHANGELOG.md` is the persistent coordination mechanism. Both agents prepend entries (newest at top). The `status` field on each entry is the work queue.

### How entries relate to each other

Every TESTING entry has a `Ref` field pointing to exactly one CODING entry by its timestamp and title. This is the permanent link between a feature and all testing activity around it. When navigating the changelog, always follow `Ref` to understand context.

A complete ROUND looks like this in the file (newest at top, read bottom-up for chronological order):

```
## [day 3, 15:00] ROUND COMPLETE — add search feature

## [day 3, 14:00] TESTING — all pass               ← status: done
  Ref: [day 1, 09:00] CODING — add search feature      all bugs: fixed

## [day 2, 11:00] TESTING — 2 bugs found           ← status: done (after bug-fix loop)
  Ref: [day 1, 09:00] CODING — add search feature      bugs: fixed/fixed

## [day 1, 09:00] CODING — add search feature      ← status: done
```

There is exactly one CODING entry per feature. There may be one or more TESTING entries, each linked via `Ref`. The CODING entry's status is the final verdict: `todo` means work is ongoing, `done` means fully signed off.

### CHANGELOG.md entry format

Entries are always **prepended** (newest first). Do not delete entries — only update `status` fields and bug statuses in place.

**Coding entry:**
```markdown
## [YYYY-MM-DD HH:MM] CODING — <one-line summary of the feature or fix>
**status**: todo

### What changed
<2–4 sentences describing the new behavior from a user perspective.
Be specific enough that the testing agent can test it without reading the code.>

### Files modified
- `src/parser.py` — <reason>
- `src/templates/viewer.html` — <reason>

### Edge cases to test
- <specific input or scenario to exercise>
- <another edge case worth covering>

---
```

**Testing entry — always references the CODING entry it responds to:**
```markdown
## [YYYY-MM-DD HH:MM] TESTING — <e.g. "6 tests added, all pass" or "3 bugs found">
**status**: todo | done

### Ref
[YYYY-MM-DD HH:MM] CODING — <exact title copied from the CODING entry>

### Tests added
- `tests/test_parser.py::test_foo` — <what behavior it covers>

### Results
Passed: X | Failed: X | Skipped: X

### Confirmed working
- <behavior that was tested and passed — fill in even when bugs exist elsewhere>

### Bugs found
#### Bug: <short title>
**status**: open | fixed
- **Repro**: <minimal input or step sequence that triggers it>
- **Expected**: <what should happen>
- **Actual**: <what does happen>

### Coverage gaps
- <area not yet covered and why it matters>

---
```

**ROUND COMPLETE entry:**
```markdown
## [YYYY-MM-DD HH:MM] ROUND COMPLETE — <same title as the CODING entry>

---
```

**Status rules — only these values are valid:**

| Entry type | Status | Meaning |
|------------|--------|---------|
| CODING | `todo` | Testing agent must test this |
| CODING | `done` | Signed off; coding agent updates PROJECT.md |
| TESTING | `todo` | Coding agent must fix open bugs |
| TESTING | `done` | No action needed; record only |
| Bug item | `open` | Not yet fixed |
| Bug item | `fixed` | Coding agent addressed it; testing agent should verify |

## PROJECT.md — verified features only

`PROJECT.md` contains only verified, fully-tested features. It is a stable record of what works, not a draft of what is in progress.

- **Coding agent** updates Section 2 (features, bug fixes) after the CODING entry is marked `done` and the user says `CHANGELOG`
- **Testing agent** updates Section 3 (test scenarios, edge cases) when it marks a CODING entry `done` (all tests pass)
- Neither agent edits the other's section

## File Ownership

| Path | Owner |
|------|-------|
| `src/**` | Coding only |
| `tests/**` | Testing only |
| `PROJECT.md` Section 2 | Coding only (after sign-off) |
| `PROJECT.md` Section 3 | Testing only (on sign-off) |
| `CHANGELOG.md` | Both prepend; both edit `status` fields in existing entries |
| `AGENTS.md`, `CODING_AGENT.md`, `TESTING_AGENT.md` | Read only |
