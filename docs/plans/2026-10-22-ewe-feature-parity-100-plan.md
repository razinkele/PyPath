# EwE Feature Parity to 100% — Implementation Plan (2026-10-22)

## Objective

Reach **100% EwE feature parity** for the scoped EwE 6.6+ target surface, with auditable evidence in CI.

## Baseline check (today)

- Repository documentation currently states **86% coverage** (`72/84`) for EwE DB tables.
- `packages/pypath/src/pypath/io/_ewe_schema.py` currently defines **72 schema tables** (`EWE_TABLES` length = 72).
- Therefore, parity closure requires both:
  1. **Schema completion** from 72 to 84 target tables.
  2. **Reader/writer/runtime behavior completion** for those tables/features.

## Definition of done (100%)

100% parity is considered achieved only when all are true:

1. **Schema parity:** all target EwE tables are represented in schema constants.
2. **I/O parity:** each target table has validated read/write support (Access and CSV bundle paths, where applicable).
3. **Behavior parity:** table-backed behavior is executed in runtime logic (not just stored).
4. **Roundtrip parity:** import -> export -> import maintains expected fields/semantics within tolerances.
5. **CI evidence:** automated parity report passes and is published as artifact.
6. **Docs parity:** README/CHANGELOG/spec docs report the same percentage and scope.

## Checkpoint plan

### Checkpoint 1 — Parity contract + gap ledger

**Deliverables**
- Add machine-readable matrix: `docs/parity/ewe_parity_matrix.csv`.
- Add one script/test that derives current support from code and fails on unmapped targets.
- Freeze the target table set for this program (`84` total).

**Validation gate**
- CI job `parity-audit` emits:
  - total target tables
  - implemented read/write tables
  - missing table list
- Build fails if matrix and code-derived counts diverge.

---

### Checkpoint 2 — Schema completion (72 -> 84)

**Deliverables**
- Extend `_ewe_schema.py` to include all missing target table definitions.
- Add strict column/type definitions and comments for compatibility assumptions.

**Validation gate**
- `len(EWE_TABLES) == 84` enforced by test.
- Schema unit tests assert required columns on newly added tables.

---

### Checkpoint 3 — Reader completion for missing tables

**Deliverables**
- Implement table readers in `io/ewemdb.py` for each missing table.
- Add normalization/parsing rules for sentinels and enums.

**Validation gate**
- Table-read smoke tests for all 84 tables against representative fixture DBs.
- Failure on any target table returning unknown/unhandled table errors.

---

### Checkpoint 4 — Writer completion (Access + CSV)

**Deliverables**
- Extend `_access_writer.py` and `_csv_bundle_writer.py` serialization for each missing table.
- Ensure foreign-key-safe ordering for write/clear operations.

**Validation gate**
- Export tests verify all target tables appear with expected columns.
- Access and CSV outputs pass schema validation and reload without loss of mandatory fields.

---

### Checkpoint 5 — Runtime semantics parity

**Deliverables**
- Wire newly covered features into runtime modules where required (Ecosim/Ecospace/auxiliary flows).
- Add deterministic feature-level behavior tests (not only I/O tests).

**Validation gate**
- Numeric parity tests against EwE/Rpath fixtures for each newly activated feature surface.
- Thresholds and acceptance tolerances documented per feature.

---

### Checkpoint 6 — Roundtrip + compatibility harness

**Deliverables**
- Add full roundtrip harness:
  - load `.eweaccdb`
  - export `.eweaccdb` and `.ewecsv.zip`
  - re-import and compare
- Generate parity report JSON/CSV artifact from tests.

**Validation gate**
- Zero missing tables.
- Roundtrip deltas within configured tolerances.

---

### Checkpoint 7 — Documentation and release discipline

**Deliverables**
- Update root and package READMEs to 100% parity claim with scope note.
- Update changelogs with concrete parity completion details.
- Add CI badge/report link for parity audit.

**Validation gate**
- Docs and parity report percentages match exactly.
- Release checklist blocks publish if parity audit is not green.

## Execution strategy

- Implement in vertical slices by domain (timeseries/calibration, mediation, Monte Carlo/pedigree, ecotracer, advanced ecospace/taxonomy/value-chain residuals).
- For each slice use strict loop:
  1. add tests + parity matrix rows (red)
  2. implement schema + read/write + behavior (green)
  3. run targeted + core fast suite + parity audit
  4. update docs for that slice

## Risks and mitigations

- **Risk:** table-level support without runtime semantics.
  - **Mitigation:** feature acceptance tests mandatory for each table group.
- **Risk:** Access-only behavior drift vs CSV path.
  - **Mitigation:** dual-path export/import assertions in CI.
- **Risk:** parity % drift in docs.
  - **Mitigation:** docs parity values generated from audit output.

## Immediate next implementation task

Start with **Checkpoint 1**:
- create `docs/parity/ewe_parity_matrix.csv`
- add parity audit test/script that computes baseline and missing list from code
- wire parity audit into CI as non-optional for release workflows.
