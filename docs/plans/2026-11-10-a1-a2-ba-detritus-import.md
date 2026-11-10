# A1 + A2: Biomass Accumulation and Detritus Import Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `read_ewemdb` carry EwE's biomass accumulation (BiomAcc, BiomAccRate) and detritus import (DtImports) into Ecopath, and make Ecosim and Ecospace add the constant detritus import, so an imported model starts balanced at t = 0.

**Architecture:** The core already has `BioAcc` and `DetInput` columns, and Ecopath already uses them. Only the reader aliases are wrong, so A1 is a reader fix. BiomAccRate is resolved to BA = rate·B inside the reader, so no new model column is needed. For A2, `Rpath` gains a `DetInput` array that is threaded through `RsimParams` into `deriv_vector`. There, both detritus kernels and the fast-group rule add it as a constant gain, as EwE's `DetInFlow` does. Ecospace gets it through the same params dict.

**Tech Stack:** Python ≥3.10, numpy, numba (optional twin kernels), pytest, mdb/ODBC reader for `.eweaccdb`.

**Spec:** `C:\Users\arturas.baziukas\OneDrive - ku.lt\HORIZON_EUROPE\PyPath\docs\reviews\2026-11-10-audit-ewe-6.7.144-review.md` §A1 (lines 79-136) and §A2 (lines 138-144). The spec is untracked and exists only in the OneDrive checkout, not in the worktree.

**Oracles (untracked, OneDrive checkout only):** call this folder `ORACLES` = `C:\Users\arturas.baziukas\OneDrive - ku.lt\HORIZON_EUROPE\PyPath\docs\reviews\audit-oracles\`. It holds `audit_main/ba_scan.py` and the EwE 6.7.144 sources (`ewe6_7144/`). The sweep script and its outputs go in `ORACLES/a1a2/`.

## Verified facts this plan relies on (checked 2026-11-10 on `bb1acbb`)

- **Slice-2 columns are unused.** `ba_scan.py` finds OtherMort, Immigration, Emigration and EmigRate zero or absent in all six databases.
  - Non-zero BA/import appears only in LT2022 (Scoters BiomAcc −0.0039 and rate −0.1, Long-tailed duck −0.0006 and −0.1, Detritus DtImports 1000) and in Edgaras (BiomAcc on 5 groups, no rate).
- **No Rpath reference model has a non-zero DetInput** (`tests/data/rpath_reference/`, `example_model_data/`). A2 therefore changes no Rpath parity result, and the import can be unconditional, with no Rpath-compatibility switch.
- **EwE behaviour.**
  - Detritus `biomeq = (DetInFlow + ToDetritus) / (loss/B)` (`cEcoSimModel.vb:3126`).
  - The export rate at first call counts `+DtImp − BA` (`:3207`).
  - The fast-group rule uses `|loss|/B > 24` (`:2795`).
  - Adding the import to the inflow in `ecosim.py:1093`, the fast-group rule, therefore matches EwE.
- **Ecopath detritus.** Detritus `PB = total inputs / biomass` (`ecopath.py:716`), so `Biomass·PB` is the detritus input.
- **Ecospace.** `_local_deriv` (`spatial/integration.py:56-71`) evaluates `deriv_vector` on densities (`B/w`) and scales by `w`. A per-km² DetInput is therefore right as is.
- **Ecosim entry point.** `rsim_run(scenario, method="RK4"|"AB", years=...)`.

## Global Constraints

- Branch off `origin/main` at `bb1acbb` in a fresh worktree. The OneDrive checkout's local `main` is at `feed495` and is stale. Main auto-deploys to laguna, so all work goes through a PR.
- Run Python as described in the `shiny-env-gotchas` memory: put `envs/shiny/Library/bin` on PATH, set a colon-separated `PYTHONPATH=<worktree>/packages/pypath/src`, and set `PYTHONIOENCODING=utf-8`.
- Use logging, not print: `logger = logging.getLogger(__name__)` and `%s` formatting. Never write a bare `except Exception: pass`.
- Format with `ruff format` (black is not installed) and lint with ruff.
- **Test data.** Database tests use `Path(__file__).parents[3] / "Data"` and skip without the database. **Every behaviour must also have a DB-free test**, because CI has no `Data/`.
- Do not commit `docs/*.pdf` or `docs/reviews/`.
- **Out of scope (slice 2):** OtherMort→EE, and the Immigration/Emigration/EmigRate terms. This change only warns when they are non-zero.
- **Out of scope:** BA on detritus groups. No database sets it, and the fast-group rule already handles it.
- **Out of scope:** `write_ewe_samples` writing BiomAcc as 0 (`ewe_writer.py:508-510`). That is the Monte Carlo samples writer, not the model writer. File it as a follow-up.

## Review Focus

1. **BiomAccRate ≠ 0 with BiomAcc = 0 and B left to Ecopath (−9999).** EwE computes BA = rate·B after B is estimated. Here a warning is expected and BA stays 0, rather than being silently wrong. Test in Task 1.
2. **DtImports = −9999 on a detritus row, or a non-zero value on a living row.** Expected DetInput is 0 on detritus and NaN on living rows. Test in Task 1.
3. **Models with no BA and no import (BlackSea, Curonian consolidated, the Rpath reference).** Expected output is bit-identical before and after. Tests in Task 3 (DB-free) and Task 5 (sweep).
4. **Detritus at fast equilibrium (NoIntegrate) with import.** Expected `biomeq` to include the import, so detritus does not fall each month, under AB and under RK4. Test in Task 3.
5. **Ecospace with detritus import.** Expected the per-patch detritus derivative to be about 0 at t = 0. Test in Task 4.

---

### Task 0: Worktree and baseline

- [ ] **Step 1:** Run `git fetch origin && git worktree add ~/worktrees/pypath-a1a2 -b fix/a1-a2-ba-detritus-import origin/main`. Expected result: `git log -1 --oneline` shows `bb1acbb`.
- [ ] **Step 2:** Copy `~/worktrees/pypath-integration/Data` to `~/worktrees/pypath-a1a2/Data`. It is untracked, so it does not come with the worktree.
- [ ] **Step 3:** Write `ORACLES/a1a2/sweep.py <out.csv> <db>...`. For each database it runs `rpath(read_ewemdb(db))` and writes one row per group: db, Group, type, Biomass, PB, EE, M0, BA. Run it on the new worktree (`bb1acbb`) to produce `ORACLES/a1a2/sweep_before.csv`.

### Task 1: Reader maps BiomAcc, BiomAccRate and DtImports, and warns on unsupported columns

**Files:**
- Modify: `packages/pypath/src/pypath/io/ewemdb.py:797-798` (column_mapping), after the loop at `:825-838`, and `:883` (VarName map)
- Test: `packages/pypath/tests/test_ewemdb_ba_detritus_import.py` (new)

**Interfaces:**
- Produces:
  - `_resolve_group_inputs(groups_df: pd.DataFrame, types: np.ndarray) -> tuple[np.ndarray, np.ndarray]`, returning `(bioacc, det_input)` per biological group. It is a pure function, so it can be tested without a database.
  - `_warn_unsupported_group_columns(groups_df: pd.DataFrame, group_names: list[str]) -> None`.
  - `params.model["BioAcc"]` and `params.model["DetInput"]`, filled from the helper.

- [ ] **Step 1: Write the failing tests.**

```python
# DB-free (synthetic EcopathGroup frames; -9999 = EwE missing)
def test_biomacc_read_directly():        # BiomAcc=-0.0039, BiomAccRate=-0.1, Biomass=0.039 -> bioacc == -0.0039
def test_biomaccrate_fallback():         # BiomAcc=0, BiomAccRate=-0.1, Biomass=2.0 -> bioacc == -0.2
def test_biomaccrate_unknown_b_warns(caplog):  # BiomAcc=0, rate=-0.1, Biomass=-9999 -> bioacc == 0, one WARNING naming the group
def test_dtimports_detritus_only():      # detritus DtImports=1000 -> 1000; detritus -9999 -> 0.0; living DtImports=5 -> NaN
def test_unsupported_columns_warn(caplog):  # OtherMort=0.2 on one group -> one WARNING naming "OtherMort" and the group;
                                            # all-zero / -9999 / absent columns -> no warning
# DB (integration; skip without Data/ or driver, pattern of test_lt2022_ewef_parity.py:44-54)
def test_lt2022_reads_ba_and_import():   # Scoters BioAcc -0.0039, Long-tailed duck -0.0006, Detritus DetInput 1000, Scoters DetInput NaN
def test_edgaras_reads_biomacc():        # "Dvigeldžiai filtratoriai" 11618.8 (rel 1e-4), "Zoobentosas" -13.3411 (rel 1e-4)
```

- [ ] **Step 2:** Run `pytest packages/pypath/tests/test_ewemdb_ba_detritus_import.py -v`. Expected result: FAIL, with the helper functions undefined.
- [ ] **Step 3: Implement.**
  - **`_resolve_group_inputs`:**
    - Turn −9999 into NaN.
    - `bioacc = BiomAcc`. Where it is 0 or NaN and BiomAccRate ≠ 0: use rate·Biomass if the input Biomass > 0. Otherwise warn and use 0. This is EwE's rule, `cEcoPathModel.vb:1564-1568`.
    - `det_input = DtImports` on type-2 rows, with NaN turned into 0. Living rows get NaN.
  - **Aliases:** drop the old `BioAcc` and `DetInput` alias entries from `column_mapping`. The helper owns these columns now, though it still accepts the legacy names `BA` and `DetInput` when `BiomAcc` or `DtImports` is absent.
  - **VarName map:** remove `"BioAccRate": "BioAcc"` at `:883`.
  - **Unmatched aliases:** log `logger.debug` when no alias matches.
  - **More ignored-setting warnings (parity roadmap, P1):**
    - Extend the warnings with one `logger.warning` per setting and per affected group or scenario, each saying that PyPath ignores the setting.
    - Settings to cover:
      - EcosimScenarioGroup: RiskTime ≠ 0, QmQo ≠ 1, CmCo ≠ 1000, FishRateMax, AdditivePredMort.
      - Stanza: BABsplit ≠ 0, LeadingCB ≠ LeadingLifeStage.
      - EcopathGroup: fractional Type.
      - EcosimScenario: StepSize, SystemRecovery, sorwt or Discount away from EwE defaults.
      - Nutrient settings.
    - Before adding the RiskTime, QmQo and CmCo checks, verify the EwE defaults in `cEcoSimDatastructures.vb`.
    - Add one DB-free test per table: a synthetic frame with one non-default value gives exactly one warning, and the defaults give none.
    - Add one LT2022 test: RiskTime warns and names the affected groups.
- [ ] **Step 4:** Run the test file again. Expected result: PASS. Then run `pytest packages/pypath/tests/ -q -k "ewemdb" --ignore=packages/pypath/tests/scripts`. Expected result: no new failures, apart from the EwE-F parity pins that Task 2 handles.
- [ ] **Step 5:** Commit with `fix(io): read EwE BiomAcc, BiomAccRate and DtImports; warn on unsupported mortality/migration columns`.

### Task 2: LT2022 Ecopath balance matches EwE; re-pin the EwE-F parity test

**Files:**
- Modify: `packages/pypath/tests/test_lt2022_ewef_parity.py`
- Test: `packages/pypath/tests/test_ewemdb_ba_detritus_import.py`

- [ ] **Step 1: Write the tests.** They should pass immediately after Task 1, because the core already uses BioAcc and DetInput. That proves no core change is needed for Ecopath.

```python
# Bird EE: derived from EwE's EE formula on EwE 6.7.144 inputs (EwE stores no output EE).
# Detritus Export 5366.09: stored by EwE (EcopathGroup output) - the strongest oracle here.
def test_lt2022_bird_ee_with_ba(lt2022_rpath):
    ee = dict(zip(lt2022_rpath.Group, lt2022_rpath.EE))
    assert ee["Scoters"] == pytest.approx(0.0043, abs=2e-4)
    assert ee["Long-tailed duck"] == pytest.approx(0.227, abs=2e-3)

def test_lt2022_detritus_with_import(lt2022_rpath):
    i = list(lt2022_rpath.Group).index("Detritus")
    inputs = lt2022_rpath.Biomass[i] * lt2022_rpath.PB[i]          # detritus PB = inputs / B
    assert inputs == pytest.approx(6131.6, rel=1e-3)
    assert lt2022_rpath.EE[i] == pytest.approx(0.125, abs=2e-3)
    assert inputs * (1 - lt2022_rpath.EE[i]) == pytest.approx(5366.09, rel=1e-3)
```

- [ ] **Step 2:** Run the tests. Expected result: PASS. If they fail, stop and report, because the core does not use the columns as the audit assumed.
- [ ] **Step 3: Re-pin the EwE-F test.**
  - Remove `Scoters`, `Long-tailed duck` and `Detritus` from `EWEF_EE`, with a comment that EwE-F's input lacked BiomAcc/DtImports and that these groups are now pinned in `test_ewemdb_ba_detritus_import.py`.
  - Run `pytest packages/pypath/tests/test_lt2022_ewef_parity.py -v`. Expected result: 19 PASS.
  - If any of the other 19 moves, stop and investigate. Don't re-pin those.
- [ ] **Step 4:** Commit with `test: pin LT2022 bird EE and detritus export to EwE; drop BA-free EwE-F values`.

### Task 3: Ecosim adds the constant detritus import (A2)

**Files:**
- Modify: `packages/pypath/src/pypath/core/ecopath.py`. Add the `Rpath` field `DetInput: np.ndarray = field(default_factory=lambda: np.zeros(0))` (length ngroups, 0 except on detritus) and fill it from `det_input` (`:657-664`).
- Modify: `packages/pypath/src/pypath/core/ecosim.py`:
  - `RsimParams` (`:56`): add `DetInput: np.ndarray`, size NUM_GROUPS+1 and indexed like `B_BaseRef`, with a default. Grep for direct `RsimParams(` constructors.
  - `rsim_params`: fill it from `getattr(rpath, "DetInput", None)`, using zeros when it is empty or missing.
  - The `rsim_run` params dict (`:~1450`): add `"DetInput": params.DetInput`.
  - Fast-group rule (`:1093`): `inflow += det_input[d]`.
- Modify: `packages/pypath/src/pypath/core/ecosim_deriv.py`:
  - `_rpath_detritus_derivs(..., det_input=None)`: add `gain[d] += det_input[d]` before computing `deriv[d]`, so `TotGain` and NoIntegrate's `biomeq` include it.
  - The legacy matrix path after `_compute_detritus_derivs(...)` (`:2124`): add `deriv[d] += det_input[d]` for the dead groups. Leave the numba signature unchanged.
  - `deriv_vector` (`:1937`): pass `params.get("DetInput")`.
- Modify: `packages/pypath/src/pypath/io/ewe_writer.py:94`. Add `params.model["DetInput"]` from `model.DetInput`, so `_rpath_to_params` round-trips it.
- Test: `packages/pypath/tests/test_ecosim_detritus_import.py` (new)

**Interfaces:**
- Produces: `Rpath.DetInput`, `RsimParams.DetInput` and the params-dict key `"DetInput"`. Task 4 uses this key.

- [ ] **Step 1: Write the failing tests.**
  - **Shared model:** a small model, `_det_import_model(det_input: float)`, built with `create_rpath_params` like the one in `tests/test_fast_group_rule.py:31`. It has one producer, one consumer and one detritus group, with Detritus `DetInput = det_input`. Each test gets the t = 0 derivative from `deriv_vector(B_BaseRef, params_dict, forcing_dict, fishing_dict)`. Build the dicts the way `rsim_run` does; if that is inline code, test through `rsim_run(..., years=range(1, 2))` and its month-0 derivative output instead.

```python
@pytest.mark.parametrize("det_input", [0.0, 50.0])
@pytest.mark.parametrize("link_based", [True, False])   # True: params with DetLinkFrom (rsim_run path);
                                                        # False: dict without DetLinkFrom (legacy matrix path)
def test_t0_detritus_balanced(det_input, link_based):   # |deriv[det]| < 1e-9 * det inputs

def test_t0_living_derivative_equals_ba():              # consumer BioAcc = 0.01 -> deriv[consumer] == approx(0.01, abs=1e-9)

@pytest.mark.parametrize("method", ["AB", "RK4"])
@pytest.mark.parametrize("noint", [True, False])        # force NoIntegrate on/off for detritus
def test_detritus_flat_over_a_year(method, noint):      # det_input=50; |B_det(m12)/B_det(m0) - 1| < 1e-6

def test_det_input_zero_is_bit_identical():             # model without DetInput: rsim_run output == output on bb1acbb
                                                        # (np.array_equal on out.out_Biomass vs the same run with "DetInput" key removed)
# DB (integration): LT2022 scenario from ecosim_scenario_from_ewemdb(DB_PATH, scenario=16, timeseries_dataset=35)
def test_lt2022_t0_birds_equal_ba():                    # deriv[Scoters] == approx(BA, abs=1e-8), same for Long-tailed duck
def test_lt2022_t0_detritus_balanced():                 # |deriv[Detritus]| < 1e-3   (was about -1000 before the fix)
```

- [ ] **Step 2:** Run `pytest packages/pypath/tests/test_ecosim_detritus_import.py -v`. Expected result: every `det_input=50` case and the LT2022 detritus test FAIL, while the `det_input=0` cases and the bird and BA tests PASS.
- [ ] **Step 3:** Implement as listed under Files.
- [ ] **Step 4:** Run `pytest packages/pypath/tests/test_ecosim_detritus_import.py packages/pypath/tests/test_fast_group_rule.py packages/pypath/tests/test_ecosim.py packages/pypath/tests/test_ecopath.py -v`. Expected result: PASS. Then run the Rpath reference parity tests (`pytest packages/pypath/tests/ -q -k "rpath_reference or parity" --ignore=packages/pypath/tests/scripts`) and expect them unchanged.
- [ ] **Step 5:** Commit with `fix(ecosim): constant detritus import inflow, matching EwE DetInFlow (A2)`.

### Task 4: Ecospace carries the import

**Files:**
- Modify: `packages/pypath/src/pypath/spatial/integration.py:~709`. Add `"DetInput": getattr(params, "DetInput", None)` to the spatial params dict.
- Test: `packages/pypath/tests/test_ecosim_detritus_import.py`

- [ ] **Step 1: Write the test.** `test_ecospace_t0_detritus_balanced` uses `_det_import_model(50.0)` on a 3-patch grid with unequal areas (1, 2, 3). Check that the t = 0 detritus derivative of every patch is below 1e-9 times the patch's inputs. Unequal areas exercise the `w` scaling in `_local_deriv`.
- [ ] **Step 2:** Run it. Expected result: FAIL, with each patch at about `−50·w`.
- [ ] **Step 3:** Add the dict key.
- [ ] **Step 4:** Run it again. Expected result: PASS. Then run `pytest packages/pypath/tests/ -q -k "spatial or ecospace" --ignore=packages/pypath/tests/scripts` and expect no failures.
- [ ] **Step 5:** Commit with `fix(ecospace): pass detritus import into per-patch derivatives`.

### Task 5: Sweep, scenario-16 re-pin, round trip, full suite, PR

**Files:**
- Modify: `packages/pypath/tests/test_ecosim_lt2022_parity.py:27` (`LT2022_SS16`, plus its history comment)
- Test: `packages/pypath/tests/test_ewe_writer_roundtrip.py`. Add `test_access_biomacc_roundtrip`: write LT2022 with `write_ewemdb`, read it back, and check that Scoters BioAcc is −0.0039 and Detritus DetInput is 1000. Skip without ODBC.

- [ ] **Step 1:** Run `ORACLES/a1a2/sweep.py` on the branch to produce `ORACLES/a1a2/sweep_after.csv`, then diff it against `sweep_before.csv`. Expected results:
  - BlackSea, Curonian consolidated and Curonian export are bit-identical, apart from the export copy's LT2022 groups.
  - In the LT2022 files, only Scoters, Long-tailed duck and Detritus change.
  - **Edgaras:** list the groups whose EE leaves [0, 1] and report them in the PR. Don't change Edgaras.
- [ ] **Step 2: Scenario-16 SS.**
  - Run `pytest packages/pypath/tests/test_ecosim_lt2022_parity.py -v`.
  - If the SS has moved by more than 2 %, confirm the cause first: rerun with Detritus DetInput and both birds' BA set to 0 on the scenario, which must reproduce 637.0 ± 0.1. Only then re-pin.
  - Add a history-comment line in the existing style: "…, and X with biomass accumulation and detritus import read (A1/A2)". Note that the value is PyPath's own output and has not been checked against EwE.
- [ ] **Step 3:** Run the round-trip test. If `write_ewemdb` drops BiomAcc or DtImports, fix the model writer in `ewe_writer.py`, then run it again. Expected result: PASS.
- [ ] **Step 4: Full verification.**
  - Run `pytest packages/pypath/tests/ -q --ignore=packages/pypath/tests/scripts`, slow tests included (about 27 minutes, so run it in the background). Expected result: 0 failures.
  - Run `pytest packages/pypath-shiny/tests/ -q --ignore=packages/pypath-shiny/tests/ui`. Expected result: 0 failures.
  - Run `ruff check` and `ruff format --check` on the touched files. Expected result: clean.
- [ ] **Step 5: Commit, push and open the PR.** Push the branch and open a PR against main. Ask the user before merging, since main deploys to laguna. In the PR body, include:
  - the sweep table;
  - the Edgaras EE list;
  - the new SS16 value;
  - the note that Ecosim dynamics after the fix are unverified against EwE until the scenario-16 export arrives;
  - follow-ups: slice 2, the samples writer, and detritus BA.
