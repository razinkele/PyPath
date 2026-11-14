# Continuous Pathway, Phase 1 (Ecosim core) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `rsim_run(scenario, method="lsoda" | "dop853" | "radau" | "bdf")` solves Ecosim's continuous equations with SciPy's adaptive solvers, month by month, beside the unchanged RK4/AB parity paths.

**Architecture:** A new module `core/ecosim_continuous.py` holds a pure right-hand side (`ContinuousRHS`), the handling-time fixed-point solver and a month integrator with Gauss–Legendre quadrature. `rsim_run` branches into it at the integration step only; `ecosim_deriv.py` gains three flags read from `params`. The scenario is never mutated, so one scenario runs on both paths.

**Tech Stack:** Python ≥ 3.10, NumPy, SciPy (`scipy.integrate.solve_ivp`, `scipy.optimize.root`), pytest.

**Spec:** `docs/plans/2026-11-14-continuous-pathway-design.md` (sections 2–6 and 9; decisions in section 11).

## Global Constraints

- The parity paths (`method="RK4"`, `"AB"`) give bit-identical results: the existing suite passes unchanged.
- Continuous methods, case-insensitive: `lsoda` (default for the path), `dop853`, `radau`, `bdf`. Any other method name raises `ValueError`.
- Default tolerances: `rtol = 1e-6`; `atol_i = 1e-8 * max(B0_i, 1e-6 * max(B0))` over the biomass states.
- `max_step` is one month (`dt = 1/12` year); the solver restarts at every month boundary; forcing is held constant within a month.
- Corrections on by default on this path: R2 (`DETRITUS_MO_UNSCALED` off), R4 (EmigRate for living groups where only a rate is entered), R5 (detritus gets moTot × B), DF applied once. Hden starts at the recalibrated Hden0; EwE's CmCo / (CmCo + 1) start is dropped.
- Positivity: the wrapper clips states at 0 before evaluating and counts clips; month-end floor 1e-20.
- IBM groups are rejected with `ValueError`.
- Style: black 88, ruff (E, F, I, W); `logger = logging.getLogger(__name__)`; no bare `except Exception: pass`; test data paths via `Path(__file__).parent / "data" / ...`; tests reading `Data/` are marked `integration`.
- No new dependencies.
- Not in phase 1 (spec section 9): mediation lookup interpolation, feeding time as a state, Hden on the kernel only, an analytic Jacobian, linear forcing, Ecospace, the Shiny control.

## Review Focus

- **A group at zero biomass** (extinct, or forced to 0): the derivative and the Hden solve stay finite; no division by zero. Test in Task 4.
- **The Hden fixed point fails to converge** at a pathological state: a `RuntimeError` that names the month and the groups, never a silent wrong answer. Test in Task 3.
- **Forced biomass** stays exactly at its forced value through the month and at the boundary. Test in Task 6.
- **A solver failure mid-run**: raises with the month unless `on_failure="warn"`, which logs and continues from the last state. Test in Task 5.
- **Scenario reuse**: an RK4 run after an LSODA run on the same scenario object equals a fresh RK4 run. Test in Task 6.

---

### Task 1: Method selection, `ValueError`, output fields

**Files:**
- Modify: `packages/pypath/src/pypath/core/ecosim.py` (`RsimOutput` near line 418; `rsim_run` signature near line 1665; the unknown-method fallback near line 2582)
- Test: `packages/pypath/tests/test_ecosim_continuous.py` (new)

**Interfaces:**
- Produces: `CONTINUOUS_METHODS = ("lsoda", "dop853", "radau", "bdf")` in `ecosim.py`; `rsim_run(scenario, method="RK4", years=None, *, mediation=None, ecotracer=None, fleet_dynamics=None, solver_options: Optional[dict] = None)`; `RsimOutput.solver_info: Optional[dict] = None` and `RsimOutput.out_Catch_month_end: Optional[np.ndarray] = None`.

- [ ] **Step 1: Write the failing tests**

```python
def test_an_unknown_method_raises(reco_plain):
    with pytest.raises(ValueError, match="lsoda"):
        rsim_run(_scenario(reco_plain), method="rk45", years=range(1, 2))

def test_the_parity_paths_have_no_solver_info(reco_plain):
    out = rsim_run(_scenario(reco_plain), method="RK4", years=range(1, 2))
    assert out.solver_info is None and out.out_Catch_month_end is None
```

Fixtures (top of the test file, reused by later tasks): `reco_plain` builds Rpath's REco from `tests/data/rpath_reference/ecopath/` **without** stanza tables (its life stages are ordinary groups); `reco_stanzas` builds it with `stanza_groups.csv` / `stanza_indiv.csv`, as in `test_spatial_stanzas.py`. `_scenario(reco, rule="ewe", effort=2.0, years=range(1, 3))` wraps `rsim_scenario(..., fast_group_rule=rule)` and sets `ForcedEffort[:] = effort`.

- [ ] **Step 2: Run, expect FAIL** — `pytest packages/pypath/tests/test_ecosim_continuous.py -q`: the first test fails (no exception: RK4 fallback), the second with `AttributeError`.
- [ ] **Step 3: Implement.** The `ValueError` message lists `"RK4", "AB"` and `CONTINUOUS_METHODS`. Until Task 6, a continuous method raises `NotImplementedError("... phase 1 in progress")`. Remove the silent RK4 fallback.
- [ ] **Step 4: Run, expect PASS**, then the fast core suite: `pytest packages/pypath/tests/ -q -m "not integration and not slow" --ignore=packages/pypath/tests/scripts`. Expect all pass apart from tests that relied on the fallback; fix such a test only if it passed an invalid name by mistake, and list it in the commit message.
- [ ] **Step 5: Commit** with the spec: `git add docs/plans/2026-11-14-continuous-pathway-*.md packages/pypath/...; git commit -m "feat(ecosim): reject unknown integration methods; solver_info field"`.

### Task 2: Derivative flags (exact Hden, R5, DF once)

**Files:**
- Modify: `packages/pypath/src/pypath/core/ecosim_deriv.py` (Hden block near 1930–1949; other mortality near 2128–2170; `_rpath_detritus_derivs` near 780–885 and its call near 2330)
- Test: `packages/pypath/tests/test_ecosim_continuous_deriv.py` (new)

**Interfaces:**
- Produces, all read from `params` and absent by default (parity unchanged):
  - `params["_hden_exact"] = True`: the kernel runs once with `predYY * (_hden0 / _hden)`; `_hden` is neither updated nor written.
  - `rates["HdenConsumption"]`: `QQ[1:, :].sum(axis=0)` (link consumption per predator, import excluded), set whenever `_hden` is present and `rates` is given.
  - `params["DETRITUS_MO_TOT"] = True`: `_M0_detritus` is multiplied by the same `_mo_tot_factor` as `_M0_effective` (R5 closure).
  - `_rpath_detritus_derivs(..., df_once: bool = False)`; `deriv_vector` passes `df_once=params.get("DETRITUS_DF_ONCE", False)`. With `df_once`, a pool's flow to another pool is `det_passed_prop[src] * BB[src] * det_frac[k] / s`, `s` = the sum of `src`'s fractions to pools other than itself (DF counted once).

- [ ] **Step 1: Write the failing tests**
  - `test_exact_hden_leaves_hden_unchanged_and_uses_it`: on a REco EwE-rule params dict with CmCo 2 for every consumer (`scenario.params.CmCo[1:NUM_LIVING+1] = 2.0`, built through `rsim_run`'s setup helper or by hand from `_ewe_handling_setup`), call `deriv_vector` twice with `_hden_exact` and `_hden = H` (H = 1.3 × Hden0): `_hden` is unchanged and both calls return identical arrays; `rates["HdenConsumption"]` equals the column sums of the kernel's QQ at that H.
  - `test_r5_closure_sends_mo_tot_to_detritus`: REco with `PaddP` 0.5 for all living groups (`rsim_scenario` reads `params.model["PaddP"]` if present; otherwise set `scenario.params.PaddP`). With `DETRITUS_MO_TOT`, Σ over living of `rates["MzeroLoss"]` equals the detritus pools' inflow from other mortality, to rtol 1e-12; without the flag it does not (assert the gap > 1e-6).
  - `test_df_once_counts_the_fate_once`: the toy of `test_ecospace_detritus_pass_on.py` (pool A passes 0.5 to B and 0.5 to itself, DetPassedProp 0.2, B_A 10): `gain[3] == 2.0` with `df_once=True`, `1.0` without.
- [ ] **Step 2: Run, expect FAIL** (missing key or flag).
- [ ] **Step 3: Implement** the three flags as above.
- [ ] **Step 4: Run, expect PASS**, plus `pytest packages/pypath/tests/test_ecosim*.py packages/pypath/tests/test_ewe_detritus_fate.py -q` unchanged.
- [ ] **Step 5: Commit** `feat(ecosim): derivative flags for the continuous path (exact Hden, R5 closure, DF once)`.

### Task 3: Handling time as a fixed point; Hden0 recalibrated

**Files:**
- Create: `packages/pypath/src/pypath/core/ecosim_continuous.py`
- Test: `packages/pypath/tests/test_ecosim_continuous_hden.py` (new)

**Interfaces:**
- Consumes: Task 2's `_hden_exact` and `rates["HdenConsumption"]`.
- Produces:
  - `recalibrated_hden0(params_dict: dict, cmco_qb: np.ndarray) -> np.ndarray`: `1 / (1 - q0 / cmco_qb)` for groups with `cmco_qb > 0`, else 1, with `q0_j = QQbase[1:, j].sum() / Bbase[j]` (link flows only; diet import is row 0 and stays out).
  - `solve_hden(y_full, params_dict, forcing, fishing, index, cmco_qb, *, h0=None, tol=1e-12, max_iter=30, fallback=True) -> tuple[np.ndarray, int]`: the H with `H = 1 + H * q(H) / (cmco_qb * index)` on groups with `cmco_qb > 0` and `index > 0` (others H = 1). `q(H)` comes from one `deriv_vector` call with `_hden_exact` on a shallow copy of `params_dict`. Undamped iteration from `h0` (default: `params_dict["_hden0"]`) up to `max_iter`. If the max relative residual is still above `tol` and `fallback` is set, `scipy.optimize.root(F, H, method="hybr", tol=tol)` with `F(H) = g(H) - H`. On failure, raise `RuntimeError` naming "handling time" and the unconverged group indices. Returns `(H, n_evaluations)`.

- [ ] **Step 1: Write the failing tests**
  - `test_the_recalibrated_hden0_is_the_base_fixed_point`: REco, EwE rule, CmCo 2 for every consumer: `solve_hden` at the base state returns `recalibrated_hden0` to max relative difference 1e-12, and at CmCo 1000 `recalibrated_hden0` equals EwE's `CmCo / (CmCo - 1)` to 1e-12 for groups without import.
  - `test_a_perturbed_state_converges`: base biomasses × `[0.2, 3.0, ...]` (alternating), CmCo 2: residual `max|g(H) - H| / H < 1e-10`.
  - `test_no_fixed_point_raises_a_clear_error`: `max_iter=1, fallback=False` at the perturbed state → `RuntimeError` matching `"handling time"`.
  - `test_solving_does_not_touch_params`: `params_dict["_hden"]` and `params_dict["_hden0"]` are unchanged (`np.array_equal`) after `solve_hden`.
- [ ] **Step 2: Run, expect FAIL** (module missing).
- [ ] **Step 3: Implement** both functions.
- [ ] **Step 4: Run, expect PASS.**
- [ ] **Step 5: Commit** `feat(ecosim): handling time as a fixed point of the state (continuous path)`.

### Task 4: The pure right-hand side

**Files:**
- Modify: `packages/pypath/src/pypath/core/ecosim_continuous.py`
- Modify: `packages/pypath/src/pypath/core/ecosim.py` (`RsimParams`: add `EmigRate: Optional[np.ndarray] = None`; fill it in `rsim_params` from `rpath.EmigRate`, living and dead, 0-padded like `Emig`; the parity paths do not read it)
- Test: `packages/pypath/tests/test_ecosim_continuous_rhs.py` (new)

**Interfaces:**
- Consumes: `recalibrated_hden0`, `solve_hden` (Task 3); the Task 2 flags.
- Produces: `class ContinuousRHS` with:
  - `__init__(self, params_dict: dict, fishing_dict: dict, n_bio: int, b0: np.ndarray, stanza_index=None)`: builds a private shallow copy of `params_dict` with `DETRITUS_MO_UNSCALED=False`, `DETRITUS_MO_TOT=True`, `DETRITUS_DF_ONCE=True`, `_hden_exact=True`, `_hden0` recalibrated when `_cmco_qb` is present, `Emig` with R4 (`continuous_emig(emig, emig_rate, emigration_zero_mask)`), and `_det_out` pre-filled once at the unforced base state when `DetPassedProp` is not None (a `deriv_vector` call at `b0` with `_det_out` all NaN and a neutral forcing dict, Hden at Hden0). It records `self.corrections: list[str]` and `self.ignored_switches: list[str]` (names of the numerics switches set True in `params_dict`: `RK4_FAST_EQUILIBRIUM`, `STANZA_STAGE_RATES`, `FTIME_START_OF_MONTH`, `RK4_STEPS`) and logs both once at INFO.
  - `set_month(self, forcing_dict: dict, state_full: np.ndarray, forced_mask: np.ndarray, forced_values: np.ndarray) -> None`: holds the month's forcing, the non-state slots (0 and gears) of `state_full`, and the forced groups.
  - `__call__(self, t: float, y: np.ndarray) -> np.ndarray`: `y` and the return are the biomass states `1..n_bio`. Clip at 0 (count clips), set forced groups, solve Hden, evaluate, zero the forced groups' derivative.
  - `rates(self, y: np.ndarray) -> dict`: the same evaluation with `deriv_vector`'s `rates` dict, plus `"Hden"`.
  - `stats: dict` with `clips`, `hden_evals_max`, `min_state`, reset by `set_month`.
  - `continuous_emig(emig, emig_rate, emigration, b0, n_living) -> np.ndarray`: living groups take `emig_rate` where `emigration == 0 and emig_rate > 0`, else `emig` unchanged.

- [ ] **Step 1: Write the failing tests**
  - `test_the_rhs_is_pure`: call at state A, then at state B, then at A: the two A results are identical (`np.array_equal`), and `params_dict["_hden"]` and `params_dict["_det_out"]` equal their values before the calls.
  - `test_a_balanced_model_starts_balanced` (parametrize `rule` in `"ewe"`, `"rpath"`): REco plain at base effort 1 with neutral forcing; at `b0` every `|dy_i| / b0_i < 1e-9`, also with CmCo 2 under `"ewe"`.
  - `test_a_zero_group_stays_finite`: one prey group set to 0 → all derivatives finite and that group's derivative ≥ 0.
  - `test_fast_groups_are_states`: under `"ewe"`, a group with `NoIntegrate` set (REco has 2) has a nonzero derivative at base biomasses × 1.5.
  - `test_the_rhs_closes_the_detritus_budget`: EwE rule, `PaddP` 0.5 and `MoPred` 0.5 for all living groups, Ftime 0.8: from `rhs.rates(y)`, Σ over living of other-mortality loss plus unassimilated consumption equals the detritus pools' inflow from living groups, to rtol 1e-10 (REco has no pool-to-pool flow; subtract the discards' fishing inflow, `FishTo` pools).
  - `test_living_emigration_uses_the_rate` (`continuous_emig` with arrays): rate-only groups take the rate; groups with an entered Emigration keep `emig`.
  - `test_the_corrections_and_ignored_switches_are_recorded`: under `"ewe"`, `rhs.corrections` contains `"R2"`, `"R4"`, `"R5"`, `"DF once"`, `"Hden0 recalibrated"`; `rhs.ignored_switches` contains `"RK4_FAST_EQUILIBRIUM"` and `"STANZA_STAGE_RATES"`.
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement** `ContinuousRHS` and `continuous_emig`, plus the `RsimParams.EmigRate` field.
- [ ] **Step 4: Run, expect PASS**, plus the fast core suite unchanged.
- [ ] **Step 5: Commit** `feat(ecosim): pure right-hand side for the continuous path`.

### Task 5: Month integrator with quadrature

**Files:**
- Modify: `packages/pypath/src/pypath/core/ecosim_continuous.py`
- Test: `packages/pypath/tests/test_ecosim_continuous_month.py` (new)

**Interfaces:**
- Consumes: any object with `__call__(t, y)`, `rates(y)` and `stats` (Task 4's `ContinuousRHS`, or a test double).
- Produces:
  - `@dataclass MonthResult`: `y_end: np.ndarray`, `y_mean: np.ndarray`, `food_gain_mean: np.ndarray`, `loss_rate_mean: np.ndarray` (mean of `LossPropToB / B`, 0 where B = 0), `nfev: int`, `njev: int`, `nlu: int`, `status: int`, `message: str`. Rate arrays are indexed like `deriv_vector`'s rates (full length); `y_*` are biomass states `1..n_bio`.
  - `integrate_month(rhs, y0, dt, method, rtol, atol, *, on_failure="raise") -> MonthResult`: `solve_ivp(rhs, (0, dt), y0, method=METHOD_NAMES[method], rtol=rtol, atol=atol, dense_output=True, max_step=dt)`; 6-point Gauss–Legendre nodes on `[0, dt]` (`np.polynomial.legendre.leggauss(6)`) on the dense output for `y_mean` and, through `rhs.rates`, for `food_gain_mean` and `loss_rate_mean`; floor `y_end` at 1e-20. `METHOD_NAMES = {"lsoda": "LSODA", "dop853": "DOP853", "radau": "Radau", "bdf": "BDF"}`. A failed solve (`status < 0`) raises `RuntimeError` with the solver message; with `on_failure="warn"` it logs a warning and returns the last state reached.
  - `default_atol(b0_bio: np.ndarray) -> np.ndarray`: `1e-8 * np.maximum(b0_bio, 1e-6 * b0_bio.max())`.

- [ ] **Step 1: Write the failing tests** (a test double `Decay`: `dy = -k * y`, `rates` returns `FoodGain = y` and `LossPropToB = k * y`):
  - `test_the_month_mean_is_the_exact_integral`: `y_mean == y0 * (1 - exp(-k dt)) / (k dt)` to rtol 1e-9 with `dop853`, rtol 1e-12; `loss_rate_mean == k`.
  - `test_every_method_solves_a_month` (parametrize the four methods): `y_end == y0 * exp(-k dt)` to rtol 1e-5.
  - `test_a_failed_month_raises_or_warns`: a double whose derivative returns NaN after t > dt/2; `on_failure="raise"` → `RuntimeError`; `"warn"` → returns finite `y_end` and logs a warning (`caplog`).
  - `test_default_atol_scales_with_biomass`: `default_atol(np.array([1e-6, 1.0, 100.0]))` equals `[1e-12, 1e-8, 1e-6]`.
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run, expect PASS.**
- [ ] **Step 5: Commit** `feat(ecosim): month integrator with quadrature (continuous path)`.

### Task 6: Wire the path into `rsim_run`

**Files:**
- Modify: `packages/pypath/src/pypath/core/ecosim.py` (the run setup before the month loop near 2336–2350, the integration step near 2499, the stanza update near 2605, feeding time near 2677, catch near 2745, the output assembly)
- Modify: `packages/pypath/docs/docs/api/` (the core API page that documents `ecosim`: add `pypath.core.ecosim_continuous`)
- Test: `packages/pypath/tests/test_ecosim_continuous.py`

**Interfaces:**
- Consumes: `ContinuousRHS` (Task 4), `integrate_month`, `MonthResult`, `default_atol` (Task 5).
- Produces: `rsim_run(..., method in CONTINUOUS_METHODS, solver_options={"rtol", "atol", "on_failure"})` returning `RsimOutput` with:
  - `solver_info = {"method", "rtol", "atol", "corrections", "ignored_switches", "months": [ {"month", "nfev", "njev", "nlu", "status", "clips", "hden_evals_max", "min_state"} ... ]}`;
  - `out_Catch` and `out_Gear_Catch` = the month's F (held) × `y_mean` / 12;
  - `out_Catch_month_end` = F × month-end biomass / 12, EwE's booking;
  - `out_Hden` = the Hden solved at the month-end state, when handling time is on.

Behaviour in the continuous branch:
- Before the loop: reject IBM groups (`ValueError`); build `ContinuousRHS` once; skip `_hden_start_pending` and `_det_out_pending`.
- Each month: `rhs.set_month(...)`; `integrate_month`; full state rebuilt from `y_end` with slot 0 and gear slots unchanged; forced groups re-applied.
- Stanza update: `split_update(stanzas_obj, stanza_state, np.ones(len(state)), res_food_gain_mean, res_loss_rate_mean, forced_recs=..., month_index=month - 1)` (unit biomass, so survival is exp(−∫Z dt)), then `split_set_pred` as now.
- Feeding time, for groups with FtimeAdj > 0: the update uses the month means (FoodGain mean over the predator index mean: B mean, or B0 × StanzaPredRatio for stanza groups) and does not skip flagged fast groups.

- [ ] **Step 1: Write the failing tests**
  - `test_lsoda_runs_reco` (`reco_stanzas`, EwE rule, 2 years): finite, positive; `len(out.solver_info["months"]) == 24`; `method="LSODA"` gives identical output (case-insensitive).
  - `test_the_catch_is_the_month_integral`: for each fished group and month, `out_Catch[m, g] == approx(F_m[g] * (B[m-1, g] + B[m, g]) / 2 / 12, rel=1e-2)` (the trapezoid of a smooth month), and `out_Catch_month_end[m, g] == F_m[g] * out_Biomass[m, g] / 12` exactly; F_m is `out.out_FishMort[m]`.
  - `test_stanza_totals_follow_the_age_structure`: `reco_stanzas`: `out.stanza_biomass[1:, idx] == out.out_Biomass[1:, idx]` (rtol 1e-12).
  - `test_forced_biomass_is_held`: force one group to 0.8 × B0 for months 1–6 through `scenario.forcing.ForcedBio`; `out_Biomass[1:7, g] == 0.8 * B0` exactly.
  - `test_ibm_groups_are_rejected`: a scenario with `params.ibm_groups = {g: object()}` → `ValueError` matching `"IBM"`.
  - `test_reusing_a_scenario_leaves_rk4_unchanged`: RK4 on a fresh scenario equals RK4 on a scenario that has run LSODA first (`np.array_equal` on `out_Biomass`).
- [ ] **Step 2: Run, expect FAIL** (`NotImplementedError` from Task 1).
- [ ] **Step 3: Implement** the branch; remove the `NotImplementedError`. Docstring of `rsim_run`: list the methods and say that continuous runs solve the continuous equations with the corrections of the spec's section 11, so they differ from EwE by design.
- [ ] **Step 4: Run, expect PASS**; then the full core and Shiny suites (`pytest packages/pypath/tests/ -q --ignore=packages/pypath/tests/scripts`; `pytest packages/pypath-shiny/tests/ -q --ignore=packages/pypath-shiny/tests/ui`); ruff; `mkdocs build --strict` in `packages/pypath/docs`.
- [ ] **Step 5: Commit** `feat(ecosim): continuous path in rsim_run (lsoda, dop853, radau, bdf)`.

### Task 7: Validation

**Files:**
- Test: `packages/pypath/tests/test_ecosim_continuous_validation.py` (new)
- Create: `docs/parity/continuous_path.md` (one page: what the path is, the corrections, the measured differences from the RK4 path on REco, LT2022 and the Black Sea, and the cost table from these runs)

**Interfaces:**
- Consumes: `rsim_run(..., method=..., solver_options=...)` (Task 6).

- [ ] **Step 1: Write the tests**
  - `test_rk4_converges_to_dop853` (`reco_plain`, `"rpath"` rule, effort 2, 1 year): reference = DOP853 with `rtol 1e-10`; `e_n = max |B_RK4(n) / B_ref - 1|` at month 12 over living groups, with `scenario.params.RK4_STEPS = n` for n in 4, 8, 16. Assert `e_4 / e_8 > 8`, `e_8 / e_16 > 8` and `e_16 < 1e-6`. If REco's fast groups make 4 sub-steps unstable, start at 8 and record that in the test's docstring.
  - `test_lsoda_agrees_with_dop853` (`reco_stanzas`, EwE rule, 2 years, both `rtol 1e-8`): max relative difference < 1e-5.
  - `test_radau_and_bdf_agree_with_lsoda` (`reco_plain`, 1 year, `rtol 1e-8`): < 1e-5.
  - `test_a_balanced_model_stays_balanced` (`reco_plain`, base effort 1, neutral forcing, 50 years, LSODA): `max |B / B0 - 1| < 1e-6` over living groups, under both rules.
  - `test_black_sea_and_lt2022_run` (`integration`; skip without `Data/` or a driver): 10 years LSODA, finite and positive; print the median and max relative difference from the RK4 path for the documentation page.
- [ ] **Step 2: Run, expect PASS.** These test behaviour Tasks 1–6 built, so they should pass on first run. A failure is a finding: investigate with superpowers:systematic-debugging before changing anything.
- [ ] **Step 3: Write `docs/parity/continuous_path.md`** from the numbers the integration test prints and the `solver_info` cost per model.
- [ ] **Step 4: Run the full suites again**, ruff and `mkdocs build --strict`.
- [ ] **Step 5: Commit** `test(ecosim): validate the continuous path; document its differences`.
