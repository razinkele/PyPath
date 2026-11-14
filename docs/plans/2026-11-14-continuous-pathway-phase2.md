# Continuous Pathway, Phase 2 (Feeding Time, Speed, Conservation) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** On the continuous path, feeding time becomes a state, handling time is solved on the consumption kernel alone, EwE mediation is interpolated, and runs report a month-integral mass budget within a stated cost budget. The parity paths stay bit-identical.

**Architecture:** `rsim_run`'s run setup and monthly forcing build move into `_run_setup` and `_month_forcing` in `ecosim.py`, which the tests then use. `ecosim_deriv.py` gains a handling-time hook in the `_hden_exact` branch, a vectorised link kernel (`HdenKernel`), a mediation flag and new `rates` keys; none of these runs on the parity paths. `ecosim_continuous.py` gains the iteration core, the feeding-time state and the budget.

**Tech Stack:** Python ≥ 3.10, NumPy, SciPy (`solve_ivp`, `optimize.root`), pytest.

**Spec:** `docs/plans/2026-11-14-continuous-pathway-design.md`: sections 2.1, 2.3, 2.8, 4, 5.5 and section 9's phase-2 row; decisions 1-9 of section 11.

## Global Constraints

- Parity: `method="RK4"` and `"AB"` stay bit-identical: `np.array_equal` against captures taken at the branch base (a035f33; its only change since 786fc41 is formatting) on the same machine (machines differ by about 2e-6); the existing suite passes unchanged.
- Hden is solved from the fixed cold start Hden0 at every evaluation and never carried between evaluations (spec 2.1).
- `solver_options` keys: `rtol`, `atol`, `corrections`, `on_failure`, `hden_tol`. `hden_tol` defaults to 1e-12 and must be in (0, 1e-8].
- Feeding time obeys `dFt/dt = 12 * (clip(target, L, FtimeMax) - Ft)` per year, for living groups with `FtimeAdj > 0` (planner's call: clipping the target, not the state, makes one forward-Euler month equal the monthly rule).
  - `target` is `_ftime_update_rpath`'s value (`ecosim.py:3350`) at the current state.
  - `L` is `ForagingTimeLowerLimit`; `FtimeMax` is 2.0 when absent.
  - A feeding-time state's `atol` is `1e-2 * rtol` (planner's call: Ft is about 1, so `rtol` governs and `atol` is a floor).
- The default biomass `atol` is `1e-8 * (rtol / 1e-6) * max(B0_i, 1e-6 * max(B0))`.
- `first_step` of month m > 1 is the last full step of month m - 1. Month 1, and any month after a failed or degraded one, leave it to the solver (planner's call: the very last step is truncated at the boundary).
- Mediation: the continuous path interpolates `EweMediation` in float64, exactly 1 at the base. The parity paths keep EwE's truncated float32 lookup.
- `ecosim_deriv.py` changes only in: `rates` keys (under `rates is not None`), the `_hden_solver` hook, `KernelInputs`, `HdenKernel`, the `MEDIATION_INTERPOLATE` flag and `_rpath_detritus_derivs(..., flows=None)`. Planner's call: spec section 6 says "only flags", but the kernel solve and the budget need values that exist only inside `deriv_vector`.
- Style: `ruff format` (0.15.x; it differs from black on some lines) and `ruff check` (E, F, I, W). A module `logger`, no bare `except Exception: pass`; tests reading `Data/` are marked `integration`; no new dependencies.
- **Before every commit**, in every task, run CI's steps (`.github/workflows/ci.yml`) with the dev interpreter (on laguna `/opt/micromamba/envs/shiny/bin/python -m …`, `PYTHONPATH=$PWD/packages/pypath/src:$PWD/packages/pypath-shiny/src`):
  - `ruff check packages/` and `ruff format --check packages/` (CI's format gate: when it fails, CI skips the tests);
  - `mypy packages/pypath/src --config-file packages/pypath/pyproject.toml` (non-blocking in CI; add no new errors in the files you touch);
  - `pytest packages/pypath/tests -q -m "not integration and not slow" --ignore=packages/pypath/tests/integration --ignore=packages/pypath/tests/scripts`.
- Parity capture: `CAP=$HOME/.cache/pypath-phase2` holds `before.npz`, taken in Task 1 Step 1 before any edit and compared against by every later task. If it is lost, regenerate it from the base: `git worktree add ../p2-base a035f33`, then run this branch's `capture_parity.py` with `PYTHONPATH=../p2-base/packages/pypath/src:$PWD/packages/pypath`.
- Not in phase 2: an analytic Jacobian, linear forcing, quasi-steady fast groups, the log transform, Shiny, Ecospace, and stanza cohorts as states (phases 3-4).

## Review Focus

- **The setup move changes a parity number.** `capture_parity.py --compare` reports every array identical (RK4 and AB; stanzas, both rules, feeding time, RiskTime, forced biomass, PaddP, mediation). Test in Task 1.
- **The vectorised kernel drifts from the link loop** (pooling, HandleSwitch, `eat_eff`, mediation, a zero-biomass prey), so Hden is silently wrong. `HdenKernel.q` equals the loop's `HdenConsumption` to 1e-13 per configuration, and the right-hand side records the true residual of its final loop pass. Test in Task 2.
- **The iteration stalls or proposes H ≤ 0.** It falls back to undamped passes, then `hybr`; a non-positive or unconverged H raises `RuntimeError` naming "handling time" and the groups. Test in Task 2.
- **Feeding time at its bounds.** A target above FtimeMax, or B = 0, keeps Ft in [L, FtimeMax], with dFt/dt = 0 and no division at B = 0. Test in Task 3.
- **Feeding-time states leak into biomass quantities** (nutrient sum, budget, `y_mean`, catch, `atol`, stanza update): with them present, `NutFree + sum(B) == NutTot * shape` at every quadrature node and the budget residual stays at solver tolerance. Test in Task 5.

---

### Task 1: Shared run setup; parity capture

**Files:**
- Modify: `packages/pypath/src/pypath/core/ecosim.py`:
  - move into helpers the option checks (1764-1814), the setup (1822-2483, except the output arrays and the ecotracer and fleet initialisation) and the forcing build (2543-2597);
  - the month-label wraps (2793, 2914) skip a message that already names the month.
- Modify: `ecosim_continuous.py` (`continuous_options`).
- Create: `packages/pypath/tests/scripts/capture_parity.py`. It is committed but not collected by pytest. Planner's call: phases 3-4 will refactor again.
- Modify: `packages/pypath/tests/continuous_support.py`, and `NUM_LIVING + 2` becomes `NUM_LIVING + NUM_DEAD` in `test_ecosim_continuous*.py` (deferred minor).
- Test: `packages/pypath/tests/test_ecosim_run_setup.py` (new).

**Interfaces:**
- Produces, private in `ecosim.py` (planner's call: no public API promise; the setup uses `globals()` and closures). The moved `warnings.warn(..., stacklevel=3)` becomes `stacklevel=4` (one more frame). Loop locals (`dt`, `rk4_steps`, crash variables, `annual_qlink`, feeding-time and risk flags, the gear lookup, `out_fishmort[0]`, `out_hden`) stay in `rsim_run`.
  - `@dataclass RunSetup`: `params_dict`, `fishing_dict`, `state` (after the stanza start), `n_months`, `n_years`, `n_groups`, `stanzas`, `stanza_state`, `stanza_idx`, `ftime` (a copy of `start_state.Ftime`), `risk0` and `handling` (`_ewe_handling_setup`'s tuple or None).
  - `_run_setup(scenario, years=None, *, mediation=None) -> RunSetup`: moved verbatim, with its side effects.
  - `_month_forcing(scenario, setup, month: int, ftime: np.ndarray) -> dict`: the dict built at 2563, including `StanzaPredRatio`. Forced biomass and forced F stay in the loop.
- Produces `continuous_options(method, solver_options: Optional[dict], n_bio: int) -> dict` with keys `method`, `rtol`, `atol` (None or an ndarray), `on_failure` and `corrections`. It raises the same `ValueError`s as now, and a `ValueError` matching "method" for a non-string method (deferred minor; planner's call: `ValueError`, as for an unknown name).
- Produces `run_dicts(scen, method="RK4", years=range(1, 2))` on the helpers (month-1 forced biomass applied, deep-copied, no monkeypatch). Docstrings: `_hden` is EwE's start value, not advanced; `_build`'s cached `(model, params)` must not be mutated (deferred minors).

- [ ] **Step 1: Capture, then write the failing tests.**
  - `capture_parity.py OUT.npz` runs, 2 years each: REco plain `"ewe"` effort 2 (RK4, AB); REco stanzas `"ewe"` and `"rpath"` (RK4, AB each); REco plain `"ewe"` with `FtimeAdj` 0.5, `RiskTime[13]` 0.3, `PaddP` 0.5, `ForcedBio[0:6, 12]` 0.8 × B0 (RK4, AB); `test_ewe_mediation._scenario` with `SHAPE_A` on (ZOO, FISH) (RK4); LT2022 s16 and Black Sea s1 (RK4) when `Data/` and a driver exist. It saves `out_Biomass`, `out_Catch`, `out_Gear_Catch`, `out_Hden`, `stanza_biomass` and `end_state`. `--compare A B` prints `identical: <n> arrays` (exit 0) or the differing keys (exit 1).
  - Before editing `ecosim.py`, run `mkdir -p $CAP && PYTHONPATH=$PWD/packages/pypath/src:$PWD/packages/pypath /opt/micromamba/envs/shiny/bin/python packages/pypath/tests/scripts/capture_parity.py $CAP/before.npz` (`CAP` as in Global Constraints); keep it, as every later task compares against it. The name does not match `test_*.py`, so pytest never collects it.

  Then write the tests:
  - `test_the_setup_reproduces_rsim_runs_dicts`, parametrised over (`reco_plain`, `"ewe"`), (`reco_stanzas`, `"ewe"`) and (`reco_stanzas`, `"rpath"`). Against the old monkeypatch capture, kept as a private helper: the same `params_dict` keys, every array `np.array_equal` (NaN equal) except `_hden` (advanced by EwE's start step), every `forcing_dict` and `fishing_dict` entry equal, the same state. It rules out a setup that drops, reorders or recomputes an entry.
  - `test_lowercase_parity_names_run`: `"rk4"` and `"ab"` give `out_Biomass` `np.array_equal` to `"RK4"` and `"AB"` (deferred minor).
  - `test_a_non_string_method_raises`: `method=4` raises `ValueError` matching "method".
  - `test_a_month_is_named_once`: patch `integrate_month` to raise `RuntimeError(f"month 1 (year {scenario.start_year}): boom")`; the message contains "month 1" exactly once.
- [ ] **Step 2: Run, expect FAIL**: `pytest packages/pypath/tests/test_ecosim_run_setup.py -q`. Expected failures: `ImportError` (`_run_setup`), `AttributeError` (`'int' object has no attribute 'upper'`) and a doubled prefix.
- [ ] **Step 3: Implement** the helpers and the label guard, and rewrite `run_dicts`.
- [ ] **Step 4: Run, expect PASS.** Then `capture_parity.py $CAP/after.npz` and `--compare $CAP/before.npz $CAP/after.npz` (`identical: …`, exit 0); the fast core suite (`pytest packages/pypath/tests/ -q -m "not integration and not slow" --ignore=packages/pypath/tests/scripts`), the Shiny suite (`pytest packages/pypath-shiny/tests/ -q --ignore=packages/pypath-shiny/tests/ui`) and the CI steps of Global Constraints: all pass.
- [ ] **Step 5: Commit** `refactor(ecosim): shared run setup and month forcing; parity capture script`.

### Task 2: Handling time on the kernel, accelerated; `hden_tol`

**Files:**
- Modify: `ecosim_deriv.py`: `KernelInputs` and `HdenKernel` after the loop kernel (206-356); the hook and `rates["Hden"]` in the Hden block (1954-1972).
- Modify: `ecosim_continuous.py` and `ecosim.py`: `solver_info["hden_tol"]`, renamed month keys.
- Test: `test_ecosim_continuous_hden.py`, `test_ecosim_continuous_rhs.py`.

**Interfaces:**
- Produces, in `ecosim_deriv.py`:
  - `@dataclass(frozen=True) KernelInputs`: `BB`, `preyYY`, `predYY` (unscaled), `hden0`, `VV`, `DD`, `QQbase` (after the switching, arena and search remaps), `link_prey`, `link_pred`, `ScrambleSelf`, `HandleSelf`, `PredPredWeight`, `PreyPreyWeight`, `HandleSwitch`, `COUPLED`, `med_multipliers`, `eat_eff`, `index` (as in the non-exact branch) and `cmco_qb`.
  - The hook: with `_hden_exact`, if `params["_hden_solver"]: Callable[[KernelInputs], np.ndarray]` is set, it returns the Hden for the final kernel pass. `rates["Hden"]` is that Hden.
  - `class HdenKernel`: `__init__(self, inputs: KernelInputs)` precomputes the Hden-independent link factors (`qbase × PYY_term × dd_term × med`) and the B ≤ 0 and qbase ≤ 0 masks; `q(self, hden: np.ndarray) -> np.ndarray` returns `Σ_{prey≥1} Q[prey, j]`, as the loop's `HdenConsumption`. Only PDY, `vv_term` and PredSuite depend on Hden. Planner's call: the loop kernel is 68 % of a derivative call (0.588 of 0.870 s over 300 REco calls), so iterating on it would save only about 30 %.
- Produces, in `ecosim_continuous.py`:
  - `iterate_hden(g, h0, active, *, tol=1e-12, max_iter=30, fallback=True, accel="bracketed", coupled=False) -> tuple[np.ndarray, int]`. It returns the iterate whose residual, `max|g(H) - H| / H` over `active`, is ≤ `tol`, and never an H ≤ 0 (deferred minor: the `hybr` acceptance lost the H > 0 guard). `accel="none"` gives plain undamped passes. `solve_hden_kernel` passes `coupled=True` when any active link has `ScrambleSelf > 0`.
  - `solve_hden(...)` keeps its signature, delegates to `iterate_hden` and serves as the full-derivative reference.
  - `solve_hden_kernel(inputs: KernelInputs, *, tol=1e-12, max_iter=30, fallback=True) -> tuple[np.ndarray, int]`.
  - `ContinuousRHS(..., hden_tol: float = 1e-12)` installs its solver as `_hden_solver`, owns its `_hden` array (deferred minor: no longer the caller's), and makes one `deriv_vector` call per evaluation. `stats["hden_passes_max"]` replaces `hden_evals_max` (planner's call: rename, since the count is now of kernel passes); `stats["hden_residual_max"]` comes from the final pass.
  - `recalibrated_hden0` raises `ValueError` naming the groups where `q0 >= cmco_qb` (deferred minor).
  - `continuous_options` accepts `hden_tol`. Planner's call on the bound 1e-8: LSODA's finite-difference Jacobian perturbs states by about 1.5e-8 relative.

**Algorithm (planner's call: a per-component bracketed secant, Illinois).** With `ScrambleSelf` 0 (the default, `ecosim.py:571`) each predator's g depends on its own H only (cross-dependence measured as exactly 0 on the Black Sea); g is non-decreasing and bounded with F(1) ≥ 0, so a bracket exists. Spikes (2026-11-14, from Hden0 at perturbed states, CmCo 2): REco 9-10 passes (undamped 24-29, Anderson m = 5 13-15, plain secant 6-8); Black Sea 3-19, all converging (undamped up to 117 and one stuck after 300; plain secant failed 2 of 8).

```
F(h) = g(h) - h, per active component
a = h0; Fa = F(a); return a if res <= tol
b = where(Fa > 0, 1e6, 1.0); Fb = F(b)        # F(1) >= 0; F(1e6) < 0
repeat up to max_iter:
    x = (a*Fb - b*Fa) / (Fb - Fa)              # midpoint where not finite
    Fx = F(x); return x if res(x, Fx) <= tol
    replace the endpoint whose F has Fx's sign; halve the kept
    endpoint's F when it is kept twice running (Illinois)
coupled=True: undamped passes up to max_iter instead
fallback: root(F, method="hybr", tol=tol); accept only finite H > 0 with
residual <= tol, else RuntimeError("handling time did not converge ...", groups)
```

- [ ] **Step 1: Write the failing tests**
  - `test_the_vector_kernel_matches_the_link_loop`, parametrised: REco at CmCo 2; one prey at 0; `HandleSelf` and `ScrambleSelf` 0.5; `HandleSwitch` 2.0; `params["_eat_eff"]` 0.7; a mediation collection with group multipliers. A stub `_hden_solver` captures `KernelInputs` and returns `h = Hden0 × [1.3, 0.8, …]`. Assert `HdenKernel(inputs).q(h)` equals that call's `rates["HdenConsumption"]` (`rtol=1e-13`, `atol=0`).
  - `test_kernel_and_full_solves_agree`: `solve_hden_kernel` and `solve_hden` differ by < 1e-11 relative at `_perturbed` and five log-normal states (σ 0.7, `np.random.default_rng(0)`).
  - `test_the_bracketed_iteration_needs_few_passes`: at the alternating state, at most 12 passes, and fewer than half of `accel="none"`. A double `g_j(h) = 1 + 0.9 h_j / (0.1 + h_j)` per component converges, as does a coupled double through `coupled=True`.
  - `test_a_non_positive_hden_is_never_accepted`: `g(h) = -1 - h` (fixed point -0.5) raises `RuntimeError` matching "handling time".
  - `test_one_derivative_per_evaluation`: 10 RHS calls make exactly 10 `ecosim_continuous.deriv_vector` calls.
  - `test_the_rhs_reports_the_true_residual`: after a call at the perturbed state, `stats["hden_residual_max"]` is finite and ≤ 1e-11.
  - `test_hden_tol_is_an_option`: `{"hden_tol": 1e-9}` runs with `solver_info["hden_tol"] == 1e-9`; 0, 1e-7 and `"x"` raise `ValueError` matching "hden_tol".
  - Deferred minors: `rhs.rates(y)["Hden"]` has a residual ≤ 1e-11 under the full-derivative `g` of `solve_hden`; `cmco_qb` of 1e-9 makes `recalibrated_hden0` raise `ValueError`; `rhs._pd["_hden0"]` is `np.array_equal` to `recalibrated_hden0`; the purity helper `same` also compares scalars, lists, and `ActiveLinkArray` by `.prey`/`.pred`.
  - `hden_evals_max` becomes `hden_passes_max` in `test_ecosim_continuous.py:143`, `test_ecosim_continuous_rhs.py:234` and `test_ecosim_continuous_validation.py:270`.
- [ ] **Step 2: Run, expect FAIL** (`ImportError: KernelInputs`).
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run, expect PASS**: `pytest packages/pypath/tests/test_ecosim_continuous*.py -q`, then the CI steps of Global Constraints. The parity compare reports identical.
- [ ] **Step 5: Commit** `perf(ecosim): handling time on a vectorised kernel with a bracketed iteration; hden_tol`.

### Task 3: Feeding time as a state

**Files:**
- Modify: `ecosim_continuous.py`.
- Modify: `ecosim.py`: the continuous branch (2768-2822) skips the monthly feeding-time update (2941-3003) and the month-start RiskTime rates.
- Test: `packages/pypath/tests/test_ecosim_continuous_ftime.py` (new).

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) FeedingTime`: `groups: np.ndarray` (int); the full-length arrays `adj`, `qbopt`, `cap`, `risk_time`, `risk0` and `mo`; and `lower: float`.
  - `feeding_time(params, risk0) -> Optional[FeedingTime]`: living groups with `FtimeAdj > 0`, or None.
  - `ftime_target(ft, adj, qopt, food_gain, index, lower, cap) -> np.ndarray`: `_ftime_update_rpath` vectorised. It keeps `ft` where `food_gain`, `index` or `qopt` is ≤ 0, and clips to `[lower, cap]`.
  - `ContinuousRHS(..., feeding=None)`: `y = [B_1..B_n_bio, Ft_groups]`, `n_state = n_bio + len(groups)`. Each evaluation writes `clip(Ft, lower, cap)` into a copy of the month's `Ftime`, never into `forcing_dict["Ftime"]` (that array is `_ftime_current`), and takes FoodGain and FoodLoss from its own `rates` and `index` from `hden_index`. Qopt is `_risk_qopt` at the current FoodLoss and B (planner's call: RiskTime becomes instantaneous, replacing phase 1's month-start predation over month-end biomass).
  - `set_month` raises `ValueError` unless the mask and the values have length `NUM_GROUPS + 1` (deferred minor).
  - `rates(y)` adds `"Ftime"`.
  - `integrate_month(..., n_bio=None)`, default `len(y0)`: the rates, loss rates and `y_mean` use the first `n_bio` states.
- Produces, in `rsim_run`: `atol` extended by `1e-2 * rtol` per feeding-time state; `_ftime_current[groups] = res.y_end[n_bio:]` each month; `solver_info["ftime_groups"]`; `months[m]["ftime_range"]` (`[min, max]` of those states at month end; planner's call: new keys, as are those of Tasks 5-6).

- [ ] **Step 1: Write the failing tests.** Setup for all: REco plain, `"ewe"`; `FtimeAdj` 0.5 for every living group; `RiskTime[13]` 0.3; state B0 × alternating 1.3, 0.7; `start_state.Ftime[g] = 2.0` for one group whose raw target exceeds 2.
  - `test_one_euler_month_is_the_monthly_rule`: for every feeding-time group, `Ft + dFt / 12 == approx(_ftime_update_rpath(Ft, adj, qopt, FoodGain, index, lower=L, cap=FtimeMax), rel=1e-14)`, with FoodGain, FoodLoss and `index` from `rhs.rates(y)` and `qopt` from `_risk_qopt` for group 13. The capped group rules out a missing clip; a group at B = 0 gives `dFt == 0`. A second case on `reco_stanzas` (`"ewe"`, `FtimeAdj` 0.5 for every living group) checks the same identity for the stanza predators, whose `index` is `Bbase * StanzaPredRatio` rather than B (as `_ftime_index` under the EwE rule).
  - `test_feeding_time_stays_within_its_bounds`: one year of LSODA with `FtimeAdj` 1.0 and effort 3. Every `ftime_range` lies within `[L, FtimeMax]`, and every state is finite.
  - `test_feeding_time_is_integrated`: a spy sees no `_ftime_update_rpath` call; `end_state.Ftime[groups]` equals the last `y_end[n_bio:]`; the `atol` given to `solve_ivp` has length `n_bio + len(groups)` with tail `1e-2 * rtol`.
  - `test_reusing_a_scenario_with_feeding_time_leaves_rk4_unchanged`: `np.array_equal`.
  - Deferred minors: `test_a_forced_group_feeds_the_others` (a prey's derivative differs with its predator forced to 0.8 × B0 rather than B0; the caller's `state_full` is unchanged); `set_month` with a mask of length `n_bio` raises `ValueError`.
- [ ] **Step 2: Run, expect FAIL** (`ImportError: FeedingTime`).
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run, expect PASS**, `test_feeding_time_and_risk_time_run` included; then the CI steps of Global Constraints and the parity compare (identical).
- [ ] **Step 5: Commit** `feat(ecosim): feeding time as a state of the continuous path`.

### Task 4: Mediation interpolation; corrections listed where they act

**Files:**
- Modify: `packages/pypath/src/pypath/core/mediation.py` (`setup` 263, `lookup` 292, `values` 300, `link_factors` 321).
- Modify: `ecosim_deriv.py:1847`; `ecosim_continuous.py`.
- Test: `packages/pypath/tests/test_ecosim_continuous_mediation.py` (new). In `test_ecosim_continuous_rhs.py:176` and `test_ecosim_continuous.py:336`, set `EmigRate[12] = 0.1` so that R4 applies.

**Interfaces:**
- Produces:
  - `EweMediation.xbase64`, a float64 value set in `setup`.
  - An `interpolate=False` keyword on `lookup(shape, x, *, interpolate=False)`, `values` and `link_factors`. With `interpolate=True`:
    - `x` sums in float64, without EwE's 1e-10;
    - `ratio = imedbase * (x / xbase64)`, so the base gives `imedbase` exactly;
    - the value is `np.interp(ratio, 1..1200, points) / points[imedbase - 1]`, clamped at the end points as EwE does.

    The `False` path is unchanged. Planner's call: dropping the 1e-10 and summing in float64 make the base value exactly 1, so the balanced start holds.
  - `deriv_vector` passes `interpolate=params.get("MEDIATION_INTERPOLATE", False)`, and the RHS sets that flag when `EweMediation` is present.
  - `rhs.corrections` lists an item only where it acts (planner's call, revisiting phase 1's fixed-list ruling: `solver_info` should say what changed): R2 if the scenario's `DETRITUS_MO_UNSCALED` is True; R5 if `_mo_tot` is present; DF once if `DetPassedProp` is present; R4 if `continuous_emig` changes a living group; "Hden0 recalibrated" if handling time is on; "mediation interpolated" if `EweMediation` is present. The last two stay under `corrections=False`.

- [ ] **Step 1: Write the failing tests**
  - `test_the_interpolated_lookup`, on `_mediation(SHAPE_A, 300, [0, 1], [], [1, xbase])`: `lookup(0, xbase64, interpolate=True) == 1.0` exactly; at `ratio = 300.5` it equals `(points[299] + points[300]) / 2 / points[299]` (`rel=1e-15`); at `x = xbase × (1 ± 1e-9)` it differs from 1 (EwE's lookup does not); `x = 0` gives `points[0] / points[299]` and `10 × xbase` gives `points[1199] / points[299]`.
  - `test_the_continuous_path_interpolates_mediation`: on the toy `_scenario` with `SHAPE_A`, a `link_factors` spy sees `interpolate=True` on every LSODA evaluation and `False` on every RK4 one; `"mediation interpolated"` is in `solver_info["corrections"]`.
  - `test_corrections_list_what_applies`: REco under `"rpath"` lists neither DF once nor R2; REco under `"ewe"` without `EmigRate` omits R4.
- [ ] **Step 2: Run, expect FAIL** (`TypeError`, unexpected keyword `interpolate`).
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run, expect PASS**: `pytest packages/pypath/tests/test_ewe_mediation.py packages/pypath/tests/test_mediation*.py packages/pypath/tests/test_ecosim_continuous*.py -q`. The parity compare reports identical.
- [ ] **Step 5: Commit** `feat(ecosim): interpolated mediation and applicable corrections on the continuous path`.

### Task 5: Conservation: month budget, nutrient check, residuals in `solver_info`

**Files:**
- Modify: `ecosim_deriv.py`: `rates` keys (blocks at 2266, 2392); `_rpath_detritus_derivs(..., flows: Optional[dict] = None)` (780); `_ewe_nutrient_free` beside `_ewe_nutrient_factor` (1469, unchanged).
- Modify: `ecosim_continuous.py`, `ecosim.py`.
- Test: `packages/pypath/tests/test_ecosim_continuous_budget.py` (new).

**Interfaces:**
- Produces `rates` keys: `"Production"` (the living kernel's production × RecruitmentMult), `"Import"` (`import_q`), `"Discards"` (catch reaching each pool), `"DetExport"` (detritus that reaches no pool: under the EwE rule DetritusOut × B less the flows passed to other pools, otherwise link flows to destinations outside 1..n_bio), `"NutFree"` (as used; NaN when off) and `"NutFloor"` (bool).
- Produces `_ewe_nutrient_free(biomass_sum, base_sum, base_free_prop, shape) -> tuple[float, float]`, returning `(nut_free, nut_free0)`.
- Produces `exchange_rate(rates, y_full, params_dict, forcing) -> float` (planner's call: the spec's term list made concrete), the net external flux of groups 1..n_bio. In: Production of producers and of groups with QB 0, Import, Immigration + ForcedMigrate, `_det_inflow`. Out: respiration (`(1 - Unassim) × FoodGain`, less Production for consumers), landed catch `FishingLoss - Discards`, `EmigLoss`, pool `Emig × B`, `DetExport`. Internal transfers are not used, so an unclosed transfer leaves a residual. A residual of ΔB − ∫Σdy would be a tautology.
- Produces `ContinuousRHS.exchange(rates, y) -> float` and `stats["nutrient_floor"]`, the number of evaluations at the floor.
- `MonthResult` gains `exchange_mean` (NaN without `rhs.exchange`) and `pool_net_mean`, the mean of `DetGain - DetLoss`. Both come from the six existing quadrature evaluations.
- `months[m]` gains `budget_residual = (Σ y_end - Σ y0 - dt × exchange_mean) / Σ y0` over the biomass states (before the stanza overwrite, forced re-apply and floor), `pool_budget_residual` (max over pools of `|ΔB_d - dt × pool_net_mean_d| / B_d0`) and `nutrient_floor`. Planner's calls: both residuals are None in a month with forced biomass (its flows are imposed); pools are checked against their own `DetGain - DetLoss`, since a source-side pool budget would duplicate `_rpath_detritus_derivs` and phase 1's `test_the_rhs_closes_the_detritus_budget` pins the source side.
- Deferred minors: quadrature nodes are clipped at 0 before `rhs.rates` and the loss-rate division; when the rates fall back on the warn path, `y_mean` falls back too; `integrate_month`'s docstring covers `stats`.

- [ ] **Step 1: Write the failing tests.** T = Σ|Production| + Σ FoodGain.
  - `test_the_derivative_closes_the_system_budget`: `|Σ dy[:n_bio] - rhs.exchange(r, y)| <= 1e-12 × T` at B0 × alternating 1.3, 0.7. Parametrised over REco `"ewe"`, REco `"rpath"`, and an `"ewe"` variant with `PaddP` 0.5, `MoPred` 0.5, start Ftime 0.8, `EmigRate[12]` 0.1 and `Immigration` zeros with 0.5 at group 12 (REco has `Immigration` None).
  - `test_unclosed_corrections_leave_a_residual`: the same variant with `corrections=False` gives a residual `> 1e-6 × T`. This rules out an `exchange_rate` derived from Σ dy.
  - `test_the_month_budget_closes`: month 1 on DOP853 with `rtol` 1e-10 and `atol = default_atol(b0) × 1e-4`. Then `|budget_residual| < 1e-8` and `pool_budget_residual < 1e-7`.
  - `test_solver_info_reports_the_budget`: one year of LSODA at the defaults, with `FtimeAdj` 0.5. Every month has `|budget_residual| < 1e-5`; with `ForcedBio[0:6, 12]` set, months 1-6 are None.
  - `test_free_plus_bound_nutrient_is_the_total` (`NutBaseFreeProp` 0.5, `FtimeAdj` 0.5): at the six month-1 nodes, `r["NutFree"] + Σ y[:n_bio] == (Σ B0 / 0.5) × NutForcing` (`rel=1e-12`); `months[0]["nutrient_floor"]` is 0 at `NutForcing` 1.0 and > 0 at 0.4. The identity holds by construction, so the test checks the wiring (summed states, shape) and the floor flag.
  - Deferred minors: `test_df_once_without_other_pools_passes_nothing` (a pool linked only to itself, with `df_once`, has finite gains and passes 0 to other pools); `test_df_once_is_inert_without_detpassedprop` (under the `"rpath"` rule the derivative is `np.array_equal` with and without `DETRITUS_DF_ONCE`).
- [ ] **Step 2: Run, expect FAIL** (`KeyError: 'Production'`).
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run, expect PASS**, then the CI steps of Global Constraints and the parity compare (identical). A residual above its bound is a finding for superpowers:systematic-debugging, not a bound to move.
- [ ] **Step 5: Commit** `feat(ecosim): month-integral mass budget, nutrient check and residuals in solver_info`.

### Task 6: First-step reuse, tolerances, timing budget

**Files:**
- Modify: `ecosim_continuous.py`, `ecosim.py`.
- Test: `packages/pypath/tests/test_ecosim_continuous_speed.py` (new).

**Interfaces:**
- Produces `integrate_month(..., first_step=None)` (forwarded to `solve_ivp` when given); `MonthResult.last_step` (`np.diff(sol.t)[-2]` with at least two steps, since the last is truncated at `dt`; else the only step); `default_atol(b0_bio, rtol=1e-6)`, scaled by `rtol / 1e-6`. Planner's call: phase 1 measured the rtol 1e-10 reference at only 4.7e-9 under the default `atol` (deferred minor).
- Produces `solver_info["rhs_evals"]` (every RHS evaluation of the run, start checks, quadrature and output rates included: the deferred minor "start check not counted") and `months[m]["first_step"]`.

**Budget (planner's call; spec section 4's 10-20x assumes a phase-3 Jacobian).** Phase 2 controls the cost of an evaluation, not their number, so the primary gate is CPU per RHS evaluation over a run: at most 1.5 × one exact single-pass `deriv_vector` call with `rates` at the base state. The RK4-ratio floor is RHS calls per month ÷ RK4's month, in single-pass derivative calls (call counts from `docs/parity/continuous_path.md`; costs measured 2026-11-14):

| Model | Floor | RK4-ratio target | Phase 1 (page) |
|---|---|---|---|
| REco, no feeding-time states | 28 / 4.3 ≈ 6.5x (`perf_counter`) | 12x | 11-21x; 18x and 75x (CmCo 1000, 2) measured today |
| Black Sea, no feeding-time states | 96 / 5.2 ≈ 19x (`process_time`) | 35x (Task 7) | 914x |
| LT2022, 17 feeding-time states | about 32x (`process_time`) | reported, not asserted | 125x |

LT2022: 223 / 9.6 ≈ 23x with phase 1's counts, plus about 17 × 5.1 finite-difference Jacobian columns a month for its 17 feeding-time states.

**Measuring on a loaded machine.** Deterministic counts gate the fast suite. The timing tests (`slow`, `benchmark`) use `time.process_time()` in one process: one untimed warm-up of each method, then 5 interleaved RK4/LSODA pairs on `copy.deepcopy` of one scenario; ratio = `min(LSODA) / min(RK4)`; a failure prints every sample and `os.getloadavg()`.

- [ ] **Step 1: Write the failing tests**
  - `test_the_next_month_starts_from_the_last_step`: a `solve_ivp` spy over 3 months of REco. Month 1 has no `first_step`; months 2-3 have `first_step == months[m-1]["last_step"]`. After a month-1 failure under `on_failure="warn"`, month 2 has no `first_step`.
  - `test_first_step_is_passed_through`: on `Decay`, `dt / 4` reaches `solve_ivp`.
  - `test_default_atol_follows_rtol`: `default_atol([1e-6, 1, 100], rtol=1e-10) == [1e-16, 1e-12, 1e-10]`. The existing call is unchanged.
  - `test_one_derivative_per_rhs_evaluation` (fast): REco `"ewe"`, CmCo 2, one year: `ecosim_continuous.deriv_vector` calls `== solver_info["rhs_evals"] + 1` (the DetritusOut pre-fill), and every month has `hden_passes_max <= 10`.
  - `test_reco_is_within_its_budget` (`slow`, `benchmark`, parametrised over CmCo 1000 and 2): REco plain, `"ewe"`, effort 2, one year. The RK4 ratio is ≤ 12, and CPU per RHS evaluation (LSODA CPU / `rhs_evals`) is ≤ 1.5 × the min-of-200 CPU of one exact `deriv_vector` call.
- [ ] **Step 2: Run, expect FAIL** (`TypeError`: `first_step`).
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run, expect PASS**, including `pytest packages/pypath/tests/test_ecosim_continuous_speed.py -q -m benchmark`. Record both ratios for Task 7.
- [ ] **Step 5: Commit** `perf(ecosim): first-step reuse, atol follows rtol, timing budget`.

### Task 7: Re-measure and document

**Files:**
- Modify: `test_ecosim_continuous_validation.py`: rename `test_black_sea_and_lt2022_run` to `test_ten_years_on_three_models` (deferred minor); print `rhs_evals` and the residuals; assert `hden_passes_max` ≤ 25 (spike: 19 at most). The timing asserts go in a separate `test_ten_year_cost` (`integration`, `slow`, `benchmark`) that uses Task 6's protocol (warm-up, interleaved pairs, min of samples, `process_time`): for all three models, CPU per RHS evaluation ≤ 1.5 × one exact derivative call; the Black Sea RK4 ratio ≤ 35; LT2022's RK4 ratio printed. Controller's ruling: a one-shot timing inside the integration test would fail on a loaded machine.
- Modify: `docs/parity/continuous_path.md`: corrections-table rows for feeding time (state), mediation (interpolated) and RiskTime (instantaneous), removed from Known limitations; a budget section; the cost tables.
- Modify: `packages/pypath/docs/docs/api/core.md`: `hden_tol`, the budget keys, and the sentence on "quadrature means", which over-claims (fleet dynamics does not use them; deferred minor).
- Modify: the design doc §11, adding item 10: this plan's planner's calls, as accepted.

- [ ] **Step 1:** Run `pytest packages/pypath/tests/test_ecosim_continuous_validation.py -q -s` (no marker filter): all pass. A failed bound is a finding, not a threshold to move.
- [ ] **Step 2:** Write the page from the printout and Task 6's ratios.
- [ ] **Step 3:** Run the full core and Shiny suites, `ruff check` and `ruff format --check packages/`, `mkdocs build --strict` in `packages/pypath/docs`, and the parity compare (identical).
- [ ] **Step 4: Commit** `docs(ecosim): continuous path phase 2: feeding time, cost, budget`.

---

## Out of scope

- Jacobian reuse across monthly restarts: SciPy's LSODA takes a Jacobian callable, not a matrix to reuse; it needs phase 3's analytic Jacobian.
- numba for the loop kernel: it would change the parity paths' arithmetic.
- Spec section 4's 10-20x for the Black Sea and LT2022: the call-count floors alone are about 19x and 32x. Lowering them needs fewer evaluations: phase 3's analytic Jacobian and linear forcing.
- A separate continuous month loop: it would duplicate the fleet, ecotracer, catch and stanza code both paths share; Task 1 extracts the setup instead.

## Deferred minors not taken

- Already fixed in phase 1's final wave: the stale Hden and moTot comments (`ecosim_deriv.py:1946-1953`, `:2180-2185`), the `atol` shape check, `degraded` means, the clips assertion, the wall-time caveat, the anchovy sentence.
- `annual_Qlink` and ecotracer month-end snapshots, `annual_QB` never written: every path, pre-existing, documented.
- The catch integral is approximate under fleet mediation: F is held for the month by design (spec 2.6).
- The unattributed Aurelia (23 %) and LT2022 0.44 % median differences: an attribution study, not code; Task 7 re-measures them.
- Phase-1 Task 3's "early return unconfirmed" and "TDD order": superseded by `iterate_hden`, or process only.
