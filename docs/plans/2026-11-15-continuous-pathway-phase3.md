# Continuous Pathway, Phase 3a (Cost, Options, Pooled Handling Time) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring the continuous path to the spec's 10-20 x RK4 on all three benchmark models, and add the spec's opt-in options: linear forcing, log-transformed states and quasi-steady fast groups. Fix the pooled handling-time failure at its root (a porting defect in the scramble weights). The parity paths stay bit-identical, except for pooled models in Task 4, which move to Rpath's values.

**Architecture:**
- **Cost per evaluation.** A safeguarded Newton solves handling time, using an analytic `dq/dH` from `HdenKernel`. A vectorised twin of the sparse link loop does the final consumption pass when `params["_KERNEL_VECTOR"]` is set, with a link-loop cross-check once a month.
- **Number of evaluations.** A finite-difference Jacobian callable (`JacobianReuse`) reuses Jacobians across monthly restarts.
- **Options.** `ecosim_continuous.py` gains the forcing interpolant, the log-state wrapper and the quasi-steady reduction.
- **Changes outside it.** `ecosim_deriv.py` changes only behind `_KERNEL_VECTOR`, and in an allocation; `rsim_run`'s continuous branch passes the new options through.
- **The analytic Jacobian** (spec section 9) is designed here as Task 9, gated on open question 1.

**Tech Stack:** Python ≥ 3.10, NumPy, SciPy 1.18 (`solve_ivp`), pytest. No new dependencies. numba is not used: it is an optional extra, `pypath-ewe[numba]`, but nothing here needs it.

**Spec:** `docs/plans/2026-11-14-continuous-pathway-design.md`: sections 2.2, 2.5, 4 and 6, section 9's phase-3 row and section 11, items 1-10. Exemplar: `docs/plans/2026-11-14-continuous-pathway-phase2.md`.

## Scope decision: 3a numerics now, 3b Shiny after

Phase 3 is split.
- **3a (this plan): the numerics.** It lives in `packages/pypath`.
- **3b (sketched at the end): Shiny.** It lives in `packages/pypath-shiny`, which is released separately (1.4.0), and has its own test stack.

3b consumes only the public API (`method`, `solver_options`, `solver_info`) and could even start on phase 2's API. The two share no files, and the review skills differ (numerical against UI).

## Measurements behind this plan

Spikes outside the repo (`/tmp/claude-1000/-home-razinka-PyPath/3de35854-f464-4f7d-a354-0e5f9d63fe72/scratchpad/p3/m1-m19*.py`), SciPy 1.18.1, laguna, 2026-11-15. "rhs/mo" counts every right-hand-side evaluation per month (`solver_info["rhs_evals"]`). Ten-year figures use the page's protocol (warm-up, three interleaved runs, minimum, `process_time`).

| Question | Finding |
|---|---|
| Where LSODA's calls go | REco and the Black Sea form no Jacobian (the Black Sea runs Adams, 42.5 steps a month). LT2022: 4.83 Jacobians × 41 states = 73 % of `nfev`; a free exact Jacobian would take it from 280 to 82 rhs/mo. |
| FD Jacobian reused across months (M7) | LT2022: BDF 102 → 61.6 rhs/mo; LSODA, also reusing while the state moved < 10 %, 280 → 104 (within 1.0e-7 of no reuse); Radau 120 → 92. |
| What drives the Black Sea | Its monthly `NutForcing` (0.7-1.15): with it constant, 87.8 → 32.7 `nfev`/mo. |
| Linear forcing (M5, M14) | Costs more everywhere: Black Sea 87.8 → 113 `nfev`/mo; REco without stanzas under seasonal forcing, LSODA, 81.2 (piecewise-constant, restarts), 102 (tent, restarts), 111 (tent, no restarts). Restarts cost only under constant forcing (20.1 → 11.3), and all three benchmark models have stanzas. |
| Per evaluation (M8, M12) | Newton handling time: 8.2 → 4.4 passes (Black Sea). With the vectorised final pass, CPU per evaluation 1.31 → 0.77 ms (Black Sea), 1.69 → 0.80 (REco), 1.69 → 0.87 (LT2022); biomass moves ≤ 1.1e-10. |
| Combined, ten years (M13) | REco LSODA 4.0 x (today 8.2 x); Black Sea LSODA 19.6 x (33.2 x); LT2022 BDF 9.9 x at 86.4 rhs/mo (today's LSODA 61.6 x at 342.7), LSODA 14.8 x at 146.8. No clips. |
| Quasi-steady (M6) | On LT2022's three flagged groups, 9.8 % off at effort × 3 (Zooplankton turns over at 25 /yr); on Detritus alone (3406 /yr), 8.3e-5 off and the largest eigenvalue 3321 → 98.5. Black Sea: 4.5 % off. An inner `hybr` eats LSODA's gain; DOP853 602 → 361 rhs/mo. |
| Log states (M10) | 5-11 % fewer `nfev`, within 1.3e-6. REco at effort 100: linear states clip 131 times; log states need an absorbing floor (else month 52 fails), then clip 0 times for +9 % evaluations. |
| Pooled failures (M9, M17) | CmCo 2 fails in month 2, CmCo 1000 in month 3 or 6. More passes, damping, Anderson, `hybr`, `lm`, Newton and `hden_tol` 1e-8 (the page's workaround) all fail. **Root cause:** PyPath's port of Rpath's scramble weights (`ecosim.py:782-791`) omits Rpath's `VV <- VV * QQ / B_prey` (`rsim.params`), so 5 weights are negative and the predator suite hits a clamp. With Rpath's formula the weights equal Rpath's (dumped by Rscript) to 1.4e-13 and both cases complete (17 and 4 passes, min B/B0 0.83). |

## Prerequisites (controller's note, 2026-11-15)

Two fixes in flight touch code this plan builds on. Execution starts only after both have merged into main, and this branch is then rebased onto that main:
- **PR #122:** the `mediation=` IndexError in `fishing_mortality` (fleet multipliers indexed by the 0-based fleet).
- **Branch `fix/old-findings`:**
  - ScrambleSelf/HandleSelf, given per group, are expanded to the links of the keyed group;
  - an offset `years` range raises `ValueError`;
  - a fished detritus pool loses its catch under the `"rpath"` rule.

  It edits `rsim_params` and `_build_link_matrix` next to Task 4's weight block.

"Branch base" below means that rebased main. Record its SHA in Task 1, and use it wherever 0677cd6 appears.

## Global Constraints

- **Parity:** `method="RK4"` and `"AB"` stay bit-identical: `capture_parity.py --compare` against `$CAP/before.npz` (`CAP=$HOME/.cache/pypath-phase3`), taken at the branch base on the same machine. If lost: `git worktree add ../p3-base <base SHA>`, then run this branch's `capture_parity.py` with `PYTHONPATH=../p3-base/packages/pypath/src:$PWD/packages/pypath`. The existing suite passes unchanged except for the tests this plan edits by name. The one planned exception: Task 4 changes the `reco_plain_ewe_pooled/*` arrays (Rpath's weights) and re-baselines.
- **`solver_options`:** phase 2's `rtol`, `atol`, `corrections`, `on_failure`, `hden_tol`, plus `jacobian` (`"reuse"`, default, or `"solver"`, today's behaviour), `forcing_interpolation` (`"constant"`, default, or `"linear"`), `positivity` (`"clip"`, default, or `"log"`), `fast_groups` (`"states"`, default, or `"quasi-steady"`) and `qs_rate` (float > 0, default 365.0 /yr). Other keys and values raise `ValueError` naming the key. The three model options are opt-in and labelled (spec section 6).
- **Handling time** is solved from the cold start Hden0 at every evaluation, never carried; `hden_tol` stays (0, 1e-8], default 1e-12; coupled (pooled) maps keep undamped passes.
- **Purity:** every evaluation is a function of `(t, y)` and of data fixed by `set_month` (the quasi-steady start and chord Jacobian, the interpolant). The Jacobian cache is not part of the right-hand side.
- **`ecosim_deriv.py`** changes only in `_compute_consumption_vector` (new), `HdenKernel.q_and_dq`, the `_KERNEL_VECTOR` dispatch in `_run_kernel` (`:2091`) and the `_det_flows` allocation under `rates is not None` (deferred minor).
- **Style:** `ruff format` (0.15.x, CI's gate, not black), `ruff check` (E, F, I, W); a module `logger`; no bare `except Exception: pass`; no new dependencies.
- **Tests:** tests reading `Data/` are `integration`; timing tests are `slow` and `benchmark` and use phase 2's protocol (one untimed warm-up per method, interleaved pairs on `copy.deepcopy` of one scenario, `COST_SAMPLES = 3`, ratio `min(candidate) / min(RK4)` in `process_time`; a failure prints every sample and `_loadavg()`).
- **Before every commit**, in every task, run CI's steps with `/opt/micromamba/envs/shiny/bin/python -m …` and `PYTHONPATH=$PWD/packages/pypath/src:$PWD/packages/pypath-shiny/src:$PWD/packages/pypath`: `ruff check packages/`; `ruff format --check packages/`; `mypy packages/pypath/src --config-file packages/pypath/pyproject.toml` (non-blocking; no new errors in touched files); `pytest packages/pypath/tests -q -m "not integration and not slow" --ignore=packages/pypath/tests/integration --ignore=packages/pypath/tests/scripts`; then `capture_parity.py $CAP/after.npz` and `--compare $CAP/before.npz $CAP/after.npz` (`identical: …`, exit 0).

## Review Focus

- **A stale Jacobian across a jump** (forcing step, stanza overwrite, year boundary) makes Newton fail repeatedly, or BDF takes a reused matrix as fresh. BDF and Radau get a fresh Jacobian at every request after the month's first; LSODA on a retry at the same `t`; with REco's effort × 5 from month 7, `"reuse"` matches `"solver"` to 1e-5 with no more evaluations. Test in Task 3.
- **The parity path picks up the vector pass, or the weight fix leaks beyond pooled models.** A spy sees no `_compute_consumption_vector` call on RK4 or AB; after Task 4 the compare differs in `reco_plain_ewe_pooled/*` only. Tests in Tasks 2 and 4.
- **Option combinations:** log states with the Jacobian callable (FD in log space), quasi-steady where a selected group is forced (excluded and listed), linear forcing with log states: each runs a year, finite, with `solver_info` naming the options. Test in Task 7.
- **Monthly means at series ends and with short forcing arrays:** every month's mean is preserved, ends included, also when an array is shorter than the run (`_month_forcing`'s defaults fill in), and non-negative keys stay ≥ 0. Test in Task 5.
- **Quasi-steady with poor separation:** LT2022 with `qs_rate=20` selects Zooplankton (25 /yr); `qs_separation < 10` and a WARNING. Test in Task 7.

---

### Task 1: Parity capture: pooled, fleet dynamics, `out_FishMort`; base from 0677cd6

**Files:**
- Modify: `packages/pypath/tests/scripts/capture_parity.py`.

**Interfaces:**
- `OUTPUTS` gains `out_FishMort` (deferred minor) and `annual_Qlink`.
- `CASES` gains three cases:
  - `"reco_plain_ewe_pooled"` (RK4, AB): `_reco(False, "ewe")` with `p.ScrambleSelf = p.HandleSelf = np.full(len(p.PreyFrom), 0.5)` and `p.CmCo[1:NL+1] = 2.0`.
  - `"fleet_dynamics"` (RK4): `test_fleet_dynamics_integration._make_fleet_model()` with `create_fleet_econ_params(1, n_links)`, price 10, `cap_base_growth` 0.3, `cap_depreciate` 0.05, run through `rsim_run(..., fleet_dynamics=fd)`.
  - `"reco_forced_f"` (RK4): `ForcedFRate[0, 5] = 0.3`.
- The pooled case changes by design in Task 4 (see there); every other key must stay identical through Task 8.

- [ ] **Step 1: Write the new cases first** (a script, not `src`). Then take the base capture:
  - `git worktree add ../p3-base 0677cd6`;
  - `mkdir -p $CAP && PYTHONPATH=../p3-base/packages/pypath/src:$PWD/packages/pypath python packages/pypath/tests/scripts/capture_parity.py $CAP/before.npz`.

  Expect more than 81 arrays and no "skipped" line on laguna.
- [ ] **Step 2: Check it against the branch:** `capture_parity.py $CAP/after.npz` and `--compare`. Expect `identical`, since `src` has not changed. This validates the new cases.
- [ ] **Step 3:** Run the CI steps of Global Constraints.
- [ ] **Step 4: Commit** `test(ecosim): parity capture covers pooling, fleet dynamics, forced F and FishMort`.

### Task 2: Cost per evaluation: Newton handling time, vectorised final pass

**Files:**
- Modify `ecosim_deriv.py`: `HdenKernel` (401-495) gains `q_and_dq`; new `_compute_consumption_vector` after `_compute_consumption_sparse_python` (206-352); the dispatch in `_run_kernel` (2091); `_det_flows` (deferred minor).
- Modify `ecosim_continuous.py` (`iterate_hden`, `solve_hden_kernel`, `ContinuousRHS`) and `ecosim.py` (`months[m]["kernel_check"]`).
- Test: `test_ecosim_continuous_hden.py`, `test_ecosim_continuous_speed.py`, new `test_ecosim_continuous_vector.py`.

**Interfaces:**
- Consumes phase 2's `HdenKernel`, `KernelInputs`, `iterate_hden` and the `_hden_solver` hook.
- Produces `HdenKernel.q_and_dq(hden) -> tuple[np.ndarray, np.ndarray]`: `q` as `q()` and `dq[j] = ∂q_j/∂H_j = Σ_links base·(vv_term − pdy·vv/den²·∂eff/∂pdy)·(−pdy/H_j)`, `den = max(vv − 1 + max(eff, 1e-10), 1e-10)`; a clamped `eff` contributes 0; `ValueError` when `coupled()`.
- Produces `_compute_consumption_vector(QQ, BB, VV, DD, QQbase, preyYY, predYY, link_prey, link_pred, n_links, **kw)`, the keywords of `_compute_consumption_sparse_python`: the same per-link arithmetic and order, writing only the links the loop writes; pooling follows the loop's `use_pooling` (HandleSelf acts with ScrambleSelf 0); suites by `np.bincount` over the same links.
- `_run_kernel` takes the twin when `params.get("_KERNEL_VECTOR", False)`; only `ContinuousRHS` sets it.
- `iterate_hden(..., accel="newton", dg=None)`, `dg(H)` the diagonal `∂g_j/∂H_j`, independent components only. `solve_hden_kernel` uses it when not coupled, with `g` and `dg` from one `q_and_dq` per pass (cached by H).
- The first counted evaluation after `set_month` also runs `deriv_vector` without the flag and records `stats["kernel_check"] = max|HdenConsumption_vec − HdenConsumption_loop| / max|HdenConsumption_loop|` (FoodGain when handling time is off); `hden_residual_max` uses the loop there. New `months[m]["kernel_check"]`.

**Algorithm (safeguarded Newton per component; the tests do not determine it):**

```
h = h0; lo = 1; hi = 1e6                 # F(1) >= 0 because g >= 1
repeat up to max_iter:
    q, dq = q_and_dq(h); F = g - h; return h if max|F|/h <= tol over active
    lo = where(F > 0, max(lo, h), lo); hi = where(F < 0, min(hi, h), hi)
    x = h - F / (dg - 1)                 # g = 1 + h q / D, dg = (q + h dq) / D, D = cmco_qb index
    h = where(not finite or x <= lo or x >= hi, (lo + hi) / 2, x)
unconverged: the bracketed secant from h, then hybr (as now)
```

- [ ] **Step 1: Write the failing tests.**
  - `test_q_and_dq_matches_central_differences`: REco, CmCo 2, alternating 1.3/0.7; `dq` against central differences with δ = 1e-6 H_j, `rtol=1e-7`.
  - `test_newton_needs_few_passes`: at `_perturbed` and five log-normal states (σ 0.7, rng 0), at most 7 passes and fewer than the bracketed secant's, within 1e-11 of it (spike: 5-6 against 9-10).
  - `test_the_vector_pass_equals_the_link_loop`: phase 2's kernel cases (pooling, a prey at 0, HandleSwitch 2.0, `_eat_eff` 0.7, mediation multipliers) plus the combined case (deferred minor), HandleSelf 0.5 with ScrambleSelf 0, and REco's own arrays; QQ `np.array_equal`, else `rtol=1e-13`; the pooling case asserts `coupled()` (deferred minor).
  - `test_the_parity_paths_never_take_the_vector_pass`: a spy sees no call of `_compute_consumption_vector` on RK4 or AB, some on LSODA.
  - `test_kernel_check_is_reported`: one year of LSODA on REco, every month `kernel_check <= 1e-13`.
  - Deferred minors: `test_a_component_without_a_bracket_converges` (`g = 1.5` for one component, `0.5 + 0.9 h` for another, `fallback=True`: converges, H > 0); `test_ecosim_continuous_rhs.py:318` becomes `0.0 <= res`; the module docstring's long line is wrapped.
  - Edits: `test_one_derivative_per_rhs_evaluation` expects `rhs_evals + 1 + n_months` calls (the monthly cross-check) and `hden_passes_max <= 6`; `_exact_call_cpu` also pops `_KERNEL_VECTOR`, so the reference stays one loop pass; `EVAL_COST_BOUND` 2.5 → 1.2 (measured: 0.74 x at CmCo 1000, 0.83 x at CmCo 2; passes at most 4 over a CmCo-2 year).
- [ ] **Step 2: Run, expect FAIL** (`AttributeError: q_and_dq`).
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run, expect PASS:** `pytest packages/pypath/tests/test_ecosim_continuous*.py -q`, the CI steps, the parity compare, and `-m benchmark test_ecosim_continuous_speed.py` (record the ratios).
- [ ] **Step 5: Commit** `perf(ecosim): Newton handling time and a vectorised consumption pass on the continuous path`.

### Task 3: Jacobian reuse across monthly restarts

**Files:** Modify `ecosim_continuous.py` (new `JacobianReuse`; `integrate_month(..., jac=None)`; `continuous_options`) and `ecosim.py` (continuous branch, `solver_info`). Test: new `test_ecosim_continuous_jacobian.py`.

**Interfaces:**
- Consumes `integrate_month`, `ContinuousRHS.evals`, `continuous_options`.
- Produces `class JacobianReuse(method: str, atol: np.ndarray, rtol: float, *, move: float = 0.1)`: `for_month(fun) -> Callable[[float, np.ndarray], np.ndarray]`, once per month; `stats = {"jac_fresh": int, "jac_reused": int}`, reset by `for_month`.
- **Rule** (planner's call, from M7): a month's first request returns the run's last Jacobian when one of the right shape exists. Later requests: BDF and Radau always compute fresh (SciPy asks again only after a failed Newton iteration); LSODA reuses unless the request repeats the previous `t` (a retry) or `max|y − y_J| / (|y_J| + 1e-12) > move`. Fresh means forward differences, `h_k = 1.49e-8 × max(|y_k|, atol_k / rtol)`, `n_state` evaluations counted in `rhs.evals`.
- `integrate_month(..., jac=None)` forwards `jac` except for DOP853. `continuous_options` accepts `jacobian`; `METHOD_NAMES` becomes the single source and `ecosim.CONTINUOUS_METHODS = tuple(METHOD_NAMES)` (deferred minor).
- `solver_info["jacobian"]`; per month `jac_fresh`, `jac_reused`. With a callable, LSODA's `nfev` excludes Jacobian columns, so cost is gated on `rhs_evals`.

- [ ] **Step 1: Write the failing tests.**
  - `test_the_fd_jacobian_matches_central_differences`: REco at CmCo 2, alternating state, `hden_tol=1e-14`, `rtol=1e-4` on columns with norm > 1e-8 (planner's estimate: the FD noise is `hden_tol / h`, about 7e-7).
  - `test_reuse_rules` (a counting fake `fun`, n = 3): BDF reuses at month 2's first request and computes fresh at the second; LSODA reuses at a new `t` within 10 %, computes fresh at the same `t` and after a 20 % move; DOP853's `solve_ivp` gets no `jac`.
  - `test_reuse_agrees_with_the_solver` (fast): REco with stanzas, effort 2, × 5 from month 7, two years, BDF, Radau and LSODA: `|ΔB| / B ≤ 1e-5`, `rhs_evals(reuse) ≤ rhs_evals(solver)`, no clips (spike: BDF 848 against 1377, 9.5e-10; Radau 920 against 1399, 1.6e-7; LSODA identical).
  - `test_jacobian_option`: `"x"` raises `ValueError` matching "jacobian"; `solver_info["jacobian"]` echoes the choice.
  - `test_lt2022_bdf_evaluations` (`integration`, `slow`): ten years of BDF, `rhs_evals / 120 <= 100` (spike: 86.4), no clips, `jac_fresh` per month mean ≤ 0.5 and max ≤ 8 (spike: 0.23, 5).
- [ ] **Step 2: Run, expect FAIL** (`ImportError: JacobianReuse`).
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run, expect PASS**, then the CI steps and the parity compare.
- [ ] **Step 5: Commit** `perf(ecosim): reuse finite-difference Jacobians across monthly restarts`.

### Task 4: Pooled handling time: Rpath's scramble weights (root cause)

**Files:**
- Modify `packages/pypath/src/pypath/core/ecosim.py` (`rsim_params`, the weight block at 775-805) and `ecosim_continuous.py` (`solve_hden`, the coupled error message).
- Create `packages/pypath/tests/data/rpath_reference/ecosim/suite_weights.csv` and a README line on how it was made.
- Modify `docs/parity/continuous_path.md` (Known limitations) and `packages/pypath/docs/docs/api/core.md`.
- Test: new `test_rpath_suite_weights.py`; `test_ecosim_continuous_hden.py`; the corrections tests in `test_ecosim_continuous_mediation.py`.

**Interfaces:**
- Consumes Task 1's pooled capture case.
- Produces, in `rsim_params`, Rpath's formula: `vv_rate = vv × qq / B_prey` (0 where B_prey ≤ 0), then `aa = 2·qq·vv_rate / (vv_rate·B_pred·B_prey − qq·B_pred)` and the normalisation as now. Its denominator is `B_pred·qq·(VV − 1)`, positive for VV > 1.
- Planner's call: fix at the source, not clip on the continuous path. The port differs from Rpath by up to 11.9 in a weight; the fix makes it equal Rpath to 1.4e-13. Weights act only through pooling, so a model with ScrambleSelf and HandleSelf 0 is bit-identical.
- `solve_hden` decides coupling with `HdenKernel`'s rule (deferred minor). A coupled failure says `"handling time did not converge for a pooled (ScrambleSelf > 0) map …"`.

- [ ] **Step 1: Make the oracle and write the failing tests.**
  - The oracle: `Rscript -e 'library(Rpath); s <- rsim.params(rpath(REco.params), mscramble=2, mhandle=1000, preyswitch=1); write.csv(data.frame(from=s$PreyFrom, to=s$PreyTo, ppw=s$PredPredWeight, pyw=s$PreyPreyWeight), "…/suite_weights.csv", row.names=FALSE)'` gives 167 rows, none negative. Rpath is installed on laguna.
  - `test_suite_weights_match_rpath`: REco's `PredPredWeight` and `PreyPreyWeight` equal the CSV's, link by link (`atol=1e-12`).
  - `test_pooled_reco_runs`, CmCo 2 and 1000: two years of LSODA on REco plain, `"ewe"`, effort 2, ScrambleSelf and HandleSelf 0.5 on all 167 links. It completes with `hden_passes_max <= 30` (spike: 17, 4) and `min(B/B0) > 0.5` over living groups (spike: 0.83). Today CmCo 1000 raises in month 3, and CmCo 2 takes 109 passes and collapses to 1e-21.
  - Deferred minor: REco `"rpath"` omits R5 and DF once; `corrections=False` with `EweMediation` keeps "mediation interpolated".
- [ ] **Step 2: Run, expect FAIL** (weights differ by up to 11.9; CmCo 1000 raises in month 3).
- [ ] **Step 3: Implement.** On the page, replace the pooled limitation and drop the `hden_tol` workaround (disproved by M9).
- [ ] **Step 4: Run, expect PASS.** Then the CI steps. The compare must list exactly the `reco_plain_ewe_pooled/*` keys as differing. Then `cp $CAP/after.npz $CAP/before.npz` (keep the old one as `before_t1.npz`). Later tasks compare against the new baseline.
- [ ] **Step 5: Commit** `fix(ecosim): Rpath's scramble-weight formula (pooled handling time)`.

### Task 5: Linear forcing (opt-in model option)

**Files:** Modify `ecosim_continuous.py` (new `ForcingInterpolant`; `ContinuousRHS.set_month`, `_evaluate`, `rates`, `exchange`; `integrate_month`) and `ecosim.py` (the monthly series from `_month_forcing`). Test: new `test_ecosim_continuous_forcing.py`.

**Interfaces:**
- Consumes `_month_forcing`, `ContinuousRHS`, `integrate_month`.
- Interpolated: `ForcedPrey`, `ForcedMort`, `ForcedSearch` with `PP_forcing`, `ForcedMigrate`, `ForcedActresp`, `NutForcing`, `ForcedSearchLink`, `ForcedVulLink`. Held monthly: `ForcedBio`, `ForcedRecs`, `ForcedEffort`, forced F and catch, fleet dynamics, quotas, `StanzaPredRatio`. Planner's call: spec 2.6 keeps effort monthly-held (against 2.5's "a step in effort becomes a ramp"), which keeps the catch integral exact.
- Produces `ForcingInterpolant(monthly: dict[str, np.ndarray], signed=frozenset({"ForcedMigrate"}))`, arrays of shape (n_months, …): `at(month: int, s: float) -> dict` (s the fraction of the month) and `mean(month: int, key) -> np.ndarray` (exact).
- **Algorithm (the tests do not determine it):** a continuous tent through `(0, b_k)`, `(½, c_k)`, `(1, b_{k+1})`; inside the series `b_k = min((m_{k−1} + m_k)/2, 2m_{k−1}, 2m_k)` for a non-negative key and `(m_{k−1} + m_k)/2` for a signed one; `b_0 = m_0`, `b_N = m_{N−1}`; `c_k = 2m_k − (b_k + b_{k+1})/2`. The mean is exactly `m_k`, the limiter keeps it ≥ 0, constant input gives exactly constant output.
- **`"smooth"`** (the user's answer 5): per month, one quadratic through the same limited boundary values: `v(s) = b_k(1 − s) + b_{k+1}s + 6(m_k − (b_k + b_{k+1})/2)·s(1 − s)`.
  - Its mean is exactly `m_k`.
  - It is continuous at the boundaries, with no kink inside the month.
  - It is ≥ 0 for non-negative keys, because `0 ≤ b ≤ 2m` on both sides.
  - It keeps the 6-point quadrature unchanged.
  - Measured (spike `m5c_interp.py`, mode "quad"): 16 % fewer evaluations than constant forcing on the Black Sea, where the tent costs 29 % more.
  - `ForcingInterpolant(..., kind="linear" | "smooth")`.
- Restarts stay at every month boundary (dropping them measured costlier: 111 against 102).
- Quadrature: the tent has a kink at mid-month, so under `"linear"` `integrate_month` uses 3-point Gauss-Legendre on each half-month (still 6 evaluations); under `"constant"` the 6-point rule is unchanged.
- `set_month(..., forcing_at=None)`; `__call__(t, y)` and `rates(y, t=0.0)` merge `forcing_at(t)`; `integrate_month` passes the node times; the rates carry the forcing used and `exchange` reads `ForcedMigrate` from it; `forcing.get("Ftime")` replaces the direct index (deferred minor).
- `solver_info` echoes `forcing_interpolation`, `positivity` and `fast_groups`; `corrections` stays the defect list.

- [ ] **Step 1: Write the failing tests.**
  - `test_the_tent_preserves_means`: random non-negative series (rng 0) and the signed `[1, −1, 2]`; every `mean` equals `m_k` (`rel=1e-14`, also by 2-point Gauss on each half); ≥ 0 at 101 points per month for non-negative input; continuous at boundaries; `[1, 0, 1]` gives exactly 0 in the middle month; a constant series gives exactly the constant.
  - `test_constant_forcing_is_unchanged`: REco with stanzas, two years, `"linear"` against `"constant"`: `rtol=1e-9` (the split quadrature moves the month means in the last digits).
  - `test_series_ends_and_short_arrays`: REco without stanzas, two years, `ForcedPrey` of 18 rows, seasonal on group 13: the forcing at a node (`rates(y, t)["_forcing"]`) equals `at(k, t / dt)`; `mean(k, key)` equals `_month_forcing`'s value in every month (`rel=1e-14`), ends included; months 19-24 are exactly 1.
  - `test_linear_forcing_runs_the_black_sea` (`integration`): two years, status 0, no clips; prints `nfev` (spike: 113 against 87.8), not asserted.
  - `test_the_smooth_interpolant`: the same properties as `test_the_tent_preserves_means` for `kind="smooth"` (mean, ≥ 0 at 101 points per month, continuity at the boundaries, exact constants), and `"smooth"` against `"constant"` on constant forcing at `rtol=1e-12`.
  - `test_smooth_forcing_runs_the_black_sea` (`integration`): two years, status 0, no clips; `nfev` printed (spike: 16 % below constant forcing), not asserted.
  - `continuous_options` accepts `forcing_interpolation` in `{"constant", "linear", "smooth"}`.
- [ ] **Step 2: Run, expect FAIL** (`ImportError: ForcingInterpolant`).
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run, expect PASS**, then the CI steps and the parity compare.
- [ ] **Step 5: Commit** `feat(ecosim): mean-preserving linear forcing on the continuous path (opt-in)`.

### Task 6: Log-transformed biomass states (opt-in)

**Files:**
- Modify: `ecosim_continuous.py` (`integrate_month(..., positivity="clip")`) and `ecosim.py`.
- Test: new `test_ecosim_continuous_log.py`.

**Interfaces and rules** (planner's calls; the absorbing floor and `atol` come from M10). Consumes `integrate_month` and `JacobianReuse` (Task 3). Produces:
- Every biomass state is `z = ln max(y, 1e-20)`; feeding-time states stay linear.
- The wrapper evaluates `B = exp(max(z, z_f))`, with `z_f = ln 1e-20`, and `dz = dy / B`. Where `z ≤ z_f`, `dz = max(dz, 0)`: a group at the floor can only grow.
- The `atol` of a log state is `max(atol_i / y0_i, rtol)`, with `y0` the month's start: the linear tolerance's equivalent at the month-start biomass.
- `sol.y` and the dense output are transformed back before the means. A forced group is held, as now.
- Under `"reuse"`, `JacobianReuse` differentiates the wrapped function (FD in z).
- New key: `solver_info["positivity"]`.

- [ ] **Step 1: Write the failing tests.**
  - `test_log_states_never_clip`: REco plain at effort 100 for 5 years. `"clip"` has `sum(clips) > 0` (spike: 131); `"log"` has `clips == 0`, every state ≥ 1e-20, and agrees with `"clip"` to 1e-4 on entries above 1e-2 × B0 (spike: 2.5e-5; 1.5e-4 above 1e-4 × B0, 8.3e-4 above 1e-6 × B0).
  - `test_log_states_agree`: REco with stanzas, two years, `"log"` against `"clip"` ≤ 1e-5 (spike: 1.3e-6).
  - `test_log_tolerance`: on a fake `rhs`, the `atol` reaching `solve_ivp` equals the rule.
  - `test_positivity_option`: `"x"` raises `ValueError`.
- [ ] **Step 2: Run, expect FAIL** (`TypeError`: `positivity`).
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run, expect PASS.** Then the CI steps and the parity compare.
- [ ] **Step 5: Commit** `feat(ecosim): log-transformed biomass states on the continuous path (opt-in)`.

### Task 7: Quasi-steady fast groups (opt-in model reduction)

**Files:** Modify `ecosim_continuous.py` (new `QuasiSteady`; `ContinuousRHS`) and `ecosim.py`. Test: new `test_ecosim_continuous_qs.py`.

**Interfaces:**
- Consumes `ContinuousRHS`, `JacobianReuse`, `ForcingInterpolant`, the log states (combination test).
- **Selection** (planner's call; M6 shows EwE's 24 /yr flag selects groups that are not fast): groups flagged by `NoIntegrate` whose base loss rate (`LossPropToB / B`, unforced base) exceeds `qs_rate` (default 365 /yr), groups forced in any month of the run and stanza groups excluded for the whole run; none qualifying logs a WARNING and runs them as states. LT2022: Detritus (3406 /yr); REco (Phytoplankton 240) and the Black Sea (291): none.
- Produces `QuasiSteady(groups, rhs)`: the reduced state drops `groups`; each evaluation solves `dB_f/dt = 0` by a chord Newton with `J_ff` by forward differences at `set_month`'s start, from the month-start `B_f`, halving steps to keep `B_f > 0`, to 1e-10 relative in at most 20 iterations, recomputing `J_ff` once at the iterate if the residual stops falling; failure raises `RuntimeError` naming "quasi-steady" and the groups. All of it is month data, so the evaluation stays pure.
- `solver_info["qs_groups"]`; `solver_info["qs_separation"]`, the smallest selected base loss rate over the largest |Re λ| of the reduced Jacobian at the month-1 start (FD, once), with a WARNING below 10 (LT2022 Detritus: 3406 / 98.5 = 35; Zooplankton: 25 / 12 = 2); per month `qs_inner_max`.
- A selected group's `out_Biomass` is its solved value. Budget residuals are None under quasi-steady groups (the reduction drops `dB_f/dt`, so the budget does not close to solver tolerance).

- [ ] **Step 1: Write the failing tests.**
  - `test_the_constraint_holds`: `_stiff_model`, `qs_rate=50` (Phyto, 100 /yr): `|dB_f/dt| ≤ 1e-10 × TotGain_f` at the six month-1 nodes (planner's estimate: the spike solved with `hybr`, not the chord Newton).
  - `test_qs_agrees_on_the_stiff_model`: one year against full states at rtol 1e-8, ≤ 1e-5 (spike: 1.1e-7).
  - `test_qs_on_lt2022` (`integration`, `slow`): effort × 3, two years, DOP853 with QS against LSODA with full states, ≤ 1e-3 on living groups (spike: 8.3e-5); `qs_groups` is Detritus only.
  - `test_poor_separation_warns` (`integration`): LT2022, `qs_rate=20`: `qs_separation < 10` and the WARNING in caplog.
  - `test_option_combinations` (Review Focus): REco with stanzas, one year, each of `{"positivity": "log", "jacobian": "reuse"}`, `{"fast_groups": "quasi-steady", "qs_rate": 30}` with Zooplankton (19) forced at 0.8 × B0 in months 1-6 (excluded and logged; Phytoplankton, 240 /yr, quasi-steady) and `{"forcing_interpolation": "linear", "positivity": "log"}`: finite, `solver_info` echoing every option.
- [ ] **Step 2: Run, expect FAIL** (`ImportError: QuasiSteady`).
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run, expect PASS**, then the CI steps and the parity compare.
- [ ] **Step 5: Commit** `feat(ecosim): quasi-steady fast groups on the continuous path (opt-in)`.

### Task 8: Re-measure, cost gates, document

**Files:**
- `test_ecosim_continuous_validation.py`:
  - `test_ten_year_cost` takes a `(model, method)` parameter: `("reco", "lsoda")`, `("black_sea", "lsoda")`, `("lt2022", "bdf")`, `("lt2022", "lsoda")`;
  - `test_ten_years_on_three_models` prints `kernel_check`, `jac_fresh` and `jac_reused`.
- `test_ecosim_continuous_speed.py`: drop `_spy_solve_ivp`'s unused `fail_first` (deferred minor).
- `docs/parity/continuous_path.md`, `packages/pypath/docs/docs/api/core.md` (also its long line at 335; deferred minor) and the design document's section 11, item 11.

Consumes every earlier task.

**Bounds** (asserted under the ten-year protocol; Global Constraints):

| Model, method | Today | Spike (M13) | Bound |
|---|---|---|---|
| REco with stanzas, LSODA | 8.2 x | 4.0 x | ≤ 6 x, and CPU per evaluation ≤ 1.2 × one exact loop call |
| Black Sea, LSODA | 32.5 x | 19.6 x | ≤ 25 x (was 40) |
| LT2022, BDF | (not measured) | 9.9 x | ≤ 14 x |
| LT2022, LSODA | 61.6 x | 14.8 x | ≤ 20 x |

- [ ] **Step 1:** Run `pytest packages/pypath/tests/test_ecosim_continuous_validation.py -q -s` with no marker filter. A bound that fails is a finding for superpowers:systematic-debugging, not a number to move.
- [ ] **Step 2: Rewrite the page.**
  - Cost tables, by method; "which method" (BDF for stiff models, per open question 2).
  - The options section, labelled "changes the model" for linear forcing and quasi-steady.
  - A Jacobian section: LSODA's `nfev` no longer counts the columns.
  - The pooled fix.
  - The final-wave residuals: merge the duplicated `annual_Qlink` bullet, and fix the `atol_ftime … and min_state` sentence.
- [ ] **Step 3:** Run the full core and Shiny suites, `ruff`, `mkdocs build --strict` (use a scratch venv if mkdocs is missing) and the parity compare.
- [ ] **Step 4: Commit** `docs(ecosim): continuous path phase 3a: cost, options, pooled fix`.

### Task 9 (designed; not built in 3a, per the user's answer 1): Analytic Jacobian

Consumes `q_and_dq` (Task 2) and `JacobianReuse` (Task 3); produces `AnalyticJacobian(rhs).for_month(fun)` with `JacobianReuse`'s signature.

**Structure.**
- **Analytic blocks:** A, `∂q_j/∂H_j` (Task 2's `q_and_dq`); B, `∂Q_l/∂(B, Ft)` at fixed H, per link, through the prey term (`PYY_term`, `dd_term`), the predator term (`pdy`, stanza index held), prey Ftime in the EwE arena remap and the switching factor `r_l` (LT2022 needs it); C, the implicit step `dH/dy = (I − ∂g/∂H)⁻¹ ∂g/∂y`, diagonal for independent predators; D, the assembly (FoodGain and FoodLoss incidence, detritus routing, M0 with MoPred·Ft, F·B, emigration, import, nutrient limitation as a rank-1 term, moTot through Eatenof, feeding-time rows).
- **Finite-difference columns** for the inputs of features left out (`EweMediation` mediators, pooled suites, UseVarPQ); above n/2 such columns, fall back to `JacobianReuse` and record why in `solver_info["jacobian"]`.

**Verification** against central differences: A and B at H held, `rtol=1e-7` (smooth away from the clamps; FD error about 1e-10); C against FD of `solve_hden_kernel` at `hden_tol=1e-14`, `rtol=1e-6` (FD noise is `hden_tol / δ`, about 2e-9 at δ = 6e-6); the assembled J against FD of `ContinuousRHS` at `hden_tol=1e-14`, `1e-6 × max|J[:, k]|` per column.

**Gain bound:** see open question 1. Effort: 3-4 days.

---

## Phase 3b (Shiny), sketch

Files: `packages/pypath-shiny/src/pypath_shiny/pages/ecosim.py` (input at 235, run at 1946-1952); new `tests/test_ecosim_continuous_ui.py`.

- Method choices grouped as "EwE parity" (RK4, AB) and "Continuous (not EwE parity)" (LSODA, BDF for stiff models, DOP853, Radau).
- A tolerance preset: Fast (rtol 1e-4; cost to be measured in 3b), Default (1e-6), Reference (1e-10, `atol` scaled by `default_atol`).
- An "Advanced: changes the model" accordion for linear forcing, log states and quasi-steady groups.
- A diagnostics panel from `solver_info`: method, tolerances, corrections, ignored switches; a per-month table (`nfev`, `rhs_evals`, `jac_fresh`, clips, `hden_passes_max`, `kernel_check`, budget residuals); a warning badge for `qs_separation < 10` or a degraded month.
- Diet rewiring (`rsim_run_advanced`, RK4 and AB only) disabled for continuous methods, with a note.
- Tests: a smoke run of REco for one year through the server function with `"lsoda"` that renders the panel; labelled choices; an invalid combination notifies instead of raising.

## Out of scope

- **Dropping monthly restarts**, even without stanzas: measured costlier with forcing (111 against 81.2 `nfev`/mo).
- **numba kernels:** the numpy twin is enough. Raise it again with open question 4.
- **Stanza cohorts as states and Ecospace** (phase 4).
- **The RsimParams per-group/per-link arrays and the `mediation=` IndexError.** Both are fixed separately, in `fix/old-findings` and PR #122 (see Prerequisites).

## Deferred minors not taken

Of phase 2's 47:
- **13 taken:** Task 1: `out_FishMort`. Task 2: the non-bracketed test, `coupled()` asserted, the combined kernel case, `0.0 <= res`, the docstring line, `_det_flows`. Task 3: `METHOD_NAMES` as the source. Task 4: the coupling rule, the corrections tests. Task 5: `forcing.get("Ftime")`. Task 8: `fail_first`, the long line in `core.md`.
- **7 already fixed** in the final wave: the pooled sentence on the page, the residual export, `atol_ftime`, the degraded-month budget, the zero-pool guard, the two first-step labels.
- **2 obsolete** (first-step reuse was removed): `_next_first_step`, the degraded-month first-step test.
- **25 not taken:** parity-shared code (3: the `RunSetup.handling[1]` alias, the recomputed `_has_ftime_adj`, the repeated `_index` block); test strengthening outside 3a's files (11: the second month-label guard, the setup-test docstring, the feeding-time bound and clip tests, the `atol` spy, the per-group cap, the NaN note, the untested migration and pool-emigration flows, the `negative_export` wording, the looser first-step assertion); mediation tidy-ups (4); mypy asserts (CI non-blocking); the `_risk_qopt` import; the `QQ_month` / Ecotracer pairing (documented); the floored `y_end` in the budget; the duplicated nutrient arithmetic; the GL6 docstring; the `last_step` conditional.

## The user's answers (2026-11-15), binding

1. **Analytic Jacobian:** not now. Task 9 stays designed but is not built in 3a. Revisit it after Task 8's measurements.
2. **Stiff method:** keep LSODA as the default. The docs (Task 8) and 3b's label recommend BDF for stiff models. No `method="auto"`.
3. **Defaults:** `jacobian="reuse"` is the default. The three model options are opt-in and labelled.
4. **numba:** later, not in 3a.
5. **Forcing:** offer both `"linear"` (the tent) and `"smooth"` (Task 5, amended).
6. **The weight fix:** applies on all methods. RK4 and AB move to Rpath's weights for pooled models, as Task 4 plans.
7. **Effort:** held monthly under either interpolation (spec 2.6).

## Open questions for the user (as asked; answered above)

1. **Analytic Jacobian:** build Task 9 (3-4 days) or stop at FD with reuse? Spec section 9 lists it; on top of reuse it would take LT2022 LSODA from 14.8 x to about 10 x and BDF from 9.9 x to about 9 x, and REco and the Black Sea nowhere.
2. **Recommended stiff method:** BDF with reuse beats LSODA on LT2022 (9.9 x against 14.8 x) and, with reuse, on REco (848 against 1035 evaluations over two years), but loses on the Black Sea (133 against 95.8 rhs/mo, today's code). Recommend BDF for stiff models, or add `method="auto"`?
3. **Defaults:** `jacobian="reuse"` by default (numerics only: LT2022 LSODA with and without reuse agree to 1.0e-7 over two years), the three model options opt-in as the spec says. Confirm.
4. **numba:** already an optional extra with a fallback; not needed here (link loop 0.294 → 0.0036 ms under numba, most of it captured by the numpy twin). Consider it later?
5. **Smoother forcing:** a C¹ mean-preserving quadratic cut the Black Sea by 16 % (linear: +29 %). Offer `forcing_interpolation="smooth"`?
6. **The weight fix on the parity paths:** Task 4 changes RK4 and AB for pooled models, towards Rpath. Accept, or apply Rpath's weights on the continuous path only? (The per-link arrays question is settled: the user chose to fix them, in `fix/old-findings`.)
7. **Effort under linear forcing:** held monthly (spec 2.6), which narrows 2.5's example of a step in effort becoming a ramp. Interpolate effort too, at the cost of an approximate catch integral?

## Spec contradictions found

1. **Section 2.5:** restarts as most of LSODA's cost and the linear option's gain. Under forcing, linear costs +26 to +37 %; restarts matter only under constant forcing; all benchmark models have stanzas.
2. **Section 2.5 against 2.6:** "a step in effort becomes a ramp" against "effort … monthly-held"; this plan follows 2.6.
3. **Section 4:** "LSODA … the only sensible choice with fast groups": BDF with reuse is cheaper on LT2022, and LSODA forms no Jacobian on REco or the Black Sea.
4. **Section 4 and phase 2's out-of-scope note** ("Jacobian reuse … needs phase 3's analytic Jacobian"): an FD callable gives the reuse; numba is not needed. The brief's "numba is not a dependency" is half true: it is an optional extra.
5. **Section 2.2:** quasi-steady "the flagged groups": the 24 /yr flag selects Zooplankton (25 /yr), 9.8 % off.
6. **The page's pooled limitation:** "more passes do not help; loosen `hden_tol`": `hden_tol` 1e-8 fails too; the cause is the port's scramble weights, which omit Rpath's `VV × QQ / B_prey`.
