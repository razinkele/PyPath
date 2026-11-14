# The continuous integration path

`rsim_run` has two kinds of integrator. RK4 (the default) and AB step EwE's and Rpath's monthly schemes; they are the parity paths and reproduce EwE 6.7.144 and Rpath. The continuous path, `method="lsoda"`, `"dop853"`, `"radau"` or `"bdf"`, solves the Ecosim equations themselves with SciPy's `solve_ivp`. It is **not EwE parity**: its numbers differ from EwE's and from the RK4 path, partly by design. This page says what it does, when to use it, how far it is from the RK4 path on three models, and what it costs. The measurements are from `packages/pypath/tests/test_ecosim_continuous_validation.py` and, where the text says so, from measurement scripts run beside it (phase 2 numbers: 2026-11-14 and 2026-11-15, SciPy 1.18.1, one process on one machine). Phase 2 added feeding time as a state, interpolated mediation, a faster handling-time solve, a mass budget and first-step reuse; the sections below describe the path as it is now. The design is in `docs/plans/2026-11-14-continuous-pathway-design.md`.

```python
out = rsim_run(scenario, method="lsoda")  # rtol 1e-6, default atol

from pypath.core.ecosim_continuous import default_atol

p = scenario.params
b0 = np.asarray(p.B_BaseRef, dtype=float)[1 : p.NUM_LIVING + p.NUM_DEAD + 1]
rtol = 1e-10
out = rsim_run(
    scenario,
    method="dop853",
    solver_options={"rtol": rtol, "atol": default_atol(b0) * rtol / 1e-6},
)
out.solver_info  # method, tolerances, corrections, ignored switches, per-month cost
```

## What it is

- **Month by month.** Each month is one initial-value problem over 1/12 year with the month's forcing, effort, forced F or catch, feeding time and forced biomass held, and the solver restarts at each month boundary. Within the month the solver chooses its own steps, up to one month, under the error control of `rtol` and `atol`.
- **Every group is a state.** Living and detritus biomass, fast groups included: `NoIntegrate` is ignored, and stiff groups are left to the stiff methods instead of EwE's blend with the equilibrium.
- **Handling time is exact.** Hden is solved at every state as the fixed point of the disc equation, from a recalibrated Hden0 and always from that cold start (never carried between evaluations), instead of EwE's one damped pass per derivative call. The solve runs on a vectorised copy of the consumption loop (`HdenKernel`): a bracketed (Illinois) secant per predator; a model with pooled predation (ScrambleSelf > 0) takes plain undamped passes over every predator; a predator the secant cannot bracket falls back to undamped passes, then SciPy's `hybr`. A handling time that is not finite, not positive or not within `hden_tol` raises `RuntimeError` naming "handling time" and the groups.
- **Feeding time is a state.** For living groups with `FtimeAdj` > 0, `dFt/dt = 12 x (clip(target, L, FtimeMax) - Ft)` per year, where `target` is EwE's feeding-time update at the current state, `L` is `ForagingTimeLowerLimit` and `FtimeMax` is 2 when absent (the target is clipped, not the state, so one forward-Euler month equals EwE's monthly rule). RiskTime is instantaneous: the predation risk is read at the current state and food loss. `solver_info["ftime_groups"]` lists the groups and `months[m]["ftime_range"]` the min and max feeding time at the month end. Of the three models below, only LT2022 has such groups (17); REco and the Black Sea have `FtimeAdj` 0.
- **Mediation is interpolated.** EwE's `EweMediation` lookup is interpolated in float64 (exactly 1 at the base state); the parity paths keep EwE's truncated float32 lookup.
- **Month-end updates use month means.** The stanza age structure takes the month's mean food gain and loss rate (6-point Gauss-Legendre quadrature of the dense output); `out_Catch` is the month's F times the month-mean biomass / 12, the integral of F x B. EwE's F x month-end biomass / 12 is kept in `out_Catch_month_end`.
- **Fixed detritus export.** DetritusOut is set once at the unforced base state, not by the first derivative of month 1.
- **Positivity.** A negative biomass state is read as 0 inside the right-hand side and counted in `clips` (the solver's trial steps can reach one; LT2022 has 6 such evaluations in 10 years, below); month-end biomass is floored at 1e-20, and month-mean biomass (hence the catch) is clipped at 0.
- **Diagnostics.** `out.solver_info` holds the method, `rtol`, `atol`, the corrections applied, the EwE switches ignored, and per month `nfev`, `njev`, `nlu`, `steps`, `status`, `t_reached`, `first_step` and `last_step`, `degraded`, `clips`, `hden_passes_max` (kernel passes of the month's largest handling-time solve; it was `hden_evals_max` in phase 1) and `min_state`, plus the budget keys below; `solver_info["hden_tol"]` is the tolerance of that solve (`solver_options`, default 1e-12, between 0 and 1e-8) and `solver_info["rhs_evals"]` counts every right-hand-side evaluation of the run. `degraded` is True when a month mean fell back to the start state (non-finite dense output or rates at a quadrature node); `status` can be 0 in that case, so check both.
- **Failures.** A failed month raises `RuntimeError` naming the month, unless `solver_options={"on_failure": "warn"}`, which logs it, with the month, and continues from the last state reached. Two failures raise even with `"warn"`: a handling-time solve that fails inside the right-hand side, and a non-finite derivative (or non-finite rates) at the start of a month.
- **The parity paths are unchanged.** One scenario can be run on both paths; the continuous run does not modify it.

## When to use it

Use **RK4** (the default) whenever the result should be EwE's: comparing with EwE output, fitting to time series (EwE's sum of squares), Monte Carlo, Ecospace, and anything that needs many runs. Use **AB** only to reproduce Rpath's `adams_run`.

Use the **continuous path** when you want the solution of the equations rather than of EwE's monthly scheme:

- to check that a result is not an artefact of one RK4 step a month or of the fast-group blend;
- for closed mass budgets (R2, R5) and a balanced model that stays balanced (R3, R4): REco (without stanzas, EwE rule) at base effort stays within 3e-7 of its base over 50 years on LSODA, against 1.16e-4 on the RK4 path, all of which comes from EwE's Hden start (R3; with R3 removed the RK4 path stays within 6e-14);
- as a reference when testing a numerical change to the parity paths.

Which method:

- **`lsoda`** (the default choice): switches between non-stiff and stiff stepping on its own, and is the only sensible choice with fast groups.
- **`dop853`**: an explicit high-order method, the reference for non-stiff models at tight tolerance. On a stiff model it is limited by stability, not accuracy.
- **`radau`** and **`bdf`**: implicit alternatives for stiff models.

Tolerances: the default is `rtol` 1e-6 and `atol` 1e-8 x (`rtol` / 1e-6) x max(B0, 1e-6 x max(B0)) per group, so **the default `atol` follows `rtol`** (in phase 1 it did not, and `rtol` 1e-10 with the default `atol` was good to 4.7e-9 only). A feeding-time state has `atol` 1e-2 x `rtol`. Pass `atol` to override (`pypath.core.ecosim_continuous.default_atol` gives the base-scale vector). The first step of every month after the first is the previous month's last full step; month 1, and any month after a failed or degraded one, leave it to the solver.

## The corrections it applies

The continuous path applies these by default and lists them in `solver_info["corrections"]`; the parity paths keep EwE's behaviour. `solver_options={"corrections": False}` leaves out R2, R4, R5 and DF once; the handling-time rows (R3, Hden0, Hden) and the rows below DF once are part of the path and stay. `solver_info["corrections"]` lists only the items that act on the scenario (R2 if `DETRITUS_MO_UNSCALED` is set, R5 if a group has moTot, DF once if there is `DetPassedProp`, R4 if an EmigRate changes a living group), plus "Hden0 recalibrated" when handling time is on and "mediation interpolated" when there is a mediation lookup. IDs refer to `docs/parity/deviations.md`.

| Item | RK4 path (EwE rule) | Continuous path |
|---|---|---|
| R2 | Detritus receives the unscaled mo x B while the group loses the feeding-time-scaled flow | Closed: detritus receives what the group loses |
| R3 | A run starts from Hden = CmCo / (CmCo + 1) | Dropped: the run starts at the base fixed point |
| Hden0 | CmCo / (CmCo - 1) | Recalibrated to 1 / (1 - q0 / (CmCo x QB)), the base fixed point of the link consumption q0; lower for predators with diet import |
| Hden | One damped pass per derivative call, carried between calls | The exact fixed point at each state |
| R4 | A living group's EmigRate is ignored | EmigRate x B where Emigration is 0 |
| R5 | With PaddP < 1 detritus receives mo x B while the group loses moTot x B | Closed: detritus receives moTot x B |
| DF once | DetPassedProp x DF counts a pool's passed share twice | Once: DetPassedProp x B x DF / (the source's DF to other pools) |
| DetritusOut | Set by the first derivative of month 1, with month 1's forcing and effort | Set at the unforced base state |
| Fast groups | `RK4_FAST_EQUILIBRIUM`: a blend with the equilibrium biomass | Integrated as states |
| Stanza update | `STANZA_STAGE_RATES`: growth from the step's last RK4 stage, survival from its first | Month means by quadrature |
| Feeding time | `FTIME_START_OF_MONTH`: from the start-of-month consumption, held within the month | A state: `dFt/dt = 12 x (clip(target, L, FtimeMax) - Ft)`, the target at the current state |
| RiskTime | The month-start predation, held within the month | Instantaneous: read at the current state and food loss |
| Mediation | EwE's lookup, truncated to its 1200 float32 points | Interpolated in float64, exactly 1 at the base |
| Sub-steps | `RK4_STEPS` | Ignored; the solver picks its steps |

The four switches (`RK4_FAST_EQUILIBRIUM`, `STANZA_STAGE_RATES`, `FTIME_START_OF_MONTH`, `RK4_STEPS`) are reported in `solver_info["ignored_switches"]` when the scenario sets them. Under the `"rpath"` rule only `RK4_STEPS` is set, and handling time is off.

## The mass budget

Each month's result carries a check of the integrated mass budget, computed from the right-hand side's own flows rather than from the biomass change, so it is not a tautology. `exchange_rate` is the net external flux of the biomass states: in, the production of producers and of groups with QB 0, import, immigration and forced migration, and the detritus inflow from outside; out, respiration, landed catch (fishing links that end outside the detritus pools, plus forced F), emigration, the pools' emigration and their DetritusOut export. Internal transfers are not used, so an unclosed transfer leaves a residual. Per month:

- `budget_residual` = (sum of end biomass - sum of start biomass - t x the month-mean exchange) / sum of start biomass, over the biomass states (feeding-time states are not summed).
- `pool_budget_residual`: the largest over the detritus pools of |end - start - t x the pool's mean (DetGain - DetLoss)| / start.
- `nutrient_floor`: evaluations where the free nutrient was at its floor. `negative_export`: evaluations where a pool's DetPassedProp exceeded its DetritusOut by more than 1e-6 relative (the export is then negative; the Black Sea's Detritus pool passes on all of its DetritusOut, and float32 rounding puts it just above, which is within that tolerance).
- Both residuals are None in a month with forced biomass: the forced groups' flows are imposed, not integrated.

The residual measures the solver and the quadrature, not a leak: at the default `rtol` it is of the order of `rtol`; on a REco variant at `rtol` 1e-10 it falls to about 1e-9, the error of the 6-point Gauss-Legendre mean of the exchange. Ten years at the default tolerances (maximum over the months checked):

| Model | Months checked | `budget_residual` | `pool_budget_residual` | `nutrient_floor` | `negative_export` |
|---|---|---|---|---|---|
| REco, with stanzas | 120 of 120 | 2.9e-9 | 1.9e-6 | 0 | 0 |
| Black Sea | 120 of 120 | 3.3e-6 | 2.3e-6 | 0 | 0 |
| LT2022 | 0 of 120 (forced biomass in every month) | None | None | 0 | 0 |

## Validation

There is no EwE oracle for this path, so it is checked against itself and against RK4 where the two define the same equations (design section 5).

| Check | Model | Result |
|---|---|---|
| RK4 converges to DOP853 (rtol 1e-12, atol 1e-14 x B0) as `RK4_STEPS` doubles | REco without stanzas, `"rpath"` rule (no stanzas, no handling time, fast groups integrated by RK4, so the equations coincide), effort 2, month 12 | e_4 = 1.73e-9, e_8 = 6.03e-11, e_16 = 2.81e-12: the error falls 28.6 and 21.5 times per doubling, fourth order |
| LSODA against DOP853, both rtol 1e-8 | REco with stanzas, EwE rule, effort 2, 2 years | 3.0e-8 max relative difference |
| Radau and BDF against LSODA, rtol 1e-8 | REco without stanzas, EwE rule, effort 2, 1 year | Radau 7.9e-8, BDF 1.0e-6 |
| A balanced model stays balanced, LSODA, default tolerances | REco without stanzas, base effort, no forcing, 50 years | 3.0e-7 (EwE rule), 3.1e-7 (`"rpath"`) from B0; flat over the 50 years, solver error at rtol 1e-6, not drift (the start derivative is 4e-14 x B) |
| Ten years run, LSODA | REco, the Black Sea, LT2022 | Finite and positive; every month status 0, none degraded; `hden_passes_max` at most 6, 9 and 6 (the test bounds it at 25); clips 0, 0 and 6 (LT2022: 6 evaluations in months 13, 25, 49, 85 and 97, trial steps with a negative biomass, none in phase 1) |

A DOP853 reference at rtol 1e-10 is not accurate enough for the convergence check. Measured against rtol 1e-13 in phase 1, it was good to 4.7e-9 only at the then default `atol` (which did not follow `rtol`), and to 7e-12 to 1.1e-11 with an `atol` of 1e-10 x B0 or less, which still exceeds e_16. At rtol 1e-12 the error is 7.5e-14.

## Differences from the RK4 path

Ten years, LSODA at the default tolerances against `method="RK4"` on the same scenario. `|B_lsoda / B_RK4 - 1|` over months 1-120 and every living group (living and detritus groups in brackets):

| Model | Scenario | Median | Max |
|---|---|---|---|
| REco (Rpath), with stanzas | EwE rule, effort 2 from month 1 | 0.017 % (0.017 %) | 18 %, Seabirds (93 %, Discards) |
| Black Sea (EwE-F test case) | scenario 1 | 5.5 % (5.0 %) | 38 %, Zooplankton (38 %) |
| LT2022 | scenario 16, time series dataset 35 | 3.6 % (3.4 %) | 122 %, Zooplankton (122 %) |

Most of these differences are definitional. To see which, the RK4 path was moved towards the continuous definitions one item at a time and compared with the same LSODA run. These were measurement scripts, not tests: `RK4_STEPS` and `RK4_FAST_EQUILIBRIUM` set on the scenario; R3 removed by starting Hden at CmCo / (CmCo - 1); DetritusOut pre-filled at the unforced base state by wrapping `deriv_vector`. Median / max over living groups. The first three rows were re-measured on 2026-11-15 against the phase 2 LSODA run; the last two rows are phase 1 measurements against the phase 1 run and were not repeated (they depend on the LSODA side only through the same run's definitions; the numbers may have moved by the amount the first three rows moved):

| RK4 path | REco | Black Sea | LT2022 |
|---|---|---|---|
| As is (one step a month, EwE rule) | 0.017 % / 18 % | 5.5 % / 38 % | 3.6 % / 122 % |
| 32 sub-steps (LT2022: 128) | 0.017 % / 18 % | 2.7 % / 32 % | 0.63 % / 154 % |
| and fast groups integrated | 0.017 % / 18 % | 2.7 % / 30 % | 0.53 % / 13 % |
| and R3 removed | 0.014 % / 18 % | 2.7 % / 31 % | 0.44 % / 9.6 % |
| and DetritusOut at the base state | 0.0017 % / 0.069 % | 2.7 % / 31 % | 0.44 % / 9.6 % |

- **REco: where DetritusOut is set.** The scenario doubles effort from month 1. The RK4 path, as EwE, sets each detritus pool's export rate in the first derivative of month 1, with the doubled discards already flowing, so Discards stays near its base (1.15 at year 10, B0 1.22). The continuous path sets it at the base state, so Discards rises to 2.21, and the Seabirds follow. DetritusOut at the base state alone brings the paths within 0.069 %; so does running month 1 at base effort on both paths (0.072 %).
- **Black Sea: one step a month, then handling time.** Half the median difference is RK4's single step a month (32 and 128 sub-steps give the same result). Most of the rest is handling time at the model's CmCo 2: with CmCo 1000 for every group on both paths, the RK4 path with 32 sub-steps and fast groups integrated comes within 0.43 % median of LSODA. The handling-time items are the recalibrated Hden0 (lower for predators with diet import) and the exact fixed point in place of EwE's damped pass carried between calls. The maximum is juvenile anchovy (Anchovy 0-0+), still 26 % at CmCo 1000 and 5.5 % apart at the end of month 1, when the scenario's nutrient forcing moves the whole web (both paths leave the base by up to 17 % in month 1; with neutral forcing the two anchovy stanzas stay within 0.2 % of their base on both over the first three months, the span measured). That is the stanza update: fed EwE's rates (consumption at the month-end state, 5.6 % below the month mean in month 1), the continuous path's anchovy follows the RK4 path's (32 sub-steps, fast groups integrated) to 4 digits, and over 10 years the paths then come within 0.38 % median. The largest remaining difference among the groups is Aurelia: 2.3e-4 in month 1, 0.46 % at month 12 and 21 % at month 120 (measured against the RK4 path with 32 sub-steps and fast groups integrated; phase 1 measured 5e-6 in month 1 and 23 % at year 10). It was not attributed further; the corrections listed for this model are R2, DF once and the recalibrated Hden0 (no feeding-time groups, no mediation, R4 and R5 not listed); phase 1 measured R2, R4, R5 and feeding time to change nothing here, and DetritusOut likewise, and phase 2 did not repeat that.
- **LT2022: one step a month, then the fast-group blend.** 128 sub-steps (its stiffest eigenvalue, about 3300 per year, needs more than about 100 for RK4 to be stable; 256 give the same result) take the median from 3.7 % to 0.56 %. The maximum is Zooplankton, a fast group, under EwE's blend with the equilibrium; integrating it takes the maximum from 161 % to 9.6 %. R3 and DetritusOut change nothing here. What remains, 0.53 % median and 13 % at most (Zooplankton; phase 1: 0.44 % and 9.6 %, before feeding time became a state), comes from the other corrections (handling time, the stanza update, feeding time as a state against RK4's monthly rule, interpolated mediation, R2, R4, R5, DF once) and was not attributed further.

Setting `STANZA_STAGE_RATES`, `FTIME_START_OF_MONTH` or `DETRITUS_MO_UNSCALED` to False on the RK4 path is not a step towards the continuous path: the first two select Rpath's rules, not month means, and moved LT2022 further away (1.9 % median, 34 % max).

## Cost

Measured on 2026-11-14 and 2026-11-15 on one machine with no other heavy jobs, except where stated. The times are CPU time (`process_time`) from `test_ten_year_cost`: a warm-up of each method, three interleaved RK4 and LSODA runs of 10 years, the minimum of each. The call counts come from `solver_info` in `test_ten_years_on_three_models`.

LSODA at the default tolerances, 10 years:

| Model | Groups | `nfev` / month (mean, max) | Right-hand-side evaluations / month | `hden_passes_max` (largest month) | CPU s / year, RK4 | CPU s / year, LSODA | Ratio | CPU / evaluation | vs one exact derivative call |
|---|---|---|---|---|---|---|---|---|---|
| REco, with stanzas | 22 | 32, 38 | 40.4 | 6 | 0.100 | 0.789 | 7.9 x | 1.63 ms | 1.59 x |
| Black Sea | 22 | 94, 122 | 102.0 | 9 | 0.047 | 1.61 | 34.5 x | 1.32 ms | 2.47 x |
| LT2022 | 24 (17 feeding-time states) | 349, 796 (6.0 Jacobians) | 357.2 | 6 | 0.105 | 6.80 | 64.6 x | 1.59 ms | 1.76 x |

Since phase 1 (same models, same scenarios; the phase 1 wall times were taken with other jobs running, so the comparison is indicative): the Black Sea went from 40.7 s to 1.6 s per simulated year (914 x RK4 to 34.5 x), REco from 2.6 s to 0.79 s, and LT2022 from 13.1 s to 6.8 s although it now carries 17 more states and about 60 % more `nfev` (349 per month against 214). The handling-time solve takes at most 9 kernel passes where the Black Sea's undamped iteration took up to 132 derivative evaluations. One evaluation on REco at CmCo 2 went from 41.5 ms to about 1.9 ms (Task 6, a separate measurement).

The test asserts for REco that CPU per evaluation is at most 2.5 x one exact single-pass derivative call (the evaluation adds the handling-time solve, 6-9 kernel passes, to a derivative call; the plan's 1.5 x could not hold) and for the Black Sea that the RK4 ratio is at most 35. The other figures are printed. **The Black Sea bound is not reliably met**: three runs of the same test gave 34.5 (quiet machine, load average about 1.2), 35.2 (inside the full suite, load average about 1.2, a failure) and 36.0 (load average 6.7, a failure), with identical call counts (12237 evaluations); the minimum LSODA CPU moved by 2 % between the first two runs, and RK4's by 1 %. The bound of 35 was kept as ruled and the test fails on some runs. Design section 4's 10-20 x is not reached for the Black Sea and LT2022; it needs fewer evaluations (phase 3: an analytic Jacobian and linear forcing).

First-step reuse (month m starts from the last full step of month m - 1) was measured on the Black Sea and LT2022, 10 years, with and without it (a measurement script that drops `first_step`; the two runs agree to 3.9e-7 and 1.3e-6 in biomass):

| Model | `nfev` / month with reuse (mean, median, max) | without reuse | Right-hand-side evaluations with | without |
|---|---|---|---|---|
| Black Sea | 94.0, 97, 122 | 87.4, 91, 115 | 12237 | 11453 |
| LT2022 | 349.2, 300, 796 | 334.7, 297, 787 | 42868 | 41127 |

Reuse costs about 7 % more evaluations on the Black Sea and 4 % more on LT2022, and gained nothing on REco in Task 6 (20.1 to 22.0 `nfev` per month at CmCo 1000, 20.8 to 20.7 at CmCo 2). The code keeps it as the design asks; turning it off would be a one-line change, and it is not worth the cost on these three models.

REco by method, 2 years at the default tolerances, EwE rule, effort 2 (CPU time, minimum of two runs against the minimum of three RK4 runs, measured 2026-11-15 after the ten-year runs):

| Model | Method | `nfev` / month | Jacobians / month | All right-hand-side calls / month | CPU s / year | Ratio to RK4 |
|---|---|---|---|---|---|---|
| REco, no stanzas | lsoda | 19.9 | 0 | 27.9 | 0.54 | 5.6 x |
| | dop853 | 32.0 | 0 | 40.1 | 0.77 | 8.1 x |
| | radau | 15.3 | 1.0 | 45.4 | 0.89 | 9.3 x |
| | bdf | 14.5 | 1.0 | 45.6 | 0.91 | 9.5 x |
| REco, stanzas | lsoda | 36.0 | 0 | 44.0 | 0.87 | 8.6 x |
| | dop853 | 32.0 | 0 | 40.1 | 0.78 | 7.8 x |
| | radau | 20.1 | 1.4 | 58.4 | 1.16 | 11.5 x |
| | bdf | 17.9 | 1.8 | 68.1 | 1.35 | 13.4 x |

RK4 takes 0.096 s per year without stanzas and 0.101 with.

What the call counts mean:

- `nfev` counts the solver's calls of the right-hand side. For LSODA it includes the finite-difference Jacobian columns; for Radau and BDF it does not (SciPy builds their Jacobian outside the count), hence the separate column.
- `rhs_evals` (the third column of the first table) counts every evaluation of the run: the solver's, one start check a month, six quadrature evaluations, and the rate evaluations for the outputs (Hden at the month end; the month-start predation when RiskTime is set; the feeding-time range). It is always above the sum of `nfev`.
- Under the EwE rule each right-hand-side evaluation is one derivative call, inside which handling time is solved first on the consumption kernel alone: a bracketed secant per predator, `hden_passes_max` kernel passes at most, with `hybr` finishing if 30 passes leave `hden_tol` unmet. A model with pooled predation takes undamped passes instead (pooled REco: 19 to 29, measured in Task 2).
- LT2022 is stiff (one eigenvalue of about 3300 per year at base, design section 2.2), so LSODA runs its stiff method there, with about 6 Jacobians a month.

## Known limitations

- `annual_Qlink` and the ecotracer are month-end snapshots, without handling time or mediation, unlike the month-integral catch. `annual_QB` is never filled, on any path.
- The catch integral is approximate under fleet mediation, because F is recomputed from the state within the month while the catch uses the month's F times the month-mean biomass.
- Forcing, effort, forced catch and forced biomass are held within the month, and fleet dynamics runs at the month boundaries.
- A forced stanza group is overwritten by the age structure, on both paths.
- Stanza models are first order in the monthly split whatever the tolerance: the age structure is updated once a month from the month's mean rates.
- The budget residuals are None in a month with forced biomass, so a model that forces biomass in every month (LT2022) is not budget-checked on this path.
- LT2022 has 6 evaluations in 10 years where a trial step of the solver reads a negative biomass (clipped to 0, counted in `clips`); the output is finite and positive and no month degrades, but phase 1 had none and the cause (the feeding-time states, or the changed step sequence) was not isolated.
- The Black Sea costs 34.5 to 36 x RK4 (the 35 x bound of `test_ten_year_cost` fails on some runs) and LT2022 64.6 x; the 10-20 x of design section 4 needs an analytic Jacobian and linear forcing (phase 3). First-step reuse does not help on these two models.
- IBM groups are rejected.
- Ecospace still uses its own scheme (RK4 sub-steps with implicit movement); the continuous path is Ecosim only.
