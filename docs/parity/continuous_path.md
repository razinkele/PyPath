# The continuous integration path

`rsim_run` has two kinds of integrator. RK4 (the default) and AB step EwE's and Rpath's monthly schemes; they are the parity paths and reproduce EwE 6.7.144 and Rpath. The continuous path, `method="lsoda"`, `"dop853"`, `"radau"` or `"bdf"`, solves the Ecosim equations themselves with SciPy's `solve_ivp`. It is **not EwE parity**: its numbers differ from EwE's and from the RK4 path, partly by design. This page says what it does, when to use it, how far it is from the RK4 path on three models, and what it costs.

The measurements are from `packages/pypath/tests/test_ecosim_continuous_validation.py` and, where the text says so, from measurement scripts run beside it (phase 2 numbers: 2026-11-14 and 2026-11-15, SciPy 1.18.1, one process on one machine).

Phase 2 added feeding time as a state, interpolated mediation, a faster handling-time solve and a mass budget. Phase 3a (2026-11-15, same SciPy and machine) cut the cost (a Newton handling-time solve on a vectorised consumption pass, finite-difference Jacobian reuse), fixed pooled models and added four opt-in options: forcing interpolation, log-transformed states, quasi-steady fast groups and the Jacobian choice. The sections below describe the path as it is now.

The numbers in the Validation, Differences and Cost sections come from one run of `test_ecosim_continuous_validation.py` on 2026-11-15; the opt-in costs under Options come from short measurement scripts run the same day (two-year Black Sea, five-year REco, two-year LT2022); where a number is older the text says so. The design is in `docs/plans/2026-11-14-continuous-pathway-design.md`.

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
- **Handling time is exact.** Hden is solved at every state as the fixed point of the disc equation, from a recalibrated Hden0 and always from that cold start (never carried between evaluations), instead of EwE's one damped pass per derivative call. The solve runs on a vectorised copy of the consumption loop (`HdenKernel`, with a vectorised consumption pass, `_KERNEL_VECTOR`): a Newton iteration per predator in which components that have converged are held (a predator that eats nothing, Hden 1, is a valid root); a predator Newton cannot settle falls back to a bracketed (Illinois) secant, then undamped passes, then SciPy's `hybr`; a model with pooled predation (ScrambleSelf > 0) takes plain undamped passes over every predator. A handling time that is not finite, not positive or not within `hden_tol` raises `RuntimeError` naming "handling time" and the groups.
- **Feeding time is a state.** For living groups with `FtimeAdj` > 0, `dFt/dt = 12 x (clip(target, L, FtimeMax) - Ft)` per year, where `target` is EwE's feeding-time update at the current state, `L` is `ForagingTimeLowerLimit` and `FtimeMax` is 2 when absent (the target is clipped, not the state, so one forward-Euler month equals EwE's monthly rule). RiskTime is instantaneous: the predation risk is read at the current state and food loss. `solver_info["ftime_groups"]` lists the groups and `months[m]["ftime_range"]` the min and max feeding time at the month end. Of the three models below, only LT2022 has such groups (17); REco and the Black Sea have `FtimeAdj` 0.
- **Mediation is interpolated.** EwE's `EweMediation` lookup is interpolated in float64 (exactly 1 at the base state); the parity paths keep EwE's truncated float32 lookup.
- **Month-end updates use month means.** The stanza age structure takes the month's mean food gain and loss rate (6-point Gauss-Legendre quadrature of the dense output); `out_Catch` is the month's F times the month-mean biomass / 12, the integral of F x B. EwE's F x month-end biomass / 12 is kept in `out_Catch_month_end`.
- **Fixed detritus export.** DetritusOut is set once at the unforced base state, not by the first derivative of month 1.
- **Positivity.** A negative biomass state is read as 0 inside the right-hand side and counted in `clips` (a solver's trial step can reach one); month-end biomass is floored at 1e-20, and month-mean biomass (hence the catch) is clipped at 0.
- **Diagnostics.** `out.solver_info` holds the method, `rtol`, `atol`, the corrections applied, the EwE switches ignored, and per month `nfev`, `njev`, `nlu`, `steps`, `status`, `t_reached`, `last_step` (the last full step, a diagnostic), `degraded`, `clips`, `hden_passes_max` (kernel passes of the month's largest handling-time solve; it was `hden_evals_max` in phase 1), `hden_residual_max` (the month's largest relative residual of the solved handling time) and `atol_ftime` (the feeding-time states' atol, the tail of the solver's vector; `[]` without feeding time), `min_state`, `kernel_check`, `jac_fresh` and `jac_reused` (below), plus the budget keys below; `solver_info["hden_tol"]` is the tolerance of that solve (`solver_options`, default 1e-12, between 0 and 1e-8) and `solver_info["rhs_evals"]` counts every right-hand-side evaluation of the run. `degraded` is True when a month mean fell back to the start state (non-finite dense output or rates at a quadrature node); `status` can be 0 in that case, so check both.
- **Kernel check.** `months[m]["kernel_check"]` is the largest relative difference between the vectorised consumption pass and the link loop (the exact reference) at the month's checked evaluation: the run-time check that the fast path agrees with the loop on your model. It was 0.0 in all 120 months on REco, the Black Sea and LT2022.
- **Failures.** A failed month raises `RuntimeError` naming the month, unless `solver_options={"on_failure": "warn"}`, which logs it, with the month, and continues from the last state reached. Two failures raise even with `"warn"`: a handling-time solve that fails inside the right-hand side, and a non-finite derivative (or non-finite rates) at the start of a month.
- **The parity paths are unchanged.** One scenario can be run on both paths; the continuous run does not modify it.

## When to use it

Use **RK4** (the default) whenever the result should be EwE's: comparing with EwE output, fitting to time series (EwE's sum of squares), Monte Carlo, Ecospace, and anything that needs many runs. Use **AB** only to reproduce Rpath's `adams_run`.

Use the **continuous path** when you want the solution of the equations rather than of EwE's monthly scheme:

- to check that a result is not an artefact of one RK4 step a month or of the fast-group blend;
- for closed mass budgets (R2, R5) and a balanced model that stays balanced (R3, R4): REco (without stanzas, EwE rule) at base effort stays within 3e-7 of its base over 50 years on LSODA, against 1.16e-4 on the RK4 path, all of which comes from EwE's Hden start (R3; with R3 removed the RK4 path stays within 6e-14);
- as a reference when testing a numerical change to the parity paths.

Which method:

- **`bdf`: use it for stiff models.** LT2022 (stiffest eigenvalue about 3300 per year) costs 11.3 x RK4 on BDF against 17.3 x on LSODA (Cost, below). `lsoda` stays the default, because it chooses between stiff and non-stiff stepping on its own and is never far off; there is no `method="auto"`, so choose BDF yourself when a model has fast groups or a stiff feeding-time response.
- **`lsoda`** (the default): switches between non-stiff and stiff stepping on its own; the best choice when you do not know the model's stiffness.
- **`dop853`**: an explicit high-order method, the reference for non-stiff models at tight tolerance. On a stiff model it is limited by stability, not accuracy, except under quasi-steady fast groups (Options).
- **`radau`**: an implicit alternative for stiff models.

Tolerances: the default is `rtol` 1e-6 and `atol` 1e-8 x (`rtol` / 1e-6) x max(B0, 1e-6 x max(B0)) per group, so **the default `atol` follows `rtol`** (in phase 1 it did not, and `rtol` 1e-10 with the default `atol` was good to 4.7e-9 only). A feeding-time state has `atol` 1e-2 x `rtol`. Pass `atol` to override (`pypath.core.ecosim_continuous.default_atol` gives the base-scale vector). Every month leaves its first step to the solver (reusing the previous month's last step was tried and dropped, see Cost).

## Options

`solver_options` takes `rtol`, `atol`, `corrections`, `on_failure`, `hden_tol` and the five below. Any other key, or a value outside the lists, raises `ValueError` naming the key. The defaults are the cheapest and the closest to the month-held formulation; the options marked **changes the model** solve different equations and are opt-in. `solver_info` names the options in force.

| Option | Values | Changes the model? |
|---|---|---|
| `jacobian` | `"reuse"` (default), `"solver"` | No: cost only |
| `forcing_interpolation` | `"constant"` (default), `"linear"`, `"smooth"` | **Yes** (linear, smooth) |
| `positivity` | `"clip"` (default), `"log"` | No: the same equations in log variables |
| `fast_groups` | `"states"` (default), `"quasi-steady"` | **Yes** (quasi-steady) |
| `qs_rate` | float > 0, default 365 per year | Selects the quasi-steady groups |

- **`jacobian`.** See the next section.
- **`forcing_interpolation`: `"linear"` and `"smooth"` change the model.** The default holds each month's forcing constant, as EwE does. `"linear"` replaces it by a continuous tent through the month's boundary values with a kink at mid-month; `"smooth"` by one quadratic per month. Both preserve every month's mean exactly (the boundary values are averages of neighbouring months, limited so a non-negative key stays at or above 0; a negative value in a key that is not signed raises). Effort stays held monthly under either. With no time-varying key the constant rule is used, whatever the setting. Linear integrates the month's means on 12 Gauss-Legendre nodes (6-point on each half-month, because of the kink). Cost on the Black Sea, LSODA, two years: linear +31 % right-hand-side evaluations (3023 against 2299), smooth 15 % fewer (1957). The biomass differs from the constant run by up to 15 % (Phytoplankton; median 0.13 %; Black Sea, two years, LSODA, measured 2026-11-15) at a month end, a different model of the forcing, not an error. The point of the options is cost and smoothness for the solver; the Black Sea does not reach the 10-20 x of design section 4 with them.
- **`positivity="log"`.** The states are `z = ln B`, so a biomass cannot go negative and the `clips` count stays 0. The same equations, not a different model; a group at the floor (1e-20) absorbs, it can only grow. The atol of a log state is `min(max(atol_i / B0_i, rtol), 0.1)`: the cap keeps error control for a group regrowing from the floor. It costs evaluations where groups sweep orders of magnitude: REco without stanzas at effort 100, five years, LSODA, needs 2303 right-hand-side evaluations against 1643 with clipping (+40 %; +29 % of the nfev over the same run without the atol cap, which the cap's error control costs), and clips 0 against 131 times; REco with stanzas at effort 2, two years, needs 914 against 1003 (all measured 2026-11-15). Use it for runs that drive a group towards zero.
- **`fast_groups="quasi-steady"`: changes the model.** Groups that EwE flags as fast (`NoIntegrate`) and whose base loss rate exceeds `qs_rate` per year (EwE's flag fires at 24 per year, so it also flags groups that are not fast) are no longer states; at every evaluation each is set to its equilibrium given the slow groups (chord Newton, then pseudo-transient continuation from the month-start biomass, then `hybr`; only an attracting root, every real part of the fast block's Jacobian eigenvalues negative, is accepted, else `RuntimeError`). That removes the stiffness, so DOP853 can be used. `solver_info["qs_groups"]` lists the groups (a forced group is excluded and listed), `qs_separation` the slowest selected base loss rate over the largest `|Re lambda|` of the reduced Jacobian at the start (a WARNING below 10: a poor separation makes the approximation poor), and per month `qs_inner_max` and `qs_fallbacks` the inner solve's cost. The budget residuals are None under it. Cost, LT2022 with effort x 3, two years (one group selected, `qs_separation` 34.6): DOP853 goes from 753 to 210 right-hand-side evaluations a month (18079 to 5046), whereas LSODA (125 to 146 a month) and BDF (64 to 152) cost more, each solver evaluation and Jacobian column becoming a solve of 2-5 evaluations. Use it only with DOP853, and check `qs_separation`.

### The Jacobian

LSODA, Radau and BDF need a Jacobian, which the path builds by finite differences; a fresh one costs `n_state + 1` right-hand-side evaluations (SciPy passes no `f`, so the base evaluation is repeated). The solver restarts every month, and a Jacobian from a nearby state is a good enough iteration matrix, so `jacobian="reuse"` (the default) hands the solver a callable that reuses the last matrix: LSODA reuses it whenever `t` strictly advances and the state has moved by at most 10 % relative since the matrix was built (a retreat in `t` is a retry after a rejected step and gets a fresh matrix); BDF and Radau take a fresh matrix at every request after the month's first. A month's first request reuses the previous month's matrix when its shape matches (the state vector has the same length). `jacobian="solver"` leaves the solver to build its own, as in phase 2. Per month `jac_fresh` and `jac_reused` count the two. On LT2022 with LSODA, ten years: 147 fresh, 573 reused (the run of 2026-11-15); the reuse costs about 3 % more evaluations than one fresh matrix per request would (146.0 a month in the phase 3a development run against 150.7 here). Two consequences for reading `solver_info`:

- With a Jacobian callable, LSODA's `nfev` **no longer includes the Jacobian columns** (SciPy counts them elsewhere), as BDF's and Radau's never did. Compare cost by `rhs_evals`, which counts every evaluation, not by `nfev`.
- The reuse does not change the answer beyond the solver's tolerance (measured 2026-11-15, REco with stanzas, effort 2 and x 5 from month 7, two years: biomass differs between `"reuse"` and `"solver"` by 9.5e-10 on BDF, 1.6e-7 on Radau and exactly 0 on LSODA, which forms no Jacobian on REco; BDF needs 848 evaluations against 1377, Radau 920 against 1399; LT2022 LSODA, two years, differs by 1.0e-7 with 2544 evaluations against 6721).

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
| REco, with stanzas | 120 of 120 | 3.9e-9 | 1.2e-7 | 0 | 0 |
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
| Ten years run, LSODA | REco, the Black Sea, LT2022 | Finite and positive; every month status 0, none degraded; `hden_passes_max` at most 3, 5 and 3 (the test bounds it at 25); clips 0 on all three; `kernel_check` 0.0 in every month |

A DOP853 reference at rtol 1e-10 is not accurate enough for the convergence check. Measured against rtol 1e-13 in phase 1, it was good to 4.7e-9 only at the then default `atol` (which did not follow `rtol`), and to 7e-12 to 1.1e-11 with an `atol` of 1e-10 x B0 or less, which still exceeds e_16. At rtol 1e-12 the error is 7.5e-14.

## Differences from the RK4 path

Ten years, LSODA at the default tolerances against `method="RK4"` on the same scenario. These differences, and rows 1-3 of the ladder below, were measured with first-step reuse on (its effect is about 1e-6; see Cost), so they hold to that order after the drop. `|B_lsoda / B_RK4 - 1|` over months 1-120 and every living group (living and detritus groups in brackets):

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

Measured on 2026-11-15 on one machine (load average 2.2-2.5 during the timed runs; no other job of ours). The times are CPU time (`process_time`) from `test_ten_year_cost`: a warm-up of each method, three interleaved RK4 and candidate runs of 10 years, the minimum of each; the test asserts the ratio against the bound. The call counts come from `solver_info` in `test_ten_years_on_three_models` (LSODA; LT2022 BDF's `rhs_evals` from `test_ten_year_cost`). Every month leaves its first step to the solver and the default options are used (`jacobian="reuse"`).

| Model, method | `nfev` / month (mean, max) | Right-hand-side evaluations / month | `hden_passes_max` (largest month) | CPU s / year, RK4 | CPU s / year | Ratio | Bound | Phase 2 ratio | CPU / evaluation | vs one exact derivative call |
|---|---|---|---|---|---|---|---|---|---|---|
| REco, with stanzas, LSODA | 33, 37 | 41.5 | 3 | 0.101 | 0.478 | 4.7 x | 6 x | 8.2 x | 0.96 ms | 0.93 x |
| Black Sea, LSODA | 87, 115 | 95.4 | 5 | 0.047 | 1.126 | 24.2 x | 25 x | 32.5 x | 0.98 ms | 1.82 x |
| LT2022, LSODA | 91, 278 (6.0 Jacobians) | 150.7 | 3 | 0.109 | 1.890 | 17.3 x | 20 x | 61.6 x | 1.05 ms | 1.16 x |
| LT2022, BDF | not recorded | 86.4 | not recorded | 0.108 | 1.221 | 11.3 x | 14 x | not measured | 1.18 ms | 1.30 x |

All four bounds held, the Black Sea's with the least margin (24.2 against 25; its three samples were 11.26, 11.29 and 11.27 s against RK4's 0.47-0.49 s). A bound that fails under this protocol is a finding, not a number to move: laguna is shared, so a failed sample is re-run once, and the test prints every sample and the load average. For REco the test also asserts CPU per evaluation at most 1.2 x one exact single-pass derivative call (`EVAL_COST_BOUND`; the phase 2 bound was 2.5): the evaluation is now a Newton handling-time solve, 3 kernel passes at most, on a vectorised consumption pass, and it measured 0.93 x. Between phase 2 and phase 3a LT2022 on LSODA fell from 61.6 x to 17.3 x, the Black Sea from 32.5 x to 24.2 x and REco from 8.2 x to 4.7 x, from cheaper evaluations (Newton handling time, the vectorised pass), and the Jacobian reuse (on LT2022, a fresh finite-difference matrix costs `n_state + 1` evaluations).

The 10-20 x of design section 4 is reached on LT2022 (BDF 11.3 x, LSODA 17.3 x) but not on the Black Sea (24.2 x): its call-count floor, about 95 evaluations a month, stays. The opt-in options (Options, above) move that floor: smooth forcing 15 % fewer evaluations; the other options cost more, except quasi-steady groups with DOP853. An analytic Jacobian is designed, not built.

**First-step reuse was tried and dropped** (phase 2 measurement, unchanged). The design asked for each month to start from the last full step of the month before. Measured on 10-year runs with and without it (a measurement script that drops `first_step`; the runs agree to 3.9e-7 and 1.3e-6 in biomass), it added evaluations on all three models:

| Model | `nfev` / month with reuse (mean) | without reuse |
|---|---|---|
| REco (1 year, CmCo 1000) | 22.0 | 20.1 |
| Black Sea | 94.0 | 87.4 |
| LT2022 | 349.2 | 334.7 |

On LT2022 it also caused 6 clipped evaluations (a trial step with a negative biomass, read as 0) in months 13, 25, 49, 85 and 97, the first months of a year, where forcing steps while the reused step was sized for December; with no reuse LT2022 clips none in 10 years. So the code leaves every month's first step to the solver. `integrate_month(..., first_step=)` remains as a pass-through, and `months[m]["last_step"]` as a diagnostic. Design section 4's step reuse is not implemented.

What the call counts mean:

- `nfev` counts the solver's calls of the right-hand side. With `jacobian="reuse"` LSODA's count excludes the Jacobian columns, as it has always for Radau and BDF (SciPy builds their Jacobian outside the count); the columns are in `rhs_evals`. Compare cost by `rhs_evals` or CPU, not by `nfev`.
- `rhs_evals` counts every evaluation of the run: the solver's, the Jacobian columns, one start check a month, the quadrature evaluations, and the rate evaluations for the outputs (month 1's start rates, for the start Hden, and the month-end Hden). It is always above the sum of `nfev`.
- Under the EwE rule each right-hand-side evaluation is one derivative call, inside which handling time is solved first on the consumption kernel alone (Newton; `hden_passes_max` kernel passes at most, with the secant, undamped passes and `hybr` behind it). A model with pooled predation takes undamped passes instead.
- LT2022 is stiff (one eigenvalue of about 3300 per year at base, design section 2.2), so LSODA runs its stiff method there, with about 6 Jacobians a month, and BDF is cheaper (11.3 x against 17.3 x).
- REco by method was measured in phase 2 only (2 years, before phase 3a); LSODA 8.2 x, DOP853 11.2 x, Radau 10.5 x and BDF 9.9 x with stanzas. Those are not re-measured here and the LSODA figure is superseded by 4.7 x.

## Known limitations

- **Pooled models (the pooled fix, phase 3a).** Pooled models (ScrambleSelf > 0) failed the handling-time solve far from base because the port's scramble weights were wrong, not because the solve is weak: `rsim_params` fed VV into the weight formula unconverted, where Rpath converts it to a rate first (`vv_rate = VV x QQ / B_prey`), and the weights differed from Rpath's by up to 11.9 (some negative). With Rpath's formula, which matches `rsim.params` on REco to 1e-12 (`tests/data/rpath_reference/ecosim/suite_weights.csv`), pooled REco (ScrambleSelf and HandleSelf 0.5 on every link, effort 2, EwE rule, two years of LSODA) completes at CmCo 2 (17 passes at most) and CmCo 1000 (4), with no group below 0.83 of its base biomass (measured when the fix landed, 2026-11-15; not repeated in the later run). No `hden_tol` workaround is needed. A coupled failure that still occurs says "handling time did not converge for a pooled (ScrambleSelf > 0) map". The fix applies on every method: RK4 and AB of a pooled model now use Rpath's weights (a model with ScrambleSelf and HandleSelf 0 is unchanged), so results of pooled models on the parity paths moved when it landed (`deviations.md`, row "Rpath-wts"). Pooled runs complete; the passes per evaluation of a pooled model are undamped, so they stay higher than the Newton solve's.
- `annual_Qlink` and the Ecotracer step are month-end snapshots, without handling time or mediation (unlike the month-integral catch), and pair the month-end biomass with the month-start feeding time, as the parity paths order it (feeding time moved at most 3.3 % in a month, median 1.5 %, on REco with FtimeAdj 0.5 and effort 2). `annual_QB` is never filled, on any path.
- The catch integral is approximate under fleet mediation, because F is recomputed from the state within the month while the catch uses the month's F times the month-mean biomass.
- Forcing is held within the month unless `forcing_interpolation` is `"linear"` or `"smooth"` (which change the model); effort, forced F and catch and forced biomass are held within the month under every setting, and fleet dynamics runs at the month boundaries.
- A forced stanza group is overwritten by the age structure, on both paths.
- Stanza models are first order in the monthly split whatever the tolerance: the age structure is updated once a month from the month's mean rates.
- The budget residuals are None in a month with forced biomass, so a model that forces biomass in every month (LT2022) is not budget-checked on this path.
- A fishery on a detritus pool closes the budget under both rules: the 'rpath' rule takes the catch from the pool and `exchange_rate` counts its landed part; the EwE rule leaves the catch and its discards in place, as EwE does (`deviations.md`, R6), and books none of it as landed. On REco with 5 landed from Detritus, or forced F 0.1 on Detritus, `budget_residual` stays within 4e-9 a month under the EwE rule (1e-8 under 'rpath').
- The Black Sea costs 24.2 x RK4, LT2022 17.3 x on LSODA and 11.3 x on BDF (Cost); the Black Sea does not reach the 10-20 x of design section 4, which would need fewer evaluations (an analytic Jacobian, designed but not built).
- Quasi-steady fast groups are a model change, cost more on LSODA and BDF than the stiff solver already costs, give no budget residuals, and an unattractive or missing root raises `RuntimeError`.
- IBM groups are rejected.
- Ecospace still uses its own scheme (RK4 sub-steps with implicit movement); the continuous path is Ecosim only.
