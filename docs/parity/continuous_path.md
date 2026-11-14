# The continuous integration path

`rsim_run` has two kinds of integrator. RK4 (the default) and AB step EwE's and Rpath's monthly schemes; they are the parity paths and reproduce EwE 6.7.144 and Rpath. The continuous path, `method="lsoda"`, `"dop853"`, `"radau"` or `"bdf"`, solves the Ecosim equations themselves with SciPy's `solve_ivp`. It is **not EwE parity**: its numbers differ from EwE's and from the RK4 path, partly by design. This page says what it does, when to use it, how far it is from the RK4 path on three models, and what it costs. The measurements are from `packages/pypath/tests/test_ecosim_continuous_validation.py` and, where the text says so, from measurement scripts run beside it (2026-11-14, SciPy 1.18.1, one process on one machine). The design is in `docs/plans/2026-11-14-continuous-pathway-design.md`.

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
- **Handling time is exact.** Hden is solved at every state as the fixed point of the disc equation, from a recalibrated Hden0, instead of EwE's one damped pass per derivative call.
- **Month-end updates use month means.** The stanza age structure takes the month's mean food gain and loss rate (6-point Gauss-Legendre quadrature of the dense output); feeding time takes the month-mean consumption; `out_Catch` is the month's F times the month-mean biomass / 12, the integral of F x B. EwE's F x month-end biomass / 12 is kept in `out_Catch_month_end`.
- **Fixed detritus export.** DetritusOut is set once at the unforced base state, not by the first derivative of month 1.
- **Positivity.** A negative state is read as 0 inside the right-hand side and counted in `clips`; month-end biomass is floored at 1e-20.
- **Diagnostics.** `out.solver_info` holds the method, `rtol`, `atol`, the corrections applied, the EwE switches ignored, and per month `nfev`, `njev`, `nlu`, `status`, `t_reached`, `clips`, `hden_evals_max` and `min_state`. A failed month raises unless `on_failure="warn"`.
- **The parity paths are unchanged.** One scenario can be run on both paths; the continuous run does not modify it.

## When to use it

Use **RK4** (the default) whenever the result should be EwE's: comparing with EwE output, fitting to time series (EwE's sum of squares), Monte Carlo, Ecospace, and anything that needs many runs. Use **AB** only to reproduce Rpath's `adams_run`.

Use the **continuous path** when you want the solution of the equations rather than of EwE's monthly scheme:

- to check that a result is not an artefact of one RK4 step a month or of the fast-group blend;
- for closed mass budgets (R2, R5) and a balanced model that stays balanced (R3, R4): REco (without stanzas, EwE rule) at base effort stays within 3e-7 of its base over 50 years on LSODA, against 1.2e-4 on the RK4 path, all of which comes from EwE's Hden start (R3; with R3 removed the RK4 path stays within 6e-14);
- as a reference when testing a numerical change to the parity paths.

Which method:

- **`lsoda`** (the default choice): switches between non-stiff and stiff stepping on its own, and is the only sensible choice with fast groups.
- **`dop853`**: an explicit high-order method, the reference for non-stiff models at tight tolerance. On a stiff model it is limited by stability, not accuracy.
- **`radau`** and **`bdf`**: implicit alternatives for stiff models.

Tolerances: the default is `rtol` 1e-6 and `atol` 1e-8 x max(B0, 1e-6 x max(B0)) per group. **`atol` does not follow `rtol`**: with `rtol` 1e-10 and the default `atol`, REco's run is good to 4.7e-9 only. To tighten a run, pass both, for example `atol = default_atol(b0) * rtol / 1e-6` (`pypath.core.ecosim_continuous.default_atol`), which keeps the defaults' ratio and their floor for groups with a tiny or zero B0.

## The corrections it applies

The continuous path applies these by default and lists them in `solver_info["corrections"]`; the parity paths keep EwE's behaviour. IDs refer to `docs/parity/deviations.md`.

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
| Feeding time | `FTIME_START_OF_MONTH`: from the start-of-month consumption | From the month-mean consumption, held within the month |
| Sub-steps | `RK4_STEPS` | Ignored; the solver picks its steps |

The four switches in the last rows are reported in `solver_info["ignored_switches"]` when the scenario sets them. Under the `"rpath"` rule only `RK4_STEPS` is set, and handling time is off.

## Validation

There is no EwE oracle for this path, so it is checked against itself and against RK4 where the two define the same equations (design section 5).

| Check | Model | Result |
|---|---|---|
| RK4 converges to DOP853 (rtol 1e-12, atol 1e-14 x B0) as `RK4_STEPS` doubles | REco without stanzas, `"rpath"` rule (no stanzas, no handling time, fast groups integrated by RK4, so the equations coincide), effort 2, month 12 | e_4 = 1.73e-9, e_8 = 6.03e-11, e_16 = 2.81e-12: the error falls 28.6 and 21.5 times per doubling, fourth order |
| LSODA against DOP853, both rtol 1e-8 | REco with stanzas, EwE rule, effort 2, 2 years | 3.0e-8 max relative difference |
| Radau and BDF against LSODA, rtol 1e-8 | REco without stanzas, EwE rule, effort 2, 1 year | Radau 7.9e-8, BDF 1.0e-6 |
| A balanced model stays balanced, LSODA, default tolerances | REco without stanzas, base effort, no forcing, 50 years | 3.0e-7 (EwE rule), 3.1e-7 (`"rpath"`) from B0; flat over the 50 years, solver error at rtol 1e-6, not drift (the start derivative is 4e-14 x B) |
| Ten years run, LSODA | REco, the Black Sea, LT2022 | Finite and positive; every month status 0, no clips |

A DOP853 reference at rtol 1e-10 is not accurate enough for the convergence check. Measured against rtol 1e-13, it is good to 4.7e-9 only at the default `atol`, and to 7e-12 to 1.1e-11 with an `atol` of 1e-10 x B0 or less, which still exceeds e_16. At rtol 1e-12 the error is 7.5e-14.

## Differences from the RK4 path

Ten years, LSODA at the default tolerances against `method="RK4"` on the same scenario. `|B_lsoda / B_RK4 - 1|` over months 1-120 and every living group (living and detritus groups in brackets):

| Model | Scenario | Median | Max |
|---|---|---|---|
| REco (Rpath), with stanzas | EwE rule, effort 2 from month 1 | 0.017 % (0.017 %) | 18 %, Seabirds (93 %, Discards) |
| Black Sea (EwE-F test case) | scenario 1 | 5.5 % (5.0 %) | 38 %, Zooplankton (38 %) |
| LT2022 | scenario 16, time series dataset 35 | 3.7 % (3.6 %) | 115 %, Zooplankton (115 %) |

Most of these differences are definitional. To see which, the RK4 path was moved towards the continuous definitions one item at a time and compared with the same LSODA run. These were measurement scripts, not tests: `RK4_STEPS` and `RK4_FAST_EQUILIBRIUM` set on the scenario; R3 removed by starting Hden at CmCo / (CmCo - 1); DetritusOut pre-filled at the unforced base state by wrapping `deriv_vector`. Median / max over living groups:

| RK4 path | REco | Black Sea | LT2022 |
|---|---|---|---|
| As is (one step a month, EwE rule) | 0.017 % / 18 % | 5.5 % / 38 % | 3.7 % / 115 % |
| 32 sub-steps (LT2022: 128) | 0.017 % / 18 % | 2.7 % / 32 % | 0.56 % / 161 % |
| and fast groups integrated | 0.017 % / 18 % | 2.7 % / 30 % | 0.44 % / 9.6 % |
| and R3 removed | 0.014 % / 18 % | 2.7 % / 31 % | 0.44 % / 9.6 % |
| and DetritusOut at the base state | 0.0017 % / 0.069 % | 2.7 % / 31 % | 0.44 % / 9.6 % |

- **REco: where DetritusOut is set.** The scenario doubles effort from month 1. The RK4 path, as EwE, sets each detritus pool's export rate in the first derivative of month 1, with the doubled discards already flowing, so Discards stays near its base (1.15 at year 10, B0 1.22). The continuous path sets it at the base state, so Discards rises to 2.21, and the Seabirds follow. DetritusOut at the base state alone brings the paths within 0.069 %; so does running month 1 at base effort on both paths (0.072 %).
- **Black Sea: one step a month, then handling time.** Half the median difference is RK4's single step a month (32 and 128 sub-steps give the same result). Most of the rest is handling time at the model's CmCo 2: with CmCo 1000 for every group on both paths, the RK4 path with 32 sub-steps and fast groups integrated comes within 0.43 % median of LSODA. The handling-time items are the recalibrated Hden0 (lower for predators with diet import) and the exact fixed point in place of EwE's damped pass carried between calls. The maximum is juvenile anchovy (Anchovy 0-0+), still 26 % at CmCo 1000 and 5.5 % apart at the end of month 1, when the scenario's nutrient forcing moves the whole web (both paths leave the base by up to 17 % in month 1; with neutral forcing the two anchovy stanzas stay within 0.2 % of their base on both). That is the stanza update: fed EwE's rates (consumption at the month-end state, 5.6 % below the month mean in month 1), the continuous path's anchovy follows the RK4 path's (32 sub-steps, fast groups integrated) to 4 digits, and over 10 years the paths then come within 0.38 % median. The largest remaining difference is Aurelia: 5e-6 in month 1, growing to 23 % by year 10. It was not attributed further; the corrections still active on this model are DF once and the exact handling time (R2, R4, R5 and feeding time do nothing here, and DetritusOut was measured to change nothing).
- **LT2022: one step a month, then the fast-group blend.** 128 sub-steps (its stiffest eigenvalue, about 3300 per year, needs more than about 100 for RK4 to be stable; 256 give the same result) take the median from 3.7 % to 0.56 %. The maximum is Zooplankton, a fast group, under EwE's blend with the equilibrium; integrating it takes the maximum from 161 % to 9.6 %. R3 and DetritusOut change nothing here. What remains, 0.44 % median, comes from the other corrections (handling time, the stanza update, feeding time from month means, R2, R4, R5, DF once) and was not attributed further.

Setting `STANZA_STAGE_RATES`, `FTIME_START_OF_MONTH` or `DETRITUS_MO_UNSCALED` to False on the RK4 path is not a step towards the continuous path: the first two select Rpath's rules, not month means, and moved LT2022 further away (1.9 % median, 34 % max).

## Cost

LSODA at the default tolerances, 10 years, from `solver_info` and the test's wall clock:

| Model | Groups | `nfev` / month (mean, max) | `hden_evals_max` (largest month) | Wall s / year, RK4 | Wall s / year, LSODA | Ratio |
|---|---|---|---|---|---|---|
| REco, with stanzas | 22 | 33, 37 | 4 | 0.12 | 2.6 | 21 x |
| Black Sea | 22 | 87, 115 | 132 | 0.045 | 40.7 | 914 x |
| LT2022 | 24 | 214, 443 (5.1 Jacobians) | 4 | 0.105 | 13.1 | 125 x |

REco by method, 2 years at the default tolerances, EwE rule, effort 2:

| Model | Method | `nfev` / month | Jacobians / month | All right-hand-side calls / month | Wall s / year | Ratio to RK4 |
|---|---|---|---|---|---|---|
| REco, no stanzas | lsoda | 19 | 0 | 20 | 1.8 | 11 x |
| | dop853 | 19 | 0 | 20 | 1.8 | 11 x |
| | radau | 16 | 1.0 | 39 | 3.1 | 18 x |
| | bdf | 18 | 1.0 | 42 | 3.3 | 20 x |
| REco, stanzas | lsoda | 34 | 0 | 35 | 2.8 | 23 x |
| | dop853 | 50 | 0 | 51 | 3.8 | 32 x |
| | radau | 23 | 1.0 | 46 | 3.5 | 29 x |
| | bdf | 19 | 1.0 | 43 | 3.3 | 28 x |

What the call counts mean:

- `nfev` counts the solver's calls of the right-hand side. For LSODA it includes the finite-difference Jacobian columns; for Radau and BDF it does not (SciPy builds their Jacobian outside the count), hence the separate column: about `nfev` + Jacobians x groups.
- Each month adds calls that are not in `nfev`: one start check, six quadrature evaluations, and the rate evaluations for the outputs (Hden at the month end; the month-start predation when RiskTime is set).
- Under the EwE rule each right-hand-side call solves handling time first: `hden_evals_max` derivative evaluations at most, then one more. At CmCo 1000 that is 4 or fewer. At the Black Sea's CmCo 2 the undamped iteration does not always reach its tolerance (1e-12) in 30 passes, and SciPy's `hybr` finishes, so a call can take up to 132 derivative evaluations. That is where the Black Sea's time goes: with CmCo 1000 for every group (a measurement script) the same 10 years take 3.3 s a year, with the same `nfev`.
- LT2022 is stiff (one eigenvalue of about 3300 per year at base, design section 2.2), so LSODA runs its stiff method there, with about 5 Jacobians a month.

## Known limitations

- `annual_Qlink` and the ecotracer are month-end snapshots, without handling time or mediation, unlike the month-integral catch. `annual_QB` is never filled, on any path.
- The catch integral is approximate under fleet mediation, because F is recomputed from the state within the month while the catch uses the month's F times the month-mean biomass.
- RiskTime reads the month-start predation over the month-end biomass.
- A forced stanza group is overwritten by the age structure, on both paths.
- Stanza models are first order in the monthly split whatever the tolerance: the age structure is updated once a month from the month's mean rates.
- EwE's mediation lookup is still truncated to its 1200 points, so a mediated model's right-hand side is piecewise constant in the state; design section 2.8 plans to interpolate it.
- Feeding time, forcing, effort and forced catch are held within the month (feeding time becomes a state in phase 2).
- IBM groups are rejected.
- Ecospace still uses its own scheme (RK4 sub-steps with implicit movement); the continuous path is Ecosim only.
