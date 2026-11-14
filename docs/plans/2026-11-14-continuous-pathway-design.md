# A third integration pathway for PyPath: continuous dynamics with DOP853 and LSODA

Status: approved 2026-11-14 (decisions in section 11); phase 1 in progress.

Paths are relative to the repository root. `ecosim.py` and `ecosim_deriv.py` are in `packages/pypath/src/pypath/core/`. Line numbers are from `main` at 36ae98b. Numbers marked "measured" come from read-only spikes (section 8).

## 1. Goal and position

| Pathway | Selected by | Aim |
|---|---|---|
| EwE parity | `fast_group_rule="ewe"`, RK4, one step a month | reproduce EwE 6.7.144 |
| Rpath parity | `fast_group_rule="rpath"`, RK4 with 4 sub-steps, or AB | reproduce Rpath |
| **Continuous (new)** | `method="lsoda"` or `"dop853"` | solve the model's equations to a stated tolerance |

The new pathway does not try to match EwE's or Rpath's numbers. Where a parity result depends on step size, stage order or the number of derivative calls, the new path defines the underlying equation and solves that. The parity paths keep their results.

Today's switches (`ecosim.py:946-959`) fall into three kinds:

- **Equation switches**, inherited from the scenario: `PREY_FTIME_ARENA`, `LINK_HANDLING_TERM` / `EWE_HANDLING_TIME`, `EWE_MO_TOT`, `STANZA_WALFORD_PLUS_GROUP`, the detritus fate, `NutBaseFreeProp`.
- **Numerics switches**, ignored by the new path: `RK4_FAST_EQUILIBRIUM`, `RK4_STEPS`, `STANZA_STAGE_RATES`, `FTIME_START_OF_MONTH`, the per-call Hden update, the first-call DetritusOut.
- **Defect corrections**, which the new path applies and which do change the equations: `DETRITUS_MO_UNSCALED` off and the other items of table 2.9. Whether these are on by default is an open question (section 10).

A scenario built under either rule can therefore run on the new path: it keeps that rule's functional forms and drops its stepping mechanics.

## 2. The continuous right-hand side

State: the biomass of groups 1..n_bio (living and detritus). Slot 0 and the gear slots are not states. Phase 2 adds feeding time.

`deriv_vector` (`ecosim_deriv.py:1578`) stays the single source of the flow terms. The new path calls it through a wrapper, `pure_rhs(t, y)`, that depends on `(t, y)` only.

### 2.1 Handling time (Hden): an algebraic function of the state

Today, under the EwE rule, each integration call advances Hden with damping 0.5 and writes it back (`ecosim_deriv.py:1929-1942`; `_ewe_hden_update`, `:1466`). The result depends on how many times the derivative is called. `_frozen_hden` (`ecosim.py:3041`) does not make a call pure either: it stops the write-back, but the consumption is still computed at the stepped value (`ecosim_deriv.py:1937-1940`).

EwE's update is one damped pass of a fixed-point iteration for the disc equation Hden_j = 1 + Htime_j x sum(a x V), where V depends on every predator's Hden (`cEcoSimModel.vb:2890-2928`). **Decision: Hden is the solution of that fixed point at the current state**, solved in the wrapper from a fixed cold start to a tight tolerance and never carried between calls. This needs a third mode in the Hden block, "use the Hden given, no update", for the final evaluation.

Measured:

- EwE's damped pass needs 29 to 55 passes to reach 1e-12; the undamped map needs 4 to 5 at CmCo 1000 (LT2022) and 20 at CmCo 2 (Black Sea, base state).
- At CmCo 2 the undamped map did not converge in 300 passes on 2 of 6 strongly perturbed states. A safeguarded solver (Anderson acceleration or Newton) is required.
- Only the consumption kernel is needed for the iteration, not the whole derivative.

Start of a run:

- **EwE's start value CmCo / (CmCo + 1) is dropped** (R3; `ecosim.py:2337-2351`, `:2431-2436`).
- **Hden0 is recalibrated to the fixed point at the base state.** EwE's Dzero = CmCo / (CmCo - 1) is that fixed point only when all food comes through links, because the update sums link consumption only (`ecosim_deriv.py:1937`). A predator with diet import has a lower one: 3e-4 lower for LT2022's Piscivorous birds, 0.33 lower for the Black Sea's importers at CmCo 2 (measured). The new path sets Hden0_j = 1 / (1 - q0_j / (CmCo_j x QB_j)), with q0 the kernel's own base link consumption per unit predator. Measured: this is an exact fixed point (residual 2e-16) and both models then start balanced.

### 2.2 Fast groups: integrated as stiff states

Today groups with loss/B above 24 /yr are flagged (`_fast_group_flags`, `ecosim.py:1165`) and, under the EwE rule, set to `(1 - SorWt) x biomeq + SorWt x B` at every stage (`_rk4_fast_equilibrium`, `ecosim_deriv.py:2672`): a lagged blend, neither the equilibrium nor the ODE.

**Decision: every group is a state; `NoIntegrate` is ignored.** The flag exists because explicit monthly RK4 cannot take these groups. Measured stiffness at base: LT2022 has one eigenvalue of 3321 /yr (then 98 and 13); the Black Sea's largest is 156 /yr. A later opt-in `fast_groups="quasi-steady"` (the exact equilibrium, gain / loss rate) would let DOP853 run stiff models; it is a model reduction and must be labelled so.

### 2.3 Feeding time (Ftime): a relaxation ODE; monthly-held in phase 1

The monthly rule `new = L + (1 - L) x Ft x ((1 - adj) + adj x Qopt / (Q/B))` (`_ftime_update_rpath`, `ecosim.py:3073`; applied at `:2677-2735`) is one forward-Euler step of one month of

    dFt/dt = 12 x [ L + (1 - L) x Ft x ((1 - adj) + adj x Qopt / (Q/B)) - Ft ]    per year

clipped to [L, FtimeMax]. **Decision: this ODE is the definition**, and Ftime becomes a state for groups with FtimeAdj > 0 in phase 2. Phase 1 holds Ftime within the month and updates it at the boundary from the month-average Q/B.

### 2.4 Detritus export (DetritusOut): a parameter fixed at setup

Today `_det_out` is filled by the first derivative of month 1 (`ecosim.py:2340-2342`, `:2437-2441`; `ecosim_deriv.py:780`, `:920`), so it depends on month-1 forcing and the Hden transient. **Decision: DetritusOut = (inflow0 - BA - eaten0) / B0 - Emig, at least 1e-20, evaluated once at the unforced base state** and pre-filled before the run. Ecospace already pre-computes it (`spatial/integration.py:78`, `:1201`).

### 2.5 Forcing in time

The caller builds one forcing dict per month (`ecosim.py:2381`). **Decision: piecewise-constant monthly forcing stays the default**, with the solver restarted at each boundary: the inputs are monthly numbers, a jump inside a step defeats error control, and results stay comparable with the parity paths. `forcing_interpolation="linear"` is a phase-3 option. It is a different model (a step in effort becomes a ramp) and must preserve monthly means; its gain is that runs without stanzas need not restart monthly, and restarts are most of LSODA's cost (section 4).

### 2.6 Fishing and effort

F = sum(effort x q) on the current biomass is already continuous (`fishing_mortality`, `ecosim_deriv.py:1499`; `:1987`). Effort, forced F, fleet dynamics and quotas stay monthly-held (`ecosim.py:2421-2429`, `:2788`). Forced catch stays a rate fixed at the start of the month (`_forced_fishing_rate`, `ecosim.py:1104`) in phase 1; the realised catch is reported and can differ from the forced one.

### 2.7 Nutrient limitation

`_ewe_nutrient_factor` (`ecosim_deriv.py:1446`, applied at `:1713-1729`) is already a pure function of the state and is kept. Its floor at 0.00101 x NutFree0 is a kink, and it couples every producer to every group.

### 2.8 Other purity items

- **Forced biomass** is overwritten in the derivative (`ecosim_deriv.py:1980`) and around the step (`ecosim.py:2415-2420`, `:2588-2589`). In the new path a forced group is held at its value for the month with a zero derivative.
- **IBM groups** (`ecosim_deriv.py:2031`) are rejected in phase 1.
- **Mediation**: EwE's truncated 1200-point lookup (`mediation.py:296`) is piecewise constant in the state; the new path interpolates it.
- **Floors** (0 per sub-step, `ecosim_deriv.py:2783`; 1e-10 at month end, `ecosim.py:2593-2594`) are replaced by section 4's positivity treatment.

### 2.9 EwE behaviours that deviations.md records as reproduced on purpose

| ID | EwE behaviour | New path |
|---|---|---|
| R1 | UseVarPQ stored True loads as False | Unchanged (a reader matter) |
| R2 | Detritus gets mo x B; the group loses the Ftime-scaled flow | Closed budget (`DETRITUS_MO_UNSCALED` off, `ecosim_deriv.py:2145`) |
| R3 | Run starts from Hden = CmCo / (CmCo + 1) | Dropped |
| R4 | Ecosim ignores EmigRate for living groups; a balanced model drifts | Use EmigRate x B |
| R5 | With PaddP < 1 detritus gets mo x B; the group loses moTot x B | Closed budget. Needs a new flag: the moTot factor (`ecosim_deriv.py:2152-2161`) is applied after `_M0_detritus` is set |
| DF | DetPassedProp x DF applies DF twice | Kept in phase 1 (DetritusOut absorbs it at base) |

## 3. Discrete events

**Decision: integrate month by month**, one `solve_ivp` call per month with dense output; month boundaries carry the discrete updates. A continuous-age stanza model was rejected: it is a different model, and Ecosim's stanzas are defined as monthly cohorts.

Order in month m:

1. Set held values: forcing, effort, forced F, forced biomass, `StanzaPredRatio`, Ftime (phase 1).
2. Integrate.
3. By 6-point Gauss-Legendre quadrature on the dense output: month-average FoodGain and LossPropToB / B per group, catch and consumption per link.
4. Stanza update from those averages, then `split_set_pred`.
5. Feeding-time update (phase 1), fleet dynamics, quotas, outputs.

**What feeds `split_update`.** The EwE path takes growth from the last RK4 stage and survival from the first (`ecosim.py:2492-2498`, `:2607-2615`). The new path passes the month's **time averages**: mean FoodGain, and mean(LossPropToB / B) presented as a loss at unit biomass. Survival, `exp(-LossPropToB / 12 / B)` (`stanzas.py:922`), is then exp(-integral of Z dt), exact for a cohort under a varying Z. `biomass` is used only in that ratio, so the signature does not change.

**Limit.** Within the month the stanza groups are integrated as ordinary states, then overwritten by sum(N x W) (`ecosim.py:2637`), with `StanzaPredRatio` held. For models with stanzas the result is first order in the monthly split whatever the tolerance. Moving cohort numbers and weights into the state vector is phase-4 material.

**Catch** is the integral of F x B over the month. EwE books F x month-end B / 12 (`ecosim.py:2751-2755`); that figure is kept in a second field for `ewe_sum_of_squares`.

## 4. Solver use

**Roles.** LSODA is the default and the only sensible choice with fast groups. DOP853 is the reference for non-stiff models at tight tolerance; on a stiff model it is limited by stability (618 derivative calls a month on LT2022, measured).

**Tolerances.** Biomasses span 1e-6 to 1e2 t/km2, so `atol` is a vector: `rtol = 1e-6`, `atol_i = 1e-8 x max(B0_i, 1e-6 x max(B0))`. A "reference" preset uses `rtol = 1e-10`.

**Steps.** `max_step` is one month by construction. `first_step` is left to the solver in phase 1; reusing the previous month's last step comes in phase 2.

**Positivity.**

| Option | For | Against |
|---|---|---|
| Integrate ln B | guaranteed; relative error control | fails at B = 0 (extinct or forced-to-zero groups); conservation error becomes relative |
| Clip in the wrapper and at month end | simple; equations unchanged for B > 0 | small negative overshoots possible |

Decision: clip in phase 1, since losses are proportional to B and the exact solution stays positive; record the minimum and any clip; floor at 1e-20 at month end. Log-transform is a phase-3 option.

**Jacobian.** Not sparse at this size: measured fill 64 % (LT2022, 24 states) and 40 % (Black Sea, 22), because nutrient limitation, shared arenas, mediation and prey switching add dense rows on top of the diet and detritus links. SciPy's LSODA takes dense or banded Jacobians only (`jac_sparsity` is "not supported by 'LSODA'", SciPy 1.18.1). Dense finite differences are fine at 20 to 60 states, but they are where the calls go: LT2022 needed 2.6 Jacobians a month at 24 calls each, because each monthly restart builds a new one. An analytic numba Jacobian of the arena kernel is worth doing in phase 3.

**Cost, measured** over one simulated year, monthly restarts, month-1 forcing held, unoptimised wrapper:

| Model | Method | Derivative calls / month | Wall s / year |
|---|---|---|---|
| LT2022 (max eigenvalue 3321 /yr) | RK4, EwE path | 4 plus auxiliary | 0.16 |
| | LSODA, rtol 1e-6 | 117 | 12.7 |
| | LSODA, rtol 1e-8 | 149 | 16.9 |
| | BDF / Radau, rtol 1e-6 | about 40 | 4.5 |
| | DOP853, rtol 1e-8 | 618 | 68 |
| Black Sea (max eigenvalue 156 /yr) | RK4, EwE path | 4 plus auxiliary | 0.055 |
| | LSODA, rtol 1e-6 | 29 | 10.3 |
| | DOP853, rtol 1e-8 | 47 | 16.5 |

LSODA makes about 7 to 30 times the derivative calls of RK4. Wall time in the spike was 80 to 190 times RK4, because the wrapper iterated the whole derivative (0.8 to 1.9 ms a call) for Hden: 5 passes at CmCo 1000, about 30 at CmCo 2. With the Hden solve on the kernel alone, an accelerated iteration and a supplied Jacobian, 10 to 20 times RK4 is a reasonable target; that figure is an estimate.

## 5. Validation without an EwE oracle as truth

1. **Balanced start.** With no forcing, dB/dt at t = 0 equals the Ecopath biomass accumulation, and a 50-year run stays there. Measured with Hden held at the recalibrated Hden0: the Black Sea's only residuals are its two detritus pools growing at exactly DetBA / B; LT2022's are two bird groups at -0.1 /yr (consistent with an entered BA, not confirmed) and Zooplankton at +0.002 /yr (not attributed). Accounting for those is the first task of phase 1.
2. **Convergence.** On a model with a pure right-hand side on both sides, RK4 with `RK4_STEPS` doubled must approach DOP853 at rtol 1e-10 with the error falling 16 times per doubling. REco qualifies under the `"rpath"` rule: it has no stanzas, and that rule integrates its 3 flagged groups in RK4 (measured: 2 flagged under `"ewe"`, 3 under `"rpath"`), with FtimeAdj 0 and enough sub-steps for stability.
3. **Self-consistency.** LSODA against DOP853 on REco; LSODA at rtol 1e-6 against 1e-10.
4. **Fast groups.** On LT2022 and the `_stiff_model` fixture of `tests/test_fast_group_rule.py`, fast groups must track gain / loss rate to within the ratio of time scales.
5. **Conservation.** With R2 and R5 closed, each month's change in total biomass equals the integral of production + import + immigration - respiration - catch - export - emigration, to solver tolerance; likewise for each detritus pool. Nutrient: free + bound = NutTot x shape(t) while the floor is inactive.
6. **Agreement where definitions coincide.** No fast groups, stanzas, feeding-time adjustment or importers, CmCo 1000, constant forcing: the EwE path with `RK4_STEPS` raised must converge to the new path. Where definitions differ, the difference is measured on LT2022 scenario 16 and the Black Sea case and recorded in a new `docs/parity/continuous_path.md`.
7. **Data.** `tests/data/rpath_reference/` (REco), `Data/LT2022_0.5ST_final7.eweaccdb` scenario 16, `Data/BlackSea_EwEF_testcase.eweaccdb` (CmCo 2, NutBaseFreeProp 0.9, detritus BA), and the LT2022 PaddP, migration and negative-mo variants. Tests reading `Data/` are marked `integration`.

## 6. API and placement

**Selection.** `rsim_run(scenario, method="lsoda" | "dop853", solver_options=None)` (`ecosim.py:1665`); options are `rtol`, `atol`, `corrections`, `on_failure`, and later `forcing_interpolation`, `fast_groups`, `positivity`.

**With EwE-rule switches.** The rule is fixed when parameters are built (`rsim_params`, `ecosim.py:541`); the method arrives at run time. The new path reads the equation switches, ignores the numerics switches, applies the defect corrections, and logs once at INFO which switches it ignored and which corrections it applied. It does not mutate the scenario, so one scenario can be run on both paths. Roadmap decision D1 is unaffected: `compat` picks the functional forms; `method` picks the stepping and, for the continuous methods, the corrections.

**Behaviour change.** An unknown method silently runs RK4 today (`ecosim.py:2582-2587`), so `method="lsoda"` currently returns RK4 output. Phase 1 makes an unknown method raise `ValueError`.

**Placement.** The run setup and monthly forcing build are inline in `rsim_run` (about `ecosim.py:1700-2440`), so an early dispatch would duplicate them. Phase 1 instead **branches at the integration step** (`ecosim.py:2500`) into a new module `core/ecosim_continuous.py` (wrapper, Hden solver, month integrator, quadrature), with guarded branches for the Hden start, the DetritusOut pre-fill, the stanza rates and the catch. The parity branches are untouched and the existing suite must pass unchanged. Extracting the setup into a shared helper is left for phase 2. `ecosim_deriv.py` gains only flags read from `params`: exact-Hden mode, R5 closure, interpolated mediation.

**Diagnostics.** `RsimOutput.solver_info`: method, tolerances, per-month `nfev`, `njev`, steps, status, Hden iterations, minimum state, clip count, ignored switches, applied corrections, conservation residual. A failed month raises unless `on_failure="warn"`.

**Shiny.** The Ecosim page already passes an `integration_method` input to `rsim_run` (`packages/pypath-shiny/src/pypath_shiny/pages/ecosim.py:235`, `:1946-1952`). Add the two methods under a "Continuous (not EwE parity)" label with a tolerance preset and a diagnostics panel, after phase 2.

## 7. Ecospace

`rsim_run_spatial` runs RK4 sub-steps with backward-Euler movement as a Lie split (`spatial/integration.py:1210-1255`; `dispersal.py:599`). The brief's `spatial/stanza_space.py` does not exist in this tree; the map-wide stanza update is in `integration.py`.

Method of lines gives about 300 x 25 = 7500 states. For LSODA that is impractical in SciPy: a dense Jacobian costs 7500 derivative calls to build; a banded one (patch-major ordering, half-bandwidth about groups x grid row length) still costs on the order of a bandwidth of spatial derivative calls per Jacobian and does not fit irregular grids. BDF and Radau accept `jac_sparsity`, so they are the feasible implicit route, at an estimated one to two orders of magnitude above the current scheme (not measured).

**Decision: Ecospace stays on its current scheme.** The useful step, in phase 4, is the continuous path for the local reaction in each patch (a 25-state LSODA solve per patch per month) inside the existing split with implicit movement, preferably as a Strang split. That removes the fast-group blend and the Hden iteration from Ecospace; the splitting error remains and must be reported.

## 8. Measurements and limits

The spikes (not committed) ran with SciPy 1.18.1, captured the arguments of `rsim_run`'s first `integrate_rk4` call and called `deriv_vector` directly. Limits: month-1 forcing held for the year; stanza groups integrated without a monthly update; Hden iterated on the full derivative.

Not verified: the cause of LT2022's three base residuals; DOP853's stability limit in theory; the AB floor (Bref x 1e-15) and EwE's 1e-20 floor, both from the earlier review; every Ecospace cost figure; the 10 to 20 times cost target.

## 9. Phased plan

| Phase | Deliverable | Tests | Effort |
|---|---|---|---|
| **1. Ecosim core** | Pure wrapper (2.1, 2.4, 2.8, 2.9); LSODA and DOP853, month-segmented; fast groups as states; Ftime monthly-held; quadrature for stanzas and catch; `solver_info`; `ValueError`; IBM rejected | Purity (same `(t, y)` gives identical output after unrelated calls); balanced start; RK4-to-DOP853 convergence; LSODA against DOP853; conservation; parity suite unchanged | 6 to 9 days |
| **2. Feeding time, speed** | Ftime as a state; Hden on the kernel only; step reuse; shared setup helper; `docs/parity/continuous_path.md` | Ftime ODE against the monthly rule; timing budget | 5 to 7 days |
| **3. Options** | Analytic Jacobian; linear forcing; quasi-steady fast groups; log-transform; Shiny | Jacobian against finite differences; monthly means preserved; Shiny smoke test | 5 to 8 days |
| **4. Stanzas, Ecospace** | Cohorts as states; per-patch continuous reaction | Stanza convergence; a uniform map reproduces Ecosim | 2 to 3 weeks; decide after phase 2 |

Risks:

- **Cost**: 80 to 190 times RK4 as spiked; fitting and Monte Carlo are impractical on this path before phase 2.
- **Hden at low CmCo**: the bare fixed point can fail at CmCo 2, and a failure inside the right-hand side fails the run.
- **Stanza models are first order in the month** whatever the tolerance.
- **Non-smooth terms** (clips, floors, lookups) cost steps and can stall an implicit solver.
- **Two semantics from one scenario**: differences between `method="RK4"` and `"lsoda"` are partly definitional; `solver_info` and the divergence document must show which.
- **`rsim_run` is edited** at the integration step, so the parity suite is the guard.

## 10. Open questions for the user

1. Are the defect corrections (R2, R4, R5; and DF applied once) on by default in the new path, or opt-in?
2. Is the Ftime ODE with rate 12 /yr the definition, or should feeding time stay a monthly rule?
3. Expose Radau and BDF as well? They were about 3 times cheaper than LSODA on LT2022.
4. Forced catch: a start-of-month rate (catch may miss the target) or a constant removal (hits the target, can empty a group)?
5. Should an unknown `method` raise? It changes behaviour for misspelt names.
6. Which catch figure is primary in the output: the integral, or EwE's month-end booking?
7. Should importers' food count in the handling budget, in place of the recalibrated Hden0? It changes their dynamics away from base.

## 11. Decisions taken (2026-11-14)

The recommendations were adopted; questions 4 and 7 had no recommendation and were settled by the implementer.

1. The defect corrections R2, R4, R5 and DF applied once are on by default on this path; the parity paths keep EwE's behaviour.
2. The feeding-time relaxation ODE is the definition (phase 2); phase 1 holds feeding time within the month.
3. Radau and BDF are offered beside LSODA (default) and DOP853.
4. Forced catch is a rate fixed at the start of the month; the realised catch is reported.
5. An unknown `method` raises `ValueError`.
6. The primary catch figure is the integral of F x B over the month; EwE's month-end booking is kept in `out_Catch_month_end`.
7. Diet imports stay out of the handling budget; Hden0 is recalibrated to the base fixed point.
8. Mediation lookup interpolation (section 2.8) moved to phase 2, with the other smoothness items; phase 1 keeps EwE's lookup.
