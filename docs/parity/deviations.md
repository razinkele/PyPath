# Known deviations from EwE 6.7.144

This file lists every place where PyPath knowingly behaves differently from EwE 6.7.144. It is part of the definition of parity in `docs/plans/2026-11-10-ewe-parity-roadmap.md`: a finding is closed only when it is fixed or listed here as an accepted deviation.

Source: the audit of 2026-11-10 against tag `v6.7.144` of Official-EwE/Eii.Ecopath. Line references point to that tag.

Status values:

- **Open:** a known difference that a roadmap phase will fix.
- **Accepted:** kept on purpose, with the reason given.

## Open (scheduled)

| ID | Area | Difference | Roadmap phase |
|---|---|---|---|
| A1-s2 | Reader / Ecopath | OtherMort, Immigration, Emigration and EmigRate are not read or modelled; the reader warns when they are non-zero | P5 |
| A1-rate | Reader / Ecopath | BiomAccRate on a group whose B is left to Ecopath: the reader warns and uses BA = 0. EwE solves BA = rate x B inside the balance (`cEcoPathModel.vb:467, 508, 2047`) | P5 |
| A1-det | Reader / Ecopath / Ecosim | Biomass accumulation on detritus groups is not modelled: the reader warns and sets it to 0. EwE subtracts it from the detritus export (`cEcoSimModel.vb:3206`) | P5 |
| B1–B4 | Ecospace | Dispersal units, capacity-driven movement, RK4 stability and static habitat capacity. **Ecospace runs of imported models are not EwE Ecospace output**, and the Shiny Ecospace page says so | P6 |
| B5 | Ecospace | EatEff and RelVulBad, migration maps, fitness-based emigration, advection mapping, frozen feeding time, zero spatial catch | P6 |
| C1–C3 | Fitting | Fixed: `ewe_sum_of_squares` is EwE's SS (`cEcoSimModel.vb:2575-2735, 3849-3966`) and `fit_to_timeseries` minimises it. Every reference series counts; Z = ln(obs / pred) at the start of month 6 for annual data (at each month's start for time-step data); catch is predicted as B x FishTime of the month before (`out_FishMort`); relative series take q = mean(Z); each series is weighted by its WtType, now read (with EwE's fallback to 1 when every weight is 0). On EwE's own biomass and FishTime it reproduces the oracle's SS to 1e-7 for LT2022 scenario 16 with dataset 35 (75.2231) and with dataset 22 (forced catches, absolute biomass, weight 100: 42447.14); PyPath's own runs follow to 0.02 % and 0.16 %. Still open: reference types 5, 7, 12, 13, 19, 20 and 104 (reported and left out; no database here has them), the time-step branch (no such dataset here to check against EwE), and EwE's search itself (v = 1 + exp(p) in blocks, the PP anomaly, AIC) | P3 |
| D1 | Ecosim | Fixed under the EwE rule: prey Ftime and vulnerability forcing enter the arena as EwE's vulnerable-pool inflow (`PREY_FTIME_ARENA`). The `"rpath"` rule keeps Rpath's linear prey term. Vulnerability-type mediation still scales consumption directly (see D3) | P4 |
| D2 | Ecosim | Fixed under the EwE rule: the timing (start-of-month state), the Q source (the derivative's consumption) and the stanza denominator (consumption per predator index, sum(N x WWa) for stanza groups; `cEcoSimModel.vb:1762, 3310`). Fast groups keep their Ftime, as in EwE (it updates integrated groups only). Still open: RiskTime | P4 |
| D3 | Ecosim | Mediation weights, normalisation, combination and application type | P4 |
| D5 | Ecosim | Fixed under the EwE rule: Rpath's per-link handling term DD / (DD - 1 + preyYY) is not used; EwE6 has none (`LINK_HANDLING_TERM`). Still open: EwE's own handling time, which acts on the predator's total intake (Hden = 1 + Htime x sum(a x V), Htime = pred / (CmCo x B0 x QB), `cEcoSimModel.vb:2896-2969`). CmCo is not read; at EwE's default 1000 the effect stays within 0.1 % unless a predator eats far more per head than at base | P4 |
| D4 | Ecopath | Entered EE is not recomputed; missing Q/B is not estimated; fractional Type is truncated | P5 |
| E | Stanzas | Fixed under the EwE rule: the age table (ends at 90 % of W_inf, `CInt(40.3978/K)`), the Ford-Walford plus group in the setup and in every month, and the static consumption weights WWa (`cEcoSimModel.vb:4288-4335, 1866, 3549`). `rpath_stanzas()` keeps Rpath's table by default. Consumption is anchored on the LeadingCB stage (both rules; `cEcoSimModel.vb:4180-4187, 4384-4398`). BABsplit is read: it adds to every stage's Z in the age table and, under the EwE rule, scales the base recruits by exp(BaB/12) (`cEcoSimModel.vb:4311, 4362`). The monthly stanza update takes EwE's rates under the EwE rule (`STANZA_STAGE_RATES`): growth from the step's last RK4 stage, survival from its first (`cEcoSimModel.vb:1700-1756`). Recruitment follows EwE's SplitUpdate under the EwE rule: RscaleSplit x egg x RzeroS x hatch, with (Eggs/BaseEggs)^RecPower only without a hatch-code shape, eggs = spawning biomass (no saturation); EcosimStanzaShape is read (`cEcoSimModel.vb:1895-1902, 3487-3498`). SpawnProp, FixedFecundity (eggs per individual at their base value; SpawnProp counted twice, as EwE) and RecStanza (linked recruitment) follow EwE's SplitInitialize/SplitUpdate (`cEcoSimModel.vb:1852-1861, 1905-1912, 3458-3466`); EggAtSpawn has no effect on EwE's model. No stanza item is open | P2 |
| H | Ecosim | Fixed: environmental response functions (`EcosimScenarioCapacityDrivers`) were not read. They are applied as EwE does (`cEcoSimModel.vb:4931, 5016`): Target 115 scales a consumer's search rate or a producer's production (`ForcedSearch`), Target 126 other mortality (`ForcedMort`). Still open: a response on a detritus group, which scales its import in EwE (reported and skipped) | — |
| F | Ecotracer | `core/ecotracer.py` is a simplified tracer, not EwE's Ecotracer (no environment pool; different uptake and losses) | P8 (relabel) |
| G | Various | Fishing, Ecosim options, Ecopath reporting and minor reader items listed in the audit's section G | P5, P7 |

## Accepted

| ID | Area | Difference | Reason |
|---|---|---|---|
| AB-fast | Ecosim | With `method="AB"`, fast groups (NoIntegrate) use Rpath adams_run's biomeq, TotGain / (TotLoss / B), which counts all consumption as gain and respiration as loss. EwE has only RK4 (`cEcoSimModel.vb:758`). Under RK4 with the EwE fast-group rule, PyPath uses EwE's net biomeq (`cEcoSimModel.vb:3112`) | AB reproduces Rpath's integrator, so its fast-group update stays Rpath's. EwE-equivalent runs use RK4, which is the default |
| ewe-default | Ecosim | The default fast-group rule is `"ewe"`, also for models built from Rpath-style CSVs. Such models therefore get EwE's fast-group biomeq, feeding-time update (start-of-month, from the derivative's consumption) and arena (D1) whenever those features are active (fast groups, FtimeAdj > 0, ForcedPrey != 1), EwE's stanza age table and plus group (E), no per-link handling term (D5), other mortality reaching detritus unscaled by feeding time (R2), and EwE's stanza-update rates (E) | `fast_group_rule="rpath"` reproduces Rpath's behaviour (roadmap decision D1, compat switch) |
| A2-space | Ecospace | Detritus import (DtImports) enters every patch's detritus, per unit area, so a balanced model stays balanced at t = 0. EwE 6.7.144 Ecospace leaves it out of the detritus inflow, `Immig x DtImpMult + ToDetritus` (`cSpaceSolver.vb:1043`, `cEcoSpace.vb:3514`), but keeps it in the export rate taken from Ecosim's first `Derivt` (`cEcoSpace.vb:2427`, `cEcoSimModel.vb:3206`), so EwE Ecospace detritus drifts down from t = 0 when there is an import | The EwE form loses mass with every step; PyPath's matches Ecosim, which EwE balances. Ecospace is not EwE-equivalent anyway (B1-B5). Rpath-compatible behaviour will be kept behind `compat="rpath"` (roadmap decision D1), not listed here |

## EwE 6.7.144 behaviour reproduced on purpose

PyPath follows what EwE 6.7.144 actually does, even where that comes from an EwE defect. The headless EwE oracle run is the evidence.

| ID | Area | EwE behaviour | PyPath |
|---|---|---|---|
| R1 | Reader / Ecosim | `EcosimScenario.UseVarPQ` stored True is loaded as False: `cEwEDatabase.ReadSafe` treats a Boolean True as its -9999 "ignore" value (`Convert.ToBoolean(-9999)` is True) and returns the default, False (`cDBDataSource.vb:3739`). The variable P/Q never runs; the oracle shows GE/GE0 = 1 for every LT2022 group in every month | `ecosim_scenario_from_ewemdb` sets `UseVarPQ = False` and warns when the stored flag is True. Set `params.UseVarPQ = True` to use the variable P/Q |
| R2 | Ecosim | Other mortality reaches detritus unscaled by feeding time: a group with MoPred > 0 loses mo x (1 - MoPred + MoPred x Ftime) x B (`cEcoSimModel.vb:3081`), but `SimDetritusMT` sends mo x B (times other-mortality forcing) to detritus (`:3160-3166`), so the detritus budget is not closed when Ftime != 1. LT2022 has MoPred = 1 for every group; the oracle's detritivores follow the unscaled flow | Under the EwE rule the same (`DETRITUS_MO_UNSCALED`); the 'rpath' rule sends the scaled flow and keeps the budget closed |

