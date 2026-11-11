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
| C1–C3 | Fitting | `fit_to_timeseries` fits catch to biomass, keeps one series per group, and uses a different q and sampling month. The LT2022 SS pin (value and history in `test_ecosim_lt2022_parity.py`) is PyPath's own linear SS, not EwE's (EwE's own SS for scenario 16 is 75.22) | P3 |
| D1 | Ecosim | Fixed under the EwE rule: prey Ftime and vulnerability forcing enter the arena as EwE's vulnerable-pool inflow (`PREY_FTIME_ARENA`). The `"rpath"` rule keeps Rpath's linear prey term. Vulnerability-type mediation still scales consumption directly (see D3) | P4 |
| D2 | Ecosim | Fixed under the EwE rule: the timing (start-of-month state), the Q source (the derivative's consumption) and the stanza denominator (consumption per predator index, sum(N x WWa) for stanza groups; `cEcoSimModel.vb:1762, 3310`). Fast groups keep their Ftime, as in EwE (it updates integrated groups only). Still open: RiskTime | P4 |
| D3 | Ecosim | Mediation weights, normalisation, combination and application type | P4 |
| D4 | Ecopath | Entered EE is not recomputed; missing Q/B is not estimated; fractional Type is truncated | P5 |
| E | Stanzas | Fixed under the EwE rule: the age table (ends at 90 % of W_inf, `CInt(40.3978/K)`), the Ford-Walford plus group in the setup and in every month, and the static consumption weights WWa (`cEcoSimModel.vb:4288-4335, 1866, 3549`). `rpath_stanzas()` keeps Rpath's table by default. Consumption is anchored on the LeadingCB stage (both rules; `cEcoSimModel.vb:4180-4187, 4384-4398`). Recruitment follows EwE's SplitUpdate under the EwE rule: RscaleSplit x egg x RzeroS x hatch, with (Eggs/BaseEggs)^RecPower only without a hatch-code shape, eggs = spawning biomass (no saturation); EcosimStanzaShape is read (`cEcoSimModel.vb:1895-1902, 3487-3498`). Still open: BABsplit, RzeroS x exp(BaB/12), SpawnProp and related settings, and the monthly rates SplitUpdate takes (EwE: BAvg and LossAvg averaged over the month; PyPath: the end-of-month state) | P2 |
| H | Ecosim | Fixed: environmental response functions (`EcosimScenarioCapacityDrivers`) were not read. They are applied as EwE does (`cEcoSimModel.vb:4931, 5016`): Target 115 scales a consumer's search rate or a producer's production (`ForcedSearch`), Target 126 other mortality (`ForcedMort`). Still open: a response on a detritus group, which scales its import in EwE (reported and skipped) | — |
| F | Ecotracer | `core/ecotracer.py` is a simplified tracer, not EwE's Ecotracer (no environment pool; different uptake and losses) | P8 (relabel) |
| G | Various | Fishing, Ecosim options, Ecopath reporting and minor reader items listed in the audit's section G | P5, P7 |

## Accepted

| ID | Area | Difference | Reason |
|---|---|---|---|
| AB-fast | Ecosim | With `method="AB"`, fast groups (NoIntegrate) use Rpath adams_run's biomeq, TotGain / (TotLoss / B), which counts all consumption as gain and respiration as loss. EwE has only RK4 (`cEcoSimModel.vb:758`). Under RK4 with the EwE fast-group rule, PyPath uses EwE's net biomeq (`cEcoSimModel.vb:3112`) | AB reproduces Rpath's integrator, so its fast-group update stays Rpath's. EwE-equivalent runs use RK4, which is the default |
| ewe-default | Ecosim | The default fast-group rule is `"ewe"`, also for models built from Rpath-style CSVs. Such models therefore get EwE's fast-group biomeq, feeding-time update (start-of-month, from the derivative's consumption) and arena (D1) whenever those features are active (fast groups, FtimeAdj > 0, ForcedPrey != 1) | `fast_group_rule="rpath"` reproduces Rpath's behaviour (roadmap decision D1, compat switch) |
| A2-space | Ecospace | Detritus import (DtImports) enters every patch's detritus, per unit area, so a balanced model stays balanced at t = 0. EwE 6.7.144 Ecospace leaves it out of the detritus inflow, `Immig x DtImpMult + ToDetritus` (`cSpaceSolver.vb:1043`, `cEcoSpace.vb:3514`), but keeps it in the export rate taken from Ecosim's first `Derivt` (`cEcoSpace.vb:2427`, `cEcoSimModel.vb:3206`), so EwE Ecospace detritus drifts down from t = 0 when there is an import | The EwE form loses mass with every step; PyPath's matches Ecosim, which EwE balances. Ecospace is not EwE-equivalent anyway (B1-B5). Rpath-compatible behaviour will be kept behind `compat="rpath"` (roadmap decision D1), not listed here |

## EwE 6.7.144 behaviour reproduced on purpose

PyPath follows what EwE 6.7.144 actually does, even where that comes from an EwE defect. The headless EwE oracle run is the evidence.

| ID | Area | EwE behaviour | PyPath |
|---|---|---|---|
| R1 | Reader / Ecosim | `EcosimScenario.UseVarPQ` stored True is loaded as False: `cEwEDatabase.ReadSafe` treats a Boolean True as its -9999 "ignore" value (`Convert.ToBoolean(-9999)` is True) and returns the default, False (`cDBDataSource.vb:3739`). The variable P/Q never runs; the oracle shows GE/GE0 = 1 for every LT2022 group in every month | `ecosim_scenario_from_ewemdb` sets `UseVarPQ = False` and warns when the stored flag is True. Set `params.UseVarPQ = True` to use the variable P/Q |

