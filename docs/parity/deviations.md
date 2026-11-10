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
| C1–C3 | Fitting | `fit_to_timeseries` fits catch to biomass, keeps one series per group, and uses a different q and sampling month. The LT2022 SS pin (631.7 since A1/A2, 637.0 before) is PyPath's own linear SS, not EwE's | P3 |
| D1 | Ecosim | Prey Ftime and vulnerability forcing scale consumption linearly; EwE saturates them in the arena | P4 |
| D2 | Ecosim | Feeding-time update timing, Q source, stanza denominator, groups updated, RiskTime | P4 |
| D3 | Ecosim | Mediation weights, normalisation, combination and application type | P4 |
| D4 | Ecopath | Entered EE is not recomputed; missing Q/B is not estimated; fractional Type is truncated | P5 |
| E | Stanzas | LeadingCB, BABsplit, plus-group weight, recruitment forcing, SpawnProp and related settings | P2 |
| F | Ecotracer | `core/ecotracer.py` is a simplified tracer, not EwE's Ecotracer (no environment pool; different uptake and losses) | P8 (relabel) |
| G | Various | Fishing, Ecosim options, Ecopath reporting and minor reader items listed in the audit's section G | P5, P7 |

## Accepted

| ID | Area | Difference | Reason |
|---|---|---|---|
| A2-space | Ecospace | Detritus import (DtImports) enters every patch's detritus, per unit area, so a balanced model stays balanced at t = 0. EwE 6.7.144 Ecospace leaves it out of the detritus inflow, `Immig x DtImpMult + ToDetritus` (`cSpaceSolver.vb:1043`, `cEcoSpace.vb:3514`), but keeps it in the export rate taken from Ecosim's first `Derivt` (`cEcoSpace.vb:2427`, `cEcoSimModel.vb:3206`), so EwE Ecospace detritus drifts down from t = 0 when there is an import | The EwE form loses mass with every step; PyPath's matches Ecosim, which EwE balances. Ecospace is not EwE-equivalent anyway (B1-B5). Rpath-compatible behaviour will be kept behind `compat="rpath"` (roadmap decision D1), not listed here |
