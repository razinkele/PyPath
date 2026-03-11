# PyPath EwE Feature Roadmap

> **For agentic workers:** This is a high-level roadmap, not an executable plan. Each phase should be expanded into its own detailed implementation plan using superpowers:writing-plans before execution.

**Goal:** Close the gap between PyPath and native EwE 6.6+ desktop, enabling full-fidelity model exchange and feature parity for the most impactful EwE capabilities.

**Current state:** PyPath implements Ecopath mass balance, Ecosim dynamics, basic Ecospace, and IBM. The I/O layer covers ~21 of 84 EwE database tables. Several missing features limit interoperability with published EwE models and standard fisheries science workflows.

---

## Phase 0: Schema Compatibility (prerequisite)

**Plan:** `docs/superpowers/plans/2026-03-11-ewe-schema-fix.md`
**Status:** Plan written, ready to execute

Fix the EwE writer to produce databases that actually load in EwE 6.6+ desktop. Without this, no exported model is usable in native EwE.

**Scope:**
- Fix 15+ table names and 40+ column names
- Replace blank template database
- Update all tests

**Estimated tasks:** 14 + 9 errata items

---

## Phase 1: Time Series & Calibration Pipeline

**Priority:** HIGH — This is the standard EwE calibration workflow used in nearly every published model.

**What's missing:**
- `EcosimTimeSeries`, `EcosimTimeSeriesDataset`, `EcosimTimeSeriesGroup`, `EcosimTimeSeriesFleet` (4 tables)
- Formal observed data loading from EwE databases
- Sum-of-squares (SS) fitting against reference time series (EwE's standard fitting approach)

**What exists:**
- `EcosimOptimizer` with Bayesian GP optimization — this works but uses a different interface than EwE's native time series fitting

**Tasks:**
1. Add time series data structures to `core/params.py` or a new `core/timeseries.py`
2. Implement reader support in `io/ewemdb.py` for the 4 time series tables
3. Implement writer support in `io/_csv_bundle_writer.py`
4. Create SS fitting objective that matches EwE 6 behavior (weighted by data type)
5. Add convenience function: `fit_to_timeseries(scenario, timeseries) -> fitted_scenario`
6. Support data types: biomass (absolute/relative), catch, fishing mortality, effort

**Dependencies:** Phase 0 (schema fix)

---

## Phase 2: Mediation Functions

**Priority:** HIGH — Required by many published EwE models; trophic mediation cannot be represented without this.

**What's missing:**
- `EcosimShapeMediation` table (shape definitions for mediation functions)
- `EcosimScenarioshapeMedWeightsGroup`, `EcosimScenarioshapeMedWeightsFleet`, `EcosimScenarioshapeMedWeightsLandings` (3 weight tables)
- Mediation function evaluation in `ecosim_deriv.py`

**What it does:**
Mediation functions modify predator-prey vulnerability (VV) based on the biomass of a third "mediating" group. Example: coral reef provides hiding places for small fish, so when coral biomass declines, predation on small fish increases. The mediation function maps mediator biomass -> vulnerability multiplier via a user-defined shape.

**Tasks:**
1. Add `MediationFunction` dataclass to `core/forcing.py` or new `core/mediation.py`
2. Read mediation shapes and weights from EwE database
3. Apply mediation multipliers in `deriv_vector()` consumption kernel
4. Write mediation tables back to export
5. Add tests with a simple 3-group mediation scenario

**Dependencies:** Phase 0

---

## Phase 3: Monte Carlo / Pedigree Uncertainty

**Priority:** HIGH — Essential for uncertainty analysis in ecosystem assessments.

**What's missing:**
- `EcopathSample`, `EcopathGroupSample`, `EcopathGroupCatchSample`, `EcopathDietCompSample` (4 tables)
- `Pedigree`, `EcopathGroupPedigree` (2 tables)
- Monte Carlo sampling engine
- Pedigree-based CV assignment

**What exists:**
- `params.pedigree` DataFrame stores pedigree values
- `EcosimOptimizer` has validation metrics

**Tasks:**
1. Implement pedigree table reader/writer (map pedigree levels to CVs)
2. Create `MonteCarlo` class: sample parameters from distributions defined by pedigree CVs
3. Run N Ecopath balance attempts, collect feasible parameterizations
4. Optionally propagate through Ecosim for dynamic uncertainty
5. Summary statistics and visualization (confidence intervals on biomass/catch)
6. Read/write sample tables for reproducibility

**Dependencies:** Phase 0

---

## Phase 4: Ecotracer (Contaminant Tracking)

**Priority:** MEDIUM — Self-contained module, important for contaminant fate studies.

**What's missing:**
- `EcotracerScenario`, `EcotracerScenarioGroup` (2 tables)
- Tracer mass balance equations in Ecosim integration
- Bioaccumulation / biomagnification tracking

**What it does:**
Tracks contaminant concentrations through the food web. Each group has: initial concentration (Czero), environmental input (Cenv), immigration input (Cimmig), decay rate (Cdecay), assimilation proportion, and metabolism rate. The tracer mass balance is solved alongside Ecosim dynamics.

**Tasks:**
1. Add `EcotracerParams` dataclass
2. Implement tracer derivative equations (linear system coupled to Ecosim biomass)
3. Add tracer state to `RsimState` and `RsimOutput`
4. Read/write Ecotracer tables from EwE database
5. Add plotting: contaminant concentration over time by group

**Dependencies:** Phase 0

---

## Phase 5: Fleet Dynamics & MSE

**Priority:** MEDIUM — Important for fisheries management applications.

**What's missing:**
- `EcosimScenarioFleet` (fleet-level effort dynamics: capacity, depreciation, profit-based allocation)
- `EcosimScenarioMSE` (assessment method, power, trials)
- `EcosimScenarioQuota` (TAC allocation by fleet and group)
- Effort dynamics model (investment/disinvestment based on profitability)
- Harvest control rules (HCR)

**What exists:**
- Forced effort time series (`ForcedEffort`)
- Spatial fishing allocation (`spatial/fishing.py`)

**Tasks:**
1. Add `FleetDynamics` to handle effort response to profit
2. Implement effort dynamics in Ecosim integration loop
3. Add quota management: TAC -> fleet allocation -> group-level F
4. Implement basic MSE loop: operating model -> assessment -> HCR -> management action
5. Read/write fleet scenario and MSE tables
6. Add fleet-level scenario configuration to Shiny app

**Dependencies:** Phase 1 (time series), Phase 0

---

## Phase 6: Advanced Ecospace Features

**Priority:** MEDIUM — Required for spatial management scenario evaluation.

**What's missing:**
- `EcospaceScenarioMPA` + `EcospaceScenarioMPAFishery` (MPA management)
- `EcospaceScenarioGroupMigration` (seasonal migration maps)
- `EcospaceScenarioMonth` (monthly wind/advection/upwelling maps)
- `EcospaceScenarioFleet` (spatial fleet effort allocation, port maps, sailing costs)
- `EcospaceScenarioGroupHabitat` (habitat preference matrix)
- `EcospaceScenarioWeightLayer` (capacity driver weight layers)
- `EcospaceScenarioCapacityDrivers` (environmental capacity functions)
- `EcospaceScenarioDataConnection` (external data source connections)

**What exists:**
- Basic grid, dispersal, habitat capacity, spatial fishing
- Environmental drivers with response functions

**Tasks (incremental — pick subsets):**
1. **MPA support:** Define no-take zones, fleet exclusions, evaluate MPA scenarios
2. **Migration maps:** Monthly spatial distribution maps per group
3. **Habitat preference matrix:** Group-habitat suitability scores
4. **Spatial fleet dynamics:** Port-based effort, sailing cost, IFD effort allocation
5. **Read/write all Ecospace tables**

**Dependencies:** Phase 0

---

## Phase 7: System-Level Ecological Indicators

**Priority:** LOW-MEDIUM — Important for ecosystem-based management reporting.

**What's missing:**
- Full Ulanowicz ascendency metrics (development capacity, overhead, redundancy)
- System maturity indices
- Formal sensitivity analysis framework (one-at-a-time, Morris method, Sobol)
- Ecosystem summary indicators (mean trophic level of catch, marine trophic index)

**What exists:**
- Finn Cycling Index
- Connectance, omnivory, keystoneness
- MTI matrix
- Network indices (linkage density, etc.)

**Tasks:**
1. Implement Ulanowicz flow analysis (ascendency, overhead, capacity)
2. Add IndiSeas-style ecosystem indicators
3. Implement sensitivity analysis methods
4. Summary report generation for ecosystem assessments

**Dependencies:** None (standalone)

---

## Phase 8: Value Chain Economics

**Priority:** LOW — Specialized use case, 21 tables, significant implementation effort.

**What's missing:**
- Full supply chain model: producer -> processor -> wholesaler -> retailer -> consumer
- 21 `c`-prefix tables in EwE database
- Economic optimization (maximize value across chain)
- Price elasticity effects on fishing effort

**What exists:**
- Basic fleet cost parameters (FixedCost, SailingCost)

**Tasks:**
This is a major standalone module (~2000+ lines). Defer unless there's specific user demand.

**Dependencies:** Phase 5 (fleet dynamics)

---

## Phase 9: Taxonomy & External Database Integration

**Priority:** LOW — Quality-of-life feature.

**What's missing:**
- `EcopathTaxon`, `EcopathGroupTaxon`, `EcopathStanzaTaxon` (3 tables)
- Formal species-to-group mapping with external database keys (FishBase, WoRMS, OBIS)

**What exists:**
- `io/biodata.py` — OBIS/WoRMS/FishBase integration (already working)
- These tables are essentially metadata; PyPath's biodata module serves the same purpose differently

**Tasks:**
1. Add taxonomy table reader/writer for EwE database exchange
2. Link existing biodata module output to EwE taxonomy format

**Dependencies:** Phase 0

---

## Implementation Priority Matrix

| Phase | Feature | Impact | Effort | Priority |
|-------|---------|--------|--------|----------|
| 0 | Schema fix | Critical | Medium | **Do first** |
| 1 | Time series & calibration | High | Medium | **Do second** |
| 2 | Mediation functions | High | Low-Med | **Do third** |
| 3 | Monte Carlo / pedigree | High | Medium | **Do fourth** |
| 4 | Ecotracer | Medium | Low | Good quick win |
| 5 | Fleet dynamics & MSE | Medium | High | After 1-3 |
| 6 | Advanced Ecospace | Medium | High | Incremental |
| 7 | Ecological indicators | Low-Med | Low | Standalone |
| 8 | Value chain economics | Low | Very High | Defer |
| 9 | Taxonomy integration | Low | Low | Nice to have |

---

## EwE Database Table Coverage After Each Phase

| Phase | Tables covered | Total (of 84) | Coverage |
|-------|---------------|---------------|----------|
| Current | 21 (wrong names) | 84 | 25% |
| Phase 0 | 17 (correct names) | 84 | 20% |
| Phase 1 | 21 | 84 | 25% |
| Phase 2 | 25 | 84 | 30% |
| Phase 3 | 31 | 84 | 37% |
| Phase 4 | 33 | 84 | 39% |
| Phase 5 | 36 | 84 | 43% |
| Phase 6 | 48 | 84 | 57% |
| Phase 7 | 48 | 84 | 57% |
| Phase 8 | 69 | 84 | 82% |
| Phase 9 | 72 | 84 | 86% |

Note: 100% coverage is not a goal. Some tables (Quote, UpdateLog, cOOPStorable, etc.) are EwE desktop UI internals with no scientific value.

---

## Recommended Execution Order

```
Phase 0 (schema fix) ──> Phase 1 (time series) ──> Phase 2 (mediation)
                                                         │
                                                         v
                                               Phase 3 (Monte Carlo)
                                                         │
                                                         v
                                          Phase 4 (ecotracer) ──> Phase 5 (fleet/MSE)
                                                                        │
                                                                        v
Phase 7 (indicators) ───────────────────────────────────> Phase 6 (ecospace)
                                                                        │
                                                                        v
                                                               Phase 8 (economics)
```

Phases 0-3 form the critical path. Phase 7 can be done anytime (no dependencies). Phase 4 is a good standalone addition whenever needed.
