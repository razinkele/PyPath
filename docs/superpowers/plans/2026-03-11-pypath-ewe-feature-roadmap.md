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
**Status:** ✅ COMPLETE — I/O gap closed, all 16 Ecospace tables supported, capacity driver weights applied at runtime.

**Implemented:**
- ✅ `EcospaceScenarioMPA` + `EcospaceScenarioMPAFishery` — MPAConfig, MPAZone classes, read_mpa_config(), write_mpa(), full test suite
- ✅ `EcospaceScenarioGroupHabitat` — habitat preference matrix, Gaussian/threshold/linear/step response functions
- ✅ `EcospaceScenarioFleet` — SpatialFishing class with uniform/gravity/port-based/habitat-based allocation
- ✅ `EcospaceScenarioCapacityDrivers` — schema + reader + writer
- ✅ External flux support — ExternalFluxTimeseries for ocean models (ROMS, MITgcm, HYCOM), NetCDF/CSV import
- ✅ Environmental drivers — EnvironmentalLayer, EnvironmentalDrivers, seasonal temperature
- ✅ 7 new tables: GroupMigration, Month, WeightLayer, DataConnection, DataConnectionDisabled, DriverDisabled, HabitatFishery
- ✅ EcospaceReadResult extended with 8 new fields (driver_layers, migration_maps, monthly_maps, etc.)
- ✅ Write support for all 16 Ecospace tables (was 2)
- ✅ MPA write support with mpa_config parameter on write_ewemdb()
- ✅ EcospaceScenarioMPAPatch removed (doesn't exist in real EwE 6.6+)
- ✅ Binary map columns preserved as raw bytes for round-trip fidelity

- ✅ Capacity driver runtime integration — weight layer scalar Weight applied to habitat_capacity at read time

**Note:** Binary LayerMap raster decoding deferred — raw bytes preserved for round-trip, spatial weight application uses scalar Weight field.

**Dependencies:** Phase 0

---

## Phase 7: System-Level Ecological Indicators

**Priority:** LOW-MEDIUM — Important for ecosystem-based management reporting.
**Status:** ✅ COMPLETE — All core features implemented and tested.

**Implemented:**
- ✅ Full Ulanowicz ascendency framework — TST, ascendency, capacity, overhead, relative ascendency (core/indicators.py)
- ✅ Finn Cycling Index — Leontief inverse method
- ✅ Transfer efficiency — per-trophic-level, integer-bin approach
- ✅ Ecosystem summary indicators — MTL catch, Marine Trophic Index, catch/biomass ratio, gross efficiency, Shannon diversity, Kempton's Q
- ✅ Dynamic indicators time series — ecosystem_indicators_timeseries() for Ecosim output
- ✅ Morris sensitivity analysis — OAT screening with mu*, sigma, mu
- ✅ Sobol sensitivity — variance-based via SALib (optional dependency)
- ✅ Network indices — connectance, omnivory, keystoneness, MTI matrix, linkage density
- ✅ System maturity indices — P/R ratio, B/TST ratio, net production, mean path length (SystemMaturityIndices, system_maturity())

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

| Phase | Feature | Impact | Effort | Priority | Status |
|-------|---------|--------|--------|----------|--------|
| 0 | Schema fix | Critical | Medium | **Do first** | ✅ Done |
| 1 | Time series & calibration | High | Medium | **Do second** | ✅ Done |
| 2 | Mediation functions | High | Low-Med | **Do third** | ✅ Done |
| 3 | Monte Carlo / pedigree | High | Medium | **Do fourth** | ✅ Done |
| 4 | Ecotracer | Medium | Low | Good quick win | ✅ Done |
| 5 | Fleet dynamics & MSE | Medium | High | After 1-3 | ✅ Done |
| 6 | Advanced Ecospace | Medium | High | Incremental | ✅ Done |
| 7 | Ecological indicators | Low-Med | Low | Standalone | ✅ Done |
| 8 | Value chain economics | Low | Very High | Defer | Not started |
| 9 | Taxonomy integration | Low | Low | Nice to have | ✅ Done |

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
Phase 0 (schema fix) ──> Phase 1 (time series) ──> Phase 2 (mediation)       ✅ ALL DONE
                                                         │
                                                         v
                                               Phase 3 (Monte Carlo)          ✅ DONE
                                                         │
                                                         v
                                          Phase 4 (ecotracer) ──> Phase 5 (fleet/MSE)  ✅ DONE
                                                                        │
                                                                        v
Phase 7 (indicators) ✅ ────────────────────────────────> Phase 6 (ecospace) ✅
                                                                        │
Phase 9 (taxonomy) ✅                                                   v
                                                               Phase 8 (economics) ❌
```

**Remaining work:**
- Phase 8: Value chain economics (21 tables, major effort — defer unless demand)
