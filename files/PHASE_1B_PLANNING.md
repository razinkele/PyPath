# Phase 1B Planning: Ecopath Enhancement (Nov 2026 – Mar 2027)

## 1. Executive Summary

**Timeline**: November 2026 – March 2027 (4 months, 18 weeks)
**Scope**: 5 major modules, 20+ subtasks, comprehensive Ecopath enhancement
**Target Parity**: 38% (v0.5.2) → 48% (v0.5.3)
**Resource Estimate**: 80-100 dev-days, ~950 lines of code
**Blockers**: None identified; Phase 1A foundation complete and verified

### Context
Phase 1B follows the successful completion of Phase 1A (Ecopath I/O foundation: read_ewe_samples(), write_ewe_samples(), 5-fleet Curonian model parity verification, 1446 passing tests). Phase 1B focuses on **enhanced Ecopath modeling capabilities** by implementing 5 missing critical table systems (Pedigree, Taxonomy, Size Distribution, Fleet/Discard, Nutrients) that are essential for uncertainty quantification, advanced multi-stanza group modeling, and realistic fleet allocation and discard routing workflows.

### Strategic Importance
- **Parity**: Bridges gap from 38% to 54% EwE feature parity
- **Reproducibility**: WoRMS integration ensures species identification traceability
- **Uncertainty**: Pedigree system enables sensitivity analysis and parameter confidence bounds
- **Realism**: Fleet/discard routing enables ecosystem-based fisheries management scenarios
- **Extensibility**: Foundation for Phase 1B.2 (Ecosim forcing) and Phase 2+ (uncertainty propagation)

---

## 2. Parity Analysis

### Current State (v0.5.2)
| System | Tables Implemented | Total Tables | Coverage | Status |
|--------|-------------------|--------------|----------|--------|
| **Ecopath** | 10 | 17 | 59% | Mature (input/output basic) |
| **Ecosim** | 5 | 27 | 19% | Minimal (derivatives only) |
| **Ecospace** | 16 | 16 | 100% | Complete |
| **Overall** | 32 | 84 | 38% | Phase 1A target: ACHIEVED |

### Missing Ecopath Tables (7) – Phase 1B Priority Order

| Priority | Table | Category | Complexity | Est. LOC | Rationale |
|----------|-------|----------|-----------|----------|-----------|
| 1 | **Pedigree** | Metadata | LOW | 120 | Enables uncertainty quantification; foundation for sensitivity analysis |
| 2 | **Taxonomic** | Metadata | MED | 180 | Species traceability; integrates WoRMS for scientific rigor |
| 3 | **Size Distribution** | Biology | MED | 280 | Multi-stanza calibration; enables growth model for Ecosim Phase 2 |
| 4 | **Discard Pathways** | Fleet | HIGH | 370 | Complete fisheries realism; enables bycatch/discard scenarios |
| 5 | **Landings** | Fleet | LOW | 100 | Enhanced catch data; temporal/type distribution |
| 6 | **Nutrients** | Chemistry | LOW | 100 | N/P cycling; stoichiometric consistency checks |
| — | Age-Length Keys | Biology | MED | — | Merged into Size Distribution (TG3.3) |

**Post-Phase 1B Target** (v0.5.3):
- Ecopath: 14-15/17 tables (~82%)
- Ecosim: 5/27 (unchanged; Phase 2+ scope)
- Ecospace: 16/16 (unchanged)
- **Overall: 45-48/84 tables (54%+)**

### Implementation Strategy
Phase 1B prioritizes **highest-impact** tables first:
1. **Pedigree + Taxonomy** (Weeks 1-3): Quick wins; enable reproducibility + uncertainty framework
2. **Size Distribution** (Weeks 3-4): Prerequisite for multi-stanza tests and Ecosim Phase 2
3. **Fleet + Discard** (Weeks 4-6): Most complex; enables realistic fisheries scenarios
4. **Nutrients + Landings** (Weeks 5-6): Finishing work; lower complexity

---

## 3. Task Groups & Detailed Breakdown

### TG3.1: Pedigree System (Weeks 1-2, 15-20 dev-days, ~120 LOC)

**Goal**: Implement parameter uncertainty quantification framework for sensitivity analysis, confidence bounds, and robust parameter estimation

**Design Rationale**: 
- Pedigree index (0-4) encodes data quality confidence
- Quality flags (low/med/high) enable grouping for ensemble methods
- Bidirectional EwE sync ensures roundtrip fidelity

**Tasks**:

1. **TG3.1.1 Pedigree Model** (3 days, 80 LOC)
   - Define `PedigreeConfig` dataclass (pydantic v2-compatible)
   - Fields: `pedigree_index` (int 0-4), `quality_flag` (enum: low/med/high), `comment` (str)
   - Reference ranges per group type & parameter (e.g., biomass pedigree typically ≥ diet composition)
   - Serialize to/from DataFrame with backward compatibility (v0.5.2 can import v0.5.3 models)
   - Location: `packages/pypath/src/pypath/core/pedigree.py`
   - Validation: pedigree_index must align with quality_flag (0-1: high, 2: med, 3-4: low)

2. **TG3.1.2 Pedigree Validation** (2 days, 40 LOC)
   - Range validation: pedigree_index ∈ [0,4]
   - Quality tier consistency: enforce monotonic confidence (higher index = lower confidence)
   - Flag missing pedigree for required groups (biomass, diet, catch)
   - Cross-check: diet_pedigree ≤ biomass_pedigree (diet data typically less precise)
   - Unit tests: 8-10 test cases covering edge cases, conflicts, boundary values

3. **TG3.1.3 EwE Database Read/Write** (2 days, ~40 LOC)
   - Extend `ewemdb.py`: `read_ewe_pedigree(dbpath: str) → pd.DataFrame`
   - Extend `ewemdb.py`: `write_ewe_pedigree(dbpath: str, pedigree_df: pd.DataFrame) → None`
   - Parse EwE Pedigree table schema (AphiaID-based or group name-based lookup)
   - Map EwE quality flags (0-high, 1-med, 2-low in EwE) → PyPath enums
   - Integration test vs EwE 6.0 reference database (Curonian or test fixture)
   - Ensure bidirectional fidelity: (EwE → PyPath → EwE) preserves values within tolerance

**Deliverable**: `pedigree.py` (120 LOC) + 15+ tests, full NumPy docstrings, zero lint errors

**Dependencies**: Phase 1A ewemdb.py, existing test fixtures (Curonian Lagoon model)

---

### TG3.2: Taxonomic Linkage (Weeks 2-3, 15-20 dev-days, ~180 LOC)

**Goal**: Link model groups to WoRMS/GBIF taxonomy for reproducibility, traceability, and scientific rigor in species identification

**Design Rationale**:
- WoRMS AphiaID is the universal taxonomic identifier for marine/aquatic species
- Darwin Core standardization enables interoperability with biodiversity data portals
- Caching layer (SQLite) minimizes API calls and enables offline operation

**Tasks**:

1. **TG3.2.1 Taxonomy Dataclass** (3 days, 90 LOC)
   - `TaxonomyData` class: group_id, common_name, scientific_name, aphia_id, kingdom, phylum, class, order, family, genus, species
   - Optional fields: aphia_id_alt (synonyms list), rank_confidence (0-1), last_checked (ISO date)
   - Darwin Core mapping (dwc:scientificName, dwc:kingdom, dwc:acceptedNameUsageID for AphiaID)
   - Serialize to Excel/CSV with proper Darwin Core headers
   - Backward compat: all fields optional (default to null); models can exist without taxonomy
   - Search/lookup methods: by AphiaID, by scientific name, by common name

2. **TG3.2.2 WoRMS Integration** (4 days, 90 LOC)
   - `biodata.py` enhancement: `query_worms_taxonomy(names: List[str], use_cache=True) → List[TaxonomyData]`
   - Batch API calls to WoRMS (1000 names per request to minimize latency; WoRMS API supports batching)
   - SQLite caching layer: `~/.pypath/cache/worms.db` (date-stamped entries, invalidate every 6 months)
   - Error handling: ambiguous names (≥2 matches), synonyms, extinct taxa, misspellings, near-matches
   - Rate limiting: 1 request/sec; exponential backoff retry (3 attempts, 2^n seconds)
   - Confidence scoring: exact match (1.0) > fuzzy match (0.7-0.9) > no match (0.0)
   - Test fixture: 50 common Baltic species (sprat, herring, cod, flounder, etc.)
   - Fallback: return cached data if API unavailable; warn user

3. **TG3.2.3 Validation & Reconciliation** (2 days, 30 LOC)
   - Cross-validate taxonomic hierarchy consistency (kingdom → phylum → class → order → family → genus → species)
   - Detect conflicts: common name vs binomial (e.g., "Sprat" should map to *Sprattus sprattus*)
   - Consistency checks: AphiaID uniqueness across model, missing parent taxa, extinct/invalid taxa
   - Report ambiguities: species with multiple AphiaIDs, synonyms that need disambiguation
   - Unit tests: 10+ cases covering ambiguous names, misspellings, extinct species, synonym resolution

**Deliverable**: `biodata.py` enhancements (180 LOC) + 12+ tests, caching infrastructure operational, offline fallback working

**Dependencies**: Phase 1A biodata.py (OBIS integration), WoRMS REST API (stable, public)

**Risks**: WoRMS API downtime (mitigated by caching), rate-limit issues (mitigated by batching + backoff)

---

### TG3.3: Size Distribution & Length-At-Age (Weeks 3-4, 20-25 dev-days, ~280 LOC)

**Goal**: Model body size variability for multi-stanza groups; enable von Bertalanffy growth integration for Ecosim Phase 2B+

**Design Rationale**:
- Age-length keys essential for calibrating multi-stanza lifespan distribution
- Length-weight allometry critical for biomass↔abundance conversions in stanzas
- Growth model enables dynamic ontogenetic niche shifts in Ecosim

**Tasks**:

1. **TG3.3.1 Size Distribution Model** (4 days, 120 LOC)
   - `SizeDistribution` dataclass: distribution_type (enum: normal, lognormal, gamma), params dict (μ, σ, shape, scale)
   - Per-group body size reference: mean_length_cm, length_cv (coefficient of variation, 0-1)
   - Length-Weight relationship: allometric formula (W = a × L^b), store (a, b, fit_r², n_obs)
   - `AgeLength` dataclass: age_months, mean_length_cm, length_sd_cm, n_samples, maturity_stage
   - Array storage: per-stanza age-length keys (12-36 rows typical for 12-36 month cohorts)
   - Growth model: von Bertalanffy parameters (L∞, k, t₀) for Ecosim integration
   - Serialization: DataFrame-backed for compatibility with existing PyPath patterns

2. **TG3.3.2 Stanza Integration** (3 days, 80 LOC)
   - Extend `RpathStanzaParams`: add `size_distribution: SizeDistribution`, `age_length_keys: List[AgeLength]`
   - Map EwE lifespan_length → age-length keys (uniform distribution if not provided)
   - Growth model: implement von Bertalanffy solver (L∞, k, t₀) for Ecosim phase 2
   - Growth rate validation: positive growth (dl/dt > 0), asymptotic approach to L∞
   - Backward compatibility: optional fields, default to uniform distribution if absent
   - Ensure existing Ecosim/Ecospace code works with/without size data

3. **TG3.3.3 EwE Database I/O** (4 days, 80 LOC)
   - `read_ewe_age_length_keys(dbpath: str, group_id: int) → List[AgeLength]`
   - `read_ewe_size_distribution(dbpath: str, group_id: int) → SizeDistribution`
   - `write_ewe_age_length_keys(dbpath: str, group_id: int, keys: List[AgeLength]) → None`
   - `write_ewe_size_distribution(dbpath: str, group_id: int, dist: SizeDistribution) → None`
   - Test data: Baltic sprat/herring stanzas (realistic 12-36 month age-length relationships)
   - Validation: length ranges biologically plausible (newborn < adult), monotonic growth, fit quality
   - Schema mapping: handle EwE v6.0 and v6.1 variations

**Deliverable**: `size_distribution.py` (280 LOC) + 10+ tests, full stanza integration, EwE I/O roundtrip validated

**Dependencies**: Phase 1A stanzas.py, RpathParams, test data (Baltic herring/sprat)

---

### TG3.4: Fleet Hierarchy & Discard Routing (Weeks 4-6, 30-35 dev-days, ~370 LOC)

**Goal**: Implement complete fleet/gear/group allocation system for realistic discard pathways and ecosystem-based fisheries management scenarios

**Design Rationale**:
- Fleet hierarchy captures fishing effort diversity (passive/active gears, by-country)
- Allocation matrix enables realistic catch composition per fleet/gear
- Discard routing models ecosystem-scale impacts (mortality, nutrient cycling)

**Tasks**:

1. **TG3.4.1 Fleet & Gear Hierarchy** (5 days, 120 LOC)
   - `FleetGroup` dataclass: fleet_id, name, country (ISO 3166), engine_type (enum: passive/active/none), fuel_type, crew_size, base_port
   - `GearType` dataclass: gear_id, name, selectivity_profile (enum: non-selective/size-selective/species-selective), mesh_size_mm, bycatch_ratio (0-1)
   - `Allocation` dataclass: fleet_id, gear_id, group_id, catch_tonnes, effort_hours, value_usd
   - Serialize to multiple tables: Fleet, Gear, FleetGear (cross-reference), CatchAllocation
   - Validation: allocation > 0 for non-zero catch; total allocation ≤ available biomass
   - Effort normalization: effort_hours as fraction of annual fishing days (250-300 typical)

2. **TG3.4.2 Discard Routing** (5 days, 100 LOC)
   - `DiscardPath` dataclass: fleet_id, gear_id, source_group_id, destination_group_id, destination_type (enum: detritus/output/cannibalism)
   - Per-path: discard_fraction (0-1), min_size_cm, max_size_cm, survival_fraction (0-1 for partial mortality)
   - Routing logic: allocation × discard_fraction × survival_fraction → destination biomass
   - Support chaining: fleet catches group A → 30% discarded (dead) → detritus; 20% discarded (live) → cannibalism; 50% → landings
   - Enable discard mortality as fleet bycatch input for Ecosim Phase 2B+ (bycatch scenarios)
   - Validation: discard_fraction + landings_fraction ≤ 1.0, destinations exist in model

3. **TG3.4.3 Allocation Validation & Balance** (4 days, 80 LOC)
   - Cross-validate: sum(allocation per group) ≤ Landings table total (within 5% tolerance)
   - Discard routing: destination groups must exist and be valid (no circular routing)
   - Effort consistency: effort × selectivity profile ≤ total fleet effort capacity
   - Zero-value handling: skip missing gears gracefully, warn user on inconsistencies
   - Unit tests: 12+ cases (missing gears, zero allocation, conflicts, routing cycles, edge cases)
   - Mass balance check: total catch = landings + discards (dead) + live releases (all must sum to catch)

4. **TG3.4.4 EwE Database I/O** (5 days, 70 LOC)
   - `read_ewe_fleets(dbpath: str) → Dict[int, FleetGroup]`
   - `read_ewe_gears(dbpath: str) → Dict[int, GearType]`
   - `read_ewe_allocation(dbpath: str) → pd.DataFrame` (fleet × gear × group matrix)
   - `read_ewe_discard_paths(dbpath: str) → List[DiscardPath]`
   - `write_ewe_fleets(dbpath: str, fleets: Dict[int, FleetGroup]) → None`
   - `write_ewe_allocation(dbpath: str, allocation: pd.DataFrame) → None`
   - Integration test: 5-fleet Curonian Lagoon model (Baltic seines, trawls, gillnets, dredges)
   - Validation vs EwE 6.0 allocation totals (row/column sums match)
   - Schema mapping: handle EwE v6.0/v6.1 fleet table variations

**Deliverable**: `fleet.py` (370 LOC) + 20+ tests, Curonian reference validation passing, EwE I/O roundtrip verified

**Dependencies**: Phase 1A params.py (RpathParams), Curonian Lagoon model, EwE 6.0 schema

**Complexity**: Highest in Phase 1B; requires careful validation to prevent mass-balance violations

---

### TG3.5: Nutrient System (Weeks 5-6, 10-15 dev-days, ~100 LOC)

**Goal**: Implement N/P cycling parameters for stoichiometric mass-balance checks and nutrient limitation modeling

**Design Rationale**:
- N/P excretion enables nutrient feedback loops in Ecosim dynamics
- Uptake rates quantify phytoplankton nutrient demand
- Consistency checks prevent unrealistic nutrient fluxes

**Tasks**:

1. **TG3.5.1 Nutrient Dataclass** (3 days, 50 LOC)
   - `NutrientExcretion`: group_id, N_fraction (%, 0-50), P_fraction (%, 0-10), excretion_rate_per_biomass_day (0.001-0.1)
   - `NutrientUptake`: phytoplankton_group_id, uptake_rate_N_per_biomass_day (mg/g/day), uptake_rate_P_per_biomass_day
   - Optional fields: regenerated_fraction (0-1, regenerated vs excreted), excretion_type (enum: fecal, urinary, both)
   - Backward compatibility: default to zero (no nutrient cycling) if omitted
   - Validation: N_fraction + P_fraction ≤ 100%, rates non-negative

2. **TG3.5.2 EwE Database Read** (2 days, 30 LOC)
   - `read_ewe_nutrients(dbpath: str) → Dict[str, Union[NutrientExcretion, NutrientUptake]]`
   - Schema mapping: parse EwE Nutrients table (v6.0+ only; return empty dict for v5.x)
   - Enum mapping: excretion_type strings → PyPath enums
   - Test fixture: Curonian model (5-10 groups, minimal nutrient data or synthetic)

3. **TG3.5.3 Validation & Diagnostics** (1 week, 20 LOC, defer if necessary for Phase 1B.1)
   - Nutrient mass balance: excretion + uptake consistency check
   - Visualization placeholder: N/P flows across trophic levels (Phase 2+ feature)
   - Stoichiometric check: N:P ratios within redfield (~16:1) ± 50%

**Deliverable**: `nutrients.py` (100 LOC) + 8+ tests, basic validation functional

**Dependencies**: Phase 1A params.py, test fixtures

---

### TG3.6: Enhanced Landings & Catch Data (Weeks 4-5, 10-15 dev-days, ~100 LOC)

**Goal**: Upgrade Landings table with catch type and temporal distribution for realistic fishing scenarios

**Tasks**:

1. **TG3.6.1 Catch Type Extension** (2 days, 40 LOC)
   - `CatchRecord`: fleet_id, gear_id, group_id, catch_type (enum: commercial, bycatch, discard), tonnes, month (1-12)
   - Size range metadata: min_size_cm, max_size_cm (enables size-selective scenarios)
   - Value tracking: value_usd per tonne (enables economic analysis)
   - Aggregate Landings from CatchRecords: sum(tonnes) by group = Landings[group]

2. **TG3.6.2 Temporal Distribution** (2 days, 40 LOC)
   - Seasonal allocation: catch_by_month (12-element array per fleet/group)
   - Integrate with Forcing (Phase 1B.2+): fishing effort time series
   - Monthly distribution must sum to annual Landings

3. **TG3.6.3 Validation** (1 day, 20 LOC)
   - Cross-checks: sum(catch_records) = Landings total (tolerance ±1%)
   - Catch type consistency: bycatch ≤ total allocation for gear
   - Temporal: sum(catch_by_month) = annual Landings

**Deliverable**: Landings extension in `params.py` (100 LOC) + 10+ tests

---

## 4. Implementation Roadmap

### Week 1 (Nov 3-9): Pedigree System Foundation
- **Mon**: Design PedigreeConfig class; review EwE 6.0 schema
- **Tue**: Implement `pedigree.py` module + 4 unit tests
- **Wed**: Extend `ewemdb.py` with `read_ewe_pedigree()`
- **Thu-Fri**: Integration tests; validation edge cases; ruff/mypy pass

**Deliverable**: `pedigree.py` (120 LOC), 15+ tests, 0 lint errors

### Week 2 (Nov 10-16): Taxonomic Linkage
- **Mon**: Design TaxonomyData; review WoRMS REST API
- **Tue-Wed**: Implement `biodata.py` `query_worms_taxonomy()` + caching
- **Thu**: Rate limiting + error handling
- **Fri**: Integration test (50 Baltic species); edge case coverage

**Deliverable**: `biodata.py` enhancements (180 LOC), 12+ tests, SQLite cache operational

### Week 3 (Nov 17-23): Size Distribution (Part 1)
- **Mon**: Design SizeDistribution, AgeLength dataclasses
- **Tue-Wed**: Implement `size_distribution.py` core module
- **Thu**: Length-weight integration
- **Fri**: Unit tests (8+ cases); growth model skeleton

**Deliverable**: `size_distribution.py` part 1 (120 LOC), 10+ tests

### Week 4 (Nov 24-30): Size Distribution (Part 2) + Fleet Foundation
- **Mon-Tue**: Stanza integration; age-length keys mapping; EwE I/O
- **Wed**: Size distribution complete; full test coverage
- **Thu**: Design FleetGroup, GearType, Allocation dataclasses
- **Fri**: Implement `fleet.py` fleet hierarchy module

**Deliverable**: `size_distribution.py` complete (280 LOC), `fleet.py` part 1 (120 LOC), 20+ tests

### Week 5 (Dec 1-7): Discard Routing & Allocation
- **Mon-Tue**: Implement DiscardPath routing logic
- **Wed**: Allocation validation + balance checks
- **Thu-Fri**: EwE I/O for fleet/gear/allocation; Curonian Lagoon test

**Deliverable**: `fleet.py` complete (370 LOC), 20+ tests, Curonian validation

### Week 6 (Dec 8-14): Nutrients & Landings
- **Mon-Tue**: Implement `nutrients.py` + EwE read
- **Wed**: Landings enhancement in `params.py`
- **Thu-Fri**: Validation tests; buffer for blockers

**Deliverable**: `nutrients.py` (100 LOC), Landings (100 LOC), 18+ tests

### Week 7-8 (Dec 15-29): Documentation & Integration
- Complete NumPy-style docstrings (100% public functions)
- Usage guide: "Advanced Ecopath Modeling" (2,000+ words)
- Jupyter notebook: "Phase 1B Tutorial" (15+ cells)
- Full integration tests (10+ multi-module scenarios)
- Version bump: 0.5.2 → 0.5.3
- CHANGELOG.md, Release notes

**Deliverable**: v0.5.3 release-ready, 1500+ tests passing, zero lint errors

---

## 5. Success Criteria

### Code Quality
- All new code passes `ruff check` (E, F, I, W; 88-char line length)
- `mypy packages/pypath/src --config-file packages/pypath/pyproject.toml` → 0 errors
- NumPy-style docstrings: 100% of public functions/classes
- Test coverage: ≥85% for all new modules
- No print() calls; all logging via `logger.debug/info/warning`
- No bare `except Exception: pass`

### Functional Requirements
- Pedigree: Read/write EwE database, validate ranges, quality consistency
- Taxonomy: Query WoRMS (100+ names), caching (6-month TTL), edge cases handled
- Size: Age-length keys for multi-stanza, von Bertalanffy implemented, backward compatible
- Fleet: Curonian 5-fleet reads correctly, discard routing validates, mass-balance checks
- Nutrients: Read/write N/P parameters, consistency checks passing
- Landings: Catch records aggregate correctly, temporal distribution valid

### Test Coverage
- Baseline: 1446 → 1500+ tests passing
- Zero regressions in Phase 1A modules
- New modules: pedigree (15) + taxonomy (12) + size (10) + fleet (20) + nutrients (8) + landings (10) = 75+ tests

### Parity Metrics
- Ecopath: 10 → 14-15/17 tables (82%)
- Overall: 38% → 54%+

### Documentation
- 2,000+ word usage guide
- Jupyter notebook (15+ cells, runnable)
- CHANGELOG.md (500+ words)
- API docs auto-generated (NumPy style)

---

## 6. Risk Mitigation & Contingency

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|-----------|
| WoRMS API downtime/rate-limits | Medium | High | SQLite caching (6-month TTL), batch queries, offline fallback |
| Complex stanza age-length mapping | Medium | High | Prototype Week 3; test fixtures ready; consult Phase 1A stanza code |
| EwE v6.1 schema incompatibility | Low | High | Test v6.0 + v6.1 alpha if available; version schema mappings in code |
| Test data unavailable | Low | Medium | Use synthetic Curonian from Phase 1A (confirmed available) |
| Scope creep (Ecosim integration) | Medium | Medium | Hard stop: Ecopath only; defer Ecosim to Phase 1B.2 (2027 Q1) |
| API integration test failures | Medium | Medium | Implement mock/fixture fallback; extensive unit tests |

**Contingency**: +2 weeks (Dec 15-29) allocated for edge cases, rework, community feedback

---

## 7. Dependencies & Assumptions

### External Dependencies (Pre-existing)
- WoRMS REST API (via biodata.py, already in Phase 1A)
- EwE 6.0 reference database (test data only)
- numpy, pandas, scipy, matplotlib (v0.5.2 stable)

### Assumptions (Validated)
1. Phase 1A complete, 1446 tests passing ✓
2. EwE v6.0 schema stable, v6.1 backward compatible
3. Curonian model available (example_model_data/) ✓
4. No breaking numpy/pandas/scipy changes (Python 3.10-3.12)
5. WoRMS API stable (rate-limit: 1 req/sec)

### Out of Scope
- Ecosim forcing (Phase 1B.2, Q1 2027)
- Ecospace nutrient diffusion (Phase 2A, Q2 2027)
- Ensemble propagation (Phase 2B, Q3 2027)

---

## 8. Acceptance Checklist

**Code**: All modules compile | `ruff check` → 0 errors | `mypy` → 0 errors | ≥85% test coverage | 1516+ tests passing  
**Functionality**: Pedigree read/write | WoRMS queries cached | Size age-length mapped | Fleet/discard validated | Nutrients consistent  
**Documentation**: 2,000+ word guide | Runnable Jupyter notebook | CHANGELOG.md v0.5.3 | API docs auto-generated  
**Release**: Version 0.5.3 | PyPI built & tested | conda recipe updated | GitHub release notes published

---

**Document Created**: 2026-10-24  
**Status**: Ready for Phase 1B execution (Nov 2026)  
**Next Review**: 2026-11-03 (Week 1 kickoff)
