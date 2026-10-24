# Phase 1C: Ecosim & Ecospace Enhancements Roadmap

**Phase**: 1C (Next after Phase 1B.7)  
**Status**: Planning  
**Created**: 2026-10-24  
**Target**: Complete Ecosim and Ecospace feature parity with EwE

## Executive Summary

Phase 1C focuses on dynamic simulation enhancements (Ecosim) and spatial modeling (Ecospace) to improve PyPath's parity from 41.7% (current Phase 1B.7) to 65%+ EwE feature coverage. This phase will address the two largest feature gaps identified in the parity audit.

### Current EwE Parity Status

| Component | Features | Status | Phase Target |
|-----------|----------|--------|--------------|
| **Ecopath** | 14/17 (82%) | ✅ Phase 1B Complete | 82% achieved |
| **Ecosim** | 12/33 (36%) | 🔄 Phase 1C | Target: 65% |
| **Ecospace** | 9/34 (26%) | 🔄 Phase 1C | Target: 60% |
| **Total** | 35/84 (41.7%) | 📈 In Progress | Target: 65% |

---

## Phase 1C.1: Ecosim Dynamic Enhancements

### Objective
Implement core Ecosim dynamic simulation features to reach 65% parity (21/33 features).

### Current Gap Analysis

**What's Working (12/33):**
- ✅ Ramp forcing (linear time series)
- ✅ Seasonal forcing patterns
- ✅ Basic predator-prey dynamics
- ✅ Vulnerability exchange
- ✅ Recruitment forcing
- ✅ Detritus flows
- ✅ Multi-stanza age groups
- ✅ Biomass dynamics
- ✅ Catch dynamics
- ✅ Economic forcing
- ✅ Diet shifts
- ✅ Catch scenarios

**What's Missing (21/33):**
1. 🔴 **Advanced Parameter Ramping** (3 features)
   - Multi-segment polynomial ramping functions
   - Breakpoint management (non-linear transitions)
   - Time-series interpolation methods

2. 🔴 **Environmental Forcing** (4 features)
   - Habitat productivity forcing
   - Oceanographic variable coupling
   - Climate variable integration
   - Spatial heterogeneity impacts

3. 🔴 **Recruitment Dynamics** (3 features)
   - Stock-recruitment relationship optimization
   - Larval connectivity modeling
   - Recruitment success factors

4. 🔴 **Consumption Dynamics** (3 features)
   - Predator aggregation effects
   - Handling time implementation
   - Satiation curves

5. 🔴 **Metabolic Scaling** (2 features)
   - Q10 temperature-dependent metabolism
   - Body size-dependent metabolic rates
   - Activity budget modeling

6. 🔴 **Fishing Management** (3 features)
   - Dynamic gear selectivity patterns
   - Effort allocation optimization
   - Mixed-fishery dynamics

7. 🔴 **Uncertainty & Sensitivity** (3 features)
   - Parameter sensitivity analysis
   - Monte Carlo uncertainty propagation
   - Risk assessment framework

### Phase 1C.1 Deliverables

#### Checkpoint 1C.1a: Parameter Ramping System (Weeks 1-2)
**Goal**: Implement advanced parameter ramping functions

`python
# New Classes
- RampingFunction (abstract base)
  - LinearRamp (current implementation)
  - PolynomialRamp (new - quadratic, cubic)
  - SplineRamp (new - smooth interpolation)
  - StepwiseRamp (new - breakpoint-based)
  
- BreakpointManager
  - add_breakpoint(time, value)
  - validate_continuity()
  - interpolate(time)

# New Methods in EcosimScenario
- set_parameter_ramp(group, parameter, ramp_function)
- add_ramping_breakpoint(group, time, value)
- get_ramping_trajectory(group, times)
`

**Tests**: 25 new tests
**Files**: core/ramping.py (350 lines)

#### Checkpoint 1C.1b: Environmental Forcing (Weeks 3-4)
**Goal**: Integrate environmental variables into dynamics

`python
# New Classes
- EnvironmentalForcing
  - habitat_productivity: float
  - oceanographic_factors: Dict[str, TimeSeries]
  - climate_variables: Dict[str, TimeSeries]
  
- HabitatProductivity
  - base_productivity: float
  - spatial_variation: np.ndarray
  - temporal_forcing: TimeSeries

# Integration with Ecosim
- apply_environmental_forcing()
- compute_habitat_suitability()
- scale_growth_by_environment()
`

**Tests**: 30 new tests
**Files**: core/environmental_forcing.py (420 lines)

#### Checkpoint 1C.1c: Advanced Recruitment (Weeks 5-6)
**Goal**: Implement stock-recruitment relationships

`python
# New Classes
- RecruitmentModel (abstract)
  - RickerRecruitment (new)
  - BevertonHoltRecruitment (new)
  - MyersRecruitment (new)
  - CustomRecruitment (new)

- LarvalConnectivity
  - dispersal_matrix: np.ndarray
  - connectivity_scaling: float
  - export_production: np.ndarray

# New Methods
- set_recruitment_model(group, model)
- compute_recruitment(parent_biomass, environmental_factors)
`

**Tests**: 28 new tests
**Files**: core/recruitment.py (380 lines)

#### Checkpoint 1C.1d: Metabolic Scaling (Weeks 7-8)
**Goal**: Implement temperature and size-dependent metabolism

`python
# New Classes
- MetabolicModel
  - base_metabolism: float
  - q10_coefficient: float
  - mass_exponent: float
  
- TemperatureDependence
  - reference_temperature: float
  - q10_values: Dict[str, float]
  - compute_q10_factor(temp)

# New Methods in EcosimDerivatives
- compute_temperature_adjusted_metabolism()
- apply_size_dependent_rates()
`

**Tests**: 20 new tests
**Files**: core/metabolism.py (280 lines)

#### Checkpoint 1C.1e: Fishing Management (Weeks 9-10)
**Goal**: Advanced multi-fleet and selectivity management

`python
# Enhancements to FleetContainer
- dynamic_selectivity_patterns: Dict[str, TimeSeries]
- effort_allocation_optimization: OptimizationModel
- mixed_fishery_interactions: NetworkModel

# New Classes
- SelectivityPattern (new types)
  - DynamicLogisticSelectivity
  - TimeVaryingSelectivity
  - SpatialSelectivity

- FishingOptimization
  - maximize_revenue()
  - minimize_bycatch()
  - constrain_sustainability()
`

**Tests**: 25 new tests
**Files**: core/fishing_management.py (340 lines)

#### Checkpoint 1C.1f: Sensitivity & Uncertainty (Weeks 11-12)
**Goal**: Parameter sensitivity and uncertainty quantification

`python
# New Module: core/sensitivity_analysis.py
- SensitivityAnalyzer
  - compute_local_sensitivity()
  - compute_global_sensitivity()
  - morris_screening()
  
- UncertaintyPropagation
  - monte_carlo_ensemble(n_samples=1000)
  - latin_hypercube_sampling()
  - quantile_analysis()
  
- RiskAssessment
  - compute_extinction_risk()
  - compute_collapse_risk()
  - worst_case_analysis()
`

**Tests**: 22 new tests
**Files**: core/sensitivity_analysis.py (450 lines)

### Phase 1C.1 Summary

- **Total Lines of Code**: ~2,000 lines
- **Total Tests**: ~150 new tests
- **Files Created**: 6 new modules
- **Estimated Duration**: 12 weeks
- **Target Parity**: 21/33 (65%)

---

## Phase 1C.2: Ecospace Spatial Modeling

### Objective
Implement core Ecospace spatial features to reach 60% parity (20/34 features).

### Current Gap Analysis

**What's Working (9/34):**
- ✅ Basic grid-based habitat
- ✅ Patch connectivity
- ✅ Species dispersal
- ✅ Spatial initial conditions
- ✅ Habitat preferences
- ✅ Fishing zones
- ✅ Marine protected areas (MPAs)
- ✅ Basic movement patterns
- ✅ Spatial catch tracking

**What's Missing (25/34):**
1. 🔴 **Habitat Mapping** (3 features)
   - Bathymetry integration
   - Substrate type mapping
   - Habitat quality gradients

2. 🔴 **Movement Dynamics** (4 features)
   - Behavioral migration patterns
   - Ontogenetic movements
   - Aggregation dynamics
   - Avoidance behaviors

3. 🔴 **Dispersal Mechanisms** (4 features)
   - Larval dispersal modeling
   - Pelagic drift tracking
   - Settlement patterns
   - Recruitment connectivity

4. 🔴 **Spatial Forcing** (3 features)
   - Coastal upwelling impacts
   - Nutrient gradients
   - Temperature gradients
   - Oxygen minimum zones

5. 🔴 **Fishing Effort** (3 features)
   - Spatial fleet deployment
   - Port-based fishing patterns
   - Fishing power distribution
   - Spillover effects

6. 🔴 **MPA Management** (3 features)
   - Dynamic MPA effectiveness
   - Spillover quantification
   - Enforcement effectiveness
   - Reserve network effects

7. 🔴 **Ecological Interactions** (3 features)
   - Spatial predator-prey dynamics
   - Habitat-dependent rates
   - Edge effects
   - Metapopulation dynamics

8. 🔴 **Advanced Outputs** (2 features)
   - Spatial abundance maps
   - Connectivity analysis
   - Biodiversity metrics

### Phase 1C.2 Deliverables

#### Checkpoint 1C.2a: Habitat Mapping (Weeks 1-2)
**Goal**: Integrate realistic bathymetry and habitat quality

`python
# New Classes
- HabitatMap
  - bathymetry: np.ndarray
  - substrate_types: np.ndarray
  - habitat_quality: np.ndarray
  - from_geotiff(filepath)
  - from_netcdf(filepath)

- SubstrateType (enum)
  - SAND, MUD, ROCK, GRAVEL, CORAL
  
- HabitatQuality
  - base_quality: float
  - spatial_gradient: np.ndarray
  - temporal_variation: TimeSeries
`

**Tests**: 20 new tests
**Files**: spatial/habitat.py (300 lines)

#### Checkpoint 1C.2b: Advanced Movement (Weeks 3-4)
**Goal**: Realistic behavioral movement patterns

`python
# New Classes
- MovementModel (abstract)
  - BehavioralMigration (new)
  - OntogeneticMovement (new)
  - AggregationDynamics (new)
  - AvoidanceBehavior (new)

- SpatialBehavior
  - attraction_to_habitat: float
  - repulsion_from_predators: float
  - aggregation_tendency: float
  - migration_triggers: List[Condition]

# Integration
- apply_behavioral_movement()
- compute_preference_distribution()
- track_individual_trajectories()
`

**Tests**: 25 new tests
**Files**: spatial/movement.py (350 lines)

#### Checkpoint 1C.2c: Larval Dispersal (Weeks 5-6)
**Goal**: Comprehensive larval connectivity modeling

`python
# New Classes
- LarvalDispersal
  - pelagic_duration: float
  - larval_mortality: float
  - connectivity_matrix: np.ndarray
  - settlement_suitability: np.ndarray

- DriftModel
  - current_velocity: np.ndarray
  - diffusivity: float
  - compute_drift_trajectory()

- SettlementModel
  - habitat_suitability: np.ndarray
  - density_dependence: float
  - recruitment_success: np.ndarray
`

**Tests**: 28 new tests
**Files**: spatial/larval_dispersal.py (380 lines)

#### Checkpoint 1C.2d: Spatial Forcing (Weeks 7-8)
**Goal**: Environmental gradients and spatial heterogeneity

`python
# New Classes
- CoastalUpwelling
  - upwelling_index: TimeSeries
  - nutrient_transport: np.ndarray
  - productivity_enhancement: float

- TemperatureGradient
  - zonal_gradient: float
  - meridional_gradient: float
  - temporal_variation: TimeSeries
  
- OxygenMinimumZones
  - oxycline_depth: float
  - oxygen_concentration: np.ndarray
  - anaerobic_area: float
`

**Tests**: 22 new tests
**Files**: spatial/forcing.py (320 lines)

#### Checkpoint 1C.2e: Spatial Fishing (Weeks 9-10)
**Goal**: Realistic fleet deployment and fishing patterns

`python
# New Classes
- FleetDeployment
  - fleet_locations: Dict[str, np.ndarray]
  - catchability_distribution: np.ndarray
  - port_based_patterns: Dict[str, Pattern]
  
- FishingEffort
  - spatial_distribution: np.ndarray
  - temporal_dynamics: TimeSeries
  - spillover_effects: np.ndarray

# New Methods
- deploy_fleet_spatially()
- compute_spillover_from_mpas()
- track_fishing_pressure()
`

**Tests**: 24 new tests
**Files**: spatial/fishing.py (310 lines)

#### Checkpoint 1C.2f: MPA Dynamics (Weeks 11-12)
**Goal**: Dynamic marine protected area management

`python
# New Classes
- MarineProtectedArea
  - enforcement_effectiveness: float
  - spillover_quantification: float
  - reserve_network_effects: np.ndarray
  - dynamic_boundary: bool

- ReserveNetwork
  - connectivity_strength: np.ndarray
  - network_efficiency: float
  - reserve_redundancy: float
  
# Analysis
- compute_spillover_benefit()
- quantify_network_effects()
- assess_enforcement_failure()
`

**Tests**: 20 new tests
**Files**: spatial/mpa.py (280 lines)

#### Checkpoint 1C.2g: Ecological Interactions (Weeks 13-14)
**Goal**: Spatial predator-prey and metapopulation dynamics

`python
# New Classes
- SpatialPredatorPrey
  - predator_distribution: np.ndarray
  - prey_vulnerability_map: np.ndarray
  - spatial_coupling: float

- EdgeEffects
  - boundary_condition: str  # 'periodic', 'absorbing', 'reflective'
  - edge_mortality: float
  - edge_recruitment: float

- Metapopulation
  - patch_occupancy: np.ndarray
  - colonization_rate: float
  - extinction_rate: float
  - source_sink_dynamics: np.ndarray
`

**Tests**: 26 new tests
**Files**: spatial/ecological_interactions.py (350 lines)

### Phase 1C.2 Summary

- **Total Lines of Code**: ~2,100 lines
- **Total Tests**: ~165 new tests
- **Files Created**: 7 new modules
- **Estimated Duration**: 14 weeks
- **Target Parity**: 20/34 (60%)

---

## Phase 1C.3: Integration & Optimization

### Objective
Integrate Ecosim and Ecospace, add optimization, and achieve 65%+ total parity.

### Deliverables

#### 1C.3a: Coupled Ecosim-Ecospace Framework (Weeks 1-2)
- Seamless integration between dynamics and spatial modules
- Unified state vector for coupled simulations
- Performance optimization for large spatial domains

#### 1C.3b: Scenario Optimization (Weeks 3-4)
- Multi-objective optimization for management strategies
- Genetic algorithms for fishing policy design
- Risk-aware decision support

#### 1C.3c: Full Test Suite (Weeks 5-6)
- 200+ integration tests
- Validation against EwE databases
- Performance benchmarking

---

## Success Criteria

### Phase 1C.1 (Ecosim)
- [ ] 21/33 features implemented (65% parity)
- [ ] 150+ new tests passing
- [ ] Validation against EwE Ecosim models
- [ ] Performance <10% overhead vs Phase 1B

### Phase 1C.2 (Ecospace)
- [ ] 20/34 features implemented (60% parity)
- [ ] 165+ new tests passing
- [ ] Realistic spatial dynamics
- [ ] Performance scales to 100x100 grids

### Phase 1C.3 (Integration)
- [ ] 35%+ overall parity (35/84 → 55+/84)
- [ ] 200+ integration tests
- [ ] Coupled Ecosim-Ecospace workflows
- [ ] Production-ready release (v0.6.0)

---

## Resource Estimates

| Phase | Code (LOC) | Tests | Duration | Team | Priority |
|-------|-----------|-------|----------|------|----------|
| 1C.1 (Ecosim) | 2,000 | 150 | 12 weeks | 2-3 | HIGH |
| 1C.2 (Ecospace) | 2,100 | 165 | 14 weeks | 2-3 | HIGH |
| 1C.3 (Integration) | 800 | 200 | 6 weeks | 1-2 | MEDIUM |
| **Total** | **4,900** | **515** | **32 weeks** | **2-3** | |

---

## Timeline

`
Q1 2027: Phase 1C.1 (Ecosim) - 12 weeks
  - Weeks 1-12: Core Ecosim features
  - Week 13: Testing & validation
  - Week 14: Buffer/contingency

Q2 2027: Phase 1C.2 (Ecospace) - 14 weeks
  - Weeks 1-14: Core Ecospace features
  - Week 15: Testing & validation
  - Week 16: Buffer/contingency

Q3 2027: Phase 1C.3 (Integration) - 6 weeks
  - Weeks 1-6: Integration & optimization
  - v0.6.0 release
`

---

## Known Constraints

1. **EwE Compatibility**: Must maintain feature parity with EwE export format
2. **Performance**: Spatial simulations must run on standard hardware
3. **Backward Compatibility**: Phase 1C must not break Phase 1A/1B APIs
4. **Testing**: All features must have ≥80% test coverage
5. **Documentation**: NumPy-style docstrings on all public APIs

---

## Next Steps

1. ✅ Review Phase 1C roadmap with team
2. ⏳ Obtain stakeholder approval for timeline
3. ⏳ Assign feature ownership
4. ⏳ Create detailed task breakdown for 1C.1a
5. ⏳ Schedule Phase 1C.1a kickoff

---

**Document Created**: 2026-10-24  
**Status**: Ready for Review  
**Target Release**: v0.6.0 (Q3 2027)
