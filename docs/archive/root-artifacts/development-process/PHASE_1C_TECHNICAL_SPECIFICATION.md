# Phase 1C Technical Specifications

**Version**: 1.0  
**Status**: Draft for Review  
**Date**: 2026-10-24  
**Target Release**: v0.6.0

---

## 1. Ecosim Dynamic Extensions (Phase 1C.1)

### 1.1 Ramping Function System

#### API Design
\\\python
from pypath.core.ramping import RampingFunction, LinearRamp, PolynomialRamp

# Base interface
class RampingFunction:
    \"\"\"Abstract base class for parameter ramping.\"\"\"
    
    def evaluate(self, time: float) -> float:
        \"\"\"Get ramp value at time t.\"\"\"
        raise NotImplementedError
    
    def derivative(self, time: float) -> float:
        \"\"\"Rate of change at time t.\"\"\"
        raise NotImplementedError
    
    def get_trajectory(self, times: np.ndarray) -> np.ndarray:
        \"\"\"Get values for multiple times.\"\"\"
        pass

# Linear ramp (existing, to be enhanced)
class LinearRamp(RampingFunction):
    def __init__(self, t0: float, v0: float, t1: float, v1: float):
        self.t0, self.v0 = t0, v0
        self.t1, self.v1 = t1, v1
    
    def evaluate(self, time: float) -> float:
        if time <= self.t0:
            return self.v0
        elif time >= self.t1:
            return self.v1
        else:
            alpha = (time - self.t0) / (self.t1 - self.t0)
            return self.v0 + alpha * (self.v1 - self.v0)

# Polynomial ramp (new)
class PolynomialRamp(RampingFunction):
    def __init__(self, t0: float, v0: float, t1: float, v1: float, 
                 degree: int = 2):
        \"\"\"Polynomial interpolation between two points.
        
        Parameters
        ----------
        t0, v0 : float
            Start time and value
        t1, v1 : float
            End time and value
        degree : int
            Polynomial degree (2=quadratic, 3=cubic, etc.)
        \"\"\"
        self.t0, self.v0 = t0, v0
        self.t1, self.v1 = t1, v1
        self.degree = degree
        self._compute_coefficients()
    
    def _compute_coefficients(self):
        # Hermite polynomial to match positions and velocities
        pass
    
    def evaluate(self, time: float) -> float:
        pass

# Spline ramp (new)
class SplineRamp(RampingFunction):
    \"\"\"Cubic spline interpolation with multiple knots.\"\"\"
    
    def __init__(self, times: np.ndarray, values: np.ndarray):
        self.times = times
        self.values = values
        self._fit_spline()
    
    def _fit_spline(self):
        # Fit cubic spline through knots
        pass
    
    def evaluate(self, time: float) -> float:
        pass

# Stepwise ramp (new)
class StepwiseRamp(RampingFunction):
    \"\"\"Step function with breakpoints.\"\"\"
    
    def __init__(self):
        self.breakpoints = []  # List of (time, value)
    
    def add_breakpoint(self, time: float, value: float):
        self.breakpoints.append((time, value))
        self.breakpoints.sort()
    
    def evaluate(self, time: float) -> float:
        for i, (t, v) in enumerate(self.breakpoints):
            if time < t:
                if i == 0:
                    return self.breakpoints[0][1]
                return self.breakpoints[i-1][1]
        return self.breakpoints[-1][1]
\\\

#### Integration with EcosimScenario

\\\python
class EcosimScenario:
    # New methods for ramping
    
    def set_parameter_ramp(self, group_name: str, parameter: str, 
                           ramp: RampingFunction) -> None:
        \"\"\"Set ramping function for parameter.\"\"\"
        pass
    
    def add_ramping_breakpoint(self, group_name: str, parameter: str,
                               time: float, value: float) -> None:
        \"\"\"Add breakpoint for stepwise/spline ramping.\"\"\"
        pass
    
    def get_ramping_trajectory(self, group_name: str, parameter: str,
                               times: np.ndarray) -> np.ndarray:
        \"\"\"Get parameter values over time.\"\"\"
        pass
\\\

#### Tests
- 25 tests covering:
  - Linear ramp evaluation
  - Polynomial ramp accuracy (degree 2, 3, 4)
  - Spline continuity and smoothness
  - Stepwise exact matching
  - Derivative continuity
  - Boundary conditions
  - Multiple ramping simultaneous

---

### 1.2 Environmental Forcing System

#### API Design

\\\python
from pypath.core.environmental_forcing import (
    EnvironmentalForcing, HabitatProductivity
)

class EnvironmentalForcing:
    \"\"\"Environmental driver for ecosystem dynamics.\"\"\"
    
    def __init__(self):
        self.habitat_productivity = HabitatProductivity()
        self.oceanographic_factors = {}  # name -> TimeSeries
        self.climate_variables = {}  # name -> TimeSeries
    
    def add_oceanographic_factor(self, name: str, time_series: TimeSeries):
        \"\"\"Add oceanographic time series (e.g., temperature, current).\"\"\"
        pass
    
    def add_climate_variable(self, name: str, time_series: TimeSeries):
        \"\"\"Add climate forcing (e.g., NAO, PDO).\"\"\"
        pass
    
    def compute_habitat_suitability(self, group_name: str, 
                                     environmental_state: Dict) -> float:
        \"\"\"0-1 habitat suitability index.\"\"\"
        pass
    
    def scale_growth_by_environment(self, base_growth: float,
                                    environmental_state: Dict) -> float:
        \"\"\"Apply environmental multiplier to growth rate.\"\"\"
        pass

class HabitatProductivity:
    def __init__(self, base: float = 1.0):
        self.base_productivity = base
        self.spatial_variation = None  # np.ndarray if spatial
        self.temporal_forcing = None  # TimeSeries
    
    def set_temporal_pattern(self, time_series: TimeSeries):
        pass
    
    def evaluate(self, time: float, space_index: int = None) -> float:
        \"\"\"Get productivity at time (and space if applicable).\"\"\"
        pass
\\\

#### Integration with EcosimDerivatives

\\\python
class EcosimDerivatives:
    def __init__(self, scenario, environmental_forcing=None):
        self.environmental_forcing = environmental_forcing
    
    def deriv_vector(self, t, y, params):
        # Existing code...
        
        # NEW: Apply environmental forcing
        if self.environmental_forcing:
            env_state = self.environmental_forcing.get_state(t)
            growth_multiplier = self.environmental_forcing.scale_growth(env_state)
            # Apply to growth terms in dydt
        
        return dydt
\\\

#### Tests
- 30 tests covering:
  - Habitat suitability calculation
  - Temporal forcing application
  - Multiple environmental variables
  - Growth scaling
  - State reconstruction

---

### 1.3 Recruitment Dynamics

#### API Design

\\\python
from pypath.core.recruitment import (
    RecruitmentModel, RickerRecruitment, 
    BevertonHoltRecruitment, LarvalConnectivity
)

class RecruitmentModel:
    \"\"\"Abstract recruitment relationship.\"\"\"
    
    def recruitment(self, parent_biomass: float, 
                   environmental_factors: Dict) -> float:
        \"\"\"Predict recruitment given parent biomass and environment.\"\"\"
        raise NotImplementedError

class RickerRecruitment(RecruitmentModel):
    \"\"\"R = a * S * exp(-b * S).\"\"\"
    
    def __init__(self, a: float, b: float):
        self.a = a  # Productivity
        self.b = b  # Density dependence
    
    def recruitment(self, parent_biomass: float, 
                   environmental_factors: Dict) -> float:
        # Ricker curve calculation
        pass

class BevertonHoltRecruitment(RecruitmentModel):
    \"\"\"R = (a*S) / (1 + b*S).\"\"\"
    
    def __init__(self, a: float, b: float):
        pass

class MyersRecruitment(RecruitmentModel):
    \"\"\"R = a*S / (1 + (b*S)^c).\"\"\"
    
    def __init__(self, a: float, b: float, c: float):
        pass

class LarvalConnectivity:
    \"\"\"Larval dispersal and settlement.\"\"\"
    
    def __init__(self, n_patches: int):
        self.dispersal_matrix = np.zeros((n_patches, n_patches))
        self.connectivity_scaling = 1.0
        self.export_production = np.zeros(n_patches)
    
    def set_dispersal_matrix(self, matrix: np.ndarray):
        pass
    
    def compute_recruitment_pool(self, parent_biomass: np.ndarray) -> np.ndarray:
        \"\"\"Larval production and dispersal.\"\"\"
        pass
\\\

#### Tests
- 28 tests covering:
  - Ricker curve properties
  - Beverton-Holt curve properties
  - Myers curve properties
  - Connectivity matrix normalization
  - Larval pool calculation

---

## 2. Ecospace Spatial Extensions (Phase 1C.2)

### 2.1 Habitat Mapping System

#### API Design

\\\python
from pypath.spatial.habitat import (
    HabitatMap, SubstrateType, HabitatQuality
)

class SubstrateType(Enum):
    SAND = 1
    MUD = 2
    ROCK = 3
    GRAVEL = 4
    CORAL = 5
    OTHER = 0

class HabitatMap:
    \"\"\"Multi-layer habitat characteristics.\"\"\"
    
    def __init__(self, nx: int, ny: int):
        self.nx, self.ny = nx, ny
        self.bathymetry = np.zeros((ny, nx))  # Depth in meters
        self.substrate = np.full((ny, nx), SubstrateType.OTHER.value)
        self.habitat_quality = np.ones((ny, nx))
    
    @classmethod
    def from_geotiff(cls, filepath: str) -> 'HabitatMap':
        \"\"\"Load from GeoTIFF file.\"\"\"
        # Use rasterio to load
        pass
    
    @classmethod
    def from_netcdf(cls, filepath: str) -> 'HabitatMap':
        \"\"\"Load from NetCDF file.\"\"\"
        # Use xarray to load
        pass
    
    def get_depth(self, i: int, j: int) -> float:
        return self.bathymetry[j, i]
    
    def get_substrate(self, i: int, j: int) -> SubstrateType:
        return SubstrateType(self.substrate[j, i])
    
    def get_habitat_quality(self, i: int, j: int) -> float:
        return self.habitat_quality[j, i]
    
    def set_quality_gradient(self, direction: str, magnitude: float):
        \"\"\"Apply spatial gradient to habitat quality.\"\"\"
        # direction: 'x', 'y', 'radial', etc.
        pass

class HabitatQuality:
    def __init__(self, base: float = 1.0):
        self.base_quality = base
        self.spatial_gradient = None
        self.temporal_variation = None
    
    def evaluate(self, i: int, j: int, time: float) -> float:
        \"\"\"Get quality at location and time.\"\"\"
        pass
\\\

#### Tests
- 20 tests covering:
  - GeoTIFF loading
  - NetCDF loading
  - Substrate type assignment
  - Quality gradient application
  - Depth interpolation

---

### 2.2 Movement Dynamics

#### API Design

\\\python
from pypath.spatial.movement import (
    MovementModel, BehavioralMigration, 
    OntogeneticMovement, AggregationDynamics
)

class MovementModel:
    \"\"\"Abstract spatial movement behavior.\"\"\"
    
    def compute_movement(self, biomass_distribution: np.ndarray,
                        habitat_map: HabitatMap,
                        time: float) -> np.ndarray:
        \"\"\"Return flow matrix (n_patches, n_patches).\"\"\"
        raise NotImplementedError

class BehavioralMigration(MovementModel):
    \"\"\"Migration triggered by environmental cues.\"\"\"
    
    def __init__(self):
        self.attraction_to_habitat = 0.5
        self.migration_triggers = []
    
    def add_trigger(self, condition, target_region):
        pass

class OntogeneticMovement(MovementModel):
    \"\"\"Size-dependent movement patterns.\"\"\"
    
    def __init__(self, size_breaks: np.ndarray):
        self.size_breaks = size_breaks  # Size class boundaries
        self.movement_by_size = {}  # size_class -> MovementPattern
    
    def set_movement_for_size_class(self, size_class: int, 
                                    movement: MovementModel):
        pass

class AggregationDynamics(MovementModel):
    \"\"\"School/aggregate formation.\"\"\"
    
    def __init__(self):
        self.aggregation_tendency = 0.3
        self.school_size_min = 1000  # kg
    
    def compute_aggregation(self, biomass: np.ndarray) -> np.ndarray:
        \"\"\"Concentration factor due to schooling.\"\"\"
        pass
\\\

#### Tests
- 25 tests covering:
  - Movement matrix stochasticity
  - Trigger conditions
  - Size class separation
  - Aggregation effects
  - Connectivity consistency

---

## 3. Quality Assurance Standards

### 3.1 Testing Requirements

**All Phase 1C modules must achieve:**
- Minimum 80% code coverage
- 100% of public APIs documented
- ≥5 real-world examples per major class
- Integration tests with existing PyPath modules

### 3.2 Documentation Requirements

**All modules must include:**
- NumPy-style docstrings (all public functions/classes)
- Usage examples in docstrings
- Jupyter notebook tutorial (if major feature)
- Integration guide section in phase1c_readme.md

### 3.3 Performance Requirements

**Acceptable performance baselines:**
- Ramping: <1ms to evaluate 10,000 time points
- Environmental forcing: <5ms per time step
- Recruitment: <2ms per group per time step
- Ecospace: <100ms per time step for 100x100 grid

### 3.4 Validation Requirements

**Required validations:**
- Comparison with published EwE models
- Sensitivity analysis for new parameters
- Uncertainty bounds quantification
- Edge case testing

---

## 4. Architecture & Design Patterns

### 4.1 Module Organization

\\\
packages/pypath/src/pypath/
├── core/
│   ├── ramping.py          [NEW - 1C.1a]
│   ├── environmental_forcing.py  [NEW - 1C.1b]
│   ├── recruitment.py       [NEW - 1C.1c]
│   ├── metabolism.py        [NEW - 1C.1d]
│   ├── fishing_management.py [NEW - 1C.1e]
│   └── sensitivity_analysis.py  [NEW - 1C.1f]
├── spatial/
│   ├── habitat.py           [NEW - 1C.2a]
│   ├── movement.py          [NEW - 1C.2b]
│   ├── larval_dispersal.py  [NEW - 1C.2c]
│   ├── forcing.py           [NEW - 1C.2d]
│   ├── fishing.py           [NEW - 1C.2e]
│   ├── mpa.py               [NEW - 1C.2f]
│   └── ecological_interactions.py  [NEW - 1C.2g]
└── analysis/
    └── scenario_optimizer.py [NEW - 1C.3a]
\\\

### 4.2 Design Patterns Used

1. **Abstract Base Classes** - Define interfaces for ramping, movement, recruitment
2. **Factory Pattern** - Create appropriate model instances
3. **Strategy Pattern** - Swap different forcing/movement implementations
4. **Observer Pattern** - Events when parameters change
5. **Data Container Pattern** - @dataclass with validation

### 4.3 Type Hints & Validation

All public functions must have:
- Complete type hints (no Any unless justified)
- Input validation with clear error messages
- Return type annotations

---

## Success Metrics

### Code Quality
- [ ] 4,900 lines of production code
- [ ] 515 tests all passing
- [ ] ≥80% code coverage
- [ ] Zero type checking errors (mypy)
- [ ] black/ruff compliance

### Feature Completeness
- [ ] 21/33 Ecosim features (65%)
- [ ] 20/34 Ecospace features (60%)
- [ ] 55+/84 total parity (65%+)

### Performance
- [ ] No >10% slowdown vs v0.5.3
- [ ] Scales to 200x200 spatial grids
- [ ] Ramping/forcing <5ms overhead per step

---

**Status**: Ready for implementation  
**Next Step**: Create detailed task breakdown for 1C.1a (Ramping)
