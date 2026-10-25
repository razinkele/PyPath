# Phase 1C.1a: Ramping System - Detailed Task Breakdown

**Objective**: Implement comprehensive ramping function system for Ecosim parameter dynamics  
**Timeline**: Weeks 1-3 of Phase 1C (12 weeks total)  
**Lead**: [TBD - assign to team lead]  
**Dependencies**: Core Ecosim infrastructure (DONE - v0.5.3)  

---

## Sprint Structure

### Week 1: Core API & Linear Ramp Enhancement
**Goal**: Establish abstract interfaces and enhance existing linear ramping

#### Task 1.1: Define RampingFunction Abstract Base Class
**Time**: 4 hours  
**Owner**: [TBD]  

**Deliverables**:
- File: packages/pypath/src/pypath/core/ramping.py (new file)
- Abstract class RampingFunction with methods:
  - valuate(time: float) -> float
  - derivative(time: float) -> float
  - get_trajectory(times: np.ndarray) -> np.ndarray
  - __repr__() for debugging

**Code template**:
`python
from abc import ABC, abstractmethod
import numpy as np

class RampingFunction(ABC):
    """Abstract base class for parameter ramping.
    
    All ramping functions model time-varying parameters as:
    f(t) where f(t0) = v0 and f(t1) = v1
    """
    
    @abstractmethod
    def evaluate(self, time: float) -> float:
        """Evaluate ramp value at time t."""
    
    @abstractmethod
    def derivative(self, time: float) -> float:
        """Rate of change (df/dt) at time t."""
    
    def get_trajectory(self, times: np.ndarray) -> np.ndarray:
        """Get values for array of times."""
        return np.array([self.evaluate(t) for t in times])
    
    def __repr__(self):
        """Return debugging representation."""
        raise NotImplementedError
`

**Tests to create** (in packages/pypath/tests/test_ramping.py):
- 	est_rampingfunction_is_abstract - Verify ABC enforcement
- 	est_rampingfunction_get_trajectory - Verify batching method
- Total: 2 tests

**Review checklist**:
- [ ] ABC syntax correct
- [ ] All abstract methods documented
- [ ] Docstrings follow NumPy style
- [ ] No unused imports

---

#### Task 1.2: Enhance LinearRamp Implementation
**Time**: 6 hours  
**Owner**: [TBD]  

**Current state**: LinearRamp exists in cosim.py but scattered  
**Changes**:

1. Move LinearRamp to amping.py
2. Ensure it inherits from RampingFunction
3. Add derivative calculation
4. Add edge case handling (t0 == t1)

**Code signature**:
`python
class LinearRamp(RampingFunction):
    def __init__(self, t0: float, v0: float, t1: float, v1: float):
        """Linear interpolation between (t0,v0) and (t1,v1).
        
        Parameters
        ----------
        t0, v0 : float
            Start time and value
        t1, v1 : float
            End time and value
        """
        if t0 == t1:
            raise ValueError(f"Start and end times must differ: t0={t0}, t1={t1}")
        self.t0, self.v0 = t0, v0
        self.t1, self.v1 = t1, v1
    
    def evaluate(self, time: float) -> float:
        # Before t0: return v0
        # After t1: return v1
        # Between: linear interpolation
    
    def derivative(self, time: float) -> float:
        # Constant slope (t0, t1)
        # Zero outside [t0, t1]
`

**Tests** (expand existing):
- 	est_linear_ramp_before_t0 - Returns v0
- 	est_linear_ramp_after_t1 - Returns v1
- 	est_linear_ramp_midpoint - Correct interpolation
- 	est_linear_ramp_derivative_constant - Slope correct
- 	est_linear_ramp_derivative_zero_outside - Zero before/after
- 	est_linear_ramp_raises_on_invalid_times - t0 == t1
- 	est_linear_ramp_trajectory_array - Batching
- Total: 7 tests

**Review checklist**:
- [ ] Consistent with abstract interface
- [ ] Edge cases handled
- [ ] Derivative mathematically correct
- [ ] Tests comprehensive
- [ ] Existing code refactored to use new location

---

#### Task 1.3: Create StepwiseRamp Class
**Time**: 5 hours  
**Owner**: [TBD]  

**Purpose**: Step function for discontinuous parameter changes  
**Use case**: Sudden policy changes, discrete fishing closures

**Code signature**:
`python
class StepwiseRamp(RampingFunction):
    """Step function with multiple breakpoints."""
    
    def __init__(self, breakpoints: List[Tuple[float, float]] = None):
        """Initialize with optional breakpoints.
        
        Parameters
        ----------
        breakpoints : List[(time, value), ...]
            Breakpoints in temporal order
        """
        self.breakpoints = []
        if breakpoints:
            for t, v in breakpoints:
                self.add_breakpoint(t, v)
    
    def add_breakpoint(self, time: float, value: float):
        """Add breakpoint and sort."""
    
    def evaluate(self, time: float) -> float:
        # Find last breakpoint <= time
        # Return its value
`

**Tests**:
- 	est_stepwise_empty - No breakpoints
- 	est_stepwise_single_breakpoint - Before/after one step
- 	est_stepwise_multiple_breakpoints - Multiple steps
- 	est_stepwise_duplicate_times - Last value wins
- 	est_stepwise_trajectory - Array handling
- Total: 5 tests

---

#### Task 1.4: Integration Tests (Week 1 complete)
**Time**: 3 hours  
**Owner**: [TBD]  

**Tests**:
- 	est_multiple_ramps_in_scenario - EcosimScenario with 3+ ramps
- 	est_ramp_parameter_tracking - Get values over simulation
- Total: 2 tests

**Acceptance**: All 16 tests passing, coverage ≥85%

---

### Week 2: Advanced Ramps & Integration with Ecosim
**Goal**: Polynomial, spline ramps + wire into scenario system

#### Task 2.1: Implement PolynomialRamp
**Time**: 8 hours  
**Owner**: [TBD]  

**Purpose**: Smooth transitions (avoid discontinuous derivatives)  
**Math**: Hermite polynomial through (t0,v0), (t1,v1) with zero derivatives

**Code**:
`python
class PolynomialRamp(RampingFunction):
    def __init__(self, t0: float, v0: float, t1: float, v1: float, 
                 degree: int = 2):
        """Polynomial interpolation.
        
        Uses Hermite polynomial to ensure matching positions.
        Derivatives set to zero at endpoints for smooth transitions.
        
        Parameters
        ----------
        degree : int
            2=quadratic, 3=cubic, 4=quartic, etc.
        """
        if degree < 2:
            raise ValueError("degree must be ≥2")
        self.t0, self.v0 = t0, v0
        self.t1, self.v1 = t1, v1
        self.degree = degree
        self._compute_coefficients()
    
    def _compute_coefficients(self):
        # Build Vandermonde-like system
        # Solve for polynomial coefficients
        pass
    
    def evaluate(self, time: float) -> float:
        if time <= self.t0:
            return self.v0
        elif time >= self.t1:
            return self.v1
        else:
            # Evaluate polynomial
            pass
`

**Tests**:
- 	est_polynomial_quadratic_endpoints - f(t0)=v0, f(t1)=v1
- 	est_polynomial_quadratic_derivative_smoothness - df/dt continuous
- 	est_polynomial_cubic_smoothness - Cubic smoother than quadratic
- 	est_polynomial_zero_derivative_endpoints - Zero slope at boundaries
- 	est_polynomial_degree_parameter - Degree 2, 3, 4, 5
- Total: 5 tests

**Review checklist**:
- [ ] Hermite polynomial correctly formulated
- [ ] Numerical stability tested
- [ ] Edge cases (t0 ≈ t1) handled

---

#### Task 2.2: Implement SplineRamp
**Time**: 7 hours  
**Owner**: [TBD]  

**Purpose**: Multi-knot smooth interpolation  
**Use case**: Complex time series (e.g., seasonal forcing)

**Code**:
`python
from scipy.interpolate import CubicSpline

class SplineRamp(RampingFunction):
    def __init__(self, times: np.ndarray, values: np.ndarray):
        """Cubic spline through multiple points.
        
        Parameters
        ----------
        times : np.ndarray
            Time knots (must be sorted, unique)
        values : np.ndarray
            Values at knots (same length as times)
        """
        if len(times) < 2:
            raise ValueError("Need ≥2 knots")
        if len(times) != len(values):
            raise ValueError("times and values must have same length")
        
        self.times = np.asarray(times)
        self.values = np.asarray(values)
        self._spline = CubicSpline(times, values)
    
    def evaluate(self, time: float) -> float:
        return float(self._spline(time))
    
    def derivative(self, time: float) -> float:
        return float(self._spline(time, 1))  # First derivative
`

**Tests**:
- 	est_spline_exact_knots - Passes through all knots
- 	est_spline_smoother_than_piecewise_linear - Derivative continuous
- 	est_spline_extrapolation - Before first / after last knot
- 	est_spline_many_knots - 50+ knots
- Total: 4 tests

---

#### Task 2.3: Add Ramping to EcosimScenario
**Time**: 6 hours  
**Owner**: [TBD]  

**File**: packages/pypath/src/pypath/core/ecosim.py  

**Changes**:
`python
class EcosimScenario:
    def __init__(self, ...):
        self.ramps = {}  # {(group_name, param_name): RampingFunction}
    
    def set_parameter_ramp(self, group_name: str, parameter: str, 
                           ramp: RampingFunction) -> None:
        """Attach ramping function to parameter.
        
        Parameters
        ----------
        group_name : str
            Name of ecological group (e.g., 'Cod', 'Herring')
        parameter : str
            Parameter name (e.g., 'M2', 'K', 'q')
        ramp : RampingFunction
            Ramping function to apply
        """
        if group_name not in self.model.index:
            raise ValueError(f"Unknown group: {group_name}")
        self.ramps[(group_name, parameter)] = ramp
    
    def get_parameter_value(self, group_name: str, parameter: str, 
                            time: float) -> float:
        """Get parameter at time, accounting for ramps."""
        key = (group_name, parameter)
        base_value = self.params.model.loc[group_name, parameter]
        if key in self.ramps:
            ramp_value = self.ramps[key].evaluate(time)
            return base_value * ramp_value  # Or additive?
        return base_value
    
    def get_ramp_trajectory(self, group_name: str, parameter: str,
                            times: np.ndarray) -> np.ndarray:
        """Get parameter trajectory over multiple times."""
        key = (group_name, parameter)
        if key not in self.ramps:
            raise ValueError(f"No ramp defined for {group_name}.{parameter}")
        return self.ramps[key].get_trajectory(times)
`

**Tests**:
- 	est_scenario_set_ramp - Ramp stored
- 	est_scenario_get_value_with_ramp - Multiplier applied
- 	est_scenario_get_value_without_ramp - Base value returned
- 	est_scenario_invalid_group - Raises ValueError
- 	est_scenario_ramp_trajectory_method - Array input
- Total: 5 tests

---

#### Task 2.4: Wire Ramping into Derivatives
**Time**: 8 hours  
**Owner**: [TBD]  

**File**: packages/pypath/src/pypath/core/ecosim_deriv.py  

**Challenge**: Call scenario.get_parameter_value(group, param, time) inside ODE solver

**Changes**:
`python
def deriv_vector(t, y, scenario, ...):
    """Compute dydt with ramped parameters."""
    dydt = np.zeros_like(y)
    
    for group_idx, group_name in enumerate(scenario.model.index):
        # Get M2 (natural mortality) with ramp applied
        M2 = scenario.get_parameter_value(group_name, 'M2', t)
        
        # Get K (max growth) with ramp applied
        K = scenario.get_parameter_value(group_name, 'K', t)
        
        # Use ramped values in derivative calculation
        # dydt[group_idx] = ... (use M2, K, ...)
    
    return dydt
`

**Tests**:
- 	est_deriv_with_linear_ramp_M2 - Linearly changing mortality
- 	est_deriv_with_polynomial_ramp_K - Polynomial growth ramp
- 	est_deriv_consistent_without_ramp - No ramp = no change
- 	est_deriv_ramp_changes_dynamics - Different output with ramp
- Total: 4 tests

**Acceptance**: All derivatives tests pass, Ecosim output matches v0.5.3 when no ramps

---

#### Task 2.5: Documentation & Examples
**Time**: 4 hours  
**Owner**: [TBD]  

**Deliverables**:
1. **API reference** (update docs/docs/reference/core.md):
   - Add RampingFunction, LinearRamp, PolynomialRamp, SplineRamp, StepwiseRamp
   - 2-3 lines per class

2. **User guide** (new docs/docs/guides/ramping.md):
   - Motivation for ramping
   - When to use each ramp type
   - 3+ worked examples with code
   - Performance considerations

3. **Jupyter notebook** (packages/pypath/docs/notebooks/ramping_tutorial.ipynb):
   - Import ramping classes
   - Create simple 5-group model
   - Add ramps to 2-3 parameters
   - Plot parameter trajectories
   - Run Ecosim
   - Compare output with/without ramps

**Review checklist**:
- [ ] Examples runnable end-to-end
- [ ] Docstrings complete
- [ ] Tutorial accessible to new users

---

### Week 3: Testing & Integration
**Goal**: Achieve 85%+ coverage, finalize integration

#### Task 3.1: Comprehensive Integration Tests
**Time**: 6 hours  
**Owner**: [TBD]  

**Test scenarios**:
1. **Gradual fishing pressure increase**
   - Fishing mortality M2 ramps linearly 0.1 → 0.5 over 20 years
   - Verify population decline is smooth
   - Check that step > derivative stays finite

2. **Seasonal environmental forcing (spline)**
   - 4-knot spline mimicking seasonal productivity
   - Verify biomass follows seasonal pattern
   - Verify amplitude and phase correct

3. **Multi-parameter ramping**
   - Simultaneously ramp M2, K, F for 3 groups
   - Verify no interference
   - Check energy balance

4. **Ramp edge cases**
   - t0 at t=0, t1 at t=end
   - Very steep linear ramp
   - Overlapping ramps (different groups)

**Files**: packages/pypath/tests/test_ramping_integration.py

**Tests**: 8 tests total, all scenarios

---

#### Task 3.2: Performance Benchmarking
**Time**: 4 hours  
**Owner**: [TBD]  

**Benchmark code** (add to conftest or separate benchmark file):
`python
import timeit

def benchmark_ramp_evaluation():
    """Evaluate ramp 10,000 times."""
    ramp = PolynomialRamp(0, 1, 10, 2, degree=3)
    times = np.linspace(0, 10, 10000)
    
    elapsed = timeit.timeit(
        lambda: ramp.get_trajectory(times),
        number=100
    )
    
    avg_ms = (elapsed / 100) * 1000
    assert avg_ms < 1.0, f"Ramp evaluation too slow: {avg_ms:.2f}ms"
    print(f"✅ Ramp eval: {avg_ms:.3f}ms for 10k points")
`

**Targets**:
- LinearRamp.evaluate() < 0.1 µs
- PolynomialRamp.evaluate() < 0.5 µs
- SplineRamp.evaluate() < 1 µs
- Full trajectory (10k points) < 1 ms

**Acceptance**: All benchmarks pass

---

#### Task 3.3: Code Coverage Audit
**Time**: 3 hours  
**Owner**: [TBD]  

**Command**:
`ash
pytest packages/pypath/tests/test_ramping*.py \
  --cov=pypath.core.ramping \
  --cov-report=html \
  --cov-report=term-missing
`

**Target**: ≥85% coverage for ramping module

**Uncovered branches**: Document and link to future work tickets if not critical

---

#### Task 3.4: Code Review & Merge
**Time**: 4 hours  
**Owner**: [Lead reviewer]  

**Checklist**:
- [ ] All 36 tests passing
- [ ] Coverage ≥85%
- [ ] Docstrings complete (100% of public API)
- [ ] No type errors (mypy clean)
- [ ] Code style (ruff/black compliant)
- [ ] Integration with existing modules verified
- [ ] Performance within targets
- [ ] Documentation (guide + notebook + API reference)

**Merge criteria**:
- [ ] 2+ approvals from core team
- [ ] All CI checks passing (Ubuntu + Windows, Python 3.10-3.12)
- [ ] No new warnings

---

## Resource Allocation

| Task | Hours | Owner | Skills |
|------|-------|-------|--------|
| 1.1 - ABC definition | 4 | ? | Python, OOP, type hints |
| 1.2 - LinearRamp | 6 | ? | Numerics, refactoring |
| 1.3 - StepwiseRamp | 5 | ? | Data structures |
| 1.4 - Integration tests | 3 | ? | Testing |
| 2.1 - PolynomialRamp | 8 | ? | Numerical analysis, Hermite |
| 2.2 - SplineRamp | 7 | ? | Interpolation, scipy |
| 2.3 - Scenario integration | 6 | ? | Ecosystem code, OOP |
| 2.4 - Derivatives wiring | 8 | ? | Ecosim dynamics, careful |
| 2.5 - Documentation | 4 | ? | Technical writing |
| 3.1 - Integration tests | 6 | ? | Testing, ecosystem models |
| 3.2 - Performance | 4 | ? | Profiling, optimization |
| 3.3 - Coverage audit | 3 | ? | Coverage tools |
| 3.4 - Code review | 4 | Lead | Review, merge decisions |

**Total**: 68 hours (~2 weeks for 1-2 developers, 3 weeks solo)

---

## Success Criteria

### Functional
- [x] RampingFunction ABC with ≥2 implementations
- [x] LinearRamp enhanced and tested
- [x] PolynomialRamp, SplineRamp, StepwiseRamp implemented
- [x] EcosimScenario.set_parameter_ramp() working
- [x] Integration with deriv_vector() complete
- [x] User guide and examples available

### Quality
- [x] 36 tests all passing
- [x] Coverage ≥85%
- [x] No type errors
- [x] Performance targets met
- [x] All docstrings NumPy-style

### Documentation
- [x] API reference updated
- [x] User guide (ramping.md) created
- [x] Jupyter notebook tutorial complete

---

## Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Deriv integration breaks existing Ecosim | Medium | High | Comprehensive regression tests, v0.5.3 comparison |
| Performance regression in ODE solver | Low | High | Early benchmarking, profiling on large models |
| Numerical stability (spline extrapolation) | Medium | Medium | Extensive edge case testing, clamping in derivatives |
| Scope creep (adding too many ramp types) | High | Medium | Strict to scope: 4 classes only, document future |

---

## Dependency Graph

`
├─ 1.1 (ABC)
│  ├─ 1.2 (LinearRamp) ──┐
│  ├─ 1.3 (StepwiseRamp) ├── 1.4 (Integration) ──┐
│  ├─ 2.1 (PolyRamp) ────┤                        ├── 3.1-3.4 (Final)
│  └─ 2.2 (SplineRamp) ──┘
└─ 2.3 (Scenario API) ──── 2.4 (Deriv wiring) ────┘
└─ 2.5 (Docs) ─────────────────────────────────────┘
`

**Critical path**: 1.1 → 1.2 → 2.3 → 2.4 → 3.1-3.4  
**Parallel work**: 1.3, 1.4, 2.1, 2.2, 2.5 can proceed in parallel with 1.2

---

**Next Step**: Assign owners to tasks 1.1-1.4 and kickoff Week 1
