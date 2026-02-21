# IBM Integration into PyPath — Design Document

## Summary

Integrate an Individual-Based Model (IBM) into PyPath's Ecosim simulation engine using a **derivative override** architecture. The IBM replaces one or more functional groups (initially **Baltic smelt**) with super-individual agents that simulate bioenergetics, spatial movement, size-structured predation, adaptive foraging, stochastic reproduction, and life history plasticity. All other groups remain aggregate. Mass balance is enforced strictly at each monthly timestep.

## Motivation

Ecopath/Ecosim models assume homogeneous populations, fixed diet matrices, no spatial memory, and statistical recruitment. These assumptions break down for species like Baltic smelt where:

- Spawning success depends on individual-level temperature sensitivity
- Larval survival depends on zooplankton match/mismatch timing (Cushing hypothesis)
- Size-structured predation from cod/salmon creates non-linear mortality
- Anadromous migration introduces spatial behavior absent from aggregate models

An IBM embedded inside Ecosim captures these emergent properties while preserving the mass-balanced food web context for all other groups.

## Architecture

### Integration Pattern: Derivative Override

PyPath's `deriv_vector()` function computes dB/dt for each group using foraging arena theory. For IBM-replaced groups, we intercept this computation and delegate to the IBM, which simulates individual dynamics and returns aggregate biomass change, consumption, and production.

```
Ecosim monthly loop
  deriv_vector(state, params, forcing, fishing, t)
    Standard groups: foraging arena theory (unchanged)
    IBM groups:
      1. Extract prey availability & predation pressure from QQ matrix
      2. ibm_group.compute_step(prey_available, predation_pressure, env, dt)
           For each super-individual:
             Forage (adaptive, size-dependent)
             Grow (bioenergetics)
             Move (spatial, anadromous migration)
             Reproduce (stochastic, temperature-dependent)
             Survive (predation mortality from Ecosim)
           Return: aggregate biomass, consumption_by_prey, production
      3. deriv[ibm_idx] = production - predation_loss - fishing - m0
      4. Subtract IBM consumption from prey group derivatives
```

### Why Derivative Override

Three architectures were considered:

1. **Derivative Override** (chosen): Replace `deriv_vector()` computation for IBM groups. Minimal disruption, reuses RK4/AB integrator, natural mass balance, spatial module works unchanged.

2. **Loop Injection**: Hook into `rsim_run()` after integration. More freedom for IBM internals but invasive to main loop, manual mass balance enforcement, risk of energy leaks.

3. **Parallel Process**: Separate IBM simulation with state exchange. Maximum isolation but complex communication, serialization overhead, overengineered for single-species use.

## Package Structure

```
packages/pypath/src/pypath/ibm/
  __init__.py              # Public API exports
  base.py                  # IBMGroup ABC + SuperIndividual dataclass + IBMStepResult
  bioenergetics.py         # Growth, metabolism, assimilation (Wisconsin model)
  behavior.py              # Spatial movement, anadromous migration, adaptive foraging
  reproduction.py          # Stochastic spawning, larval survival
  predation.py             # Size-structured predation mortality distribution
  integration.py           # Derivative override logic + mass balance checker
  smelt.py                 # SmeltIBM concrete implementation
```

## Core Data Structures

### SuperIndividual

Represents a cohort of similar fish (super-individual approach):

```python
@dataclass
class SuperIndividual:
    id: int                    # Unique identifier
    n_represented: float       # Number of real fish this agent represents
    weight: float              # Individual body weight (g)
    length: float              # Body length (cm), derived from weight via allometry
    age: float                 # Age (years)
    energy_reserve: float      # Energy storage (kJ)
    patch_idx: int             # Current ECOSPACE patch index
    is_mature: bool            # Reproductive maturity flag
    sex: int                   # 0=female, 1=male
```

### IBMGroup (Abstract Base)

```python
class IBMGroup(ABC):
    group_index: int               # Which Ecosim group index this replaces
    individuals: List[SuperIndividual]
    params: IBMParams              # Species-specific parameter set

    @abstractmethod
    def compute_step(self, prey_available, predation_pressure,
                     env_forcing, dt) -> IBMStepResult:
        """Simulate one timestep for all individuals, return aggregates."""

    @abstractmethod
    def get_aggregate_biomass(self) -> float:
        """Sum of n_represented * weight across all living individuals."""

    @abstractmethod
    def get_consumption_by_prey(self) -> np.ndarray:
        """Total consumption broken down by prey group index."""

    @abstractmethod
    def initialize_from_ecosim(self, biomass, params, n_super_individuals=500):
        """Create initial super-individual population from Ecosim equilibrium state."""
```

### IBMStepResult

```python
@dataclass
class IBMStepResult:
    biomass: float                  # Total biomass after step
    production: float               # Somatic + reproductive production during step
    consumption_by_prey: np.ndarray # Consumption from each prey group (array indexed by group)
    mortality_count: float          # Number of individuals lost (natural + predation)
    recruitment_count: float        # Number of new individuals added (spawning)
```

## SmeltIBM Behaviors

### 1. Bioenergetics (Wisconsin Model)

Growth = Assimilation - Standard Metabolism - Active Metabolism - SDA - Reproduction Cost

- **Assimilation**: Consumption * (1 - Unassimilated fraction)
- **Standard metabolism**: Allometric function of weight, temperature-dependent via Q10
- **Active metabolism**: Proportional to swimming speed (linked to movement behavior)
- **SDA (Specific Dynamic Action)**: Fixed fraction of consumption
- **Reproduction cost**: Energy diverted to gonad development when mature

Temperature dependence uses the standard Q10 formulation. Weight-to-length conversion uses the allometric relationship L = a * W^b.

### 2. Size-Structured Predation

Predation mortality from Ecosim is distributed across individuals based on body size. Smaller smelt face higher per-capita risk from gape-limited predators (cod, salmon).

- Size-selectivity curve: log-normal centered on predator gape size / prey body depth ratio
- Per-individual mortality hazard: `h_i = base_mortality * selectivity(length_i) / mean_selectivity`
- Super-individual death: stochastic process where `n_represented` decreases proportional to hazard

### 3. Adaptive Foraging

Individuals select prey types based on local availability weighted by energy content and handling time, following optimal foraging theory.

- Available prey types: zooplankton, mysids, fish larvae (from Ecosim prey groups)
- Selection probability: proportional to `(energy_content * encounter_rate) / handling_time`
- Diet composition emerges from individual decisions, replacing the fixed DC matrix for this group
- Seasonal variation: prey switching responds to relative abundance changes

### 4. Spatial Movement

Individuals move between ECOSPACE patches based on:

- **Habitat quality**: preference function based on depth, temperature, salinity
- **Food availability**: patches with higher prey density attract individuals
- **Predator avoidance**: move away from patches with high predator biomass
- **Anadromous migration**: obligate spring river migration when water temperature > 5 C
- Movement implemented as stochastic patch transitions using a weighted probability matrix

### 5. Stochastic Reproduction

- Maturity: determined by energy reserve threshold (individual-specific)
- Spawning trigger: temperature > 5 C during spring migration
- Egg production: proportional to female body weight (fecundity-weight relationship)
- Larval survival: stochastic, depends on zooplankton timing match/mismatch
- New super-individuals created from surviving larvae at end of larval phase

### 6. Life History Plasticity

- Individual Von Bertalanffy K and Linf drawn from population distributions at birth
- Growth rate modified by food availability history (good conditions = faster growth)
- Age at maturity varies among individuals (energy threshold, not fixed age)
- Under warming scenarios: faster growth, earlier maturity, smaller asymptotic size

## Integration with Existing Modules

### deriv_vector() Modification

Minimal change to `ecosim_deriv.py`:

```python
# Before the per-group derivative loop:
ibm_groups = params.get('ibm_groups', {})  # Dict[int, IBMGroup]

# In the per-group loop:
for i in range(1, NUM_LIVING + 1):
    if i in ibm_groups:
        ibm = ibm_groups[i]
        prey_avail = extract_prey_availability(QQ, i, NUM_GROUPS)
        pred_pressure = extract_predation_pressure(QQ, i, NUM_LIVING)
        env = extract_env_forcing(forcing, i)

        result = ibm.compute_step(prey_avail, pred_pressure, env, dt=1/12)

        deriv[i] = result.biomass - BB[i]  # Net biomass change as derivative
        # Subtract IBM consumption from prey derivatives
        for prey_idx in range(1, NUM_GROUPS + 1):
            if result.consumption_by_prey[prey_idx] > 0:
                deriv[prey_idx] -= result.consumption_by_prey[prey_idx]
        continue

    # ... standard foraging arena calculation (unchanged) ...
```

### ECOSPACE Integration

The IBM spatial behavior uses the same `EcospaceGrid` from the spatial module:

- `patch_idx` on each SuperIndividual maps to grid patches
- Movement probabilities derived from adjacency matrix and habitat preferences
- IBM individuals coexist with aggregate spatial flux for other groups
- No changes needed to the spatial module itself

### Scenario Setup

To enable IBM for a group, add to `RsimScenario`:

```python
scenario = rsim_scenario(rpath_model)
smelt_ibm = SmeltIBM.from_ecosim_group(
    scenario.params, group_name="Smelt",
    n_super_individuals=500,
    smelt_params=SmeltParams(...)
)
scenario.params['ibm_groups'] = {smelt_ibm.group_index: smelt_ibm}
results = rsim_run(scenario)
```

## Mass Balance Enforcement

Strict per-timestep enforcement:

1. IBM consumption from each prey group is subtracted from that prey's derivative
2. After IBM step, verify: `sum(consumption_by_prey) == total_assimilation + total_waste`
3. If imbalance exceeds 5% tolerance, scale consumption proportionally and log warning
4. Production returned by IBM feeds into the group's derivative naturally

```python
def check_mass_balance(ibm_result, tolerance=0.05):
    total_consumed = np.sum(ibm_result.consumption_by_prey)
    total_accounted = ibm_result.production + waste + respiration
    relative_error = abs(total_consumed - total_accounted) / (total_consumed + 1e-10)
    if relative_error > tolerance:
        logger.warning("IBM mass balance violation: %.1f%%", relative_error * 100)
```

## Testing Strategy

### Unit Tests (per module)

- **bioenergetics**: Growth under known temperature/food conditions matches Wisconsin model
- **predation**: Size-selectivity curve produces correct relative mortality
- **behavior**: Migration triggers at correct temperature, movement probabilities sum to 1
- **reproduction**: Fecundity-weight relationship, larval survival probability bounds

### Integration Tests

- IBM-replaced smelt in 3-group model (phyto -> zoo -> smelt) conserves total biomass over 24 months
- At equilibrium parameters, IBM smelt biomass matches aggregate Ecosim smelt (within 10%)
- External forcing (temperature, fishing) produces qualitatively correct responses

### Validation Tests

- Compare IBM smelt dynamics against published Rpath smelt simulation
- Verify recruitment variability is higher than aggregate model (key IBM advantage)
- Size distribution evolves correctly under predation pressure

### Performance Tests

- Benchmark with 100, 500, 1000, 5000, 10000 super-individuals
- Monthly step should complete in < 1 second for 1000 super-individuals
- Full 50-year simulation in < 5 minutes

## Dependencies

- **numpy**: Array operations (already in pypath)
- **scipy.stats**: Probability distributions for stochastic processes (already available)
- No new external dependencies required

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Performance degradation with many super-individuals | Vectorize operations with NumPy; optional Numba JIT for hot loops |
| Mass balance drift over long simulations | Per-timestep enforcement + periodic Ecopath consistency check |
| Derivative override doesn't capture discrete events (reproduction) | Handle events within compute_step, aggregate to smooth derivative |
| Parameter estimation for IBM is data-intensive | Start with literature values for Baltic smelt; sensitivity analysis to identify critical params |
| IBM adds stochasticity to otherwise deterministic Ecosim | Provide ensemble run capability (multiple IBM realizations) |
