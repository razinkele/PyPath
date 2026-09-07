# Individual-Based Model (IBM) Example

This guide demonstrates how to attach an IBM group to an Ecosim
simulation. The IBM replaces the standard ODE derivative for one or more
functional groups, while keeping the rest of the food web coupled via
the normal Ecosim solver.

For parameter tuning advice, see the
[IBM Parameterization Guide](../guides/ibm-parameterization.md).

## Quick Start: Ecosim + SmeltIBM

```python
from pypath import (
    create_rpath_params, rpath, rsim_scenario, rsim_run,
    SmeltIBM, SmeltParams,
)

# 1. Build and balance a model (abbreviated)
params = create_rpath_params(
    groups=["Phytoplankton", "Zooplankton", "Smelt", "Cod", "Detritus"],
    types=[1, 0, 0, 0, 2],
)
# ... fill in model table and diet matrix ...
model = rpath(params)

# 2. Create Ecosim scenario
scenario = rsim_scenario(model, params, years=range(1, 51))

# 3. Set up the IBM
ng = scenario.params.NUM_GROUPS + 1
smelt = SmeltIBM(
    group_index=3,   # Smelt is group 3 in this model
    n_groups=ng,
    params=SmeltParams.baltic_defaults(),
)

# 4. Initialize population from Ecopath biomass
smelt.initialize_from_ecosim(
    biomass=scenario.params.BB[3],
    params={},
    n_super_individuals=500,
)

# 5. Register and run
scenario.params.ibm_groups = {3: smelt}
output = rsim_run(scenario)
```

## How It Works

When `ibm_groups` is set on the scenario, the Ecosim derivative
function (`deriv_vector`) automatically:

1. Skips IBM groups in the standard consumption kernel
2. Calls `compute_step()` on each IBM group
3. Overrides the derivative: `deriv[group_idx] = (ibm_biomass - BB[group_idx]) / dt`
4. Subtracts IBM consumption from prey derivatives

The rest of the food web runs normally through the ODE solver.

## SmeltIBM 5-Phase Timestep

Each call to `compute_step()` executes:

| Phase | Module | Action |
|-------|--------|--------|
| 1. Forage + Grow | `bioenergetics`, `behavior` | Adaptive prey selection, Wisconsin growth |
| 2. Reproduce | `reproduction` | Spawning, Cushing larval survival |
| 3. Predation | `predation` | Size-structured mortality |
| 4. Bookkeeping | -- | Aging, senescence removal, recruit addition |
| 5. Movement | `behavior` | Spatial patch selection (Ecospace only) |

## Multiple IBM Groups

Multiple species can be IBM-managed simultaneously:

```python
smelt = SmeltIBM(group_index=3, n_groups=ng, params=SmeltParams.baltic_defaults())
smelt.initialize_from_ecosim(scenario.params.BB[3], {}, 500)

herring = SmeltIBM(group_index=4, n_groups=ng, params=SmeltParams.baltic_defaults())
herring.initialize_from_ecosim(scenario.params.BB[4], {}, 300)

scenario.params.ibm_groups = {3: smelt, 4: herring}
output = rsim_run(scenario)
```

Each IBM group is processed independently. They share the same prey
biomass pool, so consumption by one group affects prey available to the
other.

## Customizing SmeltParams

Override individual parameter sub-modules while keeping defaults for the
rest:

```python
from pypath.ibm.bioenergetics import BioenergParams
from pypath.ibm.smelt import SmeltParams

# Start from defaults
sp = SmeltParams.baltic_defaults()

# Override just bioenergetics for a warmer-water species
sp.bioenerg = BioenergParams(
    ra=0.0040,
    rb=-0.25,
    q10=2.5,
    t_ref=15.0,
    sda_fraction=0.15,
    unassimilated_fraction=0.22,
    a_length=0.48,
    b_length=0.333,
    energy_density=6.0,
    reproduction_fraction=0.25,
)

# Adjust growth parameters
sp.vbgf_linf_mean = 30.0  # larger asymptotic length
sp.max_age = 12.0          # longer-lived

ibm = SmeltIBM(group_index=3, n_groups=ng, params=sp)
```

## Creating a Custom IBM Species

Subclass `IBMGroup` and implement four methods:

```python
from pypath.ibm.base import IBMGroup, IBMStepResult, SpatialContext
import numpy as np

class CustomIBM(IBMGroup):
    def __init__(self, group_index, n_groups, my_params):
        super().__init__(group_index, n_groups)
        self.my_params = my_params
        self.individuals = []

    def initialize_from_ecosim(self, biomass, params, n_super_individuals=200):
        # Create initial population matching target biomass
        ...

    def compute_step(self, prey_available, predation_pressure,
                     env_forcing, dt, spatial_context=None):
        # Run one timestep: growth, mortality, reproduction, movement
        biomass = ...
        consumption = np.zeros(self.n_groups)
        return IBMStepResult(
            biomass=biomass,
            production=0.0,
            consumption_by_prey=consumption,
            mortality_count=0.0,
            recruitment_count=0.0,
        )

    def get_aggregate_biomass(self):
        return sum(ind.weight * ind.n_represented / 1e6
                   for ind in self.individuals)

    def get_consumption_by_prey(self):
        return self._last_consumption.copy()
```

See `pypath.ibm.smelt.SmeltIBM` for a complete 530-line reference
implementation.

## Inspecting IBM State

After running, inspect the super-individual population:

```python
# Total biomass
print(f"IBM biomass: {smelt.get_aggregate_biomass():.4f} t/km2")

# Population size
print(f"Super-individuals: {len(smelt.individuals)}")

# Age distribution
ages = [ind.age for ind in smelt.individuals]
print(f"Age range: {min(ages):.1f} - {max(ages):.1f} years")

# Size distribution
weights = [ind.weight for ind in smelt.individuals]
print(f"Weight range: {min(weights):.1f} - {max(weights):.1f} g")

# Consumption breakdown
consumption = smelt.get_consumption_by_prey()
for i, c in enumerate(consumption):
    if c > 0:
        print(f"  Consumed from group {i}: {c:.6f} t/km2")
```

## Visualization in the Shiny App

The PyPath Shiny app provides a dedicated IBM page with:

- **Biomass comparison**: IBM vs standard Ecosim trajectories
- **Size distribution**: Histogram of individual body weights
- **Age structure**: Population age pyramid
- **Energy reserves**: Distribution of condition factor
- **Spatial distribution**: Patch-level biomass map (Ecospace mode)

Launch the app and navigate to Advanced > Individual-Based Model.
