# IBM Bioenergetics & Spatial Movement Guide

Practical guide for parameterizing Individual-Based Model (IBM) groups
in PyPath, covering the Wisconsin bioenergetics model and the spatial
movement algorithm used in Ecospace simulations.

---

## 1. Wisconsin Bioenergetics Model

The IBM uses a classic **Wisconsin bioenergetics** framework
(`pypath.ibm.bioenergetics`). Each super-individual's energy budget per
timestep is:

```
net_energy = assimilated_consumption - metabolism - SDA - reproduction_cost
```

### The Energy Budget Pipeline

```
Consumption (C)
  |
  +---> Unassimilated fraction ---> lost (faeces + excretion)
  |       C x unassimilated_fraction
  |
  +---> Assimilated (A) = C x (1 - unassimilated_fraction)
  |
  +---> SDA = C x sda_fraction  (cost of digestion)
  |
  +---> Metabolism (M) = ra x weight^rb x Q10^((T-Tref)/10) x dt x 365
  |
  +---> Net = A - M - SDA
          |
          +---> If mature & net > 0: subtract reproduction_fraction x net
          |
          +---> weight_change = remaining_net / energy_density
```

### Parameter Reference

| Parameter | Symbol | Baltic Smelt | Typical Range | How to Find |
|-----------|--------|-------------|---------------|-------------|
| `ra` | Metabolic intercept | 0.0033 g O2/g/d | 0.001--0.01 | FishBase metabolism table, or Hanson et al. (1997) |
| `rb` | Metabolic weight exponent | -0.227 | -0.1 to -0.4 | Negative = larger fish have lower per-gram metabolism |
| `q10` | Temperature sensitivity | 2.1 | 1.5--3.0 | 2.0 is a safe default; cold-water fish tend toward 2.5+ |
| `t_ref` | Reference temperature | 10.0 C | Species-specific | Temperature at which `ra` was measured |
| `sda_fraction` | Specific dynamic action | 0.172 | 0.10--0.20 | ~17% of consumption; higher for protein-rich prey |
| `unassimilated_fraction` | Faecal + excretion loss | 0.27 | 0.20--0.35 | ~27% lost; varies by prey quality |
| `a_length` | Length-weight coefficient | 0.55 | Species-specific | From L = a x W^b; get from FishBase L-W table |
| `b_length` | Length-weight exponent | 0.333 | ~1/3 (isometric) | Cube root relationship for isometric growth |
| `energy_density` | Tissue energy content | 5.0 kJ/g | 3.0--7.0 | Higher for fatty species (herring ~6.5, lean whitefish ~4.0) |
| `reproduction_fraction` | Energy to gonads | 0.3 | 0.15--0.50 | Higher for capital breeders, lower for income breeders |

### Temperature Dependence (Q10)

The Q10 factor scales metabolic rate to the current water temperature:

```python
factor = q10 ** ((temp - t_ref) / 10.0)
```

This means:

- At `t_ref` (10 C for smelt): factor = 1.0 (no adjustment)
- At 20 C with Q10=2.1: factor = 2.1 (metabolism doubles)
- At 5 C with Q10=2.1: factor = 0.69 (metabolism drops 31%)

**Practical tip**: If your species lives in a narrow temperature range
(e.g., deep-water fish), Q10 matters less. For species with wide thermal
ranges (e.g., perch, pike), get Q10 from species-specific respirometry
studies.

### How Consumption Feeds Into Growth

In `SmeltIBM.compute_step()`, each super-individual's timestep proceeds as:

```python
# 1. Maximum consumption (allometric scaling)
max_consumption = 0.1 * (weight ** 0.7) * dt * 365.0

# 2. Adaptive foraging allocates max_consumption across prey
allocation = adaptive_forage(prey_dict, max_consumption, length, foraging_params)

# 3. Total consumed feeds into growth_step
ind_consumption = sum(allocation.values())
new_weight, new_energy = growth_step(
    weight, energy_reserve, ind_consumption, temperature, is_mature, dt, params
)

# 4. Length updated from new weight
length = allometric_length(new_weight, a_length, b_length)
```

The 0.7 exponent in `max_consumption` is a standard allometric scaling --
smaller fish eat more per gram than larger fish.

### Batch Processing (Performance)

For populations with hundreds of super-individuals, the vectorized
`growth_step_batch` processes all individuals at once using NumPy arrays
instead of a per-individual Python loop:

```python
from pypath.ibm.bioenergetics import growth_step_batch

new_weights, new_energies = growth_step_batch(
    weights=np.array([5.0, 10.0, 20.0]),
    energy_reserves=np.array([0.5, 1.0, 0.2]),
    consumptions=np.array([3.0, 5.0, 1.0]),
    temperature=15.0,
    is_mature=np.array([False, True, False]),
    dt=1.0 / 12.0,
    params=bioenerg_params,
)
```

`SmeltIBM.compute_step()` uses this automatically in Phase 1 (forage + grow).

### Customizing for a New Species

To parameterize for, say, **pike-perch (Sander lucioperca)**:

```python
from pypath.ibm.bioenergetics import BioenergParams

pikeperch_bioenerg = BioenergParams(
    ra=0.0028,                    # Lower basal metabolism than smelt
    rb=-0.20,                     # Similar weight scaling
    q10=2.3,                      # Slightly higher temperature sensitivity
    t_ref=15.0,                   # Warmer preference species
    sda_fraction=0.15,            # Lower SDA (piscivore, efficient digestion)
    unassimilated_fraction=0.20,  # Fish prey = high assimilation
    a_length=0.50,                # From FishBase L-W for pike-perch
    b_length=0.333,               # Isometric
    energy_density=5.5,           # Slightly higher than smelt
    reproduction_fraction=0.25,   # Moderate spawner
)
```

**Where to find parameters:**

1. **FishBase** -- Search species -- Metabolism tab -- ra, rb values
2. **Hanson et al. (1997)** Wisconsin bioenergetics manual -- Q10, SDA
3. **Published bioenergetics models** -- Search "[species name]
   bioenergetics model" on Google Scholar
4. **If no data**: Use a closely related species and adjust body size
   parameters

---

## 2. Spatial Movement Algorithm

The movement system (`pypath.ibm.behavior`) uses a **score-based
stochastic patch selection** over a sparse adjacency graph.

### Movement Decision Algorithm

For each super-individual each timestep:

```
1. Get reachable patches = {current_patch} U neighbors(current_patch)
2. For each reachable patch p:
     score(p) = habitat_weight x habitat_quality[p]
              + food_weight x food_density[p]
              + predator_weight x 1/(1 + predator_density[p])
3. Apply inertia bonus to current patch:
     score(current) *= (1 + (1 - base_speed))
4. Normalize scores to probabilities
5. Stochastically select destination patch
```

### Movement Parameters

| Parameter | Baltic Smelt | Effect | Tuning Guidance |
|-----------|-------------|--------|-----------------|
| `base_speed` | 0.3 | Low = sedentary, high = mobile | 0.1 for benthic species, 0.5+ for pelagic |
| `habitat_weight` | 0.4 | Importance of habitat quality | Higher for species with strong habitat preferences |
| `food_weight` | 0.4 | Importance of food availability | Higher for actively foraging species |
| `predator_weight` | 0.2 | Importance of predator avoidance | Higher for prey species, lower for apex predators |
| `migration_temp_threshold` | 4.0 C | Temperature trigger for migration | Species-specific spawning temperature |
| `migration_months` | (3,4,5) | Months when migration is allowed | Match spawning season |

### Inertia Mechanism

The key behavioral feature is the **inertia bonus**:

```python
if p == current_patch:
    score += (1.0 - base_speed) * score
```

With `base_speed = 0.3`, the current patch gets a 70% bonus. This means:

- An individual in a patch with equal quality to its neighbors has ~63%
  chance of staying
- Only significantly better neighboring patches will attract movement
- Higher `base_speed` leads to more exploratory movement

### Predator Avoidance

The sigmoid avoidance term `1/(1 + predator_density[p])`:

- When `predator_density = 0`: avoidance score = 1.0 (safe patch)
- When `predator_density = 1`: avoidance score = 0.5 (moderate risk)
- When `predator_density = 10`: avoidance score = 0.09 (high risk,
  strongly avoided)

### Adjacency Matrix Structure

The adjacency matrix is a **scipy sparse CSR matrix** representing which
patches are directly reachable:

```python
import scipy.sparse as sp

# Example: 4-patch linear arrangement (0-1-2-3)
adjacency = sp.csr_matrix([
    [0, 1, 0, 0],   # patch 0 connects to 1
    [1, 0, 1, 0],   # patch 1 connects to 0, 2
    [0, 1, 0, 1],   # patch 2 connects to 1, 3
    [0, 0, 1, 0],   # patch 3 connects to 2
])
```

Ecospace builds this automatically from grid cell adjacency. For custom
grids, provide your own matrix.

### Migration Trigger

Seasonal migration is controlled by `should_migrate()`:

```python
def should_migrate(temperature, month, params):
    return (temperature > params.migration_temp_threshold
            and month in params.migration_months)
```

This is a simple gate -- when true, the standard movement algorithm is
enabled during spawning months.

### Adaptive Foraging (Prey Selection)

The foraging algorithm uses a **profitability-based allocation** with
availability caps:

```
profitability(prey) = (energy_content / handling_time) x available_biomass
```

Consumption is allocated proportionally to profitability. If a prey group
runs out (availability cap), surplus is redistributed to remaining prey
in a second pass.

**Practical example** -- a pike-perch eating 3 prey groups:

```python
import numpy as np
from pypath.ibm.behavior import ForagingParams

foraging_params = ForagingParams(
    energy_content=np.array([0, 0, 0, 3.5, 4.0, 0, 6.0, ...]),  # kJ/g by group
    #                                  ^gr3  ^gr4       ^gr6
    handling_time=np.array([0, 0, 0, 1.0, 1.2, 0, 2.0, ...]),    # time/g
    #                                ^gr3  ^gr4       ^gr6
)
# Profitability: gr3=3.5, gr4=3.33, gr6=3.0
# Pike-perch preferentially consumes group 3, then 4, then 6
```

### Movement Profiles for Different Species Types

**Pelagic schooling fish** (herring, sprat):

```python
from pypath.ibm.behavior import MovementParams

MovementParams(
    base_speed=0.6,          # Highly mobile
    habitat_weight=0.2,      # Less tied to bottom habitat
    food_weight=0.6,         # Strongly follows plankton patches
    predator_weight=0.2,     # School provides some protection
    migration_temp_threshold=6.0,
    migration_months=(4, 5, 6, 9, 10),  # Spring + autumn migrations
)
```

**Benthic/demersal fish** (flounder, cod):

```python
MovementParams(
    base_speed=0.15,         # Sedentary
    habitat_weight=0.6,      # Strong bottom-type preference
    food_weight=0.2,         # Feeds locally
    predator_weight=0.2,
    migration_temp_threshold=8.0,
    migration_months=(3, 4),  # Short spawning migration
)
```

**Apex predator** (pike, perch):

```python
MovementParams(
    base_speed=0.25,         # Moderate movement
    habitat_weight=0.5,      # Territory/structure dependent
    food_weight=0.4,         # Follows prey concentrations
    predator_weight=0.1,     # Low predation risk
    migration_temp_threshold=12.0,
    migration_months=(4, 5),
)
```

---

## 3. Size-Structured Predation

Predation mortality is distributed across super-individuals using a
**log-normal size selectivity** (`pypath.ibm.predation`):

```
selectivity(length) = exp(-0.5 x (ln(length / optimal_prey_length) / selectivity_sd)^2)
```

- Fish at `optimal_prey_length` get selectivity = 1.0 (maximum
  vulnerability)
- Smaller/larger fish are progressively less vulnerable
- `selectivity_sd = 0.5` (Baltic smelt default) = moderately selective
  predator

### Tuning `selectivity_sd`

| Value | Predator Type | Example |
|-------|---------------|---------|
| 0.2--0.3 | Very size-selective (gape-limited) | Small-mouthed bass, zander juveniles |
| 0.4--0.6 | Moderate selectivity (default) | Most fish predators |
| 0.8--1.0 | Generalist predator | Seals, marine mammals, cormorants |

### Predation Parameters

```python
from pypath.ibm.predation import PredationParams

PredationParams(
    optimal_prey_length=10.0,  # cm -- peak vulnerability length
    selectivity_sd=0.5,        # log-normal width
)
```

---

## 4. Reproduction

Reproduction (`pypath.ibm.reproduction`) uses a **Cushing match/mismatch
hypothesis** for larval survival:

```
survival = base_survival x exp(-0.5 x (mismatch / match_window)^2)
```

where `mismatch = |spawn_day - zooplankton_peak_day|`.

### Reproduction Parameters

| Parameter | Baltic Smelt | Description |
|-----------|-------------|-------------|
| `fecundity_coefficient` | 200.0 | Eggs per g^exponent |
| `fecundity_exponent` | 1.2 | Weight-fecundity power law exponent |
| `larval_base_survival` | 0.01 | 1% survival at perfect match |
| `zooplankton_match_window` | 15.0 days | Gaussian width of match window |
| `maturity_energy_threshold` | 0.5 | Minimum energy reserve to spawn |
| `spawning_temp_threshold` | 4.0 C | Minimum temperature for spawning |
| `larval_duration_days` | 30 | Larval phase duration |
| `recruit_weight` | 0.5 g | Weight of newly recruited individual |
| `recruit_length` | 3.0 cm | Length of newly recruited individual |

---

## 5. Practical Tips

**Choosing `num_individuals`**: More super-individuals = smoother
dynamics but slower. 200--500 is typical for a single species. Each
super-individual represents many real fish via `n_represented`.

**Prey group indices**: Must match your Ecopath model numbering (1-based,
0 is "Outside"). Check `rpath_params.Group` for the mapping.

**Timestep coupling**: IBM `compute_step()` is called at each ODE solver
step. The IBM operates in the same `dt` as the Ecosim integration
(typically monthly).

**Debugging**: Run with a simple 2-group model first (1 IBM group + 1
prey) to verify biomass conservation before scaling up.

**Visualization**: The Shiny app's IBM page (`pages/ibm.py`) provides
ready-made plots for biomass comparison, size distribution, age
structure, energy reserves, and spatial distribution.

---

## References

- Hanson, P.C., Johnson, T.B., Schindler, D.E., Kitchell, J.F. (1997).
  *Fish Bioenergetics 3.0*. University of Wisconsin Sea Grant Institute.
- Cushing, D.H. (1990). Plankton production and year-class strength in
  fish populations: an update of the match/mismatch hypothesis.
  *Advances in Marine Biology*, 26, 249--293.
- FishBase (https://www.fishbase.se) -- Species-specific metabolism,
  length-weight, and growth parameters.
