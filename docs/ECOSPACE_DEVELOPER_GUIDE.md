# ECOSPACE Developer Guide

Technical documentation for developers extending or modifying PyPath ECOSPACE.

## Architecture Overview

ECOSPACE extends Ecosim from non-spatial (state = `[n_groups]`) to spatial (state = `[n_groups, n_patches]`) while maintaining full backward compatibility.

### Design Principles

1. **Optional Extension**: Spatial features are opt-in via `scenario.ecospace`
2. **Modular Components**: Each feature (grids, dispersal, habitat, fishing) is independent
3. **Conservation Laws**: Mass and flux conservation guaranteed by design
4. **Performance**: Sparse matrices, vectorized operations, minimal overhead
5. **Testability**: 140+ unit tests with >95% coverage

---

## Module Structure

```
src/pypath/spatial/
├── __init__.py              # Public API exports
├── ecospace_params.py       # Core data structures
├── gis_utils.py             # Grid creation, shapefile I/O
├── connectivity.py          # Adjacency, distances, graph operations
├── dispersal.py             # Movement kernels (diffusion, advection)
├── external_flux.py         # Ocean model integration
├── environmental.py         # Environmental layers, drivers
├── habitat.py               # Response functions, suitability
├── fishing.py               # Spatial effort allocation
├── integration.py           # Spatial RK4, main simulation loop
└── plotting.py              # Visualization (future)

tests/
├── test_grid_creation.py
├── test_dispersal.py
├── test_habitat.py
├── test_environmental.py
├── test_spatial_fishing.py
├── test_spatial_integration.py
├── test_spatial_validation.py
├── test_irregular_grids.py
└── test_backward_compatibility.py
```

---

## Core Algorithms

### Spatial Derivative Calculation

**File**: `src/pypath/spatial/integration.py`

```python
def deriv_vector_spatial(state_spatial, params, forcing, fishing, ecospace,
                         environmental_drivers, t, dt):
    """
    Spatial derivative: dB[i,p]/dt = local_dynamics + spatial_flux

    Algorithm:
    1. For each patch p:
        a. Extract local state: state_patch = state[:, p]
        b. Apply habitat capacity: K_mult = habitat_capacity[:, p]
        c. Calculate local Ecosim dynamics (predation, fishing, M0)
        d. Store in deriv[:, p]

    2. Calculate spatial fluxes:
        a. For each group with dispersal_rate > 0:
            - Compute diffusion flux (Fick's law)
            - If advection_enabled: add habitat advection
        b. For groups with external_flux:
            - Use pre-computed transport

    3. Add spatial fluxes to derivatives:
        deriv += spatial_flux

    Returns: deriv [n_groups+1, n_patches]
    """
```

**Key Implementation Details:**
- **State indexing**: Index 0 = "Outside" (always 0), indices 1+ = groups
- **Habitat capacity**: Multiplies `Bbase` locally before dynamics calculation
- **Flux priority**: External > Advection > Diffusion
- **Conservation**: Flux calculation guarantees Σ_p flux[i,p] = 0

---

### Diffusion Flux (Fick's Law)

**File**: `src/pypath/spatial/dispersal.py`

```python
def diffusion_flux(biomass_vector, dispersal_rate, grid, adjacency):
    """
    Fick's Law: flux = -D * ∇B * (border_length / distance)

    Algorithm:
    1. Initialize net_flux = zeros(n_patches)

    2. For each adjacent pair (p, q):
        a. Get border_length from grid.edge_lengths[(p,q)]
        b. Calculate distance = ||centroid[p] - centroid[q]|| * 111 km/deg
        c. Calculate gradient = biomass[p] - biomass[q]
        d. Calculate flux_rate = dispersal_rate * border_length / distance
        e. flux_value = flux_rate * gradient
        f. net_flux[p] -= flux_value  # Outflow from p
           net_flux[q] += flux_value  # Inflow to q

    3. Return net_flux

    Properties:
    - Flux is symmetric: flux(p→q) = -flux(q→p)
    - Conservation: sum(net_flux) = 0 (numerical precision)
    - Direction: Flows from high to low biomass
    """
```

**Implementation Notes:**
- Uses sparse adjacency matrix for O(n_edges) complexity, not O(n²)
- Only processes each edge once (p < q)
- Border lengths and distances cached in grid for efficiency
- Conversion factor: 1 degree ≈ 111 km at equator

---

### Habitat Advection

**File**: `src/pypath/spatial/dispersal.py`

```python
def habitat_advection(biomass_vector, habitat_preference, gravity_strength,
                      grid, adjacency):
    """
    Directed movement toward preferred habitat.

    Algorithm:
    1. If gravity_strength == 0: return zeros (no movement)

    2. For each adjacent pair (p, q):
        a. habitat_gradient = habitat[q] - habitat[p]
        b. If habitat_gradient <= 0: continue (only move toward better habitat)
        c. flux_rate = gravity_strength * biomass[p] * habitat_gradient
        d. net_flux[p] -= flux_rate  # Outflow from p
           net_flux[q] += flux_rate  # Inflow to q

    3. Return net_flux

    Properties:
    - Movement is directed (not symmetric)
    - Still conserves mass: sum(net_flux) = 0
    - Strength ∈ [0, 1]: 0 = no movement, 1 = strong preference
    """
```

---

### Spatial RK4 Integration

**File**: `src/pypath/spatial/integration.py`

```python
def spatial_rk4_step(state, deriv_func, dt):
    """
    Runge-Kutta 4th order for spatial state.

    Same algorithm as non-spatial, but operates on [n_groups, n_patches] arrays.

    k1 = deriv_func(state, t)
    k2 = deriv_func(state + 0.5*dt*k1, t + 0.5*dt)
    k3 = deriv_func(state + 0.5*dt*k2, t + 0.5*dt)
    k4 = deriv_func(state + dt*k3, t + dt)

    state_new = state + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

    Returns: state_new [n_groups+1, n_patches]
    """
```

**Why RK4 for Spatial?**
- **Accuracy**: 4th order method handles stiff equations better than Euler
- **Stability**: Critical for diffusion-dominated systems
- **Consistency**: Same algorithm as non-spatial Ecosim
- **Overhead**: Minimal (4 derivative evaluations per timestep)

---

## Data Structures

### EcospaceGrid

**Design Decisions:**

1. **Sparse Adjacency Matrix**:
   ```python
   adjacency_matrix: scipy.sparse.csr_matrix  # [n_patches, n_patches]
   ```
   - Why sparse? Most patches have 4-8 neighbors, not n_patches
   - Memory: O(n_edges) instead of O(n²)
   - Speed: Iteration over `nonzero()` is O(n_edges)

2. **Edge Lengths Dictionary**:
   ```python
   edge_lengths: Dict[Tuple[int,int], float]
   ```
   - Key = `(min(p,q), max(p,q))` ensures unique keys
   - Value = shared border length in km
   - Lookup is O(1) for diffusion calculation

3. **Centroids Array**:
   ```python
   patch_centroids: np.ndarray  # [n_patches, 2]
   ```
   - Row i = [longitude, latitude] of patch i
   - Vectorized distance calculations
   - Used for distance penalties, visualization

---

### Habitat Representation

**Choice**: Separate `habitat_preference` and `habitat_capacity`

```python
habitat_preference: np.ndarray  # [n_groups, n_patches], values 0-1
habitat_capacity: np.ndarray    # [n_groups, n_patches], multiplier
```

**Why separate?**
- **Preference**: Where organisms *want* to be (advection)
- **Capacity**: How much the patch can *support* (carrying capacity)
- Allows: High preference but low capacity (attractive but crowded)
- Allows: Low preference but high capacity (avoided but productive)

**Alternative Considered**: Single `habitat_quality` matrix
- Rejected: Conflates two distinct concepts
- Harder to calibrate: What does quality=0.5 mean?

---

## Performance Considerations

### Bottlenecks Identified

1. **Flux Calculation** (50% of spatial simulation time)
   - Solution: Sparse matrices, vectorized operations
   - Future: Numba JIT compilation (marked with `@numba.jit`)

2. **Local Dynamics Per Patch** (40% of time)
   - Solution: Reuse non-spatial `deriv_vector()` code
   - Future: Parallelize across patches (embarrassingly parallel)

3. **Output Storage** (10% of time, but 90% of memory)
   - Solution: Store aggregated output by default
   - Option: `save_spatial=True` for full `[time, groups, patches]`

### Optimization Strategies

**Spatial Loop Vectorization:**
```python
# BEFORE (slow):
for p in range(n_patches):
    state_patch = state[:, p]
    deriv[:, p] = deriv_vector(state_patch, params, ...)

# AFTER (fast):
# Vectorized across patches where possible
# Still requires per-patch call for predation (non-linear)
```

**Adjacency Iteration:**
```python
# EFFICIENT:
rows, cols = adjacency.nonzero()
for idx in range(len(rows)):
    p, q = rows[idx], cols[idx]
    if p >= q: continue  # Skip duplicates
    # Process edge (p, q)

# INEFFICIENT (avoid):
for p in range(n_patches):
    for q in range(n_patches):
        if adjacency[p, q]:
            # Process edge
```

**Memory Footprint:**
```python
# Non-spatial: state = [10 groups] = 80 bytes
# Spatial (25 patches): state = [10, 25] = 2 KB
# Spatial (100 patches): state = [10, 100] = 8 KB

# 10-year simulation, 120 months:
# Non-spatial output: 120 * 10 * 8 = 9.6 KB
# Spatial (25 patches): 120 * 10 * 25 * 8 = 240 KB
# Spatial (100 patches): 120 * 10 * 100 * 8 = 960 KB
```

---

## Extending ECOSPACE

### Adding a New Movement Type

Example: Tidal advection

**Step 1**: Create movement function in `dispersal.py`:
```python
def tidal_advection(
    biomass_vector: np.ndarray,
    tidal_strength: float,
    tidal_direction: np.ndarray,  # [n_patches, 2] - direction vectors
    grid: EcospaceGrid,
    adjacency: scipy.sparse.csr_matrix
) -> np.ndarray:
    """Calculate tidal transport."""
    n_patches = len(biomass_vector)
    net_flux = np.zeros(n_patches)

    rows, cols = adjacency.nonzero()
    for idx in range(len(rows)):
        p, q = rows[idx], cols[idx]
        if p >= q: continue

        # Calculate directional component
        edge_vector = grid.patch_centroids[q] - grid.patch_centroids[p]
        edge_direction = edge_vector / np.linalg.norm(edge_vector)

        # Project tidal direction onto edge
        tidal_p = np.dot(tidal_direction[p], edge_direction)

        if tidal_p > 0:  # Tide flows p → q
            flux = tidal_strength * biomass_vector[p] * tidal_p
            net_flux[p] -= flux
            net_flux[q] += flux

    return net_flux
```

**Step 2**: Add to `EcospaceParams`:
```python
@dataclass
class EcospaceParams:
    # ... existing fields ...
    tidal_enabled: Optional[np.ndarray] = None  # [n_groups], boolean
    tidal_strength: Optional[np.ndarray] = None  # [n_groups], 0-1
    tidal_direction: Optional[np.ndarray] = None  # [n_patches, 2]
```

**Step 3**: Integrate in `calculate_spatial_flux()`:
```python
def calculate_spatial_flux(state, ecospace, params, t):
    # ... existing diffusion and advection ...

    # Add tidal transport
    if ecospace.tidal_enabled is not None:
        for group_idx in range(1, n_groups):
            if ecospace.tidal_enabled[group_idx]:
                tidal_flux = tidal_advection(
                    state[group_idx],
                    ecospace.tidal_strength[group_idx],
                    ecospace.tidal_direction,
                    ecospace.grid,
                    ecospace.grid.adjacency_matrix
                )
                flux[group_idx] += tidal_flux

    return flux
```

**Step 4**: Write tests:
```python
def test_tidal_advection_conserves_mass():
    flux = tidal_advection(biomass, strength=0.5, direction, grid, adj)
    assert abs(flux.sum()) < 1e-10

def test_tidal_advection_direction():
    # Tide flows east
    direction = np.column_stack([np.ones(10), np.zeros(10)])
    flux = tidal_advection(biomass, 0.5, direction, grid, adj)
    # Western patches should have outflow
    assert flux[0] < 0
```

---

### Adding Environmental Response

Example: Oxygen-dependent habitat

**Step 1**: Create response function in `habitat.py`:
```python
def create_hypoxia_response(
    threshold: float = 2.0,  # mg/L
    lethal: float = 0.5      # mg/L
) -> Callable:
    """Response to dissolved oxygen.

    Optimal: > threshold
    Stressful: [lethal, threshold]
    Lethal: < lethal
    """
    def response(oxygen: float) -> float:
        if oxygen < lethal:
            return 0.0
        elif oxygen < threshold:
            return (oxygen - lethal) / (threshold - lethal)
        else:
            return 1.0

    return np.vectorize(response)
```

**Step 2**: Use in simulation:
```python
# Load oxygen data
oxygen_layer = EnvironmentalLayer(
    name='dissolved_oxygen',
    units='mg/L',
    values=oxygen_timeseries,  # [n_months, n_patches]
    times=np.arange(n_months) / 12.0
)

# Create response
cod_oxygen_response = create_hypoxia_response(threshold=3.0, lethal=1.0)

# Calculate habitat at each timestep
for month in range(n_months):
    oxygen_values = oxygen_layer.get_value_at_time(month / 12.0)
    habitat_capacity[cod_idx, :] = cod_oxygen_response(oxygen_values)
```

---

## Testing Strategy

### Test Categories

1. **Unit Tests** (fast, isolated)
   ```python
   def test_diffusion_conserves_mass():
       """Single function, simple inputs, verify properties."""
   ```

2. **Integration Tests** (medium, multiple components)
   ```python
   def test_spatial_flux_combined():
       """Diffusion + advection + external, verify total conservation."""
   ```

3. **Validation Tests** (slow, scientific correctness)
   ```python
   def test_spatial_vs_nonspatial_equivalence():
       """1-patch spatial must equal non-spatial exactly."""
   ```

4. **Performance Tests** (benchmarks)
   ```python
   def test_100_patch_simulation_speed():
       """Run 10-year simulation, assert < 60 seconds."""
   ```

### Test Data Generation

**Grid Fixtures**:
```python
@pytest.fixture
def simple_grid():
    """5x5 regular grid for fast tests."""
    return create_regular_grid((0,0,5,5), nx=5, ny=5)

@pytest.fixture
def coastal_transect():
    """1D grid for gradient tests."""
    return create_1d_grid(n_patches=10, spacing=1.0)
```

**Biomass Fixtures**:
```python
@pytest.fixture
def concentrated_biomass(simple_grid):
    """All biomass in center patch."""
    biomass = np.zeros((3, 25))  # 2 groups + Outside, 25 patches
    biomass[1, 12] = 100.0  # Center of 5x5 grid
    return biomass
```

### Numerical Precision

**Conservation Tolerances**:
```python
# Mass conservation
assert abs(total_biomass_final - total_biomass_initial) / total_biomass_initial < 0.01

# Flux conservation
assert abs(flux.sum()) < 1e-10  # Strict for single timestep

# Spatial sum
np.testing.assert_allclose(
    result.out_Biomass,
    result.out_Biomass_spatial.sum(axis=2),
    rtol=1e-8
)
```

---

## Common Pitfalls

### 1. State Indexing

❌ **WRONG**:
```python
# Assuming ecospace parameters start at 0
for group in range(n_groups):
    flux = diffusion_flux(state[group], ecospace.dispersal_rate[group], ...)
```

✅ **CORRECT**:
```python
# state[0] = Outside, ecospace params index from 0 but map to state[1:]
for group_idx in range(1, n_groups + 1):
    ecospace_idx = group_idx - 1
    flux = diffusion_flux(state[group_idx], ecospace.dispersal_rate[ecospace_idx], ...)
```

### 2. Edge Processing

❌ **WRONG**:
```python
# Processing each edge twice
rows, cols = adjacency.nonzero()
for i, j in zip(rows, cols):
    process_edge(i, j)  # Will process (i,j) and (j,i) separately
```

✅ **CORRECT**:
```python
# Process each edge once
rows, cols = adjacency.nonzero()
for idx in range(len(rows)):
    i, j = rows[idx], cols[idx]
    if i >= j: continue  # Skip duplicates and self-loops
    process_edge(i, j)
```

### 3. Flux Direction

❌ **WRONG**:
```python
# Confusing flux sign
gradient = biomass[p] - biomass[q]
flux_value = dispersal_rate * gradient
net_flux[p] += flux_value  # WRONG: high biomass gets inflow
net_flux[q] -= flux_value
```

✅ **CORRECT**:
```python
# Flux flows from high to low
gradient = biomass[p] - biomass[q]
flux_value = dispersal_rate * gradient
net_flux[p] -= flux_value  # Outflow from high
net_flux[q] += flux_value  # Inflow to low
```

---

## Future Development

### Planned Features

1. **3D Grids** (depth layers)
   - State: `[n_groups, n_patches, n_layers]`
   - Vertical diffusion/advection
   - Depth-dependent processes

2. **Adaptive Timesteps**
   - Detect stiff systems (large gradients)
   - Reduce `dt` when needed
   - Flag for user review

3. **Parallel Patches**
   - Local dynamics is embarrassingly parallel
   - Use `multiprocessing` or `joblib`
   - 4-8x speedup on multi-core systems

4. **GPU Acceleration**
   - Flux calculations on CUDA
   - For very large grids (>1000 patches)

### Contributing

**Code Style**:
- Follow PEP 8
- Type hints for all public functions
- Docstrings in NumPy format
- Maximum line length: 100 characters

**Pull Request Process**:
1. Create feature branch: `feature/new-movement-type`
2. Write tests first (TDD)
3. Implement feature
4. Ensure all tests pass: `pytest tests/`
5. Add documentation
6. Submit PR with description

**Review Checklist**:
- [ ] Tests added and passing
- [ ] Docstrings complete
- [ ] Type hints present
- [ ] Conservation laws verified
- [ ] Backward compatibility maintained
- [ ] Performance acceptable (< 10% overhead)

---

## References

### Scientific Background

- **Walters et al. (1999)**: Original Ecospace description
- **Christensen & Walters (2004)**: Ecopath with Ecosim methods
- **Steenbeek et al. (2016)**: Modern Ecospace implementation

### Implementation References

- **NumPy Sparse**: https://docs.scipy.org/doc/scipy/reference/sparse.html
- **GeoPandas**: https://geopandas.org/
- **RK4 Methods**: Press et al., "Numerical Recipes"

---

## Contact

For development questions:
- GitHub Issues: https://github.com/razinkele/PyPath/issues
- Email: razinkele@gmail.com
