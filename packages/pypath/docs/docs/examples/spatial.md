# Spatial Modeling (Ecospace)

This guide shows how to set up and run a spatial Ecospace simulation.

## Prerequisites

```bash
pip install pypath-ewe[spatial]
```

## Creating a Spatial Grid

```python
from pypath.spatial import create_1d_grid, EcospaceParams
import numpy as np

# Create a 1D grid with 10 patches
grid = create_1d_grid(n_patches=10, spacing=1.0)
```

## Setting Up Ecospace Parameters

```python
# After creating a balanced model and Ecosim scenario:
from pypath import create_rpath_params, rpath, rsim_scenario

# ... set up params, balance model, create scenario ...

ng = scenario.params.NUM_GROUPS + 1  # +1 for "Outside" group

ecospace = EcospaceParams(
    grid=grid,
    habitat_preference=np.ones((ng, 10)),
    habitat_capacity=np.ones((ng, 10)),
    dispersal_rate=np.full(ng, 2.0),
    advection_enabled=np.zeros(ng, dtype=bool),
    gravity_strength=np.zeros(ng),
)
```

## Running the Simulation

```python
from pypath.spatial.integration import run_ecospace

results = run_ecospace(scenario, ecospace, years=range(1, 11))
```

## Hexagonal Grids

For realistic spatial modeling, use hexagonal grids:

```python
from pypath.spatial import create_hexagonal_grid

grid = create_hexagonal_grid(
    bounds=(0, 0, 100, 100),
    resolution=10.0,
)
```

See the [Spatial API Reference](../api/spatial.md) for full details.
