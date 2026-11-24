# Example Models

PyPath ships three real food-web models, from EPath's EwE-F test cases, ready
to balance and run. In the app, pick one from **Open a real model…** on the
Home page, or under **Example Model** on the Ecosim page, which also balances
it; its EwE-F Ecosim settings then become the starting scenario on the Ecosim
page.

| Name | Model | Groups | Source |
|---|---|---|---|
| `Central_Baltic` | Baltic Proper, 1974, modified for a stationary baseline | 22 | Tomczak et al. (2012), *Ecological Modelling* 230, 123–147 |
| `Curonian_Lagoon` | Curonian Lagoon pelagic food web, 2005–2007 | 19 | Marine Research Institute, Klaipėda University |
| `Lithuanian_Coast` | Exposed sandy coast of Lithuania, with round goby | 44 | Razinkovas-Baziukas et al. (2017), *Estuarine, Coastal and Shelf Science* 195, 110–123 |

Each model's folder (`pypath/examples/data/<name>/`) has a README with its
source, the changes made to it and the files included: the EwE-F Ecopath
inputs and the Ecosim settings (vulnerabilities, group settings, fishing
forcing). PyPath's balance of each matches Ecopath-F's.

The Curonian Lagoon model also ships a spatial grid for Ecospace: the 29
boxes of a SHYFEM box model of the lagoon (`example_ecospace_grid`). Only the
grid: habitat maps and dispersal rates are yours to set. In the app, the
Ecospace page offers it as **Load the example's grid** while the example is
open.

## Balance and run one

```python
from pypath.core.ecopath import rpath
from pypath.core.ecosim import rsim_run
from pypath.examples import example_ecosim_scenario, load_example_model

params = load_example_model("Lithuanian_Coast")
balanced = rpath(params)

years = range(1, 11)
scenario = example_ecosim_scenario("Lithuanian_Coast", years=years)
result = rsim_run(scenario, method="RK4", years=years)
```

## The Curonian Lagoon on its 29 boxes

```python
import numpy as np
from pypath.examples import example_ecosim_scenario, example_ecospace_grid
from pypath.spatial import EcospaceParams, rsim_run_spatial

years = range(1, 3)
scenario = example_ecosim_scenario("Curonian_Lagoon", years=years)
grid = example_ecospace_grid("Curonian_Lagoon")  # 29 patches, one per box
n = scenario.params.NUM_GROUPS
ecospace = EcospaceParams(
    grid=grid,
    habitat_preference=np.ones((n, grid.n_patches)),
    habitat_capacity=np.ones((n, grid.n_patches)),
    dispersal_rate=np.full(n, 1.0),
    advection_enabled=np.zeros(n, dtype=bool),
    gravity_strength=np.zeros(n),
)
result = rsim_run_spatial(scenario, ecospace=ecospace, years=years)
```

The grid's element types are the box numbers, so the box model's own output
fits it: see [the hydrodynamic link](hydrodynamic-link.md) for fields, water
transport and a capacity by month.

## An Ecospace scenario from an EwE database

`pypath.io.read_ecospace` reads a database's Ecospace scenario: its grid (the
water cells of the depth map), each group's habitat capacity and its dispersal
rates. It has a row for each living and detritus group; a run also needs the
fleets' rows, which `with_group_rows` adds:

```python
from pypath.io import read_ecospace
from pypath.io.ewemdb import ecosim_scenario_from_ewemdb

scenario = ecosim_scenario_from_ewemdb("model.eweaccdb", scenario=1)
p = scenario.params
read = read_ecospace("model.eweaccdb", n_groups=p.NUM_LIVING + p.NUM_DEAD)
ecospace = read.ecospace.with_group_rows(p.NUM_GROUPS)
```

In the app, the Ecospace page offers the scenarios of the database you
imported under **Stored Ecospace Scenario**.

## API

::: pypath.examples
    options:
      show_root_heading: true
