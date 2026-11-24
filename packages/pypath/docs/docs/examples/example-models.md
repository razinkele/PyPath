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

The Curonian Lagoon model is also a spatial model. It ships its own Ecospace
set-up, from the same EwE-F case: a grid of 1,576 cells of 1 km (the lagoon
and the Klaipėda strait) with the cells' depth and a habitat map per group
(`example_ecospace`). As an alternative grid it has the 29 boxes of a SHYFEM
box model of the lagoon (`example_ecospace_grid`), for which habitat maps and
dispersal rates are yours to set. In the app, the Ecospace page offers both
while the example is open: **Load the example's Ecospace set-up** and **Load
the example's grid**.

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

## The Curonian Lagoon on its own grid

```python
from pypath.examples import example_ecosim_scenario, example_ecospace
from pypath.spatial import rsim_run_spatial

years = range(1, 3)
scenario = example_ecosim_scenario("Curonian_Lagoon", years=years)
ecospace = example_ecospace("Curonian_Lagoon")  # 1,576 cells, 20 group rows
result = rsim_run_spatial(scenario, ecospace=ecospace, years=years)
```

The configuration is PyPath's Ecospace on the case's grid and habitat maps:
each map, over its largest value, is its group's habitat capacity, and every
group disperses at EwE's speed for a new Ecospace group (300 km a year, 10
for detritus) over cells of 1 km. Those two are chosen, as the EwE-F files do
not hold them; `pypath.io.ewef.read_ewef_ecospace` takes others. EwE-F's own
movement, and the case's advection field, are not reproduced.

!!! note "The spatial run settles below the Ecopath biomasses"
    Over ten years the map totals settle at about 0.6 of the start for most
    groups. The phytoplankton map is the cause. In EwE 6's Ecospace, which
    PyPath follows, a producer's production in a cell is its capacity there
    times what it would be without it, and can rise only to the group's
    maximum relative P/B (2 by default): at capacity c the producer settles
    at (2c - 1) of its level at capacity 1, and at nothing where c is below
    0.5. The map follows depth and is divided by its largest value, which
    puts 43 % of the lagoon's cells below 0.5. With every map uniform the
    totals stay at 1.000; with phytoplankton's maximum relative P/B at 5
    they end between 0.82 and 0.97; with its pattern moved to EwE's relative
    primary production map (`EcospaceParams.relative_pp`) and its capacity
    uniform, between 0.80 and 1.04. This is what EwE 6's equations give for
    these maps, checked against EwE's source; there is no EwE 6 run of this
    model to compare numbers with. The maps were drawn for EwE-F, which
    rescales each to a mean of 1. The example's README has the figures.

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

EwE places a cell as the same number of degrees each way, which draws a grid
of km cells too narrow east to west (by cos(latitude): 0.57 at 55 N).
`cell_placement="ground"` puts the cells where they lie on the ground instead;
`geometry=True` keeps them as polygons for a map. Only the position changes:
the run is the same on either placement.

```python
read = read_ecospace(
    "model.eweaccdb",
    n_groups=p.NUM_LIVING + p.NUM_DEAD,
    cell_placement="ground",
    geometry=True,
)
```

In the app, the Ecospace page offers the scenarios of the database you
imported under **Stored Ecospace Scenario**, placed on the ground unless you
choose EwE's coordinates.

## API

::: pypath.examples
    options:
      show_root_heading: true
