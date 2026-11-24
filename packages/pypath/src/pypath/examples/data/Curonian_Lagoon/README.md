# Curonian Lagoon pelagic food web, 2005–2007

Ecopath mass balance of the Curonian Lagoon pelagic food web: 19 groups (17 consumers, phytoplankton and detritus) and one fleet. EwE 6 model *curonian pelagic*, Marine Research Institute, Klaipėda University; no peer-reviewed reference is attached to it. It is the model of PyPath's Curonian Lagoon case study.

Meiobenthos has both B and EE in the database; its B is left for Ecopath to estimate from the EE of 0.95 (`ecopath/README.md`).

Seven consumers take part of their diet from outside the lagoon (`EcopathGroup.ImpVar` in the database: Demersal fish 0.345, the five bird groups 0.2 each, Predatory fish 0.306), carried in the Import row of the diet file. Copies of this example made before 24 November 2026 (pypath-ewe 1.22.0 and earlier) had that row at zero: the diets of those groups did not sum to 1, and the Ecosim baseline drifted (over 50 years the birds fell to 0.38 of their start and oligochaetes rose to 1.16). With the row the baseline stays within 0.03 % of its start. The Ecopath biomasses and EE are the same in both.

## Files

- `ecopath/`: EwE-F Ecopath inputs: `*_Scenario.csv` (basic input), `*_DC.csv` (diet, Import row last), `*_DetFate.csv` (detritus fate), and `*_filenames.nml`. `ecopath/README.md` is EPath's note on the model and every change made to the source file.
- `ecosim/`: EwE-F Ecosim settings: `*_Scenario.csv` (group settings), `*_vul.csv` (vulnerabilities), `*_F.csv` (fishing forcing) and `*_filenames.nml`. `ecosim/README.md` is EPath's note on the Ecosim and Ecospace inputs and how each was made; of the files it lists, only those named here are shipped.
- `ecospace/`: the model's Ecospace set-up, from the same EwE-F case:
    - `Curonian_Lagoon_grid.csv`: the land/sea mask, 94 rows (north to south) by 51 columns, 1,576 sea cells of about 1 km: the lagoon (1,563 cells) and the Klaipėda strait up to its mouth (13).
    - `Curonian_Lagoon_depth.csv`: the cells' mean water depth in metres (land -999): 0 to 5.3 m in the lagoon, 3.3 m on average; the dredged strait 12 m.
    - `Curonian_Lagoon_georef.json`: the grid's north-west corner (55.728° N, 20.5084° E) and cell steps (0.009° of latitude, 0.0158° of longitude).
    - `Curonian_Lagoon_SpatialDistribution/Group_1.csv` to `Group_19.csv`: a habitat map per group, in the model's group order. Each follows the cell depth by a curve stated in EPath's note (water-column groups scale with depth, soft-bottom benthos favours the deeper basin, the heron the shores, gulls nowhere in particular); chironomids and oligochaetes also follow the sediment's organic carbon. These are stated assumptions, not fitted.

  `example_ecospace("Curonian_Lagoon")` makes PyPath's Ecospace configuration of them (`pypath.io.ewef.read_ewef_ecospace`): the sea cells as patches, each map over its largest value as the group's habitat capacity, and a dispersal rate per group. Two things are chosen, not read, because EwE-F's files do not hold them: the cell side (1 km, from the latitude step) and the dispersal speeds (EwE's for a new Ecospace group: 300 km a year, 10 for detritus). It is PyPath's Ecospace (EwE 6's movement) on this grid and these maps, not EwE-F's own movement, and the case's advection field (a direction per cell from the lagoon's residual circulation) is not shipped and not applied. The same set-up, written into an EwE database, is the Ecospace scenario of `Data/curonian_pelagic.sqlite` in the repository; a test checks that the two agree.

- `ecospace/curonian29.grd`: an alternative grid, not part of the EwE-F case: a SHYFEM mesh of the lagoon, the Klaipėda strait and the sea off the coast (1,309 nodes, 2,027 elements, 2,296 km²). Its element types are the 29 boxes of a hydrodynamic box model of the lagoon, so `example_ecospace_grid("Curonian_Lagoon")` gives a 29-patch Ecospace grid that the box model's output (`boxes_2d.txt`) fits as it is. Only the grid is here: no habitat maps, dispersal rates or model output. By the mesh's own depths, 21 boxes are 1.4 to 4.5 m deep on average, box 11 is 7.1 m, and boxes 1, 4, 7, 10, 12, 13 and 16 are over 10 m.

The Ecopath and Ecosim files are copied from EPath (EwE-F) test cases `ecopath-f/testcases/Curonian_Lagoon` and `ecosim-f/testcases/Curonian_Lagoon` at EPath commit 5bdbeb9 (24 November 2026; the case itself was last built on 18 November, EPath e725905). The namelists' paths are EPath's; PyPath finds the files beside them.

## Use

```python
from pypath.examples import load_example_model, example_ecosim_scenario
from pypath.core.ecopath import rpath
from pypath.core.ecosim import rsim_run

params = load_example_model("Curonian_Lagoon")
balanced = rpath(params)
scenario = example_ecosim_scenario("Curonian_Lagoon", years=range(1, 11))
result = rsim_run(scenario, method="RK4", years=range(1, 11))
```

On its own grid, with its habitat maps (a ten-year run on the 1,576 cells takes about five minutes):

```python
from pypath.examples import example_ecospace
from pypath.spatial import rsim_run_spatial

ecospace = example_ecospace("Curonian_Lagoon")
spatial = rsim_run_spatial(scenario, ecospace=ecospace, years=range(1, 11))
```

On the 29 boxes instead (habitat and movement are yours to set):

```python
import numpy as np
from pypath.examples import example_ecospace_grid
from pypath.spatial import EcospaceParams, rsim_run_spatial

grid = example_ecospace_grid("Curonian_Lagoon")
n = scenario.params.NUM_GROUPS
ecospace = EcospaceParams(
    grid=grid,
    habitat_preference=np.ones((n, grid.n_patches)),
    habitat_capacity=np.ones((n, grid.n_patches)),
    dispersal_rate=np.full(n, 1.0),
    advection_enabled=np.zeros(n, dtype=bool),
    gravity_strength=np.zeros(n),
)
spatial = rsim_run_spatial(scenario, ecospace=ecospace, years=range(1, 11))
```

PyPath's balance matches Ecopath-F's solution of this model (biomass and EE of every group; `tests/test_ewef_example_models.py`).
