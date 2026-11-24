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

## What a spatial run does, and how to read EPath's note

On its own grid the model does not stay at its Ecopath biomasses. In a ten-year run (PyPath's Ecospace, the set-up as shipped) the map totals settle at about 0.6 of the start for most groups: phytoplankton, zooplankton, bacteria and detritus at 0.60 to 0.62, the fish at 0.61 to 0.68, the birds at 0.57 to 0.63, the bivalves at 0.47. The cause is the phytoplankton map, shown by three-year runs on the same grid:

| Habitat maps | Map total after 3 years, over the start |
|---|---|
| all uniform | 1.000 for every group |
| phytoplankton uniform, the others as shipped | 0.82 to 1.19, median 0.98 |
| phytoplankton as shipped, the others uniform | 0.53 to 0.92, median 0.61; phytoplankton 0.593 |
| as shipped | 0.56 to 0.92, median 0.68; phytoplankton 0.592 |
| as shipped, with phytoplankton's maximum relative P/B at 5 | 0.82 to 0.97, median 0.89; phytoplankton 0.881 |

Why: in EwE 6's Ecospace, which PyPath follows, a producer's production in a cell is its habitat capacity there times the production it would have without it (`cSpaceSolver.vb` 1008, with the density term scaled by the base ratio, `cEcoSpace.vb` 6121). The production per biomass can rise to the group's maximum relative P/B (EwE's `PBmaxs`, 2 by default and 2 here) times its Ecopath P/B, so at capacity c a producer balancing its Ecopath mortality settles at

    B = B(capacity 1) x (c x PBmaxs - 1) / (PBmaxs - 1),   and at nothing where c < 1 / PBmaxs.

With PBmaxs 2 a producer cannot sustain itself where its capacity is below 0.5. The phytoplankton map follows depth and is divided by its largest value (EwE 6's rule), which puts 43 % of the lagoon's cells below 0.5. The formula alone gives a map total of 0.53 of the start; the run gives 0.59, as the grazers thin out too and dispersal feeds the poor cells (phytoplankton is gone from 4 % of the cells, not 43 %). That 0.59 is also the map's mean capacity, which is a coincidence: on three cells of capacity 1, 0.6 and 0.2 the formula gives 0.667 and the run 0.668, not the mean 0.6. Dispersal, the nutrient setting, the sub-steps and integrating the phytoplankton change none of it.

Two things follow:

- It is what EwE 6's equations give for these maps, checked against EwE's source. There is still no EwE 6 run of this model to compare numbers with. In the one case where PyPath's Ecospace is compared with EwE's own run (a Black Sea model), every group shares one map pattern, the grazers fall with the phytoplankton, and phytoplankton ends at 1.27 of its start in PyPath and 1.21 in EwE.
- The level depends on the producer's maximum relative P/B. With 5 in place of 2, the same three-year run ends with phytoplankton at 0.88 of its start and all groups between 0.82 and 0.97.

The maps were drawn for EwE-F, which rescales each map to a mean of 1, so that only its pattern matters. Under EwE 6's rule a map's level relative to its largest value matters as well, and a water-column group's capacity of 0.15 in the shallows says more than the map's author meant.

`ecosim/README.md` is EPath's note, copied as it is. Three of its statements describe EwE-F and do not hold for this set-up in PyPath:

- "Ecospace rescales each map to a mean of 1 over sea cells, so only the relative pattern matters": EwE-F does. Here a map is divided by its largest value, so its level matters, as above.
- "the sea means stay within 3 % of the non-spatial run": that is EwE-F's run, with its own movement and the advection field.
- "PyPath's own Ecosim demo on this model reports a zooplankton crash after year 12": not so for the files here. `example_ecosim_scenario("Curonian_Lagoon")` run for 50 years stays within 0.03 % of its start.

Its *Advection* section, and the files it lists that are not named above, concern EwE-F only.

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
