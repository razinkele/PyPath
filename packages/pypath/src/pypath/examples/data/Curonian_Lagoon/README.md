# Curonian Lagoon pelagic food web, 2005–2007

Ecopath mass balance of the Curonian Lagoon pelagic food web: 19 groups (17 consumers, phytoplankton and detritus) and one fleet. EwE 6 model *curonian pelagic*, Marine Research Institute, Klaipėda University; no peer-reviewed reference is attached to it. It is the model of PyPath's Curonian Lagoon case study.

Meiobenthos has both B and EE in the database; its B is left for Ecopath to estimate from the EE of 0.95 (`ecopath/README.md`).

## Files

- `ecopath/`: EwE-F Ecopath inputs: `*_Scenario.csv` (basic input), `*_DC.csv` (diet, Import row last), `*_DetFate.csv` (detritus fate), and `*_filenames.nml`. `ecopath/README.md` is EPath's note on the model and every change made to the source file.
- `ecosim/`: EwE-F Ecosim settings: `*_Scenario.csv` (group settings), `*_vul.csv` (vulnerabilities), `*_F.csv` (fishing forcing) and `*_filenames.nml`. `ecosim/README.md` is EPath's note; the files it lists that are not here (Ecopath-F results, the EwE database and the Ecospace grid, maps and spatial inputs) are not shipped.

- `ecospace/curonian29.grd`: a SHYFEM mesh of the lagoon, the Klaipėda strait and the sea off the coast (1,309 nodes, 2,027 elements, 2,296 km²). Its element types are the 29 boxes of a hydrodynamic box model of the lagoon, so `example_ecospace_grid("Curonian_Lagoon")` gives a 29-patch Ecospace grid that the box model's output (`boxes_2d.txt`) fits as it is. Only the grid is here: no habitat maps, dispersal rates or model output. By the mesh's own depths, 21 boxes are 1.4 to 4.5 m deep on average, box 11 is 7.1 m, and boxes 1, 4, 7, 10, 12, 13 and 16 are over 10 m.

The Ecopath and Ecosim files are copied from EPath (EwE-F) test cases `ecopath-f/testcases/Curonian_Lagoon` and `ecosim-f/testcases/Curonian_Lagoon` at EPath commit 8029f8d (13 November 2026). The namelists' paths are EPath's; PyPath finds the files beside them.

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

On the 29 boxes (habitat and movement are yours to set):

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
