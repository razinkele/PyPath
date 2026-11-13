# Curonian Lagoon pelagic food web, 2005–2007

Ecopath mass balance of the Curonian Lagoon pelagic food web: 19 groups (17 consumers, phytoplankton and detritus) and one fleet. EwE 6 model *curonian pelagic*, Marine Research Institute, Klaipėda University; no peer-reviewed reference is attached to it. It is the model of PyPath's Curonian Lagoon case study.

Meiobenthos has both B and EE in the database; its B is left for Ecopath to estimate from the EE of 0.95 (`ecopath/README.md`).

## Files

- `ecopath/`: EwE-F Ecopath inputs: `*_Scenario.csv` (basic input), `*_DC.csv` (diet, Import row last), `*_DetFate.csv` (detritus fate), and `*_filenames.nml`. `ecopath/README.md` is EPath's note on the model and every change made to the source file.
- `ecosim/`: EwE-F Ecosim settings: `*_Scenario.csv` (group settings), `*_vul.csv` (vulnerabilities), `*_F.csv` (fishing forcing) and `*_filenames.nml`. `ecosim/README.md` is EPath's note; the files it lists that are not here (Ecopath-F results, the EwE database and the Ecospace grid, maps and spatial inputs) are not shipped.

Copied from EPath (EwE-F) test cases `ecopath-f/testcases/Curonian_Lagoon` and `ecosim-f/testcases/Curonian_Lagoon` at EPath commit 8029f8d (13 November 2026). The namelists' paths are EPath's; PyPath finds the files beside them.

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

PyPath's balance matches Ecopath-F's solution of this model (biomass and EE of every group; `tests/test_ewef_example_models.py`).
