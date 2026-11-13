# Central Baltic Sea, 1974

Food web of the Baltic Proper (central Baltic Sea, 240 000 km²) in 1974: 22 groups, wet weight in t/km², with sprat, herring and cod as multi-stanza sets. It is the EwE 6 model *Central Baltic Sea* (EcoBase model 608) of

- Tomczak, M. T., Niiranen, S., Hjerne, O., & Blenckner, T. (2012). Ecosystem flow dynamics in the Baltic Proper—Using a multi-trophic dataset as a basis for food–web modelling. *Ecological Modelling*, 230, 123–147. https://doi.org/10.1016/j.ecolmodel.2011.12.014

**modified for a stationary baseline**, so its results are not those of the published model: each stanza set's biomass accumulation rate is set to 0 and added to the P/B of its stages; one stage name is completed and two diet columns are rescaled to 1 (`ecopath/README.md`).

## Files

- `ecopath/`: EwE-F Ecopath inputs: `*_Scenario.csv` (basic input), `*_DC.csv` (diet, Import row last), `*_DetFate.csv` (detritus fate), `*_GrowthParameters.csv` (one row per stanza set, in set order), and `*_filenames.nml`. `ecopath/README.md` is EPath's note on the model and every change made to the source file.
- `ecosim/`: EwE-F Ecosim settings: `*_Scenario.csv` (group settings), `*_vul.csv` (vulnerabilities), `*_F.csv` (fishing forcing) and `*_filenames.nml`. `ecosim/README.md` is EPath's note; the files it lists that are not here (Ecopath-F results) are not shipped.

Copied from EPath (EwE-F) test cases `ecopath-f/testcases/Central_Baltic` and `ecosim-f/testcases/Central_Baltic` at EPath commit 8029f8d (13 November 2026). The namelists' paths are EPath's; PyPath finds the files beside them.

## Use

```python
from pypath.examples import load_example_model, example_ecosim_scenario
from pypath.core.ecopath import rpath
from pypath.core.ecosim import rsim_run

params = load_example_model("Central_Baltic")
balanced = rpath(params)
scenario = example_ecosim_scenario("Central_Baltic", years=range(1, 11))
result = rsim_run(scenario, method="RK4", years=range(1, 11))
```

PyPath's balance matches Ecopath-F's solution of this model (biomass and EE of every group; `tests/test_ewef_example_models.py`).
