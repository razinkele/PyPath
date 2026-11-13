# Lithuanian sandy coast

Food web of the exposed sandy coast of Lithuania (south-eastern Baltic Sea, the 0–20 m zone from Nida to Būtingė): 44 groups, wet weight in g/m² (= t/km²). It is the EwE 6 model *coast* (2017) of

- Razinkovas-Baziukas, A., Morkūnė, R., Bacevičius, E., & Gasiūnaitė, Z. R. (2017). Trophic network model of exposed sandy coast: Linking continental and marine water ecosystems. *Estuarine, Coastal and Shelf Science*, 195, 110–123. https://doi.org/10.1016/j.ecss.2017.04.030

with round goby added as a three-stage multi-stanza set, from Skabeikis, A. (2019), *Role of the round goby (Neogobius melanostomus) in the food web of the south-eastern Baltic Sea coastal waters*, doctoral dissertation, Klaipėda University. It is a working version; the changes to the file are listed in `ecopath/README.md`. Seventeen groups have immigration; for two of them (three-spined stickleback and juvenile European smelt) it exceeds what the food web takes, so their EE is negative, as in Ecopath-F.

## Files

- `ecopath/`: EwE-F Ecopath inputs: `*_Scenario.csv` (basic input), `*_DC.csv` (diet, Import row last), `*_DetFate.csv` (detritus fate), `*_GrowthParameters.csv` (one row per stanza set, in set order), and `*_filenames.nml`. `ecopath/README.md` is EPath's note on the model and every change made to the source file.
- `ecosim/`: EwE-F Ecosim settings: `*_Scenario.csv` (group settings), `*_vul.csv` (vulnerabilities), `*_F.csv` (fishing forcing) and `*_filenames.nml`. `ecosim/README.md` is EPath's note; the files it lists that are not here (Ecopath-F results) are not shipped.

Copied from EPath (EwE-F) test cases `ecopath-f/testcases/Lithuanian_Coast` and `ecosim-f/testcases/Lithuanian_Coast` at EPath commit 8029f8d (13 November 2026). The namelists' paths are EPath's; PyPath finds the files beside them.

## Use

```python
from pypath.examples import load_example_model, example_ecosim_scenario
from pypath.core.ecopath import rpath
from pypath.core.ecosim import rsim_run

params = load_example_model("Lithuanian_Coast")
balanced = rpath(params)
scenario = example_ecosim_scenario("Lithuanian_Coast", years=range(1, 11))
result = rsim_run(scenario, method="RK4", years=range(1, 11))
```

PyPath's balance matches Ecopath-F's solution of this model (biomass and EE of every group; `tests/test_ewef_example_models.py`).
