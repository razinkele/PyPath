# Example Models

PyPath ships three real food-web models, from EPath's EwE-F test cases, ready
to balance and run. In the app, pick one from **Open a real model…** on the
Home page; its EwE-F Ecosim settings then become the starting scenario on the
Ecosim page.

| Name | Model | Groups | Source |
|---|---|---|---|
| `Central_Baltic` | Baltic Proper, 1974, modified for a stationary baseline | 22 | Tomczak et al. (2012), *Ecological Modelling* 230, 123–147 |
| `Curonian_Lagoon` | Curonian Lagoon pelagic food web, 2005–2007 | 19 | Marine Research Institute, Klaipėda University |
| `Lithuanian_Coast` | Exposed sandy coast of Lithuania, with round goby | 44 | Razinkovas-Baziukas et al. (2017), *Estuarine, Coastal and Shelf Science* 195, 110–123 |

Each model's folder (`pypath/examples/data/<name>/`) has a README with its
source, the changes made to it and the files included: the EwE-F Ecopath
inputs and the Ecosim settings (vulnerabilities, group settings, fishing
forcing). PyPath's balance of each matches Ecopath-F's.

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

## API

::: pypath.examples
    options:
      show_root_heading: true
