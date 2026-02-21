# PyPath EwE

Python implementation of Ecopath with Ecosim (EwE) for food web modeling.

## Installation

```bash
pip install pypath-ewe
```

### Optional dependencies

```bash
pip install pypath-ewe[spatial]      # Ecospace spatial modeling
pip install pypath-ewe[interactive]  # Plotly interactive plots
pip install pypath-ewe[biodata]      # Species data from WoRMS/OBIS
pip install pypath-ewe[all]          # Everything
```

## Quick Example

```python
from pypath import create_rpath_params, rpath, rsim_scenario, rsim_run

# Create a simple 3-group model
params = create_rpath_params(
    groups=["Phytoplankton", "Zooplankton", "Detritus"],
    types=[1, 0, 2],
)
# ... set biomass, PB, QB, diet matrix ...

# Balance the model
model = rpath(params)

# Run dynamic simulation
scenario = rsim_scenario(model, params, years=range(1, 51))
output = rsim_run(scenario)
```
