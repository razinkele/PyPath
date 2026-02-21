# pypath-ewe

Python implementation of Ecopath with Ecosim (EwE) for food web modeling.

**pypath-ewe** provides a complete Python implementation of the Rpath ecosystem modeling approach, including:

- **Ecopath** -- mass-balance parameterization of food web models
- **Ecosim** -- time-dynamic simulation with forcing functions and fishing scenarios
- **Ecospace** -- spatially explicit ecosystem modeling with habitat maps and dispersal
- **Stanza groups** -- multi-stanza (age-structured) population modeling
- **EcoBase / EwE database** -- import models from online databases

## Installation

```bash
pip install pypath-ewe
```

With optional dependencies:

```bash
pip install pypath-ewe[interactive]   # plotly, networkx
pip install pypath-ewe[spatial]       # geopandas, folium, shapely
pip install pypath-ewe[biodata]       # pyworms, pyobis for species data
pip install pypath-ewe[all]           # everything
```

## Quick Start

```python
from pypath import rpath, rsim_run, create_rpath_params

# Create a simple model
params = create_rpath_params(groups=["Phytoplankton", "Zooplankton", "Fish"], ...)
model = rpath(params)
output = rsim_run(model, years=50)
```

## Documentation

Full documentation: <https://razinkele.github.io/PyPath/>

## License

MIT
