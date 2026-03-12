# pypath-ewe

Python implementation of Ecopath with Ecosim (EwE) for food web modeling.

**pypath-ewe** provides a complete Python implementation of the Rpath ecosystem modeling approach, including:

- **Ecopath** -- mass-balance parameterization of food web models
- **Ecosim** -- time-dynamic simulation with forcing functions and fishing scenarios
- **Ecospace** -- spatially explicit ecosystem modeling with habitat maps and dispersal
- **Time Series & Calibration** -- SS fitting against observed biomass, catch, effort data
- **Mediation Functions** -- trophic mediation via third-party group biomass
- **Monte Carlo / Pedigree** -- uncertainty analysis with pedigree-based sampling
- **Ecotracer** -- contaminant tracking coupled to Ecosim dynamics
- **Fleet Dynamics** -- profit-driven effort allocation and quota management
- **Ecological Indicators** -- ascendency, cycling index, transfer efficiency, system maturity
- **Stanza groups** -- multi-stanza (age-structured) population modeling
- **EwE I/O** -- read/write 72 of 84 EwE database tables (86% coverage)
- **EcoBase** -- import models from online databases
- **Species Data** -- WoRMS, OBIS, FishBase integration

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
