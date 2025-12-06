# PyPath - Python Ecopath with Ecosim

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**PyPath** is a Python implementation of the Ecopath with Ecosim (EwE) ecosystem modeling approach. It is a port of the R package [Rpath](https://github.com/NOAA-EDAB/Rpath) developed by NOAA-EDAB.

## Features

- **Ecopath**: Mass-balance food web modeling
- **Ecosim**: Dynamic ecosystem simulation
- **Multi-stanza groups**: Age-structured populations
- **Sensitivity analysis**: Monte Carlo parameter uncertainty
- **Visualization**: Food web diagrams and time series plots

## Installation

```bash
pip install pypath-ecopath
```

Or install from source:

```bash
git clone https://github.com/pypath-ecopath/pypath.git
cd pypath
pip install -e ".[all]"
```

## Quick Start

```python
import pypath as pp

# Create model parameters
params = pp.create_rpath_params(
    groups=['Phytoplankton', 'Zooplankton', 'Fish', 'Detritus', 'Fleet'],
    types=[1, 0, 0, 2, 3]  # 1=producer, 0=consumer, 2=detritus, 3=fleet
)

# Or read from CSV files
params = pp.read_rpath_params(
    model_file='my_model.csv',
    diet_file='my_diet.csv'
)

# Balance the model
model = pp.rpath(params, eco_name='My Ecosystem')
print(model)

# Create and run simulation
scenario = pp.rsim_scenario(model, params, years=range(1, 101))
output = pp.rsim_run(scenario, method='RK4')

# Visualize results
pp.plot_biomass(output, groups=['Fish', 'Zooplankton'])
```

## Documentation

Full documentation is available at [https://pypath-ecopath.readthedocs.io](https://pypath-ecopath.readthedocs.io).

## Scientific Background

PyPath implements the Ecopath with Ecosim approach:

- **Ecopath**: Solves the mass-balance equation for food webs (Polovina, 1984; Christensen & Walters, 2004)
- **Ecosim**: Dynamic simulation using foraging arena theory (Walters et al., 2000)

### Key References

- Lucey, S. M., Gaichas, S. K., & Aydin, K. Y. (2020). Conducting reproducible ecosystem modeling using the open source mass balance model Rpath. *Ecological Modelling*, 427, 109057.
- Christensen, V., & Walters, C. J. (2004). Ecopath with Ecosim: Methods, capabilities and limitations. *Ecological Modelling*, 172(2), 109-139.

## Contributing

Contributions are welcome! Please read our [contributing guidelines](CONTRIBUTING.md) before submitting a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Original Rpath R package: [NOAA-EDAB/Rpath](https://github.com/NOAA-EDAB/Rpath)
- Ecopath with Ecosim: [www.ecopath.org](http://www.ecopath.org)
