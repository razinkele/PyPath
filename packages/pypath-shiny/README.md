# pypath-shiny

Shiny web frontend for PyPath EwE food web modeling.

**pypath-shiny** provides an interactive web dashboard for building, running, and visualizing
Ecopath with Ecosim (EwE) ecosystem models using the [pypath-ewe](https://pypi.org/project/pypath-ewe/) core library.

## Features

- Interactive model parameter editing
- Ecosim time-dynamic simulations with real-time plots
- Diet matrix visualization
- Fishing scenario management
- Species data lookup via WoRMS/OBIS/FishBase

## Installation

```bash
pip install pypath-shiny
```

## Usage

```bash
pypath-shiny
```

Or from Python:

```python
from pypath_shiny.app import app

app.run()
```

## Documentation

Full documentation: <https://razinkele.github.io/PyPath/>

## License

MIT
