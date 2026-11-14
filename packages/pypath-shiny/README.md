# pypath-shiny

Web interface to [PyPath](https://github.com/razinkele/PyPath), the Python
implementation of Ecopath with Ecosim (EwE). Use it in a browser to build,
balance, run and analyse aquatic food-web models, without writing code.

The app is built with [Shiny for Python](https://shiny.posit.co/py/) on top of
the [pypath-ewe](https://github.com/razinkele/PyPath/blob/main/packages/pypath/README.md) library, which holds the algorithms.

## Features

The left-hand menu is organised as follows.

**Getting started**

- **Home**: load one of the bundled example models, or build the small
  default model.
- **Tutorial**: a walk-through of the workflow.
- **Data Import**: read an EwE 6 database (`.ewemdb`, `.eweaccdb`, `.mdb`,
  `.accdb`), download a model from EcoBase, or build a model from WoRMS, OBIS
  and FishBase records on the Biodiversity tab.

**Core workflow**

- **Ecopath Model**: edit basic parameters, diet matrix, fisheries and
  multi-stanza groups; balance the model and inspect the results and
  diagnostics.
- **Pre-Balance**: diagnostics run before balancing (biomasses, vital rates,
  predator-prey relationships), based on the Prebal routine.
- **Ecosim**: set up and run time-dynamic scenarios, with a choice of
  Runge-Kutta 4 (RK4) or Adams-Bashforth integration. View time series, catch
  and summaries. After importing an EwE database, run its own Ecosim scenarios.

**Advanced**

- **Ecospace** and **Ecospace Wizard**: spatial modelling with grids, habitat
  preferences and capacity, dispersal, fishing effort and biomass animation.
  The wizard builds an Ecospace model in seven steps, from drawing the study
  area on a map to downloading EMODnet habitat and bathymetry data.
- **Multi-Stanza**: age-structured groups with von Bertalanffy growth.
- **Individual-Based Model**: couple super-individuals to Ecosim for a chosen
  group and compare standard and IBM runs.
- **Forcing**, **Diet Rewiring** and **Optimization**: interactive demo pages
  for those features.

**Output**

- **Analysis**: network indices, trophic structure, mixed trophic impacts,
  keystoneness and balance checks.
- **Results**: model summary, food-web diagram, simulation results, scenario
  comparison and data tables. Download model and simulation results as CSV.

**Help**

- **About** and **Feedback**: the Feedback page has a dashboard for reports,
  and **Send Feedback** in the menu opens a report form from any page.

## Requirements

Python 3.10 or later and pypath-ewe 1.6.0 or later. Ecosim and Ecospace follow
EwE 6.7.144 closely. The
[known differences](https://github.com/razinkele/PyPath/blob/main/docs/parity/deviations.md)
are listed in the repository.

## Installation

With conda, from the `razinka` channel:

```bash
conda install -c razinka -c conda-forge pypath-shiny
```

From source, in a clone of <https://github.com/razinkele/PyPath>:

```bash
pip install -e packages/pypath
pip install -e packages/pypath-shiny
```

The packages on PyPI are out of date for now, so install with conda or from
source.

Two features need more than the Python packages:

- **EwE database import** reads Microsoft Access files, so it needs an Access
  backend: [mdbtools](https://github.com/mdbtools/mdbtools) on Linux or macOS
  (`sudo apt install mdbtools`, `brew install mdbtools`), or `pyodbc` with the
  Microsoft Access ODBC driver on Windows.
- **The Biodiversity tab** (WoRMS, OBIS and FishBase lookups) needs `pyworms`
  and `pyobis`. From source, install the extra with
  `pip install -e "packages/pypath-shiny[biodata]"`; with conda, add
  `pip install pyworms pyobis`.

## Running

```bash
pypath-shiny
```

The server listens on all network interfaces (`0.0.0.0`), port 8000, so open
<http://localhost:8000>. The command takes no options. To use another host or
port, start the same ASGI app with uvicorn:

```bash
python -m uvicorn pypath_shiny.app:app --host 127.0.0.1 --port 8080
```

Set the environment variable `PYPATH_DEV=1` before `pypath-shiny` to reload
the server when source files change.

## Documentation and licence

Documentation: <https://razinkele.github.io/PyPath/>

Licence: MIT.
