"""``import pypath`` does not load what a plain run never uses.

matplotlib, networkx and scipy.stats took over half of the package's import
time, and every script, worker process and app start paid it. They are
loaded when a plot is drawn or a lognormal is sampled, not before.

Each check runs in a new interpreter: what is loaded there is what a user's
``import pypath`` loads.
"""

import json
import subprocess
import sys

import pytest

HEAVY = ["matplotlib", "matplotlib.pyplot", "networkx", "scipy.stats"]


def _loaded(statements: str) -> list:
    """The heavy modules present after ``statements`` in a new interpreter."""
    program = (
        "import sys, json\n"
        f"{statements}\n"
        f"print(json.dumps([m for m in {HEAVY!r} if m in sys.modules]))\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True
    )
    assert done.returncode == 0, done.stderr[-800:]
    return json.loads(done.stdout.strip().splitlines()[-1])


def test_importing_the_package_loads_no_plotting_or_statistics_library():
    assert _loaded("import pypath") == []
    assert _loaded("import pypath.core.plotting, pypath.core.sensitivity") == []


def test_a_run_from_files_loads_none_of_them_either():
    program = """
import warnings; warnings.simplefilter("ignore")
from pypath.examples import example_ecosim_scenario
from pypath.core.ecosim import rsim_run
out = rsim_run(example_ecosim_scenario("Curonian_Lagoon", years=range(1, 3)), years=range(1, 3))
assert out.out_Biomass.shape[0] == 25
"""
    assert _loaded(program) == []


def test_a_plot_loads_matplotlib_when_it_is_drawn():
    pytest.importorskip("matplotlib")
    program = """
import warnings; warnings.simplefilter("ignore")
import matplotlib; matplotlib.use("Agg")
import pypath, sys
from pypath.core.plotting import plot_biomass
from pypath.examples import example_ecosim_scenario
from pypath.core.ecosim import rsim_run
out = rsim_run(example_ecosim_scenario("Curonian_Lagoon", years=range(1, 3)), years=range(1, 3))
assert "matplotlib.pyplot" not in sys.modules
figure = plot_biomass(out)
assert "matplotlib.pyplot" in sys.modules and figure is not None
"""
    assert "matplotlib.pyplot" in _loaded(program)


def test_a_sensitivity_analysis_loads_scipy_stats_when_it_samples():
    program = """
import warnings, sys; warnings.simplefilter("ignore")
from pypath.examples import load_example_model
from pypath.core.sensitivity import SensitivityConfig, run_sensitivity
assert "scipy.stats" not in sys.modules
params = load_example_model("Curonian_Lagoon")
result = run_sensitivity(params, SensitivityConfig(method="morris", n_trajectories=2, seed=1))
assert "scipy.stats" in sys.modules and result is not None
"""
    assert "scipy.stats" in _loaded(program)


def test_every_public_name_is_still_there():
    program = """
import pypath, pypath.core
missing = [n for n in pypath.__all__ if not hasattr(pypath, n)]
missing += ["core." + n for n in getattr(pypath.core, "__all__", []) if not hasattr(pypath.core, n)]
assert not missing, missing
from pypath.core.plotting import HAS_NETWORKX, HAS_PLOTLY
assert isinstance(HAS_NETWORKX, bool) and isinstance(HAS_PLOTLY, bool)
"""
    _loaded(program)
