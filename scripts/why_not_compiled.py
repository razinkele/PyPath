"""Say whether rsim_run's compiled RK4 path runs a database scenario, and why not.

Usage: python scripts/why_not_compiled.py DATABASE SCENARIO
"""

import sys
import warnings

warnings.filterwarnings("ignore")
from pypath.core import ecosim as E
from pypath.core import ecosim_fast as F
from pypath.io.ewemdb import ecosim_scenario_from_ewemdb

sc = ecosim_scenario_from_ewemdb(sys.argv[1], scenario=int(sys.argv[2]))
setup = E._run_setup(sc, None)
print("numba / enabled:", F.HAS_NUMBA, F.ENABLED)
print("hooks intact:", F.python_hooks_intact(E))
print(
    "step:", F.unsupported_reason(setup.params_dict, setup.fishing_dict) or "supported"
)
fast = F.FastRK4(setup.params_dict, setup.fishing_dict)
try:
    F.CompiledRun(sc, setup, fast, 1, 1 / 12, (None,) * 7)
    print("loop: supported")
except F._Unsupported as e:
    print("loop: not supported:", e)
except (AttributeError, TypeError) as e:  # no output arrays given: past the checks
    print("loop: passed the checks", type(e).__name__)
for k in ("ForcedSearchLink", "ForcedVulLink"):
    print(k, getattr(sc.forcing, k, None) is not None)
