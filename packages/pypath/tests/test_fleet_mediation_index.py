"""Fleet mediation: ``MediationLink.fleet_idx`` is 0-based (first gear = 0).

``compute_fleet_multipliers`` returns ``NUM_GEARS`` values indexed by that
0-based fleet; ``fishing_mortality`` must read gear ``g`` (1-based, the
``ForcedEffort`` column) at ``fleet_med[g - 1]``. It used to read
``fleet_med[g]``: the multiplier of the wrong gear, and an IndexError at the
last one.
"""

import numpy as np
import pytest

from pypath.core.ecosim import rsim_run
from pypath.core.ecosim_deriv import fishing_mortality
from pypath.core.mediation import MediationCollection, MediationLink, MediationShape
from tests.continuous_support import needs_reco, run_dicts, scenario
from tests.continuous_support import reco_plain as _reco_plain

reco_plain = pytest.fixture(_reco_plain)


def _halving(fleet_idx):
    """Effort x0.5 for the 0-based fleet, whatever the mediator's biomass."""
    shape = MediationShape(1, "half", np.array([0.0, 10.0]), np.array([0.5, 0.5]))
    return MediationCollection(
        shapes=[shape], links=[MediationLink(1, 0, fleet_idx=fleet_idx)]
    )


def _gear_rates(scen, mediation):
    """Per-gear sum of the link catch rates (1-based gear number -> rate)."""
    state, params, forcing, fishing = run_dicts(scen)
    n_gears = params["NUM_GEARS"]
    fleet_med = None
    if mediation is not None:
        fleet_med = mediation.compute_fleet_multipliers(state, params["Bbase"], n_gears)
    _, link_rate = fishing_mortality(params, forcing, fishing, fleet_med=fleet_med)
    pos = {int(g): p + 1 for p, g in enumerate(params["fleet_idx"])}
    out = np.zeros(n_gears + 1)
    for i in range(1, len(fishing["FishFrom"])):
        out[pos[int(fishing["FishThrough"][i]) - 1]] += link_rate[i]
    return out, n_gears


@needs_reco
@pytest.mark.parametrize("which", ["first", "middle", "last"])
def test_the_multiplier_lands_on_its_gear_only(reco_plain, which):
    scen = scenario(reco_plain)
    base, n_gears = _gear_rates(scen, None)
    assert n_gears >= 2
    fleet = {"first": 0, "middle": n_gears // 2, "last": n_gears - 1}[which]
    got, _ = _gear_rates(scen, _halving(fleet))
    gear = fleet + 1  # ForcedEffort column of 0-based fleet
    assert base[gear] > 0
    assert got[gear] == pytest.approx(0.5 * base[gear])
    others = [g for g in range(1, n_gears + 1) if g != gear]
    np.testing.assert_allclose(got[others], base[others])


@needs_reco
def test_a_run_with_the_last_gear_mediated_completes_and_cuts_its_catch(reco_plain):
    scen = scenario(reco_plain)
    n_gears = scen.params.NUM_GEARS
    plain = rsim_run(scen, years=range(1, 2))
    med = rsim_run(scen, years=range(1, 2), mediation=_halving(n_gears - 1))
    assert np.all(np.isfinite(med.annual_Catch))
    assert med.annual_Catch.sum() < plain.annual_Catch.sum()
