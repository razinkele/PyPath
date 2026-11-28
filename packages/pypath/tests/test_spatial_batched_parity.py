"""The batched all-patch Ecospace derivative must reproduce the per-patch
deriv_vector loop bit-for-bit (output biomass of whole spatial runs)."""

import numpy as np
import pytest

import pypath.spatial.integration as si
from pypath.spatial import EcospaceParams, create_regular_grid, rsim_run_spatial


def _run(scenario, ecospace, batched, monkeypatch):
    monkeypatch.setattr(si, "USE_BATCHED_DERIV", batched)
    out = rsim_run_spatial(scenario, ecospace=ecospace, years=range(1, 3))
    return np.asarray(out.out_Biomass_spatial)


@pytest.mark.parametrize("heterogeneous", [False, True])
def test_batched_matches_per_patch(spatial_scenario, monkeypatch, heterogeneous):
    scenario, _ = spatial_scenario
    ng = scenario.params.NUM_GROUPS + 1
    grid = create_regular_grid(bounds=(0, 0, 5, 4), nx=5, ny=4)
    n = grid.n_patches
    rng = np.random.default_rng(7)
    pref = rng.uniform(0.2, 1.0, (ng, n)) if heterogeneous else np.ones((ng, n))
    ecospace = EcospaceParams(
        grid=grid,
        habitat_preference=pref,
        habitat_capacity=np.ones((ng, n)),
        dispersal_rate=np.full(ng, 2.0),
        advection_enabled=np.zeros(ng, dtype=bool),
        gravity_strength=np.zeros(ng),
    )
    a = _run(scenario, ecospace, True, monkeypatch)
    b = _run(scenario, ecospace, False, monkeypatch)
    np.testing.assert_array_equal(a, b)


def test_batched_path_is_taken(spatial_scenario, monkeypatch):
    """Guard against silently falling back for the standard feature set."""
    scenario, _ = spatial_scenario
    calls = {"n": 0}
    for name in ("deriv_batched", "deriv_spatial_fast"):
        real = getattr(si, name)

        def spy(*args, _real=real, **kwargs):
            calls["n"] += 1
            return _real(*args, **kwargs)

        monkeypatch.setattr(si, name, spy)
    ng = scenario.params.NUM_GROUPS + 1
    grid = create_regular_grid(bounds=(0, 0, 3, 3), nx=3, ny=3)
    ecospace = EcospaceParams(
        grid=grid,
        habitat_preference=np.ones((ng, grid.n_patches)),
        habitat_capacity=np.ones((ng, grid.n_patches)),
        dispersal_rate=np.full(ng, 1.0),
        advection_enabled=np.zeros(ng, dtype=bool),
        gravity_strength=np.zeros(ng),
    )
    rsim_run_spatial(scenario, ecospace=ecospace, years=range(1, 2))
    assert calls["n"] > 0
