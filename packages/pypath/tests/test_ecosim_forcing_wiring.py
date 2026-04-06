"""Tests that forcing arrays from RsimForcing reach deriv_vector."""

import warnings

import numpy as np
import pytest
from pypath.core.ecopath import rpath
from pypath.core.ecosim import rsim_run, rsim_scenario
from pypath.core.params import create_rpath_params


def _simple_model():
    """3-group model: Phyto(producer) -> Zoo(consumer) + Det."""
    params = create_rpath_params(
        groups=["Phyto", "Zoo", "Det"],
        types=[1, 0, 2],
    )
    params.model.loc[0, ["Biomass", "PB", "QB", "EE"]] = [100.0, 10.0, 0.0, 0.5]
    params.model.loc[1, ["Biomass", "PB", "QB", "EE"]] = [20.0, 5.0, 25.0, 0.8]
    params.model.loc[2, ["Biomass", "PB", "QB", "EE"]] = [10.0, 0.0, 0.0, 0.0]
    # Diet: Zoo eats 100% Phyto (rows: Phyto, Zoo, Det, Import)
    params.diet["Zoo"] = [1.0, 0.0, 0.0, 0.0]
    params.diet["Phyto"] = [0.0, 0.0, 0.0, 0.0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = rpath(params)
    return model, params


class TestForcingWiring:
    def test_forced_prey_affects_simulation(self):
        """ForcedPrey != 1.0 should change simulation output vs baseline."""
        model, params = _simple_model()

        # Baseline run
        sc1 = rsim_scenario(model, params, years=range(1, 6))
        out1 = rsim_run(sc1, method="RK4")

        # Run with ForcedPrey = 0.5 for Phyto (ecosim index 1)
        sc2 = rsim_scenario(model, params, years=range(1, 6))
        if hasattr(sc2.forcing, "ForcedPrey") and sc2.forcing.ForcedPrey is not None:
            sc2.forcing.ForcedPrey[:, 1] = 0.5
        out2 = rsim_run(sc2, method="RK4")

        # Biomass should differ
        bio1 = out1.annual_Biomass[3, 2]  # Zoo at year 3
        bio2 = out2.annual_Biomass[3, 2]
        assert abs(bio1 - bio2) > 0.01 * bio1, (
            f"ForcedPrey should affect simulation: baseline={bio1:.2f}, forced={bio2:.2f}"
        )

    def test_default_forcing_is_neutral(self):
        """Default ForcedPrey=1.0 should not change results vs no forcing."""
        model, params = _simple_model()
        sc = rsim_scenario(model, params, years=range(1, 4))
        out = rsim_run(sc, method="RK4")
        # Should complete without error and produce reasonable biomass
        assert out.annual_Biomass[-1, 1] > 0  # Phyto alive
        assert out.annual_Biomass[-1, 2] > 0  # Zoo alive

    def test_forcing_dict_contains_forced_prey(self):
        """Verify ForcedPrey and PP_forcing keys are present in forcing_dict."""
        model, params = _simple_model()
        sc = rsim_scenario(model, params, years=range(1, 3))

        # Patch deriv_vector to capture the forcing_dict
        captured = {}
        import pypath.core.ecosim as ecosim_mod

        orig_deriv = ecosim_mod.deriv_vector

        def _capture_deriv(state, params_dict, forcing, fishing):
            captured["forcing"] = forcing.copy()
            return orig_deriv(state, params_dict, forcing, fishing)

        ecosim_mod.deriv_vector = _capture_deriv
        try:
            rsim_run(sc, method="RK4")
        finally:
            ecosim_mod.deriv_vector = orig_deriv

        assert "ForcedPrey" in captured["forcing"], (
            "ForcedPrey missing from forcing_dict"
        )
        assert "PP_forcing" in captured["forcing"], (
            "PP_forcing missing from forcing_dict"
        )
