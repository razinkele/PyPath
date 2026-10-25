"""Integration tests for metabolic scaling in Ecosim runtime wiring."""

from __future__ import annotations

import warnings

import numpy as np

from pypath.core.ecopath import rpath
from pypath.core.ecosim import rsim_run, rsim_scenario
from pypath.core.metabolism import MetabolicModel, TemperatureDependence
from pypath.core.params import create_rpath_params


def _simple_model():
    params = create_rpath_params(groups=["Phyto", "Zoo", "Det"], types=[1, 0, 2])
    params.model.loc[0, ["Biomass", "PB", "QB", "EE"]] = [100.0, 10.0, 0.0, 0.5]
    params.model.loc[1, ["Biomass", "PB", "QB", "EE"]] = [20.0, 5.0, 25.0, 0.8]
    params.model.loc[2, ["Biomass", "PB", "QB", "EE"]] = [10.0, 0.0, 0.0, 0.0]
    params.diet["Zoo"] = [1.0, 0.0, 0.0, 0.0]
    params.diet["Phyto"] = [0.0, 0.0, 0.0, 0.0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = rpath(params)
    return model, params


class TestMetabolismWiring:
    def test_forcing_dict_contains_forced_actresp(self):
        model, params = _simple_model()
        sc = rsim_scenario(model, params, years=range(1, 3))

        captured = {}
        import pypath.core.ecosim as ecosim_mod

        orig_deriv = ecosim_mod.deriv_vector

        def _capture_deriv(state, params_dict, forcing, fishing):
            captured["keys"] = set(forcing.keys())
            return orig_deriv(state, params_dict, forcing, fishing)

        ecosim_mod.deriv_vector = _capture_deriv
        try:
            rsim_run(sc, method="RK4")
        finally:
            ecosim_mod.deriv_vector = orig_deriv

        assert "ForcedActresp" in captured["keys"]

    def test_forced_actresp_changes_dynamics(self):
        model, params = _simple_model()

        base = rsim_scenario(model, params, years=range(1, 6))
        out_base = rsim_run(base, method="RK4")

        boosted = rsim_scenario(model, params, years=range(1, 6))
        boosted.forcing.ForcedActresp[:, 2] = 1.4  # Zoo metabolic burden up
        out_boosted = rsim_run(boosted, method="RK4")

        base_zoo = float(np.mean(out_base.annual_Biomass[:, 2]))
        boosted_zoo = float(np.mean(out_boosted.annual_Biomass[:, 2]))
        assert abs(base_zoo - boosted_zoo) > 0.005 * base_zoo

    def test_model_based_metabolic_scaling_changes_dynamics(self):
        model, params = _simple_model()

        baseline = rsim_scenario(model, params, years=range(1, 5))
        out_baseline = rsim_run(baseline, method="RK4")

        scenario = rsim_scenario(model, params, years=range(1, 5))
        temp_model = TemperatureDependence(q10_coefficient=2.0, reference_temperature=10.0)
        met_model = MetabolicModel(
            base_metabolism=0.2,
            mass_exponent=0.75,
            reference_mass=1.0,
            temperature_dependence=temp_model,
            activity_budget=1.0,
        )

        # Attach optional derivative inputs (params_dict is built from scenario.params attrs)
        n = scenario.params.NUM_GROUPS + 1
        scenario.params.MetabolismModel = met_model
        scenario.params.MetabolicTemperature = np.full(n, 10.0)
        scenario.params.MetabolicTemperature[2] = 18.0  # warmer Zoo
        scenario.params.MetabolicBodyMass = np.ones(n)
        scenario.params.MetabolicBodyMass[2] = 2.0

        out_scaled = rsim_run(scenario, method="RK4")

        baseline_zoo = float(np.mean(out_baseline.annual_Biomass[:, 2]))
        scaled_zoo = float(np.mean(out_scaled.annual_Biomass[:, 2]))
        assert abs(baseline_zoo - scaled_zoo) > 0.005 * baseline_zoo
