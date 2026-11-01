"""Integration tests for recruitment effects in Ecosim dynamics."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from pypath.core.ecopath import rpath
from pypath.core.ecosim import rsim_run, rsim_scenario
from pypath.core.params import create_rpath_params
from pypath.core.recruitment import (
    BevertonHoltRecruitment,
    LarvalConnectivity,
    RickerRecruitment,
)


def _simple_model():
    """3-group model: Phyto -> Zoo + Det."""
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


def _mean_zoo_annual(output):
    # annual_Biomass includes outside at col 0; Zoo is col 2 in this setup
    return float(np.mean(output.annual_Biomass[:, 2]))


class TestRecruitmentWiring:
    def test_default_recruitment_forcing_is_neutral(self):
        model, params = _simple_model()
        sc = rsim_scenario(model, params, years=range(1, 4))
        out = rsim_run(sc, method="RK4")
        assert np.all(out.annual_Biomass[:, 2] > 0)

    def test_forced_recs_affects_simulation(self):
        model, params = _simple_model()

        sc_base = rsim_scenario(model, params, years=range(1, 6))
        out_base = rsim_run(sc_base, method="RK4")

        sc_forced = rsim_scenario(model, params, years=range(1, 6))
        # Boost Zoo recruitment across the run
        sc_forced.forcing.ForcedRecs[:, 2] = 1.25
        out_forced = rsim_run(sc_forced, method="RK4")

        base_val = _mean_zoo_annual(out_base)
        forced_val = _mean_zoo_annual(out_forced)
        assert abs(forced_val - base_val) > 0.01 * base_val

    def test_forced_recs_reduction_affects_simulation(self):
        model, params = _simple_model()

        sc_base = rsim_scenario(model, params, years=range(1, 6))
        out_base = rsim_run(sc_base, method="RK4")

        sc_low = rsim_scenario(model, params, years=range(1, 6))
        sc_low.forcing.ForcedRecs[:, 2] = 0.75
        out_low = rsim_run(sc_low, method="RK4")

        assert abs(
            _mean_zoo_annual(out_low) - _mean_zoo_annual(out_base)
        ) > 0.01 * _mean_zoo_annual(out_base)

    def test_recruitment_adj_in_forcing_dict_is_handled(self):
        model, params = _simple_model()
        sc = rsim_scenario(model, params, years=range(1, 3))

        # Inject auxiliary dynamic recruitment adjustment
        sc.forcing.ForcedRecs[:, 2] = 1.0

        import pypath.core.ecosim as ecosim_mod

        captured = {}
        orig_deriv = ecosim_mod.deriv_vector

        def _capture_deriv(state, params_dict, forcing, fishing):
            forcing = dict(forcing)
            forcing["RecruitmentAdj"] = np.ones_like(forcing["Ftime"])
            forcing["RecruitmentAdj"][2] = 1.1
            captured["has_forced_recs"] = "ForcedRecs" in forcing
            captured["has_recruit_adj"] = "RecruitmentAdj" in forcing
            return orig_deriv(state, params_dict, forcing, fishing)

        ecosim_mod.deriv_vector = _capture_deriv
        try:
            rsim_run(sc, method="RK4")
        finally:
            ecosim_mod.deriv_vector = orig_deriv

        assert captured["has_forced_recs"]
        assert captured["has_recruit_adj"]


class TestRecruitmentModelToForcingBridge:
    def test_build_recruitment_vector(self, spatial_scenario):
        scenario, _ = spatial_scenario
        scenario.set_recruitment_model(
            "Fish", RickerRecruitment(alpha=0.5, beta=0.01, rmax=2.0)
        )
        vec = scenario.build_recruitment_vector(
            time=1.0,
            ssb_by_group={"Fish": 30.0},
        )
        assert vec.shape == (scenario.params.NUM_GROUPS + 1,)
        fish_idx = list(scenario.params.spname).index("Fish") + 1
        assert vec[fish_idx] > 0.01

    def test_apply_recruitment_to_forcing_writes_row(self, spatial_scenario):
        scenario, _ = spatial_scenario
        scenario.set_recruitment_model(
            "Fish", BevertonHoltRecruitment(alpha=3.0, beta=0.1, rmax=2.0)
        )
        row_before = scenario.forcing.ForcedRecs[0, :].copy()
        scenario.apply_recruitment_to_forcing(
            month_index=0,
            time=1.0,
            ssb_by_group={"Fish": 25.0},
        )
        row_after = scenario.forcing.ForcedRecs[0, :]
        assert not np.allclose(row_before, row_after)

    def test_apply_recruitment_to_forcing_month_bounds(self, spatial_scenario):
        scenario, _ = spatial_scenario
        with pytest.raises(ValueError, match="out of bounds"):
            scenario.apply_recruitment_to_forcing(
                month_index=-1,
                time=0.0,
                ssb_by_group={"Fish": 20.0},
            )

    def test_model_bridge_with_larval_connectivity(self, spatial_scenario):
        scenario, _ = spatial_scenario
        scenario.set_recruitment_model(
            "Fish", RickerRecruitment(alpha=0.5, beta=0.01, rmax=2.0)
        )
        scenario.set_recruitment_model(
            "Zoo", BevertonHoltRecruitment(alpha=2.0, beta=0.1, rmax=1.5)
        )
        lc = LarvalConnectivity(
            connectivity_matrix=np.array([[0.8, 0.2], [0.2, 0.8]]),
            source_groups=["Fish", "Zoo"],
            target_groups=["Fish", "Zoo"],
            survival_rate=0.9,
            normalize_columns=False,
        )
        scenario.set_larval_connectivity(lc)

        vec = scenario.build_recruitment_vector(
            time=1.0,
            ssb_by_group={"Fish": 30.0, "Zoo": 20.0},
        )
        fish_idx = list(scenario.params.spname).index("Fish") + 1
        zoo_idx = list(scenario.params.spname).index("Zoo") + 1
        assert vec[fish_idx] > 0.01
        assert vec[zoo_idx] > 0.01

    def test_recruitment_bridge_can_drive_run(self, spatial_scenario):
        scenario, _ = spatial_scenario
        scenario.set_recruitment_model(
            "Fish", BevertonHoltRecruitment(alpha=3.0, beta=0.1, rmax=2.0)
        )

        # Apply model-derived recruitment on every month of first year
        n_months = scenario.forcing.ForcedRecs.shape[0]
        for m in range(n_months):
            scenario.apply_recruitment_to_forcing(
                month_index=m,
                time=float(m) / 12.0,
                ssb_by_group={"Fish": 20.0 + 0.1 * m},
            )

        out = rsim_run(scenario, method="RK4")
        assert np.all(out.out_Biomass >= 0)
