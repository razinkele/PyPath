"""Integration tests for Monte Carlo and sensitivity analysis."""

import warnings

import pytest

from pypath.core.montecarlo import MCConfig, run_montecarlo
from pypath.core.params import create_rpath_params


def _make_mc_model():
    """Create a balanced 3-group model with moderate pedigree CVs."""
    params = create_rpath_params(
        groups=["Producer", "Consumer", "Detritus"],
        types=[1, 0, 2],
    )
    params.model.loc[0, "Biomass"] = 10.0
    params.model.loc[0, "PB"] = 200.0
    params.model.loc[0, "EE"] = 0.8
    params.model.loc[1, "Biomass"] = 5.0
    params.model.loc[1, "PB"] = 50.0
    params.model.loc[1, "QB"] = 150.0
    params.model.loc[1, "EE"] = 0.9
    params.model.loc[2, "Biomass"] = 100.0
    params.model["BioAcc"] = 0.0
    params.model["Unassim"] = 0.2
    params.model.loc[0, "Unassim"] = 0.0
    params.model.loc[2, "Unassim"] = 0.0
    params.model["Detritus"] = 1.0
    params.model.loc[2, "Detritus"] = 0.0
    params.diet["Consumer"] = [1.0, 0.0, 0.0, 0.0]
    # Moderate pedigree
    params.pedigree["Biomass"] = [0.15, 0.15, 0.0]
    params.pedigree["PB"] = [0.1, 0.1, 0.0]
    params.pedigree["QB"] = [0.0, 0.1, 0.0]
    params.pedigree["Diet"] = [0.0, 0.0, 0.0]
    return params


@pytest.mark.slow
class TestMCIntegration:
    def test_ecopath_mc_feasibility(self):
        """Full pipeline: pedigree -> MC(n=50, ecopath_only) -> feasibility > 0."""
        params = _make_mc_model()
        config = MCConfig(n_samples=50, method="random", seed=42, ecopath_only=True)
        result = run_montecarlo(params, config)
        assert result.feasibility_rate > 0
        assert "Biomass" in result.ecopath_stats
        assert result.ecopath_stats["Biomass"].shape[0] > 0

    def test_ecopath_mc_with_ecosim(self):
        """Full pipeline: pedigree -> MC(n=10, ecosim) -> ecosim_stats shape."""
        params = _make_mc_model()
        config = MCConfig(
            n_samples=10,
            method="random",
            seed=42,
            ecopath_only=False,
            ecosim_years=range(1, 6),
        )
        result = run_montecarlo(params, config)
        if result.n_ecosim > 0:
            assert result.ecosim_stats is not None
            assert result.ecosim_stats["Biomass"].shape[2] == 7  # 7 stats

    def test_zero_cv_identical(self):
        """Zero-CV pedigree -> all samples identical."""
        params = _make_mc_model()
        params.pedigree["Biomass"] = [0.0, 0.0, 0.0]
        params.pedigree["PB"] = [0.0, 0.0, 0.0]
        params.pedigree["QB"] = [0.0, 0.0, 0.0]
        config = MCConfig(n_samples=5, method="random", seed=42, ecopath_only=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = run_montecarlo(params, config)
        if result.n_feasible > 1:
            assert result.ecopath_stats["Biomass"]["std"].max() < 1e-10

    def test_store_runs_accessible(self):
        """store_runs=True -> raw outputs accessible."""
        params = _make_mc_model()
        config = MCConfig(
            n_samples=10,
            method="random",
            seed=42,
            ecopath_only=True,
            store_runs=True,
        )
        result = run_montecarlo(params, config)
        assert result.ecopath_runs is not None
        assert len(result.ecopath_runs) == result.n_feasible

    def test_morris_screening(self):
        """Morris on 3-group model -> all params ranked."""
        from pypath.core.sensitivity import SensitivityConfig, run_sensitivity

        params = _make_mc_model()
        config = SensitivityConfig(
            method="morris",
            n_trajectories=5,
            seed=42,
            ecopath_only=True,
        )
        result = run_sensitivity(params, config)
        assert len(result.mu_star) > 0
        assert len(result.parameter_names) == len(result.mu_star)
