"""Tests for pypath.core.montecarlo module."""

import warnings as _warnings

import numpy as np
import pandas as pd
import pytest

from pypath.core.montecarlo import MCConfig, MCResult


class TestMCConfig:
    def test_defaults(self):
        config = MCConfig()
        assert config.n_samples == 1000
        assert config.method == "lhs"
        assert config.seed is None
        assert config.ecopath_only is False
        assert config.ecosim_years is None
        assert config.store_runs is False
        assert config.n_jobs == 1
        assert config.ecosim_method == "RK4"
        assert config.eco_area == 1.0

    def test_custom_values(self):
        config = MCConfig(n_samples=50, method="random", seed=42, ecopath_only=True)
        assert config.n_samples == 50
        assert config.method == "random"
        assert config.ecopath_only is True


class TestMCResult:
    def test_construction(self):
        result = MCResult(
            n_total=100,
            n_feasible=80,
            n_ecosim=0,
            ecopath_stats={"Biomass": pd.DataFrame({"mean": [1.0]})},
            ecosim_stats=None,
            ecopath_runs=None,
            ecosim_runs=None,
            feasibility_rate=0.8,
            parameter_samples=None,
        )
        assert result.n_total == 100
        assert result.n_feasible == 80
        assert result.feasibility_rate == 0.8

    def test_to_dataframe(self):
        stats = pd.DataFrame(
            {
                "mean": [10.0, 5.0],
                "std": [1.0, 0.5],
                "p5": [8.0, 4.0],
                "p25": [9.0, 4.5],
                "p50": [10.0, 5.0],
                "p75": [11.0, 5.5],
                "p95": [12.0, 6.0],
            }
        )
        result = MCResult(
            n_total=100,
            n_feasible=80,
            n_ecosim=0,
            ecopath_stats={"Biomass": stats},
            ecosim_stats=None,
            ecopath_runs=None,
            ecosim_runs=None,
            feasibility_rate=0.8,
            parameter_samples=None,
        )
        df = result.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_to_dict(self):
        result = MCResult(
            n_total=10,
            n_feasible=8,
            n_ecosim=0,
            ecopath_stats={},
            ecosim_stats=None,
            ecopath_runs=None,
            ecosim_runs=None,
            feasibility_rate=0.8,
            parameter_samples=None,
        )
        d = result.to_dict()
        assert isinstance(d, dict)
        assert d["n_total"] == 10
        assert d["feasibility_rate"] == 0.8


from pypath.core.montecarlo import run_montecarlo
from pypath.core.params import create_rpath_params


def _make_mc_params():
    """3-group model with moderate pedigree CVs."""
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
    # Set moderate pedigree
    params.pedigree["Biomass"] = [0.1, 0.1, 0.0]
    params.pedigree["PB"] = [0.1, 0.1, 0.0]
    params.pedigree["QB"] = [0.0, 0.1, 0.0]
    params.pedigree["Diet"] = [0.0, 0.0, 0.0]
    return params


class TestRunMontecarlo:
    def test_ecopath_only(self):
        params = _make_mc_params()
        config = MCConfig(n_samples=20, method="random", seed=42, ecopath_only=True)
        result = run_montecarlo(params, config)
        assert result.n_total == 20
        assert result.n_feasible > 0
        assert result.feasibility_rate > 0
        assert "Biomass" in result.ecopath_stats
        assert result.ecosim_stats is None

    def test_store_runs(self):
        params = _make_mc_params()
        config = MCConfig(
            n_samples=10, method="random", seed=42, ecopath_only=True, store_runs=True
        )
        result = run_montecarlo(params, config)
        assert result.ecopath_runs is not None
        assert len(result.ecopath_runs) == result.n_feasible

    def test_zero_cv_all_identical(self):
        params = _make_mc_params()
        # Set all CVs to 0
        params.pedigree["Biomass"] = [0.0, 0.0, 0.0]
        params.pedigree["PB"] = [0.0, 0.0, 0.0]
        params.pedigree["QB"] = [0.0, 0.0, 0.0]
        config = MCConfig(n_samples=5, method="random", seed=42, ecopath_only=True)
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")  # no distributions warning
            result = run_montecarlo(params, config)
        # With zero CV, all samples are identical → all feasible or all fail
        if result.n_feasible > 1:
            bio = result.ecopath_stats["Biomass"]
            assert bio["std"].iloc[0] == pytest.approx(0.0, abs=1e-10)

    def test_with_ecosim(self):
        params = _make_mc_params()
        config = MCConfig(
            n_samples=5,
            method="random",
            seed=42,
            ecopath_only=False,
            ecosim_years=range(1, 6),
        )
        result = run_montecarlo(params, config)
        assert result.n_ecosim > 0, "Expected at least 1 successful Ecosim run"
        assert result.ecosim_stats is not None

    def test_progress_callback(self):
        params = _make_mc_params()
        config = MCConfig(n_samples=5, method="random", seed=42, ecopath_only=True)
        calls = []
        result = run_montecarlo(
            params, config, progress_callback=lambda i, n: calls.append((i, n))
        )
        assert len(calls) > 0
