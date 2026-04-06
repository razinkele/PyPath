"""Tests for pypath.core.sensitivity module."""

import numpy as np
import pytest

from pypath.core.sensitivity import (
    MorrisResult,
    SensitivityConfig,
    _generate_morris_trajectories,
    _compute_elementary_effects,
)


class TestMorrisDesign:
    def test_trajectory_shape(self):
        """Morris trajectories have correct shape: (n_traj * (k+1), k)."""
        k = 3
        n_traj = 5
        traj = _generate_morris_trajectories(
            k, n_traj, n_levels=4, rng=np.random.default_rng(42)
        )
        assert traj.shape == (n_traj * (k + 1), k)

    def test_values_in_unit_cube(self):
        traj = _generate_morris_trajectories(
            4, 3, n_levels=4, rng=np.random.default_rng(42)
        )
        assert np.all(traj >= 0.0)
        assert np.all(traj <= 1.0)

    def test_one_param_changes_per_step(self):
        """Within each trajectory, exactly one parameter changes per step."""
        k = 3
        n_traj = 2
        traj = _generate_morris_trajectories(
            k, n_traj, n_levels=4, rng=np.random.default_rng(42)
        )
        for t in range(n_traj):
            start = t * (k + 1)
            for step in range(k):
                diff = traj[start + step + 1] - traj[start + step]
                n_changed = np.sum(np.abs(diff) > 1e-10)
                assert n_changed == 1


class TestElementaryEffects:
    def test_known_linear_function(self):
        """For y = 2*x0 + 3*x1, EE should be [2, 3]."""
        k = 2
        n_traj = 10
        traj = _generate_morris_trajectories(
            k, n_traj, n_levels=4, rng=np.random.default_rng(42)
        )
        # Evaluate y = 2*x0 + 3*x1
        y = 2.0 * traj[:, 0] + 3.0 * traj[:, 1]
        result = _compute_elementary_effects(traj, y, k, n_traj, n_levels=4)
        assert result.mu_star[0] == pytest.approx(2.0, rel=0.2)
        assert result.mu_star[1] == pytest.approx(3.0, rel=0.2)

    def test_result_structure(self):
        k = 2
        n_traj = 5
        traj = _generate_morris_trajectories(
            k, n_traj, n_levels=4, rng=np.random.default_rng(42)
        )
        y = traj[:, 0] + traj[:, 1]
        result = _compute_elementary_effects(traj, y, k, n_traj, n_levels=4)
        assert isinstance(result, MorrisResult)
        assert len(result.mu_star) == k
        assert len(result.sigma) == k
        assert len(result.mu) == k


class TestSensitivityConfig:
    def test_defaults(self):
        config = SensitivityConfig()
        assert config.method == "morris"
        assert config.n_trajectories == 10
        assert config.n_levels == 4

    def test_sobol_missing(self):
        """Sobol without SALib raises ImportError."""
        from pypath.core.sensitivity import HAS_SALIB

        if not HAS_SALIB:
            from pypath.core.sensitivity import run_sensitivity
            from pypath.core.params import create_rpath_params

            params = create_rpath_params(
                groups=["A", "B", "Det"],
                types=[1, 0, 2],
            )
            params.model.loc[0, "Biomass"] = 10.0
            params.model.loc[1, "Biomass"] = 5.0
            params.model.loc[2, "Biomass"] = 100.0
            params.pedigree["Biomass"] = [0.2, 0.2, 0.0]
            config = SensitivityConfig(method="sobol")
            with pytest.raises(ImportError, match="SALib"):
                run_sensitivity(params, config)
