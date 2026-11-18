"""Tests for pypath.core.sensitivity module."""

import numpy as np
import pytest

from pypath.core.sensitivity import (
    MorrisResult,
    SensitivityConfig,
    _compute_elementary_effects,
    _generate_morris_trajectories,
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
            from pypath.core.params import create_rpath_params
            from pypath.core.sensitivity import run_sensitivity

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


class TestSobolRuns:
    """Sobol analysis runs end to end on SALib's current API."""

    def _run(self, seed):
        pytest.importorskip("SALib")
        from pypath.core.sensitivity import run_sensitivity
        from tests.test_mc_integration import _make_mc_model

        config = SensitivityConfig(method="sobol", n_samples=8, seed=seed)
        return run_sensitivity(_make_mc_model(), config)

    def test_sobol_gives_finite_indices(self):
        from pypath.core.sensitivity import SobolResult

        result = self._run(seed=3)
        assert isinstance(result, SobolResult)
        assert len(result.parameter_names) == 5
        for indices in (result.S1, result.ST, result.S1_conf, result.ST_conf):
            assert indices.shape == (5,)
            assert np.all(np.isfinite(indices))

    # SALib reads a seed of 0 as no seed at all
    @pytest.mark.parametrize("seed", [0, 3])
    def test_sobol_is_reproducible_from_its_seed(self, seed):
        first, second = self._run(seed=seed), self._run(seed=seed)
        for name in ("S1", "ST", "S1_conf", "ST_conf"):
            assert getattr(first, name).tobytes() == getattr(second, name).tobytes()


class TestMorrisLevels:
    """The Morris unit grid maps inside the quantile endpoints.

    It maps linearly onto quantiles 1/(2p) to 1 - 1/(2p), so levels j/(p-1)
    stand for the centres of p equally likely bins. The unit grid used to
    pass straight through the lognormal quantile function, so its levels 0
    and 1 became parameter values of 0 and infinity and most evaluations
    failed.
    """

    def test_grid_maps_onto_bin_midpoint_quantiles(self):
        from pypath.core.sensitivity import _morris_quantiles

        grid = np.array([[0.0, 1 / 3], [2 / 3, 1.0]])
        q = _morris_quantiles(grid, n_levels=4)
        np.testing.assert_allclose(q, [[0.125, 0.375], [0.625, 0.875]])

    def test_trajectory_quantiles_stay_inside_the_unit_interval(self):
        from pypath.core.sensitivity import _morris_quantiles

        for p in (4, 6, 8):
            traj = _generate_morris_trajectories(
                7, 20, n_levels=p, rng=np.random.default_rng(p)
            )
            q = _morris_quantiles(traj, n_levels=p)
            assert q.min() >= 1 / (2 * p) - 1e-12
            assert q.max() <= 1 - 1 / (2 * p) + 1e-12

    def test_every_parameter_gets_a_finite_effect(self):
        from pypath.core.sensitivity import run_sensitivity
        from tests.test_mc_integration import _make_mc_model

        config = SensitivityConfig(method="morris", n_trajectories=10, seed=42)
        result = run_sensitivity(_make_mc_model(), config)
        assert np.all(np.isfinite(result.mu_star))
        assert np.all(np.isfinite(result.sigma))
        assert np.all(np.isfinite(result.mu))
        # The output is total biomass, so the two given biomasses move it
        biomass = [n.startswith("Biomass") for n in result.parameter_names]
        assert np.all(result.mu_star[biomass] > 0)
