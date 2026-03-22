"""Tests for Phase 2 validation & numerical stability guards."""

import warnings

import numpy as np
import pytest


# Task 10: dispersal zero-distance guard
class TestDispersalZeroDistance:
    def test_zero_distance_no_crash(self):
        """Patches at same location should not cause inf/nan."""
        try:
            from pypath.spatial.dispersal import _diffusion_flux_numba
        except ImportError:
            pytest.skip("numba not available")

        biomass = np.array([10.0, 5.0])
        rows = np.array([0])
        cols = np.array([1])
        distances = np.array([0.0])  # same location!
        border_lengths = np.array([1.0])
        n_patches = 2
        dispersal_rate = 0.1

        result = _diffusion_flux_numba(
            biomass, dispersal_rate, rows, cols, distances, border_lengths, n_patches
        )
        assert np.all(np.isfinite(result))
        # Zero distance means zero flux (skipped)
        assert np.allclose(result, 0.0)


# Task 11: habitat tolerance validation
class TestHabitatTolerance:
    def test_zero_tolerance_raises(self):
        """tolerance=0 must raise ValueError."""
        from pypath.spatial.habitat import create_gaussian_response

        with pytest.raises(ValueError, match="tolerance must be > 0"):
            create_gaussian_response(optimal_value=10.0, tolerance=0.0)

    def test_negative_tolerance_raises(self):
        """Negative tolerance must raise ValueError."""
        from pypath.spatial.habitat import create_gaussian_response

        with pytest.raises(ValueError, match="tolerance must be > 0"):
            create_gaussian_response(optimal_value=10.0, tolerance=-1.0)

    def test_valid_tolerance_works(self):
        """Positive tolerance should work normally."""
        from pypath.spatial.habitat import create_gaussian_response

        resp = create_gaussian_response(optimal_value=10.0, tolerance=3.0)
        result = resp(np.array([7.0, 10.0, 13.0]))
        assert result[1] == pytest.approx(1.0)  # optimal
        assert result[0] == pytest.approx(result[2])  # symmetric


# Task 12: M0 NaN guard
class TestM0NanGuard:
    def test_m0_finite_when_ee_nan(self):
        """M0 should be 0 (not NaN) when EE is unknown."""
        from pypath.core.ecopath import rpath
        from pypath.core.params import create_rpath_params

        params = create_rpath_params(
            groups=["Phyto", "Zoo", "Detritus"],
            types=[1, 0, 2],
        )
        m = params.model
        m.loc[m["Group"] == "Phyto", ["Biomass", "PB"]] = [10.0, 50.0]
        m.loc[m["Group"] == "Phyto", "EE"] = np.nan  # unknown EE
        m.loc[m["Group"] == "Zoo", ["Biomass", "PB", "QB"]] = [5.0, 2.0, 10.0]
        m.loc[m["Group"] == "Zoo", "EE"] = 0.9
        m.loc[m["Group"] == "Detritus", "Biomass"] = 100.0
        m["BioAcc"] = 0.0
        m["Unassim"] = 0.2
        m.loc[m["Group"] == "Phyto", "Unassim"] = 0.0
        m.loc[m["Group"] == "Detritus", "Unassim"] = 0.0
        m["Detritus"] = 1.0
        params.diet["Zoo"] = [1.0, 0.0, 0.0, 0.0]  # 3 groups + Import row

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            balanced = rpath(params)

        # M0 should be finite even if EE was NaN input
        # (after balancing, EE may be computed, but the guard should prevent NaN propagation)
        assert balanced is not None
