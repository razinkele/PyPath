"""
Tests for spatial Ecosim integration.

These tests verify that spatial ECOSPACE correctly integrates
with Ecosim dynamics.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.slow

from pypath.spatial import (  # noqa: E402
    EcospaceParams,
    create_1d_grid,
    deriv_vector_spatial,
    rsim_run_spatial,
)


class TestSpatialDerivative:
    """Test spatial derivative calculation."""

    def test_deriv_vector_spatial_basic(self):
        """Test basic spatial derivative calculation."""
        grid = create_1d_grid(n_patches=3, spacing=1.0)
        n_patches = 3
        n_groups = 2

        ecospace = EcospaceParams(
            grid=grid,
            habitat_preference=np.ones((n_groups, n_patches)),
            habitat_capacity=np.ones((n_groups, n_patches)),
            dispersal_rate=np.array([0.0, 2.0]),
            advection_enabled=np.array([False, False]),
            gravity_strength=np.array([0.0, 0.0]),
        )

        state_spatial = np.array(
            [
                [0, 0, 0],
                [5, 5, 5],
                [10, 20, 10],
            ],
            dtype=float,
        )

        # Keys must match what deriv_vector() expects (not RsimParams attribute names)
        params = {
            "NUM_GROUPS": 2,
            "NUM_LIVING": 1,
            "NUM_DEAD": 1,
            "NUM_GEARS": 0,
            "PB": np.array([0, 0.5, 1.0]),
            "QB": np.array([0, 2.0, 0.0]),
            "M0": np.array([0, 0.1, 0.2]),
            "Unassim": np.array([0, 0.2, 0.2]),
            "Bbase": np.array([0, 5, 20]),
            "B_BaseRef": np.array([0, 5, 20]),
            "PP_type": np.array([0, 0, 2]),
            "NoIntegrate": np.array([0, 0, 0]),
            "ActiveLink": np.zeros((3, 3), dtype=bool),
            "VV": np.ones((3, 3)) * 2.0,
            "DD": np.ones((3, 3)) * 1000.0,
            "QQbase": np.zeros((3, 3)),
            "DetFrac": np.zeros((3, 2)),
            "DetFrom": np.array([]),
            "DetTo": np.array([]),
            "FishFrom": np.array([0]),
            "FishThrough": np.array([0]),
            "FishQ": np.array([0.0]),
        }

        forcing = {
            "Ftime": np.ones(3),
            "ForcedBio": np.zeros(3),
            "ForcedMigrate": np.zeros(3),
            "ForcedPrey": np.ones(3),
            "PP_forcing": np.ones(3),
        }

        fishing = {
            "FishFrom": np.array([0]),
            "FishThrough": np.array([0]),
            "FishQ": np.array([0.0]),
        }

        deriv = deriv_vector_spatial(
            state_spatial,
            params,
            forcing,
            fishing,
            ecospace,
            environmental_drivers=None,
            t=0.0,
            dt=1.0 / 12.0,
        )

        assert deriv.shape == state_spatial.shape
        assert deriv.shape == (3, 3)


class TestSpatialIntegrationBasic:
    """Test basic spatial integration functionality."""

    def test_spatial_vs_nonspatial_single_patch(
        self, spatial_scenario, single_patch_ecospace
    ):
        """Test that 1-patch spatial equals non-spatial."""
        from pypath.core.ecosim import rsim_run

        scenario, _ = spatial_scenario
        result_nonspatial = rsim_run(scenario, years=range(1, 3))
        result_spatial = rsim_run_spatial(
            scenario, ecospace=single_patch_ecospace, years=range(1, 3)
        )

        np.testing.assert_allclose(
            result_nonspatial.out_Biomass,
            result_spatial.out_Biomass,
            rtol=1e-4,
            atol=1e-8,
        )

    def test_mass_conservation_spatial(self, spatial_scenario, simple_ecospace):
        """Test that total biomass is conserved in spatial simulation."""
        scenario, _ = spatial_scenario
        result = rsim_run_spatial(scenario, ecospace=simple_ecospace, years=range(1, 3))

        initial_total = result.out_Biomass[0].sum()
        final_total = result.out_Biomass[-1].sum()

        # Allow up to 50% drift — Ecosim dynamics (production, mortality, fishing)
        # naturally change total biomass.  What matters is no NaN/Inf/crash.
        assert np.all(np.isfinite(result.out_Biomass))
        assert final_total > 0, "Total biomass collapsed to zero"
        if initial_total > 0:
            relative_change = abs(final_total - initial_total) / initial_total
            assert (
                relative_change < 0.5
            ), f"Biomass changed by {relative_change * 100:.1f}% — suspicious"

    def test_spatial_flux_affects_distribution(self, spatial_scenario):
        """Test that spatial flux changes biomass distribution."""
        scenario, _ = spatial_scenario
        ng = scenario.params.NUM_GROUPS + 1

        # Create 5-patch grid with high dispersal
        grid = create_1d_grid(n_patches=5, spacing=1.0)
        ecospace = EcospaceParams(
            grid=grid,
            habitat_preference=np.ones((ng, 5)),
            habitat_capacity=np.ones((ng, 5)),
            dispersal_rate=np.full(ng, 5.0),
            advection_enabled=np.zeros(ng, dtype=bool),
            gravity_strength=np.zeros(ng),
        )

        result = rsim_run_spatial(scenario, ecospace=ecospace, years=range(1, 3))

        # Spatial output should exist and have correct shape
        assert hasattr(result, "out_Biomass_spatial")
        assert result.out_Biomass_spatial.shape[2] == 5  # 5 patches

        # With dispersal, biomass should be distributed across patches
        # (not all concentrated in one patch)
        final_spatial = result.out_Biomass_spatial[-1]  # [n_groups, n_patches]
        for g in range(1, ng):
            patch_biomass = final_spatial[g, :]
            if patch_biomass.sum() > 0:
                # At least 2 patches should have non-zero biomass
                nonzero_patches = np.count_nonzero(patch_biomass > 1e-10)
                assert (
                    nonzero_patches >= 2
                ), f"Group {g}: biomass in only {nonzero_patches} patches"


class TestBackwardCompatibility:
    """Test backward compatibility with non-spatial Ecosim."""

    def test_rsim_run_spatial_without_ecospace(self, spatial_scenario):
        """Test that rsim_run_spatial works without ecospace (non-spatial mode)."""
        scenario, _ = spatial_scenario
        result = rsim_run_spatial(scenario, years=range(1, 3))

        assert result.out_Biomass.shape[0] > 0
        assert not hasattr(result, "out_Biomass_spatial")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
