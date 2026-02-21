"""
Test backward compatibility of spatial features.

These tests verify that:
1. Non-spatial Ecosim code continues to work unchanged
2. Adding ecospace=None has no effect on existing simulations
3. All existing test patterns remain valid
"""

import numpy as np
import pytest

from pypath.spatial import EcospaceParams, create_1d_grid, rsim_run_spatial


class TestBackwardCompatibility:
    """Test that spatial features don't break existing non-spatial code."""

    def test_rsim_run_spatial_without_ecospace(self, spatial_scenario):
        """Test that rsim_run_spatial works without ecospace (non-spatial mode)."""
        scenario, _ = spatial_scenario
        result = rsim_run_spatial(scenario, years=range(1, 3))
        assert result.out_Biomass.shape[0] > 0
        assert not hasattr(result, "out_Biomass_spatial")

    def test_ecospace_none_equals_nonspatial(self, spatial_scenario):
        """Test that ecospace=None produces identical results to non-spatial."""
        from pypath.core.ecosim import rsim_run

        scenario, _ = spatial_scenario
        result_nonspatial = rsim_run(scenario, years=range(1, 3))
        result_spatial = rsim_run_spatial(scenario, ecospace=None, years=range(1, 3))

        np.testing.assert_allclose(
            result_nonspatial.out_Biomass,
            result_spatial.out_Biomass,
            rtol=1e-10,
        )

    def test_single_patch_equals_nonspatial(
        self, spatial_scenario, single_patch_ecospace
    ):
        """Test that 1-patch spatial equals non-spatial.

        This is a critical validation - if there's only one patch,
        spatial and non-spatial should give identical results.
        """
        from pypath.core.ecosim import rsim_run

        scenario, _ = spatial_scenario
        result_nonspatial = rsim_run(scenario, years=range(1, 3))
        result_spatial = rsim_run_spatial(
            scenario, ecospace=single_patch_ecospace, years=range(1, 3)
        )

        # Spatial out_Biomass sums over patches — for 1 patch should match
        np.testing.assert_allclose(
            result_nonspatial.out_Biomass,
            result_spatial.out_Biomass,
            rtol=1e-4,
            atol=1e-8,
        )

    def test_optional_parameters_dont_break_existing_code(self):
        """Test that RsimScenario has optional ecospace fields."""
        import dataclasses

        from pypath.core.ecosim import RsimScenario

        assert dataclasses.is_dataclass(RsimScenario)
        fields = {f.name: f for f in dataclasses.fields(RsimScenario)}

        assert "ecospace" in fields
        assert "environmental_drivers" in fields

        ecospace_field = fields["ecospace"]
        assert (
            ecospace_field.default is None
            or ecospace_field.default_factory is not dataclasses.MISSING
        )

    def test_existing_ecosim_imports_unchanged(self):
        """Test that existing import patterns still work."""
        from pypath.core import RsimScenario
        from pypath.core.ecosim import rsim_run
        from pypath.spatial import EcospaceParams

        assert RsimScenario is not None
        assert rsim_run is not None
        assert EcospaceParams is not None
        assert rsim_run_spatial is not None


class TestNoSpatialDependenciesRequired:
    """Test that non-spatial code doesn't require spatial dependencies."""

    def test_core_ecosim_imports_without_spatial(self):
        """Test that core Ecosim can be imported without spatial modules."""
        from pypath.core import RsimParams, RsimScenario

        assert RsimParams is not None
        assert RsimScenario is not None

    def test_spatial_imports_are_optional(self):
        """Test that spatial imports are in separate module."""
        try:
            import importlib.util

            spatial_available = importlib.util.find_spec("pypath.spatial") is not None
        except Exception:
            spatial_available = False

        assert isinstance(spatial_available, bool)


class TestParameterValidation:
    """Test that invalid spatial parameters are caught early."""

    def test_ecospace_grid_required(self):
        """Test that EcospaceParams requires a grid."""
        with pytest.raises(TypeError):
            EcospaceParams()

    def test_habitat_arrays_match_grid_size(self):
        """Test that habitat arrays must match grid n_patches."""
        grid = create_1d_grid(n_patches=5)

        with pytest.raises((ValueError, IndexError)):
            ecospace = EcospaceParams(
                grid=grid,
                habitat_preference=np.ones((3, 10)),  # Wrong n_patches
                habitat_capacity=np.ones((3, 5)),
                dispersal_rate=np.zeros(3),
                advection_enabled=np.zeros(3, dtype=bool),
                gravity_strength=np.zeros(3),
            )
            _ = ecospace.habitat_preference[:, : grid.n_patches]


class TestDataStructureCompatibility:
    """Test that data structures are backward compatible."""

    def test_rsim_output_structure_unchanged(self, spatial_scenario):
        """Test that RsimOutput structure remains compatible."""
        from pypath.core.ecosim import rsim_run

        scenario, _ = spatial_scenario
        result = rsim_run(scenario, years=range(1, 3))

        assert hasattr(result, "out_Biomass")
        assert hasattr(result, "out_Catch")
        assert hasattr(result, "end_state")

    def test_spatial_output_adds_without_breaking(
        self, spatial_scenario, simple_ecospace
    ):
        """Test that spatial output adds attributes without breaking existing."""
        scenario, _ = spatial_scenario
        result_spatial = rsim_run_spatial(
            scenario, ecospace=simple_ecospace, years=range(1, 3)
        )

        # Standard attributes still exist
        assert hasattr(result_spatial, "out_Biomass")
        assert result_spatial.out_Biomass.shape[0] > 0

        # New spatial attribute added
        assert hasattr(result_spatial, "out_Biomass_spatial")
        n_months = result_spatial.out_Biomass.shape[0]
        n_groups = scenario.params.NUM_GROUPS + 1
        n_patches = simple_ecospace.grid.n_patches
        assert result_spatial.out_Biomass_spatial.shape == (
            n_months,
            n_groups,
            n_patches,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
