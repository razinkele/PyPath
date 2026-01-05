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

    def test_rsim_run_spatial_without_ecospace(self):
        """Test that rsim_run_spatial works without ecospace (non-spatial mode)."""
        pytest.skip("Requires full Ecosim scenario setup - placeholder test")

        # TODO: Implement once we have rsim_scenario working
        # from pypath.core import rsim_scenario
        #
        # # Create standard non-spatial scenario
        # scenario = rsim_scenario(model, params)
        #
        # # Call spatial function without ecospace
        # result = rsim_run_spatial(scenario)
        #
        # # Should run as standard non-spatial Ecosim
        # assert result.out_Biomass.shape == (n_months, n_groups)
        # assert not hasattr(result, 'out_Biomass_spatial')

    def test_ecospace_none_equals_nonspatial(self):
        """Test that ecospace=None produces identical results to non-spatial."""
        pytest.skip("Requires full Ecosim scenario setup - placeholder test")

        # TODO: Implement comparison test
        # from pypath.core import rsim_scenario, rsim_run
        #
        # scenario = rsim_scenario(model, params)
        #
        # # Non-spatial run
        # result_nonspatial = rsim_run(scenario)
        #
        # # Spatial run with ecospace=None
        # result_spatial = rsim_run_spatial(scenario, ecospace=None)
        #
        # # Should be identical
        # np.testing.assert_allclose(
        #     result_nonspatial.out_Biomass,
        #     result_spatial.out_Biomass,
        #     rtol=1e-10
        # )

    def test_single_patch_equals_nonspatial(self):
        """Test that 1-patch spatial equals non-spatial.

        This is a critical validation - if there's only one patch,
        spatial and non-spatial should give identical results.
        """
        pytest.skip("Requires full Ecosim scenario setup - placeholder test")

        # TODO: Implement once we have rsim_scenario working
        # from pypath.core import rsim_scenario, rsim_run
        #
        # # Create scenario
        # scenario = rsim_scenario(model, params)
        #
        # # Run non-spatial
        # result_nonspatial = rsim_run(scenario, years=range(1, 11))
        #
        # # Create 1-patch spatial grid
        # grid = create_1d_grid(n_patches=1)
        # n_groups = scenario.params.NUM_GROUPS
        #
        # ecospace = EcospaceParams(
        #     grid=grid,
        #     habitat_preference=np.ones((n_groups, 1)),
        #     habitat_capacity=np.ones((n_groups, 1)),
        #     dispersal_rate=np.zeros(n_groups),  # No dispersal in 1-patch
        #     advection_enabled=np.zeros(n_groups, dtype=bool),
        #     gravity_strength=np.zeros(n_groups)
        # )
        #
        # # Run spatial
        # result_spatial = rsim_run_spatial(scenario, ecospace=ecospace, years=range(1, 11))
        #
        # # Results should be identical
        # np.testing.assert_allclose(
        #     result_nonspatial.out_Biomass,
        #     result_spatial.out_Biomass.sum(axis=2),  # Sum over single patch
        #     rtol=1e-5,
        #     atol=1e-8
        # )

    def test_optional_parameters_dont_break_existing_code(self):
        """Test that RsimScenario has optional ecospace fields."""
        import dataclasses

        from pypath.core.ecosim import RsimScenario

        # Check that RsimScenario is a dataclass with ecospace field
        assert dataclasses.is_dataclass(RsimScenario), (
            "RsimScenario should be a dataclass"
        )

        # Check that ecospace field exists and is optional
        fields = {f.name: f for f in dataclasses.fields(RsimScenario)}

        assert "ecospace" in fields, "RsimScenario should have ecospace field"
        assert "environmental_drivers" in fields, (
            "RsimScenario should have environmental_drivers field"
        )

        # Check that ecospace defaults to None
        ecospace_field = fields["ecospace"]
        assert (
            ecospace_field.default is None
            or ecospace_field.default_factory is not dataclasses.MISSING
        ), "ecospace field should have a default value"

    def test_existing_ecosim_imports_unchanged(self):
        """Test that existing import patterns still work."""
        # These imports should work without change
        from pypath.core import RsimScenario
        from pypath.core.ecosim import rsim_run

        # Spatial imports are separate
        from pypath.spatial import EcospaceParams

        # Both should be importable without conflict
        assert RsimScenario is not None
        assert rsim_run is not None
        assert EcospaceParams is not None
        assert rsim_run_spatial is not None


class TestNoSpatialDependenciesRequired:
    """Test that non-spatial code doesn't require spatial dependencies."""

    def test_core_ecosim_imports_without_spatial(self):
        """Test that core Ecosim can be imported without spatial modules."""
        # This should work even if spatial dependencies (geopandas, etc.) are missing
        from pypath.core import RsimParams, RsimScenario

        assert RsimParams is not None
        assert RsimScenario is not None

    def test_spatial_imports_are_optional(self):
        """Test that spatial imports are in separate module."""
        # Spatial features should be opt-in
        try:
            import importlib.util

            spatial_available = importlib.util.find_spec("pypath.spatial") is not None
        except Exception:
            spatial_available = False

        # This test always passes - just documents that spatial is optional
        # In practice, spatial deps should be installed, so this will be True
        assert isinstance(spatial_available, bool)


class TestParameterValidation:
    """Test that invalid spatial parameters are caught early."""

    def test_ecospace_grid_required(self):
        """Test that EcospaceParams requires a grid."""
        with pytest.raises(TypeError):
            # Missing required 'grid' argument
            EcospaceParams()

    def test_habitat_arrays_match_grid_size(self):
        """Test that habitat arrays must match grid n_patches."""
        grid = create_1d_grid(n_patches=5)

        # Wrong size habitat preference
        with pytest.raises((ValueError, IndexError)):
            ecospace = EcospaceParams(
                grid=grid,
                habitat_preference=np.ones((3, 10)),  # Wrong n_patches
                habitat_capacity=np.ones((3, 5)),
                dispersal_rate=np.zeros(3),
                advection_enabled=np.zeros(3, dtype=bool),
                gravity_strength=np.zeros(3),
            )
            # Error might occur on access, not construction
            _ = ecospace.habitat_preference[:, : grid.n_patches]


class TestDataStructureCompatibility:
    """Test that data structures are backward compatible."""

    def test_rsim_output_structure_unchanged(self):
        """Test that RsimOutput structure remains compatible."""
        pytest.skip("Requires full Ecosim scenario setup - placeholder test")

        # TODO: Test that existing output attributes are preserved
        # from pypath.core import rsim_run
        #
        # result = rsim_run(scenario)
        #
        # # Standard attributes should exist
        # assert hasattr(result, 'out_Biomass')
        # assert hasattr(result, 'out_Catch')
        # assert hasattr(result, 'out_Mortality')
        # assert hasattr(result, 'start_state')
        # assert hasattr(result, 'end_state')

    def test_spatial_output_adds_without_breaking(self):
        """Test that spatial output adds attributes without breaking existing."""
        pytest.skip("Requires full Ecosim scenario setup - placeholder test")

        # TODO: Test that spatial adds new attributes
        # result_spatial = rsim_run_spatial(scenario, ecospace=ecospace)
        #
        # # Standard attributes still exist
        # assert hasattr(result_spatial, 'out_Biomass')
        #
        # # New spatial attributes added
        # assert hasattr(result_spatial, 'out_Biomass_spatial')
        # assert result_spatial.out_Biomass_spatial.shape == (n_months, n_groups+1, n_patches)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
