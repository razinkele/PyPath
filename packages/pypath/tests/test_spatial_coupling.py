"""Tests for the 1C.3 coupled Ecosim-Ecospace framework."""

from __future__ import annotations

import numpy as np

from pypath.spatial.coupling import (
    CoupledStateVector,
    build_initial_state_vector,
    run_coupled_simulation,
)


def test_coupled_state_vector_roundtrip():
    spatial = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ]
    )
    vec = CoupledStateVector.from_spatial_biomass(spatial)
    recovered = vec.to_spatial_biomass()
    np.testing.assert_allclose(recovered, spatial)


def test_build_initial_state_vector_preserves_group_totals():
    totals = np.array([12.0, 6.0, 4.0, 2.0])
    habitat = np.array(
        [
            [2.0, 1.0, 1.0],
            [1.0, 2.0, 1.0],
            [1.0, 1.0, 2.0],
        ]
    )

    vec = build_initial_state_vector(totals, n_patches=3, habitat_capacity=habitat)
    collapsed = vec.to_total_biomass()
    np.testing.assert_allclose(collapsed, totals)


def test_rsim_run_spatial_with_state_vector_io(spatial_scenario, simple_ecospace):
    scenario, _ = spatial_scenario
    ng = scenario.params.NUM_GROUPS + 1
    n_patches = simple_ecospace.grid.n_patches

    initial = np.tile(scenario.start_state.Biomass[:, np.newaxis], (1, n_patches)) / n_patches
    initial_vec = initial.reshape(-1)

    result = run_coupled_simulation(
        scenario,
        simple_ecospace,
        years=range(1, 3),
        state_vector=initial_vec,
        store_state_vectors=True,
    )

    assert hasattr(result, "out_Biomass_spatial")
    assert hasattr(result, "out_StateVector")
    assert hasattr(result, "state_vector_final")

    assert result.out_StateVector.shape[1] == ng * n_patches
    assert result.state_vector_final.shape[0] == ng * n_patches

    np.testing.assert_allclose(
        result.out_StateVector[-1], result.state_vector_final, rtol=1e-10, atol=1e-12
    )
