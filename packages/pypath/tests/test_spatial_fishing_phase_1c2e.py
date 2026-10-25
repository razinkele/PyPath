from __future__ import annotations

import numpy as np

from pypath.spatial.fishing import (
    FleetDeployment,
    distribute_effort_with_fleet,
    mpa_spillover_potential,
)


def test_fleet_deployment_accessibility_zero_beyond_range():
    fleet = FleetDeployment(
        "trawl", home_ports=np.array([0]), max_range_km=10.0, capacity=1.0
    )
    weights = fleet.accessibility_weights(np.array([0.0, 5.0, 20.0]))
    assert weights[2] == 0.0


def test_distribute_effort_with_fleet_normalized_to_capacity():
    base = np.array([1.0, 1.0, 1.0])
    dist = np.array([2.0, 4.0, 8.0])
    fleet = FleetDeployment(
        "gillnet", home_ports=np.array([0]), max_range_km=20.0, capacity=100.0
    )
    effort = distribute_effort_with_fleet(base, fleet, dist)
    assert np.isclose(effort.sum(), 100.0)


def test_mpa_spillover_potential_positive_gradient_only():
    low = mpa_spillover_potential(
        biomass_inside_mpa=1.0,
        biomass_outside_mpa=2.0,
        boundary_length_km=10.0,
        mobility_rate=0.1,
    )
    high = mpa_spillover_potential(
        biomass_inside_mpa=4.0,
        biomass_outside_mpa=1.0,
        boundary_length_km=10.0,
        mobility_rate=0.1,
    )
    assert low == 0.0
    assert high > 0.0
