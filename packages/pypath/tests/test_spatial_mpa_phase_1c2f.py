from __future__ import annotations

import numpy as np

from pypath.spatial.mpa import (
    MarineProtectedArea,
    ReserveNetwork,
    mpa_spillover_index,
)


def test_reserve_network_coverage_and_protection():
    network = ReserveNetwork(
        reserves=[
            MarineProtectedArea("A", [0, 1], protection_level=1.0),
            MarineProtectedArea("B", [2], protection_level=0.6),
        ]
    )
    assert np.isclose(network.coverage_fraction(5), 0.6)
    assert np.isclose(network.average_protection_level(), 0.8)


def test_reserve_network_connectivity_score():
    adjacency = np.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ]
    )
    network = ReserveNetwork([MarineProtectedArea("A", [0, 1, 2])])
    assert network.connectivity_score(adjacency) > 0.0


def test_mpa_spillover_index_nonnegative():
    value = mpa_spillover_index(
        inside_biomass=np.array([3.0, 2.0]),
        outside_biomass=np.array([1.0, 1.5]),
        exchange_rate=0.2,
    )
    assert value > 0.0
