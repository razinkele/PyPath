from __future__ import annotations

import numpy as np

from pypath.spatial.movement import (
    MovementKernel,
    aggregation_index,
    behavioral_migration,
    ontogenetic_movement,
    predator_avoidance_movement,
)


def test_behavioral_migration_conserves_flux_sum():
    biomass = np.array([10.0, 10.0, 10.0])
    habitat = np.array([0.2, 1.0, 0.5])
    adjacency = np.array(
        [
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ]
    )
    flux = behavioral_migration(biomass, habitat, adjacency, MovementKernel(rate=0.2))
    assert np.isclose(flux.sum(), 0.0)


def test_ontogenetic_movement_shifts_to_adults():
    juv = np.array([10.0, 6.0, 4.0])
    adults = np.array([3.0, 2.0, 1.0])
    nursery = np.array([1.0, 0.8, 0.6])
    feeding = np.array([0.2, 0.4, 0.4])

    next_juv, next_ad = ontogenetic_movement(juv, adults, nursery, feeding, 0.5)
    assert next_ad.sum() > adults.sum()
    assert next_juv.sum() < juv.sum()


def test_predator_avoidance_reduces_in_high_risk_patch():
    prey = np.array([8.0, 8.0, 8.0])
    pred = np.array([0.1, 2.0, 0.1])
    adjacency = np.ones((3, 3)) - np.eye(3)
    flux = predator_avoidance_movement(
        prey, pred, adjacency, MovementKernel(rate=0.3, avoidance_strength=3.0)
    )
    assert flux[1] < 0


def test_aggregation_index_zero_for_uniform():
    assert aggregation_index(np.array([1.0, 1.0, 1.0])) == 0.0
