from __future__ import annotations

import numpy as np

from pypath.spatial.ecological_interactions import (
    EdgeEffectParams,
    apply_edge_effects,
    metapopulation_occupancy_step,
    spatial_predation_rate,
)


def test_spatial_predation_rate_shape():
    pred = np.array([2.0, 1.0])
    prey = np.array([4.0, 3.0])
    kernel = np.eye(2)
    rates = spatial_predation_rate(pred, prey, kernel, attack_rate=0.5)
    assert rates.shape == prey.shape


def test_apply_edge_effects_reduces_edges_only():
    quality = np.array([1.0, 1.0, 1.0])
    mask = np.array([False, True, False])
    adjusted = apply_edge_effects(quality, mask, EdgeEffectParams(edge_intensity=0.4))
    assert adjusted[1] == 0.6
    assert adjusted[0] == 1.0


def test_metapopulation_occupancy_step_bounds():
    occ = np.array([0.2, 0.5, 0.8])
    conn = np.array(
        [
            [0.0, 0.2, 0.2],
            [0.2, 0.0, 0.2],
            [0.2, 0.2, 0.0],
        ]
    )
    next_occ = metapopulation_occupancy_step(occ, 0.3, 0.1, conn)
    assert np.all((next_occ >= 0.0) & (next_occ <= 1.0))
