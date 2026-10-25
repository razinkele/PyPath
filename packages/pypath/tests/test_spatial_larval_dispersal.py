from __future__ import annotations

import numpy as np

from pypath.spatial.larval_dispersal import (
    LarvalDispersalParams,
    advect_larvae,
    larval_connectivity_matrix,
    settle_larvae,
)


def test_advect_larvae_survival_and_retention():
    larvae = np.array([10.0, 0.0])
    transport = np.array([[0.2, 0.8], [0.5, 0.5]])
    params = LarvalDispersalParams(mortality_rate=0.1, retention_rate=0.2)
    advected = advect_larvae(larvae, transport, params)
    assert np.all(advected >= 0)


def test_settle_larvae_partition():
    larvae = np.array([10.0, 5.0, 2.0])
    quality = np.array([1.0, 0.5, 0.0])
    params = LarvalDispersalParams(settlement_rate=0.4)
    settled, remaining = settle_larvae(larvae, quality, params)
    np.testing.assert_allclose(settled + remaining, larvae)
    assert settled[2] == 0.0


def test_larval_connectivity_shape():
    spawning = np.array([3.0, 2.0, 1.0])
    transport = np.eye(3)
    conn = larval_connectivity_matrix(spawning, transport, LarvalDispersalParams())
    assert conn.shape == (3, 3)
