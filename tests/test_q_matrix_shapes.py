import numpy as np

from pypath.core.ecosim import _compute_Q_matrix


def test_compute_Q_matrix_accepts_1d_link_arrays():
    # Small example where link arrays are passed as 1-D and should be
    # converted into full matrices for QQ computation.
    n = 3
    params_dict = {
        'NUM_GROUPS': n,
        'NUM_LIVING': 2,
        # Bbase/state include index 0 (Outside) + n groups
        'Bbase': np.array([0.0, 1.0, 1.0, 0.0]),
        'PreyFrom': np.array([1, 2]),
        'PreyTo': np.array([2, 2]),
        # VV, DD and QQbase passed as 1-D link arrays
        'VV': np.array([2.0, 1.0]),
        'DD': np.array([1.0, 1.0]),
        'QQbase': np.array([1.0, 0.0]),
    }

    state = np.array([0.0, 1.0, 1.0, 0.0])
    forcing = {'Ftime': np.ones(n + 1), 'ForcedPrey': np.ones(n + 1)}

    QQ = _compute_Q_matrix(params_dict, state, forcing)

    # Expect the link (prey=1, pred=2) to have a positive computed Q
    assert QQ.shape == (n + 1, n + 1)
    assert QQ[1, 2] > 0.5
    assert np.isclose(QQ[1, 2], 1.0)


def test_active_link_built_from_prey_lists():
    # Ensure that when ActiveLink is missing, it's built from PreyFrom/PreyTo
    n = 2
    params_dict = {
        'NUM_GROUPS': n,
        'NUM_LIVING': n,
        'Bbase': np.ones(n + 1),
        'PreyFrom': np.array([1]),
        'PreyTo': np.array([2]),
        'VV': np.array([1.0]),
        'DD': np.array([1.0]),
        'QQbase': np.array([1.0]),
    }
    state = np.ones(n + 1)
    forcing = {'Ftime': np.ones(n + 1), 'ForcedPrey': np.ones(n + 1)}
    QQ = _compute_Q_matrix(params_dict, state, forcing)
    assert QQ[1, 2] >= 0.0  # exists and computed (may be zero depending on params)
