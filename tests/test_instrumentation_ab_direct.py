import numpy as np
from pypath.core.ecosim_deriv import integrate_ab


def test_integrate_ab_converts_1based_numeric_groups():
    """Direct test of integrate_ab to ensure numeric 1-based INSTRUMENT_GROUPS are converted."""
    n = 6  # including outside index 0
    NUM_GROUPS = n - 1
    # minimal params dict required by deriv_vector
    params = {
        'NUM_GROUPS': NUM_GROUPS,
        'NUM_LIVING': NUM_GROUPS - 0,
        'NUM_DEAD': 0,
        'PB': np.ones(n),
        'QB': np.ones(n),
        'ActiveLink': np.ones((n, n), dtype=bool),
        'VV': np.ones((n, n)),
        'DD': np.ones((n, n)) * 2.0,
        'Unassim': np.zeros(n),
        'Bbase': np.ones(n),
        'spname': ['Outside'] + [f'G{i}' for i in range(1, n)],
    }

    state = np.ones(n)
    derivs_history = []
    forcing = {'Ftime': np.ones(n), 'ForcedBio': np.zeros(n), 'ForcedMigrate': np.zeros(n), 'ForcedEffort': np.ones(1)}
    fishing = {'FishFrom': np.array([0]), 'FishThrough': np.array([0]), 'FishQ': np.array([0.0]), 'FishingMort': np.zeros(n)}

    # Legacy 1-based index for group G3 (which has 0-based index 2 in groups)
    legacy_one_based = 3
    expected_zero_based = legacy_one_based - 1

    captured = []

    def cb(payload):
        captured.append(payload)

    params['INSTRUMENT_GROUPS'] = [legacy_one_based]
    params['instrument_callback'] = cb

    new_state, deriv_current = integrate_ab(state, derivs_history, params, forcing, fishing, dt=1.0)

    assert len(captured) > 0, "Callback was not invoked"
    assert expected_zero_based in captured[0].get('groups', []), f"Expected converted group {expected_zero_based} in payload groups, got {captured[0].get('groups', [])}"
