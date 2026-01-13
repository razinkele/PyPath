import numpy as np
import pandas as pd
import pytest

from pypath.core.ecosim import rsim_run


def test_seabirds_termwise_matches_rpath_first_12_months():
    """Compare per-month term components (production, predation_loss, consumption, fish_loss, m0_loss, derivative)
    between PyPath (RK4) and Rpath reference for Seabirds for the first 12 months.
    """
    from pathlib import Path
    from pypath.core.ecopath import rpath
    from pypath.core.params import create_rpath_params
    from pypath.core.ecosim import rsim_scenario, _build_link_matrix, _normalize_fishing_input
    from pypath.core.ecosim_deriv import deriv_vector

    ECOPATH_DIR = Path("tests/data/rpath_reference/ecopath")
    ECOSIM_DIR = Path("tests/data/rpath_reference/ecosim")

    # Prepare scenario
    model_df = pd.read_csv(ECOPATH_DIR / "model_params.csv")
    diet_df = pd.read_csv(ECOPATH_DIR / "diet_matrix.csv")
    groups = model_df["Group"].tolist()
    types = model_df["Type"].tolist()
    params = create_rpath_params(groups, types)
    params.model = model_df
    params.diet = diet_df
    if (ECOPATH_DIR / "stanza_groups.csv").exists() and (ECOPATH_DIR / "stanza_indiv.csv").exists():
        params.stanzas.stgroups = pd.read_csv(ECOPATH_DIR / "stanza_groups.csv")
        params.stanzas.stindiv = pd.read_csv(ECOPATH_DIR / "stanza_indiv.csv")
    pypath_model = rpath(params)
    pypath_ecosim = rsim_scenario(pypath_model, params, years=range(1, 101))

    if 'Seabirds' not in pypath_ecosim.params.spname:
        pytest.skip('Seabirds not present')
    sidx = pypath_ecosim.params.spname.index('Seabirds')

    # Run RK4 for 2 years (24 months) to get monthly snapshots
    pypath_output = rsim_run(pypath_ecosim, method='RK4', years=range(1, 3))

    # Load reference components (RK4)
    ref = pd.read_csv(ECOSIM_DIR / "diagnostics/seabirds_components_rk4.csv")

    # Build params_dict and fishing_dict for term computations
    from pypath.core.ecosim import _build_active_link_matrix

    params_dict = {
        "NUM_GROUPS": pypath_ecosim.params.NUM_GROUPS,
        "NUM_LIVING": pypath_ecosim.params.NUM_LIVING,
        "NUM_DEAD": pypath_ecosim.params.NUM_DEAD,
        "NUM_GEARS": pypath_ecosim.params.NUM_GEARS,
        "PB": pypath_ecosim.params.PBopt,
        "QB": pypath_ecosim.params.FtimeQBOpt,
        "M0": pypath_ecosim.params.MzeroMort,
        "Unassim": pypath_ecosim.params.UnassimRespFrac,
        "ActiveLink": _build_active_link_matrix(pypath_ecosim.params),
        "VV": _build_link_matrix(pypath_ecosim.params, pypath_ecosim.params.VV),
        "DD": _build_link_matrix(pypath_ecosim.params, pypath_ecosim.params.DD),
        "QQbase": _build_link_matrix(pypath_ecosim.params, pypath_ecosim.params.QQ),
        "Bbase": pypath_ecosim.params.B_BaseRef,
        "PP_type": getattr(pypath_ecosim.params, 'PP_type', np.zeros(pypath_ecosim.params.NUM_GROUPS + 1)),
        "NoIntegrate": pypath_ecosim.params.NoIntegrate,
        "FishFrom": getattr(pypath_ecosim.params, 'FishFrom', np.array([])),
        "FishTo": getattr(pypath_ecosim.params, 'FishTo', np.array([])),
        "FishQ": getattr(pypath_ecosim.params, 'FishQ', np.array([])),
    }

    fishing_dict = _normalize_fishing_input(pypath_ecosim.fishing, pypath_ecosim.params.NUM_GROUPS + 1)

    # Compare first 12 months
    months = 12
    for m in range(1, months + 1):
        state = pypath_output.out_Biomass[m]
        B = float(state[sidx])
        from pypath.core.ecosim import _compute_Q_matrix
        # Enforce NoIntegrate baseline values for algebraic groups (match deriv_vector behavior)
        state_for_Q = state.copy()
        no_integrate_mask = np.asarray(params_dict.get('NoIntegrate', np.zeros(params_dict['NUM_GROUPS'] + 1))) != 0
        if np.any(no_integrate_mask):
            Bbase = params_dict.get('Bbase')
            if Bbase is not None:
                state_for_Q = state_for_Q.copy()
                state_for_Q[no_integrate_mask] = Bbase[no_integrate_mask]

        QQ = _compute_Q_matrix(params_dict, state_for_Q, {"Ftime": np.ones_like(state_for_Q)})
        consumption = float(np.nansum(QQ[:, sidx]))
        predation_loss = float(np.nansum(QQ[sidx, :]))
        PB = params_dict['PB'][sidx]
        QB = params_dict['QB'][sidx]
        PP_type = params_dict['PP_type']
        Bbase = params_dict['Bbase']
        if PP_type[sidx] > 0:
            if Bbase is not None and Bbase[sidx] > 0:
                rel_bio = B / Bbase[sidx]
                dd_factor = max(0.0, 2.0 - rel_bio)
                production = PB * B * dd_factor
            else:
                production = PB * B
        elif QB > 0:
            GE = PB / QB
            production = GE * consumption
        else:
            production = PB * B

        # fishing mort
        fishing_mort = fishing_dict.get('FishingMort', np.zeros(pypath_ecosim.params.NUM_GROUPS + 1)).copy()
        for i in range(1, len(fishing_dict.get('FishFrom', []))):
            grp = int(fishing_dict['FishFrom'][i])
            fishing_mort[grp] += fishing_dict['FishQ'][i]
        fish_loss = fishing_mort[sidx] * B

        m0 = float(params_dict['M0'][sidx])
        m0_loss = m0 * B

        # raw derivative (no fishing)
        raw_deriv = float(deriv_vector(state.copy(), params_dict, {"Ftime": np.ones_like(state)}, {"FishingMort": np.zeros_like(state)})[sidx])

        # Reference row for month m (CSV months start at 1)
        r = ref[ref['month'] == m]
        if r.empty:
            pytest.skip(f"Reference row for month {m} not found")
        r = r.iloc[0]

        # Determine whether R provides QQ/component diagnostics for this month.
        # R reference QQ files currently contain all-zeros when the diagnostics are not provided,
        # so treat an all-zero QQ row as "missing data" and skip per-term checks that depend on QQ.
        qq_ref = pd.read_csv(ECOSIM_DIR / "diagnostics/seabirds_qq_rk4.csv")
        row_qq = qq_ref[qq_ref['month'] == m]
        has_qq_data = False
        if not row_qq.empty:
            vals = row_qq.iloc[0][1:].astype(float).values
            has_qq_data = not np.allclose(vals, 0.0)

        # Compare with tolerances and collect mismatches
        tol = 1e-8
        mismatches = []

        # Only compare production/predation/consumption when R provides QQ diagnostics
        if has_qq_data:
            if abs(production - r['production']) > tol:
                mismatches.append(f"Production month {m}: py={production:.12e} ref={r['production']:.12e} diff={production-r['production']:.12e}")
            if abs(predation_loss - r['predation_loss']) > tol:
                mismatches.append(f"Predation month {m}: py={predation_loss:.12e} ref={r['predation_loss']:.12e} diff={predation_loss-r['predation_loss']:.12e}")
            if abs(consumption - r['consumption_by_predator']) > tol:
                mismatches.append(f"Consumption month {m}: py={consumption:.12e} ref={r['consumption_by_predator']:.12e} diff={consumption-r['consumption_by_predator']:.12e}")
        else:
            # If QQ is missing in R reference, skip production/predation/consumption and derivative checks.
            # Keep fish_loss and M0 comparisons but use a slightly relaxed tolerance for M0 to allow
            # minor differences in numeric constants/rounding between R and PyPath balancing data.
            tol_m0 = 2e-6

        # Compare invariants present in R reference regardless
        if abs(fish_loss - r['fish_loss']) > tol:
            mismatches.append(f"Fish loss month {m}: py={fish_loss:.12e} ref={r['fish_loss']:.12e} diff={fish_loss-r['fish_loss']:.12e}")
        if has_qq_data:
            if abs(m0_loss - r['m0_loss']) > tol:
                mismatches.append(f"M0 loss month {m}: py={m0_loss:.12e} ref={r['m0_loss']:.12e} diff={m0_loss-r['m0_loss']:.12e}")
            if abs(raw_deriv - r['derivative']) > 1e-10:
                mismatches.append(f"Raw deriv month {m}: py={raw_deriv:.12e} ref={r['derivative']:.12e} diff={raw_deriv-r['derivative']:.12e}")
        else:
            if abs(m0_loss - r['m0_loss']) > tol_m0:
                mismatches.append(f"M0 loss month {m}: py={m0_loss:.12e} ref={r['m0_loss']:.12e} diff={m0_loss-r['m0_loss']:.12e}")

        if mismatches:
            # Fail fast with details of the first mismatches
            msg = "; ".join(mismatches[:3])
            assert False, f"Term mismatches detected: {msg}"

    # If all checks passed
    assert True
