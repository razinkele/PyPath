import numpy as np
import pandas as pd

from pypath.core.params import create_rpath_params
from pypath.core.ecopath import rpath
from pypath.core.ecosim import rsim_scenario, rsim_run, _build_active_link_matrix, _build_link_matrix
from pypath.core.ecosim_deriv import deriv_vector


def test_detritus_not_consumed_without_inputs():
    """Fail if any detritus group is consumed but has no DetFate sources."""
    ECOPATH_DIR = "tests/data/rpath_reference/ecopath"
    model_df = pd.read_csv(ECOPATH_DIR + "/model_params.csv")
    diet_df = pd.read_csv(ECOPATH_DIR + "/diet_matrix.csv")

    params = create_rpath_params(model_df['Group'].tolist(), model_df['Type'].tolist())
    params.model = model_df
    params.diet = diet_df

    r = rpath(params)
    scenario = rsim_scenario(r, params, years=range(1, 101))

    # Run the full 100-year rk4 run to reach late-month states
    out = rsim_run(scenario, method='rk4', years=range(1, 101))

    biom = out.out_Biomass

    # Identify living and dead group indices in global ordering
    living_global_idx = np.where((r.type == 0) | (r.type == 1))[0]
    dead_global_idx = np.where(r.type == 2)[0]

    # Check every month for any detritus consumed without DetFate or fish discard inputs
    rs = scenario.params
    det_from = rs.DetFrom
    det_to = rs.DetTo
    det_frac = rs.DetFrac
    fish_to = rs.FishTo

    for month_idx in range(biom.shape[0]):
        monthly_bio = biom[month_idx, :]
        for d_local, d_global in enumerate(dead_global_idx):
            # Compute detritus consumption by summing diet fractions times predator QB*B
            detcons = 0.0
            for col_idx, pred_global in enumerate(living_global_idx):
                frac = r.DC[d_global, col_idx]
                pred_bio_qb = monthly_bio[pred_global] * r.QB[pred_global]
                detcons += frac * pred_bio_qb

            # Links in DetFrom/DetTo are 1-indexed; expected det_to value for local index
            expected_det_to = rs.NUM_LIVING + d_local + 1
            det_link_idx = np.where(det_to == expected_det_to)[0]
            det_links_sum = float(np.sum(det_frac[det_link_idx])) if det_link_idx.size > 0 else 0.0

            # Fish links that target detritus (discards) also contribute
            fish_links_present = np.any(fish_to == expected_det_to) if fish_to.size > 0 else False

            # Fail if consumed but no incoming detritus links and no fish discard links
            assert not (
                detcons > 0 and np.isclose(det_links_sum, 0.0) and not fish_links_present
            ), (
                f"Month {month_idx}: Detritus '{r.Group[d_global]}' is consumed (detcons={detcons:.3e}) "
                f"but has no DetFrom/DetTo links (det_links_sum={det_links_sum:.3e}) and no FishTo links"
            )

    # Specific check: month 1199 Discards detritus should have positive inputs after accounting for fish links
    check_month = 1199 - 1  # zero-based index
    if biom.shape[0] > check_month:
        state = biom[check_month, :]
        # Reconstruct DetFrac matrix from DetFrom/DetTo and incorporate fish links (mirrors deriv_vector behavior)
        NUM_GROUPS = rs.NUM_GROUPS
        NUM_LIVING = rs.NUM_LIVING
        NUM_DEAD = rs.NUM_DEAD

        # base reconstruction
        mat = np.zeros((NUM_GROUPS + 1, NUM_DEAD + 1))
        det_from = rs.DetFrom
        det_to = rs.DetTo
        det_frac = rs.DetFrac
        for k in range(len(det_frac)):
            f = int(det_from[k])
            t = int(det_to[k])
            if t >= (NUM_LIVING + 1) and t <= (NUM_LIVING + NUM_DEAD) and f >= 0 and f <= NUM_GROUPS:
                det_col = t - NUM_LIVING
                mat[f, det_col] += det_frac[k]

        # incorporate fish links
        fish_from = rs.FishFrom
        fish_to = rs.FishTo
        fish_q = rs.FishQ
        for k in range(len(fish_from)):
            f = int(fish_from[k])
            t = int(fish_to[k])
            if t >= (NUM_LIVING + 1) and t <= (NUM_LIVING + NUM_DEAD) and f >= 0 and f <= NUM_GROUPS:
                det_col = t - NUM_LIVING
                src_idx = f
                fish_input = float(fish_q[k]) * float(state[src_idx])
                m0_arr = rs.MzeroMort
                qb_arr = rs.FtimeQBOpt
                unassim_arr = rs.UnassimRespFrac
                m0_pos = max(0.0, float(m0_arr[src_idx]))
                qb_loss = 0.0 if np.isnan(qb_arr[src_idx]) else float(qb_arr[src_idx])
                source_loss = m0_pos * float(state[src_idx]) + float(state[src_idx]) * qb_loss * float(unassim_arr[src_idx])
                frac = fish_input / (source_loss + 1e-30)
                mat[src_idx, det_col] += frac

        # Compute source losses as in ecopath
        living_idx = living_global_idx
        loss = np.zeros(NUM_GROUPS + 1)
        for idx in living_idx:
            m0_pos = max(0.0, float(rs.MzeroMort[idx]))
            qb_val = 0.0 if np.isnan(rs.FtimeQBOpt[idx]) else float(rs.FtimeQBOpt[idx])
            loss[idx] = m0_pos * float(state[idx]) + float(state[idx]) * qb_val * float(rs.UnassimRespFrac[idx])

        # Compute detinputs for Discards detritus (find local index)
        # Identify local index for 'Discards' group
        try:
            d_glob_idx = list(r.Group).index('Discards')
            d_local_idx = list(np.where(r.type == 2)[0]).index(d_glob_idx)
            detinputs = float(np.sum(loss[:, np.newaxis] * mat[:, d_local_idx]))
            assert detinputs > 0.0, f"Month {check_month}: Discards detritus inputs unexpectedly zero after including fish links"
        except ValueError:
            # No Discards group in this model
            pass


def test_deriv_includes_fish_discard_links():
    """Verify that deriv_vector applies fish discard mappings when provided in params dict."""
    ECOPATH_DIR = "tests/data/rpath_reference/ecopath"
    model_df = pd.read_csv(ECOPATH_DIR + "/model_params.csv")
    diet_df = pd.read_csv(ECOPATH_DIR + "/diet_matrix.csv")

    params = create_rpath_params(model_df['Group'].tolist(), model_df['Type'].tolist())
    params.model = model_df
    params.diet = diet_df

    r = rpath(params)
    scenario = rsim_scenario(r, params, years=range(1, 101))

    # Run the full 100-year rk4 run to reach late-month states
    out = rsim_run(scenario, method='rk4', years=range(1, 101))

    biom = out.out_Biomass

    # Choose a month to inspect (use last month if available)
    check_month = min(1199 - 1, biom.shape[0] - 1)
    state = biom[check_month, :]

    # Build params dict (without fish arrays)
    rs = scenario.params
    params_dict = {
        'NUM_GROUPS': rs.NUM_GROUPS,
        'NUM_LIVING': rs.NUM_LIVING,
        'NUM_DEAD': rs.NUM_DEAD,
        'NUM_GEARS': rs.NUM_GEARS,
        'PB': rs.PBopt,
        'QB': rs.FtimeQBOpt,
        'M0': rs.MzeroMort,
        'Unassim': rs.UnassimRespFrac,
        'ActiveLink': _build_active_link_matrix(rs),
        'VV': _build_link_matrix(rs, rs.VV),
        'DD': _build_link_matrix(rs, rs.DD),
        'QQbase': _build_link_matrix(rs, rs.QQ),
        'Bbase': rs.B_BaseRef,
        'PP_type': rs.PP_type,
    }

    fishing = {'FishingMort': np.zeros(rs.NUM_GROUPS + 1)}
    forcing0 = {'Ftime': scenario.start_state.Ftime.copy(), 'ForcedBio': np.zeros(rs.NUM_GROUPS + 1), 'ForcedEffort': np.ones(rs.NUM_GEARS + 1)}

    # Compute deriv without fish arrays
    d_no_fish = deriv_vector(state.copy(), params_dict.copy(), forcing0, fishing)

    # Now add fish arrays to params dict and compute again
    params_with_fish = params_dict.copy()
    params_with_fish['FishFrom'] = getattr(rs, 'FishFrom', np.array([]))
    params_with_fish['FishTo'] = getattr(rs, 'FishTo', np.array([]))
    params_with_fish['FishQ'] = getattr(rs, 'FishQ', np.array([]))

    d_with_fish = deriv_vector(state.copy(), params_with_fish, forcing0, fishing)

    # If there is a Discards group, its detritus derivative should increase when fish links are present
    try:
        d_glob_idx = list(r.Group).index('Discards')
        # Find corresponding detritus global index
        if np.any(r.type == 2):
            # detritus global indices
            dead_globals = np.where(r.type == 2)[0]
            if d_glob_idx in dead_globals:
                # detritus index in global ordering
                det_global_idx = d_glob_idx
                # Compare derivatives - ensure adding fish links does not reduce the derivative
                assert d_with_fish[det_global_idx] >= d_no_fish[det_global_idx], "With fish discard links, detritus derivative should not be smaller"
                # Verify that fish-derived DetFrac contributions would be computed (logic copied from deriv_vector)
                fish_from = getattr(rs, 'FishFrom', None)
                fish_to = getattr(rs, 'FishTo', None)
                fish_q = getattr(rs, 'FishQ', None)
                found_positive_frac = False
                if fish_from is not None and fish_to is not None and fish_q is not None:
                    for k in range(len(fish_from)):
                        f = int(fish_from[k])
                        t = int(fish_to[k])
                        if t >= (rs.NUM_LIVING + 1) and t <= (rs.NUM_LIVING + rs.NUM_DEAD) and f >= 0 and f <= rs.NUM_GROUPS:
                            det_col = t - rs.NUM_LIVING
                            src_idx = f
                            fish_input = float(fish_q[k]) * float(state[src_idx])
                            m0_arr = rs.MzeroMort
                            qb_arr = rs.FtimeQBOpt
                            unassim_arr = rs.UnassimRespFrac
                            m0_pos = max(0.0, float(m0_arr[src_idx]))
                            qb_loss = 0.0 if np.isnan(qb_arr[src_idx]) else float(qb_arr[src_idx])
                            source_loss = m0_pos * float(state[src_idx]) + float(state[src_idx]) * qb_loss * float(unassim_arr[src_idx])
                            frac = fish_input / (source_loss + 1e-30)
                            if frac > 0:
                                found_positive_frac = True
                                break
                # If fish links are present in the scenario, at least one positive fraction should be computable
                if getattr(rs, 'FishFrom', None) is not None and len(getattr(rs, 'FishFrom', [])) > 1:
                    assert found_positive_frac, "No positive fish-derived DetFrac fraction computed for available FishFrom links"
    except ValueError:
        # No Discards group - skip
        pass
