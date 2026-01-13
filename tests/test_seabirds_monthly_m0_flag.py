import numpy as np
import pandas as pd
import pytest

from pypath.core.ecosim import rsim_run
from tests.test_rpath_reference import ECOSIM_DIR


@pytest.mark.skipif(not ECOSIM_DIR.exists(), reason="Reference data not available")
def test_seabirds_monthly_m0_flag():
    """Diagnostic: run RK4 with monthly M0 on and off and report Seabirds correlation."""
    from pathlib import Path
    from pypath.core.ecopath import rpath
    from pypath.core.params import create_rpath_params
    from pypath.core.ecosim import rsim_scenario

    ECOPATH_DIR = Path("tests/data/rpath_reference/ecopath")

    # Prepare pypath_ecosim scenario locally (do not rely on external fixture)
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

    # Load Rpath trajectory
    rpath_traj = pd.read_csv(ECOSIM_DIR / "biomass_trajectory_rk4.csv")
    group_names = rpath_traj.columns[1:].tolist()
    try:
        seabirds_idx = group_names.index("Seabirds")
    except ValueError:
        pytest.skip("Seabirds not present in reference data")

    # Run with monthly M0 adjustment enabled (default)
    pypath_output_on = rsim_run(pypath_ecosim, method="RK4", years=range(1, 101))
    p_on = pypath_output_on.out_Biomass[:, seabirds_idx]

    # Run with monthly M0 adjustment disabled
    pypath_ecosim.params.MONTHLY_M0_ADJUST = False
    pypath_output_off = rsim_run(pypath_ecosim, method="RK4", years=range(1, 101))
    p_off = pypath_output_off.out_Biomass[:, seabirds_idx]

    # Reference
    r = rpath_traj["Seabirds"].values
    L = min(len(r), len(p_on), len(p_off))
    r_slice = r[:L]
    p_on_slice = p_on[:L]
    p_off_slice = p_off[:L]

    def corr(a, b):
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            return 1.0 if np.allclose(a, b, atol=1e-12) else np.nan
        return np.corrcoef(a, b)[0, 1]

    corr_on = corr(r_slice, p_on_slice)
    corr_off = corr(r_slice, p_off_slice)

    print(f"Seabirds correlation with monthly M0 ON : {corr_on:.6f}")
    print(f"Seabirds correlation with monthly M0 OFF: {corr_off:.6f}")

    # Basic sanity: correlations should be numeric
    assert not np.isnan(corr_on)
    assert not np.isnan(corr_off)


@pytest.mark.skipif(not ECOSIM_DIR.exists(), reason="Reference data not available")
def test_seabirds_find_first_divergence():
    """Find the first month where PyPath (RK4) and Rpath differ for Seabirds beyond tiny noise."""
    from pathlib import Path
    from pypath.core.ecopath import rpath
    from pypath.core.params import create_rpath_params
    from pypath.core.ecosim import rsim_scenario

    ECOPATH_DIR = Path("tests/data/rpath_reference/ecopath")
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

    rpath_traj = pd.read_csv(ECOSIM_DIR / "biomass_trajectory_rk4.csv")
    group_names = rpath_traj.columns[1:].tolist()
    try:
        seabirds_idx = group_names.index("Seabirds")
    except ValueError:
        pytest.skip("Seabirds not present in reference data")

    pypath_output = rsim_run(pypath_ecosim, method="RK4", years=range(1, 101))
    p = pypath_output.out_Biomass[:, seabirds_idx]
    r = rpath_traj["Seabirds"].values

    L = min(len(r), len(p))
    tol = 1e-12
    first = None
    for i in range(L):
        if not np.isclose(r[i], p[i], atol=tol, rtol=0):
            first = i
            break

    assert first is not None, "No divergence detected between PyPath and Rpath for Seabirds"
    print(f"First divergence at index {first}: Rpath={r[first]:.12e} PyPath={p[first]:.12e} diff={p[first]-r[first]:.12e}")


@pytest.mark.skipif(not ECOSIM_DIR.exists(), reason="Reference data not available")
def test_seabirds_initial_m0_persistence():
    """Check persisted initial M0 for Seabirds after rsim_run and compare to reference."""
    from pathlib import Path
    from pypath.core.ecopath import rpath
    from pypath.core.params import create_rpath_params
    from pypath.core.ecosim import rsim_scenario

    ECOPATH_DIR = Path("tests/data/rpath_reference/ecopath")
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

    before_m0 = float(pypath_ecosim.params.MzeroMort[sidx])
    print(f"Seabirds M0 before run: {before_m0:.12e}")

    # Run a short simulation to trigger initialization adjustments
    rsim_output = rsim_run(pypath_ecosim, method='RK4', years=range(1, 2))

    after_m0 = float(pypath_ecosim.params.MzeroMort[sidx])
    print(f"Seabirds M0 after run:  {after_m0:.12e}")

    # Check against reference value in ecosim_params.json
    import json
    ref = json.load(open('tests/data/rpath_reference/ecosim/ecosim_params.json'))
    ref_m0 = float(ref['MzeroMort'][sidx])
    print(f"Seabirds M0 reference:  {ref_m0:.12e}")

    # Ensure we persisted a numeric value
    assert np.isfinite(after_m0)
    # Report whether the persisted value equals the reference (it might differ)
    print(f"Seabirds M0 delta vs reference: {after_m0 - ref_m0:.12e}")
