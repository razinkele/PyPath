import numpy as np
import pandas as pd
from pypath.core.params import create_rpath_params
from pypath.core.ecopath import rpath
from pypath.core.ecosim import rsim_scenario, rsim_run


def test_nointegrate_groups_stay_at_baseline():
    model_df = pd.read_csv('tests/data/rpath_reference/ecopath/model_params.csv')
    diet_df = pd.read_csv('tests/data/rpath_reference/ecopath/diet_matrix.csv')
    params = create_rpath_params(model_df['Group'].tolist(), model_df['Type'].tolist())
    params.model = model_df
    params.diet = diet_df

    p = rpath(params)
    scenario = rsim_scenario(p, params, years=range(1, 3))  # 2 years = initial + 1 year
    out = rsim_run(scenario, method='RK4', years=range(1, 3))

    # For each NoIntegrate group verify month 1 biomass equals baseline
    noint = np.asarray(scenario.params.NoIntegrate) != 0
    Bbase = scenario.params.B_BaseRef
    # out.out_Biomass rows: 0 initial, 1 first month
    for idx in np.where(noint)[0]:
        got = out.out_Biomass[1, idx]
        exp = Bbase[idx]
        assert abs(got - exp) < 1e-5, (
            f"Group idx {idx} (NoIntegrate) did not remain at baseline: got {got} expected {exp} diff {abs(got-exp):.6e}"
        )
