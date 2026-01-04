import numpy as np

from pypath.core.ecosim import rsim_run, rsim_state, rsim_params
from pypath.core.params import RpathParams


def make_model_with_qlinks():
    params = RpathParams()
    params.model = [{"Group": "A", "Type": 1}, {"Group": "B", "Type": 2}]
    # Add a simple Qlink definition: flow from A->detritus
    params.qlinks = [{"from": "A", "to": "detritus", "value": 0.1}]
    params.n_groups = 2
    return params


def test_annual_qlink_accumulation():
    params = make_model_with_qlinks()
    years = range(2)
    state = rsim_state(rsim_params(params))
    out = rsim_run(state, years)

    # Expect output includes annual Qlink totals
    assert hasattr(out, "annual_qlink") or hasattr(out, "annual_Qlink"), (
        "Ecosim output must include annual Qlink accumulation (attribute 'annual_qlink' or 'annual_Qlink')"
    )
