from pathlib import Path

import numpy as np

from pypath_shiny.pages import diet_rewiring_demo as demo
from pypath_shiny.pages.diet_rewiring_demo import (
    build_prey_switching_code,
    switching_diet,
)


def _oracle(b, s, q0=(0.5, 0.3, 0.2), v=2.0):
    b, q0 = np.asarray(b, float), np.asarray(q0, float)
    r = (q0 * b**s / np.sum(q0 * b**s)) / (q0 / q0.sum())
    q = q0 * b * r * v / (r + v - 1.0)
    return q / q.sum()


def test_base_diet_at_base_abundance():
    for s in (0.0, 1.0, 2.0):
        np.testing.assert_allclose(
            switching_diet([1, 1, 1], s), [0.5, 0.3, 0.2], rtol=1e-12
        )


def test_matches_ewe_formula():
    for s in (0.0, 0.5, 1.0, 2.0):
        np.testing.assert_allclose(
            switching_diet([2.0, 1.0, 0.5], s),
            _oracle([2.0, 1.0, 0.5], s),
            rtol=1e-10,
        )


def test_stronger_power_turns_more_to_the_abundant_prey():
    assert switching_diet([2, 1, 1], 2.0)[0] > switching_diet([2, 1, 1], 0.0)[0]


def test_demo_uses_the_core_function_not_diet_rewiring():
    src = Path(demo.__file__).read_text(encoding="utf-8")
    assert "apply_prey_switching" in src and "DietRewiring" not in src


def test_code_snippet_sets_switchpower():
    code = build_prey_switching_code(1.5)
    assert "SwitchPower" in code and "1.5" in code and "rsim_run" in code
    compile(code, "<switching_example>", "exec")


def test_code_snippet_runs_as_written():
    import warnings

    code = build_prey_switching_code(1.5)
    ns: dict = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exec(compile(code, "<switching_example>", "exec"), ns)
    names = [str(n).strip() for n in ns["scenario"].params.spname]
    assert ns["scenario"].params.SwitchPower[names.index(ns["predator"])] == 1.5
    assert ns["result"].out_Biomass.shape[0] > 1
    assert np.all(np.isfinite(ns["result"].out_Biomass))
