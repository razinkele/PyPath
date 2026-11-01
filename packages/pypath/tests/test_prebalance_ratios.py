"""Pre-balance predator/prey and vital-rate ratios.

Both functions read prey names from ``diet[predator][diet > 0].index``, but
``RpathParams.diet`` keeps prey names in its ``Group`` column with a plain
0..n index. Prey "names" were integers, never matched a group, and every
predator was skipped, so the ratio tables were always empty (seen in the
Shiny Pre-Balance tabs). Nothing tested these functions before.
"""

import numpy as np
import pytest

from pypath.analysis.prebalance import (
    calculate_predator_prey_ratios,
    calculate_vital_rate_ratios,
    generate_prebalance_report,
)
from pypath.core.params import create_rpath_params


@pytest.fixture
def params():
    """Phyto <- Zoo <- Fish (Fish also eats Phyto), plus detritus."""
    p = create_rpath_params(["Phyto", "Zoo", "Fish", "Detritus"], [1, 0, 0, 2])
    m = p.model
    m["Biomass"] = [10.0, 5.0, 2.0, 100.0]
    m["PB"] = [100.0, 20.0, 1.0, np.nan]
    m["QB"] = [np.nan, 60.0, 5.0, np.nan]
    d = p.diet.set_index("Group")
    d.loc["Phyto", "Zoo"] = 1.0
    d.loc["Zoo", "Fish"] = 0.8
    d.loc["Phyto", "Fish"] = 0.2
    p.diet = d.reset_index()
    return p


def _by_predator(df):
    return df.set_index("Predator")


def test_predator_prey_ratios(params):
    r = _by_predator(calculate_predator_prey_ratios(params))
    assert set(r.index) == {"Zoo", "Fish"}  # Phyto has no diet
    assert r.loc["Zoo", "Prey_Biomass"] == pytest.approx(10.0)
    assert r.loc["Zoo", "Ratio"] == pytest.approx(0.5)
    assert r.loc["Fish", "Prey_Biomass"] == pytest.approx(15.0)
    assert r.loc["Fish", "Ratio"] == pytest.approx(2.0 / 15.0)


def test_pb_ratios(params):
    r = _by_predator(calculate_vital_rate_ratios(params, "PB"))
    assert r.loc["Zoo", "Ratio"] == pytest.approx(20.0 / 100.0)
    assert r.loc["Fish", "Prey_Rate_Mean"] == pytest.approx(60.0)
    assert r.loc["Fish", "Ratio"] == pytest.approx(1.0 / 60.0)


def test_qb_ratios_skip_prey_without_rate(params):
    r = _by_predator(calculate_vital_rate_ratios(params, "QB"))
    # Zoo's only prey (Phyto) has no Q/B, so Zoo has no ratio
    assert set(r.index) == {"Fish"}
    assert r.loc["Fish", "Ratio"] == pytest.approx(5.0 / 60.0)


def test_diet_indexed_by_group_also_works(params):
    params.diet = params.diet.set_index("Group")
    r = _by_predator(calculate_predator_prey_ratios(params))
    assert r.loc["Zoo", "Ratio"] == pytest.approx(0.5)


def test_report_has_ratio_tables(params):
    report = generate_prebalance_report(params)
    assert len(report["predator_prey_ratios"]) == 2
    assert len(report["pb_ratios"]) == 2
    assert len(report["qb_ratios"]) == 1
    assert isinstance(report["warnings"], list)
