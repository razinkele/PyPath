"""Pre-balance trophic levels (analysis.prebalance._calculate_trophic_levels).

TL = 1 for producers and detritus, and 1 + the diet-weighted trophic level
of the prey for consumers, with imports left out: the levels ``rpath``
computes for the balanced model. An RpathParams diet holds the prey names in
its "Group" column; the routine looked prey up by the table's row index, so
every group came out at TL 1 and the biomass slope and TL plots of the
pre-balance report ordered groups as entered.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from pypath.analysis.prebalance import (
    _calculate_trophic_levels,
    calculate_biomass_slope,
)
from pypath.core.ecopath import rpath
from pypath.core.params import create_rpath_params

BLACK_SEA = Path(__file__).parents[3] / "Data" / "BlackSea_EwEF_testcase.eweaccdb"


def _chain():
    """Fish eat zooplankton (0.8) and detritus (0.2); zooplankton eat phytoplankton.

    Entered top-down, with biomass falling up the chain.
    """
    params = create_rpath_params(["Fish", "Zoo", "Phyto", "Detritus"], [0, 0, 1, 2])
    params.model.loc[0, ["Biomass", "PB", "QB"]] = [1.0, 0.5, 3.0]
    params.model.loc[1, ["Biomass", "PB", "QB"]] = [10.0, 5.0, 20.0]
    params.model.loc[2, ["Biomass", "PB"]] = [100.0, 50.0]
    params.model.loc[3, "Biomass"] = 10.0
    diet = params.diet
    diet.loc[diet["Group"] == "Zoo", "Fish"] = 0.8
    diet.loc[diet["Group"] == "Detritus", "Fish"] = 0.2
    diet.loc[diet["Group"] == "Phyto", "Zoo"] = 1.0
    return params


def _rpath_tl(params):
    balanced = rpath(params)
    return dict(zip(params.model["Group"], balanced.TL))


def test_trophic_levels_follow_the_diet():
    tl = _calculate_trophic_levels(_chain())
    assert tl["Phyto"] == 1 and tl["Detritus"] == 1
    assert tl["Zoo"] == pytest.approx(2.0)
    assert tl["Fish"] == pytest.approx(1 + 0.8 * 2 + 0.2 * 1)


def test_trophic_levels_match_the_balanced_model():
    params = _chain()
    tl, want = _calculate_trophic_levels(params), _rpath_tl(params)
    for group in ("Fish", "Zoo", "Phyto", "Detritus"):
        assert tl[group] == pytest.approx(want[group], abs=1e-12), group


def test_a_diet_indexed_by_group_name_gives_the_same_levels():
    params = _chain()
    indexed = SimpleNamespace(model=params.model, diet=params.diet.set_index("Group"))
    assert (
        _calculate_trophic_levels(indexed) == _calculate_trophic_levels(params)
    ).all()


def test_the_biomass_slope_orders_groups_by_trophic_level():
    # Biomass 100 (phytoplankton), 10 (detritus), 10 (zooplankton), 1 (fish)
    # up the chain: a falling slope. In input order it would rise.
    assert calculate_biomass_slope(_chain()) < 0


@pytest.mark.integration
def test_black_sea_trophic_levels_match_the_balanced_model():
    if not BLACK_SEA.exists():
        pytest.skip(f"{BLACK_SEA.name} not found")
    from pypath.io.ewemdb import check_ewemdb_support, read_ewemdb

    if not any(check_ewemdb_support().values()):
        pytest.skip("No EwE database driver")
    params = read_ewemdb(str(BLACK_SEA))
    tl, want = _calculate_trophic_levels(params), _rpath_tl(params)
    living = params.model.loc[params.model["Type"] < 2, "Group"]
    assert max(tl[g] for g in living) > 4  # dolphins
    for group in living:
        assert tl[group] == pytest.approx(want[group], abs=1e-6), group


def test_a_mixotroph_is_weighted_by_its_consumer_share():
    # Type 0.4: 60 % consumer. rpath scales its diet by 1 - Type.
    params = create_rpath_params(["Fish", "Mixo", "Phyto", "Detritus"], [0, 0.4, 1, 2])
    params.model.loc[0, ["Biomass", "PB", "QB"]] = [1.0, 0.5, 3.0]
    params.model.loc[1, ["Biomass", "PB", "QB"]] = [10.0, 5.0, 20.0]
    params.model.loc[2, ["Biomass", "PB"]] = [100.0, 50.0]
    params.model.loc[3, "Biomass"] = 10.0
    diet = params.diet
    diet.loc[diet["Group"] == "Mixo", "Fish"] = 1.0
    diet.loc[diet["Group"] == "Phyto", "Mixo"] = 1.0
    tl, want = _calculate_trophic_levels(params), _rpath_tl(params)
    assert tl["Mixo"] == pytest.approx(1 + 0.6 * 1)
    for group in ("Fish", "Mixo"):
        assert tl[group] == pytest.approx(want[group], abs=1e-9), group
