"""EwE anchors stanza consumption on its own life stage, ``Stanza.LeadingCB``.

EwE6 takes the biomass from the leading life stage (``LeadingLifeStage``,
BaseStanza) and the consumption from the leading CB stage (``LeadingCB``,
BaseStanzaCB; cEcoSimModel.vb 4180-4187). The CB stage keeps its Q/B, and
the others follow from K = Q/B_cb x B_cb / sum(N x WWa) over its months
(vb 4384-4398), the CB stage's biomass being the derived one. Unset (<= 0)
it is the last stage, like LeadingLifeStage; Rpath has no such setting and
uses the leading stanza for both.
"""

import copy
from pathlib import Path

import numpy as np
import pytest

from pypath.core.stanzas import create_stanza_params, rpath_stanzas

from .test_stanza_plus_group import _mussel_params

DATA_DIR = Path(__file__).parents[3] / "Data"
LT2022 = DATA_DIR / "LT2022_0.5ST_final7.eweaccdb"
EDGARAS = DATA_DIR / "Edgaras model.ewemdb"

# LT2022's blue mussel as EwE stores it: adults lead the biomass (27, Q/B 8),
# and the juveniles' derived values are 15.005259 and 13.620459.
JUV_B, JUV_QB, AD_QB = 15.005259, 13.620459, 8.0


def _with_leading_cb(params, cb_stage):
    """Copy of the mussel stanza tables with the CB anchor on ``cb_stage``."""
    sp = params.stanzas
    groups = [
        {
            "stanza_group_num": 1,
            "n_stanzas": 2,
            "vbgf_ksp": sp.stanza_groups[0].vbgf_ksp,
        }
    ]
    indivs = [
        {
            "stanza_group_num": 1,
            "stanza_num": s.stanza_num,
            "group_num": s.group_num,
            "group_name": s.group_name,
            "first": s.first,
            "last": s.last,
            "z": s.z,
            "leading": s.leading,
            "leading_cb": s.stanza_num == cb_stage,
        }
        for s in sp.stanza_individuals
    ]
    params.stanzas = create_stanza_params(groups, indivs)
    return params


@pytest.mark.parametrize("rule", ["ewe", "rpath"])
def test_consumption_follows_the_leading_cb_stage(rule):
    """Juveniles anchor consumption (Q/B 13.620459) while adults lead the
    biomass (27): the adults' Q/B is derived, 8.0 under EwE's table, and the
    juveniles keep theirs."""
    params = _with_leading_cb(_mussel_params(), cb_stage=1)
    params.model.loc[1, "QB"] = JUV_QB
    params.model.loc[2, "QB"] = 99.0  # not the anchor: overwritten
    rpath_stanzas(params, rule=rule)
    model = params.model.set_index("Group")
    assert model.loc["Juv", "QB"] == pytest.approx(JUV_QB, rel=1e-12)
    assert model.loc["Adult", "Biomass"] == pytest.approx(27.0, rel=1e-12)
    if rule == "ewe":
        assert model.loc["Juv", "Biomass"] == pytest.approx(JUV_B, rel=1e-6)
        assert model.loc["Adult", "QB"] == pytest.approx(AD_QB, rel=1e-6)
    else:
        # Rpath's table, anchored the same way: the consumption index ratio.
        table = params.stanzas.st_groups[1]
        juv = table["age"] <= 11
        ratio = (
            table.loc[~juv, "Q"].sum()
            / table.loc[~juv, "B"].sum()
            / (table.loc[juv, "Q"].sum() / table.loc[juv, "B"].sum())
        )
        assert model.loc["Adult", "QB"] == pytest.approx(JUV_QB * ratio, rel=1e-12)


def test_without_leading_cb_the_leading_stage_anchors_consumption():
    params = _mussel_params()
    params.model.loc[1, "QB"] = 99.0
    rpath_stanzas(params, rule="ewe")
    model = params.model.set_index("Group")
    assert model.loc["Adult", "QB"] == pytest.approx(AD_QB, rel=1e-12)
    assert model.loc["Juv", "QB"] == pytest.approx(JUV_QB, rel=1e-6)


def test_leading_cb_column_reaches_the_stanza_parameters():
    """The DataFrame form (RpathParams.stanzas, as the EwE reader fills it)
    carries LeadingCB; without the column it follows Leading."""
    import pandas as pd

    from pypath.core.params import RpathStanzaParams
    from pypath.core.stanzas import ecosim_stanza_params

    params = _mussel_params()
    stgroups = pd.DataFrame(
        {"StGroupNum": [1], "StanzaGroup": ["Mussel"], "VBGF_Ksp": [1.175]}
    )
    stindiv = pd.DataFrame(
        {
            "StGroupNum": [1, 1],
            "StanzaNum": [1, 2],
            "Group": ["Juv", "Adult"],
            "First": [0, 12],
            "Last": [11, 999],
            "Z": [1.94, 1.75],
            "Leading": [False, True],
            "LeadingCB": [True, False],
        }
    )
    tables = RpathStanzaParams(n_stanza_groups=1, stgroups=stgroups, stindiv=stindiv)
    sp = ecosim_stanza_params(tables, params.model)
    assert [s.leading_cb for s in sp.stanza_individuals] == [True, False]
    tables.stindiv = stindiv.drop(columns=["LeadingCB"])
    sp = ecosim_stanza_params(tables, params.model)
    assert [s.leading_cb for s in sp.stanza_individuals] == [False, True]


def _read(path):
    if not path.exists():
        pytest.skip(f"{path.name} not found")
    from pypath.io.ewemdb import EwEDatabaseError, read_ewemdb

    try:
        return read_ewemdb(str(path))
    except EwEDatabaseError as e:
        pytest.skip(f"No EwE database driver: {e}")


@pytest.mark.integration
def test_reader_reads_leading_cb():
    """LT2022 anchors on stage 2 (adults); Edgaras' Karšis leads its biomass
    from stage 1 and its consumption from stage 6."""
    lt = _read(LT2022).stanzas.stindiv
    assert lt.loc[lt["LeadingCB"].astype(bool), "StanzaNum"].tolist() == [2]
    ed = _read(EDGARAS).stanzas.stindiv
    karsis = ed[ed["StanzaGroup"].str.strip() == "Karšis"]
    assert karsis.loc[karsis["Leading"].astype(bool), "StanzaNum"].tolist() == [1]
    assert karsis.loc[karsis["LeadingCB"].astype(bool), "StanzaNum"].tolist() == [6]
    other = ed[ed["StanzaGroup"].str.strip() != "Karšis"]
    assert (other["LeadingCB"].astype(bool) == other["Leading"].astype(bool)).all()


@pytest.mark.integration
def test_edgaras_karsis_consumption_follows_stage_six():
    """Raising Karšis' stage-6 Q/B by 10 % raises every other stage's derived
    Q/B by 10 %; changing stage 1 (the biomass lead) changes nothing, because
    its Q/B is derived from stage 6."""
    base = _read(EDGARAS)
    st = base.stanzas.stindiv
    karsis = st[st["StanzaGroup"].str.strip() == "Karšis"].sort_values("StanzaNum")
    names = [str(g) for g in karsis["Group"]]
    rows = {n: base.model.index[base.model["Group"] == n][0] for n in names}
    stored = {n: float(base.model.loc[rows[n], "QB"]) for n in names}

    def derived(edit_group, factor):
        p = copy.deepcopy(base)
        p.model.loc[rows[edit_group], "QB"] *= factor
        rpath_stanzas(p, rule="ewe")
        return {n: float(p.model.loc[rows[n], "QB"]) for n in names}

    up = derived(names[5], 1.1)
    for n in names:
        assert up[n] == pytest.approx(1.1 * stored[n], rel=1e-5), n
    same = derived(names[0], 1.1)
    for n in names:
        assert same[n] == pytest.approx(stored[n], rel=1e-5), n
    assert np.isfinite(list(same.values())).all()
