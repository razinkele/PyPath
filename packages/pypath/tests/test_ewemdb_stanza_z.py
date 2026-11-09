"""EwE stanza life stages take their total mortality Z from the Ecopath P/B.

``StanzaLifeStage.Mortality`` is 0 in LT2022; a Z of 0 gives a flat age
structure that puts almost no biomass in the young stages (2e-4 against an
Ecopath 15.0 for Blue mussel (juv.)). Rpath's EwE-derived REco example
carries Z == PB for every stanza.
"""

from pathlib import Path

import pytest

DATA_DIR = Path(__file__).parents[3] / "Data"
DB_PATH = DATA_DIR / "LT2022_0.5ST_final7.eweaccdb"

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def params():
    if not DB_PATH.exists():
        pytest.skip("LT2022 database not found")
    from pypath.io.ewemdb import EwEDatabaseError, read_ewemdb

    try:
        return read_ewemdb(str(DB_PATH))
    except EwEDatabaseError as e:
        pytest.skip(f"No EwE database driver: {e}")


def test_stanza_z_is_the_group_pb(params):
    model = params.model.set_index("Group")
    stindiv = params.stanzas.stindiv
    assert len(stindiv) == 2
    for _, row in stindiv.iterrows():
        assert row["Z"] == pytest.approx(float(model.loc[row["Group"], "PB"]))
    assert stindiv["Z"].min() > 0


def test_stanza_k_is_the_group_vbk(params):
    """EwE6 derives the age structure with ``EcopathGroup.vbK`` of the leading
    life stage's group (cCore.CalculateStanza, cEcoSimModel.InitStanza), not
    ``StanzaLifeStage.vbK``, where it writes 0.3 as a placeholder."""
    stgroups = params.stanzas.stgroups
    assert len(stgroups) == 1
    assert float(stgroups["VBGF_Ksp"].iloc[0]) == pytest.approx(1.175)


def test_derived_juvenile_biomass_matches_ewe6(params):
    """With K = 1.175 and Z = P/B the age structure reproduces the biomass EwE6
    wrote into the database for Blue mussel (juv.): 15.005259 (a verbatim port
    of EwE6's CalculateStanzaParameters gives 15.005255)."""
    from pypath.core.ecopath import rpath
    from pypath.core.ecosim import rsim_scenario

    model = rpath(params)
    sc = rsim_scenario(model, params, years=range(1, 3))
    st, p = sc.stanzas, sc.params
    juv = p.spname.index("Blue mussel (juv.)")
    for isp in range(1, int(st.n_split) + 1):
        for ist in range(1, int(st.n_stanzas[isp]) + 1):
            if int(st.ecopath_code[isp, ist]) != juv:
                continue
            a1, a2 = int(st.age1[isp, ist]), int(st.age2[isp, ist])
            derived = float(
                (
                    st.base_nage_s[a1 : a2 + 1, isp] * st.base_wage_s[a1 : a2 + 1, isp]
                ).sum()
            )
            # EwE6 closes the oldest age with a geometric plus-group
            # extrapolation; PyPath sums the tail explicitly. They agree to
            # 2e-5 (15.00556 vs 15.00526).
            assert derived == pytest.approx(15.005259, rel=1e-4)
            assert float(p.B_BaseRef[juv]) == pytest.approx(15.005259, rel=1e-5)


@pytest.mark.parametrize("leading", ["missing", 0, 99])
def test_stanza_table_without_leading_life_stage(params, monkeypatch, leading):
    """Older EwE6 databases (Curonian 'curonian pelagic-multistanza', converted
    from EwE5) have no Stanza.LeadingLifeStage column. EwE6 then leads with the
    last life stage (cEcoSimModel: BaseStanza <= 0 -> Nstanza), and clamps a
    value above the stage count to it (cStanzaDatastructures). The importer
    looked the column up directly, and the KeyError dropped every stanza
    group without a warning."""
    import pypath.io.ewemdb as ewemdb

    original = ewemdb.read_ewemdb_table

    def without_leading(filepath, table, *args, **kwargs):
        df = original(filepath, table, *args, **kwargs)
        if table == "Stanza":
            if leading == "missing":
                df = df.drop(columns=[c for c in df.columns if c.startswith("Leading")])
            else:
                df = df.copy()
                df["LeadingLifeStage"] = leading
        return df

    monkeypatch.setattr(ewemdb, "read_ewemdb_table", without_leading)
    old = ewemdb.read_ewemdb(str(DB_PATH))
    assert old.stanzas.n_stanza_groups == params.stanzas.n_stanza_groups == 1
    stindiv = old.stanzas.stindiv
    leading = stindiv[stindiv["Leading"].astype(bool)]
    assert leading["StanzaNum"].tolist() == [stindiv["StanzaNum"].max()]
    assert float(old.stanzas.stgroups["VBGF_Ksp"].iloc[0]) == pytest.approx(1.175)
