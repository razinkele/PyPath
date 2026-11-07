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


def test_derived_juvenile_biomass_is_not_negligible(params):
    """With Z = P/B the juvenile stanza carries real biomass (5.3 of an
    Ecopath 15.0; the remaining gap is EwE's own derivation, see #52/F0)."""
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
            assert derived > 1.0
            assert derived < float(p.B_BaseRef[juv])
