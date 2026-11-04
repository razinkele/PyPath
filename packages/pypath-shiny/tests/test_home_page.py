"""Tests for pages/home.py — UI renders and server signature."""

import inspect
from pathlib import Path

from pypath_shiny.pages.home import home_server, home_ui


def test_home_ui_renders():
    assert home_ui() is not None


def test_home_server_exact_signature():
    p = list(inspect.signature(home_server).parameters)
    assert p == ["input", "output", "session", "model_data"]


def test_home_page_uses_notification_service_not_direct_ui_calls():
    """Guardrail: migrate home page notifications to service wrappers."""
    from pypath_shiny.pages import home

    source = Path(home.__file__).read_text(encoding="utf-8")
    assert "ui.notification_show(" not in source


def test_example_model_gives_biomass_or_ee_never_both():
    """With B, PB, QB and EE all set, Ecopath echoed the EEs unchecked; the
    example was not mass-balanced and Ecosim crashed Foragefish1, then Seals."""
    import numpy as np

    from pypath_shiny.pages.home import build_example_params

    m = build_example_params().model
    living = m[m["Type"] < 2]
    both = living["Group"][living["Biomass"].notna() & living["EE"].notna()]
    assert both.empty, f"over-specified groups: {list(both)}"
    assert np.all(living["Biomass"].notna() | living["EE"].notna())


def test_example_model_starts_ecosim_at_equilibrium():
    import warnings

    import numpy as np
    from pypath.core.ecopath import rpath
    from pypath.core.ecosim import rsim_run, rsim_scenario

    from pypath_shiny.pages.home import build_example_params

    params = build_example_params()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = rpath(params)
        out = rsim_run(
            rsim_scenario(model, params, years=range(1, 3)),
            method="RK4",
            years=range(1, 3),
        )
    living = model.NUM_LIVING
    assert np.all(model.EE[:living] <= 1.0)
    b = out.out_Biomass[:, 1 : living + 1]
    drift = np.abs(b[-1] / b[0] - 1.0)
    assert drift.max() < 1e-3, dict(zip(model.Group[:living], drift.round(4)))


def test_example_stanzas_follow_rpath_conventions():
    """Wmat is a fraction of Winf, stanzas don't overlap, and Z equals P/B.

    The example used Wmat=50 (nothing ever matured) and a juvenile Last of
    24 overlapping the adults' First of 24.
    """
    import pytest

    from pypath_shiny.pages.home import build_example_params

    params = build_example_params()
    sg = params.stanzas.stgroups.iloc[0]
    assert 0 < sg["Wmat"] < 1

    si = params.stanzas.stindiv.set_index("Group")
    juv, adu = si.loc["JuvRoundfish1"], si.loc["AduRoundfish1"]
    assert juv["Last"] + 1 == adu["First"]

    model = params.model.set_index("Group")
    for name, st in (("JuvRoundfish1", juv), ("AduRoundfish1", adu)):
        assert st["Z"] == pytest.approx(model.loc[name, "PB"])

    stanzas = pytest.importorskip("pypath.core.stanzas")
    if not hasattr(stanzas, "ecosim_stanza_params"):
        pytest.skip("needs pypath-ewe with ecosim_stanza_params")
    # The juvenile Biomass and Q/B must be the ones the stanza age
    # structure implies, so the stanza inputs agree with the Ecopath balance.
    esp = stanzas.ecosim_stanza_params(params.stanzas, params.model)
    stanzas.rpath_stanzas(params, stanza_params=esp)
    got = {s.group_name: s for s in esp.stanza_individuals}
    assert got["JuvRoundfish1"].biomass == pytest.approx(
        model.loc["JuvRoundfish1", "Biomass"], rel=1e-5
    )
    assert got["JuvRoundfish1"].qb == pytest.approx(
        model.loc["JuvRoundfish1", "QB"], rel=1e-5
    )
