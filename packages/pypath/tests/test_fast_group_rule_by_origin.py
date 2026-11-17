"""The Ecosim fast-group rule defaults by the model's origin (roadmap D1).

``compat == "ewe"`` (EwE, EwE-F and EcoBase imports) gives the EwE rule;
anything else (models built from CSV or in code) gives Rpath's. An explicit
``fast_group_rule`` always wins.
"""

import copy
import warnings
from types import SimpleNamespace

import pytest

from pypath.core.ecosim import default_fast_group_rule, rsim_params, rsim_scenario
from tests.continuous_support import needs_reco, reco_plain

YEARS = range(1, 3)


def _flags(scen):
    p = scen.params
    return p.RK4_FAST_EQUILIBRIUM, p.FTIME_START_OF_MONTH


def _scenario(model, params, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return rsim_scenario(model, params, years=YEARS, **kwargs)


# --- the helper ---------------------------------------------------------------


@pytest.mark.parametrize(
    "compat, expected", [("ewe", "ewe"), ("rpath", "rpath"), ("other", "rpath")]
)
def test_helper_follows_the_models_compat(compat, expected):
    assert default_fast_group_rule(SimpleNamespace(compat=compat)) == expected


def test_helper_treats_a_missing_compat_as_rpath():
    assert default_fast_group_rule(SimpleNamespace()) == "rpath"
    assert default_fast_group_rule(None) == "rpath"


def test_params_compat_takes_precedence_over_the_balanced_models():
    ewe, rpath_ = SimpleNamespace(compat="ewe"), SimpleNamespace(compat="rpath")
    assert default_fast_group_rule(rpath_, ewe) == "ewe"
    assert default_fast_group_rule(ewe, rpath_) == "rpath"


def test_params_without_compat_fall_back_to_the_balanced_model():
    model = SimpleNamespace(compat="ewe")
    assert default_fast_group_rule(model, SimpleNamespace()) == "ewe"
    assert default_fast_group_rule(model, None) == "ewe"


def test_helper_is_exported_beside_rsim_scenario():
    import pypath
    import pypath.core

    for module in (pypath, pypath.core):
        assert module.default_fast_group_rule is default_fast_group_rule
        assert "default_fast_group_rule" in module.__all__


# --- rsim_scenario ------------------------------------------------------------


@needs_reco
def test_reco_from_csv_gets_rpaths_flags_by_default():
    model, params = reco_plain()
    assert params.compat == "rpath" and model.compat == "rpath"
    scen = _scenario(model, params)
    assert _flags(scen) == (False, False)
    assert scen.params.RK4_STEPS == 4


@needs_reco
def test_an_explicit_ewe_rule_wins_for_reco():
    model, params = reco_plain()
    scen = _scenario(model, params, fast_group_rule="ewe")
    assert _flags(scen) == (True, True)
    assert scen.params.RK4_STEPS == 1


@needs_reco
def test_reco_with_ewe_compat_gets_the_ewe_rule():
    model, params = reco_plain()
    # Copies: the cached REco objects are shared and must not change.
    params = copy.copy(params)
    params.compat = "ewe"
    scen = _scenario(model, params)
    assert _flags(scen) == (True, True)


@needs_reco
def test_an_explicit_rpath_rule_wins_over_ewe_compat():
    model, params = reco_plain()
    params = copy.copy(params)
    params.compat = "ewe"
    scen = _scenario(model, params, fast_group_rule="rpath")
    assert _flags(scen) == (False, False)


def test_an_ewef_example_still_gets_the_ewe_rule():
    from pypath.examples import example_ecosim_scenario

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scen = example_ecosim_scenario("Central_Baltic", years=YEARS)
    assert _flags(scen) == (True, True)
    assert scen.params.STANZA_WALFORD_PLUS_GROUP


# --- rsim_params --------------------------------------------------------------


@needs_reco
@pytest.mark.parametrize("compat, ewe", [("rpath", False), ("ewe", True)])
def test_rsim_params_follows_the_balanced_models_compat(compat, ewe):
    model, _ = reco_plain()
    model = copy.copy(model)
    model.compat = compat
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = rsim_params(model)
    assert (p.RK4_FAST_EQUILIBRIUM, p.FTIME_START_OF_MONTH) == (ewe, ewe)


def test_rsim_params_defaults_its_rule_to_none():
    import inspect

    param = inspect.signature(rsim_params).parameters["fast_group_rule"]
    assert param.default is None


@needs_reco
@pytest.mark.parametrize("bad", ["EwE", "both", ""])
def test_an_invalid_explicit_rule_still_raises(bad):
    model, params = reco_plain()
    with pytest.raises(ValueError, match="fast_group_rule"):
        rsim_params(model, fast_group_rule=bad)
    with pytest.raises(ValueError, match="fast_group_rule"):
        _scenario(model, params, fast_group_rule=bad)
