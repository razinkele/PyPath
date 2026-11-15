"""HandleSelf / ScrambleSelf reach the consumption kernel per predator.

Rpath's ``rsim.params`` builds both as one weight per group
(``rep(x, NUM_GROUPS + 1)``) and its C++ unrolls them by the link's predator
(``ecosim.cpp``: ``Hself = HandleSelf[PreyTo]``, ``Sself =
ScrambleSelf[PreyTo]``). ``rsim_run``'s setup must therefore give every link
of a predator that predator's weight, and no other link.
"""

import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from pypath.core import ecosim
from pypath.core.ecosim import (
    RsimScenario,
    _build_link_matrix,
    _run_setup,
    rsim_fishing,
    rsim_forcing,
    rsim_params,
    rsim_state,
)
from tests.continuous_support import needs_reco, reco_plain

YEARS = range(1, 3)


def _params(**weights):
    model, _ = reco_plain()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return rsim_params(model, **weights)


def _params_dict(p):
    scen = RsimScenario(
        params=p,
        start_state=rsim_state(p),
        forcing=rsim_forcing(p, YEARS),
        fishing=rsim_fishing(p, YEARS),
        start_year=YEARS[0],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _run_setup(scen, YEARS).params_dict


def _links(p):
    return np.asarray(p.PreyFrom, dtype=int), np.asarray(p.PreyTo, dtype=int)


@needs_reco
def test_the_weights_reach_every_link_through_rsim_params():
    p = _params(scrambleselfwt=0.5, handleselfwt=0.25)
    n = p.NUM_GROUPS + 1
    assert len(p.ScrambleSelf) == n and len(p.HandleSelf) == n  # per group
    assert len(p.PreyFrom) > n  # REco: more links than groups
    pd = _params_dict(p)
    prey, pred = _links(p)
    linked = np.zeros((n, n), dtype=bool)
    linked[prey, pred] = True
    for name, value in (("ScrambleSelf", 0.5), ("HandleSelf", 0.25)):
        m = np.asarray(pd[name])
        assert m.shape == (n, n)
        assert np.all(m[prey, pred] == value), name
        assert np.all(m[~linked] == 0.0), name


@needs_reco
def test_a_group_weight_pools_its_own_links_only():
    p = _params()
    n = p.NUM_GROUPS + 1
    prey, pred = _links(p)
    counts = np.bincount(pred[1:], minlength=n)
    g = int(np.argmax(counts))  # the predator with the most prey
    h = int(np.argsort(counts)[-2])  # another predator with several prey
    assert counts[g] >= 2 and counts[h] >= 2 and g != h
    p.ScrambleSelf = np.zeros(n)
    p.ScrambleSelf[g] = 0.7
    p.HandleSelf = np.zeros(n)
    p.HandleSelf[h] = 0.3
    pd = _params_dict(p)
    for name, grp, value in (("ScrambleSelf", g, 0.7), ("HandleSelf", h, 0.3)):
        m = np.asarray(pd[name])
        expected = np.zeros((n, n))
        expected[prey[pred == grp], grp] = value
        np.testing.assert_array_equal(m, expected, err_msg=name)


@needs_reco
def test_a_per_link_array_is_accepted_unchanged():
    p = _params()
    nlinks = len(p.PreyFrom)
    assert nlinks != p.NUM_GROUPS + 1
    p.ScrambleSelf = np.linspace(0.0, 0.9, nlinks)
    p.HandleSelf = np.linspace(0.9, 0.0, nlinks)
    pd = _params_dict(p)
    for name in ("ScrambleSelf", "HandleSelf"):
        expected = _build_link_matrix(p, getattr(p, name))
        np.testing.assert_array_equal(pd[name], expected, err_msg=name)


@needs_reco
def test_an_array_of_another_length_is_refused():
    p = _params()
    p.ScrambleSelf = np.zeros(5)
    with pytest.raises(ValueError, match="ScrambleSelf"):
        _params_dict(p)


def test_per_group_wins_when_the_lengths_coincide():
    """Three groups and four links (the dummy link 0 included): length 4 is
    both NUM_GROUPS + 1 and len(PreyFrom); it is read per group, the form
    rsim_params builds."""
    p = SimpleNamespace(
        NUM_GROUPS=3,
        PreyFrom=np.array([0, 1, 1, 2]),
        PreyTo=np.array([0, 2, 3, 3]),
    )
    with pytest.warns(UserWarning, match="per group"):
        m = ecosim._group_link_matrix(p, np.array([0.0, 0.0, 0.2, 0.4]), "ScrambleSelf")
    expected = np.zeros((4, 4))
    expected[1, 2] = 0.2
    expected[1, 3] = 0.4
    expected[2, 3] = 0.4
    np.testing.assert_array_equal(m, expected)


def test_no_ambiguity_warning_for_a_uniform_array_or_unequal_lengths():
    """A uniform array gives the same matrix read either way; with lengths
    that differ the reading is not ambiguous."""
    tie = SimpleNamespace(
        NUM_GROUPS=3, PreyFrom=np.array([0, 1, 1, 2]), PreyTo=np.array([0, 2, 3, 3])
    )
    other = SimpleNamespace(
        NUM_GROUPS=3, PreyFrom=np.array([0, 1, 1]), PreyTo=np.array([0, 2, 3])
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ecosim._group_link_matrix(tie, np.full(4, 0.3), "HandleSelf")
        ecosim._group_link_matrix(other, np.array([0.0, 0.1, 0.2, 0.4]), "HandleSelf")
        ecosim._group_link_matrix(other, np.array([0.0, 0.1, 0.2]), "HandleSelf")
