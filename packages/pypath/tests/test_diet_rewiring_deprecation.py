import warnings

from pypath.core.forcing import DietRewiring, create_diet_rewiring


def _deprecations(fn):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        fn()
    return [x for x in w if issubclass(x.category, DeprecationWarning)]


def test_enabled_diet_rewiring_warns_once_at_the_caller():
    got = _deprecations(lambda: DietRewiring(enabled=True))
    assert len(got) == 1
    assert str(got[0].message).startswith("DietRewiring is deprecated")
    assert "SwitchPower" in str(got[0].message)
    assert got[0].filename == __file__


def test_disabled_diet_rewiring_is_silent():
    assert _deprecations(lambda: DietRewiring()) == []
    assert _deprecations(lambda: DietRewiring(enabled=False)) == []


def test_create_diet_rewiring_warns_once_at_its_caller():
    got = _deprecations(lambda: create_diet_rewiring())
    assert len(got) == 1 and got[0].filename == __file__
