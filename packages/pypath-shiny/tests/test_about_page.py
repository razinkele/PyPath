"""Tests for pages/about.py — _get_version helper."""

import inspect

from pypath_shiny.pages.about import _get_version, about_server, about_ui


def test_get_version_installed_package():
    v = _get_version("pypath-ewe")
    assert isinstance(v, str) and v != "N/A"


def test_get_version_unknown_package():
    v = _get_version("nonexistent-package-xyz-abc")
    assert v == "N/A"


def test_get_version_returns_string():
    v = _get_version("pypath-shiny")
    assert isinstance(v, str)


def test_about_ui_renders():
    result = about_ui()
    assert result is not None


def test_about_server_has_three_params():
    params = list(inspect.signature(about_server).parameters.keys())
    assert "input" in params
    assert "output" in params
    assert "session" in params
    assert len(params) == 3


def test_about_shows_the_versions_of_the_running_code(monkeypatch):
    """The two PyPath versions come from the code that is running, as the top
    bar's does. Read from installed-package metadata they showed an old or
    missing version wherever the app ran from a source tree."""
    import pypath

    import pypath_shiny
    from pypath_shiny.pages import about

    def no_metadata(name):
        raise LookupError(name)

    monkeypatch.setattr(about, "pkg_version", no_metadata)
    html = str(about_ui())
    assert f">{pypath.__version__}<" in html
    assert f">{pypath_shiny.__version__}<" in html


def test_about_lists_the_shipped_example_models():
    from pypath.examples import list_example_models

    html = str(about_ui())
    models = list_example_models()
    assert models
    for model in models:
        assert model.title in html


def test_every_reference_on_about_links_to_its_doi():
    from pypath_shiny.pages import about

    html = str(about_ui())
    assert len(about.REFERENCES) >= 4
    for reference in about.REFERENCES:
        assert reference.doi.startswith("10."), reference
        assert f'href="https://doi.org/{reference.doi}"' in html
        assert reference.title in html


def test_about_page_classes_work_in_the_dark_theme():
    """``bg-light`` stays white under the dark theme, where inline code is
    light: the equations were unreadable."""
    assert "bg-light" not in str(about_ui())
