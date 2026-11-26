"""Tests for pages/analysis.py — UI renders and server signature."""

import inspect

from pypath_shiny.pages.analysis import analysis_server, analysis_ui


def test_analysis_ui_renders():
    assert analysis_ui() is not None


def test_analysis_server_exact_signature():
    p = list(inspect.signature(analysis_server).parameters)
    assert p == ["input", "output", "session", "model_data", "sim_results"]


def _texts(fig) -> str:
    return "\n".join(t.get_text() for ax in fig.axes for t in ax.texts)


def _balanced_example():
    import warnings

    from pypath.core.ecopath import rpath

    from pypath_shiny.pages.home import build_example_params

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return rpath(build_example_params())


def test_an_unbalanced_model_gets_a_message_where_the_food_web_would_be():
    # the plot area was left empty, with nothing to say why
    import matplotlib.pyplot as plt

    from pypath_shiny.pages.analysis import foodweb_figure

    fig = foodweb_figure(None)
    try:
        assert "Balance Model" in _texts(fig)
        assert "Ecopath Model" in _texts(fig)  # the page the button is on
    finally:
        plt.close(fig)


def test_an_unbalanced_model_gets_a_message_where_the_spectrum_would_be():
    import matplotlib.pyplot as plt

    from pypath_shiny.pages.analysis import trophic_spectrum_figure

    fig = trophic_spectrum_figure(None, "biomass")
    try:
        assert "Balance Model" in _texts(fig)
    finally:
        plt.close(fig)


def test_a_balanced_model_still_gets_its_plots():
    import matplotlib.pyplot as plt

    from pypath_shiny.pages.analysis import foodweb_figure, trophic_spectrum_figure

    model = _balanced_example()
    for fig in (foodweb_figure(model), trophic_spectrum_figure(model, "biomass")):
        try:
            assert "Balance Model" not in _texts(fig)
            assert "Could not plot" not in _texts(fig)
            assert any(ax.axison for ax in fig.axes)
        finally:
            plt.close(fig)


def test_a_plot_that_fails_says_so_in_its_place():
    import matplotlib.pyplot as plt

    from pypath_shiny.pages.analysis import foodweb_figure

    fig = foodweb_figure(object())  # nothing the plot can read
    try:
        assert "Could not plot food web" in _texts(fig)
    finally:
        plt.close(fig)


def test_the_analysis_plots_are_drawn_by_those_functions():
    source = inspect.getsource(analysis_server)
    assert "return foodweb_figure(get_balanced_model())" in source
    assert "trophic_spectrum_figure(get_balanced_model()," in source


def test_the_message_stays_inside_a_narrow_plot_area():
    # the page sizes the figure to its place before saving it: the spectrum's
    # card, or any plot on a phone, is narrower than the instruction's line
    import matplotlib.pyplot as plt

    from pypath_shiny.pages.analysis import BALANCE_FIRST, message_figure

    message = BALANCE_FIRST.format(what="the trophic spectrum")
    for width_px in (300, 480, 731, 1340):
        fig = message_figure(message)
        try:
            fig.set_dpi(100)
            fig.set_size_inches(width_px / 100, 4.0)
            fig.canvas.draw()
            text = fig.axes[0].texts[0]
            box = text.get_window_extent(fig.canvas.get_renderer())
            assert box.x0 >= 0 and box.x1 <= width_px, (width_px, box.x0, box.x1)
            assert box.y0 >= 0 and box.y1 <= 400, (width_px, box.y0, box.y1)
            # every word is still there, however the lines fall
            assert text._get_wrapped_text().split() == message.split()
            if width_px == 300:
                assert text._get_wrapped_text().count("\n") > message.count("\n")
        finally:
            plt.close(fig)
