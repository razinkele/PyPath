"""Results and visualization page module."""

import logging

import numpy as np
import pandas as pd
from shiny import Inputs, Outputs, Session, reactive, render, req, ui

try:
    from shinywidgets import output_widget, render_widget
except ImportError:  # pragma: no cover

    def output_widget(*args, **kwargs):
        kwargs.pop("fillable", None)  # output_plot has no fillable
        return ui.output_plot(*args, **kwargs)

    def render_widget(func):
        return render.plot(func)


# Import centralized configuration
from pypath_shiny.components import download_icon, empty_state
from pypath_shiny.config import UI
from pypath_shiny.services.cell_tables import as_cell_table, with_cell_tips

# Import great_tables formatters
from ..services import network_viz
from ..services.gt_formatters import (
    format_diet_matrix,
    format_params_table,
    format_time_series_summary,
)
from ..services.plotly_viz import (
    build_biomass_figure,
    build_catch_figure,
    build_foodweb_figure,
    build_tl_bar_figure,
    build_trophic_pyramid_figure,
    plotly_template_for,
)

# Import shared utilities (pypath path setup handled by app/__init__.py)
from .utils import get_model_info, is_balanced_model, is_rpath_params

logger = logging.getLogger(__name__)


def choose_default_results_groups(
    groups: list[str],
    previous_selected: tuple[str, ...] | list[str] | None,
    max_default: int = 3,
) -> list[str]:
    """Keep valid prior selections, otherwise choose first groups."""
    if not groups:
        return []

    previous = list(previous_selected or [])
    kept = [g for g in previous if g in groups]
    if kept:
        return kept

    limit = max(1, int(max_default))
    return groups[:limit]


def _foodweb_views() -> list:
    """The food web's outputs: the Plotly diagram, and with pyvis' Shiny
    binding installed a second tab with an interactive network (drag, zoom,
    search).

    The two views are tabs, not conditional panels: a Plotly widget redrawn
    after a conditional panel hid and showed it makes shinywidgets log an
    error in the browser, and a tab does not.
    """
    diagram = output_widget(
        "foodweb_plot", height=UI.plot_height_large_px, fillable=True
    )
    if not network_viz.HAS_PYVIS_SHINY:
        return [diagram]
    return [
        ui.navset_pill(
            ui.nav_panel("Diagram", diagram, value="diagram"),
            ui.nav_panel(
                "Interactive network",
                network_viz.output_pyvis_network(
                    "foodweb_network",
                    height="650px",
                    theme="auto",  # follows the page's dark mode
                    show_layout_switcher=False,
                ),
                value="network",
            ),
            id="foodweb_view",
            selected="diagram",
        )
    ]


def results_ui():
    """Results page UI."""
    return ui.div(
        ui.h2("Results & Visualization", class_="mb-4"),
        ui.layout_sidebar(
            ui.sidebar(
                ui.h4("Export Options"),
                ui.h5("Model Results"),
                ui.download_button(
                    "download_model_csv",
                    "Download Model (CSV)",
                    icon=download_icon(),
                    class_="btn-outline-primary w-100 mb-2",
                ),
                ui.tags.hr(),
                ui.h5("Simulation Results"),
                ui.download_button(
                    "download_sim_csv",
                    "Download Simulation (CSV)",
                    icon=download_icon(),
                    class_="btn-outline-success w-100 mb-2",
                ),
                ui.download_button(
                    "download_annual_csv",
                    "Download Annual Summary",
                    icon=download_icon(),
                    class_="btn-outline-success w-100 mb-2",
                ),
                ui.tags.hr(),
                ui.h5("Plot Settings"),
                ui.input_select(
                    "plot_style",
                    "Plot Style",
                    choices={
                        "default": "Default",
                        "seaborn-v0_8": "Seaborn",
                        "ggplot": "GGPlot",
                        "dark_background": "Dark Background",
                    },
                ),
                ui.input_select(
                    "color_palette",
                    "Color Palette",
                    choices={
                        "tab10": "Tab10",
                        "Set2": "Set2",
                        "Paired": "Paired",
                        "viridis": "Viridis",
                    },
                ),
                role="complementary",
                aria_label="Results settings",
                width=280,
            ),
            # Main content
            ui.navset_card_tab(
                ui.nav_panel(
                    "Model Summary",
                    ui.h4("Ecopath Model Summary", class_="mt-3"),
                    ui.output_ui("model_summary_status"),
                    ui.output_ui("model_summary_table"),
                    ui.tags.hr(),
                    ui.h5("Trophic Structure"),
                    ui.layout_columns(
                        output_widget(
                            "tl_bar_plot", height=UI.plot_height_small_px, fillable=True
                        ),
                        output_widget(
                            "tl_flow_plot",
                            height=UI.plot_height_small_px,
                            fillable=True,
                        ),
                        col_widths=[UI.col_width_medium, UI.col_width_medium],
                    ),
                ),
                ui.nav_panel(
                    "Food Web",
                    ui.h4("Food Web Diagram", class_="mt-3"),
                    ui.layout_columns(
                        ui.div(
                            ui.input_slider(
                                "foodweb_min_flow",
                                "Minimum Flow to Show",
                                min=0,
                                max=1,
                                value=0.01,
                                step=0.01,
                            ),
                            ui.input_checkbox(
                                "show_biomass_size",
                                "Scale nodes by biomass",
                                value=True,
                            ),
                            ui.input_checkbox(
                                "show_flow_width", "Scale edges by flow", value=True
                            ),
                        ),
                        col_widths=[12],
                    ),
                    *_foodweb_views(),
                ),
                ui.nav_panel(
                    "Simulation Results",
                    ui.h4("Ecosim Simulation Results", class_="mt-3"),
                    ui.output_ui("sim_summary_status"),
                    ui.h5("Biomass Time Series"),
                    ui.layout_columns(
                        ui.input_selectize(
                            "results_groups",
                            "Select Groups",
                            choices=[],
                            multiple=True,
                        ),
                        ui.input_checkbox("log_scale", "Log Scale", value=False),
                        ui.input_checkbox(
                            "show_uncertainty", "Show Uncertainty", value=False
                        ),
                        col_widths=[6, 3, 3],
                    ),
                    output_widget(
                        "results_biomass_plot", height="450px", fillable=True
                    ),
                    ui.tags.hr(),
                    ui.h5("Catch Time Series"),
                    output_widget("results_catch_plot", height="350px", fillable=True),
                ),
                ui.nav_panel(
                    "Comparison",
                    ui.h4("Scenario Comparison", class_="mt-3"),
                    ui.p(
                        "Scenario comparison is under development.",
                        class_="text-muted",
                    ),
                ),
                ui.nav_panel(
                    "Data Tables",
                    ui.h4("Raw Data Tables", class_="mt-3"),
                    ui.navset_card_pill(
                        ui.nav_panel(
                            "Parameters",
                            ui.output_ui(
                                "params_data_table",
                                style="max-height: 500px; overflow: auto;",
                            ),
                        ),
                        ui.nav_panel(
                            "Diet Matrix",
                            ui.output_ui(
                                "diet_data_table",
                                style="max-height: 500px; overflow: auto;",
                            ),
                        ),
                        ui.nav_panel(
                            "Biomass (Monthly)",
                            ui.output_ui(
                                "biomass_data_table",
                                style="max-height: 500px; overflow: auto;",
                            ),
                        ),
                        ui.nav_panel(
                            "Catch (Annual)",
                            ui.output_ui(
                                "catch_data_table",
                                style="max-height: 500px; overflow: auto;",
                            ),
                        ),
                    ),
                ),
                full_screen=True,
            ),
        ),
    )


def results_server(
    input: Inputs,
    output: Outputs,
    session: Session,
    model_data: reactive.Value,
    sim_results: reactive.Value,
):
    """Results page server logic."""

    @render.ui
    def model_summary_status():
        """Display model status."""
        model = model_data.get()
        info = get_model_info(model)

        if info is None:
            return ui.div(
                ui.tags.i(class_="bi bi-info-circle me-2"),
                "No model data available. Create and balance an Ecopath model first.",
                class_="alert alert-info",
            )

        status = "Balanced" if info["is_balanced"] else "Not balanced"
        return ui.div(
            ui.tags.i(class_="bi bi-check-circle me-2"),
            f"Model: {info['eco_name']} ({info['num_groups']} groups) - {status}",
            class_="alert alert-success",
        )

    @render.ui
    @as_cell_table(empty="Balance a model to see its summary.")
    def model_summary_table():
        """Display model summary table."""
        model = model_data.get()
        info = get_model_info(model)

        if info is None:
            return None

        if is_balanced_model(model) and hasattr(model, "summary"):
            return model.summary()
        elif info["params"] is not None:
            # Return params model table
            params = info["params"]
            if is_rpath_params(params):
                return params.model[["Group", "Type", "Biomass", "PB", "QB", "EE"]]
        return None

    def _styled(fig):
        """Apply the sidebar "Plot Style" to a Plotly figure."""
        fig.update_layout(template=plotly_template_for(input.plot_style()))
        return fig

    @render_widget
    def tl_bar_plot():
        """Interactive trophic-level bars (Plotly)."""
        info = get_model_info(model_data.get())
        if info is None or not info["is_balanced"] or info["trophic_level"] is None:
            return _styled(build_tl_bar_figure([], []))
        n_bio = info["num_living"] + info["num_dead"]
        return _styled(
            build_tl_bar_figure(
                info["groups"][:n_bio],
                info["trophic_level"][:n_bio],
                palette=input.color_palette(),
            )
        )

    @render_widget
    def tl_flow_plot():
        """Interactive trophic pyramid of production (Plotly)."""
        model = model_data.get()
        info = get_model_info(model)
        if info is None or not info["is_balanced"] or info["trophic_level"] is None:
            return _styled(build_trophic_pyramid_figure([], []))
        n_living = info["num_living"]
        biomass = np.asarray(model.Biomass[:n_living], dtype=float)
        production = biomass * np.asarray(model.PB[:n_living], dtype=float)
        return _styled(
            build_trophic_pyramid_figure(info["trophic_level"][:n_living], production)
        )

    def _foodweb_settings() -> tuple:
        return (
            input.foodweb_min_flow(),
            input.show_biomass_size(),
            input.show_flow_width(),
        )

    # With two food-web views, each gets the slider and switch values only
    # while its tab is the selected one, so the hidden view is left alone: no
    # wasted redraw, and no Plotly widget closed while hidden, which makes
    # shinywidgets log "Unexpected error while closing model" in the browser
    # console. A view catches up when its tab is selected. One case still
    # logs that message, with nothing wrong on screen: the diagram redrawn as
    # its tab opens, after the settings changed on the network tab.
    _view_settings = {
        "diagram": reactive.value(None),
        "network": reactive.value(None),
    }

    if network_viz.HAS_PYVIS_SHINY:

        @reactive.effect
        def _route_foodweb_settings():
            held = _view_settings[input.foodweb_view()]
            settings = _foodweb_settings()
            with reactive.isolate():
                unchanged = held.get() == settings
            if not unchanged:  # an equal tuple would still redraw the view
                held.set(settings)

    def _settings_for(view: str) -> tuple:
        """The settings ``view`` should draw with (it waits for its first)."""
        if not network_viz.HAS_PYVIS_SHINY:
            return _foodweb_settings()
        held = _view_settings[view].get()
        req(held is not None)
        return held

    if network_viz.HAS_PYVIS_SHINY:

        @network_viz.render_pyvis_network
        def foodweb_network():
            """Interactive food web (pyvis): the same data as the diagram."""
            min_flow, size_by_biomass, width_by_flow = _settings_for("network")
            model = model_data.get()
            info = get_model_info(model)
            if info is None or not info["is_balanced"]:
                return network_viz.build_foodweb_network([], [], [], [], [])
            return network_viz.build_foodweb_network(
                model.Group,
                model.type,
                model.TL,
                model.Biomass,
                model.DC,
                min_flow=min_flow,
                size_by_biomass=size_by_biomass,
                width_by_flow=width_by_flow,
            )

    @render_widget
    def foodweb_plot():
        """Interactive food web (Plotly): groups at their trophic level."""
        min_flow, size_by_biomass, width_by_flow = _settings_for("diagram")
        model = model_data.get()
        info = get_model_info(model)
        if info is None or not info["is_balanced"]:
            return _styled(build_foodweb_figure([], [], [], [], 0))
        n_bio = info["num_living"] + info["num_dead"]
        return _styled(
            build_foodweb_figure(
                info["groups"][:n_bio],
                model.TL[:n_bio],
                model.Biomass[:n_bio],
                model.DC,
                n_living=info["num_living"],
                min_flow=min_flow,
                size_by_biomass=size_by_biomass,
                width_by_flow=width_by_flow,
            )
        )

    @render.ui
    def sim_summary_status():
        """Display simulation status."""
        sim = sim_results.get()

        if sim is None:
            return ui.div(
                ui.tags.i(class_="bi bi-info-circle me-2"),
                "No simulation results available. Run an Ecosim simulation first.",
                class_="alert alert-info",
            )

        return ui.div(
            ui.tags.i(class_="bi bi-check-circle me-2"),
            f"Simulation: {sim.params['years']} years, {sim.params['NUM_GROUPS']} groups",
            class_="alert alert-success",
        )

    @reactive.effect
    def _update_group_choices():
        """Update group choices when simulation results change."""
        _sim = sim_results.get()
        model = model_data.get()
        info = get_model_info(model)

        if info is not None:
            n_living = info["num_living"]
            groups = info["groups"][:n_living]
            selected = choose_default_results_groups(groups, input.results_groups())
            ui.update_selectize(
                "results_groups",
                choices=groups,
                selected=selected,
            )

    @render_widget
    def results_biomass_plot():
        """Simulation biomass results plot."""
        sim = sim_results.get()
        model = model_data.get()
        info = get_model_info(model)

        if info is None:
            return build_biomass_figure(sim, [], [], log_scale=input.log_scale())

        selected = input.results_groups() or []
        return build_biomass_figure(
            sim,
            info["groups"],
            selected,
            log_scale=input.log_scale(),
            palette=input.color_palette(),
        ).update_layout(template=plotly_template_for(input.plot_style()))

    @render_widget
    def results_catch_plot():
        """Simulation catch results plot."""
        sim = sim_results.get()
        return build_catch_figure(sim, palette=input.color_palette()).update_layout(
            template=plotly_template_for(input.plot_style())
        )

    @render.ui
    def params_data_table():
        """Model parameters data table."""
        model = model_data.get()
        info = get_model_info(model)
        if info is None:
            return empty_state("Balance a model to see its parameters.")

        df = None
        if is_balanced_model(model) and hasattr(model, "summary"):
            df = model.summary()
        elif is_rpath_params(info["params"]):
            df = info["params"].model
        else:
            df = pd.DataFrame()

        if df.empty:
            return df
        return with_cell_tips(format_params_table(df, title="Model Parameters"), df)

    @render.ui
    def diet_data_table():
        """Diet matrix data table."""
        model = model_data.get()
        info = get_model_info(model)
        if info is None:
            return empty_state("Balance a model to see its diet matrix.")

        if not info["is_balanced"]:
            # Return diet from params
            params = info["params"]
            if hasattr(params, "diet"):
                df = params.diet
            else:
                df = pd.DataFrame({"Message": ["Balance model to see diet matrix"]})
        else:
            n_living = info["num_living"]
            n_bio = info["num_living"] + info["num_dead"]
            groups = info["groups"]

            df = pd.DataFrame(
                model.DC[:n_bio, :n_living],
                index=groups[:n_bio],
                columns=groups[:n_living],
            )
            df.insert(0, "Prey", df.index)

        if df.empty:
            return df
        return with_cell_tips(
            format_diet_matrix(df, title="Diet Composition Matrix"), df
        )

    @render.ui
    def biomass_data_table():
        """Monthly biomass data table."""
        sim = sim_results.get()
        model = model_data.get()
        info = get_model_info(model)

        if sim is None or info is None:
            return empty_state("Run an Ecosim simulation to see biomass.")

        n_bio = info["num_living"] + info["num_dead"]
        groups = info["groups"]

        df = pd.DataFrame(sim.out_Biomass[:, 1 : n_bio + 1], columns=groups[:n_bio])
        df.insert(0, "Month", range(len(df)))
        df.insert(1, "Year", df["Month"] / 12)

        # Show every 12 months
        df = df[df["Month"] % 12 == 0]

        if df.empty:
            return df
        return with_cell_tips(
            format_time_series_summary(df, title="Monthly Biomass", decimals=4), df
        )

    @render.ui
    def catch_data_table():
        """Annual catch data table."""
        sim = sim_results.get()
        model = model_data.get()
        info = get_model_info(model)

        if sim is None or info is None:
            return empty_state("Run an Ecosim simulation to see catch.")

        n_living = info["num_living"]
        groups = info["groups"]

        df = pd.DataFrame(
            sim.annual_Catch[:, 1 : n_living + 1], columns=groups[:n_living]
        )
        df.insert(0, "Year", range(1, len(df) + 1))

        if df.empty:
            return df
        return with_cell_tips(
            format_time_series_summary(df, title="Annual Catch", decimals=4), df
        )

    @render.download_button(filename="pypath_model.csv")
    def download_model_csv():
        """Download model as CSV."""
        model = model_data.get()
        info = get_model_info(model)
        if info is not None:
            if is_balanced_model(model) and hasattr(model, "summary"):
                yield model.summary().to_csv(index=False)
                return
            if is_rpath_params(info["params"]):
                yield info["params"].model.to_csv(index=False)
                return
        yield ""

    @render.download_button(filename="pypath_simulation.csv")
    def download_sim_csv():
        """Download simulation results as CSV."""
        sim = sim_results.get()
        model = model_data.get()
        info = get_model_info(model)

        if sim is not None and info is not None:
            n_bio = info["num_living"] + info["num_dead"]
            groups = info["groups"]
            df = pd.DataFrame(sim.out_Biomass[:, 1 : n_bio + 1], columns=groups[:n_bio])
            df.insert(0, "Month", range(len(df)))
            yield df.to_csv(index=False)
            return
        yield ""

    @render.download_button(filename="pypath_annual_summary.csv")
    def download_annual_csv():
        """Download annual summary as CSV."""
        sim = sim_results.get()
        model = model_data.get()
        info = get_model_info(model)

        if sim is not None and info is not None:
            n_living = info["num_living"]
            groups = info["groups"]
            df = pd.DataFrame(
                sim.annual_Catch[:, 1 : n_living + 1], columns=groups[:n_living]
            )
            df.insert(0, "Year", range(1, len(df) + 1))
            yield df.to_csv(index=False)
            return
        yield ""
