"""
PyPath - Shiny for Python Dashboard Application

A web-based frontend for the PyPath ecosystem modeling package,
implementing Ecopath mass-balance and Ecosim dynamic simulation.
"""

import logging
from datetime import datetime
from pathlib import Path

try:
    import shinyswatch
except Exception:
    # Provide a minimal shim so app can run without shinyswatch installed.
    class _NoShinyswatch:
        class theme:
            flatly = None

        @staticmethod
        def theme_picker_server():
            return None

        @staticmethod
        def theme_picker_ui():
            # Lazy import so tests without Shiny can continue to import app
            try:
                from shiny import ui as _ui

                return _ui.tags.div("Theme picker unavailable")
            except Exception:
                return "Theme picker unavailable"

    shinyswatch = _NoShinyswatch()

from shiny import App, Inputs, Outputs, Session, reactive, ui

# Get logger
logger = logging.getLogger("pypath_app")

# App directory for static assets
APP_DIR = Path(__file__).parent

# Configuration imports
from pypath_shiny.config import UI  # noqa: E402

# Import page modules - organized by category
from pypath_shiny.pages import (  # noqa: E402
    about,
    analysis,
    data_import,
    diet_rewiring_demo,
    ecopath,
    ecosim,
    ecospace,
    forcing_demo,
    home,
    ibm,
    multistanza,
    optimization_demo,
    prebalance,
    results,
)

def _icon_label(icon_class: str, text: str) -> ui.TagList:
    """Create a nav label with a Bootstrap Icon and text."""
    return ui.TagList(
        ui.tags.i(class_=f"bi {icon_class}", style="margin-right: 8px;"),
        text,
    )


# App UI with left sidebar pill list navigation
app_ui = ui.page_fluid(
    # Typography, icons, and custom styles
    ui.head_content(
        ui.tags.link(
            rel="stylesheet",
            href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&family=Lora:ital,wght@0,400;0,500;0,600;0,700;1,400&family=Outfit:wght@300;400;500;600;700&display=swap",
        ),
        ui.tags.link(
            rel="stylesheet",
            href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css",
        ),
        ui.tags.link(rel="stylesheet", href="custom.css"),
        ui.tags.style(f"""
            /* Make Group column wider in DataGrids */
            .shiny-data-grid td:first-child,
            .shiny-data-grid th:first-child {{
                min-width: {UI.table_col_min_width_px} !important;
                max-width: {UI.table_col_max_width_px} !important;
            }}
            /* Style for numeric columns */
            .shiny-data-grid td:not(:first-child) {{
                text-align: right;
                font-family: monospace;
            }}
        """),
    ),
    # Branding header
    ui.div(
        ui.tags.img(
            src="icon.svg",
            height=UI.icon_height_px,
            style="margin-right: 10px; vertical-align: middle;",
        ),
        ui.tags.span("PyPath", class_="brand-name"),
        class_="pypath-brand mb-2",
    ),
    # Main navigation — left sidebar pill list
    ui.navset_pill_list(
        # Core workflow pages
        ui.nav_panel(_icon_label("bi-house-fill", "Home"), home.home_ui(), value="Home"),
        ui.nav_panel(_icon_label("bi-download", "Data Import"), data_import.import_ui(), value="Data Import"),
        ui.nav_panel(_icon_label("bi-gear", "Ecopath Model"), ecopath.ecopath_ui(), value="Ecopath Model"),
        ui.nav_panel(
            _icon_label("bi-clipboard-check", "Pre-Balance"), prebalance.prebalance_ui(), value="Pre-Balance",
        ),
        ui.nav_panel(_icon_label("bi-graph-up", "Ecosim"), ecosim.ecosim_ui(), value="Ecosim"),
        # Advanced features (collapsible group)
        ui.nav_menu(
            _icon_label("bi-stars", "Advanced"),
            ui.nav_panel(
                _icon_label("bi-globe-americas", "Ecospace"), ecospace.ecospace_ui(), value="Ecospace",
            ),
            ui.nav_panel(
                _icon_label("bi-layers", "Multi-Stanza"), multistanza.multistanza_ui(), value="Multi-Stanza",
            ),
            ui.nav_panel(
                _icon_label("bi-lightning", "Forcing"), forcing_demo.forcing_demo_ui(), value="Forcing",
            ),
            ui.nav_panel(
                _icon_label("bi-arrow-repeat", "Diet Rewiring"),
                diet_rewiring_demo.diet_rewiring_demo_ui(),
                value="Diet Rewiring",
            ),
            ui.nav_panel(
                _icon_label("bi-bullseye", "Optimization"),
                optimization_demo.optimization_demo_ui(),
                value="Optimization",
            ),
            ui.nav_panel(
                _icon_label("bi-cpu", "Individual-Based Model"), ibm.ibm_ui(), value="Individual-Based Model",
            ),
        ),
        # Output pages
        ui.nav_panel(
            _icon_label("bi-bar-chart-line", "Analysis"), analysis.analysis_ui(), value="Analysis",
        ),
        ui.nav_panel(
            _icon_label("bi-file-earmark-text", "Results"), results.results_ui(), value="Results",
        ),
        ui.nav_panel(_icon_label("bi-info-circle", "About"), about.about_ui(), value="About"),
        ui.nav_control(
            ui.input_action_button(
                "btn_settings",
                _icon_label("bi-gear-fill", "Settings"),
                class_="btn btn-link text-start w-100 p-2",
            ),
        ),
        id="main_nav",
        widths=UI.nav_sidebar_widths,
        well=True,
    ),
    # Footer
    ui.div(
        ui.tags.hr(),
        ui.tags.p(
            f"PyPath \u00a9 {datetime.now().year} | ",
            ui.tags.a(
                "Documentation",
                href="https://github.com/razinkele/PyPath",
            ),
            " | ",
            ui.tags.a(
                "Report Issue",
                href="https://github.com/razinkele/PyPath/issues",
            ),
            class_="text-center",
        ),
        class_="pypath-footer",
    ),
    theme=(
        shinyswatch.theme.flatly
        if getattr(shinyswatch, "theme", None) is not None
        and getattr(shinyswatch.theme, "flatly", None) is not None
        else None
    ),
)


def server(input: Inputs, output: Outputs, session: Session):
    """Main server function.

    Data Flow Architecture:
    ======================

    This application uses a centralized reactive state management pattern:

    1. Primary Reactive State:
       - model_data: Contains RpathParams objects (Ecopath model parameters)
       - sim_results: Contains Ecosim simulation results

    2. State Updates:
       - Data Import page -> sets model_data (from CSV/database)
       - Ecopath page -> modifies model_data (balancing, adjustments)
       - Ecosim page -> reads model_data, sets sim_results
       - Analysis/Results pages -> read model_data and sim_results

    3. SharedData Pattern:
       - Provides structured access for advanced features
       - References (not duplicates) primary reactive values
       - Automatically syncs model_data to params for advanced features

    4. Page Communication:
       - All pages receive references to reactive values
       - Changes propagate automatically via Shiny's reactive system
       - Pages use reactive.effect to respond to state changes
    """

    # Enable theme picker (only when available)
    if hasattr(shinyswatch, "theme_picker_server") and callable(
        shinyswatch.theme_picker_server
    ):
        try:
            shinyswatch.theme_picker_server()
        except Exception:
            # Degrade gracefully if the theme picker cannot be initialized
            logger.debug(
                "shinyswatch.theme_picker_server failed to initialize", exc_info=True
            )

    # Show settings modal when settings button clicked
    @reactive.effect
    @reactive.event(input.btn_settings)
    def show_settings_modal():
        m = ui.modal(
            ui.h5("Theme Selection", class_="mb-3"),
            shinyswatch.theme_picker_ui(),
            title="Settings",
            easy_close=True,
            footer=ui.modal_button("Close"),
            size="m",
        )
        ui.modal_show(m)

    # Primary reactive state (shared across all pages)
    model_data = reactive.Value(None)  # RpathParams: Ecopath model parameters
    sim_results = reactive.Value(None)  # Dict: Ecosim simulation results
    sim_scenario = reactive.Value(None)  # RsimScenario: shared Ecosim scenario

    # Create shared data object for pages that need structured access
    class SharedData:
        """Container providing structured access to shared reactive state.

        This class consolidates access to reactive values for pages that need
        structured access. It directly exposes reactive values as attributes,
        following Shiny's reactive pattern where values are accessed via () calls.

        Attributes:
            model_data: Reactive value for model data (shared with core pages)
            sim_results: Reactive value for simulation results (shared with core pages)
            params: Reactive value for model parameters (for advanced features)
        """

        def __init__(
            self, model_data_ref: reactive.Value, sim_results_ref: reactive.Value
        ):
            # Reference the primary reactive values (no duplication)
            self.model_data = model_data_ref
            self.sim_results = sim_results_ref
            # Additional reactive value for pages expecting params structure
            self.params = reactive.Value(None)

    shared_data = SharedData(model_data, sim_results)

    # Sync model_data to shared_data.params for pages expecting params
    @reactive.effect
    def sync_model_data():
        """Synchronize model_data to shared_data.params for advanced features."""
        data = model_data()
        if data is not None:
            shared_data.params.set(data)

    # Initialize page servers with error handling
    server_modules = [
        ("Home", lambda: home.home_server(input, output, session, model_data)),
        (
            "Data Import",
            lambda: data_import.import_server(input, output, session, model_data),
        ),
        ("Ecopath", lambda: ecopath.ecopath_server(input, output, session, model_data)),
        (
            "Pre-Balance Diagnostics",
            lambda: prebalance.prebalance_server(input, output, session, model_data),
        ),
        (
            "Ecosim",
            lambda: ecosim.ecosim_server(
                input, output, session, model_data, sim_results, sim_scenario
            ),
        ),
        (
            "Ecospace",
            lambda: ecospace.ecospace_server(
                input, output, session, model_data, sim_results, sim_scenario
            ),
        ),
        (
            "Multi-Stanza",
            lambda: multistanza.multistanza_server(input, output, session, shared_data),
        ),
        (
            "Forcing Demo",
            lambda: forcing_demo.forcing_demo_server(input, output, session),
        ),
        (
            "Diet Rewiring Demo",
            lambda: diet_rewiring_demo.diet_rewiring_demo_server(
                input, output, session
            ),
        ),
        (
            "Optimization Demo",
            lambda: optimization_demo.optimization_demo_server(input, output, session),
        ),
        (
            "IBM",
            lambda: ibm.ibm_server(
                input, output, session, model_data, sim_results, sim_scenario
            ),
        ),
        (
            "Analysis",
            lambda: analysis.analysis_server(
                input, output, session, model_data, sim_results
            ),
        ),
        (
            "Results",
            lambda: results.results_server(
                input, output, session, model_data, sim_results
            ),
        ),
        ("About", lambda: about.about_server(input, output, session)),
    ]

    # Initialize all server modules with error handling
    for page_name, server_init in server_modules:
        try:
            server_init()
        except Exception as e:
            logger.error("Failed to initialize %s server: %s", page_name, e, exc_info=True)


def main():
    """Launch the PyPath Shiny application."""
    import uvicorn

    uvicorn.run("pypath_shiny.app:app", host="0.0.0.0", port=8000, reload=True)


# Create the app with static assets
app = App(app_ui, server, static_assets=APP_DIR / "static")
