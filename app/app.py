"""
PyPath - Shiny for Python Dashboard Application

A web-based frontend for the PyPath ecosystem modeling package,
implementing Ecopath mass-balance and Ecosim dynamic simulation.
"""

from shiny import App, Inputs, Outputs, Session, reactive, render, ui
from pathlib import Path
import shinyswatch

# Import page modules
from pages import home, ecopath, ecosim, results, about
from pages import data_import, analysis

# App directory for static assets
APP_DIR = Path(__file__).parent

# App UI with dashboard layout and Bootstrap theme
app_ui = ui.page_navbar(
    # Include Bootstrap Icons CSS and custom styles
    ui.head_content(
        ui.tags.link(
            rel="stylesheet",
            href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css"
        ),
        # Custom CSS for DataGrid styling
        ui.tags.style("""
            /* Make Group column wider in DataGrids */
            .shiny-data-grid td:first-child,
            .shiny-data-grid th:first-child {
                min-width: 180px !important;
                max-width: 250px !important;
            }
            /* Style for numeric columns */
            .shiny-data-grid td:not(:first-child) {
                text-align: right;
                font-family: monospace;
            }
        """)
    ),
    # Navigation pages
    ui.nav_panel("Home", home.home_ui()),
    ui.nav_panel("Data Import", data_import.import_ui()),
    ui.nav_panel("Ecopath Model", ecopath.ecopath_ui()),
    ui.nav_panel("Ecosim Simulation", ecosim.ecosim_ui()),
    ui.nav_panel("Analysis", analysis.analysis_ui()),
    ui.nav_panel("Results", results.results_ui()),
    ui.nav_spacer(),
    # Settings button that opens modal
    ui.nav_control(
        ui.input_action_button(
            "btn_settings",
            ui.tags.i(class_="bi bi-gear-fill"),
            class_="btn btn-link nav-link p-2",
            title="Settings"
        )
    ),
    ui.nav_panel("About", about.about_ui()),
    
    # Navbar settings
    title=ui.tags.span(
        ui.tags.img(src="icon.svg", height="32px", style="margin-right: 8px; vertical-align: middle;"),
        ui.tags.span("PyPath", style="font-weight: 600; vertical-align: middle;")
    ),
    id="main_navbar",
    footer=ui.div(
        ui.tags.hr(),
        ui.tags.p(
            "PyPath © 2025 | ",
            ui.tags.a("Documentation", href="https://github.com/razinkele/PyPath", class_="text-decoration-none"),
            " | ",
            ui.tags.a("Report Issue", href="https://github.com/razinkele/PyPath/issues", class_="text-decoration-none"),
            class_="text-center text-muted small"
        ),
        class_="p-2"
    ),
    fillable=True,
    # Apply a clean modern theme - 'flatly' is professional and readable
    theme=shinyswatch.theme.flatly,
)


def server(input: Inputs, output: Outputs, session: Session):
    """Main server function."""
    
    # Enable theme picker
    shinyswatch.theme_picker_server()
    
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
    
    # Shared reactive values across pages
    model_data = reactive.Value(None)
    sim_results = reactive.Value(None)
    
    # Initialize page servers
    home.home_server(input, output, session)
    data_import.import_server(input, output, session, model_data)
    ecopath.ecopath_server(input, output, session, model_data)
    ecosim.ecosim_server(input, output, session, model_data, sim_results)
    analysis.analysis_server(input, output, session, model_data, sim_results)
    results.results_server(input, output, session, model_data, sim_results)
    about.about_server(input, output, session)


# Create the app with static assets
app = App(app_ui, server, static_assets=APP_DIR / "static")
