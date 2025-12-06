"""
PyPath - Shiny for Python Dashboard Application

A web-based frontend for the PyPath ecosystem modeling package,
implementing Ecopath mass-balance and Ecosim dynamic simulation.
"""

from shiny import App, Inputs, Outputs, Session, reactive, render, ui

# Import page modules
from app.pages import home, ecopath, ecosim, results, about

# App UI with dashboard layout
app_ui = ui.page_navbar(
    # Navigation pages
    ui.nav_panel("Home", home.home_ui()),
    ui.nav_panel("Ecopath Model", ecopath.ecopath_ui()),
    ui.nav_panel("Ecosim Simulation", ecosim.ecosim_ui()),
    ui.nav_panel("Results", results.results_ui()),
    ui.nav_spacer(),
    ui.nav_panel("About", about.about_ui()),
    
    # Navbar settings
    title=ui.tags.span(
        ui.tags.span("🌊 ", style="font-size: 1.5em;"),
        "PyPath Dashboard"
    ),
    id="main_navbar",
    footer=ui.div(
        ui.tags.hr(),
        ui.tags.p(
            "PyPath © 2025 | ",
            ui.tags.a("Documentation", href="https://github.com/your-repo/pypath"),
            " | ",
            ui.tags.a("Report Issue", href="https://github.com/your-repo/pypath/issues"),
            style="text-align: center; color: #666; font-size: 0.85em;"
        ),
        style="padding: 10px;"
    ),
    fillable=True,
)


def server(input: Inputs, output: Outputs, session: Session):
    """Main server function."""
    
    # Shared reactive values across pages
    model_data = reactive.Value(None)
    sim_results = reactive.Value(None)
    
    # Initialize page servers
    home.home_server(input, output, session)
    ecopath.ecopath_server(input, output, session, model_data)
    ecosim.ecosim_server(input, output, session, model_data, sim_results)
    results.results_server(input, output, session, model_data, sim_results)
    about.about_server(input, output, session)


# Create the app
app = App(app_ui, server)
