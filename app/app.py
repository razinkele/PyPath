"""
PyPath - Shiny for Python Dashboard Application

A web-based frontend for the PyPath ecosystem modeling package,
implementing Ecopath mass-balance and Ecosim dynamic simulation.
"""

from shiny import App, Inputs, Outputs, Session, reactive, render, ui
from pathlib import Path
from datetime import datetime
import shinyswatch
import sys

# App directory for static assets
APP_DIR = Path(__file__).parent

# Add the root directory to the path so absolute imports work
root_dir = APP_DIR.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# Import page modules - organized by category
# Core pages
from pages import home, data_import, ecopath, ecosim, results, analysis, about
# Advanced features
from pages import multistanza, forcing_demo, diet_rewiring_demo, optimization_demo, ecospace

# Configuration imports
from config import UI

# App UI with dashboard layout and Bootstrap theme
app_ui = ui.page_navbar(
    # Include Bootstrap Icons CSS and custom styles
    ui.head_content(
        ui.tags.link(
            rel="stylesheet",
            href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css"
        ),
        # Load custom CSS file
        ui.tags.link(
            rel="stylesheet",
            href="custom.css"
        ),
        # Additional CSS for DataGrid styling
        ui.tags.style(f"""
            /* Make Group column wider in DataGrids */
            .shiny-data-grid td:first-child,
            .shiny-data-grid th:first-child {{
                min-width: {UI.table_col_min_width_px} !important;
                max-width: {UI.table_col_max_width_px} !important;
            }}
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
    ui.nav_menu(
        "Advanced Features",
        ui.nav_panel("ECOSPACE Spatial Modeling", ecospace.ecospace_ui()),
        ui.nav_panel("Multi-Stanza Groups", multistanza.multistanza_ui()),
        ui.nav_panel("State-Variable Forcing", forcing_demo.forcing_demo_ui()),
        ui.nav_panel("Dynamic Diet Rewiring", diet_rewiring_demo.diet_rewiring_demo_ui()),
        ui.nav_panel("Bayesian Optimization", optimization_demo.optimization_demo_ui()),
        icon=ui.tags.i(class_="bi bi-stars")
    ),
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
        ui.tags.img(src="icon.svg", height=UI.icon_height_px, style="margin-right: 8px; vertical-align: middle;"),
        ui.tags.span("PyPath", style="font-weight: 600; vertical-align: middle;")
    ),
    id="main_navbar",
    footer=ui.div(
        ui.tags.hr(),
        ui.tags.p(
            f"PyPath © {datetime.now().year} | ",
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

    # Primary reactive state (shared across all pages)
    model_data = reactive.Value(None)  # RpathParams: Ecopath model parameters
    sim_results = reactive.Value(None)  # Dict: Ecosim simulation results

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
        def __init__(self, model_data_ref: reactive.Value, sim_results_ref: reactive.Value):
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
            # For RpathParams objects (have model and diet attributes), store directly
            if hasattr(data, 'model') and hasattr(data, 'diet'):
                shared_data.params.set(data)
            else:
                # For other data structures, store as-is
                shared_data.params.set(data)

    # Initialize page servers with error handling
    server_modules = [
        ("Home", lambda: home.home_server(input, output, session, model_data)),
        ("Data Import", lambda: data_import.import_server(input, output, session, model_data)),
        ("Ecopath", lambda: ecopath.ecopath_server(input, output, session, model_data)),
        ("Ecosim", lambda: ecosim.ecosim_server(input, output, session, model_data, sim_results)),
        ("Ecospace", lambda: ecospace.ecospace_server(input, output, session, model_data, sim_results)),
        ("Multi-Stanza", lambda: multistanza.multistanza_server(input, output, session, shared_data)),
        ("Forcing Demo", lambda: forcing_demo.forcing_demo_server(input, output, session)),
        ("Diet Rewiring Demo", lambda: diet_rewiring_demo.diet_rewiring_demo_server(input, output, session)),
        ("Optimization Demo", lambda: optimization_demo.optimization_demo_server(input, output, session)),
        ("Analysis", lambda: analysis.analysis_server(input, output, session, model_data, sim_results)),
        ("Results", lambda: results.results_server(input, output, session, model_data, sim_results)),
        ("About", lambda: about.about_server(input, output, session)),
    ]

    # Initialize all server modules with error handling
    for page_name, server_init in server_modules:
        try:
            server_init()
        except Exception as e:
            print(f"ERROR: Failed to initialize {page_name} server: {e}")
            import traceback
            traceback.print_exc()


# Create the app with static assets
app = App(app_ui, server, static_assets=APP_DIR / "static")
