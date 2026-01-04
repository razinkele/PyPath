"""Ecosim simulation page module."""

import numpy as np
import pandas as pd
from shiny import Inputs, Outputs, Session, reactive, render, ui

# Import centralized configuration
try:
    from app.config import PARAM_RANGES, THRESHOLDS, UI
except ModuleNotFoundError:
    from config import PARAM_RANGES, THRESHOLDS, UI

# pypath imports (path setup handled by app/__init__.py)
from pypath.core.autofix import validate_and_fix_scenario
from pypath.core.ecosim import RsimScenario, rsim_run, rsim_scenario
from pypath.core.ecosim_advanced import rsim_run_advanced
from pypath.core.forcing import DietRewiring


# Helper to check model balance and show notification if not
def _require_balanced_model_or_notify(model) -> bool:
    """Return True if model is balanced, otherwise show a UI error and return False."""
    from app.pages.utils import is_balanced_model

    if not is_balanced_model(model):
        ui.notification_show(
            "Ecosim requires a balanced Ecopath model. Balance the model on the Ecopath page first.",
            type="error",
            duration=6,
        )
        return False
    return True


def ecosim_ui() -> ui.Tag:
    """Ecosim simulation page UI."""
    return ui.page_fluid(
        ui.h2("Ecosim Dynamic Simulation", class_="mb-4"),
        ui.layout_sidebar(
            # Sidebar for simulation settings
            ui.sidebar(
                ui.h4("Simulation Settings"),
                # Time settings
                ui.h5("Time Period"),
                ui.input_numeric(
                    "sim_years",
                    ui.span(
                        "Simulation Years ",
                        ui.tags.i(
                            class_="bi bi-info-circle",
                            title="Number of years to simulate. Longer periods show long-term dynamics but take more time to compute.",
                            style="cursor: help;",
                        ),
                    ),
                    value=PARAM_RANGES.years_default,
                    min=PARAM_RANGES.years_min,
                    max=PARAM_RANGES.years_max,
                ),
                ui.input_select(
                    "integration_method",
                    ui.span(
                        "Integration Method ",
                        ui.tags.i(
                            class_="bi bi-info-circle",
                            title="Numerical integration method. RK4 (Runge-Kutta 4th order) is more accurate but slower. AB (Adams-Bashforth) is faster but less stable.",
                            style="cursor: help;",
                        ),
                    ),
                    choices={"RK4": "Runge-Kutta 4", "AB": "Adams-Bashforth"},
                    selected="RK4",
                ),
                ui.tags.hr(),
                # Vulnerability settings
                ui.h5("Functional Response"),
                ui.input_slider(
                    "vulnerability",
                    ui.span(
                        "Default Vulnerability ",
                        ui.tags.i(
                            class_="bi bi-info-circle",
                            title="Controls predator-prey functional response. 1 = bottom-up control (prey abundance limits predators), 2 = mixed control, higher values = top-down control (predators limit prey). Range: 1-100.",
                            style="cursor: help;",
                        ),
                    ),
                    min=PARAM_RANGES.vulnerability_min,
                    max=PARAM_RANGES.vulnerability_max,
                    value=PARAM_RANGES.vulnerability_default,
                    step=0.5,
                ),
                ui.p(
                    "1 = Bottom-up, 2 = Mixed, High = Top-down",
                    class_="text-muted small",
                ),
                ui.tags.hr(),
                # Diet Rewiring settings
                ui.h5("Dynamic Diet Rewiring"),
                ui.input_checkbox(
                    "enable_diet_rewiring",
                    ui.span(
                        "Enable Diet Rewiring ",
                        ui.tags.i(
                            class_="bi bi-info-circle",
                            title="Allow predator diet preferences to change based on prey availability (prey switching, adaptive foraging). Predators shift to more abundant prey species.",
                            style="cursor: help;",
                        ),
                    ),
                    value=False,
                ),
                ui.panel_conditional(
                    "input.enable_diet_rewiring",
                    ui.input_slider(
                        "switching_power",
                        ui.span(
                            "Switching Power ",
                            ui.tags.i(
                                class_="bi bi-info-circle",
                                title="Controls strength of prey switching. 1.0 = proportional (no switching), 2-3 = moderate switching (typical), >3 = strong switching (opportunistic predators).",
                                style="cursor: help;",
                            ),
                        ),
                        min=PARAM_RANGES.switching_power_min,
                        max=PARAM_RANGES.switching_power_max,
                        value=PARAM_RANGES.switching_power_default,
                        step=0.1,
                    ),
                    ui.input_slider(
                        "rewiring_interval",
                        ui.span(
                            "Update Interval (months) ",
                            ui.tags.i(
                                class_="bi bi-info-circle",
                                title="How often diet is recalculated. 1 = monthly (responsive but slow), 12 = annual (fast but less responsive).",
                                style="cursor: help;",
                            ),
                        ),
                        min=PARAM_RANGES.rewiring_interval_min,
                        max=PARAM_RANGES.rewiring_interval_max,
                        value=PARAM_RANGES.rewiring_interval_default,
                        step=1,
                    ),
                    ui.input_numeric(
                        "min_diet_proportion",
                        ui.span(
                            "Minimum Diet Proportion ",
                            ui.tags.i(
                                class_="bi bi-info-circle",
                                title="Minimum fraction to maintain in diet. Prevents complete elimination of prey types.",
                                style="cursor: help;",
                            ),
                        ),
                        value=THRESHOLDS.min_diet_proportion_range_default,
                        min=THRESHOLDS.min_diet_proportion_range_min,
                        max=THRESHOLDS.min_diet_proportion_range_max,
                        step=0.001,
                    ),
                ),
                ui.tags.hr(),
                # Fishing scenarios
                ui.h5("Fishing Scenario"),
                ui.input_select(
                    "fishing_scenario",
                    ui.span(
                        "Scenario Type ",
                        ui.tags.i(
                            class_="bi bi-info-circle",
                            title="Fishing effort scenario: Baseline (constant), Increase (gradual ramp up), Decrease (gradual ramp down), Closure (fishing stops for period), Custom (upload CSV).",
                            style="cursor: help;",
                        ),
                    ),
                    choices={
                        "baseline": "Baseline (constant effort)",
                        "increase": "Increase effort",
                        "decrease": "Decrease effort",
                        "closure": "Fishery closure",
                        "custom": "Custom",
                    },
                    selected="baseline",
                ),
                ui.output_ui("fishing_params_ui"),
                ui.tags.hr(),
                # Stability settings
                ui.h5(
                    ui.span(
                        "Stability ",
                        ui.tags.i(
                            class_="bi bi-info-circle",
                            title="Automatic parameter calibration to prevent crashes and improve simulation stability.",
                            style="cursor: help;",
                        ),
                    )
                ),
                ui.input_checkbox(
                    "enable_autofix",
                    ui.span(
                        "Auto-fix parameters ",
                        ui.tags.i(
                            class_="bi bi-info-circle",
                            title=f"Automatically caps VV ≤ {THRESHOLDS.vv_cap}, QQ ≤ {THRESHOLDS.qq_cap}, ensures minimum biomass ≥ {THRESHOLDS.min_biomass}, and normalizes DD to 1-2. Prevents most crashes caused by extreme parameter values.",
                            style="cursor: help;",
                        ),
                    ),
                    value=True,
                ),
                ui.input_action_button(
                    "btn_autofix_help",
                    "What does autofix do?",
                    class_="btn-sm btn-outline-info w-100 mt-2",
                ),
                ui.tags.hr(),
                # Run buttons
                ui.input_action_button(
                    "btn_create_scenario", "Create Scenario", class_="btn-primary w-100"
                ),
                ui.input_action_button(
                    "btn_run_sim", "Run Simulation", class_="btn-success w-100 mt-2"
                ),
                width=UI.sidebar_width,
            ),
            # Main content
            ui.navset_card_tab(
                ui.nav_panel(
                    "Scenario Setup",
                    ui.layout_columns(
                        ui.h4("Scenario Configuration", class_="mt-3"),
                        ui.div(
                            ui.input_action_button(
                                "btn_help_scenario",
                                ui.span(
                                    ui.tags.i(class_="bi bi-question-circle me-1"),
                                    "Help",
                                ),
                                class_="btn-sm btn-outline-primary mt-3",
                            ),
                            style="text-align: right;",
                        ),
                        col_widths=[10, 2],
                    ),
                    ui.output_ui("help_scenario_setup"),
                    ui.output_ui("scenario_status"),
                    ui.layout_columns(
                        ui.card(
                            ui.card_header("Effort Forcing"),
                            ui.output_plot("effort_preview_plot"),
                        ),
                        ui.card(
                            ui.card_header("Biomass Forcing"),
                            ui.card_body(
                                ui.p(
                                    "Configure environmental forcing on prey availability"
                                ),
                                ui.input_select(
                                    "forcing_group",
                                    ui.span(
                                        "Select Group ",
                                        ui.tags.i(
                                            class_="bi bi-info-circle",
                                            title="Group to apply environmental forcing. Typically primary producers (phytoplankton) or lower trophic levels.",
                                            style="cursor: help;",
                                        ),
                                    ),
                                    choices=["(Create scenario first)"],
                                ),
                                ui.input_slider(
                                    "forcing_multiplier",
                                    ui.span(
                                        "Forcing Multiplier ",
                                        ui.tags.i(
                                            class_="bi bi-info-circle",
                                            title="Multiplier for prey availability. 1.0 = normal, >1.0 = more productive (e.g., nutrient enrichment), <1.0 = less productive (e.g., climate stress).",
                                            style="cursor: help;",
                                        ),
                                    ),
                                    min=0,
                                    max=3,
                                    value=1,
                                    step=0.1,
                                ),
                            ),
                        ),
                        col_widths=[6, 6],
                    ),
                ),
                ui.nav_panel(
                    "Simulation Progress",
                    ui.layout_columns(
                        ui.h4("Simulation Status", class_="mt-3"),
                        ui.div(
                            ui.input_action_button(
                                "btn_help_progress",
                                ui.span(
                                    ui.tags.i(class_="bi bi-question-circle me-1"),
                                    "Help",
                                ),
                                class_="btn-sm btn-outline-primary mt-3",
                            ),
                            style="text-align: right;",
                        ),
                        col_widths=[10, 2],
                    ),
                    ui.output_ui("help_progress"),
                    ui.output_ui("simulation_status"),
                    ui.output_ui("progress_display"),
                ),
                ui.nav_panel(
                    "Time Series",
                    ui.layout_columns(
                        ui.h4("Biomass Trajectories", class_="mt-3"),
                        ui.div(
                            ui.input_action_button(
                                "btn_help_timeseries",
                                ui.span(
                                    ui.tags.i(class_="bi bi-question-circle me-1"),
                                    "Help",
                                ),
                                class_="btn-sm btn-outline-primary mt-3",
                            ),
                            style="text-align: right;",
                        ),
                        col_widths=[10, 2],
                    ),
                    ui.output_ui("help_timeseries"),
                    ui.layout_columns(
                        ui.input_selectize(
                            "plot_groups",
                            ui.span(
                                "Select Groups to Plot ",
                                ui.tags.i(
                                    class_="bi bi-info-circle",
                                    title="Choose which functional groups to display in the biomass trajectory plot. Select up to 10 groups for clarity.",
                                    style="cursor: help;",
                                ),
                            ),
                            choices=[],
                            multiple=True,
                        ),
                        ui.input_checkbox(
                            "relative_biomass",
                            ui.span(
                                "Show Relative Biomass ",
                                ui.tags.i(
                                    class_="bi bi-info-circle",
                                    title="Normalize all trajectories to start at 1.0. Useful for comparing groups of different sizes. Value of 2.0 = doubled, 0.5 = halved.",
                                    style="cursor: help;",
                                ),
                            ),
                            value=False,
                        ),
                        col_widths=[9, 3],
                    ),
                    ui.output_plot(
                        "biomass_timeseries", height=UI.plot_height_medium_px
                    ),
                ),
                ui.nav_panel(
                    "Catch",
                    ui.layout_columns(
                        ui.h4("Catch Trajectories", class_="mt-3"),
                        ui.div(
                            ui.input_action_button(
                                "btn_help_catch",
                                ui.span(
                                    ui.tags.i(class_="bi bi-question-circle me-1"),
                                    "Help",
                                ),
                                class_="btn-sm btn-outline-primary mt-3",
                            ),
                            style="text-align: right;",
                        ),
                        col_widths=[10, 2],
                    ),
                    ui.output_ui("help_catch"),
                    ui.output_plot("catch_timeseries", height=UI.plot_height_small_px),
                    ui.output_table("annual_catch_table"),
                ),
                ui.nav_panel(
                    "Summary",
                    ui.layout_columns(
                        ui.h4("Simulation Summary", class_="mt-3"),
                        ui.div(
                            ui.input_action_button(
                                "btn_help_summary",
                                ui.span(
                                    ui.tags.i(class_="bi bi-question-circle me-1"),
                                    "Help",
                                ),
                                class_="btn-sm btn-outline-primary mt-3",
                            ),
                            style="text-align: right;",
                        ),
                        col_widths=[10, 2],
                    ),
                    ui.output_ui("help_summary"),
                    ui.output_ui("summary_cards"),
                    ui.layout_columns(
                        ui.output_plot("final_biomass_plot"),
                        ui.output_plot("biomass_change_plot"),
                        col_widths=[6, 6],
                    ),
                ),
            ),
        ),
    )


def ecosim_server(
    input: Inputs,
    output: Outputs,
    session: Session,
    model_data: reactive.Value,
    sim_results: reactive.Value,
) -> None:
    """Ecosim simulation page server logic.

    Handles all server-side logic for the Ecosim simulation page including
    scenario creation, simulation execution, and results visualization.

    Parameters
    ----------
    input : Inputs
        Shiny input object containing user interface values
    output : Outputs
        Shiny output object for rendering UI elements
    session : Session
        Shiny session object for reactive programming
    model_data : reactive.Value
        Reactive value containing the balanced Ecopath model
    sim_results : reactive.Value
        Reactive value for storing simulation results

    Returns
    -------
    None
        This is a server function that sets up reactive effects and outputs
    """

    # Reactive values for this page
    scenario = reactive.Value(None)
    sim_output = reactive.Value(None)
    show_help_scenario = reactive.Value(False)
    show_help_progress = reactive.Value(False)
    show_help_timeseries = reactive.Value(False)
    show_help_catch = reactive.Value(False)
    show_help_summary = reactive.Value(False)
    _show_autofix_help = reactive.Value(False)

    @reactive.effect
    @reactive.event(input.btn_help_scenario)
    def _toggle_help_scenario():
        show_help_scenario.set(not show_help_scenario.get())

    @reactive.effect
    @reactive.event(input.btn_help_progress)
    def _toggle_help_progress():
        show_help_progress.set(not show_help_progress.get())

    @reactive.effect
    @reactive.event(input.btn_help_timeseries)
    def _toggle_help_timeseries():
        show_help_timeseries.set(not show_help_timeseries.get())

    @reactive.effect
    @reactive.event(input.btn_help_catch)
    def _toggle_help_catch():
        show_help_catch.set(not show_help_catch.get())

    @reactive.effect
    @reactive.event(input.btn_help_summary)
    def _toggle_help_summary():
        show_help_summary.set(not show_help_summary.get())

    @reactive.effect
    @reactive.event(input.btn_autofix_help)
    def _toggle_autofix_help():
        ui.modal_show(
            ui.modal(
                ui.h4("Automatic Parameter Calibration (Autofix)"),
                ui.tags.hr(),
                ui.div(
                    ui.h5("What does autofix do?"),
                    ui.p(
                        "The autofix module automatically adjusts simulation parameters to prevent crashes "
                        "and improve stability. It's enabled by default and highly recommended for all simulations."
                    ),
                    ui.h5("Parameters that get fixed:"),
                    ui.tags.ul(
                        ui.tags.li(
                            ui.tags.strong("VV (Vulnerability):"),
                            f" Capped at ≤ {THRESHOLDS.vv_cap} (prevents rapid prey depletion)",
                        ),
                        ui.tags.li(
                            ui.tags.strong("QQ (Density Dependence):"),
                            f" Capped at ≤ {THRESHOLDS.qq_cap} (reduces oscillations)",
                        ),
                        ui.tags.li(
                            ui.tags.strong("Minimum Biomass:"),
                            f" Raised to ≥ {THRESHOLDS.min_biomass} (prevents instant extinction)",
                        ),
                        ui.tags.li(
                            ui.tags.strong("DD (Prey Switching):"),
                            " Normalized to 1-2 range (stabilizes predation)",
                        ),
                    ),
                    ui.h5("Why is this needed?"),
                    ui.p(
                        "Ecosim is sensitive to extreme parameter values. Small numerical errors or unrealistic "
                        "values can cause groups to crash (biomass → 0). Autofix prevents most of these issues "
                        "by constraining parameters to ecologically reasonable ranges."
                    ),
                    ui.h5("What problems does it NOT fix?"),
                    ui.tags.ul(
                        ui.tags.li(
                            ui.tags.strong("EE > 1:"),
                            " Overconsumption requires rebalancing the Ecopath model",
                        ),
                        ui.tags.li(
                            ui.tags.strong("Fundamental model issues:"),
                            " Incomplete diets, missing groups, etc.",
                        ),
                    ),
                    ui.h5("Example results:"),
                    ui.div(
                        ui.tags.table(
                            ui.tags.tr(
                                ui.tags.th("Metric"),
                                ui.tags.th("Without Autofix"),
                                ui.tags.th("With Autofix"),
                            ),
                            ui.tags.tr(
                                ui.tags.td("Crashed groups"),
                                ui.tags.td("4"),
                                ui.tags.td("1"),
                            ),
                            ui.tags.tr(
                                ui.tags.td("Fixes applied"),
                                ui.tags.td("0"),
                                ui.tags.td("164"),
                            ),
                            ui.tags.tr(
                                ui.tags.td("Improvement"),
                                ui.tags.td("-"),
                                ui.tags.td("75% reduction"),
                            ),
                            class_="table table-bordered table-sm mt-2",
                        ),
                        class_="mb-3",
                    ),
                    ui.h5("When to disable autofix:"),
                    ui.p(
                        "Only disable if you're specifically testing extreme parameter values or "
                        "debugging model behavior. For normal use, keep it enabled."
                    ),
                    class_="p-3",
                ),
                title="Autofix Help",
                easy_close=True,
                footer=ui.modal_button("Close"),
            )
        )

    @output
    @render.ui
    def help_scenario_setup():
        if not show_help_scenario.get():
            return None
        return ui.div(
            ui.tags.div(
                ui.h5(
                    ui.tags.i(class_="bi bi-info-circle me-2"), "Scenario Setup Help"
                ),
                ui.tags.hr(),
                ui.h6("Purpose"),
                ui.p(
                    "This tab allows you to configure your Ecosim simulation before running it. "
                    "You set the time period, functional response parameters, fishing effort scenarios, "
                    "and stability options."
                ),
                ui.h6("Workflow"),
                ui.tags.ol(
                    ui.tags.li(
                        ui.tags.strong("Load Ecopath model"),
                        " in the Data Import or Ecopath pages",
                    ),
                    ui.tags.li(
                        ui.tags.strong("Configure settings"), " in the left sidebar:"
                    ),
                    ui.tags.ul(
                        ui.tags.li("Simulation years (1-500)"),
                        ui.tags.li("Integration method (RK4 recommended)"),
                        ui.tags.li("Vulnerability (functional response type)"),
                        ui.tags.li(
                            "Dynamic diet rewiring (optional - for adaptive foraging)"
                        ),
                        ui.tags.li("Fishing scenario"),
                        ui.tags.li("Enable/disable autofix (keep enabled)"),
                    ),
                    ui.tags.li(
                        ui.tags.strong("Click 'Create Scenario'"),
                        " - this validates parameters and applies fixes",
                    ),
                    ui.tags.li(
                        ui.tags.strong("Review"),
                        " the effort preview and biomass forcing options",
                    ),
                    ui.tags.li(ui.tags.strong("Click 'Run Simulation'"), " when ready"),
                ),
                ui.h6("Dynamic Diet Rewiring"),
                ui.p(
                    ui.tags.strong("Optional feature:"),
                    " Allows predator diet preferences to adapt based on prey availability (prey switching).",
                ),
                ui.tags.ul(
                    ui.tags.li(
                        ui.tags.strong("Switching Power (1-5):"),
                        " Controls how strongly predators switch to abundant prey. 1.0 = no switching, 2-3 = typical, >3 = opportunistic",
                    ),
                    ui.tags.li(
                        ui.tags.strong("Update Interval:"),
                        " How often diet is recalculated. Monthly (1) = responsive but slower, Annual (12) = faster but less responsive",
                    ),
                    ui.tags.li(
                        ui.tags.strong("Min Proportion:"),
                        " Prevents complete elimination of prey types from diet",
                    ),
                ),
                ui.h6("Fishing Scenarios"),
                ui.tags.ul(
                    ui.tags.li(
                        ui.tags.strong("Baseline:"),
                        " Effort stays constant at current levels",
                    ),
                    ui.tags.li(
                        ui.tags.strong("Increase:"),
                        " Gradual ramp-up in effort (% per year, starting from specified year)",
                    ),
                    ui.tags.li(
                        ui.tags.strong("Decrease:"), " Gradual ramp-down in effort"
                    ),
                    ui.tags.li(
                        ui.tags.strong("Closure:"),
                        " Fishing stops completely for a specified period",
                    ),
                    ui.tags.li(
                        ui.tags.strong("Custom:"),
                        " Upload CSV file with custom effort trajectory",
                    ),
                ),
                ui.h6("Tips"),
                ui.tags.ul(
                    ui.tags.li("Always enable autofix for first runs"),
                    ui.tags.li(
                        "Check how many fixes are applied - many fixes suggest model issues"
                    ),
                    ui.tags.li("Preview effort trajectory before running"),
                    ui.tags.li(
                        "Start with 50 years, increase if needed for long-term dynamics"
                    ),
                ),
                class_="alert alert-info mb-3",
            )
        )

    @output
    @render.ui
    def help_progress():
        if not show_help_progress.get():
            return None
        return ui.div(
            ui.tags.div(
                ui.h5(
                    ui.tags.i(class_="bi bi-info-circle me-2"),
                    "Simulation Progress Help",
                ),
                ui.tags.hr(),
                ui.h6("Understanding Simulation Status"),
                ui.p(
                    "This tab shows whether your simulation completed successfully and provides diagnostic information "
                    "if any groups experienced low biomass (crashes)."
                ),
                ui.h6("Status Messages"),
                ui.tags.ul(
                    ui.tags.li(
                        ui.tags.strong("Simulation completed successfully:"),
                        f" No crashes detected. All groups maintained biomass above threshold ({THRESHOLDS.crash_threshold}).",
                    ),
                    ui.tags.li(
                        ui.tags.strong("Low biomass detected (groups recovered):"),
                        " Some groups briefly dipped below threshold but bounced back. This is often okay - "
                        "it's a transient adjustment rather than a real crash. Check biomass plots to confirm recovery.",
                    ),
                    ui.tags.li(
                        ui.tags.strong("Population crash (groups did not recover):"),
                        " Groups went to near-zero biomass and stayed there. This indicates a real problem - "
                        "check your model parameters and Ecopath balance.",
                    ),
                ),
                ui.h6("What is a 'Crash'?"),
                ui.p(
                    f"A crash is detected when any group's biomass falls below {THRESHOLDS.crash_threshold} (1/10,000 of reference biomass). "
                    "This threshold filters out numerical noise while catching biologically meaningful crashes."
                ),
                ui.tags.ul(
                    ui.tags.li(
                        ui.tags.strong("Crash year:"),
                        " When the first group hit low biomass",
                    ),
                    ui.tags.li(
                        ui.tags.strong("Crashed groups:"),
                        " Which specific groups had problems",
                    ),
                    ui.tags.li(
                        ui.tags.strong("Recovery:"),
                        f" Whether groups bounced back (final biomass > {THRESHOLDS.recovery_threshold})",
                    ),
                ),
                ui.h6("What to Do If Crashes Occur"),
                ui.tags.ol(
                    ui.tags.li(
                        ui.tags.strong("Check if groups recovered:"),
                        " Look at the status message and crashed groups list",
                    ),
                    ui.tags.li(
                        ui.tags.strong("View biomass plots:"),
                        " Go to Time Series tab and plot the crashed groups",
                    ),
                    ui.tags.li(
                        ui.tags.strong("If recovered:"),
                        " It's likely just numerical adjustment - simulation is fine",
                    ),
                    ui.tags.li(ui.tags.strong("If not recovered:"), " Check:"),
                    ui.tags.ul(
                        ui.tags.li("Was autofix enabled? (Should be checked)"),
                        ui.tags.li(
                            "How many fixes were applied? (Many fixes suggest model problems)"
                        ),
                        ui.tags.li(
                            "Are any EE values > 1 in Ecopath? (Requires rebalancing)"
                        ),
                        ui.tags.li("Do crashed groups have very low initial biomass?"),
                        ui.tags.li("Are predator-prey relationships extreme?"),
                    ),
                ),
                ui.h6("Simulation Details Table"),
                ui.p("Shows key metrics:"),
                ui.tags.ul(
                    ui.tags.li("Years simulated - total time period"),
                    ui.tags.li("Groups / Living groups - model size"),
                    ui.tags.li("Crash year - when first crash occurred (or 'None')"),
                    ui.tags.li("Crashed groups - names or count of affected groups"),
                ),
                class_="alert alert-info mb-3",
            )
        )

    @output
    @render.ui
    def help_timeseries():
        if not show_help_timeseries.get():
            return None
        return ui.div(
            ui.tags.div(
                ui.h5(ui.tags.i(class_="bi bi-info-circle me-2"), "Time Series Help"),
                ui.tags.hr(),
                ui.h6("Purpose"),
                ui.p(
                    "Biomass trajectories show how each group's population changes over time. "
                    "This is the most important output for understanding ecosystem dynamics."
                ),
                ui.h6("How to Use"),
                ui.tags.ol(
                    ui.tags.li(
                        ui.tags.strong("Select groups:"),
                        " Use the dropdown to choose which groups to plot (up to ~10 for clarity)",
                    ),
                    ui.tags.li(
                        ui.tags.strong("Relative vs Absolute:"),
                        " Toggle 'Show Relative Biomass' to normalize to initial values",
                    ),
                    ui.tags.li(
                        ui.tags.strong("Interpret patterns:"),
                        " Look for trends, oscillations, crashes, or equilibria",
                    ),
                ),
                ui.h6("What to Look For"),
                ui.tags.ul(
                    ui.tags.li(
                        ui.tags.strong("Equilibrium:"),
                        " Biomass stays relatively constant (flat line) - system is stable",
                    ),
                    ui.tags.li(
                        ui.tags.strong("Oscillations:"),
                        " Regular up-and-down cycles - often predator-prey dynamics",
                    ),
                    ui.tags.li(
                        ui.tags.strong("Trends:"),
                        " Steady increase or decrease - shows long-term directional change",
                    ),
                    ui.tags.li(
                        ui.tags.strong("Crashes:"),
                        " Biomass drops to near-zero - indicates extinction or severe depletion",
                    ),
                    ui.tags.li(
                        ui.tags.strong("Recovery:"),
                        " Brief dip followed by return to normal - transient perturbation",
                    ),
                ),
                ui.h6("Interpreting Relative Biomass"),
                ui.p(
                    "When 'Show Relative Biomass' is checked, all trajectories start at 1.0. This makes it easier to compare "
                    "groups of different sizes:"
                ),
                ui.tags.ul(
                    ui.tags.li("Value = 1.0: No change from initial"),
                    ui.tags.li("Value = 2.0: Doubled since start"),
                    ui.tags.li("Value = 0.5: Halved since start"),
                    ui.tags.li("Value near 0: Crashed (went extinct)"),
                ),
                ui.h6("Common Patterns"),
                ui.tags.ul(
                    ui.tags.li(
                        ui.tags.strong("Predator-Prey Cycles:"),
                        " Predator and prey oscillate out of phase (prey peaks → predator peaks → prey crashes → predator crashes → repeat)",
                    ),
                    ui.tags.li(
                        ui.tags.strong("Fishing Impact:"),
                        " Target species decline, prey species increase, predator species may decline (loss of food)",
                    ),
                    ui.tags.li(
                        ui.tags.strong("Trophic Cascade:"),
                        " Changes propagate up/down food web (e.g., remove top predator → mesopredator increase → prey decrease)",
                    ),
                ),
                ui.h6("Tips"),
                ui.tags.ul(
                    ui.tags.li("Plot crashed groups first to see if they recovered"),
                    ui.tags.li("Plot predator-prey pairs together to see dynamics"),
                    ui.tags.li(
                        "Use relative biomass to focus on patterns not magnitudes"
                    ),
                    ui.tags.li(
                        "Compare scenarios by running multiple times with different settings"
                    ),
                ),
                class_="alert alert-info mb-3",
            )
        )

    @output
    @render.ui
    def help_catch():
        if not show_help_catch.get():
            return None
        return ui.div(
            ui.tags.div(
                ui.h5(ui.tags.i(class_="bi bi-info-circle me-2"), "Catch Data Help"),
                ui.tags.hr(),
                ui.h6("Purpose"),
                ui.p(
                    "Catch trajectories show how total fishery yield changes over time based on your fishing effort scenario."
                ),
                ui.h6("Understanding the Plot"),
                ui.p(
                    "The plot shows total annual catch (sum across all groups and gears). Catch depends on:"
                ),
                ui.tags.ul(
                    ui.tags.li(
                        ui.tags.strong("Effort:"), " Fishing pressure (set in scenario)"
                    ),
                    ui.tags.li(ui.tags.strong("Biomass:"), " Available stock"),
                    ui.tags.li(
                        ui.tags.strong("Catchability:"), " How easily fish are caught"
                    ),
                ),
                ui.h6("Interpreting the Table"),
                ui.p("The table shows catch by group for every 5 years. Use this to:"),
                ui.tags.ul(
                    ui.tags.li("Identify which groups contribute most to total catch"),
                    ui.tags.li("See how catch composition changes over time"),
                    ui.tags.li("Detect when a fishery is collapsing (catch → 0)"),
                ),
                ui.h6("Fishing Scenario Effects"),
                ui.tags.ul(
                    ui.tags.li(
                        ui.tags.strong("Baseline:"),
                        " Catch should roughly track biomass changes (if biomass stable, catch stable)",
                    ),
                    ui.tags.li(
                        ui.tags.strong("Increase effort:"),
                        " Catch may initially increase but often decreases as stocks are depleted",
                    ),
                    ui.tags.li(
                        ui.tags.strong("Decrease effort:"),
                        " Catch decreases but stocks may recover, leading to stable long-term yield",
                    ),
                    ui.tags.li(
                        ui.tags.strong("Closure:"),
                        " Catch drops to zero during closure, may rebound after if stocks recover",
                    ),
                ),
                ui.h6("Warning Signs"),
                ui.tags.ul(
                    ui.tags.li(
                        "Catch declining despite constant or increasing effort → stock depletion"
                    ),
                    ui.tags.li("Catch → 0 → fishery collapse"),
                    ui.tags.li(
                        "High variability → unstable fishery or extreme predator-prey dynamics"
                    ),
                ),
                ui.h6("Tips"),
                ui.tags.ul(
                    ui.tags.li(
                        "Compare catch plot with biomass trajectories to understand dynamics"
                    ),
                    ui.tags.li("Sustainable fishery: catch stable over time"),
                    ui.tags.li("Overfishing: catch peaks early then crashes"),
                    ui.tags.li("Use catch data to evaluate management scenarios"),
                ),
                class_="alert alert-info mb-3",
            )
        )

    @output
    @render.ui
    def help_summary():
        if not show_help_summary.get():
            return None
        return ui.div(
            ui.tags.div(
                ui.h5(ui.tags.i(class_="bi bi-info-circle me-2"), "Summary Help"),
                ui.tags.hr(),
                ui.h6("Purpose"),
                ui.p(
                    "The summary tab provides a quick overview of simulation outcomes, comparing initial and final states."
                ),
                ui.h6("Value Boxes"),
                ui.tags.ul(
                    ui.tags.li(
                        ui.tags.strong("Initial Biomass:"),
                        " Total system biomass at start (sum of all living groups)",
                    ),
                    ui.tags.li(
                        ui.tags.strong("Final Biomass:"), " Total system biomass at end"
                    ),
                    ui.tags.li(
                        ui.tags.strong("Biomass Change:"),
                        " Percent change from initial to final. "
                        "Green (positive) = system grew, Red (negative) = system declined",
                    ),
                    ui.tags.li(
                        ui.tags.strong("Total Catch:"),
                        " Sum of all catch over entire simulation period",
                    ),
                ),
                ui.h6("Initial vs Final Biomass Plot"),
                ui.p(
                    "Bar chart comparing each group's biomass at start (blue) and end (green). Use this to:"
                ),
                ui.tags.ul(
                    ui.tags.li("Identify winners (groups that increased)"),
                    ui.tags.li("Identify losers (groups that decreased)"),
                    ui.tags.li("Spot extinctions (final bar missing/tiny)"),
                ),
                ui.h6("Biomass Change Plot"),
                ui.p(
                    "Horizontal bar chart showing percent change for each group. "
                    "Green bars = increase, Red bars = decrease. Use this to:"
                ),
                ui.tags.ul(
                    ui.tags.li("Quickly see which groups changed most"),
                    ui.tags.li("Identify disproportionate impacts"),
                    ui.tags.li("Compare relative winners and losers"),
                ),
                ui.h6("Interpreting Results"),
                ui.tags.ul(
                    ui.tags.li(
                        ui.tags.strong("Healthy ecosystem:"),
                        " Small to moderate changes, no extinctions, total biomass stable or increasing",
                    ),
                    ui.tags.li(
                        ui.tags.strong("Stressed ecosystem:"),
                        " Large changes, some groups declining significantly, total biomass decreasing",
                    ),
                    ui.tags.li(
                        ui.tags.strong("Collapsed ecosystem:"),
                        " Multiple extinctions, total biomass down >50%, extreme changes",
                    ),
                ),
                ui.h6("What to Do Next"),
                ui.tags.ol(
                    ui.tags.li(
                        "If results look reasonable, save or export them (Results page)"
                    ),
                    ui.tags.li(
                        "If unexpected changes, check Time Series tab for dynamics"
                    ),
                    ui.tags.li(
                        "If crashes occurred, check Simulation Progress tab for diagnostics"
                    ),
                    ui.tags.li("Try different scenarios to compare outcomes"),
                    ui.tags.li(
                        "Adjust fishing effort or other parameters to explore management options"
                    ),
                ),
                class_="alert alert-info mb-3",
            )
        )

    @output
    @render.ui
    def fishing_params_ui():
        """Dynamic UI for fishing scenario parameters."""
        scenario_type = input.fishing_scenario()

        if scenario_type == "baseline":
            return ui.p(
                "Effort remains constant at baseline levels.", class_="text-muted"
            )

        elif scenario_type == "increase":
            return ui.div(
                ui.input_slider(
                    "effort_change_rate",
                    ui.span(
                        "Annual Increase (%) ",
                        ui.tags.i(
                            class_="bi bi-info-circle",
                            title="Percent increase in fishing effort per year. Example: 5% means effort multiplier increases by 0.05 each year.",
                            style="cursor: help;",
                        ),
                    ),
                    min=0,
                    max=50,
                    value=5,
                ),
                ui.input_numeric(
                    "effort_start_year",
                    ui.span(
                        "Start Year ",
                        ui.tags.i(
                            class_="bi bi-info-circle",
                            title="Year when effort starts increasing. Effort stays constant until this year.",
                            style="cursor: help;",
                        ),
                    ),
                    value=10,
                    min=1,
                ),
            )

        elif scenario_type == "decrease":
            return ui.div(
                ui.input_slider(
                    "effort_change_rate",
                    ui.span(
                        "Annual Decrease (%) ",
                        ui.tags.i(
                            class_="bi bi-info-circle",
                            title="Percent decrease in fishing effort per year. Example: 5% means effort multiplier decreases by 0.05 each year.",
                            style="cursor: help;",
                        ),
                    ),
                    min=0,
                    max=50,
                    value=5,
                ),
                ui.input_numeric(
                    "effort_start_year",
                    ui.span(
                        "Start Year ",
                        ui.tags.i(
                            class_="bi bi-info-circle",
                            title="Year when effort starts decreasing. Effort stays constant until this year.",
                            style="cursor: help;",
                        ),
                    ),
                    value=10,
                    min=1,
                ),
            )

        elif scenario_type == "closure":
            return ui.div(
                ui.input_numeric(
                    "closure_start_year",
                    ui.span(
                        "Closure Start Year ",
                        ui.tags.i(
                            class_="bi bi-info-circle",
                            title="Year when fishing stops completely. Effort = 0 during closure period.",
                            style="cursor: help;",
                        ),
                    ),
                    value=10,
                    min=1,
                ),
                ui.input_numeric(
                    "closure_duration",
                    ui.span(
                        "Duration (years) ",
                        ui.tags.i(
                            class_="bi bi-info-circle",
                            title="How many years the fishery closure lasts. After this, effort returns to baseline.",
                            style="cursor: help;",
                        ),
                    ),
                    value=10,
                    min=1,
                ),
            )

        else:  # custom
            return ui.p(
                "Upload custom effort CSV or define in Results tab.",
                class_="text-muted",
            )

    @reactive.effect
    @reactive.event(input.btn_create_scenario)
    def _create_scenario():
        """Create simulation scenario from model."""
        model = model_data.get()

        if model is None:
            ui.notification_show(
                "No Ecopath model available. Please balance a model first.",
                type="error",
            )
            return

        # Require a balanced Rpath model for Ecosim
        if not _require_balanced_model_or_notify(model):
            return

        try:
            years = range(1, input.sim_years() + 1)

            # Create scenario
            # Need original params - recreate from balanced model
            from pypath.core.params import create_rpath_params

            # Get groups and types safely
            if hasattr(model, "Group"):
                # It's a balanced Rpath object
                groups = list(model.Group)
                types = list(model.type)
            elif hasattr(model, "model") and "Group" in model.model.columns:
                # It's an RpathParams object
                groups = list(model.model["Group"])
                types = list(model.model["Type"])
            else:
                raise ValueError(
                    "Model object must be either Rpath or RpathParams type"
                )

            orig_params = create_rpath_params(groups, types)

            # Fill in the balanced parameter values
            orig_params.model["Biomass"] = model.Biomass
            orig_params.model["PB"] = model.PB
            orig_params.model["QB"] = model.QB
            orig_params.model["EE"] = model.EE
            orig_params.model["Unassim"] = model.Unassim
            orig_params.model["BioAcc"] = model.BA
            orig_params.model["Type"] = types

            # Reconstruct diet matrix from DC (diet composition)
            # DC is (ngroups + 1, nliving) where last row is import
            nliving = model.NUM_LIVING
            for i in range(model.NUM_GROUPS):
                for j in range(nliving):
                    if i < nliving:  # Living groups eat
                        orig_params.diet.iloc[i, j + 1] = model.DC[i, j]

            new_scenario = rsim_scenario(
                model, orig_params, years=years, vulnerability=input.vulnerability()
            )

            # Apply fishing scenario
            _apply_fishing_scenario(new_scenario, input)

            # Apply autofix if enabled
            if input.enable_autofix():
                new_scenario, report = validate_and_fix_scenario(
                    new_scenario, model, auto_fix=True, verbose=False
                )

                # Show what was fixed
                if report["fixes"]:
                    fix_count = len(report["fixes"])
                    ui.notification_show(
                        f"Applied {fix_count} stability fix{'es' if fix_count > 1 else ''}",
                        type="info",
                        duration=5,
                    )

            scenario.set(new_scenario)

            # Update group choices (use groups extracted earlier)
            num_living_dead = (
                model.NUM_LIVING + model.NUM_DEAD
                if hasattr(model, "NUM_LIVING")
                else len(groups)
            )
            group_names = groups[:num_living_dead]
            ui.update_selectize(
                "plot_groups", choices=group_names, selected=group_names[:3]
            )
            ui.update_select("forcing_group", choices=group_names)

            ui.notification_show("Scenario created successfully!", type="message")

        except Exception as e:
            ui.notification_show(f"Error creating scenario: {str(e)}", type="error")

    def _apply_fishing_scenario(scen: RsimScenario, input: Inputs):
        """Apply fishing scenario settings to scenario."""
        scenario_type = input.fishing_scenario()
        n_months = scen.fishing.ForcedEffort.shape[0]
        n_gears = (
            scen.fishing.ForcedEffort.shape[1] - 1
        )  # Subtract 1 for "Outside" column

        # If no fishing gears, nothing to modify
        if n_gears <= 0:
            return

        if scenario_type == "baseline":
            # Keep at 1.0
            pass

        elif scenario_type == "increase":
            rate = input.effort_change_rate() / 100
            start_year = input.effort_start_year()
            start_month = (start_year - 1) * 12

            for m in range(start_month, n_months):
                years_since = (m - start_month) / 12
                multiplier = 1.0 + rate * years_since
                scen.fishing.ForcedEffort[m, 1:] = multiplier

        elif scenario_type == "decrease":
            rate = input.effort_change_rate() / 100
            start_year = input.effort_start_year()
            start_month = (start_year - 1) * 12

            for m in range(start_month, n_months):
                years_since = (m - start_month) / 12
                multiplier = max(
                    THRESHOLDS.minimum_effort_multiplier, 1.0 - rate * years_since
                )
                scen.fishing.ForcedEffort[m, 1:] = multiplier

        elif scenario_type == "closure":
            start_year = input.closure_start_year()
            duration = input.closure_duration()
            start_month = (start_year - 1) * 12
            end_month = min(start_month + duration * 12, n_months)

            scen.fishing.ForcedEffort[start_month:end_month, 1:] = 0.0

    @output
    @render.ui
    def scenario_status():
        """Display scenario status."""
        scen = scenario.get()

        if scen is None:
            return ui.div(
                ui.tags.i(class_="bi bi-info-circle me-2"),
                "No scenario created. Load an Ecopath model and click 'Create Scenario'.",
                class_="alert alert-info",
            )

        return ui.div(
            ui.tags.i(class_="bi bi-check-circle me-2"),
            f"Scenario ready: {scen.params.NUM_GROUPS} groups, {input.sim_years()} years",
            class_="alert alert-success",
        )

    @output
    @render.plot
    def effort_preview_plot():
        """Preview effort forcing trajectory."""
        import matplotlib.pyplot as plt

        scen = scenario.get()

        fig, ax = plt.subplots(figsize=(8, 4))

        if scen is None:
            ax.text(
                0.5,
                0.5,
                "Create scenario to preview effort",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return fig

        # Check if there are any gears
        n_gears = scen.fishing.ForcedEffort.shape[1] - 1
        if n_gears <= 0:
            ax.text(
                0.5,
                0.5,
                "No fishing fleets in model",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return fig

        effort = scen.fishing.ForcedEffort[:, 1]  # First fleet
        months = np.arange(len(effort)) / 12

        ax.plot(months, effort, "b-", linewidth=2)
        ax.set_xlabel("Year")
        ax.set_ylabel("Effort Multiplier")
        ax.set_title("Fishing Effort Trajectory")
        ax.axhline(y=1, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlim(0, len(effort) / 12)
        ax.set_ylim(0, max(1.5, max(effort) * 1.1))

        plt.tight_layout()
        return fig

    @reactive.effect
    @reactive.event(input.btn_run_sim)
    def _run_simulation():
        """Run the Ecosim simulation."""
        scen = scenario.get()

        if scen is None:
            ui.notification_show("Create a scenario first", type="warning")
            return

        try:
            # Check if diet rewiring is enabled
            diet_rewiring_enabled = input.enable_diet_rewiring()

            if diet_rewiring_enabled:
                ui.notification_show(
                    "Running simulation with diet rewiring...",
                    type="message",
                    duration=2,
                )

                # Create diet rewiring configuration
                diet_rewiring = DietRewiring(
                    enabled=True,
                    switching_power=input.switching_power(),
                    update_interval=int(input.rewiring_interval()),
                    min_proportion=input.min_diet_proportion(),
                )

                # Run advanced simulation with diet rewiring
                output = rsim_run_advanced(
                    scen,
                    state_forcing=None,
                    diet_rewiring=diet_rewiring,
                    method=input.integration_method(),
                )
            else:
                ui.notification_show(
                    "Running simulation...", type="message", duration=2
                )

                # Run standard simulation
                output = rsim_run(scen, method=input.integration_method())

            sim_output.set(output)
            sim_results.set(output)

            # Build informative crash message
            if output.crash_year > 0:
                # Get crashed group names
                crashed_names = [scen.params.spname[i] for i in output.crashed_groups]

                # Format group list
                if len(crashed_names) <= 3:
                    groups_str = ", ".join(crashed_names)
                else:
                    groups_str = f"{', '.join(crashed_names[:3])}, +{len(crashed_names) - 3} more"

                # Check if groups recovered
                _final_biomass = {
                    i: output.end_state.Biomass[i] for i in output.crashed_groups
                }
                recovered = [
                    name
                    for i, name in zip(output.crashed_groups, crashed_names)
                    if output.end_state.Biomass[i] > THRESHOLDS.recovery_threshold
                ]

                if recovered:
                    msg = f"Low biomass detected in year {output.crash_year} ({groups_str}). "
                    msg += "Groups recovered - check plots for details."
                    msg_type = "info"
                else:
                    msg = (
                        f"Population crash at year {output.crash_year}: {groups_str}. "
                    )
                    msg += "Groups did not recover."
                    msg_type = "warning"

                ui.notification_show(msg, type=msg_type, duration=10)
            else:
                success_msg = "Simulation completed successfully!"
                if diet_rewiring_enabled:
                    success_msg += " Diet rewiring was applied."
                ui.notification_show(success_msg, type="message")

        except Exception as e:
            ui.notification_show(f"Simulation error: {str(e)}", type="error")

    @output
    @render.ui
    def simulation_status():
        """Display simulation status."""
        output = sim_output.get()
        scen = scenario.get()

        if output is None:
            return ui.div(
                ui.tags.i(class_="bi bi-hourglass me-2"),
                "Simulation not yet run. Create scenario and click 'Run Simulation'.",
                class_="alert alert-secondary",
            )

        if output.crash_year > 0:
            # Get crashed group names
            crashed_names = [scen.params.spname[i] for i in output.crashed_groups]

            # Format group list
            if len(crashed_names) <= 3:
                groups_str = ", ".join(crashed_names)
            else:
                groups_str = (
                    f"{', '.join(crashed_names[:3])}, +{len(crashed_names) - 3} more"
                )

            # Check if groups recovered
            recovered = [
                name
                for i, name in zip(output.crashed_groups, crashed_names)
                if output.end_state.Biomass[i] > THRESHOLDS.recovery_threshold
            ]

            if recovered:
                msg = f"Low biomass detected in year {output.crash_year} for: {groups_str}. "
                msg += f"{len(recovered)}/{len(crashed_names)} group(s) recovered."
                alert_class = "alert alert-info"
                icon_class = "bi bi-info-circle me-2"
            else:
                msg = (
                    f"Population crash at year {output.crash_year} for: {groups_str}. "
                )
                msg += "Groups did not recover."
                alert_class = "alert alert-warning"
                icon_class = "bi bi-exclamation-triangle me-2"

            return ui.div(ui.tags.i(class_=icon_class), msg, class_=alert_class)

        return ui.div(
            ui.tags.i(class_="bi bi-check-circle me-2"),
            f"Simulation completed successfully: {output.params['years']} years simulated",
            class_="alert alert-success",
        )

    @output
    @render.ui
    def progress_display():
        """Display progress/completion info."""
        output = sim_output.get()
        scen = scenario.get()
        if output is None:
            return None

        # Build crashed groups info
        if output.crash_year > 0 and scen is not None:
            crashed_names = [scen.params.spname[i] for i in output.crashed_groups]
            crashed_info = (
                ", ".join(crashed_names)
                if len(crashed_names) <= 5
                else f"{len(crashed_names)} groups"
            )
        else:
            crashed_info = "None"

        return ui.div(
            ui.tags.h5("Simulation Details"),
            ui.tags.table(
                ui.tags.tr(
                    ui.tags.td("Years simulated:"),
                    ui.tags.td(str(output.params["years"])),
                ),
                ui.tags.tr(
                    ui.tags.td("Groups:"), ui.tags.td(str(output.params["NUM_GROUPS"]))
                ),
                ui.tags.tr(
                    ui.tags.td("Living groups:"),
                    ui.tags.td(str(output.params["NUM_LIVING"])),
                ),
                ui.tags.tr(
                    ui.tags.td("Crash year:"),
                    ui.tags.td(
                        "None" if output.crash_year < 0 else str(output.crash_year)
                    ),
                ),
                ui.tags.tr(ui.tags.td("Crashed groups:"), ui.tags.td(crashed_info)),
                class_="table table-sm",
            ),
        )

    @output
    @render.plot
    def biomass_timeseries():
        """Plot biomass time series."""
        import matplotlib.pyplot as plt

        output = sim_output.get()
        scen = scenario.get()

        fig, ax = plt.subplots(figsize=(12, 6))

        if output is None or scen is None:
            ax.text(
                0.5,
                0.5,
                "Run simulation to see results",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return fig

        selected_groups = input.plot_groups()
        if not selected_groups:
            ax.text(
                0.5,
                0.5,
                "Select groups to plot",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return fig

        # Get time and biomass data
        n_months = output.out_Biomass.shape[0]
        time = np.arange(n_months) / 12

        # Get group indices
        group_names = scen.params.spname[1:]  # Skip "Outside"

        for group in selected_groups:
            if group in group_names:
                idx = group_names.index(group) + 1  # +1 for Outside offset

                biomass = output.out_Biomass[:, idx]

                if input.relative_biomass():
                    if biomass[0] > 0:
                        biomass = biomass / biomass[0]
                    else:
                        biomass = np.zeros_like(biomass)

                ax.plot(time, biomass, label=group, linewidth=2)

        ax.set_xlabel("Year")
        ax.set_ylabel("Relative Biomass" if input.relative_biomass() else "Biomass")
        ax.set_title("Biomass Trajectories")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
        ax.set_xlim(0, max(time))

        if input.relative_biomass():
            ax.axhline(y=1, color="gray", linestyle="--", alpha=0.5)

        plt.tight_layout()
        return fig

    @output
    @render.plot
    def catch_timeseries():
        """Plot catch time series."""
        import matplotlib.pyplot as plt

        output = sim_output.get()
        scen = scenario.get()

        fig, ax = plt.subplots(figsize=(12, 5))

        if output is None or scen is None:
            ax.text(
                0.5,
                0.5,
                "Run simulation to see catch data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return fig

        # Annual catch data
        years = np.arange(output.annual_Catch.shape[0]) + 1
        total_catch = np.sum(output.annual_Catch[:, 1:], axis=1)

        ax.fill_between(years, total_catch, alpha=0.3)
        ax.plot(years, total_catch, "b-", linewidth=2, label="Total Catch")

        ax.set_xlabel("Year")
        ax.set_ylabel("Catch")
        ax.set_title("Total Annual Catch")
        ax.set_xlim(1, len(years))

        plt.tight_layout()
        return fig

    @output
    @render.table
    def annual_catch_table():
        """Display annual catch summary."""
        output = sim_output.get()
        scen = scenario.get()

        if output is None or scen is None:
            return pd.DataFrame()

        # Create summary table
        group_names = scen.params.spname[1 : scen.params.NUM_LIVING + 1]

        catch_df = pd.DataFrame(
            output.annual_Catch[:, 1 : scen.params.NUM_LIVING + 1], columns=group_names
        )
        catch_df.insert(0, "Year", range(1, len(catch_df) + 1))

        # Show every 5 years
        return catch_df[catch_df["Year"] % 5 == 0].round(3)

    @output
    @render.ui
    def summary_cards():
        """Display summary statistics cards."""
        output = sim_output.get()
        scen = scenario.get()

        if output is None or scen is None:
            return ui.p("Run simulation to see summary.", class_="text-muted")

        # Calculate summary stats
        initial_biomass = np.sum(output.out_Biomass[0, 1 : scen.params.NUM_LIVING + 1])
        final_biomass = np.sum(output.out_Biomass[-1, 1 : scen.params.NUM_LIVING + 1])
        total_catch = np.sum(output.annual_Catch[:, 1:])
        biomass_change = (final_biomass - initial_biomass) / initial_biomass * 100

        return ui.layout_columns(
            ui.value_box(
                "Initial Biomass",
                f"{initial_biomass:.2f}",
                showcase=ui.tags.i(class_="bi bi-box"),
            ),
            ui.value_box(
                "Final Biomass",
                f"{final_biomass:.2f}",
                showcase=ui.tags.i(class_="bi bi-box-fill"),
                theme="primary" if biomass_change >= 0 else "danger",
            ),
            ui.value_box(
                "Biomass Change",
                f"{biomass_change:+.1f}%",
                showcase=ui.tags.i(
                    class_=(
                        "bi bi-arrow-up" if biomass_change >= 0 else "bi bi-arrow-down"
                    )
                ),
                theme="success" if biomass_change >= 0 else "danger",
            ),
            ui.value_box(
                "Total Catch",
                f"{total_catch:.2f}",
                showcase=ui.tags.i(class_="bi bi-basket"),
            ),
            col_widths=[3, 3, 3, 3],
        )

    @output
    @render.plot
    def final_biomass_plot():
        """Plot final vs initial biomass."""
        import matplotlib.pyplot as plt

        output = sim_output.get()
        scen = scenario.get()

        fig, ax = plt.subplots(figsize=(8, 5))

        if output is None or scen is None:
            ax.text(
                0.5,
                0.5,
                "Run simulation first",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return fig

        n_living = scen.params.NUM_LIVING
        group_names = scen.params.spname[1 : n_living + 1]

        initial = output.out_Biomass[0, 1 : n_living + 1]
        final = output.out_Biomass[-1, 1 : n_living + 1]

        x = np.arange(len(group_names))
        width = 0.35

        ax.bar(x - width / 2, initial, width, label="Initial", color="#3498db")
        ax.bar(x + width / 2, final, width, label="Final", color="#2ecc71")

        ax.set_xlabel("Group")
        ax.set_ylabel("Biomass")
        ax.set_title("Initial vs Final Biomass")
        ax.set_xticks(x)
        ax.set_xticklabels(group_names, rotation=45, ha="right")
        ax.legend()

        plt.tight_layout()
        return fig

    @output
    @render.plot
    def biomass_change_plot():
        """Plot percent biomass change."""
        import matplotlib.pyplot as plt

        output = sim_output.get()
        scen = scenario.get()

        fig, ax = plt.subplots(figsize=(8, 5))

        if output is None or scen is None:
            ax.text(
                0.5,
                0.5,
                "Run simulation first",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return fig

        n_living = scen.params.NUM_LIVING
        group_names = scen.params.spname[1 : n_living + 1]

        initial = output.out_Biomass[0, 1 : n_living + 1]
        final = output.out_Biomass[-1, 1 : n_living + 1]

        pct_change = np.where(initial > 0, (final - initial) / initial * 100, 0)

        colors = ["#2ecc71" if c >= 0 else "#e74c3c" for c in pct_change]

        ax.barh(group_names, pct_change, color=colors)
        ax.axvline(x=0, color="gray", linestyle="-")
        ax.set_xlabel("Percent Change (%)")
        ax.set_title("Biomass Change by Group")

        plt.tight_layout()
        return fig
