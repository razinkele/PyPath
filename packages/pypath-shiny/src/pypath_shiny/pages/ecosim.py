"""Ecosim simulation page module."""

import logging

import numpy as np
import pandas as pd

# pypath imports (path setup handled by app/__init__.py)
from pypath.core.autofix import validate_and_fix_scenario
from pypath.core.ecosim import RsimScenario, rsim_run, rsim_scenario
from pypath.core.ecosim_advanced import rsim_run_advanced
from pypath.core.forcing import DietRewiring
from shiny import Inputs, Outputs, Session, reactive, render, ui
from shiny.types import SilentException

try:
    from shinywidgets import output_widget, render_widget
except ImportError:  # pragma: no cover

    def output_widget(*args, **kwargs):
        kwargs.pop("fillable", None)  # output_plot has no fillable
        return ui.output_plot(*args, **kwargs)

    def render_widget(func):
        return render.plot(func)


# Import centralized configuration
from pypath_shiny.config import PARAM_RANGES, THRESHOLDS, UI
from pypath_shiny.pages.ewe_scenario import (
    build_ewe_scenario,
    ewe_source_matches,
)
from pypath_shiny.services.ecosim_controls import (
    apply_fishing_management_control,
    apply_metabolism_control,
    apply_recruitment_control,
    build_control_metadata,
    validate_fishing_management_control,
    validate_metabolism_control,
    validate_recruitment_control,
)
from pypath_shiny.services.notifications import (
    notify_error,
    notify_info,
    notify_success,
    notify_warning,
)
from pypath_shiny.services.plotly_viz import (
    build_biomass_change_figure,
    build_biomass_figure,
    build_catch_figure,
    build_effort_figure,
    build_initial_final_figure,
)
from pypath_shiny.services.tasks import (
    TaskState,
    TaskStatus,
    apply_if_latest,
    finish_task,
    new_running_task,
    update_task_progress,
)


# Helper to check model balance and show notification if not
def _require_balanced_model_or_notify(model) -> bool:
    """Return True if model is balanced, otherwise show a UI error and return False."""
    from pypath_shiny.pages.utils import is_balanced_model

    if not is_balanced_model(model):
        notify_error(
            "Ecosim requires a balanced Ecopath model. Balance the model on the Ecopath page first.",
            duration=6,
        )
        return False
    return True


logger = logging.getLogger(__name__)


def detritus_self_fate_message(model) -> str | None:
    """User-facing warning for detritus pools that flow to themselves."""
    from pypath.core.ecosim import detritus_self_fates

    try:
        self_fates = detritus_self_fates(model, model.NUM_LIVING, model.NUM_DEAD)
    except Exception as e:
        logger.debug("detritus self-fate check failed: %s", e)
        return None
    if not self_fates:
        return None
    names = ", ".join(self_fates)
    return (
        f"Detritus fate to itself is above 0 for {names}. Ecosim keeps that "
        "share of the pool's surplus in the pool, so it grows away from the "
        "Ecopath balance. On the Ecopath page, set the group's fate to itself "
        "to 0."
    )


def _notify_detritus_self_fate(model) -> None:
    msg = detritus_self_fate_message(model)
    if msg:
        notify_warning(msg, duration=12)


def build_task_status_alert(task_state: TaskState) -> tuple[str, str, str]:
    """Map task state to bootstrap alert class, icon, and message."""
    if task_state.status is TaskStatus.RUNNING:
        return (
            "alert alert-info",
            "bi bi-arrow-repeat me-2",
            f"{task_state.message} ({task_state.progress:.0f}%)",
        )
    if task_state.status is TaskStatus.ERROR:
        msg = task_state.error or task_state.message or "Simulation failed."
        return ("alert alert-danger", "bi bi-exclamation-triangle me-2", msg)
    if task_state.status is TaskStatus.SUCCESS:
        return (
            "alert alert-success",
            "bi bi-check-circle me-2",
            task_state.message or "Simulation completed.",
        )
    if task_state.status is TaskStatus.CANCELLED:
        return (
            "alert alert-warning",
            "bi bi-pause-circle me-2",
            task_state.message or "Simulation cancelled.",
        )
    return (
        "alert alert-secondary",
        "bi bi-hourglass me-2",
        "Simulation not yet run. Create scenario and click 'Run Simulation'.",
    )


def apply_prey_forcing(scen, group, multiplier):
    """Set the prey-availability multiplier for one group on a scenario.

    `ForcedPrey` is (n_months, NUM_GROUPS + 1) and 1-based in the group axis;
    the core multiplies prey availability by it each month. Each call starts
    again from the scenario's own forcing (kept on the first call), so runs
    do not accumulate and an EwE scenario's forcing shapes survive; the
    group's forcing is scaled by the multiplier.

    Returns True when a multiplier was applied.
    """
    forcing = getattr(scen, "forcing", None)
    if forcing is None or getattr(forcing, "ForcedPrey", None) is None:
        return False

    base = getattr(scen, "_base_forced_prey", None)
    if base is None or np.shape(base) != np.shape(forcing.ForcedPrey):
        base = np.array(forcing.ForcedPrey, dtype=float, copy=True)
        scen._base_forced_prey = base
    forcing.ForcedPrey[:] = base

    if multiplier is None or float(multiplier) == 1.0 or not group:
        return False

    spname = [str(n) for n in getattr(scen.params, "spname", [])]
    if group not in spname:
        return False

    col = spname.index(group)
    forcing.ForcedPrey[:, col] = base[:, col] * float(multiplier)
    return True


def scenario_from_balanced_model(
    model, groups, types, years, vulnerability, fast_group_rule="ewe"
) -> RsimScenario:
    """Ecosim scenario from the app's balanced model.

    One vulnerability for every link (the page's slider) and the chosen
    fast-group rule ('ewe' or 'rpath').
    """
    # rsim_scenario needs a params object next to the balanced model
    from pypath.core.params import create_rpath_params

    orig_params = create_rpath_params(groups, types)

    # Fill in the balanced parameter values
    orig_params.model["Biomass"] = model.Biomass
    orig_params.model["PB"] = model.PB
    orig_params.model["QB"] = model.QB
    orig_params.model["EE"] = model.EE
    orig_params.model["Unassim"] = model.Unassim
    orig_params.model["BioAcc"] = model.BA
    orig_params.model["Type"] = types

    # Note: the diet matrix is not copied back into orig_params -
    # rsim_scenario rebuilds it from rpath.DC via rsim_params and only
    # reads the 'model' attribute of the params object.

    return rsim_scenario(
        model,
        orig_params,
        years=years,
        vulnerability=vulnerability,
        fast_group_rule=fast_group_rule,
    )


def ecosim_ui() -> ui.Tag:
    """Ecosim simulation page UI."""
    return ui.div(
        ui.h2("Ecosim Dynamic Simulation", class_="mb-4"),
        ui.layout_sidebar(
            # Sidebar for simulation settings
            ui.sidebar(
                ui.h4("Simulation Settings"),
                # Scenario stored in an imported EwE database (when it matches)
                ui.output_ui("ewe_scenario_source_ui"),
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
                ui.input_select(
                    "fast_group_rule",
                    ui.span(
                        "Fast Groups ",
                        ui.tags.i(
                            class_="bi bi-info-circle",
                            title="Groups that turn over much faster than the time step (plankton, detritus). EwE6 solves them at equilibrium in every step and takes one RK4 step per month; Rpath integrates them like every other group with four RK4 steps per month.",
                            style="cursor: help;",
                        ),
                    ),
                    choices={
                        "ewe": "EwE6: at equilibrium",
                        "rpath": "Rpath: integrated",
                    },
                    selected="ewe",
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
                    },
                    selected="baseline",
                ),
                ui.output_ui("fishing_params_ui"),
                ui.tags.hr(),
                # Recruitment controls
                ui.h5("Recruitment Dynamics"),
                ui.input_checkbox(
                    "enable_recruitment_control",
                    "Enable stock-recruitment model",
                    value=False,
                ),
                ui.panel_conditional(
                    "input.enable_recruitment_control",
                    ui.input_select(
                        "recruitment_group",
                        "Target Group",
                        choices=["(Create scenario first)"],
                    ),
                    ui.input_select(
                        "recruitment_model",
                        "Model Type",
                        choices={
                            "ricker": "Ricker",
                            "bevertonholt": "Beverton-Holt",
                            "myers": "Myers",
                        },
                        selected="ricker",
                    ),
                    ui.output_ui("recruitment_params_ui"),
                    ui.input_numeric(
                        "recruitment_env_multiplier",
                        "Environmental Multiplier",
                        value=1.0,
                        min=PARAM_RANGES.recruitment_env_multiplier_min,
                        max=PARAM_RANGES.recruitment_env_multiplier_max,
                        step=0.1,
                    ),
                ),
                ui.tags.hr(),
                # Metabolism controls
                ui.h5("Metabolism Scaling"),
                ui.input_checkbox(
                    "enable_metabolism_control",
                    "Enable temperature/size scaling",
                    value=False,
                ),
                ui.panel_conditional(
                    "input.enable_metabolism_control",
                    ui.input_slider(
                        "metabolism_temperature_delta",
                        "Temperature Delta (°C)",
                        min=PARAM_RANGES.metabolism_temp_delta_min,
                        max=PARAM_RANGES.metabolism_temp_delta_max,
                        value=PARAM_RANGES.metabolism_temp_delta_default,
                        step=0.5,
                    ),
                    ui.input_slider(
                        "metabolism_q10",
                        "Q10 Coefficient",
                        min=PARAM_RANGES.metabolism_q10_min,
                        max=PARAM_RANGES.metabolism_q10_max,
                        value=PARAM_RANGES.metabolism_q10_default,
                        step=0.1,
                    ),
                    ui.input_numeric(
                        "metabolism_mass_exponent",
                        "Mass Exponent",
                        value=PARAM_RANGES.metabolism_mass_exponent_default,
                        min=PARAM_RANGES.metabolism_mass_exponent_min,
                        max=PARAM_RANGES.metabolism_mass_exponent_max,
                        step=0.05,
                    ),
                    ui.input_numeric(
                        "metabolism_reference_mass",
                        "Reference Mass",
                        value=PARAM_RANGES.metabolism_reference_mass_default,
                        min=PARAM_RANGES.metabolism_reference_mass_min,
                        step=0.1,
                    ),
                    ui.input_numeric(
                        "metabolism_activity_budget",
                        "Activity Budget",
                        value=PARAM_RANGES.metabolism_activity_budget_default,
                        min=PARAM_RANGES.metabolism_activity_budget_min,
                        max=PARAM_RANGES.metabolism_activity_budget_max,
                        step=0.1,
                    ),
                    ui.layout_columns(
                        ui.input_checkbox(
                            "metabolism_apply_consumers",
                            "Apply to consumers",
                            value=True,
                        ),
                        ui.input_checkbox(
                            "metabolism_apply_producers",
                            "Apply to producers",
                            value=True,
                        ),
                        col_widths=[6, 6],
                    ),
                ),
                ui.tags.hr(),
                # Adaptive fishing management controls
                ui.h5("Adaptive Fishing Management"),
                ui.input_checkbox(
                    "enable_fishing_management_control",
                    "Enable fleet management strategy",
                    value=False,
                ),
                ui.panel_conditional(
                    "input.enable_fishing_management_control",
                    ui.input_select(
                        "fishing_management_strategy",
                        "Management Strategy",
                        choices={
                            "sustainability": "Sustainability reduction",
                            "revenue": "Revenue-focused allocation",
                            "bycatch": "Bycatch minimization",
                        },
                        selected="sustainability",
                    ),
                    ui.output_ui("fishing_management_params_ui"),
                ),
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
                            title=f"Automatically caps VV ≤ {THRESHOLDS.vv_cap}, QQ ≤ {THRESHOLDS.qq_cap}, ensures minimum biomass ≥ {THRESHOLDS.min_biomass}, and normalizes DD to 1-2. Can stop crashes caused by extreme parameter values, but QQ and DD are derived from the balanced model, so changing them moves the run off its starting equilibrium. Off by default.",
                            style="cursor: help;",
                        ),
                    ),
                    # Off by default: capping the derived QQ/DD made balanced
                    # models drift (example Seals 0.025 -> 0.0106 in 10 years).
                    value=False,
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
                        fill=False,  # header row: do not stretch to the tab height
                    ),
                    ui.output_ui("help_scenario_setup"),
                    ui.output_ui("scenario_status"),
                    ui.layout_columns(
                        ui.card(
                            ui.card_header("Effort Forcing"),
                            output_widget(
                                "effort_preview_plot",
                                height=UI.plot_height_small_px,
                                fillable=True,
                            ),
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
                        fill=False,  # header row: do not stretch to the tab height
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
                        fill=False,  # header row: do not stretch to the tab height
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
                        fill=False,  # header row: do not stretch to the tab height
                    ),
                    output_widget(
                        "biomass_timeseries",
                        height=UI.plot_height_medium_px,
                        fillable=True,
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
                        fill=False,  # header row: do not stretch to the tab height
                    ),
                    ui.output_ui("help_catch"),
                    output_widget(
                        "catch_timeseries",
                        height=UI.plot_height_small_px,
                        fillable=True,
                    ),
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
                        fill=False,  # header row: do not stretch to the tab height
                    ),
                    ui.output_ui("help_summary"),
                    ui.output_ui("summary_cards"),
                    ui.layout_columns(
                        output_widget(
                            "final_biomass_plot",
                            height=UI.plot_height_medium_px,
                            fillable=True,
                        ),
                        output_widget(
                            "biomass_change_plot",
                            height=UI.plot_height_medium_px,
                            fillable=True,
                        ),
                        col_widths=[6, 6],
                        fill=False,  # do not stretch to the tab height
                    ),
                ),
            ),
        ),
    )


def read_optional_input(input, name: str, default):
    """Read an input that may not exist in the browser.

    Inputs created by an ``output_ui`` inside a hidden conditional panel are
    never rendered, and reading one raises ``SilentException``.
    ``hasattr(input, name)`` is always True for Shiny's Inputs, so it is no
    guard. Returns ``default`` when the input is missing or None.
    """
    try:
        value = getattr(input, name)()
    except (SilentException, AttributeError, KeyError, TypeError):
        return default
    return default if value is None else value


def ecosim_server(
    input: Inputs,
    output: Outputs,
    session: Session,
    model_data: reactive.Value,
    sim_results: reactive.Value,
    sim_scenario: reactive.Value = None,
    ewe_source: reactive.Value = None,
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
    ewe_source : reactive.Value, optional
        The EwE database last imported on the Data Import page (see
        :func:`pypath_shiny.pages.ewe_scenario.ewe_source_record`); its
        Ecosim scenarios are offered while the model has its groups.

    Returns
    -------
    None
        This is a server function that sets up reactive effects and outputs
    """

    # Reactive values for this page
    scenario = reactive.Value(None)
    sim_output = reactive.Value(None)
    sim_task_state = reactive.Value(TaskState(run_id="ecosim-idle"))
    # Years the current scenario was actually built with (input.sim_years() may
    # have been changed since).
    scenario_years = reactive.Value(None)
    scenario_controls_metadata = reactive.Value({})
    show_help_scenario = reactive.Value(False)
    show_help_progress = reactive.Value(False)
    show_help_timeseries = reactive.Value(False)
    show_help_catch = reactive.Value(False)
    show_help_summary = reactive.Value(False)

    def _input_or_default(name: str, default):
        """Read a UI input value with fallback for conditional controls."""
        return read_optional_input(input, name, default)

    def _matching_ewe_source():
        """The imported EwE database, when the current model has its groups."""
        if ewe_source is None:
            return None
        source = ewe_source.get()
        return source if ewe_source_matches(source, model_data.get()) else None

    @output
    @render.ui
    def ewe_scenario_source_ui():
        """Offer the Ecosim scenarios stored in the imported EwE database."""
        source = _matching_ewe_source()
        if source is None:
            return ui.div()
        scenarios = {str(k): v for k, v in source["scenarios"].items()}
        default = str(source.get("scenario"))
        if default not in scenarios:
            default = next(iter(scenarios))
        datasets = {"": "None"}
        datasets.update({str(k): v for k, v in source.get("datasets", {}).items()})
        return ui.div(
            ui.h5("Scenario Source"),
            ui.input_radio_buttons(
                "scenario_source",
                "Build the scenario from",
                {
                    "model": "The balanced model and the settings below",
                    "ewe": f"An Ecosim scenario stored in {source['name']}",
                },
                selected="model",
            ),
            ui.panel_conditional(
                "input.scenario_source === 'ewe'",
                ui.input_select(
                    "ewe_scenario_id", "EwE scenario", scenarios, selected=default
                ),
                ui.input_select(
                    "ewe_dataset_id",
                    "Time-series dataset (drivers)",
                    datasets,
                    selected="",
                ),
                ui.p(
                    "Runs the database's Ecopath inputs with the scenario's own "
                    "vulnerabilities, feeding time, switching, forcing and "
                    "length, as EwE does. Edits made in this app and the "
                    "vulnerability slider are not used.",
                    class_="text-muted small",
                ),
            ),
            ui.tags.hr(),
        )

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
                        ui.tags.li("Simulation years (2-500)"),
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
                    f"A crash is detected when any group's biomass falls below the absolute "
                    f"threshold {THRESHOLDS.crash_threshold} t/km². "
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

        return ui.p("", class_="text-muted")

    @output
    @render.ui
    def recruitment_params_ui():
        """Dynamic UI for recruitment model parameters."""
        model_type = input.recruitment_model()

        if model_type in {"ricker", "bevertonholt"}:
            return ui.div(
                ui.input_numeric(
                    "recruitment_alpha", "Alpha", value=1.2, min=0.001, step=0.05
                ),
                ui.input_numeric(
                    "recruitment_beta", "Beta", value=0.1, min=0.001, step=0.01
                ),
                ui.input_numeric(
                    "recruitment_rmax", "Rmax", value=1.5, min=0.1, step=0.1
                ),
            )

        if model_type == "myers":
            return ui.div(
                ui.input_numeric(
                    "recruitment_rmax", "Rmax", value=1.5, min=0.1, step=0.1
                ),
                ui.input_numeric("recruitment_k", "K", value=1.0, min=0.001, step=0.05),
                ui.input_numeric(
                    "recruitment_power", "Power", value=1.0, min=0.1, max=4.0, step=0.1
                ),
            )

        return ui.p("", class_="text-muted")

    @output
    @render.ui
    def fishing_management_params_ui():
        """Dynamic UI for adaptive fishing-management parameters."""
        strategy = input.fishing_management_strategy()

        if strategy == "sustainability":
            return ui.div(
                ui.input_slider(
                    "sustainability_reduction",
                    "Effort Retention",
                    min=PARAM_RANGES.sustainability_reduction_min,
                    max=PARAM_RANGES.sustainability_reduction_max,
                    value=PARAM_RANGES.sustainability_reduction_default,
                    step=0.05,
                ),
                ui.input_slider(
                    "sustainability_floor",
                    "Minimum Effort Floor",
                    min=PARAM_RANGES.sustainability_floor_min,
                    max=PARAM_RANGES.sustainability_floor_max,
                    value=PARAM_RANGES.sustainability_floor_default,
                    step=0.05,
                ),
            )

        if strategy == "revenue":
            return ui.input_numeric(
                "fishing_effort_budget",
                "Effort Budget",
                value=PARAM_RANGES.fishing_effort_budget_default,
                min=PARAM_RANGES.fishing_effort_budget_min,
                step=0.1,
            )

        if strategy == "bycatch":
            return ui.input_slider(
                "bycatch_reduction",
                "Bycatch Reduction Fraction",
                min=PARAM_RANGES.bycatch_reduction_min,
                max=PARAM_RANGES.bycatch_reduction_max,
                value=PARAM_RANGES.bycatch_reduction_default,
                step=0.05,
            )

        return ui.p("", class_="text-muted")

    @reactive.effect
    @reactive.event(input.btn_create_scenario)
    def _create_scenario():
        """Create simulation scenario from model."""
        model = model_data.get()

        if model is None:
            notify_error("No Ecopath model available. Please balance a model first.")
            return

        # Require a balanced Rpath model for Ecosim
        if not _require_balanced_model_or_notify(model):
            return

        try:
            # Get groups and types safely
            from pypath_shiny.pages.utils import (
                is_balanced_model,
                is_rpath_params,
            )

            if is_balanced_model(model):
                # It's a balanced Rpath object
                groups = list(model.Group)
                types = list(model.type)
            elif is_rpath_params(model):
                # It's an RpathParams object
                groups = list(model.model["Group"])
                types = list(model.model["Type"])
            else:
                raise ValueError(
                    "Model object must be either Rpath or RpathParams type"
                )

            rule = _input_or_default("fast_group_rule", "ewe")
            source = _matching_ewe_source()
            if (
                source is not None
                and _input_or_default("scenario_source", "model") == "ewe"
            ):
                # The database's own scenario, as EwE6 runs it: its Ecopath
                # inputs, settings, forcing and length.
                new_scenario = build_ewe_scenario(
                    source,
                    _input_or_default("ewe_scenario_id", source["scenario"]),
                    _input_or_default("ewe_dataset_id", ""),
                    rule,
                )
                built_years = new_scenario.fishing.ForcedEffort.shape[0] // 12
            else:
                new_scenario = scenario_from_balanced_model(
                    model,
                    groups,
                    types,
                    years=range(1, input.sim_years() + 1),
                    vulnerability=input.vulnerability(),
                    fast_group_rule=rule,
                )
                built_years = input.sim_years()
            _notify_detritus_self_fate(model)

            # Apply fishing scenario
            _apply_fishing_scenario(new_scenario, input)

            # Apply advanced control adapters
            recruitment_summary = apply_recruitment_control(
                new_scenario,
                validate_recruitment_control(
                    enabled=input.enable_recruitment_control(),
                    group_name=input.recruitment_group(),
                    model_type=input.recruitment_model(),
                    alpha=read_optional_input(input, "recruitment_alpha", 1.2),
                    beta=read_optional_input(input, "recruitment_beta", 0.1),
                    rmax=read_optional_input(input, "recruitment_rmax", 1.5),
                    k=_input_or_default("recruitment_k", 1.0),
                    power=read_optional_input(input, "recruitment_power", 1.0),
                    environment_multiplier=input.recruitment_env_multiplier(),
                ),
            )
            metabolism_summary = apply_metabolism_control(
                new_scenario,
                validate_metabolism_control(
                    enabled=input.enable_metabolism_control(),
                    temperature_delta=input.metabolism_temperature_delta(),
                    q10=input.metabolism_q10(),
                    mass_exponent=input.metabolism_mass_exponent(),
                    reference_mass=input.metabolism_reference_mass(),
                    activity_budget=input.metabolism_activity_budget(),
                    apply_to_consumers=input.metabolism_apply_consumers(),
                    apply_to_producers=input.metabolism_apply_producers(),
                ),
            )
            fishing_mgmt_summary = apply_fishing_management_control(
                new_scenario,
                validate_fishing_management_control(
                    enabled=input.enable_fishing_management_control(),
                    strategy=input.fishing_management_strategy(),
                    sustainability_reduction=read_optional_input(
                        input,
                        "sustainability_reduction",
                        PARAM_RANGES.sustainability_reduction_default,
                    ),
                    sustainability_floor=read_optional_input(
                        input,
                        "sustainability_floor",
                        PARAM_RANGES.sustainability_floor_default,
                    ),
                    effort_budget=read_optional_input(
                        input,
                        "fishing_effort_budget",
                        PARAM_RANGES.fishing_effort_budget_default,
                    ),
                    bycatch_reduction=read_optional_input(
                        input,
                        "bycatch_reduction",
                        PARAM_RANGES.bycatch_reduction_default,
                    ),
                ),
            )
            controls_metadata = build_control_metadata(
                recruitment=recruitment_summary,
                metabolism=metabolism_summary,
                fishing_management=fishing_mgmt_summary,
            )
            scenario_controls_metadata.set(controls_metadata)
            setattr(new_scenario, "applied_controls_metadata", controls_metadata)

            # Apply autofix if enabled
            if input.enable_autofix():
                new_scenario, report = validate_and_fix_scenario(
                    new_scenario, model, auto_fix=True, verbose=False
                )

                # Show what was fixed
                if report["fixes"]:
                    fix_count = len(report["fixes"])
                    notify_info(
                        f"Applied {fix_count} stability fix{'es' if fix_count > 1 else ''}",
                        duration=5,
                    )

            scenario.set(new_scenario)
            if sim_scenario is not None:
                sim_scenario.set(new_scenario)
            # A new scenario invalidates results from the previous one
            sim_output.set(None)
            if sim_results is not None:
                sim_results.set(None)

            # Update group choices. Select biomass groups by type rather than
            # by position: the core supports non-canonical group ordering.
            if is_balanced_model(model) and hasattr(model, "type"):
                type_arr = np.asarray(model.type)
                group_names = [str(g) for g, t in zip(groups, type_arr) if t < 3]
            else:
                group_names = list(groups)
            ui.update_selectize(
                "plot_groups", choices=group_names, selected=group_names[:3]
            )
            ui.update_select("forcing_group", choices=group_names)
            ui.update_select("recruitment_group", choices=group_names)

            scenario_years.set(built_years)
            notify_success("Scenario created successfully!")

        except Exception as e:
            logger.exception("Scenario creation failed")
            notify_error(f"Error creating scenario: {str(e)}")

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

        metadata = scenario_controls_metadata.get() or {}
        enabled_controls = [
            name.replace("_", " ")
            for name, value in metadata.items()
            if isinstance(value, dict) and value.get("enabled")
        ]
        controls_note = (
            f" Controls: {', '.join(enabled_controls)}." if enabled_controls else ""
        )

        return ui.div(
            ui.tags.i(class_="bi bi-check-circle me-2"),
            f"Scenario ready: {scen.params.NUM_GROUPS} groups, "
            f"{scenario_years.get()} years.{controls_note}",
            class_="alert alert-success",
        )

    @output
    @render_widget
    def effort_preview_plot():
        """Interactive effort forcing preview, one line per fleet (Plotly)."""
        scen = scenario.get()
        if scen is None:
            return build_effort_figure(None, [])
        p = scen.params
        first_fleet = 1 + p.NUM_LIVING + p.NUM_DEAD  # spname[0] is "Outside"
        fleets = list(p.spname[first_fleet : first_fleet + p.NUM_GEARS])
        return build_effort_figure(scen.fishing.ForcedEffort, fleets)

    def _apply_biomass_forcing(scen):
        """Apply the Biomass Forcing card to the scenario's prey multiplier."""
        if apply_prey_forcing(scen, input.forcing_group(), input.forcing_multiplier()):
            notify_success(
                f"Prey availability for {input.forcing_group()} forced to "
                f"{input.forcing_multiplier():g}x",
                duration=4,
            )

    @reactive.effect
    @reactive.event(input.btn_run_sim)
    def _run_simulation():
        """Run the Ecosim simulation."""
        scen = scenario.get()

        if scen is None:
            notify_warning("Create a scenario first")
            return

        run_id = f"ecosim-run-{input.btn_run_sim()}"
        sim_task_state.set(new_running_task(run_id, "Preparing simulation"))

        def _update_latest(updater):
            sim_task_state.set(
                apply_if_latest(sim_task_state.get(), run_id=run_id, updater=updater)
            )

        try:
            _update_latest(
                lambda s: update_task_progress(
                    s,
                    message="Applying forcing and calibration settings",
                    progress=20.0,
                )
            )
            _apply_biomass_forcing(scen)

            # Check if diet rewiring is enabled
            diet_rewiring_enabled = input.enable_diet_rewiring()

            _update_latest(
                lambda s: update_task_progress(
                    s,
                    message="Integrating Ecosim equations",
                    progress=50.0,
                )
            )

            if diet_rewiring_enabled:
                notify_info(
                    "Running simulation with diet rewiring...",
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
                notify_info("Running simulation...", duration=2)

                # Run standard simulation
                output = rsim_run(scen, method=input.integration_method())

            _update_latest(
                lambda s: update_task_progress(
                    s,
                    message="Post-processing outputs",
                    progress=85.0,
                )
            )

            controls_metadata = scenario_controls_metadata.get() or {}
            if isinstance(output.params, dict):
                output.params["applied_controls"] = controls_metadata
            setattr(output, "applied_controls_metadata", controls_metadata)

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

                # Check if groups recovered, relative to their baseline biomass:
                # an absolute floor would exclude naturally small groups.
                base = getattr(scen.params, "B_BaseRef", None)

                def _recovered(i):
                    end = output.end_state.Biomass[i]
                    if base is not None and len(base) > i and base[i] > 0:
                        return end > THRESHOLDS.recovery_threshold * base[i]
                    return end > THRESHOLDS.crash_threshold

                recovered = [
                    name
                    for i, name in zip(output.crashed_groups, crashed_names)
                    if _recovered(i)
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

                if msg_type == "warning":
                    notify_warning(msg, duration=10)
                else:
                    notify_info(msg, duration=10)
            else:
                success_msg = "Simulation completed successfully!"
                if diet_rewiring_enabled:
                    success_msg += " Diet rewiring was applied."
                notify_success(success_msg)

            final_msg = (
                f"Simulation complete ({output.params['years']} years)."
                if output is not None
                else "Simulation complete."
            )
            _update_latest(lambda s: finish_task(s, success=True, message=final_msg))

        except Exception as e:
            logger.exception("Simulation failed")
            err_text = str(e)
            notify_error(f"Simulation error: {err_text}")
            _update_latest(
                lambda s, err=err_text: finish_task(
                    s,
                    success=False,
                    message="Simulation failed",
                    error=err,
                )
            )

    @reactive.effect
    @reactive.event(input.btn_cancel_sim)
    def _cancel_simulation():
        """Handle simulation cancel button."""
        task_state = sim_task_state.get()

        # Only allow cancellation if task is running
        if task_state.status is TaskStatus.RUNNING:
            # Mark as cancelled
            from pypath_shiny.services.tasks import finish_task

            sim_task_state.set(
                finish_task(
                    task_state,
                    success=False,
                    message="Cancelled by user",
                    error="Simulation was cancelled",
                )
            )
            notify_warning("Simulation cancelled by user")

    @output
    @render.ui
    def simulation_status():
        """Display simulation status."""
        output = sim_output.get()
        scen = scenario.get()
        task_state = sim_task_state.get()

        if output is None or task_state.status in {
            TaskStatus.RUNNING,
            TaskStatus.ERROR,
        }:
            alert_class, icon_class, msg = build_task_status_alert(task_state)
            return ui.div(ui.tags.i(class_=icon_class), msg, class_=alert_class)

        if output.crash_year > 0:
            # Get crashed group names
            crashed_names = (
                [scen.params.spname[i] for i in output.crashed_groups]
                if scen is not None
                else []
            )

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
        task_state = sim_task_state.get()

        if task_state.status is TaskStatus.RUNNING:
            return ui.div(
                ui.tags.h5("Simulation Progress"),
                ui.div(
                    ui.div(
                        f"{task_state.progress:.0f}%",
                        class_="progress-bar",
                        role="progressbar",
                        style=f"width: {task_state.progress:.0f}%",
                    ),
                    class_="progress",
                ),
                ui.p(task_state.message, class_="text-muted mt-2"),
                ui.input_action_button(
                    "btn_cancel_sim",
                    ui.span(
                        ui.tags.i(class_="bi bi-stop-circle me-2"),
                        "Cancel Simulation",
                    ),
                    class_="btn btn-warning w-100 mt-3",
                )
                if task_state.can_cancel
                else None,
            )

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
    @render_widget
    def biomass_timeseries():
        """Interactive biomass time series (Plotly)."""
        output = sim_output.get()
        scen = scenario.get()
        if output is None or scen is None:
            return build_biomass_figure(
                None,
                [],
                [],
                title="Biomass Trajectories",
                empty_message="Run simulation to see results",
            )
        return build_biomass_figure(
            output,
            list(scen.params.spname[1:]),  # skip "Outside"
            list(input.plot_groups() or []),
            relative=bool(input.relative_biomass()),
            title="Biomass Trajectories",
            empty_message="Select groups to plot",
        )

    @output
    @render_widget
    def catch_timeseries():
        """Interactive total annual catch (Plotly)."""
        output = sim_output.get()
        if scenario.get() is None:
            output = None
        return build_catch_figure(output, title="Total Annual Catch")

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

    def _start_end_living_biomass():
        """(group names, initial, final) for living groups, or None."""
        output = sim_output.get()
        scen = scenario.get()
        if output is None or scen is None:
            return None
        n_living = scen.params.NUM_LIVING
        return (
            list(scen.params.spname[1 : n_living + 1]),
            output.out_Biomass[0, 1 : n_living + 1],
            output.out_Biomass[-1, 1 : n_living + 1],
        )

    @output
    @render_widget
    def final_biomass_plot():
        """Interactive initial vs final biomass (Plotly)."""
        data = _start_end_living_biomass()
        if data is None:
            return build_initial_final_figure([], [], [])
        return build_initial_final_figure(*data)

    @output
    @render_widget
    def biomass_change_plot():
        """Interactive percent biomass change (Plotly)."""
        data = _start_end_living_biomass()
        if data is None:
            return build_biomass_change_figure([], [], [])
        return build_biomass_change_figure(*data)
