"""Ecosim simulation page module."""

import asyncio
import copy
import itertools
import logging
import time
from collections.abc import Mapping
from typing import NamedTuple

import numpy as np
import pandas as pd

# pypath imports (path setup handled by app/__init__.py)
from pypath.core.autofix import validate_and_fix_scenario
from pypath.core.ecosim import RsimScenario, rsim_scenario
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
from pypath.core.ecosim_continuous import QS_RATE_DEFAULT

from pypath_shiny.config import PARAM_RANGES, THRESHOLDS, UI
from pypath_shiny.pages.ewe_scenario import (
    build_ewe_scenario,
    ewe_source_matches,
    scenario_source_choices,
)
from pypath_shiny.pages.utils import is_balanced_model, is_rpath_params
from pypath_shiny.services.ecosim_controls import (
    apply_fishing_management_control,
    apply_metabolism_control,
    apply_recruitment_control,
    build_control_metadata,
    validate_fishing_management_control,
    validate_metabolism_control,
    validate_recruitment_control,
)
from pypath_shiny.services.ecosim_solver import (
    CONTINUOUS_JS,
    INTERPOLATION_CHOICES,
    MAX_CONCURRENT_RUNS,
    METHOD_CHOICES,
    PRESET_CHOICES,
    RUN_EXECUTOR,
    RunOutcome,
    RunSlot,
    SolverRequest,
    build_solver_request,
    continuous_failure_hint,
    is_continuous,
    result_is_current,
    run_ecosim,
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
from pypath_shiny.services.prey_switching import (
    BALANCED_SOURCE,
    DISPLAY_DECIMALS,
    SWITCH_POWER_MAX,
    SwitchingStore,
    above_range,
    apply_at_create,
    consumers_from_model,
    consumers_from_params,
    effective_values,
    fold_edits,
    on_model_change,
    pending,
    snapshot_source,
    source_key,
    summary,
    summary_line,
)
from pypath_shiny.services.solver_diagnostics import (
    flagged_months as _flagged_months,
)
from pypath_shiny.services.solver_diagnostics import (
    solver_badges as _solver_badges,
)
from pypath_shiny.services.solver_diagnostics import (
    solver_month_table as _month_table,
)
from pypath_shiny.services.solver_diagnostics import (
    solver_summary as _solver_summary,
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
def _group_names(scen) -> list[str] | None:
    """Group names of a scenario (index 0 is Outside), or None."""
    spname = getattr(getattr(scen, "params", None), "spname", None)
    return None if spname is None else [str(n) for n in spname]


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


def _solver_request_or_notify(
    method,
    *,
    diet_rewiring,
    preset,
    log_states,
    forcing_interpolation,
    quasi_steady,
    qs_rate,
) -> SolverRequest | None:
    """Build the solver request, or notify and return None on a bad combination."""
    try:
        request = build_solver_request(
            method,
            preset=preset,
            log_states=log_states,
            forcing_interpolation=forcing_interpolation,
            quasi_steady=quasi_steady,
            qs_rate=qs_rate,
            diet_rewiring=diet_rewiring,
        )
    except ValueError as e:
        notify_warning(str(e), duration=8)
        return None
    for note in request.notes:
        notify_info(note, duration=8)
    return request


MODEL_CHANGED_NOTE = "Model changed: create the scenario again before running."


def _model_change_notice(*cleared) -> str | None:
    """The note for a model load, or None when it cleared nothing.

    ``cleared`` are the scenario and results the load dropped; a first model
    load, with none of them, says nothing.
    """
    return MODEL_CHANGED_NOTE if any(v is not None for v in cleared) else None


# Short method names for messages: LSODA, BDF, DOP853, Radau, Runge-Kutta 4...
_METHOD_NAMES = {
    method: label.split(" (")[0]
    for group in METHOD_CHOICES.values()
    for method, label in group.items()
}


def _success_message(method: str) -> str:
    """The notification for a run that finished without a crash."""
    if is_continuous(method):
        name = _METHOD_NAMES.get(method, method)
        return f"Simulation completed with {name} (continuous, not EwE parity)."
    return "Simulation completed successfully!"


def _failure_message(error: str, method: str, preset: str) -> str:
    """The notification for a failed run, with advice only where it can help."""
    hint = continuous_failure_hint(method, preset, error)
    return f"Simulation error: {error}" + (f" {hint}" if hint else "")


def _flagged_months_view(flagged):
    """The flagged-months table, scrolling inside a box of capped height.

    A long run can flag hundreds of months; the box keeps the panel short,
    and every month stays in the table.
    """
    if flagged.empty:
        return ui.p("No month flagged.", class_="text-muted mb-2")
    return ui.div(
        ui.h6("Flagged months"),
        ui.div(
            ui.HTML(
                flagged.to_html(
                    index=False,
                    classes="table table-sm",
                    border=0,
                    float_format=lambda x: f"{x:.3g}",
                )
            ),
            style="max-height: 320px; overflow-y: auto; overflow-x: auto;",
        ),
        class_="mb-2",
    )


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
        # No percentage at 0: the Ecosim bar is indeterminate.
        pct = f" ({task_state.progress:.0f}%)" if task_state.progress > 0 else ""
        return (
            "alert alert-info",
            "bi bi-arrow-repeat me-2",
            f"{task_state.message}{pct}",
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
                            title="The EwE parity methods reproduce EwE (RK4) and Rpath (AB). The continuous methods solve the equations to a tolerance and differ from EwE by design. BDF suits stiff models.",
                            style="cursor: help;",
                        ),
                    ),
                    choices=METHOD_CHOICES,
                    selected="RK4",
                ),
                ui.panel_conditional(
                    CONTINUOUS_JS,
                    ui.h6("Numerics (same model)"),
                    ui.input_radio_buttons(
                        "solver_preset",
                        "Tolerance",
                        PRESET_CHOICES,
                        selected="default",
                    ),
                    ui.input_checkbox(
                        "solver_log_states",
                        ui.span(
                            "Log-transformed biomass ",
                            ui.tags.i(
                                class_="bi bi-info-circle",
                                title="Solves the same equations on the log of biomass. Use it for runs that drive a group towards zero. It costs about 40 % more where groups sweep orders of magnitude.",
                                style="cursor: help;",
                            ),
                        ),
                        value=False,
                    ),
                    ui.p(
                        "Fast saves time only where the solver takes many steps a "
                        "month (Black Sea 37 %, REco 8 %); Reference costs 2-3.5 "
                        "\u00d7 Default; continuous runs cost about 5-25 \u00d7 RK4.",
                        class_="text-muted small",
                    ),
                    ui.accordion(
                        ui.accordion_panel(
                            "Advanced: changes the model",
                            ui.input_select(
                                "solver_forcing_interpolation",
                                "Forcing within the month",
                                INTERPOLATION_CHOICES,
                                selected="constant",
                            ),
                            ui.p(
                                "Interpolates forcing that varies month to month "
                                "(an imported EwE scenario's forcing functions); "
                                "effort stays monthly; the page's own forcing card "
                                "is constant in time",
                                class_="text-muted small",
                            ),
                            ui.input_checkbox(
                                "solver_quasi_steady",
                                "Quasi-steady fast groups",
                                value=False,
                            ),
                            ui.panel_conditional(
                                "input.solver_quasi_steady",
                                ui.input_numeric(
                                    "solver_qs_rate",
                                    "Quasi-steady above (per year)",
                                    value=QS_RATE_DEFAULT,
                                    min=1,
                                    step=5,
                                ),
                            ),
                        ),
                        id="solver_advanced",
                        open=False,
                    ),
                ),
                ui.input_select(
                    "fast_group_rule",
                    ui.span(
                        "Equation rule (EwE6 / Rpath) ",
                        ui.tags.i(
                            class_="bi bi-info-circle",
                            title="Groups that turn over much faster than the time step (plankton, detritus). EwE6 solves them at equilibrium in every step and takes one RK4 step per month; Rpath integrates them like every other group with four RK4 steps per month. Under a continuous method this picks the equations only (handling time, detritus fate): every group is integrated, or solved quasi-steady if you ask for it below.",
                            style="cursor: help;",
                        ),
                    ),
                    choices={
                        "ewe": "EwE6 (fast groups at equilibrium under RK4/AB)",
                        "rpath": "Rpath (fast groups integrated)",
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
                ui.h5("Prey Switching"),
                ui.p(
                    "Predators turn towards prey that become more abundant. "
                    "0 = off; higher values switch more sharply (EwE allows 0 "
                    "to 2). Values apply when you create the scenario.",
                    class_="text-muted small",
                ),
                ui.output_ui("switching_control"),
                ui.output_ui("switching_pending_note"),
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
                    ui.output_ui("switching_line"),
                    ui.output_ui("progress_display"),
                    ui.output_ui("solver_diagnostics"),
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


def read_optional_input(input, name: str, default, *, keep_none: bool = False):
    """Read an input that may not exist in the browser.

    Inputs created by an ``output_ui`` inside a hidden conditional panel are
    never rendered, and reading one raises ``SilentException``.
    ``hasattr(input, name)`` is always True for Shiny's Inputs, so it is no
    guard. Returns ``default`` when the input is missing or None; with
    ``keep_none``, a None value (an empty numeric field) is returned as None.
    """
    try:
        value = getattr(input, name)()
    except (SilentException, AttributeError, KeyError, TypeError):
        return default
    if value is None and not keep_none:
        return default
    return value


def _solver_inputs(input, method: str) -> dict:
    """The solver keyword arguments the page's inputs give for ``method``.

    A parity method reads none of the continuous inputs, so a stale one can
    never block its run. With quasi-steady ticked, an empty rate field reaches
    the request builder as None and is refused there, not run at the default.
    """
    if not is_continuous(method):
        return dict(
            preset="default",
            log_states=False,
            forcing_interpolation="constant",
            quasi_steady=False,
            qs_rate=QS_RATE_DEFAULT,
        )
    quasi_steady = bool(read_optional_input(input, "solver_quasi_steady", False))
    return dict(
        preset=read_optional_input(input, "solver_preset", "default"),
        log_states=bool(read_optional_input(input, "solver_log_states", False)),
        forcing_interpolation=read_optional_input(
            input, "solver_forcing_interpolation", "constant"
        ),
        quasi_steady=quasi_steady,
        qs_rate=read_optional_input(
            input, "solver_qs_rate", QS_RATE_DEFAULT, keep_none=True
        )
        if quasi_steady
        else QS_RATE_DEFAULT,
    )


NO_CONSUMERS_TEXT = "This model has no consumers to switch."
NO_MODEL_SWITCHING_TEXT = "Balance a model to set prey switching."


def balanced_switching_groups(model) -> tuple[list[str], str | None]:
    """The balanced route's consumers, or the message shown instead of boxes."""
    if not (is_balanced_model(model) or is_rpath_params(model)):
        return [], NO_MODEL_SWITCHING_TEXT
    groups = consumers_from_model(model)
    return groups, None if groups else NO_CONSUMERS_TEXT


def switching_pending_text(
    built_from_current_key: bool, is_pending: bool, has_overrides: bool
) -> str | None:
    """The note under the switching control, or None for no note.

    Once a scenario is built from the current source, the note shows only
    while the settings differ from it; before that, only when there are edits.
    """
    if built_from_current_key:
        if is_pending:
            return "Changed values apply when you create the scenario again."
        return None
    if has_overrides:
        return "Values apply when you create the scenario."
    return None


def switching_inputs(
    prefix: str,
    groups: list[str],
    values: Mapping[str, float],
    extra_readonly: Mapping[str, float],
) -> ui.Tag:
    """The prey-switching control's body: one box per consumer.

    ``values`` are the values in force (the source's, with the edits applied).
    ``extra_readonly`` holds imported nonzero values of groups that cannot
    switch: they are shown and kept, not offered.
    """
    if groups:
        body = ui.div(
            *[
                ui.input_numeric(
                    f"{prefix}{i}",
                    name,
                    value=round(values[name], DISPLAY_DECIMALS),
                    min=0,
                    max=SWITCH_POWER_MAX,
                    step=0.1,
                )
                for i, name in enumerate(groups)
            ],
            style="max-height: 320px; overflow-y: auto;",
        )
    else:
        body = ui.p(NO_CONSUMERS_TEXT, class_="text-muted small")
    readonly = [
        ui.p(
            f"{name}: {round(value, DISPLAY_DECIMALS):g} (imported; this group "
            "cannot switch, kept as imported)",
            class_="text-muted small",
        )
        for name, value in extra_readonly.items()
    ]
    return ui.div(body, *readonly)


class _SwitchingSpec(NamedTuple):
    """What the prey-switching control shows, and under which input ids."""

    prefix: str
    key: str
    groups: list[str]
    base: dict[str, float]
    values: dict[str, float]
    readonly: dict[str, float]
    message: str | None


def ecosim_server(
    input: Inputs,
    output: Outputs,
    session: Session,
    model_data: reactive.Value,
    sim_results: reactive.Value,
    sim_scenario: reactive.Value = None,
    ewe_source: reactive.Value = None,
    rebalance_mark: reactive.Value | None = None,
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
    rebalance_mark : reactive.Value, optional
        Set by the Ecopath page's re-balance just before it publishes its
        model; a model change that finds it set keeps the prey-switching
        overrides, any other clears them.

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
    # Prey switching: the edits by source key; a counter that the model-change
    # effect and Create Scenario bump once they have updated the store, so
    # the control re-renders after them; and the source key that built the
    # current scenario.
    switching_store = reactive.Value(SwitchingStore())
    switching_generation = reactive.Value(0)
    scenario_switching_key = reactive.Value(None)
    # A fresh input-id prefix on every render of the control, so that a stale
    # browser value can never land on a new list of groups.
    switching_renders = itertools.count()

    # The run last launched: its method and preset advise on a failure that
    # has no outcome to read; its label and its start and integration-start times feed the
    # progress texts. Plain state, written on the loop only; the worker never
    # reads it.
    launched = {
        "run_id": None,
        "method": None,
        "preset": None,
        "label": None,
        "started": None,
        "integrating_since": None,
    }
    retired_runs = itertools.count(1)
    # This session's run in the process-wide executor. It stays taken until
    # the worker thread finishes, through a cancel (the thread cannot be
    # stopped), so Run-then-Cancel cannot pile up threads.
    run_slot = RunSlot()

    @reactive.extended_task
    async def _ecosim_task(run_id, source, scen, request, diet, slot):
        """Integrate on a worker thread, off the event loop.

        ``scen`` is a deep copy of ``source``, the page's scenario object. The
        run goes to the bounded process-wide executor and ``slot`` holds its
        future until the thread finishes. The worker reads no input and sends
        no notification: the result effect does both, on the loop.
        """
        future = RUN_EXECUTOR.submit(run_ecosim, scen, request, diet_rewiring=diet)
        slot.acquire(future)
        out = await asyncio.wrap_future(future)
        return RunOutcome(run_id, source, out, request, diet is not None)

    # A closed or reloaded tab must not leave its run queued for a worker with
    # no owner; a run already computing finishes and its result is discarded.
    session.on_ended(_ecosim_task.cancel)

    def _finish_run(
        run_id: str, *, success: bool, message: str, error: str | None = None
    ) -> None:
        """Finish the running task, if it is still ``run_id``."""
        state = sim_task_state.get()
        if state.status is not TaskStatus.RUNNING:
            return
        sim_task_state.set(
            apply_if_latest(
                state,
                run_id=run_id,
                updater=lambda s: finish_task(
                    s, success=success, message=message, error=error
                ),
            )
        )

    @reactive.effect
    @reactive.event(model_data, ignore_init=True)
    def _retire_run_on_model_change():
        """A new model retires the run and drops the old model's scenario and
        results, so neither the Results pages nor a Run use them."""
        note = _model_change_notice(
            scenario.get(),
            sim_output.get(),
            sim_results.get() if sim_results is not None else None,
        )
        _ecosim_task.cancel()
        sim_task_state.set(TaskState(run_id=f"ecosim-retired-{next(retired_runs)}"))
        sim_output.set(None)
        if sim_results is not None:
            sim_results.set(None)
        scenario.set(None)
        if sim_scenario is not None:
            sim_scenario.set(None)
        if note:
            notify_info(note, duration=6)
        scenario_switching_key.set(None)
        with reactive.isolate():
            # One-shot: only the re-balance that published this model keeps the
            # overrides; the mark is consumed either way.
            rebalance = (
                bool(rebalance_mark.get()) if rebalance_mark is not None else False
            )
            if rebalance_mark is not None:
                rebalance_mark.set(False)
            new_store, switching_note = on_model_change(
                switching_store.get(),
                rebalance=rebalance,
                consumers=consumers_from_model(model_data.get()),
                keep_ewe=_matching_ewe_source() is not None,
            )
            switching_store.set(new_store)
        if switching_note is not None:
            notify_info(switching_note, duration=6)
        # Last: the control re-renders once the switching state is updated.
        switching_generation.set(switching_generation.get() + 1)

    def _input_or_default(name: str, default):
        """Read a UI input value with fallback for conditional controls."""
        return read_optional_input(input, name, default)

    def _matching_ewe_source():
        """The imported EwE database, when the current model has its groups."""
        if ewe_source is None:
            return None
        source = ewe_source.get()
        return source if ewe_source_matches(source, model_data.get()) else None

    def _current_switching_key() -> str:
        """The source key Create Scenario would build from now.

        The same route test as Create Scenario's: the "ewe" source selected
        and matching the model, else the balanced model.
        """
        source = _matching_ewe_source()
        if source is None or _input_or_default("scenario_source", "model") != "ewe":
            return BALANCED_SOURCE
        return source_key(
            source, _input_or_default("ewe_scenario_id", source["scenario"])
        )

    @reactive.calc
    def _switching_spec() -> _SwitchingSpec:
        """What the prey-switching control shows.

        Recomputes on a generation bump (after a model change or Create
        Scenario has updated the store), a new scenario key, a new imported
        source, or a change of scenario source. The model, the scenario and
        the store are read isolated, so an edit never re-renders the control.
        """
        switching_generation.get()
        built_key = scenario_switching_key.get()
        if ewe_source is not None:
            ewe_source.get()
        _input_or_default("scenario_source", "model")
        _input_or_default("ewe_scenario_id", None)
        with reactive.isolate():
            key = _current_switching_key()
            model = model_data.get()
            scen = scenario.get()
            store = switching_store.get()
        prefix = f"switch_power_{next(switching_renders)}_"
        groups: list[str] = []
        base: dict[str, float] = {}
        readonly: dict[str, float] = {}
        message = None
        if key == BALANCED_SOURCE:
            groups, message = balanced_switching_groups(model)
            base = dict.fromkeys(groups, 0.0)
        elif built_key != key or scen is None:
            message = "Create the scenario to see and change its switching powers."
        else:
            groups = consumers_from_params(scen.params)
            base = dict(store.source_values.get(key, {}))
            consumers = set(groups)
            readonly = {
                name: value
                for name, value in base.items()
                if name not in consumers and value != 0
            }
        values = effective_values(store, key, base)
        return _SwitchingSpec(prefix, key, groups, base, values, readonly, message)

    @output
    @render.ui
    def switching_control():
        spec = _switching_spec()
        if spec.message:
            return ui.p(spec.message, class_="text-muted small")
        return switching_inputs(spec.prefix, spec.groups, spec.values, spec.readonly)

    @reactive.effect(priority=1)
    def _collect_switching_edits():
        """Fold the boxes into the store; refuse and reset values out of range.

        The priority runs this before Create Scenario when a box's new value
        and the click arrive together.
        """
        spec = _switching_spec()
        if not spec.groups:
            return
        boxes = {
            name: input[f"{spec.prefix}{i}"]() for i, name in enumerate(spec.groups)
        }
        with reactive.isolate():
            old = switching_store.get()
            new, errors = fold_edits(old, spec.key, boxes, spec.base)
            if new != old:
                switching_store.set(new)
        if errors:
            in_force = effective_values(new, spec.key, spec.base)
            for name, message in errors.items():
                notify_warning(message)
                ui.update_numeric(
                    f"{spec.prefix}{spec.groups.index(name)}",
                    value=round(in_force[name], DISPLAY_DECIMALS),
                )

    @output
    @render.ui
    def switching_pending_note():
        spec = _switching_spec()
        store = switching_store.get()
        scen = scenario.get()
        built = scen is not None and scenario_switching_key.get() == spec.key
        text = switching_pending_text(
            built,
            built and pending(store, spec.key, spec.base, scen.params),
            bool(store.overrides.get(spec.key)),
        )
        if text is None:
            return ui.div()
        return ui.p(text, class_="text-muted small fst-italic")

    @output
    @render.ui
    def ewe_scenario_source_ui():
        """Offer the Ecosim scenarios of the imported EwE database, or the
        Ecosim settings of the example model opened from the Home page."""
        source = _matching_ewe_source()
        if source is None:
            return ui.div()
        choices, selected, offers_datasets = scenario_source_choices(source)
        if offers_datasets:
            scenarios = {str(k): v for k, v in source["scenarios"].items()}
            default = str(source.get("scenario"))
            if default not in scenarios:
                default = next(iter(scenarios))
            datasets = {"": "None"}
            datasets.update({str(k): v for k, v in source.get("datasets", {}).items()})
            details = [
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
                    "vulnerability slider are not used; prey switching set in "
                    "the sidebar is.",
                    class_="text-muted small",
                ),
            ]
        else:
            details = [
                ui.p(
                    "Runs the example's Ecopath inputs with its own EwE-F "
                    "vulnerabilities, group settings, fishing and run length. "
                    "Edits made in this app and the vulnerability slider are "
                    "not used; prey switching set in the sidebar is.",
                    class_="text-muted small",
                )
            ]
        return ui.div(
            ui.h5("Scenario Source"),
            ui.input_radio_buttons(
                "scenario_source",
                "Build the scenario from",
                choices,
                selected=selected,
            ),
            ui.panel_conditional("input.scenario_source === 'ewe'", *details),
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
                            ui.tags.strong("DD (handling time):"),
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
                        ui.tags.li(
                            "Integration method (RK4 for EwE results; a continuous "
                            "method to solve the equations to a tolerance)"
                        ),
                        ui.tags.li("Vulnerability (functional response type)"),
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
                ui.h6("Integration Methods"),
                ui.tags.ul(
                    ui.tags.li(
                        ui.tags.strong("RK4 (default) and AB:"),
                        " the EwE parity methods. Use RK4 for results that "
                        "should match EwE; AB reproduces Rpath's Adams-Bashforth.",
                    ),
                    ui.tags.li(
                        ui.tags.strong("LSODA, BDF, DOP853, Radau:"),
                        " continuous methods (not EwE parity). They solve the "
                        "Ecosim equations to a tolerance, so their numbers "
                        "differ from EwE's. BDF suits stiff models; LSODA picks "
                        "stiff or non-stiff stepping itself.",
                    ),
                    ui.tags.li(
                        ui.tags.strong("Tolerance (Fast / Default / Reference):"),
                        " the same model solved to a looser or tighter "
                        "tolerance. Fast saves time only where the solver takes "
                        "many steps a month (measured 2026-11-15, ten years: "
                        "37 % on the Black Sea, 8 % on REco, 1-4 % on the three "
                        "shipped examples); Reference costs 2-3.5 \u00d7 Default.",
                    ),
                    ui.tags.li(
                        ui.tags.strong("Log-transformed biomass:"),
                        " the same equations on the log of biomass; for runs "
                        "that drive a group towards zero.",
                    ),
                    ui.tags.li(
                        ui.tags.strong("Advanced: changes the model:"),
                        " forcing interpolation and quasi-steady fast groups "
                        "solve a different model from EwE's.",
                    ),
                    ui.tags.li(
                        ui.tags.strong("Equation rule (EwE6 / Rpath):"),
                        " under a continuous method this picks only the "
                        "equations; every group is integrated.",
                    ),
                ),
                ui.h6("Running a Simulation"),
                ui.p(
                    "A continuous run costs about 5-25 \u00d7 RK4. At the "
                    "default 50 years one takes about a minute (Lithuanian "
                    "Coast, LSODA: 58 s against 15 s for RK4, measured "
                    "2026-11-15). Runs go in the background, up to 2 at a "
                    "time on the server, so a run may wait in a queue. Cancel "
                    "discards the result, but a cancelled run still finishes "
                    "in the background, so a new run starts only when it has "
                    "finished. Closing the tab cancels its run. Loading or "
                    "re-balancing a model clears the results and the "
                    "scenario: create the scenario again."
                ),
                ui.h6("Prey Switching"),
                ui.p(
                    "Set a predator's switching power in the sidebar to let it "
                    "turn towards prey that become more abundant (EwE's "
                    "SwitchPower, 0 to 2; 0 = off). Values apply when the "
                    "scenario is created; an imported EwE scenario starts from "
                    "its own values. Every run shows the switching it used. "
                    "Diet rewiring is deprecated: it never integrated."
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
                ui.h6("Solver diagnostics"),
                ui.p(
                    "Continuous methods (LSODA, BDF, DOP853, Radau) add a "
                    "Solver diagnostics card below. Each badge reads one "
                    "number the solver recorded:"
                ),
                ui.tags.ul(
                    ui.tags.li(
                        ui.tags.strong("Continuous path (not EwE parity):"),
                        " the run does not reproduce EwE's RK4 numbers.",
                    ),
                    ui.tags.li(
                        ui.tags.strong("Solver did not finish (red):"),
                        " the solver stopped before the month ended; "
                        "the trajectory after it is unreliable.",
                    ),
                    ui.tags.li(
                        ui.tags.strong("Degraded month:"),
                        " the solver fell back to a cruder step in that month.",
                    ),
                    ui.tags.li(
                        ui.tags.strong("Kernel check:"),
                        " the right-hand side disagreed with the reference "
                        "kernel by more than 1e-9.",
                    ),
                    ui.tags.li(
                        ui.tags.strong("qs_separation:"),
                        " below 10, the quasi-steady approximation is poor; "
                        "raise the rate or integrate the fast groups.",
                    ),
                    ui.tags.li(
                        ui.tags.strong("No group qualified:"),
                        " quasi-steady was chosen but no group is fast "
                        "enough, so every group was integrated.",
                    ),
                    ui.tags.li(
                        ui.tags.strong("Changes the model:"),
                        " linear or smooth forcing, or quasi-steady groups, "
                        "solve a different model from EwE's.",
                    ),
                    ui.tags.li(
                        ui.tags.strong("Clipped evaluations:"),
                        " a biomass went negative inside a step and was "
                        "clipped; log states avoid this.",
                    ),
                ),
                ui.p(
                    "Below the badges: a run summary, the flagged months, and "
                    "(collapsed) a table of every month."
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
            # The source key, computed once, also picks the route: an EwE key
            # only when the "ewe" source is selected and matches the model.
            switching_key = _current_switching_key()
            if switching_key != BALANCED_SOURCE:
                source = _matching_ewe_source()
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

            # Prey switching: on the EwE route, record the source's own values
            # before any edit is applied; then write the edits. Untouched, the
            # scenario keeps the very SwitchPower array its source built. The
            # store is committed after scenario.set, below.
            new_store = apply_at_create(
                new_scenario.params, switching_store.get(), switching_key
            )
            for name, value in above_range(
                snapshot_source(new_scenario.params)
            ).items():
                notify_warning(
                    f"{name} has an imported switching power of {value:g}, "
                    "above EwE's range of 0 to 2; it is kept."
                )
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
            switching_store.set(new_store)
            scenario_switching_key.set(switching_key)
            switching_generation.set(switching_generation.get() + 1)
            # A new scenario invalidates results from the previous one, and
            # retires a run still integrating it: a late outcome then fails
            # result_is_current on both its run_id and its scenario.
            _ecosim_task.cancel()
            sim_task_state.set(TaskState(run_id=f"ecosim-retired-{next(retired_runs)}"))
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
        """Start the Ecosim simulation on a worker thread (``_ecosim_task``)."""
        scen = scenario.get()

        if scen is None:
            notify_warning("Create a scenario first")
            return

        # invoke() would queue behind a running task. Refuse instead, before
        # any state change, so the live run keeps its run_id.
        if (
            sim_task_state.get().status is TaskStatus.RUNNING
            or _ecosim_task.status() == "running"
        ):
            notify_warning("A simulation is already running. Cancel it first.")
            return
        # A cancelled run's thread computes on until it ends, holding the slot.
        if run_slot.busy:
            notify_warning(
                "The previous simulation is still finishing in the background (a "
                "cancelled run keeps computing until it would have ended). Try "
                "again when it finishes."
            )
            return

        run_id = f"ecosim-run-{input.btn_run_sim()}"
        method = _input_or_default("integration_method", "RK4")
        preset = "default"
        try:
            solver_inputs = _solver_inputs(input, method)
            preset = solver_inputs["preset"]
            # Diet rewiring is not implemented, so it is never requested.
            request = _solver_request_or_notify(
                method, diet_rewiring=False, **solver_inputs
            )
            if request is None:
                return

            _apply_biomass_forcing(scen)
            # The worker integrates its own copy; ``scen`` stays the page's
            # scenario object, which the result is checked against.
            run_scen = copy.deepcopy(scen)
            years = run_scen.fishing.ForcedEffort.shape[0] // 12
            name = _METHOD_NAMES.get(request.method, request.method)
            label = f"{name}, {years} years"
            launched.update(
                run_id=run_id,
                method=request.method,
                preset=preset,
                label=label,
                started=time.monotonic(),
                integrating_since=None,
            )
            # "Starting" until a worker picks the run up (_note_worker_start):
            # with the process-wide executor full, the run queues first.
            sim_task_state.set(new_running_task(run_id, f"Starting {label}"))
            notify_info("Running simulation...", duration=2)
            _ecosim_task.invoke(
                run_id, scen, run_scen, request, diet=None, slot=run_slot
            )

        except Exception as e:
            logger.exception("Simulation could not start")
            err_text = str(e)
            notify_error(_failure_message(err_text, method, preset))
            _finish_run(
                run_id, success=False, message="Simulation failed", error=err_text
            )

    def _land_result(outcome: RunOutcome) -> None:
        """Show a current run's output and its messages (on the loop)."""
        output = outcome.output
        scen = outcome.source
        method = outcome.request.method
        try:
            output.run_request = outcome.request

            controls_metadata = scenario_controls_metadata.get() or {}
            # What this run used, read from the scenario that ran
            output.switching_used = summary(outcome.source.params)
            if isinstance(output.params, dict):
                output.params["applied_controls"] = {
                    **controls_metadata,
                    "prey_switching": output.switching_used,
                }
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
                notify_success(_success_message(method))

            final_msg = (
                f"Simulation complete ({output.params['years']} years)."
                if output is not None
                else "Simulation complete."
            )
            _finish_run(outcome.run_id, success=True, message=final_msg)

        except Exception as e:
            logger.exception("Showing the simulation result failed")
            err_text = str(e)
            notify_error(f"Simulation error: {err_text}")
            _finish_run(
                outcome.run_id,
                success=False,
                message="Simulation failed",
                error=err_text,
            )

    @reactive.effect
    def _apply_ecosim_result():
        """Land the task's outcome. Depends on the task's status only."""
        status = _ecosim_task.status()
        # Everything else is read isolated, so this effect's own state
        # changes (finish_task) do not re-trigger it.
        with reactive.isolate():
            if status == "success":
                outcome = _ecosim_task.result()
                if result_is_current(sim_task_state.get(), outcome, scenario.get()):
                    _land_result(outcome)
                else:
                    logger.debug(
                        "Dropped a stale Ecosim result (run %s)", outcome.run_id
                    )
            elif status == "error":
                state = sim_task_state.get()
                if (
                    state.status is not TaskStatus.RUNNING
                    or state.run_id != launched["run_id"]
                ):
                    logger.debug("Dropped a stale Ecosim error (state %s)", state)
                    return
                try:
                    _ecosim_task.result()
                except Exception as e:
                    logger.error("Simulation failed", exc_info=e)
                    err_text = str(e)
                    notify_error(
                        _failure_message(
                            err_text, launched["method"], launched["preset"]
                        )
                    )
                    _finish_run(
                        state.run_id,
                        success=False,
                        message="Simulation failed",
                        error=err_text,
                    )
            # "cancelled": whoever cancelled has already set the state.

    @reactive.effect
    @reactive.event(input.btn_cancel_sim)
    def _cancel_simulation():
        """Cancel the running simulation; its result is discarded."""
        task_state = sim_task_state.get()

        # Only allow cancellation if task is running
        if task_state.status is TaskStatus.RUNNING:
            # A worker thread cannot be stopped: a run already computing goes
            # on to its end in the background (holding the session's slot),
            # and the run_id guard (result_is_current) drops its result. A
            # run still queued for a worker is cancelled outright.
            _ecosim_task.cancel()
            sim_task_state.set(
                finish_task(
                    task_state,
                    success=False,
                    message="Cancelled by user",
                    error="Simulation was cancelled",
                )
            )
            notify_warning(
                "Cancelled: the result will be discarded, but the computation "
                "continues in the background until it would have finished; a "
                "new run can start after that."
            )

    @output
    @render.ui
    def switching_line():
        """The prey switching the shown run used."""
        out = sim_output.get()
        if out is None:
            return None
        return ui.p(
            summary_line(getattr(out, "switching_used", None)),
            class_="text-muted small",
        )

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
    @render.text
    def sim_elapsed():
        """Queue wait, then integration time, ticking every second."""
        state = sim_task_state.get()
        if state.status is not TaskStatus.RUNNING or state.run_id != launched["run_id"]:
            return ""
        reactive.invalidate_later(1)
        now = time.monotonic()
        if run_slot.waiting:
            return (
                f"Queued: the server runs at most {MAX_CONCURRENT_RUNS} "
                "simulations at once, and yours starts when one finishes "
                f"(waiting {now - launched['started']:.0f} s)."
            )
        since = launched["integrating_since"]
        if since is None:
            return "Starting..."
        # From the worker's start: queue time is not counted.
        return f"Elapsed: {now - since:.0f} s"

    @reactive.effect
    def _note_worker_start():
        """Switch "Starting" to "Integrating" once a worker picks the run up.

        Polls only until then; the progress block (and its Cancel button) is
        rebuilt once, at the switch.
        """
        state = sim_task_state.get()
        if (
            state.status is not TaskStatus.RUNNING
            or state.run_id != launched["run_id"]
            or launched["integrating_since"] is not None
        ):
            return
        if not run_slot.running:
            reactive.invalidate_later(0.5)
            return
        launched["integrating_since"] = time.monotonic()
        sim_task_state.set(
            update_task_progress(state, message=f"Integrating with {launched['label']}")
        )

    @output
    @render.ui
    def progress_display():
        """Display progress/completion info."""
        output = sim_output.get()
        scen = scenario.get()
        task_state = sim_task_state.get()

        if task_state.status is TaskStatus.RUNNING:
            # Indeterminate: the core reports no per-month progress. The
            # elapsed seconds tick in their own output, so this block and its
            # Cancel button are not rebuilt every second.
            return ui.div(
                ui.tags.h5("Simulation Progress"),
                ui.div(
                    ui.div(
                        class_="progress-bar progress-bar-striped progress-bar-animated",
                        role="progressbar",
                        style="width: 100%",
                        aria_label="Simulation running",
                    ),
                    class_="progress",
                ),
                ui.p(task_state.message, class_="text-muted mt-2 mb-0"),
                ui.p(ui.output_text("sim_elapsed", inline=True), class_="text-muted"),
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

        request = getattr(output, "run_request", None)
        info = getattr(output, "solver_info", None)
        method_rows = []
        if request is not None:
            summary = dict(_solver_summary(info, _group_names(scen)))
            shown = (
                summary["Method"]
                if "Method" in summary
                else {"RK4": "Runge-Kutta 4", "AB": "Adams-Bashforth"}.get(
                    request.method, str(request.method)
                )
            )
            method_rows.append(ui.tags.tr(ui.tags.td("Method:"), ui.tags.td(shown)))
            for label in ("Tolerance", "Model options"):
                if label in summary:
                    method_rows.append(
                        ui.tags.tr(ui.tags.td(f"{label}:"), ui.tags.td(summary[label]))
                    )

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
                *method_rows,
                class_="table table-sm",
            ),
        )

    @output
    @render.ui
    def solver_diagnostics():
        """Badges, summary and flagged months for a continuous run."""
        output = sim_output.get()
        info = getattr(output, "solver_info", None)
        if info is None:
            return None
        names = _group_names(scenario.get())
        badges = _solver_badges(info, names)
        flagged_view = _flagged_months_view(_flagged_months(info))
        return ui.card(
            ui.card_header("Solver diagnostics"),
            ui.div(
                *[
                    ui.span(b.text, class_=f"badge text-bg-{b.level} me-1 mb-1")
                    for b in badges
                ],
                class_="mb-2",
            ),
            ui.tags.table(
                *[
                    ui.tags.tr(ui.tags.th(label), ui.tags.td(value))
                    for label, value in _solver_summary(info, names)
                ],
                class_="table table-sm",
            ),
            flagged_view,
            ui.accordion(
                ui.accordion_panel(
                    "Every month", ui.output_data_frame("solver_month_table")
                ),
                open=False,
            ),
        )

    @output
    @render.data_frame
    def solver_month_table():
        """Every month's solver counters (a pandas frame, never GT)."""
        info = getattr(sim_output.get(), "solver_info", None)
        return render.DataGrid(_month_table(info), height="400px")

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
