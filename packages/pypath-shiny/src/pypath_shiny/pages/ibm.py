"""
Individual-Based Model (IBM) Page

Interactive configuration, execution, and visualization of IBM simulations
coupled with Ecosim. Users can:
- Select a functional group to model with super-individuals
- Configure VBGF growth, bioenergetics, predation, movement, and reproduction
- Initialize an age-structured population from Ecopath equilibrium
- Run standard vs. IBM-enhanced Ecosim and compare results
"""

import copy
import logging

import numpy as np
import pandas as pd
from shiny import Inputs, Outputs, Session, reactive, render, req, ui

from pypath_shiny.config import IBM

logger = logging.getLogger(__name__)


def ibm_ui():
    """UI for the Individual-Based Model page."""
    return ui.page_fluid(
        ui.layout_sidebar(
            ui.sidebar(
                ui.h4("IBM Configuration", class_="mb-3"),
                # Group selection
                ui.input_select(
                    "ibm_group_select",
                    "Functional Group",
                    choices={},
                    width="100%",
                ),
                ui.p(
                    "Select a consumer group to model with super-individuals.",
                    class_="text-muted small mb-3",
                ),
                # Parameter accordion
                ui.accordion(
                    # Growth parameters
                    ui.accordion_panel(
                        "Growth (VBGF)",
                        ui.input_slider(
                            "ibm_vbgf_k",
                            "Growth coefficient K (yr⁻¹)",
                            min=IBM.vbgf_k_min,
                            max=IBM.vbgf_k_max,
                            value=IBM.vbgf_k_default,
                            step=0.01,
                        ),
                        ui.input_slider(
                            "ibm_vbgf_linf",
                            "Asymptotic length Linf (cm)",
                            min=IBM.vbgf_linf_min,
                            max=IBM.vbgf_linf_max,
                            value=IBM.vbgf_linf_default,
                            step=1.0,
                        ),
                        ui.input_slider(
                            "ibm_max_age",
                            "Maximum age (years)",
                            min=IBM.max_age_min,
                            max=IBM.max_age_max,
                            value=IBM.max_age_default,
                            step=1.0,
                        ),
                        icon=ui.tags.i(class_="bi bi-graph-up"),
                    ),
                    # Bioenergetics parameters
                    ui.accordion_panel(
                        "Bioenergetics",
                        ui.input_slider(
                            "ibm_q10",
                            "Q10 temperature coefficient",
                            min=IBM.q10_min,
                            max=IBM.q10_max,
                            value=IBM.q10_default,
                            step=0.1,
                        ),
                        ui.input_numeric(
                            "ibm_t_ref",
                            "Reference temperature (°C)",
                            value=IBM.t_ref_default,
                            min=0,
                            max=30,
                            step=1,
                        ),
                        ui.input_numeric(
                            "ibm_ra",
                            "Metabolic rate intercept (ra)",
                            value=IBM.ra_default,
                            min=0.0001,
                            step=0.0001,
                        ),
                        ui.input_slider(
                            "ibm_energy_density",
                            "Energy density (kJ/g)",
                            min=IBM.energy_density_min,
                            max=IBM.energy_density_max,
                            value=IBM.energy_density_default,
                            step=0.5,
                        ),
                        icon=ui.tags.i(class_="bi bi-thermometer-half"),
                    ),
                    # Predation parameters
                    ui.accordion_panel(
                        "Predation",
                        ui.input_slider(
                            "ibm_optimal_prey_length",
                            "Optimal prey length (cm)",
                            min=IBM.optimal_prey_length_min,
                            max=IBM.optimal_prey_length_max,
                            value=IBM.optimal_prey_length_default,
                            step=1.0,
                        ),
                        ui.input_slider(
                            "ibm_selectivity_sd",
                            "Selectivity σ",
                            min=IBM.selectivity_sd_min,
                            max=IBM.selectivity_sd_max,
                            value=IBM.selectivity_sd_default,
                            step=0.1,
                        ),
                        icon=ui.tags.i(class_="bi bi-crosshair"),
                    ),
                    # Movement parameters
                    ui.accordion_panel(
                        "Movement",
                        ui.input_numeric(
                            "ibm_base_speed",
                            "Base speed (0-1)",
                            value=IBM.base_speed_default,
                            min=0,
                            max=1,
                            step=0.1,
                        ),
                        ui.input_numeric(
                            "ibm_habitat_weight",
                            "Habitat weight (0-1)",
                            value=IBM.habitat_weight_default,
                            min=0,
                            max=1,
                            step=0.1,
                        ),
                        ui.input_numeric(
                            "ibm_food_weight",
                            "Food weight (0-1)",
                            value=IBM.food_weight_default,
                            min=0,
                            max=1,
                            step=0.1,
                        ),
                        ui.input_numeric(
                            "ibm_predator_weight",
                            "Predator weight (0-1)",
                            value=IBM.predator_weight_default,
                            min=0,
                            max=1,
                            step=0.1,
                        ),
                        icon=ui.tags.i(class_="bi bi-arrows-move"),
                    ),
                    # Reproduction parameters
                    ui.accordion_panel(
                        "Reproduction",
                        ui.input_slider(
                            "ibm_fecundity_coeff",
                            "Fecundity coefficient",
                            min=IBM.fecundity_coefficient_min,
                            max=IBM.fecundity_coefficient_max,
                            value=IBM.fecundity_coefficient_default,
                            step=10.0,
                        ),
                        ui.input_slider(
                            "ibm_larval_survival",
                            "Larval base survival",
                            min=IBM.larval_base_survival_min,
                            max=IBM.larval_base_survival_max,
                            value=IBM.larval_base_survival_default,
                            step=0.001,
                        ),
                        ui.input_slider(
                            "ibm_spawning_temp",
                            "Spawning temp. threshold (°C)",
                            min=IBM.spawning_temp_threshold_min,
                            max=IBM.spawning_temp_threshold_max,
                            value=IBM.spawning_temp_threshold_default,
                            step=0.5,
                        ),
                        icon=ui.tags.i(class_="bi bi-egg"),
                    ),
                    id="ibm_accordion",
                    open=["Growth (VBGF)"],
                    multiple=True,
                ),
                ui.hr(),
                # Super-individual count
                ui.input_numeric(
                    "ibm_n_super",
                    "Number of super-individuals",
                    value=IBM.n_super_individuals_default,
                    min=IBM.n_super_individuals_min,
                    max=IBM.n_super_individuals_max,
                    step=IBM.n_super_individuals_step,
                ),
                # Action buttons
                ui.input_action_button(
                    "ibm_initialize",
                    ui.tags.span(
                        ui.tags.i(class_="bi bi-play-circle me-2"),
                        "Initialize IBM",
                    ),
                    class_="btn btn-primary w-100 mt-2",
                ),
                ui.input_action_button(
                    "ibm_run",
                    ui.tags.span(
                        ui.tags.i(class_="bi bi-lightning-fill me-2"),
                        "Run Comparison",
                    ),
                    class_="btn btn-success w-100 mt-2",
                ),
                width=350,
            ),
            # Main content tabs
            ui.navset_card_tab(
                ui.nav_panel(
                    "Configuration",
                    ui.h5("Parameter Summary", class_="mt-3 mb-3"),
                    ui.output_table("ibm_param_summary"),
                    ui.h5("Initialization Status", class_="mt-4 mb-3"),
                    ui.output_ui("ibm_init_status"),
                ),
                ui.nav_panel(
                    "Population Structure",
                    ui.output_plot("ibm_size_distribution", height="400px"),
                    ui.output_plot("ibm_age_distribution", height="400px"),
                    ui.output_table("ibm_population_stats"),
                ),
                ui.nav_panel(
                    "Biomass Comparison",
                    ui.output_plot("ibm_biomass_comparison", height="500px"),
                    ui.p(
                        "Compares standard Ecosim trajectory with IBM-enhanced "
                        "simulation for the selected group.",
                        class_="text-muted small mt-2",
                    ),
                ),
                ui.nav_panel(
                    "Energy & Growth",
                    ui.output_plot("ibm_growth_curves", height="450px"),
                    ui.output_plot("ibm_energy_budget", height="450px"),
                ),
                ui.nav_panel(
                    "Reproduction & Mortality",
                    ui.output_plot("ibm_recruitment_mortality", height="500px"),
                ),
                id="ibm_tabs",
            ),
        ),
        # Page header
        ui.div(
            ui.h2(
                ui.tags.i(class_="bi bi-people-fill me-2"),
                "Individual-Based Model",
                class_="mb-2",
            ),
            ui.p(
                "Configure and run individual-based population models coupled "
                "with Ecosim. Compare standard aggregate dynamics against "
                "size-structured super-individual simulations.",
                class_="text-muted mb-4",
            ),
            class_="mb-4",
        ),
    )


def ibm_server(
    input: Inputs,
    _output: Outputs,
    _session: Session,
    _model_data: reactive.Value,
    _sim_results: reactive.Value,
    _sim_scenario: reactive.Value = None,
) -> None:
    """Server logic for the IBM page."""

    # Reactive values for IBM state
    ibm_group_instance = reactive.Value(None)
    ibm_params = reactive.Value(None)
    ibm_enhanced_output = reactive.Value(None)
    standard_output = reactive.Value(None)

    # ----------------------------------------------------------------
    # Update group choices when scenario changes
    # ----------------------------------------------------------------
    @reactive.effect
    def _update_group_choices():
        """Populate group dropdown with consumer groups from the model."""
        model = _model_data()
        if model is None:
            return
        if not hasattr(model, "model") or model.model is None:
            return

        df = model.model
        # Filter to consumer groups (Type == 0)
        choices = {}
        for idx, row in df.iterrows():
            group_type = row.get("Type", row.get("type", None))
            group_name = row.get("Group", row.get("group", f"Group {idx}"))
            if group_type == 0:
                choices[str(idx)] = group_name

        if not choices:
            # Fallback: offer all groups
            for idx, row in df.iterrows():
                group_name = row.get("Group", row.get("group", f"Group {idx}"))
                choices[str(idx)] = group_name

        ui.update_select("ibm_group_select", choices=choices)

    # ----------------------------------------------------------------
    # Build SmeltParams from current UI inputs
    # ----------------------------------------------------------------
    def _build_params(n_groups: int):
        """Construct SmeltParams from the current slider/input values."""
        from pypath.ibm import (
            BioenergParams,
            ForagingParams,
            MovementParams,
            PredationParams,
            ReproductionParams,
            SmeltParams,
        )

        bioenerg = BioenergParams(
            ra=float(input.ibm_ra()),
            rb=IBM.rb_default,
            q10=float(input.ibm_q10()),
            t_ref=float(input.ibm_t_ref()),
            sda_fraction=IBM.sda_fraction_default,
            unassimilated_fraction=IBM.unassimilated_fraction_default,
            a_length=0.55,
            b_length=0.333,
            energy_density=float(input.ibm_energy_density()),
            reproduction_fraction=IBM.reproduction_fraction_default,
        )

        predation = PredationParams(
            optimal_prey_length=float(input.ibm_optimal_prey_length()),
            selectivity_sd=float(input.ibm_selectivity_sd()),
        )

        # Size foraging arrays to actual model group count
        foraging = ForagingParams(
            energy_content=np.full(n_groups, 4.0),
            handling_time=np.full(n_groups, 1.0),
        )

        movement = MovementParams(
            base_speed=float(input.ibm_base_speed()),
            habitat_weight=float(input.ibm_habitat_weight()),
            food_weight=float(input.ibm_food_weight()),
            predator_weight=float(input.ibm_predator_weight()),
            migration_temp_threshold=IBM.migration_temp_threshold_default,
            migration_months=(3, 4, 5),
        )

        reproduction = ReproductionParams(
            fecundity_coefficient=float(input.ibm_fecundity_coeff()),
            fecundity_exponent=IBM.fecundity_exponent_default,
            larval_base_survival=float(input.ibm_larval_survival()),
            zooplankton_match_window=15.0,
            maturity_energy_threshold=0.5,
            spawning_temp_threshold=float(input.ibm_spawning_temp()),
            larval_duration_days=30,
            recruit_weight=0.5,
            recruit_length=3.0,
        )

        return SmeltParams(
            bioenerg=bioenerg,
            predation=predation,
            foraging=foraging,
            movement=movement,
            reproduction=reproduction,
            vbgf_k_mean=float(input.ibm_vbgf_k()),
            vbgf_k_sd=IBM.vbgf_k_sd_default,
            vbgf_linf_mean=float(input.ibm_vbgf_linf()),
            vbgf_linf_sd=IBM.vbgf_linf_sd_default,
            max_age=float(input.ibm_max_age()),
        )

    # ----------------------------------------------------------------
    # Initialize IBM
    # ----------------------------------------------------------------
    @reactive.effect
    @reactive.event(input.ibm_initialize)
    def _initialize_ibm():
        """Create SmeltIBM and initialize from Ecopath equilibrium biomass."""
        from pypath.ibm import SmeltIBM

        model = _model_data()
        if model is None:
            ui.notification_show(
                "Please load an Ecopath model first (Data Import tab).",
                type="warning",
                duration=5,
            )
            return

        scenario = _sim_scenario() if _sim_scenario is not None else None
        if scenario is None:
            ui.notification_show(
                "Please create an Ecosim scenario first (Ecosim Simulation tab).",
                type="warning",
                duration=5,
            )
            return

        group_str = input.ibm_group_select()
        if not group_str:
            ui.notification_show(
                "Please select a functional group.",
                type="warning",
                duration=5,
            )
            return

        try:
            group_idx = int(group_str)
            n_groups = scenario.params.NUM_GROUPS

            # Build parameters
            params = _build_params(n_groups)
            ibm_params.set(params)

            # Create IBM instance
            ibm = SmeltIBM(
                group_index=group_idx,
                n_groups=n_groups,
                params=params,
            )

            # Get equilibrium biomass for this group (1-based index in Ecosim)
            biomass = float(scenario.params.B_BaseRef[group_idx + 1])

            n_super = int(input.ibm_n_super())
            ibm.initialize_from_ecosim(
                biomass=biomass,
                params={},
                n_super_individuals=n_super,
            )

            ibm_group_instance.set(ibm)

            # Get group name for notification
            df = model.model
            group_name = df.iloc[group_idx].get(
                "Group", df.iloc[group_idx].get("group", f"Group {group_idx}")
            )

            ui.notification_show(
                f"IBM initialized for {group_name}: "
                f"{len(ibm.individuals)} super-individuals, "
                f"biomass = {ibm.get_aggregate_biomass():.4f} t",
                type="message",
                duration=5,
            )

        except Exception as e:
            logger.error("IBM initialization failed: %s", e, exc_info=True)
            ui.notification_show(
                f"IBM initialization failed: {e!s}",
                type="error",
                duration=8,
            )

    # ----------------------------------------------------------------
    # Run comparison: standard vs. IBM-enhanced Ecosim
    # ----------------------------------------------------------------
    @reactive.effect
    @reactive.event(input.ibm_run)
    def _run_comparison():
        """Run both standard and IBM-enhanced Ecosim, store results."""
        from pypath.core.ecosim import rsim_run

        ibm = ibm_group_instance()
        if ibm is None:
            ui.notification_show(
                "Please initialize the IBM first.",
                type="warning",
                duration=5,
            )
            return

        scenario = _sim_scenario() if _sim_scenario is not None else None
        if scenario is None:
            ui.notification_show(
                "No Ecosim scenario available.",
                type="warning",
                duration=5,
            )
            return

        try:
            ui.notification_show(
                "Running standard Ecosim...", type="message", duration=2
            )

            # Run standard Ecosim (on a deep copy)
            std_scenario = copy.deepcopy(scenario)
            std_result = rsim_run(std_scenario)
            standard_output.set(std_result)

            ui.notification_show(
                "Running IBM-enhanced Ecosim...", type="message", duration=2
            )

            # Run IBM-enhanced Ecosim
            ibm_scenario = copy.deepcopy(scenario)

            # Inject IBM group into scenario
            if not hasattr(ibm_scenario.params, "ibm_groups"):
                ibm_scenario.params.ibm_groups = {}
            # Re-initialize IBM for the fresh run
            from pypath.ibm import SmeltIBM

            ibm_fresh = SmeltIBM(
                group_index=ibm.group_index,
                n_groups=ibm.n_groups,
                params=ibm.params,
            )
            biomass = float(ibm_scenario.params.B_BaseRef[ibm.group_index + 1])
            ibm_fresh.initialize_from_ecosim(
                biomass=biomass,
                params={},
                n_super_individuals=len(ibm.individuals),
            )
            ibm_scenario.params.ibm_groups[ibm.group_index] = ibm_fresh

            ibm_result = rsim_run(ibm_scenario)
            ibm_enhanced_output.set(ibm_result)

            ui.notification_show("Comparison run complete!", type="message", duration=3)

        except Exception as e:
            logger.error("IBM comparison run failed: %s", e, exc_info=True)
            ui.notification_show(
                f"IBM run failed: {e!s}",
                type="error",
                duration=8,
            )

    # ----------------------------------------------------------------
    # Renderers
    # ----------------------------------------------------------------

    @render.table
    def ibm_param_summary():
        """Show parameter summary table."""
        rows = [
            ("VBGF K", f"{input.ibm_vbgf_k():.3f}", "yr⁻¹"),
            ("VBGF Linf", f"{input.ibm_vbgf_linf():.1f}", "cm"),
            ("Max age", f"{input.ibm_max_age():.0f}", "years"),
            ("Q10", f"{input.ibm_q10():.1f}", ""),
            ("Reference temp.", f"{input.ibm_t_ref():.1f}", "°C"),
            ("Ra", f"{input.ibm_ra():.4f}", "g O₂/g/day"),
            ("Energy density", f"{input.ibm_energy_density():.1f}", "kJ/g"),
            ("Optimal prey L", f"{input.ibm_optimal_prey_length():.1f}", "cm"),
            ("Selectivity σ", f"{input.ibm_selectivity_sd():.2f}", ""),
            ("Fecundity coeff.", f"{input.ibm_fecundity_coeff():.0f}", "eggs/g^exp"),
            ("Larval survival", f"{input.ibm_larval_survival():.3f}", ""),
            ("Spawning temp.", f"{input.ibm_spawning_temp():.1f}", "°C"),
            ("N super-ind.", f"{input.ibm_n_super()}", ""),
        ]
        return pd.DataFrame(rows, columns=["Parameter", "Value", "Unit"])

    @render.ui
    def ibm_init_status():
        """Show initialization status card."""
        ibm = ibm_group_instance()
        if ibm is None:
            return ui.div(
                ui.tags.i(class_="bi bi-info-circle me-2"),
                "IBM not yet initialized. Configure parameters and click "
                "'Initialize IBM'.",
                class_="alert alert-info",
            )

        n_ind = len(ibm.individuals)
        total_biomass = ibm.get_aggregate_biomass()
        ages = [ind.age for ind in ibm.individuals]
        weights = [ind.weight for ind in ibm.individuals]
        n_mature = sum(1 for ind in ibm.individuals if ind.is_mature)

        return ui.div(
            ui.div(
                ui.tags.i(class_="bi bi-check-circle-fill me-2"),
                f"IBM initialized: {n_ind} super-individuals",
                class_="alert alert-success mb-3",
            ),
            ui.tags.table(
                ui.tags.tbody(
                    ui.tags.tr(
                        ui.tags.td("Total biomass:", class_="pe-3 fw-bold"),
                        ui.tags.td(f"{total_biomass:.4f} tonnes"),
                    ),
                    ui.tags.tr(
                        ui.tags.td("Age range:", class_="pe-3 fw-bold"),
                        ui.tags.td(f"{min(ages):.1f} – {max(ages):.1f} years"),
                    ),
                    ui.tags.tr(
                        ui.tags.td("Weight range:", class_="pe-3 fw-bold"),
                        ui.tags.td(f"{min(weights):.2f} – {max(weights):.2f} g"),
                    ),
                    ui.tags.tr(
                        ui.tags.td("Mature:", class_="pe-3 fw-bold"),
                        ui.tags.td(
                            f"{n_mature} / {n_ind} ({100 * n_mature / n_ind:.0f}%)"
                        ),
                    ),
                ),
                class_="table table-sm table-borderless",
            ),
        )

    # -- Population Structure tab --

    @render.plot
    def ibm_size_distribution():
        """Plot size (length) distribution of super-individuals."""
        import matplotlib.pyplot as plt

        ibm = ibm_group_instance()
        req(ibm is not None and len(ibm.individuals) > 0)

        lengths = [ind.length for ind in ibm.individuals]
        n_represented = [ind.n_represented for ind in ibm.individuals]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(
            range(len(lengths)),
            n_represented,
            color="steelblue",
            alpha=0.7,
            edgecolor="white",
            linewidth=0.3,
        )
        # Add a twin axis for length
        ax2 = ax.twinx()
        ax2.plot(
            range(len(lengths)),
            lengths,
            color="darkorange",
            marker=".",
            markersize=3,
            linewidth=1,
            label="Length (cm)",
        )
        ax2.set_ylabel("Length (cm)", color="darkorange")

        ax.set_xlabel("Super-individual index")
        ax.set_ylabel("N represented", color="steelblue")
        ax.set_title("Population Size Structure")
        ax2.legend(loc="upper right")
        fig.tight_layout()
        return fig

    @render.plot
    def ibm_age_distribution():
        """Plot age distribution of super-individuals."""
        import matplotlib.pyplot as plt

        ibm = ibm_group_instance()
        req(ibm is not None and len(ibm.individuals) > 0)

        ages = [ind.age for ind in ibm.individuals]
        n_represented = [ind.n_represented for ind in ibm.individuals]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(
            ages,
            n_represented,
            width=0.15,
            color="teal",
            alpha=0.7,
            edgecolor="white",
            linewidth=0.3,
        )
        ax.set_xlabel("Age (years)")
        ax.set_ylabel("N represented")
        ax.set_title("Age Distribution")
        fig.tight_layout()
        return fig

    @render.table
    def ibm_population_stats():
        """Summary statistics for the IBM population."""
        ibm = ibm_group_instance()
        if ibm is None or len(ibm.individuals) == 0:
            return pd.DataFrame(
                {"Statistic": ["N/A"], "Value": ["Initialize IBM first"]}
            )

        weights = np.array([ind.weight for ind in ibm.individuals])
        lengths = np.array([ind.length for ind in ibm.individuals])
        ages = np.array([ind.age for ind in ibm.individuals])
        n_rep = np.array([ind.n_represented for ind in ibm.individuals])

        rows = [
            ("Super-individuals", f"{len(ibm.individuals)}"),
            ("Total N represented", f"{n_rep.sum():.0f}"),
            ("Total biomass (t)", f"{ibm.get_aggregate_biomass():.4f}"),
            ("Mean weight (g)", f"{np.average(weights, weights=n_rep):.2f}"),
            ("Mean length (cm)", f"{np.average(lengths, weights=n_rep):.2f}"),
            ("Mean age (yr)", f"{np.average(ages, weights=n_rep):.2f}"),
            (
                "Mature fraction",
                f"{sum(1 for i in ibm.individuals if i.is_mature) / len(ibm.individuals):.1%}",
            ),
        ]
        return pd.DataFrame(rows, columns=["Statistic", "Value"])

    # -- Biomass Comparison tab --

    @render.plot
    def ibm_biomass_comparison():
        """Overlay standard vs. IBM-enhanced biomass trajectories."""
        import matplotlib.pyplot as plt

        std = standard_output()
        ibm_out = ibm_enhanced_output()
        req(std is not None and ibm_out is not None)

        ibm = ibm_group_instance()
        req(ibm is not None)

        # Ecosim uses 1-based indexing: group_index + 1
        col = ibm.group_index + 1

        fig, ax = plt.subplots(figsize=(10, 5))

        # Standard Ecosim
        std_biomass = std.out_Biomass[:, col]
        time_months = np.arange(len(std_biomass))
        ax.plot(
            time_months,
            std_biomass,
            label="Standard Ecosim",
            color="steelblue",
            linewidth=2,
        )

        # IBM-enhanced
        ibm_biomass = ibm_out.out_Biomass[:, col]
        ax.plot(
            time_months[: len(ibm_biomass)],
            ibm_biomass,
            label="IBM-enhanced",
            color="darkorange",
            linewidth=2,
            linestyle="--",
        )

        # Get group name
        model = _model_data()
        group_name = f"Group {ibm.group_index}"
        if model is not None and hasattr(model, "model"):
            try:
                group_name = model.model.iloc[ibm.group_index].get(
                    "Group",
                    model.model.iloc[ibm.group_index].get("group", group_name),
                )
            except (IndexError, AttributeError):
                pass

        ax.set_xlabel("Time (months)")
        ax.set_ylabel("Biomass (t/km²)")
        ax.set_title(f"Biomass Comparison — {group_name}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    # -- Energy & Growth tab --

    @render.plot
    def ibm_growth_curves():
        """Plot Von Bertalanffy growth curves for the IBM population."""
        import matplotlib.pyplot as plt

        ibm = ibm_group_instance()
        params_val = ibm_params()
        req(ibm is not None and params_val is not None)

        fig, ax = plt.subplots(figsize=(8, 4.5))

        # Theoretical VBGF curve
        ages_theory = np.linspace(0, params_val.max_age, 200)
        lengths_theory = params_val.vbgf_linf_mean * (
            1 - np.exp(-params_val.vbgf_k_mean * ages_theory)
        )
        ax.plot(
            ages_theory,
            lengths_theory,
            color="black",
            linewidth=2,
            label=f"VBGF (K={params_val.vbgf_k_mean:.2f}, "
            f"Linf={params_val.vbgf_linf_mean:.0f})",
        )

        # Individual data points
        ages = [ind.age for ind in ibm.individuals]
        lengths = [ind.length for ind in ibm.individuals]
        ax.scatter(
            ages,
            lengths,
            c="darkorange",
            s=20,
            alpha=0.6,
            label="Super-individuals",
            zorder=5,
        )

        ax.set_xlabel("Age (years)")
        ax.set_ylabel("Length (cm)")
        ax.set_title("Von Bertalanffy Growth")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    @render.plot
    def ibm_energy_budget():
        """Visualize energy budget components from bioenergetics parameters."""
        import matplotlib.pyplot as plt

        params_val = ibm_params()
        req(params_val is not None)

        bp = params_val.bioenerg
        categories = ["Assimilated", "SDA", "Unassimilated", "Reproduction"]
        assimilated = 1.0 - bp.unassimilated_fraction
        values = [
            assimilated * (1 - bp.sda_fraction) * (1 - bp.reproduction_fraction),
            assimilated * bp.sda_fraction,
            bp.unassimilated_fraction,
            assimilated * (1 - bp.sda_fraction) * bp.reproduction_fraction,
        ]
        colors = ["#2ecc71", "#e74c3c", "#95a5a6", "#f39c12"]

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.barh(categories, values, color=colors, edgecolor="white", height=0.5)
        ax.set_xlabel("Fraction of consumption")
        ax.set_title("Energy Budget Allocation")
        ax.set_xlim(0, 1)
        for i, v in enumerate(values):
            ax.text(v + 0.01, i, f"{v:.1%}", va="center", fontsize=9)
        fig.tight_layout()
        return fig

    # -- Reproduction & Mortality tab --

    @render.plot
    def ibm_recruitment_mortality():
        """Plot recruitment and mortality from IBM-enhanced run."""
        import matplotlib.pyplot as plt

        ibm_out = ibm_enhanced_output()
        req(ibm_out is not None)

        ibm = ibm_group_instance()
        req(ibm is not None)

        col = ibm.group_index + 1

        fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)

        # Biomass trajectory (top)
        biomass = ibm_out.out_Biomass[:, col]
        time = np.arange(len(biomass))
        axes[0].plot(time, biomass, color="steelblue", linewidth=1.5)
        axes[0].set_ylabel("Biomass (t/km²)")
        axes[0].set_title("IBM-Enhanced Biomass & Dynamics")
        axes[0].grid(True, alpha=0.3)

        # Biomass change as proxy for recruitment/mortality (bottom)
        delta = np.diff(biomass)
        positive = np.where(delta > 0, delta, 0)
        negative = np.where(delta < 0, delta, 0)
        axes[1].bar(
            time[1:],
            positive,
            color="green",
            alpha=0.6,
            label="Net growth",
            width=1.0,
        )
        axes[1].bar(
            time[1:],
            negative,
            color="red",
            alpha=0.6,
            label="Net decline",
            width=1.0,
        )
        axes[1].set_xlabel("Time (months)")
        axes[1].set_ylabel("ΔBiomass (t/km²)")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        return fig
