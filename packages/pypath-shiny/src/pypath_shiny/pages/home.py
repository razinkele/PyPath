"""Home page module."""

import logging
import warnings

import pandas as pd
from pypath.core.ecopath import rpath
from pypath.core.params import create_rpath_params
from shiny import Inputs, Outputs, Session, reactive, ui

from pypath_shiny.components import feature_card, quick_start_step
from pypath_shiny.config import DEFAULTS

# Import centralized configuration
from pypath_shiny.services.notifications import (
    notify_error,
    notify_info,
    notify_success,
    notify_warning,
)

logger = logging.getLogger(__name__)


def _notify_legacy(message: str, type: str = "default", duration: int | None = None):
    """Compatibility wrapper while migrating this page to notification services."""
    if type == "error":
        notify_error(message, duration=duration)
    elif type == "warning":
        notify_warning(message, duration=duration)
    elif type == "message":
        notify_success(message, duration=duration)
    else:
        notify_info(message, duration=duration)


def build_example_params():
    """The Home page example model (10 groups, Roundfish multi-stanza).

    Each living group gets *either* Biomass *or* EE (never both), so Ecopath
    solves the other and the model is mass-balanced: Ecosim then starts at
    equilibrium. With both given for every group, Ecopath echoed the EEs
    unchecked (Foragefish1 was really eaten at EE ~2.4) and Ecosim crashed
    Foragefish1, then Seals.
    """
    # Create example marine ecosystem model
    groups = [
        "Seals",  # Top predator
        "JuvRoundfish1",  # Juvenile fish
        "AduRoundfish1",  # Adult fish
        "OtherGroundfish",  # Groundfish
        "Foragefish1",  # Forage fish
        "Megabenthos",  # Large benthos
        "Zooplankton",  # Zooplankton
        "Phytoplankton",  # Primary producer
        "Detritus",  # Detritus
        "Trawlers",  # Fishing fleet
    ]

    types = [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        2,
        3,
    ]  # consumer, producer, detritus, fleet

    # Define stanza groups - make JuvRoundfish1 and AduRoundfish1 into multi-stanza group
    stgroups_list = [
        None,
        "Roundfish",
        "Roundfish",
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ]

    params = create_rpath_params(groups, types, stgroups=stgroups_list)

    # Initialize remarks DataFrame (same structure as model)
    remarks_cols = ["Group"] + [col for col in params.model.columns if col != "Group"]
    params.remarks = pd.DataFrame({col: [""] * len(groups) for col in remarks_cols})

    # Populate multi-stanza parameters for the Roundfish stanza group
    if params.stanzas.n_stanza_groups > 0:
        # Set stanza group parameters (von Bertalanffy growth, maturity weight, etc.)
        params.stanzas.stgroups.loc[0, "VBGF_Ksp"] = 0.4  # von Bertalanffy growth rate
        params.stanzas.stgroups.loc[0, "VBGF_d"] = 0.66667  # VBGF allometric parameter
        # Weight at maturity as a fraction of Winf (Rpath convention), not grams
        params.stanzas.stgroups.loc[0, "Wmat"] = 0.1
        params.stanzas.stgroups.loc[0, "BAB"] = 0.0  # Biomass accumulation rate
        params.stanzas.stgroups.loc[0, "RecPower"] = 1.0  # Recruitment power

        # Set individual stanza parameters (age ranges and mortality)
        # Z of each stanza equals the P/B of its group (Rpath convention).
        # Juvenile: months 0-23, Z=1.5
        juv_idx = params.stanzas.stindiv[
            params.stanzas.stindiv["Group"] == "JuvRoundfish1"
        ].index[0]
        params.stanzas.stindiv.loc[juv_idx, "StanzaNum"] = 1
        params.stanzas.stindiv.loc[juv_idx, "First"] = 0  # Start at birth
        params.stanzas.stindiv.loc[juv_idx, "Last"] = 23  # Adults start at 24
        params.stanzas.stindiv.loc[juv_idx, "Z"] = 1.5  # Total mortality
        params.stanzas.stindiv.loc[juv_idx, "Leading"] = 0  # Not leading stanza

        # Adult: 24+ months, Z=0.35, leading stanza
        adu_idx = params.stanzas.stindiv[
            params.stanzas.stindiv["Group"] == "AduRoundfish1"
        ].index[0]
        params.stanzas.stindiv.loc[adu_idx, "StanzaNum"] = 2
        params.stanzas.stindiv.loc[adu_idx, "First"] = 24  # Start at 24 months
        params.stanzas.stindiv.loc[adu_idx, "Last"] = (
            DEFAULTS.default_months
        )  # End at default months (10 years)
        params.stanzas.stindiv.loc[adu_idx, "Z"] = 0.35  # Total mortality
        params.stanzas.stindiv.loc[adu_idx, "Leading"] = (
            1  # Leading stanza (plus group)
        )

        # Add StanzaGroup column to stindiv for clarity
        params.stanzas.stindiv["StanzaGroup"] = "Roundfish"

    # Add helpful remarks/tooltips for key parameters
    # Find group indices
    seal_idx = groups.index("Seals")
    juv_idx = groups.index("JuvRoundfish1")
    adu_idx = groups.index("AduRoundfish1")
    phyto_idx = groups.index("Phytoplankton")
    det_idx = groups.index("Detritus")

    params.remarks.loc[seal_idx, "Biomass"] = "Low biomass typical for top predator"
    params.remarks.loc[seal_idx, "EE"] = (
        "Left blank: Ecopath computes it (top predator, mostly catch)"
    )
    params.remarks.loc[juv_idx, "Biomass"] = (
        "Part of Roundfish multi-stanza group; derived from the adult stanza"
    )
    params.remarks.loc[juv_idx, "EE"] = (
        "Left blank: Ecopath computes it from the given Biomass"
    )
    params.remarks.loc[adu_idx, "Biomass"] = "Leading stanza of Roundfish group"
    params.remarks.loc[adu_idx, "PB"] = "Lower P/B for adult stage"
    params.remarks.loc[phyto_idx, "Type"] = "Primary producer (Type=1)"
    params.remarks.loc[phyto_idx, "Biomass"] = "Autotroph - no QB value needed"
    params.remarks.loc[det_idx, "Type"] = "Detritus pool (Type=2)"
    params.remarks.loc[det_idx, "DetInput"] = "Import of detritus from outside system"

    # Set model parameters
    # Biomass for the top predator and the stanza groups; Ecopath solves
    # their EE. The other living groups get an EE and Ecopath solves Biomass.
    biomass_data = {
        "Seals": 0.025,
        # Juvenile Biomass and Q/B are what rpath_stanzas derives from the
        # adult (leading) stanza, so Ecopath and the stanza age structure agree.
        "JuvRoundfish1": 0.219007,
        "AduRoundfish1": 1.39,
        "Detritus": 500.0,
    }
    pb_data = {
        "Seals": 0.15,
        "JuvRoundfish1": 1.5,
        "AduRoundfish1": 0.35,
        "OtherGroundfish": 0.4,
        "Foragefish1": 0.7,
        "Megabenthos": 0.2,
        "Zooplankton": 30.0,
        "Phytoplankton": 200.0,
    }
    qb_data = {
        "Seals": 25.0,
        "JuvRoundfish1": 8.253302,
        "AduRoundfish1": 3.5,
        "OtherGroundfish": 2.0,
        "Foragefish1": 5.0,
        "Megabenthos": 1.5,
        "Zooplankton": 100.0,
    }
    ee_data = {
        "OtherGroundfish": 0.8,
        "Foragefish1": 0.9,
        "Megabenthos": 0.6,
        "Zooplankton": 0.9,
        "Phytoplankton": 0.8,
    }

    for i, group in enumerate(groups):
        if group in biomass_data:
            params.model.loc[i, "Biomass"] = biomass_data[group]
        if group in pb_data:
            params.model.loc[i, "PB"] = pb_data[group]
        if group in qb_data:
            params.model.loc[i, "QB"] = qb_data[group]
        if group in ee_data:
            params.model.loc[i, "EE"] = ee_data[group]

    # Set defaults - consumers get config value, producers/detritus get 0.0
    params.model["BioAcc"] = 0.0
    params.model.loc[params.model["Type"] == 0, "Unassim"] = (
        DEFAULTS.unassim_consumers
    )  # Consumers
    params.model.loc[params.model["Type"] == 1, "Unassim"] = (
        DEFAULTS.unassim_producers
    )  # Producers
    params.model.loc[params.model["Type"] == 2, "Unassim"] = (
        DEFAULTS.unassim_producers
    )  # Detritus
    params.model.loc[params.model["Type"] == 3, "BioAcc"] = float("nan")
    params.model.loc[params.model["Type"] == 3, "Unassim"] = float("nan")
    params.model["Detritus"] = 1.0
    # Detritus does not flow to itself: its unused surplus leaves the system
    # (as in Rpath's REco). With a self-fate of 1, Ecosim keeps that surplus
    # in the pool and Detritus grows ~10%/yr from the Ecopath balance.
    params.model.loc[params.model["Type"] == 2, "Detritus"] = 0.0
    params.model.loc[params.model["Type"] == 3, "Detritus"] = float("nan")

    # Set diet matrix
    prey_names = list(params.diet["Group"])
    n_prey = len(prey_names)

    def make_diet(diet_dict):
        diet = [0.0] * n_prey
        for prey, prop in diet_dict.items():
            if prey in prey_names:
                diet[prey_names.index(prey)] = prop
        return diet

    params.diet["Seals"] = make_diet(
        {"Foragefish1": 0.4, "AduRoundfish1": 0.3, "OtherGroundfish": 0.3}
    )
    params.diet["JuvRoundfish1"] = make_diet({"Zooplankton": 0.9, "Megabenthos": 0.1})
    params.diet["AduRoundfish1"] = make_diet(
        {"Foragefish1": 0.5, "Zooplankton": 0.3, "Megabenthos": 0.2}
    )
    params.diet["OtherGroundfish"] = make_diet(
        {"Foragefish1": 0.4, "Megabenthos": 0.3, "Zooplankton": 0.3}
    )
    params.diet["Foragefish1"] = make_diet({"Zooplankton": 1.0})
    params.diet["Megabenthos"] = make_diet({"Phytoplankton": 0.3, "Detritus": 0.7})
    params.diet["Zooplankton"] = make_diet({"Phytoplankton": 0.9, "Detritus": 0.1})
    params.diet["Phytoplankton"] = [0.0] * n_prey

    # Set fishing catches
    catches = {
        "AduRoundfish1": 0.145,
        "OtherGroundfish": 0.38,
        "Megabenthos": 0.19,
        "Seals": 0.002,
        "JuvRoundfish1": 0.003,
        "Foragefish1": 0.1,
    }
    for group, catch in catches.items():
        if group in groups:
            idx = groups.index(group)
            params.model.loc[idx, "Trawlers"] = catch

    return params


def home_ui():
    """Home page UI."""
    return ui.div(
        ui.div(
            # Hero section — ocean gradient with caustic light effects
            ui.div(
                ui.p("Ecosystem Modeling Framework", class_="hero-tagline"),
                ui.h1("Ecopath with Ecosim in Python", class_="hero-title"),
                ui.p(
                    "Build mass-balance food web models, run dynamic ecosystem simulations, "
                    "and explore spatial dynamics with irregular grids \u2014 all from your browser.",
                    class_="hero-description",
                ),
                ui.div(
                    ui.input_action_button(
                        "btn_start_ecopath",
                        ui.TagList(
                            ui.tags.i(class_="bi bi-arrow-right-circle-fill me-2"),
                            "Start Modeling",
                        ),
                        class_="btn-hero-primary",
                    ),
                    ui.input_action_button(
                        "btn_load_example",
                        ui.TagList(
                            ui.tags.i(class_="bi bi-play-circle me-2"),
                            "Load Example",
                        ),
                        class_="btn-hero-secondary",
                    ),
                    class_="hero-actions",
                ),
                class_="hero-section",
            ),
            # What's New section
            ui.div(
                ui.h2(
                    ui.tags.i(class_="bi bi-star-fill text-warning me-2"),
                    "What's New in PyPath",
                    class_="mb-3",
                ),
                ui.card(
                    ui.card_body(
                        ui.layout_columns(
                            ui.div(
                                ui.h5(
                                    ui.tags.i(class_="bi bi-geo-alt text-success me-2"),
                                    "Irregular Grid Support",
                                    class_="mb-2",
                                ),
                                ui.p(
                                    "Upload custom polygon geometries for realistic spatial modeling! "
                                    "ECOSPACE now supports shapefiles, GeoJSON, and GeoPackage formats.",
                                    class_="mb-1",
                                ),
                                ui.p(
                                    ui.tags.strong("Try it: "),
                                    "Navigate to the Ecospace page and upload ",
                                    ui.tags.code(
                                        "examples/coastal_grid_example.geojson"
                                    ),
                                    class_="text-muted small",
                                ),
                            ),
                            ui.div(
                                ui.h5(
                                    ui.tags.i(class_="bi bi-shuffle text-primary me-2"),
                                    "Diet Rewiring",
                                    class_="mb-2",
                                ),
                                ui.p(
                                    "Advanced prey switching behavior! Predators now adapt their diet "
                                    "based on prey availability with configurable switching power.",
                                    class_="mb-1",
                                ),
                                ui.p(
                                    ui.tags.strong("Explore: "),
                                    "Check out the Diet Rewiring Demo page for interactive examples.",
                                    class_="text-muted small",
                                ),
                            ),
                            ui.div(
                                ui.h5(
                                    ui.tags.i(
                                        class_="bi bi-lightning text-warning me-2"
                                    ),
                                    "Enhanced Ecosim",
                                    class_="mb-2",
                                ),
                                ui.p(
                                    "Environmental forcing, optimization tools, and improved "
                                    "multi-stanza support for age-structured populations.",
                                    class_="mb-1",
                                ),
                                ui.p(
                                    ui.tags.strong("Learn more: "),
                                    "See the Advanced Features demos for detailed examples.",
                                    class_="text-muted small",
                                ),
                            ),
                            ui.div(
                                ui.h5(
                                    ui.tags.i(class_="bi bi-cpu text-info me-2"),
                                    "Individual-Based Modeling",
                                    class_="mb-2",
                                ),
                                ui.p(
                                    "New IBM module with bioenergetics, predation, reproduction, "
                                    "and spatial movement for super-individual tracking.",
                                    class_="mb-1",
                                ),
                                ui.p(
                                    ui.tags.strong("Try it: "),
                                    "Navigate to the IBM page to configure super-individual groups.",
                                    class_="text-muted small",
                                ),
                            ),
                            col_widths=[3, 3, 3, 3],
                        )
                    ),
                    class_="border-success",
                ),
                class_="mb-4",
            ),
            # Feature cards - Row 1
            ui.h2("Features", class_="mb-4"),
            ui.layout_columns(
                feature_card(
                    "Data Import",
                    [
                        "Connect to EcoBase database (350+ models)",
                        "Search and download published models",
                        "Import EwE database files (.ewemdb, .eweaccdb)",
                        "Automatic format conversion",
                        "Pre-configured ecosystem models",
                    ],
                    icon_class="bi-cloud-download",
                    button_label="Import Model",
                    button_id="btn_goto_import",
                    button_class="btn-success",
                ),
                feature_card(
                    "Ecopath Mass Balance",
                    [
                        "Define functional groups and food web",
                        "Set biomass, P/B, Q/B ratios",
                        "Automatic mass-balance calculations",
                        "Trophic level computation",
                        "Multi-stanza age structure",
                    ],
                    icon_class="bi-diagram-3",
                    button_label="Create Ecopath Model",
                    button_id="btn_goto_ecopath",
                ),
                feature_card(
                    "Ecosim Simulation",
                    [
                        "Dynamic ecosystem simulations",
                        "Foraging arena functional response",
                        "Fishing scenarios and forcing",
                        "Environmental forcing effects",
                        "Multiple integration methods",
                    ],
                    icon_class="bi-graph-up",
                    button_label="Run Simulation",
                    button_id="btn_goto_ecosim",
                ),
                col_widths=[4, 4, 4],
            ),
            # Feature cards - Row 2
            ui.layout_columns(
                feature_card(
                    "ECOSPACE Spatial Modeling",
                    [
                        "Irregular grid support (GeoJSON, Shapefile)",
                        "Realistic spatial geometries",
                        "Habitat preference mapping",
                        "Dispersal and movement dynamics",
                        "Spatial fishing effort allocation",
                    ],
                    icon_class="bi-geo-alt",
                    button_label="Explore Spatial",
                    button_id="btn_goto_ecospace",
                    button_class="btn-success",
                    badge="NEW",
                ),
                feature_card(
                    "Individual-Based Model",
                    [
                        "Super-individual tracking",
                        "Wisconsin bioenergetics model",
                        "Size-structured predation",
                        "Stochastic reproduction",
                        "Spatial movement & foraging",
                    ],
                    icon_class="bi-cpu",
                    button_label="Explore IBM",
                    button_id="btn_goto_ibm",
                    button_class="btn-info",
                    badge="NEW",
                ),
                feature_card(
                    "Network Analysis",
                    [
                        "Food web topology metrics",
                        "Ecosystem health indicators",
                        "Trophic impact analysis (MTI)",
                        "Keystone species identification",
                        "Lindeman spine diagrams",
                    ],
                    icon_class="bi-diagram-2",
                    button_label="Run Analysis",
                    button_id="btn_goto_analysis",
                    button_class="btn-info",
                ),
                feature_card(
                    "Results & Visualization",
                    [
                        "Interactive time series plots",
                        "Food web diagrams",
                        "Biomass and catch trajectories",
                        "Export results to CSV/Excel",
                        "Comparative scenario analysis",
                    ],
                    icon_class="bi-bar-chart",
                    button_label="View Results",
                    button_id="btn_goto_results",
                ),
                col_widths=[3, 3, 3, 3],
                class_="mt-4",
            ),
            # Feature cards - Row 3
            ui.layout_columns(
                feature_card(
                    "Advanced Features",
                    [
                        "Diet rewiring (prey switching)",
                        "Environmental forcing functions",
                        "Multi-stanza age structure",
                        "Optimization and sensitivity analysis",
                        "Custom fishing scenarios",
                    ],
                    icon_class="bi-gear-wide-connected",
                ),
                feature_card(
                    "About PyPath",
                    [
                        "Python port of R Rpath package",
                        "NOAA/EwE ecosystem modeling",
                        "Open source (MIT License)",
                        "GitHub repository",
                        "Active development",
                    ],
                    icon_class="bi-info-circle",
                    button_label="Learn More",
                    button_id="btn_goto_about",
                    button_class="btn-secondary",
                ),
                feature_card(
                    "Documentation",
                    [
                        "User guides and tutorials",
                        "API reference documentation",
                        "Example models and scripts",
                        "Irregular grid guide (NEW)",
                        "Video tutorials (coming soon)",
                    ],
                    icon_class="bi-file-text",
                ),
                col_widths=[4, 4, 4],
                class_="mt-4",
            ),
            # Quick start section
            ui.h2("Quick Start", class_="mt-5 mb-4"),
            ui.layout_columns(
                quick_start_step(
                    "Step 1: Define Groups",
                    "Start by defining your ecosystem's functional groups - producers, consumers, detritus, and fishing fleets.",
                    "create_rpath_params(groups, types)",
                ),
                quick_start_step(
                    "Step 2: Set Parameters",
                    "Enter biomass, P/B, Q/B ratios, diet composition, and fishery catches for each group.",
                    "params.model['Biomass'] = ...",
                ),
                quick_start_step(
                    "Step 3: Balance Model",
                    "Run the mass-balance calculations to solve for missing parameters and validate the model.",
                    "model = rpath(params)",
                ),
                quick_start_step(
                    "Step 4: Run Simulation",
                    "Create a scenario and run dynamic Ecosim simulations to explore ecosystem dynamics.",
                    "rsim_run(scenario)",
                ),
                quick_start_step(
                    "Step 5: Add Spatial Dynamics",
                    "Upload spatial grids and run Ecospace simulations with habitat preferences and dispersal.",
                    "rsim_run_spatial(ecospace)",
                    badge="Optional",
                ),
                quick_start_step(
                    "Step 6: Individual-Based Modeling",
                    "Configure super-individual groups with bioenergetics, predation, and spatial movement parameters.",
                    "SmeltIBM.compute_step(state)",
                    badge="Optional",
                ),
                col_widths=[2, 2, 2, 2, 2, 2],
            ),
            class_="container py-4",
        )
    )


def home_server(
    input: Inputs, output: Outputs, session: Session, model_data: reactive.Value
):
    """Home page server logic."""

    @reactive.effect
    @reactive.event(input.btn_goto_import)
    def _goto_import():
        ui.update_navs("main_nav", selected="Data Import")

    @reactive.effect
    @reactive.event(input.btn_start_ecopath, input.btn_goto_ecopath)
    def _goto_ecopath():
        ui.update_navs("main_nav", selected="Ecopath Model")

    @reactive.effect
    @reactive.event(input.btn_goto_ecosim)
    def _goto_ecosim():
        ui.update_navs("main_nav", selected="Ecosim")

    @reactive.effect
    @reactive.event(input.btn_goto_ecospace)
    def _goto_ecospace():
        ui.update_navs("main_nav", selected="Ecospace")

    @reactive.effect
    @reactive.event(input.btn_goto_ibm)
    def _goto_ibm():
        ui.update_navs("main_nav", selected="Individual-Based Model")

    @reactive.effect
    @reactive.event(input.btn_goto_analysis)
    def _goto_analysis():
        ui.update_navs("main_nav", selected="Analysis")

    @reactive.effect
    @reactive.event(input.btn_goto_results)
    def _goto_results():
        ui.update_navs("main_nav", selected="Results")

    @reactive.effect
    @reactive.event(input.btn_goto_about)
    def _goto_about():
        ui.update_navs("main_nav", selected="About")

    @reactive.effect
    @reactive.event(input.btn_load_example)
    def _load_example_model():
        """Load an example marine ecosystem model."""
        try:
            params = build_example_params()

            # Validate the example model can balance (result intentionally discarded;
            # the ecopath page needs editable params, not the balanced model).
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rpath(params)
            model_data.set(params)

            _notify_legacy(
                "Example model loaded with multi-stanza groups (Roundfish) and sample remarks! Navigate to Ecopath tab or explore Advanced Features.",
                type="message",
                duration=7,
            )

            # Navigate to Ecopath tab
            ui.update_navs("main_nav", selected="Ecopath Model")

        except Exception as e:
            logger.exception("Loading the example model failed")
            _notify_legacy(f"Error loading example model: {str(e)}", type="error")
