"""Home page module."""

import logging
import warnings

import pandas as pd
from pypath.core.ecopath import rpath
from pypath.core.params import create_rpath_params
from shiny import Inputs, Outputs, Session, reactive, ui

from pypath_shiny.components import feature_card, quick_start_step
from pypath_shiny.config import CHANGELOG_URL, DEFAULTS, DOCS_URL

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


def example_choices() -> dict:
    """The shipped example models for the Home and the Ecosim page pickers."""
    from pypath.examples import list_example_models

    choices = {"": "Open a real model..."}
    choices.update({m.name: m.title for m in list_example_models()})
    return choices


#: What a model opened from this page still needs. Home hands it over
#: unbalanced, for the Ecopath page to show and edit; until it is balanced,
#: Ecosim refuses it and Analysis and Results have nothing to draw.
_NOT_BALANCED_YET = (
    "It is not balanced yet: click Balance Model on the Ecopath Model page "
    "before Ecosim, Analysis or Results."
)

EXAMPLE_LOADED_NOTICE = (
    "Example model loaded, with multi-stanza groups (Roundfish) and sample "
    "remarks. " + _NOT_BALANCED_YET
)


def example_opened_notice(name: str) -> str:
    """The notice for a shipped example opened with the picker."""
    return (
        f"{name} opened. {_NOT_BALANCED_YET} Its Ecosim settings are the "
        "starting scenario on the Ecosim page."
    )


def open_example_model(name: str):
    """A shipped example model, checked to balance, and its Ecosim source.

    Returns
    -------
    (params, record)
        The model's RpathParams and the record the Ecosim page uses to offer
        the model's own EwE-F Ecosim settings.
    """
    from pypath.examples import load_example_model

    from pypath_shiny.pages.ewe_scenario import example_source_record

    params = load_example_model(name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rpath(params)  # raises if the model cannot be balanced
    return params, example_source_record(name, params)


def publish_example_model(
    name: str,
    *,
    model_data,
    ewe_source=None,
    sim_results=None,
    balance_request=None,
):
    """Open a shipped example and put it on the app's shared state.

    The Home page and the Ecosim page open examples through this, so both
    drop the previous model's simulation and record the example's Ecosim
    source the same way. With ``balance_request`` the model is handed to the
    Ecopath page to adopt and balance in one step (the Ecosim page's
    route); without it the unbalanced model is published, for the user to
    look at and balance on the Ecopath page (the Home page's route).

    Returns the source record. Raises whatever reading or balancing the
    example raises; nothing is published then.
    """
    params, record = open_example_model(name)
    if sim_results is not None:
        sim_results.set(None)
    if ewe_source is not None:
        ewe_source.set(record)
    if balance_request is not None:
        balance_request.set(params)
    else:
        model_data.set(params)
    return record


def _news_item(
    icon_class: str, title: str, text: str, pointer: str, *, label: str = "Where"
) -> ui.Tag:
    """One entry of the Home page's "Recent Additions" card.

    ``pointer`` says where in the app the addition is found; ``label`` is the
    word in front of it.
    """
    return ui.div(
        ui.h5(ui.tags.i(class_=f"bi {icon_class} me-2"), title, class_="mb-2"),
        ui.p(text, class_="mb-1"),
        ui.p(ui.tags.strong(f"{label}: "), pointer, class_="text-muted small mb-0"),
    )


def home_ui():
    """Home page UI."""
    return ui.div(
        ui.div(
            # Hero section — ocean gradient with caustic light effects
            ui.div(
                ui.p("Ecosystem Modeling Framework", class_="hero-tagline"),
                ui.h1("Ecopath with Ecosim in Python", class_="hero-title"),
                ui.p(
                    "Build mass-balance food web models, run them through time, "
                    "and explore them in space on grids, polygons or a "
                    "hydrodynamic model's mesh \u2014 all from your browser.",
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
                    ui.div(
                        ui.input_select(
                            "example_model",
                            None,
                            example_choices(),
                            selected="",
                        ),
                        ui.input_action_button(
                            "btn_load_example_model",
                            ui.TagList(
                                ui.tags.i(class_="bi bi-folder2-open me-2"),
                                "Open",
                            ),
                            class_="btn-hero-secondary",
                        ),
                        class_="hero-example-picker",
                    ),
                    class_="hero-actions",
                ),
                class_="hero-section",
            ),
            # Recent additions
            ui.div(
                ui.div(
                    ui.h2("Recent Additions", class_="home-section-title"),
                    ui.tags.a(
                        "Full changelog",
                        ui.tags.i(
                            class_="bi bi-box-arrow-up-right ms-1", aria_hidden="true"
                        ),
                        href=CHANGELOG_URL,
                        target="_blank",
                        rel="noopener noreferrer",
                        class_="home-section-link",
                    ),
                    class_="home-section-head",
                ),
                ui.card(
                    ui.card_body(
                        ui.layout_columns(
                            _news_item(
                                "bi-water",
                                "Hydrodynamic Model Link",
                                "A SHYFEM mesh can be the Ecospace grid. A box "
                                "model's output, as text or as a monthly box file, "
                                "shows its fields on the grid and lets its water "
                                "flows carry chosen groups.",
                                "Ecospace page, grid type SHYFEM mesh and the "
                                "Hydrodynamic Model panel.",
                            ),
                            _news_item(
                                "bi-folder2-open",
                                "Real Models to Open",
                                "The example models are real food webs. Each opens "
                                "with its own Ecosim settings, and the Curonian "
                                "Lagoon one brings its Ecospace set-up. An EwE "
                                "database's stored Ecospace scenario loads too.",
                                "The picker in the banner above, the Ecosim page "
                                "and the Ecospace page.",
                            ),
                            _news_item(
                                "bi-shuffle",
                                "Prey Switching",
                                "Predators turn towards prey that become more "
                                "abundant: EwE's switching power, set per predator "
                                "on the Ecosim page.",
                                "The Prey Switching demo page shows how the "
                                "switching power shapes a predator's diet.",
                                label="Explore",
                            ),
                            _news_item(
                                "bi-hourglass-split",
                                "Runs That Do Not Block",
                                "Ecospace and IBM runs work in the background. An "
                                "Ecospace run gives a run-time estimate, shows the "
                                "month it has reached and can be cancelled.",
                                "Ecospace and Individual-Based Model pages.",
                            ),
                            col_widths={"sm": 12, "md": 6, "xl": 3},
                        )
                    ),
                    class_="border-success home-news",
                ),
                class_="home-section",
            ),
            # Feature cards: one per page, in the order of the workflow
            ui.div(
                ui.h2("Features", class_="home-section-title"),
                ui.layout_columns(
                    feature_card(
                        "Data Import",
                        [
                            "Published models from the EcoBase repository",
                            "EwE 6 database files (.ewemdb, .eweaccdb, .mdb, .accdb)",
                            "SQLite copies of EwE databases",
                            "Species data from WoRMS, OBIS, FishBase and SeaLifeBase",
                            "Groups, diet and stanzas shown before the model is used",
                        ],
                        icon_class="bi-cloud-download",
                        button_label="Open Data Import",
                        button_id="btn_goto_import",
                    ),
                    feature_card(
                        "Ecopath Model",
                        [
                            "Functional groups and the diet matrix in editable tables",
                            "Biomass, P/B, Q/B, EE and catches by fleet",
                            "Mass balance solves each group's missing parameter",
                            "Trophic levels",
                            "Multi-stanza age structure",
                        ],
                        icon_class="bi-diagram-3",
                        button_label="Open Ecopath Model",
                        button_id="btn_goto_ecopath",
                    ),
                    feature_card(
                        "Pre-Balance",
                        [
                            "Checks of the inputs before the model is balanced",
                            "Biomass across trophic levels",
                            "Predator-prey ratios",
                            "Vital rate ratios",
                            "Warnings listed group by group",
                        ],
                        icon_class="bi-clipboard-check",
                        button_label="Open Pre-Balance",
                        button_id="btn_goto_prebalance",
                    ),
                    feature_card(
                        "Ecosim",
                        [
                            "Foraging arena functional response",
                            "Runge-Kutta 4, Adams-Bashforth, or the continuous "
                            "solvers LSODA, BDF, DOP853 and Radau",
                            "EwE 6 or Rpath equation rules, set by the model's origin",
                            "An EwE database's own scenario and time series",
                            "Prey switching (EwE's switching power)",
                        ],
                        icon_class="bi-graph-up",
                        button_label="Open Ecosim",
                        button_id="btn_goto_ecosim",
                    ),
                    feature_card(
                        "Ecospace",
                        [
                            "Regular grids, transects, uploaded polygons, "
                            "hexagons or a SHYFEM mesh",
                            "Habitat preferences, dispersal and spatial fishing",
                            "Fields and water transport of a hydrodynamic box model",
                            "Stored Ecospace scenarios of EwE databases",
                            "A wizard that builds a new set-up step by step",
                        ],
                        icon_class="bi-geo-alt",
                        button_label="Open Ecospace",
                        button_id="btn_goto_ecospace",
                    ),
                    feature_card(
                        "Individual-Based Model",
                        [
                            "One group followed as super-individuals inside Ecosim",
                            "Wisconsin bioenergetics model",
                            "Size-structured predation",
                            "Stochastic reproduction",
                            "Comparison with the standard Ecosim run",
                        ],
                        icon_class="bi-cpu",
                        button_label="Open Individual-Based Model",
                        button_id="btn_goto_ibm",
                    ),
                    feature_card(
                        "Analysis",
                        [
                            "System and flow indices",
                            "Trophic levels and the trophic spectrum",
                            "Mixed trophic impacts (MTI)",
                            "Keystoneness of each group",
                            "Balance check and CSV export",
                        ],
                        icon_class="bi-diagram-2",
                        button_label="Open Analysis",
                        button_id="btn_goto_analysis",
                    ),
                    feature_card(
                        "Results",
                        [
                            "Model summary and data tables",
                            "Food web diagram and interactive network",
                            "Biomass and catch through time",
                            "Scenario comparison",
                            "CSV downloads of the model and the simulation",
                        ],
                        icon_class="bi-bar-chart",
                        button_label="Open Results",
                        button_id="btn_goto_results",
                    ),
                    feature_card(
                        "Tutorial and Documentation",
                        [
                            "A walkthrough from an empty model to fishing scenarios",
                            "Code for every step, to run with the pypath library",
                            "API reference of the library",
                            "Worked examples: EwE databases, spatial models, "
                            "the hydrodynamic link, the IBM",
                        ],
                        icon_class="bi-mortarboard",
                        button_label="Open Tutorial",
                        button_id="btn_goto_tutorial",
                        link_label="Documentation",
                        link_href=DOCS_URL,
                    ),
                    col_widths={"sm": 12, "md": 6, "xl": 4},
                ),
                class_="home-section",
            ),
            # Quick start section
            ui.div(
                ui.h2("Quick Start", class_="home-section-title"),
                ui.layout_columns(
                    quick_start_step(
                        "Step 1: Define Groups",
                        "Name the functional groups of the food web: producers, "
                        "consumers, detritus and fishing fleets.",
                        "create_rpath_params(groups, types)",
                    ),
                    quick_start_step(
                        "Step 2: Set Parameters",
                        "Enter biomass, P/B, Q/B, the diet matrix and the catches "
                        "of each group.",
                        'params.model["Biomass"] = ...',
                    ),
                    quick_start_step(
                        "Step 3: Balance the Model",
                        "Ecopath solves the missing parameter of each group and "
                        "reports the balance.",
                        "model = rpath(params)",
                    ),
                    quick_start_step(
                        "Step 4: Run a Simulation",
                        "Build an Ecosim scenario from the balanced model and run "
                        "it through time.",
                        "sim = rsim_scenario(model, params)\nout = rsim_run(sim)",
                    ),
                    quick_start_step(
                        "Step 5: Add Space",
                        "Give the scenario a grid, habitat preferences and "
                        "dispersal, and run it as Ecospace.",
                        "rsim_run_spatial(\n    sim, ecospace=ecospace)",
                        badge="Optional",
                    ),
                    quick_start_step(
                        "Step 6: Follow Individuals",
                        "Hand a group to the individual-based model: Ecosim then "
                        "takes that group's change from its super-individuals.",
                        "sim.params.ibm_groups = {3: smelt}",
                        badge="Optional",
                    ),
                    col_widths={"sm": 12, "md": 6, "xl": 4},
                ),
                class_="home-section",
            ),
            class_="container home-page",
        )
    )


def home_server(
    input: Inputs,
    output: Outputs,
    session: Session,
    model_data: reactive.Value,
    ewe_source: reactive.Value = None,
    sim_results: reactive.Value = None,
):
    """Home page server logic.

    ``ewe_source``, when given, receives the record of an example model
    opened from the picker, so the Ecosim page offers its Ecosim settings.
    ``sim_results``, when given, is cleared whenever a model is loaded here,
    so no page shows the previous model's simulation under the new groups.
    """

    @reactive.effect
    @reactive.event(input.btn_load_example_model)
    def _open_example_model():
        """Open the example model chosen in the picker."""
        name = input.example_model()
        if not name:
            _notify_legacy("Choose a model to open first.", type="warning")
            return
        try:
            record = publish_example_model(
                name,
                model_data=model_data,
                ewe_source=ewe_source,
                sim_results=sim_results,
            )
        except Exception as e:  # report any reader or balance failure
            logger.exception("Opening the example model %s failed", name)
            _notify_legacy(f"Could not open {name}: {e}", type="error")
            return
        _notify_legacy(
            example_opened_notice(record["name"]),
            type="message",
            duration=12,
        )
        ui.update_navset("main_nav", selected="Ecopath Model")

    @reactive.effect
    @reactive.event(input.btn_goto_import)
    def _goto_import():
        ui.update_navset("main_nav", selected="Data Import")

    @reactive.effect
    @reactive.event(input.btn_start_ecopath, input.btn_goto_ecopath)
    def _goto_ecopath():
        ui.update_navset("main_nav", selected="Ecopath Model")

    @reactive.effect
    @reactive.event(input.btn_goto_ecosim)
    def _goto_ecosim():
        ui.update_navset("main_nav", selected="Ecosim")

    @reactive.effect
    @reactive.event(input.btn_goto_ecospace)
    def _goto_ecospace():
        ui.update_navset("main_nav", selected="Ecospace")

    @reactive.effect
    @reactive.event(input.btn_goto_ibm)
    def _goto_ibm():
        ui.update_navset("main_nav", selected="Individual-Based Model")

    @reactive.effect
    @reactive.event(input.btn_goto_analysis)
    def _goto_analysis():
        ui.update_navset("main_nav", selected="Analysis")

    @reactive.effect
    @reactive.event(input.btn_goto_results)
    def _goto_results():
        ui.update_navset("main_nav", selected="Results")

    @reactive.effect
    @reactive.event(input.btn_goto_prebalance)
    def _goto_prebalance():
        ui.update_navset("main_nav", selected="Pre-Balance")

    @reactive.effect
    @reactive.event(input.btn_goto_tutorial)
    def _goto_tutorial():
        ui.update_navset("main_nav", selected="Tutorial")

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
            if sim_results is not None:
                sim_results.set(None)

            _notify_legacy(EXAMPLE_LOADED_NOTICE, type="message", duration=12)

            # Navigate to Ecopath tab
            ui.update_navset("main_nav", selected="Ecopath Model")

        except Exception as e:
            logger.exception("Loading the example model failed")
            _notify_legacy(f"Error loading example model: {str(e)}", type="error")
