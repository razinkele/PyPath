"""Ecospace Data Wizard — 7-step guided ecospace model creation.

Steps:
1. Select Area — draw polygon on map
2. Configure Grid — choose grid type and resolution
3. Download Data — fetch EMODnet habitats and bathymetry
4. Review Habitats — inspect and merge EUNIS categories
5. Assign Preferences — semi-auto habitat preferences per species group
6. Set Dispersal — per-group dispersal parameters
7. Review & Launch — summary and build EcospaceParams
"""

import logging

from shiny import Inputs, Outputs, Session, reactive, render, ui

logger = logging.getLogger(__name__)

_STEPS = [
    "Select Area",
    "Configure Grid",
    "Download Data",
    "Review Habitats",
    "Assign Preferences",
    "Set Dispersal",
    "Review & Launch",
]


def _step_progress_ui():
    """Render the step progress bar."""
    items = []
    for i, label in enumerate(_STEPS, 1):
        items.append(
            ui.span(
                f"{i}. {label}",
                class_="badge bg-secondary me-1",
                id=f"wizard_step_badge_{i}",
            )
        )
    return ui.div(*items, class_="mb-3")


def ecospace_wizard_ui():
    """Wizard page UI."""
    return ui.page_fluid(
        ui.h3("Ecospace Data Wizard"),
        _step_progress_ui(),
        ui.output_ui("wizard_step_content"),
        ui.div(
            ui.input_action_button("wizard_back", "Back", class_="btn-secondary me-2"),
            ui.input_action_button("wizard_next", "Next", class_="btn-primary"),
            class_="mt-3",
        ),
    )


def ecospace_wizard_server(
    input: Inputs, output: Outputs, session: Session, shared_data=None
):
    """Wizard page server logic."""
    wizard_step = reactive.value(1)

    @reactive.effect
    @reactive.event(input.wizard_next)
    def _next():
        current = wizard_step.get()
        if current < len(_STEPS):
            wizard_step.set(current + 1)

    @reactive.effect
    @reactive.event(input.wizard_back)
    def _back():
        current = wizard_step.get()
        if current > 1:
            wizard_step.set(current - 1)

    @render.ui
    def wizard_step_content():
        step = wizard_step.get()
        if step == 1:
            return _step1_select_area_ui()
        elif step == 2:
            return _step2_configure_grid_ui()
        elif step == 3:
            return _step3_download_data_ui()
        elif step == 4:
            return _step4_review_habitats_ui()
        elif step == 5:
            return _step5_assign_preferences_ui()
        elif step == 6:
            return _step6_set_dispersal_ui()
        elif step == 7:
            return _step7_review_launch_ui()
        return ui.p("Unknown step")


def _step1_select_area_ui():
    return ui.card(
        ui.card_header("Step 1: Select Study Area"),
        ui.p("Draw a polygon on the map to define your study area."),
        ui.output_ui("wizard_map"),
    )


def _step2_configure_grid_ui():
    return ui.card(
        ui.card_header("Step 2: Configure Grid"),
        ui.input_radio_buttons(
            "wizard_grid_type",
            "Grid Type",
            choices={"regular": "Regular Rectangular", "hexagonal": "Hexagonal"},
            selected="regular",
        ),
        ui.input_numeric(
            "wizard_cell_size", "Cell Size (km)", value=5, min=0.5, max=100
        ),
    )


def _step3_download_data_ui():
    return ui.card(
        ui.card_header("Step 3: Download Data"),
        ui.p("Download EMODnet seabed habitats and bathymetry for your study area."),
        ui.input_action_button(
            "wizard_download", "Download Data", class_="btn-primary"
        ),
        ui.output_ui("wizard_download_status"),
        ui.hr(),
        ui.p("Salinity (optional):"),
        ui.input_file(
            "wizard_salinity_file",
            "Upload salinity file (CSV or NetCDF)",
            accept=[".csv", ".nc", ".nc4"],
        ),
    )


def _step4_review_habitats_ui():
    return ui.card(
        ui.card_header("Step 4: Review Habitats"),
        ui.p("Review EUNIS habitat types assigned to each grid patch."),
        ui.output_ui("wizard_habitat_map"),
        ui.output_table("wizard_habitat_table"),
    )


def _step5_assign_preferences_ui():
    return ui.card(
        ui.card_header("Step 5: Assign Habitat Preferences"),
        ui.input_select(
            "wizard_preset",
            "Quick Preset",
            choices={
                "none": "-- Manual --",
                "pelagic": "Pelagic",
                "demersal": "Demersal",
                "benthic": "Benthic",
                "auto": "Auto-suggest (biodata)",
            },
        ),
        ui.output_ui("wizard_preference_editor"),
    )


def _step6_set_dispersal_ui():
    return ui.card(
        ui.card_header("Step 6: Set Dispersal Parameters"),
        ui.input_slider(
            "wizard_dispersal_default",
            "Default Dispersal Rate (km\u00b2/month)",
            min=0.0,
            max=100.0,
            value=10.0,
            step=0.5,
        ),
        ui.input_slider(
            "wizard_gravity",
            "Gravity Strength",
            min=0.0,
            max=1.0,
            value=0.3,
            step=0.05,
        ),
        ui.output_ui("wizard_dispersal_table"),
    )


def _step7_review_launch_ui():
    return ui.card(
        ui.card_header("Step 7: Review & Launch"),
        ui.output_ui("wizard_summary"),
        ui.input_action_button(
            "wizard_create",
            "Create Ecospace Model",
            class_="btn-success btn-lg",
        ),
    )
