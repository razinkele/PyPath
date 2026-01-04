"""Data Import page module - EcoBase and EwE database import."""

import pandas as pd
from shiny import Inputs, Outputs, Session, reactive, render, ui

from pypath.io.biodata import (
    APIConnectionError,
    SpeciesNotFoundError,
    batch_get_species_info,
    biodata_to_rpath,
)

# pypath imports (path setup handled by app/__init__.py)
from pypath.io.ecobase import (
    ecobase_to_rpath,
    get_ecobase_model,
    list_ecobase_models,
    search_ecobase_models,
)
from pypath.io.ewemdb import (
    EwEDatabaseError,
    check_ewemdb_support,
    get_ewemdb_metadata,
    read_ewemdb,
)

# Import shared utilities
from .utils import (
    create_cell_styles,
    format_dataframe_for_display,
)

# Configuration imports
try:
    from app.config import PARAM_RANGES, UI
except ModuleNotFoundError:
    from config import PARAM_RANGES, UI


def import_ui():
    """Data import page UI."""
    return ui.page_sidebar(
        # Sidebar with import options (tabs)
        ui.sidebar(
            ui.navset_pill(
                # EcoBase tab
                ui.nav_panel(
                    "EcoBase",
                    ui.div(
                        ui.p(
                            "Download models from ",
                            ui.tags.a(
                                "EcoBase",
                                href="http://ecobase.ecopath.org/",
                                target="_blank",
                            ),
                            class_="small text-muted",
                        ),
                        # Search section
                        ui.input_text(
                            "ecobase_search",
                            "Search",
                            placeholder="e.g., Baltic, coral",
                        ),
                        ui.input_select(
                            "ecobase_ecosystem",
                            "Ecosystem Type",
                            choices={
                                "": "All Types",
                                "marine": "Marine",
                                "freshwater": "Freshwater",
                                "estuarine": "Estuarine",
                            },
                        ),
                        ui.div(
                            ui.input_action_button(
                                "btn_search_ecobase",
                                ui.tags.span(
                                    ui.tags.i(class_="bi bi-search me-1"), "Search"
                                ),
                                class_="btn-primary btn-sm",
                            ),
                            ui.input_action_button(
                                "btn_list_all",
                                "All Models",
                                class_="btn-outline-secondary btn-sm ms-1",
                            ),
                            class_="mb-3",
                        ),
                        ui.tags.hr(),
                        # Results table
                        ui.h6("Available Models"),
                        ui.p(
                            "Click a row to select, then download.",
                            class_="small text-muted",
                        ),
                        ui.output_data_frame("ecobase_models_table"),
                        ui.tags.hr(),
                        # Selected model info and download
                        ui.output_ui("ecobase_selected_info"),
                        ui.tags.hr(),
                        # Use imported model button
                        ui.output_ui("use_model_button_ecobase"),
                        class_="mt-2",
                    ),
                ),
                # EwE File tab
                ui.nav_panel(
                    "EwE File",
                    ui.div(
                        ui.p(
                            "Import from EwE 6.x database files (.ewemdb, .eweaccdb, .mdb, .accdb)",
                            class_="small text-muted",
                        ),
                        # Check driver support
                        ui.output_ui("ewemdb_support_status"),
                        ui.tags.hr(),
                        # File upload
                        ui.input_file(
                            "ewemdb_upload",
                            "Select file",
                            accept=[".ewemdb", ".eweaccdb", ".mdb", ".accdb", ".ewe"],
                            multiple=False,
                        ),
                        ui.input_numeric(
                            "ewemdb_scenario", "Scenario Number", value=1, min=1
                        ),
                        ui.input_action_button(
                            "btn_import_ewemdb",
                            ui.tags.span(
                                ui.tags.i(class_="bi bi-upload me-1"), "Import Model"
                            ),
                            class_="btn-success mt-2",
                        ),
                        ui.tags.hr(),
                        # Metadata preview
                        ui.h6("File Information"),
                        ui.output_ui("ewemdb_metadata_ui"),
                        ui.tags.hr(),
                        # Use imported model button
                        ui.output_ui("use_model_button_ewe"),
                        class_="mt-2",
                    ),
                ),
                # Biodiversity Data tab
                ui.nav_panel(
                    "Biodiversity",
                    ui.div(
                        ui.p(
                            "Build models from global biodiversity databases: ",
                            ui.tags.a(
                                "WoRMS",
                                href="https://www.marinespecies.org/",
                                target="_blank",
                            ),
                            ", ",
                            ui.tags.a(
                                "OBIS", href="https://obis.org/", target="_blank"
                            ),
                            ", ",
                            ui.tags.a(
                                "FishBase",
                                href="https://www.fishbase.org/",
                                target="_blank",
                            ),
                            class_="small text-muted",
                        ),
                        # Example data button
                        ui.input_action_button(
                            "btn_load_example_species",
                            ui.tags.span(
                                ui.tags.i(class_="bi bi-file-earmark-text me-1"),
                                "Load Example",
                            ),
                            class_="btn-outline-secondary btn-sm mb-2",
                        ),
                        # Species list input
                        ui.input_text_area(
                            "biodata_species_list",
                            "Species List (one per line, common names)",
                            placeholder="Atlantic cod\nAtlantic herring\nEuropean sprat\nZooplankton\nPhytoplankton",
                            rows=UI.textarea_rows_default,
                            resize="vertical",
                        ),
                        # Model area
                        ui.input_numeric(
                            "biodata_area",
                            "Model Area (km²)",
                            value=1000,
                            min=1,
                            step=100,
                        ),
                        # Options
                        ui.input_checkbox(
                            "biodata_include_occurrences",
                            "Include OBIS occurrence data",
                            value=True,
                        ),
                        ui.input_checkbox(
                            "biodata_include_traits",
                            "Include FishBase trait data",
                            value=True,
                        ),
                        ui.tags.hr(),
                        # Fetch data button
                        ui.input_action_button(
                            "btn_fetch_biodata",
                            ui.tags.span(
                                ui.tags.i(class_="bi bi-cloud-download me-1"),
                                "Fetch Species Data",
                            ),
                            class_="btn-primary w-100 mb-2",
                        ),
                        # Progress and status
                        ui.output_ui("biodata_fetch_status"),
                        ui.tags.hr(),
                        # Results preview
                        ui.h6("Fetched Species Data"),
                        ui.output_data_frame("biodata_results_table"),
                        ui.tags.hr(),
                        # Biomass estimates section
                        ui.output_ui("biodata_biomass_section"),
                        ui.tags.hr(),
                        # Create model button
                        ui.output_ui("biodata_create_button"),
                        # Use imported model button
                        ui.output_ui("use_model_button_biodata"),
                        class_="mt-2",
                    ),
                ),
                id="import_tabs",
            ),
            width=400,
            title="Import Source",
        ),
        # Main content - Preview pane
        ui.h3("Model Preview", class_="mb-3"),
        ui.output_ui("import_preview_status"),
        ui.navset_card_tab(
            ui.nav_panel(
                "Groups",
                ui.output_data_frame("imported_groups_table"),
            ),
            ui.nav_panel(
                "Diet Matrix",
                ui.output_data_frame("imported_diet_table"),
            ),
            ui.nav_panel(
                "Multi-Stanza",
                ui.output_ui("imported_stanza_status"),
                ui.h6("Stanza Groups", class_="mt-3"),
                ui.output_data_frame("imported_stanza_groups_table"),
                ui.h6("Life Stages", class_="mt-3"),
                ui.output_data_frame("imported_stanza_indiv_table"),
            ),
            ui.nav_panel(
                "Summary",
                ui.output_ui("imported_summary"),
            ),
        ),
        title="Import Ecopath Models",
        fillable=True,
    )


def import_server(
    input: Inputs, output: Outputs, session: Session, model_data: reactive.Value
):
    """Data import page server logic."""

    # Reactive values
    ecobase_models = reactive.Value(None)
    imported_params = reactive.Value(None)
    selected_model_id = reactive.Value(None)

    # === EcoBase Functions ===

    @reactive.effect
    @reactive.event(input.btn_list_all)
    def _list_all_models():
        """List all models from EcoBase."""
        try:
            ui.notification_show("Fetching models from EcoBase...", duration=3)
            models_df = list_ecobase_models()
            ecobase_models.set(models_df)
            selected_model_id.set(None)
            ui.notification_show(
                f"Found {len(models_df)} public models", type="message"
            )
        except Exception as e:
            ui.notification_show(f"Error: {str(e)}", type="error")

    @reactive.effect
    @reactive.event(input.btn_search_ecobase)
    def _search_models():
        """Search EcoBase models."""
        try:
            search_term = input.ecobase_search()
            ecosystem = input.ecobase_ecosystem()

            if not search_term and not ecosystem:
                ui.notification_show(
                    "Enter a search term or select ecosystem type", type="warning"
                )
                return

            ui.notification_show("Searching EcoBase...", duration=3)

            # Get all models first, then filter
            all_models = list_ecobase_models()

            if search_term:
                results = search_ecobase_models(search_term, models_df=all_models)
            else:
                results = all_models.copy()

            # Reset index before filtering to avoid alignment issues
            results = results.reset_index(drop=True)

            if ecosystem:
                mask = (
                    results["ecosystem_type"]
                    .str.lower()
                    .str.contains(ecosystem.lower(), na=False)
                )
                results = results[mask].reset_index(drop=True)

            ecobase_models.set(results)
            selected_model_id.set(None)
            ui.notification_show(
                f"Found {len(results)} matching models", type="message"
            )
        except Exception as e:
            ui.notification_show(f"Error: {str(e)}", type="error")

    @output
    @render.data_frame
    def ecobase_models_table():
        """Render EcoBase models as data frame."""
        models = ecobase_models.get()
        if models is None or len(models) == 0:
            return render.DataGrid(
                pd.DataFrame(
                    {"Message": ["Click 'All Models' or search to load models"]}
                )
            )

        display_cols = [
            "model_number",
            "model_name",
            "country",
            "ecosystem_type",
            "num_groups",
        ]
        display_cols = [c for c in display_cols if c in models.columns]
        return render.DataGrid(
            models[display_cols].head(100),
            selection_mode="row",
            height=UI.datagrid_height_default_px,
        )

    @reactive.effect
    def _update_selected_model():
        """Update selected model when row is clicked."""
        models = ecobase_models.get()
        selected_rows = input.ecobase_models_table_selected_rows()

        if models is not None and selected_rows and len(selected_rows) > 0:
            row_idx = selected_rows[0]
            if row_idx < len(models):
                model_id = models.iloc[row_idx]["model_number"]
                selected_model_id.set(int(model_id))

    @output
    @render.ui
    def ecobase_selected_info():
        """Show selected model info and download button."""
        model_id = selected_model_id.get()
        models = ecobase_models.get()

        if model_id is None or models is None:
            return ui.p(
                "Select a model from the table above", class_="text-muted small"
            )

        # Find model info
        model_row = models[models["model_number"] == model_id]
        if len(model_row) == 0:
            return ui.p(
                "Select a model from the table above", class_="text-muted small"
            )

        model_name = model_row.iloc[0].get("model_name", f"Model {model_id}")

        return ui.div(
            ui.div(
                ui.tags.strong("Selected: "),
                f"{model_name} (ID: {model_id})",
                class_="mb-2",
            ),
            ui.input_action_button(
                "btn_download_ecobase",
                ui.tags.span(
                    ui.tags.i(class_="bi bi-cloud-download me-1"),
                    "Download Selected Model",
                ),
                class_="btn-success w-100",
            ),
            class_="p-2 bg-light rounded",
        )

    @reactive.effect
    @reactive.event(input.btn_download_ecobase)
    def _download_ecobase():
        """Download and import model from EcoBase."""
        try:
            model_id = selected_model_id.get()
            if not model_id:
                ui.notification_show("Select a model first", type="warning")
                return

            ui.notification_show(f"Downloading model {model_id}...", duration=5)

            # Download model data
            model_data_dict = get_ecobase_model(int(model_id))

            # Convert to RpathParams
            params = ecobase_to_rpath(model_data_dict)

            # Debug: count diet values
            diet_count = 0
            for col in params.diet.columns:
                if col != "Group":
                    for idx in params.diet.index:
                        val = params.diet.at[idx, col]
                        if pd.notna(val) and val > 0:
                            diet_count += 1

            imported_params.set(params)

            ui.notification_show(
                f"Downloaded model: {len(params.model)} groups, {diet_count} diet entries",
                type="message",
            )
        except Exception as e:
            ui.notification_show(f"Download error: {str(e)}", type="error")

    # === EwE Database Functions ===

    @output
    @render.ui
    def ewemdb_support_status():
        """Check and display ewemdb driver support."""
        support = check_ewemdb_support()

        if support["any_available"]:
            drivers = []
            if support["pyodbc"]:
                drivers.append("pyodbc")
            if support["pypyodbc"]:
                drivers.append("pypyodbc")
            if support["mdb_tools"]:
                drivers.append("mdb-tools")

            return ui.div(
                ui.tags.i(class_="bi bi-check-circle-fill text-success me-2"),
                f"Database drivers available: {', '.join(drivers)}",
                class_="alert alert-success",
            )
        else:
            return ui.div(
                ui.tags.i(class_="bi bi-exclamation-triangle-fill text-warning me-2"),
                "No database drivers found. Install pyodbc or mdb-tools to read .ewemdb files.",
                class_="alert alert-warning",
            )

    @reactive.effect
    @reactive.event(input.btn_import_ewemdb)
    def _import_ewemdb():
        """Import EwE database file."""
        file_info = input.ewemdb_upload()
        if not file_info:
            ui.notification_show("Select a file first", type="warning")
            return

        try:
            filepath = file_info[0]["datapath"]
            scenario = input.ewemdb_scenario() or 1

            ui.notification_show("Importing EwE database...", duration=5)

            params = read_ewemdb(filepath, scenario=scenario)
            imported_params.set(params)

            # Debug: Check for remarks
            has_remarks = hasattr(params, "remarks") and params.remarks is not None
            remarks_info = ""
            if has_remarks:
                # Count non-empty remarks
                non_empty_count = 0
                for col in params.remarks.columns:
                    if col != "Group":
                        non_empty_count += sum(
                            1 for v in params.remarks[col] if str(v).strip()
                        )
                remarks_info = f", {non_empty_count} remarks"

            # Check for stanza data
            stanza_info = ""
            if (
                hasattr(params, "stanzas")
                and params.stanzas is not None
                and params.stanzas.n_stanza_groups > 0
            ):
                n_stanza = params.stanzas.n_stanza_groups
                n_stages = (
                    len(params.stanzas.stindiv)
                    if params.stanzas.stindiv is not None
                    else 0
                )
                stanza_info = f", {n_stanza} stanza group(s)"

            ui.notification_show(
                f"Imported model with {len(params.model)} groups{remarks_info}{stanza_info}",
                type="message",
            )
        except EwEDatabaseError as e:
            ui.notification_show(f"Database error: {str(e)}", type="error")
        except Exception as e:
            ui.notification_show(f"Import error: {str(e)}", type="error")

    @output
    @render.ui
    def ewemdb_metadata_ui():
        """Show EwE file metadata."""
        file_info = input.ewemdb_upload()
        if not file_info:
            return ui.p("Upload a file to see metadata.", class_="text-muted")

        try:
            filepath = file_info[0]["datapath"]
            metadata = get_ewemdb_metadata(filepath)

            # Build ecosim/ecospace indicators
            ecosim_badge = ""
            if metadata.get("has_ecosim"):
                n_scen = metadata.get("num_scenarios", 0)
                ecosim_badge = ui.tags.span(
                    f"Ecosim ({n_scen} scenarios)", class_="badge bg-success me-1"
                )

            ecospace_badge = ""
            if metadata.get("has_ecospace"):
                ecospace_badge = ui.tags.span("Ecospace", class_="badge bg-info")

            return ui.div(
                ui.tags.table(
                    ui.tags.tr(
                        ui.tags.td("Name:"), ui.tags.td(metadata.get("name", "N/A"))
                    ),
                    ui.tags.tr(
                        ui.tags.td("Author:"), ui.tags.td(metadata.get("author", "N/A"))
                    ),
                    ui.tags.tr(
                        ui.tags.td("Groups:"),
                        ui.tags.td(str(metadata.get("num_groups", "N/A"))),
                    ),
                    ui.tags.tr(
                        ui.tags.td("Fleets:"),
                        ui.tags.td(str(metadata.get("num_fleets", "N/A"))),
                    ),
                    ui.tags.tr(
                        ui.tags.td("Contains:"),
                        ui.tags.td(
                            ecosim_badge,
                            ecospace_badge if ecospace_badge else "Ecopath only",
                        ),
                    ),
                    class_="table table-sm",
                )
            )
        except Exception as e:
            return ui.p(f"Could not read metadata: {str(e)}", class_="text-warning")

    # === Imported Model Preview ===

    @output
    @render.ui
    def import_preview_status():
        """Show import status."""
        params = imported_params.get()
        if params is None:
            return ui.div(
                ui.tags.i(class_="bi bi-info-circle me-2"),
                "No model imported yet. Use EcoBase or upload an EwE database file above.",
                class_="alert alert-info",
            )

        n_groups = len(params.model)
        n_living = len(params.model[params.model["Type"] <= 1])
        n_detritus = len(params.model[params.model["Type"] == 2])
        n_fleets = len(params.model[params.model["Type"] == 3])

        # Check for stanza data
        stanza_info = ""
        if (
            hasattr(params, "stanzas")
            and params.stanzas is not None
            and params.stanzas.n_stanza_groups > 0
        ):
            n_stanza = params.stanzas.n_stanza_groups
            n_stages = (
                len(params.stanzas.stindiv) if params.stanzas.stindiv is not None else 0
            )
            stanza_info = (
                f", {n_stanza} multi-stanza group(s) with {n_stages} life stages"
            )

        return ui.div(
            ui.tags.i(class_="bi bi-check-circle-fill text-success me-2"),
            f"Model loaded: {n_groups} groups ({n_living} living, {n_detritus} detritus, {n_fleets} fleets){stanza_info}",
            class_="alert alert-success",
        )

    @output
    @render.data_frame
    def imported_groups_table():
        """Show imported groups."""
        params = imported_params.get()
        if params is None:
            return pd.DataFrame()

        # Select key columns
        cols = ["Group", "Type", "Biomass", "PB", "QB", "EE", "ProdCons"]
        cols = [c for c in cols if c in params.model.columns]

        df = params.model[cols].copy()

        # Get remarks if available
        remarks_df = (
            params.remarks
            if hasattr(params, "remarks") and params.remarks is not None
            else None
        )

        # Format for display: handle 9999 values and round to 3 decimals
        formatted_df, no_data_mask, remarks_mask, _ = format_dataframe_for_display(
            df, decimal_places=3, remarks_df=remarks_df
        )
        styles = create_cell_styles(formatted_df, no_data_mask, remarks_mask)

        return render.DataGrid(formatted_df, styles=styles)

    @output
    @render.data_frame
    def imported_diet_table():
        """Show imported diet matrix."""
        params = imported_params.get()
        if params is None:
            return pd.DataFrame()

        # Format for display: handle 9999 values and round to 3 decimals
        formatted_df, no_data_mask, remarks_mask, _ = format_dataframe_for_display(
            params.diet.copy(), decimal_places=3
        )
        styles = create_cell_styles(formatted_df, no_data_mask, remarks_mask)

        return render.DataGrid(formatted_df, styles=styles)

    @output
    @render.ui
    def imported_stanza_status():
        """Show multi-stanza status in import preview."""
        params = imported_params.get()
        if params is None:
            return ui.p("No model imported.", class_="text-muted")

        has_stanzas = (
            hasattr(params, "stanzas")
            and params.stanzas is not None
            and params.stanzas.n_stanza_groups > 0
        )

        if not has_stanzas:
            return ui.div(
                ui.tags.i(class_="bi bi-info-circle me-2"),
                "This model has no multi-stanza groups defined.",
                class_="alert alert-info",
            )

        n_groups = params.stanzas.n_stanza_groups
        n_stages = (
            len(params.stanzas.stindiv) if params.stanzas.stindiv is not None else 0
        )

        return ui.div(
            ui.tags.i(class_="bi bi-check-circle-fill text-success me-2"),
            f"Found {n_groups} multi-stanza group(s) with {n_stages} total life stages.",
            class_="alert alert-success",
        )

    @output
    @render.data_frame
    def imported_stanza_groups_table():
        """Show imported stanza groups."""
        params = imported_params.get()
        if params is None:
            return pd.DataFrame()

        has_stanzas = (
            hasattr(params, "stanzas")
            and params.stanzas is not None
            and params.stanzas.stgroups is not None
            and len(params.stanzas.stgroups) > 0
        )

        if not has_stanzas:
            return render.DataGrid(
                pd.DataFrame({"Message": ["No multi-stanza groups"]})
            )

        df = params.stanzas.stgroups.copy()
        formatted_df, no_data_mask, _, _ = format_dataframe_for_display(
            df, decimal_places=3
        )
        styles = create_cell_styles(formatted_df, no_data_mask, None)

        return render.DataGrid(formatted_df, styles=styles)

    @output
    @render.data_frame
    def imported_stanza_indiv_table():
        """Show imported stanza life stages."""
        params = imported_params.get()
        if params is None:
            return pd.DataFrame()

        has_stanzas = (
            hasattr(params, "stanzas")
            and params.stanzas is not None
            and params.stanzas.stindiv is not None
            and len(params.stanzas.stindiv) > 0
        )

        if not has_stanzas:
            return render.DataGrid(
                pd.DataFrame({"Message": ["No multi-stanza life stages"]})
            )

        df = params.stanzas.stindiv.copy()

        # Reorder columns for better display
        preferred_order = [
            "StanzaGroup",
            "Group",
            "StanzaNum",
            "First",
            "Last",
            "Z",
            "Leading",
        ]
        cols = [c for c in preferred_order if c in df.columns]
        cols += [c for c in df.columns if c not in cols]
        df = df[cols]

        formatted_df, no_data_mask, _, _ = format_dataframe_for_display(
            df, decimal_places=3
        )
        styles = create_cell_styles(formatted_df, no_data_mask, None)

        return render.DataGrid(formatted_df, styles=styles)

    @output
    @render.ui
    def imported_summary():
        """Show model summary statistics."""
        params = imported_params.get()
        if params is None:
            return ui.p("No model loaded.", class_="text-muted")

        model = params.model

        # Calculate summary stats
        biomass_sum = model["Biomass"].sum() if "Biomass" in model.columns else 0

        living = model[model["Type"] <= 1]
        if len(living) > 0 and "Biomass" in living.columns and "PB" in living.columns:
            production = (living["Biomass"] * living["PB"]).sum()
        else:
            production = 0

        # Count stanzas
        n_stanza = 0
        if hasattr(params, "stanzas") and params.stanzas is not None:
            n_stanza = params.stanzas.n_stanza_groups

        return ui.div(
            ui.h5("Model Summary"),
            ui.layout_columns(
                ui.value_box(
                    "Total Biomass",
                    f"{biomass_sum:.2f} t/km²",
                    showcase=ui.tags.i(class_="bi bi-box"),
                ),
                ui.value_box(
                    "Total Production",
                    f"{production:.2f} t/km²/yr",
                    showcase=ui.tags.i(class_="bi bi-graph-up-arrow"),
                ),
                ui.value_box(
                    "Living Groups",
                    str(len(living)),
                    showcase=ui.tags.i(class_="bi bi-circle-fill"),
                ),
                col_widths=[
                    UI.col_width_narrow,
                    UI.col_width_narrow,
                    UI.col_width_narrow,
                ],
            ),
            ui.layout_columns(
                ui.value_box(
                    "Detritus Groups",
                    str(len(model[model["Type"] == 2])),
                    showcase=ui.tags.i(class_="bi bi-recycle"),
                    theme="bg-secondary",
                ),
                ui.value_box(
                    "Fleets",
                    str(len(model[model["Type"] == 3])),
                    showcase=ui.tags.i(class_="bi bi-tsunami"),
                    theme="bg-secondary",
                ),
                ui.value_box(
                    "Multi-Stanza Groups",
                    str(n_stanza),
                    showcase=ui.tags.i(class_="bi bi-diagram-3"),
                    theme="bg-secondary",
                ),
                col_widths=[
                    UI.col_width_narrow,
                    UI.col_width_narrow,
                    UI.col_width_narrow,
                ],
            ),
        )

    @output
    @render.ui
    def use_model_button_ecobase():
        """Show 'Use Model' button in EcoBase tab when model is loaded."""
        params = imported_params.get()
        if params is None:
            return ui.div()  # Return empty div instead of None

        return ui.input_action_button(
            "btn_use_imported",
            ui.tags.span(
                ui.tags.i(class_="bi bi-arrow-right-circle me-1"),
                "Use This Model in Ecopath",
            ),
            class_="btn-primary w-100",
        )

    @output
    @render.ui
    def use_model_button_ewe():
        """Show 'Use Model' button in EwE tab when model is loaded."""
        params = imported_params.get()
        if params is None:
            return ui.div()  # Return empty div instead of None

        return ui.input_action_button(
            "btn_use_imported",
            ui.tags.span(
                ui.tags.i(class_="bi bi-arrow-right-circle me-1"),
                "Use This Model in Ecopath",
            ),
            class_="btn-primary w-100",
        )

    @reactive.effect
    @reactive.event(input.btn_use_imported)
    def _use_imported():
        """Transfer imported model to main model data."""
        params = imported_params.get()
        if params is None:
            ui.notification_show("No model to use", type="warning")
            return

        model_data.set(params)
        ui.notification_show(
            "Model transferred! Go to 'Ecopath Model' tab to edit and balance.",
            type="message",
        )

    # === Biodiversity Database Functions ===

    # Reactive values for biodiversity data
    biodata_df = reactive.Value(None)
    biodata_model = reactive.Value(None)

    @reactive.effect
    @reactive.event(input.btn_load_example_species)
    def _load_example_species():
        """Load example species list."""
        example_species = """Atlantic cod
Atlantic herring
European sprat
Zooplankton
Phytoplankton"""
        ui.update_text_area("biodata_species_list", value=example_species)
        ui.notification_show("Example species loaded", type="message", duration=2)

    @reactive.effect
    @reactive.event(input.btn_fetch_biodata)
    def _fetch_biodata():
        """Fetch species data from biodiversity databases."""
        species_text = input.biodata_species_list()
        if not species_text or not species_text.strip():
            ui.notification_show("Please enter species names", type="warning")
            return

        # Parse species list
        species_list = [s.strip() for s in species_text.split("\n") if s.strip()]

        if len(species_list) == 0:
            ui.notification_show("No valid species names found", type="warning")
            return

        try:
            ui.notification_show(
                f"Fetching data for {len(species_list)} species from WoRMS, OBIS, and FishBase...",
                duration=5,
            )

            # Fetch data using batch function
            df = batch_get_species_info(
                species_list,
                include_occurrences=input.biodata_include_occurrences(),
                include_traits=input.biodata_include_traits(),
                strict=False,  # Allow partial data
                max_workers=5,
                timeout=45,
            )

            if df is None or len(df) == 0:
                ui.notification_show(
                    "No species data retrieved. Check species names and try again.",
                    type="warning",
                    duration=5,
                )
                return

            biodata_df.set(df)
            ui.notification_show(
                f"Successfully fetched data for {len(df)}/{len(species_list)} species!",
                type="message",
                duration=3,
            )

        except SpeciesNotFoundError as e:
            ui.notification_show(
                f"Species not found: {str(e)}", type="warning", duration=5
            )
        except APIConnectionError as e:
            ui.notification_show(
                f"API connection error: {str(e)}", type="error", duration=5
            )
        except Exception as e:
            ui.notification_show(
                f"Error fetching data: {str(e)}", type="error", duration=5
            )

    @output
    @render.ui
    def biodata_fetch_status():
        """Show fetch status."""
        df = biodata_df.get()
        if df is None:
            return ui.div()

        n_species = len(df)
        n_with_tl = df["trophic_level"].notna().sum()
        n_with_obis = df["occurrence_count"].notna().sum()

        return ui.div(
            ui.tags.i(class_="bi bi-check-circle-fill text-success me-2"),
            f"Retrieved: {n_species} species, {n_with_tl} with trophic level, {n_with_obis} with OBIS data",
            class_="alert alert-success small",
        )

    @output
    @render.data_frame
    def biodata_results_table():
        """Show fetched species data."""
        df = biodata_df.get()
        if df is None:
            return pd.DataFrame(
                {
                    "Message": [
                        "Click 'Fetch Species Data' to retrieve biodiversity data"
                    ]
                }
            )

        # Select key columns for display
        display_cols = [
            "common_name",
            "scientific_name",
            "trophic_level",
            "max_length",
            "occurrence_count",
        ]
        display_cols = [c for c in display_cols if c in df.columns]
        display_df = df[display_cols].copy()

        # Rename for better display
        display_df = display_df.rename(
            columns={
                "common_name": "Common Name",
                "scientific_name": "Scientific Name",
                "trophic_level": "TL",
                "max_length": "Max Length (cm)",
                "occurrence_count": "OBIS Records",
            }
        )

        return render.DataGrid(display_df, height=UI.datagrid_height_default_px)

    @output
    @render.ui
    def biodata_biomass_section():
        """Show biomass input section."""
        df = biodata_df.get()
        if df is None:
            return ui.p(
                "Fetch species data first to enter biomass estimates.",
                class_="text-muted small",
            )

        # Create biomass inputs for each species
        inputs = []
        inputs.append(ui.h6("Biomass Estimates (t/km²)", class_="mb-2"))
        inputs.append(
            ui.p(
                "Enter estimated biomass for each species:",
                class_="small text-muted mb-2",
            )
        )

        for idx, row in df.iterrows():
            sp_name = row["common_name"]
            # Create safe input ID
            input_id = f"biomass_{sp_name.replace(' ', '_').replace('-', '_').lower()}"

            inputs.append(
                ui.input_numeric(
                    input_id,
                    sp_name,
                    value=1.0,
                    min=PARAM_RANGES.biomass_input_min,
                    step=PARAM_RANGES.biomass_input_step,
                    width="100%",
                )
            )

        return ui.div(*inputs)

    @output
    @render.ui
    def biodata_create_button():
        """Show create model button when data is fetched."""
        df = biodata_df.get()
        if df is None:
            return ui.div()

        return ui.input_action_button(
            "btn_create_biodata_model",
            ui.tags.span(ui.tags.i(class_="bi bi-gear me-1"), "Create Ecopath Model"),
            class_="btn-success w-100",
        )

    @reactive.effect
    @reactive.event(input.btn_create_biodata_model)
    def _create_biodata_model():
        """Create Ecopath model from biodiversity data."""
        df = biodata_df.get()
        if df is None:
            ui.notification_show("No species data available", type="warning")
            return

        try:
            # Collect biomass estimates from inputs
            biomass_estimates = {}
            for idx, row in df.iterrows():
                sp_name = row["common_name"]
                input_id = (
                    f"biomass_{sp_name.replace(' ', '_').replace('-', '_').lower()}"
                )

                # Get the input value
                try:
                    biomass_val = input[input_id]()
                    if biomass_val is not None and biomass_val > 0:
                        biomass_estimates[sp_name] = biomass_val
                except Exception:
                    # If input doesn't exist or error, use default
                    biomass_estimates[sp_name] = 1.0

            ui.notification_show("Creating Ecopath model...", duration=3)

            # Create model using biodata_to_rpath
            params = biodata_to_rpath(
                df, biomass_estimates=biomass_estimates, area_km2=input.biodata_area()
            )

            biodata_model.set(params)
            imported_params.set(params)  # Also set imported_params for preview

            ui.notification_show(
                f"Model created successfully with {len(params.model)} groups!",
                type="message",
                duration=3,
            )

        except Exception as e:
            ui.notification_show(
                f"Error creating model: {str(e)}", type="error", duration=5
            )

    @output
    @render.ui
    def use_model_button_biodata():
        """Show 'Use Model' button in Biodiversity tab when model is created."""
        params = biodata_model.get()
        if params is None:
            return ui.div()

        return ui.input_action_button(
            "btn_use_biodata_model",
            ui.tags.span(
                ui.tags.i(class_="bi bi-arrow-right-circle me-1"),
                "Use This Model in Ecopath",
            ),
            class_="btn-primary w-100",
        )

    @reactive.effect
    @reactive.event(input.btn_use_biodata_model)
    def _use_biodata_model():
        """Transfer biodata model to main model data."""
        params = biodata_model.get()
        if params is None:
            ui.notification_show("No model to use", type="warning")
            return

        model_data.set(params)
        ui.notification_show(
            "Model transferred! Go to 'Ecopath Model' tab to edit and balance.",
            type="message",
        )
