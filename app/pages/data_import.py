"""Data Import page module - EcoBase and EwE database import."""

from shiny import Inputs, Outputs, Session, reactive, render, ui, req
import pandas as pd
import numpy as np
from pathlib import Path

# Import pypath
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from pypath.core.params import RpathParams
from pypath.io.ecobase import (
    list_ecobase_models,
    get_ecobase_model,
    ecobase_to_rpath,
    search_ecobase_models,
)
from pypath.io.ewemdb import (
    read_ewemdb,
    get_ewemdb_metadata,
    check_ewemdb_support,
    EwEDatabaseError,
)


def import_ui():
    """Data import page UI."""
    return ui.page_fluid(
        ui.h2("Import Ecopath Models", class_="mb-4"),
        
        ui.layout_columns(
            # EcoBase section
            ui.card(
                ui.card_header(
                    ui.tags.i(class_="bi bi-cloud-download me-2"),
                    "EcoBase Online Database"
                ),
                ui.card_body(
                    ui.p(
                        "Download published Ecopath models from the ",
                        ui.tags.a("EcoBase database", href="http://ecobase.ecopath.org/", target="_blank"),
                        ". This database contains hundreds of models from ecosystems worldwide."
                    ),
                    
                    ui.tags.hr(),
                    
                    # Search section
                    ui.h5("Search Models"),
                    ui.layout_columns(
                        ui.input_text("ecobase_search", "Search Term", placeholder="e.g., Baltic, coral, tropical"),
                        ui.input_select(
                            "ecobase_ecosystem",
                            "Ecosystem Type",
                            choices={
                                "": "All Types",
                                "marine": "Marine",
                                "freshwater": "Freshwater",
                                "estuarine": "Estuarine",
                            }
                        ),
                        col_widths=[6, 6]
                    ),
                    ui.input_action_button(
                        "btn_search_ecobase",
                        "Search EcoBase",
                        class_="btn-primary mb-3"
                    ),
                    ui.input_action_button(
                        "btn_list_all",
                        "List All Models",
                        class_="btn-outline-secondary mb-3 ms-2"
                    ),
                    
                    ui.tags.hr(),
                    
                    # Results table
                    ui.h5("Available Models"),
                    ui.output_ui("ecobase_results_ui"),
                    
                    ui.tags.hr(),
                    
                    # Download section
                    ui.layout_columns(
                        ui.input_numeric(
                            "ecobase_model_id",
                            "Model ID to Download",
                            value=None,
                            min=1
                        ),
                        ui.input_action_button(
                            "btn_download_ecobase",
                            "Download Model",
                            class_="btn-success mt-4"
                        ),
                        col_widths=[6, 6]
                    ),
                    ui.output_ui("ecobase_download_status"),
                ),
            ),
            
            # EwE Database section
            ui.card(
                ui.card_header(
                    ui.tags.i(class_="bi bi-database me-2"),
                    "EwE Database Files (.ewemdb)"
                ),
                ui.card_body(
                    ui.p(
                        "Import models from Ecopath with Ecosim 6.x database files. "
                        "These are Microsoft Access database files (.ewemdb, .mdb, .accdb)."
                    ),
                    
                    # Check driver support
                    ui.output_ui("ewemdb_support_status"),
                    
                    ui.tags.hr(),
                    
                    # File upload
                    ui.h5("Upload EwE Database"),
                    ui.input_file(
                        "ewemdb_upload",
                        "Select .ewemdb file",
                        accept=[".ewemdb", ".mdb", ".accdb", ".ewe"],
                        multiple=False
                    ),
                    
                    ui.input_numeric(
                        "ewemdb_scenario",
                        "Scenario Number",
                        value=1,
                        min=1
                    ),
                    
                    ui.input_action_button(
                        "btn_import_ewemdb",
                        "Import Model",
                        class_="btn-success mt-3"
                    ),
                    
                    ui.tags.hr(),
                    
                    # Metadata preview
                    ui.h5("File Information"),
                    ui.output_ui("ewemdb_metadata_ui"),
                ),
            ),
            col_widths=[6, 6]
        ),
        
        ui.tags.hr(class_="my-4"),
        
        # Preview section
        ui.h3("Imported Model Preview", class_="mb-3"),
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
                "Summary",
                ui.output_ui("imported_summary"),
            ),
        ),
        
        # Use imported model
        ui.div(
            ui.input_action_button(
                "btn_use_imported",
                "Use This Model in Ecopath →",
                class_="btn-primary btn-lg mt-3"
            ),
            class_="text-center mt-4"
        ),
    )


def import_server(
    input: Inputs,
    output: Outputs,
    session: Session,
    model_data: reactive.Value
):
    """Data import page server logic."""
    
    # Reactive values
    ecobase_models = reactive.Value(None)
    imported_params = reactive.Value(None)
    
    # === EcoBase Functions ===
    
    @reactive.effect
    @reactive.event(input.btn_list_all)
    def _list_all_models():
        """List all models from EcoBase."""
        try:
            ui.notification_show("Fetching models from EcoBase...", duration=3)
            models_df = list_ecobase_models()
            ecobase_models.set(models_df)
            ui.notification_show(
                f"Found {len(models_df)} public models",
                type="message"
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
                ui.notification_show("Enter a search term or select ecosystem type", type="warning")
                return
            
            ui.notification_show("Searching EcoBase...", duration=3)
            
            # Get all models first, then filter
            all_models = list_ecobase_models()
            
            if search_term:
                results = search_ecobase_models(search_term, models_df=all_models)
            else:
                results = all_models
            
            if ecosystem:
                results = results[
                    results['ecosystem_type'].str.lower().str.contains(ecosystem.lower(), na=False)
                ]
            
            ecobase_models.set(results)
            ui.notification_show(f"Found {len(results)} matching models", type="message")
        except Exception as e:
            ui.notification_show(f"Error: {str(e)}", type="error")
    
    @output
    @render.ui
    def ecobase_results_ui():
        """Render EcoBase search results."""
        models = ecobase_models.get()
        if models is None:
            return ui.p("Click 'List All Models' or search to see available models.", class_="text-muted")
        
        if len(models) == 0:
            return ui.p("No models found matching your criteria.", class_="text-muted")
        
        # Show subset of columns
        display_cols = ['model_number', 'model_name', 'country', 'ecosystem_type', 'num_groups']
        display_cols = [c for c in display_cols if c in models.columns]
        
        return ui.output_data_frame("ecobase_models_table")
    
    @output
    @render.data_frame
    def ecobase_models_table():
        """Render EcoBase models as data frame."""
        models = ecobase_models.get()
        if models is None or len(models) == 0:
            return pd.DataFrame()
        
        display_cols = ['model_number', 'model_name', 'country', 'ecosystem_type', 'num_groups', 'author']
        display_cols = [c for c in display_cols if c in models.columns]
        return render.DataGrid(models[display_cols].head(100), selection_mode="row")
    
    @reactive.effect
    @reactive.event(input.btn_download_ecobase)
    def _download_ecobase():
        """Download and import model from EcoBase."""
        try:
            model_id = input.ecobase_model_id()
            if not model_id:
                ui.notification_show("Enter a model ID", type="warning")
                return
            
            ui.notification_show(f"Downloading model {model_id}...", duration=5)
            
            # Download model data
            model_data_dict = get_ecobase_model(int(model_id))
            
            # Convert to RpathParams
            params = ecobase_to_rpath(model_data_dict)
            imported_params.set(params)
            
            ui.notification_show(
                f"Downloaded model with {len(params.model)} groups",
                type="message"
            )
        except Exception as e:
            ui.notification_show(f"Download error: {str(e)}", type="error")
    
    @output
    @render.ui
    def ecobase_download_status():
        """Show download status."""
        params = imported_params.get()
        if params is not None:
            return ui.div(
                ui.tags.i(class_="bi bi-check-circle-fill text-success me-2"),
                f"Model loaded: {len(params.model)} groups",
                class_="alert alert-success mt-2"
            )
        return None
    
    # === EwE Database Functions ===
    
    @output
    @render.ui
    def ewemdb_support_status():
        """Check and display ewemdb driver support."""
        support = check_ewemdb_support()
        
        if support['any_available']:
            drivers = []
            if support['pyodbc']:
                drivers.append("pyodbc")
            if support['pypyodbc']:
                drivers.append("pypyodbc")
            if support['mdb_tools']:
                drivers.append("mdb-tools")
            
            return ui.div(
                ui.tags.i(class_="bi bi-check-circle-fill text-success me-2"),
                f"Database drivers available: {', '.join(drivers)}",
                class_="alert alert-success"
            )
        else:
            return ui.div(
                ui.tags.i(class_="bi bi-exclamation-triangle-fill text-warning me-2"),
                "No database drivers found. Install pyodbc or mdb-tools to read .ewemdb files.",
                class_="alert alert-warning"
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
            
            ui.notification_show(
                f"Imported model with {len(params.model)} groups",
                type="message"
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
            
            return ui.div(
                ui.tags.table(
                    ui.tags.tr(ui.tags.td("Name:"), ui.tags.td(metadata.get('name', 'N/A'))),
                    ui.tags.tr(ui.tags.td("Author:"), ui.tags.td(metadata.get('author', 'N/A'))),
                    ui.tags.tr(ui.tags.td("Groups:"), ui.tags.td(str(metadata.get('num_groups', 'N/A')))),
                    ui.tags.tr(ui.tags.td("Fleets:"), ui.tags.td(str(metadata.get('num_fleets', 'N/A')))),
                    class_="table table-sm"
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
                class_="alert alert-info"
            )
        
        n_groups = len(params.model)
        n_living = len(params.model[params.model['Type'] <= 1])
        n_detritus = len(params.model[params.model['Type'] == 2])
        n_fleets = len(params.model[params.model['Type'] == 3])
        
        return ui.div(
            ui.tags.i(class_="bi bi-check-circle-fill text-success me-2"),
            f"Model loaded: {n_groups} groups ({n_living} living, {n_detritus} detritus, {n_fleets} fleets)",
            class_="alert alert-success"
        )
    
    @output
    @render.data_frame
    def imported_groups_table():
        """Show imported groups."""
        params = imported_params.get()
        if params is None:
            return pd.DataFrame()
        
        # Select key columns
        cols = ['Group', 'Type', 'Biomass', 'PB', 'QB', 'EE', 'ProdCons']
        cols = [c for c in cols if c in params.model.columns]
        
        return render.DataGrid(params.model[cols])
    
    @output
    @render.data_frame
    def imported_diet_table():
        """Show imported diet matrix."""
        params = imported_params.get()
        if params is None:
            return pd.DataFrame()
        
        return render.DataGrid(params.diet)
    
    @output
    @render.ui
    def imported_summary():
        """Show model summary statistics."""
        params = imported_params.get()
        if params is None:
            return ui.p("No model loaded.", class_="text-muted")
        
        model = params.model
        
        # Calculate summary stats
        biomass_sum = model['Biomass'].sum() if 'Biomass' in model.columns else 0
        
        living = model[model['Type'] <= 1]
        if len(living) > 0 and 'Biomass' in living.columns and 'PB' in living.columns:
            production = (living['Biomass'] * living['PB']).sum()
        else:
            production = 0
        
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
                col_widths=[4, 4, 4]
            ),
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
            type="message"
        )
