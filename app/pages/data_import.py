"""Data Import page module - EcoBase and EwE database import."""

from shiny import Inputs, Outputs, Session, reactive, render, ui, req
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict

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

# Import shared utilities
from .utils import (
    format_dataframe_for_display,
    create_cell_styles,
    TYPE_LABELS,
    NO_DATA_VALUE,
    NO_DATA_STYLE,
    REMARK_STYLE,
)


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
                            ui.tags.a("EcoBase", href="http://ecobase.ecopath.org/", target="_blank"),
                            class_="small text-muted"
                        ),
                        
                        # Search section
                        ui.input_text("ecobase_search", "Search", placeholder="e.g., Baltic, coral"),
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
                        ui.div(
                            ui.input_action_button(
                                "btn_search_ecobase",
                                ui.tags.span(ui.tags.i(class_="bi bi-search me-1"), "Search"),
                                class_="btn-primary btn-sm"
                            ),
                            ui.input_action_button(
                                "btn_list_all",
                                "All Models",
                                class_="btn-outline-secondary btn-sm ms-1"
                            ),
                            class_="mb-3"
                        ),
                        
                        ui.tags.hr(),
                        
                        # Results table
                        ui.h6("Available Models"),
                        ui.p("Click a row to select, then download.", class_="small text-muted"),
                        ui.output_data_frame("ecobase_models_table"),
                        
                        ui.tags.hr(),
                        
                        # Selected model info and download
                        ui.output_ui("ecobase_selected_info"),
                        
                        ui.tags.hr(),
                        
                        # Use imported model button
                        ui.output_ui("use_model_button_ecobase"),
                        
                        class_="mt-2"
                    ),
                ),
                
                # EwE File tab
                ui.nav_panel(
                    "EwE File",
                    ui.div(
                        ui.p(
                            "Import from EwE 6.x database files (.ewemdb, .eweaccdb, .mdb, .accdb)",
                            class_="small text-muted"
                        ),
                        
                        # Check driver support
                        ui.output_ui("ewemdb_support_status"),
                        
                        ui.tags.hr(),
                        
                        # File upload
                        ui.input_file(
                            "ewemdb_upload",
                            "Select file",
                            accept=[".ewemdb", ".eweaccdb", ".mdb", ".accdb", ".ewe"],
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
                            ui.tags.span(ui.tags.i(class_="bi bi-upload me-1"), "Import Model"),
                            class_="btn-success mt-2"
                        ),
                        
                        ui.tags.hr(),
                        
                        # Metadata preview
                        ui.h6("File Information"),
                        ui.output_ui("ewemdb_metadata_ui"),
                        
                        ui.tags.hr(),
                        
                        # Use imported model button
                        ui.output_ui("use_model_button_ewe"),
                        
                        class_="mt-2"
                    ),
                ),
                id="import_tabs"
            ),
            width=400,
            title="Import Source"
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
    input: Inputs,
    output: Outputs,
    session: Session,
    model_data: reactive.Value
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
                results = all_models.copy()
            
            # Reset index before filtering to avoid alignment issues
            results = results.reset_index(drop=True)
            
            if ecosystem:
                mask = results['ecosystem_type'].str.lower().str.contains(ecosystem.lower(), na=False)
                results = results[mask].reset_index(drop=True)
            
            ecobase_models.set(results)
            selected_model_id.set(None)
            ui.notification_show(f"Found {len(results)} matching models", type="message")
        except Exception as e:
            ui.notification_show(f"Error: {str(e)}", type="error")
    
    @output
    @render.data_frame
    def ecobase_models_table():
        """Render EcoBase models as data frame."""
        models = ecobase_models.get()
        if models is None or len(models) == 0:
            return render.DataGrid(pd.DataFrame({"Message": ["Click 'All Models' or search to load models"]}))
        
        display_cols = ['model_number', 'model_name', 'country', 'ecosystem_type', 'num_groups']
        display_cols = [c for c in display_cols if c in models.columns]
        return render.DataGrid(
            models[display_cols].head(100), 
            selection_mode="row",
            height="300px"
        )
    
    @reactive.effect
    def _update_selected_model():
        """Update selected model when row is clicked."""
        models = ecobase_models.get()
        selected_rows = input.ecobase_models_table_selected_rows()
        
        if models is not None and selected_rows and len(selected_rows) > 0:
            row_idx = selected_rows[0]
            if row_idx < len(models):
                model_id = models.iloc[row_idx]['model_number']
                selected_model_id.set(int(model_id))
    
    @output
    @render.ui
    def ecobase_selected_info():
        """Show selected model info and download button."""
        model_id = selected_model_id.get()
        models = ecobase_models.get()
        
        if model_id is None or models is None:
            return ui.p("Select a model from the table above", class_="text-muted small")
        
        # Find model info
        model_row = models[models['model_number'] == model_id]
        if len(model_row) == 0:
            return ui.p("Select a model from the table above", class_="text-muted small")
        
        model_name = model_row.iloc[0].get('model_name', f'Model {model_id}')
        
        return ui.div(
            ui.div(
                ui.tags.strong("Selected: "),
                f"{model_name} (ID: {model_id})",
                class_="mb-2"
            ),
            ui.input_action_button(
                "btn_download_ecobase",
                ui.tags.span(ui.tags.i(class_="bi bi-cloud-download me-1"), "Download Selected Model"),
                class_="btn-success w-100"
            ),
            class_="p-2 bg-light rounded"
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
                if col != 'Group':
                    for idx in params.diet.index:
                        val = params.diet.at[idx, col]
                        if pd.notna(val) and val > 0:
                            diet_count += 1
            
            imported_params.set(params)
            
            ui.notification_show(
                f"Downloaded model: {len(params.model)} groups, {diet_count} diet entries",
                type="message"
            )
        except Exception as e:
            ui.notification_show(f"Download error: {str(e)}", type="error")
    
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
            
            # Debug: Check for remarks
            has_remarks = hasattr(params, 'remarks') and params.remarks is not None
            remarks_info = ""
            if has_remarks:
                # Count non-empty remarks
                non_empty_count = 0
                for col in params.remarks.columns:
                    if col != 'Group':
                        non_empty_count += sum(1 for v in params.remarks[col] if str(v).strip())
                remarks_info = f", {non_empty_count} remarks"
                print(f"[DEBUG] Imported model has remarks: {params.remarks.columns.tolist()}")
            else:
                print(f"[DEBUG] Imported model has NO remarks")
            
            # Check for stanza data
            stanza_info = ""
            if hasattr(params, 'stanzas') and params.stanzas is not None and params.stanzas.n_stanza_groups > 0:
                n_stanza = params.stanzas.n_stanza_groups
                n_stages = len(params.stanzas.stindiv) if params.stanzas.stindiv is not None else 0
                stanza_info = f", {n_stanza} stanza group(s)"
                print(f"[DEBUG] Imported model has {n_stanza} stanza groups with {n_stages} life stages")
            
            ui.notification_show(
                f"Imported model with {len(params.model)} groups{remarks_info}{stanza_info}",
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
            
            # Build ecosim/ecospace indicators
            ecosim_badge = ""
            if metadata.get('has_ecosim'):
                n_scen = metadata.get('num_scenarios', 0)
                ecosim_badge = ui.tags.span(
                    f"Ecosim ({n_scen} scenarios)",
                    class_="badge bg-success me-1"
                )
            
            ecospace_badge = ""
            if metadata.get('has_ecospace'):
                ecospace_badge = ui.tags.span(
                    "Ecospace",
                    class_="badge bg-info"
                )
            
            return ui.div(
                ui.tags.table(
                    ui.tags.tr(ui.tags.td("Name:"), ui.tags.td(metadata.get('name', 'N/A'))),
                    ui.tags.tr(ui.tags.td("Author:"), ui.tags.td(metadata.get('author', 'N/A'))),
                    ui.tags.tr(ui.tags.td("Groups:"), ui.tags.td(str(metadata.get('num_groups', 'N/A')))),
                    ui.tags.tr(ui.tags.td("Fleets:"), ui.tags.td(str(metadata.get('num_fleets', 'N/A')))),
                    ui.tags.tr(ui.tags.td("Contains:"), ui.tags.td(ecosim_badge, ecospace_badge if ecospace_badge else "Ecopath only")),
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
        
        # Check for stanza data
        stanza_info = ""
        if hasattr(params, 'stanzas') and params.stanzas is not None and params.stanzas.n_stanza_groups > 0:
            n_stanza = params.stanzas.n_stanza_groups
            n_stages = len(params.stanzas.stindiv) if params.stanzas.stindiv is not None else 0
            stanza_info = f", {n_stanza} multi-stanza group(s) with {n_stages} life stages"
        
        return ui.div(
            ui.tags.i(class_="bi bi-check-circle-fill text-success me-2"),
            f"Model loaded: {n_groups} groups ({n_living} living, {n_detritus} detritus, {n_fleets} fleets){stanza_info}",
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
        
        df = params.model[cols].copy()
        
        # Get remarks if available
        remarks_df = params.remarks if hasattr(params, 'remarks') and params.remarks is not None else None
        
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
            hasattr(params, 'stanzas') and 
            params.stanzas is not None and 
            params.stanzas.n_stanza_groups > 0
        )
        
        if not has_stanzas:
            return ui.div(
                ui.tags.i(class_="bi bi-info-circle me-2"),
                "This model has no multi-stanza groups defined.",
                class_="alert alert-info"
            )
        
        n_groups = params.stanzas.n_stanza_groups
        n_stages = len(params.stanzas.stindiv) if params.stanzas.stindiv is not None else 0
        
        return ui.div(
            ui.tags.i(class_="bi bi-check-circle-fill text-success me-2"),
            f"Found {n_groups} multi-stanza group(s) with {n_stages} total life stages.",
            class_="alert alert-success"
        )
    
    @output
    @render.data_frame
    def imported_stanza_groups_table():
        """Show imported stanza groups."""
        params = imported_params.get()
        if params is None:
            return pd.DataFrame()
        
        has_stanzas = (
            hasattr(params, 'stanzas') and 
            params.stanzas is not None and 
            params.stanzas.stgroups is not None and
            len(params.stanzas.stgroups) > 0
        )
        
        if not has_stanzas:
            return render.DataGrid(pd.DataFrame({'Message': ['No multi-stanza groups']}))
        
        df = params.stanzas.stgroups.copy()
        formatted_df, no_data_mask, _, _ = format_dataframe_for_display(df, decimal_places=3)
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
            hasattr(params, 'stanzas') and 
            params.stanzas is not None and 
            params.stanzas.stindiv is not None and
            len(params.stanzas.stindiv) > 0
        )
        
        if not has_stanzas:
            return render.DataGrid(pd.DataFrame({'Message': ['No multi-stanza life stages']}))
        
        df = params.stanzas.stindiv.copy()
        
        # Reorder columns for better display
        preferred_order = ['StanzaGroup', 'Group', 'StanzaNum', 'First', 'Last', 'Z', 'Leading']
        cols = [c for c in preferred_order if c in df.columns]
        cols += [c for c in df.columns if c not in cols]
        df = df[cols]
        
        formatted_df, no_data_mask, _, _ = format_dataframe_for_display(df, decimal_places=3)
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
        biomass_sum = model['Biomass'].sum() if 'Biomass' in model.columns else 0
        
        living = model[model['Type'] <= 1]
        if len(living) > 0 and 'Biomass' in living.columns and 'PB' in living.columns:
            production = (living['Biomass'] * living['PB']).sum()
        else:
            production = 0
        
        # Count stanzas
        n_stanza = 0
        if hasattr(params, 'stanzas') and params.stanzas is not None:
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
                col_widths=[4, 4, 4]
            ),
            ui.layout_columns(
                ui.value_box(
                    "Detritus Groups",
                    str(len(model[model['Type'] == 2])),
                    showcase=ui.tags.i(class_="bi bi-recycle"),
                    theme="bg-secondary",
                ),
                ui.value_box(
                    "Fleets",
                    str(len(model[model['Type'] == 3])),
                    showcase=ui.tags.i(class_="bi bi-tsunami"),
                    theme="bg-secondary",
                ),
                ui.value_box(
                    "Multi-Stanza Groups",
                    str(n_stanza),
                    showcase=ui.tags.i(class_="bi bi-diagram-3"),
                    theme="bg-secondary",
                ),
                col_widths=[4, 4, 4]
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
            ui.tags.span(ui.tags.i(class_="bi bi-arrow-right-circle me-1"), "Use This Model in Ecopath"),
            class_="btn-primary w-100"
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
            ui.tags.span(ui.tags.i(class_="bi bi-arrow-right-circle me-1"), "Use This Model in Ecopath"),
            class_="btn-primary w-100"
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
