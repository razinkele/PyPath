"""Ecopath model page module."""

from shiny import Inputs, Outputs, Session, reactive, render, ui, req
import pandas as pd
import numpy as np

# Import pypath
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from pypath.core.params import create_rpath_params, check_rpath_params, RpathParams
from pypath.core.ecopath import rpath, Rpath


def ecopath_ui():
    """Ecopath model page UI."""
    return ui.page_fluid(
        ui.h2("Ecopath Mass-Balance Model", class_="mb-4"),
        
        ui.layout_sidebar(
            # Sidebar for model setup
            ui.sidebar(
                ui.h4("Model Setup"),
                
                # Model name
                ui.input_text("eco_name", "Model Name", value="My Ecosystem"),
                
                ui.tags.hr(),
                
                # Group definition section
                ui.h5("Define Groups"),
                ui.input_text_area(
                    "group_names",
                    "Group Names (one per line)",
                    value="Phytoplankton\nZooplankton\nSmall Fish\nLarge Fish\nDetritus\nFleet",
                    rows=6
                ),
                ui.input_text_area(
                    "group_types",
                    "Group Types (one per line: 1=producer, 0=consumer, 2=detritus, 3=fleet)",
                    value="1\n0\n0\n0\n2\n3",
                    rows=6
                ),
                
                ui.input_action_button(
                    "btn_create_params",
                    "Create Parameter Template",
                    class_="btn-primary w-100 mt-3"
                ),
                
                ui.tags.hr(),
                
                # File upload
                ui.h5("Or Load from File"),
                ui.input_file(
                    "upload_params",
                    "Upload Parameters (CSV)",
                    accept=[".csv"],
                    multiple=False
                ),
                
                ui.tags.hr(),
                
                # Balance model
                ui.h5("Run Model"),
                ui.input_action_button(
                    "btn_balance",
                    "Balance Model",
                    class_="btn-success w-100"
                ),
                
                ui.download_button(
                    "download_params",
                    "Download Parameters",
                    class_="btn-outline-secondary w-100 mt-2"
                ),
                
                width=300,
            ),
            
            # Main content area
            ui.navset_card_tab(
                ui.nav_panel(
                    "Model Parameters",
                    ui.h4("Basic Parameters", class_="mt-3"),
                    ui.output_ui("model_params_table"),
                ),
                ui.nav_panel(
                    "Diet Matrix",
                    ui.h4("Diet Composition", class_="mt-3"),
                    ui.p("Enter diet fractions (columns must sum to 1.0 for each predator)"),
                    ui.output_ui("diet_matrix_table"),
                ),
                ui.nav_panel(
                    "Fisheries",
                    ui.h4("Landings & Discards", class_="mt-3"),
                    ui.output_ui("fisheries_table"),
                ),
                ui.nav_panel(
                    "Model Results",
                    ui.h4("Balanced Model Results", class_="mt-3"),
                    ui.output_ui("balance_status"),
                    ui.output_table("model_results_table"),
                ),
                ui.nav_panel(
                    "Diagnostics",
                    ui.h4("Model Diagnostics", class_="mt-3"),
                    ui.output_ui("diagnostics_output"),
                    ui.layout_columns(
                        ui.output_plot("trophic_level_plot"),
                        ui.output_plot("ee_plot"),
                        col_widths=[6, 6]
                    ),
                ),
            ),
        ),
    )


def ecopath_server(
    input: Inputs, 
    output: Outputs, 
    session: Session,
    model_data: reactive.Value
):
    """Ecopath model page server logic."""
    
    # Reactive values for this page
    params = reactive.Value(None)
    balanced_model = reactive.Value(None)
    
    @reactive.effect
    @reactive.event(input.btn_create_params)
    def _create_params():
        """Create parameter template from group definitions."""
        try:
            # Parse group names
            names = [n.strip() for n in input.group_names().split('\n') if n.strip()]
            types_str = [t.strip() for t in input.group_types().split('\n') if t.strip()]
            
            if len(names) != len(types_str):
                ui.notification_show(
                    f"Number of names ({len(names)}) must match number of types ({len(types_str)})",
                    type="error"
                )
                return
            
            types = [int(t) for t in types_str]
            
            # Create parameters
            new_params = create_rpath_params(names, types)
            params.set(new_params)
            
            ui.notification_show(
                f"Created parameter template with {len(names)} groups",
                type="message"
            )
        except Exception as e:
            ui.notification_show(f"Error creating parameters: {str(e)}", type="error")
    
    @output
    @render.ui
    def model_params_table():
        """Render editable model parameters table."""
        p = params.get()
        if p is None:
            return ui.p("Click 'Create Parameter Template' to start, or upload a file.", 
                       class_="text-muted")
        
        # Create input fields for each parameter
        model_df = p.model
        n_groups = len(model_df)
        
        # Build table rows
        rows = []
        for i in range(n_groups):
            group = model_df.loc[i, 'Group']
            gtype = model_df.loc[i, 'Type']
            
            # Different inputs based on group type
            if gtype == 3:  # Fleet - minimal inputs
                row = ui.tags.tr(
                    ui.tags.td(group),
                    ui.tags.td(f"Fleet ({gtype})"),
                    ui.tags.td("-"),
                    ui.tags.td("-"),
                    ui.tags.td("-"),
                    ui.tags.td("-"),
                )
            elif gtype == 2:  # Detritus
                row = ui.tags.tr(
                    ui.tags.td(group),
                    ui.tags.td(f"Detritus ({gtype})"),
                    ui.tags.td(
                        ui.input_numeric(f"biomass_{i}", None, value=100.0, min=0, width="100px")
                    ),
                    ui.tags.td("-"),
                    ui.tags.td("-"),
                    ui.tags.td("-"),
                )
            else:  # Living groups
                row = ui.tags.tr(
                    ui.tags.td(group),
                    ui.tags.td(f"{'Producer' if gtype == 1 else 'Consumer'} ({gtype})"),
                    ui.tags.td(
                        ui.input_numeric(f"biomass_{i}", None, value=None, min=0, width="100px")
                    ),
                    ui.tags.td(
                        ui.input_numeric(f"pb_{i}", None, value=None, min=0, width="100px")
                    ),
                    ui.tags.td(
                        ui.input_numeric(f"qb_{i}", None, value=None, min=0, width="100px")
                    ) if gtype < 1 else ui.tags.td("-"),
                    ui.tags.td(
                        ui.input_numeric(f"ee_{i}", None, value=None, min=0, max=1, width="100px")
                    ),
                )
            rows.append(row)
        
        return ui.tags.table(
            ui.tags.thead(
                ui.tags.tr(
                    ui.tags.th("Group"),
                    ui.tags.th("Type"),
                    ui.tags.th("Biomass"),
                    ui.tags.th("P/B"),
                    ui.tags.th("Q/B"),
                    ui.tags.th("EE"),
                )
            ),
            ui.tags.tbody(*rows),
            class_="table table-striped table-hover"
        )
    
    @output
    @render.ui
    def diet_matrix_table():
        """Render editable diet matrix."""
        p = params.get()
        if p is None:
            return ui.p("Create or load parameters first.", class_="text-muted")
        
        diet_df = p.diet
        
        # Build header with predator names
        header_cells = [ui.tags.th("Prey / Predator")]
        pred_cols = [c for c in diet_df.columns if c != 'Group']
        for pred in pred_cols:
            header_cells.append(ui.tags.th(pred))
        
        # Build rows
        rows = []
        for i, row_data in diet_df.iterrows():
            prey = row_data['Group']
            cells = [ui.tags.td(prey)]
            
            for j, pred in enumerate(pred_cols):
                input_id = f"diet_{i}_{j}"
                cells.append(
                    ui.tags.td(
                        ui.input_numeric(input_id, None, value=0.0, min=0, max=1, 
                                        step=0.01, width="80px")
                    )
                )
            rows.append(ui.tags.tr(*cells))
        
        return ui.tags.div(
            ui.tags.table(
                ui.tags.thead(ui.tags.tr(*header_cells)),
                ui.tags.tbody(*rows),
                class_="table table-sm table-bordered"
            ),
            style="overflow-x: auto;"
        )
    
    @output
    @render.ui
    def fisheries_table():
        """Render fisheries (landings/discards) table."""
        p = params.get()
        if p is None:
            return ui.p("Create or load parameters first.", class_="text-muted")
        
        model_df = p.model
        
        # Find fleet columns
        fleet_groups = model_df[model_df['Type'] == 3]['Group'].tolist()
        
        if not fleet_groups:
            return ui.p("No fleets defined in the model.", class_="text-muted")
        
        # Build table
        header_cells = [ui.tags.th("Group")]
        for fleet in fleet_groups:
            header_cells.append(ui.tags.th(f"{fleet} Landings"))
            header_cells.append(ui.tags.th(f"{fleet} Discards"))
        
        rows = []
        living_groups = model_df[model_df['Type'] < 2]['Group'].tolist()
        
        for i, group in enumerate(living_groups):
            cells = [ui.tags.td(group)]
            for j, fleet in enumerate(fleet_groups):
                cells.append(
                    ui.tags.td(
                        ui.input_numeric(f"landing_{i}_{j}", None, value=0.0, min=0, width="80px")
                    )
                )
                cells.append(
                    ui.tags.td(
                        ui.input_numeric(f"discard_{i}_{j}", None, value=0.0, min=0, width="80px")
                    )
                )
            rows.append(ui.tags.tr(*cells))
        
        return ui.tags.table(
            ui.tags.thead(ui.tags.tr(*header_cells)),
            ui.tags.tbody(*rows),
            class_="table table-sm table-bordered"
        )
    
    @reactive.effect
    @reactive.event(input.btn_balance)
    def _balance_model():
        """Balance the Ecopath model."""
        p = params.get()
        if p is None:
            ui.notification_show("Create parameters first", type="warning")
            return
        
        try:
            # Update parameters from inputs
            model_df = p.model.copy()
            n_groups = len(model_df)
            
            for i in range(n_groups):
                gtype = model_df.loc[i, 'Type']
                
                if gtype < 2:  # Living groups
                    biomass_val = input[f"biomass_{i}"]()
                    pb_val = input[f"pb_{i}"]()
                    ee_val = input[f"ee_{i}"]()
                    
                    if biomass_val is not None:
                        model_df.loc[i, 'Biomass'] = biomass_val
                    if pb_val is not None:
                        model_df.loc[i, 'PB'] = pb_val
                    if ee_val is not None:
                        model_df.loc[i, 'EE'] = ee_val
                    
                    if gtype < 1:  # Consumer
                        qb_val = input[f"qb_{i}"]()
                        if qb_val is not None:
                            model_df.loc[i, 'QB'] = qb_val
                
                elif gtype == 2:  # Detritus
                    biomass_val = input[f"biomass_{i}"]()
                    if biomass_val is not None:
                        model_df.loc[i, 'Biomass'] = biomass_val
            
            # Update diet matrix
            diet_df = p.diet.copy()
            pred_cols = [c for c in diet_df.columns if c != 'Group']
            
            for i in range(len(diet_df)):
                for j, pred in enumerate(pred_cols):
                    input_id = f"diet_{i}_{j}"
                    val = input[input_id]()
                    if val is not None:
                        diet_df.loc[i, pred] = val
            
            # Create updated params
            p.model = model_df
            p.diet = diet_df
            
            # Set defaults for missing values
            p.model['BioAcc'] = p.model['BioAcc'].fillna(0.0)
            p.model['Unassim'] = p.model['Unassim'].fillna(0.2)
            
            # Set detritus fate to first detritus group
            det_groups = p.model[p.model['Type'] == 2]['Group'].tolist()
            if det_groups:
                for det in det_groups:
                    if det in p.model.columns:
                        p.model[det] = p.model[det].fillna(1.0 / len(det_groups))
            
            # Balance the model
            model = rpath(p, eco_name=input.eco_name())
            balanced_model.set(model)
            model_data.set(model)
            
            ui.notification_show("Model balanced successfully!", type="message")
            
        except Exception as e:
            ui.notification_show(f"Error balancing model: {str(e)}", type="error")
    
    @output
    @render.ui
    def balance_status():
        """Show balance status."""
        model = balanced_model.get()
        if model is None:
            return ui.div(
                ui.tags.i(class_="bi bi-exclamation-circle me-2"),
                "Model not yet balanced. Enter parameters and click 'Balance Model'.",
                class_="alert alert-info"
            )
        
        # Check for issues
        ee_issues = np.sum((model.EE > 1) | (model.EE < 0))
        
        if ee_issues > 0:
            return ui.div(
                ui.tags.i(class_="bi bi-exclamation-triangle me-2"),
                f"Model balanced with warnings: {ee_issues} groups have EE outside [0,1]",
                class_="alert alert-warning"
            )
        
        return ui.div(
            ui.tags.i(class_="bi bi-check-circle me-2"),
            f"Model '{model.eco_name}' balanced successfully!",
            class_="alert alert-success"
        )
    
    @output
    @render.table
    def model_results_table():
        """Display balanced model results."""
        model = balanced_model.get()
        if model is None:
            return pd.DataFrame()
        
        return model.summary()
    
    @output
    @render.ui
    def diagnostics_output():
        """Display model diagnostics."""
        model = balanced_model.get()
        if model is None:
            return ui.p("Balance the model to see diagnostics.", class_="text-muted")
        
        # Calculate diagnostics
        total_biomass = np.sum(model.Biomass[:model.NUM_LIVING])
        total_production = np.sum(model.Biomass[:model.NUM_LIVING] * model.PB[:model.NUM_LIVING])
        
        return ui.div(
            ui.layout_columns(
                ui.value_box(
                    "Total Groups",
                    model.NUM_GROUPS,
                    showcase=ui.tags.i(class_="bi bi-diagram-3"),
                ),
                ui.value_box(
                    "Living Groups",
                    model.NUM_LIVING,
                    showcase=ui.tags.i(class_="bi bi-heart"),
                ),
                ui.value_box(
                    "Total Biomass",
                    f"{total_biomass:.2f}",
                    showcase=ui.tags.i(class_="bi bi-box"),
                ),
                ui.value_box(
                    "Total Production",
                    f"{total_production:.2f}",
                    showcase=ui.tags.i(class_="bi bi-arrow-up-circle"),
                ),
                col_widths=[3, 3, 3, 3]
            ),
        )
    
    @output
    @render.plot
    def trophic_level_plot():
        """Plot trophic levels."""
        import matplotlib.pyplot as plt
        
        model = balanced_model.get()
        if model is None:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No model data", ha='center', va='center')
            return fig
        
        fig, ax = plt.subplots(figsize=(8, 5))
        
        groups = model.Group[:model.NUM_LIVING + model.NUM_DEAD]
        tl = model.TL[:model.NUM_LIVING + model.NUM_DEAD]
        
        colors = ['#2ecc71' if t == 1 else '#3498db' if t < 2.5 else '#e74c3c' 
                  for t in tl]
        
        ax.barh(groups, tl, color=colors)
        ax.set_xlabel('Trophic Level')
        ax.set_title('Trophic Levels by Group')
        ax.axvline(x=1, color='gray', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        return fig
    
    @output
    @render.plot
    def ee_plot():
        """Plot ecotrophic efficiency."""
        import matplotlib.pyplot as plt
        
        model = balanced_model.get()
        if model is None:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No model data", ha='center', va='center')
            return fig
        
        fig, ax = plt.subplots(figsize=(8, 5))
        
        groups = model.Group[:model.NUM_LIVING + model.NUM_DEAD]
        ee = model.EE[:model.NUM_LIVING + model.NUM_DEAD]
        
        colors = ['#2ecc71' if 0 <= e <= 1 else '#e74c3c' for e in ee]
        
        ax.barh(groups, ee, color=colors)
        ax.set_xlabel('Ecotrophic Efficiency')
        ax.set_title('Ecotrophic Efficiency by Group')
        ax.axvline(x=1, color='red', linestyle='--', alpha=0.5, label='EE=1')
        ax.set_xlim(0, max(1.1, max(ee) * 1.1))
        
        plt.tight_layout()
        return fig
    
    @render.download(filename="pypath_params.csv")
    def download_params():
        """Download parameters as CSV."""
        p = params.get()
        if p is not None:
            return p.model.to_csv(index=False)
        return ""
