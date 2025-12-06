"""Ecosim simulation page module."""

from shiny import Inputs, Outputs, Session, reactive, render, ui, req
import pandas as pd
import numpy as np

# Import pypath
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from pypath.core.ecosim import (
    rsim_params, rsim_state, rsim_forcing, rsim_fishing,
    rsim_scenario, rsim_run, RsimScenario
)


def ecosim_ui():
    """Ecosim simulation page UI."""
    return ui.page_fluid(
        ui.h2("Ecosim Dynamic Simulation", class_="mb-4"),
        
        ui.layout_sidebar(
            # Sidebar for simulation settings
            ui.sidebar(
                ui.h4("Simulation Settings"),
                
                # Time settings
                ui.h5("Time Period"),
                ui.input_numeric(
                    "sim_years",
                    "Simulation Years",
                    value=50,
                    min=1,
                    max=500
                ),
                ui.input_select(
                    "integration_method",
                    "Integration Method",
                    choices={"RK4": "Runge-Kutta 4", "AB": "Adams-Bashforth"},
                    selected="RK4"
                ),
                
                ui.tags.hr(),
                
                # Vulnerability settings
                ui.h5("Functional Response"),
                ui.input_slider(
                    "vulnerability",
                    "Default Vulnerability",
                    min=1,
                    max=100,
                    value=2,
                    step=0.5
                ),
                ui.p(
                    "1 = Bottom-up, 2 = Mixed, High = Top-down",
                    class_="text-muted small"
                ),
                
                ui.tags.hr(),
                
                # Fishing scenarios
                ui.h5("Fishing Scenario"),
                ui.input_select(
                    "fishing_scenario",
                    "Scenario Type",
                    choices={
                        "baseline": "Baseline (constant effort)",
                        "increase": "Increase effort",
                        "decrease": "Decrease effort",
                        "closure": "Fishery closure",
                        "custom": "Custom"
                    },
                    selected="baseline"
                ),
                
                ui.output_ui("fishing_params_ui"),
                
                ui.tags.hr(),
                
                # Run buttons
                ui.input_action_button(
                    "btn_create_scenario",
                    "Create Scenario",
                    class_="btn-primary w-100"
                ),
                ui.input_action_button(
                    "btn_run_sim",
                    "Run Simulation",
                    class_="btn-success w-100 mt-2"
                ),
                
                width=300,
            ),
            
            # Main content
            ui.navset_card_tab(
                ui.nav_panel(
                    "Scenario Setup",
                    ui.h4("Scenario Configuration", class_="mt-3"),
                    ui.output_ui("scenario_status"),
                    
                    ui.layout_columns(
                        ui.card(
                            ui.card_header("Effort Forcing"),
                            ui.output_plot("effort_preview_plot"),
                        ),
                        ui.card(
                            ui.card_header("Biomass Forcing"),
                            ui.card_body(
                                ui.p("Configure environmental forcing on prey availability"),
                                ui.input_select(
                                    "forcing_group",
                                    "Select Group",
                                    choices=["(Create scenario first)"],
                                ),
                                ui.input_slider(
                                    "forcing_multiplier",
                                    "Forcing Multiplier",
                                    min=0,
                                    max=3,
                                    value=1,
                                    step=0.1
                                ),
                            ),
                        ),
                        col_widths=[6, 6]
                    ),
                ),
                ui.nav_panel(
                    "Simulation Progress",
                    ui.h4("Simulation Status", class_="mt-3"),
                    ui.output_ui("simulation_status"),
                    ui.output_ui("progress_display"),
                ),
                ui.nav_panel(
                    "Time Series",
                    ui.h4("Biomass Trajectories", class_="mt-3"),
                    ui.layout_columns(
                        ui.input_selectize(
                            "plot_groups",
                            "Select Groups to Plot",
                            choices=[],
                            multiple=True,
                        ),
                        ui.input_checkbox("relative_biomass", "Show Relative Biomass", value=False),
                        col_widths=[9, 3]
                    ),
                    ui.output_plot("biomass_timeseries", height="500px"),
                ),
                ui.nav_panel(
                    "Catch",
                    ui.h4("Catch Trajectories", class_="mt-3"),
                    ui.output_plot("catch_timeseries", height="400px"),
                    ui.output_table("annual_catch_table"),
                ),
                ui.nav_panel(
                    "Summary",
                    ui.h4("Simulation Summary", class_="mt-3"),
                    ui.output_ui("summary_cards"),
                    ui.layout_columns(
                        ui.output_plot("final_biomass_plot"),
                        ui.output_plot("biomass_change_plot"),
                        col_widths=[6, 6]
                    ),
                ),
            ),
        ),
    )


def ecosim_server(
    input: Inputs,
    output: Outputs,
    session: Session,
    model_data: reactive.Value,
    sim_results: reactive.Value
):
    """Ecosim simulation page server logic."""
    
    # Reactive values for this page
    scenario = reactive.Value(None)
    sim_output = reactive.Value(None)
    
    @output
    @render.ui
    def fishing_params_ui():
        """Dynamic UI for fishing scenario parameters."""
        scenario_type = input.fishing_scenario()
        
        if scenario_type == "baseline":
            return ui.p("Effort remains constant at baseline levels.", class_="text-muted")
        
        elif scenario_type == "increase":
            return ui.div(
                ui.input_slider(
                    "effort_change_rate",
                    "Annual Increase (%)",
                    min=0,
                    max=50,
                    value=5
                ),
                ui.input_numeric(
                    "effort_start_year",
                    "Start Year",
                    value=10,
                    min=1
                ),
            )
        
        elif scenario_type == "decrease":
            return ui.div(
                ui.input_slider(
                    "effort_change_rate",
                    "Annual Decrease (%)",
                    min=0,
                    max=50,
                    value=5
                ),
                ui.input_numeric(
                    "effort_start_year",
                    "Start Year",
                    value=10,
                    min=1
                ),
            )
        
        elif scenario_type == "closure":
            return ui.div(
                ui.input_numeric(
                    "closure_start_year",
                    "Closure Start Year",
                    value=10,
                    min=1
                ),
                ui.input_numeric(
                    "closure_duration",
                    "Duration (years)",
                    value=10,
                    min=1
                ),
            )
        
        else:  # custom
            return ui.p("Upload custom effort CSV or define in Results tab.", class_="text-muted")
    
    @reactive.effect
    @reactive.event(input.btn_create_scenario)
    def _create_scenario():
        """Create simulation scenario from model."""
        model = model_data.get()
        
        if model is None:
            ui.notification_show(
                "No Ecopath model available. Please balance a model first.",
                type="error"
            )
            return
        
        try:
            years = range(1, input.sim_years() + 1)
            
            # Create scenario
            # Need original params - create placeholder
            from pypath.core.params import create_rpath_params
            
            # Recreate params from model
            groups = list(model.Group)
            types = list(model.type)
            orig_params = create_rpath_params(groups, types)
            
            new_scenario = rsim_scenario(model, orig_params, years=years)
            
            # Apply fishing scenario
            _apply_fishing_scenario(new_scenario, input)
            
            scenario.set(new_scenario)
            
            # Update group choices
            group_names = list(model.Group[:model.NUM_LIVING + model.NUM_DEAD])
            ui.update_selectize("plot_groups", choices=group_names, selected=group_names[:3])
            ui.update_select("forcing_group", choices=group_names)
            
            ui.notification_show("Scenario created successfully!", type="message")
            
        except Exception as e:
            ui.notification_show(f"Error creating scenario: {str(e)}", type="error")
    
    def _apply_fishing_scenario(scen: RsimScenario, input: Inputs):
        """Apply fishing scenario settings to scenario."""
        scenario_type = input.fishing_scenario()
        n_months = scen.fishing.ForcedEffort.shape[0]
        n_gears = scen.fishing.ForcedEffort.shape[1]
        
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
                multiplier = max(0.01, 1.0 - rate * years_since)
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
                class_="alert alert-info"
            )
        
        return ui.div(
            ui.tags.i(class_="bi bi-check-circle me-2"),
            f"Scenario ready: {scen.params.NUM_GROUPS} groups, {input.sim_years()} years",
            class_="alert alert-success"
        )
    
    @output
    @render.plot
    def effort_preview_plot():
        """Preview effort forcing trajectory."""
        import matplotlib.pyplot as plt
        
        scen = scenario.get()
        
        fig, ax = plt.subplots(figsize=(8, 4))
        
        if scen is None:
            ax.text(0.5, 0.5, "Create scenario to preview effort", 
                   ha='center', va='center', transform=ax.transAxes)
            return fig
        
        effort = scen.fishing.ForcedEffort[:, 1]  # First fleet
        months = np.arange(len(effort)) / 12
        
        ax.plot(months, effort, 'b-', linewidth=2)
        ax.set_xlabel('Year')
        ax.set_ylabel('Effort Multiplier')
        ax.set_title('Fishing Effort Trajectory')
        ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlim(0, len(effort) / 12)
        ax.set_ylim(0, max(1.5, max(effort) * 1.1))
        
        plt.tight_layout()
        return fig
    
    @reactive.effect
    @reactive.event(input.btn_run_sim)
    def _run_simulation():
        """Run the Ecosim simulation."""
        scen = scenario.get()
        
        if scen is None:
            ui.notification_show("Create a scenario first", type="warning")
            return
        
        try:
            ui.notification_show("Running simulation...", type="message", duration=2)
            
            # Run simulation
            output = rsim_run(scen, method=input.integration_method())
            
            sim_output.set(output)
            sim_results.set(output)
            
            if output.crash_year > 0:
                ui.notification_show(
                    f"Simulation completed with crash at year {output.crash_year}",
                    type="warning"
                )
            else:
                ui.notification_show("Simulation completed successfully!", type="message")
            
        except Exception as e:
            ui.notification_show(f"Simulation error: {str(e)}", type="error")
    
    @output
    @render.ui
    def simulation_status():
        """Display simulation status."""
        output = sim_output.get()
        
        if output is None:
            return ui.div(
                ui.tags.i(class_="bi bi-hourglass me-2"),
                "Simulation not yet run. Create scenario and click 'Run Simulation'.",
                class_="alert alert-secondary"
            )
        
        if output.crash_year > 0:
            return ui.div(
                ui.tags.i(class_="bi bi-exclamation-triangle me-2"),
                f"Simulation crashed at year {output.crash_year}. Check results for details.",
                class_="alert alert-warning"
            )
        
        return ui.div(
            ui.tags.i(class_="bi bi-check-circle me-2"),
            f"Simulation completed: {output.params['years']} years simulated",
            class_="alert alert-success"
        )
    
    @output
    @render.ui
    def progress_display():
        """Display progress/completion info."""
        output = sim_output.get()
        if output is None:
            return None
        
        return ui.div(
            ui.tags.h5("Simulation Details"),
            ui.tags.table(
                ui.tags.tr(ui.tags.td("Years simulated:"), ui.tags.td(str(output.params['years']))),
                ui.tags.tr(ui.tags.td("Groups:"), ui.tags.td(str(output.params['NUM_GROUPS']))),
                ui.tags.tr(ui.tags.td("Living groups:"), ui.tags.td(str(output.params['NUM_LIVING']))),
                ui.tags.tr(ui.tags.td("Crash year:"), 
                          ui.tags.td("None" if output.crash_year < 0 else str(output.crash_year))),
                class_="table table-sm"
            ),
        )
    
    @output
    @render.plot
    def biomass_timeseries():
        """Plot biomass time series."""
        import matplotlib.pyplot as plt
        
        output = sim_output.get()
        scen = scenario.get()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        if output is None or scen is None:
            ax.text(0.5, 0.5, "Run simulation to see results", 
                   ha='center', va='center', transform=ax.transAxes)
            return fig
        
        selected_groups = input.plot_groups()
        if not selected_groups:
            ax.text(0.5, 0.5, "Select groups to plot", 
                   ha='center', va='center', transform=ax.transAxes)
            return fig
        
        # Get time and biomass data
        n_months = output.out_Biomass.shape[0]
        time = np.arange(n_months) / 12
        
        # Get group indices
        group_names = scen.params.spname[1:]  # Skip "Outside"
        
        for group in selected_groups:
            if group in group_names:
                idx = group_names.index(group) + 1  # +1 for Outside offset
                
                biomass = output.out_Biomass[:, idx]
                
                if input.relative_biomass():
                    if biomass[0] > 0:
                        biomass = biomass / biomass[0]
                    else:
                        biomass = np.zeros_like(biomass)
                
                ax.plot(time, biomass, label=group, linewidth=2)
        
        ax.set_xlabel('Year')
        ax.set_ylabel('Relative Biomass' if input.relative_biomass() else 'Biomass')
        ax.set_title('Biomass Trajectories')
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        ax.set_xlim(0, max(time))
        
        if input.relative_biomass():
            ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        return fig
    
    @output
    @render.plot
    def catch_timeseries():
        """Plot catch time series."""
        import matplotlib.pyplot as plt
        
        output = sim_output.get()
        scen = scenario.get()
        
        fig, ax = plt.subplots(figsize=(12, 5))
        
        if output is None or scen is None:
            ax.text(0.5, 0.5, "Run simulation to see catch data", 
                   ha='center', va='center', transform=ax.transAxes)
            return fig
        
        # Annual catch data
        years = np.arange(output.annual_Catch.shape[0]) + 1
        total_catch = np.sum(output.annual_Catch[:, 1:], axis=1)
        
        ax.fill_between(years, total_catch, alpha=0.3)
        ax.plot(years, total_catch, 'b-', linewidth=2, label='Total Catch')
        
        ax.set_xlabel('Year')
        ax.set_ylabel('Catch')
        ax.set_title('Total Annual Catch')
        ax.set_xlim(1, len(years))
        
        plt.tight_layout()
        return fig
    
    @output
    @render.table
    def annual_catch_table():
        """Display annual catch summary."""
        output = sim_output.get()
        scen = scenario.get()
        
        if output is None or scen is None:
            return pd.DataFrame()
        
        # Create summary table
        group_names = scen.params.spname[1:scen.params.NUM_LIVING + 1]
        
        catch_df = pd.DataFrame(
            output.annual_Catch[:, 1:scen.params.NUM_LIVING + 1],
            columns=group_names
        )
        catch_df.insert(0, 'Year', range(1, len(catch_df) + 1))
        
        # Show every 5 years
        return catch_df[catch_df['Year'] % 5 == 0].round(3)
    
    @output
    @render.ui
    def summary_cards():
        """Display summary statistics cards."""
        output = sim_output.get()
        scen = scenario.get()
        
        if output is None or scen is None:
            return ui.p("Run simulation to see summary.", class_="text-muted")
        
        # Calculate summary stats
        initial_biomass = np.sum(output.out_Biomass[0, 1:scen.params.NUM_LIVING + 1])
        final_biomass = np.sum(output.out_Biomass[-1, 1:scen.params.NUM_LIVING + 1])
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
                theme="primary" if biomass_change >= 0 else "danger"
            ),
            ui.value_box(
                "Biomass Change",
                f"{biomass_change:+.1f}%",
                showcase=ui.tags.i(class_="bi bi-arrow-up" if biomass_change >= 0 else "bi bi-arrow-down"),
                theme="success" if biomass_change >= 0 else "danger"
            ),
            ui.value_box(
                "Total Catch",
                f"{total_catch:.2f}",
                showcase=ui.tags.i(class_="bi bi-basket"),
            ),
            col_widths=[3, 3, 3, 3]
        )
    
    @output
    @render.plot
    def final_biomass_plot():
        """Plot final vs initial biomass."""
        import matplotlib.pyplot as plt
        
        output = sim_output.get()
        scen = scenario.get()
        
        fig, ax = plt.subplots(figsize=(8, 5))
        
        if output is None or scen is None:
            ax.text(0.5, 0.5, "Run simulation first", 
                   ha='center', va='center', transform=ax.transAxes)
            return fig
        
        n_living = scen.params.NUM_LIVING
        group_names = scen.params.spname[1:n_living + 1]
        
        initial = output.out_Biomass[0, 1:n_living + 1]
        final = output.out_Biomass[-1, 1:n_living + 1]
        
        x = np.arange(len(group_names))
        width = 0.35
        
        ax.bar(x - width/2, initial, width, label='Initial', color='#3498db')
        ax.bar(x + width/2, final, width, label='Final', color='#2ecc71')
        
        ax.set_xlabel('Group')
        ax.set_ylabel('Biomass')
        ax.set_title('Initial vs Final Biomass')
        ax.set_xticks(x)
        ax.set_xticklabels(group_names, rotation=45, ha='right')
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    @output
    @render.plot
    def biomass_change_plot():
        """Plot percent biomass change."""
        import matplotlib.pyplot as plt
        
        output = sim_output.get()
        scen = scenario.get()
        
        fig, ax = plt.subplots(figsize=(8, 5))
        
        if output is None or scen is None:
            ax.text(0.5, 0.5, "Run simulation first", 
                   ha='center', va='center', transform=ax.transAxes)
            return fig
        
        n_living = scen.params.NUM_LIVING
        group_names = scen.params.spname[1:n_living + 1]
        
        initial = output.out_Biomass[0, 1:n_living + 1]
        final = output.out_Biomass[-1, 1:n_living + 1]
        
        pct_change = np.where(initial > 0, (final - initial) / initial * 100, 0)
        
        colors = ['#2ecc71' if c >= 0 else '#e74c3c' for c in pct_change]
        
        ax.barh(group_names, pct_change, color=colors)
        ax.axvline(x=0, color='gray', linestyle='-')
        ax.set_xlabel('Percent Change (%)')
        ax.set_title('Biomass Change by Group')
        
        plt.tight_layout()
        return fig
