"""Results and visualization page module."""

from shiny import Inputs, Outputs, Session, reactive, render, ui, req
import pandas as pd
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def results_ui():
    """Results page UI."""
    return ui.page_fluid(
        ui.h2("Results & Visualization", class_="mb-4"),
        
        ui.layout_sidebar(
            ui.sidebar(
                ui.h4("Export Options"),
                
                ui.h5("Model Results"),
                ui.download_button(
                    "download_model_csv",
                    "Download Model (CSV)",
                    class_="btn-outline-primary w-100 mb-2"
                ),
                ui.download_button(
                    "download_model_excel",
                    "Download Model (Excel)",
                    class_="btn-outline-primary w-100 mb-2"
                ),
                
                ui.tags.hr(),
                
                ui.h5("Simulation Results"),
                ui.download_button(
                    "download_sim_csv",
                    "Download Simulation (CSV)",
                    class_="btn-outline-success w-100 mb-2"
                ),
                ui.download_button(
                    "download_annual_csv",
                    "Download Annual Summary",
                    class_="btn-outline-success w-100 mb-2"
                ),
                
                ui.tags.hr(),
                
                ui.h5("Plot Settings"),
                ui.input_select(
                    "plot_style",
                    "Plot Style",
                    choices={
                        "default": "Default",
                        "seaborn": "Seaborn",
                        "ggplot": "GGPlot",
                        "dark": "Dark Background"
                    }
                ),
                ui.input_select(
                    "color_palette",
                    "Color Palette",
                    choices={
                        "tab10": "Tab10",
                        "Set2": "Set2",
                        "Paired": "Paired",
                        "husl": "HUSL"
                    }
                ),
                
                width=280,
            ),
            
            # Main content
            ui.navset_card_tab(
                ui.nav_panel(
                    "Model Summary",
                    ui.h4("Ecopath Model Summary", class_="mt-3"),
                    ui.output_ui("model_summary_status"),
                    ui.output_table("model_summary_table"),
                    
                    ui.tags.hr(),
                    
                    ui.h5("Trophic Structure"),
                    ui.layout_columns(
                        ui.output_plot("tl_bar_plot"),
                        ui.output_plot("tl_flow_plot"),
                        col_widths=[6, 6]
                    ),
                ),
                ui.nav_panel(
                    "Food Web",
                    ui.h4("Food Web Diagram", class_="mt-3"),
                    ui.layout_columns(
                        ui.div(
                            ui.input_slider(
                                "foodweb_min_flow",
                                "Minimum Flow to Show",
                                min=0,
                                max=1,
                                value=0.01,
                                step=0.01
                            ),
                            ui.input_checkbox("show_biomass_size", "Scale nodes by biomass", value=True),
                            ui.input_checkbox("show_flow_width", "Scale edges by flow", value=True),
                        ),
                        col_widths=[12]
                    ),
                    ui.output_plot("foodweb_plot", height="600px"),
                ),
                ui.nav_panel(
                    "Simulation Results",
                    ui.h4("Ecosim Simulation Results", class_="mt-3"),
                    ui.output_ui("sim_summary_status"),
                    
                    ui.h5("Biomass Time Series"),
                    ui.layout_columns(
                        ui.input_selectize(
                            "results_groups",
                            "Select Groups",
                            choices=[],
                            multiple=True,
                        ),
                        ui.input_checkbox("log_scale", "Log Scale", value=False),
                        ui.input_checkbox("show_uncertainty", "Show Uncertainty", value=False),
                        col_widths=[6, 3, 3]
                    ),
                    ui.output_plot("results_biomass_plot", height="450px"),
                    
                    ui.tags.hr(),
                    
                    ui.h5("Catch Time Series"),
                    ui.output_plot("results_catch_plot", height="350px"),
                ),
                ui.nav_panel(
                    "Comparison",
                    ui.h4("Scenario Comparison", class_="mt-3"),
                    ui.p("Compare results from different scenarios (future feature)", class_="text-muted"),
                    
                    ui.layout_columns(
                        ui.card(
                            ui.card_header("Scenario A"),
                            ui.card_body(
                                ui.input_file("upload_scenario_a", "Upload Results A"),
                            ),
                        ),
                        ui.card(
                            ui.card_header("Scenario B"),
                            ui.card_body(
                                ui.input_file("upload_scenario_b", "Upload Results B"),
                            ),
                        ),
                        col_widths=[6, 6]
                    ),
                    
                    ui.output_plot("comparison_plot", height="400px"),
                ),
                ui.nav_panel(
                    "Data Tables",
                    ui.h4("Raw Data Tables", class_="mt-3"),
                    
                    ui.navset_card_pill(
                        ui.nav_panel(
                            "Parameters",
                            ui.output_data_frame("params_data_table"),
                        ),
                        ui.nav_panel(
                            "Diet Matrix",
                            ui.output_data_frame("diet_data_table"),
                        ),
                        ui.nav_panel(
                            "Biomass (Monthly)",
                            ui.output_data_frame("biomass_data_table"),
                        ),
                        ui.nav_panel(
                            "Catch (Annual)",
                            ui.output_data_frame("catch_data_table"),
                        ),
                    ),
                ),
            ),
        ),
    )


def results_server(
    input: Inputs,
    output: Outputs,
    session: Session,
    model_data: reactive.Value,
    sim_results: reactive.Value
):
    """Results page server logic."""
    
    @output
    @render.ui
    def model_summary_status():
        """Display model status."""
        model = model_data.get()
        
        if model is None:
            return ui.div(
                ui.tags.i(class_="bi bi-info-circle me-2"),
                "No model data available. Create and balance an Ecopath model first.",
                class_="alert alert-info"
            )
        
        return ui.div(
            ui.tags.i(class_="bi bi-check-circle me-2"),
            f"Model: {model.eco_name} ({model.NUM_GROUPS} groups)",
            class_="alert alert-success"
        )
    
    @output
    @render.table
    def model_summary_table():
        """Display model summary table."""
        model = model_data.get()
        
        if model is None:
            return pd.DataFrame()
        
        return model.summary()
    
    @output
    @render.plot
    def tl_bar_plot():
        """Trophic level bar plot."""
        import matplotlib.pyplot as plt
        
        model = model_data.get()
        
        fig, ax = plt.subplots(figsize=(8, 5))
        
        if model is None:
            ax.text(0.5, 0.5, "No model data", ha='center', va='center', transform=ax.transAxes)
            return fig
        
        # Apply plot style
        style = input.plot_style()
        if style != "default":
            plt.style.use(style)
        
        n_bio = model.NUM_LIVING + model.NUM_DEAD
        groups = model.Group[:n_bio]
        tl = model.TL[:n_bio]
        
        # Sort by TL
        sorted_idx = np.argsort(tl)[::-1]
        groups = [groups[i] for i in sorted_idx]
        tl = tl[sorted_idx]
        
        colors = plt.cm.get_cmap(input.color_palette())(np.linspace(0, 1, len(groups)))
        
        ax.barh(groups, tl, color=colors)
        ax.set_xlabel('Trophic Level')
        ax.set_title('Trophic Levels')
        ax.axvline(x=1, color='gray', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        return fig
    
    @output
    @render.plot
    def tl_flow_plot():
        """Trophic flow pyramid."""
        import matplotlib.pyplot as plt
        
        model = model_data.get()
        
        fig, ax = plt.subplots(figsize=(8, 5))
        
        if model is None:
            ax.text(0.5, 0.5, "No model data", ha='center', va='center', transform=ax.transAxes)
            return fig
        
        n_living = model.NUM_LIVING
        
        # Group by trophic level bins
        tl = model.TL[:n_living]
        biomass = model.Biomass[:n_living]
        production = biomass * model.PB[:n_living]
        
        # Create TL bins
        tl_bins = [1, 2, 3, 4, 5, 6]
        tl_labels = ['TL 1-2', 'TL 2-3', 'TL 3-4', 'TL 4-5', 'TL 5+']
        
        prod_by_tl = []
        for i in range(len(tl_bins) - 1):
            mask = (tl >= tl_bins[i]) & (tl < tl_bins[i + 1])
            prod_by_tl.append(np.sum(production[mask]))
        
        # Pyramid plot (horizontal bars, smallest at top)
        y_pos = np.arange(len(tl_labels))
        
        colors = plt.cm.get_cmap('YlGn')(np.linspace(0.3, 0.9, len(tl_labels)))
        
        ax.barh(y_pos, prod_by_tl, color=colors, height=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(tl_labels)
        ax.set_xlabel('Production')
        ax.set_title('Trophic Pyramid (Production)')
        ax.invert_yaxis()
        
        plt.tight_layout()
        return fig
    
    @output
    @render.plot
    def foodweb_plot():
        """Food web diagram."""
        import matplotlib.pyplot as plt
        
        model = model_data.get()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        if model is None:
            ax.text(0.5, 0.5, "No model data", ha='center', va='center', transform=ax.transAxes)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            return fig
        
        try:
            import networkx as nx
        except ImportError:
            ax.text(0.5, 0.5, "NetworkX not installed.\nInstall with: pip install networkx", 
                   ha='center', va='center', transform=ax.transAxes)
            return fig
        
        n_bio = model.NUM_LIVING + model.NUM_DEAD
        n_living = model.NUM_LIVING
        
        # Create graph
        G = nx.DiGraph()
        
        # Add nodes
        for i in range(n_bio):
            G.add_node(model.Group[i], 
                      tl=model.TL[i], 
                      biomass=model.Biomass[i])
        
        # Add edges from diet matrix
        min_flow = input.foodweb_min_flow()
        
        for pred_idx in range(n_living):
            pred = model.Group[pred_idx]
            for prey_idx in range(n_bio):
                if prey_idx < model.DC.shape[0] and pred_idx < model.DC.shape[1]:
                    flow = model.DC[prey_idx, pred_idx]
                    if flow > min_flow:
                        prey = model.Group[prey_idx]
                        G.add_edge(prey, pred, weight=flow)
        
        if len(G.edges()) == 0:
            ax.text(0.5, 0.5, "No diet connections found.\nCheck diet matrix.", 
                   ha='center', va='center', transform=ax.transAxes)
            return fig
        
        # Layout based on trophic level
        pos = {}
        tl_groups = {}
        for node in G.nodes():
            tl = int(G.nodes[node]['tl'])
            if tl not in tl_groups:
                tl_groups[tl] = []
            tl_groups[tl].append(node)
        
        for tl, nodes in tl_groups.items():
            n = len(nodes)
            for i, node in enumerate(nodes):
                x = (i + 1) / (n + 1)
                pos[node] = (x, tl)
        
        # Node sizes based on biomass
        if input.show_biomass_size():
            node_sizes = [300 + G.nodes[n]['biomass'] * 50 for n in G.nodes()]
        else:
            node_sizes = [500] * len(G.nodes())
        
        # Edge widths based on flow
        if input.show_flow_width():
            edge_widths = [G.edges[e]['weight'] * 3 for e in G.edges()]
        else:
            edge_widths = [1] * len(G.edges())
        
        # Colors by trophic level
        node_colors = [G.nodes[n]['tl'] for n in G.nodes()]
        
        # Draw
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes, 
                              node_color=node_colors, cmap='YlGnBu',
                              alpha=0.8)
        nx.draw_networkx_edges(G, pos, ax=ax, width=edge_widths,
                              alpha=0.5, arrows=True, 
                              connectionstyle="arc3,rad=0.1")
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=9)
        
        ax.set_title('Food Web Structure')
        ax.set_ylabel('Trophic Level')
        ax.set_xlim(-0.1, 1.1)
        
        plt.tight_layout()
        return fig
    
    @output
    @render.ui
    def sim_summary_status():
        """Display simulation status."""
        sim = sim_results.get()
        
        if sim is None:
            return ui.div(
                ui.tags.i(class_="bi bi-info-circle me-2"),
                "No simulation results available. Run an Ecosim simulation first.",
                class_="alert alert-info"
            )
        
        return ui.div(
            ui.tags.i(class_="bi bi-check-circle me-2"),
            f"Simulation: {sim.params['years']} years, {sim.params['NUM_GROUPS']} groups",
            class_="alert alert-success"
        )
    
    @reactive.effect
    def _update_group_choices():
        """Update group choices when simulation results change."""
        sim = sim_results.get()
        model = model_data.get()
        
        if model is not None:
            groups = list(model.Group[:model.NUM_LIVING + model.NUM_DEAD])
            ui.update_selectize("results_groups", choices=groups, selected=groups[:3])
    
    @output
    @render.plot
    def results_biomass_plot():
        """Simulation biomass results plot."""
        import matplotlib.pyplot as plt
        
        sim = sim_results.get()
        model = model_data.get()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        if sim is None or model is None:
            ax.text(0.5, 0.5, "No simulation data", ha='center', va='center', transform=ax.transAxes)
            return fig
        
        selected = input.results_groups()
        if not selected:
            ax.text(0.5, 0.5, "Select groups to display", ha='center', va='center', transform=ax.transAxes)
            return fig
        
        # Apply plot style
        style = input.plot_style()
        if style != "default":
            try:
                plt.style.use(style)
            except:
                pass
        
        n_months = sim.out_Biomass.shape[0]
        time = np.arange(n_months) / 12
        
        group_list = list(model.Group)
        colors = plt.cm.get_cmap(input.color_palette())(np.linspace(0, 1, len(selected)))
        
        for i, group in enumerate(selected):
            if group in group_list:
                idx = group_list.index(group) + 1
                biomass = sim.out_Biomass[:, idx]
                
                if input.log_scale():
                    biomass = np.log10(np.maximum(biomass, 1e-10))
                
                ax.plot(time, biomass, label=group, color=colors[i], linewidth=2)
        
        ax.set_xlabel('Year')
        ax.set_ylabel('Log10(Biomass)' if input.log_scale() else 'Biomass')
        ax.set_title('Biomass Time Series')
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        ax.set_xlim(0, max(time))
        
        plt.tight_layout()
        return fig
    
    @output
    @render.plot
    def results_catch_plot():
        """Simulation catch results plot."""
        import matplotlib.pyplot as plt
        
        sim = sim_results.get()
        
        fig, ax = plt.subplots(figsize=(12, 5))
        
        if sim is None:
            ax.text(0.5, 0.5, "No simulation data", ha='center', va='center', transform=ax.transAxes)
            return fig
        
        years = np.arange(sim.annual_Catch.shape[0]) + 1
        total_catch = np.sum(sim.annual_Catch[:, 1:], axis=1)
        
        ax.fill_between(years, total_catch, alpha=0.3, color='steelblue')
        ax.plot(years, total_catch, 'b-', linewidth=2)
        
        ax.set_xlabel('Year')
        ax.set_ylabel('Total Catch')
        ax.set_title('Annual Catch')
        ax.set_xlim(1, len(years))
        
        plt.tight_layout()
        return fig
    
    @output
    @render.plot
    def comparison_plot():
        """Scenario comparison plot."""
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "Upload scenarios to compare", ha='center', va='center', transform=ax.transAxes)
        return fig
    
    @output
    @render.data_frame
    def params_data_table():
        """Model parameters data table."""
        model = model_data.get()
        if model is None:
            return pd.DataFrame()
        return model.summary()
    
    @output
    @render.data_frame
    def diet_data_table():
        """Diet matrix data table."""
        model = model_data.get()
        if model is None:
            return pd.DataFrame()
        
        n_living = model.NUM_LIVING
        n_bio = model.NUM_LIVING + model.NUM_DEAD
        
        diet_df = pd.DataFrame(
            model.DC[:n_bio, :n_living],
            index=model.Group[:n_bio],
            columns=model.Group[:n_living]
        ).round(4)
        diet_df.insert(0, 'Prey', diet_df.index)
        
        return diet_df
    
    @output
    @render.data_frame
    def biomass_data_table():
        """Monthly biomass data table."""
        sim = sim_results.get()
        model = model_data.get()
        
        if sim is None or model is None:
            return pd.DataFrame()
        
        n_bio = model.NUM_LIVING + model.NUM_DEAD
        
        df = pd.DataFrame(
            sim.out_Biomass[:, 1:n_bio + 1],
            columns=model.Group[:n_bio]
        ).round(4)
        df.insert(0, 'Month', range(len(df)))
        df.insert(1, 'Year', df['Month'] / 12)
        
        # Show every 12 months
        return df[df['Month'] % 12 == 0]
    
    @output
    @render.data_frame
    def catch_data_table():
        """Annual catch data table."""
        sim = sim_results.get()
        model = model_data.get()
        
        if sim is None or model is None:
            return pd.DataFrame()
        
        n_living = model.NUM_LIVING
        
        df = pd.DataFrame(
            sim.annual_Catch[:, 1:n_living + 1],
            columns=model.Group[:n_living]
        ).round(4)
        df.insert(0, 'Year', range(1, len(df) + 1))
        
        return df
    
    @render.download(filename="pypath_model.csv")
    def download_model_csv():
        """Download model as CSV."""
        model = model_data.get()
        if model is not None:
            return model.summary().to_csv(index=False)
        return ""
    
    @render.download(filename="pypath_simulation.csv")
    def download_sim_csv():
        """Download simulation results as CSV."""
        sim = sim_results.get()
        model = model_data.get()
        
        if sim is not None and model is not None:
            n_bio = model.NUM_LIVING + model.NUM_DEAD
            df = pd.DataFrame(
                sim.out_Biomass[:, 1:n_bio + 1],
                columns=model.Group[:n_bio]
            )
            df.insert(0, 'Month', range(len(df)))
            return df.to_csv(index=False)
        return ""
    
    @render.download(filename="pypath_annual_summary.csv")
    def download_annual_csv():
        """Download annual summary as CSV."""
        sim = sim_results.get()
        model = model_data.get()
        
        if sim is not None and model is not None:
            n_living = model.NUM_LIVING
            df = pd.DataFrame(
                sim.annual_Catch[:, 1:n_living + 1],
                columns=model.Group[:n_living]
            )
            df.insert(0, 'Year', range(1, len(df) + 1))
            return df.to_csv(index=False)
        return ""
