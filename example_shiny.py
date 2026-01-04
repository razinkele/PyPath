import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shinyswatch
from shiny import App, reactive, render, ui

# --- UI Definition ---
app_ui = ui.page_sidebar(
    # 1. The Left Sidebar
    ui.sidebar(
        ui.h4("Control Panel"),
        ui.hr(),
        ui.input_select("region", "Select Region:", ["North America", "Europe", "Asia"]),
        ui.input_slider("n", "Data Points", 10, 100, 50),
        ui.hr(),
        ui.input_action_button("reset", "Reset View", class_="btn-primary w-100"),
        title="App Menu",
        width=300
    ),

    # 2. Top Navigation Bar (Within the main area)
    ui.navset_bar(
        # Page 1
        ui.nav_panel("Analytics",
            ui.layout_columns(
                ui.value_box(
                    "Selected Region",
                    ui.output_text("txt_region"),
                    show_full_screen=True
                ),
                ui.value_box(
                    "Current Mean",
                    ui.output_text("txt_mean"),
                    show_full_screen=True
                ),
                fill=False
            ),
            ui.card(
                ui.card_header("Performance Visualization"),
                ui.output_plot("main_plot"),
                full_screen=True
            )
        ),

        # Page 2
        ui.nav_panel("Data Explorer",
            ui.card(
                ui.card_header("Raw Dataset"),
                ui.output_data_frame("data_table")
            )
        ),

        title="Project Nexus",
        id="main_nav"
    ),

    # Applying the theme
    theme=shinyswatch.theme.flatly,
    title="Core Shiny Dashboard"
)

# --- Server Logic ---
def server(input, output, session):

    # Reactive calculation for data
    @reactive.calc
    def filtered_data():
        # Create dummy data based on inputs
        np.random.seed(42)
        data = np.random.randn(input.n())
        return pd.DataFrame({"Value": data, "Index": range(len(data))})

    @render.text
    def txt_region():
        return input.region()

    @render.text
    def txt_mean():
        val = filtered_data()["Value"].mean()
        return f"{val:.2f}"

    @render.plot
    def main_plot():
        df = filtered_data()
        fig, ax = plt.subplots()
        ax.plot(df["Index"], df["Value"], marker='o', color='#2c3e50')
        ax.set_title(f"Trend for {input.region()}")
        ax.grid(True, alpha=0.3)
        return fig

    @render.data_frame
    def data_table():
        return filtered_data()

# --- App Initialization ---
app = App(app_ui, server)
