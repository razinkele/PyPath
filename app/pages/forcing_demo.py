"""
State-Variable Forcing Demonstration Page

Interactive demonstration of forcing state variables to observed or prescribed time series.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from shiny import Inputs, Outputs, Session, reactive, render, ui

# pypath imports (path setup handled by app/__init__.py)
from pypath.core.forcing import (
    StateForcing,
    create_biomass_forcing,
    create_recruitment_forcing,
)

# Configuration imports
try:
    from app.config import PARAM_RANGES
except ModuleNotFoundError:
    from config import PARAM_RANGES


def forcing_demo_ui():
    """UI for forcing demonstration page."""
    return ui.page_fluid(
        ui.layout_sidebar(
            ui.sidebar(
                ui.h4("Forcing Configuration"),
                ui.input_select(
                    "forcing_type",
                    "Forcing Type",
                    choices={
                        "biomass": "Biomass Forcing",
                        "recruitment": "Recruitment Forcing",
                        "fishing": "Fishing Mortality",
                        "primary_production": "Primary Production",
                    },
                    selected="biomass",
                ),
                ui.input_select(
                    "forcing_mode",
                    "Forcing Mode",
                    choices={
                        "replace": "REPLACE - Override computed value",
                        "add": "ADD - Add to computed value",
                        "multiply": "MULTIPLY - Multiply computed value",
                        "rescale": "RESCALE - Rescale to target",
                    },
                    selected="replace",
                ),
                ui.input_numeric("group_idx", "Group Index", value=0, min=0, max=20),
                ui.hr(),
                ui.h5("Time Series Pattern"),
                ui.input_select(
                    "pattern_type",
                    "Pattern",
                    choices={
                        "seasonal": "Seasonal Cycle",
                        "trend": "Linear Trend",
                        "pulse": "Recruitment Pulses",
                        "step": "Step Change",
                        "custom": "Custom Values",
                    },
                    selected="seasonal",
                ),
                ui.panel_conditional(
                    "input.pattern_type === 'seasonal'",
                    ui.input_slider(
                        "seasonal_amplitude",
                        "Amplitude",
                        min=0.1,
                        max=PARAM_RANGES.seasonal_amplitude_max,
                        value=0.5,
                        step=0.1,
                    ),
                    ui.input_numeric(
                        "seasonal_baseline",
                        "Baseline Value",
                        value=PARAM_RANGES.seasonal_baseline_default,
                        min=0.1,
                        step=0.5,
                    ),
                ),
                ui.panel_conditional(
                    "input.pattern_type === 'trend'",
                    ui.input_numeric("trend_start", "Start Value", value=10.0, min=0.1),
                    ui.input_numeric("trend_end", "End Value", value=20.0, min=0.1),
                ),
                ui.panel_conditional(
                    "input.pattern_type === 'pulse'",
                    ui.input_slider(
                        "pulse_strength",
                        "Pulse Strength (multiplier)",
                        min=PARAM_RANGES.pulse_strength_min,
                        max=PARAM_RANGES.pulse_strength_max,
                        value=PARAM_RANGES.pulse_strength_default,
                        step=0.1,
                    ),
                ),
                ui.hr(),
                ui.input_action_button(
                    "generate_forcing", "Generate Forcing", class_="btn-primary w-100"
                ),
                ui.input_action_button(
                    "forcing_run_demo",
                    "Run Demo Simulation",
                    class_="btn-success w-100 mt-2",
                ),
                width=300,
            ),
            # Main content
            ui.navset_tab(
                ui.nav_panel(
                    "Forcing Time Series",
                    ui.card(
                        ui.card_header("Forced Values Over Time"),
                        ui.output_ui("forcing_plot"),
                        ui.output_text_verbatim("forcing_summary"),
                    ),
                ),
                ui.nav_panel(
                    "Simulation Comparison",
                    ui.card(
                        ui.card_header("Effect of Forcing on Simulation"),
                        ui.output_ui("forcing_comparison_plot"),
                        ui.markdown(
                            """
                        **Blue**: Standard simulation (no forcing)

                        **Red**: Simulation with forcing applied

                        **Forcing Effect**: Shows how forcing modifies the baseline simulation
                        """
                        ),
                    ),
                ),
                ui.nav_panel(
                    "Code Example",
                    ui.card(
                        ui.card_header("Python Code for This Configuration"),
                        ui.output_code("forcing_code_example"),
                        ui.download_button(
                            "forcing_download_code", "Download Code", class_="mt-2"
                        ),
                    ),
                ),
                ui.nav_panel(
                    "Use Cases",
                    ui.card(
                        ui.card_header(
                            ui.tags.i(class_="bi bi-lightbulb me-2"),
                            "State-Variable Forcing Use Cases",
                        ),
                        ui.markdown(
                            """
                        ## What is State-Variable Forcing?

                        State-variable forcing allows you to **override computed values** with observed
                        or prescribed values. This is a powerful technique for:

                        ### 1. Calibration to Observations

                        **Example**: Force phytoplankton biomass to satellite chlorophyll data

                        ```python
                        forcing = create_biomass_forcing(
                            group_idx=0,
                            observed_biomass=satellite_data,
                            mode='replace'
                        )
                        ```

                        **When to use**:
                        - You have reliable observations
                        - Want to match empirical patterns
                        - Testing bottom-up vs top-down control

                        ### 2. Climate Change Scenarios

                        **Example**: Gradually increase primary production

                        ```python
                        forcing = StateForcing()
                        forcing.add_forcing(
                            group_idx=0,
                            variable='primary_production',
                            time_series={2000: 1.0, 2050: 1.3, 2100: 1.6},
                            mode='multiply'
                        )
                        ```

                        **When to use**:
                        - Climate-driven changes
                        - Temperature effects
                        - Ocean acidification impacts

                        ### 3. Fishing Management

                        **Example**: Fishing moratorium

                        ```python
                        forcing = StateForcing()
                        forcing.add_forcing(
                            group_idx=5,
                            variable='fishing_mortality',
                            time_series={2000: 0.3, 2010: 0.0, 2020: 0.15},
                            mode='replace'
                        )
                        ```

                        **When to use**:
                        - Marine protected areas
                        - Fishing bans
                        - Quota changes
                        - Recovery scenarios

                        ### 4. Recruitment Variability

                        **Example**: Strong and weak year-classes

                        ```python
                        forcing = create_recruitment_forcing(
                            group_idx=3,
                            recruitment_multiplier={2005: 3.0, 2010: 0.5},
                            interpolate=False
                        )
                        ```

                        **When to use**:
                        - Environmental recruitment drivers
                        - Regime shifts
                        - El Niño effects
                        - Temperature-dependent recruitment

                        ### 5. Hybrid Models

                        Combine empirical data with process-based simulation

                        **When to use**:
                        - Mix observations with mechanistic models
                        - Data-rich for some variables, not others
                        - Testing specific hypotheses
                        - Filling knowledge gaps

                        ## Forcing Modes Explained

                        ### REPLACE Mode
                        - **What it does**: Completely overrides computed value
                        - **Use when**: You have reliable absolute measurements
                        - **Example**: Satellite biomass, catch statistics

                        ### ADD Mode
                        - **What it does**: Adds to computed value
                        - **Use when**: Representing additive processes
                        - **Example**: Immigration, supplementation

                        ### MULTIPLY Mode
                        - **What it does**: Scales computed value
                        - **Use when**: Representing proportional changes
                        - **Example**: Environmental multipliers, recruitment strength

                        ### RESCALE Mode
                        - **What it does**: Rescales to match target while preserving proportions
                        - **Use when**: Want to match total while keeping ratios
                        - **Example**: Total catch with fleet proportions

                        ## Best Practices

                        ✅ **DO:**
                        - Validate forced data quality
                        - Check for ecological realism
                        - Document assumptions
                        - Test sensitivity to forcing
                        - Compare with and without forcing

                        ⚠️ **DON'T:**
                        - Force too many variables (reduces model predictive power)
                        - Ignore mass balance violations
                        - Use poor quality data
                        - Forget to document what was forced

                        ## Performance

                        State-variable forcing has **minimal computational overhead** (~1%),
                        making it suitable for production use.
                        """
                        ),
                    ),
                ),
            ),
        )
    )


def forcing_demo_server(input: Inputs, output: Outputs, session: Session):
    """Server logic for forcing demonstration."""

    # Reactive values
    forcing_obj = reactive.Value(None)
    time_series_data = reactive.Value(None)

    @reactive.effect
    @reactive.event(input.generate_forcing)
    def generate_forcing():
        """Generate forcing time series based on selected pattern."""
        # Generate years
        years = np.linspace(2000, 2020, 241)  # Monthly for 20 years

        # Generate pattern
        pattern_type = input.pattern_type()

        if pattern_type == "seasonal":
            amplitude = input.seasonal_amplitude()
            baseline = input.seasonal_baseline()
            values = baseline + amplitude * baseline * np.sin(2 * np.pi * years)

        elif pattern_type == "trend":
            start = input.trend_start()
            end = input.trend_end()
            values = np.linspace(start, end, len(years))

        elif pattern_type == "pulse":
            strength = input.pulse_strength()
            values = np.ones(len(years))
            # Add pulses every 5 years
            for year in [2005, 2010, 2015]:
                idx = np.argmin(np.abs(years - year))
                values[max(0, idx - 6) : min(len(values), idx + 6)] = strength

        elif pattern_type == "step":
            values = np.ones(len(years)) * 10.0
            # Step change at 2010
            values[years >= 2010] = 15.0

        else:  # custom
            values = np.ones(len(years)) * 15.0

        # Store data
        df = pd.DataFrame({"Year": years, "Value": values})
        time_series_data.set(df)

        # Create forcing object
        forcing_type = input.forcing_type()
        mode = input.forcing_mode()
        group_idx = input.group_idx()

        if forcing_type == "biomass":
            forcing = create_biomass_forcing(
                group_idx=group_idx,
                observed_biomass=values,
                years=years,
                mode=mode,
                interpolate=True,
            )
        elif forcing_type == "recruitment":
            forcing = create_recruitment_forcing(
                group_idx=group_idx,
                recruitment_multiplier=values,
                years=years,
                interpolate=(pattern_type != "pulse"),
            )
        else:
            forcing = StateForcing()
            variable = forcing_type
            forcing.add_forcing(
                group_idx=group_idx,
                variable=variable,
                time_series=values,
                years=years,
                mode=mode,
                interpolate=True,
            )

        forcing_obj.set(forcing)

    @output
    @render.ui
    def forcing_plot():
        """Plot forcing time series."""
        df = time_series_data()
        if df is None:
            return ui.div(
                ui.tags.p(
                    "Click 'Generate Forcing' to create forcing time series",
                    class_="text-muted text-center p-5",
                )
            )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=df["Year"],
                y=df["Value"],
                mode="lines",
                name="Forced Values",
                line=dict(color="#E63946", width=3),
            )
        )

        fig.update_layout(
            xaxis_title="Year",
            yaxis_title="Forced Value",
            template="plotly_white",
            height=400,
            showlegend=True,
            hovermode="x unified",
        )

        return ui.HTML(fig.to_html(include_plotlyjs="cdn"))

    @output
    @render.text
    def forcing_summary():
        """Display summary of forcing configuration."""
        df = time_series_data()
        if df is None:
            return ""

        forcing_type = input.forcing_type()
        mode = input.forcing_mode()
        group_idx = input.group_idx()
        pattern = input.pattern_type()

        summary = f"""
Forcing Configuration:
----------------------
Type: {forcing_type.upper()}
Mode: {mode.upper()}
Group: {group_idx}
Pattern: {pattern}

Time Series Statistics:
-----------------------
Mean: {df["Value"].mean():.2f}
Min: {df["Value"].min():.2f}
Max: {df["Value"].max():.2f}
Range: {df["Value"].max() - df["Value"].min():.2f}
Data Points: {len(df)}
        """
        return summary

    @output
    @render.ui
    def forcing_comparison_plot():
        """Plot comparison of forced vs unforced simulation."""
        df = time_series_data()
        if df is None:
            return ui.div(
                ui.tags.p(
                    "Generate forcing first, then run demo simulation",
                    class_="text-muted text-center p-5",
                )
            )

        # Simulate baseline (simple exponential)
        years = df["Year"].values
        baseline = 10.0 * np.exp(0.01 * (years - years[0]))

        # Apply forcing
        forced = df["Value"].values

        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=("Biomass Comparison", "Forcing Effect"),
            row_heights=[0.6, 0.4],
        )

        # Baseline simulation
        fig.add_trace(
            go.Scatter(
                x=years,
                y=baseline,
                mode="lines",
                name="Without Forcing",
                line=dict(color="#1D3557", width=2),
            ),
            row=1,
            col=1,
        )

        # Forced simulation (for REPLACE mode, this is the forced values)
        mode = input.forcing_mode()
        if mode == "replace":
            forced_sim = forced
        elif mode == "multiply":
            forced_sim = baseline * forced
        elif mode == "add":
            forced_sim = baseline + forced
        else:  # rescale
            forced_sim = forced

        fig.add_trace(
            go.Scatter(
                x=years,
                y=forced_sim,
                mode="lines",
                name="With Forcing",
                line=dict(color="#E63946", width=2),
            ),
            row=1,
            col=1,
        )

        # Effect of forcing
        effect = forced_sim - baseline
        fig.add_trace(
            go.Scatter(
                x=years,
                y=effect,
                mode="lines",
                name="Forcing Effect",
                fill="tozeroy",
                line=dict(color="#2A9D8F", width=2),
            ),
            row=2,
            col=1,
        )

        fig.update_xaxes(title_text="Year", row=2, col=1)
        fig.update_yaxes(title_text="Biomass", row=1, col=1)
        fig.update_yaxes(title_text="Effect", row=2, col=1)

        fig.update_layout(
            height=600, template="plotly_white", showlegend=True, hovermode="x unified"
        )

        return ui.HTML(fig.to_html(include_plotlyjs="cdn"))

    @output
    @render.code
    def forcing_code_example():
        """Generate Python code example."""
        forcing_type = input.forcing_type()
        mode = input.forcing_mode()
        group_idx = input.group_idx()
        pattern = input.pattern_type()

        code = f"""# State-Variable Forcing Example
# Generated from PyPath Demo

import numpy as np
from pypath.core.forcing import create_biomass_forcing, StateForcing
from pypath.core.ecosim_advanced import rsim_run_advanced

# Generate {pattern} pattern
years = np.linspace(2000, 2020, 241)
"""

        if pattern == "seasonal":
            amplitude = input.seasonal_amplitude()
            baseline = input.seasonal_baseline()
            code += f"""values = {baseline} + {amplitude} * {baseline} * np.sin(2 * np.pi * years)
"""
        elif pattern == "trend":
            start = input.trend_start()
            end = input.trend_end()
            code += f"""values = np.linspace({start}, {end}, len(years))
"""

        if forcing_type == "biomass":
            code += f"""
# Create biomass forcing
forcing = create_biomass_forcing(
    group_idx={group_idx},
    observed_biomass=values,
    years=years,
    mode='{mode}',
    interpolate=True
)
"""
        else:
            code += f"""
# Create custom forcing
forcing = StateForcing()
forcing.add_forcing(
    group_idx={group_idx},
    variable='{forcing_type}',
    time_series=values,
    years=years,
    mode='{mode}',
    interpolate=True
)
"""

        code += """
# Run simulation with forcing
result = rsim_run_advanced(
    scenario,
    state_forcing=forcing,
    verbose=True
)

# Compare with baseline
baseline_result = rsim_run(scenario)

# Plot results
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.plot(baseline_result.annual_Biomass[:, group_idx], label='Baseline')
plt.plot(result.annual_Biomass[:, group_idx], label='With Forcing')
plt.legend()
plt.xlabel('Year')
plt.ylabel('Biomass')
plt.title('Effect of Forcing')
plt.show()
"""
        return code

    @render.download(filename="forcing_example.py")
    def forcing_download_code():
        """Download code example."""
        code = forcing_code_example()
        return code
