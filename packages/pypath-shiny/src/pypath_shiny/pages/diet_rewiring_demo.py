"""
Dynamic Diet Rewiring Demonstration Page

Interactive demonstration of adaptive foraging and prey switching dynamics.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from shiny import Inputs, Outputs, Session, reactive, render, ui

# Import centralized configuration
try:
    from app.config import DEFAULTS
except ModuleNotFoundError:
    from config import DEFAULTS

# pypath imports (path setup handled by app/__init__.py)
from pypath.core.forcing import DietRewiring


def diet_rewiring_demo_ui():
    """UI for diet rewiring demonstration page."""
    return ui.page_fluid(
        ui.layout_sidebar(
            ui.sidebar(
                ui.h4("Diet Rewiring Configuration"),
                ui.input_slider(
                    "demo_switching_power",
                    "Switching Power",
                    min=1.0,
                    max=DEFAULTS.max_dc,  # was: 5.0
                    value=DEFAULTS.switching_power,  # was: 2.5
                    step=0.1,
                ),
                ui.input_slider(
                    "update_interval",
                    "Update Interval (months)",
                    min=1,
                    max=24,
                    value=DEFAULTS.diet_update_interval,  # was: 12
                    step=1,
                ),
                ui.input_numeric(
                    "min_proportion",
                    "Minimum Diet Proportion",
                    value=DEFAULTS.min_diet_proportion,  # was: 0.001
                    min=0.0001,
                    max=0.1,
                    step=0.001,
                ),
                ui.hr(),
                ui.h5("Scenario Selection"),
                ui.input_radio_buttons(
                    "scenario",
                    "Test Scenario",
                    choices={
                        "normal": "Normal Conditions",
                        "prey1_collapse": "Prey 1 Collapse",
                        "prey2_bloom": "Prey 2 Bloom",
                        "alternating": "Alternating Abundance",
                        "custom": "Custom Biomass",
                    },
                    selected="normal",
                ),
                ui.panel_conditional(
                    "input.scenario === 'custom'",
                    ui.input_slider(
                        "prey1_biomass",
                        "Prey 1 Biomass",
                        min=0,
                        max=50,
                        value=10,
                        step=1,
                    ),
                    ui.input_slider(
                        "prey2_biomass",
                        "Prey 2 Biomass",
                        min=0,
                        max=50,
                        value=10,
                        step=1,
                    ),
                    ui.input_slider(
                        "prey3_biomass",
                        "Prey 3 Biomass",
                        min=0,
                        max=50,
                        value=10,
                        step=1,
                    ),
                ),
                ui.hr(),
                ui.input_action_button(
                    "run_rewiring", "Calculate Diet Shift", class_="btn-primary w-100"
                ),
                ui.input_action_button(
                    "reset_diet",
                    "Reset to Base Diet",
                    class_="btn-secondary w-100 mt-2",
                ),
                width=300,
            ),
            # Main content
            ui.navset_tab(
                ui.nav_panel(
                    "Diet Composition",
                    ui.card(
                        ui.card_header("Diet Shift Visualization"),
                        ui.output_ui("diet_comparison_plot"),
                        ui.output_text_verbatim("diet_summary"),
                    ),
                ),
                ui.nav_panel(
                    "Prey Switching Response",
                    ui.card(
                        ui.card_header("How Diet Changes with Biomass"),
                        ui.output_ui("switching_curve_plot"),
                        ui.markdown(
                            """
                        **Prey Switching Model:**

                        $\\text{new\\_diet}[i] = \\text{base\\_diet}[i] \\times \\left(\\frac{B_i}{\\bar{B}}\\right)^p$

                        Where:
                        - $B_i$ = current prey biomass
                        - $\\bar{B}$ = mean prey biomass
                        - $p$ = switching power

                        Then normalized so diet sums to 1.
                        """
                        ),
                    ),
                ),
                ui.nav_panel(
                    "Time Series",
                    ui.card(
                        ui.card_header("Diet Evolution Over Time"),
                        ui.output_ui("time_series_plot"),
                        ui.markdown(
                            """
                        Shows how diet composition changes as prey biomass varies over time.
                        """
                        ),
                    ),
                ),
                ui.nav_panel(
                    "Code Example",
                    ui.card(
                        ui.card_header("Python Code"),
                        ui.output_code("diet_code_example"),
                        ui.download_button(
                            "diet_download_code", "Download Code", class_="mt-2"
                        ),
                    ),
                ),
                ui.nav_panel(
                    "Help",
                    ui.card(
                        ui.card_header(
                            ui.tags.i(class_="bi bi-question-circle me-2"),
                            "Dynamic Diet Rewiring Guide",
                        ),
                        ui.markdown(
                            """
                        ## What is Dynamic Diet Rewiring?

                        Diet rewiring allows **predator diet preferences to change over time**
                        in response to changing prey abundance. This implements:

                        - **Prey switching**: Predators shift to more abundant prey
                        - **Adaptive foraging**: Optimize feeding efficiency
                        - **Functional responses**: Beyond static diet matrices

                        ## Key Parameters

                        ### Switching Power (p)

                        Controls the **strength** of the prey switching response:

                        - **p = 1.0**: Proportional response (no switching)
                          - Diet changes proportionally with biomass
                          - Weak adaptive behavior

                        - **p = 2.0-3.0**: Moderate switching (typical)
                          - Realistic for most predators
                          - Balanced foraging strategy

                        - **p > 3.0**: Strong switching
                          - Highly opportunistic predators
                          - Can create alternative stable states

                        ### Update Interval

                        How often diet is recalculated:

                        - **Monthly (1)**: Fast-changing systems
                          - High computational cost
                          - Very responsive

                        - **Quarterly (3)**: Seasonal changes
                          - Moderate cost
                          - Seasonal adaptation

                        - **Annual (12)**: Slow-changing systems
                          - Low computational cost
                          - Long-term adaptation

                        ### Minimum Proportion

                        Prevents diet proportions from going to zero:

                        - **0.001**: Standard value
                        - Prevents division by zero
                        - Ensures some consumption of all prey

                        ## Ecological Scenarios

                        ### 1. Normal Conditions
                        - All prey at similar biomass
                        - Diet close to base preferences
                        - Minimal switching

                        ### 2. Prey Collapse
                        - One prey becomes scarce
                        - Predator shifts away from scarce prey
                        - Increases other prey consumption
                        - **Prey refuge effect**

                        ### 3. Prey Bloom
                        - One prey becomes very abundant
                        - Predator shifts heavily to abundant prey
                        - **Opportunistic feeding**

                        ### 4. Alternating Abundance
                        - Prey biomass cycles
                        - Diet tracks abundance patterns
                        - **Dynamic adaptation**

                        ## Example Use Cases

                        ### Fish Predators
                        ```python
                        # Cod switching between herring and sprat
                        diet_rewiring = create_diet_rewiring(
                            switching_power=2.5,  # Moderate opportunism
                            update_interval=3      # Quarterly
                        )
                        ```

                        ### Opportunistic Feeders
                        ```python
                        # Jellyfish switching between zooplankton types
                        diet_rewiring = create_diet_rewiring(
                            switching_power=4.0,  # High opportunism
                            update_interval=1      # Monthly
                        )
                        ```

                        ### Specialist Predators
                        ```python
                        # Killer whales with preferred prey
                        diet_rewiring = create_diet_rewiring(
                            switching_power=1.5,  # Weak switching
                            update_interval=12     # Annual
                        )
                        ```

                        ## Scientific Background

                        ### Theoretical Basis

                        - **Murdoch (1969)**: Original prey switching experiments
                        - **Chesson (1983)**: Frequency-dependent predation theory
                        - **Gentleman et al. (2003)**: Functional responses in Ecosim

                        ### When Prey Switching Matters

                        ✅ **Important when:**
                        - Prey abundance varies substantially
                        - Predators are generalists
                        - Multiple prey species available
                        - System shows regime shifts

                        ⚠️ **Less important when:**
                        - Prey abundance is stable
                        - Predators are specialists
                        - Strong habitat segregation
                        - Morphological constraints on diet

                        ## Best Practices

                        ### Choosing Switching Power

                        1. **Start with 2.0** (moderate switching)
                        2. **Increase for**: Opportunistic predators, variable prey
                        3. **Decrease for**: Specialists, morphological constraints
                        4. **Validate with**: Observed diet data, gut content analysis

                        ### Setting Update Interval

                        1. **Consider timescales**: Monthly for plankton, annual for fish
                        2. **Balance**: Realism vs. computational cost
                        3. **Start with 12 months**: Good for most applications

                        ### Validation

                        ✅ **Check:**
                        - Diet sums to 1 (automatically ensured)
                        - Ecological realism of shifts
                        - Comparison with observed diets
                        - Sensitivity to parameters

                        ## Performance

                        - **Annual updates**: ~1% overhead (negligible)
                        - **Monthly updates**: ~5-10% overhead (acceptable)
                        - **Real-time updates**: Use with caution

                        ## Limitations

                        ⚠️ **Simplified model**:
                        - No handling time
                        - No search costs
                        - No learning
                        - No spatial effects
                        - Assumes biomass = availability

                        Despite simplifications, provides **realistic adaptive foraging dynamics**
                        for ecosystem models.
                        """
                        ),
                    ),
                ),
            ),
        )
    )


def diet_rewiring_demo_server(input: Inputs, output: Outputs, session: Session):
    """Server logic for diet rewiring demonstration."""

    # Base diet (3 prey, 1 predator)
    base_diet = np.array(
        [
            [0.5],  # Prey 1: Herring (50%)
            [0.3],  # Prey 2: Sprat (30%)
            [0.2],  # Prey 3: Zooplankton (20%)
        ]
    )

    # Reactive values
    current_diet = reactive.Value(base_diet.copy())
    _diet_history = reactive.Value(None)

    @reactive.effect
    @reactive.event(input.run_rewiring)
    def calculate_diet_shift():
        """Calculate diet shift based on scenario."""
        # Get scenario biomass
        scenario = input.scenario()

        if scenario == "normal":
            biomass = np.array([10.0, 10.0, 10.0, 0.0])
        elif scenario == "prey1_collapse":
            biomass = np.array([2.0, 10.0, 10.0, 0.0])
        elif scenario == "prey2_bloom":
            biomass = np.array([10.0, 30.0, 10.0, 0.0])
        elif scenario == "alternating":
            biomass = np.array([15.0, 5.0, 10.0, 0.0])
        else:  # custom
            biomass = np.array(
                [
                    input.prey1_biomass(),
                    input.prey2_biomass(),
                    input.prey3_biomass(),
                    0.0,
                ]
            )

        # Create diet rewiring object
        switching_power = input.demo_switching_power()
        min_proportion = input.min_proportion()

        rewiring = DietRewiring(
            enabled=True,
            switching_power=switching_power,
            min_proportion=min_proportion,
            update_interval=input.update_interval(),
        )

        rewiring.initialize(base_diet)
        new_diet = rewiring.update_diet(biomass)

        if new_diet is not None:
            current_diet.set(new_diet)

    @reactive.effect
    @reactive.event(input.reset_diet)
    def reset_diet():
        """Reset to base diet."""
        current_diet.set(base_diet.copy())

    @output
    @render.ui
    def diet_comparison_plot():
        """Plot diet composition comparison."""
        new_diet = current_diet()

        fig = go.Figure()

        prey_names = ["Herring", "Sprat", "Zooplankton"]
        x = np.arange(len(prey_names))
        width = 0.35

        fig.add_trace(
            go.Bar(
                x=x - width / 2,
                y=base_diet[:, 0] * 100,
                name="Base Diet",
                marker_color="#457B9D",
                width=width,
            )
        )

        fig.add_trace(
            go.Bar(
                x=x + width / 2,
                y=new_diet[:, 0] * 100,
                name="Current Diet",
                marker_color="#E63946",
                width=width,
            )
        )

        fig.update_layout(
            xaxis=dict(tickmode="array", tickvals=x, ticktext=prey_names),
            yaxis_title="Diet Proportion (%)",
            template="plotly_white",
            height=400,
            barmode="group",
            showlegend=True,
        )

        return ui.HTML(fig.to_html(include_plotlyjs="cdn"))

    @output
    @render.text
    def diet_summary():
        """Display diet summary statistics."""
        new_diet = current_diet()

        summary = "Diet Composition:\n"
        summary += "-" * 50 + "\n"
        summary += f"{'Prey':<15} {'Base':<12} {'Current':<12} {'Change':<10}\n"
        summary += "-" * 50 + "\n"

        prey_names = ["Herring", "Sprat", "Zooplankton"]
        for i, name in enumerate(prey_names):
            base_pct = base_diet[i, 0] * 100
            curr_pct = new_diet[i, 0] * 100
            change = curr_pct - base_pct
            arrow = "^" if change > 0.5 else "v" if change < -0.5 else "-"

            summary += f"{name:<15} {base_pct:>6.1f}%    {curr_pct:>6.1f}%    {change:>+6.1f}% {arrow}\n"

        summary += "-" * 50 + "\n"
        summary += f"Switching Power: {input.demo_switching_power():.1f}\n"
        summary += f"Update Interval: {input.update_interval()} months\n"

        return summary

    @output
    @render.ui
    def switching_curve_plot():
        """Plot prey switching response curves."""
        switching_power = input.demo_switching_power()

        # Range of relative biomass
        relative_biomass = np.linspace(0.1, 5.0, 100)

        # Calculate diet response for different base proportions
        base_props = [0.5, 0.3, 0.2]
        colors = ["#457B9D", "#E63946", "#2A9D8F"]

        fig = go.Figure()

        for i, (base_prop, color) in enumerate(zip(base_props, colors)):
            # Response before normalization
            response = base_prop * (relative_biomass**switching_power)

            fig.add_trace(
                go.Scatter(
                    x=relative_biomass,
                    y=response,
                    mode="lines",
                    name=f"Base = {base_prop * 100:.0f}%",
                    line=dict(color=color, width=2),
                )
            )

        fig.update_layout(
            xaxis_title="Relative Prey Biomass (B/B_mean)",
            yaxis_title="Diet Response (before normalization)",
            template="plotly_white",
            height=400,
            showlegend=True,
            hovermode="x unified",
        )

        # Add reference line at 1.0
        fig.add_vline(x=1.0, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_hline(y=0.5, line_dash="dash", line_color="gray", opacity=0.5)

        return ui.HTML(fig.to_html(include_plotlyjs="cdn"))

    @output
    @render.ui
    def time_series_plot():
        """Plot diet evolution over time."""
        # Generate time series scenario
        months = np.arange(0, 120)  # 10 years

        # Simulate prey biomass cycling
        prey1 = 10 + 5 * np.sin(2 * np.pi * months / 24)
        prey2 = 10 + 3 * np.sin(2 * np.pi * months / 24 + np.pi)
        prey3 = np.ones_like(months) * 10

        switching_power = input.demo_switching_power()
        min_proportion = input.min_proportion()

        # Calculate diet over time
        diet_over_time = []

        rewiring = DietRewiring(
            enabled=True,
            switching_power=switching_power,
            min_proportion=min_proportion,
            update_interval=input.update_interval(),
        )
        rewiring.initialize(base_diet)

        for i in range(len(months)):
            biomass = np.array([prey1[i], prey2[i], prey3[i], 0.0])

            # Only update at intervals
            if i % input.update_interval() == 0:
                new_diet = rewiring.update_diet(biomass)
                if new_diet is not None:
                    diet_over_time.append(new_diet[:, 0])
                else:
                    diet_over_time.append(base_diet[:, 0])
            else:
                diet_over_time.append(rewiring.current_diet[:, 0])

        diet_array = np.array(diet_over_time)

        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=("Prey Biomass", "Diet Composition"),
            row_heights=[0.4, 0.6],
            vertical_spacing=0.12,
        )

        # Prey biomass
        fig.add_trace(
            go.Scatter(x=months, y=prey1, name="Herring", line=dict(color="#457B9D")),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=months, y=prey2, name="Sprat", line=dict(color="#E63946")),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=months, y=prey3, name="Zooplankton", line=dict(color="#2A9D8F")
            ),
            row=1,
            col=1,
        )

        # Diet proportions
        fig.add_trace(
            go.Scatter(
                x=months,
                y=diet_array[:, 0] * 100,
                name="Herring Diet",
                line=dict(color="#457B9D"),
                showlegend=False,
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=months,
                y=diet_array[:, 1] * 100,
                name="Sprat Diet",
                line=dict(color="#E63946"),
                showlegend=False,
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=months,
                y=diet_array[:, 2] * 100,
                name="Zooplankton Diet",
                line=dict(color="#2A9D8F"),
                showlegend=False,
            ),
            row=2,
            col=1,
        )

        fig.update_xaxes(title_text="Month", row=2, col=1)
        fig.update_yaxes(title_text="Biomass", row=1, col=1)
        fig.update_yaxes(title_text="Diet %", row=2, col=1)

        fig.update_layout(
            height=600, template="plotly_white", showlegend=True, hovermode="x unified"
        )

        return ui.HTML(fig.to_html(include_plotlyjs="cdn"))

    @output
    @render.code
    def diet_code_example():
        """Generate Python code example."""
        switching_power = input.demo_switching_power()
        update_interval = input.update_interval()
        min_proportion = input.min_proportion()

        code = f"""# Dynamic Diet Rewiring Example
# Generated from PyPath Demo

from pypath.core.forcing import create_diet_rewiring
from pypath.core.ecosim_advanced import rsim_run_advanced

# Create diet rewiring configuration
diet_rewiring = create_diet_rewiring(
    switching_power={switching_power},
    min_proportion={min_proportion},
    update_interval={update_interval}
)

# Run simulation with diet rewiring
result = rsim_run_advanced(
    scenario,
    diet_rewiring=diet_rewiring,
    verbose=True
)

# Access updated diet matrix
# The diet will adapt as prey biomass changes during simulation

# Plot diet evolution
import matplotlib.pyplot as plt
import numpy as np

# Extract diet from simulation (if saved)
# This is conceptual - actual implementation depends on output structure
months = np.arange(len(result.out_Biomass))
diet_history = []  # Would be extracted from simulation

plt.figure(figsize=(12, 6))
plt.plot(months, diet_history)
plt.xlabel('Month')
plt.ylabel('Diet Proportion')
plt.title('Diet Evolution with Prey Switching')
plt.legend(['Prey 1', 'Prey 2', 'Prey 3'])
plt.show()
"""
        return code

    @render.download(filename="diet_rewiring_example.py")
    def diet_download_code():
        """Download code example."""
        code = diet_code_example()
        return code
