"""
Multi-stanza Groups Page

Interactive setup and visualization of age-structured populations with von Bertalanffy growth.
"""

from shiny import ui, render, reactive, Inputs, Outputs, Session
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def multistanza_ui():
    """UI for multi-stanza groups page."""
    return ui.page_fluid(
        ui.layout_sidebar(
            ui.sidebar(
                ui.h4("Multi-Stanza Setup"),
                ui.input_select(
                    "stanza_group",
                    "Select Group",
                    choices=[],
                    selected=None
                ),
                ui.hr(),
                ui.h5("Stanza Parameters"),
                ui.input_numeric(
                    "n_stanzas",
                    "Number of Stanzas",
                    value=3,
                    min=1,
                    max=10
                ),
                ui.input_numeric(
                    "vb_k",
                    "von Bertalanffy K (growth rate)",
                    value=0.5,
                    min=0.01,
                    max=2.0,
                    step=0.01
                ),
                ui.input_numeric(
                    "vb_linf",
                    "L∞ (asymptotic length, cm)",
                    value=100,
                    min=1,
                    max=500,
                    step=1
                ),
                ui.input_numeric(
                    "vb_t0",
                    "t₀ (theoretical age at length 0)",
                    value=0,
                    min=-5,
                    max=5,
                    step=0.1
                ),
                ui.input_numeric(
                    "length_weight_a",
                    "Length-Weight a",
                    value=0.01,
                    min=0.0001,
                    max=1.0,
                    step=0.001
                ),
                ui.input_numeric(
                    "length_weight_b",
                    "Length-Weight b",
                    value=3.0,
                    min=1.0,
                    max=5.0,
                    step=0.1
                ),
                ui.hr(),
                ui.input_action_button(
                    "calculate_stanzas",
                    "Calculate Stanza Properties",
                    class_="btn-primary w-100"
                ),
                ui.input_action_button(
                    "save_stanzas",
                    "Save Configuration",
                    class_="btn-success w-100 mt-2"
                ),
                width=300
            ),
            # Main content
            ui.navset_tab(
                ui.nav_panel(
                    "Growth Curves",
                    ui.card(
                        ui.card_header("von Bertalanffy Growth Model"),
                        ui.output_ui("growth_plot"),
                        ui.markdown("""
                        **von Bertalanffy Growth Equation:**

                        Length: $L(t) = L_\\infty (1 - e^{-K(t - t_0)})$

                        Weight: $W(t) = a \\cdot L(t)^b$
                        """)
                    )
                ),
                ui.nav_panel(
                    "Stanza Properties",
                    ui.card(
                        ui.card_header("Calculated Stanza Parameters"),
                        ui.output_data_frame("stanza_table"),
                        ui.download_button("download_stanzas", "Download CSV", class_="mt-2")
                    )
                ),
                ui.nav_panel(
                    "Biomass Distribution",
                    ui.card(
                        ui.card_header("Biomass by Stanza"),
                        ui.output_ui("biomass_plot"),
                        ui.markdown("""
                        Shows the distribution of biomass across age stanzas based on:
                        - Growth rate (von Bertalanffy K)
                        - Natural mortality (Z)
                        - Recruitment patterns
                        """)
                    )
                ),
                ui.nav_panel(
                    "Help",
                    ui.card(
                        ui.card_header(ui.tags.i(class_="bi bi-info-circle me-2"), "Multi-Stanza Groups"),
                        ui.markdown("""
                        ## What are Multi-Stanza Groups?

                        Multi-stanza groups represent **age-structured populations** where different
                        life stages (stanzas) have distinct ecological properties.

                        ### Key Concepts

                        **von Bertalanffy Growth Model**
                        - K: Growth rate coefficient (higher = faster growth)
                        - L∞: Asymptotic (maximum) length
                        - t₀: Theoretical age at zero length

                        **Length-Weight Relationship**
                        - W = a × L^b
                        - Typical b values: 2.5-3.5 (3.0 for isometric growth)

                        **Stanza Properties**
                        - Each stanza represents an age range
                        - Properties calculated based on growth model
                        - Biomass distributed across stanzas

                        ### Example Use Cases

                        1. **Fish Populations**
                           - Juveniles, Sub-adults, Adults
                           - Different vulnerability to predation
                           - Size-dependent diet preferences

                        2. **Marine Mammals**
                           - Pups, Juveniles, Adults
                           - Age-specific reproduction
                           - Different energy requirements

                        3. **Invertebrates**
                           - Larvae, Juveniles, Adults
                           - Metamorphosis stages
                           - Size-structured populations

                        ### How to Use

                        1. **Select a group** from your model
                        2. **Set number of stanzas** (typically 2-5)
                        3. **Enter growth parameters** (K, L∞, t₀)
                        4. **Set length-weight relationship** (a, b)
                        5. **Calculate** to see growth curves and properties
                        6. **Review** stanza properties and biomass distribution
                        7. **Save** configuration to apply to your model

                        ### Tips

                        - **Start simple**: Begin with 2-3 stanzas
                        - **Use literature values**: Find K, L∞ from FishBase or literature
                        - **Check biomass**: Ensure distribution makes ecological sense
                        - **Validate**: Compare with observed age structure if available
                        """)
                    )
                )
            )
        )
    )


def multistanza_server(input: Inputs, output: Outputs, session: Session, shared_data):
    """Server logic for multi-stanza page."""

    # Reactive value to store stanza calculations
    stanza_data = reactive.Value(None)

    @reactive.effect
    def update_group_choices():
        """Update available groups when model changes."""
        if shared_data.params() is not None:
            params = shared_data.params()
            if hasattr(params, 'Group'):
                groups = params.Group.tolist()
                ui.update_select("stanza_group", choices=groups)

    @reactive.effect
    @reactive.event(input.calculate_stanzas)
    def calculate_stanzas():
        """Calculate stanza properties based on growth parameters."""
        n_stanzas = input.n_stanzas()
        K = input.vb_k()
        Linf = input.vb_linf()
        t0 = input.vb_t0()
        a = input.length_weight_a()
        b = input.length_weight_b()

        # Calculate age ranges for stanzas
        # Assume max age is when length reaches 95% of Linf
        t_max = -np.log(0.05) / K + t0

        # Create age bins
        ages = np.linspace(0, t_max, n_stanzas + 1)

        # Calculate properties for each stanza
        stanzas = []
        for i in range(n_stanzas):
            t_start = ages[i]
            t_end = ages[i + 1]
            t_mid = (t_start + t_end) / 2

            # von Bertalanffy growth
            L_start = Linf * (1 - np.exp(-K * (t_start - t0)))
            L_end = Linf * (1 - np.exp(-K * (t_end - t0)))
            L_mid = Linf * (1 - np.exp(-K * (t_mid - t0)))

            # Length-weight relationship
            W_start = a * (L_start ** b)
            W_end = a * (L_end ** b)
            W_mid = a * (L_mid ** b)

            stanzas.append({
                'Stanza': i + 1,
                'Age_Start': round(t_start, 2),
                'Age_End': round(t_end, 2),
                'Age_Mid': round(t_mid, 2),
                'Length_Start_cm': round(L_start, 2),
                'Length_End_cm': round(L_end, 2),
                'Length_Mid_cm': round(L_mid, 2),
                'Weight_Start_g': round(W_start, 2),
                'Weight_End_g': round(W_end, 2),
                'Weight_Mid_g': round(W_mid, 2),
            })

        df = pd.DataFrame(stanzas)
        stanza_data.set(df)

    @output
    @render.ui
    def growth_plot():
        """Render growth curves plot."""
        if input.calculate_stanzas() == 0:
            return ui.div(
                ui.tags.p("Click 'Calculate Stanza Properties' to generate growth curves",
                         class_="text-muted text-center p-5")
            )

        K = input.vb_k()
        Linf = input.vb_linf()
        t0 = input.vb_t0()
        a = input.length_weight_a()
        b = input.length_weight_b()

        # Calculate max age
        t_max = -np.log(0.05) / K + t0
        ages = np.linspace(0, t_max, 100)

        # von Bertalanffy length
        lengths = Linf * (1 - np.exp(-K * (ages - t0)))

        # Weight from length-weight relationship
        weights = a * (lengths ** b)

        # Create subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Length Growth', 'Weight Growth')
        )

        # Length curve
        fig.add_trace(
            go.Scatter(
                x=ages, y=lengths,
                mode='lines',
                name='Length',
                line=dict(color='#2E86AB', width=3)
            ),
            row=1, col=1
        )

        # Weight curve
        fig.add_trace(
            go.Scatter(
                x=ages, y=weights,
                mode='lines',
                name='Weight',
                line=dict(color='#A23B72', width=3),
                showlegend=False
            ),
            row=1, col=2
        )

        # Add stanza boundaries if calculated
        df = stanza_data()
        if df is not None:
            for _, row in df.iterrows():
                # Add vertical line for age boundary
                fig.add_vline(
                    x=row['Age_End'],
                    line_dash="dash",
                    line_color="gray",
                    opacity=0.5,
                    row=1, col=1
                )
                fig.add_vline(
                    x=row['Age_End'],
                    line_dash="dash",
                    line_color="gray",
                    opacity=0.5,
                    row=1, col=2
                )

        fig.update_xaxes(title_text="Age (years)", row=1, col=1)
        fig.update_xaxes(title_text="Age (years)", row=1, col=2)
        fig.update_yaxes(title_text="Length (cm)", row=1, col=1)
        fig.update_yaxes(title_text="Weight (g)", row=1, col=2)

        fig.update_layout(
            height=400,
            showlegend=False,
            template='plotly_white',
            margin=dict(l=50, r=50, t=50, b=50)
        )

        return ui.HTML(fig.to_html(include_plotlyjs="cdn", div_id="growth_plot"))

    @output
    @render.data_frame
    def stanza_table():
        """Render stanza properties table."""
        df = stanza_data()
        if df is None:
            return pd.DataFrame({
                'Message': ['Click "Calculate Stanza Properties" to generate table']
            })
        return render.DataGrid(df, width="100%", height="400px")

    @output
    @render.ui
    def biomass_plot():
        """Render biomass distribution plot."""
        df = stanza_data()
        if df is None:
            return ui.div(
                ui.tags.p("Calculate stanza properties first to see biomass distribution",
                         class_="text-muted text-center p-5")
            )

        # Simple biomass distribution (can be enhanced with mortality)
        # Assume exponential decline with age
        Z = 0.3  # Natural mortality (can be made adjustable)

        biomass = []
        for _, row in df.iterrows():
            t_mid = row['Age_Mid']
            W_mid = row['Weight_Mid_g']
            # Numbers decline exponentially with age
            N = np.exp(-Z * t_mid)
            B = N * W_mid
            biomass.append(B)

        # Normalize to sum to 1
        biomass = np.array(biomass)
        biomass = biomass / biomass.sum()

        fig = go.Figure(data=[
            go.Bar(
                x=df['Stanza'].astype(str),
                y=biomass * 100,
                marker_color='#2E86AB',
                text=[f'{b*100:.1f}%' for b in biomass],
                textposition='outside'
            )
        ])

        fig.update_layout(
            xaxis_title="Stanza",
            yaxis_title="Biomass Proportion (%)",
            template='plotly_white',
            height=400,
            showlegend=False,
            margin=dict(l=50, r=50, t=50, b=50)
        )

        return ui.HTML(fig.to_html(include_plotlyjs="cdn", div_id="biomass_plot"))

    @session.download(filename="stanza_configuration.csv")
    def download_stanzas():
        """Download stanza configuration as CSV."""
        df = stanza_data()
        if df is not None:
            yield df.to_csv(index=False)
