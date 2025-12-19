"""Pre-balance Diagnostics Page.

This module provides an interactive interface for pre-balance diagnostic analysis
of Ecopath models before balancing. It helps identify potential issues with
biomasses, vital rates, and predator-prey relationships.

Based on the Prebal routine by Barbara Bauer (SU, 2016).
"""

from shiny import ui, render, reactive, Inputs, Outputs, Session
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Get logger
logger = logging.getLogger('pypath_app.prebalance')

try:
    from app.config import UI, PLOTS, COLORS
    from app.pages.utils import is_rpath_params
except ModuleNotFoundError:
    from config import UI, PLOTS, COLORS
    from pages.utils import is_rpath_params

# Import prebalance functions
import sys
root_dir = Path(__file__).parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.pypath.analysis.prebalance import (
    calculate_biomass_slope,
    calculate_biomass_range,
    calculate_predator_prey_ratios,
    calculate_vital_rate_ratios,
    plot_biomass_vs_trophic_level,
    plot_vital_rate_vs_trophic_level,
    generate_prebalance_report,
)


def prebalance_ui():
    """Pre-balance diagnostics UI."""
    return ui.page_fluid(
        ui.layout_sidebar(
            ui.sidebar(
                ui.h4("Pre-Balance Diagnostics"),
                ui.p(
                    "Run diagnostic checks on your unbalanced model to identify "
                    "potential issues before balancing.",
                    class_="text-muted"
                ),
                ui.hr(),

                ui.input_action_button(
                    "btn_run_diagnostics",
                    "Run Diagnostics",
                    class_="btn-primary w-100 mb-3",
                    icon=ui.tags.i(class_="bi bi-play-circle")
                ),

                ui.hr(),

                ui.panel_well(
                    ui.h6("Visualization Options"),

                    ui.input_select(
                        "plot_type",
                        "Plot Type",
                        choices={
                            "biomass": "Biomass vs Trophic Level",
                            "pb": "P/B vs Trophic Level",
                            "qb": "Q/B vs Trophic Level"
                        },
                        selected="biomass"
                    ),

                    ui.input_text(
                        "exclude_groups",
                        "Exclude Groups (comma-separated)",
                        value="",
                        placeholder="e.g., Whales, Seabirds"
                    ),
                ),

                ui.hr(),

                ui.panel_well(
                    ui.h6("About Pre-Balance Diagnostics"),
                    ui.tags.small(
                        ui.tags.ul(
                            ui.tags.li(
                                ui.tags.strong("Biomass Slope:"),
                                " Indicates top-down control strength (-0.5 to -1.5 typical)"
                            ),
                            ui.tags.li(
                                ui.tags.strong("Biomass Range:"),
                                " Large ranges (>6 orders) may indicate missing groups"
                            ),
                            ui.tags.li(
                                ui.tags.strong("Predator/Prey Ratio:"),
                                " High ratios (>1) suggest unsustainable predation"
                            ),
                            ui.tags.li(
                                ui.tags.strong("Vital Rate Ratios:"),
                                " Predator rates should be lower than prey rates"
                            ),
                            class_="small"
                        ),
                        class_="text-muted"
                    )
                ),

                width=UI.sidebar_width,
                position="left"
            ),

            # Main content area
            ui.navset_card_tab(
                ui.nav_panel(
                    "Summary Report",
                    ui.output_ui("report_summary"),
                ),
                ui.nav_panel(
                    "Warnings",
                    ui.output_ui("report_warnings"),
                ),
                ui.nav_panel(
                    "Predator-Prey Ratios",
                    ui.output_data_frame("table_predator_prey"),
                ),
                ui.nav_panel(
                    "Vital Rate Ratios",
                    ui.tags.div(
                        ui.h5("P/B Ratios"),
                        ui.output_data_frame("table_pb_ratios"),
                        ui.hr(),
                        ui.h5("Q/B Ratios"),
                        ui.output_data_frame("table_qb_ratios"),
                    )
                ),
                ui.nav_panel(
                    "Visualization",
                    ui.output_plot("diagnostic_plot", height=UI.plot_height_large_px),
                ),
                ui.nav_panel(
                    "Help",
                    ui.markdown(
                        """
                        ## Pre-Balance Diagnostics Help

                        ### Overview
                        Pre-balance diagnostics help identify potential issues with your Ecopath model
                        **before** attempting to balance it. This can save time and help you understand
                        your model's structure better.

                        ### Diagnostic Metrics

                        #### 1. Biomass Slope
                        - Measures how biomass changes across trophic levels
                        - **Typical range**: -0.5 to -1.5 (negative slope expected)
                        - **Interpretation**:
                          - Steep slope (< -2): Very strong top-down control
                          - Flat slope (> -0.3): Weak trophic structure

                        #### 2. Biomass Range
                        - Measures the span of biomasses (log10 scale)
                        - **Warning threshold**: > 6 orders of magnitude
                        - **Issues**:
                          - Large ranges may indicate missing functional groups
                          - Could suggest unrealistic biomass values

                        #### 3. Predator-Prey Biomass Ratios
                        - Compares predator biomass to total prey biomass
                        - **Typical range**: 0.01 to 0.5
                        - **Warning threshold**: > 1.0
                        - **Interpretation**:
                          - Ratio > 1: Predator biomass exceeds prey (unsustainable)
                          - High ratios indicate insufficient prey support

                        #### 4. Vital Rate Ratios (P/B, Q/B)
                        - Compares predator rates to mean prey rates
                        - **Expected pattern**: Predators have lower rates than prey
                        - **Interpretation**:
                          - Follows metabolic theory (larger animals = slower rates)
                          - Violations may indicate data errors

                        ### How to Use

                        1. **Load Model**: Import your unbalanced Ecopath model on the Data Import page
                        2. **Run Diagnostics**: Click "Run Diagnostics" button
                        3. **Review Summary**: Check overall metrics and ranges
                        4. **Check Warnings**: Address any flagged issues
                        5. **Examine Ratios**: Look for suspicious predator-prey relationships
                        6. **Visualize**: Use plots to identify outliers
                        7. **Fix Issues**: Return to Data Import or Ecopath pages to adjust values
                        8. **Re-run**: Run diagnostics again until warnings are resolved

                        ### Visualization Options

                        - **Biomass vs TL**: Shows biomass distribution across food web
                        - **P/B vs TL**: Production rates should decrease with trophic level
                        - **Q/B vs TL**: Consumption rates should decrease with trophic level
                        - **Exclude Groups**: Optionally remove groups from visualization (e.g., marine mammals)

                        ### Common Issues and Solutions

                        | Issue | Likely Cause | Solution |
                        |-------|-------------|----------|
                        | High predator/prey ratio | Predator biomass too high | Reduce predator biomass or increase prey biomass |
                        | Large biomass range | Missing functional groups | Add intermediate groups or check for data entry errors |
                        | Steep biomass slope | Strong top-down control | May be realistic (verify with literature) |
                        | Inverted vital rates | Data entry error | Check P/B and Q/B values against literature |

                        ### References

                        - Bauer, B. (2016). Prebal routine for Rpath. Stockholm University.
                        - Link, J. S. (2010). Adding rigor to ecological network models by evaluating
                          a set of pre-balance diagnostics: A plea for PREBAL. *Ecological Modelling*,
                          221(12), 1580-1591.
                        - Christensen, V., & Walters, C. J. (2004). Ecopath with Ecosim: Methods,
                          capabilities and limitations. *Ecological Modelling*, 172(2-4), 109-139.
                        """
                    )
                ),
            )
        )
    )


def prebalance_server(
    input: Inputs,
    output: Outputs,
    session: Session,
    model_data: reactive.Value
):
    """Pre-balance diagnostics server logic.

    Parameters
    ----------
    input : Inputs
        Shiny inputs
    output : Outputs
        Shiny outputs
    session : Session
        Shiny session
    model_data : reactive.Value
        Reactive value containing model data (RpathParams)
    """

    # Store diagnostic report
    diagnostic_report = reactive.Value(None)

    @reactive.effect
    @reactive.event(input.btn_run_diagnostics)
    def _run_diagnostics():
        """Run pre-balance diagnostics on current model."""
        try:
            data = model_data()

            if data is None:
                ui.notification_show(
                    "No model data available. Please import a model first.",
                    type="warning",
                    duration=5
                )
                return

            # Check if model is unbalanced (RpathParams)
            if not is_rpath_params(data):
                ui.notification_show(
                    "Pre-balance diagnostics require an unbalanced model (RpathParams). "
                    "The current model appears to be already balanced.",
                    type="warning",
                    duration=5
                )
                return

            ui.notification_show("Running diagnostics...", duration=3)

            # Generate diagnostic report
            report = generate_prebalance_report(data)
            diagnostic_report.set(report)

            # Show completion notification
            num_warnings = len(report['warnings'])
            if num_warnings == 0:
                ui.notification_show(
                    "Diagnostics complete! No major issues detected.",
                    type="message",
                    duration=5
                )
            else:
                ui.notification_show(
                    f"Diagnostics complete. Found {num_warnings} warning(s). Check the Warnings tab.",
                    type="warning",
                    duration=5
                )

        except Exception as e:
            logger.error(f"Error running diagnostics: {e}", exc_info=True)
            ui.notification_show(
                f"Error running diagnostics: {str(e)}",
                type="error",
                duration=5
            )

    @output
    @render.ui
    def report_summary():
        """Render diagnostic summary report."""
        report = diagnostic_report()

        if report is None:
            return ui.tags.div(
                ui.tags.p(
                    "No diagnostics run yet. Click 'Run Diagnostics' to analyze your model.",
                    class_="text-muted text-center p-5"
                )
            )

        # Format summary statistics
        summary_cards = [
            ui.div(
                ui.h5("Biomass Diagnostics", class_="card-title"),
                ui.tags.dl(
                    ui.tags.dt("Biomass Range:"),
                    ui.tags.dd(f"{report['biomass_range']:.2f} orders of magnitude"),
                    ui.tags.dt("Biomass Slope:"),
                    ui.tags.dd(f"{report['biomass_slope']:.3f}"),
                ),
                class_="card-body"
            ),
        ]

        # Predator-prey summary
        if len(report['predator_prey_ratios']) > 0:
            pp_ratios = report['predator_prey_ratios']['Ratio']
            summary_cards.append(
                ui.div(
                    ui.h5("Predator-Prey Ratios", class_="card-title"),
                    ui.tags.dl(
                        ui.tags.dt("Number of predators analyzed:"),
                        ui.tags.dd(f"{len(pp_ratios)}"),
                        ui.tags.dt("Mean ratio:"),
                        ui.tags.dd(f"{pp_ratios.mean():.3f}"),
                        ui.tags.dt("Max ratio:"),
                        ui.tags.dd(f"{pp_ratios.max():.3f}"),
                        ui.tags.dt("Ratios > 1.0:"),
                        ui.tags.dd(f"{(pp_ratios > 1.0).sum()} (potentially unsustainable)"),
                    ),
                    class_="card-body"
                )
            )

        # Vital rate summaries
        if len(report.get('pb_ratios', [])) > 0:
            pb_ratios = report['pb_ratios']['Ratio']
            summary_cards.append(
                ui.div(
                    ui.h5("P/B Rate Ratios", class_="card-title"),
                    ui.tags.dl(
                        ui.tags.dt("Mean P/B ratio (Predator/Prey):"),
                        ui.tags.dd(f"{pb_ratios.mean():.3f}"),
                        ui.tags.dt("Number analyzed:"),
                        ui.tags.dd(f"{len(pb_ratios)}"),
                    ),
                    class_="card-body"
                )
            )

        if len(report.get('qb_ratios', [])) > 0:
            qb_ratios = report['qb_ratios']['Ratio']
            summary_cards.append(
                ui.div(
                    ui.h5("Q/B Rate Ratios", class_="card-title"),
                    ui.tags.dl(
                        ui.tags.dt("Mean Q/B ratio (Predator/Prey):"),
                        ui.tags.dd(f"{qb_ratios.mean():.3f}"),
                        ui.tags.dt("Number analyzed:"),
                        ui.tags.dd(f"{len(qb_ratios)}"),
                    ),
                    class_="card-body"
                )
            )

        return ui.tags.div(
            ui.row(
                *[ui.column(6, ui.div(card, class_="card mb-3")) for card in summary_cards]
            )
        )

    @output
    @render.ui
    def report_warnings():
        """Render diagnostic warnings."""
        report = diagnostic_report()

        if report is None:
            return ui.tags.div(
                ui.tags.p(
                    "No diagnostics run yet.",
                    class_="text-muted text-center p-5"
                )
            )

        warnings = report['warnings']

        if len(warnings) == 0:
            return ui.tags.div(
                ui.div(
                    ui.tags.i(class_="bi bi-check-circle-fill text-success", style="font-size: 3rem;"),
                    ui.h4("No major issues detected!", class_="mt-3"),
                    ui.p(
                        "Your model passed all pre-balance diagnostic checks. "
                        "You can proceed with balancing.",
                        class_="text-muted"
                    ),
                    class_="text-center p-5"
                )
            )

        # Format warnings as alert boxes
        warning_items = []
        for i, warning in enumerate(warnings, 1):
            warning_items.append(
                ui.div(
                    ui.tags.strong(f"Warning {i}:"),
                    " ",
                    warning,
                    class_="alert alert-warning mb-3",
                    role="alert"
                )
            )

        return ui.tags.div(
            ui.h5(f"Found {len(warnings)} Warning(s)"),
            ui.hr(),
            *warning_items
        )

    @output
    @render.data_frame
    def table_predator_prey():
        """Render predator-prey ratios table."""
        report = diagnostic_report()

        if report is None or len(report['predator_prey_ratios']) == 0:
            return pd.DataFrame()

        df = report['predator_prey_ratios'].copy()

        # Format numeric columns
        df['Prey_Biomass'] = df['Prey_Biomass'].apply(lambda x: f"{x:.2f}")
        df['Predator_Biomass'] = df['Predator_Biomass'].apply(lambda x: f"{x:.2f}")
        df['Ratio'] = df['Ratio'].apply(lambda x: f"{x:.3f}")

        # Sort by ratio descending
        df = df.sort_values('Ratio', ascending=False, key=lambda x: x.astype(float))

        return render.DataGrid(df, width="100%", height=UI.datagrid_height_tall_px)

    @output
    @render.data_frame
    def table_pb_ratios():
        """Render P/B ratios table."""
        report = diagnostic_report()

        if report is None or len(report.get('pb_ratios', [])) == 0:
            return pd.DataFrame()

        df = report['pb_ratios'].copy()

        # Format numeric columns
        df['Prey_Rate_Mean'] = df['Prey_Rate_Mean'].apply(lambda x: f"{x:.3f}")
        df['Predator_Rate'] = df['Predator_Rate'].apply(lambda x: f"{x:.3f}")
        df['Ratio'] = df['Ratio'].apply(lambda x: f"{x:.3f}")

        return render.DataGrid(df, width="100%", height="300px")

    @output
    @render.data_frame
    def table_qb_ratios():
        """Render Q/B ratios table."""
        report = diagnostic_report()

        if report is None or len(report.get('qb_ratios', [])) == 0:
            return pd.DataFrame()

        df = report['qb_ratios'].copy()

        # Format numeric columns
        df['Prey_Rate_Mean'] = df['Prey_Rate_Mean'].apply(lambda x: f"{x:.3f}")
        df['Predator_Rate'] = df['Predator_Rate'].apply(lambda x: f"{x:.3f}")
        df['Ratio'] = df['Ratio'].apply(lambda x: f"{x:.3f}")

        return render.DataGrid(df, width="100%", height="300px")

    @output
    @render.plot
    def diagnostic_plot():
        """Render diagnostic visualization."""
        report = diagnostic_report()
        data = model_data()

        if report is None or data is None:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(PLOTS.default_width, PLOTS.default_height))
            ax.text(
                0.5, 0.5,
                'No diagnostics run yet',
                ha='center', va='center',
                fontsize=14, color='gray'
            )
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            return fig

        # Parse excluded groups
        exclude_str = input.exclude_groups().strip()
        exclude_groups = [g.strip() for g in exclude_str.split(',') if g.strip()] if exclude_str else None

        # Generate plot based on selection
        plot_type = input.plot_type()

        try:
            if plot_type == "biomass":
                fig = plot_biomass_vs_trophic_level(
                    data,
                    exclude_groups=exclude_groups,
                    figsize=(PLOTS.default_width, PLOTS.default_height)
                )
            elif plot_type in ["pb", "qb"]:
                rate_name = plot_type.upper()
                fig = plot_vital_rate_vs_trophic_level(
                    data,
                    rate_name=rate_name,
                    exclude_groups=exclude_groups,
                    figsize=(PLOTS.default_width, PLOTS.default_height)
                )
            else:
                raise ValueError(f"Unknown plot type: {plot_type}")

            return fig

        except Exception as e:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(PLOTS.default_width, PLOTS.default_height))
            ax.text(
                0.5, 0.5,
                f'Error generating plot:\n{str(e)}',
                ha='center', va='center',
                fontsize=12, color='red'
            )
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            logger.error(f"Error generating diagnostic plot: {e}", exc_info=True)
            return fig
