"""Analysis page module - Network analysis, indicators, and advanced plots."""

import io

import matplotlib
import numpy as np
import pandas as pd
from shiny import Inputs, Outputs, Session, reactive, render, ui

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

# pypath imports (path setup handled by app/__init__.py)
from pypath.core.analysis import (
    calculate_network_indices,
    check_ecopath_balance,
    export_ecopath_to_dataframe,
    keystoneness_index,
    mixed_trophic_impacts,
)
from pypath.core.plotting import (
    plot_foodweb,
    plot_mti_heatmap,
    plot_trophic_spectrum,
)

# Import centralized logger and config
try:
    from app.config import THRESHOLDS, UI
    from app.logger import get_logger
    from app.pages.utils import is_balanced_model

    logger = get_logger(__name__)
except ModuleNotFoundError:
    import sys
    from pathlib import Path as PathLib

    app_dir = PathLib(__file__).parent.parent
    if str(app_dir) not in sys.path:
        sys.path.insert(0, str(app_dir))
    from config import THRESHOLDS, UI
    from logger import get_logger
    from pages.utils import is_balanced_model

    logger = get_logger(__name__)


def analysis_ui():
    """Analysis page UI."""
    return ui.page_fluid(
        ui.h2("Ecosystem Analysis", class_="mb-4"),
        ui.navset_card_tab(
            # Network Analysis
            ui.nav_panel(
                "Network Analysis",
                ui.h4("Network Indices", class_="mt-3"),
                ui.p(
                    "Ecological network analysis indices following Ulanowicz methodology."
                ),
                ui.output_ui("network_status"),
                ui.layout_columns(
                    ui.card(
                        ui.card_header("System Indices"),
                        ui.card_body(
                            ui.output_ui("system_indices"),
                        ),
                    ),
                    ui.card(
                        ui.card_header("Flow Indices"),
                        ui.card_body(
                            ui.output_ui("flow_indices"),
                        ),
                    ),
                    col_widths=[UI.col_width_medium, UI.col_width_medium],
                ),
                ui.h5("Food Web Structure", class_="mt-4"),
                ui.output_plot(
                    "analysis_foodweb_plot", height=UI.plot_height_medium_px
                ),
            ),
            # Trophic Analysis
            ui.nav_panel(
                "Trophic Analysis",
                ui.h4("Trophic Structure", class_="mt-3"),
                ui.output_ui("trophic_status"),
                ui.layout_columns(
                    ui.card(
                        ui.card_header("Trophic Level Summary"),
                        ui.card_body(
                            ui.output_table("trophic_summary_table"),
                        ),
                    ),
                    ui.card(
                        ui.card_header("Trophic Spectrum"),
                        ui.card_body(
                            ui.input_select(
                                "spectrum_metric",
                                "Metric",
                                choices={
                                    "biomass": "Biomass",
                                    "production": "Production",
                                },
                            ),
                            ui.output_plot(
                                "trophic_spectrum_plot", height=UI.plot_height_small_px
                            ),
                        ),
                    ),
                    col_widths=[5, 7],
                ),
            ),
            # Mixed Trophic Impacts
            ui.nav_panel(
                "Trophic Impacts",
                ui.h4("Mixed Trophic Impacts (MTI)", class_="mt-3"),
                ui.p(
                    "MTI quantifies the direct and indirect effects of changes in one group's biomass "
                    "on all other groups. Positive values indicate positive impacts."
                ),
                ui.output_ui("mti_status"),
                ui.output_plot("mti_heatmap_plot", height=UI.plot_height_large_px),
                ui.tags.hr(),
                ui.h5("MTI Details"),
                ui.layout_columns(
                    ui.card(
                        ui.card_header("Top Positive Impacts"),
                        ui.card_body(
                            ui.output_table("mti_positive_table"),
                        ),
                    ),
                    ui.card(
                        ui.card_header("Top Negative Impacts"),
                        ui.card_body(
                            ui.output_table("mti_negative_table"),
                        ),
                    ),
                    col_widths=[UI.col_width_medium, UI.col_width_medium],
                ),
            ),
            # Keystoneness
            ui.nav_panel(
                "Keystoneness",
                ui.h4("Keystone Species Analysis", class_="mt-3"),
                ui.p(
                    "Keystoneness identifies species with disproportionately large ecological effects "
                    "relative to their biomass (Power et al. 1996, Libralato et al. 2006)."
                ),
                ui.output_ui("keystone_status"),
                ui.layout_columns(
                    ui.card(
                        ui.card_header("Top Keystone Species"),
                        ui.card_body(
                            ui.output_table("keystoneness_table"),
                        ),
                    ),
                    ui.card(
                        ui.card_header("Keystoneness vs Biomass"),
                        ui.card_body(
                            ui.output_plot(
                                "keystoneness_plot", height=UI.plot_height_small_px
                            ),
                        ),
                    ),
                    col_widths=[5, 7],
                ),
            ),
            # Model Balance Check
            ui.nav_panel(
                "Balance Check",
                ui.h4("Ecopath Balance Diagnostics", class_="mt-3"),
                ui.p("Check mass balance status and identify potential issues."),
                ui.output_ui("analysis_balance_status"),
                ui.layout_columns(
                    ui.card(
                        ui.card_header("Balance Summary"),
                        ui.card_body(
                            ui.output_ui("balance_summary"),
                        ),
                    ),
                    ui.card(
                        ui.card_header("EE Values"),
                        ui.card_body(
                            ui.output_plot(
                                "analysis_ee_plot", height=UI.plot_height_small_px
                            ),
                        ),
                    ),
                    col_widths=[5, 7],
                ),
                ui.h5("Detailed Diagnostics", class_="mt-4"),
                ui.output_table("balance_details_table"),
            ),
            # Model Export
            ui.nav_panel(
                "Export Data",
                ui.h4("Export Model Data", class_="mt-3"),
                ui.p("Export Ecopath model data to DataFrames for further analysis."),
                ui.output_ui("export_status"),
                ui.layout_columns(
                    ui.card(
                        ui.card_header("Basic Parameters"),
                        ui.card_body(
                            ui.output_table("export_params_table"),
                        ),
                    ),
                    ui.card(
                        ui.card_header("Diet Matrix"),
                        ui.card_body(
                            ui.output_table("export_diet_table"),
                        ),
                    ),
                    col_widths=[UI.col_width_medium, UI.col_width_medium],
                ),
                ui.tags.hr(),
                ui.download_button(
                    "download_model_data",
                    "Download All Data (CSV)",
                    class_="btn-primary",
                ),
            ),
        ),
    )


def analysis_server(
    input: Inputs,
    output: Outputs,
    session: Session,
    model_data: reactive.Value,
    sim_results: reactive.Value,
):
    """Analysis page server logic."""

    # Reactive calculations
    @reactive.calc
    def get_balanced_model():
        """Get the balanced Rpath model if available."""
        data = model_data.get()
        if data is None:
            return None
        # Check if it's a balanced model (Rpath) or just params
        if is_balanced_model(data):
            return data
        return None

    @reactive.calc
    def get_network_indices():
        """Calculate network indices."""
        model = get_balanced_model()
        if model is None:
            return None
        try:
            return calculate_network_indices(model)
        except Exception as e:
            logger.error(f"Error calculating network indices: {e}", exc_info=True)
            return None

    @reactive.calc
    def get_mti_matrix():
        """Calculate MTI matrix."""
        model = get_balanced_model()
        if model is None:
            return None
        try:
            return mixed_trophic_impacts(model)
        except Exception as e:
            logger.error(f"Error calculating MTI: {e}", exc_info=True)
            return None

    @reactive.calc
    def get_keystoneness():
        """Calculate keystoneness index."""
        model = get_balanced_model()
        if model is None:
            return None
        try:
            return keystoneness_index(model)
        except Exception as e:
            logger.error(f"Error calculating keystoneness: {e}", exc_info=True)
            return None

    @reactive.calc
    def get_balance_check():
        """Run balance check."""
        model = get_balanced_model()
        if model is None:
            return None
        try:
            return check_ecopath_balance(model)
        except Exception as e:
            logger.error(f"Error checking balance: {e}", exc_info=True)
            return None

    # === Network Analysis ===

    @output
    @render.ui
    def network_status():
        """Show network analysis status."""
        model = get_balanced_model()
        if model is None:
            return ui.div(
                ui.tags.i(class_="bi bi-info-circle me-2"),
                "Balance an Ecopath model first to see network analysis.",
                class_="alert alert-info",
            )
        return None

    @output
    @render.ui
    def system_indices():
        """Display system-level indices."""
        indices = get_network_indices()
        if indices is None:
            return ui.p("No indices available.", class_="text-muted")

        rows = []
        # NetworkIndices is a dataclass, access via attributes
        system_attrs = [
            ("total_throughput", "Total System Throughput", "t/km²/yr"),
            ("total_production", "Total Production", "t/km²/yr"),
            ("total_consumption", "Total Consumption", "t/km²/yr"),
            ("total_respiration", "Total Respiration", "t/km²/yr"),
            ("total_biomass", "Total Biomass", "t/km²"),
            ("system_omnivory_index", "System Omnivory Index", ""),
        ]

        for attr, label, unit in system_attrs:
            if hasattr(indices, attr):
                value = getattr(indices, attr)
                if value is not None:
                    rows.append(
                        ui.tags.tr(
                            ui.tags.td(label), ui.tags.td(f"{value:.3f} {unit}".strip())
                        )
                    )

        if not rows:
            return ui.p("No system indices calculated.", class_="text-muted")

        return ui.tags.table(ui.tags.tbody(*rows), class_="table table-sm")

    @output
    @render.ui
    def flow_indices():
        """Display flow-related indices."""
        indices = get_network_indices()
        if indices is None:
            return ui.p("No indices available.", class_="text-muted")

        rows = []
        flow_attrs = [
            ("ascendency", "Ascendency", ""),
            ("development_capacity", "Development Capacity", ""),
            ("overhead", "Overhead", ""),
            ("finn_cycling_index", "Finn Cycling Index", "%"),
            ("connectance", "Connectance", ""),
            ("num_links", "Number of Links", ""),
        ]

        for attr, label, unit in flow_attrs:
            if hasattr(indices, attr):
                value = getattr(indices, attr)
                if value is not None:
                    if unit == "%":
                        rows.append(
                            ui.tags.tr(
                                ui.tags.td(label), ui.tags.td(f"{value * 100:.1f}%")
                            )
                        )
                    else:
                        rows.append(
                            ui.tags.tr(
                                ui.tags.td(label),
                                ui.tags.td(
                                    f"{value:.3f}"
                                    if isinstance(value, float)
                                    else str(value)
                                ),
                            )
                        )

        if not rows:
            return ui.p("No flow indices calculated.", class_="text-muted")

        return ui.tags.table(ui.tags.tbody(*rows), class_="table table-sm")

    @output
    @render.plot
    def analysis_foodweb_plot():
        """Plot food web structure."""
        model = get_balanced_model()
        if model is None:
            return None
        try:
            fig = plot_foodweb(model)
            return fig
        except Exception as e:
            fig, ax = plt.subplots()
            ax.text(
                0.5,
                0.5,
                f"Could not plot food web:\n{str(e)[:50]}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.axis("off")
            return fig

    # === Trophic Analysis ===

    @output
    @render.ui
    def trophic_status():
        """Show trophic analysis status."""
        model = get_balanced_model()
        if model is None:
            return ui.div(
                ui.tags.i(class_="bi bi-info-circle me-2"),
                "Balance an Ecopath model to see trophic analysis.",
                class_="alert alert-info",
            )
        return None

    @output
    @render.table
    def trophic_summary_table():
        """Display trophic level summary."""
        model = get_balanced_model()
        if model is None:
            return pd.DataFrame({"Message": ["Balance model first"]})

        try:
            groups = model.params.model["Group"].values
            tl = model.trophic_level
            biomass = model.params.model["Biomass"].values

            df = pd.DataFrame(
                {
                    "Group": groups,
                    "Trophic Level": np.round(tl, 2),
                    "Biomass": np.round(biomass, 3),
                }
            )
            df = df.sort_values("Trophic Level", ascending=False).head(15)
            return df
        except Exception as e:
            logger.error(f"Error extracting trophic data: {e}", exc_info=True)
            return pd.DataFrame({"Message": ["Could not extract trophic data"]})

    @output
    @render.plot
    def trophic_spectrum_plot():
        """Plot trophic spectrum."""
        model = get_balanced_model()
        if model is None:
            return None

        try:
            metric = input.spectrum_metric()
            fig = plot_trophic_spectrum(model, metric=metric)
            return fig
        except Exception as e:
            fig, ax = plt.subplots()
            ax.text(
                0.5,
                0.5,
                f"Could not plot spectrum:\n{str(e)[:50]}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.axis("off")
            return fig

    # === Mixed Trophic Impacts ===

    @output
    @render.ui
    def mti_status():
        """Show MTI analysis status."""
        mti = get_mti_matrix()
        if mti is None:
            return ui.div(
                ui.tags.i(class_="bi bi-info-circle me-2"),
                "Balance an Ecopath model to calculate Mixed Trophic Impacts.",
                class_="alert alert-info",
            )
        return None

    @output
    @render.plot
    def mti_heatmap_plot():
        """Plot MTI heatmap."""
        mti = get_mti_matrix()
        model = get_balanced_model()
        if mti is None or model is None:
            return None

        try:
            fig = plot_mti_heatmap(mti, model)
            return fig
        except Exception as e:
            fig, ax = plt.subplots()
            ax.text(
                0.5,
                0.5,
                f"Could not plot MTI:\n{str(e)[:50]}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.axis("off")
            return fig

    @output
    @render.table
    def mti_positive_table():
        """Show top positive MTI impacts."""
        mti = get_mti_matrix()
        model = get_balanced_model()
        if mti is None or model is None:
            return pd.DataFrame({"Message": ["No MTI available"]})

        try:
            groups = model.params.model["Group"].values

            # Flatten matrix and find top impacts
            impacts = []
            for i, g1 in enumerate(groups):
                for j, g2 in enumerate(groups):
                    if i != j:
                        impacts.append({"From": g1, "To": g2, "Impact": mti[i, j]})

            df = pd.DataFrame(impacts)
            df = df.nlargest(10, "Impact")
            df["Impact"] = df["Impact"].round(4)
            return df
        except Exception as e:
            logger.error(f"Error extracting positive impacts: {e}", exc_info=True)
            return pd.DataFrame({"Message": ["Could not extract impacts"]})

    @output
    @render.table
    def mti_negative_table():
        """Show top negative MTI impacts."""
        mti = get_mti_matrix()
        model = get_balanced_model()
        if mti is None or model is None:
            return pd.DataFrame({"Message": ["No MTI available"]})

        try:
            groups = model.params.model["Group"].values

            # Flatten matrix and find top negative impacts
            impacts = []
            for i, g1 in enumerate(groups):
                for j, g2 in enumerate(groups):
                    if i != j:
                        impacts.append({"From": g1, "To": g2, "Impact": mti[i, j]})

            df = pd.DataFrame(impacts)
            df = df.nsmallest(10, "Impact")
            df["Impact"] = df["Impact"].round(4)
            return df
        except Exception as e:
            logger.error(f"Error extracting negative impacts: {e}", exc_info=True)
            return pd.DataFrame({"Message": ["Could not extract impacts"]})

    # === Keystoneness ===

    @output
    @render.ui
    def keystone_status():
        """Show keystoneness status."""
        ks = get_keystoneness()
        if ks is None:
            return ui.div(
                ui.tags.i(class_="bi bi-info-circle me-2"),
                "Balance an Ecopath model to calculate keystoneness.",
                class_="alert alert-info",
            )
        return None

    @output
    @render.table
    def keystoneness_table():
        """Display keystoneness table."""
        ks = get_keystoneness()
        model = get_balanced_model()
        if ks is None or model is None:
            return pd.DataFrame({"Message": ["Balance model first"]})

        try:
            groups = model.params.model["Group"].values

            # Create DataFrame with group names
            df = pd.DataFrame(
                {"Group": groups[: len(ks)], "Keystoneness": np.round(ks, 4)}
            )
            df = df.sort_values("Keystoneness", ascending=False).head(10)
            return df
        except Exception as e:
            logger.error(f"Error extracting keystoneness data: {e}", exc_info=True)
            return pd.DataFrame({"Message": ["Could not extract keystoneness"]})

    @output
    @render.plot
    def keystoneness_plot():
        """Plot keystoneness vs biomass."""
        model = get_balanced_model()
        ks = get_keystoneness()
        if model is None or ks is None:
            return None

        try:
            groups = model.params.model["Group"].values
            biomass = model.params.model["Biomass"].values

            fig, ax = plt.subplots(figsize=(8, 6))

            # Filter valid points
            valid = (biomass > 0) & (~np.isnan(ks[: len(biomass)]))

            _scatter = ax.scatter(
                np.log10(biomass[valid] + THRESHOLDS.log_offset_small),
                ks[: len(biomass)][valid],
                s=100,
                alpha=0.7,
                c="steelblue",
            )

            # Annotate top species
            for i, (g, b, k) in enumerate(
                zip(groups[valid], biomass[valid], ks[: len(biomass)][valid])
            ):
                if k > np.percentile(ks[~np.isnan(ks)], 75):
                    ax.annotate(
                        g,
                        (np.log10(b + THRESHOLDS.log_offset_small), k),
                        fontsize=8,
                        ha="left",
                    )

            ax.set_xlabel("Log10(Biomass)")
            ax.set_ylabel("Keystoneness Index")
            ax.set_title("Keystoneness vs Biomass")
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            return fig
        except Exception as e:
            fig, ax = plt.subplots()
            ax.text(
                0.5,
                0.5,
                f"Could not plot:\n{str(e)[:50]}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.axis("off")
            return fig

    # === Balance Check ===

    @output
    @render.ui
    def analysis_balance_status():
        """Show balance check status."""
        model = get_balanced_model()
        if model is None:
            return ui.div(
                ui.tags.i(class_="bi bi-info-circle me-2"),
                "Balance an Ecopath model to see diagnostics.",
                class_="alert alert-info",
            )
        return None

    @output
    @render.ui
    def balance_summary():
        """Display balance summary."""
        check = get_balance_check()
        if check is None:
            return ui.p("No balance check available.", class_="text-muted")

        is_balanced = check.get("balanced", False)
        issues = check.get("issues", [])

        badge_class = "bg-success" if is_balanced else "bg-warning"
        status_text = "Balanced" if is_balanced else "Issues Found"

        items = [
            ui.div(
                ui.tags.span(status_text, class_=f"badge {badge_class} fs-6"),
                class_="mb-3",
            )
        ]

        if issues:
            items.append(ui.h6("Issues:"))
            items.append(
                ui.tags.ul(
                    *[ui.tags.li(issue) for issue in issues[:10]], class_="text-warning"
                )
            )

        return ui.div(*items)

    @output
    @render.plot
    def analysis_ee_plot():
        """Plot ecotrophic efficiency values."""
        model = get_balanced_model()
        if model is None:
            return None

        try:
            groups = model.params.model["Group"].values
            ee = model.params.model["EE"].values

            fig, ax = plt.subplots(figsize=(10, 6))

            # Color by EE value
            colors = ["green" if e <= 1 else "red" for e in ee]

            y_pos = range(len(groups))
            ax.barh(y_pos, ee, color=colors, alpha=0.7)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(groups, fontsize=8)
            ax.axvline(1, color="red", linestyle="--", linewidth=2, label="EE = 1")
            ax.set_xlabel("Ecotrophic Efficiency (EE)")
            ax.set_title("Ecotrophic Efficiency by Group")
            ax.legend()
            ax.grid(True, alpha=0.3, axis="x")

            plt.tight_layout()
            return fig
        except Exception as e:
            fig, ax = plt.subplots()
            ax.text(
                0.5,
                0.5,
                f"Could not plot EE:\n{str(e)[:50]}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.axis("off")
            return fig

    @output
    @render.table
    def balance_details_table():
        """Display detailed balance diagnostics."""
        check = get_balance_check()
        model = get_balanced_model()
        if check is None or model is None:
            return pd.DataFrame({"Message": ["No diagnostics available"]})

        try:
            groups = model.params.model["Group"].values
            ee = model.params.model["EE"].values
            biomass = model.params.model["Biomass"].values
            pb = model.params.model["PB"].values
            qb = model.params.model["QB"].values

            df = pd.DataFrame(
                {
                    "Group": groups,
                    "EE": np.round(ee, 3),
                    "Biomass": np.round(biomass, 3),
                    "P/B": np.round(pb, 3),
                    "Q/B": np.round(qb, 3),
                    "P/Q": np.round(pb / np.where(qb > 0, qb, np.nan), 3),
                }
            )

            # Mark issues
            df["Status"] = np.where(ee > 1, "⚠️ EE>1", "✓")

            return df
        except Exception as e:
            logger.error(f"Error extracting balance diagnostics: {e}", exc_info=True)
            return pd.DataFrame({"Message": ["Could not extract diagnostics"]})

    # === Export Data ===

    @output
    @render.ui
    def export_status():
        """Show export status."""
        model = get_balanced_model()
        if model is None:
            return ui.div(
                ui.tags.i(class_="bi bi-info-circle me-2"),
                "Balance an Ecopath model to export data.",
                class_="alert alert-info",
            )
        return None

    @output
    @render.table
    def export_params_table():
        """Display basic parameters table."""
        model = get_balanced_model()
        if model is None:
            return pd.DataFrame({"Message": ["No model available"]})

        try:
            df = model.params.model[
                ["Group", "Type", "Biomass", "PB", "QB", "EE"]
            ].copy()
            df = df.round(3)
            return df.head(15)
        except Exception as e:
            logger.error(f"Error extracting model parameters: {e}", exc_info=True)
            return pd.DataFrame({"Message": ["Could not extract parameters"]})

    @output
    @render.table
    def export_diet_table():
        """Display diet matrix preview."""
        model = get_balanced_model()
        if model is None:
            return pd.DataFrame({"Message": ["No model available"]})

        try:
            diet = model.params.diet.copy()
            diet = diet.round(3)
            # Show only first 10 columns
            cols = diet.columns[:10].tolist()
            return diet[cols].head(10)
        except Exception as e:
            logger.error(f"Error extracting diet matrix: {e}", exc_info=True)
            return pd.DataFrame({"Message": ["Could not extract diet matrix"]})

    @output
    @render.download(filename="pypath_model_data.csv")
    def download_model_data():
        """Download model data as CSV."""
        model = get_balanced_model()
        if model is None:
            yield "No model data available"
            return

        try:
            dfs = export_ecopath_to_dataframe(model)

            # Combine into single file with sections
            output = io.StringIO()

            for name, df in dfs.items():
                output.write(f"\n=== {name.upper()} ===\n")
                df.to_csv(output, index=True)
                output.write("\n")

            yield output.getvalue()
        except Exception as e:
            yield f"Error exporting: {str(e)}"
