"""Pre-balance Diagnostics Page.

This module provides an interactive interface for pre-balance diagnostic analysis
of Ecopath models before balancing. It helps identify potential issues with
biomasses, vital rates, and predator-prey relationships.

Based on the Prebal routine by Barbara Bauer (SU, 2016).
"""

import logging
from pathlib import Path

import pandas as pd
from shiny import Inputs, Outputs, Session, reactive, render, ui

# Get logger
logger = logging.getLogger("pypath_app.prebalance")

try:
    from app.config import PLOTS, UI
    from app.pages.utils import is_rpath_params
except ModuleNotFoundError:
    from config import PLOTS, UI
    from pages.utils import is_rpath_params

# Prebalance functions are imported lazily inside the diagnostics handler to avoid path issues
# and to keep top-level imports clean.

# Diagnostics helper (uses loader from utils)
try:
    from app.pages.utils import load_rpath_diagnostics
except Exception:
    # Fallback import path
    from pages.utils import load_rpath_diagnostics


def rpath_diagnostics_summary(diag_dir: str | Path = "tests/data/rpath_reference/ecosim/diagnostics") -> str:
    """Return a short summary string about the Rpath diagnostics state.

    This is a pure function intended for server-side checks and unit tests.
    It does not access UI elements.
    """
    out = load_rpath_diagnostics(Path(diag_dir))
    if out.get("meta") is None:
        return "No diagnostics available"
    if out.get("errors"):
        return f"Diagnostics incomplete: {len(out.get('errors'))} error(s)"
    if out.get("qq_provided"):
        return "Rpath QQ diagnostics provided"
    return "Rpath QQ not provided"


def make_rpath_status_badge(
    status: str, note: str | None = None, link: str | None = None
) -> "ui.Tag":
    """Return a small Bootstrap badge UI Tag reflecting diagnostics `status`.

    - Provided -> green badge
    - Incomplete -> yellow badge
    - Not provided / No diagnostics -> gray badge

    Optional `note` will be used as a tooltip (`title` attribute). Optional `link`
    will wrap the badge in an anchor to the diagnostics location.
    """
    """Return a small Bootstrap badge UI Tag reflecting diagnostics `status`.

    - Provided -> green badge
    - Incomplete -> yellow badge
    - Not provided / No diagnostics -> gray badge

    Optional `note` will be used as a tooltip (`title` attribute). Optional `link`
    will wrap the badge in an anchor to the diagnostics location.
    """
    # Lazy import to keep module import-safe in tests
    try:
        from shiny import ui as _ui
    except Exception:
        # Fallback dummy representation if Shiny not available
        parts = [f"Status: {status}"]
        if note:
            parts.append(f"note: {note}")
        if link:
            parts.append(f"link: {link}")
        return " | ".join(parts)

    cls = "badge bg-secondary"
    if "provided" in status.lower():
        cls = "badge bg-success"
    elif "incomplete" in status.lower():
        cls = "badge bg-warning text-dark"

    badge = _ui.tags.span(status, class_=cls, title=note if note else None)

    if link:
        # Wrap in an anchor that opens in a new tab/window
        return _ui.tags.a(badge, href=link, target="_blank", rel="noopener noreferrer")

    return badge


def run_verify_rpath(diag_dir: str | Path = "tests/data/rpath_reference/ecosim/diagnostics") -> dict:
    """Execute the verification script and return structured results.

    Returns a dict with keys:
      - returncode: int
      - output: str (combined stdout+stderr)
      - error: optional error message on failure

    This runs the script using the current Python interpreter for portability.
    """
    import subprocess
    import sys

    script = Path("scripts/verify_rpath_reference.py")
    if not script.exists():
        return {"returncode": -1, "output": "verify script not found", "error": "missing_script"}

    try:
        proc = subprocess.run(
            [sys.executable, str(script)], capture_output=True, text=True, check=False, timeout=30
        )
        out = (proc.stdout or "") + "\n" + (proc.stderr or "")
        # Truncate to reasonable length
        if len(out) > 20000:
            out = out[:20000] + "\n...output truncated..."
        return {"returncode": int(proc.returncode), "output": out}
    except subprocess.TimeoutExpired:
        return {"returncode": -2, "output": "", "error": "timeout"}
    except Exception as e:
        return {"returncode": -3, "output": "", "error": str(e)}





def prebalance_ui():
    """Pre-balance diagnostics UI."""
    return ui.page_fluid(
        ui.layout_sidebar(
            ui.sidebar(
                ui.h4("Pre-Balance Diagnostics"),
                # Rpath diagnostics status badge + info button
                ui.div(
                    ui.output_ui("rpath_diag_status"),
                    ui.input_action_button(
                        "btn_rpath_diag_info",
                        "",
                        class_="btn-sm btn-outline-info ms-2",
                        icon=ui.tags.i(class_="bi bi-info-circle"),
                    ),
                    # Inline fallback content (rendered when `session.show_modal` is unavailable)
                    ui.output_ui("rpath_modal_inline"),
                    style="display:inline-flex; align-items:center; gap:0.5rem;",
                ),
                ui.p(
                    "Run diagnostic checks on your unbalanced model to identify "
                    "potential issues before balancing.",
                    class_="text-muted",
                ),
                ui.hr(),
                ui.input_action_button(
                    "btn_run_diagnostics",
                    "Run Diagnostics",
                    class_="btn-primary w-100 mb-3",
                    icon=ui.tags.i(class_="bi bi-play-circle"),
                ),
                ui.hr(),
                ui.card(
                    ui.card_body(
                        ui.h6("Visualization Options"),
                        ui.input_select(
                            "plot_type",
                            "Plot Type",
                            choices={
                                "biomass": "Biomass vs Trophic Level",
                                "pb": "P/B vs Trophic Level",
                                "qb": "Q/B vs Trophic Level",
                            },
                            selected="biomass",
                        ),
                        ui.input_text(
                            "exclude_groups",
                            "Exclude Groups (comma-separated)",
                            value="",
                            placeholder="e.g., Whales, Seabirds",
                        ),
                    )
                ),
                ui.hr(),
                ui.card(
                    ui.card_body(
                        ui.h6("About Pre-Balance Diagnostics"),
                        ui.tags.small(
                            ui.tags.ul(
                                ui.tags.li(
                                    ui.tags.strong("Biomass Slope:"),
                                    " Indicates top-down control strength (-0.5 to -1.5 typical)",
                                ),
                                ui.tags.li(
                                    ui.tags.strong("Biomass Range:"),
                                    " Large ranges (>6 orders) may indicate missing groups",
                                ),
                                ui.tags.li(
                                    ui.tags.strong("Predator/Prey Ratio:"),
                                    " High ratios (>1) suggest unsustainable predation",
                                ),
                                ui.tags.li(
                                    ui.tags.strong("Vital Rate Ratios:"),
                                    " Predator rates should be lower than prey rates",
                                ),
                                class_="small",
                            ),
                            class_="text-muted",
                        ),
                    )
                ),
                width=UI.sidebar_width,
                position="left",
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
                    ),
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
                    ),
                ),
            ),
        )
    )


def prebalance_server(
    input: Inputs, output: Outputs, session: Session, model_data: reactive.Value
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
    # Fallback container for modal content when session.show_modal() is not available
    rpath_modal_content = reactive.Value(None)

    # Populate an initial verification summary so the UI always contains the
    # verification output text even when `session.show_modal` is unsupported or
    # button clicks don't register in certain runtime variants.
    try:
        _res = run_verify_rpath(Path("tests/data/rpath_reference/ecosim/diagnostics"))
        _note = load_rpath_diagnostics(Path("tests/data/rpath_reference/ecosim/diagnostics")).get("note")
        _body = ui.tags.div(
            ui.h5("Meta note:"),
            ui.tags.pre(str(_note) if _note is not None else "(none)"),
            ui.hr(),
            ui.h5("Verification output:"),
            ui.tags.pre(_res.get("output", "")),
        )
        rpath_modal_content.set(_body)
    except Exception as _e:
        rpath_modal_content.set(ui.tags.div(ui.tags.p(str(_e))))

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
                    duration=5,
                )
                return

            # Check if model is unbalanced (RpathParams)
            if not is_rpath_params(data):
                ui.notification_show(
                    "Pre-balance diagnostics require an unbalanced model (RpathParams). "
                    "The current model appears to be already balanced.",
                    type="warning",
                    duration=5,
                )
                return

            ui.notification_show("Running diagnostics...", duration=3)

            # Lazy import to avoid top-level path manipulation and E402
            try:
                from pypath.analysis.prebalance import generate_prebalance_report
            except Exception:
                import sys

                root_dir = Path(__file__).parent.parent.parent
                if str(root_dir) not in sys.path:
                    sys.path.insert(0, str(root_dir))
                from src.pypath.analysis.prebalance import generate_prebalance_report

            # Generate diagnostic report
            report = generate_prebalance_report(data)
            diagnostic_report.set(report)

            # Show completion notification
            num_warnings = len(report["warnings"])
            if num_warnings == 0:
                ui.notification_show(
                    "Diagnostics complete! No major issues detected.",
                    type="message",
                    duration=5,
                )
            else:
                ui.notification_show(
                    f"Diagnostics complete. Found {num_warnings} warning(s). Check the Warnings tab.",
                    type="warning",
                    duration=5,
                )

        except Exception as e:
            logger.error(f"Error running diagnostics: {e}", exc_info=True)
            ui.notification_show(
                f"Error running diagnostics: {str(e)}", type="error", duration=5
            )

    @output
    @render.ui
    def report_summary():
        """Render diagnostic summary report."""
        report = diagnostic_report()

        # Defensive: if diagnostics haven't been run yet, show a friendly placeholder
        if report is None:
            return ui.tags.div(
                ui.tags.p("No diagnostics run yet.", class_="text-muted text-center p-5")
            )

    # rpath_diag_status removed from here and defined after report_summary to avoid
    # shadowing report rendering. The function renders a small status badge in the
    # sidebar using `make_rpath_status_badge`.

        # Format summary statistics defensively
        def _safe_fmt(value, fmt="{:.3f}", na_text="n/a"):
            try:
                if value is None:
                    return na_text
                return fmt.format(value)
            except Exception:
                return na_text

        br_text = _safe_fmt(report.get("biomass_range"), "{:.2f}", "n/a")
        bs_text = _safe_fmt(report.get("biomass_slope"), "{:.3f}", "n/a")

        summary_cards = [
            ui.div(
                ui.h5("Biomass Diagnostics", class_="card-title"),
                ui.tags.dl(
                    ui.tags.dt("Biomass Range:"),
                    ui.tags.dd(f"{br_text} orders of magnitude"),
                    ui.tags.dt("Biomass Slope:"),
                    ui.tags.dd(f"{bs_text}"),
                ),
                class_="card-body",
            ),
        ]

        # Predator-prey summary
        if report.get("predator_prey_ratios") is not None and len(report.get("predator_prey_ratios")) > 0:
            pp_ratios = report["predator_prey_ratios"].get("Ratio")
            # Defensive aggregation
            try:
                n_pp = len(pp_ratios)
                mean_pp = f"{pp_ratios.mean():.3f}" if n_pp > 0 else "n/a"
                max_pp = f"{pp_ratios.max():.3f}" if n_pp > 0 else "n/a"
                n_gt1 = int((pp_ratios > 1.0).sum()) if n_pp > 0 else 0
            except Exception:
                n_pp = 0
                mean_pp = "n/a"
                max_pp = "n/a"
                n_gt1 = 0

            summary_cards.append(
                ui.div(
                    ui.h5("Predator-Prey Ratios", class_="card-title"),
                    ui.tags.dl(
                        ui.tags.dt("Number of predators analyzed:"),
                        ui.tags.dd(f"{n_pp}"),
                        ui.tags.dt("Mean ratio:"),
                        ui.tags.dd(mean_pp),
                        ui.tags.dt("Max ratio:"),
                        ui.tags.dd(max_pp),
                        ui.tags.dt("Ratios > 1.0:"),
                        ui.tags.dd(f"{n_gt1} (potentially unsustainable)"),
                    ),
                    class_="card-body",
                )
            )

        # Vital rate summaries
        if len(report.get("pb_ratios", [])) > 0:
            pb_ratios = report["pb_ratios"]["Ratio"]
            summary_cards.append(
                ui.div(
                    ui.h5("P/B Rate Ratios", class_="card-title"),
                    ui.tags.dl(
                        ui.tags.dt("Mean P/B ratio (Predator/Prey):"),
                        ui.tags.dd(f"{pb_ratios.mean():.3f}"),
                        ui.tags.dt("Number analyzed:"),
                        ui.tags.dd(f"{len(pb_ratios)}"),
                    ),
                    class_="card-body",
                )
            )

        if len(report.get("qb_ratios", [])) > 0:
            qb_ratios = report["qb_ratios"]["Ratio"]
            summary_cards.append(
                ui.div(
                    ui.h5("Q/B Rate Ratios", class_="card-title"),
                    ui.tags.dl(
                        ui.tags.dt("Mean Q/B ratio (Predator/Prey):"),
                        ui.tags.dd(f"{qb_ratios.mean():.3f}"),
                        ui.tags.dt("Number analyzed:"),
                        ui.tags.dd(f"{len(qb_ratios)}"),
                    ),
                    class_="card-body",
                )
            )

        return ui.tags.div(
            ui.row(
                *[
                    ui.column(6, ui.div(card, class_="card mb-3"))
                    for card in summary_cards
                ]
            )
        )

    @output
    @render.ui
    def rpath_diag_status():
        """Render an inline UI badge describing Rpath diagnostics state.

        Uses the diagnostics loader to show any `meta.note` as a tooltip and
        wraps the badge in a link to the diagnostics folder using a `file://` URI
        when available.
        """
        diag_dir = Path("tests/data/rpath_reference/ecosim/diagnostics")
        diag = load_rpath_diagnostics(diag_dir)
        status = rpath_diagnostics_summary(diag_dir)
        note = diag.get("note") if diag else None
        try:
            link = Path(diag_dir).resolve().as_uri()
        except Exception:
            link = None
        # When info button is clicked we show details. If the runtime doesn't
        # support modals we'll expose the output inline in `rpath_modal_inline`.
        return make_rpath_status_badge(status, note=note, link=link)

    @output
    @render.ui
    def rpath_modal_inline():
        """Inline fallback for modal content when `session.show_modal` is not present.

        This provides a stable, testable location containing the verification
        output so headless smoke tests can find the content even in older
        Shiny runtimes that lack the `show_modal` helper.
        """
        body = rpath_modal_content()
        if body is None:
            return ui.tags.div()
        return ui.tags.div(body, id="rpath_modal_inline", class_="card mb-3 p-3")

    @reactive.effect
    @reactive.event(input.btn_rpath_diag_info)
    def _show_rpath_modal():
        """Show a modal dialog containing `meta.note` and the output of the verifier.

        If `session.show_modal` is not available (older Shiny runtime), fall back
        to setting `rpath_modal_content` which is rendered inline via a UI output
        (`rpath_modal_inline`). This avoids unhandled exceptions on the server
        and keeps the diagnostic content discoverable by tests and Playwright.
        """
        try:
            result = run_verify_rpath(Path("tests/data/rpath_reference/ecosim/diagnostics"))
            note = load_rpath_diagnostics(Path("tests/data/rpath_reference/ecosim/diagnostics")).get("note")
            title = "Rpath Diagnostics"
            body = ui.tags.div(
                ui.h5("Meta note:"),
                ui.tags.pre(str(note) if note is not None else "(none)"),
                ui.hr(),
                ui.h5("Verification output:"),
                ui.tags.pre(result.get("output", "")),
            )
            # Preferred: show modal if session supports it
            try:
                session.show_modal(ui.modal_dialog(body, title=title, size="lg"))
            except AttributeError:
                # Fallback: expose the body via a reactive value rendered inline
                rpath_modal_content.set(body)
        except Exception as e:
            err_body = ui.tags.div(ui.tags.p(str(e)))
            try:
                session.show_modal(ui.modal_dialog(err_body, title="Rpath Diagnostics - Error"))
            except AttributeError:
                rpath_modal_content.set(err_body)

    @output
    @render.ui
    def report_warnings():
        """Render diagnostic warnings."""
        report = diagnostic_report()

        if report is None:
            return ui.tags.div(
                ui.tags.p(
                    "No diagnostics run yet.", class_="text-muted text-center p-5"
                )
            )

        warnings = report["warnings"]

        if len(warnings) == 0:
            return ui.tags.div(
                ui.div(
                    ui.tags.i(
                        class_="bi bi-check-circle-fill text-success",
                        style="font-size: 3rem;",
                    ),
                    ui.h4("No major issues detected!", class_="mt-3"),
                    ui.p(
                        "Your model passed all pre-balance diagnostic checks. "
                        "You can proceed with balancing.",
                        class_="text-muted",
                    ),
                    class_="text-center p-5",
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
                    role="alert",
                )
            )

        return ui.tags.div(
            ui.h5(f"Found {len(warnings)} Warning(s)"), ui.hr(), *warning_items
        )

    @output
    @render.data_frame
    def table_predator_prey():
        """Render predator-prey ratios table."""
        report = diagnostic_report()

        if report is None or len(report["predator_prey_ratios"]) == 0:
            return pd.DataFrame()

        df = report["predator_prey_ratios"].copy()

        # Format numeric columns
        df["Prey_Biomass"] = df["Prey_Biomass"].apply(lambda x: f"{x:.2f}")
        df["Predator_Biomass"] = df["Predator_Biomass"].apply(lambda x: f"{x:.2f}")
        df["Ratio"] = df["Ratio"].apply(lambda x: f"{x:.3f}")

        # Sort by ratio descending
        df = df.sort_values("Ratio", ascending=False, key=lambda x: x.astype(float))

        return render.DataGrid(df, width="100%", height=UI.datagrid_height_tall_px)

    @output
    @render.data_frame
    def table_pb_ratios():
        """Render P/B ratios table."""
        report = diagnostic_report()

        if report is None or len(report.get("pb_ratios", [])) == 0:
            return pd.DataFrame()

        df = report["pb_ratios"].copy()

        # Format numeric columns
        df["Prey_Rate_Mean"] = df["Prey_Rate_Mean"].apply(lambda x: f"{x:.3f}")
        df["Predator_Rate"] = df["Predator_Rate"].apply(lambda x: f"{x:.3f}")
        df["Ratio"] = df["Ratio"].apply(lambda x: f"{x:.3f}")

        return render.DataGrid(df, width="100%", height="300px")

    @output
    @render.data_frame
    def table_qb_ratios():
        """Render Q/B ratios table."""
        report = diagnostic_report()

        if report is None or len(report.get("qb_ratios", [])) == 0:
            return pd.DataFrame()

        df = report["qb_ratios"].copy()

        # Format numeric columns
        df["Prey_Rate_Mean"] = df["Prey_Rate_Mean"].apply(lambda x: f"{x:.3f}")
        df["Predator_Rate"] = df["Predator_Rate"].apply(lambda x: f"{x:.3f}")
        df["Ratio"] = df["Ratio"].apply(lambda x: f"{x:.3f}")

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
                0.5,
                0.5,
                "No diagnostics run yet",
                ha="center",
                va="center",
                fontsize=14,
                color="gray",
            )
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")
            return fig

        # Parse excluded groups
        exclude_str = input.exclude_groups().strip()
        exclude_groups = (
            [g.strip() for g in exclude_str.split(",") if g.strip()]
            if exclude_str
            else None
        )

        # Generate plot based on selection
        plot_type = input.plot_type()

        try:
            # Lazy-import plotting helpers to avoid E402 and path issues
            try:
                from pypath.analysis.prebalance import (
                    plot_biomass_vs_trophic_level,
                    plot_vital_rate_vs_trophic_level,
                )
            except Exception:
                import sys

                root_dir = Path(__file__).parent.parent.parent
                if str(root_dir) not in sys.path:
                    sys.path.insert(0, str(root_dir))
                from src.pypath.analysis.prebalance import (
                    plot_biomass_vs_trophic_level,
                    plot_vital_rate_vs_trophic_level,
                )

            if plot_type == "biomass":
                fig = plot_biomass_vs_trophic_level(
                    data,
                    exclude_groups=exclude_groups,
                    figsize=(PLOTS.default_width, PLOTS.default_height),
                )
            elif plot_type in ["pb", "qb"]:
                rate_name = plot_type.upper()
                fig = plot_vital_rate_vs_trophic_level(
                    data,
                    rate_name=rate_name,
                    exclude_groups=exclude_groups,
                    figsize=(PLOTS.default_width, PLOTS.default_height),
                )
            else:
                raise ValueError(f"Unknown plot type: {plot_type}")

            return fig

        except Exception as e:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(PLOTS.default_width, PLOTS.default_height))
            ax.text(
                0.5,
                0.5,
                f"Error generating plot:\n{str(e)}",
                ha="center",
                va="center",
                fontsize=12,
                color="red",
            )
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")
            logger.error(f"Error generating diagnostic plot: {e}", exc_info=True)
            return fig
