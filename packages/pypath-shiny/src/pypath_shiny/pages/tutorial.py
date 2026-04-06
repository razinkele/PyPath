"""Tutorial page module — guided walkthrough of the PyPath modeling workflow."""

import logging

from shiny import Inputs, Outputs, Session, reactive, ui

logger = logging.getLogger(__name__)


def _code_block(*lines: str, language: str = "python") -> ui.TagList:
    """Render a code block with syntax-aware styling."""
    code = "\n".join(lines)
    return ui.tags.div(
        ui.tags.pre(
            ui.tags.code(code, class_=f"language-{language}"),
            class_="bg-light p-3 rounded",
            style="font-size: 0.85em; overflow-x: auto;",
        ),
        class_="mb-3",
    )


def _step_card(
    number: int, title: str, icon: str, *body_elements, badge: str | None = None
) -> ui.Tag:
    """Render a numbered tutorial step as a card."""
    header_content = ui.TagList(
        ui.tags.span(
            str(number),
            class_="badge bg-primary rounded-circle me-2",
            style="width: 28px; height: 28px; line-height: 20px; display: inline-block; text-align: center;",
        ),
        ui.tags.i(class_=f"bi {icon} me-2"),
        title,
    )
    if badge:
        header_content = ui.TagList(
            header_content,
            ui.tags.span(badge, class_="badge bg-info ms-2", style="font-size: 0.7em;"),
        )
    return ui.card(
        ui.card_header(header_content),
        ui.card_body(*body_elements),
        class_="mb-4",
    )


def tutorial_ui():
    """Tutorial page UI."""
    return ui.page_fluid(
        ui.div(
            # Header
            ui.div(
                ui.h2(
                    ui.tags.i(class_="bi bi-mortarboard-fill text-primary me-2"),
                    "PyPath Tutorial",
                ),
                ui.p(
                    "A step-by-step guide to ecosystem modeling with PyPath. "
                    "Follow along using either the web interface or the Python API.",
                    class_="lead text-muted",
                ),
                class_="mb-4",
            ),
            # Table of Contents
            ui.card(
                ui.card_header(
                    ui.tags.i(class_="bi bi-list-ol me-2"),
                    "Contents",
                ),
                ui.card_body(
                    ui.layout_columns(
                        ui.tags.ol(
                            ui.tags.li(
                                ui.tags.a("Create a Model from Scratch", href="#step1")
                            ),
                            ui.tags.li(
                                ui.tags.a("Set Ecopath Parameters", href="#step2")
                            ),
                            ui.tags.li(
                                ui.tags.a("Define the Diet Matrix", href="#step3")
                            ),
                            ui.tags.li(ui.tags.a("Balance the Model", href="#step4")),
                        ),
                        ui.tags.ol(
                            ui.tags.li(
                                ui.tags.a("Run an Ecosim Simulation", href="#step5"),
                                value="5",
                            ),
                            ui.tags.li(
                                ui.tags.a("Apply Fishing Scenarios", href="#step6"),
                                value="6",
                            ),
                            ui.tags.li(
                                ui.tags.a("Adjust Vulnerability", href="#step7"),
                                value="7",
                            ),
                            ui.tags.li(
                                ui.tags.a("Load from EwE Database", href="#step8"),
                                value="8",
                            ),
                        ),
                        col_widths=[6, 6],
                    ),
                ),
                class_="mb-4",
            ),
            # ── Workflow Overview ───────────────────────────────────────
            ui.card(
                ui.card_header(
                    ui.tags.i(class_="bi bi-diagram-3 me-2"),
                    "Modeling Workflow Overview",
                ),
                ui.card_body(
                    ui.p(
                        "The Ecopath with Ecosim framework follows a natural progression:"
                    ),
                    ui.layout_columns(
                        ui.div(
                            ui.h6(
                                ui.tags.span("1", class_="badge bg-primary me-1"),
                                "Ecopath",
                            ),
                            ui.p(
                                "Static mass-balance snapshot. Define groups, biomass, "
                                "production rates, consumption rates, and diet.",
                                class_="small text-muted",
                            ),
                        ),
                        ui.div(
                            ui.h6(
                                ui.tags.span("2", class_="badge bg-primary me-1"),
                                "Ecosim",
                            ),
                            ui.p(
                                "Dynamic simulation. Project the balanced model forward "
                                "in time with fishing, forcing, and vulnerability settings.",
                                class_="small text-muted",
                            ),
                        ),
                        ui.div(
                            ui.h6(
                                ui.tags.span("3", class_="badge bg-success me-1"),
                                "Ecospace",
                            ),
                            ui.p(
                                "Spatial dynamics. Add habitat maps, dispersal, and "
                                "spatial fishing effort on irregular polygon grids.",
                                class_="small text-muted",
                            ),
                        ),
                        ui.div(
                            ui.h6(
                                ui.tags.span("4", class_="badge bg-info me-1"),
                                "IBM",
                            ),
                            ui.p(
                                "Individual-based modeling. Track super-individuals "
                                "with bioenergetics, predation, and reproduction.",
                                class_="small text-muted",
                            ),
                        ),
                        col_widths=[3, 3, 3, 3],
                    ),
                    ui.tags.div(
                        ui.tags.i(class_="bi bi-info-circle me-1"),
                        "This tutorial covers steps 1-2. See the Ecospace and IBM pages "
                        "for spatial and individual-based extensions.",
                        class_="alert alert-info mt-2 small",
                    ),
                ),
                class_="mb-4",
            ),
            # ── Step 1: Create Model ───────────────────────────────────
            ui.tags.div(id="step1"),
            _step_card(
                1,
                "Create a Model from Scratch",
                "bi-plus-circle",
                ui.p(
                    "Every model starts by defining functional groups. Each group has a ",
                    ui.tags.strong("name"),
                    " and a ",
                    ui.tags.strong("type code"),
                    ":",
                ),
                ui.tags.table(
                    ui.tags.thead(
                        ui.tags.tr(
                            ui.tags.th("Type"),
                            ui.tags.th("Code"),
                            ui.tags.th("Description"),
                            ui.tags.th("Example"),
                        ),
                    ),
                    ui.tags.tbody(
                        ui.tags.tr(
                            ui.tags.td("Consumer"),
                            ui.tags.td("0"),
                            ui.tags.td("Heterotroph (eats other groups)"),
                            ui.tags.td("Fish, Zooplankton, Benthos"),
                        ),
                        ui.tags.tr(
                            ui.tags.td("Producer"),
                            ui.tags.td("1"),
                            ui.tags.td("Autotroph (primary production)"),
                            ui.tags.td("Phytoplankton, Macroalgae"),
                        ),
                        ui.tags.tr(
                            ui.tags.td("Detritus"),
                            ui.tags.td("2"),
                            ui.tags.td("Non-living organic matter"),
                            ui.tags.td("Detritus, Discards"),
                        ),
                        ui.tags.tr(
                            ui.tags.td("Fleet"),
                            ui.tags.td("3"),
                            ui.tags.td("Fishing gear type"),
                            ui.tags.td("Trawlers, Gillnets"),
                        ),
                    ),
                    class_="table table-sm table-striped",
                ),
                ui.h6("Python API", class_="mt-3"),
                _code_block(
                    "from pypath import create_rpath_params",
                    "",
                    "params = create_rpath_params(",
                    '    groups=["Phytoplankton", "Zooplankton", "Small Fish",',
                    '           "Detritus", "Trawlers"],',
                    "    types=[1, 0, 0, 2, 3],",
                    ")",
                ),
                ui.h6("In the Web App", class_="mt-3"),
                ui.p(
                    "Navigate to ",
                    ui.tags.strong("Ecopath Model"),
                    " and use the table editor to add groups. "
                    "Set the Type column for each group (0, 1, 2, or 3).",
                ),
            ),
            # ── Step 2: Set Parameters ─────────────────────────────────
            ui.tags.div(id="step2"),
            _step_card(
                2,
                "Set Ecopath Parameters",
                "bi-sliders",
                ui.p("For each group, provide the key ecological rates:"),
                ui.tags.table(
                    ui.tags.thead(
                        ui.tags.tr(
                            ui.tags.th("Parameter"),
                            ui.tags.th("Symbol"),
                            ui.tags.th("Unit"),
                            ui.tags.th("Notes"),
                        ),
                    ),
                    ui.tags.tbody(
                        ui.tags.tr(
                            ui.tags.td("Biomass"),
                            ui.tags.td(ui.tags.em("B")),
                            ui.tags.td("t/km\u00b2"),
                            ui.tags.td("Standing stock biomass"),
                        ),
                        ui.tags.tr(
                            ui.tags.td("Production/Biomass"),
                            ui.tags.td(ui.tags.em("P/B")),
                            ui.tags.td("yr\u207b\u00b9"),
                            ui.tags.td("Total mortality rate (= Z for fish)"),
                        ),
                        ui.tags.tr(
                            ui.tags.td("Consumption/Biomass"),
                            ui.tags.td(ui.tags.em("Q/B")),
                            ui.tags.td("yr\u207b\u00b9"),
                            ui.tags.td("Consumers only (not producers)"),
                        ),
                        ui.tags.tr(
                            ui.tags.td("Ecotrophic Efficiency"),
                            ui.tags.td(ui.tags.em("EE")),
                            ui.tags.td("0\u20131"),
                            ui.tags.td("Fraction of production consumed in system"),
                        ),
                    ),
                    class_="table table-sm table-striped",
                ),
                ui.tags.div(
                    ui.tags.i(class_="bi bi-lightbulb me-1"),
                    "You can leave one parameter unknown per group. "
                    "The mass-balance solver will calculate it from the others.",
                    class_="alert alert-warning small",
                ),
                ui.h6("Python API", class_="mt-3"),
                _code_block(
                    "# Biomass (t/km\u00b2)",
                    'params.model.loc[0, "Biomass"] = 10.0   # Phytoplankton',
                    'params.model.loc[1, "Biomass"] = 5.0    # Zooplankton',
                    'params.model.loc[2, "Biomass"] = 2.0    # Small Fish',
                    "",
                    "# Production/Biomass (yr\u207b\u00b9)",
                    'params.model.loc[0, "PB"] = 200.0  # Phytoplankton: high turnover',
                    'params.model.loc[1, "PB"] = 50.0   # Zooplankton',
                    'params.model.loc[2, "PB"] = 1.0    # Small Fish',
                    "",
                    "# Consumption/Biomass (yr\u207b\u00b9, consumers only)",
                    'params.model.loc[1, "QB"] = 150.0  # Zooplankton',
                    'params.model.loc[2, "QB"] = 5.0    # Small Fish',
                    "",
                    "# Ecotrophic Efficiency",
                    'params.model.loc[0, "EE"] = 0.8',
                    'params.model.loc[1, "EE"] = 0.9',
                    'params.model.loc[2, "EE"] = 0.5',
                ),
                ui.h6("In the Web App", class_="mt-3"),
                ui.p(
                    "Edit cells directly in the Ecopath Model table. "
                    "Leave one cell empty per group for the solver to fill in.",
                ),
            ),
            # ── Step 3: Diet Matrix ────────────────────────────────────
            ui.tags.div(id="step3"),
            _step_card(
                3,
                "Define the Diet Matrix",
                "bi-grid-3x3",
                ui.p(
                    "The diet matrix defines who eats whom. Each ",
                    ui.tags.strong("column"),
                    " is a predator, each ",
                    ui.tags.strong("row"),
                    " is a prey item. Columns must sum to 1.0 (100% of the diet).",
                ),
                ui.tags.div(
                    ui.tags.table(
                        ui.tags.thead(
                            ui.tags.tr(
                                ui.tags.th("Prey \\ Predator", style="min-width:120px"),
                                ui.tags.th("Zooplankton"),
                                ui.tags.th("Small Fish"),
                            ),
                        ),
                        ui.tags.tbody(
                            ui.tags.tr(
                                ui.tags.td("Phytoplankton"),
                                ui.tags.td("0.90"),
                                ui.tags.td("-"),
                            ),
                            ui.tags.tr(
                                ui.tags.td("Zooplankton"),
                                ui.tags.td("-"),
                                ui.tags.td("0.80"),
                            ),
                            ui.tags.tr(
                                ui.tags.td("Detritus"),
                                ui.tags.td("0.10"),
                                ui.tags.td("0.20"),
                            ),
                            ui.tags.tr(
                                ui.tags.td(ui.tags.strong("Total")),
                                ui.tags.td(ui.tags.strong("1.00")),
                                ui.tags.td(ui.tags.strong("1.00")),
                                class_="table-active",
                            ),
                        ),
                        class_="table table-sm table-bordered",
                    ),
                    class_="mb-3",
                    style="overflow-x: auto;",
                ),
                ui.h6("Python API", class_="mt-3"),
                _code_block(
                    "# Each list = diet column for that predator",
                    "# Order matches group list: [Phyto, Zoo, SmallFish, Detritus, Trawlers]",
                    'params.diet["Zooplankton"] = [0.90, 0.0, 0.0, 0.10, 0.0]',
                    'params.diet["Small Fish"] = [0.0, 0.80, 0.0, 0.20, 0.0]',
                ),
                ui.h6("In the Web App", class_="mt-3"),
                ui.p(
                    "Switch to the ",
                    ui.tags.strong("Diet"),
                    " tab on the Ecopath Model page. "
                    "Edit diet fractions in the table. A warning appears if any "
                    "column does not sum to 1.0.",
                ),
            ),
            # ── Step 4: Balance ────────────────────────────────────────
            ui.tags.div(id="step4"),
            _step_card(
                4,
                "Balance the Model (Ecopath)",
                "bi-check-circle",
                ui.p(
                    "The Ecopath solver checks mass balance and fills in missing "
                    "parameters. The fundamental equation is:"
                ),
                ui.tags.div(
                    ui.tags.code(
                        "B\u1d62 \u00d7 PB\u1d62 \u00d7 EE\u1d62 = "
                        "\u03a3\u2c7c(B\u2c7c \u00d7 QB\u2c7c \u00d7 DC\u2c7c\u1d62) "
                        "+ Y\u1d62 + E\u1d62 + BA\u1d62"
                    ),
                    class_="bg-light p-3 rounded mb-3",
                    style="font-size: 1.05em;",
                ),
                ui.p("Where:"),
                ui.tags.ul(
                    ui.tags.li(
                        ui.tags.em("B\u1d62 \u00d7 PB\u1d62 \u00d7 EE\u1d62"),
                        " = production of group ",
                        ui.tags.em("i"),
                        " consumed in the system",
                    ),
                    ui.tags.li(
                        "\u03a3\u2c7c(B\u2c7c \u00d7 QB\u2c7c \u00d7 DC\u2c7c\u1d62) "
                        "= total predation on group ",
                        ui.tags.em("i"),
                    ),
                    ui.tags.li(
                        ui.tags.em("Y\u1d62"),
                        " = fishery catch, ",
                        ui.tags.em("E\u1d62"),
                        " = net emigration, ",
                        ui.tags.em("BA\u1d62"),
                        " = biomass accumulation",
                    ),
                ),
                ui.tags.div(
                    ui.tags.i(class_="bi bi-exclamation-triangle me-1"),
                    "If any EE > 1.0, the model is unbalanced. Use the ",
                    ui.tags.strong("Pre-Balance"),
                    " page to diagnose issues before proceeding.",
                    class_="alert alert-danger small",
                ),
                ui.h6("Python API", class_="mt-3"),
                _code_block(
                    "from pypath import rpath",
                    "",
                    "model = rpath(params)",
                    "print(model)  # Shows balance status and group summary",
                ),
                ui.h6("In the Web App", class_="mt-3"),
                ui.p(
                    'Click the "Balance Model" button on the Ecopath Model page. '
                    "The app runs pre-balance diagnostics and solves the linear system. "
                    "Missing parameters appear automatically in the table.",
                ),
            ),
            # ── Step 5: Ecosim ─────────────────────────────────────────
            ui.tags.div(id="step5"),
            _step_card(
                5,
                "Run an Ecosim Simulation",
                "bi-graph-up",
                ui.p(
                    "Ecosim projects the balanced Ecopath model forward in time. "
                    "It uses the ",
                    ui.tags.strong("foraging arena"),
                    " functional response, where prey availability is split into "
                    "a vulnerable and an invulnerable pool.",
                ),
                ui.h6("Key Concepts"),
                ui.tags.ul(
                    ui.tags.li(
                        ui.tags.strong("Vulnerability (VV)"),
                        " \u2014 Controls functional response shape: ",
                        "1.0 = donor-controlled (bottom-up), ",
                        "2.0 = mixed (default), ",
                        ">10 = recipient-controlled (top-down)",
                    ),
                    ui.tags.li(
                        ui.tags.strong("Integration methods"),
                        " \u2014 RK4 (Runge-Kutta 4th order) for accuracy, "
                        "or AB (Adams-Bashforth 2-step) to match Rpath/EwE output",
                    ),
                    ui.tags.li(
                        ui.tags.strong("NoIntegrate groups"),
                        " \u2014 Detritus and fast-turnover groups use "
                        "algebraic equilibrium instead of numerical integration",
                    ),
                ),
                ui.h6("Python API", class_="mt-3"),
                _code_block(
                    "from pypath import rsim_scenario, rsim_run",
                    "",
                    "# Create a 50-year scenario from the balanced model",
                    "scenario = rsim_scenario(model, params, years=range(1, 51))",
                    "",
                    "# Run with Adams-Bashforth (matches Rpath/EwE)",
                    'output = rsim_run(scenario, method="AB")',
                    "",
                    "# Biomass trajectories: shape (n_months, n_groups+1)",
                    "print(output.out_Biomass.shape)",
                ),
                ui.h6("In the Web App", class_="mt-3"),
                ui.p(
                    "Navigate to the ",
                    ui.tags.strong("Ecosim"),
                    " page. Set the simulation years, choose the integration method, "
                    'and click "Run Ecosim". Biomass trajectories appear automatically.',
                ),
            ),
            # ── Step 6: Fishing ────────────────────────────────────────
            ui.tags.div(id="step6"),
            _step_card(
                6,
                "Apply Fishing Scenarios",
                "bi-tsunami",
                ui.p(
                    "Modify fishing effort over time to explore management scenarios. "
                    "Effort is a multiplier on the base catch rate (1.0 = status quo)."
                ),
                ui.h6("Python API", class_="mt-3"),
                _code_block(
                    "from pypath import adjust_fishing",
                    "",
                    "# Double fishing effort on Small Fish from year 10 to 30",
                    'adjust_fishing(scenario, group="Small Fish",',
                    "               value=2.0, years=range(10, 31))",
                    "",
                    "output = rsim_run(scenario)",
                ),
                ui.h6("In the Web App", class_="mt-3"),
                ui.p(
                    "On the Ecosim page, use the Fishing Effort panel to set "
                    "effort multipliers for each fleet across time periods. "
                    "Re-run the simulation to see the effect.",
                ),
            ),
            # ── Step 7: Vulnerability ──────────────────────────────────
            ui.tags.div(id="step7"),
            _step_card(
                7,
                "Adjust Vulnerability Parameters",
                "bi-shield-exclamation",
                ui.p(
                    "Vulnerability (VV) is the most important Ecosim calibration parameter. "
                    "It controls how strongly predator biomass affects prey consumption:"
                ),
                ui.tags.table(
                    ui.tags.thead(
                        ui.tags.tr(
                            ui.tags.th("VV Value"),
                            ui.tags.th("Control Type"),
                            ui.tags.th("Behavior"),
                        ),
                    ),
                    ui.tags.tbody(
                        ui.tags.tr(
                            ui.tags.td("1.0"),
                            ui.tags.td("Donor-controlled"),
                            ui.tags.td(
                                "Prey production limits consumption (bottom-up)"
                            ),
                        ),
                        ui.tags.tr(
                            ui.tags.td("2.0"),
                            ui.tags.td("Mixed (default)"),
                            ui.tags.td("Both prey and predator influence flow rates"),
                        ),
                        ui.tags.tr(
                            ui.tags.td(">10"),
                            ui.tags.td("Recipient-controlled"),
                            ui.tags.td(
                                "Predator biomass drives consumption (top-down)"
                            ),
                        ),
                    ),
                    class_="table table-sm table-striped",
                ),
                ui.h6("Python API", class_="mt-3"),
                _code_block(
                    "from pypath import set_vulnerability",
                    "",
                    "# Make Zooplankton highly vulnerable to Small Fish predation",
                    'set_vulnerability(scenario, prey="Zooplankton",',
                    '                  pred="Small Fish", value=5.0)',
                ),
                ui.h6("Calibration Tips", class_="mt-3"),
                ui.tags.ul(
                    ui.tags.li("Start with VV=2.0 (default) for all links"),
                    ui.tags.li(
                        "Lower VV (toward 1.0) for stable benthic prey or detritus feeders"
                    ),
                    ui.tags.li(
                        "Raise VV (toward 10+) for pelagic prey subject to strong top-down control"
                    ),
                    ui.tags.li(
                        "Use the Optimization page to fit VV to time series data"
                    ),
                ),
            ),
            # ── Step 8: Load from EwE ──────────────────────────────────
            ui.tags.div(id="step8"),
            _step_card(
                8,
                "Load from an EwE Database",
                "bi-database",
                ui.p(
                    "PyPath can read existing Ecopath with Ecosim models directly from "
                    "EwE's native Access database format (",
                    ui.tags.code(".eweaccdb"),
                    " or ",
                    ui.tags.code(".ewemdb"),
                    " files). This loads all parameters, diet matrices, stanzas, "
                    "fleet catches, vulnerability overrides, and forcing functions.",
                ),
                ui.h6("Option A: Ecopath Parameters Only", class_="mt-3"),
                _code_block(
                    "from pypath import read_ewemdb, rpath",
                    "",
                    "# Load Ecopath parameters from EwE database",
                    'params = read_ewemdb("path/to/model.eweaccdb")',
                    "model = rpath(params)",
                ),
                ui.h6("Option B: Complete Ecosim Scenario", class_="mt-3"),
                _code_block(
                    "from pypath.io.ewemdb import ecosim_scenario_from_ewemdb",
                    "from pypath import rsim_run",
                    "",
                    "# Load a ready-to-run Ecosim scenario (with all EwE settings)",
                    'scenario = ecosim_scenario_from_ewemdb("model.eweaccdb", scenario=16)',
                    "",
                    "# This loads: vulnerability overrides, fishing effort shapes,",
                    "#   environmental forcing, foraging time adjustments, and",
                    "#   forced biomass from the EwE scenario ID.",
                    "",
                    'output = rsim_run(scenario, method="AB")',
                ),
                ui.h6("Option C: From EcoBase Online Repository", class_="mt-3"),
                _code_block(
                    "from pypath import search_ecobase_models, get_ecobase_model, ecobase_to_rpath",
                    "",
                    "# Search 350+ published Ecopath models",
                    'results = search_ecobase_models("Baltic Sea")',
                    "print(results)",
                    "",
                    "# Download and convert",
                    "model_data = get_ecobase_model(model_id=123)",
                    "params = ecobase_to_rpath(model_data)",
                ),
                ui.h6("In the Web App", class_="mt-3"),
                ui.p(
                    "Navigate to ",
                    ui.tags.strong("Data Import"),
                    " and select your data source: EwE database file, "
                    "EcoBase online repository, or CSV files. The importer "
                    "handles format conversion automatically.",
                ),
                badge="Advanced",
            ),
            # ── Plotting Results ───────────────────────────────────────
            ui.card(
                ui.card_header(
                    ui.tags.i(class_="bi bi-bar-chart me-2"),
                    "Plotting Results",
                ),
                ui.card_body(
                    ui.p(
                        "After running a simulation, plot biomass trajectories "
                        "for all living groups:"
                    ),
                    _code_block(
                        "import matplotlib.pyplot as plt",
                        "",
                        "bio = output.out_Biomass",
                        "months = range(bio.shape[0])",
                        "",
                        "for i, name in enumerate(model.Group):",
                        "    if model.type[i] in (0, 1):  # consumers + producers",
                        "        b = bio[:, i + 1]  # +1 for Outside offset",
                        "        if b[0] > 0:",
                        "            plt.plot(months, b / b[0], label=name)",
                        "",
                        'plt.xlabel("Month")',
                        'plt.ylabel("Relative Biomass (B/B\u2080)")',
                        "plt.legend(loc='best', fontsize=8)",
                        'plt.title("Ecosim Biomass Trajectories")',
                        "plt.tight_layout()",
                        "plt.show()",
                    ),
                    ui.p(
                        "In the web app, biomass plots are generated automatically "
                        "on the ",
                        ui.tags.strong("Results"),
                        " page after an Ecosim run.",
                    ),
                ),
                class_="mb-4",
            ),
            # ── Next Steps ─────────────────────────────────────────────
            ui.card(
                ui.card_header(
                    ui.tags.i(class_="bi bi-arrow-right-circle me-2"),
                    "Next Steps",
                ),
                ui.card_body(
                    ui.layout_columns(
                        ui.div(
                            ui.h6(
                                ui.tags.i(
                                    class_="bi bi-clipboard-check text-primary me-2"
                                ),
                                "Pre-Balance Diagnostics",
                            ),
                            ui.p(
                                "Run diagnostics before balancing to catch data entry "
                                "errors and thermodynamic violations early.",
                                class_="small",
                            ),
                            ui.input_action_button(
                                "btn_tut_prebalance",
                                "Pre-Balance",
                                class_="btn-outline-primary btn-sm",
                            ),
                        ),
                        ui.div(
                            ui.h6(
                                ui.tags.i(
                                    class_="bi bi-globe-americas text-success me-2"
                                ),
                                "Ecospace Spatial Modeling",
                            ),
                            ui.p(
                                "Add spatial dynamics with irregular polygon grids, "
                                "habitat maps, and dispersal.",
                                class_="small",
                            ),
                            ui.input_action_button(
                                "btn_tut_ecospace",
                                "Ecospace",
                                class_="btn-outline-success btn-sm",
                            ),
                        ),
                        ui.div(
                            ui.h6(
                                ui.tags.i(class_="bi bi-cpu text-info me-2"),
                                "Individual-Based Modeling",
                            ),
                            ui.p(
                                "Track super-individuals with bioenergetics, "
                                "predation, reproduction, and movement.",
                                class_="small",
                            ),
                            ui.input_action_button(
                                "btn_tut_ibm",
                                "IBM",
                                class_="btn-outline-info btn-sm",
                            ),
                        ),
                        ui.div(
                            ui.h6(
                                ui.tags.i(class_="bi bi-book text-secondary me-2"),
                                "API Documentation",
                            ),
                            ui.p(
                                "Full API reference for all PyPath functions, "
                                "classes, and modules.",
                                class_="small",
                            ),
                            ui.tags.a(
                                "View Docs",
                                href="https://razinkele.github.io/PyPath/",
                                target="_blank",
                                class_="btn btn-outline-secondary btn-sm",
                            ),
                        ),
                        col_widths=[3, 3, 3, 3],
                    ),
                ),
                class_="mb-4",
            ),
            class_="container py-4",
        )
    )


def tutorial_server(input: Inputs, output: Outputs, session: Session):
    """Tutorial page server logic."""

    @reactive.effect
    @reactive.event(input.btn_tut_prebalance)
    def _goto_prebalance():
        ui.update_navs("main_nav", selected="Pre-Balance")

    @reactive.effect
    @reactive.event(input.btn_tut_ecospace)
    def _goto_ecospace():
        ui.update_navs("main_nav", selected="Ecospace")

    @reactive.effect
    @reactive.event(input.btn_tut_ibm)
    def _goto_ibm():
        ui.update_navs("main_nav", selected="Individual-Based Model")
