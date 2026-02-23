"""About page module."""

import logging
import platform
import sys
from importlib.metadata import version as pkg_version

from shiny import Inputs, Outputs, Session, ui

logger = logging.getLogger(__name__)


def _get_version(package: str) -> str:
    """Get installed package version, returning 'N/A' if not found."""
    try:
        return pkg_version(package)
    except Exception as e:
        logger.debug("Failed to get version for %s: %s", package, e)
        return "N/A"


def about_ui():
    """About page UI."""
    return ui.page_fluid(
        ui.div(
            ui.h2("About PyPath", class_="mb-4"),
            ui.card(
                ui.card_body(
                    ui.h4("What is PyPath?"),
                    ui.p(
                        "PyPath is a Python implementation of the Ecopath with Ecosim (EwE) "
                        "ecosystem modeling approach. It provides tools for:"
                    ),
                    ui.tags.ul(
                        ui.tags.li(
                            ui.tags.strong("Ecopath"),
                            " - Static mass-balance modeling of food webs",
                        ),
                        ui.tags.li(
                            ui.tags.strong("Ecosim"),
                            " - Time-dynamic simulation of ecosystem changes",
                        ),
                        ui.tags.li(
                            ui.tags.strong("Ecospace"),
                            " - Spatial-temporal modeling with irregular polygon grids",
                        ),
                        ui.tags.li(
                            ui.tags.strong("IBM"),
                            " - Individual-based modeling with super-individual tracking",
                        ),
                    ),
                    ui.p(
                        "PyPath is based on the R package ",
                        ui.tags.a(
                            "Rpath",
                            href="https://github.com/NOAA-EDAB/Rpath/",
                            target="_blank",
                        ),
                        " developed by NOAA's Northeast Fisheries Science Center.",
                    ),
                ),
            ),
            ui.card(
                ui.card_header("Key Features"),
                ui.card_body(
                    ui.layout_columns(
                        ui.div(
                            ui.h5("Ecopath Mass Balance"),
                            ui.tags.ul(
                                ui.tags.li(
                                    "Define functional groups and food web structure"
                                ),
                                ui.tags.li(
                                    "Set biomass, production, and consumption rates"
                                ),
                                ui.tags.li(
                                    "Automatic calculation of missing parameters"
                                ),
                                ui.tags.li("Trophic level computation"),
                                ui.tags.li("Multi-stanza age structure"),
                            ),
                        ),
                        ui.div(
                            ui.h5("Ecosim Simulation"),
                            ui.tags.ul(
                                ui.tags.li("Foraging arena-based functional response"),
                                ui.tags.li(
                                    "Vulnerability parameters for top-down/bottom-up control"
                                ),
                                ui.tags.li("Fishing effort scenarios"),
                                ui.tags.li("Environmental forcing"),
                                ui.tags.li("Diet rewiring (prey switching)"),
                            ),
                        ),
                        ui.div(
                            ui.h5("Ecospace Spatial Modeling"),
                            ui.tags.ul(
                                ui.tags.li(
                                    "Irregular polygon grids (GeoJSON, Shapefile)"
                                ),
                                ui.tags.li("Habitat preference mapping"),
                                ui.tags.li("Dispersal and movement dynamics"),
                                ui.tags.li("Environmental drivers"),
                                ui.tags.li("Spatial fishing effort allocation"),
                            ),
                        ),
                        col_widths=[4, 4, 4],
                    ),
                    ui.layout_columns(
                        ui.div(
                            ui.h5("Individual-Based Model (IBM)"),
                            ui.tags.ul(
                                ui.tags.li("Super-individual tracking"),
                                ui.tags.li("Wisconsin bioenergetics model"),
                                ui.tags.li("Size-structured predation"),
                                ui.tags.li("Stochastic reproduction"),
                                ui.tags.li("Spatial movement and adaptive foraging"),
                            ),
                        ),
                        ui.div(
                            ui.h5("Network Analysis"),
                            ui.tags.ul(
                                ui.tags.li("Food web topology metrics"),
                                ui.tags.li("Trophic impact analysis (MTI)"),
                                ui.tags.li("Keystone species identification"),
                                ui.tags.li("Lindeman spine diagrams"),
                                ui.tags.li("Pre-balance diagnostics"),
                            ),
                        ),
                        ui.div(
                            ui.h5("Visualization & Export"),
                            ui.tags.ul(
                                ui.tags.li("Interactive time series plots"),
                                ui.tags.li("Food web diagrams"),
                                ui.tags.li("Scenario comparisons"),
                                ui.tags.li("Export to CSV/Excel"),
                                ui.tags.li("Spatial map visualizations"),
                            ),
                        ),
                        col_widths=[4, 4, 4],
                        class_="mt-3",
                    ),
                ),
            ),
            ui.card(
                ui.card_header("Scientific Background"),
                ui.card_body(
                    ui.h5("Ecopath with Ecosim"),
                    ui.p(
                        "Ecopath with Ecosim is a widely-used ecosystem modeling approach developed "
                        "by Villy Christensen and Carl Walters. The approach consists of:"
                    ),
                    ui.tags.ol(
                        ui.tags.li(
                            ui.tags.strong("Ecopath"),
                            " - Creates a static, mass-balanced snapshot of an ecosystem",
                        ),
                        ui.tags.li(
                            ui.tags.strong("Ecosim"),
                            " - Projects the ecosystem forward in time under various scenarios",
                        ),
                        ui.tags.li(
                            ui.tags.strong("Ecospace"),
                            " - Adds spatial dynamics with irregular polygon grids",
                        ),
                        ui.tags.li(
                            ui.tags.strong("IBM"),
                            " - Individual-based modeling for tracking super-individuals "
                            "with bioenergetics and life history",
                        ),
                    ),
                    ui.h5("Key Equations", class_="mt-4"),
                    ui.p("The Ecopath mass-balance equation:"),
                    ui.tags.div(
                        ui.tags.code(
                            "Production = Predation + Catch + Net Migration + Biomass Accumulation + Other Mortality"
                        ),
                        class_="bg-light p-3 rounded",
                    ),
                    ui.p("Or mathematically:", class_="mt-2"),
                    ui.tags.div(
                        ui.tags.code(
                            "Bᵢ × PBᵢ × EEᵢ = Σⱼ(Bⱼ × QBⱼ × DCⱼᵢ) + Yᵢ + Eᵢ + BAᵢ"
                        ),
                        class_="bg-light p-3 rounded",
                    ),
                    ui.h5("References", class_="mt-4"),
                    ui.tags.ul(
                        ui.tags.li(
                            "Christensen, V., & Walters, C. J. (2004). Ecopath with Ecosim: methods, "
                            "capabilities and limitations. ",
                            ui.tags.em("Ecological Modelling"),
                            ", 172(2-4), 109-139.",
                        ),
                        ui.tags.li(
                            "Lucey, S. M., et al. (2020). Conducting Management Strategy Evaluation "
                            "for the Northeast US Continental Shelf. ",
                            ui.tags.em("Frontiers in Marine Science"),
                            ", 7, 1029.",
                        ),
                    ),
                ),
            ),
            ui.card(
                ui.card_header("Development"),
                ui.card_body(
                    ui.layout_columns(
                        ui.div(
                            ui.h5("Technology Stack"),
                            ui.tags.ul(
                                ui.tags.li("Python 3.10+"),
                                ui.tags.li("NumPy for numerical computations"),
                                ui.tags.li("SciPy for optimization and integration"),
                                ui.tags.li("Pandas for data handling"),
                                ui.tags.li("Shiny for Python for the web interface"),
                                ui.tags.li("Matplotlib for visualization"),
                            ),
                        ),
                        ui.div(
                            ui.h5("Links"),
                            ui.tags.ul(
                                ui.tags.li(
                                    ui.tags.a(
                                        "GitHub Repository",
                                        href="https://github.com/razinkele/PyPath",
                                        target="_blank",
                                    )
                                ),
                                ui.tags.li(
                                    ui.tags.a(
                                        "Documentation",
                                        href="https://razinkele.github.io/PyPath/",
                                        target="_blank",
                                    )
                                ),
                                ui.tags.li(
                                    ui.tags.a(
                                        "Original Rpath Package",
                                        href="https://github.com/NOAA-EDAB/Rpath/",
                                        target="_blank",
                                    )
                                ),
                                ui.tags.li(
                                    ui.tags.a(
                                        "EwE Official Site",
                                        href="https://ecopath.org/",
                                        target="_blank",
                                    )
                                ),
                            ),
                        ),
                        ui.div(
                            ui.h5("License"),
                            ui.p("MIT License"),
                            ui.p(
                                "PyPath is free and open source software. "
                                "Contributions are welcome!"
                            ),
                        ),
                        col_widths=[4, 4, 4],
                    ),
                ),
            ),
            ui.card(
                ui.card_header("Version Information"),
                ui.card_body(
                    ui.tags.table(
                        ui.tags.tr(
                            ui.tags.td("Python:"),
                            ui.tags.td(platform.python_version()),
                        ),
                        ui.tags.tr(
                            ui.tags.td("PyPath-EwE:"),
                            ui.tags.td(_get_version("pypath-ewe")),
                        ),
                        ui.tags.tr(
                            ui.tags.td("PyPath-Shiny:"),
                            ui.tags.td(_get_version("pypath-shiny")),
                        ),
                        ui.tags.tr(
                            ui.tags.td("Shiny:"),
                            ui.tags.td(_get_version("shiny")),
                        ),
                        ui.tags.tr(
                            ui.tags.td("NumPy:"),
                            ui.tags.td(_get_version("numpy")),
                        ),
                        ui.tags.tr(
                            ui.tags.td("Pandas:"),
                            ui.tags.td(_get_version("pandas")),
                        ),
                        ui.tags.tr(
                            ui.tags.td("SciPy:"),
                            ui.tags.td(_get_version("scipy")),
                        ),
                        class_="table table-sm",
                    ),
                ),
            ),
            class_="container py-4",
        )
    )


def about_server(input: Inputs, output: Outputs, session: Session):
    """About page server logic (minimal)."""
    pass
