"""About page module."""

import logging
import platform
import re
from dataclasses import dataclass
from importlib.metadata import version as pkg_version

import pypath
from shiny import Inputs, Outputs, Session, ui

import pypath_shiny
from pypath_shiny.config import CHANGELOG_URL, DOCS_URL, REPO_URL

logger = logging.getLogger(__name__)

RPATH_URL = "https://github.com/NOAA-EDAB/Rpath/"
EWE_URL = "https://ecopath.org/"
CONDA_URL = "https://anaconda.org/razinka/pypath-shiny"
INSTALL_COMMAND = "conda install pypath-shiny -c razinka -c conda-forge"

# Installed packages whose version the page reports: (label, distribution).
_THIRD_PARTY = (
    ("Shiny", "shiny"),
    ("NumPy", "numpy"),
    ("pandas", "pandas"),
    ("SciPy", "scipy"),
    ("Plotly", "plotly"),
    ("Numba (optional)", "numba"),
)


@dataclass(frozen=True)
class Reference:
    """A published work the page cites; ``doi`` is what makes it checkable."""

    authors: str
    year: int
    title: str
    journal: str
    locator: str  # volume, issue and pages
    doi: str


# Each entry was looked up by its DOI (title, year, journal, volume and
# pages) before it was put here. Keep it that way: a reference that cannot
# be looked up does not belong on the page.
REFERENCES = (
    Reference(
        "Christensen, V., & Walters, C. J.",
        2004,
        "Ecopath with Ecosim: methods, capabilities and limitations",
        "Ecological Modelling",
        "172(2-4), 109-139",
        "10.1016/j.ecolmodel.2003.09.003",
    ),
    Reference(
        "Walters, C., Christensen, V., & Pauly, D.",
        1997,
        "Structuring dynamic models of exploited ecosystems from trophic "
        "mass-balance assessments",
        "Reviews in Fish Biology and Fisheries",
        "7(2), 139-172",
        "10.1023/a:1018479526149",
    ),
    Reference(
        "Walters, C., Pauly, D., & Christensen, V.",
        1999,
        "Ecospace: prediction of mesoscale spatial patterns in trophic "
        "relationships of exploited ecosystems, with emphasis on the impacts "
        "of marine protected areas",
        "Ecosystems",
        "2(6), 539-554",
        "10.1007/s100219900101",
    ),
    Reference(
        "Lucey, S. M., Gaichas, S. K., & Aydin, K. Y.",
        2020,
        "Conducting reproducible ecosystem modeling using the open source "
        "mass balance model Rpath",
        "Ecological Modelling",
        "427, 109057",
        "10.1016/j.ecolmodel.2020.109057",
    ),
    Reference(
        "Akoglu, E., Libralato, S., Salihoglu, B., Oguz, T., & Solidoro, C.",
        2015,
        "EwE-F 1.0: an implementation of Ecopath with Ecosim in Fortran "
        "95/2003 for coupling and integration with other models",
        "Geoscientific Model Development",
        "8(8), 2687-2699",
        "10.5194/gmd-8-2687-2015",
    ),
)

_DOI_URL = re.compile(r"https://doi\.org/[^\s;,)]+")


def _get_version(package: str) -> str:
    """Get installed package version, returning 'N/A' if not found."""
    try:
        return pkg_version(package)
    except Exception as e:
        logger.debug("Failed to get version for %s: %s", package, e)
        return "N/A"


def _version_rows() -> list[tuple[str, str]]:
    """The rows of the Version Information table.

    The two PyPath versions are those of the code that is running, which is
    also what the top bar shows. Installed-package metadata is not used for
    them: an app run from a source tree has none, or an old copy's.
    """
    rows = [
        ("pypath-ewe (the library)", pypath.__version__),
        ("pypath-shiny (this app)", pypath_shiny.__version__),
        ("Python", platform.python_version()),
    ]
    for label, distribution in _THIRD_PARTY:
        found = _get_version(distribution)
        rows.append((label, "not installed" if found == "N/A" else found))
    return rows


def _external_link(label: str, href: str) -> ui.Tag:
    return ui.tags.a(label, href=href, target="_blank", rel="noopener noreferrer")


def _with_doi_links(text: str) -> list:
    """``text`` with every https://doi.org/... address turned into a link."""
    parts: list = []
    last = 0
    for match in _DOI_URL.finditer(text):
        parts.append(text[last : match.start()])
        parts.append(_external_link(match.group(0), match.group(0)))
        last = match.end()
    parts.append(text[last:])
    return [part for part in parts if part != ""]


def _reference_item(reference: Reference) -> ui.Tag:
    return ui.tags.li(
        f"{reference.authors} ({reference.year}). {reference.title}. ",
        ui.tags.em(reference.journal),
        f", {reference.locator}. ",
        _external_link(
            f"https://doi.org/{reference.doi}", f"https://doi.org/{reference.doi}"
        ),
    )


def _feature_list(title: str, *items: str) -> ui.Tag:
    return ui.div(ui.h5(title), ui.tags.ul(*(ui.tags.li(item) for item in items)))


def _example_models() -> list:
    """The example models the library ships, described in its own words."""
    from pypath.examples import list_example_models

    blocks = []
    for model in list_example_models():
        blocks.append(
            ui.div(
                ui.h5(model.title),
                ui.p(model.description, class_="mb-1"),
                ui.p(
                    ui.tags.strong("Source: "),
                    *_with_doi_links(model.source),
                    class_="text-muted small",
                ),
            )
        )
    return blocks


def about_ui():
    """About page UI."""
    return ui.div(
        ui.div(
            ui.h2("About PyPath", class_="mb-3"),
            ui.card(
                ui.card_body(
                    ui.h4("What is PyPath?"),
                    ui.p(
                        "PyPath is a Python implementation of the Ecopath with "
                        "Ecosim (EwE) approach to food web modelling. It comes as "
                        "two packages: ",
                        ui.tags.strong("pypath-ewe"),
                        ", the library that does the calculations, and ",
                        ui.tags.strong("pypath-shiny"),
                        ", this web application. Together they cover:",
                    ),
                    ui.tags.ul(
                        ui.tags.li(
                            ui.tags.strong("Ecopath"),
                            " - a static, mass-balanced model of a food web",
                        ),
                        ui.tags.li(
                            ui.tags.strong("Ecosim"),
                            " - the food web simulated through time",
                        ),
                        ui.tags.li(
                            ui.tags.strong("Ecospace"),
                            " - Ecosim in space, on regular grids, polygons or "
                            "the mesh of a hydrodynamic model",
                        ),
                        ui.tags.li(
                            ui.tags.strong("Individual-based model"),
                            " - a group followed as super-individuals inside "
                            "an Ecosim run",
                        ),
                    ),
                    ui.p(
                        "PyPath began as a port of the R package ",
                        _external_link("Rpath", RPATH_URL),
                        ", developed at NOAA. It also reads EwE 6 databases "
                        "and EwE-F model files. Ecosim can follow the equation "
                        "rules of EwE 6 or of Rpath: a model read from an EwE "
                        "database starts with EwE 6's, a model built here or "
                        "from CSV files with Rpath's.",
                        class_="mb-0",
                    ),
                ),
            ),
            ui.card(
                ui.card_header("Key Features"),
                ui.card_body(
                    ui.layout_columns(
                        _feature_list(
                            "Ecopath Mass Balance",
                            "Functional groups, diet matrix and catches by fleet",
                            "The missing parameter of each group solved",
                            "Trophic levels",
                            "Multi-stanza age structure",
                            "Pre-balance diagnostics",
                        ),
                        _feature_list(
                            "Ecosim Simulation",
                            "Foraging arena functional response",
                            "Runge-Kutta 4, Adams-Bashforth and continuous "
                            "solvers (LSODA, BDF, DOP853, Radau)",
                            "EwE 6 or Rpath equation rules",
                            "An EwE database's own scenarios and time series",
                            "Prey switching (EwE's switching power)",
                        ),
                        _feature_list(
                            "Ecospace Spatial Modeling",
                            "Regular grids, transects, uploaded polygons, "
                            "hexagons or a SHYFEM mesh",
                            "Habitat preferences, dispersal and spatial fishing",
                            "Fields and water transport of a hydrodynamic box model",
                            "Stored Ecospace scenarios of EwE databases",
                            "A wizard for a new set-up",
                        ),
                        _feature_list(
                            "Individual-Based Model",
                            "Super-individual tracking",
                            "Wisconsin bioenergetics model",
                            "Size-structured predation",
                            "Stochastic reproduction",
                            "Comparison with the standard Ecosim run",
                        ),
                        _feature_list(
                            "Analysis and Results",
                            "System and flow indices",
                            "Mixed trophic impacts (MTI) and keystoneness",
                            "Food web diagram and interactive network",
                            "Scenario comparison",
                            "CSV downloads",
                        ),
                        _feature_list(
                            "In the Library, Without a Page Here",
                            "Monte Carlo runs and ensembles",
                            "Sensitivity analysis (Morris, Sobol)",
                            "Ecotracer, mediation and fleet dynamics",
                            "Readers for EwE-F model files and results",
                            "Runs on several processor cores",
                        ),
                        col_widths={"sm": 12, "md": 6, "xl": 4},
                    ),
                ),
            ),
            ui.card(
                ui.card_header("Example Models"),
                ui.card_body(
                    ui.p(
                        "These food webs come with the library. Open one from the "
                        "Home page or the Ecosim page.",
                    ),
                    ui.layout_columns(
                        *_example_models(),
                        col_widths={"sm": 12, "xl": 4},
                    ),
                ),
            ),
            ui.card(
                ui.card_header("Scientific Background"),
                ui.card_body(
                    ui.h5("Ecopath with Ecosim"),
                    ui.p(
                        "Ecopath with Ecosim is a widely used approach to "
                        "ecosystem modelling (Christensen & Walters, 2004). It "
                        "has three parts:"
                    ),
                    ui.tags.ol(
                        ui.tags.li(
                            ui.tags.strong("Ecopath"),
                            " - a static, mass-balanced snapshot of an ecosystem",
                        ),
                        ui.tags.li(
                            ui.tags.strong("Ecosim"),
                            " - the ecosystem projected through time under "
                            "fishing and environmental scenarios (Walters et "
                            "al., 1997)",
                        ),
                        ui.tags.li(
                            ui.tags.strong("Ecospace"),
                            " - Ecosim repeated over the cells of a map, with "
                            "movement between them (Walters et al., 1999)",
                        ),
                    ),
                    ui.p(
                        "The individual-based model is PyPath's own addition "
                        "and not part of Ecopath with Ecosim. So are the "
                        "polygon and mesh grids of its Ecospace: EwE's Ecospace "
                        "works on a regular grid.",
                    ),
                    ui.h5("Key Equations", class_="mt-4"),
                    ui.p("The Ecopath mass-balance equation:"),
                    ui.tags.div(
                        ui.tags.code(
                            "Production = Predation + Catch + Net Migration + Biomass Accumulation + Other Mortality"
                        ),
                        class_="bg-body-tertiary p-3 rounded",
                    ),
                    ui.p("Or mathematically:", class_="mt-2"),
                    ui.tags.div(
                        ui.tags.code(
                            "Bᵢ × PBᵢ × EEᵢ = Σⱼ(Bⱼ × QBⱼ × DCⱼᵢ) + Yᵢ + Eᵢ + BAᵢ"
                        ),
                        class_="bg-body-tertiary p-3 rounded",
                    ),
                    ui.h5("References", class_="mt-4"),
                    ui.tags.ul(*(_reference_item(r) for r in REFERENCES)),
                ),
            ),
            ui.card(
                ui.card_header("Development"),
                ui.card_body(
                    ui.layout_columns(
                        _feature_list(
                            "Technology",
                            "Python 3.11 or newer",
                            "NumPy, SciPy and pandas for the calculations",
                            "Numba, optional, for a compiled consumption kernel",
                            "Shiny for Python for this app",
                            "Plotly and Matplotlib for charts, deck.gl for maps",
                            "GeoPandas and Shapely for spatial grids",
                        ),
                        ui.div(
                            ui.h5("Links"),
                            ui.tags.ul(
                                ui.tags.li(
                                    _external_link("GitHub Repository", REPO_URL)
                                ),
                                ui.tags.li(_external_link("Documentation", DOCS_URL)),
                                ui.tags.li(_external_link("Changelog", CHANGELOG_URL)),
                                ui.tags.li(_external_link("Conda Packages", CONDA_URL)),
                                ui.tags.li(
                                    _external_link("Original Rpath Package", RPATH_URL)
                                ),
                                ui.tags.li(
                                    _external_link("EwE Official Site", EWE_URL)
                                ),
                            ),
                        ),
                        ui.div(
                            ui.h5("Install and License"),
                            ui.p("Both packages install from the conda channel:"),
                            ui.tags.div(
                                ui.tags.code(INSTALL_COMMAND),
                                class_="bg-body-tertiary p-2 rounded mb-3",
                            ),
                            ui.p(
                                "MIT License. PyPath is free and open source "
                                "software. Contributions are welcome!",
                                class_="mb-0",
                            ),
                        ),
                        col_widths={"sm": 12, "md": 6, "xl": 4},
                    ),
                ),
            ),
            ui.card(
                ui.card_header("Version Information"),
                ui.card_body(
                    ui.tags.table(
                        *(
                            ui.tags.tr(
                                ui.tags.th(label, scope="row"), ui.tags.td(value)
                            )
                            for label, value in _version_rows()
                        ),
                        class_="table table-sm about-versions",
                    ),
                ),
            ),
            class_="container py-4",
        )
    )


def about_server(input: Inputs, output: Outputs, session: Session):
    """About page server logic (minimal)."""
    pass
