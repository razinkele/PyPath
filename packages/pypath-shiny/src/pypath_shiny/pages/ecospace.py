"""
ECOSPACE Spatial Modeling Page

Interactive spatial ecosystem modeling with:
- Grid configuration (regular or irregular polygons)
- Habitat preferences and capacity
- Dispersal and movement parameters
- Spatial fishing effort allocation
- Spatial simulation and visualization
"""

import asyncio
import copy
import dataclasses
import logging
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

# pypath imports (path setup handled by app/__init__.py)
from pypath.core.ecosim import RunCancelled
from pypath.spatial import (
    EcospaceGrid,
    SpatialFishing,
    allocate_gravity,
    allocate_habitat_based,
    allocate_port_based,
    allocate_uniform,
    create_1d_grid,
    create_regular_grid,
    load_spatial_grid,
)
from shiny import Inputs, Outputs, Session, reactive, render, ui
from shiny_deckgl import CARTO_DARK, MapWidget

from pypath_shiny.config import SPATIAL
from pypath_shiny.pages.ecosim import month_progress_tags
from pypath_shiny.pages.ewe_scenario import (
    EXAMPLE_KIND,
    ewe_source_matches,
    model_group_names,
)
from pypath_shiny.services.background_runs import (
    SpatialOutcome,
    run_spatial_checked,
    scenario_changed_since,
    scenario_fingerprint,
    spatial_jobs,
    spatial_result_is_current,
)
from pypath_shiny.services.cell_tables import as_cell_table

# Import centralized configuration
from pypath_shiny.services.deckgl_maps import (
    BIOMASS_PALETTE,
    EFFORT_PALETTE,
    HABITAT_PALETTE,
    MAP_FIT_PADDING,
    MAP_TOOLTIP,
    effort_layers,
    gradient_legend,
    grid_bounds,
    grid_overview_layers,
    patch_value_layers,
)
from pypath_shiny.services.ecosim_solver import RUN_EXECUTOR, RunControl, RunSlot
from pypath_shiny.services.hydro_fields import (
    BEYOND_CHOICES,
    DEFAULT_SUBSTEPS,
    FIELD_LABELS,
    MAX_SUBSTEPS,
    build_transport,
    field_on_patches,
    field_range,
    month_labels,
    read_uploaded_boxes,
    summarise_boxes,
    transport_groups,
    transport_scenario,
)
from pypath_shiny.services.mesh_grids import (
    UNIT_LABELS,
    build_mesh_grid,
    estimated_run_seconds,
    read_uploaded_mesh,
    summarise_mesh,
)
from pypath_shiny.services.notifications import (
    notify_error,
    notify_info,
    notify_success,
    notify_warning,
)
from pypath_shiny.services.prey_switching import summary, summary_line
from pypath_shiny.services.stored_ecospace import (
    CELL_PLACEMENTS,
    GRID_CORNERS,
    load_example_grid,
    load_stored_ecospace,
)
from pypath_shiny.services.tasks import (
    TaskState,
    TaskStatus,
    apply_if_latest,
    finish_task,
    new_running_task,
    transition_task_state,
    update_task_progress,
)

try:
    import geopandas as gpd
    from shapely.geometry import Polygon

    _HAS_GIS = True
except ImportError:
    _HAS_GIS = False
    gpd = None
    Polygon = None

logger = logging.getLogger(__name__)

# deck.gl maps (shiny-deckgl). Module-level like the rest of the page UI;
# shiny-deckgl keeps per-session state, so sessions do not share layers.
# The maps follow the page's dark mode (the header toggle sets data-bs-theme):
# with a dark basemap named, they swap to it and back.
_MAP_OPTIONS = {"tooltip": MAP_TOOLTIP, "dark_style": CARTO_DARK}
GRID_MAP = MapWidget("ecospace_grid_map", **_MAP_OPTIONS)
HABITAT_MAP = MapWidget("ecospace_habitat_map", **_MAP_OPTIONS)
EFFORT_MAP = MapWidget("ecospace_effort_map", **_MAP_OPTIONS)
BIOMASS_MAP = MapWidget("ecospace_biomass_map", **_MAP_OPTIONS)
FIELD_MAP = MapWidget("ecospace_field_map", **_MAP_OPTIONS)
_MAPS = (GRID_MAP, HABITAT_MAP, EFFORT_MAP, BIOMASS_MAP, FIELD_MAP)


def spatial_task_alert(task_state: TaskState) -> tuple[str, str, str]:
    """Map spatial run task state to Bootstrap alert class, icon, and message."""
    if task_state.status is TaskStatus.RUNNING:
        return (
            "alert alert-info mt-2",
            "bi bi-arrow-repeat me-2",
            f"{task_state.message} ({task_state.progress:.0f}%)",
        )
    if task_state.status is TaskStatus.ERROR:
        msg = task_state.error or task_state.message or "Spatial simulation failed."
        return ("alert alert-danger mt-2", "bi bi-exclamation-triangle me-2", msg)
    if task_state.status is TaskStatus.SUCCESS:
        return (
            "alert alert-success mt-2",
            "bi bi-check-circle me-2",
            task_state.message or "Spatial simulation completed.",
        )
    if task_state.status is TaskStatus.CANCELLED:
        return (
            "alert alert-warning mt-2",
            "bi bi-pause-circle me-2",
            task_state.message or "Spatial simulation cancelled.",
        )
    return ("", "", "")


def page_ecospace_arrays(
    habitat_vec: np.ndarray,
    n_groups: int,
    dispersal_rate: float,
    advection: bool,
    gravity: float,
) -> dict:
    """Per-group EcospaceParams arrays for the page's run, from its controls.

    Every group gets the page's habitat pattern (as preference and capacity)
    and its movement settings. Returns the keyword arguments for
    ``EcospaceParams`` other than the grid.

    Ecospace arrays are indexed from 0 by group: row g - 1 is Ecosim group g
    (``pypath.spatial.dispersal.calculate_spatial_flux``). Unlike the state
    arrays, they have no "Outside" row, so there are ``n_groups`` rows.
    """
    habitat_pref = np.tile(np.asarray(habitat_vec, dtype=float), (n_groups, 1))
    return {
        "habitat_preference": habitat_pref,
        "habitat_capacity": habitat_pref.copy(),
        "dispersal_rate": np.full(n_groups, float(dispersal_rate)),
        "advection_enabled": np.full(n_groups, bool(advection), dtype=bool),
        "gravity_strength": np.full(n_groups, float(gravity)),
    }


def wizard_params_on_grid(wizard_params, grid, n_groups=None):
    """The Ecospace Wizard's EcospaceParams while their grid is the page's grid.

    The wizard hands its grid over with its configuration. Building a new
    grid on this page replaces that grid, and the page's own controls apply
    again. Returns None when there is no wizard configuration or it is for
    another grid, and, given the scenario's ``n_groups``, when it does not
    have one row per group (a run would not use it).
    """
    if wizard_params is None or grid is None:
        return None
    if getattr(wizard_params, "grid", None) is not grid:
        return None
    if n_groups is not None and wizard_params.habitat_preference.shape[0] != n_groups:
        return None
    return wizard_params


def habitat_for_display(wizard_params, grid, page_vector, n_groups=None):
    """Habitat quality per patch for the habitat map and the effort preview.

    The mean of the wizard's per-group preferences while its configuration
    applies (``wizard_params_on_grid``), as habitat-based fishing averages
    them; otherwise ``page_vector``, the page's own habitat pattern.
    """
    active = wizard_params_on_grid(wizard_params, grid, n_groups)
    if active is None:
        return page_vector
    return np.asarray(active.habitat_preference, dtype=float).mean(axis=0)


# A whole configuration (grid, habitat, movement) the page runs instead of its
# own controls comes from the Ecospace Wizard or is loaded from an imported
# database (``services.stored_ecospace``); its origin is named in the notes.
WIZARD_ORIGIN = "the Ecospace Wizard's configuration"


def _duration_text(seconds: float) -> str:
    """A run time for a note: seconds below two minutes, then minutes, hours."""
    if seconds < 120:
        return f"{max(seconds, 1):.0f} seconds"
    if seconds < 2 * 3600:
        return f"{seconds / 60:.0f} minutes"
    return f"{seconds / 3600:.1f} hours"


def _renew_hint(origin: str) -> str:
    if origin == WIZARD_ORIGIN:
        return "Build the wizard configuration again for this model."
    return "Load it again for this model."


def run_ecospace_params(
    current_grid, n_groups, page_arrays, wizard_params=None, origin=WIZARD_ORIGIN
):
    """The EcospaceParams a run on this page uses, and a note for the user.

    A whole configuration (the Ecospace Wizard's, or one loaded from an
    imported database; ``origin`` names it) is used while it applies
    (``wizard_params_on_grid``) and has one row per scenario group. Otherwise
    the page's controls apply: ``page_arrays``, from ``page_ecospace_arrays``.
    The note (None when there is nothing to say) tells which was used, or why
    the configuration was not.
    """
    from pypath.spatial import EcospaceParams

    page = EcospaceParams(grid=current_grid, **page_arrays)
    wizard = wizard_params_on_grid(wizard_params, current_grid)
    if wizard is None:
        return page, None
    n_wizard = wizard.habitat_preference.shape[0]
    if n_wizard != n_groups:
        return page, (
            f"{origin[0].upper()}{origin[1:]} has {n_wizard} groups but "
            f"this scenario has {n_groups}, so this page's settings are used. "
            + _renew_hint(origin)
        )
    return wizard, f"Running with {origin}."


def parse_habitat_csv(path, n_patches):
    """Parse a habitat preference CSV into an array of `n_patches` values.

    The file must hold one numeric column with one row per patch (a header is
    optional). Values are clipped to [0, 1]. Raises ValueError when the shape
    does not match the current grid.
    """
    with open(path, encoding="utf-8-sig") as fh:
        first_line = fh.readline().strip()

    # A headerless file would otherwise lose its first value to the header row
    def _is_numeric(line):
        parts = [p.strip() for p in line.split(",") if p.strip()]
        if not parts:
            return False
        try:
            [float(p) for p in parts]
        except ValueError:
            return False
        return True

    df = pd.read_csv(path, header=None if _is_numeric(first_line) else "infer")
    numeric = df.select_dtypes(include="number")
    if numeric.shape[1] == 0:
        raise ValueError("no numeric column found")
    values = numeric.iloc[:, 0].to_numpy(dtype=float)
    if len(values) != n_patches:
        raise ValueError(
            f"expected {n_patches} rows (one per patch), found {len(values)}"
        )
    if np.isnan(values).any():
        raise ValueError("habitat values must all be numeric")
    return np.clip(values, 0.0, 1.0)


def _iter_polygons(geom):
    """Yield the Polygon parts of a geometry.

    A patch loaded from a user shapefile can be a MultiPolygon (an island
    group, say); iterating its parts keeps those patches visible on the maps.
    """
    if geom is None:
        return
    if geom.geom_type == "Polygon":
        yield geom
    elif geom.geom_type == "MultiPolygon":
        yield from geom.geoms


def create_hexagonal_grid_in_boundary(
    boundary_gdf, hexagon_size_km=None, auto_scale=True
):
    """Create a hexagonal grid within a boundary polygon.

    Parameters
    ----------
    boundary_gdf : geopandas.GeoDataFrame
        Boundary polygon(s) to fill with hexagons
    hexagon_size_km : float, optional
        Size of hexagons in kilometers (radius from center to vertex)
        If None, uses default from SPATIAL config
    auto_scale : bool, optional
        If True (default), automatically upscale hex size when the expected
        patch count is very large, to keep interactive grids manageable.
        Set to False to preserve the exact requested hex size.

    Returns
    -------
    EcospaceGrid
        Grid of hexagonal patches
    """
    # Use config default if not specified
    if hexagon_size_km is None:
        hexagon_size_km = SPATIAL.default_hexagon_size_km

    if not _HAS_GIS:
        raise ImportError("geopandas is required for hexagonal grid generation")

    from pypath.spatial.connectivity import build_adjacency_from_gdf

    # Get the union of all boundary polygons
    boundary_union = (
        boundary_gdf.union_all()
        if hasattr(boundary_gdf, "union_all")
        else boundary_gdf.unary_union
    )

    # Get bounds
    minx, miny, maxx, maxy = boundary_union.bounds

    # Project to metric CRS for accurate hexagon creation
    # Use UTM zone based on centroid longitude
    centroid_lon = (minx + maxx) / 2
    utm_zone = int((centroid_lon + 180) / 6) + 1
    utm_crs = (
        f"EPSG:{32600 + utm_zone}"
        if (miny + maxy) / 2 >= 0
        else f"EPSG:{32700 + utm_zone}"
    )

    # Project boundary to UTM
    boundary_gdf_utm = boundary_gdf.to_crs(utm_crs)
    boundary_union_utm = (
        boundary_gdf_utm.union_all()
        if hasattr(boundary_gdf_utm, "union_all")
        else boundary_gdf_utm.unary_union
    )

    # Convert km to meters for UTM
    hexagon_size_m = hexagon_size_km * 1000.0

    # Quick heuristic: if the boundary area is large relative to requested hex size,
    # upscale the requested size to keep the resulting number of patches within a
    # reasonable limit (avoid huge grids for typical interactive use).
    # Desired maximum patches for default behavior
    desired_max_patches = 200
    try:
        area_m2 = boundary_union_utm.area
        # Regular hexagon area = (3 * sqrt(3) / 2) * r^2
        hex_area_m2 = (3.0 * np.sqrt(3) / 2.0) * (hexagon_size_m**2)
        expected = area_m2 / hex_area_m2 if hex_area_m2 > 0 else float("inf")
        # Only auto-scale for very large expected grids to avoid inverting
        # the relative resolution requested by callers (scale threshold helps
        # keep fine/medium/coarse ordering intact for moderate cases).
        auto_scale_threshold = 10000
        if auto_scale and expected > auto_scale_threshold:
            scale = np.sqrt(expected / desired_max_patches)
            alpha = 0.25
            scale_applied = 1.0 + alpha * (scale - 1.0)
            hexagon_size_m = float(hexagon_size_m * scale_applied)
            logger.debug(
                "HEX-SCALE: expected=%.1f > %d; scale=%.3f alpha=%s -> applied=%.3f new_r=%.1f m",
                expected,
                auto_scale_threshold,
                scale,
                alpha,
                scale_applied,
                hexagon_size_m,
            )
        else:
            # Keep user-requested sizes for moderate areas to preserve monotonicity
            pass
    except (ValueError, AttributeError):
        logger.debug("HEX-SCALE: could not compute auto-scale heuristic", exc_info=True)

    # Calculate hexagon dimensions
    # For a regular hexagon with "radius" r (center to vertex):
    # - Width (flat-to-flat) = r * sqrt(3)
    # - Height (vertex-to-vertex) = 2 * r
    hex_width = hexagon_size_m * np.sqrt(3)
    hex_height = hexagon_size_m * 2.0

    # Get bounds in UTM
    minx_utm, miny_utm, maxx_utm, maxy_utm = boundary_union_utm.bounds

    # Generate hexagon centers
    hexagons = []

    # Anchor the grid to a consistent origin based on spacing to preserve
    # monotonic behavior (larger hex sizes -> fewer patches)
    row_step = hex_height * 0.75
    y0 = np.floor(miny_utm / row_step) * row_step
    x0 = np.floor(minx_utm / hex_width) * hex_width

    def _generate_centers_and_hexagons(x0_offset=0.0, y0_offset=0.0):
        """Generate hexagons with optional offsets applied to the anchored origin.

        Returns list of hexagon dicts and count.
        """
        local_hexagons = []
        local_hex_id = 0
        ry0 = y0 + y0_offset
        rx0 = x0 + x0_offset
        rrow = 0
        ry = ry0
        while ry < maxy_utm + row_step:
            x_offset = (hex_width / 2.0) if rrow % 2 == 1 else 0.0
            rx = rx0 + x_offset
            while rx < maxx_utm + hex_width:
                hx = create_hexagon(rx, ry, hexagon_size_m)
                if hx.intersects(boundary_union_utm):
                    clipped = hx.intersection(boundary_union_utm)
                    center_pt = (
                        type(hx).centroid.fget(hx)
                        if hasattr(type(hx), "centroid")
                        else hx.centroid
                    )
                    center_inside = center_pt.within(boundary_union_utm)
                    keep = False
                    if center_inside:
                        keep = True
                    elif clipped.area > (hx.area * 0.9):
                        keep = True
                    elif clipped.area > (hx.area * 0.5):
                        keep = True
                    if keep:
                        if clipped.geom_type == "Polygon":
                            local_hexagons.append(
                                {"id": local_hex_id, "geometry": clipped}
                            )
                            local_hex_id += 1
                        elif clipped.geom_type == "MultiPolygon":
                            largest = max(clipped.geoms, key=lambda p: p.area)
                            local_hexagons.append(
                                {"id": local_hex_id, "geometry": largest}
                            )
                            local_hex_id += 1
                rx += hex_width
            ry += row_step
            rrow += 1
        return local_hexagons, len(local_hexagons)

    # First pass with default anchoring
    hexagons, n_patches = _generate_centers_and_hexagons()

    if not hexagons:
        raise ValueError(
            "No hexagons fit within the boundary. Try a smaller hexagon size."
        )

    # Create GeoDataFrame
    hex_gdf = gpd.GeoDataFrame(hexagons, crs=utm_crs)

    # Calculate areas (in km²) and centroids
    # For area, use projected (UTM) coordinates
    areas_m2 = hex_gdf.geometry.area
    areas_km2 = areas_m2 / 1e6

    # Every hexagon kept here already passed the coverage test above, so the
    # tessellation is returned in full: silently dropping edge patches would
    # leave holes in the study area and understate its total area.

    # Centroids: calculate in UTM, then convert to WGS84
    centroids_utm = hex_gdf.geometry.centroid
    centroids_utm_coords = np.array([[c.x, c.y] for c in centroids_utm])

    # Convert centroid coordinates to WGS84
    from pyproj import Transformer

    transformer = Transformer.from_crs(utm_crs, "EPSG:4326", always_xy=True)
    centroids_lon, centroids_lat = transformer.transform(
        centroids_utm_coords[:, 0], centroids_utm_coords[:, 1]
    )
    centroids = np.column_stack([centroids_lon, centroids_lat])

    # Recompute WGS84 projection from final (possibly pruned) UTM GeoDataFrame
    hex_gdf_wgs84 = hex_gdf.to_crs("EPSG:4326")

    # Recompute adjacency from the (possibly pruned) GeoDataFrame so
    # indices match the finalized geometry list
    adjacency, metadata = build_adjacency_from_gdf(hex_gdf_wgs84, method="rook")

    # Metadata contains 'border_lengths' (keys are (i,j) tuples where i<j)
    raw_border_lengths = (
        metadata.get("border_lengths", {}) if isinstance(metadata, dict) else {}
    )
    # Normalize keys to (min, max) and ensure float values
    edge_lengths = {(min(k), max(k)): float(v) for k, v in raw_border_lengths.items()}

    # Ensure hexagons list matches final GeoDataFrame (ids sequential)
    hexagons = [{"id": i, "geometry": geom} for i, geom in enumerate(hex_gdf.geometry)]

    # Create EcospaceGrid
    grid = EcospaceGrid(
        n_patches=len(hexagons),
        patch_ids=np.arange(len(hexagons)),
        patch_areas=areas_km2.values,
        patch_centroids=centroids,
        adjacency_matrix=adjacency,
        edge_lengths=edge_lengths,
        crs="EPSG:4326",
        geometry=hex_gdf_wgs84,
    )

    return grid


def create_hexagon(center_x: float, center_y: float, radius: float) -> "Polygon":
    """Create a regular hexagon polygon (pointy-top orientation).

    Generates a hexagonal polygon with vertices arranged in a "pointy-top"
    orientation (flat sides on left and right, pointed vertices on top and bottom).
    This orientation is optimal for hexagonal tessellation with minimal gaps.

    Parameters
    ----------
    center_x : float
        X-coordinate of hexagon center (in same CRS as target grid)
    center_y : float
        Y-coordinate of hexagon center (in same CRS as target grid)
    radius : float
        Distance from hexagon center to any vertex (meters for UTM, degrees for WGS84).
        This is the "apothem" for the circumscribed circle.

    Returns
    -------
    shapely.geometry.Polygon
        Hexagon polygon with 6 vertices in pointy-top orientation.
        The polygon is closed (first and last vertices are identical).

    Notes
    -----
    **Hexagon Orientation:**
    - Pointy-top: Flat sides horizontal (left-right), points vertical (top-bottom)
    - Achieved by rotating vertices by 30° (π/6 radians) from standard orientation
    - This orientation tessellates with 3/4 row offset pattern

    **Hexagon Dimensions:**
    - Radius (center to vertex): r
    - Width (flat-to-flat): r × √3
    - Height (point-to-point): 2r
    - Area: (3√3/2) × r²

    Examples
    --------
    >>> from shapely.geometry import Polygon
    >>> hex = create_hexagon(100.0, 50.0, 10.0)
    >>> hex.area  # doctest: +SKIP
    259.8076...  # Approximately 3√3/2 × 10² = 259.81
    >>> len(list(hex.exterior.coords))
    7  # 6 vertices + closing point
    """
    # Create pointy-top hexagon by starting at 30 degrees (pi/6).
    # Use 6 vertex angles (exclude endpoint) to avoid duplicating the first
    # vertex before Shapely automatically closes the polygon.
    angles = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi / 6  # 6 vertices
    x_coords = center_x + radius * np.cos(angles)
    y_coords = center_y + radius * np.sin(angles)
    # Construct polygon from 6 vertices; Shapely will close the polygon (7 coords total)
    return Polygon(list(zip(x_coords, y_coords)))


def _notify_legacy(message: str, type: str = "default", duration: int | None = None):
    """Compatibility wrapper while migrating this page to notification services."""
    if type == "error":
        notify_error(message, duration=duration)
    elif type == "warning":
        notify_warning(message, duration=duration)
    elif type == "message":
        notify_success(message, duration=duration)
    else:
        notify_info(message, duration=duration)


def ecospace_ui():
    """UI for ECOSPACE spatial modeling page."""
    return ui.div(
        ui.layout_sidebar(
            ui.sidebar(
                ui.h4("ECOSPACE Configuration", class_="mb-3"),
                # A stored configuration of the model that is open: the
                # Ecospace scenario of an imported database, or an example's
                # grid
                ui.output_ui("stored_ecospace_ui"),
                # Grid setup
                ui.accordion(
                    ui.accordion_panel(
                        "Spatial Grid",
                        ui.input_select(
                            "grid_type",
                            "Grid Type",
                            choices={
                                "regular_2d": "Regular 2D Grid",
                                "1d_transect": "1D Transect (Linear)",
                                "custom": "Custom Polygons (Upload Shapefile)",
                                "shyfem": "SHYFEM mesh (.grd)",
                            },
                            selected="regular_2d",
                        ),
                        ui.panel_conditional(
                            "input.grid_type === 'regular_2d'",
                            ui.input_numeric(
                                "grid_nx",
                                "Number of Columns (nx)",
                                value=5,
                                min=2,
                                max=20,
                            ),
                            ui.input_numeric(
                                "grid_ny", "Number of Rows (ny)", value=5, min=2, max=20
                            ),
                        ),
                        ui.panel_conditional(
                            "input.grid_type === '1d_transect'",
                            ui.input_numeric(
                                "grid_n_patches",
                                "Number of Patches",
                                value=10,
                                min=3,
                                max=50,
                            ),
                        ),
                        ui.panel_conditional(
                            "input.grid_type === 'custom'",
                            ui.input_file(
                                "spatial_file_upload",
                                "Upload Spatial File",
                                accept=[".zip", ".geojson", ".json", ".gpkg"],
                                multiple=False,
                            ),
                            ui.p(
                                ui.tags.strong("Supported formats: "),
                                "Shapefile (.zip), GeoJSON (.geojson/.json), GeoPackage (.gpkg). "
                                "File must contain polygon geometries.",
                                class_="text-muted small",
                            ),
                            ui.p(
                                ui.tags.strong("Note: "),
                                "For 'Use polygons' mode, an 'id' field is required. "
                                "For 'Create hexagons' mode, any boundary polygon works.",
                                class_="text-info small",
                            ),
                            ui.hr(),
                            ui.input_radio_buttons(
                                "custom_grid_mode",
                                "Grid Mode",
                                choices={
                                    "use_polygons": "Use uploaded polygons as-is",
                                    "create_hexagons": "Create hexagonal grid within boundary",
                                },
                                selected="use_polygons",
                            ),
                            ui.panel_conditional(
                                "input.custom_grid_mode === 'use_polygons'",
                                ui.input_text(
                                    "id_field_name",
                                    "ID Field Name (optional)",
                                    value="id",
                                    placeholder="id",
                                ),
                                ui.p(
                                    "Name of the field containing unique patch IDs (default: 'id').",
                                    class_="text-muted small",
                                ),
                            ),
                            ui.panel_conditional(
                                "input.custom_grid_mode === 'create_hexagons'",
                                ui.input_slider(
                                    "hexagon_size_km",
                                    "Hexagon Size (km)",
                                    min=SPATIAL.min_hexagon_size_km,
                                    max=SPATIAL.max_hexagon_size_km,
                                    value=SPATIAL.default_hexagon_size_km,
                                    step=0.25,
                                ),
                                ui.p(
                                    ui.tags.strong("Info: "),
                                    "Hexagonal grids provide better spatial isotropy (no directional bias) "
                                    "and each cell has 6 equidistant neighbors.",
                                    class_="text-info small",
                                ),
                                ui.p(
                                    "Size determines the distance from hexagon center to vertex. "
                                    "Smaller hexagons = more patches = slower computation.",
                                    class_="text-muted small",
                                ),
                            ),
                        ),
                        ui.panel_conditional(
                            "input.grid_type === 'shyfem'",
                            ui.input_file(
                                "mesh_file_upload",
                                "Upload a SHYFEM grid",
                                accept=[".grd"],
                                multiple=False,
                            ),
                            ui.input_radio_buttons(
                                "mesh_patches",
                                "Patches",
                                choices={
                                    "type": "One per element type (the boxes of a box model)",
                                    "element": "One per element",
                                },
                                selected="type",
                            ),
                            ui.input_select(
                                "mesh_units",
                                "Coordinates in the file",
                                choices={
                                    "auto": "Detect",
                                    "degrees": "Degrees (longitude, latitude)",
                                    "m": "Metres",
                                    "km": "km",
                                },
                                selected="auto",
                            ),
                            ui.input_text(
                                "mesh_epsg",
                                "EPSG code (for metres or km)",
                                value="",
                                placeholder="e.g. 32634",
                            ),
                            ui.output_ui("mesh_summary"),
                        ),
                        ui.input_action_button(
                            "create_grid",
                            "Create Grid",
                            class_="btn btn-primary w-100 mt-2",
                        ),
                        icon=ui.tags.i(class_="bi bi-grid-3x3-gap"),
                    ),
                    # Dispersal parameters
                    ui.accordion_panel(
                        "Movement & Dispersal",
                        ui.p(
                            "Configure how organisms move between patches.",
                            class_="text-muted small mb-3",
                        ),
                        ui.input_numeric(
                            "dispersal_rate_default",
                            "Dispersal Rate (per year)",
                            value=5.0,
                            min=0,
                            max=100,
                            step=1,
                        ),
                        ui.input_numeric(
                            "gravity_strength",
                            "Habitat Attraction Strength (0-1)",
                            value=0.5,
                            min=0,
                            max=1,
                            step=0.1,
                        ),
                        ui.input_checkbox(
                            "enable_advection",
                            "Enable Habitat-Directed Movement",
                            value=True,
                        ),
                        ui.p(
                            "Without an active Ecospace Wizard configuration "
                            "(its rates take precedence), every group "
                            "disperses at this rate: between two neighbouring "
                            "square cells, biomass flows at this rate times "
                            "their current biomass difference, per year. For "
                            "a non-migratory group, EwE's Mvel of 300 km/year "
                            "on 36.7 km cells is about 2.6 per year.",
                            class_="text-muted small",
                        ),
                        icon=ui.tags.i(class_="bi bi-arrows-move"),
                    ),
                    # A hydrodynamic model's box output: fields and transport
                    ui.accordion_panel(
                        "Hydrodynamic Model",
                        ui.p(
                            "Upload a SHYFEM box model's output (boxes_2d.txt) "
                            "to see its fields on the grid and to let its "
                            "water flows carry chosen groups. Make the grid "
                            "from the same model's .grd first.",
                            class_="text-muted small mb-3",
                        ),
                        ui.input_file(
                            "box_file_upload",
                            "Upload box output",
                            accept=[".txt"],
                            multiple=False,
                        ),
                        ui.output_ui("box_summary"),
                        ui.input_selectize(
                            "transport_groups",
                            "Groups the water carries",
                            choices={},
                            multiple=True,
                        ),
                        ui.output_ui("transport_note"),
                        ui.input_numeric(
                            "transport_substeps",
                            "Sub-steps a month (when a fast group is carried)",
                            value=DEFAULT_SUBSTEPS,
                            min=1,
                            max=MAX_SUBSTEPS,
                            step=1,
                        ),
                        ui.input_checkbox(
                            "transport_check",
                            "Check the sub-steps when a fast group is carried "
                            "(the run takes three times as long)",
                            value=True,
                        ),
                        ui.input_select(
                            "transport_beyond",
                            "When the run is longer than the file",
                            choices=BEYOND_CHOICES,
                            selected="repeat_year",
                        ),
                        icon=ui.tags.i(class_="bi bi-water"),
                    ),
                    # Habitat configuration
                    ui.accordion_panel(
                        "Habitat Preferences",
                        ui.input_select(
                            "habitat_pattern",
                            "Habitat Pattern",
                            choices={
                                "uniform": "Uniform (all patches equal)",
                                "gradient": "Linear Gradient",
                                "patchy": "Patchy (random variation)",
                                "core_periphery": "Core-Periphery",
                                "custom": "Custom (upload CSV)",
                            },
                            selected="gradient",
                        ),
                        ui.panel_conditional(
                            "input.habitat_pattern === 'gradient'",
                            ui.input_select(
                                "gradient_direction",
                                "Gradient Direction",
                                choices={
                                    "horizontal": "Horizontal (West → East)",
                                    "vertical": "Vertical (South → North)",
                                    "radial": "Radial (Center → Edge)",
                                },
                                selected="horizontal",
                            ),
                        ),
                        ui.panel_conditional(
                            "input.habitat_pattern === 'custom'",
                            ui.input_file(
                                "habitat_upload",
                                "Upload Habitat Matrix (CSV)",
                                accept=[".csv"],
                                multiple=False,
                            ),
                        ),
                        icon=ui.tags.i(class_="bi bi-geo-alt"),
                    ),
                    # Spatial fishing
                    ui.accordion_panel(
                        "Spatial Fishing",
                        ui.input_select(
                            "fishing_allocation",
                            "Effort Allocation Method",
                            choices={
                                "uniform": "Uniform (equal across patches)",
                                "gravity": "Gravity (follow biomass)",
                                "port": "Port-based (distance decay)",
                                "habitat": "Habitat-based (target quality)",
                            },
                            selected="gravity",
                        ),
                        ui.panel_conditional(
                            "input.fishing_allocation === 'gravity'",
                            ui.input_slider(
                                "gravity_alpha",
                                "Biomass Attraction (α)",
                                min=0,
                                max=2,
                                value=1.0,
                                step=0.1,
                            ),
                            ui.input_selectize(
                                "fishing_target_groups",
                                "Target Groups",
                                choices={},
                                multiple=True,
                            ),
                            ui.p(
                                "Effort follows the biomass of these groups. "
                                "Leave empty to follow total biomass.",
                                class_="text-muted small",
                            ),
                        ),
                        ui.panel_conditional(
                            "input.fishing_allocation === 'port'",
                            ui.input_text(
                                "port_patches",
                                "Port Patch Indices (comma-separated)",
                                value="0",
                            ),
                            ui.input_slider(
                                "port_beta",
                                "Distance Decay (β)",
                                min=0,
                                max=3,
                                value=1.0,
                                step=0.1,
                            ),
                        ),
                        icon=ui.tags.i(class_="bi bi-gear"),
                    ),
                    id="ecospace_accordion",
                    open=["Spatial Grid"],
                    multiple=True,
                ),
                ui.hr(),
                ui.output_ui("wizard_config_note"),
                # Run simulation button
                # A task button: busy while the run computes. It is never
                # disabled; without a scenario or a grid the handler says so.
                ui.input_task_button(
                    "run_spatial_sim",
                    "Run Spatial Simulation",
                    icon=ui.tags.i(class_="bi bi-play-fill"),
                    label_busy="Running...",
                    type="success",
                    class_="w-100 mt-3",
                ),
                ui.output_ui("spatial_run_status"),
                ui.output_ui("spatial_switching_line"),
                ui.output_ui("spatial_transport_line"),
                role="complementary",
                aria_label="Ecospace settings",
                width=350,
            ),
            # Main panel with tabs
            ui.navset_card_tab(
                ui.nav_panel(
                    "Grid Visualization",
                    ui.output_ui("grid_plot_note"),
                    GRID_MAP.ui(height="500px"),
                    ui.div(ui.output_text("grid_info"), class_="alert alert-info mt-3"),
                ),
                ui.nav_panel(
                    "Habitat Map",
                    HABITAT_MAP.ui(height="500px"),
                ),
                ui.nav_panel(
                    "Hydrodynamic Fields",
                    ui.output_ui("field_map_note"),
                    FIELD_MAP.ui(height="500px"),
                    ui.div(
                        ui.input_select(
                            "box_variable",
                            "Field:",
                            choices={},
                            width="300px",
                        ),
                        ui.input_slider(
                            "box_month",
                            "Month of the file",
                            min=1,
                            max=12,
                            value=1,
                            step=1,
                            animate=True,
                        ),
                        class_="mt-3",
                    ),
                ),
                ui.nav_panel(
                    "Fishing Effort",
                    EFFORT_MAP.ui(height="500px"),
                    ui.p(
                        "Spatial distribution of fishing effort based on selected allocation method.",
                        class_="text-muted small mt-2",
                    ),
                    ui.p(
                        "Effort is redistributed relative to the grid mean, so total "
                        "fleet effort is unchanged. 'Uniform' matches a run with no "
                        "spatial allocation.",
                        class_="text-muted small",
                    ),
                ),
                ui.nav_panel(
                    "Biomass Animation",
                    ui.output_ui("biomass_animation_ui"),
                    BIOMASS_MAP.ui(height="500px"),
                    ui.div(
                        ui.input_slider(
                            "animation_time",
                            "Time Step",
                            min=0,
                            max=100,
                            value=0,
                            step=1,
                            animate=True,
                        ),
                        ui.input_select(
                            "biomass_view_group",
                            "View Biomass for Group:",
                            choices={},
                            width="300px",
                        ),
                        class_="mt-3",
                    ),
                ),
                ui.nav_panel(
                    "Spatial Metrics",
                    ui.output_ui("spatial_metrics_table"),
                    ui.p(
                        "Summary statistics for spatial distribution of biomass.",
                        class_="text-muted small mt-2",
                    ),
                ),
                id="ecospace_tabs",
                full_screen=True,
            ),
        ),
        # Page header
        ui.div(
            ui.h2(
                ui.tags.i(class_="bi bi-map me-2"),
                "ECOSPACE - Spatial Ecosystem Modeling",
                class_="mb-2",
            ),
            ui.p(
                "Configure spatial grids, habitat preferences, movement parameters, and fishing allocation. "
                "Run spatially-explicit ecosystem simulations and visualize results.",
                class_="text-muted mb-4",
            ),
            # Audit vs EwE 6.7.144 (2026-11-10), findings B1-B4. Remove once
            # roadmap phase P6 lands.
            ui.div(
                ui.tags.i(class_="bi bi-exclamation-triangle me-2"),
                ui.strong("Not EwE-equivalent yet. "),
                "Ecospace runs here are not equivalent to EwE Ecospace output: "
                "dispersal units, capacity-driven movement and dynamic habitat "
                "capacity still differ from EwE 6.7.144. Use them for exploration, "
                "not for comparison with EwE results.",
                id="ecospace_parity_warning",
                class_="alert alert-warning",
                role="alert",
            ),
            class_="mb-4",
        ),
    )


def ecospace_server(
    input: Inputs,
    _output: Outputs,
    _session: Session,
    _model_data: reactive.Value,
    _sim_results: reactive.Value,
    _sim_scenario: reactive.Value = None,
    wizard_params: reactive.Value = None,
    ewe_source: reactive.Value = None,
):
    """Server logic for ECOSPACE page.

    ``ewe_source``, when given, is the record of the imported EwE database
    or of the example model that is open: its Ecospace scenarios, or its
    grid, are offered to load while the current model has its groups.
    """

    # Reactive values for spatial state
    grid = reactive.Value(None)
    boundary_polygon = reactive.Value(None)  # Store uploaded boundary for visualization
    _ecospace_params = reactive.Value(None)
    # where _ecospace_params comes from, as the notes name it
    _config_origin = reactive.Value(WIZARD_ORIGIN)
    # the groups a loaded scenario's rows belong to, in order (None for the
    # wizard's configuration, which is built for the model that is open)
    _config_groups = reactive.Value(None)
    _spatial_results = reactive.Value(None)
    _spatial_switching = reactive.Value(None)
    # what the shown run did with the water transport: (css class, text) lines
    _spatial_notes = reactive.Value(())
    spatial_task_state = reactive.Value(TaskState(run_id="spatial-idle"))

    @reactive.effect
    def _adopt_wizard_params():
        """Pick up an EcospaceParams built by the Ecospace Wizard."""
        if wizard_params is None:
            return
        params_obj = wizard_params.get()
        if params_obj is None:
            return
        _ecospace_params.set(params_obj)
        _config_origin.set(WIZARD_ORIGIN)
        _config_groups.set(None)
        wizard_grid = getattr(params_obj, "grid", None)
        if wizard_grid is not None:
            grid.set(wizard_grid)
            _spatial_results.set(None)
            _spatial_switching.set(None)
            _notify_legacy(
                f"Loaded wizard configuration: {wizard_grid.n_patches} patches",
                type="message",
                duration=5,
            )

    def _scenario_n_groups():
        """The Ecosim scenario's group count, or None before there is one."""
        scenario = _sim_scenario() if _sim_scenario is not None else None
        return getattr(getattr(scenario, "params", None), "NUM_GROUPS", None)

    @render.ui
    def wizard_config_note():
        """Say, beside the Run button, whether the wizard's configuration runs."""
        wizard = wizard_params_on_grid(_ecospace_params(), grid())
        if wizard is None:
            return None
        n_groups = _scenario_n_groups()
        origin = _config_origin()
        if (
            n_groups is not None
            and wizard_params_on_grid(wizard, grid(), n_groups) is None
        ):
            return ui.div(
                ui.tags.i(class_="bi bi-exclamation-triangle me-2"),
                f"{origin[0].upper()}{origin[1:]} has "
                f"{wizard.habitat_preference.shape[0]} groups but the scenario "
                f"has {n_groups}, so runs use this page's settings. "
                + _renew_hint(origin),
                class_="alert alert-warning small mt-2 mb-0",
            )
        if origin == WIZARD_ORIGIN:
            used = (
                "Runs use the Ecospace Wizard's habitat preferences and "
                "movement settings for this grid."
            )
        else:
            used = (
                f"Runs use the grid, habitat capacity and dispersal rates of {origin}."
            )
        return ui.div(
            ui.tags.i(class_="bi bi-magic me-2"),
            f"{used} The Habitat Preferences and Movement & "
            "Dispersal controls apply after you create a new grid here.",
            id="ecospace_config_note",
            class_="alert alert-info small mt-2 mb-0",
        )

    # --- a stored configuration of the open model --------------------------

    def _matching_source():
        """The source record, while the current model has its groups."""
        if ewe_source is None:
            return None
        source = ewe_source.get()
        return source if ewe_source_matches(source, _model_data.get()) else None

    @render.ui
    def stored_ecospace_ui():
        """Offer the open model's stored Ecospace scenario, or its grid."""
        source = _matching_source()
        if source is None:
            return ui.p(
                "To load a spatial model: import an EwE database that holds an "
                "Ecospace scenario (Data Import), or open the Curonian Lagoon "
                "example (Ecosim page), which brings its 29-box grid. Either "
                "is then offered here.",
                id="stored_ecospace_hint",
                class_="text-muted small",
            )
        if source.get("kind") == EXAMPLE_KIND:
            if not source.get("ecospace_grid"):
                return ui.p(
                    f"The example {source['name']} has no spatial grid: build "
                    "one below.",
                    id="stored_ecospace_hint",
                    class_="text-muted small",
                )
            return ui.div(
                ui.h5("Example Grid"),
                ui.input_action_button(
                    "btn_load_example_grid",
                    ui.TagList(
                        ui.tags.i(class_="bi bi-grid-3x3-gap me-2"),
                        "Load the example's grid",
                    ),
                    class_="btn btn-outline-primary btn-sm w-100",
                ),
                ui.p(
                    "The boxes of the lagoon's hydrodynamic box model, one "
                    "patch each. Only the grid is loaded: habitat and "
                    "movement are this page's settings, and the habitat "
                    "pattern is set to uniform. The box model's output fits "
                    "this grid (Hydrodynamic Model).",
                    class_="text-muted small mt-2",
                ),
                ui.tags.hr(),
                id="stored_ecospace_block",
            )
        scenarios = {
            str(k): v for k, v in (source.get("ecospace_scenarios") or {}).items()
        }
        if not scenarios:
            return ui.p(
                f"{source['name']} holds no Ecospace scenario with a grid: "
                "build the grid below.",
                id="stored_ecospace_hint",
                class_="text-muted small",
            )
        return ui.div(
            ui.h5("Stored Ecospace Scenario"),
            ui.input_select("stored_ecospace_id", f"In {source['name']}", scenarios),
            ui.input_select(
                "stored_ecospace_corner",
                "The scenario's coordinates give the grid's",
                GRID_CORNERS,
                selected="north_west",
            ),
            ui.input_select(
                "stored_ecospace_placement",
                "Draw the cells",
                CELL_PLACEMENTS,
                selected="ground",
            ),
            ui.input_action_button(
                "btn_load_stored_ecospace",
                ui.TagList(
                    ui.tags.i(class_="bi bi-box-arrow-in-down me-2"),
                    "Load this scenario",
                ),
                class_="btn btn-outline-primary btn-sm w-100",
            ),
            ui.p(
                "Loads the scenario's grid (the water cells of its depth map), "
                "each group's habitat capacity and its dispersal rates, and "
                "runs with them. Not used in a run here: environmental "
                "drivers, migration and monthly maps, the fleets' habitat "
                "rules, protected areas and ports.",
                class_="text-muted small mt-2",
            ),
            ui.p(
                "EwE takes a cell as the same number of degrees each way, "
                "which draws a grid of km cells too narrow east to west (by "
                "the cosine of the latitude: about half as wide at 55° N). "
                "The cells are drawn where they lie on the ground unless you "
                "choose EwE's coordinates; the run is the same either way.",
                class_="text-muted small",
            ),
            ui.tags.hr(),
            id="stored_ecospace_block",
        )

    @reactive.effect
    @reactive.event(input.btn_load_stored_ecospace)
    def _load_stored_ecospace():
        """Load the chosen Ecospace scenario of the imported database."""
        source = _matching_source()
        if source is None:
            _notify_legacy(
                "The imported database no longer matches the model that is open.",
                type="warning",
            )
            return
        try:
            stored = load_stored_ecospace(
                source,
                input.stored_ecospace_id(),
                _model_data.get(),
                grid_corner=input.stored_ecospace_corner() or "north_west",
                cell_placement=input.stored_ecospace_placement() or "ground",
            )
        except Exception as e:
            logger.debug("Stored Ecospace scenario not loaded: %s", e)
            _notify_legacy(
                f"Could not load the scenario: {e!s}", type="error", duration=10
            )
            return
        _ecospace_params.set(stored.params)
        _config_origin.set(stored.origin)
        _config_groups.set(list(stored.groups))
        grid.set(stored.params.grid)
        _spatial_results.set(None)
        _spatial_switching.set(None)
        scenario = _sim_scenario() if _sim_scenario is not None else None
        months = getattr(getattr(scenario, "fishing", None), "ForcedEffort", None)
        years = months.shape[0] / 12.0 if months is not None else 1.0
        run_params = getattr(scenario, "params", None)
        seconds = estimated_run_seconds(
            stored.n_patches,
            stored.n_groups,
            years,
            substeps=int(getattr(run_params, "RK4_STEPS", 1) or 1),
            fast_equilibrium=bool(getattr(run_params, "RK4_FAST_EQUILIBRIUM", True)),
        )
        span = f"{years:g} years" if months is not None else "a simulated year"
        skipped = (
            " Not used here: " + ", ".join(stored.not_applied) + "."
            if stored.not_applied
            else ""
        )
        _notify_legacy(
            f"Loaded {stored.origin}: {stored.summary}. A run of {span} takes "
            f"about {_duration_text(seconds)}.{skipped}",
            type="message",
            duration=12,
        )

    @reactive.effect
    def _drop_a_scenario_of_another_model():
        """A loaded scenario's rows belong to the groups it was read for.

        Another model, or the same one with a group added, removed, renamed
        or moved, must not run with them: a re-balance of the same groups
        keeps the scenario.
        """
        model = _model_data.get()
        with reactive.isolate():
            groups = _config_groups.get()
            origin = _config_origin.get()
        if groups is None or model_group_names(model) == groups:
            return
        _ecospace_params.set(None)
        _config_origin.set(WIZARD_ORIGIN)
        _config_groups.set(None)
        _spatial_results.set(None)
        _spatial_switching.set(None)
        _notify_legacy(
            f"{origin[0].upper()}{origin[1:]} was loaded for another set of "
            "groups and is no longer used: this page's settings apply. Load "
            "it again for the model that is open.",
            type="warning",
            duration=10,
        )

    @reactive.effect
    @reactive.event(input.btn_load_example_grid)
    def _load_example_grid():
        """Load the grid shipped with the example model that is open."""
        source = _matching_source()
        try:
            new_grid = load_example_grid(source)
        except Exception as e:
            logger.debug("Example grid not loaded: %s", e)
            _notify_legacy(f"Could not load the example's grid: {e!s}", type="error")
            return
        grid.set(new_grid)
        _spatial_results.set(None)
        _spatial_switching.set(None)
        # the default west-to-east gradient would leave the western boxes
        # with no habitat at all
        ui.update_select("habitat_pattern", selected="uniform")
        _notify_legacy(
            f"Loaded the grid of {source['name']}: {new_grid.n_patches} patches "
            f"({new_grid.patch_areas.sum():,.0f} km²). The habitat pattern is "
            "set to uniform.",
            type="message",
            duration=8,
        )

    # Load and display boundary polygon immediately on file upload
    @reactive.effect
    def load_boundary_on_upload():
        """Load boundary polygon for visualization when file is uploaded."""
        # Only process if in custom grid mode
        if input.grid_type() != "custom":
            # Drop any previously uploaded boundary: leaving it set makes the
            # grid map centre on a boundary that no longer applies.
            boundary_polygon.set(None)
            return

        # Check if file is uploaded
        file_info = input.spatial_file_upload()
        if file_info is None or len(file_info) == 0:
            # No file uploaded, clear boundary
            boundary_polygon.set(None)
            return

        try:
            uploaded_file = file_info[0]
            file_path = uploaded_file["datapath"]
            file_name = uploaded_file["name"]

            # Create temporary directory for processing
            temp_dir = tempfile.mkdtemp()

            try:
                # Handle different file types
                if file_name.endswith(".zip"):
                    # Extract shapefile from zip
                    with zipfile.ZipFile(file_path, "r") as zip_ref:
                        zip_ref.extractall(temp_dir)

                    # Find the .shp file
                    shp_files = list(Path(temp_dir).glob("**/*.shp"))
                    if not shp_files:
                        raise ValueError("No .shp file found in zip archive")

                    spatial_file = str(shp_files[0])

                elif file_name.endswith((".geojson", ".json", ".gpkg")):
                    # Copy file to temp directory
                    spatial_file = str(Path(temp_dir) / file_name)
                    shutil.copy(file_path, spatial_file)

                else:
                    raise ValueError(f"Unsupported file format: {file_name}")

                # Load boundary file for visualization
                if not _HAS_GIS:
                    raise ImportError(
                        "geopandas is required for spatial file processing"
                    )

                boundary_gdf = gpd.read_file(spatial_file)
                if boundary_gdf.crs is None:
                    boundary_gdf = boundary_gdf.set_crs("EPSG:4326")
                else:
                    boundary_gdf = boundary_gdf.to_crs("EPSG:4326")

                # Store boundary for visualization
                boundary_polygon.set(boundary_gdf)

                _notify_legacy(
                    f"Boundary loaded: {len(boundary_gdf)} feature(s) from {file_name}",
                    type="message",
                    duration=3,
                )

            finally:
                # Clean up temporary directory
                shutil.rmtree(temp_dir, ignore_errors=True)

        except (
            ValueError,
            TypeError,
            OSError,
            RuntimeError,
            ImportError,
            zipfile.BadZipFile,
        ) as e:
            # RuntimeError covers pyogrio DataSourceError; ImportError covers a
            # missing geopandas. An uncaught exception here would close the
            # whole Shiny session.
            logger.debug("Boundary load failed: %s", e)
            _notify_legacy(f"Error loading boundary: {e!s}", type="warning", duration=5)
            boundary_polygon.set(None)

    @reactive.calc
    def _uploaded_mesh():
        """The uploaded .grd as a mesh, or None; a file that cannot be read is reported."""
        file_info = input.mesh_file_upload()
        if not file_info:
            return None
        try:
            return read_uploaded_mesh(file_info[0]["datapath"])
        except (ValueError, OSError, MemoryError) as e:
            logger.debug("Could not read the mesh: %s", e)
            _notify_legacy(f"Could not read the mesh: {e!s}", type="error", duration=6)
            return None

    @render.ui
    def mesh_summary():
        """What the uploaded mesh holds, to check before a grid is made."""
        mesh = _uploaded_mesh()
        if mesh is None:
            return ui.p(
                "A .grd file names no coordinate system: check the units "
                "shown here after the upload.",
                class_="text-muted small",
            )
        try:
            info = summarise_mesh(mesh, input.mesh_units() or "auto")
        except ValueError as e:
            return ui.p(f"This mesh cannot be used: {e!s}", class_="text-danger small")
        n_patches = (
            info.n_elements if input.mesh_patches() == "element" else info.n_types
        )
        lines = [
            f"{info.n_nodes:,} nodes, {info.n_elements:,} elements, "
            f"{info.n_types:,} element types",
            f"Coordinates read as {UNIT_LABELS[info.units]}"
            + (" (detected)" if (input.mesh_units() or "auto") == "auto" else "")
            + f": x {info.x_range[0]:,.4g} to {info.x_range[1]:,.4g}, "
            f"y {info.y_range[0]:,.4g} to {info.y_range[1]:,.4g}",
            f"Area {info.area_km2:,.1f} km²"
            + (
                f", depth {info.depth_min:.1f} to {info.depth_max:.1f} m (means per element type)"
                if info.depth_min is not None
                else ", no depth in the file"
            ),
            f"This choice makes {n_patches:,} patches.",
        ]
        notes = []
        if info.units != "degrees":
            notes.append(
                ui.p(
                    "Metres or km: enter the EPSG code so the map can place the mesh.",
                    class_="text-info small mb-1",
                )
            )
        if n_patches > SPATIAL.huge_grid_threshold:
            minutes = estimated_run_seconds(n_patches, 20, 10) / 60.0
            notes.append(
                ui.p(
                    f"A run on {n_patches:,} patches is slow: about "
                    f"{minutes:,.0f} minutes for a 20-group model over 10 years "
                    "under the EwE rule (one sub-step a month; more sub-steps "
                    "take that many times as long).",
                    class_="text-warning small mb-1",
                )
            )
        return ui.div(
            *[ui.p(line, class_="small mb-1") for line in lines],
            *notes,
            id="mesh_summary_text",
        )

    # Create grid based on configuration
    # --- a hydrodynamic model's box output ---------------------------------

    @reactive.calc
    def _uploaded_boxes():
        """The uploaded box output as monthly means, or None; errors are reported."""
        file_info = input.box_file_upload()
        if not file_info:
            return None
        try:
            return read_uploaded_boxes(file_info[0]["datapath"])
        except Exception as e:
            logger.debug("Could not read the box output: %s", e)
            _notify_legacy(
                f"Could not read the box output: {e!s}", type="error", duration=8
            )
            return None

    @reactive.calc
    def _box_summary():
        series = _uploaded_boxes()
        return None if series is None else summarise_boxes(series, grid())

    @render.ui
    def box_summary():
        """What the uploaded file holds and whether it fits the grid."""
        info = _box_summary()
        if info is None:
            return ui.p(
                "Files of up to 64 MB are read, about a year of hourly output "
                "of a 29-box model. The file is not kept on the server.",
                class_="text-muted small",
            )
        lines = [
            f"{info.n_months} months, {info.first_month} to {info.last_month} "
            "(monthly means)",
            f"{info.n_boxes} boxes, {info.n_sections} sections",
            "Fields: " + ", ".join(info.variables),
        ]
        if info.missing_zones is None:
            lines.append("Create a grid to place the fields.")
        elif info.fits_grid:
            lines.append(
                f"Every one of the grid's {info.n_patches:,} patches lies in a "
                "box of the file."
            )
        return ui.div(
            *[ui.div(line) for line in lines],
            *[ui.div(problem, class_="text-danger") for problem in info.problems],
            ui.div(
                "The file's first month is taken as the run's first month.",
                class_="text-muted",
            ),
            id="box_summary_text",
            class_="small mb-2",
        )

    @reactive.effect
    def _offer_box_fields():
        """The file's fields and months, for the field map."""
        series = _uploaded_boxes()
        if series is None:
            ui.update_select("box_variable", choices={})
            return
        choices = {name: FIELD_LABELS.get(name, name) for name in series.variables}
        selected = "salt" if "salt" in choices else series.variables[0]
        ui.update_select("box_variable", choices=choices, selected=selected)
        ui.update_slider("box_month", min=1, max=len(series.times), value=1)

    def _box_month_index(series):
        try:
            month = int(input.box_month() or 1)
        except (TypeError, ValueError):
            month = 1
        return min(max(month, 1), len(series.times)) - 1

    @render.ui
    def field_map_note():
        series, info = _uploaded_boxes(), _box_summary()
        if series is None:
            return ui.p(
                "Upload a box model's output under Hydrodynamic Model to see "
                "its fields here.",
                class_="text-muted small",
            )
        if info.missing_zones is None:
            return ui.p("Create a grid to place the fields.", class_="text-muted small")
        if not info.fits_grid:
            return ui.p(info.problems[-1], class_="text-danger small")
        variable = input.box_variable()
        if variable not in series.variables:
            return None
        return ui.p(
            f"{FIELD_LABELS.get(variable, variable)}, mean of "
            f"{month_labels(series)[_box_month_index(series)]}",
            id="field_map_title",
            class_="fw-semibold mb-1",
        )

    @reactive.effect
    async def _push_field_map():
        series, g = _uploaded_boxes(), grid()
        variable = input.box_variable()
        info = _box_summary()
        if (
            series is None
            or g is None
            or info is None
            or not info.fits_grid
            or variable not in series.variables
        ):
            await FIELD_MAP.update(_session, [], widgets=[])
            return
        values = field_on_patches(series, g, variable, _box_month_index(series))
        vmin, vmax = field_range(series, variable)
        label = FIELD_LABELS.get(variable, variable)
        await FIELD_MAP.update(
            _session,
            patch_value_layers(
                "field",
                g,
                values,
                palette=BIOMASS_PALETTE,
                vmin=vmin,
                vmax=vmax,
                value_name=label,
                value_format=".3g",
                labels=g.geometry is not None
                and g.n_patches <= SPATIAL.large_grid_threshold,
            ),
            widgets=[gradient_legend(label, BIOMASS_PALETTE, vmin, vmax, "field")],
        )

    @reactive.calc
    def _transport_choices():
        scenario = _sim_scenario() if _sim_scenario is not None else None
        if scenario is None:
            return []
        try:
            return transport_groups(scenario)
        except Exception as e:
            logger.debug("Could not list the groups for water transport: %s", e)
            return []

    @reactive.effect
    def _offer_transport_groups():
        """The groups water can carry; a new scenario clears the choice."""
        choices = {
            str(c.row): c.name + (" (fast: will be integrated)" if c.fast else "")
            for c in _transport_choices()
            if c.eligible
        }
        ui.update_selectize("transport_groups", choices=choices, selected=[])

    def _transport_rows():
        try:
            return [int(r) for r in (input.transport_groups() or ())]
        except (TypeError, ValueError) as e:
            logger.debug("Transport group selection rejected: %s", e)
            return []

    @render.ui
    def transport_note():
        """What carrying the chosen groups needs and costs."""
        choices = _transport_choices()
        if not choices:
            return ui.p(
                "Create an Ecosim scenario to choose groups.",
                class_="text-muted small",
            )
        left_out = [c for c in choices if not c.eligible]
        rows = _transport_rows()
        by_row = {c.row: c for c in choices}
        fast = [by_row[r].name for r in rows if r in by_row and by_row[r].fast]
        lines = []
        info = _box_summary()
        if rows and info is None:
            lines.append(
                ui.div("Upload the box output: without it nothing is carried.")
            )
        elif rows and not (info.has_flows and info.fits_grid):
            lines.append(
                ui.div(
                    "This file cannot carry groups on this grid (see above).",
                    class_="text-danger",
                )
            )
        if fast:
            lines.append(
                ui.div(
                    ", ".join(fast) + " will be integrated instead of solved at "
                    "equilibrium, at the sub-steps below; the run takes that "
                    "many times as long."
                )
            )
        if left_out:
            lines.append(
                ui.div(
                    "Not offered: "
                    + "; ".join(f"{c.name} ({c.reason})" for c in left_out[:12])
                    + (" and others" if len(left_out) > 12 else "")
                    + ".",
                    class_="text-muted",
                )
            )
        lines.append(
            ui.div(
                "Only water that goes both ways between two boxes carries a "
                "group (exchange); the net flow through the lagoon does not.",
                class_="text-muted",
            )
        )
        return ui.div(*lines, id="transport_note_text", class_="small mb-2")

    @reactive.effect
    @reactive.event(input.create_grid)
    def create_spatial_grid():
        """Create spatial grid based on user configuration."""
        try:
            grid_type = input.grid_type()

            if grid_type == "regular_2d":
                nx = input.grid_nx()
                ny = input.grid_ny()
                # A cleared numeric input arrives as None
                if not nx or not ny or nx < 1 or ny < 1:
                    _notify_legacy(
                        "Enter positive grid dimensions (nx, ny) first",
                        type="warning",
                        duration=4,
                    )
                    return
                nx, ny = int(nx), int(ny)

                # Create regular grid
                new_grid = create_regular_grid(bounds=(0, 0, nx, ny), nx=nx, ny=ny)
                grid.set(new_grid)
                # Results from a previous grid no longer match this geometry
                _spatial_results.set(None)
                _spatial_switching.set(None)

                _notify_legacy(
                    f"Created {nx}×{ny} regular grid ({new_grid.n_patches} patches)",
                    type="message",
                    duration=3,
                )

            elif grid_type == "1d_transect":
                n_patches = input.grid_n_patches()
                if not n_patches or n_patches < 1:
                    _notify_legacy(
                        "Enter a positive number of patches first",
                        type="warning",
                        duration=4,
                    )
                    return
                n_patches = int(n_patches)

                # Create 1D grid
                new_grid = create_1d_grid(n_patches=n_patches, spacing=1.0)
                grid.set(new_grid)
                # Results from a previous grid no longer match this geometry
                _spatial_results.set(None)
                _spatial_switching.set(None)

                _notify_legacy(
                    f"Created 1D transect with {n_patches} patches",
                    type="message",
                    duration=3,
                )

            elif grid_type == "shyfem":
                mesh = _uploaded_mesh()
                if mesh is None:
                    _notify_legacy(
                        "Upload a SHYFEM grid file (.grd) first.",
                        type="warning",
                        duration=4,
                    )
                    return
                patches = input.mesh_patches() or "type"
                new_grid = build_mesh_grid(
                    mesh,
                    patches=patches,
                    units=input.mesh_units() or "auto",
                    epsg=input.mesh_epsg(),
                )
                grid.set(new_grid)
                # Results from a previous grid no longer match this geometry
                _spatial_results.set(None)
                _spatial_switching.set(None)
                what = "elements" if patches == "element" else "element types"
                _notify_legacy(
                    f"Created a grid of {new_grid.n_patches} patches from the "
                    f"mesh's {what} ({new_grid.patch_areas.sum():,.0f} km²)",
                    type="message",
                    duration=4,
                )

            elif grid_type == "custom":
                # Handle custom spatial file upload
                file_info = input.spatial_file_upload()
                if file_info is None or len(file_info) == 0:
                    _notify_legacy(
                        "Please upload a spatial file (shapefile, GeoJSON, or GeoPackage).",
                        type="warning",
                        duration=5,
                    )
                    return

                # Get the uploaded file
                uploaded_file = file_info[0]
                file_path = uploaded_file["datapath"]
                file_name = uploaded_file["name"]

                # Get grid mode
                grid_mode = input.custom_grid_mode()

                try:
                    # Create temporary directory for processing
                    temp_dir = tempfile.mkdtemp()

                    try:
                        # Handle different file types
                        if file_name.endswith(".zip"):
                            # Extract shapefile from zip
                            with zipfile.ZipFile(file_path, "r") as zip_ref:
                                zip_ref.extractall(temp_dir)

                            # Find the .shp file
                            shp_files = list(Path(temp_dir).glob("**/*.shp"))
                            if not shp_files:
                                raise ValueError("No .shp file found in zip archive")

                            spatial_file = str(shp_files[0])

                        elif file_name.endswith((".geojson", ".json", ".gpkg")):
                            # Copy file to temp directory
                            spatial_file = str(Path(temp_dir) / file_name)
                            shutil.copy(file_path, spatial_file)

                        else:
                            raise ValueError(f"Unsupported file format: {file_name}")

                        # Load boundary file first (for visualization and processing)
                        if not _HAS_GIS:
                            raise ImportError(
                                "geopandas is required for spatial file processing"
                            )

                        boundary_gdf = gpd.read_file(spatial_file)
                        if boundary_gdf.crs is None:
                            boundary_gdf = boundary_gdf.set_crs("EPSG:4326")
                        else:
                            boundary_gdf = boundary_gdf.to_crs("EPSG:4326")

                        # Store boundary for visualization
                        boundary_polygon.set(boundary_gdf)

                        # Check grid mode
                        if grid_mode == "use_polygons":
                            # Use uploaded polygons as-is
                            id_field = input.id_field_name() or "id"
                            new_grid = load_spatial_grid(
                                filepath=spatial_file,
                                id_field=id_field,
                                crs="EPSG:4326",  # WGS84
                            )

                            _notify_legacy(
                                f"Loaded irregular grid: {new_grid.n_patches} patches from {file_name}",
                                type="message",
                                duration=4,
                            )

                        elif grid_mode == "create_hexagons":
                            # Create hexagonal grid within boundary
                            hexagon_size = input.hexagon_size_km()

                            # Estimate patch count before generation
                            bounds = boundary_gdf.total_bounds
                            area_degrees = (bounds[2] - bounds[0]) * (
                                bounds[3] - bounds[1]
                            )
                            # Rough conversion: 1 degree at 55°N ≈ 70 km
                            area_km2 = area_degrees * 70 * 70
                            hex_area = 2.598 * (
                                hexagon_size**2
                            )  # Area of regular hexagon
                            estimated_patches = int(area_km2 / hex_area)

                            # Warn if very large grid
                            if estimated_patches > SPATIAL.huge_grid_threshold:
                                _notify_legacy(
                                    f"Warning: Estimated {estimated_patches:,} hexagons! This may take several minutes and cause browser slowdown. "
                                    f"Consider using a larger hexagon size (≥1 km).",
                                    type="warning",
                                    duration=10,
                                )
                            elif estimated_patches > SPATIAL.large_grid_threshold:
                                _notify_legacy(
                                    f"Large grid: Estimated ~{estimated_patches:,} hexagons. Generation may take 30-60 seconds.",
                                    type="warning",
                                    duration=7,
                                )

                            _notify_legacy(
                                f"Generating hexagonal grid ({hexagon_size} km hexagons)...",
                                type="message",
                                duration=3,
                            )

                            # Generate hexagonal grid
                            new_grid = create_hexagonal_grid_in_boundary(
                                boundary_gdf, hexagon_size_km=hexagon_size
                            )

                            if new_grid.n_patches > SPATIAL.large_grid_threshold:
                                _notify_legacy(
                                    f"Created large hexagonal grid: {new_grid.n_patches:,} hexagons. "
                                    f"Map rendering may be slow. Use zoom/pan to explore.",
                                    type="info",
                                    duration=6,
                                )
                            else:
                                _notify_legacy(
                                    f"Created hexagonal grid: {new_grid.n_patches} hexagons within {file_name} boundary",
                                    type="message",
                                    duration=4,
                                )

                        grid.set(new_grid)
                        # Results from a previous grid no longer match this geometry
                        _spatial_results.set(None)
                        _spatial_switching.set(None)

                    finally:
                        # Clean up temporary directory
                        shutil.rmtree(temp_dir, ignore_errors=True)

                except (
                    ValueError,
                    TypeError,
                    OSError,
                    RuntimeError,
                    ImportError,
                    zipfile.BadZipFile,
                ) as e:
                    logger.debug("Spatial file processing failed: %s", e)
                    _notify_legacy(
                        f"Error processing spatial file: {e!s}",
                        type="error",
                        duration=6,
                    )
                    return

        except (ValueError, TypeError, OSError, RuntimeError, ImportError) as e:
            logger.debug("Grid creation failed: %s", e)
            _notify_legacy(f"Error creating grid: {e!s}", type="error", duration=5)

    def _compute_habitat_vector(g):
        """Compute habitat preference vector for all patches from current UI state.

        Returns a 1D array of habitat quality values [n_patches].
        """
        n_patches = g.n_patches
        pattern = input.habitat_pattern()

        if pattern == "uniform":
            return np.ones(n_patches) * 0.8
        elif pattern == "gradient":
            direction = input.gradient_direction()
            centroids = g.patch_centroids
            if direction == "horizontal":
                span = centroids[:, 0].max() - centroids[:, 0].min()
                if span > 0:
                    return (centroids[:, 0] - centroids[:, 0].min()) / span
                return np.ones(n_patches) * 0.5
            elif direction == "vertical":
                span = centroids[:, 1].max() - centroids[:, 1].min()
                if span > 0:
                    return (centroids[:, 1] - centroids[:, 1].min()) / span
                return np.ones(n_patches) * 0.5
            else:  # radial
                center = centroids.mean(axis=0)
                distances = np.linalg.norm(centroids - center, axis=1)
                max_dist = distances.max()
                if max_dist > 0:
                    return 1 - (distances / max_dist)
                return np.ones(n_patches) * 0.5
        elif pattern == "patchy":
            # Local generator: np.random.seed() would reseed the process-global
            # RNG shared with other sessions and with the IBM/dispersal code.
            return np.random.default_rng(42).uniform(0.2, 1.0, n_patches)
        elif pattern == "core_periphery":
            center = g.patch_centroids.mean(axis=0)
            distances = np.linalg.norm(g.patch_centroids - center, axis=1)
            max_dist = distances.max()
            if max_dist > 0:
                return 1 - (distances / max_dist) ** 2
            return np.ones(n_patches) * 0.5
        elif pattern == "custom":
            values = _read_habitat_csv(n_patches)
            if values is not None:
                return values
            return np.ones(n_patches) * 0.5
        else:
            return np.ones(n_patches) * 0.5

    def _read_habitat_csv(n_patches):
        """Read the uploaded habitat CSV: one value per patch, in patch order.

        Returns None (and warns) when nothing usable was uploaded, so callers
        fall back to a uniform habitat.
        """
        file_info = input.habitat_upload()
        if not file_info:
            _notify_legacy(
                "Select 'Custom (upload CSV)' and upload a habitat file, "
                "or choose another habitat pattern.",
                type="warning",
                duration=5,
            )
            return None
        try:
            values = parse_habitat_csv(file_info[0]["datapath"], n_patches)
        except (OSError, ValueError, pd.errors.ParserError) as e:
            logger.debug("Habitat CSV rejected: %s", e)
            _notify_legacy(
                f"Habitat file not usable: {e!s}", type="warning", duration=6
            )
            return None
        return values

    @reactive.effect
    def _populate_target_groups():
        """Offer the model's groups as gravity targets.

        Keys are Ecosim indices: the biomass array the allocator sees is
        1-based with index 0 = "Outside", so Ecopath group g is at g + 1.
        """
        choices = {}
        scenario = _sim_scenario() if _sim_scenario is not None else None
        spname = getattr(getattr(scenario, "params", None), "spname", None)
        if spname is not None:
            choices = {str(i): str(name) for i, name in enumerate(spname) if i > 0}
        else:
            model = _model_data()
            names = getattr(model, "Group", None)
            n_living = getattr(model, "NUM_LIVING", 0)
            n_dead = getattr(model, "NUM_DEAD", 0)
            if names is not None:
                choices = {
                    str(i + 1): str(names[i]) for i in range(int(n_living + n_dead))
                }
        if choices:
            ui.update_selectize("fishing_target_groups", choices=choices)

    def _selected_target_groups():
        """Ecosim indices of the chosen target groups, or None for all."""
        try:
            selected = input.fishing_target_groups()
        except Exception as e:  # control not yet rendered
            logger.debug("Target group selection unavailable: %s", e)
            return None
        if not selected:
            return None
        try:
            return [int(i) for i in selected]
        except (TypeError, ValueError) as e:
            logger.debug("Target group selection rejected: %s", e)
            return None

    def _build_spatial_fishing():
        """Build a SpatialFishing object from the Spatial Fishing controls.

        Returns None when the allocation is uniform, which is exactly the
        no-spatial-fishing case, so the simulation skips the extra work.
        """
        method = input.fishing_allocation()
        if method == "uniform":
            return None

        if method == "gravity":
            return SpatialFishing(
                allocation_type="gravity",
                gravity_alpha=float(input.gravity_alpha()),
                gravity_beta=0.0,
                target_groups=_selected_target_groups(),
            )

        if method == "port":
            ports = _parse_port_patches()
            if ports is None:
                return None
            return SpatialFishing(
                allocation_type="port",
                port_patches=ports,
                gravity_beta=float(input.port_beta()),
            )

        if method == "habitat":
            return SpatialFishing(allocation_type="habitat")

        return None

    def _parse_port_patches():
        """Parse the comma-separated port patch indices, or warn and return None."""
        g = grid()
        try:
            ports = np.array(
                [int(x.strip()) for x in input.port_patches().split(",") if x.strip()]
            )
        except (ValueError, AttributeError) as e:
            logger.debug("Port patch list rejected: %s", e)
            _notify_legacy(
                "Port patch indices must be a comma-separated list of whole "
                "numbers; using uniform effort.",
                type="warning",
                duration=6,
            )
            return None
        if len(ports) == 0 or (g is not None and (ports >= g.n_patches).any()):
            _notify_legacy(
                f"Port patch indices must be between 0 and "
                f"{(g.n_patches - 1) if g else 0}; using uniform effort.",
                type="warning",
                duration=6,
            )
            return None
        return ports

    # This session's spatial run in the process-wide executor. A run a worker
    # has started stops only between months, so after a cancel the slot stays
    # taken until the month in progress ends; a click is refused meanwhile.
    run_slot = RunSlot()
    # The live run's RunControl: the worker records each month on it and the
    # loop reads the progress and asks for a stop through it.
    launched = {"control": None}

    @ui.bind_task_button(button_id="run_spatial_sim")
    @reactive.extended_task
    async def _spatial_task(
        run_id,
        source,
        source_grid,
        scen,
        ecospace_params,
        spatial_fishing,
        used,
        slot,
        control,
        fingerprint,
        check_substeps=False,
        notes=(),
    ):
        """Integrate on a worker thread, off the event loop.

        ``scen`` is a deep copy of ``source``, the page's scenario object. The
        run goes to the bounded executor shared with the Ecosim page. The
        worker reads no reactive value and shows nothing: the launch effect
        gathers everything first and ``_apply_spatial_result`` lands the
        outcome, both on the loop.
        """
        future = RUN_EXECUTOR.submit(
            run_spatial_checked,
            scen,
            ecospace_params,
            spatial_fishing,
            control=control,
            # worker processes, where asked for (PYPATH_ECOSPACE_JOBS)
            n_jobs=spatial_jobs(ecospace_params.grid.n_patches),
            check_substeps=check_substeps,
        )
        slot.acquire(future)
        out, substeps = await asyncio.wrap_future(future)
        return SpatialOutcome(
            run_id, source, source_grid, out, used, fingerprint, substeps, notes
        )

    def _stop_spatial_run() -> None:
        """Stop the session's spatial run, on the loop.

        The control's stop ends a computing run at its next month boundary
        (its future then completes with ``RunCancelled``, which frees the
        slot); the task's cancel drops the outcome, and cancels a run still
        queued for a worker outright.
        """
        control = launched["control"]
        if control is not None:
            control.request_stop()
        _spatial_task.cancel()

    # A closed or reloaded tab must not leave its run queued for a worker with
    # no owner; a run already computing stops at its next month boundary and
    # its result is dropped.
    _session.on_ended(_stop_spatial_run)

    # Run spatial simulation handler
    @reactive.effect
    @reactive.event(input.run_spatial_sim)
    def _run_spatial_simulation():
        """Prepare a spatial run and start it on a worker (``_spatial_task``)."""
        # invoke() would queue behind a running task. Refuse instead, before
        # any state change, so the live run keeps its run_id.
        if (
            spatial_task_state.get().status is TaskStatus.RUNNING
            or _spatial_task.status() == "running"
        ):
            _notify_legacy(
                "A spatial simulation is already running. Cancel it first, or "
                "wait for it.",
                type="warning",
                duration=5,
            )
            return
        if run_slot.busy:
            # A cancelled run's thread computes on to its next month boundary.
            _notify_legacy(
                "The cancelled run is finishing its current month. "
                "Try again in a moment.",
                type="warning",
                duration=5,
            )
            return

        # Guard: need both scenario and grid
        if _sim_scenario is None or _sim_scenario() is None:
            msg = "Please create an Ecosim scenario first (Ecosim Simulation tab)."
            _notify_legacy(
                msg,
                type="warning",
                duration=5,
            )
            spatial_task_state.set(
                TaskState(
                    run_id="spatial-idle",
                    status=TaskStatus.ERROR,
                    message="Cannot start spatial simulation",
                    error=msg,
                )
            )
            return

        current_grid = grid()
        if current_grid is None:
            msg = "Please create a spatial grid first."
            _notify_legacy(
                msg,
                type="warning",
                duration=5,
            )
            spatial_task_state.set(
                TaskState(
                    run_id="spatial-idle",
                    status=TaskStatus.ERROR,
                    message="Cannot start spatial simulation",
                    error=msg,
                )
            )
            return

        run_id = f"spatial-run-{input.run_spatial_sim()}"
        spatial_task_state.set(
            new_running_task(run_id, "Preparing spatial configuration")
        )

        def _update_latest(updater):
            spatial_task_state.set(
                apply_if_latest(
                    spatial_task_state.get(), run_id=run_id, updater=updater
                )
            )

        try:
            scenario = _sim_scenario()
            used = summary(scenario.params)
            n_groups = scenario.params.NUM_GROUPS

            _update_latest(
                lambda s: update_task_progress(
                    s,
                    message="Building habitat and movement matrices",
                    progress=25.0,
                )
            )

            ecospace_params, config_note = run_ecospace_params(
                current_grid,
                n_groups,
                page_ecospace_arrays(
                    _compute_habitat_vector(current_grid),
                    n_groups,
                    dispersal_rate=input.dispersal_rate_default(),
                    advection=input.enable_advection(),
                    gravity=input.gravity_strength(),
                ),
                wizard_params=_ecospace_params.get(),
                origin=_config_origin.get(),
            )
            if config_note:
                _notify_legacy(config_note, type="message", duration=6)

            _notify_legacy("Running spatial simulation...", type="message", duration=2)
            _update_latest(
                lambda s: update_task_progress(
                    s,
                    message="Running spatial integration",
                    progress=70.0,
                )
            )

            spatial_fishing = _build_spatial_fishing()

            # Water transport from the uploaded box output, for the groups
            # chosen: the fast ones among them are integrated in the run's
            # copy of the scenario.
            run_scenario, notes, check_substeps = scenario, [], False
            rows = _transport_rows()
            if rows:
                series, info = _uploaded_boxes(), _box_summary()
                if series is None:
                    raise ValueError(
                        "groups are chosen for water transport but no box "
                        "output is uploaded (Hydrodynamic Model)"
                    )
                if not (info.has_flows and info.fits_grid):
                    raise ValueError(
                        "the uploaded box output cannot carry groups on this "
                        "grid: " + " ".join(info.problems)
                    )
                choices = _transport_choices()
                substeps = int(input.transport_substeps() or DEFAULT_SUBSTEPS)
                run_scenario, integrated = transport_scenario(
                    scenario, choices, rows, substeps=substeps
                )
                transport = build_transport(
                    series, rows, beyond=input.transport_beyond() or "repeat_year"
                )
                ecospace_params = dataclasses.replace(
                    ecospace_params, water_transport=transport
                )
                names = {c.row: c.name for c in choices}
                notes.append("Water carried " + ", ".join(names[r] for r in rows) + ".")
                if integrated:
                    notes.append(
                        ", ".join(integrated)
                        + f" integrated at {substeps} sub-steps a month."
                    )
                # the check is about the integrated groups: without one
                # the sub-steps are the scenario's own and nothing is asked
                check_substeps = bool(integrated) and bool(input.transport_check())

            # The worker integrates a copy: the page's scenario stays usable
            # (and editable) while the run computes.
            control = RunControl()
            launched["control"] = control
            _spatial_task.invoke(
                run_id,
                scenario,
                current_grid,
                copy.deepcopy(run_scenario),
                ecospace_params,
                spatial_fishing=spatial_fishing,
                used=used,
                slot=run_slot,
                control=control,
                fingerprint=scenario_fingerprint(scenario),
                check_substeps=check_substeps,
                notes=tuple(notes),
            )

        except Exception as e:
            logger.error("Spatial simulation could not start: %s", e, exc_info=True)
            _fail_spatial_run(run_id, f"Spatial simulation failed: {e!s}")

    def _finish_spatial_run(run_id: str, **finish) -> None:
        """Finish the running task, if it is still ``run_id``."""
        spatial_task_state.set(
            apply_if_latest(
                spatial_task_state.get(),
                run_id=run_id,
                updater=lambda s: finish_task(s, **finish),
            )
        )

    def _fail_spatial_run(run_id: str, err: str) -> None:
        _notify_legacy(err, type="error", duration=8)
        _finish_spatial_run(
            run_id, success=False, message="Spatial simulation failed", error=err
        )

    def _land_spatial_result(outcome: SpatialOutcome) -> None:
        """Put a finished run on the page."""
        result = outcome.output
        scenario = outcome.source
        n_groups = scenario.params.NUM_GROUPS
        _spatial_results.set(result)
        _spatial_switching.set(outcome.switching)
        lines = [("text-muted", note) for note in outcome.notes]
        if outcome.substeps is not None:
            level = outcome.substeps.level
            css = {"passed": "text-success", "moderate": "text-muted"}.get(
                level, "text-danger"
            )
            lines.append((css, outcome.substeps.message))
            if level == "failed":
                _notify_legacy(outcome.substeps.message, type="warning", duration=12)
        _spatial_notes.set(tuple(lines))

        # Post-run: update biomass group dropdown with group names
        spname = getattr(scenario.params, "spname", None)
        if spname is not None:
            group_choices = {
                str(i): name for i, name in enumerate(spname) if i > 0 and i <= n_groups
            }
        else:
            group_choices = {str(i): f"Group {i}" for i in range(1, n_groups + 1)}
        ui.update_select("biomass_view_group", choices=group_choices)

        # Update animation slider max to number of timesteps
        n_timesteps = result.out_Biomass_spatial.shape[0] - 1
        ui.update_slider("animation_time", max=n_timesteps, value=0)

        _notify_legacy("Spatial simulation complete!", type="message", duration=3)
        if scenario_changed_since(outcome, scenario):
            # The page's scenario is edited in place (the Ecosim page's forcing
            # card, for one), so the same object can hold other values now.
            _notify_legacy(
                "The Ecosim scenario's forcing, fishing or parameters changed "
                "after this run started; the result shows the settings at the "
                "start.",
                type="warning",
                duration=10,
            )
        _finish_spatial_run(
            outcome.run_id,
            success=True,
            message=f"Spatial simulation complete ({n_timesteps + 1} time steps).",
        )

    @reactive.effect
    def _apply_spatial_result():
        """Land the task's outcome. Depends on the task's status only."""
        status = _spatial_task.status()
        # Everything else is read isolated, so this effect's own state
        # changes do not re-trigger it.
        with reactive.isolate():
            state = spatial_task_state.get()
            if status == "success":
                outcome = _spatial_task.result()
                current_scenario = (
                    _sim_scenario() if _sim_scenario is not None else None
                )
                if spatial_result_is_current(state, outcome, current_scenario, grid()):
                    try:
                        _land_spatial_result(outcome)
                    except Exception as e:
                        logger.error(
                            "Spatial result could not be shown: %s", e, exc_info=True
                        )
                        _fail_spatial_run(
                            outcome.run_id, f"Spatial simulation failed: {e!s}"
                        )
                elif (
                    state.status is TaskStatus.RUNNING
                    and state.run_id == outcome.run_id
                ):
                    # The page was not frozen during the run, and its grid or
                    # scenario was replaced: this result belongs to neither.
                    msg = (
                        "The grid or the Ecosim scenario changed during the run, "
                        "so its result was discarded. Run the simulation again."
                    )
                    _notify_legacy(msg, type="warning", duration=8)
                    _finish_spatial_run(
                        outcome.run_id,
                        success=False,
                        message="Spatial result discarded",
                        error=msg,
                    )
                else:
                    logger.debug(
                        "Dropped a stale spatial result (run %s)", outcome.run_id
                    )
            elif status == "error":
                if state.status is not TaskStatus.RUNNING:
                    logger.debug("Dropped a stale spatial error (state %s)", state)
                    return
                try:
                    _spatial_task.result()
                except RunCancelled as e:
                    # Defensive: _stop_spatial_run cancels the task before the
                    # worker reaches its month boundary, so the task ends
                    # "cancelled" and does not land here. If a stopped run's
                    # outcome ever does, it is a cancel, never an error.
                    logger.debug("Spatial run %s stopped: %s", state.run_id, e)
                    spatial_task_state.set(_cancelled_state(state))
                except Exception as e:
                    logger.error("Spatial simulation failed: %s", e, exc_info=True)
                    _fail_spatial_run(state.run_id, f"Spatial simulation failed: {e!s}")
            # "cancelled": whoever cancelled has already set the state.

    def _cancelled_state(state: TaskState) -> TaskState:
        return transition_task_state(
            state,
            TaskStatus.CANCELLED,
            message="Spatial simulation cancelled.",
            can_cancel=False,
        )

    @reactive.effect
    @reactive.event(input.btn_cancel_spatial)
    def _cancel_spatial_simulation():
        """Cancel the running spatial simulation; its result is discarded."""
        state = spatial_task_state.get()
        if state.status is not TaskStatus.RUNNING:
            return
        # A run already computing stops at its next month boundary (a month
        # in progress is not interrupted) and holds the session's slot until
        # then; a run still queued for a worker is cancelled outright.
        _stop_spatial_run()
        spatial_task_state.set(_cancelled_state(state))
        _notify_legacy(
            "Cancelled: the run stops at the end of its current month and "
            "its result is discarded.",
            type="warning",
            duration=6,
        )

    @render.ui
    def spatial_month_progress():
        """The running run's ``Month m of n (p%)`` bar, ticking every second.

        Reads the run's ``RunControl`` here, on the loop (the worker only
        writes it). Empty until the first month completes: the status line
        above it says what the run is doing until then.
        """
        state = spatial_task_state.get()
        if state.status is not TaskStatus.RUNNING:
            return None
        reactive.invalidate_later(1)
        control = launched["control"]
        progress = control.progress() if control is not None else None
        if progress is None:
            return None
        return month_progress_tags(progress, state.message)

    @render.ui
    def spatial_switching_line():
        """The prey switching the shown spatial run used."""
        if _spatial_results.get() is None:
            return None
        return ui.p(
            summary_line(_spatial_switching.get()),
            class_="text-muted small",
        )

    @render.ui
    def spatial_transport_line():
        """What the shown run did with the water transport, and its check."""
        if _spatial_results.get() is None:
            return None
        lines = _spatial_notes.get()
        if not lines:
            return None
        return ui.div(
            *[ui.div(text, class_=css) for css, text in lines],
            id="spatial_transport_text",
            class_="small",
        )

    @render.ui
    def spatial_run_status():
        """Display live status for spatial simulation runs."""
        task_state = spatial_task_state.get()
        alert_class, icon_class, message = spatial_task_alert(task_state)

        if not message:
            return ui.div()

        alert = ui.div(
            ui.tags.i(class_=icon_class),
            message,
            class_=alert_class,
        )
        if task_state.status is not TaskStatus.RUNNING:
            return alert
        # The month progress ticks in its own output, so this block and its
        # Cancel button are not rebuilt every second.
        return ui.div(
            alert,
            ui.output_ui("spatial_month_progress"),
            ui.input_action_button(
                "btn_cancel_spatial",
                ui.span(
                    ui.tags.i(class_="bi bi-stop-circle me-2"),
                    "Cancel Simulation",
                ),
                class_="btn btn-warning w-100 mt-2",
            )
            if task_state.can_cancel
            else None,
        )

    # Grid visualization (deck.gl)
    @render.ui
    def grid_plot_note():
        """A hint above the grid map while there is nothing to draw."""
        if grid() is None and boundary_polygon() is None:
            return ui.p(
                "No grid or boundary loaded. Upload a file or create a grid.",
                class_="text-muted",
            )
        return None

    def _boundary_geojson():
        b = boundary_polygon()
        return b.__geo_interface__ if b is not None else None

    @reactive.effect
    async def _push_grid_map():
        await GRID_MAP.update(
            _session, grid_overview_layers(grid(), _boundary_geojson())
        )

    @reactive.effect
    async def _fit_maps_to_grid():
        """Zoom every map to a new grid or boundary (not on every redraw)."""
        b = boundary_polygon()
        bounds = grid_bounds(grid(), b.total_bounds if b is not None else None)
        if bounds is None:
            return
        for widget in _MAPS:
            await widget.fit_bounds(
                _session, bounds, padding=MAP_FIT_PADDING, max_zoom=12
            )

    # Grid info text
    @render.text
    def grid_info():
        """Display grid and boundary information."""
        has_grid = grid() is not None
        has_boundary = boundary_polygon() is not None

        if not has_grid and not has_boundary:
            return "No grid or boundary loaded. Upload a file or create a grid."

        info_lines = []

        # Boundary information
        if has_boundary:
            boundary_gdf = boundary_polygon()
            n_features = len(boundary_gdf)

            # Calculate total boundary area
            boundary_gdf_utm = boundary_gdf.to_crs(boundary_gdf.estimate_utm_crs())
            boundary_area_km2 = boundary_gdf_utm.geometry.area.sum() / 1e6

            info_lines.append("Boundary Information:")
            info_lines.append(f"  • Features: {n_features}")
            info_lines.append(f"  • Total area: {boundary_area_km2:.2f} km²")

            # Get bounds
            bounds = boundary_gdf.total_bounds
            info_lines.append(
                f"  • Extent: {bounds[2] - bounds[0]:.3f}° × {bounds[3] - bounds[1]:.3f}°"
            )

        # Grid information
        if has_grid:
            g = grid()

            if has_boundary:
                info_lines.append("")  # Add spacing

            # Count connections
            n_edges = g.adjacency_matrix.nnz // 2  # Divide by 2 for undirected
            avg_neighbors = n_edges * 2 / g.n_patches

            info_lines.append("Grid Configuration:")
            info_lines.append(f"  • Patches: {g.n_patches}")
            info_lines.append(f"  • Connections: {n_edges}")
            info_lines.append(f"  • Average neighbors: {avg_neighbors:.1f}")
            info_lines.append(f"  • Total area: {g.patch_areas.sum():.2f} km²")

        return "\n".join(info_lines)

    # Habitat visualization (deck.gl)
    @reactive.effect
    async def _push_habitat_map():
        g = grid()
        if g is None:
            await HABITAT_MAP.update(_session, [], widgets=[])
            return
        wizard = _ecospace_params()
        n_groups = _scenario_n_groups()
        habitat = habitat_for_display(wizard, g, _compute_habitat_vector(g), n_groups)
        active = wizard_params_on_grid(wizard, g, n_groups)
        if active is not None and _config_origin() != WIZARD_ORIGIN:
            # a stored scenario carries its pattern in the capacity
            habitat = np.asarray(active.habitat_capacity, dtype=float).mean(axis=0)
            pattern = "stored scenario's capacity, mean over groups"
        elif active is not None:
            pattern = "Ecospace Wizard, mean over groups"
        else:
            pattern = input.habitat_pattern().replace("_", " ")
        await HABITAT_MAP.update(
            _session,
            patch_value_layers(
                "habitat",
                g,
                habitat,
                palette=HABITAT_PALETTE,
                vmin=0.0,
                vmax=1.0,
                value_name="Habitat quality",
                value_format=".2f",
                labels=g.geometry is not None
                and g.n_patches <= SPATIAL.large_grid_threshold,
                outline=[0, 100, 0, 220],
            ),
            widgets=[
                gradient_legend(
                    f"Habitat quality ({pattern})",
                    HABITAT_PALETTE,
                    0.0,
                    1.0,
                    "habitat",
                )
            ],
        )

    # Fishing effort visualization (deck.gl)
    @reactive.effect
    async def _push_effort_map():
        """Preview the spatial fishing effort allocation."""
        g = grid()
        if g is None:
            await EFFORT_MAP.update(_session, [], widgets=[])
            return
        n_patches = g.n_patches

        allocation_method = input.fishing_allocation()
        total_effort = 100.0

        # Preview the allocation the simulation would use. Gravity depends on
        # biomass, so use the latest spatial result when there is one; before a
        # run, the Ecopath biomass spread evenly over patches is the best guess.
        biomass = _preview_biomass(n_patches)

        if allocation_method == "gravity":
            effort = allocate_gravity(
                biomass,
                target_groups=_selected_target_groups(),
                total_effort=total_effort,
                alpha=float(input.gravity_alpha()),
                beta=0,
            )
        elif allocation_method == "port":
            ports = _parse_port_patches()
            if ports is None:
                effort = allocate_uniform(n_patches, total_effort)
            else:
                effort = allocate_port_based(
                    g, ports, total_effort, beta=float(input.port_beta())
                )
        elif allocation_method == "habitat":
            effort = allocate_habitat_based(
                habitat_for_display(
                    _ecospace_params(),
                    g,
                    _compute_habitat_vector(g),
                    _scenario_n_groups(),
                ),
                total_effort=total_effort,
            )
        else:
            effort = allocate_uniform(n_patches, total_effort)

        effort = np.asarray(effort, dtype=float)
        peak = float(effort.max()) if effort.size else 0.0
        await EFFORT_MAP.update(
            _session,
            effort_layers("effort", g, effort),
            widgets=[
                gradient_legend(
                    f"Fishing effort ({allocation_method})",
                    EFFORT_PALETTE,
                    0.0,
                    peak,
                    "effort",
                )
            ],
        )

    def _preview_biomass(n_patches):
        """Per-patch biomass for the effort preview [n_groups + 1, n_patches]."""
        results = _spatial_results()
        spatial = getattr(results, "out_Biomass_spatial", None) if results else None
        if spatial is not None and spatial.shape[-1] == n_patches:
            return np.asarray(spatial[-1], dtype=float)

        model = _model_data()
        biomass = getattr(model, "Biomass", None)
        if biomass is None:
            return np.ones((2, n_patches))
        # Ecopath biomass is 0-indexed with no "Outside" slot, while the
        # allocation helpers expect the Ecosim layout where row 0 is Outside.
        # Prepend a zero row so group 0 is not silently dropped.
        per_patch = np.asarray(biomass, dtype=float) / max(n_patches, 1)
        tiled = np.tile(per_patch[:, None], (1, n_patches))
        return np.vstack([np.zeros((1, n_patches)), tiled])

    # Biomass animation visualization (deck.gl)
    @render.ui
    def biomass_animation_ui():
        """A hint above the biomass map until there is a spatial run."""
        results = _spatial_results()
        if results is None or not hasattr(results, "out_Biomass_spatial"):
            return ui.div(
                ui.tags.i(class_="bi bi-info-circle me-2"),
                "Run a spatial simulation to view biomass dynamics over time.",
                class_="alert alert-info",
            )
        return None

    @reactive.effect
    async def _push_biomass_map():
        """Per-patch biomass of the selected group at the selected time step."""
        results = _spatial_results()
        g = grid()
        if (
            results is None
            or not hasattr(results, "out_Biomass_spatial")
            or g is None
            or results.out_Biomass_spatial.shape[-1] != g.n_patches
        ):
            await BIOMASS_MAP.update(_session, [], widgets=[])
            return

        # [n_months, n_groups+1, n_patches]
        biomass_spatial = results.out_Biomass_spatial

        group_str = input.biomass_view_group()
        group_idx = int(group_str) if group_str else 1

        timestep = int(input.animation_time())
        timestep = min(timestep, biomass_spatial.shape[0] - 1)

        # Fixed colour range over the whole run, so time steps compare
        vmax = max(float(biomass_spatial[:, group_idx, :].max()), 1e-10)

        scenario = _sim_scenario() if _sim_scenario is not None else None
        if scenario and hasattr(scenario.params, "spname"):
            group_name = scenario.params.spname[group_idx]
        else:
            group_name = f"Group {group_idx}"

        await BIOMASS_MAP.update(
            _session,
            patch_value_layers(
                "biomass",
                g,
                biomass_spatial[timestep, group_idx, :],
                palette=BIOMASS_PALETTE,
                vmin=0.0,
                vmax=vmax,
                value_name=f"{group_name} biomass (t={timestep})",
            ),
            widgets=[
                gradient_legend(
                    f"{group_name} biomass", BIOMASS_PALETTE, 0.0, vmax, "biomass"
                )
            ],
        )

    # Spatial metrics table
    @render.ui
    @as_cell_table()
    def spatial_metrics_table():
        """Display spatial metrics from grid and simulation results."""
        g = grid()
        results = _spatial_results()

        if g is None and results is None:
            return pd.DataFrame(
                {
                    "Metric": [
                        "Total Patches",
                        "Occupied Patches",
                        "Center of Biomass (X)",
                        "Center of Biomass (Y)",
                        "Spatial Variance",
                    ],
                    "Value": ["N/A - Create a grid first", "N/A", "N/A", "N/A", "N/A"],
                }
            )

        metrics = []
        values = []

        if g is not None:
            metrics.append("Total Patches")
            values.append(str(g.n_patches))

            n_edges = g.adjacency_matrix.nnz // 2
            avg_neighbors = n_edges * 2 / g.n_patches if g.n_patches > 0 else 0
            metrics.append("Avg Neighbors")
            values.append(f"{avg_neighbors:.1f}")

            metrics.append("Total Area (km²)")
            values.append(f"{g.patch_areas.sum():.2f}")

        if results is not None and hasattr(results, "out_Biomass_spatial"):
            biomass_spatial = results.out_Biomass_spatial

            # Use animation slider timestep if available, else final timestep
            try:
                timestep = int(input.animation_time())
                timestep = min(timestep, biomass_spatial.shape[0] - 1)
            except Exception as e:
                logger.debug("Failed to parse animation timestep: %s", e)
                timestep = biomass_spatial.shape[0] - 1

            # Sum biomass across all groups for each patch
            total_per_patch = biomass_spatial[timestep, 1:, :].sum(axis=0)

            threshold = total_per_patch.max() * 0.01 if total_per_patch.max() > 0 else 0
            occupied = int(np.sum(total_per_patch > threshold))
            metrics.append("Occupied Patches")
            values.append(str(occupied))

            if g is not None and total_per_patch.sum() > 0:
                weights = total_per_patch / total_per_patch.sum()
                center_x = np.dot(weights, g.patch_centroids[:, 0])
                center_y = np.dot(weights, g.patch_centroids[:, 1])
                metrics.append("Center of Biomass (X)")
                values.append(f"{center_x:.4f}")
                metrics.append("Center of Biomass (Y)")
                values.append(f"{center_y:.4f}")

            spatial_var = np.var(total_per_patch)
            metrics.append("Spatial Variance")
            values.append(f"{spatial_var:.4f}")

            metrics.append("Timestep")
            values.append(str(timestep))

        return pd.DataFrame({"Metric": metrics, "Value": values})
