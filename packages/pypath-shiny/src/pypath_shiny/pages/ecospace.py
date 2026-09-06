"""
ECOSPACE Spatial Modeling Page

Interactive spatial ecosystem modeling with:
- Grid configuration (regular or irregular polygons)
- Habitat preferences and capacity
- Dispersal and movement parameters
- Spatial fishing effort allocation
- Spatial simulation and visualization
"""

import logging
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

# pypath imports (path setup handled by app/__init__.py)
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
from shiny import Inputs, Outputs, Session, reactive, render, req, ui

# Import centralized configuration
from pypath_shiny.config import PARAM_RANGES, SPATIAL

try:
    import geopandas as gpd
    from shapely.geometry import Polygon

    _HAS_GIS = True
except ImportError:
    _HAS_GIS = False
    gpd = None
    Polygon = None

logger = logging.getLogger(__name__)


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


def ecospace_ui():
    """UI for ECOSPACE spatial modeling page."""
    return ui.page_fluid(
        ui.layout_sidebar(
            ui.sidebar(
                ui.h4("ECOSPACE Configuration", class_="mb-3"),
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
                            "Default Dispersal Rate (km²/month)",
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
                            "Note: Dispersal rates can be set per-group in advanced settings.",
                            class_="text-muted small",
                        ),
                        icon=ui.tags.i(class_="bi bi-arrows-move"),
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
                # Run simulation button
                ui.input_action_button(
                    "run_spatial_sim",
                    ui.tags.span(
                        ui.tags.i(class_="bi bi-play-fill me-2"),
                        "Run Spatial Simulation",
                    ),
                    class_="btn btn-success w-100 mt-3",
                    disabled=True,
                ),
                width=350,
            ),
            # Main panel with tabs
            ui.navset_card_tab(
                ui.nav_panel(
                    "Grid Visualization",
                    ui.output_ui("grid_plot"),
                    ui.div(ui.output_text("grid_info"), class_="alert alert-info mt-3"),
                ),
                ui.nav_panel(
                    "Habitat Map",
                    ui.output_plot("habitat_plot", height="500px"),
                ),
                ui.nav_panel(
                    "Fishing Effort",
                    ui.output_plot("fishing_effort_plot", height="500px"),
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
                    ui.output_table("spatial_metrics_table"),
                    ui.p(
                        "Summary statistics for spatial distribution of biomass.",
                        class_="text-muted small mt-2",
                    ),
                ),
                id="ecospace_tabs",
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
):
    """Server logic for ECOSPACE page."""

    # Reactive values for spatial state
    grid = reactive.Value(None)
    boundary_polygon = reactive.Value(None)  # Store uploaded boundary for visualization
    _ecospace_params = reactive.Value(None)
    _spatial_results = reactive.Value(None)

    @reactive.effect
    def _adopt_wizard_params():
        """Pick up an EcospaceParams built by the Ecospace Wizard."""
        if wizard_params is None:
            return
        params_obj = wizard_params.get()
        if params_obj is None:
            return
        _ecospace_params.set(params_obj)
        wizard_grid = getattr(params_obj, "grid", None)
        if wizard_grid is not None:
            grid.set(wizard_grid)
            _spatial_results.set(None)
            ui.notification_show(
                f"Loaded wizard configuration: {wizard_grid.n_patches} patches",
                type="message",
                duration=5,
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

                ui.notification_show(
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
            ui.notification_show(
                f"Error loading boundary: {e!s}", type="warning", duration=5
            )
            boundary_polygon.set(None)

    # Create grid based on configuration
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
                    ui.notification_show(
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

                ui.notification_show(
                    f"Created {nx}×{ny} regular grid ({new_grid.n_patches} patches)",
                    type="message",
                    duration=3,
                )

            elif grid_type == "1d_transect":
                n_patches = input.grid_n_patches()
                if not n_patches or n_patches < 1:
                    ui.notification_show(
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

                ui.notification_show(
                    f"Created 1D transect with {n_patches} patches",
                    type="message",
                    duration=3,
                )

            elif grid_type == "custom":
                # Handle custom spatial file upload
                file_info = input.spatial_file_upload()
                if file_info is None or len(file_info) == 0:
                    ui.notification_show(
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

                            ui.notification_show(
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
                                ui.notification_show(
                                    f"Warning: Estimated {estimated_patches:,} hexagons! This may take several minutes and cause browser slowdown. "
                                    f"Consider using a larger hexagon size (≥1 km).",
                                    type="warning",
                                    duration=10,
                                )
                            elif estimated_patches > SPATIAL.large_grid_threshold:
                                ui.notification_show(
                                    f"Large grid: Estimated ~{estimated_patches:,} hexagons. Generation may take 30-60 seconds.",
                                    type="warning",
                                    duration=7,
                                )

                            ui.notification_show(
                                f"Generating hexagonal grid ({hexagon_size} km hexagons)...",
                                type="message",
                                duration=3,
                            )

                            # Generate hexagonal grid
                            new_grid = create_hexagonal_grid_in_boundary(
                                boundary_gdf, hexagon_size_km=hexagon_size
                            )

                            if new_grid.n_patches > SPATIAL.large_grid_threshold:
                                ui.notification_show(
                                    f"Created large hexagonal grid: {new_grid.n_patches:,} hexagons. "
                                    f"Map rendering may be slow. Use zoom/pan to explore.",
                                    type="info",
                                    duration=6,
                                )
                            else:
                                ui.notification_show(
                                    f"Created hexagonal grid: {new_grid.n_patches} hexagons within {file_name} boundary",
                                    type="message",
                                    duration=4,
                                )

                        grid.set(new_grid)
                        # Results from a previous grid no longer match this geometry
                        _spatial_results.set(None)

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
                    ui.notification_show(
                        f"Error processing spatial file: {e!s}",
                        type="error",
                        duration=6,
                    )
                    return

            # Enable run button
            ui.update_action_button("run_spatial_sim", disabled=False)

        except (ValueError, TypeError, OSError, RuntimeError, ImportError) as e:
            logger.debug("Grid creation failed: %s", e)
            ui.notification_show(
                f"Error creating grid: {e!s}", type="error", duration=5
            )

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
            ui.notification_show(
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
            ui.notification_show(
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
            choices = {
                str(i): str(name) for i, name in enumerate(spname) if i > 0
            }
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
            ui.notification_show(
                "Port patch indices must be a comma-separated list of whole "
                "numbers; using uniform effort.",
                type="warning",
                duration=6,
            )
            return None
        if len(ports) == 0 or (g is not None and (ports >= g.n_patches).any()):
            ui.notification_show(
                f"Port patch indices must be between 0 and "
                f"{(g.n_patches - 1) if g else 0}; using uniform effort.",
                type="warning",
                duration=6,
            )
            return None
        return ports

    # Run spatial simulation handler
    @reactive.effect
    @reactive.event(input.run_spatial_sim)
    def _run_spatial_simulation():
        """Run spatial ECOSPACE simulation."""
        from pypath.spatial import EcospaceParams as EcoParams
        from pypath.spatial import rsim_run_spatial

        # Guard: need both scenario and grid
        if _sim_scenario is None or _sim_scenario() is None:
            ui.notification_show(
                "Please create an Ecosim scenario first (Ecosim Simulation tab).",
                type="warning",
                duration=5,
            )
            return

        current_grid = grid()
        if current_grid is None:
            ui.notification_show(
                "Please create a spatial grid first.",
                type="warning",
                duration=5,
            )
            return

        try:
            scenario = _sim_scenario()
            n_groups = scenario.params.NUM_GROUPS

            # Build habitat preference matrix [n_groups+1, n_patches]
            habitat_vec = _compute_habitat_vector(current_grid)
            habitat_pref = np.tile(habitat_vec, (n_groups + 1, 1))
            habitat_pref[0, :] = 0.0  # Index 0 = Outside

            # Build per-group dispersal rates
            dispersal_rate = np.full(
                n_groups + 1, float(input.dispersal_rate_default())
            )
            dispersal_rate[0] = 0.0  # No dispersal for Outside

            # Build advection flags
            advection_enabled = np.full(
                n_groups + 1, bool(input.enable_advection()), dtype=bool
            )
            advection_enabled[0] = False

            # Build gravity strength
            gravity = np.full(n_groups + 1, float(input.gravity_strength()))
            gravity[0] = 0.0

            ecospace_params = EcoParams(
                grid=current_grid,
                habitat_preference=habitat_pref,
                habitat_capacity=habitat_pref.copy(),
                dispersal_rate=dispersal_rate,
                advection_enabled=advection_enabled,
                gravity_strength=gravity,
            )

            ui.notification_show(
                "Running spatial simulation...", type="message", duration=2
            )

            spatial_fishing = _build_spatial_fishing()
            result = rsim_run_spatial(
                scenario,
                ecospace=ecospace_params,
                spatial_fishing=spatial_fishing,
            )
            _spatial_results.set(result)

            # Post-run: update biomass group dropdown with group names
            spname = getattr(scenario.params, "spname", None)
            if spname is not None:
                group_choices = {
                    str(i): name
                    for i, name in enumerate(spname)
                    if i > 0 and i <= n_groups
                }
            else:
                group_choices = {str(i): f"Group {i}" for i in range(1, n_groups + 1)}
            ui.update_select("biomass_view_group", choices=group_choices)

            # Update animation slider max to number of timesteps
            n_timesteps = result.out_Biomass_spatial.shape[0] - 1
            ui.update_slider("animation_time", max=n_timesteps, value=0)

            ui.notification_show(
                "Spatial simulation complete!", type="message", duration=3
            )

        except Exception as e:
            logger.error("Spatial simulation failed: %s", e, exc_info=True)
            ui.notification_show(
                f"Spatial simulation failed: {e!s}",
                type="error",
                duration=8,
            )

    # Grid visualization
    @render.ui
    def grid_plot():
        """Plot the spatial grid and boundary polygon with interactive Leaflet map."""
        from shiny import ui

        # Check if we have grid or boundary to display
        has_grid = grid() is not None
        has_boundary = boundary_polygon() is not None

        if not has_grid and not has_boundary:
            # Nothing to display
            return ui.div(
                ui.p(
                    "No grid or boundary loaded. Upload a file or create a grid.",
                    class_="text-muted text-center mt-5",
                ),
                style="height: 500px;",
            )

        try:
            import folium
            from folium import plugins
        except ImportError:
            return ui.div(
                ui.p(
                    "Folium is required for interactive maps. Install with: pip install folium",
                    class_="text-danger text-center mt-5",
                ),
                style="height: 500px;",
            )

        # Calculate map center and bounds
        if has_boundary:
            boundary_gdf = boundary_polygon()
            bounds = boundary_gdf.total_bounds  # minx, miny, maxx, maxy
            center_lat = (bounds[1] + bounds[3]) / 2
            center_lon = (bounds[0] + bounds[2]) / 2
        elif has_grid:
            g = grid()
            centroids = g.patch_centroids
            center_lat = np.mean(centroids[:, 1])
            center_lon = np.mean(centroids[:, 0])
        else:
            center_lat, center_lon = (
                PARAM_RANGES.default_center_lat,
                PARAM_RANGES.default_center_lon,
            )

        # Create folium map with OpenStreetMap tiles
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=10,
            tiles="OpenStreetMap",
            control_scale=True,
        )

        # Add additional tile layers
        folium.TileLayer("CartoDB positron", name="Light Map").add_to(m)
        folium.TileLayer("CartoDB dark_matter", name="Dark Map").add_to(m)

        # Add satellite imagery option
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri",
            name="Satellite",
            overlay=False,
            control=True,
        ).add_to(m)

        # Plot boundary polygon if available
        if has_boundary:
            boundary_gdf = boundary_polygon()

            # Convert to GeoJSON for folium
            boundary_geojson = boundary_gdf.__geo_interface__

            folium.GeoJson(
                boundary_geojson,
                name="Boundary",
                style_function=lambda _x: {
                    "fillColor": "red",
                    "color": "red",
                    "weight": 2.5,
                    "fillOpacity": 0.05,
                    "dashArray": "5, 5",
                },
                tooltip=folium.Tooltip("Study Area Boundary"),
            ).add_to(m)

        # Plot grid if available
        if has_grid:
            g = grid()

            # Check if we have polygon geometries (irregular grid)
            if g.geometry is not None:
                is_large_grid = g.n_patches > SPATIAL.large_grid_threshold

                # For large grids, use optimized rendering
                if is_large_grid:
                    # Create a single GeoJSON with all features (much faster)
                    features = []
                    for idx, row in g.geometry.iterrows():
                        for part in _iter_polygons(row.geometry):
                            features.append(
                                {
                                    "type": "Feature",
                                    "geometry": part.__geo_interface__,
                                    "properties": {
                                        "patch_id": idx,
                                        "area_km2": float(g.patch_areas[idx]),
                                    },
                                }
                            )

                    geojson_data = {"type": "FeatureCollection", "features": features}

                    # Add all polygons in one layer
                    folium.GeoJson(
                        geojson_data,
                        name="Grid Patches",
                        style_function=lambda _x: {
                            "fillColor": "lightblue",
                            "color": "steelblue",
                            "weight": 0.5,  # Thinner lines for large grids
                            "fillOpacity": 0.4,
                        },
                        tooltip=folium.GeoJsonTooltip(
                            fields=["patch_id", "area_km2"],
                            aliases=["Patch:", "Area (km²):"],
                            localize=True,
                        ),
                    ).add_to(m)
                    # No labels for large grids (too cluttered)

                else:
                    # Small grid: render individually with labels
                    for idx, row in g.geometry.iterrows():
                        for geom in _iter_polygons(row.geometry):
                            # Create GeoJSON for this polygon
                            geojson_data = {
                                "type": "Feature",
                                "geometry": geom.__geo_interface__,
                                "properties": {
                                    "patch_id": idx,
                                    "area_km2": g.patch_areas[idx],
                                },
                            }

                            # Add polygon to map
                            folium.GeoJson(
                                geojson_data,
                                style_function=lambda _x: {
                                    "fillColor": "lightblue",
                                    "color": "steelblue",
                                    "weight": 1.5,
                                    "fillOpacity": 0.6,
                                },
                                tooltip=folium.Tooltip(
                                    f"<b>Patch {idx}</b><br>Area: {g.patch_areas[idx]:.2f} km²"
                                ),
                                popup=folium.Popup(
                                    f"<b>Patch {idx}</b><br>"
                                    f"Area: {g.patch_areas[idx]:.2f} km²<br>"
                                    f"Center: ({g.patch_centroids[idx][0]:.4f}, {g.patch_centroids[idx][1]:.4f})"
                                ),
                            ).add_to(m)

                            # Add patch ID label at centroid
                            centroid = g.patch_centroids[idx]
                            folium.Marker(
                                location=[centroid[1], centroid[0]],  # lat, lon
                                icon=folium.DivIcon(
                                    html=f"""
                                    <div style="
                                        font-size: 10px;
                                        color: darkblue;
                                        font-weight: bold;
                                        text-align: center;
                                        text-shadow: 1px 1px 2px white, -1px -1px 2px white;
                                    ">{idx}</div>
                                """
                                ),
                            ).add_to(m)

                # Add info panel
                title_text = f"Irregular Grid: {g.n_patches:,} Patches"
                if has_boundary:
                    title_text += " (within boundary)"
                if is_large_grid:
                    title_text += " - Zoom in for details"

            else:
                # Regular grid - plot centroids and edges
                # Create feature group for edges
                edges_layer = folium.FeatureGroup(name="Connections")

                # Plot edges
                rows, cols = g.adjacency_matrix.nonzero()
                for idx in range(len(rows)):
                    i, j = rows[idx], cols[idx]
                    if i < j:  # Only plot each edge once
                        p1 = g.patch_centroids[i]
                        p2 = g.patch_centroids[j]
                        folium.PolyLine(
                            locations=[[p1[1], p1[0]], [p2[1], p2[0]]],
                            color="gray",
                            weight=1,
                            opacity=0.3,
                        ).add_to(edges_layer)

                edges_layer.add_to(m)

                # Plot patches as circle markers
                for i in range(g.n_patches):
                    centroid = g.patch_centroids[i]
                    folium.CircleMarker(
                        location=[centroid[1], centroid[0]],
                        radius=8,
                        color="steelblue",
                        fill=True,
                        fillColor="steelblue",
                        fillOpacity=0.8,
                        tooltip=folium.Tooltip(
                            f"<b>Patch {i}</b><br>Area: {g.patch_areas[i]:.2f} km²"
                        ),
                        popup=folium.Popup(
                            f"<b>Patch {i}</b><br>"
                            f"Area: {g.patch_areas[i]:.2f} km²<br>"
                            f"Location: ({centroid[0]:.4f}, {centroid[1]:.4f})"
                        ),
                    ).add_to(m)

                    # Add label
                    folium.Marker(
                        location=[centroid[1], centroid[0]],
                        icon=folium.DivIcon(
                            html=f"""
                            <div style="
                                font-size: 8px;
                                color: white;
                                font-weight: bold;
                                text-align: center;
                            ">{i}</div>
                        """
                        ),
                    ).add_to(m)

                title_text = f"Spatial Grid: {g.n_patches} Patches"

            # Add statistics overlay
            n_edges = g.adjacency_matrix.nnz // 2
            avg_neighbors = n_edges * 2 / g.n_patches if g.n_patches > 0 else 0

            # Add custom HTML overlay with stats
            stats_html = f"""
            <div style="
                position: fixed;
                top: 80px;
                left: 10px;
                width: 200px;
                background-color: rgba(245, 222, 179, 0.9);
                border: 2px solid black;
                border-radius: 5px;
                padding: 10px;
                font-size: 11px;
                z-index: 1000;
            ">
                <b>{title_text}</b><br>
                Patches: {g.n_patches}<br>
                Connections: {n_edges}<br>
                Avg neighbors: {avg_neighbors:.1f}
            </div>
            """
            m.get_root().html.add_child(folium.Element(stats_html))
        else:
            # Only boundary, no grid yet
            title_html = """
            <div style="
                position: fixed;
                top: 80px;
                left: 10px;
                width: 250px;
                background-color: rgba(255, 255, 255, 0.9);
                border: 2px solid red;
                border-radius: 5px;
                padding: 10px;
                font-size: 12px;
                font-weight: bold;
                z-index: 1000;
            ">
                Boundary Polygon<br>
                <span style="font-size: 10px; font-weight: normal;">Ready for Grid Generation</span>
            </div>
            """
            m.get_root().html.add_child(folium.Element(title_html))

        # Add layer control
        folium.LayerControl().add_to(m)

        # Add fullscreen button
        plugins.Fullscreen().add_to(m)

        # Add mouse position
        plugins.MousePosition().add_to(m)

        # Add measure control
        plugins.MeasureControl(
            primary_length_unit="kilometers",
            secondary_length_unit="meters",
            primary_area_unit="sqkilometers",
        ).add_to(m)

        # Fit bounds to show all features
        if has_boundary or has_grid:
            if has_boundary:
                bounds = boundary_polygon().total_bounds
            elif has_grid:
                g = grid()
                lons = g.patch_centroids[:, 0]
                lats = g.patch_centroids[:, 1]
                bounds = [lons.min(), lats.min(), lons.max(), lats.max()]

            # Fit map to bounds with padding
            m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

        # Return Leaflet map as HTML
        return ui.HTML(m._repr_html_())

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

    # Habitat visualization
    @render.plot
    def habitat_plot():
        """Plot habitat preference map."""
        req(grid())

        import matplotlib.pyplot as plt

        g = grid()

        # Use shared helper for habitat computation
        habitat = _compute_habitat_vector(g)

        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
        from matplotlib.patches import Polygon as MplPolygon

        # Normalize habitat values for colormap
        norm = Normalize(vmin=0, vmax=1)
        cmap = plt.colormaps["YlGn"]

        # Check if we have polygon geometries (irregular grid)
        if g.geometry is not None:
            # Plot actual polygon shapes with habitat colors
            for idx, row in g.geometry.iterrows():
                for geom in _iter_polygons(row.geometry):
                    x, y = geom.exterior.xy
                    color = cmap(norm(habitat[idx]))
                    polygon = MplPolygon(
                        list(zip(x, y, strict=True)),
                        facecolor=color,
                        edgecolor="darkgreen",
                        linewidth=1.2,
                        alpha=0.8,
                        zorder=1,
                    )
                    ax.add_patch(polygon)

                    # Add habitat value label
                    centroid = g.patch_centroids[idx]
                    ax.text(
                        centroid[0],
                        centroid[1],
                        f"{habitat[idx]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="black",
                        weight="bold",
                        zorder=3,
                    )

        else:
            # Regular grid - use scatter plot
            _scatter = ax.scatter(
                g.patch_centroids[:, 0],
                g.patch_centroids[:, 1],
                c=habitat,
                s=200,
                cmap="YlGn",
                vmin=0,
                vmax=1,
                edgecolors="black",
                linewidths=0.5,
            )

        # Add colorbar
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label="Habitat Quality (0-1)")

        ax.set_xlabel("Longitude (degrees)", fontsize=10)
        ax.set_ylabel("Latitude (degrees)", fontsize=10)
        ax.set_title(
            f"Habitat Preference Map - {input.habitat_pattern().replace('_', ' ').title()}",
            fontsize=12,
            weight="bold",
        )
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_aspect("equal")

        return fig

    # Fishing effort visualization
    @render.plot
    def fishing_effort_plot():
        """Plot spatial fishing effort allocation."""
        req(grid())

        import matplotlib.pyplot as plt

        g = grid()
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
                _compute_habitat_vector(g), total_effort=total_effort
            )
        else:
            effort = allocate_uniform(n_patches, total_effort)

        # Plot
        fig, ax = plt.subplots(figsize=(8, 6))

        scatter = ax.scatter(
            g.patch_centroids[:, 0],
            g.patch_centroids[:, 1],
            c=effort,
            s=effort * 10,  # Size proportional to effort
            cmap="Reds",
            edgecolors="black",
            linewidths=0.5,
            alpha=0.7,
        )

        plt.colorbar(scatter, ax=ax, label="Fishing Effort")

        ax.set_xlabel("X (degrees longitude)")
        ax.set_ylabel("Y (degrees latitude)")
        ax.set_title(f"Spatial Fishing Effort ({allocation_method})")
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")

        return fig

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

    # Biomass animation visualization
    @render.ui
    def biomass_animation_ui():
        """Render biomass animation: placeholder or spatial heatmap."""
        results = _spatial_results()
        if results is None or not hasattr(results, "out_Biomass_spatial"):
            return ui.div(
                ui.div(
                    ui.tags.i(class_="bi bi-info-circle me-2"),
                    "Run a spatial simulation to view biomass dynamics over time.",
                    class_="alert alert-info",
                    style="margin-top: 20px;",
                )
            )

        # Render the heatmap as an embedded plot
        return ui.output_plot("biomass_animation_plot", height="500px")

    @render.plot
    def biomass_animation_plot():
        """Plot spatial biomass heatmap for selected group and timestep."""
        import matplotlib.pyplot as plt

        results = _spatial_results()
        req(results is not None and hasattr(results, "out_Biomass_spatial"))

        g = grid()
        req(g is not None)

        biomass_spatial = (
            results.out_Biomass_spatial
        )  # [n_months, n_groups+1, n_patches]

        # Get selected group index
        group_str = input.biomass_view_group()
        group_idx = int(group_str) if group_str else 1

        # Get selected timestep
        timestep = int(input.animation_time())
        timestep = min(timestep, biomass_spatial.shape[0] - 1)

        # Extract per-patch biomass for this group/timestep
        patch_biomass = biomass_spatial[timestep, group_idx, :]

        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
        from matplotlib.patches import Polygon as MplPolygon

        fig, ax = plt.subplots(figsize=(10, 8))

        vmax = max(biomass_spatial[:, group_idx, :].max(), 1e-10)
        norm = Normalize(vmin=0, vmax=vmax)
        cmap = plt.colormaps["YlOrRd"]

        if g.geometry is not None:
            for idx, row in g.geometry.iterrows():
                for geom in _iter_polygons(row.geometry):
                    x, y = geom.exterior.xy
                    color = cmap(norm(patch_biomass[idx]))
                    polygon = MplPolygon(
                        list(zip(x, y, strict=True)),
                        facecolor=color,
                        edgecolor="gray",
                        linewidth=0.8,
                        alpha=0.9,
                    )
                    ax.add_patch(polygon)
            ax.autoscale_view()
        else:
            ax.scatter(
                g.patch_centroids[:, 0],
                g.patch_centroids[:, 1],
                c=patch_biomass,
                s=200,
                cmap="YlOrRd",
                vmin=0,
                vmax=vmax,
                edgecolors="black",
                linewidths=0.5,
            )

        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label="Biomass")

        # Get group name for title
        scenario = _sim_scenario() if _sim_scenario is not None else None
        if scenario and hasattr(scenario.params, "spname"):
            group_name = scenario.params.spname[group_idx]
        else:
            group_name = f"Group {group_idx}"

        ax.set_xlabel("Longitude (degrees)", fontsize=10)
        ax.set_ylabel("Latitude (degrees)", fontsize=10)
        ax.set_title(
            f"Biomass: {group_name} (t={timestep})",
            fontsize=12,
            weight="bold",
        )
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_aspect("equal")

        return fig

    # Spatial metrics table
    @render.table
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
