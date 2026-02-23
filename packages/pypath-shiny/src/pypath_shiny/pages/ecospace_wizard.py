"""Ecospace Data Wizard — 7-step guided ecospace model creation.

Steps:
1. Select Area — draw polygon on map
2. Configure Grid — choose grid type and resolution
3. Download Data — fetch EMODnet habitats and bathymetry
4. Review Habitats — inspect and merge EUNIS categories
5. Assign Preferences — semi-auto habitat preferences per species group
6. Set Dispersal — per-group dispersal parameters
7. Review & Launch — summary and build EcospaceParams
"""

import json
import logging

from shiny import Inputs, Outputs, Session, reactive, render, ui

logger = logging.getLogger(__name__)

_STEPS = [
    "Select Area",
    "Configure Grid",
    "Download Data",
    "Review Habitats",
    "Assign Preferences",
    "Set Dispersal",
    "Review & Launch",
]


def _step_progress_ui():
    """Render the step progress bar."""
    items = []
    for i, label in enumerate(_STEPS, 1):
        items.append(
            ui.span(
                f"{i}. {label}",
                class_="badge bg-secondary me-1",
                id=f"wizard_step_badge_{i}",
            )
        )
    return ui.div(*items, class_="mb-3")


def ecospace_wizard_ui():
    """Wizard page UI."""
    return ui.page_fluid(
        ui.h3("Ecospace Data Wizard"),
        _step_progress_ui(),
        ui.output_ui("wizard_step_content"),
        ui.div(
            ui.input_action_button("wizard_back", "Back", class_="btn-secondary me-2"),
            ui.input_action_button("wizard_next", "Next", class_="btn-primary"),
            class_="mt-3",
        ),
    )


def ecospace_wizard_server(
    input: Inputs, output: Outputs, session: Session, shared_data=None
):
    """Wizard page server logic."""
    wizard_step = reactive.value(1)

    # Reactive state for wizard data
    drawn_polygon = reactive.value(None)
    ecospace_grid = reactive.value(None)
    habitat_gdf = reactive.value(None)
    depth_per_patch = reactive.value(None)
    salinity_layer = reactive.value(None)
    habitat_types_arr = reactive.value(None)
    download_status = reactive.value("")

    @reactive.effect
    @reactive.event(input.wizard_drawn_polygon)
    def _capture_polygon():
        raw = input.wizard_drawn_polygon()
        if raw:
            drawn_polygon.set(json.loads(raw))
            logger.info("Polygon captured with coordinates")
        else:
            drawn_polygon.set(None)

    @reactive.effect
    @reactive.event(input.wizard_next)
    def _next():
        current = wizard_step.get()
        if current < len(_STEPS):
            wizard_step.set(current + 1)

    @reactive.effect
    @reactive.event(input.wizard_back)
    def _back():
        current = wizard_step.get()
        if current > 1:
            wizard_step.set(current - 1)

    @reactive.effect
    @reactive.event(input.wizard_next)
    def _create_grid_on_step3():
        """Create grid when advancing from Step 2 to Step 3."""
        if wizard_step.get() != 3:
            return
        poly = drawn_polygon.get()
        if poly is None:
            return

        try:
            from shapely.geometry import shape

            from pypath.spatial.ecospace_params import EcospaceGrid

            polygon = shape(poly)
            bounds = polygon.bounds  # (minx, miny, maxx, maxy)
            cell_size_km = input.wizard_cell_size()
            grid_type = input.wizard_grid_type()

            # Convert cell size from km to approximate degrees (1 deg lat ~ 111 km)
            cell_deg = cell_size_km / 111.0
            nx = max(2, int((bounds[2] - bounds[0]) / cell_deg))
            ny = max(2, int((bounds[3] - bounds[1]) / cell_deg))

            grid = EcospaceGrid.from_regular_grid(
                bounds=(bounds[0], bounds[1], bounds[2], bounds[3]),
                nx=nx,
                ny=ny,
            )
            ecospace_grid.set(grid)
            logger.info(
                "Created %s grid: %d patches (%dx%d)",
                grid_type,
                grid.n_patches,
                nx,
                ny,
            )
        except Exception as e:
            logger.error("Grid creation failed: %s", e)

    @reactive.effect
    @reactive.event(input.wizard_download)
    def _download_data():
        """Download EMODnet habitats and bathymetry."""
        grid = ecospace_grid.get()
        poly = drawn_polygon.get()
        if grid is None or poly is None:
            download_status.set(
                "Please draw a polygon and configure the grid first."
            )
            return

        try:
            from shapely.geometry import shape

            from pypath.io.marine_data import (
                EMODnetBathymetryClient,
                EMODnetHabitatsClient,
                MarineDataCache,
            )

            polygon = shape(poly)
            bbox = polygon.bounds  # (minx, miny, maxx, maxy)

            download_status.set("Downloading habitats...")
            cache = MarineDataCache()

            # Fetch habitats
            hab_client = EMODnetHabitatsClient(cache=cache)
            gdf = hab_client.fetch_euseamap(bbox=bbox)
            habitat_gdf.set(gdf)

            # Rasterize habitats onto grid
            hab_map = hab_client.rasterize_habitats(gdf, grid)
            habitat_types_arr.set(hab_map)

            download_status.set("Downloading bathymetry...")

            # Fetch depth
            bathy_client = EMODnetBathymetryClient(cache=cache)
            raster, transform = bathy_client.fetch_depth(bbox=bbox)
            depth = bathy_client.sample_to_grid(raster, transform, grid)
            depth_per_patch.set(depth)

            download_status.set(
                f"Done! {len(gdf)} habitat features, "
                f"depth range: {depth.min():.0f}m to {depth.max():.0f}m"
            )
            logger.info(
                "Download complete: %d habitats, depth range %.0f-%.0f",
                len(gdf),
                depth.min(),
                depth.max(),
            )
        except Exception as e:
            download_status.set(f"Download failed: {e}")
            logger.error("Download failed: %s", e)

    @reactive.effect
    @reactive.event(input.wizard_salinity_file)
    def _load_salinity():
        """Handle salinity file upload."""
        file_info = input.wizard_salinity_file()
        if not file_info:
            return
        grid = ecospace_grid.get()
        if grid is None:
            return
        try:
            from pypath.io.marine_data import SalinityLoader

            filepath = file_info[0]["datapath"]
            if filepath.endswith(".csv"):
                layer = SalinityLoader.load_from_csv(filepath, grid)
            else:
                layer = SalinityLoader.load_from_netcdf(filepath, grid)
            salinity_layer.set(layer)
            logger.info("Loaded salinity data")
        except Exception as e:
            logger.error("Salinity load failed: %s", e)

    @render.ui
    def wizard_download_status():
        """Display download progress/status."""
        status = download_status.get()
        if not status:
            return ui.p()
        if status.startswith("Done"):
            return ui.p(status, class_="text-success mt-2")
        elif status.startswith("Download failed") or status.startswith(
            "Please"
        ):
            return ui.p(status, class_="text-danger mt-2")
        return ui.p(status, class_="text-info mt-2")

    @render.ui
    def wizard_step_content():
        step = wizard_step.get()
        if step == 1:
            return _step1_select_area_ui()
        elif step == 2:
            return _step2_configure_grid_ui()
        elif step == 3:
            return _step3_download_data_ui()
        elif step == 4:
            return _step4_review_habitats_ui()
        elif step == 5:
            return _step5_assign_preferences_ui()
        elif step == 6:
            return _step6_set_dispersal_ui()
        elif step == 7:
            return _step7_review_launch_ui()
        return ui.p("Unknown step")


def _step1_select_area_ui():
    leaflet_html = """
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <link rel="stylesheet"
          href="https://cdnjs.cloudflare.com/ajax/libs/leaflet.draw/1.0.4/leaflet.draw.css" />
    <div id="wizard-map"
         style="height: 500px; width: 100%; border: 1px solid #ddd;
                border-radius: 4px;"></div>
    <div id="wizard-area-info" class="mt-2 text-muted"></div>
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script
        src="https://cdnjs.cloudflare.com/ajax/libs/leaflet.draw/1.0.4/leaflet.draw.js">
    </script>
    <script>
    (function() {
        // Wait for container to be ready
        setTimeout(function() {
            var map = L.map('wizard-map').setView([55, 15], 5);
            L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                attribution: '&copy; OpenStreetMap contributors',
                maxZoom: 18,
            }).addTo(map);

            var drawnItems = new L.FeatureGroup();
            map.addLayer(drawnItems);

            var drawControl = new L.Control.Draw({
                draw: {
                    polygon: {
                        allowIntersection: false,
                        shapeOptions: { color: '#3388ff' }
                    },
                    polyline: false,
                    rectangle: false,
                    circle: false,
                    marker: false,
                    circlemarker: false,
                },
                edit: { featureGroup: drawnItems }
            });
            map.addControl(drawControl);

            // Approximate geodesic area using the shoelace formula on
            // WGS-84 coordinates (Gauss / trapezoidal spherical excess).
            // Falls back to a simple lat/lon bounding-box estimate when
            // L.GeometryUtil is unavailable.
            function computeAreaKm2(latlngs) {
                if (L.GeometryUtil && L.GeometryUtil.geodesicArea) {
                    return L.GeometryUtil.geodesicArea(latlngs) / 1e6;
                }
                // Fallback: spherical polygon area via shoelace on radians
                var RAD = Math.PI / 180;
                var R = 6371; // Earth radius in km
                var n = latlngs.length;
                if (n < 3) return 0;
                var total = 0;
                for (var i = 0; i < n; i++) {
                    var j = (i + 1) % n;
                    var lat1 = latlngs[i].lat * RAD;
                    var lon1 = latlngs[i].lng * RAD;
                    var lat2 = latlngs[j].lat * RAD;
                    var lon2 = latlngs[j].lng * RAD;
                    total += (lon2 - lon1) * (2 + Math.sin(lat1) + Math.sin(lat2));
                }
                return Math.abs(total * R * R / 2);
            }

            function updateArea(layer) {
                var latlngs = layer.getLatLngs()[0];
                var geojson = layer.toGeoJSON();
                var areaKm2 = computeAreaKm2(latlngs).toFixed(1);
                document.getElementById('wizard-area-info').innerHTML =
                    '<strong>Study area:</strong> ~' + areaKm2 +
                    ' km&sup2; (' + latlngs.length + ' vertices)';
                // Send to Shiny
                if (window.Shiny) {
                    Shiny.setInputValue('wizard_drawn_polygon',
                                        JSON.stringify(geojson.geometry));
                }
            }

            map.on(L.Draw.Event.CREATED, function(e) {
                drawnItems.clearLayers();
                drawnItems.addLayer(e.layer);
                updateArea(e.layer);
            });

            map.on(L.Draw.Event.EDITED, function(e) {
                e.layers.eachLayer(function(layer) { updateArea(layer); });
            });

            map.on(L.Draw.Event.DELETED, function(e) {
                document.getElementById('wizard-area-info').innerHTML = '';
                if (window.Shiny) {
                    Shiny.setInputValue('wizard_drawn_polygon', null);
                }
            });
        }, 100);
    })();
    </script>
    """
    return ui.card(
        ui.card_header("Step 1: Select Study Area"),
        ui.p("Draw a polygon on the map to define your study area."),
        ui.HTML(leaflet_html),
    )


def _step2_configure_grid_ui():
    return ui.card(
        ui.card_header("Step 2: Configure Grid"),
        ui.input_radio_buttons(
            "wizard_grid_type",
            "Grid Type",
            choices={"regular": "Regular Rectangular", "hexagonal": "Hexagonal"},
            selected="regular",
        ),
        ui.input_numeric(
            "wizard_cell_size", "Cell Size (km)", value=5, min=0.5, max=100
        ),
    )


def _step3_download_data_ui():
    return ui.card(
        ui.card_header("Step 3: Download Data"),
        ui.p("Download EMODnet seabed habitats and bathymetry for your study area."),
        ui.input_action_button(
            "wizard_download", "Download Data", class_="btn-primary"
        ),
        ui.output_ui("wizard_download_status"),
        ui.hr(),
        ui.p("Salinity (optional):"),
        ui.input_file(
            "wizard_salinity_file",
            "Upload salinity file (CSV or NetCDF)",
            accept=[".csv", ".nc", ".nc4"],
        ),
    )


def _step4_review_habitats_ui():
    return ui.card(
        ui.card_header("Step 4: Review Habitats"),
        ui.p("Review EUNIS habitat types assigned to each grid patch."),
        ui.output_ui("wizard_habitat_map"),
        ui.output_table("wizard_habitat_table"),
    )


def _step5_assign_preferences_ui():
    return ui.card(
        ui.card_header("Step 5: Assign Habitat Preferences"),
        ui.input_select(
            "wizard_preset",
            "Quick Preset",
            choices={
                "none": "-- Manual --",
                "pelagic": "Pelagic",
                "demersal": "Demersal",
                "benthic": "Benthic",
                "auto": "Auto-suggest (biodata)",
            },
        ),
        ui.output_ui("wizard_preference_editor"),
    )


def _step6_set_dispersal_ui():
    return ui.card(
        ui.card_header("Step 6: Set Dispersal Parameters"),
        ui.input_slider(
            "wizard_dispersal_default",
            "Default Dispersal Rate (km\u00b2/month)",
            min=0.0,
            max=100.0,
            value=10.0,
            step=0.5,
        ),
        ui.input_slider(
            "wizard_gravity",
            "Gravity Strength",
            min=0.0,
            max=1.0,
            value=0.3,
            step=0.05,
        ),
        ui.output_ui("wizard_dispersal_table"),
    )


def _step7_review_launch_ui():
    return ui.card(
        ui.card_header("Step 7: Review & Launch"),
        ui.output_ui("wizard_summary"),
        ui.input_action_button(
            "wizard_create",
            "Create Ecospace Model",
            class_="btn-success btn-lg",
        ),
    )
