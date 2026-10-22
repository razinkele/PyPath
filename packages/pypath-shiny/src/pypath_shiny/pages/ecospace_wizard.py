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

from pypath_shiny.services.notifications import (
    notify_error,
    notify_info,
    notify_success,
    notify_warning,
)
from pypath_shiny.services.tasks import (
    TaskState,
    TaskStatus,
    apply_if_latest,
    finish_task,
    new_running_task,
    update_task_progress,
)
from pypath_shiny.services.ui_state import (
    deserialize_wizard_draft,
    serialize_wizard_draft,
)
from pypath_shiny.services.validation import (
    ValidationIssue,
    ValidationSeverity,
    sort_issues,
    summarize_issues,
)

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


def serialize_wizard_draft_payload(
    *,
    step: int,
    polygon: dict | None,
    grid_type: str | None,
    cell_size_km: float | None,
    dispersal_default: float | None,
    gravity_strength: float | None,
) -> str:
    """Build and serialize a wizard draft payload."""
    return serialize_wizard_draft(
        {
            "step": step,
            "polygon": polygon,
            "grid_type": grid_type,
            "cell_size_km": cell_size_km,
            "dispersal_default": dispersal_default,
            "gravity_strength": gravity_strength,
        }
    )


def deserialize_wizard_draft_payload(data: str) -> dict:
    """Deserialize a wizard draft payload and normalize expected fields."""
    draft = deserialize_wizard_draft(data)
    return {
        "draft_version": draft.get("draft_version"),
        "step": draft.get("step", 1),
        "polygon": draft.get("polygon"),
        "grid_type": draft.get("grid_type"),
        "cell_size_km": draft.get("cell_size_km"),
        "dispersal_default": draft.get("dispersal_default"),
        "gravity_strength": draft.get("gravity_strength"),
    }


def collect_wizard_validation_issues(
    *,
    grid,
    habitat_types,
    preference_matrix,
    group_names: list[str],
    salinity_layer,
) -> list[ValidationIssue]:
    """Collect pre-launch validation issues for the wizard summary."""
    issues: list[ValidationIssue] = []

    if grid is None:
        issues.append(
            ValidationIssue(
                field="grid",
                message="Grid is not configured.",
                severity=ValidationSeverity.ERROR,
                section="Setup",
            )
        )

    if habitat_types is None:
        issues.append(
            ValidationIssue(
                field="habitats",
                message="Habitat data is missing. Complete the data download step.",
                severity=ValidationSeverity.ERROR,
                section="Data",
            )
        )
    elif grid is not None and len(habitat_types) != getattr(grid, "n_patches", -1):
        issues.append(
            ValidationIssue(
                field="habitats",
                message="Habitat-to-grid mapping does not match patch count.",
                severity=ValidationSeverity.ERROR,
                section="Data",
            )
        )

    if preference_matrix is None:
        issues.append(
            ValidationIssue(
                field="preferences",
                message="Habitat preference matrix is missing.",
                severity=ValidationSeverity.WARNING,
                section="Configuration",
            )
        )
    else:
        n_groups = len(group_names)
        if preference_matrix.shape[0] != n_groups:
            issues.append(
                ValidationIssue(
                    field="preferences",
                    message="Preference matrix rows do not match number of groups.",
                    severity=ValidationSeverity.ERROR,
                    section="Configuration",
                )
            )

    if salinity_layer is None:
        issues.append(
            ValidationIssue(
                field="salinity",
                message="No salinity layer loaded (optional).",
                severity=ValidationSeverity.INFO,
                section="Environment",
            )
        )

    return sort_issues(issues)


def summarize_wizard_validation_issues(issues: list[ValidationIssue]) -> str:
    """Summarize wizard validation issues for dashboard rendering."""
    return summarize_issues(issues)


def wizard_download_task_alert(task_state: TaskState) -> tuple[str, str, str]:
    """Map wizard download task state to alert style, icon, and message."""
    if task_state.status is TaskStatus.RUNNING:
        return (
            "text-info mt-2",
            "bi bi-arrow-repeat me-2",
            f"{task_state.message} ({task_state.progress:.0f}%)",
        )
    if task_state.status is TaskStatus.ERROR:
        msg = task_state.error or task_state.message or "Download failed."
        return ("text-danger mt-2", "bi bi-exclamation-triangle me-2", msg)
    if task_state.status is TaskStatus.SUCCESS:
        return ("text-success mt-2", "bi bi-check-circle me-2", task_state.message)
    return ("", "", "")


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
        ui.hr(),
        ui.card(
            ui.card_header("Draft State"),
            ui.input_text_area(
                "wizard_draft_json",
                "Draft JSON",
                rows=5,
                placeholder="Paste draft JSON here or click Save Draft",
            ),
            ui.div(
                ui.input_action_button(
                    "wizard_save_draft",
                    "Save Draft",
                    class_="btn-outline-primary me-2",
                ),
                ui.input_action_button(
                    "wizard_load_draft",
                    "Load Draft",
                    class_="btn-outline-secondary me-2",
                ),
                ui.input_action_button(
                    "wizard_reset_draft",
                    "Reset Draft",
                    class_="btn-outline-danger",
                ),
                class_="mt-2",
            ),
        ),
    )


def shared_group_names(shared_data, default_n=5):
    """Group names from ``shared_data.params`` (a reactive.Value) with a fallback.

    The stored object may be an RpathParams (``.model`` DataFrame) or a balanced
    Rpath (``.Group`` array). Must be called inside a reactive context.
    """
    p = None
    if shared_data is not None and hasattr(shared_data, "params"):
        val = shared_data.params
        p = val() if callable(val) else val
    if p is not None:
        df = getattr(p, "model", None)
        if df is not None and hasattr(df, "columns") and "Group" in df.columns:
            return [str(g) for g in df["Group"]]
        if hasattr(p, "Group"):
            return [str(g) for g in p.Group]
    return [f"Group {i + 1}" for i in range(default_n)]


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
    download_task_state = reactive.value(TaskState(run_id="wizard-download-idle"))
    preference_matrix = reactive.value(None)

    @reactive.effect
    @reactive.event(input.wizard_save_draft)
    def _save_draft():
        try:
            payload = serialize_wizard_draft_payload(
                step=wizard_step.get(),
                polygon=drawn_polygon.get(),
                grid_type=input.wizard_grid_type(),
                cell_size_km=input.wizard_cell_size(),
                dispersal_default=input.wizard_dispersal_default(),
                gravity_strength=input.wizard_gravity(),
            )
            ui.update_text_area("wizard_draft_json", value=payload)
            _notify_legacy("Wizard draft saved.", type="message")
        except Exception as exc:
            logger.exception("Failed to save wizard draft")
            _notify_legacy(f"Could not save draft: {exc!s}", type="error")

    @reactive.effect
    @reactive.event(input.wizard_load_draft)
    def _load_draft():
        raw = input.wizard_draft_json()
        if not raw:
            _notify_legacy("Paste a draft JSON payload first.", type="warning")
            return
        try:
            draft = deserialize_wizard_draft_payload(raw)
            step = int(draft.get("step", 1) or 1)
            step = max(1, min(len(_STEPS), step))
            wizard_step.set(step)
            drawn_polygon.set(draft.get("polygon"))

            if draft.get("grid_type") in {"regular", "hexagonal"}:
                ui.update_select("wizard_grid_type", selected=draft["grid_type"])
            if draft.get("cell_size_km") is not None:
                ui.update_numeric(
                    "wizard_cell_size", value=float(draft["cell_size_km"])
                )
            if draft.get("dispersal_default") is not None:
                ui.update_slider(
                    "wizard_dispersal_default", value=float(draft["dispersal_default"])
                )
            if draft.get("gravity_strength") is not None:
                ui.update_slider(
                    "wizard_gravity", value=float(draft["gravity_strength"])
                )

            _notify_legacy("Wizard draft loaded.", type="message")
        except Exception as exc:
            logger.exception("Failed to load wizard draft")
            _notify_legacy(f"Could not load draft: {exc!s}", type="error")

    @reactive.effect
    @reactive.event(input.wizard_reset_draft)
    def _reset_draft():
        wizard_step.set(1)
        drawn_polygon.set(None)
        ui.update_text_area("wizard_draft_json", value="")
        _notify_legacy("Wizard draft reset.", type="message")

    @reactive.effect
    @reactive.event(input.wizard_drawn_polygon, ignore_none=False)
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
            from pypath.spatial.ecospace_params import EcospaceGrid
            from shapely.geometry import shape

            polygon = shape(poly)
            bounds = polygon.bounds  # (minx, miny, maxx, maxy)
            cell_size_km = input.wizard_cell_size()
            grid_type = input.wizard_grid_type()

            # Convert cell size from km to approximate degrees (1 deg lat ~ 111 km)
            cell_deg = cell_size_km / 111.0
            nx = max(2, int((bounds[2] - bounds[0]) / cell_deg))
            ny = max(2, int((bounds[3] - bounds[1]) / cell_deg))

            grid = None
            if grid_type == "hexagonal":
                # Tessellate the drawn polygon with hexagons of the chosen size
                try:
                    import geopandas as gpd

                    from pypath_shiny.pages.ecospace import (
                        create_hexagonal_grid_in_boundary,
                    )

                    boundary_gdf = gpd.GeoDataFrame(
                        [{"geometry": polygon}], crs="EPSG:4326"
                    )
                    grid = create_hexagonal_grid_in_boundary(
                        boundary_gdf, hexagon_size_km=cell_size_km
                    )
                except Exception as e:
                    logger.warning(
                        "Hexagonal grid failed (%s); falling back to a regular grid",
                        e,
                    )
                    grid = None

            if grid is None:
                grid = EcospaceGrid.from_regular_grid(
                    bounds=(bounds[0], bounds[1], bounds[2], bounds[3]),
                    nx=nx,
                    ny=ny,
                )
                grid_type = "regular"

            ecospace_grid.set(grid)
            logger.info(
                "Created %s grid: %d patches",
                grid_type,
                grid.n_patches,
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
            msg = "Please draw a polygon and configure the grid first."
            download_status.set(msg)
            download_task_state.set(
                TaskState(
                    run_id="wizard-download-idle",
                    status=TaskStatus.ERROR,
                    message="Wizard setup incomplete",
                    error=msg,
                )
            )
            return

        run_id = f"wizard-download-{input.wizard_download()}"
        download_task_state.set(new_running_task(run_id, "Preparing EMODnet requests"))

        def _update_latest(updater):
            download_task_state.set(
                apply_if_latest(
                    download_task_state.get(), run_id=run_id, updater=updater
                )
            )

        try:
            from pypath.io.marine_data import (
                EMODnetBathymetryClient,
                EMODnetHabitatsClient,
                MarineDataCache,
            )
            from shapely.geometry import shape

            polygon = shape(poly)
            bbox = polygon.bounds  # (minx, miny, maxx, maxy)

            download_status.set("Downloading habitats...")
            _update_latest(
                lambda s: update_task_progress(
                    s,
                    message="Downloading habitats",
                    progress=20.0,
                )
            )
            cache = MarineDataCache()

            # Fetch habitats
            hab_client = EMODnetHabitatsClient(cache=cache)
            gdf = hab_client.fetch_euseamap(bbox=bbox)
            habitat_gdf.set(gdf)

            # Rasterize habitats onto grid
            hab_map = hab_client.rasterize_habitats(gdf, grid)
            habitat_types_arr.set(hab_map)

            download_status.set("Downloading bathymetry...")
            _update_latest(
                lambda s: update_task_progress(
                    s,
                    message="Downloading bathymetry",
                    progress=65.0,
                )
            )

            # Fetch depth
            bathy_client = EMODnetBathymetryClient(cache=cache)
            raster, transform = bathy_client.fetch_depth(bbox=bbox)
            depth = bathy_client.sample_to_grid(raster, transform, grid)
            depth_per_patch.set(depth)

            done = (
                f"Done! {len(gdf)} habitat features, "
                f"depth range: {depth.min():.0f}m to {depth.max():.0f}m"
            )
            download_status.set(done)
            _update_latest(
                lambda s, msg=done: finish_task(s, success=True, message=msg)
            )
            logger.info(
                "Download complete: %d habitats, depth range %.0f-%.0f",
                len(gdf),
                depth.min(),
                depth.max(),
            )
        except Exception as e:
            err = f"Download failed: {e}"
            download_status.set(err)
            _update_latest(
                lambda s, msg=err: finish_task(
                    s,
                    success=False,
                    message="Download failed",
                    error=msg,
                )
            )
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
        task_state = download_task_state.get()

        alert_class, icon_class, task_message = wizard_download_task_alert(task_state)
        if task_message:
            return ui.p(ui.tags.i(class_=icon_class), task_message, class_=alert_class)

        if not status:
            return ui.p()
        if status.startswith("Done"):
            return ui.p(status, class_="text-success mt-2")
        elif status.startswith("Download failed") or status.startswith("Please"):
            return ui.p(status, class_="text-danger mt-2")
        return ui.p(status, class_="text-info mt-2")

    @render.ui
    def btn_cancel_download_ui():
        """Render cancel button only when download is running."""
        task_state = download_task_state.get()
        if task_state.status == TaskStatus.RUNNING and task_state.can_cancel:
            return ui.input_action_button(
                "wizard_cancel_download",
                "Cancel Download",
                class_="btn-sm btn-warning ms-2",
            )
        return None

    @reactive.effect
    @reactive.event(input.wizard_cancel_download)
    def _cancel_download():
        """Cancel the download operation."""
        task_state = download_task_state.get()
        if task_state.status == TaskStatus.RUNNING:
            download_task_state.set(
                finish_task(
                    task_state,
                    success=False,
                    message="Download cancelled",
                    error="User cancelled download",
                )
            )
            notify_warning("Download cancelled")

    @render.ui
    def wizard_habitat_map():
        """Display habitat type summary."""
        hab = habitat_types_arr.get()
        if hab is None:
            return ui.p(
                "No habitat data yet. Complete Step 3 first.",
                class_="text-muted",
            )

        import numpy as np

        unique_types = sorted(set(hab))
        items = []
        for htype in unique_types:
            count = int(np.sum(hab == htype))
            pct = 100.0 * count / len(hab)
            items.append(
                ui.div(
                    ui.tags.span(htype, class_="badge bg-info me-2"),
                    f"{count} patches ({pct:.1f}%)",
                    class_="mb-1",
                )
            )
        return ui.div(
            ui.h5(f"{len(unique_types)} habitat types found across {len(hab)} patches"),
            *items,
        )

    @render.table
    def wizard_habitat_table():
        """Habitat types summary table."""
        hab = habitat_types_arr.get()
        if hab is None:
            import pandas as pd

            return pd.DataFrame()

        import numpy as np
        import pandas as pd

        unique_types = sorted(set(hab))
        rows = []
        depth = depth_per_patch.get()
        for htype in unique_types:
            mask = hab == htype
            count = int(np.sum(mask))
            pct = 100.0 * count / len(hab)
            row = {
                "EUNIS Type": htype,
                "Patches": count,
                "Coverage (%)": f"{pct:.1f}",
            }
            if depth is not None:
                d = depth[mask]
                row["Depth Range (m)"] = f"{d.min():.0f} to {d.max():.0f}"
            rows.append(row)
        return pd.DataFrame(rows)

    @reactive.calc
    def _preference_prefs():
        """Preference matrix for the current habitat types and preset.

        A reactive calc rather than a render side effect, so the matrix used by
        'Create Ecospace Model' does not depend on whether the editor rendered.
        """
        hab = habitat_types_arr.get()
        if hab is None:
            return None, None, None

        import numpy as np
        from pypath.io.marine_data import HabitatPreferenceBuilder

        habitat_types = sorted(set(hab))
        group_names = shared_group_names(shared_data)

        builder = HabitatPreferenceBuilder()
        preset = input.wizard_preset()

        if preset == "auto":
            prefs = builder.suggest_preferences(group_names, habitat_types)
        elif preset != "none":
            prefs = builder.apply_preset(len(group_names), habitat_types, preset)
        else:
            # Default moderate preferences
            prefs = np.ones((len(group_names), len(habitat_types))) * 0.5
        return prefs, group_names, habitat_types

    @reactive.effect
    def _store_preference_matrix():
        """Keep `preference_matrix` in step with the inputs, not with rendering."""
        prefs, _, _ = _preference_prefs()
        preference_matrix.set(prefs)

    @render.ui
    def wizard_preference_editor():
        """Editable preference matrix."""
        prefs, group_names, habitat_types = _preference_prefs()
        if prefs is None:
            return ui.p("Complete Steps 3-4 first.", class_="text-muted")

        # Build an HTML table showing the preference matrix
        import pandas as pd

        df = pd.DataFrame(prefs, index=group_names, columns=habitat_types)
        df = df.round(2)
        html = df.to_html(classes="table table-sm table-striped", border=0)

        return ui.div(
            ui.p(
                f"Preference values for {len(group_names)} groups "
                f"\u00d7 {len(habitat_types)} habitats:"
            ),
            ui.HTML(html),
            ui.p(
                "Values range from 0 (avoid) to 1 (prefer).",
                class_="text-muted small",
            ),
        )

    @render.ui
    def wizard_dispersal_table():
        """Per-group dispersal settings table."""
        grid = ecospace_grid.get()
        if grid is None:
            return ui.p("Complete earlier steps first.", class_="text-muted")

        import pandas as pd

        # Get group names
        group_names = shared_group_names(shared_data)

        default_rate = input.wizard_dispersal_default()
        gravity = input.wizard_gravity()

        n_groups = len(group_names)
        df = pd.DataFrame(
            {
                "Group": group_names,
                "Dispersal Rate (km\u00b2/month)": [default_rate] * n_groups,
                "Gravity Strength": [gravity] * n_groups,
            }
        )
        html = df.to_html(classes="table table-sm table-striped", border=0, index=False)
        return ui.div(
            ui.p(f"Dispersal settings for {n_groups} groups:"),
            ui.HTML(html),
        )

    @render.ui
    def wizard_summary():
        """Summary dashboard before model creation."""
        grid = ecospace_grid.get()
        hab = habitat_types_arr.get()
        depth = depth_per_patch.get()
        sal = salinity_layer.get()
        prefs = preference_matrix.get()

        group_names = shared_group_names(shared_data)
        validation_issues = collect_wizard_validation_issues(
            grid=grid,
            habitat_types=hab,
            preference_matrix=prefs,
            group_names=group_names,
            salinity_layer=sal,
        )

        items = []

        if validation_issues:
            summary = summarize_wizard_validation_issues(validation_issues)
            has_error = any(
                i.severity is ValidationSeverity.ERROR for i in validation_issues
            )
            alert_class = "alert-danger" if has_error else "alert-warning"
            issue_items = [
                ui.tags.li(f"[{issue.section}] {issue.message}")
                for issue in validation_issues
            ]
            items.append(
                ui.div(
                    ui.h5("Validation Summary"),
                    ui.p(summary),
                    ui.tags.ul(*issue_items),
                    class_=f"alert {alert_class}",
                )
            )

        # Grid summary
        if grid is not None:
            items.append(
                ui.div(
                    ui.h5("Grid"),
                    ui.p(f"Patches: {grid.n_patches}"),
                    ui.p(f"Type: {input.wizard_grid_type()}"),
                    ui.p(f"Cell size: {input.wizard_cell_size()} km"),
                )
            )
        else:
            items.append(ui.p("Grid not configured.", class_="text-warning"))

        # Habitats summary
        if hab is not None:
            unique = sorted(set(hab))
            items.append(
                ui.div(
                    ui.h5("Habitats"),
                    ui.p(f"Types: {len(unique)} ({', '.join(unique)})"),
                )
            )

        # Environment summary
        env_items = []
        if depth is not None:
            env_items.append(f"Depth: {depth.min():.0f}m to {depth.max():.0f}m")
        if sal is not None:
            env_items.append(f"Salinity: loaded ({sal.name})")
        if env_items:
            items.append(
                ui.div(
                    ui.h5("Environment"),
                    *[ui.p(e) for e in env_items],
                )
            )

        # Preferences summary
        if prefs is not None:
            items.append(
                ui.div(
                    ui.h5("Preferences"),
                    ui.p(
                        f"Matrix: {prefs.shape[0]} groups "
                        f"\u00d7 {prefs.shape[1]} habitats"
                    ),
                )
            )

        # Dispersal summary
        items.append(
            ui.div(
                ui.h5("Dispersal"),
                ui.p(
                    f"Default rate: {input.wizard_dispersal_default()} km\u00b2/month"
                ),
                ui.p(f"Gravity: {input.wizard_gravity()}"),
            )
        )

        return ui.div(*items)

    @reactive.effect
    @reactive.event(input.wizard_create)
    def _create_ecospace_model():
        """Build EcospaceParams from wizard state."""
        grid = ecospace_grid.get()
        hab = habitat_types_arr.get()
        depth = depth_per_patch.get()
        prefs = preference_matrix.get()

        if grid is None or hab is None:
            logger.warning("Cannot create model: missing grid or habitat data")
            return

        try:
            import numpy as np
            from pypath.io.marine_data import HabitatPreferenceBuilder
            from pypath.spatial.ecospace_params import EcospaceParams
            from pypath.spatial.environmental import (
                EnvironmentalDrivers,
                EnvironmentalLayer,
            )

            # Get group names
            group_names = shared_group_names(shared_data)

            n_groups = len(group_names)
            habitat_types = sorted(set(hab))

            # Build per-patch preference matrix if we have type-level prefs
            if prefs is not None:
                builder = HabitatPreferenceBuilder()
                hab_pref = builder.build_preference_matrix(
                    prefs, habitat_types, hab, grid
                )
            else:
                hab_pref = np.ones((n_groups, grid.n_patches))

            # Build dispersal arrays
            default_rate = input.wizard_dispersal_default()
            gravity = input.wizard_gravity()
            dispersal_rate = np.full(n_groups, default_rate)
            advection_enabled = np.ones(n_groups, dtype=bool)
            gravity_strength = np.full(n_groups, gravity)

            # Build environmental drivers
            layers = []
            if depth is not None:
                layers.append(EnvironmentalLayer(name="depth", units="m", values=depth))
            sal = salinity_layer.get()
            if sal is not None:
                layers.append(sal)
            env = EnvironmentalDrivers(layers=layers) if layers else None

            # Build EcospaceParams
            params = EcospaceParams(
                grid=grid,
                habitat_preference=hab_pref,
                habitat_capacity=hab_pref.copy(),
                dispersal_rate=dispersal_rate,
                advection_enabled=advection_enabled,
                gravity_strength=gravity_strength,
                environmental_drivers=env,
            )

            # Hand the configuration to the Ecospace page via the shared
            # reactive value it watches.
            if shared_data is not None and hasattr(shared_data, "ecospace_params"):
                shared_data.ecospace_params.set(params)

            logger.info(
                "Created EcospaceParams: %d patches, %d groups, %d habitat types",
                grid.n_patches,
                n_groups,
                len(habitat_types),
            )
            _notify_legacy(
                f"Ecospace configuration ready: {grid.n_patches} patches, "
                f"{n_groups} groups. Open the Ecospace page to run it.",
                type="message",
                duration=8,
            )
        except Exception as e:
            logger.exception("Failed to create EcospaceParams")
            _notify_legacy(
                f"Could not build the Ecospace configuration: {e!s}",
                type="error",
                duration=8,
            )

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
        ui.div(
            ui.input_action_button(
                "wizard_download", "Download Data", class_="btn-primary"
            ),
            ui.output_ui("btn_cancel_download_ui"),
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
