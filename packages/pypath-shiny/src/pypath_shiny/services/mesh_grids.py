"""SHYFEM meshes for the Ecospace page: what an uploaded ``.grd`` holds, and
the grid built from it.

A ``.grd`` file names no coordinate system. The page shows what was read
(units as detected, area, depth) before a grid is made, and a mesh in metres
needs an EPSG code: the maps are drawn in longitude and latitude.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from pypath.io.shyfem import ShyfemMesh, read_grd
from pypath.spatial import EcospaceGrid
from pypath.spatial.mesh import mesh_units

__all__ = [
    "MeshSummary",
    "build_mesh_grid",
    "estimated_run_seconds",
    "mesh_crs",
    "summarise_mesh",
]

# Seconds per patch, month and group of a spatial run on the server, from a
# 5-group model on 29 and on 2027 patches (about 1.2 ms per patch-month).
_SECONDS_PER_PATCH_MONTH_GROUP = 2.4e-4

UNIT_LABELS = {"degrees": "degrees (longitude, latitude)", "m": "metres", "km": "km"}


@dataclass
class MeshSummary:
    """What a ``.grd`` file holds, for the user to check before making a grid."""

    n_nodes: int
    n_elements: int
    n_types: int
    units: str  # "degrees", "m" or "km": as named, or as detected
    detected_units: str
    area_km2: float
    depth_min: Optional[float]
    depth_max: Optional[float]
    x_range: tuple[float, float]
    y_range: tuple[float, float]


def mesh_crs(units: str, epsg: Optional[str]) -> Optional[str]:
    """The coordinate system to build the grid with, or None for degrees.

    Raises ValueError for a mesh in metres or km without a usable EPSG code.
    """
    if units == "degrees":
        return None
    text = str(epsg or "").strip().upper().replace("EPSG:", "").strip()
    if not text.isdigit():
        raise ValueError(
            "This mesh is in metres or km. Enter the EPSG code of its "
            "coordinate system (for example 32634 for UTM zone 34N), so the "
            "map can place it."
        )
    return f"EPSG:{int(text)}"


def summarise_mesh(mesh: ShyfemMesh, units: str = "auto") -> MeshSummary:
    """Counts, units, area and depth range of a mesh."""
    detected = mesh_units(mesh.xy)
    used = detected if units == "auto" else units
    grid = EcospaceGrid.from_shyfem_grd(mesh, patches="type", units=used)
    depth = (
        grid.cell_metadata["depth"].to_numpy(dtype=float)
        if "depth" in grid.cell_metadata
        else np.array([np.nan])
    )
    known = np.isfinite(depth)
    return MeshSummary(
        n_nodes=mesh.n_nodes,
        n_elements=mesh.n_elements,
        n_types=int(len(np.unique(mesh.element_type))),
        units=used,
        detected_units=detected,
        area_km2=float(grid.patch_areas.sum()),
        depth_min=float(depth[known].min()) if known.any() else None,
        depth_max=float(depth[known].max()) if known.any() else None,
        x_range=(float(mesh.xy[:, 0].min()), float(mesh.xy[:, 0].max())),
        y_range=(float(mesh.xy[:, 1].min()), float(mesh.xy[:, 1].max())),
    )


def build_mesh_grid(
    path_or_mesh,
    patches: str = "type",
    units: str = "auto",
    epsg: Optional[str] = None,
) -> EcospaceGrid:
    """The Ecospace grid of an uploaded mesh, with polygons for the maps.

    ``patches`` is "type" (one patch per element type, the boxes of a SHYFEM
    box model) or "element". Raises ValueError when the mesh is in metres or
    km and no EPSG code is given.
    """
    mesh = (
        path_or_mesh if isinstance(path_or_mesh, ShyfemMesh) else read_grd(path_or_mesh)
    )
    used = mesh_units(mesh.xy) if units == "auto" else units
    return EcospaceGrid.from_shyfem_grd(
        mesh, patches=patches, units=used, crs=mesh_crs(used, epsg), geometry=True
    )


def estimated_run_seconds(n_patches: int, n_groups: int, years: float) -> float:
    """A rough run time of a spatial simulation on this many patches."""
    return _SECONDS_PER_PATCH_MONTH_GROUP * n_patches * n_groups * 12.0 * years
