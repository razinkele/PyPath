"""Ecospace grids from finite-element meshes.

A mesh of triangles (or other polygons) becomes an ``EcospaceGrid`` with one
patch per element, or one patch per zone of elements: SHYFEM marks zones by
the element type, which a box model uses as the box number. A zone's area is
the sum of its elements', the border between two zones is the total length
of the element sides they share, and its depth the area-weighted mean.

Lengths and areas are in km. Degrees are projected first, by an azimuthal
equidistant projection about the middle of the mesh: within about 500 km of
that point lengths and areas are right to about 0.1 %. The sphere's radius
is 6371.0088 km; SHYFEM's own element areas are about 0.2 % larger.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import scipy.sparse

if TYPE_CHECKING:
    from pypath.io.shyfem import ShyfemMesh
    from pypath.spatial.ecospace_params import EcospaceGrid

logger = logging.getLogger(__name__)

__all__ = [
    "dispersal_rate_for_grid",
    "grid_from_mesh",
    "grid_from_shyfem_grd",
    "mesh_units",
]

_EARTH_RADIUS_KM = 6371.0088
_UNITS = ("auto", "degrees", "m", "km")


def dispersal_rate_for_grid(rate, from_grid: "EcospaceGrid", to_grid: "EcospaceGrid"):
    """The dispersal rate on ``to_grid`` that spreads biomass as ``rate`` does on ``from_grid``.

    Ecospace's dispersal rate is an exchange rate between neighbouring
    patches: the flow is rate x mean patch area x border / distance x the
    density difference (``pypath.spatial.dispersal.diffusion_flux``). The
    diffusivity in km2 per year is therefore rate x mean patch area, and the
    same rate spreads biomass more slowly on a finer grid. This returns
    ``rate x mean area of from_grid / mean area of to_grid``, which keeps
    the diffusivity: a release in one box of a 29-box mesh then spreads the
    same, within a few percent, on its 2027 elements.
    """
    ratio = float(np.mean(from_grid.patch_areas)) / float(np.mean(to_grid.patch_areas))
    return np.asarray(rate, dtype=float) * ratio


def mesh_units(xy: np.ndarray) -> str:
    """ "degrees" if the coordinates can be longitude and latitude, else "m"."""
    xy = np.asarray(xy, dtype=float)
    if np.all(np.abs(xy[:, 0]) <= 360.0) and np.all(np.abs(xy[:, 1]) <= 90.0):
        return "degrees"
    return "m"


def _to_km(xy: np.ndarray, units: str) -> np.ndarray:
    """Plane coordinates in km."""
    if units == "km":
        return xy.astype(float)
    if units == "m":
        return xy / 1000.0
    lon, lat = np.deg2rad(xy[:, 0]), np.deg2rad(xy[:, 1])
    lon0 = 0.5 * (lon.min() + lon.max())
    lat0 = 0.5 * (lat.min() + lat.max())
    cos_c = np.sin(lat0) * np.sin(lat) + np.cos(lat0) * np.cos(lat) * np.cos(lon - lon0)
    c = np.arccos(np.clip(cos_c, -1.0, 1.0))
    with np.errstate(invalid="ignore", divide="ignore"):
        k = np.where(c > 1e-12, c / np.sin(c), 1.0)
    x = k * np.cos(lat) * np.sin(lon - lon0)
    y = k * (
        np.cos(lat0) * np.sin(lat) - np.sin(lat0) * np.cos(lat) * np.cos(lon - lon0)
    )
    return _EARTH_RADIUS_KM * np.column_stack([x, y])


def _polygon_area_centroid(points: np.ndarray, elements: np.ndarray):
    """Area and centroid of each element (padded with -1) by the shoelace sums."""
    n = (elements >= 0).sum(axis=1)
    safe = np.where(elements >= 0, elements, elements[:, :1])
    p = points[safe]  # [n_elements, width, 2]; padding repeats the first node
    q = np.roll(p, -1, axis=1)
    # the vertex after the last real one is the first
    last = n - 1
    rows = np.arange(len(elements))
    q[rows, last] = p[rows, 0]
    cross = p[..., 0] * q[..., 1] - q[..., 0] * p[..., 1]
    valid = np.arange(elements.shape[1])[None, :] < n[:, None]
    cross = np.where(valid, cross, 0.0)
    signed = 0.5 * cross.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cx = ((p[..., 0] + q[..., 0]) * cross).sum(axis=1) / (6.0 * signed)
        cy = ((p[..., 1] + q[..., 1]) * cross).sum(axis=1) / (6.0 * signed)
    return np.abs(signed), np.column_stack([cx, cy])


def grid_from_mesh(
    xy: np.ndarray,
    elements: np.ndarray,
    zones: Optional[np.ndarray] = None,
    *,
    units: str = "auto",
    element_depth: Optional[np.ndarray] = None,
    crs: Optional[str] = None,
    geometry: bool = False,
) -> "EcospaceGrid":
    """Build an Ecospace grid from mesh nodes and elements.

    Parameters
    ----------
    xy : np.ndarray
        Node coordinates [n_nodes, 2]: longitude and latitude in degrees, or
        x and y in metres or km.
    elements : np.ndarray
        Node positions of each element [n_elements, n]; pad a shorter
        element with -1.
    zones : np.ndarray, optional
        A zone id per element. Elements of one zone make one patch. Without
        it every element is a patch.
    units : {"auto", "degrees", "m", "km"}
        Units of ``xy``. "auto" takes degrees when every coordinate can be a
        longitude and latitude, metres otherwise.
    element_depth : np.ndarray, optional
        Depth of each element; a patch gets the area-weighted mean.
    crs : str, optional
        Coordinate system of ``xy``. Degrees are "EPSG:4326" unless named.
        Metres and km without one make a "local" grid, whose centroids stay
        in the mesh's units: the run is right, but maps and the samplers of
        ``pypath.io.marine_data`` need longitude and latitude. With one
        (e.g. "EPSG:32630"; needs pyproj) the centroids and polygons are
        given in longitude and latitude and the grid is "EPSG:4326". Areas
        and lengths come from the mesh's own plane either way.
    geometry : bool
        Also build the patches' polygons as a GeoDataFrame (needs geopandas;
        a zone's polygon is the union of its elements).

    Returns
    -------
    EcospaceGrid
        ``patch_ids`` are the zone ids (or 0..n-1 per element),
        ``patch_centroids`` are in the units of ``xy`` (see ``crs``), ``centroid_km``
        holds them in km for distances, and ``cell_metadata`` has ``zone``,
        ``n_elements`` and ``depth`` per patch.
    """
    import pandas as pd

    from pypath.spatial.ecospace_params import EcospaceGrid

    if units not in _UNITS:
        raise ValueError(f"units must be one of {_UNITS}, not {units!r}")
    xy = np.asarray(xy, dtype=float)
    elements = np.asarray(elements, dtype=int)
    if units == "auto":
        units = mesh_units(xy)
    n_elements = len(elements)
    if zones is None:
        zone_ids = np.arange(n_elements)
        patch_of = np.arange(n_elements)
    else:
        zones = np.asarray(zones)
        if zones.shape != (n_elements,):
            raise ValueError(
                f"zones needs one value per element ({n_elements}), "
                f"got shape {zones.shape}"
            )
        zone_ids, patch_of = np.unique(zones, return_inverse=True)
    n_patches = len(zone_ids)

    km = _to_km(xy, units)
    area_e, centroid_e_km = _polygon_area_centroid(km, elements)
    if np.any(area_e <= 0) or not np.all(np.isfinite(centroid_e_km)):
        bad = int(np.flatnonzero(~(area_e > 0))[0]) if np.any(area_e <= 0) else -1
        raise ValueError(f"element at position {bad} has no area")
    _, centroid_e_xy = _polygon_area_centroid(xy - xy.mean(axis=0), elements)
    centroid_e_xy = centroid_e_xy + xy.mean(axis=0)

    area = np.bincount(patch_of, weights=area_e, minlength=n_patches)

    def _mean(values):
        return np.column_stack(
            [
                np.bincount(
                    patch_of, weights=area_e * values[:, k], minlength=n_patches
                )
                / area
                for k in range(values.shape[1])
            ]
        )

    centroid_km = _mean(centroid_e_km)
    centroid_xy = _mean(centroid_e_xy)

    # element sides: a side two elements share is an interface
    n_vert = (elements >= 0).sum(axis=1)
    width = elements.shape[1]
    owner = np.repeat(np.arange(n_elements), width)
    a = elements.reshape(-1)
    nxt = np.roll(elements, -1, axis=1)
    nxt[np.arange(n_elements), n_vert - 1] = elements[:, 0]
    b = nxt.reshape(-1)
    real = (np.arange(width)[None, :] < n_vert[:, None]).reshape(-1)
    owner, a, b = owner[real], a[real], b[real]
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    key = lo.astype(np.int64) * (len(xy) + 1) + hi
    order = np.argsort(key, kind="stable")
    key_s, owner_s, lo_s, hi_s = key[order], owner[order], lo[order], hi[order]
    same = key_s[1:] == key_s[:-1]
    if np.any(same[1:] & same[:-1]):
        raise ValueError("a side belongs to more than two elements")
    first = np.flatnonzero(same)
    p, q = patch_of[owner_s[first]], patch_of[owner_s[first + 1]]
    length = np.linalg.norm(km[lo_s[first]] - km[hi_s[first]], axis=1)
    between = p != q
    p, q, length = p[between], q[between], length[between]
    i, j = np.minimum(p, q), np.maximum(p, q)
    border = scipy.sparse.coo_matrix(
        (length, (i, j)), shape=(n_patches, n_patches)
    ).tocsr()  # duplicate (i, j) entries are summed
    border.sum_duplicates()
    rows, cols = border.nonzero()
    lengths = np.asarray(border[rows, cols]).reshape(-1)
    edge_lengths = {(int(r), int(c)): float(v) for r, c, v in zip(rows, cols, lengths)}
    adjacency = scipy.sparse.csr_matrix(
        (np.ones(2 * len(rows)), (np.r_[rows, cols], np.r_[cols, rows])),
        shape=(n_patches, n_patches),
    )

    meta = pd.DataFrame(
        {
            "zone": zone_ids,
            "n_elements": np.bincount(patch_of, minlength=n_patches),
        }
    )
    if element_depth is not None:
        depth = np.asarray(element_depth, dtype=float)
        known = np.isfinite(depth)
        weight = np.bincount(patch_of, weights=area_e * known, minlength=n_patches)
        total = np.bincount(
            patch_of, weights=area_e * np.where(known, depth, 0.0), minlength=n_patches
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            meta["depth"] = np.where(weight > 0, total / weight, np.nan)

    map_xy = xy
    if units == "degrees":
        crs = crs or "EPSG:4326"
    elif crs is None:
        crs = "local"
    else:
        # a projected mesh with a named system: centroids and polygons go to
        # longitude and latitude, which the maps and the data samplers read
        map_xy = _to_lonlat(xy * (1000.0 if units == "km" else 1.0), crs)
        _, centroid_e_xy = _polygon_area_centroid(
            map_xy - map_xy.mean(axis=0), elements
        )
        centroid_xy = _mean(centroid_e_xy + map_xy.mean(axis=0))
        crs = "EPSG:4326"
    geo = None
    if geometry:
        geo = _geometry(map_xy, elements, n_vert, patch_of, zone_ids, crs)

    return EcospaceGrid(
        n_patches=n_patches,
        patch_ids=zone_ids,
        patch_areas=area,
        patch_centroids=centroid_xy,
        adjacency_matrix=adjacency,
        edge_lengths=edge_lengths,
        crs=crs,
        geometry=geo,
        cell_metadata=meta,
        centroid_km=centroid_km,
    )


def _to_lonlat(xy_m: np.ndarray, crs: str) -> np.ndarray:
    """Coordinates in metres of ``crs`` as longitude and latitude."""
    try:
        from pyproj import Transformer
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "a mesh in metres with a crs needs pyproj to find longitude and "
            "latitude: pip install pyproj (or leave crs out for a local grid)"
        ) from exc
    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(xy_m[:, 0], xy_m[:, 1])
    return np.column_stack([lon, lat])


def _geometry(xy, elements, n_vert, patch_of, zone_ids, crs):
    """Patch polygons as a GeoDataFrame, in the mesh's own coordinates."""
    try:
        import geopandas as gpd
        from shapely import union_all
        from shapely.geometry import Polygon
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "geometry=True needs geopandas and shapely: pip install geopandas shapely"
        ) from exc
    polygons = [Polygon(xy[e[:n]]) for e, n in zip(elements, n_vert)]
    if len(zone_ids) == len(polygons):
        shapes = polygons
    else:
        members = [[] for _ in zone_ids]
        for polygon, patch in zip(polygons, patch_of):
            members[patch].append(polygon)
        shapes = [union_all(group) for group in members]
    return gpd.GeoDataFrame(
        {"id": zone_ids}, geometry=shapes, crs=None if crs == "local" else crs
    )


def grid_from_shyfem_grd(
    path: Union[str, Path, "ShyfemMesh"],
    patches: Union[str, np.ndarray] = "element",
    *,
    units: str = "auto",
    crs: Optional[str] = None,
    geometry: bool = False,
) -> "EcospaceGrid":
    """Build an Ecospace grid from a SHYFEM ``.grd`` mesh.

    Depth is each element's own value, or else the mean of its nodes'.

    Parameters
    ----------
    path : str, Path or ShyfemMesh
        The ``.grd`` file, or a mesh already read by ``pypath.io.shyfem.read_grd``.
    patches : {"element", "type"} or np.ndarray
        "element": one patch per element. "type": one patch per element
        type, which is the box number in a SHYFEM box model. An array gives
        a zone id per element, in the file's element order.
    units, crs, geometry
        As in ``grid_from_mesh``. A ``.grd`` file names no coordinate
        system, so check ``units`` when the automatic choice could be wrong.
    """
    from pypath.io.shyfem import ShyfemMesh, read_grd

    mesh = path if isinstance(path, ShyfemMesh) else read_grd(path)
    if isinstance(patches, str):
        if patches == "element":
            zones = None
        elif patches == "type":
            zones = mesh.element_type
        else:
            raise ValueError(
                f'patches must be "element", "type" or an array, not {patches!r}'
            )
    else:
        zones = np.asarray(patches)
        if zones.shape != (mesh.n_elements,):
            raise ValueError(
                f"patches needs one value per element ({mesh.n_elements}), "
                f"got shape {zones.shape}"
            )
    safe = np.where(mesh.elements >= 0, mesh.elements, mesh.elements[:, :1])
    node_mean = np.where(mesh.elements >= 0, mesh.node_depth[safe], np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows stay NaN
        from_nodes = np.nanmean(node_mean, axis=1)
    depth = np.where(np.isfinite(mesh.element_depth), mesh.element_depth, from_nodes)
    return grid_from_mesh(
        mesh.xy,
        mesh.elements,
        zones,
        units=units,
        element_depth=depth if np.any(np.isfinite(depth)) else None,
        crs=crs,
        geometry=geometry,
    )
