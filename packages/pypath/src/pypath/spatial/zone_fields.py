"""Fields given per zone, on the patches of an Ecospace grid.

A hydrodynamic box model, a monitoring programme or a table of basin means
gives a value per zone and time. ``zone_series_layer`` turns such a series
into an ``EnvironmentalLayer`` of monthly means on a grid: each patch takes
the value of its zone, so the same series serves a grid of boxes and a grid
of the mesh elements inside them. ``shyfem_box_layers`` does it for every
variable of a SHYFEM box model's output. ``capacity_from_layers`` makes a
habitat capacity from layers and response functions.

Time in a layer is in years from a start date, as in a spatial run; a
month's mean is held for the whole month (month k covers k/12 up to
(k+1)/12).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Mapping, Optional, Sequence, Union

import numpy as np

from pypath.spatial.environmental import EnvironmentalDrivers, EnvironmentalLayer

if TYPE_CHECKING:
    from pypath.io.shyfem import ShyfemBoxSeries
    from pypath.spatial.ecospace_params import EcospaceGrid

logger = logging.getLogger(__name__)

__all__ = [
    "capacity_from_layers",
    "patch_zones",
    "shyfem_box_layers",
    "zone_series_layer",
]

# Units of the variables SHYFEM's box model writes
_SHYFEM_UNITS = {
    "temp": "celsius",
    "salt": "psu",
    "cur_speed": "m/s",
    "act_eta": "m",
    "aver_eta": "m",
    "bstress": "N/m2",
    "volume": "m3",
}


def patch_zones(grid: "EcospaceGrid") -> np.ndarray:
    """The zone each patch of a grid lies in.

    A grid made from a SHYFEM mesh knows it (``cell_metadata["type"]``: the
    element type, which is the box number in a box model). Other grids are
    taken to have one patch per zone, numbered by ``patch_ids``.
    """
    meta = getattr(grid, "cell_metadata", None)
    if meta is not None and "type" in meta:
        return np.asarray(meta["type"])
    return np.asarray(grid.patch_ids)


def _month_index(times: np.ndarray) -> np.ndarray:
    """Calendar month of each stamp as months since 1970-01."""
    return times.astype("datetime64[M]").astype(int)


def zone_series_layer(
    times,
    values,
    zone_ids,
    grid: "EcospaceGrid",
    *,
    name: str,
    units: str = "",
    start=None,
    zones_of_patches: Optional[np.ndarray] = None,
    stamped_at_end: bool = True,
) -> EnvironmentalLayer:
    """Monthly means of a per-zone time series as a layer on a grid.

    Parameters
    ----------
    times : array of datetime64
        Time stamp of each record [n_times].
    values : np.ndarray
        The series [n_times, n_zones].
    zone_ids : np.ndarray
        The zone of each column of ``values`` [n_zones], in any order.
    grid : EcospaceGrid
        The grid whose patches get the values.
    name, units : str
        Name and units of the layer.
    start : str or datetime64, optional
        The date of time 0 of the run the layer is for. Months before it
        are dropped. Default: the first month in the series.
    zones_of_patches : np.ndarray, optional
        The zone of each patch [n_patches]. Default: ``patch_zones(grid)``.
    stamped_at_end : bool
        Records are stamped at the end of their interval (as SHYFEM's are):
        each counts in the month of its stamp, except one stamped at the
        first instant of a month, which closes the month before. With False
        a record always counts in the month of its stamp.

    Returns
    -------
    EnvironmentalLayer
        ``values`` [n_months, n_patches] with ``times`` at the start of
        each month, in years from ``start``, and ``hold`` set: month k's
        mean holds from k/12 up to (k+1)/12. Before the first month the
        first mean is given, after the last month the last.

    Raises
    ------
    ValueError
        When a patch's zone is not in ``zone_ids``, or the series has no
        data on or after ``start``.
    """
    times = np.asarray(times).astype("datetime64[s]")
    values = np.asarray(values, dtype=float)
    zone_ids = np.asarray(zone_ids)
    if values.shape != (len(times), len(zone_ids)):
        raise ValueError(
            f"values must be [n_times, n_zones] = ({len(times)}, {len(zone_ids)}), "
            f"got {values.shape}"
        )
    zones = (
        patch_zones(grid) if zones_of_patches is None else np.asarray(zones_of_patches)
    )
    if zones.shape != (grid.n_patches,):
        raise ValueError(
            f"zones_of_patches needs one value per patch ({grid.n_patches})"
        )
    order = np.argsort(zone_ids, kind="stable")
    sorted_ids = zone_ids[order]
    where = np.searchsorted(sorted_ids, zones)
    where = np.clip(where, 0, len(sorted_ids) - 1)
    missing = np.unique(zones[sorted_ids[where] != zones])
    if len(missing):
        listed = ", ".join(str(z) for z in missing[:10])
        more = "" if len(missing) <= 10 else f" and {len(missing) - 10} more"
        raise ValueError(
            f"the series has no values for zone{'s' if len(missing) > 1 else ''} "
            f"{listed}{more}, which the grid's patches lie in"
        )
    column = order[where]  # the series column of each patch

    # A record stamped at the end of its interval counts in the month that
    # interval ends in; one stamped at the first instant of a month closes
    # the month before (the rule of read_box_series(monthly=True)).
    stamps = times - np.timedelta64(1, "s") if stamped_at_end else times
    month = _month_index(stamps)
    first = (
        int(month.min())
        if start is None
        else int(_month_index(np.array([np.datetime64(start, "s")]))[0])
    )
    keep = month >= first
    if not keep.any():
        raise ValueError(f"the series has no data on or after {start}")
    months = np.unique(month[keep])
    means = np.empty((len(months), len(zone_ids)))
    for k, m in enumerate(months):
        means[k] = values[month == m].mean(axis=0)
    return EnvironmentalLayer(
        name=name,
        units=units,
        values=means[:, column],
        times=(months - first) / 12.0,
        hold=True,
    )


def shyfem_box_layers(
    series: "ShyfemBoxSeries",
    grid: "EcospaceGrid",
    *,
    start=None,
    variables: Optional[Sequence[str]] = None,
    zones_of_patches: Optional[np.ndarray] = None,
) -> EnvironmentalDrivers:
    """A SHYFEM box model's output as monthly layers on a grid.

    Parameters
    ----------
    series : ShyfemBoxSeries
        From ``pypath.io.shyfem.read_box_series``.
    grid : EcospaceGrid
        A grid of the boxes (``from_shyfem_grd(..., patches="type")``), of
        the mesh's elements, or any grid with ``zones_of_patches``.
    start : str or datetime64, optional
        The date of time 0 of the run. Default: the series' first month.
    variables : sequence of str, optional
        The variables to take. Default: all in the series.
    zones_of_patches : np.ndarray, optional
        The box of each patch. Default: ``patch_zones(grid)``.

    Returns
    -------
    EnvironmentalDrivers
        One layer per variable, named as in the series (``temp``, ``salt``,
        ``cur_speed``, ...).
    """
    names = list(series.variables) if variables is None else list(variables)
    layers = {}
    for name in names:
        layers[name] = zone_series_layer(
            series.times,
            series.variable(name),
            series.boxes,
            grid,
            name=name,
            units=_SHYFEM_UNITS.get(name, ""),
            start=start,
            zones_of_patches=zones_of_patches,
            # monthly means made while reading are stamped with their month
            stamped_at_end=not getattr(series, "monthly", False),
        )
    return EnvironmentalDrivers(layers)


def capacity_from_layers(
    drivers: Union[EnvironmentalDrivers, Mapping[str, EnvironmentalLayer]],
    responses: Mapping[Union[int, str], Mapping[str, Callable]],
    *,
    n_groups: Optional[int] = None,
    group_names: Optional[Sequence[str]] = None,
    when: Union[str, float] = "mean",
    combine: str = "multiplicative",
) -> np.ndarray:
    """Habitat capacity of each group in each patch from environmental layers.

    A group's capacity is the product (or the minimum) of its responses to
    the layers it is given one for; a group without responses has capacity 1
    everywhere. The result is what ``EcospaceParams.habitat_capacity`` takes:
    it does not change during a run. Ecospace uses the pattern of a group's
    capacity over the map, not its level.

    Parameters
    ----------
    drivers : EnvironmentalDrivers or mapping of name to EnvironmentalLayer
        The layers.
    responses : mapping
        ``{group: {layer name: response function}}``. A group is its row
        (0 is the first group) or, with ``group_names``, its name. A
        response function takes the layer's values and returns suitability
        between 0 and 1 (``create_linear_response`` and the like).
    n_groups : int, optional
        Number of rows of the result. Default: ``len(group_names)``.
    group_names : sequence of str, optional
        The groups' names, in row order.
    when : "mean" or float
        "mean": the mean over the layers' months of the suitability (not
        the suitability at the mean conditions). A number: the time, in
        years, at which the layers are read.
    combine : {"multiplicative", "minimum"}
        How a group's responses to several layers are combined.

    Returns
    -------
    np.ndarray
        Capacity [n_groups, n_patches], between 0 and 1.
    """
    layers = (
        drivers.layers if isinstance(drivers, EnvironmentalDrivers) else dict(drivers)
    )
    if combine not in ("multiplicative", "minimum"):
        raise ValueError(
            f'combine must be "multiplicative" or "minimum", not {combine!r}'
        )
    if n_groups is None:
        if group_names is None:
            raise ValueError("give n_groups or group_names")
        n_groups = len(group_names)
    if not layers:
        raise ValueError("no layers given")
    n_patches = next(iter(layers.values())).n_patches
    capacity = np.ones((n_groups, n_patches))
    for group, by_layer in responses.items():
        if isinstance(group, str):
            if group_names is None or group not in list(group_names):
                raise KeyError(f"no group named {group!r} in group_names")
            row = list(group_names).index(group)
        else:
            row = int(group)
            if not 0 <= row < n_groups:
                raise KeyError(f"group row {row} is outside 0..{n_groups - 1}")
        if not by_layer:
            continue
        for layer_name in by_layer:
            if layer_name not in layers:
                raise KeyError(
                    f"no layer {layer_name!r}; the layers are {sorted(layers)}"
                )
        if when == "mean":
            moments = sorted(
                {
                    float(t)
                    for name in by_layer
                    if layers[name].is_time_varying
                    for t in layers[name].times
                }
            ) or [0.0]
        else:
            moments = [float(when)]
        total = np.zeros(n_patches)
        for t in moments:
            parts = [
                np.clip(
                    np.asarray(fn(layers[name].get_value_at_time(t)), dtype=float),
                    0.0,
                    1.0,
                )
                for name, fn in by_layer.items()
            ]
            total += (
                np.prod(parts, axis=0)
                if combine == "multiplicative"
                else np.min(parts, axis=0)
            )
        capacity[row] = total / len(moments)
    return capacity
