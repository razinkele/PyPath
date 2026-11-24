"""Example models shipped with PyPath.

Three EwE-F models from EPath's test cases, each with its Ecopath inputs
(scenario, diet, detritus fate, growth) and its Ecosim settings
(vulnerabilities, group settings, fishing forcing). Each model's folder has a
README with its source and the changes made to it. The Curonian Lagoon model
is also a spatial model: its own Ecospace set-up on a grid of 1 km cells
with a habitat map per group (``example_ecospace``), and as an alternative
grid the 29 boxes of a SHYFEM box model (``example_ecospace_grid``).

Examples
--------
>>> from pypath.examples import list_example_models, load_example_model
>>> [m.name for m in list_example_models()]
['Central_Baltic', 'Curonian_Lagoon', 'Lithuanian_Coast']
>>> params = load_example_model("Central_Baltic")  # RpathParams, for rpath()
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from pypath.core.ecosim import RsimScenario
    from pypath.core.params import RpathParams
    from pypath.spatial import EcospaceGrid, EcospaceParams

DATA = Path(__file__).parent / "data"


@dataclass(frozen=True)
class ExampleModel:
    """An example model: its name, what it is and where it comes from.

    Attributes
    ----------
    name : str
        The name to load it by, also its folder in ``pypath/examples/data``.
    title : str
        A short title.
    description : str
        The food web, its groups and the changes made to the source model.
    source : str
        The publication or institution the model comes from.
    """

    name: str
    title: str
    description: str
    source: str

    @property
    def path(self) -> Path:
        """The model's folder."""
        return DATA / self.name

    @property
    def ecopath_path(self) -> Path:
        """The EwE-F Ecopath inputs (folder with the ``filenames.nml``)."""
        return self.path / "ecopath"

    @property
    def ecosim_path(self) -> Path:
        """The EwE-F Ecosim inputs (folder with the ``filenames.nml``)."""
        return self.path / "ecosim"

    @property
    def ecospace_path(self) -> Optional[Path]:
        """The EwE-F Ecospace inputs (grid, depth, habitat maps), or None."""
        folder = self.path / "ecospace"
        return folder if any(folder.glob("*_grid.csv")) else None

    @property
    def grid_path(self) -> Optional[Path]:
        """The model's SHYFEM mesh (``.grd``) for Ecospace, or None."""
        meshes = sorted((self.path / "ecospace").glob("*.grd"))
        return meshes[0] if meshes else None


_CATALOG = (
    ExampleModel(
        name="Central_Baltic",
        title="Central Baltic Sea, 1974",
        description=(
            "Food web of the Baltic Proper, 22 groups, with sprat, herring and "
            "cod as multi-stanza sets. Modified for a stationary baseline: each "
            "set's biomass accumulation is added to its stages' P/B."
        ),
        source=(
            "Tomczak, M. T., Niiranen, S., Hjerne, O., & Blenckner, T. (2012). "
            "Ecosystem flow dynamics in the Baltic Proper: Using a multi-trophic "
            "dataset as a basis for food-web modelling. Ecological Modelling, "
            "230, 123-147. https://doi.org/10.1016/j.ecolmodel.2011.12.014 "
            "(modified)"
        ),
    ),
    ExampleModel(
        name="Curonian_Lagoon",
        title="Curonian Lagoon pelagic food web, 2005-2007",
        description=(
            "Pelagic food web of the Curonian Lagoon, 19 groups (17 consumers, "
            "phytoplankton and detritus) and one fleet."
        ),
        source=(
            "EwE 6 model 'curonian pelagic', Marine Research Institute, "
            "Klaipeda University (no peer-reviewed reference)"
        ),
    ),
    ExampleModel(
        name="Lithuanian_Coast",
        title="Lithuanian sandy coast",
        description=(
            "Food web of the exposed sandy coast of Lithuania (0-20 m, Nida to "
            "Butinge), 44 groups, with round goby added as a three-stage "
            "multi-stanza set and immigration for 17 groups."
        ),
        source=(
            "Razinkovas-Baziukas, A., Morkune, R., Bacevicius, E., & "
            "Gasiunaite, Z. R. (2017). Trophic network model of exposed sandy "
            "coast: Linking continental and marine water ecosystems. Estuarine, "
            "Coastal and Shelf Science, 195, 110-123. "
            "https://doi.org/10.1016/j.ecss.2017.04.030; round goby from "
            "Skabeikis, A. (2019), doctoral dissertation, Klaipeda University"
        ),
    ),
)


def list_example_models() -> List[ExampleModel]:
    """The example models shipped with PyPath, in name order."""
    return list(_CATALOG)


def example_model(name: str) -> ExampleModel:
    """The example model of that name (case does not matter).

    Raises
    ------
    ValueError
        If there is no example of that name; the message lists the names.
    """
    wanted = str(name).strip().lower()
    for model in _CATALOG:
        if model.name.lower() == wanted:
            return model
    names = ", ".join(m.name for m in _CATALOG)
    raise ValueError(f"No example model {name!r}; the examples are: {names}")


def load_example_model(name: str) -> "RpathParams":
    """An example model's Ecopath parameters, ready for ``rpath``.

    Parameters
    ----------
    name : str
        An example's name (see :func:`list_example_models`).

    Returns
    -------
    RpathParams
        Read with :func:`pypath.io.ewef.read_ewef`.
    """
    from pypath.io.ewef import read_ewef

    return read_ewef(example_model(name).ecopath_path)


def example_ecosim_scenario(
    name: str,
    years: Optional[range] = None,
    fast_group_rule: Optional[str] = None,
) -> "RsimScenario":
    """An Ecosim scenario with the example's own Ecosim settings.

    Parameters
    ----------
    name : str
        An example's name (see :func:`list_example_models`).
    years : range, optional
        Simulation years; the default is the model's EwE-F run length.
    fast_group_rule : {'ewe', 'rpath'}, optional
        Passed to :func:`pypath.core.ecosim.rsim_scenario`. Default: by the
        model's origin (``compat``): ``'ewe'`` for EwE imports, ``'rpath'``
        otherwise, so ``'ewe'`` for these EwE-F models.

    Returns
    -------
    RsimScenario
        From :func:`pypath.io.ewef.ecosim_scenario_from_ewef`: the balanced
        model with the example's vulnerabilities, group settings and fishing.
    """
    from pypath.io.ewef import ecosim_scenario_from_ewef

    model = example_model(name)
    return ecosim_scenario_from_ewef(
        model.ecopath_path,
        model.ecosim_path,
        years=years,
        fast_group_rule=fast_group_rule,
    )


def example_ecospace(
    name: str, geometry: bool = False, cell_placement: str = "ground"
) -> "EcospaceParams":
    """An example's own Ecospace set-up, ready for a spatial run.

    The grid's water cells as patches, each group's habitat capacity from
    its habitat map, and a dispersal rate per group, with a row for every
    group of the model (fleets included). Use it with
    :func:`example_ecosim_scenario`:

    >>> scenario = example_ecosim_scenario("Curonian_Lagoon")
    >>> ecospace = example_ecospace("Curonian_Lagoon")
    >>> out = rsim_run_spatial(scenario, ecospace=ecospace)  # doctest: +SKIP

    It is PyPath's Ecospace on the case's grid and maps
    (:func:`pypath.io.ewef.read_ewef_ecospace` says what is read and what
    is chosen), not EwE-F's own movement.

    Parameters
    ----------
    name : str
        An example's name (see :func:`list_example_models`).
    geometry : bool
        Keep the cells as polygons, for maps (needs geopandas and shapely).
    cell_placement : {"ground", "ewe"}
        Where the cells are put; the run is the same either way.

    Returns
    -------
    EcospaceParams

    Raises
    ------
    ValueError
        If the example has no Ecospace set-up; the message names those
        that have one.
    """
    from pypath.io.ewef import read_ewef, read_ewef_ecospace

    model = example_model(name)
    if model.ecospace_path is None:
        with_one = ", ".join(m.name for m in _CATALOG if m.ecospace_path is not None)
        raise ValueError(
            f"The example {model.name} has no Ecospace set-up; the examples "
            f"with one are: {with_one}"
        )
    params = read_ewef(model.ecopath_path)
    ecospace = read_ewef_ecospace(
        model.ecospace_path,
        params,
        ecosim_path=model.ecosim_path,
        cell_placement=cell_placement,
        geometry=geometry,
    )
    return ecospace.with_group_rows(len(params.model))


def example_ecospace_grid(
    name: str, patches: str = "type", geometry: bool = False
) -> "EcospaceGrid":
    """An example's alternative grid for Ecospace, from its SHYFEM mesh.

    The boxes of a hydrodynamic box model, not the grid of the example's
    own Ecospace set-up (:func:`example_ecospace`). Only the grid: habitat
    maps and dispersal rates are yours to set (``EcospaceParams``). The element types of the mesh are the boxes of
    the hydrodynamic box model, so the model's box output
    (``pypath.io.shyfem.read_box_series``) fits the grid as it is.

    Parameters
    ----------
    name : str
        An example's name (see :func:`list_example_models`).
    patches : {'type', 'element'}
        One patch per element type (the boxes; default) or per element.
    geometry : bool
        Also make the patches' polygons (needs geopandas and shapely).

    Returns
    -------
    EcospaceGrid
        From :meth:`pypath.spatial.EcospaceGrid.from_shyfem_grd`.

    Raises
    ------
    ValueError
        If the example has no grid; the message names those that have one.
    """
    from pypath.spatial import EcospaceGrid

    model = example_model(name)
    if model.grid_path is None:
        with_grid = ", ".join(m.name for m in _CATALOG if m.grid_path is not None)
        raise ValueError(
            f"The example {model.name} has no spatial grid; the examples with "
            f"one are: {with_grid}"
        )
    return EcospaceGrid.from_shyfem_grd(
        model.grid_path, patches=patches, geometry=geometry
    )
