"""Example models shipped with PyPath.

Three EwE-F models from EPath's test cases, each with its Ecopath inputs
(scenario, diet, detritus fate, growth) and its Ecosim settings
(vulnerabilities, group settings, fishing forcing). Each model's folder has a
README with its source and the changes made to it.

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
    fast_group_rule : str, optional
        Passed to :func:`pypath.core.ecosim.rsim_scenario`.

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
