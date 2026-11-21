"""The how-to "Ecospace on a hydrodynamic model's boxes" runs as written.

The page's Python blocks are run in order, as a reader would run them, and a
few things that must hold are checked on what they leave behind.
"""

import re
from pathlib import Path

import numpy as np
import pytest

PAGE = (
    Path(__file__).parent.parent / "docs" / "docs" / "examples" / "hydrodynamic-link.md"
)

pytestmark = [
    pytest.mark.skipif(not PAGE.is_file(), reason="the docs are not in this tree"),
    pytest.mark.filterwarnings("ignore"),
]


def _blocks(text):
    """The page's python blocks, dedented (some sit inside an admonition)."""
    out = []
    for match in re.finditer(r"^( *)```python\n(.*?)^\1```", text, re.S | re.M):
        indent = match.group(1)
        out.append(
            "\n".join(
                line[len(indent) :] if line.startswith(indent) else line
                for line in match.group(2).splitlines()
            )
        )
    return out


@pytest.fixture(scope="module")
def page():
    blocks = _blocks(PAGE.read_text(encoding="utf-8"))
    assert len(blocks) >= 10
    namespace = {"__name__": "howto"}
    for number, block in enumerate(blocks, 1):
        try:
            exec(compile(block, f"{PAGE.name} block {number}", "exec"), namespace)
        except Exception as error:  # the block's number is what helps here
            raise AssertionError(f"block {number} of {PAGE.name} failed: {error!r}")
    return namespace


def test_the_grids_and_layers_are_what_the_page_says(page):
    assert page["boxes"].n_patches == 4 and page["elements"].n_patches == 32
    assert page["layers"].layer_names == ["temp", "salt", "volume"]
    np.testing.assert_allclose(
        page["layers"].get_layer_at_time("salt", 0.0), [0.1, 0.6, 3.0, 7.0]
    )
    cod = page["capacity"][page["names"].index("Ad. Cod") - 1]
    assert cod[0] == 0.0 and cod[-1] == 1.0 and np.all(np.diff(cod) >= 0)


def test_the_substeps_the_page_uses_are_enough(page):
    assert page["fast"]  # the example has fast groups to integrate
    assert page["check"].converged and page["check"].substeps == 4


def test_the_three_runs_are_finite_and_differ_as_the_page_says(page):
    still, mixed, flushed = page["still"], page["mixed"], page["flushed"]
    boxes, phyto = page["boxes"], page["phyto"]
    for out in (still, mixed, flushed):
        assert np.isfinite(out.out_Biomass_spatial).all()

    def spread(out):
        density = out.out_Biomass_spatial[-1, phyto] / boxes.patch_areas
        return density.max() / density.min()

    # exchange evens the plankton out and barely moves its total
    assert spread(mixed) < spread(still)
    assert mixed.out_Biomass[-1, phyto] == pytest.approx(
        still.out_Biomass[-1, phyto], rel=0.01
    )
    # throughflow flushes it: the total falls, the river end most
    assert flushed.out_Biomass[-1, phyto] < 0.98 * mixed.out_Biomass[-1, phyto]
    density = flushed.out_Biomass_spatial[-1, phyto] / boxes.patch_areas
    assert density[0] < density[-1]


def test_exchange_itself_keeps_a_total_and_throughflow_dilutes(page):
    exchange, throughflow = page["exchange"], page["throughflow"]
    assert exchange.throughflow is False and exchange.rest == "capacity"
    assert throughflow.open_boundaries == [-2]  # the river
    totals = np.array([4.0, 1.0, 0.5, 0.0])
    moved = exchange.step_zones(totals, 0, exchange.volume[0], 1 / 12)
    assert moved.sum() == pytest.approx(totals.sum(), rel=1e-12)
    flushed = throughflow.step_zones(totals, 0, throughflow.volume[0], 1 / 12)
    assert flushed.sum() < totals.sum()


def test_the_other_sources_and_the_longer_run(page):
    assert page["longer"].first_month == 60 and page["longer"].beyond == "mean_year"
    assert page["links"].zones.tolist() == [1, 2]
    assert page["links"].boundary_id.tolist() == [-2]
