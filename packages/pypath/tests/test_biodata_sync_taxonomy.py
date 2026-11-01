"""sync_taxonomy_with_rpath used to raise UnboundLocalError on every call.

A function-local ``if TYPE_CHECKING: from ... import RpathParams`` made the
name local to the function, so the isinstance check failed at runtime.
"""

import pytest

from pypath.core.params import create_rpath_params
from pypath.io.biodata import sync_taxonomy_with_rpath


def test_sync_taxonomy_attaches_linkages_in_order():
    params = create_rpath_params(["Fish", "Plankton", "Detritus"], [0, 1, 2])
    links = ["link-fish", "link-plankton"]
    out = sync_taxonomy_with_rpath(params, links)
    assert out is params
    assert list(out.model["taxonomy_linkage"][:2]) == links
    assert out.model["taxonomy_linkage"].iloc[2] is None


def test_sync_taxonomy_rejects_non_params():
    with pytest.raises(ValueError, match="RpathParams"):
        sync_taxonomy_with_rpath(object(), [])
