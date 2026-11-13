"""EwE-F HDF5 results, in the layouts EPath's API writes and reads.

* ``ecopath_results.h5``: compound dataset ``ep_data`` (one record per group;
  -999 = not applicable), plus ``ep_diet`` and ``ep_detfate``. Group names are
  not in the file; an imported model's run has ``group_names.json`` beside it.
* ``ecospace_results.h5``: ``biomass_spatial`` (group, month, x, y), with
  -999 in land cells.

The fixtures write those layouts; set ``PYPATH_EWEF_ECOPATH_H5`` to a real
``ecopath_results.h5`` to run the check against one.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from pypath.io import ewef_hdf5  # noqa: E402

NA = -999.0
EP_FIELDS = [
    ("biomass", "<f4"),
    ("PoB", "<f4"),
    ("QoB", "<f4"),
    ("EE", "<f4"),
    ("unass_Q", "<f4"),
    ("landings", "<f4"),
    ("discards", "<f4"),
    ("org_type", "<i4"),
    ("production", "<f4"),
    ("consumption", "<f4"),
    ("respiration", "<f4"),
    ("assimilation", "<f4"),
]


def _ep_records():
    """Two consumers, a producer and detritus.

    EPath's org_type: 0 detritus, 1 producer, 2 consumer (the reverse of EwE's
    group Type codes).
    """
    rows = []
    for b, pob, qob, landings, discards in (
        (2.0, 0.5, 4.0, 0.3, 0.1),
        (5.0, 1.0, 6.0, NA, NA),
    ):
        p, q = b * pob, b * qob
        a = q * (1 - 0.2)
        rows.append((b, pob, qob, 0.8, 0.2, landings, discards, 2, p, q, a - p, a))
    # as EPath writes them: a producer and detritus have zero Q, R and A
    rows.append((10.0, 20.0, NA, 0.5, NA, NA, NA, 1, 200.0, 0.0, 0.0, 0.0))
    rows.append((50.0, NA, NA, 0.3, NA, NA, NA, 0, 0.0, 0.0, 0.0, 0.0))
    return np.array(rows, dtype=EP_FIELDS)


def _write_ecopath(path, records=None, names=("Cod", "Sprat", "Phyto", "Detritus")):
    records = _ep_records() if records is None else records
    with h5py.File(path, "w") as f:
        f["ep_data"] = records
        f["ep_diet"] = np.zeros((1,), dtype=("<f4", (len(records) + 1, 2)))
        f["ep_detfate"] = np.ones((1,), dtype=("<f4", (len(records), 1)))
    if names is not None:
        (Path(path).parent / "group_names.json").write_text(json.dumps(list(names)))
    return path


def _write_ecospace(path):
    # 2 groups, 3 months, a 2 x 2 grid with one land cell
    spatial = np.arange(24, dtype="<f4").reshape(2, 3, 2, 2)
    spatial[:, :, 1, 1] = NA
    with h5py.File(path, "w") as f:
        f["biomass_spatial"] = spatial
    return path, spatial


# --- Ecopath results ---


def test_ecopath_results_give_biomass_and_catch_per_group(tmp_path):
    results = ewef_hdf5.read_ewef_results(
        _write_ecopath(tmp_path / "ecopath_results.h5")
    )
    assert results.kind == "ecopath"
    assert results.time_steps == 1 and results.n_groups == 4
    assert list(results.group_names) == ["Cod", "Sprat", "Phyto", "Detritus"]
    np.testing.assert_allclose(results.biomass, [[2.0, 5.0, 10.0, 50.0]])
    # catch = landings + discards; -999 (not applicable) is NaN
    np.testing.assert_allclose(
        results.catches, [[0.4, np.nan, np.nan, np.nan]], rtol=1e-6
    )


def test_ecopath_groups_are_numbered_without_a_names_file(tmp_path):
    path = _write_ecopath(tmp_path / "ecopath_results.h5", names=None)
    results = ewef_hdf5.read_ewef_results(path)
    assert list(results.group_names) == ["group_1", "group_2", "group_3", "group_4"]


def test_ecopath_records_are_kept_for_further_use(tmp_path):
    results = ewef_hdf5.read_ewef_results(
        _write_ecopath(tmp_path / "ecopath_results.h5")
    )
    ep = results.diagnostics["ep_data"]
    assert ep.dtype.names[:2] == ("biomass", "PoB")
    assert results.diagnostics["ep_diet"].shape == (5, 2)


# --- Ecospace results ---


def test_ecospace_results_give_the_domain_mean_over_sea_cells(tmp_path):
    path, spatial = _write_ecospace(tmp_path / "ecospace_results.h5")
    results = ewef_hdf5.read_ewef_results(path)
    assert results.kind == "ecospace"
    assert results.time_steps == 3 and results.n_groups == 2
    sea = spatial.reshape(2, 3, 4)[:, :, :3]  # the three sea cells
    np.testing.assert_allclose(results.biomass, sea.mean(axis=2).T)
    assert results.catches is None


def test_an_ecospace_file_has_no_catches(tmp_path):
    path, _ = _write_ecospace(tmp_path / "ecospace_results.h5")
    with pytest.raises(ValueError, match="No catch data"):
        ewef_hdf5.read_ewef_hdf5_catches(path)


# --- other files ---


def test_a_file_in_neither_layout_is_rejected(tmp_path):
    path = tmp_path / "other.h5"
    with h5py.File(path, "w") as f:
        f["something"] = np.zeros(3)
    with pytest.raises(ValueError, match="ep_data|biomass_spatial"):
        ewef_hdf5.read_ewef_results(path)


def test_a_missing_file_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError):
        ewef_hdf5.read_ewef_results(tmp_path / "missing.h5")


def test_inspect_lists_datasets_and_compound_fields(tmp_path):
    structure = ewef_hdf5.inspect_hdf5_structure(
        _write_ecopath(tmp_path / "ecopath_results.h5"), verbose=False
    )
    by_path = {d["path"]: d for d in structure["datasets"]}
    assert set(by_path) == {"ep_data", "ep_diet", "ep_detfate"}
    assert by_path["ep_data"]["fields"][:3] == ["biomass", "PoB", "QoB"]


def test_biomass_helper_returns_the_array_and_names(tmp_path):
    biomass, names = ewef_hdf5.read_ewef_hdf5_biomass(
        _write_ecopath(tmp_path / "ecopath_results.h5")
    )
    assert biomass.shape == (1, 4) and names[0] == "Cod"


# --- energy balance ---


def test_a_balanced_ecopath_file_passes_the_energy_balance(tmp_path):
    results = ewef_hdf5.read_ewef_results(
        _write_ecopath(tmp_path / "ecopath_results.h5")
    )
    ok, message = ewef_hdf5.validate_hdf5_energy_balance(results)
    assert ok, message


def test_an_unbalanced_consumer_is_named(tmp_path):
    # consumers assimilate Q x (1 - unass_Q) = P + R; break Sprat's respiration
    records = _ep_records()
    records["respiration"][1] += 1.0
    path = _write_ecopath(tmp_path / "ecopath_results.h5", records=records)
    ok, message = ewef_hdf5.validate_hdf5_energy_balance(
        ewef_hdf5.read_ewef_results(path)
    )
    assert not ok and "Sprat" in message


def test_infinite_biomass_fails_the_check(tmp_path):
    # -999 (not applicable) reads as NaN and is exempt; infinity never is
    records = _ep_records()
    records["biomass"][1] = np.inf
    path = _write_ecopath(tmp_path / "ecopath_results.h5", records=records)
    ok, message = ewef_hdf5.validate_hdf5_energy_balance(
        ewef_hdf5.read_ewef_results(path)
    )
    assert not ok and "finite" in message


@pytest.mark.parametrize("assimilation", [0.0, NA], ids=["zero", "not_applicable"])
def test_a_consumer_without_a_valid_assimilation_fails(tmp_path, assimilation):
    # Sprat produces and respires, so it must assimilate: 0 or -999 is a failure
    records = _ep_records()
    records["assimilation"][1] = assimilation
    path = _write_ecopath(tmp_path / "ecopath_results.h5", records=records)
    ok, message = ewef_hdf5.validate_hdf5_energy_balance(
        ewef_hdf5.read_ewef_results(path)
    )
    assert not ok and "Sprat" in message


@pytest.mark.parametrize("unass_q", [NA, np.inf], ids=["not_applicable", "inf"])
def test_a_consumer_without_a_valid_unassimilated_fraction_fails(tmp_path, unass_q):
    records = _ep_records()
    records["unass_Q"][1] = unass_q
    path = _write_ecopath(tmp_path / "ecopath_results.h5", records=records)
    ok, message = ewef_hdf5.validate_hdf5_energy_balance(
        ewef_hdf5.read_ewef_results(path)
    )
    assert not ok and "Sprat" in message


def test_zero_assimilation_needs_zero_production_and_respiration(tmp_path):
    # P + R = 0 and Q = 0, but P and R are not each zero
    records = _ep_records()
    for field, value in (
        ("assimilation", 0.0),
        ("production", 1.0),
        ("respiration", -1.0),
        ("consumption", 0.0),
    ):
        records[field][1] = value
    path = _write_ecopath(tmp_path / "ecopath_results.h5", records=records)
    ok, message = ewef_hdf5.validate_hdf5_energy_balance(
        ewef_hdf5.read_ewef_results(path)
    )
    assert not ok and "Sprat" in message


def test_ecospace_results_get_the_basic_checks(tmp_path):
    path, _ = _write_ecospace(tmp_path / "ecospace_results.h5")
    ok, message = ewef_hdf5.validate_hdf5_energy_balance(
        ewef_hdf5.read_ewef_results(path)
    )
    assert ok, message


@pytest.mark.skipif(
    not os.environ.get("PYPATH_EWEF_ECOPATH_H5"),
    reason="set PYPATH_EWEF_ECOPATH_H5 to a real EwE-F ecopath_results.h5",
)
def test_a_real_ecopath_results_file():
    results = ewef_hdf5.read_ewef_results(os.environ["PYPATH_EWEF_ECOPATH_H5"])
    assert results.kind == "ecopath" and results.n_groups > 0
    ok, message = ewef_hdf5.validate_hdf5_energy_balance(results)
    assert ok, message
