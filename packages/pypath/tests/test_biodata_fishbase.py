"""FishBase traits from the published parquet snapshots.

The tables here are made up for the tests: small, with the layout of the
FishBase tables (its ``diet`` table spells the species key ``Speccode``), and
no real FishBase values.
"""

import io
import threading
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from pypath.io import biodata
from pypath.io.biodata import FishBaseTraits, _fetch_fishbase_traits

CODE = 7  # the made-up species "Piscis exemplar"


def _species():
    return pd.DataFrame(
        {
            "SpecCode": [7, 8],
            "Genus": ["Piscis", "Piscis"],
            "Species": ["exemplar", "alter"],
            "Length": [120.0, 30.0],
            "DemersPelag": ["benthopelagic", None],
        }
    )


def _ecology():
    # Two stocks of the species: the first has no food-based trophic level.
    return pd.DataFrame(
        {
            "SpecCode": [7, 7, 8],
            "FoodTroph": [None, 4.2, None],
            "DietTroph": [3.9, 4.0, 3.1],
        }
    )


def _diet():
    # Three studies of species 7: one of adults, two of juveniles.
    return pd.DataFrame(
        {
            "DietCode": [1, 2, 3, 4],
            "SpecCode": [7, 7, 7, 8],
            "SampleStage": ["recruits/juv.", "adults", "recruits/juv.", "larvae"],
        }
    )


def _diet_items():
    return pd.DataFrame(
        {
            "DietCode": [1, 1, 2, 2, 2, 3, 4],
            "FoodIII": [
                "zooplankton",
                "bony fish",
                "bony fish",
                "shrimps/prawns",
                "bony fish",
                "zooplankton",
                "zooplankton",
            ],
            # Study 2 sums to 50, not 100: its items are rescaled.
            "DietPercent": [90.0, 10.0, 30.0, 10.0, 10.0, 100.0, 100.0],
        }
    )


def _popgrowth():
    # Five TL records and one SL record for species 7. Sorted by K the TL
    # records are 0.10, 0.15, 0.20, 0.30, 0.40: the median study has K 0.20.
    return pd.DataFrame(
        {
            "SpecCode": [7, 7, 7, 7, 7, 7, 8],
            "Loo": [150.0, 110.0, 100.0, 90.0, 80.0, 60.0, 25.0],
            "K": [0.10, 0.20, 0.30, 0.15, 0.40, 0.90, 0.50],
            "to": [-1.0, None, -0.2, -0.5, 0.1, 0.0, None],
            "Type": ["TL", "TL", "TL", "TL", "TL", "SL", "TL"],
        }
    )


TABLES = {
    "species": _species,
    "ecology": _ecology,
    "diet": _diet,
    "diet_items": _diet_items,
    "popgrowth": _popgrowth,
}


@pytest.fixture
def tables():
    """Serve the made-up tables in place of a download."""
    with patch.object(
        biodata, "_fishbase_table", side_effect=lambda name, timeout=60: TABLES[name]()
    ) as loader:
        yield loader


# --- The lookup -----------------------------------------------------------


def test_traits_of_a_species(tables):
    traits = _fetch_fishbase_traits("Piscis exemplar", cache=False)

    assert isinstance(traits, FishBaseTraits)
    assert traits.species_code == 7
    assert traits.max_length == 120.0
    assert traits.habitat == "benthopelagic"
    # The first stock has no FoodTroph: the first value there is, is taken.
    assert traits.trophic_level == 4.2


def test_values_are_plain_python_not_numpy(tables):
    traits = _fetch_fishbase_traits("Piscis exemplar", cache=False)
    assert type(traits.species_code) is int
    assert type(traits.max_length) is float
    assert type(traits.trophic_level) is float
    assert all(type(v) is float for v in traits.growth_params.values())
    assert all(type(item["percentage"]) is float for item in traits.diet_items)


def test_trophic_level_falls_back_to_the_diet_based_one(tables):
    traits = _fetch_fishbase_traits("Piscis alter", cache=False)
    assert traits.trophic_level == 3.1
    assert traits.habitat is None


def test_growth_is_one_real_study_the_one_with_the_median_k(tables):
    # Among the commonest length type (TL), the record with the median K.
    # Loo, K and to belong together; this one reported no to.
    traits = _fetch_fishbase_traits("Piscis exemplar", cache=False)
    assert traits.growth_params == {"Loo": 110.0, "K": 0.20}


def test_growth_keeps_to_when_the_median_study_reports_it():
    table = _popgrowth()
    table.loc[table["K"] == 0.20, "to"] = -0.3
    assert biodata._fishbase_growth(table, CODE) == {"Loo": 110.0, "K": 0.2, "to": -0.3}


def test_growth_ignores_records_without_k_or_loo():
    table = _popgrowth()
    table.loc[table["K"] == 0.10, "K"] = None
    table.loc[table["K"] == 0.40, "Loo"] = None
    # Left: K 0.15, 0.20, 0.30 -> the median is still 0.20.
    assert biodata._fishbase_growth(table, CODE)["K"] == 0.2


def test_numbers_lose_their_single_precision_noise():
    import numpy as np

    # The tables store float32: 3.29 arrives as 3.2899999618...
    assert biodata._fishbase_number(np.float32(3.29)) == 3.29
    assert biodata._fishbase_number(float("nan")) is None
    assert biodata._fishbase_number(None) is None
    assert biodata._fishbase_number("n.a.") is None
    table = _popgrowth().astype({"Loo": "float32", "K": "float32", "to": "float32"})
    table.loc[table["K"].round(2) == 0.20, "to"] = -0.01
    assert biodata._fishbase_growth(table, CODE) == {
        "Loo": 110.0,
        "K": 0.2,
        "to": -0.01,
    }


def test_growth_is_none_without_records():
    assert biodata._fishbase_growth(_popgrowth(), 999) is None


def test_diet_prefers_studies_of_adults(tables):
    # Only study 2 is of adults. Its items sum to 50 and are rescaled to 100:
    # bony fish 30 + 10 of 50, shrimps 10 of 50.
    traits = _fetch_fishbase_traits("Piscis exemplar", cache=False)
    assert traits.diet_items == [
        {"prey": "bony fish", "percentage": 80.0},
        {"prey": "shrimps/prawns", "percentage": 20.0},
    ]


def test_diet_averages_studies_with_an_absent_prey_as_zero():
    diet = _diet()
    diet["SampleStage"] = "recruits/juv."  # no adult study: all are used
    items = biodata._fishbase_diet(diet, _diet_items(), CODE)
    # Study 1: zooplankton 90, bony fish 10. Study 2 (rescaled): bony fish 80,
    # shrimps 20. Study 3: zooplankton 100. Means over the three studies.
    assert items == [
        {"prey": "zooplankton", "percentage": 63.33},
        {"prey": "bony fish", "percentage": 30.0},
        {"prey": "shrimps/prawns", "percentage": 6.67},
    ]
    assert sum(i["percentage"] for i in items) == pytest.approx(100.0)


def test_diet_uses_mixed_stage_studies_before_juvenile_ones():
    diet = _diet()
    diet["SampleStage"] = ["recruits/juv.", "larvae", "juv./adults", "larvae"]
    items = biodata._fishbase_diet(diet, _diet_items(), CODE)
    assert items == [{"prey": "zooplankton", "percentage": 100.0}]


def test_diet_is_empty_without_studies():
    assert biodata._fishbase_diet(_diet(), _diet_items(), 999) == []


def test_a_species_that_is_not_a_fish_is_none(tables):
    assert _fetch_fishbase_traits("Mytilus edulis", cache=False) is None


def test_a_name_without_a_species_part_is_none(tables):
    with pytest.warns(UserWarning, match="Invalid scientific name"):
        assert _fetch_fishbase_traits("Piscis", cache=False) is None
    tables.assert_not_called()


def test_a_failed_download_warns_and_gives_none():
    with patch.object(biodata, "_fishbase_table", side_effect=OSError("no route")):
        with pytest.warns(UserWarning, match="Failed to query FishBase.*no route"):
            assert _fetch_fishbase_traits("Piscis exemplar", cache=False) is None


def test_a_missing_part_does_not_lose_the_rest():
    def loader(name, timeout=60):
        if name == "popgrowth":
            raise OSError("growth table unavailable")
        return TABLES[name]()

    with patch.object(biodata, "_fishbase_table", side_effect=loader):
        with pytest.warns(UserWarning, match="growth"):
            traits = _fetch_fishbase_traits("Piscis exemplar", cache=False)
    assert traits.trophic_level == 4.2
    assert traits.growth_params is None
    assert traits.diet_items


# --- The table loader -------------------------------------------------------


def _parquet(frame):
    pytest.importorskip("pyarrow")
    buffer = io.BytesIO()
    frame.to_parquet(buffer)
    return buffer.getvalue()


@pytest.fixture
def no_tables_loaded():
    biodata._fishbase_tables.clear()
    yield
    biodata._fishbase_tables.clear()


def _reply(content):
    reply = MagicMock()
    reply.content = content
    return reply


def test_loader_downloads_a_table_once_and_keeps_only_what_is_read(no_tables_loaded):
    # FishBase's diet table spells the species key "Speccode".
    raw = _diet().rename(columns={"SpecCode": "Speccode"})
    raw["Locality"] = "somewhere"
    with patch.object(biodata, "requests") as requests_:
        requests_.get.return_value = _reply(_parquet(raw))
        first = biodata._fishbase_table("diet", timeout=5)
        second = biodata._fishbase_table("diet", timeout=5)

    assert list(first.columns) == ["DietCode", "SpecCode", "SampleStage"]
    assert second is first
    requests_.get.assert_called_once_with(
        "https://data.source.coop/cboettig/fishbase/fb/"
        f"{biodata.FISHBASE_RELEASE}/parquet/diet.parquet",
        timeout=5,
    )
    requests_.get.return_value.raise_for_status.assert_called_once()


def test_loader_tolerates_a_column_the_snapshot_lacks(no_tables_loaded):
    raw = _species().drop(columns=["DemersPelag"])
    with patch.object(biodata, "requests") as requests_:
        requests_.get.return_value = _reply(_parquet(raw))
        table = biodata._fishbase_table("species")
    assert "DemersPelag" not in table.columns

    with patch.object(
        biodata,
        "_fishbase_table",
        side_effect=lambda n, timeout=60: table if n == "species" else TABLES[n](),
    ):
        traits = _fetch_fishbase_traits("Piscis exemplar", cache=False)
    assert traits.habitat is None and traits.max_length == 120.0


def test_loader_downloads_once_for_many_threads(no_tables_loaded):
    content = _parquet(_species())
    started = threading.Event()

    def slow_get(url, timeout):
        started.wait(0.2)  # give the other threads time to arrive
        return _reply(content)

    with patch.object(biodata, "requests") as requests_:
        requests_.get.side_effect = slow_get
        threads = [
            threading.Thread(target=biodata._fishbase_table, args=("species",))
            for _ in range(8)
        ]
        for thread in threads:
            thread.start()
        started.set()
        for thread in threads:
            thread.join()
    assert requests_.get.call_count == 1


def test_loader_names_the_missing_parquet_reader(no_tables_loaded):
    with patch.object(biodata, "requests") as requests_:
        requests_.get.return_value = _reply(b"not parquet")
        with patch.object(
            biodata.pd, "read_parquet", side_effect=ImportError("no engine")
        ):
            with pytest.raises(ImportError, match="pyarrow"):
                biodata._fishbase_table("species")


def test_loader_needs_requests(no_tables_loaded):
    with patch.object(biodata, "HAS_REQUESTS", False):
        with pytest.raises(ImportError, match="requests is required"):
            biodata._fishbase_table("species")


def test_loader_rejects_a_table_it_does_not_know():
    with pytest.raises(KeyError):
        biodata._fishbase_table("countries")


def test_clear_cache_keeps_the_tables(no_tables_loaded):
    # The tables are a 6 MB download: clearing the per-species cache, which
    # the test suites do before every test, must not throw them away.
    biodata._fishbase_tables["species"] = _species()
    biodata.clear_cache()
    assert "species" in biodata._fishbase_tables


# --- Live, against the published snapshot -----------------------------------


@pytest.mark.integration
@pytest.mark.fishbase
def test_live_atlantic_cod():
    """The real snapshot: stable facts about Atlantic cod.

    The snapshot is a static file host, so this belongs in the weekly run.
    """
    pytest.importorskip("pyarrow")
    try:
        biodata._fishbase_table("species", timeout=60)
    except Exception as err:  # the host cannot be reached from here
        pytest.skip(f"FishBase snapshot not reachable: {type(err).__name__}")

    traits = _fetch_fishbase_traits("Gadus morhua", cache=False, timeout=60)

    assert traits is not None and traits.species_code == 69
    assert 100 < traits.max_length < 300
    assert 3.5 <= traits.trophic_level <= 5.0
    assert traits.habitat == "benthopelagic"
    assert 0.05 < traits.growth_params["K"] < 0.5
    assert 50 < traits.growth_params["Loo"] < 250
    assert traits.diet_items
    assert sum(i["percentage"] for i in traits.diet_items) == pytest.approx(
        100, abs=0.5
    )
    # Not a fish: FishBase does not have it.
    assert _fetch_fishbase_traits("Mytilus edulis", cache=False, timeout=60) is None
