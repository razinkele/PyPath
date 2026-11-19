"""FishBase traits from the published parquet snapshots.

The tables here are made up for the tests: small, with the layout of the
FishBase tables (its ``diet`` table spells the species key ``Speccode``), and
no real FishBase values.
"""

import contextlib
import io
import threading
import warnings
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


def _synonyms():
    # Names that are not the accepted one in the species table.
    return pd.DataFrame(
        {
            "SpecCode": [7, 8, 7, 8, 7, 7, 99, 7],
            "SynGenus": [
                "Vetus",  # a plain synonym of species 7
                "Duplex",  # the same old name is listed for two species
                "Duplex",
                "Falsus",  # a name wrongly applied to species 8
                "Piscis",  # a subspecies name sharing a binomial
                "Vetus",  # a trinomial: not a binomial match
                "Orbus",  # points at a species the species table lacks
                "Piscis",  # the accepted name itself
            ],
            "SynSpecies": [
                "nomen",
                "anceps",
                "anceps",
                "usus",
                "minor",
                "nomen antiquum",
                "solus",
                "exemplar",
            ],
            "Status": [
                "synonym",
                "synonym",
                "synonym",
                "misapplied name",
                "synonym",
                "synonym",
                "synonym",
                "accepted name",
            ],
            "TaxonLevel": [
                "Species",
                "species",
                "Species",
                "Species",
                "Subspecies",
                "Subspecies",
                "Species",
                "Species",
            ],
        }
    )


# The made-up SeaLifeBase: a mussel-like "Concha testa" (code 7, the same
# number as the fish: the two databases number their species independently)
# and a seal-like "Phoca ficta".
def _slb_species():
    return pd.DataFrame(
        {
            "SpecCode": [7, 20],
            "Genus": ["Concha", "Phoca"],
            "Species": ["testa", "ficta"],
            "Length": [9.0, 180.0],
            "DemersPelag": ["benthic", "pelagic"],
        }
    )


def _slb_ecology():
    return pd.DataFrame(
        {"SpecCode": [7, 20], "FoodTroph": [2.1, 4.4], "DietTroph": [None, 4.5]}
    )


def _slb_diet():
    return pd.DataFrame(
        {"DietCode": [50], "SpecCode": [20], "SampleStage": ["juv./adults"]}
    )


def _slb_diet_items():
    return pd.DataFrame(
        {
            "DietCode": [50, 50],
            "FoodIII": ["bony fish", "squids"],
            "DietPercent": [70.0, 30.0],
        }
    )


def _slb_popgrowth():
    return pd.DataFrame(
        {
            "SpecCode": [7, 7, 7, 20],
            "Loo": [8.0, 7.0, 9.0, 170.0],
            "K": [0.2, 0.3, 0.4, 0.25],
            "to": [None, None, None, -1.0],
            "Type": ["ShL", "ShL", "ShL", "TL"],
        }
    )


def _slb_synonyms():
    return pd.DataFrame(
        {
            "SpecCode": [7, 20],
            "SynGenus": ["Mytilus", "Vetus"],
            "SynSpecies": ["priscus", "nomen"],  # "Vetus nomen" is in FishBase too
            "Status": ["synonym", "synonym"],
            "TaxonLevel": ["Species", "Species"],
        }
    )


@contextlib.contextmanager
def _no_user_warnings():
    """Fail if the block warns the user."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        yield


TABLES = {
    "species": _species,
    "ecology": _ecology,
    "diet": _diet,
    "diet_items": _diet_items,
    "popgrowth": _popgrowth,
    "synonyms": _synonyms,
}
SLB_TABLES = {
    "species": _slb_species,
    "ecology": _slb_ecology,
    "diet": _slb_diet,
    "diet_items": _slb_diet_items,
    "popgrowth": _slb_popgrowth,
    "synonyms": _slb_synonyms,
}
DATABASES = {"fishbase": TABLES, "sealifebase": SLB_TABLES}


def _serve(name, timeout=60, source="fishbase"):
    """The made-up tables, in place of a download."""
    return DATABASES[source][name]()


@pytest.fixture
def tables():
    """Serve the made-up tables in place of a download."""
    with patch.object(biodata, "_fishbase_table", side_effect=_serve) as loader:
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


def test_a_name_in_neither_database_is_none(tables):
    assert _fetch_fishbase_traits("Ignotus ignotus", cache=False) is None


def test_a_name_without_a_species_part_is_none(tables):
    with pytest.warns(UserWarning, match="Invalid scientific name"):
        assert _fetch_fishbase_traits("Piscis", cache=False) is None
    tables.assert_not_called()


def test_a_failed_download_warns_and_gives_none():
    with patch.object(biodata, "_fishbase_table", side_effect=OSError("no route")):
        with pytest.warns(UserWarning, match="Failed to query FishBase.*no route"):
            assert _fetch_fishbase_traits("Piscis exemplar", cache=False) is None


def test_a_missing_part_does_not_lose_the_rest():
    def loader(name, timeout=60, source="fishbase"):
        if name == "popgrowth":
            raise OSError("growth table unavailable")
        return _serve(name, source=source)

    with patch.object(biodata, "_fishbase_table", side_effect=loader):
        with pytest.warns(UserWarning, match="growth"):
            traits = _fetch_fishbase_traits("Piscis exemplar", cache=False)
    assert traits.trophic_level == 4.2
    assert traits.growth_params is None
    assert traits.diet_items


def test_an_incomplete_result_is_not_cached():
    # A table that failed once must be tried again: the incomplete result
    # would otherwise hide the species' growth for the cache's whole lifetime.
    biodata.clear_cache()
    failing = True

    def loader(name, timeout=60, source="fishbase"):
        if name == "popgrowth" and failing:
            raise OSError("timed out")
        return _serve(name, source=source)

    with patch.object(biodata, "_fishbase_table", side_effect=loader):
        with pytest.warns(UserWarning, match="growth"):
            first = _fetch_fishbase_traits("Piscis exemplar", cache=True)
        assert first.growth_params is None and first.trophic_level == 4.2
        failing = False
        second = _fetch_fishbase_traits("Piscis exemplar", cache=True)
        assert second.growth_params == {"Loo": 110.0, "K": 0.2}
        # A complete result is cached: the tables are not read again.
        failing = True
        third = _fetch_fishbase_traits("Piscis exemplar", cache=True)
        assert third is second
    biodata.clear_cache()


# --- SeaLifeBase, for what is not a fish -----------------------------------


def test_a_fish_is_marked_as_from_fishbase(tables):
    traits = _fetch_fishbase_traits("Piscis exemplar", cache=False)
    assert traits.source == "fishbase"
    assert traits.scientific_name == "Piscis exemplar"
    assert traits.growth_length_type == "TL"
    # A fish never needs SeaLifeBase, or a synonyms table.
    asked = {
        (c.kwargs.get("source", "fishbase"), c.args[0]) for c in tables.call_args_list
    }
    assert all(source == "fishbase" for source, _ in asked)
    assert ("fishbase", "synonyms") not in asked


def test_a_non_fish_comes_from_sealifebase(tables):
    traits = _fetch_fishbase_traits("Concha testa", cache=False)

    assert traits.source == "sealifebase"
    # The same species code as the fish, in the other database's tables.
    assert traits.species_code == 7
    assert traits.max_length == 9.0
    assert traits.habitat == "benthic"
    assert traits.trophic_level == 2.1
    assert traits.diet_items == []
    # The median-K study, with the length it measured: a shell length here.
    assert traits.growth_params == {"Loo": 7.0, "K": 0.3}
    assert traits.growth_length_type == "ShL"


def test_a_non_fish_with_a_diet_study(tables):
    traits = _fetch_fishbase_traits("Phoca ficta", cache=False)
    assert traits.source == "sealifebase"
    assert traits.diet_items == [
        {"prey": "bony fish", "percentage": 70.0},
        {"prey": "squids", "percentage": 30.0},
    ]
    assert traits.growth_params == {"Loo": 170.0, "K": 0.25, "to": -1.0}


def test_a_failing_sealifebase_is_not_reported_as_an_unknown_species():
    def loader(name, timeout=60, source="fishbase"):
        if source == "sealifebase":
            raise OSError("no route")
        return _serve(name, source=source)

    with patch.object(biodata, "_fishbase_table", side_effect=loader):
        with pytest.warns(UserWarning, match="Failed to query FishBase.*no route"):
            assert _fetch_fishbase_traits("Concha testa", cache=False) is None


# --- Synonyms ----------------------------------------------------------------


def test_a_synonym_leads_to_the_accepted_species(tables):
    traits = _fetch_fishbase_traits("Mytilus priscus", cache=False)
    assert traits.source == "sealifebase"
    assert traits.scientific_name == "Concha testa"
    assert traits.max_length == 9.0


def test_an_accepted_name_is_never_looked_up_as_a_synonym(tables):
    # "Concha testa" is accepted in SeaLifeBase; no synonyms table is read.
    _fetch_fishbase_traits("Concha testa", cache=False)
    asked = {c.args[0] for c in tables.call_args_list}
    assert "synonyms" not in asked


def test_a_synonym_of_two_species_is_refused(tables):
    with pytest.warns(UserWarning, match="Duplex anceps.*more than one species"):
        assert _fetch_fishbase_traits("Duplex anceps", cache=False) is None


def test_a_synonym_in_both_databases_is_refused(tables):
    # "Vetus nomen" is a synonym of a fish and of a seal.
    with pytest.warns(UserWarning, match="Vetus nomen.*more than one species"):
        assert _fetch_fishbase_traits("Vetus nomen", cache=False) is None


def test_a_misapplied_name_is_not_followed(tables):
    # It was used for the species by mistake: it names another one.
    assert _fetch_fishbase_traits("Falsus usus", cache=False) is None


def test_a_subspecies_synonym_does_not_answer_for_its_binomial(tables):
    assert _fetch_fishbase_traits("Piscis minor", cache=False) is None


def test_a_synonym_of_a_species_missing_from_the_species_table_is_none(tables):
    assert _fetch_fishbase_traits("Orbus solus", cache=False) is None


# --- P/B from growth is for fish only ------------------------------------------


def _info(name, source, k):
    from pypath.io.biodata import SpeciesInfo

    return SpeciesInfo(
        common_name=name,
        scientific_name=name,
        aphia_id=1,
        authority="",
        trophic_level=3.0,
        growth_params={"Loo": 50.0, "K": k},
        traits_source=source,
    )


def test_a_fish_gets_pb_and_qb_from_its_growth():
    params = biodata.biodata_to_rpath(
        _info("Piscis exemplar", "fishbase", 0.2),
        biomass_estimates={"Piscis exemplar": 1.0},
    )
    row = params.model[params.model["Group"] == "Piscis exemplar"].iloc[0]
    assert row["PB"] == pytest.approx(0.5)  # 2.5 x K
    assert pd.notna(row["QB"])


def test_a_non_fish_keeps_pb_and_qb_blank():
    # The 2.5 x K rule is a fish rule: it gives a seal several times the P/B
    # an Ecopath model uses. The growth values are still reported.
    params = biodata.biodata_to_rpath(
        _info("Phoca ficta", "sealifebase", 0.25),
        biomass_estimates={"Phoca ficta": 1.0},
    )
    row = params.model[params.model["Group"] == "Phoca ficta"].iloc[0]
    assert pd.isna(row["PB"]) and pd.isna(row["QB"])


def test_a_table_without_the_source_column_is_treated_as_fish():
    # A caller's own table with a k column, as before.
    table = pd.DataFrame(
        {
            "common_name": ["x"],
            "scientific_name": ["Piscis exemplar"],
            "trophic_level": [3.0],
            "k": [0.2],
        }
    )
    params = biodata.biodata_to_rpath(table, biomass_estimates={"Piscis exemplar": 1.0})
    assert params.model.loc[0, "PB"] == pytest.approx(0.5)


@pytest.mark.parametrize("missing", [None, float("nan"), pd.NA])
def test_a_missing_source_value_is_treated_as_fish(missing):
    # A nullable string column holds pd.NA, whose truth value is an error:
    # the check must ask "is it missing?" before comparing it.
    table = pd.DataFrame(
        {
            "common_name": ["x", "y"],
            "scientific_name": ["Piscis exemplar", "Phoca ficta"],
            "trophic_level": [3.0, 4.0],
            "k": [0.2, 0.25],
            "traits_source": pd.array([missing, "sealifebase"], dtype="object"),
        }
    )
    params = biodata.biodata_to_rpath(
        table, biomass_estimates={"Piscis exemplar": 1.0, "Phoca ficta": 1.0}
    )
    assert params.model.loc[0, "PB"] == pytest.approx(0.5)
    # An explicit SeaLifeBase row still keeps its P/B blank.
    assert pd.isna(params.model.loc[1, "PB"])


def test_a_nullable_string_source_column_works():
    table = pd.DataFrame(
        {
            "common_name": ["x"],
            "scientific_name": ["Piscis exemplar"],
            "trophic_level": [3.0],
            "k": [0.2],
            "traits_source": pd.array([pd.NA], dtype="string"),
        }
    )
    params = biodata.biodata_to_rpath(table, biomass_estimates={"Piscis exemplar": 1.0})
    assert params.model.loc[0, "PB"] == pytest.approx(0.5)


def test_species_info_carries_the_source_and_length_type(tables):
    info = biodata._merge_species_data(
        {"scientificname": "Concha testa", "AphiaID": 5, "authority": ""},
        fishbase_data=_fetch_fishbase_traits("Concha testa", cache=False),
        common_name="shell",
    )
    assert info.traits_source == "sealifebase"
    assert info.growth_length_type == "ShL"


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


def _reply(content, status_code=200):
    reply = MagicMock()
    reply.content = content
    reply.status_code = status_code
    if status_code >= 400:
        import requests

        error = requests.exceptions.HTTPError(f"{status_code} error", response=reply)
        reply.raise_for_status.side_effect = error
    return reply


# --- Transient download failures are retried ---------------------------------


@pytest.fixture
def no_waiting():
    """Record the waits between attempts without sleeping through them."""
    with patch.object(biodata.time, "sleep") as sleep:
        yield sleep


def _failing_then(*outcomes):
    """A ``requests.get`` that raises or returns each outcome in turn."""
    queue = list(outcomes)

    def get(url, timeout):
        outcome = queue.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    return get


def test_a_dropped_connection_is_tried_again(no_tables_loaded, no_waiting):
    import requests

    good = _reply(_parquet(_species()))
    with patch.object(biodata, "requests") as requests_:
        requests_.get.side_effect = _failing_then(
            requests.exceptions.ConnectionError("reset by peer"),
            requests.exceptions.ReadTimeout("read timed out"),
            good,
        )
        table = biodata._fishbase_table("species", timeout=5)

    assert list(table["Genus"]) == ["Piscis", "Piscis"]
    assert requests_.get.call_count == 3
    # A short wait that grows: 1 s, then 2 s.
    assert [c.args[0] for c in no_waiting.call_args_list] == [1.0, 2.0]


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_a_busy_or_failing_server_is_tried_again(no_tables_loaded, no_waiting, status):
    with patch.object(biodata, "requests") as requests_:
        requests_.get.side_effect = _failing_then(
            _reply(b"", status_code=status), _reply(_parquet(_species()))
        )
        table = biodata._fishbase_table("species")
    assert len(table) == 2 and requests_.get.call_count == 2


@pytest.mark.parametrize("status", [400, 403, 404])
def test_a_missing_file_is_not_tried_again(no_tables_loaded, no_waiting, status):
    # The release moved or the table was renamed: asking again cannot help.
    import requests

    with patch.object(biodata, "requests") as requests_:
        requests_.get.side_effect = _failing_then(_reply(b"", status_code=status))
        with pytest.raises(requests.exceptions.HTTPError, match=str(status)):
            biodata._fishbase_table("species")
    assert requests_.get.call_count == 1
    no_waiting.assert_not_called()


def test_the_download_gives_up_after_three_attempts(no_tables_loaded, no_waiting):
    import requests

    errors = [requests.exceptions.ConnectionError(f"refused {i}") for i in (1, 2, 3)]
    with patch.object(biodata, "requests") as requests_:
        requests_.get.side_effect = _failing_then(*errors)
        with pytest.raises(requests.exceptions.ConnectionError, match="refused 3"):
            biodata._fishbase_table("species")
    assert requests_.get.call_count == 3
    assert len(no_waiting.call_args_list) == 2  # no wait after the last attempt
    assert ("fishbase", "species") not in biodata._fishbase_tables


def test_a_failed_download_can_be_asked_for_again(no_tables_loaded, no_waiting):
    import requests

    errors = [requests.exceptions.ConnectionError("down")] * 3
    with patch.object(biodata, "requests") as requests_:
        requests_.get.side_effect = _failing_then(*errors, _reply(_parquet(_species())))
        with pytest.raises(requests.exceptions.ConnectionError):
            biodata._fishbase_table("species")
        assert len(biodata._fishbase_table("species")) == 2


def test_a_lookup_survives_one_failed_download(no_tables_loaded, no_waiting):
    # The whole point: one dropped connection must not cost the species.
    import requests

    contents = {
        "species": _species(),
        "ecology": _ecology(),
        "diet": _diet(),
        "diet_items": _diet_items(),
        "popgrowth": _popgrowth(),
    }
    failed = set()

    def get(url, timeout):
        name = url.rsplit("/", 1)[1].removesuffix(".parquet")
        if name not in failed:  # every table fails once
            failed.add(name)
            raise requests.exceptions.ConnectionError("reset by peer")
        return _reply(_parquet(contents[name]))

    with patch.object(biodata, "requests") as requests_:
        requests_.get.side_effect = get
        with _no_user_warnings():
            traits = _fetch_fishbase_traits("Piscis exemplar", cache=False)
    assert traits.trophic_level == 4.2
    assert traits.growth_params == {"Loo": 110.0, "K": 0.2}
    assert traits.diet_items


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

    def loader(name, timeout=60, source="fishbase"):
        if (source, name) == ("fishbase", "species"):
            return table
        return _serve(name, source=source)

    with patch.object(biodata, "_fishbase_table", side_effect=loader):
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
    with pytest.raises(KeyError):
        biodata._fishbase_table("species", source="birdbase")


def test_loader_reads_sealifebase_from_its_own_folder(no_tables_loaded):
    with patch.object(biodata, "requests") as requests_:
        requests_.get.return_value = _reply(_parquet(_slb_species()))
        slb = biodata._fishbase_table("species", timeout=5, source="sealifebase")
        requests_.get.assert_called_once_with(
            "https://data.source.coop/cboettig/fishbase/slb/"
            f"{biodata.FISHBASE_RELEASE}/parquet/species.parquet",
            timeout=5,
        )
        # The two databases' tables are kept apart.
        requests_.get.return_value = _reply(_parquet(_species()))
        fb = biodata._fishbase_table("species", timeout=5)
    assert list(slb["Genus"]) == ["Concha", "Phoca"]
    assert list(fb["Genus"]) == ["Piscis", "Piscis"]


def test_a_loaded_table_is_served_while_another_downloads(no_tables_loaded):
    # A lookup of a fish must not wait behind another thread's download of a
    # synonyms table.
    biodata._fishbase_tables[("fishbase", "species")] = _species()
    release = threading.Event()
    downloading = threading.Event()
    content = _parquet(_synonyms())

    def slow_get(url, timeout):
        downloading.set()
        release.wait(5)
        return _reply(content)

    with patch.object(biodata, "requests") as requests_:
        requests_.get.side_effect = slow_get
        worker = threading.Thread(target=biodata._fishbase_table, args=("synonyms",))
        worker.start()
        assert downloading.wait(5)
        result = []
        reader = threading.Thread(
            target=lambda: result.append(biodata._fishbase_table("species"))
        )
        reader.start()
        reader.join(2)
        served = not reader.is_alive()
        release.set()
        worker.join(5)
    assert served and len(result) == 1


def test_clear_cache_keeps_the_tables(no_tables_loaded):
    # The tables are a 6 MB download: clearing the per-species cache, which
    # the test suites do before every test, must not throw them away.
    biodata._fishbase_tables[("fishbase", "species")] = _species()
    biodata.clear_cache()
    assert ("fishbase", "species") in biodata._fishbase_tables


# --- Live, against the published snapshot -----------------------------------


@pytest.mark.integration
@pytest.mark.fishbase
def test_live_atlantic_cod():
    """The real snapshot: stable facts about Atlantic cod.

    The snapshot is a static file host, so this belongs in the weekly run.
    """
    import requests

    pytest.importorskip("pyarrow")
    try:
        biodata._fishbase_table("species", timeout=60)
    except (
        requests.exceptions.ConnectionError,  # refused, DNS, TLS
        requests.exceptions.Timeout,
    ) as err:  # the host cannot be reached from here
        pytest.skip(f"FishBase snapshot not reachable: {type(err).__name__}")
    # Anything else fails the test: a moved release (HTTP error), a file that
    # is not parquet or a changed layout is what the weekly run is there for.

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
    assert traits.source == "fishbase"
    assert traits.growth_length_type == "TL"


@pytest.mark.integration
@pytest.mark.fishbase
def test_live_non_fish_and_synonym():
    """The real snapshots: a mussel from SeaLifeBase, and a synonym followed.

    The synonym is a row of the FishBase synonyms table: a plain synonym of
    one species, and not an accepted name in either database.
    """
    import requests

    pytest.importorskip("pyarrow")
    try:
        biodata._fishbase_table("species", timeout=60, source="sealifebase")
    except (
        requests.exceptions.ConnectionError,
        requests.exceptions.Timeout,
    ) as err:
        pytest.skip(f"SeaLifeBase snapshot not reachable: {type(err).__name__}")

    mussel = _fetch_fishbase_traits("Mytilus edulis", cache=False, timeout=60)
    assert mussel is not None and mussel.source == "sealifebase"
    assert mussel.scientific_name == "Mytilus edulis"
    assert mussel.habitat == "benthic"
    assert 1.9 <= mussel.trophic_level <= 2.5
    assert mussel.growth_length_type == "ShL"  # a shell length
    assert 0.05 < mussel.growth_params["K"] < 1.2

    renamed = _fetch_fishbase_traits("Ageneiosus quadrifilis", cache=False, timeout=60)
    assert renamed is not None and renamed.source == "fishbase"
    assert renamed.scientific_name == "Tetranematichthys quadrifilis"

    assert _fetch_fishbase_traits("Ignotus ignotus", cache=False, timeout=60) is None
