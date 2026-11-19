"""Integration tests for the biodiversity data workflow.

Tests the complete workflow used by the Shiny app:
1. Fetch species info from WoRMS/OBIS/FishBase
2. Create Ecopath model from biodiversity data
3. Verify API connectivity

All tests are marked as integration and slow since they require network access.
"""

import logging

import pytest

logger = logging.getLogger(__name__)

pyworms = pytest.importorskip("pyworms")
requests = pytest.importorskip("requests")

from pypath.io.biodata import (  # noqa: E402
    _fetch_worms_vernacular,
    batch_get_species_info,
    biodata_to_rpath,
    get_species_info,
)


@pytest.mark.integration
@pytest.mark.slow
class TestWoRMSLookup:
    """Test individual WoRMS vernacular name lookups."""

    @pytest.mark.parametrize(
        "species_name",
        [
            "Atlantic cod",
            "cod",
            "Atlantic herring",
            "herring",
            "European sprat",
            "sprat",
        ],
    )
    def test_worms_vernacular_search(self, species_name):
        results = _fetch_worms_vernacular(species_name, cache=False, timeout=30)
        assert results is not None
        logger.debug(
            "Search '%s': found %d result(s)",
            species_name,
            len(results) if results else 0,
        )
        if results:
            for r in results[:3]:
                logger.debug(
                    "  %s (AphiaID: %s)",
                    r.get("scientificname"),
                    r.get("AphiaID"),
                )


@pytest.mark.integration
@pytest.mark.slow
class TestSpeciesInfoWorkflow:
    """Test single and batch species info retrieval."""

    def test_single_species_info(self):
        info = get_species_info("cod", strict=False, timeout=30)
        assert info is not None
        assert info.common_name is not None
        assert info.scientific_name is not None
        logger.debug(
            "Species: %s (%s), TL=%.2f, AphiaID=%s",
            info.common_name,
            info.scientific_name,
            info.trophic_level if info.trophic_level else 0,
            info.aphia_id,
        )

    def test_batch_species_info(self):
        species_list = ["cod", "herring", "sprat"]
        df = batch_get_species_info(
            species_list,
            include_occurrences=True,
            include_traits=True,
            strict=False,
            max_workers=5,
            timeout=45,
        )
        assert df is not None
        assert len(df) > 0
        logger.debug("Retrieved data for %d species", len(df))
        for _, row in df.iterrows():
            logger.debug(
                "  %s: TL=%s, max_length=%s cm",
                row["common_name"],
                row["trophic_level"],
                row["max_length"],
            )

    def test_model_creation_from_biodata(self):
        simple_species = ["cod", "herring"]
        df = batch_get_species_info(
            simple_species,
            include_occurrences=True,
            include_traits=True,
            strict=False,
            timeout=45,
        )
        assert df is not None and len(df) > 0

        biomass_estimates = {row["common_name"]: 1.0 for _, row in df.iterrows()}
        params = biodata_to_rpath(
            df, biomass_estimates=biomass_estimates, area_km2=1000
        )
        assert params is not None
        assert len(params.model) > 0
        logger.debug(
            "Created model with %d groups, %d diet entries",
            len(params.model),
            (params.diet.iloc[:, 1:] > 0).sum().sum(),
        )


def _get_or_skip(service: str, url: str, **kwargs):
    """GET ``url``; skip the test when the service cannot be reached.

    A connection that is refused, times out or fails its TLS handshake says
    something about the network or the service, not about PyPath: a GitHub
    runner cannot reach WoRMS, for one. An answer with a wrong status still
    fails the test.
    """
    try:
        return requests.get(url, timeout=10, **kwargs)
    except requests.exceptions.RequestException as err:
        pytest.skip(f"{service} not reachable from here: {type(err).__name__}")


@pytest.mark.integration
@pytest.mark.slow
class TestAPIConnectivity:
    """Test that external biodiversity APIs answer, where they can be reached."""

    def test_worms_api(self):
        response = _get_or_skip(
            "WoRMS",
            "https://www.marinespecies.org/rest/AphiaRecordsByVernacular/cod",
            params={"like": "false", "offset": 1},
        )
        assert response.status_code == 200
        data = response.json()
        logger.debug(
            "WoRMS API: status=%d, results=%d", response.status_code, len(data)
        )

    def test_obis_api(self):
        response = _get_or_skip(
            "OBIS",
            "https://api.obis.org/v3/occurrence",
            params={"scientificname": "Gadus morhua", "size": 1},
        )
        assert response.status_code == 200
        logger.debug("OBIS API: status=%d", response.status_code)

    def test_fishbase_api(self):
        # FishBase is read from parquet snapshots; the diet table is a small one.
        response = _get_or_skip(
            "FishBase",
            "https://data.source.coop/cboettig/fishbase/fb/v26.06/parquet/diet.parquet",
        )
        assert response.status_code == 200
        assert response.content[:4] == b"PAR1"  # a parquet file
        logger.debug("FishBase snapshot: status=%d", response.status_code)
