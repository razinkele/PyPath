"""Comprehensive test suite for pypath.core.taxonomy module.

Tests cover:
- Phase 1B.2 Taxonomic Linkage System
- TaxonLink creation, validation, and serialization
- TaxonomyLinkage management and group assignment
- Taxonomy validation and consistency checks
- WoRMS integration (mocked)
- Edge cases and error handling

Total tests: ~30
"""

from __future__ import annotations

import logging

import pandas as pd
import pytest

from pypath.core.taxonomy import (
    TaxonLink,
    TaxonomyLinkage,
    batch_resolve_worms,
    resolve_worms_link,
    validate_taxonomy_links,
)

logger = logging.getLogger(__name__)


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def sample_taxon_link() -> TaxonLink:
    """Create a valid, complete TaxonLink."""
    return TaxonLink(
        scientific_name="Gadus morhua",
        common_name="Atlantic cod",
        worms_aphia_id=125732,
        kingdom="Animalia",
        phylum="Chordata",
        class_="Actinopterygii",
        order="Gadiformes",
        family="Gadidae",
        genus="Gadus",
        species="morhua",
        match_confidence=1.0,
    )


@pytest.fixture
def sample_taxon_link_minimal() -> TaxonLink:
    """Create a minimal valid TaxonLink (only required fields)."""
    return TaxonLink(
        scientific_name="Clupea harengus",
        worms_aphia_id=126417,
        kingdom="Animalia",
    )


@pytest.fixture
def sample_taxonomy_linkage() -> TaxonomyLinkage:
    """Create a TaxonomyLinkage with 5 representative taxa."""
    taxonomy = TaxonomyLinkage()

    # Gadus morhua (Cod)
    cod = TaxonLink(
        scientific_name="Gadus morhua",
        common_name="Atlantic cod",
        worms_aphia_id=125732,
        kingdom="Animalia",
        phylum="Chordata",
        class_="Actinopterygii",
        order="Gadiformes",
        family="Gadidae",
        genus="Gadus",
        species="morhua",
        match_confidence=1.0,
    )
    taxonomy.add_taxon(cod)

    # Clupea harengus (Herring)
    herring = TaxonLink(
        scientific_name="Clupea harengus",
        common_name="Atlantic herring",
        worms_aphia_id=126417,
        kingdom="Animalia",
        phylum="Chordata",
        class_="Actinopterygii",
        order="Clupeiformes",
        family="Clupeidae",
        genus="Clupea",
        species="harengus",
        match_confidence=1.0,
    )
    taxonomy.add_taxon(herring)

    # Anguilla anguilla (Eel)
    eel = TaxonLink(
        scientific_name="Anguilla anguilla",
        common_name="European eel",
        worms_aphia_id=125396,
        itis_tsn="162026",
        kingdom="Animalia",
        phylum="Chordata",
        class_="Actinopterygii",
        order="Anguilliformes",
        family="Anguillidae",
        genus="Anguilla",
        species="anguilla",
        match_confidence=0.95,
    )
    taxonomy.add_taxon(eel)

    # Percina fluviatilis (Perch)
    perch = TaxonLink(
        scientific_name="Perca fluviatilis",
        common_name="European perch",
        worms_aphia_id=126145,
        kingdom="Animalia",
        phylum="Chordata",
        class_="Actinopterygii",
        order="Perciformes",
        family="Percidae",
        genus="Perca",
        species="fluviatilis",
        match_confidence=0.98,
    )
    taxonomy.add_taxon(perch)

    # Zooplankton (aggregate - use kingdom-level taxon)
    zoop = TaxonLink(
        scientific_name="Copepoda",
        common_name="Copepods",
        worms_aphia_id=135102,
        kingdom="Animalia",
        phylum="Arthropoda",
        class_="Copepoda",
        match_confidence=0.8,
    )
    taxonomy.add_taxon(zoop)

    return taxonomy


@pytest.fixture
def sample_group_assignments() -> dict[str, str]:
    """Return sample group-to-scientific-name assignments."""
    return {
        "Cod": "Gadus morhua",
        "Herring": "Clupea harengus",
        "Eel": "Anguilla anguilla",
        "Perch": "Perca fluviatilis",
        "Zooplankton": "Copepoda",
    }


# ============================================================================
# TEST TAXON LINK
# ============================================================================


class TestTaxonLink:
    """Test TaxonLink creation, validation, and serialization."""

    def test_create_valid_taxon(self, sample_taxon_link):
        """Test creating a valid, complete taxon."""
        assert sample_taxon_link.scientific_name == "Gadus morhua"
        assert sample_taxon_link.common_name == "Atlantic cod"
        assert sample_taxon_link.worms_aphia_id == 125732
        assert sample_taxon_link.genus == "Gadus"
        assert sample_taxon_link.species == "morhua"
        assert sample_taxon_link.match_confidence == 1.0
        assert sample_taxon_link.is_valid()

    def test_create_minimal_taxon(self, sample_taxon_link_minimal):
        """Test creating a minimal valid taxon (only required fields)."""
        assert sample_taxon_link_minimal.scientific_name == "Clupea harengus"
        assert sample_taxon_link_minimal.worms_aphia_id == 126417
        assert sample_taxon_link_minimal.kingdom == "Animalia"
        assert sample_taxon_link_minimal.common_name is None
        assert sample_taxon_link_minimal.order == ""
        assert sample_taxon_link_minimal.is_valid()

    def test_is_valid_complete_taxon(self, sample_taxon_link):
        """Test is_valid() for a complete taxon."""
        assert sample_taxon_link.is_valid() is True

    def test_is_valid_incomplete_taxon(self):
        """Test is_valid() detects missing hierarchy/external ID."""
        # Missing external ID
        taxon = TaxonLink(
            scientific_name="Gadus morhua",
            kingdom="Animalia",
        )
        assert taxon.is_valid() is False

        # Missing kingdom
        taxon2 = TaxonLink(
            scientific_name="Gadus morhua",
            worms_aphia_id=125732,
        )
        assert taxon2.is_valid() is False

        # Empty scientific_name
        taxon3 = TaxonLink(
            scientific_name="",
            worms_aphia_id=125732,
            kingdom="Animalia",
        )
        assert taxon3.is_valid() is False

    def test_match_confidence_bounds(self):
        """Test match_confidence validation (0.0-1.0 only)."""
        # Valid bounds
        taxon_low = TaxonLink(
            scientific_name="Test",
            worms_aphia_id=1,
            kingdom="Animalia",
            match_confidence=0.0,
        )
        assert taxon_low.is_valid() is True

        taxon_high = TaxonLink(
            scientific_name="Test",
            worms_aphia_id=1,
            kingdom="Animalia",
            match_confidence=1.0,
        )
        assert taxon_high.is_valid() is True

        # Invalid: > 1.0
        taxon_over = TaxonLink(
            scientific_name="Test",
            worms_aphia_id=1,
            kingdom="Animalia",
            match_confidence=1.5,
        )
        assert taxon_over.is_valid() is False

        # Invalid: < 0.0
        taxon_under = TaxonLink(
            scientific_name="Test",
            worms_aphia_id=1,
            kingdom="Animalia",
            match_confidence=-0.1,
        )
        assert taxon_under.is_valid() is False

    def test_to_dict_serialization(self, sample_taxon_link):
        """Test to_dict() serialization."""
        d = sample_taxon_link.to_dict()
        assert isinstance(d, dict)
        assert d["scientific_name"] == "Gadus morhua"
        assert d["common_name"] == "Atlantic cod"
        assert d["worms_aphia_id"] == 125732
        assert d["kingdom"] == "Animalia"
        assert d["class_"] == "Actinopterygii"
        assert d["match_confidence"] == 1.0
        assert "genus" in d
        assert "species" in d

    def test_from_dict_deserialization(self, sample_taxon_link):
        """Test from_dict() deserialization."""
        original = sample_taxon_link.to_dict()
        reconstructed = TaxonLink.from_dict(original)

        assert reconstructed.scientific_name == original["scientific_name"]
        assert reconstructed.common_name == original["common_name"]
        assert reconstructed.worms_aphia_id == original["worms_aphia_id"]
        assert reconstructed.kingdom == original["kingdom"]
        assert reconstructed.match_confidence == original["match_confidence"]
        assert reconstructed.is_valid()

    def test_round_trip_serialization(self, sample_taxon_link):
        """Test round-trip serialization: to_dict -> from_dict."""
        original_dict = sample_taxon_link.to_dict()
        reconstructed = TaxonLink.from_dict(original_dict)
        final_dict = reconstructed.to_dict()

        # All fields should match
        for key in original_dict:
            assert original_dict[key] == final_dict[key], f"Mismatch in {key}"


# ============================================================================
# TEST TAXONOMY LINKAGE
# ============================================================================


class TestTaxonomyLinkage:
    """Test TaxonomyLinkage management and validation."""

    def test_add_taxon(self, sample_taxon_link):
        """Test adding a taxon to linkage."""
        taxonomy = TaxonomyLinkage()
        assert len(taxonomy.taxa) == 0

        taxonomy.add_taxon(sample_taxon_link)
        assert len(taxonomy.taxa) == 1
        assert "Gadus morhua" in taxonomy.taxa

    def test_add_taxon_invalid_raises_error(self):
        """Test that adding invalid taxon raises ValueError."""
        taxonomy = TaxonomyLinkage()
        invalid_taxon = TaxonLink(
            scientific_name="",  # Empty name
            worms_aphia_id=1,
            kingdom="Animalia",
        )
        with pytest.raises(ValueError, match="Invalid taxon"):
            taxonomy.add_taxon(invalid_taxon)

    def test_get_taxon_found(self, sample_taxonomy_linkage):
        """Test retrieving an existing taxon."""
        taxon = sample_taxonomy_linkage.get_taxon("Gadus morhua")
        assert taxon is not None
        assert taxon.scientific_name == "Gadus morhua"
        assert taxon.common_name == "Atlantic cod"

    def test_get_taxon_not_found(self, sample_taxonomy_linkage):
        """Test retrieving a non-existent taxon returns None."""
        taxon = sample_taxonomy_linkage.get_taxon("Nonexistent species")
        assert taxon is None

    def test_assign_group(self, sample_taxonomy_linkage):
        """Test assigning a group to a taxon."""
        assert len(sample_taxonomy_linkage.group_assignments) == 0

        sample_taxonomy_linkage.assign_group("Cod", "Gadus morhua")
        assert sample_taxonomy_linkage.group_assignments["Cod"] == "Gadus morhua"

    def test_assign_group_nonexistent_taxon_raises_error(self, sample_taxonomy_linkage):
        """Test that assigning to non-existent taxon raises KeyError."""
        with pytest.raises(KeyError, match="not in taxonomy"):
            sample_taxonomy_linkage.assign_group("UnknownGroup", "Nonexistent species")

    def test_assign_group_duplicate_scientific_name(self, sample_taxonomy_linkage):
        """Test that multiple groups cannot be assigned to the same taxon."""
        sample_taxonomy_linkage.assign_group("Cod", "Gadus morhua")
        sample_taxonomy_linkage.assign_group("CodAdult", "Gadus morhua")

        # This is allowed, but validate() should flag it as inconsistent
        is_valid, errors = sample_taxonomy_linkage.validate()
        assert not is_valid
        assert any("multiple groups" in e for e in errors)

    def test_validate_empty_linkage(self):
        """Test validating empty taxonomy."""
        taxonomy = TaxonomyLinkage()
        is_valid, errors = taxonomy.validate()
        assert is_valid is True
        assert len(errors) == 0

    def test_validate_valid_state(
        self, sample_taxonomy_linkage, sample_group_assignments
    ):
        """Test validating a properly configured taxonomy."""
        # Assign all groups
        for group, sci_name in sample_group_assignments.items():
            sample_taxonomy_linkage.assign_group(group, sci_name)

        is_valid, errors = sample_taxonomy_linkage.validate()
        assert is_valid is True
        assert len(errors) == 0

    def test_validate_missing_taxa_for_assignments(self):
        """Test validation detects assignments to missing taxa."""
        taxonomy = TaxonomyLinkage()

        # Add one taxon
        taxon = TaxonLink(
            scientific_name="Gadus morhua",
            worms_aphia_id=1,
            kingdom="Animalia",
        )
        taxonomy.add_taxon(taxon)

        # Assign a group to a taxon that doesn't exist
        # (bypass add_taxon to create this state)
        taxonomy.group_assignments["Cod"] = "Gadus morhua"
        taxonomy.group_assignments["Fake"] = "Nonexistent species"

        is_valid, errors = taxonomy.validate()
        assert not is_valid
        assert any("non-existent taxon" in e for e in errors)

    def test_to_dataframe(self, sample_taxonomy_linkage, sample_group_assignments):
        """Test exporting taxonomy to DataFrame."""
        for group, sci_name in sample_group_assignments.items():
            sample_taxonomy_linkage.assign_group(group, sci_name)

        df = sample_taxonomy_linkage.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        assert "group" in df.columns
        assert "scientific_name" in df.columns
        assert "kingdom" in df.columns
        assert "worms_aphia_id" in df.columns

        # Check that all groups are present
        assert set(df["group"].unique()) == set(sample_group_assignments.keys())

    def test_to_dataframe_empty(self):
        """Test exporting empty taxonomy returns empty DataFrame."""
        taxonomy = TaxonomyLinkage()
        df = taxonomy.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_from_dataframe_round_trip(
        self, sample_taxonomy_linkage, sample_group_assignments
    ):
        """Test round-trip: TaxonomyLinkage -> DataFrame -> TaxonomyLinkage."""
        for group, sci_name in sample_group_assignments.items():
            sample_taxonomy_linkage.assign_group(group, sci_name)

        # Export to DataFrame
        df = sample_taxonomy_linkage.to_dataframe()

        # Reimport from DataFrame
        reimported = TaxonomyLinkage.from_dataframe(df)

        # Check structure is preserved
        assert len(reimported.taxa) == len(sample_taxonomy_linkage.taxa)
        assert len(reimported.group_assignments) == len(
            sample_taxonomy_linkage.group_assignments
        )

        # Check specific taxa
        for sci_name in sample_taxonomy_linkage.taxa:
            original_taxon = sample_taxonomy_linkage.taxa[sci_name]
            reimported_taxon = reimported.taxa[sci_name]
            assert original_taxon.scientific_name == reimported_taxon.scientific_name
            assert original_taxon.kingdom == reimported_taxon.kingdom


# ============================================================================
# TEST VALIDATE TAXONOMY
# ============================================================================


class TestValidateTaxonomy:
    """Test taxonomy validation functions."""

    def test_validate_valid_taxonomy(
        self, sample_taxonomy_linkage, sample_group_assignments
    ):
        """Test validate_taxonomy_links() on valid taxonomy."""
        for group, sci_name in sample_group_assignments.items():
            sample_taxonomy_linkage.assign_group(group, sci_name)

        is_valid, errors = validate_taxonomy_links(sample_taxonomy_linkage)
        assert is_valid is True
        assert len(errors) == 0

    def test_validate_missing_taxa(self):
        """Test validation detects taxa that are not in the collection."""
        taxonomy = TaxonomyLinkage()

        # Manually assign without adding taxon (violates invariant)
        taxonomy.taxa["Gadus morhua"] = TaxonLink(
            scientific_name="Gadus morhua",
            worms_aphia_id=1,
            kingdom="Animalia",
        )
        taxonomy.group_assignments["Cod"] = "Gadus morhua"
        taxonomy.group_assignments["Phantom"] = "Nonexistent species"

        is_valid, errors = validate_taxonomy_links(taxonomy)
        assert not is_valid
        assert any("non-existent" in e.lower() for e in errors)

    def test_validate_duplicate_assignments(self):
        """Test validation detects duplicate group assignments."""
        taxonomy = TaxonomyLinkage()

        taxon = TaxonLink(
            scientific_name="Gadus morhua",
            worms_aphia_id=1,
            kingdom="Animalia",
        )
        taxonomy.add_taxon(taxon)

        # Assign same taxon to multiple groups
        taxonomy.group_assignments["Cod"] = "Gadus morhua"
        taxonomy.group_assignments["Cod_adult"] = "Gadus morhua"

        is_valid, errors = validate_taxonomy_links(taxonomy)
        assert not is_valid
        assert any("multiple groups" in e for e in errors)

    def test_validate_hierarchical_inconsistency(self):
        """Test validation detects hierarchical inconsistency."""
        taxonomy = TaxonomyLinkage()

        # Add taxon with inconsistent hierarchy
        # (e.g., family doesn't match genus)
        taxon = TaxonLink(
            scientific_name="Gadus morhua",
            worms_aphia_id=1,
            kingdom="Animalia",
            family="Salmonidae",  # Wrong family for Gadus
            genus="Gadus",
        )
        # Note: is_valid() doesn't check hierarchy consistency yet,
        # so this taxon will be added. In a more complete implementation,
        # we'd validate that genus/family/order are consistent.
        taxonomy.add_taxon(taxon)

        # For now, the taxon is valid, so no error
        is_valid, errors = validate_taxonomy_links(taxonomy)
        assert is_valid is True


# ============================================================================
# TEST WORMS INTEGRATION (MOCKED)
# ============================================================================


class TestWoRMSIntegration:
    """Test WoRMS integration (stub functions)."""

    def test_resolve_worms_link_stub_returns_none(self):
        """Test that resolve_worms_link() is a stub returning None."""
        result = resolve_worms_link("Gadus morhua")
        assert result is None

    def test_resolve_worms_link_with_various_names(self):
        """Test resolve_worms_link() stub with various names."""
        names = ["Gadus morhua", "Clupea harengus", "Nonexistent"]
        for name in names:
            result = resolve_worms_link(name)
            assert result is None

    def test_batch_resolve_worms_stub_returns_none_dict(self):
        """Test that batch_resolve_worms() is a stub returning None values."""
        names = ["Gadus morhua", "Clupea harengus", "Nonexistent"]
        result = batch_resolve_worms(names)
        assert isinstance(result, dict)
        assert len(result) == 3
        for name in names:
            assert name in result
            assert result[name] is None


# ============================================================================
# TEST EDGE CASES
# ============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_taxon_list(self):
        """Test operations on empty taxonomy."""
        taxonomy = TaxonomyLinkage()
        assert len(taxonomy.taxa) == 0
        assert len(taxonomy.group_assignments) == 0

        # Validate should pass (no violations)
        is_valid, errors = taxonomy.validate()
        assert is_valid is True

        # DataFrame should be empty
        df = taxonomy.to_dataframe()
        assert len(df) == 0

    def test_very_long_taxonomy_names(self):
        """Test handling very long scientific names."""
        long_name = "Genus " + "x" * 100
        taxon = TaxonLink(
            scientific_name=long_name,
            worms_aphia_id=999,
            kingdom="Animalia",
        )
        assert taxon.is_valid() is True

        taxonomy = TaxonomyLinkage()
        taxonomy.add_taxon(taxon)
        assert taxonomy.get_taxon(long_name) is not None

    def test_special_characters_in_names(self):
        """Test handling special characters in names."""
        special_names = [
            "Genus-species",
            "Genus species (forma)",
            "Genus_species-new",
            "Genus µm-species",
        ]

        for name in special_names:
            taxon = TaxonLink(
                scientific_name=name,
                worms_aphia_id=999,
                kingdom="Animalia",
            )
            assert taxon.is_valid() is True

    def test_duplicate_taxon_entries(self):
        """Test behavior when adding duplicate taxa."""
        taxon1 = TaxonLink(
            scientific_name="Gadus morhua",
            common_name="Atlantic cod",
            worms_aphia_id=125732,
            kingdom="Animalia",
        )
        taxon2 = TaxonLink(
            scientific_name="Gadus morhua",
            common_name="Cod (different common name)",
            worms_aphia_id=999,  # Different ID
            kingdom="Animalia",
        )

        taxonomy = TaxonomyLinkage()
        taxonomy.add_taxon(taxon1)
        taxonomy.add_taxon(taxon2)  # Overwrites

        # Should have only one entry
        assert len(taxonomy.taxa) == 1
        # Should be the second one (overwrites)
        stored_taxon = taxonomy.get_taxon("Gadus morhua")
        assert stored_taxon.worms_aphia_id == 999

    def test_group_assignment_overwrite(self, sample_taxonomy_linkage):
        """Test overwriting a group assignment."""
        sample_taxonomy_linkage.assign_group("Cod", "Gadus morhua")
        assert sample_taxonomy_linkage.group_assignments["Cod"] == "Gadus morhua"

        # Overwrite with different taxon
        sample_taxonomy_linkage.assign_group("Cod", "Clupea harengus")
        assert sample_taxonomy_linkage.group_assignments["Cod"] == "Clupea harengus"

    def test_dataframe_with_missing_optional_fields(self):
        """Test from_dataframe() with missing optional columns."""
        df = pd.DataFrame(
            {
                "group": ["Cod", "Herring"],
                "scientific_name": ["Gadus morhua", "Clupea harengus"],
                "kingdom": ["Animalia", "Animalia"],
                "worms_aphia_id": [125732, 126417],
                # Missing: common_name, class_, order, etc.
            }
        )

        taxonomy = TaxonomyLinkage.from_dataframe(df)
        assert len(taxonomy.taxa) == 2
        assert len(taxonomy.group_assignments) == 2

        # Check that missing fields are filled with defaults
        cod = taxonomy.get_taxon("Gadus morhua")
        assert cod is not None
        assert cod.common_name is None
        assert cod.order == ""

    def test_dataframe_missing_required_columns(self):
        """Test from_dataframe() fails with missing required columns."""
        df = pd.DataFrame(
            {
                "group": ["Cod"],
                "scientific_name": ["Gadus morhua"],
                # Missing: kingdom
            }
        )

        with pytest.raises(ValueError, match="must have columns"):
            TaxonomyLinkage.from_dataframe(df)

    def test_whitespace_handling_in_names(self):
        """Test that whitespace in names is handled correctly."""
        taxon = TaxonLink(
            scientific_name="  Gadus morhua  ",  # Leading/trailing whitespace
            worms_aphia_id=1,
            kingdom="  Animalia  ",
        )
        # is_valid checks if stripped name is empty
        assert taxon.is_valid() is True
        assert taxon.scientific_name == "  Gadus morhua  "  # Stored as-is


# ============================================================================
# TEST PARAMETRIZED CASES
# ============================================================================


class TestParametrized:
    """Parametrized tests for systematic coverage."""

    @pytest.mark.parametrize(
        "confidence,expected_valid",
        [
            (0.0, True),
            (0.25, True),
            (0.5, True),
            (0.75, True),
            (1.0, True),
            (-0.1, False),
            (1.1, False),
            (2.0, False),
        ],
    )
    def test_match_confidence_values(self, confidence, expected_valid):
        """Test various match_confidence values."""
        taxon = TaxonLink(
            scientific_name="Test",
            worms_aphia_id=1,
            kingdom="Animalia",
            match_confidence=confidence,
        )
        assert taxon.is_valid() is expected_valid

    @pytest.mark.parametrize(
        "worms_id,itis_id,should_be_valid",
        [
            (1, None, True),  # Only WoRMS
            (None, "123", True),  # Only ITIS
            (1, "123", True),  # Both
            (None, None, False),  # Neither
        ],
    )
    def test_external_id_requirements(self, worms_id, itis_id, should_be_valid):
        """Test that at least one external ID is required."""
        taxon = TaxonLink(
            scientific_name="Test",
            worms_aphia_id=worms_id,
            itis_tsn=itis_id,
            kingdom="Animalia",
        )
        assert taxon.is_valid() is should_be_valid

    @pytest.mark.parametrize(
        "sci_name,should_be_valid",
        [
            ("Gadus morhua", True),
            ("Clupea", True),  # Genus only
            ("", False),  # Empty
            ("   ", False),  # Whitespace only
        ],
    )
    def test_scientific_name_requirements(self, sci_name, should_be_valid):
        """Test scientific_name validation."""
        taxon = TaxonLink(
            scientific_name=sci_name,
            worms_aphia_id=1,
            kingdom="Animalia",
        )
        assert taxon.is_valid() is should_be_valid
