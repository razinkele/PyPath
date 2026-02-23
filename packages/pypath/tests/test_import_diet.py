"""
Tests for diet matrix import from EcoBase and ewemdb files.

These tests verify that diet composition data is correctly parsed
and loaded into RpathParams.
"""

import logging
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from pypath.core.params import RpathParams
from pypath.io.ecobase import (
    ecobase_to_rpath,
    get_ecobase_model,
)

logger = logging.getLogger(__name__)


class TestEcoBaseDietParsing:
    """Test diet matrix parsing from EcoBase XML."""

    @pytest.fixture
    def sample_model_data(self):
        """Download a sample model for testing."""
        # Model 403 is Western Channel, a well-documented model
        return get_ecobase_model(403)

    def test_model_download(self, sample_model_data):
        """Test that model data can be downloaded."""
        assert sample_model_data is not None
        assert "groups" in sample_model_data
        assert "diet" in sample_model_data
        assert "raw_xml" in sample_model_data
        assert len(sample_model_data["groups"]) > 0

    def test_diet_data_extracted(self, sample_model_data):
        """Test that diet data is extracted from model."""
        diet = sample_model_data["diet"]
        logger.debug("Diet data found: %d predators", len(diet))

        if diet:
            for pred, prey_dict in list(diet.items())[:3]:
                logger.debug("  %s: %s...", pred, list(prey_dict.keys())[:5])
        else:
            logger.debug("WARNING: Diet dictionary is EMPTY")

            # Debug: Look at raw XML for diet-related tags
            root = ET.fromstring(sample_model_data["raw_xml"])

            # Find all unique tags
            all_tags = set()
            for elem in root.iter():
                all_tags.add(elem.tag)

            diet_tags = [
                t
                for t in all_tags
                if "diet" in t.lower() or "dc" in t.lower() or "prey" in t.lower()
            ]
            logger.debug("Diet-related tags: %s", diet_tags)

            for group in root.iter("group"):
                for child in group:
                    logger.debug(
                        "  %s: %s",
                        child.tag,
                        child.text[:50]
                        if child.text and len(child.text) > 50
                        else child.text,
                    )
                break

        # This assertion will fail if diet is empty, helping us debug
        assert len(diet) > 0, "Diet dictionary should not be empty"

    def test_group_structure(self, sample_model_data):
        """Test group data structure for diet-related fields."""
        groups = sample_model_data["groups"]

        logger.debug("Checking %d groups for diet fields", len(groups))

        # Check first few groups for dc/diet fields
        diet_fields_found = []
        for i, g in enumerate(groups[:5]):
            group_name = g.get("group_name", g.get("name", f"Group {i}"))
            dc_fields = {
                k: v for k, v in g.items() if "dc" in k.lower() or "diet" in k.lower()
            }

            if dc_fields:
                logger.debug("  %s: %s", group_name, dc_fields)
                diet_fields_found.append((group_name, dc_fields))
            else:
                logger.debug("  %s fields: %s", group_name, list(g.keys()))

        logger.debug("Groups with diet fields: %d", len(diet_fields_found))

    def test_xml_diet_elements(self, sample_model_data):
        """Test for diet elements in raw XML."""
        root = ET.fromstring(sample_model_data["raw_xml"])

        # Count different potential diet element types
        diet_elements = list(root.iter("diet"))
        diet_item_elements = list(root.iter("diet_item"))
        dc_elements = list(root.iter("dc"))

        logger.debug("<diet> elements: %d", len(diet_elements))
        logger.debug("<diet_item> elements: %d", len(diet_item_elements))
        logger.debug("<dc> elements: %d", len(dc_elements))

        # Look for any element containing 'diet' in tag
        diet_related = []
        for elem in root.iter():
            if "diet" in elem.tag.lower():
                diet_related.append(elem.tag)

        logger.debug("All diet-related tags: %s", set(diet_related))

    def test_ecobase_to_rpath_diet(self, sample_model_data):
        """Test that diet matrix is populated in RpathParams."""
        params = ecobase_to_rpath(sample_model_data)

        assert isinstance(params, RpathParams)
        assert params.diet is not None

        # Exclude 'Group' column for numeric comparisons
        diet_numeric = params.diet.drop(columns=["Group"], errors="ignore")

        # Check if diet matrix has any non-zero values
        non_zero = (diet_numeric > 0).sum().sum()

        logger.debug("RpathParams diet matrix shape: %s", params.diet.shape)
        logger.debug("Non-zero entries: %d", non_zero)
        logger.debug("Columns (predators): %s...", list(params.diet.columns)[:5])
        logger.debug("Groups (prey): %s...", params.diet["Group"].tolist()[:5])

        if non_zero > 0:
            for col in diet_numeric.columns[:3]:
                col_data = diet_numeric[col]
                non_zero_prey = col_data[col_data > 0]
                if len(non_zero_prey) > 0:
                    prey_names = [
                        params.diet.loc[idx, "Group"] for idx in non_zero_prey.index[:3]
                    ]
                    values = non_zero_prey.head(3).tolist()
                    logger.debug("  %s: %s", col, dict(zip(prey_names, values)))
        else:
            logger.debug("WARNING: Diet matrix is all zeros!")

        assert non_zero > 0, "Diet matrix should have non-zero entries"


class TestEcoBaseXMLStructure:
    """Deep dive into EcoBase XML structure to find diet data."""

    def test_find_diet_in_xml(self):
        """Thoroughly search for diet data in EcoBase XML."""
        model_data = get_ecobase_model(403)
        root = ET.fromstring(model_data["raw_xml"])

        tag_counts = {}
        for elem in root.iter():
            tag_counts[elem.tag] = tag_counts.get(elem.tag, 0) + 1

        for tag, count in sorted(tag_counts.items()):
            logger.debug("  %s: %d", tag, count)

        # In EcoBase, diet might be stored as numbered children like dc1, dc2, etc.
        for i, group in enumerate(root.iter("group")):
            if i >= 2:
                break
            logger.debug("Group %d:", i)
            for child in group:
                tag = child.tag
                text = child.text
                # Look for tags that might be diet-related
                if any(x in tag.lower() for x in ["dc", "diet", "prey", "prop"]):
                    logger.debug("  DIET? %s: %s", tag, text)
                elif tag.startswith("dc") or tag[0].isdigit():
                    logger.debug("  NUM? %s: %s", tag, text)

    def test_raw_xml_snippet(self):
        """Print raw XML snippet to see actual structure."""
        model_data = get_ecobase_model(403)
        xml = model_data["raw_xml"]

        logger.debug("Raw XML (first 5000 chars): %s", xml[:5000])

        if "Diet" in xml or "diet" in xml:
            idx = xml.lower().find("diet")
            if idx > 0:
                start = max(0, idx - 100)
                end = min(len(xml), idx + 200)
                logger.debug("Context: ...%s...", xml[start:end])


class TestEwemdbDietParsing:
    """Test diet matrix parsing from ewemdb files."""

    def test_ewemdb_imports_available(self):
        """Test that ewemdb imports work."""
        from pypath.io.ewemdb import (
            check_ewemdb_support,
        )

        support = check_ewemdb_support()
        logger.debug(
            "ewemdb driver support: pyodbc=%s pypyodbc=%s mdb_tools=%s any=%s",
            support["pyodbc"],
            support["pypyodbc"],
            support["mdb_tools"],
            support["any_available"],
        )

    def test_diet_table_reading(self):
        """Test reading diet table from ewemdb file."""
        from pypath.io.ewemdb import check_ewemdb_support, read_ewemdb_table

        support = check_ewemdb_support()
        if not support["any_available"]:
            pytest.skip("No ewemdb drivers available")

        # Look for test files
        test_files = list(Path(__file__).parent.parent.glob("**/*.ewemdb"))
        if not test_files:
            test_files = list(Path(__file__).parent.parent.glob("**/*.mdb"))

        if not test_files:
            pytest.skip("No ewemdb test files found")

        filepath = test_files[0]
        logger.debug("Reading from %s", filepath.name)

        # Try to read diet table
        try:
            diet_df = read_ewemdb_table(str(filepath), "EcopathDietComp")
            logger.debug("EcopathDietComp columns: %s", diet_df.columns.tolist())
            logger.debug("EcopathDietComp shape: %s", diet_df.shape)
            logger.debug("First few rows:\n%s", diet_df.head())
        except Exception as e:
            logger.debug("Could not read EcopathDietComp: %s", e)

            # Try alternative names
            for table_name in ["DietComp", "Diet", "EcopathDiet"]:
                try:
                    diet_df = read_ewemdb_table(str(filepath), table_name)
                    logger.debug("%s columns: %s", table_name, diet_df.columns.tolist())
                    logger.debug("%s shape: %s", table_name, diet_df.shape)
                    break
                except Exception:
                    continue

    def test_full_ecobase_import(self):
        """Test complete EcoBase import pipeline."""
        # Download model
        model_data = get_ecobase_model(403)
        logger.debug("Downloaded model with %d groups", len(model_data["groups"]))
        logger.debug("Diet entries in model_data: %d", len(model_data["diet"]))

        # Convert to RpathParams
        params = ecobase_to_rpath(model_data)
        logger.debug("Created RpathParams with %d groups", len(params.model))

        # Check diet matrix (exclude Group column for numeric operations)
        diet_numeric = params.diet.drop(columns=["Group"], errors="ignore")
        diet_sum = diet_numeric.sum().sum()
        non_zero = (diet_numeric > 0).sum().sum()

        logger.debug("Diet matrix sum: %s", diet_sum)
        logger.debug("Diet matrix non-zero cells: %d", non_zero)
        logger.debug("Diet matrix preview:\n%s", params.diet.iloc[:5, :5])

        assert non_zero > 0, "Diet matrix should have non-zero entries"

        return params


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s", "--tb=short"])
