"""
Tests for diet matrix import from EcoBase and ewemdb files.

These tests verify that diet composition data is correctly parsed
and loaded into RpathParams.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pypath.io.ecobase import (
    get_ecobase_model,
    ecobase_to_rpath,
    list_ecobase_models,
)
from pypath.core.params import RpathParams
import xml.etree.ElementTree as ET


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
        print(f"\n=== Diet data found: {len(diet)} predators ===")

        if diet:
            for pred, prey_dict in list(diet.items())[:3]:
                print(f"  {pred}: {list(prey_dict.keys())[:5]}...")
        else:
            print("  WARNING: Diet dictionary is EMPTY")

            # Debug: Look at raw XML for diet-related tags
            print("\n=== Debugging XML structure ===")
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
            print(f"Diet-related tags: {diet_tags}")

            # Look at first group's fields
            print("\n=== First group fields ===")
            for group in root.iter("group"):
                for child in group:
                    print(
                        f"  {child.tag}: {child.text[:50] if child.text and len(child.text) > 50 else child.text}"
                    )
                break

        # This assertion will fail if diet is empty, helping us debug
        assert len(diet) > 0, "Diet dictionary should not be empty"

    def test_group_structure(self, sample_model_data):
        """Test group data structure for diet-related fields."""
        groups = sample_model_data["groups"]

        print(f"\n=== Checking {len(groups)} groups for diet fields ===")

        # Check first few groups for dc/diet fields
        diet_fields_found = []
        for i, g in enumerate(groups[:5]):
            group_name = g.get("group_name", g.get("name", f"Group {i}"))
            dc_fields = {
                k: v for k, v in g.items() if "dc" in k.lower() or "diet" in k.lower()
            }

            if dc_fields:
                print(f"  {group_name}: {dc_fields}")
                diet_fields_found.append((group_name, dc_fields))
            else:
                # Show all fields
                print(f"  {group_name} fields: {list(g.keys())}")

        print(f"\nGroups with diet fields: {len(diet_fields_found)}")

    def test_xml_diet_elements(self, sample_model_data):
        """Test for diet elements in raw XML."""
        root = ET.fromstring(sample_model_data["raw_xml"])

        # Count different potential diet element types
        diet_elements = list(root.iter("diet"))
        diet_item_elements = list(root.iter("diet_item"))
        dc_elements = list(root.iter("dc"))

        print(f"\n=== Diet XML elements ===")
        print(f"  <diet> elements: {len(diet_elements)}")
        print(f"  <diet_item> elements: {len(diet_item_elements)}")
        print(f"  <dc> elements: {len(dc_elements)}")

        # Look for any element containing 'diet' in tag
        diet_related = []
        for elem in root.iter():
            if "diet" in elem.tag.lower():
                diet_related.append(elem.tag)

        print(f"  All diet-related tags: {set(diet_related)}")

    def test_ecobase_to_rpath_diet(self, sample_model_data):
        """Test that diet matrix is populated in RpathParams."""
        params = ecobase_to_rpath(sample_model_data)

        assert isinstance(params, RpathParams)
        assert params.diet is not None

        # Exclude 'Group' column for numeric comparisons
        diet_numeric = params.diet.drop(columns=["Group"], errors="ignore")

        # Check if diet matrix has any non-zero values
        non_zero = (diet_numeric > 0).sum().sum()

        print(f"\n=== RpathParams diet matrix ===")
        print(f"  Shape: {params.diet.shape}")
        print(f"  Non-zero entries: {non_zero}")
        print(f"  Columns (predators): {list(params.diet.columns)[:5]}...")
        print(f"  Groups (prey): {params.diet['Group'].tolist()[:5]}...")

        if non_zero > 0:
            # Show some non-zero entries
            print("\n  Sample diet entries:")
            for col in diet_numeric.columns[:3]:
                col_data = diet_numeric[col]
                non_zero_prey = col_data[col_data > 0]
                if len(non_zero_prey) > 0:
                    # Get prey names for these indices
                    prey_names = [
                        params.diet.loc[idx, "Group"] for idx in non_zero_prey.index[:3]
                    ]
                    values = non_zero_prey.head(3).tolist()
                    print(f"    {col}: {dict(zip(prey_names, values))}")
        else:
            print("\n  WARNING: Diet matrix is all zeros!")

        assert non_zero > 0, "Diet matrix should have non-zero entries"


class TestEcoBaseXMLStructure:
    """Deep dive into EcoBase XML structure to find diet data."""

    def test_find_diet_in_xml(self):
        """Thoroughly search for diet data in EcoBase XML."""
        model_data = get_ecobase_model(403)
        root = ET.fromstring(model_data["raw_xml"])

        print("\n=== Complete tag inventory ===")
        tag_counts = {}
        for elem in root.iter():
            tag_counts[elem.tag] = tag_counts.get(elem.tag, 0) + 1

        for tag, count in sorted(tag_counts.items()):
            print(f"  {tag}: {count}")

        print("\n=== Looking for numeric sequences in group children ===")
        # In EcoBase, diet might be stored as numbered children like dc1, dc2, etc.
        for i, group in enumerate(root.iter("group")):
            if i >= 2:
                break
            print(f"\nGroup {i}:")
            for child in group:
                tag = child.tag
                text = child.text
                # Look for tags that might be diet-related
                if any(x in tag.lower() for x in ["dc", "diet", "prey", "prop"]):
                    print(f"  DIET? {tag}: {text}")
                elif tag.startswith("dc") or tag[0].isdigit():
                    print(f"  NUM? {tag}: {text}")

    def test_raw_xml_snippet(self):
        """Print raw XML snippet to see actual structure."""
        model_data = get_ecobase_model(403)
        xml = model_data["raw_xml"]

        print("\n=== Raw XML (first 5000 chars) ===")
        print(xml[:5000])

        print("\n=== Looking for 'Diet' in XML ===")
        if "Diet" in xml or "diet" in xml:
            # Find context around 'diet'
            idx = xml.lower().find("diet")
            if idx > 0:
                start = max(0, idx - 100)
                end = min(len(xml), idx + 200)
                print(f"Context: ...{xml[start:end]}...")


class TestEwemdbDietParsing:
    """Test diet matrix parsing from ewemdb files."""

    def test_ewemdb_imports_available(self):
        """Test that ewemdb imports work."""
        from pypath.io.ewemdb import (
            check_ewemdb_support,
            read_ewemdb_table,
            read_ewemdb,
        )

        support = check_ewemdb_support()
        print(f"\n=== ewemdb driver support ===")
        print(f"  pyodbc: {support['pyodbc']}")
        print(f"  pypyodbc: {support['pypyodbc']}")
        print(f"  mdb_tools: {support['mdb_tools']}")
        print(f"  any_available: {support['any_available']}")

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
        print(f"\n=== Reading from {filepath.name} ===")

        # Try to read diet table
        try:
            diet_df = read_ewemdb_table(str(filepath), "EcopathDietComp")
            print(f"EcopathDietComp columns: {diet_df.columns.tolist()}")
            print(f"EcopathDietComp shape: {diet_df.shape}")
            print(f"First few rows:\n{diet_df.head()}")
        except Exception as e:
            print(f"Could not read EcopathDietComp: {e}")

            # Try alternative names
            for table_name in ["DietComp", "Diet", "EcopathDiet"]:
                try:
                    diet_df = read_ewemdb_table(str(filepath), table_name)
                    print(f"\n{table_name} columns: {diet_df.columns.tolist()}")
                    print(f"{table_name} shape: {diet_df.shape}")
                    break
                except:
                    continue


class TestDietMatrixIntegration:
    """Integration tests for diet matrix through full import pipeline."""

    def test_full_ecobase_import(self):
        """Test complete EcoBase import pipeline."""
        print("\n=== Full EcoBase import test ===")

        # Download model
        model_data = get_ecobase_model(403)
        print(f"Downloaded model with {len(model_data['groups'])} groups")
        print(f"Diet entries in model_data: {len(model_data['diet'])}")

        # Convert to RpathParams
        params = ecobase_to_rpath(model_data)
        print(f"Created RpathParams with {len(params.model)} groups")

        # Check diet matrix (exclude Group column for numeric operations)
        diet_numeric = params.diet.drop(columns=["Group"], errors="ignore")
        diet_sum = diet_numeric.sum().sum()
        non_zero = (diet_numeric > 0).sum().sum()

        print(f"Diet matrix sum: {diet_sum}")
        print(f"Diet matrix non-zero cells: {non_zero}")

        # Print diet matrix summary
        print("\nDiet matrix preview:")
        print(params.diet.iloc[:5, :5])

        assert non_zero > 0, "Diet matrix should have non-zero entries"

        return params


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s", "--tb=short"])
