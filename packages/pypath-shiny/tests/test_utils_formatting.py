"""Tests for format_dataframe_for_display and create_cell_styles."""

import pandas as pd
import pytest

from pypath_shiny.pages.utils import create_cell_styles, format_dataframe_for_display


@pytest.fixture
def basic_df():
    return pd.DataFrame(
        {
            "Group": ["Fish", "Plankton", "Detritus"],
            "Type": [0, 1, 2],
            "Biomass": [10.123456, 9999, 1.0],
            "QB": [5.0, 9999, 9999],
        }
    )


class TestFormatDataframeForDisplay:
    def test_no_data_sentinel_becomes_nan(self, basic_df):
        fmt, nd, rm, sm = format_dataframe_for_display(basic_df)
        assert pd.isna(fmt["Biomass"].iloc[1])
        assert nd["Biomass"].iloc[1]

    def test_no_data_mask_false_for_valid(self, basic_df):
        fmt, nd, rm, sm = format_dataframe_for_display(basic_df)
        assert not nd["Biomass"].iloc[0]

    def test_negative_sentinel_also_masked(self):
        df = pd.DataFrame({"Group": ["Fish"], "Biomass": [-9999]})
        fmt, nd, rm, sm = format_dataframe_for_display(df)
        assert nd["Biomass"].iloc[0]
        assert pd.isna(fmt["Biomass"].iloc[0])

    def test_decimal_rounding_custom(self, basic_df):
        fmt, _, _, _ = format_dataframe_for_display(basic_df, decimal_places=2)
        assert fmt["Biomass"].iloc[0] == pytest.approx(10.12)

    def test_default_decimal_places_3(self):
        df = pd.DataFrame({"Biomass": [1.23456789]})
        fmt, _, _, _ = format_dataframe_for_display(df)
        assert fmt["Biomass"].iloc[0] == pytest.approx(1.235)

    def test_type_column_numeric_to_label(self, basic_df):
        fmt, _, _, _ = format_dataframe_for_display(basic_df)
        assert fmt["Type"].iloc[0] == "Consumer"
        assert fmt["Type"].iloc[1] == "Producer"
        assert fmt["Type"].iloc[2] == "Detritus"

    def test_stanza_groups_mask(self):
        df = pd.DataFrame(
            {
                "Group": ["SmeltJuv", "SmeltAdult", "Plankton"],
                "Biomass": [1.0, 2.0, 3.0],
            }
        )
        _, _, _, sm = format_dataframe_for_display(
            df, stanza_groups=["SmeltJuv", "SmeltAdult"]
        )
        assert sm["Biomass"].iloc[0]
        assert sm["Biomass"].iloc[1]
        assert not sm["Biomass"].iloc[2]

    def test_stanza_mask_none_no_effect(self, basic_df):
        _, _, _, sm = format_dataframe_for_display(basic_df, stanza_groups=None)
        assert sm.values.sum() == 0

    def test_remarks_mask(self):
        df = pd.DataFrame({"Group": ["Fish"], "Biomass": [1.0]})
        remarks_df = pd.DataFrame({"Group": [""], "Biomass": ["some remark"]})
        _, _, rm, _ = format_dataframe_for_display(df, remarks_df=remarks_df)
        assert rm["Biomass"].iloc[0]
        assert not rm["Group"].iloc[0]

    def test_empty_dataframe(self):
        df = pd.DataFrame({"Group": [], "Biomass": []})
        fmt, nd, rm, sm = format_dataframe_for_display(df)
        assert fmt.shape == (0, 2)

    def test_returns_four_dataframes(self, basic_df):
        result = format_dataframe_for_display(basic_df)
        assert len(result) == 4
        for r in result:
            assert isinstance(r, pd.DataFrame)


class TestCreateCellStyles:
    def test_no_data_cell_gets_gray_style(self):
        df = pd.DataFrame(
            {"Group": ["Fish"], "Type": ["Consumer"], "Biomass": [float("nan")]}
        )
        no_data = pd.DataFrame({"Group": [False], "Type": [False], "Biomass": [True]})
        styles = create_cell_styles(df, no_data)
        bio_col = df.columns.get_loc("Biomass")
        bio_styles = [s for s in styles if s["cols"] == bio_col]
        assert len(bio_styles) == 1
        assert bio_styles[0]["style"]["background-color"] == "#f0f0f0"

    def test_no_styles_for_valid_data(self):
        df = pd.DataFrame({"Group": ["Fish"], "Type": ["Consumer"], "Biomass": [10.0]})
        no_data = pd.DataFrame({"Group": [False], "Type": [False], "Biomass": [False]})
        styles = create_cell_styles(df, no_data)
        assert len(styles) == 0

    def test_qb_not_applicable_to_producer(self):
        df = pd.DataFrame({"Type": ["Consumer", "Producer"], "QB": [5.0, float("nan")]})
        no_data = pd.DataFrame({"Type": [False, False], "QB": [False, False]})
        styles = create_cell_styles(df, no_data)
        qb_col = df.columns.get_loc("QB")
        producer_styles = [s for s in styles if s["rows"] == 1 and s["cols"] == qb_col]
        assert len(producer_styles) >= 1
        assert "font-style" in producer_styles[0]["style"]

    def test_qb_not_applicable_to_detritus(self):
        df = pd.DataFrame({"Type": ["Consumer", "Detritus"], "QB": [5.0, float("nan")]})
        no_data = pd.DataFrame({"Type": [False, False], "QB": [False, False]})
        styles = create_cell_styles(df, no_data)
        qb_col = df.columns.get_loc("QB")
        det_styles = [s for s in styles if s["rows"] == 1 and s["cols"] == qb_col]
        assert len(det_styles) >= 1

    def test_no_data_takes_priority_over_non_applicable(self):
        # Cell is both no_data AND non-applicable → no_data style wins (#f0f0f0)
        df = pd.DataFrame({"Type": ["Producer"], "QB": [float("nan")]})
        no_data = pd.DataFrame({"Type": [False], "QB": [True]})
        styles = create_cell_styles(df, no_data)
        qb_col = df.columns.get_loc("QB")
        qb_styles = [s for s in styles if s["cols"] == qb_col]
        assert qb_styles[0]["style"]["background-color"] == "#f0f0f0"

    def test_remarks_style(self):
        df = pd.DataFrame({"Type": ["Consumer"], "Biomass": [10.0]})
        no_data = pd.DataFrame({"Type": [False], "Biomass": [False]})
        remarks = pd.DataFrame({"Type": [False], "Biomass": [True]})
        styles = create_cell_styles(df, no_data, remarks)
        bio_col = df.columns.get_loc("Biomass")
        bio_styles = [s for s in styles if s["cols"] == bio_col]
        assert bio_styles[0]["style"]["background-color"] == "#fff9e6"

    def test_stanza_style(self):
        df = pd.DataFrame({"Type": ["Consumer"], "Biomass": [10.0]})
        no_data = pd.DataFrame({"Type": [False], "Biomass": [False]})
        stanza = pd.DataFrame({"Type": [True], "Biomass": [True]})
        styles = create_cell_styles(df, no_data, stanza_mask=stanza)
        assert any(s["style"].get("background-color") == "#e6f3ff" for s in styles)

    def test_style_dicts_have_location_body(self):
        df = pd.DataFrame({"Biomass": [float("nan")]})
        no_data = pd.DataFrame({"Biomass": [True]})
        styles = create_cell_styles(df, no_data)
        for s in styles:
            assert s["location"] == "body"
            assert "rows" in s and "cols" in s and "style" in s
