"""Tests for write_ewe_samples() function."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from pypath.core.params import create_rpath_params
from pypath.core.sampling import (
    EcopathDietCompSample,
    EcopathGroupSample,
    SampleContainer,
)
from pypath.io.ewe_writer import write_ewe_samples, write_ewemdb
from pypath.io.ewemdb import read_ewe_samples, read_ewemdb_table


class TestWriteSamplesBasics:
    """Test basic write_ewe_samples() functionality."""

    def test_write_samples_creates_output(self):
        """Verify write_ewe_samples creates EcopathSample record."""
        container = SampleContainer(n_replicates=10)
        container.add_group_sample(
            EcopathGroupSample(
                group_name="Fish",
                biomass_samples=np.array([100.0] * 10),
                pb_samples=np.array([0.5] * 10),
                qb_samples=np.array([1.2] * 10),
                ee_samples=np.array([0.8] * 10),
            )
        )

        params = create_rpath_params(
            groups=["Fish", "Phyto", "Detritus"], types=[0, 1, 2]
        )
        params.model.loc[params.model["Group"] == "Fish", "Biomass"] = 100.0

        with tempfile.NamedTemporaryFile(suffix=".eweaccdb", delete=False) as f:
            temp_db = f.name

        try:
            try:
                write_ewemdb(params, temp_db, backend="access")
            except RuntimeError:
                pytest.skip("Access ODBC driver not available")

            sample_id = write_ewe_samples(container, temp_db)
            assert isinstance(sample_id, int)
            assert sample_id > 0

            samples_df = read_ewemdb_table(temp_db, "EcopathSample")
            assert len(samples_df) > 0
            assert sample_id in samples_df["SampleID"].values

        finally:
            Path(temp_db).unlink(missing_ok=True)

    def test_write_samples_with_diet_samples(self):
        """Test write_ewe_samples with diet samples."""
        container = SampleContainer(n_replicates=5)

        container.add_group_sample(
            EcopathGroupSample(
                group_name="Fish",
                biomass_samples=np.array([100.0] * 5),
                pb_samples=np.array([0.5] * 5),
                qb_samples=np.array([1.2] * 5),
                ee_samples=np.array([0.8] * 5),
            )
        )

        container.add_diet_sample(
            EcopathDietCompSample(
                predator_name="Fish",
                prey_name="Phyto",
                diet_fraction_samples=np.array([0.5, 0.52, 0.48, 0.51, 0.49]),
            )
        )

        params = create_rpath_params(
            groups=["Fish", "Phyto", "Detritus"], types=[0, 1, 2]
        )
        params.model.loc[params.model["Group"] == "Fish", "Biomass"] = 100.0

        with tempfile.NamedTemporaryFile(suffix=".eweaccdb", delete=False) as f:
            temp_db = f.name

        try:
            try:
                write_ewemdb(params, temp_db, backend="access")
            except RuntimeError:
                pytest.skip("Access ODBC driver not available")

            sample_id = write_ewe_samples(container, temp_db)

            group_df = read_ewemdb_table(temp_db, "EcopathGroupSample")
            diet_df = read_ewemdb_table(temp_db, "EcopathDietCompSample")

            # 5 replicates means 5 EcopathSample rows (SampleID sample_id to sample_id+4)
            # Each sample has 1 group and 1 diet link
            sample_ids = list(range(sample_id, sample_id + 5))
            assert all(sid in group_df["SampleID"].values for sid in sample_ids)
            assert all(sid in diet_df["SampleID"].values for sid in sample_ids)
            assert len(group_df[group_df["SampleID"].isin(sample_ids)]) == 5
            assert len(diet_df[diet_df["SampleID"].isin(sample_ids)]) == 5

        finally:
            Path(temp_db).unlink(missing_ok=True)

    def test_write_samples_error_on_empty_container(self):
        """Test that empty container raises ValueError."""
        container = SampleContainer(n_replicates=10)

        params = create_rpath_params(
            groups=["Fish", "Phyto", "Detritus"], types=[0, 1, 2]
        )

        with tempfile.NamedTemporaryFile(suffix=".eweaccdb", delete=False) as f:
            temp_db = f.name

        try:
            try:
                write_ewemdb(params, temp_db, backend="access")
            except RuntimeError:
                pytest.skip("Access ODBC driver not available")

            with pytest.raises(ValueError, match="must have at least"):
                write_ewe_samples(container, temp_db)

        finally:
            Path(temp_db).unlink(missing_ok=True)

    def test_write_samples_nonexistent_database(self):
        """Test that nonexistent database raises FileNotFoundError."""
        container = SampleContainer(n_replicates=5)
        container.add_group_sample(
            EcopathGroupSample(
                group_name="Fish",
                biomass_samples=np.array([100.0] * 5),
                pb_samples=np.array([0.5] * 5),
                qb_samples=np.array([1.2] * 5),
                ee_samples=np.array([0.8] * 5),
            )
        )

        with pytest.raises(FileNotFoundError):
            write_ewe_samples(container, "/nonexistent/path/db.eweaccdb")


class TestWriteSamplesRoundTrip:
    """Test write → read → compare round-trip consistency."""

    def test_roundtrip_group_samples(self):
        """Test group samples survive write → read cycle."""
        container = SampleContainer(n_replicates=4)
        container.add_group_sample(
            EcopathGroupSample(
                group_name="Fish",
                biomass_samples=np.array([100.0, 110.0, 90.0, 105.0]),
                pb_samples=np.array([0.5, 0.52, 0.48, 0.51]),
                qb_samples=np.array([1.2, 1.22, 1.18, 1.21]),
                ee_samples=np.array([0.8, 0.82, 0.78, 0.81]),
            )
        )

        params = create_rpath_params(
            groups=["Fish", "Phyto", "Detritus"], types=[0, 1, 2]
        )
        params.model.loc[params.model["Group"] == "Fish", "Biomass"] = 100.0

        with tempfile.NamedTemporaryFile(suffix=".eweaccdb", delete=False) as f:
            temp_db = f.name

        try:
            try:
                write_ewemdb(params, temp_db, backend="access")
            except RuntimeError:
                pytest.skip("Access ODBC driver not available")

            starting_sample_id = write_ewe_samples(container, temp_db)

            # Read all samples written (read_ewe_samples reads one SampleID at a time)
            # We wrote 4 replicates starting at starting_sample_id
            sample_ids = list(range(starting_sample_id, starting_sample_id + 4))
            biomass_values = []

            for sid in sample_ids:
                read_container = read_ewe_samples(temp_db, sample_id=sid)
                assert "Fish" in read_container.group_samples
                read_group = read_container.group_samples["Fish"]
                # Each SampleID has 1 row per group (n_replicates=1)
                assert read_group.n_replicates == 1
                biomass_values.append(read_group.biomass_samples[0])

            # Verify all biomass values match the original samples
            expected = container.group_samples["Fish"].biomass_samples
            np.testing.assert_array_almost_equal(np.array(biomass_values), expected)

        finally:
            Path(temp_db).unlink(missing_ok=True)
