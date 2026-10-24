"""Enhanced data loader that extracts actual CPUE values from CSV files."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from pypath.core.timeseries import (
    DATTYPE_ABS_BIOMASS,
    DATTYPE_CATCH,
    DATTYPE_EFFORT,
    DATTYPE_FORCED_BIOMASS,
    EweTimeSeries,
    EweTimeSeriesCollection,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def extract_cpue_timeseries(cpue_df: pd.DataFrame, species_name: str) -> Optional[np.ndarray]:
    """Extract annual CPUE values from CPUE CSV.

    Parameters
    ----------
    cpue_df : pd.DataFrame
        Loaded CPUE data frame
    species_name : str
        Species code (e.g., 'bream')

    Returns
    -------
    np.ndarray or None
        Annual CPUE values, or None if extraction fails
    """
    # CPUE data is in format: Year, Month, ..., CPUE column (and species-specific columns)
    if "Metai" not in cpue_df.columns:
        return None

    # Group by year and calculate mean CPUE
    if "CPUE" in cpue_df.columns:
        annual_cpue = cpue_df.groupby("Metai")["CPUE"].mean().values
        return annual_cpue

    return None


def extract_landings_timeseries(landings_df: pd.DataFrame, species_code: str) -> Optional[np.ndarray]:
    """Extract annual landings for a species from landings CSV.

    Parameters
    ----------
    landings_df : pd.DataFrame
        Loaded landings data
    species_code : str
        Species code (e.g., 'Karšis')

    Returns
    -------
    np.ndarray or None
        Annual landings, or None if species not found
    """
    # Landings data has years as rows, species as columns
    if species_code in landings_df.columns:
        values = landings_df[species_code].dropna().values
        return values.astype(float)

    return None


class CurionianDataLoaderV2:
    """Enhanced loader with actual CPUE/landings extraction."""

    def __init__(self, data_dir: Path | str):
        """Initialize with path to raw_csv directory."""
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        self.collection = EweTimeSeriesCollection()
        self._series_counter = 0
        self.log = []

    def _next_series_id(self) -> int:
        """Get next unique series ID."""
        self._series_counter += 1
        return self._series_counter

    def _log(self, message: str) -> None:
        """Log message and store in log list."""
        logger.info(message)
        self.log.append(message)

    def load_cpue(self, min_year: int = 2001, max_year: int = 2018) -> dict[str, pd.DataFrame]:
        """Load CPUE data."""
        self._log("=" * 60)
        self._log("LOADING CPUE DATA (2001-2018)")
        self._log("=" * 60)

        cpue_data = {}
        species_files = [
            ("bream", "Bream"),
            ("perch", "Perch"),
            ("roach", "Roach"),
            ("vimba", "Vimba"),
            ("zander", "Zander"),
        ]

        for code, species_name in species_files:
            cpue_file = self.data_dir / f"cpue_bar_{code}.csv"
            if not cpue_file.exists():
                self._log(f"WARNING: CPUE file not found: {cpue_file.name}")
                continue

            try:
                df = pd.read_csv(cpue_file)
                self._log(f"Loaded {species_name} ({code}): {cpue_file.name}")
                self._log(f"  Shape: {df.shape[0]} rows, {df.shape[1]} columns")
                if "Metai" in df.columns:
                    years = df["Metai"].dropna().unique()
                    self._log(f"  Years: {int(min(years))} - {int(max(years))}")
                cpue_data[code] = df
            except Exception as e:
                self._log(f"ERROR loading {species_name}: {str(e)}")

        self._log(f"Total CPUE species loaded: {len(cpue_data)}")
        return cpue_data

    def load_landings(self, min_year: int = 1947, max_year: int = 2011) -> dict[str, pd.DataFrame]:
        """Load landings data."""
        self._log("\n" + "=" * 60)
        self._log("LOADING LANDINGS DATA (1947-2011)")
        self._log("=" * 60)

        landings_data = {}
        landings_file = self.data_dir / "landings_1947_2011_sugavimai.csv"

        if not landings_file.exists():
            self._log(f"WARNING: Landings file not found: {landings_file}")
            return landings_data

        try:
            df = pd.read_csv(landings_file, skiprows=3)
            self._log(f"Loaded main landings file: {landings_file.name}")
            self._log(f"  Shape: {df.shape[0]} rows, {df.shape[1]} columns")
            landings_data["main"] = df
        except Exception as e:
            self._log(f"ERROR loading landings: {str(e)}")

        return landings_data

    def load_discharge(self, min_year: int = 1888, max_year: int = 2011) -> Optional[np.ndarray]:
        """Load discharge data."""
        self._log("\n" + "=" * 60)
        self._log("LOADING DISCHARGE DATA (1888-2011)")
        self._log("=" * 60)

        discharge_file = self.data_dir / "discharge_1888_2011_dischrges.csv"

        if not discharge_file.exists():
            self._log(f"WARNING: Discharge file not found: {discharge_file}")
            return None

        try:
            df = pd.read_csv(discharge_file)
            self._log(f"Loaded discharge: {discharge_file.name}")
            self._log(f"  Shape: {df.shape[0]} rows")

            if "Year" in df.columns and "Discharge" in df.columns:
                year_min, year_max = df["Year"].min(), df["Year"].max()
                self._log(f"  Year range: {int(year_min)} - {int(year_max)}")
                
                df_filtered = df[(df["Year"] >= min_year) & (df["Year"] <= max_year)]
                discharge_values = df_filtered["Discharge"].values.astype(float)
                self._log(f"  Filtered ({min_year}-{max_year}): {len(discharge_values)} months")
                return discharge_values
        except Exception as e:
            self._log(f"ERROR loading discharge: {str(e)}")

        return None

    def load_monitoring(self, min_year: int = 2001, max_year: int = 2018) -> list[dict]:
        """Load monitoring data."""
        self._log("\n" + "=" * 60)
        self._log("LOADING ANNUAL MONITORING DATA (2001-2018)")
        self._log("=" * 60)

        monitoring_data = []
        for year in range(min_year, max_year + 1):
            pattern_files = [
                self.data_dir / f"annual_db_{year}_sheet1.csv",
                self.data_dir / f"annual_db_{year}_data.csv",
                self.data_dir / f"annual_db_{year}_all_data.csv",
            ]

            for pfile in pattern_files:
                if pfile.exists():
                    try:
                        df = pd.read_csv(pfile)
                        self._log(f"Loaded {year}: {pfile.name} ({df.shape[0]} rows)")
                        monitoring_data.append({"year": year, "file": pfile.name, "data": df})
                        break
                    except Exception as e:
                        self._log(f"  ERROR reading {pfile.name}: {str(e)}")

        self._log(f"Total monitoring files loaded: {len(monitoring_data)}")
        return monitoring_data

    def create_timeseries_collection(
        self,
        cpue_data: dict[str, pd.DataFrame],
        landings_data: dict[str, pd.DataFrame],
        discharge_data: Optional[np.ndarray],
        monitoring_data: list[dict],
    ) -> EweTimeSeriesCollection:
        """Create collection with actual extracted data."""
        self._log("\n" + "=" * 60)
        self._log("CREATING TIMESERIES COLLECTION")
        self._log("=" * 60)

        series_list = []

        # Add discharge as forced biomass driver
        if discharge_data is not None:
            discharge_series = EweTimeSeries(
                series_id=self._next_series_id(),
                name="Nemunas Discharge (monthly)",
                dat_type=DATTYPE_FORCED_BIOMASS,
                group_idx=None,
                fleet_idx=None,
                values=discharge_data.astype(float),
                weight=1.0,
                dataset_id=0,
            )
            series_list.append(discharge_series)
            self._log(f"Added discharge forcing: {len(discharge_data)} monthly values")

        # Add CPUE as observed catch
        species_codes_cpue = list(cpue_data.keys())
        for i, code in enumerate(species_codes_cpue):
            # Extract annual CPUE values
            cpue_values_extracted = extract_cpue_timeseries(cpue_data[code], code)
            
            if cpue_values_extracted is not None:
                # Pad/interpolate to 216 months (18 years)
                cpue_monthly = np.full(216, np.nan)
                n_years = len(cpue_values_extracted)
                for year_idx in range(n_years):
                    year_start = year_idx * 12
                    year_end = (year_idx + 1) * 12
                    if year_end <= 216:
                        cpue_monthly[year_start:year_end] = cpue_values_extracted[year_idx]
            else:
                cpue_monthly = np.full(216, np.nan)

            catch_series = EweTimeSeries(
                series_id=self._next_series_id(),
                name=f"CPUE {code.capitalize()}",
                dat_type=DATTYPE_CATCH,
                group_idx=i,
                fleet_idx=None,
                values=cpue_monthly,
                weight=1.0,
                dataset_id=1,
            )
            series_list.append(catch_series)

        self._log(f"Added {len(series_list)} timeseries to collection")

        # Create collection
        collection = EweTimeSeriesCollection(series=series_list)
        self._log(f"Collection: {len(collection.series)} series, {collection.n_timesteps} timesteps")

        return collection

    def validate(self, collection: EweTimeSeriesCollection) -> dict[str, object]:
        """Validate collection."""
        self._log("\n" + "=" * 60)
        self._log("DATA VALIDATION")
        self._log("=" * 60)

        results = {
            "total_series": len(collection.series),
            "timesteps": collection.n_timesteps,
            "observed_catch_count": len(collection.observed_catch),
            "forced_biomass_count": len(collection.forced_biomass),
            "data_quality": {},
        }

        for s in collection.series:
            non_nan = np.sum(~np.isnan(s.values))
            if non_nan == 0:
                self._log(f"WARNING: Series '{s.name}' is all NaN")
            else:
                pct_complete = 100.0 * non_nan / len(s.values)
                results["data_quality"][s.name] = pct_complete
                self._log(f"  {s.name}: {pct_complete:.1f}% complete")

        self._log(f"\nValidation Summary:")
        self._log(f"  Total series: {results['total_series']}")
        self._log(f"  Timesteps: {results['timesteps']}")
        self._log(f"  Observed catch: {results['observed_catch_count']}")
        self._log(f"  Forced biomass: {results['forced_biomass_count']}")

        return results

    def load_all(self) -> tuple[EweTimeSeriesCollection, dict[str, object]]:
        """Load all Phase 1 data."""
        self._log("\n" + "=" * 58)
        self._log("CURONIAN LAGOON PHASE 1 DATA LOADER (V2)")
        self._log("Loading standardized CSV datasets")
        self._log("=" * 58)

        cpue_data = self.load_cpue()
        landings_data = self.load_landings()
        discharge_data = self.load_discharge()
        monitoring_data = self.load_monitoring()

        collection = self.create_timeseries_collection(
            cpue_data, landings_data, discharge_data, monitoring_data
        )

        validation_results = self.validate(collection)

        self._log("\n" + "=" * 60)
        self._log("LOADING COMPLETE")
        self._log("=" * 60)

        return collection, validation_results

    def get_log_text(self) -> str:
        """Return complete log as text."""
        return "\n".join(self.log)
