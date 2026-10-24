# PHASE 2 QUICK START GUIDE

## Data Location
All extracted datasets are located in:
\\\
C:\Users\DELL\OneDrive - ku.lt\HORIZON_EUROPE\PyPath\curonian_data_phase1\raw_csv\
\\\

## Loading Data in Python

### CPUE Data (2001-2018)
\\\python
import pandas as pd
import os

data_dir = r'C:\Users\DELL\OneDrive - ku.lt\HORIZON_EUROPE\PyPath\curonian_data_phase1\raw_csv'

# Main CPUE time series
cpue_data = pd.read_csv(os.path.join(data_dir, 'cpue_bar_data.csv'))

# Species-specific CPUE
cpue_bream = pd.read_csv(os.path.join(data_dir, 'cpue_bar_bream.csv'))
cpue_zander = pd.read_csv(os.path.join(data_dir, 'cpue_bar_zander.csv'))
cpue_roach = pd.read_csv(os.path.join(data_dir, 'cpue_bar_roach.csv'))
cpue_vimba = pd.read_csv(os.path.join(data_dir, 'cpue_bar_vimba.csv'))
cpue_perch = pd.read_csv(os.path.join(data_dir, 'cpue_bar_perch.csv'))
\\\

### Landings Data (1947-2011)
\\\python
# Main landings records
landings = pd.read_csv(os.path.join(data_dir, 'landings_1947_2011_sugavimai.csv'))

# Economic value data
landings_value = pd.read_csv(os.path.join(data_dir, 'landings_1947_2011_value.csv'))

# Lithuanian sector
landings_lt = pd.read_csv(os.path.join(data_dir, 'landings_1947_2011_lt_graph.csv'))

# Russian sector
landings_rus = pd.read_csv(os.path.join(data_dir, 'landings_1947_2011_rus_graph.csv'))
\\\

### Environmental Forcing (1888-2011)
\\\python
# Nemunas River discharge
discharge = pd.read_csv(os.path.join(data_dir, 'discharge_1888_2011_dischrges.csv'))
\\\

### Annual Monitoring Database (2001-2015)
\\\python
# Load all annual database files
annual_files = [f for f in os.listdir(data_dir) if f.startswith('annual_db_')]
annual_data = {f.split('_')[2]: pd.read_csv(os.path.join(data_dir, f)) for f in annual_files}

# Access specific year
year_2010_data = annual_data['2010']
\\\

## Key File References

### Metadata
- **PHASE_1_METADATA.json** - Complete dataset inventory
- **CPUE_METADATA.md** - CPUE documentation
- **LANDINGS_METADATA.md** - Landings documentation
- **DISCHARGE_METADATA.md** - Discharge documentation
- **ANNUAL_DB_METADATA.md** - Annual database documentation

### Quality
- **DATA_QUALITY_ASSESSMENT.json** - Quality metrics (93.3% avg)
- **EXTRACTION_LOG.json** - Detailed extraction record

## Primary Calibration Window: 2001-2011

This 11-year period has all four dataset types synchronously available:
- CPUE: Complete (2001-2011)
- Landings: Available (1947-2011)
- Discharge: Complete (1888-2011)
- Annual DB: Complete (2001-2015)

## Extended Validation Windows

- **Long-term trends (1947-2011):** Use landings + discharge
- **Recent dynamics (2001-2015):** Use CPUE + annual DB + discharge
- **Environmental baseline (1888-2011):** Use discharge only

## Species Mapping

### CPUE Species (2001-2018)
- Bream (Cyprinidae)
- Zander (Percidae - pike-perch)
- Roach (Cyprinidae)
- Vimba (Cyprinidae)
- Perch (Percidae)
- Plus 9 additional analysis sheets

### Landings Records (1947-2011)
See landings_1947_2011_sugavimai.csv for species list

### Annual Monitoring (2001-2015)
Check ANNUAL_DB_METADATA.md for species and metrics per year

## Expected Next Steps (Phase 2)

1. **Data Integration**
   - Map species codes across datasets
   - Align temporal indices
   - Handle missing values

2. **Ecopath Configuration**
   - Use 2001-2015 annual DB for biomass estimates
   - Use landings data for fleet configuration
   - Use historical landings for economic baseline

3. **Ecosim Fitting**
   - Load CPUE data via fit_to_timeseries()
   - Configure discharge as forcing time series
   - Validate against observed trajectories

4. **Validation**
   - Use 1947-2011 landings for regime shift analysis
   - Verify mass balance with annual monitoring
   - Test environmental sensitivity with discharge variations

## Notes

- All files are UTF-8 encoded
- Headers are included in all CSV files
- Original data preserved in EDGARAS (not modified)
- All temporal coverage verified and documented
- Quality average: 93.3% (all datasets ready)

---

Ready to proceed with Phase 2: Calibration & Validation
