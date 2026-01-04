# Biodiversity Database Setup Guide

## Issue Identified

The biodiversity database integration requires additional Python packages that are not currently installed:

- ❌ `pyworms` - NOT installed
- ❌ `pyobis` - NOT installed
- ✅ `requests` - Already installed

This is why all species lookups are failing with "Could not find species" errors.

## Solution

Install the biodiversity database dependencies.

### Option 1: Install from pyproject.toml (Recommended)

```bash
# From the PyPath root directory
pip install -e .[biodata]
```

This installs:
- `pyworms>=0.2.1`
- `pyobis>=0.3.0`
- `requests>=2.28`

### Option 2: Install Manually

```bash
pip install pyworms>=0.2.1
pip install pyobis>=0.3.0
pip install requests>=2.28
```

### Verification

After installation, verify with:

```bash
python -c "import pyworms; print('pyworms:', pyworms.__version__)"
python -c "import pyobis; print('pyobis:', pyobis.__version__)"
python -c "import requests; print('requests:', requests.__version__)"
```

Expected output:
```
pyworms: 0.2.1
pyobis: 0.3.0
requests: 2.31.0
```

## Testing the Workflow

### Quick Test

```bash
python test_biodata_workflow.py
```

This comprehensive test script will:
1. Test individual WoRMS lookups
2. Test single species workflow
3. Test batch workflow (Shiny app scenario)
4. Test model creation
5. Test API connectivity

### Expected Results

After installing dependencies, you should see:

```
1. Testing individual WoRMS vernacular search...
----------------------------------------------------------------------

Searching for: 'Atlantic cod'
  [OK] Found 1 result(s)
    [1] Gadus morhua (AphiaID: 126436)

Searching for: 'cod'
  [OK] Found 20+ result(s)
    [1] Gadus morhua (AphiaID: 126436)
    [2] Gadus macrocephalus (AphiaID: 126437)
    ...

2. Testing single species workflow...
----------------------------------------------------------------------

Fetching info for 'cod'...
[OK] Success!
  Common name: cod
  Scientific name: Gadus morhua
  AphiaID: 126436
  Trophic level: 4.4
  Max length: 180.0
  OBIS occurrences: 15000+
```

## Testing in Shiny App

Once dependencies are installed:

1. **Start the app:**
   ```bash
   shiny run app/app.py
   ```

2. **Navigate to Data Import → Biodiversity tab**

3. **Click "Load Example"**

4. **Click "Fetch Species Data"**

5. **Wait 30-60 seconds** (normal for API calls)

6. **Verify results table shows:**
   - Common names
   - Scientific names
   - Trophic levels
   - OBIS occurrence counts

7. **Adjust biomass values**

8. **Click "Create Ecopath Model"**

9. **Click "Use This Model in Ecopath"**

10. **Navigate to Ecopath Model tab** to see the generated model

## Troubleshooting

### Error: "pyworms is required"

**Cause:** `pyworms` package not installed

**Solution:**
```bash
pip install pyworms
```

### Error: "pyobis module not found"

**Cause:** `pyobis` package not installed

**Solution:**
```bash
pip install pyobis
```

### Error: "Could not find species: [name]"

**Possible causes:**

1. **Dependencies not installed** - Install pyworms/pyobis (see above)

2. **API timeout** - Increase timeout in code or try again

3. **Incorrect species name** - Try:
   - Just genus/common name: "cod" instead of "Atlantic cod"
   - Scientific name: "Gadus morhua"
   - Check spelling

4. **API down** - Test connectivity:
   ```bash
   curl https://www.marinespecies.org/rest/AphiaRecordsByVernacular/cod
   ```

### Error: "API connection timeout"

**Causes:**
- Network issues
- API server slow/down
- Firewall blocking

**Solutions:**
- Check internet connection
- Try again later
- Use VPN if corporate firewall

## Package Information

### pyworms

- **Purpose:** Access WoRMS (World Register of Marine Species) database
- **Version:** 0.2.1+
- **PyPI:** https://pypi.org/project/pyworms/
- **Docs:** https://github.com/iobis/pyworms
- **What it does:**
  - Searches species by common/vernacular names
  - Retrieves AphiaID (unique species identifier)
  - Gets accepted scientific names
  - Resolves synonyms

### pyobis

- **Purpose:** Access OBIS (Ocean Biodiversity Information System)
- **Version:** 0.3.0+
- **PyPI:** https://pypi.org/project/pyobis/
- **Docs:** https://github.com/iobis/pyobis
- **What it does:**
  - Searches occurrence records
  - Gets geographic distribution
  - Retrieves depth ranges
  - Provides temporal data

### requests

- **Purpose:** HTTP library for FishBase API calls
- **Version:** 2.28+
- **PyPI:** https://pypi.org/project/requests/
- **Already installed** in most Python environments

## API Rate Limits

### WoRMS
- **Limit:** None officially stated
- **Recommended:** < 1000 requests/hour
- **Our usage:** ~1-5 requests per species

### OBIS
- **Limit:** None officially stated
- **Recommended:** < 100 requests/minute
- **Our usage:** ~1 request per species

### FishBase
- **Limit:** None officially stated
- **Recommended:** < 50 requests/minute
- **Our usage:** ~3-4 requests per species

**Total:** With 5 species, expect ~20-25 API calls total

## File Checklist

Before testing, ensure these files exist:

- ✅ `src/pypath/io/biodata.py` - Main module
- ✅ `src/pypath/io/utils.py` - Shared utilities
- ✅ `app/pages/data_import.py` - Shiny integration
- ✅ `test_biodata_workflow.py` - Test script
- ✅ `tests/test_biodata.py` - Unit tests
- ✅ `tests/test_biodata_integration.py` - Integration tests

## Quick Reference Commands

```bash
# Install dependencies
pip install -e .[biodata]

# Verify installation
python -c "import pyworms, pyobis, requests; print('OK')"

# Run workflow test
python test_biodata_workflow.py

# Run unit tests
pytest tests/test_biodata.py -v -m "not integration"

# Run integration tests (requires internet)
pytest tests/test_biodata_integration.py -v -m integration

# Start Shiny app
shiny run app/app.py

# Validate database connections
python scripts/test_database_connections.py --quick
```

## Next Steps

1. **Install dependencies:**
   ```bash
   pip install -e .[biodata]
   ```

2. **Run test script:**
   ```bash
   python test_biodata_workflow.py
   ```

3. **If test passes, restart Shiny app:**
   ```bash
   shiny run app/app.py
   ```

4. **Test in browser:**
   - Go to Data Import → Biodiversity tab
   - Load example species
   - Fetch data
   - Create model

## Expected Timeline

- **Install dependencies:** 1-2 minutes
- **Run tests:** 2-3 minutes
- **Fetch 5 species in app:** 30-60 seconds
- **Create model:** 1-2 seconds
- **Total:** ~5 minutes to working integration

---

**Status:** Dependencies missing - install required
**Action:** Run `pip install -e .[biodata]`
**Then:** Run `python test_biodata_workflow.py` to verify
