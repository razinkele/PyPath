# Biodiversity Database Setup - Conda Shiny Environment

## Quick Setup (3 commands)

```bash
# 1. Activate your shiny environment
conda activate shiny

# 2. Install biodiversity database dependencies
pip install pyworms pyobis

# 3. Verify installation
python verify_biodata_deps.py
```

That's it! Then restart your Shiny app.

---

## Detailed Instructions

### Step 1: Activate Conda Environment

```bash
conda activate shiny
```

**Verify you're in the right environment:**
```bash
# Should show: shiny
conda env list | grep "*"

# Or check Python path
python -c "import sys; print(sys.executable)"
# Should show: C:\Users\DELL\.conda\envs\shiny\python.exe
```

### Step 2: Install Biodiversity Dependencies

The packages aren't available via conda, so use pip within conda:

```bash
pip install pyworms>=0.2.1
pip install pyobis>=0.3.0
```

**Or install both at once:**
```bash
pip install pyworms>=0.2.1 pyobis>=0.3.0
```

**Or install with pyproject.toml:**
```bash
# From PyPath root directory
pip install -e .[biodata]
```

### Step 3: Verify Installation

```bash
python verify_biodata_deps.py
```

**Expected output:**
```
======================================================================
Biodiversity Database Dependencies - Verification
======================================================================

Python version: 3.13.7 ...

1. Checking pyworms...
   [OK] pyworms installed (version: 0.2.1)

2. Checking pyobis...
   [OK] pyobis installed (version: 0.3.0)

3. Checking requests...
   [OK] requests installed (version: 2.31.0)

4. Checking pypath.io.biodata module...
   [OK] biodata module can be imported

======================================================================
Summary
======================================================================

[OK] All dependencies installed!
```

### Step 4: Test Workflow

```bash
python test_biodata_workflow.py
```

**This tests:**
- WoRMS API connectivity
- OBIS API connectivity
- FishBase API connectivity
- Species lookup (individual and batch)
- Model creation from biodiversity data

**Expected to take:** 1-2 minutes (making real API calls)

### Step 5: Start Shiny App

```bash
# Make sure you're still in shiny environment
conda activate shiny

# Start app
shiny run app/app.py
```

**Or with specific port:**
```bash
shiny run --port 57006 app/app.py
```

### Step 6: Test in Browser

1. Open browser to http://127.0.0.1:8000 (or whatever port shown)
2. Navigate to **"Data Import"** tab
3. Click **"Biodiversity"** sub-tab
4. Click **"Load Example"** button
5. Click **"Fetch Species Data"** button
6. Wait 30-60 seconds (fetching from WoRMS, OBIS, FishBase)
7. Review results in table
8. Adjust biomass values if desired
9. Click **"Create Ecopath Model"**
10. Click **"Use This Model in Ecopath"**
11. Navigate to **"Ecopath Model"** tab to see your model

---

## Troubleshooting

### Issue: "conda: command not found"

**Solution:** Use Anaconda Prompt or Conda shell

**Windows:**
- Start Menu → Anaconda Prompt
- Or: Anaconda PowerShell Prompt

### Issue: "pip: command not found" in conda environment

**Solution:** Install pip in conda environment
```bash
conda activate shiny
conda install pip
```

### Issue: Environment activation doesn't work

**PowerShell specific:**
```powershell
conda init powershell
# Close and reopen PowerShell
conda activate shiny
```

**Command Prompt:**
```cmd
conda activate shiny
```

### Issue: Package conflicts after pip install

**Solution:** Create fresh environment (if needed)
```bash
# Export current environment
conda env export > shiny_backup.yml

# Create new environment with packages
conda create -n shiny_new python=3.13 -y
conda activate shiny_new
conda install shiny pandas numpy plotly -y
pip install pyworms pyobis
pip install -e .
```

### Issue: "Could not find species" still happening

**Check dependencies are actually installed:**
```bash
conda activate shiny
python -c "import pyworms; print('OK')"
python -c "import pyobis; print('OK')"
```

**If import fails:**
```bash
# Verify you're in shiny environment
conda env list

# Reinstall
pip install --force-reinstall pyworms pyobis
```

### Issue: Different Python being used

**Check which Python:**
```bash
conda activate shiny
python -c "import sys; print(sys.executable)"
```

**Should show:**
```
C:\Users\DELL\.conda\envs\shiny\python.exe
```

**If it shows a different Python:**
```bash
# Use conda's python explicitly
C:\Users\DELL\.conda\envs\shiny\python.exe verify_biodata_deps.py
```

---

## Package Information

### pyworms
- **Purpose:** WoRMS (World Register of Marine Species) API client
- **Install:** `pip install pyworms`
- **Not available via conda** - must use pip
- **Size:** ~50 KB
- **Dependencies:** requests

### pyobis
- **Purpose:** OBIS (Ocean Biodiversity Information System) API client
- **Install:** `pip install pyobis`
- **Not available via conda** - must use pip
- **Size:** ~100 KB
- **Dependencies:** pandas, requests

### Why pip in conda?

These packages are **only available on PyPI**, not conda-forge or anaconda channels. Using pip within conda is the recommended approach for such packages.

**This is safe and recommended by conda:**
- https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#using-pip-in-an-environment

---

## Complete Installation Script

**Save as `install_biodata_deps.bat` (Windows):**

```batch
@echo off
echo ========================================
echo Installing Biodiversity Dependencies
echo ========================================
echo.

echo Activating conda environment: shiny
call conda activate shiny
if errorlevel 1 (
    echo ERROR: Failed to activate shiny environment
    pause
    exit /b 1
)

echo.
echo Installing pyworms...
pip install pyworms>=0.2.1
if errorlevel 1 (
    echo ERROR: Failed to install pyworms
    pause
    exit /b 1
)

echo.
echo Installing pyobis...
pip install pyobis>=0.3.0
if errorlevel 1 (
    echo ERROR: Failed to install pyobis
    pause
    exit /b 1
)

echo.
echo ========================================
echo Verifying installation...
echo ========================================
python verify_biodata_deps.py

echo.
echo ========================================
echo Installation complete!
echo ========================================
echo.
echo Next steps:
echo   1. Run: python test_biodata_workflow.py
echo   2. Run: shiny run app/app.py
echo.
pause
```

**Run it:**
```bash
install_biodata_deps.bat
```

**Or for PowerShell (`install_biodata_deps.ps1`):**

```powershell
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Installing Biodiversity Dependencies" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Activating conda environment: shiny" -ForegroundColor Yellow
conda activate shiny

Write-Host ""
Write-Host "Installing pyworms..." -ForegroundColor Yellow
pip install pyworms>=0.2.1

Write-Host ""
Write-Host "Installing pyobis..." -ForegroundColor Yellow
pip install pyobis>=0.3.0

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Verifying installation..." -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
python verify_biodata_deps.py

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "Installation complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Run: python test_biodata_workflow.py"
Write-Host "  2. Run: shiny run app/app.py"
```

---

## Quick Command Reference

```bash
# Activate environment
conda activate shiny

# Install dependencies
pip install pyworms pyobis

# Verify
python verify_biodata_deps.py

# Test workflow
python test_biodata_workflow.py

# Run app
shiny run app/app.py

# Check what's installed
pip list | grep -i "pyworms\|pyobis"

# Or on Windows:
pip list | findstr /i "pyworms pyobis"
```

---

## Expected Timeline

| Step | Time | Details |
|------|------|---------|
| Activate conda env | 5 seconds | `conda activate shiny` |
| Install pyworms | 10-30 seconds | Downloads from PyPI |
| Install pyobis | 10-30 seconds | Downloads from PyPI |
| Verify installation | 5 seconds | `python verify_biodata_deps.py` |
| Test workflow | 1-2 minutes | Makes real API calls |
| Start Shiny app | 5-10 seconds | `shiny run app/app.py` |
| **Total** | **2-3 minutes** | Ready to use! |

---

## After Installation

### Test Individual Components

**Test WoRMS:**
```python
python -c "from pypath.io.biodata import _fetch_worms_vernacular; print(_fetch_worms_vernacular('cod', cache=False))"
```

**Test OBIS:**
```python
python -c "from pypath.io.biodata import _fetch_obis_occurrences; print(_fetch_obis_occurrences('Gadus morhua', cache=False))"
```

**Test FishBase:**
```python
python -c "from pypath.io.biodata import _fetch_fishbase_traits; print(_fetch_fishbase_traits('Gadus morhua', cache=False))"
```

### First Real Test in App

1. Load example: **cod, herring, sprat**
2. Fetch data (may take 60 seconds)
3. Verify results show:
   - Scientific names
   - Trophic levels
   - OBIS occurrence counts
4. Create model
5. Use in Ecopath

---

## Summary

### What to run:

```bash
conda activate shiny
pip install pyworms pyobis
python verify_biodata_deps.py
python test_biodata_workflow.py
shiny run app/app.py
```

### What you'll get:

✅ Access to WoRMS (1.4+ million marine species)
✅ Access to OBIS (130+ million occurrence records)
✅ Access to FishBase (35,000+ fish species)
✅ Automatic parameter estimation for Ecopath models
✅ Build models from scratch using biodiversity data

**You're 3 commands away from a working integration!** 🎉
