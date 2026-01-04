@echo off
REM Installation script for biodiversity database dependencies
REM For PyPath Shiny app

echo.
echo ========================================================================
echo PyPath - Biodiversity Database Dependencies Installation
echo ========================================================================
echo.
echo This script will install the required packages for biodiversity
echo database integration (WoRMS, OBIS, FishBase).
echo.
echo Packages to be installed:
echo   - pyworms (WoRMS API client)
echo   - pyobis (OBIS API client)
echo.
pause

echo.
echo [1/4] Activating conda environment: shiny
echo ========================================================================
call conda activate shiny
if errorlevel 1 (
    echo.
    echo [ERROR] Failed to activate conda environment 'shiny'
    echo.
    echo Please ensure you have:
    echo   1. Anaconda/Miniconda installed
    echo   2. A conda environment named 'shiny' created
    echo.
    echo Create environment with:
    echo   conda create -n shiny python=3.13 -y
    echo.
    pause
    exit /b 1
)
echo [OK] Environment activated
echo.

echo.
echo [2/4] Installing pyworms (WoRMS API client)
echo ========================================================================
pip install pyworms>=0.2.1
if errorlevel 1 (
    echo.
    echo [ERROR] Failed to install pyworms
    echo Check your internet connection and try again.
    pause
    exit /b 1
)
echo [OK] pyworms installed
echo.

echo.
echo [3/4] Installing pyobis (OBIS API client)
echo ========================================================================
pip install pyobis>=0.3.0
if errorlevel 1 (
    echo.
    echo [ERROR] Failed to install pyobis
    echo Check your internet connection and try again.
    pause
    exit /b 1
)
echo [OK] pyobis installed
echo.

echo.
echo [4/4] Verifying installation
echo ========================================================================
python verify_biodata_deps.py
if errorlevel 1 (
    echo.
    echo [WARNING] Verification detected issues
    echo Please review the output above.
    echo.
) else (
    echo.
    echo [SUCCESS] All dependencies verified!
    echo.
)

echo.
echo ========================================================================
echo Installation Complete!
echo ========================================================================
echo.
echo Next steps:
echo.
echo   1. Test the workflow:
echo      python test_biodata_workflow.py
echo.
echo   2. Start the Shiny app:
echo      shiny run app/app.py
echo.
echo   3. In the app:
echo      - Go to Data Import tab
echo      - Click Biodiversity sub-tab
echo      - Click "Load Example"
echo      - Click "Fetch Species Data"
echo      - Wait 30-60 seconds
echo      - Click "Create Ecopath Model"
echo.
echo Documentation:
echo   - Setup guide: CONDA_BIODATA_SETUP.md
echo   - Full guide: BIODATA_SETUP_GUIDE.md
echo   - Integration: BIODATA_SHINY_INTEGRATION_COMPLETE.md
echo.
pause
