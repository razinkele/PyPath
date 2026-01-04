@echo off
REM Quick restart script for PyPath Shiny app (Windows)

echo ========================================
echo PyPath App Restart Script
echo ========================================
echo.

REM Stop any running Python processes running shiny
echo Stopping Shiny processes...
taskkill /F /IM python.exe /FI "WINDOWTITLE eq shiny*" >nul 2>&1
timeout /t 2 >nul

REM Clear Python cache
echo Clearing Python cache...
for /d /r app %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"
del /s /q app\*.pyc >nul 2>&1
echo Cache cleared!

echo.
echo Verifying bug fixes...
findstr /C:"@render.download" app\pages\multistanza.py >nul && (
    echo   [OK] multistanza.py updated
) || (
    echo   [!!] multistanza.py NOT updated
)

findstr /C:"@render.download" app\pages\forcing_demo.py >nul && (
    echo   [OK] forcing_demo.py updated
) || (
    echo   [!!] forcing_demo.py NOT updated
)

findstr /C:"@render.download" app\pages\diet_rewiring_demo.py >nul && (
    echo   [OK] diet_rewiring_demo.py updated
) || (
    echo   [!!] diet_rewiring_demo.py NOT updated
)

findstr /C:"@render.download" app\pages\optimization_demo.py >nul && (
    echo   [OK] optimization_demo.py updated
) || (
    echo   [!!] optimization_demo.py NOT updated
)

echo.
echo ========================================
echo Starting app...
echo ========================================
echo.
echo Press Ctrl+C to stop the app
echo.

REM Start app with no bytecode caching
python -B -m shiny run app\app.py
