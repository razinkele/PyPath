@echo off
setlocal
SET SCRIPT_DIR=%~dp0
SET R_SCRIPT=%SCRIPT_DIR%extract_rpath_data.R
SET VERIFY_SCRIPT=%SCRIPT_DIR%verify_rpath_reference.py
SET COMMIT=false

:parse_args
if "%~1"=="" goto after_parse
if "%~1"=="--commit" (
  set COMMIT=true
  shift
  goto parse_args
)
if "%~1"=="-h" (
  echo Usage: %~nx0 [--commit]
  exit /b 0
)

echo Running R extraction script: %R_SCRIPT%
where Rscript >nul 2>&1
if errorlevel 1 (
  echo Error: Rscript not found on PATH. Install R and ensure Rscript is available.
  exit /b 1
)

Rscript "%R_SCRIPT%"

where python >nul 2>&1
if errorlevel 1 (
  where python3 >nul 2>&1 || (
    echo Warning: Python not found on PATH; skipping verification.
    exit /b 0
  )
  set PY=python3
) else (
  set PY=python
)

echo Verifying generated reference files with %VERIFY_SCRIPT%
%PY% "%VERIFY_SCRIPT%"

if /I "%COMMIT%"=="true" (
  where git >nul 2>&1 || (
    echo git not found; cannot commit files. Exiting.
    exit /b 1
  )
  echo Staging generated reference files...
  git add tests\data\rpath_reference || echo No files to add
  git commit -m "Regenerate Rpath reference diagnostics (QQ/components)" || echo No changes to commit or commit failed.
)

echo Done.
endlocal