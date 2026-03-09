#Requires -Version 5.1
<#
.SYNOPSIS
    Prepares PyPath application for deployment to laguna.ku.lt

.DESCRIPTION
    This script creates a deployment package (tarball) containing all files
    needed to deploy the PyPath Shiny application to a Linux server.

    The repository is a monorepo with two packages:
      packages/pypath/       -> pypath-ewe  (core algorithms)
      packages/pypath-shiny/ -> pypath-shiny (Shiny web frontend)

.PARAMETER OutputPath
    Path where the deployment package will be created.
    Default: .\pypath_deploy.tar.gz

.PARAMETER SkipTests
    Skip running tests before packaging.

.EXAMPLE
    .\prepare_package.ps1

.EXAMPLE
    .\prepare_package.ps1 -OutputPath "C:\Deploy\pypath.tar.gz" -SkipTests
#>

param(
    [string]$OutputPath = ".\pypath_deploy.tar.gz",
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"

# Colors for output
function Write-Info { Write-Host "[INFO] $args" -ForegroundColor Green }
function Write-Warn { Write-Host "[WARN] $args" -ForegroundColor Yellow }
function Write-Err { Write-Host "[ERROR] $args" -ForegroundColor Red }

# Get project root
$ProjectRoot = Split-Path -Parent $PSScriptRoot
if (-not $ProjectRoot) {
    $ProjectRoot = Split-Path -Parent (Get-Location)
}

Write-Host "=============================================="
Write-Host "  PyPath Deployment Package Builder"
Write-Host "=============================================="
Write-Host ""

# Verify we're in the right directory (monorepo layout)
if (-not (Test-Path (Join-Path $ProjectRoot "packages\pypath-shiny\src\pypath_shiny\app.py"))) {
    Write-Err "Cannot find packages\pypath-shiny\src\pypath_shiny\app.py. Are you in the PyPath project directory?"
    exit 1
}

Write-Info "Project root: $ProjectRoot"

# Run tests unless skipped
if (-not $SkipTests) {
    Write-Info "Running core tests..."
    Push-Location $ProjectRoot
    try {
        $testResult = python -m pytest packages/pypath/tests/ -q --tb=no --ignore=packages/pypath/tests/scripts 2>&1
        if ($LASTEXITCODE -ne 0) {
            Write-Warn "Some tests failed. Continue anyway? (y/N)"
            $continue = Read-Host
            if ($continue -ne "y" -and $continue -ne "Y") {
                Write-Err "Aborting due to test failures"
                exit 1
            }
        } else {
            Write-Info "All tests passed!"
        }
    } finally {
        Pop-Location
    }
}

# Create temporary staging directory
$StagingDir = Join-Path $env:TEMP "pypath_deploy_$(Get-Date -Format 'yyyyMMddHHmmss')"
$PackageDir = Join-Path $StagingDir "pypath_deploy"

Write-Info "Creating staging directory: $StagingDir"
New-Item -ItemType Directory -Path $PackageDir -Force | Out-Null

# Copy application files
Write-Info "Copying application files..."

# Copy packages directory (monorepo layout)
Copy-Item -Path (Join-Path $ProjectRoot "packages") -Destination $PackageDir -Recurse
Write-Info "  - packages/"

# Remove tests and caches from deployed copy
Get-ChildItem -Path (Join-Path $PackageDir "packages") -Include "tests", "__pycache__", ".pytest_cache" -Recurse -Directory -Force |
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
Get-ChildItem -Path (Join-Path $PackageDir "packages") -Include "*.pyc" -Recurse -Force |
    Remove-Item -Force -ErrorAction SilentlyContinue
Write-Info "  - Cleaned tests/ and __pycache__/ from packages"

# Deployment scripts (copy to root of package, not subdirectory)
$DeployFiles = @("deploy.sh", "pypath_manage.sh", "README.md")
foreach ($file in $DeployFiles) {
    $srcPath = Join-Path $ProjectRoot "deploy\$file"
    if (Test-Path $srcPath) {
        Copy-Item -Path $srcPath -Destination $PackageDir
        Write-Info "  - $file"
    }
}

# Generate thin app.py entry point
$AppPyContent = @"
import sys
from pathlib import Path

# Add package source paths so imports work regardless of how Python is launched.
# Shiny Server uses su --login which prevents .pth editable installs from working.
app_dir = Path(__file__).parent
sys.path.insert(0, str(app_dir / "packages" / "pypath-shiny" / "src"))
sys.path.insert(0, str(app_dir / "packages" / "pypath" / "src"))

from pypath_shiny.app import app

__all__ = ["app"]
"@
$AppPyContent | Out-File -FilePath (Join-Path $PackageDir "app.py") -Encoding UTF8
Write-Info "  - app.py (generated entry point)"

# Remove Python cache files (final cleanup)
Write-Info "Cleaning up cache files..."
Get-ChildItem -Path $PackageDir -Include "__pycache__", "*.pyc", ".pytest_cache" -Recurse -Force |
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

# Read version from pyproject.toml
$PyprojectPath = Join-Path $ProjectRoot "packages\pypath\pyproject.toml"
$Version = "unknown"
if (Test-Path $PyprojectPath) {
    $match = Select-String -Path $PyprojectPath -Pattern '^version\s*=\s*"(.+)"' | Select-Object -First 1
    if ($match) { $Version = $match.Matches.Groups[1].Value }
}
$GitHash = ""
try {
    $GitHash = (git -C $ProjectRoot rev-parse --short HEAD 2>$null)
} catch {}

@"
# PyPath Deployment Package
version=$Version
git_commit=$GitHash
build_date=$(Get-Date -Format "yyyy-MM-dd HH:mm:ss")
build_machine=$env:COMPUTERNAME
"@ | Out-File -FilePath (Join-Path $PackageDir "VERSION") -Encoding UTF8

# Create the tarball
Write-Info "Creating deployment package..."

# Check if tar is available (Windows 10+ has it)
if (Get-Command tar -ErrorAction SilentlyContinue) {
    Push-Location $StagingDir
    try {
        tar -czf $OutputPath "pypath_deploy"
        if ($LASTEXITCODE -eq 0) {
            Write-Info "Package created: $OutputPath"
        } else {
            Write-Err "Failed to create tarball"
            exit 1
        }
    } finally {
        Pop-Location
    }
} else {
    # Fallback: create zip file instead
    $ZipPath = $OutputPath -replace "\.tar\.gz$", ".zip"
    Write-Warn "tar not found, creating ZIP file instead: $ZipPath"
    Compress-Archive -Path $PackageDir -DestinationPath $ZipPath -Force
    Write-Info "Package created: $ZipPath"
    $OutputPath = $ZipPath
}

# Cleanup staging directory
Write-Info "Cleaning up..."
Remove-Item -Path $StagingDir -Recurse -Force

# Get file size
$PackageSize = (Get-Item $OutputPath).Length / 1MB
Write-Info "Package size: $([math]::Round($PackageSize, 2)) MB"

Write-Host ""
Write-Host "=============================================="
Write-Host "  Package Ready!"
Write-Host "=============================================="
Write-Host ""
Write-Host "  Output: $OutputPath"
Write-Host ""
Write-Host "  Next steps:"
Write-Host "  1. Upload to server:"
Write-Host "     scp $OutputPath user@laguna.ku.lt:/tmp/"
Write-Host ""
Write-Host "  2. SSH to server and deploy:"
Write-Host "     ssh user@laguna.ku.lt"
Write-Host "     cd /tmp"
Write-Host "     tar -xzf pypath_deploy.tar.gz"
Write-Host "     cd pypath_deploy"
Write-Host "     sudo ./deploy.sh"
Write-Host ""
