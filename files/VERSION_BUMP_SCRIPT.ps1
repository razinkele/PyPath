# Version Bump Script for PyPath v0.5.3

# Update packages/pypath/pyproject.toml
(Get-Content 'packages\pypath\pyproject.toml') -replace 'version = "0\.5\.2"', 'version = "0.5.3"' | Set-Content 'packages\pypath\pyproject.toml'

# Update packages/pypath/src/pypath/__init__.py
(Get-Content 'packages\pypath\src\pypath\__init__.py') -replace '__version__ = "0\.5\.2"', '__version__ = "0.5.3"' | Set-Content 'packages\pypath\src\pypath\__init__.py'

# Update packages/pypath-shiny/pyproject.toml
(Get-Content 'packages\pypath-shiny\pyproject.toml') -replace 'version = "0\.5\.2"', 'version = "0.5.3"' | Set-Content 'packages\pypath-shiny\pyproject.toml'

# Update packages/pypath-shiny/src/pypath_shiny/__init__.py
(Get-Content 'packages\pypath-shiny\src\pypath_shiny\__init__.py') -replace '__version__ = "0\.5\.2"', '__version__ = "0.5.3"' | Set-Content 'packages\pypath-shiny\src\pypath_shiny\__init__.py'

# Update CLAUDE.md version table
(Get-Content 'CLAUDE.md') -replace 'v0\.5\.2 \| v0\.5\.2', 'v0.5.3 | v0.5.3' | Set-Content 'CLAUDE.md'

Write-Host "✅ Version bumped to 0.5.3 across all packages" -ForegroundColor Green
