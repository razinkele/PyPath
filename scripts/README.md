Regenerating Rpath reference data

This directory contains helpers to re-run the R extraction and verify the generated diagnostics used by tests.

Scripts:
- run_extract_rpath_data.sh / run_extract_rpath_data.bat
  - Runs scripts/extract_rpath_data.R (requires Rscript on PATH).
  - After extraction runs scripts/verify_rpath_reference.py to sanity-check outputs.
  - Pass `--commit` to stage and commit changes to git (if you want to create a PR with regenerated data).

- verify_rpath_reference.py
  - Verifies `tests/data/rpath_reference/ecosim/diagnostics/meta.json` and that `seabirds_qq_rk4.csv` and `seabirds_components_rk4.csv` contain non-NA per-term data when `meta.json` indicates QQ diagnostics were produced.

Usage (Linux/macOS):
  ./scripts/run_extract_rpath_data.sh
  ./scripts/run_extract_rpath_data.sh --commit

Usage (Windows):
  scripts\run_extract_rpath_data.bat
  scripts\run_extract_rpath_data.bat --commit

Notes:
- Running the extraction requires R + Rpath package available in your R environment.
- The verifier runs with Python and will exit non-zero if the generated files are inconsistent with the metadata.
- Regenerating reference data may be large; prefer to review diffs before committing and creating a PR.
