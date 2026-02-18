#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
R_SCRIPT="$SCRIPT_DIR/extract_rpath_data.R"
VERIFY_SCRIPT="$SCRIPT_DIR/verify_rpath_reference.py"
COMMIT=false

usage() {
  cat <<EOF
Usage: $(basename "$0") [--commit]

Runs the R extraction script to regenerate the Rpath reference data (ecosim diagnostics).
By default the script runs the extraction and runs the Python verifier. If --commit is
provided, the generated files under tests/data/rpath_reference will be staged and
committed with a standard message.

Requirements:
 - Rscript must be on PATH
 - Python 3 available for verification
 - (optional) git if using --commit

EOF
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --commit)
      COMMIT=true
      shift
      ;;
    -h|--help)
      usage
      ;;
    *)
      echo "Unknown arg: $1" >&2
      usage
      ;;
  esac
done

if ! command -v Rscript >/dev/null 2>&1; then
  echo "Error: Rscript not found on PATH. Install R and ensure Rscript is available." >&2
  exit 2
fi

echo "Running R extraction script: $R_SCRIPT"
Rscript "$R_SCRIPT"

if [ -x "$(command -v python3)" ]; then
  PY=python3
elif [ -x "$(command -v python)" ]; then
  PY=python
else
  echo "Warning: Python not found on PATH; skipping verification." >&2
  exit 0
fi

echo "Verifying generated reference files with $VERIFY_SCRIPT"
$PY "$VERIFY_SCRIPT"

if [ "$COMMIT" = true ]; then
  if ! command -v git >/dev/null 2>&1; then
    echo "git not found; cannot commit files. Exiting." >&2
    exit 3
  fi
  echo "Staging generated reference files..."
  git add tests/data/rpath_reference || true
  COMMIT_MSG="Regenerate Rpath reference diagnostics (QQ/components)"
  git commit -m "$COMMIT_MSG" || echo "No changes to commit or commit failed."
fi

echo "Done."