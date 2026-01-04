#!/bin/bash
# Quick restart script for PyPath Shiny app

echo "========================================"
echo "PyPath App Restart Script"
echo "========================================"

# Stop any running Shiny processes
echo ""
echo "Stopping Shiny processes..."
pkill -f "shiny run" 2>/dev/null
pkill -f "app.py" 2>/dev/null
sleep 1

# Clear Python cache
echo "Clearing Python cache..."
find app -name "*.pyc" -delete 2>/dev/null
find app -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
echo "Cache cleared!"

# Verify files are updated
echo ""
echo "Verifying bug fixes..."
if grep -q "@render.download" app/pages/multistanza.py; then
    echo "  [OK] multistanza.py updated"
else
    echo "  [!!] multistanza.py NOT updated"
fi

if grep -q "@render.download" app/pages/forcing_demo.py; then
    echo "  [OK] forcing_demo.py updated"
else
    echo "  [!!] forcing_demo.py NOT updated"
fi

if grep -q "@render.download" app/pages/diet_rewiring_demo.py; then
    echo "  [OK] diet_rewiring_demo.py updated"
else
    echo "  [!!] diet_rewiring_demo.py NOT updated"
fi

if grep -q "@render.download" app/pages/optimization_demo.py; then
    echo "  [OK] optimization_demo.py updated"
else
    echo "  [!!] optimization_demo.py NOT updated"
fi

echo ""
echo "========================================"
echo "Starting app..."
echo "========================================"
echo ""

# Start app with no bytecode caching
python -B -m shiny run app/app.py
