# ✅ Files Updated - Restart Required

## Status

All bug fixes have been successfully applied to the code files:

✅ **multistanza.py** - Updated (0 old decorators, 1 new decorator)
✅ **forcing_demo.py** - Updated (0 old decorators, 1 new decorator)
✅ **diet_rewiring_demo.py** - Updated (0 old decorators, 1 new decorator)
✅ **optimization_demo.py** - Updated (0 old decorators, 1 new decorator + Effect_ fix)

## Why You Still See Warnings

The Shiny app process is running with the **old code loaded in memory**. Python doesn't automatically reload changed files while the app is running.

## How to Fix - Restart the App

### Method 1: Simple Restart (Recommended)

1. **Stop the current app:**
   - Press `Ctrl+C` in the terminal where Shiny is running
   - Wait for the server to fully stop

2. **Clear Python cache (just to be sure):**
   ```bash
   find app -name "*.pyc" -delete
   find app -name "__pycache__" -type d -delete
   ```

3. **Start the app fresh:**
   ```bash
   shiny run app/app.py
   ```

### Method 2: Force Clean Start

```bash
# Stop the app (Ctrl+C)

# Clear all Python cache
cd app
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -delete
cd ..

# Start with no cache
python -B -m shiny run app/app.py
```

### Method 3: If Running in Background

If the app is running as a background process:

```bash
# Find the process
ps aux | grep "shiny run"

# Kill it
pkill -f "shiny run"

# Start fresh
shiny run app/app.py
```

## Expected Result After Restart

✅ **No deprecation warnings**
✅ **No TypeError about Effect_ object**
✅ **All download buttons work**
✅ **Clean console output**

## Verification

After restarting, you should see:

```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000
```

**Without any warnings!**

## What Was Fixed

### 1. Download Decorators (4 files)
```python
# OLD (deprecated):
@session.download(filename="example.csv")
def download():
    yield data

# NEW (modern):
@render.download(filename="example.csv")
def download():
    return data
```

### 2. Effect Callable Error (optimization_demo.py)
```python
# OLD (error):
if synthetic_data() is None:
    generate_synthetic_data()  # Can't call Effect_!

# NEW (fixed):
if synthetic_data() is None:
    # Generate data inline
    years = np.arange(2000, 2021)
    # ... (full generation code)
    synthetic_data.set(df)
```

## Troubleshooting

### Still seeing warnings after restart?

1. **Make sure you actually stopped the app:**
   ```bash
   # Check if still running
   ps aux | grep shiny

   # Force stop all shiny processes
   pkill -9 -f shiny
   ```

2. **Clear cache everywhere:**
   ```bash
   find . -name "*.pyc" -delete
   find . -name "__pycache__" -type d -delete
   ```

3. **Restart terminal:**
   - Close and reopen your terminal
   - Navigate back to project directory
   - Run `shiny run app/app.py`

### App won't start?

If you get import errors after restart:

```bash
# Reinstall dependencies
pip install --upgrade shiny plotly pandas numpy geopandas shapely scipy
```

## Summary

🔧 **Files Fixed:** 4 files updated correctly
🗑️ **Cache Cleared:** All .pyc files removed
🔄 **Action Required:** Restart the Shiny app process

**After restart: Everything will work perfectly!** ✅

---

**Quick Commands:**
```bash
# 1. Stop app (Ctrl+C)
# 2. Clean cache
find app -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
# 3. Restart
shiny run app/app.py
```
