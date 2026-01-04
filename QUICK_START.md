# PyPath Biodiversity Integration - Quick Start

## Installation (One Time Only)

```bash
# 1. Activate conda environment
conda activate shiny

# 2. Install dependencies
pip install pyworms pyobis

# 3. Verify (should see [OK] for all checks)
python verify_biodata_deps.py
```

**Or run automated installer:**
```bash
install_biodata_deps.bat
```

---

## Testing (After Installation)

```bash
# Test the workflow (takes 1-2 minutes)
python test_biodata_workflow.py

# Should see:
# [OK] WoRMS API accessible
# [OK] OBIS API accessible
# [OK] FishBase API accessible
# [OK] Species data retrieved
# [OK] Model created
```

---

## Using in Shiny App

```bash
# 1. Start app
conda activate shiny
shiny run app/app.py

# 2. In browser (http://127.0.0.1:8000):
#    - Click "Data Import" tab
#    - Click "Biodiversity" sub-tab
#    - Click "Load Example"
#    - Click "Fetch Species Data" (wait 30-60 sec)
#    - Review results
#    - Click "Create Ecopath Model"
#    - Click "Use This Model in Ecopath"
#    - Go to "Ecopath Model" tab
```

---

## What You Can Do

### Build Models From Scratch
- Enter any marine species names (common names)
- Automatic lookup in WoRMS, OBIS, FishBase
- Get trophic levels, diet, growth data
- Generate complete Ecopath model

### Example Species
- Fish: cod, herring, mackerel, tuna, salmon
- Invertebrates: shrimp, crab, squid, krill
- Primary producers: phytoplankton, seaweed
- Zooplankton: zooplankton, copepods

### Data Sources
- **1.4+ million species** (WoRMS)
- **130+ million occurrences** (OBIS)
- **35,000+ fish species** (FishBase)

---

## Troubleshooting

### "Could not find species"
**Fix:** Install dependencies
```bash
conda activate shiny
pip install pyworms pyobis
```

### "Module not found"
**Fix:** Verify environment
```bash
conda activate shiny
python verify_biodata_deps.py
```

### "API timeout"
**Fix:** Try again (APIs sometimes slow)
- Increase timeout in settings
- Use fewer species
- Check internet connection

---

## Quick Reference

| Command | Purpose |
|---------|---------|
| `conda activate shiny` | Activate environment |
| `pip install pyworms pyobis` | Install dependencies |
| `python verify_biodata_deps.py` | Check installation |
| `python test_biodata_workflow.py` | Test workflow |
| `shiny run app/app.py` | Start app |
| `pytest tests/test_biodata.py -v` | Run unit tests |

---

## Documentation

| File | Purpose |
|------|---------|
| `CONDA_BIODATA_SETUP.md` | Detailed conda setup |
| `BIODATA_SETUP_GUIDE.md` | General setup & troubleshooting |
| `verify_biodata_deps.py` | Check dependencies |
| `test_biodata_workflow.py` | Test everything |
| `install_biodata_deps.bat` | Automated installer |

---

## Need Help?

1. Check `CONDA_BIODATA_SETUP.md` for detailed instructions
2. Run `python verify_biodata_deps.py` to check setup
3. Run `python test_biodata_workflow.py` to test
4. Check session summary: `SESSION_SUMMARY_2025-12-17.md`

---

**Ready in 3 commands:**
```bash
conda activate shiny
pip install pyworms pyobis
shiny run app/app.py
```

**Then:** Data Import → Biodiversity → Load Example → Fetch Data 🎉
