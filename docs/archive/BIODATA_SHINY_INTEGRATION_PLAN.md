# Biodiversity Database Integration for Shiny App - Analysis & Plan

## Current Status

**Finding:** The biodiversity database routines (WoRMS, OBIS, FishBase) are **NOT yet integrated** into the Shiny app.

### What Exists

✅ **Backend Module Complete:**
- `src/pypath/io/biodata.py` - Fully implemented (1,312 lines)
- Functions: `get_species_info()`, `batch_get_species_info()`, `biodata_to_rpath()`
- All tests passing (32 unit + 50+ integration tests)
- Comprehensive documentation

✅ **Current Shiny App Data Import:**
- **EcoBase** integration - Complete ✓
- **EwE Database (.ewemdb)** integration - Complete ✓
- **Biodiversity Databases** - **Missing** ✗

### Gap Analysis

The Shiny app's Data Import page (`app/pages/data_import.py`) currently has:
1. **EcoBase tab** - Download models from online database
2. **EwE File tab** - Upload local .ewemdb files

**Missing:** A third tab for biodiversity databases (WoRMS/OBIS/FishBase)

## Proposed Integration

### Option 1: Add Tab to Existing Data Import Page (Recommended)

Add a third tab "Biodiversity Data" to `app/pages/data_import.py`

**UI Components Needed:**
```python
ui.nav_panel(
    "Biodiversity Data",
    ui.div(
        # Species input section
        ui.p("Build models from global biodiversity databases", class_="small text-muted"),
        ui.input_text_area(
            "biodata_species_list",
            "Species List (one per line)",
            placeholder="Atlantic cod\nHerring\nSprat\nZooplankton",
            rows=6
        ),

        # Options
        ui.input_numeric("biodata_area", "Model Area (km²)", value=1000, min=1),
        ui.input_checkbox("biodata_include_occurrences", "Include OBIS occurrence data", value=True),
        ui.input_checkbox("biodata_include_traits", "Include FishBase traits", value=True),

        # Biomass estimates
        ui.h6("Biomass Estimates (t/km²)"),
        ui.output_ui("biodata_biomass_inputs"),

        # Action buttons
        ui.div(
            ui.input_action_button(
                "btn_fetch_biodata",
                ui.tags.span(ui.tags.i(class_="bi bi-download me-1"), "Fetch Data"),
                class_="btn-primary btn-sm"
            ),
            ui.input_action_button(
                "btn_create_model",
                ui.tags.span(ui.tags.i(class_="bi bi-gear me-1"), "Create Model"),
                class_="btn-success btn-sm ms-2"
            ),
            class_="mb-3"
        ),

        # Progress and results
        ui.output_ui("biodata_progress"),
        ui.output_data_frame("biodata_results_table"),

        # Use model button
        ui.output_ui("use_model_button_biodata"),
    )
)
```

**Server Functions Needed:**
```python
# In import_server()

# Reactive value to store fetched species data
biodata_df = reactive.Value(None)
biodata_model = reactive.Value(None)

# Fetch species data
@reactive.effect
@reactive.event(input.btn_fetch_biodata)
def fetch_biodata():
    species_text = input.biodata_species_list()
    if not species_text:
        return

    species_list = [s.strip() for s in species_text.split('\n') if s.strip()]

    with ui.Progress(min=0, max=len(species_list)) as p:
        p.set(message="Fetching biodiversity data...", detail="This may take a minute")

        try:
            df = batch_get_species_info(
                species_list,
                include_occurrences=input.biodata_include_occurrences(),
                include_traits=input.biodata_include_traits(),
                strict=False,
                max_workers=5
            )
            biodata_df.set(df)
            ui.notification_show(
                f"Successfully fetched data for {len(df)} species!",
                type="message",
                duration=3
            )
        except Exception as e:
            ui.notification_show(
                f"Error fetching data: {str(e)}",
                type="error",
                duration=5
            )

# Display results table
@render.data_frame
def biodata_results_table():
    df = biodata_df()
    if df is None:
        return pd.DataFrame()

    # Show key columns
    display_df = df[[
        'common_name', 'scientific_name',
        'trophic_level', 'max_length', 'occurrence_count'
    ]].copy()
    return render.DataGrid(display_df, row_selection_mode="single")

# Create dynamic biomass inputs
@render.ui
def biodata_biomass_inputs():
    df = biodata_df()
    if df is None:
        return ui.p("Fetch species data first", class_="text-muted small")

    inputs = []
    for _, row in df.iterrows():
        sp_name = row['common_name']
        input_id = f"biomass_{sp_name.replace(' ', '_')}"
        inputs.append(
            ui.input_numeric(
                input_id,
                sp_name,
                value=1.0,
                min=0.001,
                step=0.1
            )
        )
    return ui.div(*inputs)

# Create Ecopath model from biodata
@reactive.effect
@reactive.event(input.btn_create_model)
def create_biodata_model():
    df = biodata_df()
    if df is None:
        ui.notification_show("No species data available", type="warning")
        return

    # Collect biomass estimates
    biomass_estimates = {}
    for _, row in df.iterrows():
        sp_name = row['common_name']
        input_id = f"biomass_{sp_name.replace(' ', '_')}"
        biomass_estimates[sp_name] = input[input_id]()

    try:
        params = biodata_to_rpath(
            df,
            biomass_estimates=biomass_estimates,
            area_km2=input.biodata_area()
        )
        biodata_model.set(params)
        ui.notification_show(
            "Model created successfully!",
            type="message",
            duration=3
        )
    except Exception as e:
        ui.notification_show(
            f"Error creating model: {str(e)}",
            type="error",
            duration=5
        )

# Use model button
@render.ui
def use_model_button_biodata():
    if biodata_model() is None:
        return None

    return ui.input_action_button(
        "btn_use_biodata_model",
        ui.tags.span(ui.tags.i(class_="bi bi-check-circle me-1"), "Use This Model"),
        class_="btn-success w-100"
    )

@reactive.effect
@reactive.event(input.btn_use_biodata_model)
def use_biodata_model():
    model_data.set(biodata_model())
    ui.notification_show("Model loaded! Go to Ecopath tab to balance.", type="message")
```

### Option 2: Create Separate Page (Alternative)

Create `app/pages/biodata_import.py` as a standalone page.

**Pros:**
- More space for advanced features
- Cleaner separation of concerns
- Can add data visualization (maps, trait plots, etc.)

**Cons:**
- Another navigation item (UI clutter)
- Less integrated workflow

**Recommendation:** Use Option 1 unless advanced features are planned

## Implementation Steps

### Step 1: Update data_import.py (2-3 hours)

1. Add imports:
```python
from pypath.io.biodata import (
    get_species_info,
    batch_get_species_info,
    biodata_to_rpath,
    BiodataError,
)
```

2. Add UI tab (as shown above)

3. Add server functions (as shown above)

4. Test functionality

### Step 2: Add Documentation (30 min)

1. Update app help text
2. Add tooltips for biodiversity databases
3. Link to WoRMS/OBIS/FishBase websites

### Step 3: Add Example Dataset (15 min)

Create example species list button:
```python
ui.input_action_button("btn_load_example", "Load Example", class_="btn-sm")

@reactive.effect
@reactive.event(input.btn_load_example)
def load_example():
    example_species = """Atlantic cod
Herring
Sprat
Zooplankton
Phytoplankton"""
    ui.update_text_area("biodata_species_list", value=example_species)
```

### Step 4: Add Error Handling (30 min)

- Handle missing species
- Handle API timeouts
- Show partial results
- Cache management UI

### Step 5: Testing (1 hour)

- Test with real APIs
- Test error scenarios
- Test model creation
- Test workflow integration

## Advanced Features (Optional)

### Phase 2 Enhancements:

1. **Interactive Species Explorer**
   - Dropdown with autocomplete
   - Common name search
   - Taxonomy browser

2. **Data Visualization**
   - OBIS occurrence map (Leaflet)
   - Trophic level chart
   - Size distribution plot

3. **Data Quality Indicators**
   - Show data completeness
   - Flag missing traits
   - Suggest parameter estimates

4. **Cache Management**
   - Show cached species
   - Clear cache button
   - Cache statistics

5. **Batch Import**
   - Upload CSV with species list
   - Download results as CSV
   - Template download

## Benefits of Integration

### For Users:
- **Build models from scratch** using global databases
- **No manual parameter entry** - auto-populated from science
- **Access to 1000+ species** with trait data
- **Standardized methodology** - reproducible models

### For Science:
- **Data provenance** - clear source of all parameters
- **Reproducibility** - parameters from databases
- **Up-to-date data** - always latest from APIs
- **Quality assured** - peer-reviewed database content

## Dependencies

Already satisfied:
- ✅ `pyworms` - installed
- ✅ `pyobis` - installed
- ✅ `requests` - installed
- ✅ Backend module complete
- ✅ Tests passing

## Risks & Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| API timeouts | Medium | Show progress, allow partial results |
| Rate limiting | Low | Use caching, limit concurrent requests |
| Missing species | Medium | Show warnings, allow manual entry |
| Network errors | Medium | Graceful fallback, retry logic |

## Timeline Estimate

| Task | Time | Priority |
|------|------|----------|
| Add basic tab UI | 1 hour | High |
| Add server functions | 1.5 hours | High |
| Add biomass input UI | 0.5 hour | High |
| Error handling | 0.5 hour | High |
| Testing | 1 hour | High |
| Documentation | 0.5 hour | Medium |
| Example data | 0.25 hour | Medium |
| Advanced features | 4+ hours | Low |

**Total for MVP:** ~5 hours
**Total with polish:** ~6-7 hours

## Recommendation

**Implement Option 1 (Add tab to Data Import page)** as the minimum viable integration:

1. Add third tab "Biodiversity Data" to existing Data Import page
2. Simple species list input + biomass estimates
3. Fetch data button with progress indicator
4. Results table showing key parameters
5. Create model button → loads into Ecopath tab

This provides:
- ✅ Complete workflow integration
- ✅ Minimal UI changes
- ✅ ~5 hours implementation
- ✅ High user value
- ✅ Leverages existing tested backend

## Next Steps

If approved:
1. Create feature branch: `feature/biodata-shiny-integration`
2. Implement basic tab (Steps 1-2)
3. Test with real data
4. Add example + documentation (Steps 3-4)
5. Full testing (Step 5)
6. Merge to main

---

**Status:** Proposed - Awaiting approval
**Effort:** ~5-7 hours for complete integration
**Value:** High - enables model building from scratch
**Risk:** Low - backend fully tested, frontend simple
