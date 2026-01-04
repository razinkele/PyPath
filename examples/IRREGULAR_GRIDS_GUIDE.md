# Irregular Grids in PyPath ECOSPACE

This guide explains how to use irregular (custom polygon) grids in the PyPath ECOSPACE Shiny app.

## Overview

Irregular grids allow you to use realistic spatial geometries for ecosystem modeling, such as:
- Actual coastlines and marine management zones
- Different depth zones (nearshore, shelf, deep water)
- Marine protected areas
- Statistical fishing areas
- Watershed or estuary boundaries

## Supported File Formats

The ECOSPACE page supports three spatial file formats:

1. **GeoJSON** (.geojson, .json) - **RECOMMENDED**
   - Easy to create and edit
   - Human-readable text format
   - Widely supported by GIS tools

2. **Shapefile** (.zip)
   - Traditional GIS format
   - Must be zipped with all components (.shp, .shx, .dbf, .prj)
   - Upload as a single .zip file

3. **GeoPackage** (.gpkg)
   - Modern single-file format
   - Can contain multiple layers

## File Requirements

Your spatial file must contain:

1. **Polygon geometries** - Each feature must be a Polygon (not Point or LineString)
2. **ID field** - A unique identifier for each patch (default field name: "id")
3. **Valid coordinates** - Geographic coordinates (longitude/latitude) in WGS84 (EPSG:4326)

### Example GeoJSON Structure

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {
        "id": 0,
        "name": "Coastal Zone"
      },
      "geometry": {
        "type": "Polygon",
        "coordinates": [[
          [20.0, 55.0],
          [20.5, 55.0],
          [20.5, 55.5],
          [20.0, 55.5],
          [20.0, 55.0]
        ]]
      }
    }
  ]
}
```

## Using Irregular Grids in the App

### Step 1: Prepare Your Spatial File

1. Create or obtain a spatial file with polygon geometries
2. Ensure each polygon has a unique ID field
3. Use WGS84 coordinates (longitude/latitude)
4. Save as GeoJSON, shapefile (zipped), or GeoPackage

**Example file included:** `examples/coastal_grid_example.geojson`
- 10 patches representing nearshore, shelf, and offshore zones
- Demonstrates typical coastal ecosystem structure

### Step 2: Upload to ECOSPACE Page

1. Navigate to the ECOSPACE page in the Shiny app
2. In the "Spatial Grid" accordion panel:
   - Select **"Custom Polygons (Upload Shapefile)"** from Grid Type dropdown
   - Click **"Upload Spatial File"** and select your file
   - (Optional) Specify the ID field name if different from "id"
   - Click **"Create Grid"**

3. The grid will be loaded and visualized showing:
   - Actual polygon shapes
   - Patch ID labels
   - Connectivity information

### Step 3: Configure Habitat and Dispersal

After loading your grid:

1. **Movement & Dispersal**:
   - Set dispersal rates (km²/month) - typical values: 1-100
   - Enable habitat-directed movement if desired
   - Adjust habitat attraction strength (0-1)

2. **Habitat Preferences**:
   - Choose a habitat pattern (uniform, gradient, patchy, etc.)
   - Visualize habitat quality across your irregular grid
   - Different patterns work better for different grid structures

3. **Spatial Fishing**:
   - Select effort allocation method
   - Configure gravity or port-based parameters
   - Preview fishing effort distribution

### Step 4: Run Spatial Simulation

1. Load an Ecopath model on the Home page first
2. Return to ECOSPACE and click **"Run Spatial Simulation"**
3. View results in the simulation tabs

## Creating Your Own Irregular Grids

### Using QGIS (Free GIS Software)

1. **Download QGIS**: https://qgis.org/
2. **Create New Shapefile Layer**:
   - Layer → Create Layer → New Shapefile Layer
   - Geometry type: Polygon
   - CRS: EPSG:4326 (WGS 84)
   - Add field: "id" (Integer)

3. **Draw Polygons**:
   - Use polygon tool to draw your patches
   - Assign unique ID numbers (0, 1, 2, ...)
   - Add other attributes as desired (name, habitat_type, etc.)

4. **Export as GeoJSON**:
   - Right-click layer → Export → Save Features As
   - Format: GeoJSON
   - CRS: EPSG:4326
   - Save

### Using Python (Programmatic Creation)

```python
import geopandas as gpd
from shapely.geometry import Polygon

# Create polygons
patches = [
    Polygon([(20.0, 55.0), (20.5, 55.0), (20.5, 55.5), (20.0, 55.5)]),
    Polygon([(20.5, 55.0), (21.0, 55.0), (21.0, 55.5), (20.5, 55.5)]),
    # ... more patches
]

# Create GeoDataFrame
gdf = gpd.GeoDataFrame({
    'id': range(len(patches)),
    'name': ['Patch 0', 'Patch 1', ...],
}, geometry=patches, crs='EPSG:4326')

# Save as GeoJSON
gdf.to_file('my_grid.geojson', driver='GeoJSON')
```

## Grid Design Tips

### Connectivity
- Patches must share a border (edge) to be connected
- Use "rook" adjacency (shared edges only)
- Avoid very small or very large patches (aim for similar sizes)

### Number of Patches
- **Small grids (3-10 patches)**: Good for testing and simple scenarios
- **Medium grids (10-50 patches)**: Suitable for most applications
- **Large grids (50-200 patches)**: Detailed spatial resolution, slower computation

### Patch Size
- Keep patch areas relatively consistent (within 1:10 ratio)
- Very small patches can cause numerical issues
- Very large patches may not resolve spatial patterns well

### Spatial Scale
- Match patch size to species dispersal distances
- High-mobility species: Use larger patches
- Sedentary species: Use smaller patches

## Troubleshooting

### "Field 'id' not found in shapefile"
- Your file doesn't have an "id" field
- Solution: Either add an "id" field or specify the correct field name in the app

### "No .shp file found in zip archive"
- Your zip file doesn't contain a shapefile
- Solution: Ensure you've zipped all shapefile components (.shp, .shx, .dbf)

### Patches appear disconnected
- Polygons don't share edges (small gaps between them)
- Solution: Ensure polygons touch exactly (use snapping in QGIS)

### Irregular grid visualization looks wrong
- Coordinate system mismatch
- Solution: Ensure your file uses EPSG:4326 (WGS84 lat/lon)

## Advanced Features

### Custom Habitat Attributes
You can include habitat attributes in your spatial file's properties:
```json
"properties": {
  "id": 0,
  "depth": 50.0,
  "temperature": 12.5,
  "habitat_quality": 0.8
}
```

Future versions may support using these attributes directly for habitat preferences.

### Irregular Grid Physics

The ECOSPACE model uses:
- **Diffusion (Fick's Law)**: Movement proportional to biomass gradient
- **Habitat Advection**: Movement toward higher quality habitat
- **Variable patch areas**: Properly accounts for different polygon sizes
- **Border lengths**: Uses actual shared border lengths for flux calculations

This makes irregular grids more realistic than regular rectangular grids.

## Example Use Cases

1. **Coastal Management Zone Modeling**
   - Use actual MPA boundaries from GIS databases
   - Compare management scenarios (MPAs vs open areas)

2. **Depth Zone Analysis**
   - Create zones based on bathymetry (0-50m, 50-200m, >200m)
   - Model depth-specific communities

3. **Estuary-to-Ocean Gradients**
   - Model salinity gradients with irregular patches
   - Represent mixing zones accurately

4. **Fishing Ground Analysis**
   - Use statistical fishing areas as patches
   - Model fleet behavior and effort allocation

## References

- **PyPath Spatial Documentation**: See `src/pypath/spatial/` for implementation details
- **GeoJSON Specification**: https://geojson.org/
- **QGIS Tutorial**: https://docs.qgis.org/
- **Shapely Documentation**: https://shapely.readthedocs.io/

## Need Help?

- Check the example file: `examples/coastal_grid_example.geojson`
- Review test files: `tests/test_irregular_grids.py`
- See visualization examples: `examples/ecospace_demo.py`

---

*Last updated: 2025-12-15*
