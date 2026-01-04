# Spatial File Format Support in ECOSPACE

**Date:** 2025-12-15
**Status:** ✅ Complete - All Major Formats Supported

## Overview

PyPath ECOSPACE fully supports the three major spatial vector file formats:
1. ✅ **GeoJSON** (.geojson, .json)
2. ✅ **GeoPackage** (.gpkg)
3. ✅ **Shapefile** (.zip)

## Supported Formats

### 1. GeoJSON (.geojson, .json)

**Advantages:**
- ✅ Human-readable text format
- ✅ Easy to create and edit
- ✅ Works in web applications
- ✅ No compression needed (single file)
- ✅ Widely supported by GIS tools

**Use Cases:**
- Simple boundaries
- Web mapping applications
- GitHub/version control (text-based)
- Quick prototyping

**Example:**
```json
{
  "type": "FeatureCollection",
  "features": [{
    "type": "Feature",
    "properties": {"id": 0, "name": "Study Area"},
    "geometry": {
      "type": "Polygon",
      "coordinates": [[[20.0, 55.0], [20.2, 55.0], [20.2, 55.2], [20.0, 55.2], [20.0, 55.0]]]
    }
  }]
}
```

**File Size:** Medium (text-based, can be large for complex geometries)

### 2. GeoPackage (.gpkg)

**Advantages:**
- ✅ Single-file format (no separate components)
- ✅ Efficient storage (SQLite-based)
- ✅ Supports multiple layers
- ✅ Rich attribute support
- ✅ Modern OGC standard

**Use Cases:**
- Multiple boundary layers
- Complex attribute data
- Large datasets
- Professional GIS workflows

**Example Creation:**
```python
import geopandas as gpd
from shapely.geometry import Polygon

# Create GeoDataFrame
gdf = gpd.GeoDataFrame([{
    'id': 0,
    'name': 'Marine Protected Area',
    'area_km2': 150.5,
    'protection_level': 'High'
}], geometry=[polygon], crs='EPSG:4326')

# Save as GeoPackage
gdf.to_file('study_area.gpkg', driver='GPKG')
```

**File Size:** Small-Medium (compressed binary format)

### 3. Shapefile (.zip)

**Advantages:**
- ✅ Industry standard (widely used)
- ✅ Supported by all GIS software
- ✅ Well-documented format
- ✅ Reliable and stable

**Disadvantages:**
- ❌ Multiple files (.shp, .shx, .dbf, .prj)
- ❌ Requires zipping for upload
- ❌ 10-character field name limit
- ❌ Limited attribute types

**Use Cases:**
- Legacy GIS data
- Government/agency data
- Maximum compatibility

**Required Components:**
- `.shp` - Shape geometry (required)
- `.shx` - Shape index (required)
- `.dbf` - Attribute data (required)
- `.prj` - Projection info (recommended)

**File Size:** Medium (binary format)

## Format Comparison Table

| Feature | GeoJSON | GeoPackage | Shapefile |
|---------|---------|------------|-----------|
| **Single file** | ✅ Yes | ✅ Yes | ❌ No (needs zip) |
| **Human readable** | ✅ Yes | ❌ No | ❌ No |
| **Compression** | ❌ No | ✅ Yes | ❌ No |
| **Multiple layers** | ❌ No | ✅ Yes | ❌ No |
| **Attribute support** | ✅ Good | ✅ Excellent | ⚠️ Limited |
| **File size** | Large | Small | Medium |
| **Web compatibility** | ✅ Excellent | ❌ Poor | ❌ Poor |
| **GIS software support** | ✅ Good | ✅ Excellent | ✅ Excellent |
| **Modern standard** | ✅ Yes | ✅ Yes | ❌ No (legacy) |

## Implementation in ECOSPACE

### File Upload UI

```python
ui.input_file(
    "spatial_file_upload",
    "Upload Spatial File",
    accept=[".zip", ".geojson", ".json", ".gpkg"],
    multiple=False
)
```

**Accepted Extensions:**
- `.geojson` - GeoJSON format
- `.json` - GeoJSON format (alternative extension)
- `.gpkg` - GeoPackage format
- `.zip` - Zipped Shapefile

### File Processing Logic

**GeoJSON/GeoPackage:**
```python
elif file_name.endswith(('.geojson', '.json', '.gpkg')):
    # Copy file to temp directory
    spatial_file = str(Path(temp_dir) / file_name)
    shutil.copy(file_path, spatial_file)
```

**Shapefile:**
```python
if file_name.endswith('.zip'):
    # Extract shapefile from zip
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)

    # Find the .shp file
    shp_files = list(Path(temp_dir).glob('**/*.shp'))
    spatial_file = str(shp_files[0])
```

**Universal Loading:**
```python
# Load with geopandas (works for all formats)
boundary_gdf = gpd.read_file(spatial_file)
```

GeoPandas automatically detects the format based on file extension!

## Usage Examples

### Using GeoJSON

```bash
# 1. Create GeoJSON in QGIS or online tool
# 2. Upload directly (no zipping needed)
# 3. Select grid mode and generate
```

### Using GeoPackage

```bash
# 1. Export from QGIS: Layer → Save As → GeoPackage
# 2. Upload .gpkg file
# 3. Works with multiple layers (first layer used)
```

### Using Shapefile

```bash
# 1. Export from QGIS: Layer → Save As → ESRI Shapefile
# 2. Zip all components (.shp, .shx, .dbf, .prj)
# 3. Upload .zip file
```

## Requirements

### For GeoJSON
- ✅ Valid JSON syntax
- ✅ "FeatureCollection" or "Feature" type
- ✅ Polygon or MultiPolygon geometry
- ✅ (Optional) "id" field for polygon mode

### For GeoPackage
- ✅ Valid GPKG file (SQLite format)
- ✅ At least one vector layer
- ✅ Polygon or MultiPolygon geometry
- ✅ (Optional) "id" field for polygon mode

### For Shapefile
- ✅ All required components (.shp, .shx, .dbf)
- ✅ Zipped into single .zip file
- ✅ Polygon or MultiPolygon geometry
- ✅ (Optional) .prj file for CRS

## Coordinate Reference Systems

### Automatic CRS Handling

All formats support automatic CRS detection and conversion:

```python
boundary_gdf = gpd.read_file(spatial_file)
if boundary_gdf.crs is None:
    boundary_gdf = boundary_gdf.set_crs("EPSG:4326")  # Assume WGS84
else:
    boundary_gdf = boundary_gdf.to_crs("EPSG:4326")  # Convert to WGS84
```

**Supported Input CRS:**
- Any CRS recognized by PROJ/GDAL
- Automatically converted to WGS84 (EPSG:4326)
- UTM zones (converted for processing)
- Local/regional CRS

## Testing

### Test Suite

**File:** `tests/test_file_format_support.py`

**Test Classes:**
1. `TestGeoJSONSupport` - GeoJSON reading and attributes
2. `TestGeoPackageSupport` - GPKG reading and layers
3. `TestShapefileSupport` - Shapefile reading
4. `TestFormatComparison` - Cross-format consistency
5. `TestCRSHandling` - CRS preservation and conversion

**Run Tests:**
```bash
pytest tests/test_file_format_support.py -v
```

## Common Issues and Solutions

### Issue 1: "Unsupported file format"
**Cause:** File extension not recognized
**Solution:** Ensure file ends with .geojson, .json, .gpkg, or .zip

### Issue 2: "No .shp file found in zip archive"
**Cause:** Shapefile components not properly zipped
**Solution:** Zip all files (.shp, .shx, .dbf, .prj) together

### Issue 3: Invalid geometry
**Cause:** Polygon is self-intersecting or invalid
**Solution:** Use QGIS "Fix Geometries" tool before export

### Issue 4: Missing CRS
**Cause:** File doesn't specify coordinate system
**Solution:** PyPath assumes WGS84, or specify CRS in GIS tool

### Issue 5: Large file size
**Cause:** GeoJSON with complex geometries
**Solution:** Use GeoPackage instead (more efficient)

## Recommendations

### For Simple Boundaries
**Use:** GeoJSON
- Easy to create online (geojson.io)
- Easy to edit in text editor
- Works well in Git

### For Complex Boundaries
**Use:** GeoPackage
- Better compression
- Faster loading
- More efficient storage

### For Legacy/Shared Data
**Use:** Shapefile
- Maximum compatibility
- Widely distributed format
- Safe choice for collaboration

### For Production Use
**Recommended Order:**
1. **GeoPackage** - Best overall choice
2. **GeoJSON** - For simple cases
3. **Shapefile** - For compatibility

## Creating Example Files

### GeoJSON Example

```bash
# Use online tool
https://geojson.io

# Or Python
import geopandas as gpd
from shapely.geometry import Polygon

poly = Polygon([(20.0, 55.0), (20.2, 55.0), (20.2, 55.2), (20.0, 55.2), (20.0, 55.0)])
gdf = gpd.GeoDataFrame([{'id': 0}], geometry=[poly], crs='EPSG:4326')
gdf.to_file('boundary.geojson', driver='GeoJSON')
```

### GeoPackage Example

```bash
# QGIS: Layer → Save As...
# - Format: GeoPackage
# - CRS: EPSG:4326 (WGS 84)
# - Save

# Or Python (as above, change driver)
gdf.to_file('boundary.gpkg', driver='GPKG')
```

### Shapefile Example

```bash
# QGIS: Layer → Save As...
# - Format: ESRI Shapefile
# - CRS: EPSG:4326 (WGS 84)
# - Save
# - Zip all files: boundary.zip
```

## Format Support Matrix

| Operation | GeoJSON | GPKG | Shapefile |
|-----------|---------|------|-----------|
| **Load boundary** | ✅ | ✅ | ✅ |
| **Visualize** | ✅ | ✅ | ✅ |
| **Use as polygons** | ✅ | ✅ | ✅ |
| **Generate hexagons** | ✅ | ✅ | ✅ |
| **Multi-polygon** | ✅ | ✅ | ✅ |
| **Multiple layers** | ❌ | ✅ | ❌ |
| **Attributes** | ✅ | ✅ | ✅ |
| **CRS conversion** | ✅ | ✅ | ✅ |

## Summary

✅ **Full support** for all three major vector formats
✅ **Automatic format detection** based on file extension
✅ **CRS handling** with automatic conversion to WGS84
✅ **Comprehensive testing** for all formats
✅ **User-friendly** error messages and help text

**Recommendation:** Use **GeoPackage** for new projects, **GeoJSON** for simple cases, and **Shapefile** when sharing with others.

---

**Implementation completed:** 2025-12-15
**Formats supported:** 3 (GeoJSON, GeoPackage, Shapefile)
**Test coverage:** Complete
**Documentation:** ✅ This document

*For questions or issues, see ECOSPACE page or open a GitHub issue.*
