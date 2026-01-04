# Hexagonal Grid Generation in PyPath ECOSPACE

## Overview

PyPath ECOSPACE now supports **automatic hexagonal grid generation** within custom boundary polygons. This feature allows you to define a study area boundary and automatically tessellate it with regular hexagons of your chosen size.

## Why Hexagonal Grids?

Hexagonal grids offer several advantages over rectangular grids for spatial ecosystem modeling:

### 1. **Spatial Isotropy**
- No directional bias (rectangular grids favor horizontal/vertical movement)
- Uniform connectivity in all directions
- Better approximation of circular dispersal patterns

### 2. **Equidistant Neighbors**
- Each hexagon has 6 neighbors at equal distances
- Simplifies dispersal calculations
- More realistic representation of spatial processes

### 3. **Better Area Coverage**
- Hexagons tile more efficiently than circles
- Less edge effects than rectangular grids
- Smoother gradients and transitions

### 4. **Natural for Marine Systems**
- Matches radial dispersal patterns of larvae and propagules
- Better represents ocean currents and eddies
- Commonly used in fisheries management (e.g., ICES statistical rectangles)

## How to Use

### Step 1: Prepare a Boundary File

Create or obtain a polygon file defining your study area boundary. This can be:
- **GeoJSON** (.geojson, .json) - Recommended
- **Shapefile** (.zip with .shp, .shx, .dbf, .prj)
- **GeoPackage** (.gpkg)

**Example boundary included**: `examples/baltic_sea_boundary.geojson`

### Step 2: Upload and Configure in ECOSPACE

1. Navigate to **ECOSPACE** page
2. Select **"Custom Polygons (Upload Shapefile)"** from Grid Type
3. Upload your boundary file
4. Select **"Create hexagonal grid within boundary"** as Grid Mode
5. Choose hexagon size using the slider (0.25 - 3.0 km)
6. Click **"Create Grid"**

### Step 3: View and Use the Grid

The app will:
- Generate hexagons covering your boundary
- Clip hexagons to fit exactly within the boundary
- Calculate connectivity (6 neighbors per hexagon)
- Visualize the hexagonal tessellation
- Enable spatial simulation

## Hexagon Size Selection

### Size Parameter
The hexagon size is the **radius from center to vertex** in kilometers.

**Geometric relationships:**
- Flat-to-flat width = size × √3
- Point-to-point height = size × 2
- Area ≈ 2.598 × size²

### Recommended Sizes by Application

| Hexagon Size | Patch Count | Use Case |
|--------------|-------------|----------|
| **0.25 km (250m)** | Very high | Fine-scale coastal habitats, coral reefs, estuaries |
| **0.5 km (500m)** | High | Nearshore ecosystems, bays, small MPAs |
| **1.0 km** | Medium | Regional coastal studies (default) |
| **2.0 km** | Low | Large marine ecosystems, shelf areas |
| **3.0 km** | Very low | Basin-scale studies, coarse resolution |

### Performance Considerations

**Smaller hexagons:**
- ✅ Higher spatial resolution
- ✅ Better representation of habitat heterogeneity
- ❌ More patches = slower computation
- ❌ Longer simulation times

**Larger hexagons:**
- ✅ Fewer patches = faster computation
- ✅ Suitable for large study areas
- ❌ Lower spatial resolution
- ❌ May miss fine-scale patterns

**Rule of thumb**: Hexagon size should be **similar to** or **smaller than** the typical dispersal distance of your focal species.

## Technical Details

### Hexagonal Tessellation Algorithm

The implementation uses a standard hexagonal tiling pattern:

1. **Projection**: Boundary is projected to UTM (automatic zone detection)
2. **Grid Generation**: Hexagons are generated in rows with offset pattern
   - Odd rows offset by half hex-width
   - Vertical spacing = 0.75 × hex height
3. **Clipping**: Each hexagon is intersected with boundary
4. **Filtering**: Hexagons with <10% overlap are discarded
5. **Adjacency**: Neighbor connections detected via spatial overlap
6. **Reprojection**: Final grid converted back to WGS84

### Coordinate Systems

- **Input**: WGS84 (EPSG:4326) geographic coordinates
- **Processing**: UTM projection for accurate metric calculations
- **Output**: WGS84 for compatibility with other tools

The UTM zone is automatically selected based on boundary centroid longitude.

### Edge Handling

Hexagons at the boundary edge are clipped to fit exactly:
- Maintains hexagon IDs (0, 1, 2, ...)
- Edge hexagons may be irregular polygons
- Connectivity preserved where hexagons touch
- Areas calculated accurately for clipped hexagons

## Example: Baltic Sea Study

Using the included example:

```python
# In the ECOSPACE UI:
# 1. Upload: examples/baltic_sea_boundary.geojson
# 2. Mode: Create hexagonal grid within boundary
# 3. Size: 1.0 km
# 4. Result: ~800-1200 hexagonal patches covering the area
```

Expected characteristics:
- Patch count depends on boundary area and hexagon size
- Connectivity: Most hexagons have 6 neighbors
- Edge hexagons: 2-5 neighbors (boundary edge)
- Area variation: Center hexagons uniform, edge hexagons smaller

## Comparison with Other Grid Types

| Grid Type | Neighbors | Isotropy | Complexity | Best For |
|-----------|-----------|----------|------------|----------|
| **Hexagonal** | 6 equal | Excellent | Medium | General spatial modeling |
| **Rectangular** | 4 (rook) or 8 (queen) | Poor | Low | Simple testing, land applications |
| **Triangular** | 3-12 varied | Good | High | Irregular coastlines, high detail |
| **Custom Polygons** | Variable | Depends | Low (provided) | Real management zones, MPAs |

## Advanced Tips

### 1. Boundary Simplification

For very complex coastlines, consider simplifying your boundary:
```python
import geopandas as gpd
gdf = gpd.read_file("complex_boundary.geojson")
gdf_simple = gdf.simplify(tolerance=0.01)  # degrees
gdf_simple.to_file("simplified_boundary.geojson")
```

### 2. Multi-Resolution Grids

Create different hexagon sizes for different areas:
1. Generate fine hexagons for nearshore
2. Generate coarse hexagons for offshore
3. Merge the two grids (advanced - requires custom code)

### 3. Habitat Integration

Match hexagon size to habitat patch sizes:
- Small hexagons for heterogeneous habitats (kelp forests, seagrass)
- Large hexagons for homogeneous habitats (open water)

### 4. Dispersal Distance Matching

Rule of thumb for pelagic larvae:
- Hexagon size ≈ 0.1 × dispersal distance
- Allows ~10 patches for typical larval trajectory
- Example: 10 km dispersal → 1 km hexagons

## Troubleshooting

### "No hexagons fit within the boundary"
**Cause**: Hexagon size too large for boundary area
**Solution**: Reduce hexagon size or increase boundary area

### Too many hexagons (>500)
**Cause**: Hexagon size too small for boundary area
**Solution**: Increase hexagon size for faster computation

### Irregular hexagons at edges
**Behavior**: Expected and correct
**Explanation**: Edge hexagons are clipped to boundary

### Some hexagons have <6 neighbors
**Behavior**: Expected at edges
**Explanation**: Boundary hexagons have fewer neighbors

### Slow grid generation (>30 seconds)
**Cause**: Very small hexagons or large boundary area
**Solution**: Increase hexagon size or reduce boundary area

## Validation

After creating a hexagonal grid, check:

✅ **Patch count**: Reasonable for boundary size and hexagon size
✅ **Connectivity**: Most hexagons have 6 neighbors (check Grid Info)
✅ **Visualization**: Hexagonal pattern visible in Grid Plot
✅ **Edge clipping**: Boundary edges match uploaded polygon

## Spatial Simulations with Hexagonal Grids

Hexagonal grids work seamlessly with all ECOSPACE features:

- ✅ **Habitat preferences**: Map habitat quality to hexagons
- ✅ **Dispersal**: Diffusion and advection between hexagons
- ✅ **Fishing effort**: Allocate fishing across hexagonal patches
- ✅ **Environmental forcing**: Time-varying conditions per hexagon
- ✅ **Results visualization**: Biomass heatmaps on hexagons

## References

### Scientific Literature
- Birch, C.P.D. et al. (2007). "Rectangular and hexagonal grids used for observation, experiment and simulation in ecology." *Ecological Modelling*, 206(3-4), 347-359.
- Carr, M.H. et al. (2003). "Comparing marine and terrestrial ecosystems: implications for the design of coastal marine reserves." *Ecological Applications*, 13(sp1), S90-S107.

### GIS Resources
- H3 Hexagonal Hierarchical Geospatial Indexing System: https://h3geo.org/
- QGIS Hexagonal Grid Plugin
- GDAL/OGR grid generation utilities

## Code Example (Python)

For programmatic hexagon generation:

```python
from pypath.spatial import create_hexagonal_grid_in_boundary
import geopandas as gpd

# Load boundary
boundary = gpd.read_file("my_study_area.geojson")

# Generate hexagonal grid
hex_grid = create_hexagonal_grid_in_boundary(
    boundary_gdf=boundary,
    hexagon_size_km=1.0  # 1 km hexagons
)

print(f"Created {hex_grid.n_patches} hexagons")
print(f"Average neighbors: {hex_grid.adjacency_matrix.nnz / hex_grid.n_patches:.1f}")
```

## Future Enhancements

Planned features:
- Variable hexagon sizes (finer in areas of interest)
- Hierarchical hexagonal grids (H3 integration)
- Export hexagonal grids to shapefile
- Pre-computed hexagon grids for common regions

## Need Help?

- Check the example file: `examples/baltic_sea_boundary.geojson`
- Review the irregular grids guide: `examples/IRREGULAR_GRIDS_GUIDE.md`
- Open an issue on GitHub with your boundary file

---

**Version**: 1.0
**Last Updated**: 2025-12-15
**Author**: PyPath Development Team
