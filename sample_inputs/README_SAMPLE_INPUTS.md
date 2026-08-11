# Sample GeoTIFF inputs

These files are provided to test the web app's required single input format:

```text
Georeferenced GeoTIFF / TIFF with CRS
```

They are software test uploads only. They are not synthetic obstruction examples, not fake labels, and not validation accuracy records.

Recommended test order:

1. `test_white_river_harding_st_area_crop_512_georeferenced_epsg4326.tif`  
   Small first test. Should load quickly and exercise GeoTIFF/CRS behavior.

2. `test_marion_center_area_crop_512_georeferenced_epsg4326.tif`  
   Small second test.

3. `test_marion_ne_area_crop_512_georeferenced_epsg4326.tif`  
   Small third test.

4. `test_marion_project_imagery_overlay_full_georeferenced_epsg4326.tif`  
   Larger test. Increase maximum tiles if you want broader scanning.

Expected behavior:

- app reads CRS and image bounds;
- app builds the 9-channel model stack;
- app runs tiled inference;
- app creates the input image preview;
- app creates the interactive output map;
- app enables CSV/GeoJSON/HTML-map downloads when detections are available.
