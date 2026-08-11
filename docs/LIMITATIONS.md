# Limitations

- This is a model inference app, not an official confirmation system.
- Model-only candidate detections require human/agency review.
- The app does not invent or synthesize obstruction examples.
- Image quality, resolution, season, cloud/shadow/tree cover, and sensor differences can affect predictions.
- RGB-only images lack NIR and DEM context; the app zero-fills those missing channels.
- PNG/JPG uploads do not contain true georeferencing, so their output locations are pixel-only.
- GeoTIFF/TIFF with CRS is required for latitude/longitude and GeoJSON output.
- The app reports inference scores from the model; it does not compute or invent validation accuracy for the uploaded image.
