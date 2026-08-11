# Indiana River/Stream Blockade Detection Streamlit App

This Streamlit app runs the trained Phase 3 PyTorch checkpoint from the Indiana river/stream blockade detection project on **one uploaded georeferenced GeoTIFF/TIFF image**.

The app produces **model-generated candidate detections**, not official confirmed obstruction records. Non-inventory detections require human/agency review before becoming official records.

## Recommended input format

The simplified app intentionally accepts one imagery format:

```text
Georeferenced GeoTIFF / TIFF with CRS
```

Why this is the best single format for this project:

- it preserves image pixels without relying on screenshots;
- it stores CRS, affine transform, bounds, and resolution;
- it supports interactive map output and GeoJSON candidate export;
- it can contain 3 RGB bands or 4 RGB+NIR bands;
- if band 4 exists, the app uses it as NIR and computes NDVI;
- if band 4 does not exist, NIR and NDVI are zero-filled transparently.

The app no longer accepts separate PNG/JPG/NIR/DEM uploads in the main interface. DEM and slope channels are zero-filled in this simplified version.

## Included sample input files

The folder `sample_inputs/` contains georeferenced GeoTIFF examples for software testing of the required input format.

These files are **not synthetic training data** and are **not validation labels**. They are test-upload files for verifying that the app can load a GeoTIFF, read CRS, run model inference, and build the interactive map/exports.

Suggested first test:

```text
sample_inputs/test_white_river_harding_st_area_crop_512_georeferenced_epsg4326.tif
```

Larger test:

```text
sample_inputs/test_marion_project_imagery_overlay_full_georeferenced_epsg4326.tif
```

## Model input channels

The app builds the same 9-channel input stack used by the Phase 3 model:

1. red
2. green
3. blue
4. nir_or_zero
5. ndvi01_or_zero
6. water_like_proxy
7. dem01_or_zero
8. dem_slope01_or_zero
9. brightness

For a 3-band RGB GeoTIFF:

```text
red, green, blue = from uploaded GeoTIFF
nir_or_zero = 0
ndvi01_or_zero = 0
dem01_or_zero = 0
dem_slope01_or_zero = 0
water_like_proxy = RGB-derived proxy
brightness = mean RGB
```

For a 4-band RGB+NIR GeoTIFF:

```text
red, green, blue = bands 1, 2, 3
nir_or_zero = band 4
ndvi01_or_zero = computed from band 4 and red band
```

No fake NIR, DEM, or synthetic obstruction imagery is generated.

## Visual output

The main visual output is intentionally limited to two items:

1. the input image preview, and
2. an interactive notebook-style output map.

The interactive output map uses:

```text
blue = official flowline / no final model Yes
yellow = review-required candidate
red = final model Yes candidate
```

Hover tooltips show detection/model details.

## Decision logic

A tile becomes final model `blockade_yn = Yes` only if:

1. `obstruction_probability >= threshold`, and
2. the top predicted class is in the true-obstruction class list.

False-positive or non-obstruction classes are blocked from final Yes, including:

- no_obstruction
- bridge_or_road_crossing_false_positive
- possible_rock_riffle_or_natural_drop
- shadow_tree_canopy_false_positive
- vegetation_aquatic_growth
- dry_or_low_water_channel_artifact
- insufficient_data

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Cloud

1. Push this folder to a GitHub repository.
2. Open Streamlit Community Cloud.
3. Create a new app from the repository.
4. Set the main file path to `app.py`.
5. Deploy.

## Scientific integrity notes

- The app does not retrain the model.
- The app does not create fake/synthetic obstruction images.
- The app does not invent model performance metrics.
- Model-only outputs are candidate detections and should remain review-required until accepted by a human/agency review process.

## v6 note: notebook-style flowline map only

The main visual output intentionally shows only:

1. the uploaded GeoTIFF image preview, and
2. the interactive flowline decision map.

The model still runs internally on image tiles because the neural network requires fixed-size chips. However, raw tile boxes are not displayed as the main output. When official Indiana NHD flowlines are available for the uploaded GeoTIFF bounds, detections are transferred back to those flowlines and displayed as:

- blue: official flowline / no final model Yes
- yellow: review-required candidate
- red: final model Yes candidate

If official NHD flowlines cannot be queried or are not returned for the upload bounds, the app does not fabricate stream lines and does not show tile polygons as a substitute. It displays the uploaded image and a clear message explaining that official flowlines were unavailable for that run.


## v7 update

This version fixes a Streamlit `ValueError: Circular reference detected` that occurred while displaying the preprocessing/official-context diagnostics. The issue was caused by a self-referential diagnostics dictionary in the hydrography query status, not by the model or the uploaded GeoTIFF. The visual-output policy remains: show the uploaded GeoTIFF and the notebook-style flowline decision map; do not show raw model tile boxes as the main map output.


## v8 update: Flowlines plus candidate obstruction spots

The main visual output remains intentionally simple:

1. the uploaded GeoTIFF image preview; and
2. a notebook-style interactive map.

The map now includes official NHD flowline decisions and model candidate spot markers:

- Blue flowline: official flowline / no final model Yes.
- Yellow flowline or spot: review-required model candidate.
- Red flowline or spot: final model Yes candidate after the production decision gate.

The app still does not show raw tile boxes as the main output. The neural network uses tiles internally, but the public map summarizes candidates as direct flowline decisions and centroid spot markers with hover/click details.

The app also exports a reviewer label CSV template. Reviewers fill `reviewed_blockade_yn`, `reviewed_obstruction_type`, `review_status`, `reviewer_name`, `review_date`, and notes. Confirmed records can then be used as training labels in a later model update.

## v9 high-quality defaults

This version removes the public Decision Mode selector and uses the fixed obstruction threshold stored in the model bundle/checkpoint metadata. It also defaults to the most complete scan behavior intended for demonstration and review use:

- tile size fixed at 256 px, matching the trained model chip size;
- tile overlap fixed at 0.75;
- maximum tile scan budget fixed at 5000;
- water-like/suspicious tile prioritization enabled;
- official NHD hydrography context enabled when the GeoTIFF has CRS;
- explicit subsurface/underground flowline exclusion enabled;
- public visual output limited to uploaded image preview plus interactive flowline/candidate-spot map.

These settings increase scan coverage and candidate sensitivity. They do not create new validated accuracy or retrain the model. Larger GeoTIFFs may take longer to process on CPU or Streamlit Community Cloud.


## v11 map behavior

The interactive map no longer places review dots at raw model tile centroids. The neural network still scans fixed-size tiles internally, but any public candidate spot is now snapped to the official NHD/flowline feature that inherits the model decision. This prevents row/column grid-looking dots and keeps review markers on the mapped water-channel/flowline network. Model candidates that cannot be assigned to an official flowline are hidden from the public map and remain available in the detection table/downloads.
