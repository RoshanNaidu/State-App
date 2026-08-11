from __future__ import annotations

from pathlib import Path
import tempfile

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import torch

from config import DEFAULT_MODEL_PATH, FALSE_OR_NONBLOCKING_CLASSES, TRUE_OBSTRUCTION_CLASSES, PHASE3_CHANNEL_NAMES
from hydrography import (
    feature_pixel_bboxes,
    geojson_from_nhd_features,
    image_bounds_wgs84,
    query_official_nhd_flowlines,
)
from interactive_map import create_flowline_decision_map
from model import load_phase3_checkpoint
from preprocess import (
    build_phase3_stack,
    load_uploaded_image,
    stack_to_rgb_preview,
)
from predict import detections_to_geojson, pil_to_bytes, render_mask_heatmap, render_overlay, run_tiled_inference

st.set_page_config(
    page_title="Indiana River Blockade Detection",
    page_icon="🌊",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main .block-container {padding-top: 1.6rem; padding-bottom: 2.0rem;}
    .metric-card {background: #f7f9fb; border: 1px solid #e5e9ef; border-radius: 10px; padding: 12px;}
    .warning-box {background: #fff7e6; border-left: 5px solid #f0a000; padding: 12px; border-radius: 6px;}
    .info-box {background: #eef6ff; border-left: 5px solid #2b7bb9; padding: 12px; border-radius: 6px;}
    .danger-box {background: #fff0f0; border-left: 5px solid #cc3333; padding: 12px; border-radius: 6px;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Indiana River/Stream Blockade Detection")
st.caption("Upload one georeferenced GeoTIFF/TIFF orthophoto or satellite image and run the trained Phase 3 model to produce obstruction candidate predictions.")

st.markdown(
    """
    <div class="warning-box">
    <b>Government-project safety note:</b> This application produces <b>model-generated candidate detections</b>. 
    Non-inventory detections require human/agency review before becoming official records. The app does not generate fake data, 
    synthetic obstruction images, or invented performance metrics.
    </div>
    """,
    unsafe_allow_html=True,
)


def make_json_safe(obj, _seen=None):
    """Return a JSON-safe copy with circular references and complex objects removed.

    Streamlit st.json calls json.dumps internally. Rasterio/pyproj objects,
    numpy scalars, and accidental self-referential diagnostics dictionaries can
    otherwise crash the app. This helper preserves only display diagnostics; it
    does not change model inference.
    """
    import numpy as _np
    if _seen is None:
        _seen = set()
    oid = id(obj)
    if isinstance(obj, (dict, list, tuple, set)):
        if oid in _seen:
            return "<circular_reference_removed>"
        _seen.add(oid)
    if obj is None or isinstance(obj, (str, int, float, bool)):
        try:
            if isinstance(obj, float) and (not _np.isfinite(obj)):
                return None
        except Exception:
            pass
        return obj
    if isinstance(obj, _np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v, _seen) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [make_json_safe(v, _seen) for v in obj]
    # Bounds, CRS, Affine transforms, paths, etc. are display-only here.
    return str(obj)

@st.cache_resource(show_spinner=False)
def cached_load_model(model_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return load_phase3_checkpoint(model_path, device=device)


def write_uploaded_checkpoint(uploaded_model) -> str | None:
    if uploaded_model is None:
        return None
    suffix = Path(uploaded_model.name).suffix or ".pt"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_model.getbuffer())
    tmp.flush(); tmp.close()
    return tmp.name


with st.sidebar:
    st.header("Model")
    default_model_exists = Path(DEFAULT_MODEL_PATH).exists()
    uploaded_model = st.file_uploader("Optional: upload Phase 3 .pt checkpoint", type=["pt", "pth"], help="Leave empty to use model_bundle/phase3_statewide_deep_learning_model.pt.")
    model_path = write_uploaded_checkpoint(uploaded_model) if uploaded_model else DEFAULT_MODEL_PATH
    st.write("Default model found:", "✅" if default_model_exists else "❌")
    st.write("Model path:", model_path)

    st.header("High-quality inference settings")
    st.info(
        "High-quality mode is enabled by default: 256 px model chips, maximum tile overlap, maximum tile scan budget, official hydrography context, and underground/subsurface flowline exclusion. The neural network still runs on tiles internally, but the public map shows flowline decisions and candidate spots, not tile boxes."
    )
    tile_size = 256
    overlap = 0.75
    max_tiles = 5000
    priority_limit = True
    st.write("Tile size:", tile_size)
    st.write("Tile overlap:", overlap)
    st.write("Maximum tiles to scan:", max_tiles)
    st.write("Water-like/suspicious tile priority:", "On")

    st.header("Official context filters")
    use_official_hydrography = True
    exclude_underground = True
    hydro_buffer_px = st.slider("Hydrography corridor buffer, pixels", min_value=8, max_value=128, value=40, step=8, help="Wider buffers scan more context around official NHD flowlines; narrower buffers reduce noise but can miss nearby obstruction context.")
    st.write("Official visible-channel hydrography corridor:", "On")
    st.write("Exclude non-visible/subsurface flowline types:", "On")

try:
    model, model_meta = cached_load_model(model_path)
    model_loaded = True
except Exception as exc:
    model = None
    model_meta = {}
    model_loaded = False
    st.error(f"Model could not be loaded: {exc}")

if model_loaded:
    threshold = float(model_meta.get("obstruction_threshold", 0.5))
    with st.sidebar:
        st.header("Decision gate")
        st.write("Fixed model threshold:", f"{threshold:.3f}")
        st.caption("Decision mode and manual threshold controls are intentionally removed for consistent high-sensitivity government-demo behavior. The threshold is loaded from the model bundle/checkpoint metadata.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model loaded", "Yes")
    c2.metric("Classes", model_meta.get("num_classes", "?"))
    c3.metric("Threshold", f"{threshold:.3f}")
    c4.metric("Device", model_meta.get("device", "unknown"))
    with st.expander("Loaded model metadata", expanded=False):
        st.json({
            "run_utc": model_meta.get("run_utc"),
            "decision_rule_version": model_meta.get("decision_rule_version"),
            "training_label_file_used": model_meta.get("training_label_file_used"),
            "phase3_production_label_readiness": model_meta.get("phase3_production_label_readiness"),
            "split_method": model_meta.get("split_method"),
            "channel_names": model_meta.get("channel_names"),
            "class_names": model_meta.get("class_names"),
            "true_obstruction_classes": sorted(TRUE_OBSTRUCTION_CLASSES),
            "false_or_nonblocking_classes": sorted(FALSE_OR_NONBLOCKING_CLASSES),
        })
else:
    st.stop()

st.subheader("1. Upload input imagery")
st.markdown(
    """
    <div class="info-box">
    <b>Recommended and required app input:</b> upload one <b>georeferenced GeoTIFF/TIFF with CRS</b>.
    This is the most appropriate single format for this model because it preserves image pixels, map coordinates, bounds, CRS, and optional extra bands.
    A 3-band RGB GeoTIFF is accepted. A 4-band RGB+NIR GeoTIFF is preferred when available because the app can populate the NIR and NDVI channels from band 4.
    DEM is not uploaded separately in this simplified interface; DEM and slope channels are transparently zero-filled.
    </div>
    """,
    unsafe_allow_html=True,
)
image_upload = st.file_uploader(
    "Upload one georeferenced GeoTIFF/TIFF image",
    type=["tif", "tiff"],
    help="Use a single georeferenced GeoTIFF/TIFF. Band 1-3 are interpreted as RGB; band 4 is interpreted as NIR when present. PNG/JPG and separate DEM/NIR uploads are intentionally removed from this simplified app.",
)

st.caption("Best test format: georeferenced RGB or RGB+NIR GeoTIFF/TIFF. Use the sample GeoTIFF files in sample_inputs/ to test the app.")

run = st.button("Run obstruction detection", type="primary", disabled=image_upload is None)

if run and image_upload is not None:
    with st.spinner("Loading GeoTIFF/TIFF and building Phase 3 channel stack..."):
        image_meta = load_uploaded_image(image_upload)
        if image_meta.get("kind") != "geotiff":
            st.error("This simplified production-demo app accepts only GeoTIFF/TIFF input.")
            st.stop()
        if not image_meta.get("georeferenced"):
            st.error("The uploaded TIFF does not contain a usable CRS/geotransform. Please upload a georeferenced GeoTIFF/TIFF so the app can produce the interactive map and GeoJSON correctly.")
            st.stop()
        stack, stack_meta = build_phase3_stack(image_meta, dem_grid=None, nir_grid=None)
        input_format_status = "4_BAND_RGB_NIR_GEOTIFF" if int(image_meta.get("band_count", 0)) >= 4 else "3_BAND_RGB_GEOTIFF"

    hydro_features = []
    hydro_geojson = None
    hydro_bboxes = None
    hydro_status = {"status": "NOT_REQUESTED"}
    if use_official_hydrography and image_meta.get("georeferenced"):
        with st.spinner("Querying official Indiana NHD hydrography context for uploaded GeoTIFF bounds..."):
            bbox_wgs84, bbox_status = image_bounds_wgs84(image_meta)
            if bbox_wgs84 is None:
                hydro_status = {"status": bbox_status}
            else:
                hydro_features, hydro_status = query_official_nhd_flowlines(
                    bbox_wgs84=bbox_wgs84,
                    exclude_underground=exclude_underground,
                    max_records=5000,
                )
                hydro_bboxes, bbox_px_status = feature_pixel_bboxes(hydro_features, image_meta, buffer_px=int(hydro_buffer_px))
                hydro_status["pixel_bbox_status"] = bbox_px_status
                hydro_status["pixel_bbox_count"] = len(hydro_bboxes or [])
                hydro_geojson = geojson_from_nhd_features(hydro_features)

    st.subheader("2. Image metadata")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Width", image_meta["width"])
    m2.metric("Height", image_meta["height"])
    m3.metric("Bands", image_meta["band_count"])
    m4.metric("Georeferenced", "Yes" if image_meta.get("georeferenced") else "No")
    with st.expander("Preprocessing and official context details", expanded=False):
        details_payload = {
            "filename": image_meta.get("filename"),
            "image_kind": image_meta.get("kind"),
            "georeferenced": image_meta.get("georeferenced"),
            "crs": str(image_meta.get("crs")) if image_meta.get("crs") else None,
            "resolution": image_meta.get("resolution"),
            "input_format_status": input_format_status,
            "channel_stack_status": stack_meta,
            "single_input_policy": "Only one georeferenced GeoTIFF/TIFF is accepted. Band 4 is used as NIR when present; DEM/slope are zero-filled in this simplified app.",
            "official_hydrography_context_status": hydro_status,
            "channels": PHASE3_CHANNEL_NAMES,
        }
        st.json(make_json_safe(details_payload))

    if use_official_hydrography and image_meta.get("georeferenced"):
        if hydro_bboxes:
            if hydro_status.get("live_service_fallback_used"):
                st.info(f"Flowline context loaded from the packaged Phase 3 notebook official-flowline cache: {hydro_status.get('feature_count', 0)} features; {len(hydro_bboxes)} pixel corridors available. This fallback is used only when live official NHD queries return no usable features for the demo bounds.")
            else:
                st.success(f"Official surface-channel NHD context loaded from live service: {hydro_status.get('feature_count', 0)} visible/surface flowline features after filtering; {len(hydro_bboxes)} pixel corridors available for scan filtering and flowline mapping.")
        else:
            st.warning("Official hydrography context was requested, but no usable visible/surface-flowline corridor was available. The model can still scan the image internally, but the main map will not show raw tile boxes because the requested output is direct flowline mapping.")

    with st.spinner("Running tiled deep-learning inference..."):
        detections, aggregate_mask, summary, aux = run_tiled_inference(
            model=model,
            metadata=model_meta,
            stack=stack,
            image_meta=image_meta,
            tile_size=int(tile_size),
            overlap=float(overlap),
            threshold=float(threshold),
            max_tiles=int(max_tiles),
            priority_limit=priority_limit,
            hydro_pixel_bboxes=hydro_bboxes if hydro_bboxes else None,
        )

    # Build the map before showing the summary so the public-facing numbers can
    # distinguish raw model tile candidates from candidates that actually map to
    # official visible/surface flowlines. This avoids the confusing situation
    # where raw candidates exist but the flowline map correctly has zero mapped
    # obstruction spots.
    map_html, map_stats = create_flowline_decision_map(stack, image_meta, detections, hydro_geojson=hydro_geojson, return_stats=True)

    raw_possible = int(summary.get("possible_obstruction_count", 0) or 0)
    raw_review = int(summary.get("review_required_count", 0) or 0)
    mapped_candidates = int(map_stats.get("candidate_rows_assigned_to_flowline", 0) or 0)
    mapped_review = int(map_stats.get("review_spots", 0) or 0)
    mapped_yes = int(map_stats.get("final_spots", 0) or 0)
    unmapped_candidates = max(0, raw_possible - mapped_candidates)

    st.subheader("3. Prediction summary")
    s1, s2, s3, s4, s5, s6 = st.columns(6)
    s1.metric("Tiles available", summary.get("total_windows_available", 0))
    s2.metric("Tiles scanned", summary.get("scanned_windows", 0))
    s3.metric("Raw model candidates", raw_possible, help="Tile-level model candidates before official flowline assignment. These are not shown on the public map as tile boxes.")
    s4.metric("Mapped to flowlines", mapped_candidates, help="Candidates successfully assigned/snapped to official visible/surface NHD flowlines.")
    s5.metric("Final model Yes", mapped_yes)
    s6.metric("Review required", mapped_review)

    if unmapped_candidates:
        st.info(f"The model produced {raw_possible} raw tile-level candidate(s), but {unmapped_candidates} were not close enough to official visible/surface flowlines to display as public obstruction spots. They remain in the downloadable detection table for QA, but the main map only shows candidates mapped to official flowlines.")

    if summary.get("scan_was_limited") == "Yes":
        st.warning("The image had more context-filtered tiles than the configured limit. The app scanned a prioritized subset, not the entire image. The app is already using the maximum configured tile budget. Use a smaller GeoTIFF/AOI or run on stronger hardware for complete coverage if this warning appears.")
    # Non-georeferenced TIFFs are stopped earlier in this simplified app.

    st.markdown(
        f"""
        <div class="info-box">
        <b>How the scores are computed:</b><br>
        <ul>
          <li><b>obstruction_probability</b> is the sigmoid output of the model's binary obstruction head for each tile.</li>
          <li><b>top1/top2/top3 probabilities</b> are the softmax probabilities from the model's obstruction-type classification head.</li>
          <li><b>confidence_score</b> is an operational joint score: obstruction_probability × top1_probability.</li>
          <li><b>blockade_yn</b> becomes Yes only if obstruction_probability ≥ {threshold:.3f} and the top predicted class is allowed by the true-obstruction class gate. False-positive/non-obstruction classes are blocked from final Yes.</li>
          <li><b>Raw model candidates</b> are tile-level outputs from the neural network. <b>Mapped candidates</b> are the subset that can be assigned to official visible/surface NHD flowlines.</li>
          <li><b>Flowline map</b> uses official Indiana NHD flowlines for the uploaded GeoTIFF bounds. The model still runs on image tiles internally, but the public map only shows official flowline decisions and snapped candidate spots, not tile boxes.</li>
          <li>These are inference scores, not new validation accuracy/precision metrics. The app does not invent model performance.</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("4. Visual outputs")
    preview_img = stack_to_rgb_preview(stack)

    st.markdown(
        """
        <div class="info-box">
        The visual section is intentionally limited to two products: the uploaded GeoTIFF image preview and the final interactive flowline decision map. The map follows the notebook-style output: uploaded imagery as the base layer plus blue/yellow/red official flowline decisions with hover details. Candidate obstruction spots are snapped onto the nearest official flowline so they do not appear as raw tile-center grid points. Raw model tile boxes, heatmaps, and tile-debug overlays are not shown in the main visual output.
        </div>
        """,
        unsafe_allow_html=True,
    )

    v1, v2 = st.columns([1, 1.25])
    with v1:
        st.caption("Input image uploaded by the user")
        st.image(preview_img, use_container_width=True)
    with v2:
        st.caption("Output map: blue = official flowline/no final Yes, yellow = review-required candidate flowline, red = final model Yes candidate; circle spots are snapped to official flowlines")
        if map_html and hydro_geojson and hydro_geojson.get("features"):
            st.success("Interactive flowline decision map created from uploaded GeoTIFF/TIFF and official visible/surface NHD flowlines. Red/yellow spot markers are snapped onto official flowlines rather than raw tile centers.")
            with st.expander("Flowline assignment diagnostics", expanded=False):
                st.json(make_json_safe(map_stats))
            components.html(map_html, height=720, scrolling=True)
        elif map_html:
            st.warning("The uploaded image was mapped, but official NHD flowlines were not loaded for this run. The app is intentionally not showing model tile boxes. Check the official context details above.")
            components.html(map_html, height=720, scrolling=True)
        else:
            st.error("Interactive flowline map could not be created. The requested notebook-style output requires a georeferenced GeoTIFF/TIFF and folium support.")

    st.subheader("5. Detection records and downloads")
    with st.expander("Detection records table", expanded=False):
        if detections.empty:
            st.info("No tiles were scanned or no detections were produced.")
        else:
            display_cols = [
                "detection_id", "possible_obstruction_yn", "blockade_yn", "detected_obstruction_item",
                "obstruction_probability", "confidence_score", "top1_type", "top1_probability", "top2_type", "top2_probability",
                "top3_type", "top3_probability", "review_required_yn", "decision_status",
                "pixel_xmin", "pixel_ymin", "pixel_xmax", "pixel_ymax", "model_peak_pixel_x", "model_peak_pixel_y", "longitude", "latitude", "model_peak_longitude", "model_peak_latitude",
            ]
            display_cols = [c for c in display_cols if c in detections.columns]
            display_df = detections[display_cols].copy()
            try:
                st.dataframe(display_df, use_container_width=True, height=420)
            except Exception as exc:
                st.warning(f"Interactive dataframe rendering failed in this Python environment: {exc}. Showing HTML table fallback.")
                st.markdown(display_df.to_html(index=False, escape=True), unsafe_allow_html=True)

    csv_bytes = detections.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", data=csv_bytes, file_name="indiana_blockade_candidate_predictions.csv", mime="text/csv")
    geojson = detections_to_geojson(detections)
    if geojson:
        st.download_button("Download GeoJSON", data=geojson.encode("utf-8"), file_name="indiana_blockade_candidate_predictions.geojson", mime="application/geo+json")
    else:
        st.caption("GeoJSON is available only when the uploaded image is a georeferenced GeoTIFF/TIFF with CRS.")
    # Reviewer-ready CSV: same detections with blank human-review fields.
    review_df = detections.copy()
    for col in [
        "reviewed_blockade_yn", "reviewed_obstruction_type", "review_status",
        "review_confidence", "reviewer_name", "review_date", "review_notes",
        "accepted_for_training_yn", "needs_field_check_yn"
    ]:
        if col not in review_df.columns:
            review_df[col] = ""
    review_cols_front = [
        "detection_id", "blockade_yn", "possible_obstruction_yn", "detected_obstruction_item",
        "obstruction_probability", "confidence_score", "top1_type", "top1_probability",
        "top2_type", "top2_probability", "top3_type", "top3_probability", "review_required_yn",
        "longitude", "latitude", "pixel_xmin", "pixel_ymin", "pixel_xmax", "pixel_ymax",
        "reviewed_blockade_yn", "reviewed_obstruction_type", "review_status",
        "review_confidence", "reviewer_name", "review_date", "review_notes",
        "accepted_for_training_yn", "needs_field_check_yn"
    ]
    review_cols = [c for c in review_cols_front if c in review_df.columns] + [c for c in review_df.columns if c not in review_cols_front]
    st.download_button(
        "Download reviewer label CSV template",
        data=review_df[review_cols].to_csv(index=False).encode("utf-8"),
        file_name="review_labels_from_webapp_candidates.csv",
        mime="text/csv",
        help="Send this CSV to the reviewer. They fill reviewed_blockade_yn, reviewed_obstruction_type, review_status, reviewer/date, and notes."
    )
    if map_html:
        st.download_button("Download interactive HTML map", data=map_html.encode("utf-8"), file_name="indiana_blockade_candidate_interactive_map.html", mime="text/html")

    st.subheader("6. Reviewer workflow")
    st.markdown(
        """
        <div class="info-box">
        <b>How a state reviewer uses this output:</b>
        <ol>
          <li>Open the interactive map and inspect the <b>red</b> and <b>yellow</b> candidate spots/flowlines first.</li>
          <li>Click or hover each spot/flowline to see the model's obstruction probability, predicted item, top-1/top-2/top-3 classes, and location.</li>
          <li>Download the <b>reviewer label CSV template</b> and fill: reviewed_blockade_yn, reviewed_obstruction_type, review_status, reviewer_name, review_date, and notes.</li>
          <li>Accepted/confirmed labels can be added to the next model-training cycle. Rejected labels become false-positive examples. Uncertain labels remain review/field-check items.</li>
        </ol>
        <b>Important:</b> model-only candidates are not official confirmed obstruction records until agency review accepts them.
        </div>
        """,
        unsafe_allow_html=True,
    )

else:
    st.info("Upload an image, confirm the model has loaded, then click **Run obstruction detection**.")

st.divider()
st.markdown(
    """
    **Scientific integrity notes**
    - This app does not train a model and does not generate synthetic obstruction examples.
    - The app applies the uploaded Phase 3 checkpoint to uploaded imagery from any Indiana region.
    - Model-only results are candidate detections; non-inventory detections remain review-required.
    - This simplified app accepts one input format: georeferenced GeoTIFF/TIFF. This prevents fake coordinates and keeps the interactive map/GeoJSON workflow consistent.
    """
)
