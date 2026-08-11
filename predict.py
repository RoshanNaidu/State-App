"""Tiled model inference and geospatial/pixel output generation."""
from __future__ import annotations

import io
import json
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import torch

try:
    from pyproj import Transformer
    PYPROJ_AVAILABLE = True
except Exception:  # pragma: no cover
    Transformer = None
    PYPROJ_AVAILABLE = False

try:
    import rasterio
    from rasterio.transform import xy as transform_xy
    RASTERIO_AVAILABLE = True
except Exception:  # pragma: no cover
    rasterio = None
    transform_xy = None
    RASTERIO_AVAILABLE = False

from config import FALSE_OR_NONBLOCKING_CLASSES, TRUE_OBSTRUCTION_CLASSES, REVIEW_ONLY_CLASSES


def generate_windows(height: int, width: int, tile_size: int = 256, overlap: float = 0.25) -> List[Tuple[int, int, int, int]]:
    overlap = min(max(float(overlap), 0.0), 0.8)
    stride = max(1, int(round(tile_size * (1.0 - overlap))))
    ys = list(range(0, max(height - tile_size + 1, 1), stride))
    xs = list(range(0, max(width - tile_size + 1, 1), stride))
    if not ys or ys[-1] != max(height - tile_size, 0):
        ys.append(max(height - tile_size, 0))
    if not xs or xs[-1] != max(width - tile_size, 0):
        xs.append(max(width - tile_size, 0))
    windows = []
    for y in sorted(set(ys)):
        for x in sorted(set(xs)):
            windows.append((x, y, min(x + tile_size, width), min(y + tile_size, height)))
    return windows


def _phase2_priority_score(stack: np.ndarray, window: Tuple[int, int, int, int]) -> float:
    x0, y0, x1, y1 = window
    sub = stack[:, y0:y1, x0:x1]
    water = sub[5]
    brightness = sub[8]
    if water.size == 0:
        return 0.0
    h, w = water.shape
    center_band = water[int(h * 0.4): max(int(h * 0.6), int(h * 0.4)+1), :]
    water_fraction = float((water > 0.20).mean())
    center_low_water_fraction = float((center_band < 0.12).mean()) if center_band.size else 0.0
    bright_spike = float(np.percentile(brightness, 95) - np.percentile(brightness, 50)) if brightness.size else 0.0
    score = 0.45 * center_low_water_fraction + 0.30 * bright_spike + 0.25 * (1 - water_fraction)
    return float(np.clip(score, 0, 1))



def _bbox_intersects(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return not (ax1 <= bx0 or bx1 <= ax0 or ay1 <= by0 or by1 <= ay0)


def filter_windows_by_pixel_bboxes(
    windows: List[Tuple[int, int, int, int]],
    pixel_bboxes: Optional[List[Tuple[int, int, int, int]]],
) -> List[Tuple[int, int, int, int]]:
    """Keep only windows intersecting at least one official hydrography pixel bbox.

    If no bboxes are supplied, return the original windows. This prevents an
    unavailable official service from silently deleting all scan windows.
    """
    if not pixel_bboxes:
        return windows
    kept = []
    for w in windows:
        if any(_bbox_intersects(w, b) for b in pixel_bboxes):
            kept.append(w)
    return kept or windows


def select_windows(
    stack: np.ndarray,
    tile_size: int,
    overlap: float,
    max_tiles: int,
    priority_limit: bool = True,
    hydro_pixel_bboxes: Optional[List[Tuple[int, int, int, int]]] = None,
) -> Tuple[List[Tuple[int, int, int, int]], int, int]:
    height, width = int(stack.shape[1]), int(stack.shape[2])
    windows = generate_windows(height, width, tile_size=tile_size, overlap=overlap)
    total = len(windows)
    windows = filter_windows_by_pixel_bboxes(windows, hydro_pixel_bboxes)
    total_after_context_filter = len(windows)
    if max_tiles and len(windows) > max_tiles:
        if priority_limit:
            scored = [(w, _phase2_priority_score(stack, w)) for w in windows]
            scored.sort(key=lambda x: x[1], reverse=True)
            windows = [w for w, _ in scored[:max_tiles]]
        else:
            windows = windows[:max_tiles]
    return windows, total, total_after_context_filter


def _crop_and_pad(stack: np.ndarray, window: Tuple[int, int, int, int], tile_size: int) -> np.ndarray:
    x0, y0, x1, y1 = window
    sub = stack[:, y0:y1, x0:x1]
    out = np.zeros((stack.shape[0], tile_size, tile_size), dtype="float32")
    out[:, : sub.shape[1], : sub.shape[2]] = sub.astype("float32")
    return out


def _resize_mask(mask: np.ndarray, out_size: Tuple[int, int]) -> np.ndarray:
    mask = np.clip(mask, 0, 1).astype("float32")
    im = Image.fromarray(mask, mode="F")
    im = im.resize((out_size[0], out_size[1]), resample=Image.BILINEAR)
    return np.asarray(im, dtype="float32")


def _pixel_bbox_to_geojson_polygon(image_meta: Dict[str, Any], bbox: Tuple[int, int, int, int]) -> Tuple[Optional[Dict[str, Any]], Optional[float], Optional[float], Optional[str]]:
    if not image_meta.get("georeferenced") or not RASTERIO_AVAILABLE:
        return None, None, None, None
    transform = image_meta.get("transform")
    crs = image_meta.get("crs")
    if transform is None or crs is None:
        return None, None, None, None
    x0, y0, x1, y1 = bbox
    # rasterio xy uses row, col. Use UL/LR corners via offset.
    coords_src = [
        transform_xy(transform, y0, x0, offset="ul"),
        transform_xy(transform, y0, x1, offset="ul"),
        transform_xy(transform, y1, x1, offset="ul"),
        transform_xy(transform, y1, x0, offset="ul"),
        transform_xy(transform, y0, x0, offset="ul"),
    ]
    centroid_src = transform_xy(transform, (y0 + y1) / 2.0, (x0 + x1) / 2.0)
    if str(crs).upper() not in {"EPSG:4326", "OGC:CRS84"}:
        if not PYPROJ_AVAILABLE:
            return None, None, None, "pyproj unavailable; cannot transform to EPSG:4326"
        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        coords_ll = [transformer.transform(float(x), float(y)) for x, y in coords_src]
        lon, lat = transformer.transform(float(centroid_src[0]), float(centroid_src[1]))
    else:
        coords_ll = [(float(x), float(y)) for x, y in coords_src]
        lon, lat = float(centroid_src[0]), float(centroid_src[1])
    geom = {"type": "Polygon", "coordinates": [[list(c) for c in coords_ll]]}
    return geom, float(lon), float(lat), None


def _pixel_point_to_lonlat(image_meta: Dict[str, Any], x: float, y: float) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    """Convert a pixel coordinate to EPSG:4326 lon/lat.

    This is used only to place a candidate spot at the model mask's strongest
    pixel inside a tile. It does not alter the tile-level prediction itself.
    """
    if not image_meta.get("georeferenced") or not RASTERIO_AVAILABLE:
        return None, None, None
    transform = image_meta.get("transform")
    crs = image_meta.get("crs")
    if transform is None or crs is None:
        return None, None, None
    try:
        sx, sy = transform_xy(transform, float(y), float(x), offset="center")
        if str(crs).upper() not in {"EPSG:4326", "OGC:CRS84"}:
            if not PYPROJ_AVAILABLE:
                return None, None, "pyproj unavailable; cannot transform point to EPSG:4326"
            transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
            lon, lat = transformer.transform(float(sx), float(sy))
        else:
            lon, lat = float(sx), float(sy)
        return float(lon), float(lat), None
    except Exception as exc:
        return None, None, f"point transform failed: {exc}"


def apply_decision_gate(obstruction_probability: float, top1_type: str, threshold: float) -> Dict[str, str]:
    probability_gate = obstruction_probability >= threshold
    top_is_true = top1_type in TRUE_OBSTRUCTION_CLASSES
    top_is_false = top1_type in FALSE_OR_NONBLOCKING_CLASSES
    if probability_gate and top_is_true:
        return {
            "possible_obstruction_yn": "Yes",
            "blockade_yn": "Yes",
            "review_required_yn": "Yes",
            "production_publishable_yn": "No - model-only candidate requires human/agency review",
            "decision_status": "model_candidate_review_required",
            "decision_reason": "Obstruction probability is above threshold and top class is a true-obstruction class.",
            "predicted_class_is_true_obstruction": "Yes",
            "predicted_class_is_false_positive": "No",
        }
    if probability_gate and top_is_false:
        return {
            "possible_obstruction_yn": "Yes",
            "blockade_yn": "No",
            "review_required_yn": "Yes",
            "production_publishable_yn": "No - high model probability conflicts with false/non-blocking class",
            "decision_status": "model_probability_type_conflict_review_required",
            "decision_reason": "Obstruction probability is above threshold, but top class is false-positive/non-obstruction, so final Yes is blocked.",
            "predicted_class_is_true_obstruction": "No",
            "predicted_class_is_false_positive": "Yes",
        }
    if (not probability_gate) and top_is_true:
        return {
            "possible_obstruction_yn": "No",
            "blockade_yn": "No",
            "review_required_yn": "Yes" if top1_type in REVIEW_ONLY_CLASSES else "No",
            "production_publishable_yn": "No - probability below threshold",
            "decision_status": "below_threshold_obstruction_like_class",
            "decision_reason": "Top class is obstruction-like, but obstruction probability is below threshold.",
            "predicted_class_is_true_obstruction": "Yes",
            "predicted_class_is_false_positive": "No",
        }
    return {
        "possible_obstruction_yn": "No",
        "blockade_yn": "No",
        "review_required_yn": "No",
        "production_publishable_yn": "No - model-only image scan is not official confirmation",
        "decision_status": "model_no_candidate",
        "decision_reason": "Probability below threshold or top class is non-obstruction/false-positive.",
        "predicted_class_is_true_obstruction": "No",
        "predicted_class_is_false_positive": "Yes" if top_is_false else "No",
    }


def run_tiled_inference(
    model: torch.nn.Module,
    metadata: Dict[str, Any],
    stack: np.ndarray,
    image_meta: Dict[str, Any],
    tile_size: int = 256,
    overlap: float = 0.25,
    threshold: Optional[float] = None,
    max_tiles: int = 150,
    priority_limit: bool = True,
    hydro_pixel_bboxes: Optional[List[Tuple[int, int, int, int]]] = None,
) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, Any], Dict[str, Any]]:
    device = next(model.parameters()).device
    class_names = list(metadata["class_names"])
    mean = np.asarray(metadata["channel_mean"], dtype="float32")
    std = np.asarray(metadata["channel_std"], dtype="float32")
    threshold = float(metadata.get("obstruction_threshold", 0.5) if threshold is None else threshold)
    windows, total_windows, total_after_context_filter = select_windows(
        stack,
        tile_size=tile_size,
        overlap=overlap,
        max_tiles=max_tiles,
        priority_limit=priority_limit,
        hydro_pixel_bboxes=hydro_pixel_bboxes,
    )
    if not windows:
        return pd.DataFrame(), np.zeros((stack.shape[1], stack.shape[2]), dtype="float32"), {"total_windows_available": 0, "windows_after_context_filter": 0, "scanned_windows": 0}, {}

    rows: List[Dict[str, Any]] = []
    aggregate_mask = np.zeros((stack.shape[1], stack.shape[2]), dtype="float32")
    model.eval()
    with torch.no_grad():
        for idx, w in enumerate(windows):
            chip = _crop_and_pad(stack, w, tile_size=tile_size)
            x = (chip - mean[:, None, None]) / np.maximum(std[:, None, None], 1e-6)
            xt = torch.tensor(x[None, ...], dtype=torch.float32, device=device)
            out = model(xt)
            class_probs = torch.softmax(out["class_logits"], dim=1).cpu().numpy()[0]
            obs_prob = float(torch.sigmoid(out["obstruction_logit"]).cpu().numpy()[0])
            mask_prob = torch.sigmoid(out["mask_logits"]).cpu().numpy()[0, 0]
            # The classifier decision is tile-level, but the public map should
            # not place candidate markers at tile centroids. Use the strongest
            # mask pixel as a better candidate point before snapping to official
            # flowlines in the map layer.
            try:
                peak_y_rel, peak_x_rel = np.unravel_index(int(np.nanargmax(mask_prob)), mask_prob.shape)
            except Exception:
                peak_y_rel, peak_x_rel = mask_prob.shape[0] // 2, mask_prob.shape[1] // 2
            order = np.argsort(class_probs)[::-1]
            top1, top2, top3 = [int(i) for i in order[:3]]
            top1_type = class_names[top1]
            top2_type = class_names[top2]
            top3_type = class_names[top3]
            gate = apply_decision_gate(obs_prob, top1_type, threshold)
            x0, y0, x1, y1 = w
            # Convert mask-peak location from tile pixels to full-image pixels.
            peak_x = float(x0 + min(max(peak_x_rel, 0), tile_size - 1) * max((x1 - x0), 1) / float(tile_size))
            peak_y = float(y0 + min(max(peak_y_rel, 0), tile_size - 1) * max((y1 - y0), 1) / float(tile_size))
            geom, lon, lat, geo_error = _pixel_bbox_to_geojson_polygon(image_meta, w)
            peak_lon, peak_lat, peak_geo_error = _pixel_point_to_lonlat(image_meta, peak_x, peak_y)
            confidence_score = float(obs_prob * class_probs[top1])
            row = {
                "detection_id": f"det_{idx:05d}",
                "tile_id": f"tile_{idx:05d}",
                "possible_obstruction_yn": gate["possible_obstruction_yn"],
                "blockade_yn": gate["blockade_yn"],
                "detected_obstruction_item": top1_type,
                "obstruction_probability": obs_prob,
                "confidence_score": confidence_score,
                "top1_type": top1_type,
                "top1_probability": float(class_probs[top1]),
                "top2_type": top2_type,
                "top2_probability": float(class_probs[top2]),
                "top3_type": top3_type,
                "top3_probability": float(class_probs[top3]),
                "review_required_yn": gate["review_required_yn"],
                "production_publishable_yn": gate["production_publishable_yn"],
                "decision_status": gate["decision_status"],
                "decision_reason": gate["decision_reason"],
                "predicted_class_is_true_obstruction": gate["predicted_class_is_true_obstruction"],
                "predicted_class_is_false_positive": gate["predicted_class_is_false_positive"],
                "model_version": metadata.get("decision_rule_version", "unknown"),
                "threshold_used": threshold,
                "image_mode": "georeferenced_geotiff" if image_meta.get("georeferenced") else "pixel_only_image",
                "location_type": "geospatial_bbox" if geom is not None else "pixel_bbox_only",
                "pixel_xmin": int(x0),
                "pixel_ymin": int(y0),
                "pixel_xmax": int(x1),
                "pixel_ymax": int(y1),
                "pixel_centroid_x": float((x0 + x1) / 2.0),
                "pixel_centroid_y": float((y0 + y1) / 2.0),
                "model_peak_pixel_x": float(peak_x),
                "model_peak_pixel_y": float(peak_y),
                "longitude": lon,
                "latitude": lat,
                "model_peak_longitude": peak_lon,
                "model_peak_latitude": peak_lat,
                "geometry_json": json.dumps(geom) if geom else "",
                "geometry_error": geo_error or peak_geo_error or "",
                "notes": "Model-generated candidate only; human review required for non-inventory official acceptance.",
            }
            for c, p in zip(class_names, class_probs):
                row[f"prob_{c}"] = float(p)
            rows.append(row)
            # Aggregate mask in native pixel space.
            mh = max(1, y1 - y0)
            mw = max(1, x1 - x0)
            resized = _resize_mask(mask_prob, (mw, mh))
            aggregate_mask[y0:y1, x0:x1] = np.maximum(aggregate_mask[y0:y1, x0:x1], resized[:mh, :mw])
    df = pd.DataFrame(rows)
    summary = {
        "total_windows_available": int(total_windows),
        "windows_after_context_filter": int(total_after_context_filter),
        "official_hydrography_filter_applied": "Yes" if hydro_pixel_bboxes else "No",
        "scanned_windows": int(len(windows)),
        "scan_was_limited": "Yes" if len(windows) < total_after_context_filter else "No",
        "threshold_used": float(threshold),
        "blockade_yes_count": int((df["blockade_yn"] == "Yes").sum()) if len(df) else 0,
        "possible_obstruction_count": int((df["possible_obstruction_yn"] == "Yes").sum()) if len(df) else 0,
        "review_required_count": int((df["review_required_yn"] == "Yes").sum()) if len(df) else 0,
        "max_obstruction_probability": float(df["obstruction_probability"].max()) if len(df) else np.nan,
        "mean_obstruction_probability": float(df["obstruction_probability"].mean()) if len(df) else np.nan,
    }
    return df, aggregate_mask, summary, {"windows": windows}


def render_overlay(stack: np.ndarray, detections: pd.DataFrame, threshold: float, max_boxes: int = 50) -> Image.Image:
    rgb = np.clip(stack[:3].transpose(1, 2, 0), 0, 1)
    base = Image.fromarray((rgb * 255).astype("uint8")).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    if detections is not None and len(detections):
        show = detections.sort_values("obstruction_probability", ascending=False).head(max_boxes)
        for _, r in show.iterrows():
            x0, y0, x1, y1 = [int(r[c]) for c in ["pixel_xmin", "pixel_ymin", "pixel_xmax", "pixel_ymax"]]
            if r.get("blockade_yn") == "Yes":
                color = (220, 30, 30, 230)
                label = f"YES {r['obstruction_probability']:.2f}"
            elif r.get("possible_obstruction_yn") == "Yes":
                color = (255, 150, 0, 220)
                label = f"REVIEW {r['obstruction_probability']:.2f}"
            else:
                continue
            width = 4 if r.get("blockade_yn") == "Yes" else 3
            for i in range(width):
                draw.rectangle([x0+i, y0+i, x1-i, y1-i], outline=color)
            draw.rectangle([x0, max(0, y0-18), min(x0 + 155, base.size[0]), y0], fill=(0, 0, 0, 160))
            draw.text((x0 + 3, max(0, y0 - 16)), label, fill=(255, 255, 255, 255))
    composed = Image.alpha_composite(base, overlay).convert("RGB")
    return composed


def render_mask_heatmap(mask: np.ndarray) -> Image.Image:
    """Render a relative model mask/probability heatmap.

    The raw segmentation head can be nearly uniform on out-of-distribution or
    RGB-only imagery. To avoid a visually useless solid red sheet, this display
    uses robust percentile normalization. This is a visualization of relative
    model attention, not a calibrated probability surface.
    """
    m = np.asarray(mask, dtype="float32")
    finite = np.isfinite(m)
    if not finite.any():
        z = np.zeros_like(m, dtype="uint8")
        return Image.fromarray(np.stack([z, z, z], axis=2), mode="RGB")
    vals = m[finite]
    lo, hi = np.percentile(vals, [5, 99])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(vals.min()), float(vals.max())
    if hi <= lo:
        scaled = np.zeros_like(m, dtype="float32")
    else:
        scaled = np.clip((m - lo) / (hi - lo), 0, 1)
    # Black -> purple/red -> yellow high-intensity ramp.
    r = (np.clip(scaled * 1.6, 0, 1) * 255).astype("uint8")
    g = (np.clip((scaled - 0.45) * 1.8, 0, 1) * 220).astype("uint8")
    b = (np.clip(0.35 - scaled * 0.35, 0, 0.35) * 255).astype("uint8")
    return Image.fromarray(np.stack([r, g, b], axis=2), mode="RGB")


def pil_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    bio = io.BytesIO()
    img.save(bio, format=fmt)
    return bio.getvalue()


def detections_to_geojson(df: pd.DataFrame) -> Optional[str]:
    if df is None or df.empty or "geometry_json" not in df.columns:
        return None
    features = []
    for _, r in df.iterrows():
        if not r.get("geometry_json"):
            continue
        try:
            geom = json.loads(r["geometry_json"])
        except Exception:
            continue
        props = {k: (None if pd.isna(v) else v) for k, v in r.drop(labels=["geometry_json"], errors="ignore").to_dict().items()}
        features.append({"type": "Feature", "geometry": geom, "properties": props})
    if not features:
        return None
    return json.dumps({"type": "FeatureCollection", "features": features}, indent=2)
