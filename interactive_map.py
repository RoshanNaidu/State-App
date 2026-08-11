"""Notebook-style Folium map for the Streamlit app.

Main display policy:
- show the uploaded GeoTIFF image
- show OFFICIAL hydrography flowlines colored by model decision status
- do NOT display large model tile boxes as the main map output
- do NOT place candidate spots at raw tile centroids

The model itself still runs on tiles, but tile windows are only the internal
inference mechanism. This module translates tile-level model candidates back to
nearest/intersecting official flowlines. Public markers are snapped to official
flowlines, which avoids the grid pattern created by tile-centroid markers.
"""
from __future__ import annotations

import base64
import io
import json
import math
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from PIL import Image

try:
    import folium
    FOLIUM_AVAILABLE = True
except Exception:  # pragma: no cover
    folium = None
    FOLIUM_AVAILABLE = False

try:
    from pyproj import Transformer
except Exception:  # pragma: no cover
    Transformer = None

try:
    from shapely.geometry import Point, shape
    from shapely.ops import transform as shapely_transform
    SHAPELY_AVAILABLE = True
except Exception:  # pragma: no cover
    Point = None
    shape = None
    shapely_transform = None
    SHAPELY_AVAILABLE = False

from preprocess import stack_to_rgb_preview

# Maximum distance allowed when snapping a tile-level candidate to an official
# flowline. Candidates farther away are omitted from the public flowline map;
# they remain in the downloadable detection table for QA.
MAX_SNAP_DISTANCE_M = 250.0


def _image_to_data_url(img: Image.Image) -> str:
    bio = io.BytesIO()
    img.save(bio, format="PNG")
    return "data:image/png;base64," + base64.b64encode(bio.getvalue()).decode("ascii")


def _bounds_to_wgs84(image_meta: Dict[str, Any]):
    if not image_meta.get("georeferenced") or image_meta.get("bounds") is None or image_meta.get("crs") is None:
        return None
    b = image_meta["bounds"]
    crs_text = str(image_meta.get("crs", "")).upper()
    if "4326" in crs_text or "CRS84" in crs_text:
        return [[float(b.bottom), float(b.left)], [float(b.top), float(b.right)]]
    if Transformer is None:
        return None
    try:
        t = Transformer.from_crs(image_meta["crs"], "EPSG:4326", always_xy=True)
        pts = [
            t.transform(b.left, b.bottom),
            t.transform(b.right, b.top),
            t.transform(b.left, b.top),
            t.transform(b.right, b.bottom),
        ]
        lons = [p[0] for p in pts]
        lats = [p[1] for p in pts]
        return [[float(min(lats)), float(min(lons))], [float(max(lats)), float(max(lons))]]
    except Exception:
        return None


def _finite_value(v):
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    if isinstance(v, float):
        if not math.isfinite(v):
            return None
        return round(v, 6)
    return v


def _safe_float(value):
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _is_candidate_row(r: pd.Series) -> bool:
    return (
        str(r.get("blockade_yn", "")).lower() == "yes"
        or str(r.get("review_required_yn", "")).lower() == "yes"
        or str(r.get("possible_obstruction_yn", "")).lower() == "yes"
    )


def _popup_table_html(props: dict, title: str = "Candidate snapped to official flowline") -> str:
    keep = [
        "detection_id", "assigned_flowline_index", "snap_distance_m", "snap_method",
        "blockade_yn", "possible_obstruction_yn", "detected_obstruction_item",
        "obstruction_probability", "confidence_score", "top1_type", "top1_probability",
        "top2_type", "top2_probability", "top3_type", "top3_probability",
        "review_required_yn", "decision_status", "decision_reason", "longitude", "latitude",
    ]
    rows = []
    for k in keep:
        if k not in props:
            continue
        v = _finite_value(props.get(k))
        rows.append(f"<tr><th style='text-align:left;padding:3px 8px 3px 0;'>{k}</th><td style='padding:3px 0;'>{v}</td></tr>")
    return f"<div style='font-size:12px;'><b>{title}</b><table>" + "".join(rows) + "</table></div>"


def _project_to_3857(geom):
    if not SHAPELY_AVAILABLE or Transformer is None or geom is None:
        return None
    try:
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        return shapely_transform(lambda x, y, z=None: transformer.transform(x, y), geom)
    except Exception:
        return None


def _project_from_3857(geom):
    if not SHAPELY_AVAILABLE or Transformer is None or geom is None:
        return None
    try:
        transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
        return shapely_transform(lambda x, y, z=None: transformer.transform(x, y), geom)
    except Exception:
        return None


def _make_flowline_geometries(hydro_geojson: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not SHAPELY_AVAILABLE or not hydro_geojson or not hydro_geojson.get("features"):
        return out
    for idx, feat in enumerate(hydro_geojson.get("features", [])):
        geom_obj = feat.get("geometry")
        if not geom_obj:
            continue
        try:
            g = shape(geom_obj)
            if g.is_empty:
                continue
            gp = _project_to_3857(g)
            if gp is None or gp.is_empty:
                continue
            out.append({"index": idx, "feature": feat, "geom_wgs84": g, "geom_3857": gp})
        except Exception:
            continue
    return out


def _detection_geometry_and_centroid(r: pd.Series):
    if not SHAPELY_AVAILABLE:
        return None, None, None
    geom = None
    geom_txt = r.get("geometry_json")
    if geom_txt:
        try:
            geom = shape(json.loads(geom_txt))
        except Exception:
            geom = None
    lon = _safe_float(r.get("longitude"))
    lat = _safe_float(r.get("latitude"))
    if lon is not None and lat is not None:
        centroid = Point(lon, lat)
    elif geom is not None:
        centroid = geom.centroid
    else:
        centroid = None
    geom_3857 = _project_to_3857(geom) if geom is not None else None
    return geom, centroid, geom_3857


def _priority_score(props: Dict[str, Any]) -> float:
    prob = _safe_float(props.get("obstruction_probability")) or 0.0
    score = prob
    if str(props.get("blockade_yn", "")).lower() == "yes":
        score += 100.0
    elif str(props.get("review_required_yn", "")).lower() == "yes" or str(props.get("possible_obstruction_yn", "")).lower() == "yes":
        score += 10.0
    return score


def assign_candidates_to_flowlines(
    hydro_geojson: Dict[str, Any],
    detections: pd.DataFrame,
    max_snap_distance_m: float = MAX_SNAP_DISTANCE_M,
) -> Tuple[Dict[int, Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Snap candidate detections to official flowlines.

    The model predicts by image tile. This function avoids showing tile centers
    by assigning each candidate tile to a real official flowline only when the
    tile intersects a flowline or is near one. The public map then shows the
    snapped point and highlighted flowline, not the tile box/grid.
    """
    flowlines = _make_flowline_geometries(hydro_geojson)
    diagnostics = {
        "flowline_count": len(flowlines),
        "candidate_rows_considered": 0,
        "candidate_rows_assigned_to_flowline": 0,
        "candidate_rows_not_mapped_no_near_flowline": 0,
        "max_snap_distance_m": float(max_snap_distance_m),
    }
    assignments_by_flowline: Dict[int, Dict[str, Any]] = {}
    spot_records: List[Dict[str, Any]] = []
    if not flowlines or detections is None or not len(detections) or not SHAPELY_AVAILABLE:
        return assignments_by_flowline, spot_records, diagnostics

    for _, r in detections.iterrows():
        if not _is_candidate_row(r):
            continue
        diagnostics["candidate_rows_considered"] += 1
        _, centroid_wgs84, tile_geom_3857 = _detection_geometry_and_centroid(r)
        if centroid_wgs84 is None:
            diagnostics["candidate_rows_not_mapped_no_near_flowline"] += 1
            continue
        centroid_3857 = _project_to_3857(centroid_wgs84)
        if centroid_3857 is None:
            diagnostics["candidate_rows_not_mapped_no_near_flowline"] += 1
            continue

        best = None
        best_sort_score = None
        for fl in flowlines:
            line_3857 = fl["geom_3857"]
            intersects_tile = False
            if tile_geom_3857 is not None:
                try:
                    intersects_tile = bool(line_3857.intersects(tile_geom_3857))
                except Exception:
                    intersects_tile = False
            try:
                dist_m = float(line_3857.distance(centroid_3857))
            except Exception:
                continue
            if not intersects_tile and dist_m > max_snap_distance_m:
                continue
            sort_score = (0.0 if intersects_tile else 1_000_000.0) + dist_m
            if best is None or sort_score < best_sort_score:
                best = (fl, dist_m, intersects_tile)
                best_sort_score = sort_score
        if best is None:
            diagnostics["candidate_rows_not_mapped_no_near_flowline"] += 1
            continue

        fl, dist_m, intersects_tile = best
        try:
            snapped_3857 = fl["geom_3857"].interpolate(fl["geom_3857"].project(centroid_3857))
            snapped_wgs84 = _project_from_3857(snapped_3857)
            if snapped_wgs84 is None:
                diagnostics["candidate_rows_not_mapped_no_near_flowline"] += 1
                continue
            lon = float(snapped_wgs84.x)
            lat = float(snapped_wgs84.y)
        except Exception:
            diagnostics["candidate_rows_not_mapped_no_near_flowline"] += 1
            continue

        props = {k: _finite_value(v) for k, v in r.to_dict().items() if k != "geometry_json"}
        props.update({
            "assigned_flowline_index": int(fl["index"]),
            "snap_distance_m": round(float(dist_m), 2),
            "snapped_to_official_flowline": "Yes",
            "snap_method": "tile_intersects_flowline" if intersects_tile else "nearest_flowline_within_threshold",
            "longitude": lon,
            "latitude": lat,
        })
        spot = {
            "flowline_index": int(fl["index"]),
            "lat": lat,
            "lon": lon,
            "props": props,
            "score": _priority_score(props),
        }
        spot_records.append(spot)
        diagnostics["candidate_rows_assigned_to_flowline"] += 1
        prev = assignments_by_flowline.get(int(fl["index"]))
        if prev is None or spot["score"] > prev["score"]:
            assignments_by_flowline[int(fl["index"])] = spot
    return assignments_by_flowline, spot_records, diagnostics


def enrich_hydro_geojson_with_assignments(hydro_geojson: Dict[str, Any], assignments_by_flowline: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    if not hydro_geojson or not hydro_geojson.get("features"):
        return {"type": "FeatureCollection", "features": []}
    out = []
    for idx, feat in enumerate(hydro_geojson.get("features", [])):
        props = dict(feat.get("properties", {}) or {})
        geom = feat.get("geometry")
        assign = assignments_by_flowline.get(idx)
        status = "Official flowline - No final model Yes"
        color_class = "blue_no_candidate"
        if assign:
            det_props = assign["props"]
            if str(det_props.get("blockade_yn", "")).lower() == "yes":
                status = "Final model Yes candidate"
                color_class = "red_model_yes"
            elif str(det_props.get("review_required_yn", "")).lower() == "yes" or str(det_props.get("possible_obstruction_yn", "")).lower() == "yes":
                status = "Review-required candidate"
                color_class = "yellow_review_candidate"
            props.update({
                "model_status": status,
                "map_color_class": color_class,
                "detection_id": det_props.get("detection_id"),
                "blockade_yn": det_props.get("blockade_yn"),
                "detected_obstruction_item": det_props.get("detected_obstruction_item"),
                "obstruction_probability": det_props.get("obstruction_probability"),
                "confidence_score": det_props.get("confidence_score"),
                "top1_type": det_props.get("top1_type"),
                "top1_probability": det_props.get("top1_probability"),
                "top2_type": det_props.get("top2_type"),
                "top2_probability": det_props.get("top2_probability"),
                "top3_type": det_props.get("top3_type"),
                "top3_probability": det_props.get("top3_probability"),
                "review_required_yn": det_props.get("review_required_yn"),
                "decision_status": det_props.get("decision_status"),
                "latitude": det_props.get("latitude"),
                "longitude": det_props.get("longitude"),
                "snap_distance_m": det_props.get("snap_distance_m"),
                "snap_method": det_props.get("snap_method"),
            })
        else:
            props.update({"model_status": status, "map_color_class": color_class, "blockade_yn": "No", "review_required_yn": "No"})
        # Folium GeoJsonTooltip expects selected fields to exist on every feature.
        # Populate missing model fields with None so blue/no-candidate flowlines
        # can share the same tooltip schema as yellow/red assigned flowlines.
        for _k in [
            "detection_id", "detected_obstruction_item", "obstruction_probability",
            "confidence_score", "top1_type", "top1_probability", "top2_type",
            "top2_probability", "top3_type", "top3_probability", "decision_status",
            "snap_distance_m", "snap_method", "latitude", "longitude",
        ]:
            props.setdefault(_k, None)
        out.append({"type": "Feature", "geometry": geom, "properties": props})
    return {"type": "FeatureCollection", "features": out}


def add_snapped_candidate_spot_markers(m, spot_records: List[Dict[str, Any]]) -> Tuple[int, int]:
    if not FOLIUM_AVAILABLE or not spot_records:
        return 0, 0
    final_group = folium.FeatureGroup(name="Red obstruction spots - final model Yes candidates", show=True)
    review_group = folium.FeatureGroup(name="Yellow obstruction spots - review-required candidates", show=True)
    final_count = 0
    review_count = 0
    for spot in spot_records:
        props = dict(spot.get("props", {}) or {})
        lat = float(spot["lat"])
        lon = float(spot["lon"])
        is_yes = str(props.get("blockade_yn", "")).lower() == "yes"
        prob = _safe_float(props.get("obstruction_probability")) or 0.0
        item = str(props.get("detected_obstruction_item") or "unknown")
        if is_yes:
            color = "#e60000"; fill_color = "#e60000"; radius = 8; weight = 3
            label = f"YES {prob:.2f} - {item}"; target = final_group; final_count += 1
        else:
            color = "#ff9900"; fill_color = "#ffcc00"; radius = 6; weight = 2
            label = f"REVIEW {prob:.2f} - {item}"; target = review_group; review_count += 1
        folium.CircleMarker(
            location=[lat, lon],
            radius=radius,
            color=color,
            fill=True,
            fill_color=fill_color,
            fill_opacity=0.9,
            weight=weight,
            tooltip=label,
            popup=folium.Popup(_popup_table_html(props), max_width=500),
        ).add_to(target)
    if final_count:
        final_group.add_to(m)
    if review_count:
        review_group.add_to(m)
    return final_count, review_count




def get_flowline_assignment_summary(
    hydro_geojson: Optional[Dict[str, Any]],
    detections: pd.DataFrame,
    image_meta: Dict[str, Any],
) -> Dict[str, Any]:
    """Return candidate-to-flowline assignment counts for app summary display.

    Raw model candidate tiles are not the same as public obstruction spots.
    This helper counts only candidates that can be assigned to official flowline
    geometry; those are the detections that appear on the notebook-style map.
    """
    bounds = _bounds_to_wgs84(image_meta)
    if not bounds or not hydro_geojson or not hydro_geojson.get("features"):
        return {
            "candidate_rows": int((detections.apply(_is_candidate_row, axis=1)).sum()) if detections is not None and len(detections) else 0,
            "assigned_candidates": 0,
            "unassigned_candidates": int((detections.apply(_is_candidate_row, axis=1)).sum()) if detections is not None and len(detections) else 0,
            "flowline_review_candidates": 0,
            "flowline_final_yes_candidates": 0,
            "flowlines_with_candidate_status": 0,
            "assignment_method": "NO_FLOWLINES_OR_BOUNDS",
        }
    enriched, spot_records, diag = assign_candidates_to_flowlines(hydro_geojson, detections, bounds, image_meta)
    final_yes = 0
    review = 0
    for rec in spot_records:
        props = rec.get("props", {}) or {}
        if str(props.get("blockade_yn", "")).lower() == "yes":
            final_yes += 1
        elif str(props.get("review_required_yn", "")).lower() == "yes" or str(props.get("possible_obstruction_yn", "")).lower() == "yes":
            review += 1
    out = dict(diag or {})
    out["flowline_review_candidates"] = int(review)
    out["flowline_final_yes_candidates"] = int(final_yes)
    return out


def create_flowline_decision_map(stack, image_meta: Dict[str, Any], detections: pd.DataFrame, hydro_geojson: Optional[Dict[str, Any]] = None, return_stats: bool = False):
    stats = {"map_created": False, "hydro_feature_count": 0, "candidate_rows_considered": 0, "candidate_rows_assigned_to_flowline": 0, "candidate_rows_not_mapped_no_near_flowline": 0, "final_spots": 0, "review_spots": 0}
    if not FOLIUM_AVAILABLE or not image_meta.get("georeferenced"):
        return (None, stats) if return_stats else None
    bounds = _bounds_to_wgs84(image_meta)
    if not bounds:
        return (None, stats) if return_stats else None
    center = [(bounds[0][0] + bounds[1][0]) / 2.0, (bounds[0][1] + bounds[1][1]) / 2.0]
    m = folium.Map(location=center, zoom_start=13, tiles=None, control_scale=True)

    rgb = stack_to_rgb_preview(stack, max_size=1600)
    folium.raster_layers.ImageOverlay(
        image=_image_to_data_url(rgb),
        bounds=bounds,
        opacity=1.0,
        name="Uploaded GeoTIFF imagery",
        interactive=True,
        cross_origin=False,
        zindex=1,
        show=True,
    ).add_to(m)

    final_spots = 0
    review_spots = 0
    stats["hydro_feature_count"] = len((hydro_geojson or {}).get("features", []) or [])
    if hydro_geojson and hydro_geojson.get("features"):
        assignments_by_flowline, spot_records, snap_diag = assign_candidates_to_flowlines(hydro_geojson, detections)
        stats.update({k: snap_diag.get(k, stats.get(k, 0)) for k in ["candidate_rows_considered", "candidate_rows_assigned_to_flowline", "candidate_rows_not_mapped_no_near_flowline"]})
        enriched = enrich_hydro_geojson_with_assignments(hydro_geojson, assignments_by_flowline)

        def hydro_style(feature):
            p = feature.get("properties", {}) or {}
            cls = p.get("map_color_class")
            if cls == "red_model_yes":
                return {"color": "#e60000", "weight": 6, "opacity": 1.0}
            if cls == "yellow_review_candidate":
                return {"color": "#ffcc00", "weight": 5, "opacity": 1.0}
            return {"color": "#1565c0", "weight": 2, "opacity": 0.65}

        tooltip_fields = [
            "gnis_name", "GNIS_NAME", "fcode", "FCode", "ftype", "FType",
            "model_status", "blockade_yn", "detected_obstruction_item",
            "obstruction_probability", "confidence_score", "top1_type", "top1_probability",
            "top2_type", "top2_probability", "top3_type", "top3_probability",
            "review_required_yn", "decision_status", "snap_distance_m", "snap_method", "latitude", "longitude",
        ]
        available = set()
        for feat in enriched.get("features", []):
            available.update((feat.get("properties", {}) or {}).keys())
        fields = [f for f in tooltip_fields if f in available]
        aliases = [f.replace("_", " ").title() for f in fields]
        folium.GeoJson(
            enriched,
            name="Final blockade decision flowlines",
            style_function=hydro_style,
            tooltip=folium.GeoJsonTooltip(fields=fields, aliases=aliases, localize=True, sticky=True) if fields else None,
            highlight_function=lambda feature: {"weight": 8, "opacity": 1.0},
        ).add_to(m)

        final_spots, review_spots = add_snapped_candidate_spot_markers(m, spot_records)
        stats["final_spots"] = int(final_spots)
        stats["review_spots"] = int(review_spots)

        legend_html = """
        <div style="position: fixed; bottom: 28px; left: 28px; z-index: 9999; background: white; padding: 10px 12px; border: 1px solid #777; border-radius: 6px; font-size: 13px;">
          <b>Flowline decision map</b><br>
          <span style="color:#1565c0;font-weight:700;">━━</span> Official flowline / No final model Yes<br>
          <span style="color:#ffcc00;font-weight:700;">━━</span> Review-required candidate assigned to flowline<br>
          <span style="color:#e60000;font-weight:700;">━━</span> Final model Yes candidate assigned to flowline<br>
          <small>Spots are snapped to official flowlines, not raw tile centers.</small>
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))

        if snap_diag.get("candidate_rows_considered", 0) and not (final_spots or review_spots):
            notice_html = f"""
            <div style="position: fixed; top: 18px; left: 50px; z-index: 9999; background: #fff3cd; padding: 10px 12px; border: 1px solid #cc9a06; border-radius: 6px; font-size: 13px; max-width: 560px;">
              <b>Model candidates were produced, but none were close enough to official flowlines to map as obstruction spots.</b><br>
              Candidate rows considered: {snap_diag.get('candidate_rows_considered', 0)}; maximum snap distance: {MAX_SNAP_DISTANCE_M:.0f} m.
              The app does not place off-channel tile centers on the public map.
            </div>
            """
            m.get_root().html.add_child(folium.Element(notice_html))
    else:
        msg_html = """
        <div style="position: fixed; top: 18px; left: 50px; z-index: 9999; background: #fff3cd; padding: 10px 12px; border: 1px solid #cc9a06; border-radius: 6px; font-size: 13px; max-width: 560px;">
          <b>No official flowlines were loaded for this upload.</b><br>
          The app is intentionally not displaying model tile boxes or off-channel candidate spots. Direct flowline mapping requires official NHD flowline geometries for the uploaded GeoTIFF bounds.
        </div>
        """
        m.get_root().html.add_child(folium.Element(msg_html))

    spot_legend_html = f"""
    <div style="position: fixed; bottom: 28px; right: 28px; z-index: 9999; background: white; padding: 10px 12px; border: 1px solid #777; border-radius: 6px; font-size: 13px;">
      <b>Candidate obstruction spots</b><br>
      <span style="color:#e60000;font-size:18px;">●</span> Final model Yes candidates: {final_spots}<br>
      <span style="color:#ff9900;font-size:18px;">●</span> Review-required candidates: {review_spots}<br>
      <small>Spots are snapped to official flowlines. Click a spot for model details.</small>
    </div>
    """
    m.get_root().html.add_child(folium.Element(spot_legend_html))
    folium.LayerControl(collapsed=False).add_to(m)
    html = m.get_root().render()
    stats["map_created"] = True
    return (html, stats) if return_stats else html
