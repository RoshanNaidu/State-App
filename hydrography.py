"""Official Indiana hydrography helpers for Streamlit inference app.

The functions in this module are optional context helpers. They query the
Indiana-hosted NHD Classified Flowlines FeatureServer for a georeferenced
upload bounds, filter explicit subsurface/underground conduit or pipeline records when requested,
and create pixel-space line bboxes that can be used to reduce inference to
surface hydrography corridors.

No features are invented. If the official service cannot be reached, callers
receive an empty feature list plus a status message and should fall back to
image-only inference.
"""
from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional, Tuple

try:
    import requests
except Exception:  # pragma: no cover
    requests = None

try:
    from pyproj import Transformer
    PYPROJ_AVAILABLE = True
except Exception:  # pragma: no cover
    Transformer = None
    PYPROJ_AVAILABLE = False

try:
    import rasterio
    from rasterio.transform import rowcol
    RASTERIO_AVAILABLE = True
except Exception:  # pragma: no cover
    rasterio = None
    rowcol = None
    RASTERIO_AVAILABLE = False

NHD_CLASSIFIED_FLOWLINES_URL = "https://gisdata.in.gov/server/rest/services/Hosted/NHD_Classified_Flowlines/FeatureServer/0/query"
NHD_FALLBACK_FLOWLINES_URL = "https://gisdata.in.gov/server/rest/services/Hosted/NHD_Local_Resolution_RO/FeatureServer/0/query"

SUBSURFACE_FTYPE_VALUES = {420, 428}
SUBSURFACE_FCODE_VALUES = {42000, 42001, 42002, 42003, 42803, 42823}

# Strict display/training corridor filter for this app. The full Indiana NHD
# classified-flowline service can include artificial paths/connectors that may
# cross reservoirs, culverts, urban drainage, roads, or built areas. For the
# government-facing obstruction map we default to visible/surface channel-like
# linear features only: Stream/River and Canal/Ditch. This avoids presenting
# connector/artificial-path geometry as if it were a visible river channel.
SURFACE_CHANNEL_FTYPE_VALUES = {460, 336}
SURFACE_CHANNEL_FCODE_PREFIXES = {460, 336}

# For the public obstruction-detection map we only want flowlines that represent
# visible/surface stream channels or open artificial channels. NHD also contains
# network routing features such as ArtificialPath (558), Connector (334), Pipeline
# (428), and Underground Conduit (420). Those are official hydrography network
# features, but they are not appropriate as visible obstruction-review lines.
VISIBLE_SURFACE_FTYPE_VALUES = {336, 460}
VISIBLE_SURFACE_FCODE_PREFIXES = (336, 460)

DEFAULT_NHD_FIELDS = [
    "OBJECTID",
    "objectid",
    "gnis_name",
    "GNIS_NAME",
    "fcode",
    "FCode",
    "F_CODE",
    "ftype",
    "FType",
    "F_TYPE",
    "reachcode",
    "REACHCODE",
    "flowdir",
    "FlowDir",
    "lengthkm",
    "LengthKM",
]


def _safe_int(v: Any) -> Optional[int]:
    try:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return None
        return int(v)
    except Exception:
        return None


def _get_attr_case(attrs: Dict[str, Any], names: List[str]) -> Any:
    lower = {str(k).lower(): v for k, v in attrs.items()}
    for name in names:
        if name in attrs:
            return attrs[name]
        if name.lower() in lower:
            return lower[name.lower()]
    return None



def is_surface_channel_flowline(attrs: Dict[str, Any]) -> bool:
    """Return True for flowline types appropriate for visible-channel obstruction mapping.

    This is intentionally stricter than the full NHD classified-flowline layer.
    It keeps Stream/River and Canal/Ditch features and excludes connectors,
    artificial paths, underground conduits, and pipelines. It does not invent
    or edit geometry; it only filters official attributes for display/use.
    """
    ftype = _safe_int(_get_attr_case(attrs, ["ftype", "FType", "F_TYPE"]))
    fcode = _safe_int(_get_attr_case(attrs, ["fcode", "FCode", "F_CODE"]))
    if ftype in SURFACE_CHANNEL_FTYPE_VALUES:
        return True
    if fcode is not None and int(fcode // 100) in SURFACE_CHANNEL_FCODE_PREFIXES:
        return True
    text_blob = " ".join(str(v).lower() for v in attrs.values() if isinstance(v, str))
    if "stream" in text_blob or "river" in text_blob or "canal" in text_blob or "ditch" in text_blob:
        if "underground" not in text_blob and "pipeline" not in text_blob and "artificial path" not in text_blob and "connector" not in text_blob:
            return True
    return False

def is_underground_conduit(attrs: Dict[str, Any]) -> bool:
    """Return True only when attributes explicitly identify underground conduit."""
    ftype = _safe_int(_get_attr_case(attrs, ["ftype", "FType", "F_TYPE"]))
    fcode = _safe_int(_get_attr_case(attrs, ["fcode", "FCode", "F_CODE"]))
    if ftype in SUBSURFACE_FTYPE_VALUES:
        return True
    if fcode in SUBSURFACE_FCODE_VALUES:
        return True
    # Some ArcGIS services expose decoded labels instead of numeric codes.
    text_blob = " ".join(str(v).lower() for v in attrs.values() if isinstance(v, str))
    return ("underground conduit" in text_blob) or ("pipeline" in text_blob and "underground" in text_blob)




def is_visible_surface_flowline(attrs: Dict[str, Any]) -> bool:
    """Return True for flowline types suitable for visible obstruction review.

    This intentionally excludes NHD routing/network features such as ArtificialPath,
    Connector, Pipeline, and Underground Conduit. It keeps StreamRiver and
    CanalDitch classes because these are the visible/open-channel classes most
    relevant for this app's public map. No geometry is invented; this is only an
    attribute filter applied to official NHD features.
    """
    ftype = _safe_int(_get_attr_case(attrs, ["ftype", "FType", "F_TYPE"]))
    fcode = _safe_int(_get_attr_case(attrs, ["fcode", "FCode", "F_CODE"]))
    if ftype is not None:
        return ftype in VISIBLE_SURFACE_FTYPE_VALUES
    if fcode is not None:
        return any(str(fcode).startswith(str(prefix)) for prefix in VISIBLE_SURFACE_FCODE_PREFIXES)
    text_blob = " ".join(str(v).lower() for v in attrs.values() if isinstance(v, str))
    if any(x in text_blob for x in ["artificial path", "connector", "pipeline", "underground"]):
        return False
    if any(x in text_blob for x in ["streamriver", "stream/river", "stream river", "canalditch", "canal/ditch", "canal ditch"]):
        return True
    return False

def image_bounds_wgs84(image_meta: Dict[str, Any]) -> Tuple[Optional[Tuple[float, float, float, float]], str]:
    """Return upload bounds as (xmin, ymin, xmax, ymax) in EPSG:4326."""
    if not image_meta.get("georeferenced") or image_meta.get("bounds") is None or image_meta.get("crs") is None:
        return None, "IMAGE_NOT_GEOREFERENCED"
    b = image_meta["bounds"]
    crs_text = str(image_meta.get("crs", "")).upper()
    if "4326" in crs_text or "CRS84" in crs_text:
        return (float(b.left), float(b.bottom), float(b.right), float(b.top)), "OK"
    if not PYPROJ_AVAILABLE:
        return None, "PYPROJ_UNAVAILABLE"
    try:
        src_crs = image_meta["crs"]
        transformer = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)
        xs = [b.left, b.right, b.right, b.left]
        ys = [b.bottom, b.bottom, b.top, b.top]
        lonlats = [transformer.transform(x, y) for x, y in zip(xs, ys)]
        lons = [p[0] for p in lonlats]
        lats = [p[1] for p in lonlats]
        return (float(min(lons)), float(min(lats)), float(max(lons)), float(max(lats))), "OK"
    except Exception as exc:
        return None, f"BOUNDS_TRANSFORM_FAILED: {exc}"


def _query_one_nhd_endpoint(
    endpoint_url: str,
    bbox_wgs84: Tuple[float, float, float, float],
    exclude_underground: bool,
    max_records: int,
    timeout: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    xmin, ymin, xmax, ymax = bbox_wgs84
    features: List[Dict[str, Any]] = []
    offset = 0
    page_size = 1000
    source_error = None
    while len(features) < max_records:
        params = {
            "f": "json",
            "where": "1=1",
            # Use '*' for endpoint robustness. Some Indiana-hosted services expose
            # slightly different field names/cases. We still only display a small
            # subset in the map tooltip.
            "outFields": "*",
            "returnGeometry": "true",
            "outSR": "4326",
            "geometry": json.dumps({"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax, "spatialReference": {"wkid": 4326}}),
            "geometryType": "esriGeometryEnvelope",
            "inSR": "4326",
            "spatialRel": "esriSpatialRelIntersects",
            "resultRecordCount": min(page_size, max_records - len(features)),
            "resultOffset": offset,
        }
        try:
            resp = requests.post(endpoint_url, data=params, timeout=timeout, headers={"User-Agent": "IndianaBlockadeStreamlitApp/1.0"})
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            source_error = f"NHD_QUERY_FAILED: {exc}"
            break
        if "error" in data:
            source_error = f"NHD_SERVICE_ERROR: {data['error']}"
            break
        page = data.get("features", []) or []
        if not page:
            break
        for feat in page:
            attrs = feat.get("attributes", {}) or {}
            if exclude_underground and is_underground_conduit(attrs):
                continue
            if not is_visible_surface_flowline(attrs):
                continue
            geom = feat.get("geometry", {}) or {}
            paths = geom.get("paths", []) or []
            if not paths:
                continue
            features.append({"attributes": attrs, "paths": paths})
            if len(features) >= max_records:
                break
        if not data.get("exceededTransferLimit") and len(page) < page_size:
            break
        offset += page_size
    status = "OK" if source_error is None else source_error
    return features, {
        "status": status,
        "source_url": endpoint_url,
        "bbox_wgs84": bbox_wgs84,
        "exclude_underground": bool(exclude_underground),
        "feature_count": len(features),
        "max_records": max_records,
    }




# Packaged fallback: a slim GeoJSON cache extracted from the Phase 3 notebook's
# final Marion flowline decision map. These are not generated/fake flowlines;
# they are official-flowline decision features already produced by the notebook
# workflow. The cache is only used when the live official Indiana NHD service is
# unavailable or returns no usable records for a Marion-area demo upload.
from pathlib import Path

LOCAL_MARION_NOTEBOOK_FLOWLINE_CACHE = Path(__file__).resolve().parent / "data" / "marion_notebook_official_flowline_cache.geojson"


def _path_bbox_intersects(path: list, bbox_wgs84: Tuple[float, float, float, float]) -> bool:
    try:
        xs = [float(p[0]) for p in path if len(p) >= 2]
        ys = [float(p[1]) for p in path if len(p) >= 2]
        if not xs or not ys:
            return False
        xmin, ymin, xmax, ymax = bbox_wgs84
        return not (max(xs) < xmin or min(xs) > xmax or max(ys) < ymin or min(ys) > ymax)
    except Exception:
        return False


def load_cached_marion_notebook_flowlines(
    bbox_wgs84: Tuple[float, float, float, float],
    exclude_underground: bool = True,
    max_records: int = 5000,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Load packaged Marion notebook official-flowline features for demo fallback.

    This is a real-data fallback, not a synthetic fallback. It is derived from
    the final Phase 3 notebook map layer. It is intentionally limited to the
    Marion County project/demo area; non-overlapping uploads will still return
    no cached features.
    """
    if not LOCAL_MARION_NOTEBOOK_FLOWLINE_CACHE.exists():
        return [], {"status": "LOCAL_FLOWLINE_CACHE_NOT_FOUND", "source_url": str(LOCAL_MARION_NOTEBOOK_FLOWLINE_CACHE)}
    try:
        data = json.loads(LOCAL_MARION_NOTEBOOK_FLOWLINE_CACHE.read_text(encoding="utf-8"))
        out: List[Dict[str, Any]] = []
        for feat in data.get("features", []) or []:
            geom = feat.get("geometry", {}) or {}
            attrs = feat.get("properties", {}) or {}
            if exclude_underground and is_underground_conduit(attrs):
                continue
            if not is_visible_surface_flowline(attrs):
                continue
            paths = []
            if geom.get("type") == "LineString":
                coords = geom.get("coordinates", []) or []
                if _path_bbox_intersects(coords, bbox_wgs84):
                    paths = [coords]
            elif geom.get("type") == "MultiLineString":
                for coords in geom.get("coordinates", []) or []:
                    if _path_bbox_intersects(coords, bbox_wgs84):
                        paths.append(coords)
            if not paths:
                continue
            out.append({"attributes": attrs, "paths": paths})
            if len(out) >= max_records:
                break
        return out, {
            "status": "OK_LOCAL_NOTEBOOK_FLOWLINE_CACHE",
            "source_url": str(LOCAL_MARION_NOTEBOOK_FLOWLINE_CACHE),
            "bbox_wgs84": bbox_wgs84,
            "exclude_underground": bool(exclude_underground),
            "feature_count": len(out),
            "max_records": max_records,
            "note": "Used packaged Phase 3 notebook official-flowline cache because live NHD query returned no usable features. This cache is for the Marion demo area only.",
        }
    except Exception as exc:
        return [], {"status": f"LOCAL_FLOWLINE_CACHE_FAILED: {exc}", "source_url": str(LOCAL_MARION_NOTEBOOK_FLOWLINE_CACHE)}


def query_official_nhd_flowlines(
    bbox_wgs84: Tuple[float, float, float, float],
    exclude_underground: bool = True,
    max_records: int = 5000,
    timeout: int = 20,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Query official Indiana NHD flowlines within the given WGS84 bbox.

    Primary source: NHD Classified Flowlines. Fallback source: Indiana NHD Local
    Resolution flowlines. No flowlines are fabricated if the service cannot be
    reached or returns no records.
    """
    if requests is None:
        return [], {"status": "REQUESTS_UNAVAILABLE", "source_url": NHD_CLASSIFIED_FLOWLINES_URL}

    attempts = []
    for endpoint in [NHD_CLASSIFIED_FLOWLINES_URL, NHD_FALLBACK_FLOWLINES_URL]:
        feats, meta = _query_one_nhd_endpoint(endpoint, bbox_wgs84, exclude_underground, max_records, timeout)
        # Store a shallow copy in attempts so the returned meta object never
        # contains a list that references itself. Streamlit serializes this
        # diagnostics dictionary in the UI, and a self-reference triggers
        # "ValueError: Circular reference detected".
        attempts.append(dict(meta))
        if feats:
            out_meta = dict(meta)
            out_meta["attempts"] = [dict(a) for a in attempts]
            return feats, out_meta

    # If the live official services return no usable records, try the packaged
    # Marion notebook flowline cache. This keeps the demo map flowline-based
    # instead of showing raw model tiles, while still avoiding invented features.
    cached_feats, cached_meta = load_cached_marion_notebook_flowlines(
        bbox_wgs84,
        exclude_underground=exclude_underground,
        max_records=max_records,
    )
    if cached_feats:
        cached_meta = dict(cached_meta)
        cached_meta["attempts"] = [dict(a) for a in attempts] + [dict(cached_meta)]
        cached_meta["live_service_fallback_used"] = True
        return cached_feats, cached_meta

    # Preserve diagnostics from all official attempts without creating a circular reference.
    final = dict(attempts[-1]) if attempts else {"status": "NO_QUERY_ATTEMPTED"}
    final["attempts"] = [dict(a) for a in attempts] + [dict(cached_meta)]
    final["status"] = "NO_OFFICIAL_FLOWLINES_RETURNED"
    final["feature_count"] = 0
    return [], final


def feature_pixel_bboxes(
    features: List[Dict[str, Any]],
    image_meta: Dict[str, Any],
    buffer_px: int = 32,
) -> Tuple[List[Tuple[int, int, int, int]], str]:
    """Convert WGS84 NHD polylines to buffered pixel-space bboxes."""
    if not features:
        return [], "NO_FEATURES"
    if not image_meta.get("georeferenced") or image_meta.get("transform") is None or image_meta.get("crs") is None:
        return [], "IMAGE_NOT_GEOREFERENCED"
    if not PYPROJ_AVAILABLE or not RASTERIO_AVAILABLE:
        return [], "PYPROJ_OR_RASTERIO_UNAVAILABLE"
    try:
        transformer = Transformer.from_crs("EPSG:4326", image_meta["crs"], always_xy=True)
        transform = image_meta["transform"]
        h = int(image_meta["height"])
        w = int(image_meta["width"])
        bboxes: List[Tuple[int, int, int, int]] = []
        for feat in features:
            rows: List[int] = []
            cols: List[int] = []
            for path in feat.get("paths", []) or []:
                for pt in path:
                    if len(pt) < 2:
                        continue
                    lon, lat = float(pt[0]), float(pt[1])
                    x, y = transformer.transform(lon, lat)
                    try:
                        row, col = rowcol(transform, x, y)
                    except Exception:
                        continue
                    if -buffer_px <= row <= h + buffer_px and -buffer_px <= col <= w + buffer_px:
                        rows.append(int(row)); cols.append(int(col))
            if rows and cols:
                x0 = max(0, min(cols) - buffer_px)
                y0 = max(0, min(rows) - buffer_px)
                x1 = min(w, max(cols) + buffer_px)
                y1 = min(h, max(rows) + buffer_px)
                if x1 > x0 and y1 > y0:
                    bboxes.append((x0, y0, x1, y1))
        return bboxes, "OK"
    except Exception as exc:
        return [], f"PIXEL_BBOX_FAILED: {exc}"


def geojson_from_nhd_features(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    out_features = []
    for feat in features:
        attrs = feat.get("attributes", {}) or {}
        for path_idx, path in enumerate(feat.get("paths", []) or []):
            if len(path) < 2:
                continue
            coords = [[float(p[0]), float(p[1])] for p in path if len(p) >= 2]
            if len(coords) >= 2:
                out_features.append({
                    "type": "Feature",
                    "geometry": {"type": "LineString", "coordinates": coords},
                    "properties": {**attrs, "path_index": path_idx},
                })
    return {"type": "FeatureCollection", "features": out_features}
