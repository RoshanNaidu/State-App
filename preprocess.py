"""Image loading and 9-channel preprocessing for the Phase 3 model."""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import io

import numpy as np
from PIL import Image

try:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject
    RASTERIO_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency fallback
    rasterio = None
    Resampling = None
    reproject = None
    RASTERIO_AVAILABLE = False

from config import PHASE3_CHANNEL_NAMES


def robust01(a: np.ndarray, p_low: float = 2, p_high: float = 98) -> np.ndarray:
    a = np.asarray(a, dtype="float32")
    finite = np.isfinite(a)
    if not finite.any():
        return np.zeros_like(a, dtype="float32")
    fill = float(np.nanmedian(a[finite]))
    a = np.nan_to_num(a, nan=fill, posinf=float(np.nanmax(a[finite])), neginf=float(np.nanmin(a[finite])))
    lo, hi = np.nanpercentile(a, [p_low, p_high])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
        if hi <= lo:
            return np.zeros_like(a, dtype="float32")
    return np.clip((a - lo) / (hi - lo), 0, 1).astype("float32")


def as_bands_first(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return arr[None, :, :]
    if arr.ndim == 3:
        if arr.shape[0] <= 10:
            return arr
        return arr.transpose(2, 0, 1)
    raise ValueError(f"Unsupported raster/image shape: {arr.shape}")


def _save_upload_to_temp(uploaded_file, suffix: str) -> Path:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getbuffer())
    tmp.flush()
    tmp.close()
    return Path(tmp.name)


def load_uploaded_image(uploaded_file) -> Dict[str, Any]:
    """Load image upload.

    Returns a dictionary with bands_first array and geospatial metadata when available.
    GeoTIFF/TIFF requires rasterio. PNG/JPG uses PIL and is pixel-only.
    """
    if uploaded_file is None:
        raise ValueError("No image uploaded.")
    name = uploaded_file.name or "uploaded_image"
    ext = Path(name).suffix.lower()
    if ext in {".tif", ".tiff"}:
        if not RASTERIO_AVAILABLE:
            raise RuntimeError("GeoTIFF/TIFF upload requires rasterio. Install requirements.txt.")
        tmp = _save_upload_to_temp(uploaded_file, ext)
        with rasterio.open(tmp) as ds:
            count = min(ds.count, 4)
            arr = ds.read(indexes=list(range(1, count + 1))).astype("float32")
            meta = {
                "filename": name,
                "tmp_path": str(tmp),
                "kind": "geotiff",
                "array": arr,
                "crs": ds.crs,
                "transform": ds.transform,
                "bounds": ds.bounds,
                "width": ds.width,
                "height": ds.height,
                "band_count": ds.count,
                "resolution": ds.res,
                "georeferenced": bool(ds.crs and ds.transform),
            }
        return meta
    # Non-georeferenced image mode.
    # Streamlit UploadedFile is file-like, but tests/custom callers may only expose getbuffer().
    try:
        uploaded_file.seek(0)
        img = Image.open(uploaded_file).convert("RGB")
    except Exception:
        img = Image.open(io.BytesIO(bytes(uploaded_file.getbuffer()))).convert("RGB")
    arr = np.asarray(img).astype("float32").transpose(2, 0, 1)
    return {
        "filename": name,
        "kind": "image",
        "array": arr,
        "crs": None,
        "transform": None,
        "bounds": None,
        "width": img.size[0],
        "height": img.size[1],
        "band_count": 3,
        "resolution": None,
        "georeferenced": False,
    }


def _resize_2d(arr: np.ndarray, out_shape: Tuple[int, int]) -> np.ndarray:
    """Resize float 2-D array to (height, width) using PIL bilinear."""
    arr = np.nan_to_num(np.asarray(arr, dtype="float32"), nan=0.0, posinf=1.0, neginf=0.0)
    h, w = out_shape
    im = Image.fromarray(arr, mode="F")
    im = im.resize((w, h), resample=Image.BILINEAR)
    return np.asarray(im, dtype="float32")


def load_dem_to_match(dem_upload, image_meta: Dict[str, Any]) -> Tuple[Optional[np.ndarray], str]:
    if dem_upload is None:
        return None, "DEM_NOT_UPLOADED"
    if not RASTERIO_AVAILABLE:
        return None, "DEM_IGNORED_RASTERIO_UNAVAILABLE"
    tmp = _save_upload_to_temp(dem_upload, Path(dem_upload.name).suffix.lower() or ".tif")
    try:
        with rasterio.open(tmp) as dem_ds:
            src = dem_ds.read(1).astype("float32")
            target_h = int(image_meta["height"])
            target_w = int(image_meta["width"])
            if image_meta.get("georeferenced") and dem_ds.crs and image_meta.get("transform") is not None:
                dst = np.zeros((target_h, target_w), dtype="float32")
                reproject(
                    source=src,
                    destination=dst,
                    src_transform=dem_ds.transform,
                    src_crs=dem_ds.crs,
                    dst_transform=image_meta["transform"],
                    dst_crs=image_meta["crs"],
                    resampling=Resampling.bilinear,
                )
                return dst, "DEM_REPROJECTED_TO_IMAGE_GRID"
            return _resize_2d(src, (target_h, target_w)), "DEM_RESIZED_NO_SHARED_GEOREFERENCE"
    except Exception as exc:
        return None, f"DEM_LOAD_FAILED: {exc}"




def load_nir_to_match(nir_upload, image_meta: Dict[str, Any]) -> Tuple[Optional[np.ndarray], str]:
    """Load optional separate NIR GeoTIFF and align it to the uploaded image grid.

    This is optional. If unavailable, the model's NIR and NDVI channels remain
    zero-filled, matching the production notebook's explicit *_or_zero naming.
    """
    if nir_upload is None:
        return None, "NIR_NOT_UPLOADED"
    if not RASTERIO_AVAILABLE:
        return None, "NIR_IGNORED_RASTERIO_UNAVAILABLE"
    tmp = _save_upload_to_temp(nir_upload, Path(nir_upload.name).suffix.lower() or ".tif")
    try:
        with rasterio.open(tmp) as nir_ds:
            src = nir_ds.read(1).astype("float32")
            target_h = int(image_meta["height"])
            target_w = int(image_meta["width"])
            if image_meta.get("georeferenced") and nir_ds.crs and image_meta.get("transform") is not None:
                dst = np.zeros((target_h, target_w), dtype="float32")
                reproject(
                    source=src,
                    destination=dst,
                    src_transform=nir_ds.transform,
                    src_crs=nir_ds.crs,
                    dst_transform=image_meta["transform"],
                    dst_crs=image_meta["crs"],
                    resampling=Resampling.bilinear,
                )
                return dst, "NIR_REPROJECTED_TO_IMAGE_GRID"
            return _resize_2d(src, (target_h, target_w)), "NIR_RESIZED_NO_SHARED_GEOREFERENCE"
    except Exception as exc:
        return None, f"NIR_LOAD_FAILED: {exc}"

def build_phase3_stack(image_meta: Dict[str, Any], dem_grid: Optional[np.ndarray] = None, nir_grid: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Build the same 9 channels used by the Phase 3 notebook.

    Channels: red, green, blue, nir_or_zero, ndvi01_or_zero, water_like_proxy,
    dem01_or_zero, dem_slope01_or_zero, brightness.
    """
    raw = as_bands_first(image_meta["array"]).astype("float32")
    if raw.shape[0] < 3:
        raise ValueError("At least 3 bands/RGB channels are required.")
    # Use first three bands as RGB, robustly normalized.
    rgb = np.stack([robust01(raw[0]), robust01(raw[1]), robust01(raw[2])], axis=0)
    h, w = rgb.shape[1], rgb.shape[2]
    zero = np.zeros((h, w), dtype="float32")

    # Treat GeoTIFF/TIFF fourth band as NIR if present, or use a separate
    # optional NIR grid if provided. Otherwise the *_or_zero channels remain zero.
    if nir_grid is not None and np.asarray(nir_grid).shape == (h, w):
        nir_raw = np.asarray(nir_grid, dtype="float32")
        nir01 = robust01(nir_raw)
        red = raw[0].astype("float32")
        ndvi = (nir_raw - red) / np.maximum(nir_raw + red, 1e-6)
        ndvi01 = np.clip((ndvi + 1.0) / 2.0, 0, 1).astype("float32")
        nir_status = "NIR_USED_FROM_SEPARATE_UPLOAD"
    elif image_meta.get("kind") == "geotiff" and raw.shape[0] >= 4:
        nir01 = robust01(raw[3])
        red = raw[0].astype("float32")
        nir = raw[3].astype("float32")
        ndvi = (nir - red) / np.maximum(nir + red, 1e-6)
        ndvi01 = np.clip((ndvi + 1.0) / 2.0, 0, 1).astype("float32")
        nir_status = "NIR_USED_FROM_BAND_4"
    else:
        nir01 = zero.copy()
        ndvi01 = zero.copy()
        nir_status = "NIR_NOT_AVAILABLE_ZERO_FILLED"

    brightness = np.clip(rgb.mean(axis=0), 0, 1).astype("float32")
    blue_green = np.maximum(rgb[1], rgb[2])
    water_like = np.clip(blue_green * (1.0 - brightness), 0, 1).astype("float32")

    if dem_grid is not None and np.asarray(dem_grid).shape == (h, w):
        dem_raw = np.asarray(dem_grid, dtype="float32")
        dem01 = robust01(dem_raw)
        finite = np.isfinite(dem_raw)
        fill = float(np.nanmedian(dem_raw[finite])) if finite.any() else 0.0
        gy, gx = np.gradient(np.nan_to_num(dem_raw, nan=fill))
        slope01 = robust01(np.sqrt(gx * gx + gy * gy))
        dem_status = "DEM_USED"
    else:
        dem01 = zero.copy()
        slope01 = zero.copy()
        dem_status = "DEM_NOT_AVAILABLE_ZERO_FILLED"

    stack = np.stack([rgb[0], rgb[1], rgb[2], nir01, ndvi01, water_like, dem01, slope01, brightness], axis=0)
    stack = np.nan_to_num(stack, nan=0.0, posinf=1.0, neginf=0.0).astype("float32")
    stack = np.clip(stack, 0, 1)
    meta = {
        "status": "OK",
        "height": h,
        "width": w,
        "channel_names": PHASE3_CHANNEL_NAMES,
        "nir_status": nir_status,
        "dem_status": dem_status,
        "image_mode": "georeferenced_geotiff" if image_meta.get("georeferenced") else "pixel_only_image",
    }
    return stack, meta


def stack_to_rgb_preview(stack: np.ndarray, max_size: int = 1100) -> Image.Image:
    rgb = np.clip(stack[:3].transpose(1, 2, 0), 0, 1)
    im = Image.fromarray((rgb * 255).astype("uint8"))
    im.thumbnail((max_size, max_size), Image.LANCZOS)
    return im
