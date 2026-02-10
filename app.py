"""
Sentinel-2 Monthly Composite Explorer
======================================
PURPOSE:
  - Show which individual images are used to create each monthly median composite
  - Allow downloading those individual images at 10m, all 12 bands
  - Show patch validity maps (green/red) for each downloaded monthly composite
  - NO classification model needed

PIPELINE:
  1. GEE cloud masking (same as original app)
  2. For each month: list component images, create median composite, gap-fill
  3. Download composites + track component image metadata
  4. Patch validity analysis per month
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from shapely.geometry import Polygon, box
import rasterio
from patchify import patchify
import datetime
import math
import ee
import tempfile
import requests
import time
import warnings
import sys
import base64
import json
import subprocess
from datetime import date
from PIL import Image
from io import BytesIO

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import streamlit as st

st.set_page_config(
    layout="wide",
    page_title="Sentinel-2 Composite Explorer",
    page_icon="🛰️"
)

import folium
from folium import plugins
from streamlit_folium import st_folium

# =============================================================================
# CONSTANTS
# =============================================================================
SPECTRAL_BANDS = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
PATCH_SIZE = 224

MAX_RETRIES = 3
RETRY_DELAY_BASE = 2
DOWNLOAD_TIMEOUT = 120
CHUNK_SIZE = 8192

MIN_BAND_FILE_SIZE = 10000
MIN_MULTIBAND_FILE_SIZE = 100000

CLOUD_PROB_THRESHOLD = 50
CDI_THRESHOLD = -0.5
MAX_MASKED_PERCENT_FOR_GAPFILL = 30

STATUS_NO_DATA = "no_data"
STATUS_SKIPPED = "skipped"
STATUS_COMPLETE = "complete"
STATUS_REJECTED = "rejected"

# =============================================================================
# Session State Initialization
# =============================================================================
_defaults = {
    'drawn_polygons': [],
    'last_drawn_polygon': None,
    'ee_initialized': False,
    'selected_region_index': 0,
    'processing_in_progress': False,
    'processing_config': None,
    'processing_complete': False,
    'current_temp_dir': None,
    # Core data stores
    'downloaded_composites': {},        # {month_name: file_path}
    'month_statuses': {},               # {month_name: {status, message}}
    'component_image_metadata': {},     # {month_name: [{id, date, cloud_pct, system_index}, ...]}
    'patch_validity_maps': {},          # {month_name: {mask, n_valid, n_total, original_size}}
    # For component image downloads
    'component_download_queue': {},     # {month_name: {image_id: file_path}}
    # Logging
    'processing_log': [],
}

for key, default in _defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default


# =============================================================================
# Logging
# =============================================================================
def add_log(msg, level="INFO"):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    st.session_state.processing_log.append({'ts': ts, 'level': level, 'msg': msg})
    print(f"[{ts}] [{level}] {msg}")


# =============================================================================
# File Validation
# =============================================================================
def validate_geotiff(path, expected_bands=1):
    try:
        if not os.path.exists(path):
            return False, "File missing"
        sz = os.path.getsize(path)
        min_sz = MIN_BAND_FILE_SIZE if expected_bands == 1 else MIN_MULTIBAND_FILE_SIZE
        if sz < min_sz:
            return False, f"Too small ({sz}B)"
        with rasterio.open(path) as src:
            if src.count < expected_bands:
                return False, f"Bands: {src.count} < {expected_bands}"
        return True, "OK"
    except Exception as e:
        return False, str(e)


# =============================================================================
# Earth Engine Auth
# =============================================================================
@st.cache_resource
def initialize_earth_engine():
    try:
        ee.Initialize()
        return True, "Earth Engine initialized"
    except Exception:
        try:
            b64key = os.environ.get('GOOGLE_EARTH_ENGINE_KEY_BASE64')
            if b64key:
                kd = json.loads(base64.b64decode(b64key).decode())
                kf = tempfile.NamedTemporaryFile(suffix='.json', delete=False)
                with open(kf.name, 'w') as f:
                    json.dump(kd, f)
                creds = ee.ServiceAccountCredentials(kd['client_email'], kf.name)
                ee.Initialize(creds)
                os.unlink(kf.name)
                return True, "Service Account auth"
            else:
                ee.Authenticate()
                ee.Initialize()
                return True, "Authenticated"
        except Exception as e:
            return False, f"Auth failed: {e}"


# =============================================================================
# Helpers
# =============================================================================
def get_utm_zone(lon):
    return math.floor((lon + 180) / 6) + 1


# =============================================================================
# GEE Cloud-Free Collection (same as original)
# =============================================================================
def create_cloud_free_collection(aoi, extended_start, extended_end, cloudy_pixel_percentage=10):
    s2_sr = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
             .filterBounds(aoi)
             .filterDate(extended_start, extended_end)
             .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloudy_pixel_percentage))
             .select(SPECTRAL_BANDS + ['SCL']))

    s2_cloud_prob = (ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
                     .filterBounds(aoi)
                     .filterDate(extended_start, extended_end))

    join_filter = ee.Filter.equals(leftField='system:index', rightField='system:index')
    joined = ee.Join.saveFirst('cloud_probability').apply(
        primary=s2_sr, secondary=s2_cloud_prob, condition=join_filter
    )

    def add_cloud_band(feature):
        img = ee.Image(feature)
        cp = ee.Image(img.get('cloud_probability'))
        return img.addBands(cp)

    s2_joined = ee.ImageCollection(joined.map(add_cloud_band))

    def mask_cloud_and_shadow(img):
        cloud_prob = img.select('probability')
        cdi = ee.Algorithms.Sentinel2.CDI(img)
        is_cloud = cloud_prob.gt(CLOUD_PROB_THRESHOLD).And(cdi.lt(CDI_THRESHOLD))
        kernel = ee.Kernel.circle(radius=20, units='meters')
        cloud_dilated = is_cloud.focal_max(kernel=kernel, iterations=2)
        masked = img.updateMask(cloud_dilated.Not())
        scaled = masked.select(SPECTRAL_BANDS).multiply(0.0001).clip(aoi)
        return scaled.copyProperties(img, ['system:time_start', 'system:index',
                                            'CLOUDY_PIXEL_PERCENTAGE',
                                            'SPACECRAFT_NAME',
                                            'SENSING_ORBIT_NUMBER'])

    return s2_joined.map(mask_cloud_and_shadow)


# =============================================================================
# Download helpers (band-by-band with retry, same as original)
# =============================================================================
def download_band_with_retry(image, band, aoi, output_path, scale=10):
    try:
        region = aoi.bounds().getInfo()['coordinates']
    except Exception as e:
        return False, f"AOI error: {e}"

    tmp = output_path + '.tmp'
    if os.path.exists(tmp):
        os.remove(tmp)

    if os.path.exists(output_path):
        ok, _ = validate_geotiff(output_path, 1)
        if ok:
            return True, "cached"
        os.remove(output_path)

    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            url = image.select(band).getDownloadURL({
                'scale': scale, 'region': region, 'format': 'GEO_TIFF', 'bands': [band]
            })
            resp = requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT)
            if resp.status_code == 200:
                if 'text/html' in resp.headers.get('content-type', ''):
                    raise Exception("Rate limit")
                sz = 0
                with open(tmp, 'wb') as f:
                    for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                        if chunk:
                            f.write(chunk)
                            sz += len(chunk)
                if sz < MIN_BAND_FILE_SIZE:
                    raise Exception(f"Too small ({sz}B)")
                ok, msg = validate_geotiff(tmp, 1)
                if ok:
                    os.replace(tmp, output_path)
                    return True, "success"
                raise Exception(f"Validation: {msg}")
            raise Exception(f"HTTP {resp.status_code}")
        except Exception as e:
            last_err = str(e)
        for f in [output_path, tmp]:
            if os.path.exists(f):
                try: os.remove(f)
                except: pass
        if attempt < MAX_RETRIES - 1:
            time.sleep(RETRY_DELAY_BASE ** (attempt + 1))
    return False, last_err


def download_composite(composite, aoi, output_path, month_name, scale=10, status_ph=None):
    """Download composite image (12 bands) to GeoTIFF."""
    if os.path.exists(output_path):
        ok, msg = validate_geotiff(output_path, len(SPECTRAL_BANDS))
        if ok:
            return output_path
        os.remove(output_path)

    temp_dir = os.path.dirname(output_path)
    bands_dir = os.path.join(temp_dir, f"bands_{month_name}")
    os.makedirs(bands_dir, exist_ok=True)

    band_files = []
    failed = []
    for i, band in enumerate(SPECTRAL_BANDS):
        bf = os.path.join(bands_dir, f"{band}.tif")
        if status_ph:
            status_ph.text(f"📥 {month_name}: {band} ({i+1}/{len(SPECTRAL_BANDS)})...")
        ok, err = download_band_with_retry(composite, band, aoi, bf, scale)
        if ok:
            band_files.append(bf)
        else:
            failed.append(f"{band}: {err}")

    if failed:
        st.error(f"❌ {month_name}: Failed bands - {'; '.join(failed)}")
        return None

    if len(band_files) == len(SPECTRAL_BANDS):
        if status_ph:
            status_ph.text(f"📦 {month_name}: Merging bands...")
        with rasterio.open(band_files[0]) as src:
            meta = src.meta.copy()
        meta.update(count=len(band_files))
        with rasterio.open(output_path, 'w', **meta) as dst:
            for i, bf in enumerate(band_files):
                with rasterio.open(bf) as src:
                    dst.write(src.read(1), i + 1)
        ok, msg = validate_geotiff(output_path, len(SPECTRAL_BANDS))
        if ok:
            return output_path
        if os.path.exists(output_path):
            os.remove(output_path)
    return None


# =============================================================================
# Download a single component image (individual scene, 12 bands, 10m)
# =============================================================================
def download_component_image(cloud_free_collection, system_index, aoi, output_path, scale=10, status_ph=None):
    """
    Download an individual cloud-masked scene at 10m, 12 bands.
    The image is identified by system:index within the cloud_free_collection.
    """
    try:
        # Filter to the specific image by system:index
        img = cloud_free_collection.filter(
            ee.Filter.eq('system:index', system_index)
        ).first()

        # Check it exists
        info = img.getInfo()
        if info is None:
            return None, "Image not found in collection"

        return download_composite(img, aoi, output_path, f"component_{system_index}", scale, status_ph)
    except Exception as e:
        return None, str(e)


# =============================================================================
# Gather component image metadata for a month
# =============================================================================
def get_month_component_metadata(cloud_free_collection, month_start, month_end):
    """
    Get metadata for all images in a month from the cloud-free collection.
    Returns list of dicts with date, system_index, cloud_pct, spacecraft.
    """
    monthly = cloud_free_collection.filterDate(month_start, month_end)
    count = monthly.size().getInfo()
    if count == 0:
        return []

    # Get metadata for each image (limit to avoid huge GEE calls)
    max_fetch = 50
    img_list = monthly.toList(min(count, max_fetch))

    metadata = []
    for i in range(min(count, max_fetch)):
        try:
            img = ee.Image(img_list.get(i))
            props = img.toDictionary([
                'system:time_start', 'system:index',
                'CLOUDY_PIXEL_PERCENTAGE', 'SPACECRAFT_NAME'
            ]).getInfo()

            ts = props.get('system:time_start', 0)
            dt = datetime.datetime.utcfromtimestamp(ts / 1000)

            metadata.append({
                'system_index': props.get('system:index', f'unknown_{i}'),
                'date': dt.strftime('%Y-%m-%d'),
                'datetime': dt,
                'cloud_pct': props.get('CLOUDY_PIXEL_PERCENTAGE', -1),
                'spacecraft': props.get('SPACECRAFT_NAME', 'unknown'),
            })
        except Exception as e:
            add_log(f"Failed to get metadata for image {i}: {e}", "WARNING")

    # Sort by date
    metadata.sort(key=lambda x: x['date'])
    return metadata


# =============================================================================
# Monthly processing: composite + metadata + download
# =============================================================================
def process_month(aoi, cloud_free_collection, month_info, temp_dir, scale=10, status_ph=None):
    """
    Process a single month:
    1. Get component image metadata
    2. Create median composite
    3. Check masked pixels, gap-fill if needed
    4. Download composite if 0% masked

    Returns: (output_path, status, message, component_metadata)
    """
    month_name = month_info['month_name']
    month_index = month_info['month_index']
    origin = month_info['origin']

    output_file = os.path.join(temp_dir, f"sentinel2_{month_name}.tif")
    add_log(f"Processing: {month_name}")

    # Cache check
    if os.path.exists(output_file):
        ok, msg = validate_geotiff(output_file, len(SPECTRAL_BANDS))
        if ok:
            add_log(f"{month_name}: cached")
            # Still need metadata even if cached
            origin_date = ee.Date(origin)
            ms = origin_date.advance(month_index, 'month')
            me = origin_date.advance(ee.Number(month_index).add(1), 'month')
            meta = get_month_component_metadata(cloud_free_collection, ms, me)
            return output_file, STATUS_COMPLETE, "Cached", meta

    try:
        origin_date = ee.Date(origin)
        month_start = origin_date.advance(month_index, 'month')
        month_end = origin_date.advance(ee.Number(month_index).add(1), 'month')
        month_middle = month_start.advance(15, 'day')

        if status_ph:
            status_ph.text(f"📥 {month_name}: Analyzing...")

        # Get component images for this month
        monthly_images = cloud_free_collection.filterDate(month_start, month_end)
        image_count = monthly_images.size().getInfo()

        add_log(f"{month_name}: {image_count} cloud-free images")

        # Gather metadata
        if status_ph:
            status_ph.text(f"📥 {month_name}: Gathering metadata for {image_count} images...")
        component_meta = get_month_component_metadata(cloud_free_collection, month_start, month_end)
        add_log(f"{month_name}: Got metadata for {len(component_meta)} images")

        if image_count == 0:
            return None, STATUS_NO_DATA, "No images", component_meta

        if status_ph:
            status_ph.text(f"📥 {month_name}: Creating composite from {image_count} images...")

        # Frequency + composite
        def valid_mask_fn(img):
            return ee.Image(1).updateMask(img.select('B4').mask()).unmask(0).toInt()

        frequency = monthly_images.map(valid_mask_fn).sum().toInt().rename('frequency')
        composite = monthly_images.median()

        # Masked pixel stats
        masked_stats = frequency.eq(0).reduceRegion(
            reducer=ee.Reducer.sum(), geometry=aoi, scale=10, maxPixels=1e13)
        total_stats = frequency.gte(0).reduceRegion(
            reducer=ee.Reducer.sum(), geometry=aoi, scale=10, maxPixels=1e13)

        masked_count = ee.Number(masked_stats.get('frequency')).getInfo()
        total_count = ee.Number(total_stats.get('frequency')).getInfo()

        if total_count == 0:
            return None, STATUS_NO_DATA, "No valid pixels", component_meta

        masked_pct = (masked_count / total_count) * 100
        add_log(f"{month_name}: Masked {masked_pct:.2f}%")

        if masked_pct > MAX_MASKED_PERCENT_FOR_GAPFILL:
            return None, STATUS_SKIPPED, f"Masked {masked_pct:.1f}% > {MAX_MASKED_PERCENT_FOR_GAPFILL}%", component_meta

        if masked_pct == 0:
            if status_ph:
                status_ph.text(f"📥 {month_name}: Complete (0% masked), downloading...")
            path = download_composite(composite, aoi, output_file, month_name, scale, status_ph)
            if path:
                return path, STATUS_COMPLETE, "0% masked", component_meta
            return None, STATUS_REJECTED, "Download failed", component_meta

        # Gap-fill
        add_log(f"{month_name}: Gap-filling ({masked_pct:.1f}% masked)")
        if status_ph:
            status_ph.text(f"📥 {month_name}: Gap-filling ({masked_pct:.1f}% masked)...")

        gap_mask = frequency.eq(0)
        month_middle_millis = month_middle.millis()

        m1_past_start = origin_date.advance(ee.Number(month_index).subtract(1), 'month')
        m1_past_end = month_start
        m1_future_start = month_end
        m1_future_end = origin_date.advance(ee.Number(month_index).add(2), 'month')

        m1_past = cloud_free_collection.filterDate(m1_past_start, m1_past_end)
        m1_future = cloud_free_collection.filterDate(m1_future_start, m1_future_end)
        candidates = m1_past.merge(m1_future)

        def add_dist(img):
            t = ee.Number(img.get('system:time_start'))
            return img.set('time_distance', t.subtract(month_middle_millis).abs())

        sorted_cands = candidates.map(add_dist).sort('time_distance', True)
        cand_count = sorted_cands.size().getInfo()
        add_log(f"{month_name}: {cand_count} gap-fill candidates")

        if cand_count == 0:
            return None, STATUS_REJECTED, f"No gap-fill candidates, {masked_pct:.1f}% masked", component_meta

        closest_mosaic = sorted_cands.mosaic().select(SPECTRAL_BANDS)
        has_closest = closest_mosaic.select('B4').mask()
        fill_from = gap_mask.And(has_closest)
        still_masked = gap_mask.And(has_closest.Not())

        filled = composite.unmask(closest_mosaic.updateMask(fill_from))

        fill_source = (ee.Image.constant(0).clip(aoi).toInt8()
                       .where(fill_from, 1).where(still_masked, 2).rename('fill_source'))
        sm_result = fill_source.eq(2).reduceRegion(
            reducer=ee.Reducer.sum(), geometry=aoi, scale=10, maxPixels=1e13
        ).get('fill_source')
        sm_count = ee.Number(ee.Algorithms.If(
            ee.Algorithms.IsEqual(sm_result, None), 0, sm_result)).getInfo()

        if sm_count == 0:
            if status_ph:
                status_ph.text(f"📥 {month_name}: Gap-filled, downloading...")
            path = download_composite(filled, aoi, output_file, month_name, scale, status_ph)
            if path:
                return path, STATUS_COMPLETE, f"Gap-filled (was {masked_pct:.1f}%)", component_meta
            return None, STATUS_REJECTED, "Download failed after gap-fill", component_meta
        else:
            sm_pct = (sm_count / total_count) * 100
            return None, STATUS_REJECTED, f"{sm_pct:.1f}% still masked after gap-fill", component_meta

    except Exception as e:
        add_log(f"{month_name}: ERROR - {e}", "ERROR")
        return None, STATUS_NO_DATA, f"Error: {e}", []


# =============================================================================
# Patch validity analysis
# =============================================================================
def check_patch_validity(patch, nodata_threshold_percent=0):
    if np.any(np.isnan(patch)):
        return False
    if np.all(patch == 0):
        return False
    zero_pct = (np.sum(patch == 0) / patch.size) * 100
    if zero_pct > nodata_threshold_percent:
        return False
    if patch.ndim == 3:
        for b in range(patch.shape[-1]):
            if np.all(patch[:, :, b] == 0):
                return False
    return True


def compute_patch_validity_map(image_path, patch_size=224):
    """
    Compute a patch-level validity mask for a single image.
    Returns dict with mask, counts, original_size, or None on failure.
    """
    try:
        with rasterio.open(image_path) as src:
            data = src.read()  # (bands, h, w)
        img = np.moveaxis(data, 0, -1)  # (h, w, bands)
        h, w, c = img.shape

        new_h = int(np.ceil(h / patch_size) * patch_size)
        new_w = int(np.ceil(w / patch_size) * patch_size)
        if h != new_h or w != new_w:
            padded = np.zeros((new_h, new_w, c), dtype=img.dtype)
            padded[:h, :w, :] = img
            img = padded

        patches = patchify(img, (patch_size, patch_size, c), step=patch_size)
        nph, npw = patches.shape[0], patches.shape[1]
        mask = np.zeros((nph, npw), dtype=bool)

        for i in range(nph):
            for j in range(npw):
                mask[i, j] = check_patch_validity(patches[i, j, 0])

        return {
            'mask': mask,
            'n_valid': int(np.sum(mask)),
            'n_total': nph * npw,
            'n_patches_h': nph,
            'n_patches_w': npw,
            'original_size': (h, w),
        }
    except Exception as e:
        add_log(f"Patch validity error: {e}", "ERROR")
        return None


def save_patch_validity_geotiff(validity_info, reference_image_path, output_path, patch_size=224):
    """
    Save a patch validity map as a GeoTIFF where each pixel within a patch
    gets the patch's validity value (1=valid, 0=invalid).
    Uses the georeferencing from the reference image.
    """
    try:
        mask = validity_info['mask']
        orig_h, orig_w = validity_info['original_size']

        # Expand patch mask to pixel level
        pixel_mask = np.repeat(np.repeat(mask.astype(np.uint8), patch_size, axis=0), patch_size, axis=1)
        pixel_mask = pixel_mask[:orig_h, :orig_w]

        with rasterio.open(reference_image_path) as src:
            meta = src.meta.copy()

        meta.update({
            'count': 1,
            'dtype': 'uint8',
            'height': orig_h,
            'width': orig_w,
        })

        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(pixel_mask, 1)

        return output_path
    except Exception as e:
        add_log(f"Save patch GeoTIFF error: {e}", "ERROR")
        return None


# =============================================================================
# RGB thumbnail generation
# =============================================================================
def generate_rgb_thumbnail(image_path, max_size=256):
    try:
        with rasterio.open(image_path) as src:
            red = src.read(4)
            green = src.read(3)
            blue = src.read(2)

        rgb = np.stack([red, green, blue], axis=-1)
        rgb = np.nan_to_num(rgb, nan=0.0)

        def pstretch(band):
            v = band[band > 0]
            if len(v) == 0:
                return np.zeros_like(band, dtype=np.uint8)
            lo, hi = np.percentile(v, 2), np.percentile(v, 98)
            if hi <= lo:
                hi = lo + 0.001
            return (np.clip((band - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)

        out = np.stack([pstretch(rgb[:, :, i]) for i in range(3)], axis=-1)
        pil = Image.fromarray(out, 'RGB')
        h, w = pil.size[1], pil.size[0]
        if h > max_size or w > max_size:
            s = max_size / max(h, w)
            pil = pil.resize((int(w * s), int(h * s)), Image.LANCZOS)
        return pil
    except:
        return None


def plot_patch_validity(validity_info, title="Patch Validity"):
    """Create a matplotlib figure showing patch validity as green/red grid."""
    mask = validity_info['mask']
    nv = validity_info['n_valid']
    nt = validity_info['n_total']

    fig, ax = plt.subplots(figsize=(6, 5))
    cmap = ListedColormap(['#e74c3c', '#2ecc71'])  # red=invalid, green=valid
    ax.imshow(mask.astype(int), cmap=cmap, vmin=0, vmax=1, interpolation='nearest')

    # Grid lines
    ax.set_xticks(np.arange(-0.5, mask.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, mask.shape[0], 1), minor=True)
    ax.grid(which='minor', color='black', linewidth=0.5)
    ax.tick_params(which='minor', size=0)

    ax.set_title(f"{title}\n{nv}/{nt} valid ({100*nv/nt:.1f}%)", fontsize=11)
    ax.set_xlabel("Patch Column")
    ax.set_ylabel("Patch Row")

    plt.tight_layout()
    return fig


# =============================================================================
# Main Processing Pipeline
# =============================================================================
def run_pipeline(aoi, start_date, end_date, cloudy_pixel_percentage=10, scale=10):
    """
    Main pipeline:
      Phase 1: Create cloud-free collection
      Phase 2: For each month - get metadata, create composite, gap-fill, download
      Phase 3: Patch validity analysis for each downloaded composite
    """
    st.session_state.processing_log = []
    add_log("Pipeline started")
    add_log(f"Date range: {start_date} to {end_date}, cloud%: {cloudy_pixel_percentage}")

    if st.session_state.current_temp_dir is None or not os.path.exists(st.session_state.current_temp_dir):
        st.session_state.current_temp_dir = tempfile.mkdtemp()
    temp_dir = st.session_state.current_temp_dir

    start_dt = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.datetime.strptime(end_date, '%Y-%m-%d')
    total_months = (end_dt.year - start_dt.year) * 12 + (end_dt.month - start_dt.month)

    extended_start = (start_dt - datetime.timedelta(days=31)).strftime('%Y-%m-%d')
    extended_end = (end_dt + datetime.timedelta(days=31)).strftime('%Y-%m-%d')

    # ── Phase 1 ──
    st.header("Phase 1: Cloud-Free Collection")
    st.info(f"☁️ Cloud mask: prob > {CLOUD_PROB_THRESHOLD}, CDI < {CDI_THRESHOLD}")
    add_log("Creating cloud-free collection")

    cfc = create_cloud_free_collection(aoi, extended_start, extended_end, cloudy_pixel_percentage)
    add_log("Cloud-free collection ready")

    # Store for later component downloads
    st.session_state._cloud_free_collection_obj = cfc
    st.session_state._aoi_obj = aoi

    # ── Phase 2 ──
    st.header("Phase 2: Monthly Composites")

    month_infos = []
    for mi in range(total_months):
        y = start_dt.year + (start_dt.month - 1 + mi) // 12
        m = (start_dt.month - 1 + mi) % 12 + 1
        month_infos.append({
            'month_name': f"{y}-{m:02d}",
            'month_index': mi,
            'origin': start_date,
        })

    st.info(f"📅 {len(month_infos)} months: {month_infos[0]['month_name']} → {month_infos[-1]['month_name']}")

    downloaded = {}
    statuses = {}
    all_meta = {}

    progress = st.progress(0)
    status_text = st.empty()

    for idx, mi in enumerate(month_infos):
        mn = mi['month_name']
        path, status, message, comp_meta = process_month(
            aoi, cfc, mi, temp_dir, scale, status_text
        )

        statuses[mn] = {'status': status, 'message': message}
        all_meta[mn] = comp_meta

        icon = {"no_data": "⚫", "skipped": "🟡", "complete": "🟢", "rejected": "🔴"}.get(status, "❓")
        st.write(f"{icon} **{mn}**: {status} — {message} | {len(comp_meta)} component images")

        if path:
            downloaded[mn] = path

        progress.progress((idx + 1) / len(month_infos))

    progress.empty()
    status_text.empty()

    # Summary
    st.divider()
    counts = {s: sum(1 for ms in statuses.values() if ms['status'] == s)
              for s in [STATUS_NO_DATA, STATUS_SKIPPED, STATUS_COMPLETE, STATUS_REJECTED]}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("✅ Complete", counts[STATUS_COMPLETE])
    c2.metric("🔴 Rejected", counts[STATUS_REJECTED])
    c3.metric("🟡 Skipped", counts[STATUS_SKIPPED])
    c4.metric("⚫ No Data", counts[STATUS_NO_DATA])

    st.session_state.downloaded_composites = downloaded
    st.session_state.month_statuses = statuses
    st.session_state.component_image_metadata = all_meta

    if not downloaded:
        st.error("❌ No composites downloaded!")
        return

    # ── Phase 3: Patch Validity ──
    st.header("Phase 3: Patch Validity Analysis")
    add_log(f"Analyzing patch validity for {len(downloaded)} composites")

    pv_maps = {}
    pv_progress = st.progress(0)
    sorted_months = sorted(downloaded.keys())

    for idx, mn in enumerate(sorted_months):
        pv = compute_patch_validity_map(downloaded[mn], PATCH_SIZE)
        if pv:
            pv_maps[mn] = pv
            add_log(f"{mn}: {pv['n_valid']}/{pv['n_total']} valid patches")
        else:
            add_log(f"{mn}: Patch analysis failed", "ERROR")
        pv_progress.progress((idx + 1) / len(sorted_months))

    pv_progress.empty()
    st.session_state.patch_validity_maps = pv_maps
    st.session_state.processing_complete = True
    st.session_state.processing_in_progress = False

    st.success(f"✅ Pipeline complete! {len(downloaded)} composites, {len(pv_maps)} patch maps")
    add_log("Pipeline complete")


# =============================================================================
# Display: Component Images Per Month
# =============================================================================
def display_month_components(month_name, metadata, composite_path):
    """Display component images for a single month in an expander."""
    status_info = st.session_state.month_statuses.get(month_name, {})
    status = status_info.get('status', 'unknown')
    message = status_info.get('message', '')
    icon = {"no_data": "⚫", "skipped": "🟡", "complete": "🟢", "rejected": "🔴"}.get(status, "❓")

    has_composite = composite_path is not None
    n_images = len(metadata)

    label = f"{icon} **{month_name}** — {status} — {n_images} images"
    if has_composite:
        label += " — 📥 Composite available"

    with st.expander(label, expanded=False):
        st.write(f"**Status**: {status} — {message}")
        st.write(f"**Component images**: {n_images}")

        # Show composite thumbnail if available
        if has_composite:
            col_thumb, col_info = st.columns([1, 2])
            thumb = generate_rgb_thumbnail(composite_path, 300)
            if thumb:
                col_thumb.image(thumb, caption=f"{month_name} composite (RGB)")
            col_info.write(f"**Composite file**: `{os.path.basename(composite_path)}`")

            # Download composite
            with open(composite_path, 'rb') as f:
                col_info.download_button(
                    "⬇️ Download Composite (12 bands, 10m)",
                    data=f.read(),
                    file_name=f"composite_{month_name}_12bands.tif",
                    mime="image/tiff",
                    key=f"dl_comp_{month_name}"
                )

        # Component image table
        if metadata:
            st.write("---")
            st.write("**Individual cloud-free scenes used in this composite:**")

            for i, m in enumerate(metadata):
                c1, c2, c3 = st.columns([2, 1, 1])
                c1.write(f"📷 **{m['date']}** | {m['spacecraft']} | Cloud: {m['cloud_pct']:.1f}%")
                c2.code(m['system_index'][:30] + "..." if len(m['system_index']) > 30 else m['system_index'],
                        language=None)

                # Download button for this individual image
                dl_key = f"dl_component_{month_name}_{i}"
                if c3.button(f"⬇️ Download", key=dl_key):
                    _download_component_on_demand(month_name, m, i)
        else:
            st.info("No component images for this month.")


def _download_component_on_demand(month_name, meta, idx):
    """Download a single component image on button click."""
    cfc = st.session_state.get('_cloud_free_collection_obj')
    aoi = st.session_state.get('_aoi_obj')

    if cfc is None or aoi is None:
        st.error("❌ Cloud-free collection not available. Please re-run the pipeline.")
        return

    temp_dir = st.session_state.current_temp_dir
    si = meta['system_index']
    safe_name = si.replace('/', '_').replace('\\', '_')
    out_path = os.path.join(temp_dir, f"component_{month_name}_{safe_name}.tif")

    if os.path.exists(out_path):
        ok, _ = validate_geotiff(out_path, len(SPECTRAL_BANDS))
        if ok:
            with open(out_path, 'rb') as f:
                st.download_button(
                    f"⬇️ Ready: {meta['date']}",
                    data=f.read(),
                    file_name=f"S2_{meta['date']}_{safe_name[:20]}_12bands.tif",
                    mime="image/tiff",
                    key=f"ready_{month_name}_{idx}"
                )
            return

    with st.spinner(f"Downloading {meta['date']} at 10m (12 bands)... This may take a minute."):
        status_ph = st.empty()
        try:
            img = cfc.filter(ee.Filter.eq('system:index', si)).first()
            info = img.getInfo()
            if info is None:
                st.error(f"❌ Image {si} not found in collection")
                return

            result = download_composite(img, aoi, out_path, f"comp_{meta['date']}", 10, status_ph)
            status_ph.empty()

            if result and os.path.exists(result):
                with open(result, 'rb') as f:
                    data = f.read()
                st.success(f"✅ Downloaded {meta['date']}")
                st.download_button(
                    f"⬇️ Save: {meta['date']}",
                    data=data,
                    file_name=f"S2_{meta['date']}_{safe_name[:20]}_12bands.tif",
                    mime="image/tiff",
                    key=f"save_{month_name}_{idx}"
                )
            else:
                st.error(f"❌ Download failed for {meta['date']}")
        except Exception as e:
            status_ph.empty()
            st.error(f"❌ Error: {e}")


# =============================================================================
# Display: Patch Validity Maps
# =============================================================================
def display_patch_validity_section():
    """Show patch validity maps for all downloaded composites."""
    pv_maps = st.session_state.patch_validity_maps
    downloaded = st.session_state.downloaded_composites

    if not pv_maps:
        st.info("No patch validity data available.")
        return

    st.subheader("🧩 Patch Validity Maps")
    st.write(f"Showing {PATCH_SIZE}×{PATCH_SIZE} patch validity for {len(pv_maps)} composites")

    # Summary table
    summary_data = []
    for mn in sorted(pv_maps.keys()):
        pv = pv_maps[mn]
        summary_data.append({
            'Month': mn,
            'Valid': pv['n_valid'],
            'Total': pv['n_total'],
            'Valid %': f"{100 * pv['n_valid'] / pv['n_total']:.1f}%",
            'Grid': f"{pv['n_patches_h']}×{pv['n_patches_w']}",
        })

    st.dataframe(summary_data, use_container_width=True)

    # Grid of patch maps
    sorted_months = sorted(pv_maps.keys())

    # Display in rows of 3
    for row_start in range(0, len(sorted_months), 3):
        cols = st.columns(3)
        for j in range(3):
            idx = row_start + j
            if idx >= len(sorted_months):
                break
            mn = sorted_months[idx]
            pv = pv_maps[mn]

            with cols[j]:
                fig = plot_patch_validity(pv, title=mn)
                st.pyplot(fig)
                plt.close(fig)

                # Download GeoTIFF of patch validity
                if mn in downloaded:
                    temp_dir = st.session_state.current_temp_dir
                    pv_tif_path = os.path.join(temp_dir, f"patch_validity_{mn}.tif")

                    if not os.path.exists(pv_tif_path):
                        save_patch_validity_geotiff(pv, downloaded[mn], pv_tif_path, PATCH_SIZE)

                    if os.path.exists(pv_tif_path):
                        with open(pv_tif_path, 'rb') as f:
                            st.download_button(
                                f"⬇️ Patch Map GeoTIFF",
                                data=f.read(),
                                file_name=f"patch_validity_{mn}.tif",
                                mime="image/tiff",
                                key=f"dl_pv_{mn}"
                            )


# =============================================================================
# Main App
# =============================================================================
def main():
    st.title("🛰️ Sentinel-2 Composite Explorer")
    st.caption("Explore which images compose each monthly median, download individual scenes, and inspect patch validity")

    # ── Earth Engine ──
    ee_ok, ee_msg = initialize_earth_engine()
    if not ee_ok:
        st.error(ee_msg)
        st.stop()
    st.sidebar.success(ee_msg)

    # ── Sidebar ──
    st.sidebar.header("⚙️ Parameters")
    cloudy_pct = st.sidebar.slider("Max Cloud % (metadata)", 0, 50, 10, 5,
                                   disabled=st.session_state.processing_in_progress)

    st.sidebar.header("🗂️ Cache")
    if st.session_state.downloaded_composites:
        st.sidebar.success(f"📥 {len(st.session_state.downloaded_composites)} composites")
    if st.session_state.patch_validity_maps:
        st.sidebar.success(f"🧩 {len(st.session_state.patch_validity_maps)} patch maps")
    if not st.session_state.downloaded_composites and not st.session_state.patch_validity_maps:
        st.sidebar.info("No cached data")

    if st.sidebar.button("🗑️ Clear All Cache", disabled=st.session_state.processing_in_progress):
        for k, v in _defaults.items():
            if isinstance(v, dict):
                st.session_state[k] = {}
            elif isinstance(v, list):
                st.session_state[k] = []
            else:
                st.session_state[k] = v
        st.session_state.processing_complete = False
        st.session_state.processing_in_progress = False
        # Also clear internal objects
        for k in ['_cloud_free_collection_obj', '_aoi_obj']:
            if k in st.session_state:
                del st.session_state[k]
        st.rerun()

    # ── Region ──
    st.header("1️⃣ Region")
    if not st.session_state.processing_in_progress:
        m = folium.Map(location=[35.6892, 51.3890], zoom_start=8)
        plugins.Draw(export=True, position='topleft', draw_options={
            'polyline': False, 'rectangle': True, 'polygon': True,
            'circle': False, 'marker': False, 'circlemarker': False
        }).add_to(m)
        folium.TileLayer(tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
                         attr='Google', name='Satellite').add_to(m)
        folium.LayerControl().add_to(m)

        map_data = st_folium(m, width=800, height=500, key="main_map")

        if map_data and map_data.get('last_active_drawing'):
            geom = map_data['last_active_drawing'].get('geometry', {})
            if geom.get('type') == 'Polygon':
                st.session_state.last_drawn_polygon = Polygon(geom['coordinates'][0])
                st.success("✅ Region selected")

        if st.button("💾 Save Region"):
            if st.session_state.last_drawn_polygon:
                dup = any(p.equals(st.session_state.last_drawn_polygon) for p in st.session_state.drawn_polygons)
                if not dup:
                    st.session_state.drawn_polygons.append(st.session_state.last_drawn_polygon)
                    st.success("✅ Saved!")
                    st.rerun()
                else:
                    st.warning("Already saved")
            else:
                st.warning("Draw a region first")
    else:
        st.info("🔒 Map locked during processing")

    # Saved regions
    if st.session_state.drawn_polygons:
        st.subheader("📍 Saved Regions")
        for i, p in enumerate(st.session_state.drawn_polygons):
            c1, c2 = st.columns([4, 1])
            cen = p.centroid
            c1.write(f"**Region {i+1}**: ~{p.area * 111 * 111:.2f} km² | ({cen.y:.4f}, {cen.x:.4f})")
            if c2.button("🗑️", key=f"del_{i}", disabled=st.session_state.processing_in_progress):
                st.session_state.drawn_polygons.pop(i)
                if st.session_state.selected_region_index >= len(st.session_state.drawn_polygons):
                    st.session_state.selected_region_index = max(0, len(st.session_state.drawn_polygons) - 1)
                st.rerun()

    # ── Date ──
    st.header("2️⃣ Time Period")
    c1, c2 = st.columns(2)
    start = c1.date_input("Start (inclusive)", value=date(2024, 1, 1),
                          disabled=st.session_state.processing_in_progress)
    end = c2.date_input("End (exclusive)", value=date(2025, 1, 1),
                        disabled=st.session_state.processing_in_progress)

    if start >= end:
        st.error("Invalid dates")
        st.stop()

    months = (end.year - start.year) * 12 + (end.month - start.month)
    first_mn = f"{start.year}-{start.month:02d}"
    last_y = start.year + (start.month - 1 + months - 1) // 12
    last_m = (start.month - 1 + months - 1) % 12 + 1
    last_mn = f"{last_y}-{last_m:02d}"
    st.info(f"📅 **{months} months**: {first_mn} → {last_mn}")

    # ── Process ──
    st.header("3️⃣ Process")

    selected_polygon = None
    if st.session_state.drawn_polygons:
        opts = [f"Region {i+1} (~{p.area*111*111:.2f} km²)" for i, p in enumerate(st.session_state.drawn_polygons)]
        if st.session_state.selected_region_index >= len(opts):
            st.session_state.selected_region_index = 0
        sel = st.selectbox("🎯 Select Region", range(len(opts)), format_func=lambda i: opts[i],
                           index=st.session_state.selected_region_index,
                           disabled=st.session_state.processing_in_progress)
        st.session_state.selected_region_index = sel
        selected_polygon = st.session_state.drawn_polygons[sel]
    elif st.session_state.last_drawn_polygon:
        selected_polygon = st.session_state.last_drawn_polygon
        st.info("Using unsaved drawn region")
    else:
        st.warning("⚠️ Draw a region first")

    st.divider()

    if st.button("🚀 Start Processing", type="primary",
                 disabled=st.session_state.processing_in_progress or selected_polygon is None):
        st.session_state.processing_in_progress = True
        st.session_state.processing_complete = False
        st.session_state.downloaded_composites = {}
        st.session_state.month_statuses = {}
        st.session_state.component_image_metadata = {}
        st.session_state.patch_validity_maps = {}

        coords = list(selected_polygon.exterior.coords)
        aoi = ee.Geometry.Polygon([coords])

        run_pipeline(
            aoi,
            start.strftime('%Y-%m-%d'),
            end.strftime('%Y-%m-%d'),
            cloudy_pct,
            10
        )
        st.rerun()

    # ── Results ──
    if st.session_state.processing_complete:
        st.divider()
        st.header("📊 Results")

        # Tab layout for results
        tab_comp, tab_patch, tab_log = st.tabs([
            "📷 Component Images",
            "🧩 Patch Validity",
            "📋 Processing Log"
        ])

        with tab_comp:
            st.subheader("Monthly Composites & Component Images")
            st.write("Expand each month to see which individual scenes were used and download them at full resolution (12 bands, 10m).")

            all_meta = st.session_state.component_image_metadata
            downloaded = st.session_state.downloaded_composites
            statuses = st.session_state.month_statuses

            for mn in sorted(all_meta.keys()):
                meta = all_meta[mn]
                comp_path = downloaded.get(mn)
                display_month_components(mn, meta, comp_path)

        with tab_patch:
            display_patch_validity_section()

        with tab_log:
            st.subheader("📋 Processing Log")
            log = st.session_state.processing_log
            if log:
                for entry in log:
                    color = {'INFO': 'gray', 'WARNING': 'orange', 'ERROR': 'red'}.get(entry['level'], 'gray')
                    st.markdown(f"<span style='color:{color}'>[{entry['ts']}] [{entry['level']}] {entry['msg']}</span>",
                               unsafe_allow_html=True)
            else:
                st.info("No log entries")


if __name__ == "__main__":
    main()
