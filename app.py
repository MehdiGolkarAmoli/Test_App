"""
Sentinel-2 Monthly Composite Explorer  (v2 — with robust resume)
=================================================================
PURPOSE:
  - Show which individual images are used to create each monthly median composite
  - Allow downloading individual scenes at 10 m / 12 bands
  - Show per-month patch-validity maps (green / red)
  - NO classification model

RESUME STRATEGY:
  Every completed month is persisted in st.session_state immediately:
    • month_statuses[month] = {status, message}
    • component_image_metadata[month] = [...]
    • downloaded_composites[month] = filepath
    • patch_validity_maps[month] = {...}
  On resume the pipeline recreates the (lazy) GEE collection from the stored
  processing_config and only processes months that are missing or failed.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from shapely.geometry import Polygon
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

# ═════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═════════════════════════════════════════════════════════════════════════════
SPECTRAL_BANDS = ['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B11','B12']
PATCH_SIZE = 224
MAX_RETRIES = 3
RETRY_DELAY_BASE = 2
DOWNLOAD_TIMEOUT = 120
CHUNK_SIZE = 8192
MIN_BAND_FILE_SIZE = 10_000
MIN_MULTIBAND_FILE_SIZE = 100_000
CLOUD_PROB_THRESHOLD = 50
CDI_THRESHOLD = -0.5
MAX_MASKED_PERCENT_FOR_GAPFILL = 30
STATUS_NO_DATA  = "no_data"
STATUS_SKIPPED  = "skipped"
STATUS_COMPLETE = "complete"
STATUS_REJECTED = "rejected"

# ═════════════════════════════════════════════════════════════════════════════
# SESSION STATE — every piece of progress lives here and survives reruns
# ═════════════════════════════════════════════════════════════════════════════
_DEFAULTS = {
    # UI
    'drawn_polygons': [],
    'last_drawn_polygon': None,
    'selected_region_index': 0,
    # Processing flags
    'processing_in_progress': False,
    'processing_complete': False,
    # Config (serialisable — no ee objects)
    'processing_config': None,   # {polygon_coords, start_date, end_date, cloudy_pct}
    'current_temp_dir': None,
    # Per-month results — INCREMENTALLY updated
    'downloaded_composites': {},        # month → filepath
    'month_statuses': {},               # month → {status, message}
    'component_image_metadata': {},     # month → [list of dicts]
    'patch_validity_maps': {},          # month → {mask, n_valid, …}
    # Log
    'processing_log': [],
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        if isinstance(_v, (dict, list)):
            st.session_state[_k] = type(_v)(_v)   # shallow copy
        else:
            st.session_state[_k] = _v


# ═════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═════════════════════════════════════════════════════════════════════════════
def add_log(msg, level="INFO"):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    st.session_state.processing_log.append({'ts': ts, 'level': level, 'msg': msg})
    print(f"[{ts}] [{level}] {msg}")


# ═════════════════════════════════════════════════════════════════════════════
# FILE VALIDATION
# ═════════════════════════════════════════════════════════════════════════════
def validate_geotiff(path, expected_bands=1):
    try:
        if not os.path.exists(path): return False, "Missing"
        sz = os.path.getsize(path)
        if sz < (MIN_BAND_FILE_SIZE if expected_bands == 1 else MIN_MULTIBAND_FILE_SIZE):
            return False, f"Small ({sz}B)"
        with rasterio.open(path) as src:
            if src.count < expected_bands: return False, f"Bands {src.count}<{expected_bands}"
        return True, "OK"
    except Exception as e:
        return False, str(e)


# ═════════════════════════════════════════════════════════════════════════════
# EARTH ENGINE AUTH
# ═════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def initialize_earth_engine():
    try:
        ee.Initialize(); return True, "EE initialised"
    except Exception:
        try:
            b64 = os.environ.get('GOOGLE_EARTH_ENGINE_KEY_BASE64')
            if b64:
                kd = json.loads(base64.b64decode(b64).decode())
                kf = tempfile.NamedTemporaryFile(suffix='.json', delete=False)
                with open(kf.name,'w') as f: json.dump(kd,f)
                ee.Initialize(ee.ServiceAccountCredentials(kd['client_email'], kf.name))
                os.unlink(kf.name); return True, "Service-account auth"
            ee.Authenticate(); ee.Initialize(); return True, "Authenticated"
        except Exception as e:
            return False, f"Auth failed: {e}"


# ═════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════════════
def get_utm_zone(lon): return math.floor((lon+180)/6)+1


def _month_list(start_date, end_date):
    """Return list of {month_name, month_index, origin}."""
    sd = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    ed = datetime.datetime.strptime(end_date, '%Y-%m-%d')
    n = (ed.year-sd.year)*12 + (ed.month-sd.month)
    out = []
    for i in range(n):
        y = sd.year + (sd.month-1+i)//12
        m = (sd.month-1+i)%12+1
        out.append({'month_name': f"{y}-{m:02d}", 'month_index': i, 'origin': start_date})
    return out


# ═════════════════════════════════════════════════════════════════════════════
# GEE CLOUD-FREE COLLECTION  (lazy server-side object — cheap to recreate)
# ═════════════════════════════════════════════════════════════════════════════
def _build_cloud_free_collection(aoi, extended_start, extended_end, cloudy_pct):
    s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
          .filterBounds(aoi).filterDate(extended_start, extended_end)
          .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloudy_pct))
          .select(SPECTRAL_BANDS+['SCL']))
    cp = (ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
          .filterBounds(aoi).filterDate(extended_start, extended_end))
    joined = ee.Join.saveFirst('cloud_probability').apply(
        primary=s2, secondary=cp,
        condition=ee.Filter.equals(leftField='system:index', rightField='system:index'))
    def _add(f):
        img = ee.Image(f); return img.addBands(ee.Image(img.get('cloud_probability')))
    col = ee.ImageCollection(joined.map(_add))
    def _mask(img):
        prob = img.select('probability')
        cdi = ee.Algorithms.Sentinel2.CDI(img)
        cloud = prob.gt(CLOUD_PROB_THRESHOLD).And(cdi.lt(CDI_THRESHOLD))
        dilated = cloud.focal_max(kernel=ee.Kernel.circle(20,'meters'), iterations=2)
        masked = img.updateMask(dilated.Not())
        scaled = masked.select(SPECTRAL_BANDS).multiply(0.0001).clip(aoi)
        return scaled.copyProperties(img, ['system:time_start','system:index',
                                            'CLOUDY_PIXEL_PERCENTAGE','SPACECRAFT_NAME'])
    return col.map(_mask)


def get_cloud_free_collection(config):
    """(Re)build from serialisable config — safe across reruns."""
    coords = config['polygon_coords']
    aoi = ee.Geometry.Polygon([coords])
    sd = datetime.datetime.strptime(config['start_date'], '%Y-%m-%d')
    ed = datetime.datetime.strptime(config['end_date'], '%Y-%m-%d')
    ext_s = (sd - datetime.timedelta(days=31)).strftime('%Y-%m-%d')
    ext_e = (ed + datetime.timedelta(days=31)).strftime('%Y-%m-%d')
    return _build_cloud_free_collection(aoi, ext_s, ext_e, config['cloudy_pct']), aoi


# ═════════════════════════════════════════════════════════════════════════════
# DOWNLOAD (band-by-band with retry — unchanged from original)
# ═════════════════════════════════════════════════════════════════════════════
def _download_band(image, band, aoi, out, scale=10):
    region = aoi.bounds().getInfo()['coordinates']
    tmp = out+'.tmp'
    for f in [tmp]: 
        if os.path.exists(f): os.remove(f)
    if os.path.exists(out):
        ok,_ = validate_geotiff(out,1)
        if ok: return True,"cached"
        os.remove(out)
    last = None
    for attempt in range(MAX_RETRIES):
        try:
            url = image.select(band).getDownloadURL({'scale':scale,'region':region,'format':'GEO_TIFF','bands':[band]})
            r = requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT)
            if r.status_code != 200: raise Exception(f"HTTP {r.status_code}")
            if 'text/html' in r.headers.get('content-type',''): raise Exception("Rate limit")
            sz=0
            with open(tmp,'wb') as f:
                for c in r.iter_content(CHUNK_SIZE):
                    if c: f.write(c); sz+=len(c)
            if sz < MIN_BAND_FILE_SIZE: raise Exception(f"Small ({sz}B)")
            ok,msg = validate_geotiff(tmp,1)
            if ok: os.replace(tmp,out); return True,"ok"
            raise Exception(msg)
        except Exception as e: last=str(e)
        for f in [out,tmp]:
            if os.path.exists(f):
                try: os.remove(f)
                except: pass
        if attempt < MAX_RETRIES-1: time.sleep(RETRY_DELAY_BASE**(attempt+1))
    return False, last


def download_composite(composite, aoi, out_path, label, scale=10, status_ph=None):
    if os.path.exists(out_path):
        ok,_ = validate_geotiff(out_path, len(SPECTRAL_BANDS))
        if ok: return out_path
        os.remove(out_path)
    bd = os.path.join(os.path.dirname(out_path), f"bands_{label}")
    os.makedirs(bd, exist_ok=True)
    files=[]; fails=[]
    for i,b in enumerate(SPECTRAL_BANDS):
        bf = os.path.join(bd, f"{b}.tif")
        if status_ph: status_ph.text(f"📥 {label}: {b} ({i+1}/{len(SPECTRAL_BANDS)})…")
        ok,err = _download_band(composite,b,aoi,bf,scale)
        (files if ok else fails).append(bf if ok else f"{b}:{err}")
    if fails:
        st.error(f"❌ {label}: {'; '.join(fails)}")
        return None
    if status_ph: status_ph.text(f"📦 {label}: merging…")
    with rasterio.open(files[0]) as s: meta=s.meta.copy()
    meta.update(count=len(files))
    with rasterio.open(out_path,'w',**meta) as dst:
        for i,bf in enumerate(files):
            with rasterio.open(bf) as s: dst.write(s.read(1),i+1)
    ok,msg = validate_geotiff(out_path, len(SPECTRAL_BANDS))
    if ok: return out_path
    if os.path.exists(out_path): os.remove(out_path)
    return None


# ═════════════════════════════════════════════════════════════════════════════
# COMPONENT IMAGE METADATA
# ═════════════════════════════════════════════════════════════════════════════
def _get_component_metadata(cfc, month_start, month_end):
    monthly = cfc.filterDate(month_start, month_end)
    cnt = monthly.size().getInfo()
    if cnt == 0: return []
    lst = monthly.toList(min(cnt,50))
    out=[]
    for i in range(min(cnt,50)):
        try:
            img = ee.Image(lst.get(i))
            p = img.toDictionary(['system:time_start','system:index',
                                  'CLOUDY_PIXEL_PERCENTAGE','SPACECRAFT_NAME']).getInfo()
            ts = p.get('system:time_start',0)
            dt = datetime.datetime.utcfromtimestamp(ts/1000)
            out.append({
                'system_index': p.get('system:index',f'unk_{i}'),
                'date': dt.strftime('%Y-%m-%d'),
                'cloud_pct': p.get('CLOUDY_PIXEL_PERCENTAGE',-1),
                'spacecraft': p.get('SPACECRAFT_NAME','?'),
            })
        except: pass
    out.sort(key=lambda x: x['date'])
    return out


# ═════════════════════════════════════════════════════════════════════════════
# PROCESS ONE MONTH  (composite + gap-fill + download)
# returns (path|None, status, message, metadata_list)
# ═════════════════════════════════════════════════════════════════════════════
def process_single_month(aoi, cfc, month_info, temp_dir, scale=10, status_ph=None):
    mn = month_info['month_name']
    mi = month_info['month_index']
    origin = month_info['origin']
    out_file = os.path.join(temp_dir, f"sentinel2_{mn}.tif")

    # ── cached file on disk? ──
    if os.path.exists(out_file):
        ok,_ = validate_geotiff(out_file, len(SPECTRAL_BANDS))
        if ok:
            add_log(f"{mn}: disk cache hit")
            od = ee.Date(origin)
            ms = od.advance(mi,'month'); me = od.advance(ee.Number(mi).add(1),'month')
            meta = _get_component_metadata(cfc, ms, me)
            return out_file, STATUS_COMPLETE, "Cached (disk)", meta

    try:
        od = ee.Date(origin)
        ms = od.advance(mi,'month')
        me = od.advance(ee.Number(mi).add(1),'month')
        mm = ms.advance(15,'day')

        if status_ph: status_ph.text(f"📥 {mn}: analysing…")
        monthly = cfc.filterDate(ms, me)
        ic = monthly.size().getInfo()
        add_log(f"{mn}: {ic} cloud-free images")

        if status_ph: status_ph.text(f"📥 {mn}: metadata ({ic} images)…")
        meta = _get_component_metadata(cfc, ms, me)

        if ic == 0:
            return None, STATUS_NO_DATA, "No images", meta

        # frequency & composite
        freq = monthly.map(lambda img: ee.Image(1).updateMask(img.select('B4').mask()).unmask(0).toInt()).sum().toInt().rename('frequency')
        composite = monthly.median()

        m_stats = freq.eq(0).reduceRegion(ee.Reducer.sum(), aoi, 10, maxPixels=1e13)
        t_stats = freq.gte(0).reduceRegion(ee.Reducer.sum(), aoi, 10, maxPixels=1e13)
        mc = ee.Number(m_stats.get('frequency')).getInfo()
        tc = ee.Number(t_stats.get('frequency')).getInfo()
        if tc == 0: return None, STATUS_NO_DATA, "No pixels", meta
        mp = mc/tc*100
        add_log(f"{mn}: masked {mp:.2f}%")

        if mp > MAX_MASKED_PERCENT_FOR_GAPFILL:
            return None, STATUS_SKIPPED, f"Masked {mp:.1f}%>{MAX_MASKED_PERCENT_FOR_GAPFILL}%", meta

        if mp == 0:
            if status_ph: status_ph.text(f"📥 {mn}: 0% masked, downloading…")
            p = download_composite(composite, aoi, out_file, mn, scale, status_ph)
            return (p, STATUS_COMPLETE, "0% masked", meta) if p else (None, STATUS_REJECTED, "Download failed", meta)

        # ── gap-fill ──
        add_log(f"{mn}: gap-filling ({mp:.1f}%)")
        if status_ph: status_ph.text(f"📥 {mn}: gap-filling…")
        gap = freq.eq(0)
        mmm = mm.millis()
        ps = od.advance(ee.Number(mi).subtract(1),'month')
        pe = ms
        fs = me
        fe = od.advance(ee.Number(mi).add(2),'month')
        cands = cfc.filterDate(ps,pe).merge(cfc.filterDate(fs,fe))
        cands = cands.map(lambda img: img.set('td', ee.Number(img.get('system:time_start')).subtract(mmm).abs()))
        cands = cands.sort('td', True)
        nc = cands.size().getInfo()
        add_log(f"{mn}: {nc} gap-fill candidates")
        if nc == 0:
            return None, STATUS_REJECTED, f"No gap candidates, {mp:.1f}% masked", meta

        mos = cands.mosaic().select(SPECTRAL_BANDS)
        has = mos.select('B4').mask()
        fill = gap.And(has); still = gap.And(has.Not())
        filled = composite.unmask(mos.updateMask(fill))

        fs_img = ee.Image.constant(0).clip(aoi).toInt8().where(fill,1).where(still,2).rename('fs')
        sr = fs_img.eq(2).reduceRegion(ee.Reducer.sum(), aoi, 10, maxPixels=1e13).get('fs')
        sc = ee.Number(ee.Algorithms.If(ee.Algorithms.IsEqual(sr,None),0,sr)).getInfo()

        if sc == 0:
            if status_ph: status_ph.text(f"📥 {mn}: gap-filled → downloading…")
            p = download_composite(filled, aoi, out_file, mn, scale, status_ph)
            return (p, STATUS_COMPLETE, f"Gap-filled (was {mp:.1f}%)", meta) if p else (None, STATUS_REJECTED, "DL fail post gap", meta)
        sp = sc/tc*100
        return None, STATUS_REJECTED, f"{sp:.1f}% still masked after gap-fill", meta

    except Exception as e:
        add_log(f"{mn}: ERROR {e}", "ERROR")
        return None, STATUS_NO_DATA, f"Error: {e}", []


# ═════════════════════════════════════════════════════════════════════════════
# PATCH VALIDITY
# ═════════════════════════════════════════════════════════════════════════════
def _check_patch(patch):
    if np.any(np.isnan(patch)): return False
    if np.all(patch==0): return False
    if (np.sum(patch==0)/patch.size)*100 > 0: return False
    if patch.ndim==3:
        for b in range(patch.shape[-1]):
            if np.all(patch[:,:,b]==0): return False
    return True


def compute_patch_validity(image_path, ps=PATCH_SIZE):
    try:
        with rasterio.open(image_path) as s: data=s.read()
        img = np.moveaxis(data,0,-1)
        h,w,c = img.shape
        nh = int(np.ceil(h/ps)*ps); nw = int(np.ceil(w/ps)*ps)
        if h!=nh or w!=nw:
            pad = np.zeros((nh,nw,c), dtype=img.dtype); pad[:h,:w,:]=img; img=pad
        patches = patchify(img,(ps,ps,c),step=ps)
        nph,npw = patches.shape[:2]
        mask = np.zeros((nph,npw),dtype=bool)
        for i in range(nph):
            for j in range(npw):
                mask[i,j] = _check_patch(patches[i,j,0])
        return {'mask':mask, 'n_valid':int(np.sum(mask)), 'n_total':nph*npw,
                'n_patches_h':nph, 'n_patches_w':npw, 'original_size':(h,w)}
    except Exception as e:
        add_log(f"Patch error: {e}","ERROR"); return None


def save_patch_geotiff(pv, ref_path, out_path, ps=PATCH_SIZE):
    try:
        mask = pv['mask']; oh,ow = pv['original_size']
        px = np.repeat(np.repeat(mask.astype(np.uint8),ps,0),ps,1)[:oh,:ow]
        with rasterio.open(ref_path) as s: meta=s.meta.copy()
        meta.update(count=1, dtype='uint8', height=oh, width=ow)
        with rasterio.open(out_path,'w',**meta) as d: d.write(px,1)
        return out_path
    except Exception as e:
        add_log(f"Save PV tif: {e}","ERROR"); return None


# ═════════════════════════════════════════════════════════════════════════════
# RGB THUMBNAIL
# ═════════════════════════════════════════════════════════════════════════════
def _rgb_thumb(path, max_sz=256):
    try:
        with rasterio.open(path) as s:
            r,g,b = s.read(4), s.read(3), s.read(2)
        rgb = np.nan_to_num(np.stack([r,g,b],-1),nan=0.)
        def ps(band):
            v=band[band>0]
            if len(v)==0: return np.zeros_like(band,dtype=np.uint8)
            lo,hi=np.percentile(v,2),np.percentile(v,98)
            if hi<=lo: hi=lo+.001
            return (np.clip((band-lo)/(hi-lo),0,1)*255).astype(np.uint8)
        out = np.stack([ps(rgb[:,:,i]) for i in range(3)],-1)
        pil = Image.fromarray(out,'RGB')
        h,w=pil.size[1],pil.size[0]
        if max(h,w)>max_sz:
            s=max_sz/max(h,w); pil=pil.resize((int(w*s),int(h*s)),Image.LANCZOS)
        return pil
    except: return None


def _plot_pv(pv, title=""):
    mask=pv['mask']; nv=pv['n_valid']; nt=pv['n_total']
    fig,ax=plt.subplots(figsize=(6,5))
    ax.imshow(mask.astype(int), cmap=ListedColormap(['#e74c3c','#2ecc71']),
              vmin=0, vmax=1, interpolation='nearest')
    ax.set_xticks(np.arange(-.5,mask.shape[1],1),minor=True)
    ax.set_yticks(np.arange(-.5,mask.shape[0],1),minor=True)
    ax.grid(which='minor',color='black',linewidth=.5)
    ax.tick_params(which='minor',size=0)
    ax.set_title(f"{title}\n{nv}/{nt} valid ({100*nv/nt:.1f}%)",fontsize=11)
    ax.set_xlabel("Patch Col"); ax.set_ylabel("Patch Row")
    plt.tight_layout(); return fig


# ═════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE  (with incremental save + resume)
# ═════════════════════════════════════════════════════════════════════════════
def run_pipeline(config, resume=False):
    """
    Runs (or resumes) the full pipeline.
    `config` is the serialisable dict from session_state.processing_config.
    """
    if not resume:
        st.session_state.processing_log = []
    add_log(f"{'RESUME' if resume else 'START'} pipeline")

    # Temp dir — reuse across resumes
    if st.session_state.current_temp_dir is None or not os.path.exists(st.session_state.current_temp_dir):
        st.session_state.current_temp_dir = tempfile.mkdtemp()
    temp_dir = st.session_state.current_temp_dir

    months = _month_list(config['start_date'], config['end_date'])
    month_names_set = {m['month_name'] for m in months}

    # ── Phase 1: GEE collection (cheap lazy object) ──
    st.header("Phase 1: Cloud-Free Collection")
    add_log("Building GEE collection")
    cfc, aoi = get_cloud_free_collection(config)
    st.success(f"☁️ Collection ready (prob>{CLOUD_PROB_THRESHOLD}, CDI<{CDI_THRESHOLD})")

    # ── Phase 2: Monthly composites ──
    st.header("Phase 2: Monthly Composites")
    st.info(f"📅 {len(months)} months: {months[0]['month_name']} → {months[-1]['month_name']}")

    # Figure out which months still need processing
    already_done = set()
    for mn in month_names_set:
        status_info = st.session_state.month_statuses.get(mn)
        if status_info is None:
            continue
        s = status_info['status']
        # "complete" with a valid file on disk → truly done
        if s == STATUS_COMPLETE:
            fp = st.session_state.downloaded_composites.get(mn)
            if fp and os.path.exists(fp):
                ok,_ = validate_geotiff(fp, len(SPECTRAL_BANDS))
                if ok:
                    already_done.add(mn)
                    continue
            # file gone → need to re-process
        # non-downloadable statuses (no_data, skipped, rejected) → done (nothing to download)
        elif s in (STATUS_NO_DATA, STATUS_SKIPPED, STATUS_REJECTED):
            already_done.add(mn)

    to_process = [m for m in months if m['month_name'] not in already_done]

    # Show cached results
    if already_done:
        with st.expander(f"✅ {len(already_done)} months already processed (cached)", expanded=False):
            for mn in sorted(already_done):
                si = st.session_state.month_statuses.get(mn, {})
                s = si.get('status','?')
                msg = si.get('message','')
                ic = {"no_data":"⚫","skipped":"🟡","complete":"🟢","rejected":"🔴"}.get(s,"❓")
                n_comp = len(st.session_state.component_image_metadata.get(mn, []))
                st.write(f"{ic} **{mn}**: {s} — {msg} | {n_comp} images")

    if not to_process:
        st.success("All months already processed!")
    else:
        st.info(f"📥 {len(to_process)} months to process")
        progress = st.progress(0)
        status_ph = st.empty()

        for idx, mi in enumerate(to_process):
            mn = mi['month_name']

            # Double-check — another rerun might have written it
            if mn in st.session_state.month_statuses:
                si = st.session_state.month_statuses[mn]
                if si['status'] in (STATUS_NO_DATA, STATUS_SKIPPED, STATUS_REJECTED):
                    progress.progress((idx+1)/len(to_process))
                    continue
                if si['status'] == STATUS_COMPLETE:
                    fp = st.session_state.downloaded_composites.get(mn)
                    if fp and os.path.exists(fp):
                        ok,_ = validate_geotiff(fp, len(SPECTRAL_BANDS))
                        if ok:
                            progress.progress((idx+1)/len(to_process))
                            continue

            path, status, message, comp_meta = process_single_month(
                aoi, cfc, mi, temp_dir, 10, status_ph
            )

            # ── SAVE IMMEDIATELY to session state ──
            st.session_state.month_statuses[mn] = {'status': status, 'message': message}
            st.session_state.component_image_metadata[mn] = comp_meta
            if path:
                st.session_state.downloaded_composites[mn] = path

            ic = {"no_data":"⚫","skipped":"🟡","complete":"🟢","rejected":"🔴"}.get(status,"❓")
            st.write(f"{ic} **{mn}**: {status} — {message} | {len(comp_meta)} images")

            progress.progress((idx+1)/len(to_process))

        progress.empty()
        status_ph.empty()

    # Summary
    st.divider()
    all_statuses = {mn: st.session_state.month_statuses.get(mn,{}).get('status','?')
                    for mn in month_names_set}
    counts = {s: sum(1 for v in all_statuses.values() if v==s)
              for s in [STATUS_COMPLETE, STATUS_REJECTED, STATUS_SKIPPED, STATUS_NO_DATA]}
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("✅ Complete", counts[STATUS_COMPLETE])
    c2.metric("🔴 Rejected", counts[STATUS_REJECTED])
    c3.metric("🟡 Skipped", counts[STATUS_SKIPPED])
    c4.metric("⚫ No Data", counts[STATUS_NO_DATA])

    downloaded = {mn: fp for mn,fp in st.session_state.downloaded_composites.items()
                  if mn in month_names_set}

    if not downloaded:
        st.error("❌ No composites downloaded!")
        st.session_state.processing_in_progress = False
        return

    # ── Phase 3: Patch validity (only for new ones) ──
    st.header("Phase 3: Patch Validity")
    need_pv = [mn for mn in sorted(downloaded) if mn not in st.session_state.patch_validity_maps]
    if need_pv:
        add_log(f"Computing patch validity for {len(need_pv)} months")
        pvp = st.progress(0)
        for idx, mn in enumerate(need_pv):
            pv = compute_patch_validity(downloaded[mn])
            if pv:
                st.session_state.patch_validity_maps[mn] = pv
                add_log(f"{mn}: {pv['n_valid']}/{pv['n_total']} valid")
            pvp.progress((idx+1)/len(need_pv))
        pvp.empty()
    else:
        st.success("Patch validity already computed for all composites")

    st.session_state.processing_complete = True
    st.session_state.processing_in_progress = False
    st.success(f"✅ Done — {len(downloaded)} composites, {len(st.session_state.patch_validity_maps)} patch maps")
    add_log("Pipeline finished")


# ═════════════════════════════════════════════════════════════════════════════
# DISPLAY: component images per month
# ═════════════════════════════════════════════════════════════════════════════
def _display_month(mn):
    si = st.session_state.month_statuses.get(mn,{})
    status = si.get('status','?')
    message = si.get('message','')
    meta = st.session_state.component_image_metadata.get(mn,[])
    comp = st.session_state.downloaded_composites.get(mn)
    ic = {"no_data":"⚫","skipped":"🟡","complete":"🟢","rejected":"🔴"}.get(status,"❓")

    label = f"{ic} {mn} — {status} — {len(meta)} images"
    if comp: label += " — 📥 composite"

    with st.expander(label, expanded=False):
        st.write(f"**Status**: {status} — {message}")

        if comp:
            c1,c2 = st.columns([1,2])
            thumb = _rgb_thumb(comp, 300)
            if thumb: c1.image(thumb, caption=f"{mn} composite")
            c2.write(f"`{os.path.basename(comp)}`")
            with open(comp,'rb') as f:
                c2.download_button("⬇️ Composite (12 bands 10m)", f.read(),
                                   f"composite_{mn}_12bands.tif", "image/tiff",
                                   key=f"dlc_{mn}")

        if meta:
            st.markdown("---")
            st.write("**Component scenes (cloud-free, used in median):**")
            for i, m in enumerate(meta):
                c1,c2,c3 = st.columns([3,1,1])
                c1.write(f"📷 **{m['date']}** | {m['spacecraft']} | Cloud: {m['cloud_pct']:.1f}%")
                c2.code(m['system_index'][:25]+"…" if len(m['system_index'])>25 else m['system_index'])
                if c3.button("⬇️ Download", key=f"dlcomp_{mn}_{i}"):
                    _download_component(mn, m, i)
        else:
            st.info("No component images")


def _download_component(mn, meta, idx):
    """On-demand download of a single component scene."""
    config = st.session_state.processing_config
    if config is None:
        st.error("No processing config — please re-run"); return

    cfc, aoi = get_cloud_free_collection(config)
    td = st.session_state.current_temp_dir
    si = meta['system_index']
    safe = si.replace('/','_').replace('\\','_')
    out = os.path.join(td, f"comp_{mn}_{safe}.tif")

    if os.path.exists(out):
        ok,_ = validate_geotiff(out, len(SPECTRAL_BANDS))
        if ok:
            with open(out,'rb') as f:
                st.download_button(f"✅ Ready: {meta['date']}", f.read(),
                                   f"S2_{meta['date']}_{safe[:20]}_12bands.tif",
                                   "image/tiff", key=f"rdy_{mn}_{idx}")
            return

    with st.spinner(f"Downloading {meta['date']} at 10m/12bands…"):
        sph = st.empty()
        try:
            img = cfc.filter(ee.Filter.eq('system:index', si)).first()
            if img.getInfo() is None:
                st.error(f"Image {si} not in collection"); sph.empty(); return
            r = download_composite(img, aoi, out, f"comp_{meta['date']}", 10, sph)
            sph.empty()
            if r and os.path.exists(r):
                with open(r,'rb') as f:
                    st.download_button(f"✅ Save: {meta['date']}", f.read(),
                                       f"S2_{meta['date']}_{safe[:20]}_12bands.tif",
                                       "image/tiff", key=f"sav_{mn}_{idx}")
            else:
                st.error(f"Download failed for {meta['date']}")
        except Exception as e:
            sph.empty(); st.error(f"Error: {e}")


# ═════════════════════════════════════════════════════════════════════════════
# DISPLAY: patch validity section
# ═════════════════════════════════════════════════════════════════════════════
def _display_patch_section():
    pv = st.session_state.patch_validity_maps
    dl = st.session_state.downloaded_composites
    if not pv:
        st.info("No patch data yet"); return

    st.subheader("🧩 Patch Validity Maps")

    # Summary table
    rows = []
    for mn in sorted(pv):
        p = pv[mn]
        rows.append({'Month':mn, 'Valid':p['n_valid'], 'Total':p['n_total'],
                     '%':f"{100*p['n_valid']/p['n_total']:.1f}",
                     'Grid':f"{p['n_patches_h']}×{p['n_patches_w']}"})
    st.dataframe(rows, use_container_width=True)

    sm = sorted(pv)
    for rs in range(0,len(sm),3):
        cols = st.columns(3)
        for j in range(3):
            idx = rs+j
            if idx >= len(sm): break
            mn = sm[idx]
            with cols[j]:
                fig = _plot_pv(pv[mn], mn)
                st.pyplot(fig); plt.close(fig)

                if mn in dl:
                    td = st.session_state.current_temp_dir
                    tp = os.path.join(td, f"pv_{mn}.tif")
                    if not os.path.exists(tp):
                        save_patch_geotiff(pv[mn], dl[mn], tp)
                    if os.path.exists(tp):
                        with open(tp,'rb') as f:
                            st.download_button("⬇️ GeoTIFF", f.read(),
                                               f"patch_validity_{mn}.tif", "image/tiff",
                                               key=f"dlpv_{mn}")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
def main():
    st.title("🛰️ Sentinel-2 Composite Explorer")
    st.caption("Explore component images · Download scenes · Inspect patch validity")

    # ── EE ──
    ok, msg = initialize_earth_engine()
    if not ok: st.error(msg); st.stop()
    st.sidebar.success(msg)

    # ── Sidebar ──
    st.sidebar.header("⚙️ Parameters")
    cloudy_pct = st.sidebar.slider("Max Cloud % (metadata)", 0, 50, 10, 5,
                                   disabled=st.session_state.processing_in_progress)

    st.sidebar.header("🗂️ Cache")
    nd = len(st.session_state.downloaded_composites)
    np_ = len(st.session_state.patch_validity_maps)
    nm = len(st.session_state.month_statuses)
    if nm: st.sidebar.success(f"📊 {nm} months analysed")
    if nd: st.sidebar.success(f"📥 {nd} composites")
    if np_: st.sidebar.success(f"🧩 {np_} patch maps")
    if not nm: st.sidebar.info("No cached data")

    # Failed months
    failed = [mn for mn, si in st.session_state.month_statuses.items()
              if si['status'] in (STATUS_REJECTED,)]
    if failed:
        st.sidebar.warning(f"🔴 Rejected: {', '.join(sorted(failed))}")

    if st.session_state.processing_in_progress:
        st.sidebar.error("⏳ Processing…")
        if st.sidebar.button("🛑 Stop", type="primary"):
            add_log("Stopped by user","WARNING")
            st.session_state.processing_in_progress = False
            st.rerun()

    if st.sidebar.button("🗑️ Clear All", disabled=st.session_state.processing_in_progress):
        for k,v in _DEFAULTS.items():
            if isinstance(v,(dict,list)):
                st.session_state[k] = type(v)(v)
            else:
                st.session_state[k] = v
        st.rerun()

    # ── Region ──
    st.header("1️⃣ Region")
    if not st.session_state.processing_in_progress:
        fm = folium.Map(location=[35.6892,51.3890], zoom_start=8)
        plugins.Draw(export=True, position='topleft', draw_options={
            'polyline':False,'rectangle':True,'polygon':True,
            'circle':False,'marker':False,'circlemarker':False}).add_to(fm)
        folium.TileLayer(tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
                         attr='Google',name='Satellite').add_to(fm)
        folium.LayerControl().add_to(fm)
        md = st_folium(fm, width=800, height=500, key="map")
        if md and md.get('last_active_drawing'):
            g = md['last_active_drawing'].get('geometry',{})
            if g.get('type')=='Polygon':
                st.session_state.last_drawn_polygon = Polygon(g['coordinates'][0])
                st.success("✅ Region selected")
        if st.button("💾 Save Region"):
            lp = st.session_state.last_drawn_polygon
            if lp:
                if not any(p.equals(lp) for p in st.session_state.drawn_polygons):
                    st.session_state.drawn_polygons.append(lp)
                    st.rerun()
                else: st.warning("Already saved")
            else: st.warning("Draw first")
    else:
        st.info("🔒 Map locked during processing")

    if st.session_state.drawn_polygons:
        st.subheader("📍 Saved Regions")
        for i,p in enumerate(st.session_state.drawn_polygons):
            c1,c2 = st.columns([4,1])
            cn = p.centroid
            c1.write(f"**Region {i+1}**: ~{p.area*111*111:.2f} km² | ({cn.y:.4f}, {cn.x:.4f})")
            if c2.button("🗑️",key=f"del_{i}",disabled=st.session_state.processing_in_progress):
                st.session_state.drawn_polygons.pop(i)
                if st.session_state.selected_region_index >= len(st.session_state.drawn_polygons):
                    st.session_state.selected_region_index = max(0,len(st.session_state.drawn_polygons)-1)
                st.rerun()

    # ── Date ──
    st.header("2️⃣ Time Period")
    c1,c2 = st.columns(2)
    start = c1.date_input("Start (inclusive)", value=date(2024,1,1),
                          disabled=st.session_state.processing_in_progress)
    end = c2.date_input("End (exclusive)", value=date(2025,1,1),
                        disabled=st.session_state.processing_in_progress)
    if start >= end: st.error("Invalid dates"); st.stop()
    ml = _month_list(start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))
    st.info(f"📅 **{len(ml)} months**: {ml[0]['month_name']} → {ml[-1]['month_name']}")

    # ── Process ──
    st.header("3️⃣ Process")

    sel_poly = None
    if st.session_state.drawn_polygons:
        opts = [f"Region {i+1} (~{p.area*111*111:.2f} km²)"
                for i,p in enumerate(st.session_state.drawn_polygons)]
        if st.session_state.selected_region_index >= len(opts):
            st.session_state.selected_region_index = 0
        sel = st.selectbox("🎯 Select Region", range(len(opts)),
                           format_func=lambda i: opts[i],
                           index=st.session_state.selected_region_index,
                           disabled=st.session_state.processing_in_progress)
        st.session_state.selected_region_index = sel
        sel_poly = st.session_state.drawn_polygons[sel]
    elif st.session_state.last_drawn_polygon:
        sel_poly = st.session_state.last_drawn_polygon
    else:
        st.warning("⚠️ Draw a region first")

    st.divider()

    # ── Three buttons ──
    bc1, bc2, bc3 = st.columns(3)

    with bc1:
        btn_new = st.button("🚀 Start New", type="primary",
                            disabled=st.session_state.processing_in_progress or sel_poly is None)
    with bc2:
        has_cache = bool(st.session_state.month_statuses or st.session_state.downloaded_composites)
        btn_resume = st.button("🔄 Resume",
                               disabled=not has_cache or st.session_state.processing_in_progress)
    with bc3:
        has_failed = any(si['status']==STATUS_REJECTED
                        for si in st.session_state.month_statuses.values())
        btn_retry = st.button("🔁 Retry Failed",
                              disabled=not has_failed or st.session_state.processing_in_progress)

    should_run = False
    resume = False

    if btn_new:
        # Fresh start — clear per-month caches
        st.session_state.downloaded_composites = {}
        st.session_state.month_statuses = {}
        st.session_state.component_image_metadata = {}
        st.session_state.patch_validity_maps = {}
        st.session_state.processing_complete = False
        st.session_state.processing_log = []
        st.session_state.processing_config = {
            'polygon_coords': list(sel_poly.exterior.coords),
            'start_date': start.strftime('%Y-%m-%d'),
            'end_date': end.strftime('%Y-%m-%d'),
            'cloudy_pct': cloudy_pct,
        }
        st.session_state.processing_in_progress = True
        should_run = True; resume = False

    elif btn_resume:
        if st.session_state.processing_config is None:
            st.session_state.processing_config = {
                'polygon_coords': list(sel_poly.exterior.coords),
                'start_date': start.strftime('%Y-%m-%d'),
                'end_date': end.strftime('%Y-%m-%d'),
                'cloudy_pct': cloudy_pct,
            }
        st.session_state.processing_in_progress = True
        should_run = True; resume = True

    elif btn_retry:
        # Clear rejected so they get re-processed
        to_clear = [mn for mn, si in st.session_state.month_statuses.items()
                    if si['status'] == STATUS_REJECTED]
        for mn in to_clear:
            del st.session_state.month_statuses[mn]
            st.session_state.downloaded_composites.pop(mn, None)
            st.session_state.patch_validity_maps.pop(mn, None)
        if st.session_state.processing_config is None:
            st.session_state.processing_config = {
                'polygon_coords': list(sel_poly.exterior.coords),
                'start_date': start.strftime('%Y-%m-%d'),
                'end_date': end.strftime('%Y-%m-%d'),
                'cloudy_pct': cloudy_pct,
            }
        st.session_state.processing_in_progress = True
        should_run = True; resume = True

    # Auto-continue on rerun (handles internet glitch → page reload)
    elif st.session_state.processing_in_progress and st.session_state.processing_config:
        st.info("🔄 Auto-resuming interrupted processing…")
        should_run = True; resume = True

    if should_run:
        config = st.session_state.processing_config
        if config is None:
            st.error("No config!"); st.session_state.processing_in_progress = False; st.stop()
        try:
            run_pipeline(config, resume=resume)
        except Exception as e:
            st.error(f"❌ Pipeline error: {e}")
            import traceback; st.code(traceback.format_exc())
            st.session_state.processing_in_progress = False
        st.rerun()

    # ═════════════════════════════════════════════════════════════════════════
    # RESULTS
    # ═════════════════════════════════════════════════════════════════════════
    if st.session_state.processing_complete:
        st.divider()
        st.header("📊 Results")

        tab1, tab2, tab3 = st.tabs(["📷 Component Images", "🧩 Patch Validity", "📋 Log"])

        with tab1:
            st.subheader("Monthly Composites & Component Images")
            st.write("Expand each month to see individual scenes and download them (12 bands, 10 m).")
            config = st.session_state.processing_config
            if config:
                ml = _month_list(config['start_date'], config['end_date'])
                for mi in ml:
                    mn = mi['month_name']
                    if mn in st.session_state.month_statuses:
                        _display_month(mn)

        with tab2:
            _display_patch_section()

        with tab3:
            st.subheader("📋 Processing Log")
            log = st.session_state.processing_log
            if log:
                for e in log:
                    col = {'INFO':'gray','WARNING':'orange','ERROR':'red'}.get(e['level'],'gray')
                    st.markdown(f"<span style='color:{col}'>[{e['ts']}] [{e['level']}] {e['msg']}</span>",
                               unsafe_allow_html=True)
            else:
                st.info("No log entries")


if __name__ == "__main__":
    main()
