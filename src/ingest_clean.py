# ---- SIMPLE: CNPV 2018 (DANE) -> solo PER/HOG CSVs, sin borrar tmp ----
import os, csv, time, zipfile, tempfile, shutil
from pathlib import Path
from typing import List, Tuple, Optional
import requests
from urllib.parse import urlencode
import pandas as pd

# ---- SIMPLE: CNPV 2018 (DANE) -> only PER/HOG CSVs for selected IDs ----
import os, csv, time, zipfile, tempfile, shutil
from pathlib import Path
from typing import List, Tuple, Optional
import requests
import os
from pathlib import Path
import io
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import regionmask
import time, gc
import requests
from tqdm import tqdm

def cnpv_fetch_per_hog_simple(ids: Optional[List[int]] = None,
                              start_id: int = 12143,
                              end_id: int = 12175) -> Path:
    """
    Download selected ZIPs from:
      https://microdatos.dane.gov.co/index.php/catalog/643/download/<id>
    Extract nested ZIPs, copy ONLY CSV files whose name contains 'PER' (persons) or 'HOG' (households),
    and write a manifest.

    Outputs:
      - data_raw/cnpv2018_tables/<id>/*.csv
      - data_raw/cnpv2018_manifest_per_hog.csv

    Notes:
      - No folders are deleted during the run (Windows-friendly).
      - If the site requires accepting terms, set env var DANE_PHPSESSID with your browser cookie.
    """
    # Resolve project paths
    try:
        here = Path(__file__).resolve()
    except NameError:
        here = Path.cwd()
    data_raw = here.parent.parent / "data_raw"
    out_root = data_raw / "cnpv2018_tables"
    manifest_path = data_raw / "cnpv2018_manifest_per_hog.csv"
    out_root.mkdir(parents=True, exist_ok=True)
    data_raw.mkdir(parents=True, exist_ok=True)

    # Unique temp dir per run (kept after run; delete manually when you want)
    tmp_dir = Path(tempfile.mkdtemp(prefix="cnpv2018_"))
    zips_dir = tmp_dir / "zips"
    zips_dir.mkdir(parents=True, exist_ok=True)

    # Session (optional cookie)
    sess = requests.Session()
    sess.headers.update({"User-Agent": "cnpv2018-simple/1.0"})
    phpsessid = os.getenv("DANE_PHPSESSID", "").strip()
    if phpsessid:
        sess.cookies.set("PHPSESSID", phpsessid, domain="microdatos.dane.gov.co")

    def download_zip(zip_id: int) -> Optional[Path]:
        url = f"https://microdatos.dane.gov.co/index.php/catalog/643/download/{zip_id}"
        out = zips_dir / f"{zip_id}.zip"
        if out.exists() and out.stat().st_size > 0:
            print(f"[SKIP] {zip_id}: exists {out.name}")
            return out
        try:
            with sess.get(url, stream=True, timeout=180) as r:
                if r.status_code == 404:
                    print(f"[INFO] {zip_id}: 404 (skip)")
                    return None
                r.raise_for_status()
                with out.open("wb") as f:
                    for chunk in r.iter_content(chunk_size=1<<20):
                        if chunk:
                            f.write(chunk)
            print(f"[OK] {zip_id}: downloaded -> {out.name}")
            return out
        except Exception as e:
            print(f"[WARN] {zip_id}: download failed: {e}")
            return None

    def extract_all_levels(top_zip: Path, work_dir: Path) -> List[Path]:
        """Extract nested ZIPs iteratively. Return all non-zip files."""
        pending = [top_zip]
        nonzips: List[Path] = []
        level = 0
        while pending:
            next_pending: List[Path] = []
            lvl_dir = work_dir / f"level_{level}"
            lvl_dir.mkdir(parents=True, exist_ok=True)
            for zp in pending:
                try:
                    with zipfile.ZipFile(zp, "r") as zf:
                        for info in zf.infolist():
                            name = Path(info.filename).name  # keep only basename
                            if not name:
                                continue
                            tgt = lvl_dir / f"{zp.stem}__{name}"
                            with zf.open(info) as src, open(tgt, "wb") as dst:
                                dst.write(src.read())
                            if tgt.suffix.lower() == ".zip":
                                next_pending.append(tgt)
                            else:
                                nonzips.append(tgt)
                except zipfile.BadZipFile:
                    pass
            pending = next_pending
            level += 1
        return nonzips

    def is_per_hog_csv(p: Path) -> Optional[str]:
        if p.suffix.lower() != ".csv":
            return None
        up = p.stem.upper()
        if "PER" in up:
            return "PER"
        if "HOG" in up:
            return "HOG"
        return None

    def count_csv(p: Path) -> Tuple[int,int]:
        rows, cols = 0, 0
        with p.open("r", encoding="latin1", errors="ignore", newline="") as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if i == 0:
                    cols = len(row)
                rows += 1
        return rows, cols

    # Determine which IDs to process
    iter_ids = list(ids) if ids is not None else list(range(start_id, end_id + 1))

    manifest_rows: List[dict] = []

    for zip_id in iter_ids:
        zpath = download_zip(zip_id)
        if not zpath:
            continue

        work_dir = tmp_dir / f"extract_{zip_id}"
        work_dir.mkdir(parents=True, exist_ok=True)

        files = extract_all_levels(zpath, work_dir)
        targets = [p for p in files if is_per_hog_csv(p)]

        if not targets:
            print(f"[INFO] {zip_id}: no PER/HOG CSV found")
            continue

        out_dir = out_root / str(zip_id)
        out_dir.mkdir(parents=True, exist_ok=True)

        for src in targets:
            kind = is_per_hog_csv(src)  # 'PER' or 'HOG'
            dst = out_dir / src.name
            if not dst.exists():
                shutil.copyfile(src, dst)
            r, c = count_csv(dst)
            manifest_rows.append({
                "zip_id": zip_id,
                "kind": kind,
                "file": dst.name,
                "path": str(dst),
                "rows_including_header": r,
                "columns": c
            })
            print(f"[OK] {zip_id}: {kind} -> {dst.name} (rows~{r}, cols={c})")

    # Write manifest (even if empty, to indicate the run)
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["zip_id","kind","file","path","rows_including_header","columns"])
        w.writeheader()
        for row in manifest_rows:
            w.writerow(row)

    print(f"[DONE] Manifest -> {manifest_path} (records={len(manifest_rows)})")
    print(f"[NOTE] Temp kept at: {tmp_dir}  (delete manually when you want)")
    return manifest_path

def cnpv_fetch_per_hog_top5() -> Path:
    """
    Fetch ONLY these departments (higher malaria presence):
      - 12143 (Antioquia)
      - 12152 (Córdoba)
      - 12154 (Chocó)
      - 12159 (Nariño)
      - 12166 (Cauca)
    Keeps the same output structure:
      data_raw/cnpv2018_tables/<id>/*.csv  +  data_raw/cnpv2018_manifest_per_hog.csv
    """
    top_ids = [12143, 12152, 12154, 12159, 12166]
    return cnpv_fetch_per_hog_simple(ids=top_ids)

from pathlib import Path
import pandas as pd

def combine_cnpv_per_hog(
    data_raw_dir="data_raw",
    start_id=12143,
    end_id=12175,
    out_personas="cnper2018.csv",
    out_hogares="cnhog2018.csv",
):
    """
    Scan data_raw/cnpv2018_tables/<id> (id range inclusive) and concatenate separately:
      - PER (persons)  -> data_raw/cnper2018.csv
      - HOG (households)-> data_raw/cnhog2018.csv
    """
    base = Path(data_raw_dir)
    tables_root = 'data_processed'/ "cnpv2018_tables"
    out_personas_path = 'data_processed'/ out_personas
    out_hogares_path = 'data_processed'/ out_hogares

    per_files, hog_files = [], []

    # 1) Collect PER/HOG CSV paths
    for zid in range(int(start_id), int(end_id) + 1):
        d = tables_root / str(zid)
        if not d.exists():
            continue
        for p in d.glob("*.csv"):
            stem_up = p.stem.upper()
            if "PER" in stem_up:
                per_files.append((zid, p))
            elif "HOG" in stem_up:
                hog_files.append((zid, p))

    print(f"[INFO] PER files: {len(per_files)} | HOG files: {len(hog_files)}")

    # 2) CSV reader (tries a few common encodings/delimiters)
    def _read_csv_any(path: Path) -> pd.DataFrame:
        tries = [
            dict(sep=None, engine="python", encoding="utf-8",  dtype="string", low_memory=False),
            dict(sep=None, engine="python", encoding="latin1", dtype="string", low_memory=False),
            dict(sep=",",  encoding="latin1", dtype="string", low_memory=False),
            dict(sep=";",  encoding="latin1", dtype="string", low_memory=False),
            dict(sep=",",  encoding="utf-8",  dtype="string", low_memory=False),
            dict(sep=";",  encoding="utf-8",  dtype="string", low_memory=False),
        ]
        last_err = None
        for kw in tries:
            try:
                return pd.read_csv(path, **kw)
            except Exception as e:
                last_err = e
        raise RuntimeError(f"Could not read {path.name}: {last_err}")

    # 3) Concatenate and write helper
    def _concat_and_write(file_tuples, out_path: Path, label: str):
        if not file_tuples:
            print(f"[WARN] No {label} files found -> skipping write.")
            return None

        frames = []
        for zid, p in file_tuples:
            df = _read_csv_any(p)
            # Add provenance columns
            df.insert(0, "zip_id", zid)
            df.insert(1, "source_file", p.name)
            frames.append(df)

        combined = pd.concat(frames, ignore_index=True, sort=False)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(out_path, index=False, encoding="utf-8")
        print(f"[OK] Wrote {label}: {out_path} (rows={len(combined):,}, cols={combined.shape[1]})")
        return out_path

    _concat_and_write(per_files, out_personas_path, "PERSONAS")
    _concat_and_write(hog_files, out_hogares_path, "HOGARES")

def fetch_malaria_agg(year_from=None, year_to=None):
    """
    Download aggregated data from datos.gov.co (Socrata).
    Returns a dataframe with selected columns.

    Parameters:
    - year_from, year_to: to add year filter if both are present 'ano between ...'
    """
    base = "https://www.datos.gov.co/resource/4hyg-wa9d.csv"

    where = "upper(nombre_evento) like 'MALARIA%'"
    if year_from is not None and year_to is not None:
        where += f" AND ano between {int(year_from)} and {int(year_to)}"

    params = {
        "$select": (
            "cod_eve,nombre_evento,semana,ano,"
            "cod_dpto_o,cod_mun_o,departamento_ocurrencia,municipio_ocurrencia,conteo"

        ),
        "$where": where,

        "$order": "ano,semana,departamento_ocurrencia,municipio_ocurrencia",
        "$limit": 500000,
    }

    app_token = os.getenv("SOCRATA_APP_TOKEN", "")
    if app_token:
        params["$$app_token"] = app_token

    url = base + "?" + urlencode(params)
    df = pd.read_csv(url)
    return df

DATA_DIR = Path("data_raw")
OUT_DIR = Path("data_processed")
SPATIAL_PATH = Path("data_raw/colombia_mpios/colombia_mpios_opendatasoft.gpkg")  # <-- ajusta a tu ruta
SPATIAL_LAYER = None  # si es GPKG con varias capas, pon el nombre de la capa; si es SHP, deja None

YEARS = list(range(2017, 2023))  # 2017-2022 inclusive

# TerraClimate tmean monthly
TERRACLIMATE_BASE = "https://climate.northwestknowledge.net/TERRACLIMATE-DATA"

def safe_remove(path: Path, retries: int = 5, delay: float = 0.6):
    for i in range(retries):
        try:
            os.remove(path)
            return True
        except PermissionError as e:
            gc.collect()
            time.sleep(delay)
        except Exception:
            break
    print(f"[WARN] No se pudo borrar {path.name} tras {retries} intentos.")
    return False

# Columnas de ID/nombre municipal en tu geodata
MUN_ID_COL = "mpio_code"       # <-- ajusta a tu nombre de columna ID
MUN_NAME_COL = "municipality"  # <-- ajusta a tu nombre de columna

def ensure_dirs():
    (DATA_DIR / "chirps").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "terraclimate").mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

def _download(url: str, out_path: Path, chunk=1 << 20):
    if out_path.exists():
        return
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0)) or None
        with open(out_path, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=out_path.name
        ) as pbar:
            for part in r.iter_content(chunk_size=chunk):
                if part:
                    f.write(part)
                    pbar.update(len(part))
HDF5_SIG = b"\x89HDF\r\n\x1a\n"

def is_valid_netcdf(path: Path) -> bool:
    """Quick signature check for NetCDF4/HDF5 or classic NetCDF."""
    if not path.exists() or path.stat().st_size < 32:
        return False
    with open(path, "rb") as f:
        head = f.read(8)
    if head.startswith(HDF5_SIG):
        return True
    if head[0:3] == b"CDF":
        return True
    return False

def download_nc_with_validation(url: str, out_path: Path, max_retries: int = 3, sleep_sec: float = 1.5):
    """Download to a temp file, validate NetCDF signature, atomically move to out_path. Retry if needed."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".part")
    for attempt in range(1, max_retries + 1):
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        _download(url, tmp)
        if is_valid_netcdf(tmp):
            os.replace(tmp, out_path)
            return
        else:
            tmp.unlink(missing_ok=True)
            time.sleep(sleep_sec)
    raise IOError(f"Failed to fetch valid NetCDF after {max_retries} attempts: {url}")                    

def load_municipalities(sp_path=SPATIAL_PATH, layer=SPATIAL_LAYER) -> gpd.GeoDataFrame:
    if layer:
        gdf = gpd.read_file(sp_path, layer=layer)
    else:
        gdf = gpd.read_file(sp_path)
    # Validaciones básicas
    if MUN_ID_COL not in gdf.columns:
        raise ValueError(f"No se encontró la columna ID municipal '{MUN_ID_COL}' en {sp_path}")
    if MUN_NAME_COL not in gdf.columns:
        warnings.warn(f"No se encontró '{MUN_NAME_COL}'. Se creará desde el ID.")
        gdf[MUN_NAME_COL] = gdf[MUN_ID_COL].astype(str)
    # Arreglar geometrías inválidas
    gdf = gdf.set_index(MUN_ID_COL, drop=False)
    gdf = gdf[~gdf.geometry.is_empty].copy()
    gdf["geometry"] = gdf.geometry.buffer(0)
    return gdf

# === CHIRPS p05 (more dissagregation) ===
CHIRPS_RES = "p05"
CHIRPS_BASE_TMPL = "https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/netcdf/{res}"
WEEKLY_FREQ = "W-SUN"  # weekly accumulation ending on Sunday

def build_regionmask_from_gdf(gdf: gpd.GeoDataFrame):
    """Return (Regions, GeoDataFrame in EPSG:4326) with 5-digit municipal codes."""
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)
    gdf[MUN_ID_COL] = gdf[MUN_ID_COL].astype(str).str.zfill(5)
    names   = gdf[MUN_NAME_COL].astype(str).tolist() if MUN_NAME_COL in gdf else gdf[MUN_ID_COL].tolist()
    numbers = list(range(len(gdf)))
    regs = regionmask.Regions(
        name="municipios_colombia",
        numbers=numbers,
        names=names,
        outlines=list(gdf.geometry),
        abbrevs=gdf[MUN_ID_COL].tolist()
    )
    regs.overlap = False   
    return regs, gdf

def ds_to_municipal_means(ds: xr.Dataset, var: str, regions: regionmask.Regions) -> pd.DataFrame:
    """Compute polygon means for `var` over all municipalities. Output: rows=time, cols=mpio_code."""
    if "longitude" in ds.coords: ds = ds.rename({"longitude": "lon"})
    if "latitude"  in ds.coords: ds = ds.rename({"latitude": "lat"})
    if (ds.lon > 180).any(): ds = ds.assign_coords(lon=((ds.lon + 180) % 360) - 180)
    ds = ds.sortby("lon")

    mask = regions.mask(ds)
    arr = ds[var]

    means = []
    for ridx in range(len(regions.numbers)):
        reg_mean = arr.where(mask == ridx).mean(dim=("lat", "lon"), skipna=True)
        means.append(reg_mean)

    out = xr.concat(means, dim=pd.Index(regions.abbrevs, name="mpio_code"))  

    df = out.to_pandas().T           
    df.index.name = "time"
    df.columns.name = "mpio_code"
    return df

def bbox_from_gdf(gdf: gpd.GeoDataFrame):
    """Return bounding box (lon_min, lat_min, lon_max, lat_max) in EPSG:4326."""
    g = gdf.to_crs(4326)
    lon_min, lat_min, lon_max, lat_max = g.total_bounds
    return float(lon_min), float(lat_min), float(lon_max), float(lat_max)

def _subset_bbox(ds: xr.Dataset, lon_min, lat_min, lon_max, lat_max) -> xr.Dataset:
    """Subset dataset by bounding box, handling ascending/descending axes."""
    lon_name = "lon" if "lon" in ds.coords else "longitude"
    lat_name = "lat" if "lat" in ds.coords else "latitude"
    lon_vals, lat_vals = ds[lon_name].values, ds[lat_name].values
    lon_slice = slice(lon_min, lon_max) if lon_vals[0] < lon_vals[-1] else slice(lon_max, lon_min)
    lat_slice = slice(lat_min, lat_max) if lat_vals[0] < lat_vals[-1] else slice(lat_max, lat_min)
    return ds.sel({lon_name: lon_slice, lat_name: lat_slice})

def fetch_chirps_p05(year: int, month: int | None = None) -> Path:
    """Download CHIRPS p05 NetCDF (monthly if month given, else yearly). Return local path."""
    base = CHIRPS_BASE_TMPL.format(res=CHIRPS_RES)
    if month is None:
        url = f"{base}/chirps-v2.0.{year}.days_{CHIRPS_RES}.nc"
        out = DATA_DIR / "chirps" / f"chirps-v2.0.{year}.days_{CHIRPS_RES}.nc"
    else:
        url = f"{base}/by_month/chirps-v2.0.{year}.{month:02d}.days_{CHIRPS_RES}.nc"
        out = DATA_DIR / "chirps" / f"chirps-v2.0.{year}.{month:02d}.days_{CHIRPS_RES}.nc"
    if out.exists() and not is_valid_netcdf(out):
        out.unlink(missing_ok=True)

    if not out.exists():
        download_nc_with_validation(url, out)
    return out

def process_chirps_nc_to_weekly_means(nc_path: Path,
                                      regions: regionmask.Regions,
                                      lon_min, lat_min, lon_max, lat_max) -> pd.DataFrame:
    """Open one CHIRPS NetCDF, crop to Colombia, sum to weekly, compute municipal means. Return weekly DF."""
    with xr.open_dataset(nc_path, chunks="auto") as ds:
        if "longitude" in ds.coords: ds = ds.rename({"longitude": "lon"})
        if "latitude"  in ds.coords: ds = ds.rename({"latitude": "lat"})
        if (ds.lon > 180).any(): ds = ds.assign_coords(lon=((ds.lon + 180) % 360) - 180)
        ds = ds.sortby("lon")
        ds = _subset_bbox(ds, lon_min, lat_min, lon_max, lat_max)
        var = "precip" if "precip" in ds.data_vars else [v for v in ds.data_vars if "precip" in v.lower()][0]
        arr_week = ds[var].resample(time=WEEKLY_FREQ).sum(skipna=True)
        ds_week  = arr_week.to_dataset(name=var)
        df_week  = ds_to_municipal_means(ds_week, var, regions)
        try:
            df_week.index = pd.DatetimeIndex(ds_week.indexes["time"].to_datetimeindex())
        except Exception:
            df_week.index = pd.to_datetime(df_week.index, errors="coerce")
    return df_week


def run_chirps_weekly_colombia_p05(years=YEARS, mode: str = "year", append_final: bool = True):
    """Process CHIRPS p05 by year, compute weekly municipal sums, persist yearly CSVs,
    and append each year to a cumulative final CSV safely (idempotent)."""
    ensure_dirs()
    gdf = load_municipalities()
    regions, gdf_ll = build_regionmask_from_gdf(gdf)
    lon_min, lat_min, lon_max, lat_max = bbox_from_gdf(gdf_ll)

    codes_all = list(regions.abbrevs)
    out_year_tpl = OUT_DIR / "chirps_weekly_p05_{year}.csv"
    out_final    = OUT_DIR / "climate_chirps_weekly_precip_sum_mpio_p05.csv"

    for y in years:
        out_y = Path(str(out_year_tpl).format(year=y))
        if out_y.exists() and out_y.stat().st_size > 0:
            print(f"[SKIP] Year {y} already written: {out_y.name}")
        else:
            print(f"[CHIRPS p05] {y} downloading/processing…")
            if mode == "year":
                p = fetch_chirps_p05(y, month=None)
                try:
                    df_week = process_chirps_nc_to_weekly_means(p, regions, lon_min, lat_min, lon_max, lat_max)
                finally:
                    safe_remove(p)
            else:
                parts = []
                for m in range(1, 13):
                    p = fetch_chirps_p05(y, month=m)
                    try:
                        parts.append(process_chirps_nc_to_weekly_means(p, regions, lon_min, lat_min, lon_max, lat_max))
                    finally:
                        safe_remove(p)
                if not parts:
                    print(f"[WARN] No monthly parts for {y}."); continue
                df_week = pd.concat(parts).sort_index()

            # ensure consistent columns order across years
            df_week = df_week.reindex(columns=codes_all)
            df_week.index.name = "week_end"
            df_week.columns.name = "mpio_code"

            tmp_y = out_y.with_suffix(".tmp.csv")
            w = df_week.copy()
            w.insert(0, "week_end", w.index)
            w.to_csv(tmp_y, index=False, float_format="%.2f", date_format="%Y-%m-%d")
            os.replace(tmp_y, out_y)
            print(f"[WRITE] {out_y.name} weeks={len(w)} {w['week_end'].min().date()}→{w['week_end'].max().date()}")


        df_y = pd.read_csv(out_y, parse_dates=["week_end"])
        df_y.to_csv(
            out_final,
            mode="a",
            header=not out_final.exists(),
            index=False,
            float_format="%.2f",
            date_format="%Y-%m-%d",
        )
        df_y = df_y.drop_duplicates().reset_index()
        print(f"[APPEND] {y} → {out_final.name}")
        



# === TerraClimate ===
def fetch_terraclimate(years=YEARS) -> list[Path]:
    """Download TerraClimate tmean monthly NetCDFs. Return list of local paths."""
    paths = []
    for y in years:
        url = f"{TERRACLIMATE_BASE}/TerraClimate_aet_{y}.nc"
        out = DATA_DIR / "terraclimate" / f"TerraClimate_aet_{y}.nc"
        _download(url, out)
        paths.append(out)
    return paths

def process_terraclimate_to_municipal(paths_nc: list[Path], regions: regionmask.Regions, var_key: str = "aet") -> pd.DataFrame:
    """Municipal polygon means for TerraClimate `var_key`; rows=time, cols=mpio_code. Crops to Colombia before masking."""
    # bbox Colombia (sin recrear regions)
    gdf_bbox = load_municipalities()
    lon_min, lat_min, lon_max, lat_max = bbox_from_gdf(gdf_bbox)

    # asegúrate de 2D mask
    try:
        regions.overlap = False
    except Exception:
        pass

    dfs = []
    for p in paths_nc:
        with xr.open_dataset(p, chunks="auto") as ds:
            # normaliza coords y recorta a Colombia (reduce drásticamente el grid)
            if "longitude" in ds.coords: ds = ds.rename({"longitude": "lon"})
            if "latitude"  in ds.coords: ds = ds.rename({"latitude": "lat"})
            ds = _subset_bbox(ds, lon_min, lat_min, lon_max, lat_max)

            # elige la variable (aet, tmax, tmin, etc.)
            cand = [v for v in ds.data_vars if var_key in v.lower()]
            if not cand:
                raise KeyError(f"{var_key} not found in {p.name}; vars={list(ds.data_vars)}")
            v = cand[0]

            # usa máscara 2D (rasterize) → memoria pequeña
            mask2d = regions.mask(ds, method="rasterize")
            arr = ds[v]

            # promedio municipal (loop por región usando mask 2D)
            means = []
            for ridx in range(len(regions.numbers)):
                reg_mean = arr.where(mask2d == ridx).mean(dim=("lat","lon"), skipna=True)
                means.append(reg_mean)

            out = xr.concat(means, dim=pd.Index(regions.abbrevs, name="mpio_code"))
            df_m = out.to_pandas().T
            df_m.index.name = "time"
            df_m.columns.name = "mpio_code"
            dfs.append(df_m)

    out = pd.concat(dfs).sort_index()
    out.index = pd.to_datetime(out.index)
    out.index.name = "date"
    return out

def concat_chirps_yearlies_p05(years=YEARS, make_long: bool = True):
    """Concatenate all yearly CHIRPS p05 weekly CSVs into a single final CSV.
    Input files: OUT_DIR / f"chirps_weekly_p05_{year}.csv"
    Output wide: OUT_DIR / "climate_chirps_weekly_precip_sum_mpio_p05.csv" (rows=week_end, cols=mpio_code)
    Output long: OUT_DIR / "climate_chirps_weekly_precip_sum_mpio_p05_long.csv" (optional)"""
    import re
    import pandas as pd
    from pathlib import Path

    pat = re.compile(r"chirps_weekly_p05_(\d{4})\.csv$", re.IGNORECASE)
    # Prefer explicit years in order; fall back to glob if missing
    files = [OUT_DIR / f"chirps_weekly_p05_{y}.csv" for y in years if (OUT_DIR / f"chirps_weekly_p05_{y}.csv").exists()]
    if not files:
        files = sorted([p for p in OUT_DIR.glob("chirps_weekly_p05_*.csv") if pat.search(p.name)],
                       key=lambda p: int(pat.search(p.name).group(1)))
    if not files:
        raise FileNotFoundError("No yearly CHIRPS files found in data_processed (chirps_weekly_p05_YYYY.csv).")

    def _read_year_csv(p: Path) -> pd.DataFrame:
        df = pd.read_csv(p)
        first = df.columns[0]
        if first != "week_end":
            df = df.rename(columns={first: "week_end"})
        df["week_end"] = pd.to_datetime(df["week_end"], errors="coerce")
        drop = [c for c in df.columns if isinstance(c, str) and c.lower().startswith("unnamed")]
        if drop:
            df = df.drop(columns=drop)
        rename_map = {}
        for c in df.columns:
            if c == "week_end":
                continue
            cc = str(c).strip()
            rename_map[c] = cc.zfill(5) if cc.isdigit() else cc
        df = df.rename(columns=rename_map)
        df = df.sort_values("week_end").drop_duplicates(subset=["week_end"], keep="last")
        return df.set_index("week_end")

    dfs, all_cols = [], set()
    for p in files:
        d = _read_year_csv(p)
        dfs.append(d)
        all_cols |= set(d.columns)

    cols_sorted = sorted(all_cols)
    dfs_aligned = [d.reindex(columns=cols_sorted) for d in dfs]
    final_wide = pd.concat(dfs_aligned, axis=0).sort_index()
    final_wide = final_wide[~final_wide.index.duplicated(keep="last")]
    final_wide.index.name = "week_end"
    final_wide.columns.name = "mpio_code"

    out_wide = OUT_DIR / "climate_chirps_weekly_precip_sum_mpio_p05.csv"
    final_wide.to_csv(out_wide, float_format="%.2f", date_format="%Y-%m-%d")
    print(f"[WRITE] {out_wide} | weeks={len(final_wide)} | mpios={final_wide.shape[1]} | "
          f"{final_wide.index.min().date()}→{final_wide.index.max().date()}")

    if make_long:
        final_long = final_wide.reset_index().melt(
            id_vars="week_end", var_name="mpio_code", value_name="precip_week_mm"
        ).sort_values(["mpio_code", "week_end"])
        out_long = OUT_DIR / "climate_chirps_weekly_precip_sum_mpio_p05_long.csv"
        final_long.to_csv(out_long, index=False, float_format="%.2f", date_format="%Y-%m-%d")
        print(f"[WRITE] {out_long} | rows={len(final_long)}")

def download_climate_(download_chirps: bool = True, download_terraclimate: bool = True):
    """Run pipeline: CHIRPS p05 weekly (yearly processing + final concat) and optional TerraClimate."""

    if download_chirps:
        print("[1/3] CHIRPS p05 → weekly by municipality (per-year)…")
        run_chirps_weekly_colombia_p05(YEARS, mode="year", append_final=False)
        print("[1b/3] Concatenating yearly CHIRPS into final wide/long…")
        concat_chirps_yearlies_p05(YEARS, make_long=True)

    if download_terraclimate:
        print("[2/3] TerraClimate tmean monthly by municipality…")
        gdf = load_municipalities()
        regions, _ = build_regionmask_from_gdf(gdf)
        aet_paths = fetch_terraclimate(YEARS)
        df_month = process_terraclimate_to_municipal(aet_paths, regions).astype(float)
        df_month.to_csv(
            OUT_DIR / "climate_terraclimate_monthly_mpio.csv",
            float_format="%.2f",
            date_format="%Y-%m",
        )
    print("[3/3] Done.")

import re


