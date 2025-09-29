import os
import pandas as pd
import matplotlib.pyplot as plt

# download_col_mpios.py
import os, io, zipfile
import requests
import geopandas as gpd
import pandas as pd
from pathlib import Path

# === Config ===
OUT_DIR = "data_raw/colombia_mpios"
os.makedirs(OUT_DIR, exist_ok=True)

ODS_GEOJSON = (
    "https://data.opendatasoft.com/api/explore/v2.1/catalog/datasets/"
    "shapes@bogota-laburbano/exports/geojson?lang=en&timezone=UTC"
)

def download_mpios_gdf(out_dir="data_raw/colombia_mpios"):
    os.makedirs(out_dir, exist_ok=True)
    print("[*] Downloading GeoJSON from OpenDataSoft…")
    r = requests.get(ODS_GEOJSON, timeout=180)
    r.raise_for_status()
    gdf = gpd.read_file(io.BytesIO(r.content))
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)
    cols_lower = {c.lower(): c for c in gdf.columns}
    if "mpios" not in cols_lower:
        raise ValueError("Column 'MPIOS' not found in dataset.")
    gdf["mpio_code"] = (
        pd.to_numeric(gdf[cols_lower["mpios"]], errors="coerce")
        .astype("Int64")
        .astype(str)
        .str.replace("<NA>", "", regex=False)
        .str.zfill(5)
    )
    ok = gdf["mpio_code"].str.fullmatch(r"\d{5}", na=False).mean()
    print(f"[*] Valid 5-digit mpio_code ratio: {ok:.1%}")
    gpkg_path = os.path.join(out_dir, "colombia_mpios_opendatasoft.gpkg")
    shp_path  = os.path.join(out_dir, "colombia_mpios_opendatasoft.shp")
    gdf.to_file(gpkg_path, driver="GPKG")
    gdf.to_file(shp_path)
    print(f"[✓] Saved: {gpkg_path} | {shp_path}")
    return gdf[["mpio_code", "geometry"]].drop_duplicates("mpio_code")



def download_col_departments_ne10() -> gpd.GeoDataFrame:
    """
    Download Natural Earth admin-1 (all countries), filter to Colombia, return GeoDataFrame.
    """
    url = "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_1_states_provinces.zip"
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    # Find the .shp inside the zip and read with GeoPandas
    shp_name = [n for n in z.namelist() if n.endswith(".shp")][0]
    tmp = Path("data_raw/_ne_tmp")
    tmp.mkdir(exist_ok=True)
    z.extractall(tmp)
    gdf = gpd.read_file(tmp / shp_name)
    # keep only Colombia
    if "adm0_a3" in gdf.columns:
        gdf = gdf[gdf["adm0_a3"] == "COL"].copy()
    elif "admin" in gdf.columns:
        gdf = gdf[gdf["admin"].str.upper() == "COLOMBIA"].copy()
    else:
        raise RuntimeError("Country field not found in Natural Earth layer.")
    # Use 'name' as department label (exists in NE)
    if "name" not in gdf.columns:
        raise RuntimeError("'name' field not found in Natural Earth layer.")
    return gdf


def zero_pad_codes(s, width):
    """Numeric -> string with left zeros (keeps <NA>)."""
    s = pd.to_numeric(s, errors="coerce").astype("Int64")
    return s.astype("string").str.zfill(width)