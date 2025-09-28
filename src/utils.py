import os
from urllib.parse import urlencode
import pandas as pd
import matplotlib.pyplot as plt

# download_col_mpios.py
import os, io, zipfile, re, json, sys
import requests
import geopandas as gpd
import pandas as pd


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



def make_maps(df_vuln_mpio, df_mal, gdf=None, malaria_col="inc_total_pop"):
    if gdf is None:
        gdf = download_mpios_gdf()

    merged = (
        gdf.merge(df_vuln_mpio[["mpio_code", "IV_mpio"]], on="mpio_code")
           .merge(df_mal, on="mpio_code")
    )

    value_col = malaria_col
    per_100k = False
    if malaria_col.startswith("inc"):
        value_col = f"{malaria_col}_100k"
        merged[value_col] = merged[malaria_col] * 100000
        per_100k = True

    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    merged.plot(column="IV_mpio", cmap="Reds", legend=True, ax=axes[0], edgecolor="none")
    axes[0].set_title("Vulnerability Index (IV_mpio)")
    axes[0].axis("off")
    merged.plot(column=value_col, cmap="PuBu", legend=True, ax=axes[1], edgecolor="none")
    axes[1].set_title(f"Malaria {'Incidence per 100,000' if per_100k else 'Counts'} ({malaria_col})")
    axes[1].axis("off")
    plt.tight_layout()
    plt.show()
    return merged