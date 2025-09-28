from pathlib import Path
import pandas as pd
import gc
import numpy as np
import matplotlib.pyplot as plt

# --- helpers ---
def _zero_pad_codes(s, width):
    """Numeric -> string with left zeros (keeps <NA>)."""
    s = pd.to_numeric(s, errors="coerce").astype("Int64")
    return s.astype("string").str.zfill(width)

def census_pop_by_municipality(
    persons_csv="data_raw/cnper2018.csv",
    drop_after=True,
) -> pd.DataFrame:
    """
    Build municipal population using PERSONAS with known columns:
      - U_DPTO (department code, numeric-like)
      - U_MPIO (municipality code, numeric-like)
      - P_EDADR (age groups; 1 = 0–4)
    Returns: ['mpio_code','U_DPTO','U_MPIO','pop_total','pop_u5']
    """
    usecols = ["U_DPTO", "U_MPIO", "P_EDADR"]
    df = pd.read_csv(
        persons_csv, encoding="latin1",
        usecols=usecols, dtype="string"
    )

    dpto2 = pd.to_numeric(df["U_DPTO"], errors="coerce").astype("Int64").astype("string").str.zfill(2)
    mpio3 = pd.to_numeric(df["U_MPIO"], errors="coerce").astype("Int64").astype("string").str.zfill(3)
    df["_mpio"] = dpto2 + mpio3

    # Under-5 flag (P_EDADR == 1)
    df["_u5"] = pd.to_numeric(df["P_EDADR"], errors="coerce").eq(1).astype("int64")

    grp = (
        df.groupby("_mpio")
          .agg(pop_total=("U_DPTO", "size"),  # count rows
               pop_u5=("_u5", "sum"))
          .reset_index()
          .rename(columns={"_mpio": "mpio_code"})
    )

    # Split back department/municipality strings
    grp["U_DPTO"] = grp["mpio_code"].str[:2].astype("string")
    grp["U_MPIO"] = grp["mpio_code"].str[2:].astype("string")

    # Reorder columns
    out = grp[["mpio_code", "U_DPTO", "U_MPIO", "pop_total", "pop_u5"]].copy()

    # Free memory
    if drop_after:
        del df, grp
        gc.collect()

    return out

def malaria_weekly_indicators(
    malaria_csv="data_raw/malaria_agg.csv",
    persons_csv="data_raw/cnper2018.csv",
    year_from=None, year_to=None,
) -> pd.DataFrame:
    """
    Aggregates weekly malaria counts and computes indicators per total pop and ages 0–4.
    Assumes malaria CSV has: ano, semana, cod_dpto_o, cod_mun_o, departamento_ocurrencia,
                              municipio_ocurrencia, nombre_evento, conteo
    """
    # population
    pop = census_pop_by_municipality(persons_csv)

    # malaria
    usecols = [
        "ano", "semana", "cod_dpto_o", "cod_mun_o",
        "departamento_ocurrencia", "municipio_ocurrencia",
        "nombre_evento", "conteo"
    ]
    m = pd.read_csv(
        malaria_csv, usecols=usecols, dtype="string", low_memory=False)

    # types & filters
    m = m[m['departamento_ocurrencia'].isin(['ANTIOQUIA','CAUCA','CHOCO','CORDOBA','NARIÑO'])].reset_index(drop=True)
    m["ano"] = pd.to_numeric(m["ano"], errors="coerce").astype("Int64")
    m["semana"] = pd.to_numeric(m["semana"], errors="coerce").astype("Int64")
    if year_from is not None: 
        m = m[m["ano"] >= int(year_from)]
        if year_to is not None: 
            [m["ano"] <= int(year_to)]

    m["conteo"] = pd.to_numeric(m["conteo"], errors="coerce").fillna(0).astype(int)
    m["mpio_code"] = _zero_pad_codes(m["cod_mun_o"], 5)

    ev = m["nombre_evento"].astype("string").str.upper()
    m["_is_comp"]  = ev.str.contains("COMPLICAD", na=False)
    m["_is_vivax"] = ev.str.contains("VIVAX", na=False)
    m["_is_falci"] = ev.str.contains("FALCIPAR", na=False)

    # precompute subclass counts (correct: sum conteo where flag True)
    m["cnt_comp"]  = m["_is_comp"].astype(int)  * m["conteo"]
    m["cnt_vivax"] = m["_is_vivax"].astype(int) * m["conteo"]
    m["cnt_falci"] = m["_is_falci"].astype(int) * m["conteo"]

    grp = (
        m.groupby(["ano","semana","mpio_code","departamento_ocurrencia","municipio_ocurrencia"], dropna=False)
         .agg(
            cases_total=("conteo","sum"),
            cases_complicated=("cnt_comp","sum"),
            cases_vivax=("cnt_vivax","sum"),
            cases_falciparum=("cnt_falci","sum"),
         )
         .reset_index()
    )
    grp = grp[~grp["municipio_ocurrencia"].str.contains("DESCONOCID", case=False, na=False)].reset_index(drop=True)

    # merge population
    out = grp.merge(pop[["mpio_code","pop_total","pop_u5"]], on="mpio_code", how="left")

    # incidence (handle zero pop -> NaN)
    denom_pop = out["pop_total"].replace({0: np.nan})
    denom_u5  = out["pop_u5"].replace({0: np.nan})

    out["inc_total_pop"] = out["cases_total"]       / denom_pop
    out["inc_comp_pop"]  = out["cases_complicated"] / denom_pop
    out["inc_vivax_pop"] = out["cases_vivax"]       / denom_pop
    out["inc_falci_pop"] = out["cases_falciparum"]  / denom_pop

    out = out.rename(columns={
        "departamento_ocurrencia":"department",
        "municipio_ocurrencia":"municipality"
    }).sort_values(["ano","semana","department","municipality"]).reset_index(drop=True)
    return out


# --- 3) Quick lineplot ---
def plot_weekly_lines(
    df, metric="inc_total_pop", per=100000, top_k=10, figsize=(11,6), title=None
):
    """
    Plots top_k municipalities by total cases for the chosen weekly metric.
    """
    data = df.copy()
    data["week_key"] = data["ano"].astype(int).astype(str) + "-" + data["semana"].astype(int).astype(str).str.zfill(2)
    data["metric_disp"] = data[metric] * (per if per else 1.0)

    top = (data.groupby("mpio_code")["cases_total"].sum().sort_values(ascending=False).head(top_k).index)
    sub = data[data["mpio_code"].isin(top)].copy()
    sub["label"] = sub["department"].str.title() + " - " + sub["municipality"].str.title()

    wide = sub.pivot_table(index="week_key", columns="label", values="metric_disp", aggfunc="sum").sort_index()

    ax = wide.plot(figsize=figsize)
    ax.set_xlabel("Year-Week")
    ax.set_ylabel(f"{metric} (per {per:,})" if per else metric)
    ax.set_title(title or f"Weekly {metric}")
    plt.tight_layout()
    return ax, wide


from pathlib import Path
from collections import defaultdict
import unicodedata as ud
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
import pandas as pd
import unicodedata as ud

def agg_malaria_by_department(
    csv_path="data_raw/malaria_agg.csv",
    year_from=None,
    year_to=None,
    event_prefix="MALARIA",   # keep only rows whose nombre_evento starts with this (case-insensitive)
) -> pd.DataFrame:
    """
    Aggregate malaria cases by department in one pass (no chunking).
    Returns a DataFrame with columns: ['department','cases','share','rank'].
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Not found: {csv_path}")

    usecols = ["ano", "departamento_ocurrencia", "nombre_evento", "conteo"]

    # Read once (no chunking). sep=None -> needs engine="python".
    df = pd.read_csv(
        csv_path,
        usecols=usecols,
        dtype="string",
        sep=None, engine="python", encoding="latin1"
    )

    # Year filter (optional)
    df["ano"] = pd.to_numeric(df["ano"], errors="coerce").astype("Int64")
    if year_from is not None and year_to is not None:
        df = df[(df["ano"] >= int(year_from)) & (df["ano"] <= int(year_to))]

    if df.empty:
        return pd.DataFrame(columns=["department","cases","share","rank"])

    if event_prefix:
        ev = df["nombre_evento"]
        df = df[ev.str.startswith(event_prefix.upper())]

    if df.empty:
        return pd.DataFrame(columns=["department","cases","share","rank"])

    df["departamento_ocurrencia"] = df["departamento_ocurrencia"]
    df["conteo"] = pd.to_numeric(df["conteo"], errors="coerce").fillna(0).astype(int)

    out = (
        df.groupby("departamento_ocurrencia", as_index=False)["conteo"]
          .sum()
          .rename(columns={"departamento_ocurrencia":"department", "conteo":"cases"})
          .sort_values("cases", ascending=False, ignore_index=True)
    )

    total = out["cases"].sum()
    out["share"] = out["cases"] / total if total else 0.0
    out["rank"] = out["cases"].rank(method="dense", ascending=False).astype(int)

    return out

def plot_top_departments(df: pd.DataFrame, top=12, figsize=(9,5), title=None):
    """
    Quick bar chart of top-N departments by cases (uses matplotlib).
    """
    topdf = df.nlargest(top, "cases").iloc[::-1]  # reverse for horizontal bar
    ax = topdf.plot(kind="barh", x="department", y="cases", figsize=figsize, legend=False)
    ax.set_xlabel("Cases")
    ax.set_ylabel("Department")
    ax.set_title(title or f"Top {top} departments by malaria cases")
    for i, v in enumerate(topdf["cases"].tolist()):
        ax.text(v, i, f" {v:,}", va="center")
    plt.tight_layout()
    return ax


import io, zipfile, requests, unicodedata as ud
from pathlib import Path
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

def _norm_name(x: str) -> str:
    """Uppercase, strip accents/punct/spaces for robust joins."""
    if x is None:
        return ""
    x = str(x).strip().upper()
    x = "".join(c for c in ud.normalize("NFKD", x) if not ud.combining(c))
    x = x.replace(".", " ").replace("-", " ")
    x = " ".join(x.split())
    return x

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
    gdf["dept_norm"] = gdf["name"].apply(_norm_name)
    return gdf

def plot_malaria_map_departments(df_dep: pd.DataFrame, title="Malaria cases by department"):
    """
    Merge aggregated department cases with Colombia admin-1 polygons and plot a choropleth.
    df_dep must have columns: ['department','cases'] (and optionally 'share').
    """
    # Prepare agg dataframe
    df = df_dep.copy()
    if "department" not in df.columns or "cases" not in df.columns:
        raise ValueError("df_dep must have columns: 'department' and 'cases'")
    df["dept_norm"] = df["department"].apply(_norm_name)

    # Manual name fixes for tricky ones
    # (Natural Earth uses 'DISTRITO CAPITAL DE BOGOTA' and 'SAN ANDRES PROVIDENCIA Y SANTA CATALINA')
    fixes = {
        "BOGOTA DC": "DISTRITO CAPITAL DE BOGOTA",
        "ARCHIPIELAGO DE SAN ANDRES PROVIDENCIA Y SANTA CATALINA": "SAN ANDRES PROVIDENCIA Y SANTA CATALINA",
        "SAN ANDRES PROVIDENCIA Y SANTA CATALINA": "SAN ANDRES PROVIDENCIA Y SANTA CATALINA",
    }
    df["dept_norm"] = df["dept_norm"].replace(fixes)

    # Get polygons and merge
    gdf = download_col_departments_ne10()
    merged = gdf.merge(df[["dept_norm","cases"]], on="dept_norm", how="left")
    merged["cases"] = merged["cases"].fillna(0)

    # Plot
    ax = merged.plot(column="cases", legend=True, figsize=(8, 9))
    ax.set_title(title)
    ax.axis("off")

    # Annotate top few
    top = df.sort_values("cases", ascending=False).head(5)
    print("Top 5 departments by cases:")
    print(top[["department","cases"]].to_string(index=False))
    return ax, merged
