from pathlib import Path
import pandas as pd
import gc
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import importlib
import warnings
warnings.filterwarnings('ignore')
import src.utils 
importlib.reload(src.utils)
from src.utils import *
import src.eval_visuals 
importlib.reload(src.eval_visuals)
from src.eval_visuals import *

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
    Aggregates weekly malaria counts and computes indicators per total population.
    Assumes malaria CSV has: ano, semana, cod_dpto_o, cod_mun_o, departamento_ocurrencia,
                              municipio_ocurrencia, nombre_evento, conteo
    """
    print('Constructing output variables, relative malaria cases... ')
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
    m["mpio_code"] = zero_pad_codes(m["cod_mun_o"], 5)

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
    df_week = out.copy()
    df_week.to_csv('data_processed/output_weekly_mpio.csv',index=False)
    # Plot: incidence per 100k, auto-pick top 8 municipalities by cases
    ax, wide = plot_weekly_lines(df_week, metric="inc_total_pop", per=100000, top_k=8,
                                     title="Malaria incidence per 100k (weekly)")

    return out

import pandas as pd

def build_malaria_features_total(df_week: pd.DataFrame) -> pd.DataFrame:
    """
    -------------------------------------------------------------------------
    Name: build_malaria_features_total
    Purpose:
        Aggregate malaria cases (2017–2022) by municipality and compute
        cumulative incidences using a fixed population (2018). Returns a
        DataFrame ready for feature engineering/modeling at municipal level.

    Inputs:
        df_week : pd.DataFrame
            Weekly data with columns:
            - 'mpio_code', 'municipality'
            - 'cases_total', 'cases_complicated', 'cases_vivax', 'cases_falciparum'
            - 'pop_total' (fixed population per municipality; first value is used)

    Outputs:
        pd.DataFrame with columns:
            ['mpio_code', 'municipality',
             'cases_total','cases_complicated','cases_vivax','cases_falciparum',
             'inc_total_pop','inc_comp_pop','inc_vivax_pop','inc_falci_pop']

    Notes:
        - Incidences are per 100,000 inhabitants.
        - Population is taken as fixed per municipality (first observed value).
    -------------------------------------------------------------------------
    """
    print('Constructing output variables, total malaria cases... ')
    case_cols_base = ["cases_total", "cases_complicated", "cases_vivax", "cases_falciparum"]

    # 1) Aggregate total cases (2017–2022) by municipality
    cases_sum = (
        df_week.groupby(["mpio_code", "municipality"], as_index=False)[case_cols_base]
               .sum()
    )

    # 2) Fixed population per municipality (2018): take the first observed value
    pop_by_mpio = (
        df_week.groupby("mpio_code")["pop_total"]
               .first()
               .reset_index()
    )

    # 3) Join and compute cumulative incidences (per 100,000)
    out = cases_sum.merge(pop_by_mpio, on="mpio_code", how="left")

    out["inc_total_pop"] = (out["cases_total"]       / out["pop_total"]) * 100000
    out["inc_comp_pop"]  = (out["cases_complicated"] / out["pop_total"]) * 100000
    out["inc_vivax_pop"] = (out["cases_vivax"]       / out["pop_total"]) * 100000
    out["inc_falci_pop"] = (out["cases_falciparum"]  / out["pop_total"]) * 100000

    # 4) Final tidy frame
    cols_keep = (
        ["mpio_code", "municipality"] +
        case_cols_base +
        ["inc_total_pop","inc_comp_pop","inc_vivax_pop","inc_falci_pop"]
    )
    df_malaria_total = out[cols_keep].copy()
    df_malaria_total.to_csv('data_processed/malaria_total_mpio.csv',index=False)
    return df_malaria_total



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


def build_vulnerability_index(df_malaria_total: pd.DataFrame, personas_csv: str = "data_raw/cnper2018.csv",
                                hogares_csv: str = "data_raw/cnhog2018.csv",) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    -------------------------------------------------------------------------
    Name: build_vulnerability_index
    Purpose:
        Construct household/person vulnerability flags, aggregate to municipality,
        compute correlation-informed weights versus malaria indicators, and build
        a weighted municipal vulnerability index (IV_mpio).

    Inputs:
        df_personas : pd.DataFrame
            Person-level microdata with at least:
            ['COD_ENCUESTAS','P_ALFABETA','P_NIVEL_ANOSR','P_EDADR','P_TRABAJO',
             'PA1_CALIDAD_SERV','U_DPTO','U_MPIO']
        df_hogares  : pd.DataFrame
            Household-level microdata with at least:
            ['COD_ENCUESTAS','H_DONDE_PREPALIM','H_AGUA_COCIN','HA_TOT_PER','H_NRO_DORMIT']
        df_malaria_total : pd.DataFrame
            Municipal malaria features with at least:
            ['mpio_code','inc_total_pop','inc_comp_pop','inc_vivax_pop','inc_falci_pop']

    Outputs:
        (df_vuln_mpio, corr, weights)
            df_vuln_mpio: municipal-level vulnerability dataframe with 'mpio_code' and 'IV_mpio'
            corr: Spearman correlation matrix (vulnerabilities x malaria indicators)
            weights: pd.Series of normalized weights for vulnerability variables (sum to 1)

    Notes:
        - Vulnerability flags are binary; municipal aggregation uses means (proportions).
        - Weights are derived by rescaling the average Spearman correlation with malaria
          into [0,1] and normalizing to sum to 1.
    -------------------------------------------------------------------------
    """
    df_personas = pd.read_csv(personas_csv)
    df_hogares  = pd.read_csv(hogares_csv)

    df_personas["v_analfabeta"]   = (df_personas["P_ALFABETA"] == 2).astype(int)
    df_personas["v_bajo_nivel"]   = df_personas["P_NIVEL_ANOSR"].isin([1, 2, 10]).astype(int)
    df_personas["v_edad_riesgo"]  = df_personas["P_EDADR"].isin([1, 13, 14, 15, 16, 17, 18, 19, 20, 21]).astype(int)
    df_personas["v_trabajo"]      = df_personas["P_TRABAJO"].isin([4, 5, 7, 8, 9]).astype(int)
    df_personas["v_calidad_salud"]= df_personas["PA1_CALIDAD_SERV"].isin([3, 4]).astype(int)

    df_hogares["v_cocina"]        = df_hogares["H_DONDE_PREPALIM"].isin([5, 6]).astype(int)
    df_hogares["v_agua"]          = (~df_hogares["H_AGUA_COCIN"].isin([1, 2, 11])).astype(int)
    df_hogares["v_hacinamiento"]  = (df_hogares["HA_TOT_PER"] / df_hogares["H_NRO_DORMIT"] > 3).astype(int)

    df_vuln = df_personas.merge(
        df_hogares[["COD_ENCUESTAS", "v_cocina", "v_agua", "v_hacinamiento"]],
        on="COD_ENCUESTAS",
        how="left"
    )

    vuln_vars = [
        "v_analfabeta", "v_bajo_nivel", "v_edad_riesgo",
        "v_trabajo", "v_calidad_salud", "v_cocina", "v_agua", "v_hacinamiento"
    ]

    df_vuln_mpio = df_vuln.groupby(["U_DPTO", "U_MPIO"])[vuln_vars].mean().reset_index()
    df_vuln_mpio["mpio_code"] = (
        df_vuln_mpio["U_DPTO"].astype(str).str.zfill(2) +
        df_vuln_mpio["U_MPIO"].astype(str).str.zfill(3)
    )

    df_merged = df_malaria_total.merge(df_vuln_mpio, on="mpio_code", how="inner")
    malaria_vars = ["inc_total_pop", "inc_comp_pop", "inc_vivax_pop", "inc_falci_pop"]

    corr = df_merged[vuln_vars + malaria_vars].corr(method="spearman").loc[vuln_vars, malaria_vars]
    weights_rescale = (corr.mean(axis=1) + 1) / 2
    weights_rescale = weights_rescale / weights_rescale.sum()

    w = weights_rescale.reindex(vuln_vars).fillna(0).astype(float)
    w_sum = w.sum()
    if w_sum <= 0:
        raise ValueError("Weights sum to 0. Check the correlations and input data.")
    w = w / w_sum

    df_vuln_mpio = df_vuln_mpio.copy()
    df_vuln_mpio["IV_mpio"] = (df_vuln_mpio[vuln_vars] * w.values).sum(axis=1)
    df_vuln_mpio.to_csv('data_processed/vuln_index_mpio.csv',index=False)
    plot_heatmap(corr)
    return df_vuln_mpio


def attach_climate_to_gdf(
    gdf_map: gpd.GeoDataFrame,
    chirps_path: str = "data_processed/climate_chirps_weekly_p05_2017_2022_long.csv",
    terra_path:  str = "data_processed/climate_terraclimate_monthly_mpio.csv",
) -> gpd.GeoDataFrame:
    """
    -------------------------------------------------------------------------
    Name: attach_climate_to_gdf
    Purpose:
        Read climate datasets (CHIRPS weekly precip; TerraClimate monthly temp),
        aggregate to municipality, convert temperature from °F to °C, and merge
        both climate feature sets into gdf_map by 'mpio_code'.

    Inputs:
        gdf_map : GeoDataFrame with 'mpio_code'
        chirps_path : CSV path for CHIRPS weekly long data
            Required cols: ['mpio_code','precip', ...]
        terra_path  : CSV path for TerraClimate monthly wide data
            Required cols: ['date', <five-digit municipal columns>]

    Outputs:
        GeoDataFrame with added columns:
            - precip_mean, precip_sum, precip_std
            - tmean_mean_c, tmean_std_c   (temperature in °C)
    -------------------------------------------------------------------------
    """
    gdf = gdf_map.copy()
    gdf["mpio_code"] = gdf["mpio_code"].astype(str).str.zfill(5)

    # --- CHIRPS: precip aggregation ---
    df_chirps = pd.read_csv(chirps_path)
    df_chirps["mpio_code"] = (
        df_chirps["mpio_code"].astype(float).astype(int).astype(str).str.zfill(5)
    )
    df_chirps_agg = (
        df_chirps.groupby("mpio_code")["precip_week_mm"]
        .agg(precip_mean="mean", precip_sum="sum", precip_std="std")
        .reset_index()
    )

    # --- TerraClimate: tmean wide -> long -> aggregate (to Celsius) ---
    df_terra = pd.read_csv(terra_path)
    id_cols = ["date"]
    df_terra_long = df_terra.melt(
        id_vars=id_cols, var_name="mpio_code", value_name="tmean_f"
    )
    df_terra_long["mpio_code"] = df_terra_long["mpio_code"].astype(str).str.zfill(5)
    df_terra_long["tmean_c"] = (df_terra_long["tmean_f"] - 32) * (5 / 9)

    df_temp = (
        df_terra_long.groupby("mpio_code")["tmean_c"]
        .agg(tmean_mean_c="mean", tmean_std_c="std")
        .reset_index()
    )

    # --- Merge both climate tables into gdf_map ---
    gdf = gdf.merge(df_chirps_agg, on="mpio_code", how="left")
    gdf = gdf.merge(df_temp,      on="mpio_code", how="left")

    return gdf

def make_maps(df_vuln_mpio, df_mal, gdf=None):
    """
    Produce side-by-side choropleth maps for a vulnerability index and malaria outcomes.

    Workflow
    --------
    1) Ensure a municipalities GeoDataFrame is available (download if not provided).
    2) Join municipal shapes with:
       - Vulnerability index at municipality level (df_vuln_mpio[['mpio_code','IV_mpio']]).
       - Malaria outcomes and ancillary climate summaries (df_mal, keyed by 'mpio_code').
    3) Attach climate attributes needed for mapping (via `attach_climate_to_gdf`).
    4) For each (indicator, malaria outcome) pair, prepare a value column:
       - If the malaria column is an incidence rate (name starts with 'inc'),
         convert to per-100k population for readability.
       - Otherwise, plot the raw variable.
    5) Render maps using `plot_outvar_maps` with appropriate titles and scaling.

    Parameters
    ----------
    df_vuln_mpio : pd.DataFrame
        Municipality-level table with columns ['mpio_code', 'IV_mpio'].
    df_mal : pd.DataFrame
        Municipality-level table keyed by 'mpio_code' containing malaria outcomes and
        climate summaries (e.g., 'inc_total_pop', 'precip_mean', 'tmean_mean_c').
    gdf : geopandas.GeoDataFrame or None
        Municipal boundary geometries keyed by 'mpio_code'. If None, shapes are
        retrieved via `download_mpios_gdf()`.

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame with merged attributes used for mapping (including derived
        per-100k columns where applicable).
    """
    if gdf is None:
        gdf = download_mpios_gdf()

    merged = (
        gdf.merge(df_vuln_mpio[["mpio_code", "IV_mpio"]], on="mpio_code")
           .merge(df_mal, on="mpio_code")
    )

    ind_cols = ["IV_mpio", 'precip_mean', 'precip_sum', 'tmean_mean_c']
    malaria_cols = ['inc_total_pop','inc_comp_pop', 'inc_vivax_pop', 'inc_falci_pop']
    dic_titles = {"IV_mpio": "Vulnerability Index (IV_mpio)", 'precip_mean': 'Precipitation mean', 'precip_sum': 'Precipitation sum', 'tmean_mean_c': 'Temperature mean (°C)'}

    gdf_map = attach_climate_to_gdf(merged)
    
    for ind_col, malaria_col in zip(ind_cols, malaria_cols):
        per_100k = False
        if malaria_col.startswith("inc"):
            value_col = f"{malaria_col}_100k"
            gdf_map[value_col] = gdf_map[malaria_col] * 100000
            per_100k = True
        plot_outvar_maps(gdf_map, per_100k, ind_col, malaria_col, value_col, dic_titles[ind_col])

    return gdf_map


def build_time_series_df(
    df_week: pd.DataFrame,
) -> pd.DataFrame:
    """
    Joins weekly malaria incidence (df_week) with weekly precipitation (CSV) by municipality and week.
    Converts df_week’s year/week into an ISO week-end (Sunday) date in week_end.
    Merges with the CSV, which already contains week_end in YYYY-MM-DD format.
    Returns a modeling-ready DataFrame (weekly spatio-temporal panel).
    """

    dfw = df_week.copy()
    climate_csv_path = "data_processed/climate_chirps_weekly_p05_2017_2022_long.csv"
    incidence_cols = ["inc_total_pop", "inc_comp_pop", "inc_vivax_pop", "inc_falci_pop"]

    dfw = dfw.rename(columns={'ano':'year','semana':'week'})

    dfw["year"] = pd.to_numeric(dfw["year"], errors="coerce").astype("Int64")
    dfw["week"] = pd.to_numeric(dfw["week"], errors="coerce").astype("Int64")

    dfw = dfw.dropna(subset=["year", "week"]).copy()
    dfw["week_end"] = pd.to_datetime(
        dfw["year"].astype(str) + "-W" + dfw["week"].astype(str).str.zfill(2) + "-7",
        format="%G-W%V-%u",
        errors="coerce"
    )

    dfw['mpio_code'] = dfw['mpio_code'].astype(str).str.strip()

    clima = pd.read_csv(climate_csv_path)

    clima["week_end"] = pd.to_datetime(clima["week_end"], errors="coerce", utc=False).dt.tz_localize(None)
    clima = clima.dropna(subset=["week_end"]).copy()
    clima['mpio_code'] = pd.to_numeric(clima['mpio_code'], errors='coerce').astype('int').astype(str).str.zfill(5)
    clima = clima.groupby(["mpio_code", "week_end"], as_index=False)["precip_week_mm"].mean()

    keep_cols = ['mpio_code', "week_end"] + list(incidence_cols)
    dfw_keep = dfw[keep_cols].copy()
    merged = dfw_keep.merge(
        clima[['mpio_code', "week_end", "precip_week_mm"]],
        on=['mpio_code', "week_end"],
        how="left",
        validate="m:1"
    )

    missing_precip = merged["precip_week_mm"].isna().mean()
    print(f"[INFO] % non merged rows: {missing_precip:.2%}")

    merged = merged.sort_values(['mpio_code', "week_end"]).reset_index(drop=True)

    merged = merged.dropna().rename(columns={'week_end':'date'}).sort_values('date')
    return merged