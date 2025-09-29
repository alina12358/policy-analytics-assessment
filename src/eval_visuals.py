import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
import seaborn as sns
import importlib
import unicodedata
import re
import src.utils 
importlib.reload(src.utils)
from src.utils import *
import src.modeling 
importlib.reload(src.modeling)
from src.modeling import *


def plot_malaria_map_departments(df_dep: pd.DataFrame, title="Malaria cases by department"):
    """
    Merge aggregated department cases with Colombia admin-1 polygons and plot a choropleth.
    df_dep must have columns: ['department','cases'] (and optionally 'share').
    """
    # Prepare agg dataframe
    df = df_dep.copy()
    def _norm(s):
        if pd.isna(s): return s
        s = str(s).strip().upper()
        s = ''.join(c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c))
        s = re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', ' ', s)).strip()
        return s
    # Manual name fixes for tricky ones
    # (Natural Earth uses 'DISTRITO CAPITAL DE BOGOTA' and 'SAN ANDRES PROVIDENCIA Y SANTA CATALINA')
    fixes = {
        "BOGOTA DC": "DISTRITO CAPITAL DE BOGOTA",
        "ARCHIPIELAGO DE SAN ANDRES PROVIDENCIA Y SANTA CATALINA": "SAN ANDRES PROVIDENCIA Y SANTA CATALINA",
        "SAN ANDRES PROVIDENCIA Y SANTA CATALINA": "SAN ANDRES PROVIDENCIA Y SANTA CATALINA",
    }

    # Get polygons and merge
    gdf = download_col_departments_ne10()
    gdf.rename(columns={'name':'department'},inplace=True)
    gdf["dep_norm"] = gdf["department"].apply(_norm)
    df["dep_norm"]  = df["department"].apply(_norm).replace(fixes)

    merged = gdf.merge(df[["dep_norm", "cases"]], on="dep_norm", how="left")
    merged["cases"] = merged["cases"].fillna(0)

    # Plot
    ax = merged.plot(column="cases", legend=True, figsize=(8, 9))
    ax.set_title(title)
    ax.axis("off")

    # Annotate top few
    top = df.sort_values("cases", ascending=False).head(5)
    print("Top 5 departments by cases:")
    print(top[["department","cases"]].to_string(index=False))


def plot_heatmap(corr):
    plt.figure(figsize=(12,6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                cbar_kws={'label': 'Correlación de Spearman'})
    plt.title("Correlación entre vulnerabilidades y malaria por municipio")
    plt.tight_layout()
    plt.show()

def print_moran_summary(summary: pd.DataFrame, alpha: float = 0.05) -> None:
    """
    Pretty-print global Moran's I results.
    Expects columns: ['variable','I','p_sim','z','EI','VI','n'] from compute_spatial_autocorr.
    """
    if summary.empty:
        print("No Moran's results to display.")
        return

    # Significance stars
    def stars(p):
        return "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))

    cols = ["variable", "I", "p_sim", "z", "EI", "VI", "n"]
    view = summary[cols].copy()
    # Format numbers
    view["I"]     = view["I"].map(lambda x: f"{x: .4f}")
    view["p_sim"] = view["p_sim"].map(lambda x: f"{x: .4f}{stars(x)}")
    view["z"]     = view["z"].map(lambda x: f"{x: .3f}")
    view["EI"]    = view["EI"].map(lambda x: f"{x: .4f}")
    view["VI"]    = view["VI"].map(lambda x: f"{x: .4f}")
    view["n"]     = view["n"].astype(int)

    print("\n=== Global Moran's I (per variable) ===")
    print(view.to_string(index=False))
    print(f"\nSignificance at α={alpha}: * p<0.05, ** p<0.01, *** p<0.001")


def print_lisa_cluster_counts(
    gdf: gpd.GeoDataFrame,
    variables: list,
    p_threshold: float = 0.05,
    suffix_q: str = "__lisa_q",
    suffix_p: str = "__lisa_p"
) -> None:
    """
    Print counts of LISA cluster categories per variable.
    Categories: 1=HH, 2=LH, 3=LL, 4=HL; 0=not significant (masked by p_threshold).
    """
    print("\n=== LISA cluster counts (masked by significance) ===")
    header = f"{'variable':<24} {'HH':>5} {'LH':>5} {'LL':>5} {'HL':>5} {'ns':>6} {'total':>7}"
    print(header)
    print("-" * len(header))

    for var in variables:
        q_col = f"{var}{suffix_q}"
        p_col = f"{var}{suffix_p}"
        if q_col not in gdf.columns:
            print(f"{var:<24} (missing LISA columns)")
            continue

        tmp = gdf[[q_col]].copy()
        if p_col in gdf.columns and p_threshold is not None:
            sig = gdf[p_col] <= p_threshold
            tmp.loc[~sig, q_col] = 0  # mark non-significant as 0

        counts = tmp[q_col].value_counts(dropna=False)
        HH = int(counts.get(1, 0))
        LH = int(counts.get(2, 0))
        LL = int(counts.get(3, 0))
        HL = int(counts.get(4, 0))
        NS = int(counts.get(0, 0))
        TOT = int(len(tmp))
        print(f"{var:<24} {HH:5d} {LH:5d} {LL:5d} {HL:5d} {NS:6d} {TOT:7d}")

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

def plot_outvar_maps(gdf, per_100k, ind_col, malaria_col, value_col, title): 
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    gdf.plot(column=ind_col, cmap="Reds", legend=True, ax=axes[0], edgecolor="none")
    axes[0].set_title(title)
    axes[0].axis("off")
    gdf.plot(column=value_col, cmap="PuBu", legend=True, ax=axes[1], edgecolor="none")
    axes[1].set_title(f"Malaria {'Incidence per 100,000' if per_100k else 'Counts'} ({malaria_col})")
    axes[1].axis("off")
    plt.tight_layout()
    plt.show()

def plot_iv_and_climate_vs_malaria(
    gdf_map,
    inc_vars=None,
    iv_col="IV_mpio",
    climate_vars=None,
    cell_size=(4, 3),
):
    """
    Scatter + OLS line grids:
      Row 0:  IV_mpio vs malaria incidence columns.
      Rows 1+: each climate var vs the same malaria columns.
    """
    if inc_vars is None:
        inc_vars = ["inc_total_pop","inc_comp_pop","inc_vivax_pop","inc_falci_pop"]

    if climate_vars is None:
        # prefer Celsius if available
        climate_vars = [v for v in ["precip_mean","tmean_mean_c"] if v and v in gdf_map.columns]

    nrows, ncols = 1 + len(climate_vars), len(inc_vars)
    fig, axes = plt.subplots(nrows, ncols, figsize=(cell_size[0]*ncols, cell_size[1]*nrows))
    axes = axes if nrows > 1 else [axes]  # make row-iterable

    # Row 0: IV vs malaria
    for j, m in enumerate(inc_vars):
        sns.regplot(data=gdf_map, x=iv_col, y=m, ax=axes[0][j],
                    scatter_kws={"alpha":0.5}, line_kws={"lw":2})
        axes[0][j].set_title(f"{iv_col} vs {m}")
        axes[0][j].set_xlabel(iv_col)
        axes[0][j].set_ylabel(m)

    # Next rows: climate vs malaria
    for i, c in enumerate(climate_vars, start=1):
        for j, m in enumerate(inc_vars):
            ax = axes[i][j]
            sns.regplot(data=gdf_map, x=c, y=m, ax=ax,
                        scatter_kws={"alpha":0.5}, line_kws={"lw":2})
            ax.set_title(f"{c} vs {m}")
            ax.set_xlabel(c)
            ax.set_ylabel(m)

    plt.tight_layout()
    plt.show()