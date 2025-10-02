import math
from typing import List, Optional, Tuple, Dict

import geopandas as gpd
import pandas as pd
import libpysal
from esda.moran import Moran, Moran_Local
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import numpy as np
import importlib
import numpy as np, pandas as pd, math
import libpysal
from spreg import ML_Lag

def compute_spatial_autocorr(
    gdf_map: gpd.GeoDataFrame,
    malaria_cols: List[str] = None,
    climate_cols: List[str] = None,     # e.g., ["precip_mean", "tmean_mean"]
    vuln_col: Optional[str] = "IV_mpio",
    weights: Optional[libpysal.weights.W] = None,
    weight_kind: str = "Queen",         # "Queen" or "Rook"
    id_cols: Optional[List[str]] = None # for reference in outputs
) -> Tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """
    Compute global Moran's I and LISA for selected variables and append per-variable
    LISA outputs to the GeoDataFrame. Returns (augmented_gdf, summary_df) where
    summary_df lists global Moran stats by variable.

    Variables:
      - malaria_cols: ['inc_total_pop','inc_comp_pop','inc_vivax_pop','inc_falci_pop']
      - climate_cols: place your precipitation/temperature vars here
      - vuln_col:     municipal vulnerability index (e.g., 'IV_mpio')

    LISA outputs added per variable with suffix pattern:
      <var>__lisa_I, <var>__lisa_p, <var>__lisa_q  (q ∈ {1,2,3,4}: HH, LH, LL, HL)

    Notes:
      - Weights are row-standardized ("r").
      - Returns a tidy summary with: variable, I, p_sim, n.
    """
    gdf = gdf_map.copy()

    malaria_cols = malaria_cols or ['inc_total_pop','inc_comp_pop','inc_vivax_pop','inc_falci_pop']
    climate_cols = climate_cols or []   # e.g., ['precip_mean','tmean_mean']
    vars_all: List[str] = malaria_cols + climate_cols + ([vuln_col] if vuln_col and vuln_col in gdf.columns else [])

    if not vars_all:
        raise ValueError("No variables to analyze. Provide malaria_cols/climate_cols and/or a valid vuln_col.")

    # Build spatial weights if not provided
    if weights is None:
        if weight_kind.lower() == "rook":
            w = libpysal.weights.Rook.from_dataframe(gdf)
        else:
            w = libpysal.weights.Queen.from_dataframe(gdf)
    else:
        w = weights
    w.transform = "r"

    # Compute global Moran and LISA per variable
    rows: List[Dict] = []
    for var in vars_all:
        if var not in gdf.columns:
            # silently skip missing columns to keep pipeline robust
            continue
        y = gdf[var].astype(float).values
        mor = Moran(y, w)
        lis = Moran_Local(y, w)

        # attach LISA results with var-specific suffix
        gdf[f"{var}__lisa_I"] = lis.Is
        gdf[f"{var}__lisa_p"] = lis.p_sim
        gdf[f"{var}__lisa_q"] = lis.q.astype(int)

        row = {
            "variable": var,
            "I": mor.I,
            "p_sim": mor.p_sim,
            "EI": mor.EI,
            "VI": mor.VI_norm,
            "z": mor.z_norm,
            "n": len(y),
        }
        # attach optional IDs for traceability (first few to hint provenance)
        if id_cols:
            for c in id_cols:
                if c in gdf.columns:
                    row[f"example_{c}"] = gdf[c].iloc[0]
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values("variable").reset_index(drop=True)
    return gdf, summary


def plot_lisa_grid(
    gdf_map: gpd.GeoDataFrame,
    variables: List[str],
    ncols: int = 3,
    figsize: Tuple[int, int] = (16, 10),
    cluster_suffix: str = "__lisa_q",
    pval_suffix: str = "__lisa_p",
    p_threshold: Optional[float] = 0.05,
    show_titles: bool = True,
):
    """
    Plot LISA cluster maps with consistent colors & human-readable labels.
    Cluster codes: 1=HH, 2=LH, 3=LL, 4=HL, and 0=Not significant (when masked).
    """
    # Fixed order and labels (keep this order for consistent colors)
    code_to_label = {
        0: "Not significant",
        1: "High-High",
        2: "Low-High",
        3: "Low-Low",
        4: "High-Low",
    }
    # Fixed colors matching the order above
    # 0 grey, 1 red, 2 orange, 3 blue, 4 green
    colors = ["#BDBDBD", "#D73027", "#FC8D59", "#4575B4", "#1A9850"]
    cmap = ListedColormap(colors)
    categories = list(code_to_label.values())  # ordered labels
    codes_order = list(code_to_label.keys())   # [0,1,2,3,4]

    # Grid
    n = len(variables)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
    axes = [ax for ax in (axes.ravel() if hasattr(axes, "ravel") else axes)]

    for i, var in enumerate(variables):
        ax = axes[i]
        q_col = f"{var}{cluster_suffix}"
        p_col = f"{var}{pval_suffix}"

        if q_col not in gdf_map.columns:
            ax.set_axis_off()
            ax.set_title(f"{var}: missing LISA")
            continue

        plot_gdf = gdf_map.copy()

        # Mask by significance -> set 0 (Not significant)
        if p_col in plot_gdf.columns and p_threshold is not None:
            sig = plot_gdf[p_col] <= p_threshold
            plot_gdf.loc[~sig, q_col] = 0

        # Map numeric codes to labels and enforce categorical order
        lab_col = f"{var}__lisa_label"
        plot_gdf[lab_col] = plot_gdf[q_col].map(code_to_label).fillna("Not significant")
        plot_gdf[lab_col] = pd.Categorical(plot_gdf[lab_col], categories=categories, ordered=True)

        # Plot categorical by labels using the fixed colormap
        plot_gdf.plot(
            column=lab_col,
            categorical=True,
            legend=False,          # we'll draw a shared legend
            ax=ax,
            cmap=cmap,
            linewidth=0.2,
            edgecolor="black"
        )

        if show_titles:
            ax.set_title(var)
        ax.set_axis_off()

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_axis_off()

    # Shared legend (same for all subplots)
    legend_handles = [mpatches.Patch(color=colors[k], label=label) for k, label in enumerate(categories)]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=len(categories),
        frameon=True,
        title="LISA clusters"
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for legend on top
    plt.show()

def annualize_and_standardize(
    gdf,
    inc_cols=("inc_total_pop","inc_comp_pop","inc_vivax_pop","inc_falci_pop"),
    years=6,
    pred_cols=("IV_mpio","precip_mean","tmean_mean_c"),
):
    
    """
    Annualize and z-standardize incidence rates and selected predictors.
    --------------
    - Creates annualized incidence metrics by dividing weekly (or multi-period) incidence by the number of years.
    - Computes z-scores (mean 0, std 1) for the annualized incidence columns and for selected predictor columns.

    Inputs
    ------
    gdf : pandas.DataFrame or GeoDataFrame
        Input table with incidence columns and (optionally) predictor columns.
    inc_cols : tuple[str]
        Incidence columns to annualize and standardize.
    years : int or float
        Number of years used to annualize incidence (e.g., 6 for 2017–2022).
    pred_cols : tuple[str]
        Predictor columns to standardize if present (silently skipped if absent).

    Outputs
    -------
    DataFrame (same type as input) with added columns:
        - "{inc}_ann"   : annualized incidence = inc / years
        - "{inc}_ann_z" : standardized annualized incidence
        - "{pred}_z"    : standardized predictor (for each present in pred_cols)

    Notes
    -----
    - Standardization uses population (ddof=0) standard deviation.
    - If the standard deviation is zero or NaN, the corresponding z-score column is set to 0 (avoids division by zero).
    - Assumes `numpy as np` is imported.
    """
    g = gdf.copy()
    for c in inc_cols:
        g[f"{c}_ann"]   = g[c] / years
        s = g[f"{c}_ann"].std(ddof=0); m = g[f"{c}_ann"].mean()
        g[f"{c}_ann_z"] = (g[f"{c}_ann"] - m) / s if s not in (0, np.nan) else g[f"{c}_ann"]*0
    for c in pred_cols:
        if c in g.columns:
            s = g[c].std(ddof=0); m = g[c].mean()
            g[f"{c}_z"] = (g[c] - m) / s if s not in (0, np.nan) else g[c]*0
    return g

def _p_from_z(z):
    # two-sided p using erf (sin SciPy)
    if z is None or not np.isfinite(z): return np.nan
    Phi = 0.5*(1.0 + math.erf(abs(z)/math.sqrt(2.0)))
    return 2.0*(1.0 - Phi)

def fit_sdm_with_wx(gdf, y_col, x_cols, weight_kind="Queen"):
    """
    Spatial Durbin Model (SDM) via ML_Lag with WX terms.
    
    --------------
    - Builds a spatial weights matrix W from the input GeoDataFrame (Queen or Rook contiguity).
    - Constructs the response vector y and covariate matrix X.
    - Computes spatially-lagged covariates WX and augments X -> [X, WX].
    - Fits an ML spatial lag model (ML_Lag) using the augmented design matrix (SDM flavor).
    - Extracts direct (X) and indirect (WX) effects with z- and p-values.
    - Returns the fitted model object and a tidy effects DataFrame.
    
    Inputs
    ------
    gdf : GeoDataFrame with polygons (required for contiguity weights).
    y_col : str, name of dependent variable column in gdf.
    x_cols : list[str], names of explanatory variables in gdf.
    weight_kind : {"Queen","Rook"}, type of contiguity to use.

    Outputs
    -------
    sdm : pysal.model.spreg.ML_Lag object (fitted).
    effects : pandas.DataFrame with columns:
        - variable, direct, p_direct, indirect_WX, p_indirect, total(direct+indirect)
      and attributes:
        - effects.attrs["rho"], effects.attrs["pseudo_r2"], effects.attrs["n"]

    Notes
    -----
    - Assumes `libpysal`, `numpy as np`, `pandas as pd`, and `ML_Lag` are imported.
    - Assumes a helper `_p_from_z(z)` is available and returns a two-sided p-value.
    - W is row-standardized (W.transform = "r") so indirect effects are interpretable.
    """

    W = (libpysal.weights.Rook.from_dataframe(gdf) if weight_kind.lower()=="rook"
         else libpysal.weights.Queen.from_dataframe(gdf))
    W.transform = "r"

    y = gdf[[y_col]].to_numpy()
    X = gdf[x_cols].to_numpy()

    # WX
    try:    WX = W.sparse @ X
    except AttributeError:
        WX = W.full()[0] @ X

    X_aug  = np.hstack([X, WX])
    name_x = list(x_cols) + [f"W_{c}" for c in x_cols]

    sdm = ML_Lag(y, X_aug, W, name_y=y_col, name_x=name_x, name_w="W", method="full")

    # betas / SE (constante + X_aug)  -> z, p
    betas = np.asarray(sdm.betas).ravel()
    ses   = np.asarray(sdm.std_err).ravel()
    names = ["CONSTANT"] + name_x

    coef_map = {}
    for nm, b, se in zip(names, betas, ses):
        z = (b / se) if (se and np.isfinite(se) and se!=0) else np.nan
        coef_map[nm] = {"coef": float(b), "z": float(z) if np.isfinite(z) else np.nan, "p": _p_from_z(z)}

    rows = []
    for c in x_cols:
        d   = coef_map.get(c,  {}).get("coef", np.nan)
        pd_ = coef_map.get(c,  {}).get("p",    np.nan)
        wi  = f"W_{c}"
        ind = coef_map.get(wi, {}).get("coef", np.nan)
        pi_ = coef_map.get(wi, {}).get("p",    np.nan)
        rows.append({
            "variable": c,
            "direct": d,
            "p_direct": pd_,
            "indirect_WX": ind,
            "p_indirect": pi_,
            "total(direct+indirect)": d + ind if np.isfinite(d) and np.isfinite(ind) else np.nan
        })

    effects = pd.DataFrame(rows)
    # rho (Wy)
    rho = getattr(sdm, "rho", np.nan)
    effects.attrs["rho"] = float(rho) if np.isfinite(rho) else np.nan
    effects.attrs["pseudo_r2"] = getattr(sdm, "pr2", None)
    effects.attrs["n"] = len(gdf)
    return sdm, effects

def print_sdm_effects_table(effects: pd.DataFrame, title="SDM effects (ML_Lag + WX)"):
    if effects is None or effects.empty:
        print("No effects to display."); return
    rho = effects.attrs.get("rho", np.nan)
    pr2 = effects.attrs.get("pseudo_r2", None)
    n   = effects.attrs.get("n", None)

    tbl = effects.copy()
    for c in ["direct","indirect_WX","total(direct+indirect)"]:
        if c in tbl: tbl[c] = tbl[c].map(lambda x: f"{x: .4f}" if pd.notna(x) else "")
    for c in ["p_direct","p_indirect"]:
        if c in tbl: tbl[c] = tbl[c].map(lambda x: f"{x: .4f}" if pd.notna(x) else "")

    print(f"\n=== {title} ===")
    if n is not None and pr2 is not None:
        print(f"n={n}  |  pseudo R²={pr2:.3f}")
    if np.isfinite(rho):
        print(f"rho (W*y) = {rho:.4f}")
    print(tbl[["variable","direct","p_direct","indirect_WX","p_indirect","total(direct+indirect)"]]
          .to_string(index=False))    
    
import numpy as np
import pandas as pd
import statsmodels.api as sm
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def ensure_panel_types(
    df: pd.DataFrame,
    group_col: str = "mpio_code",
    time_col: str = "date",
) -> pd.DataFrame:
    """
    Ensure panel keys have consistent dtypes.
    - Parses the time column to naive pandas datetime.
    - Casts the group column to 5-character zero-padded strings.

    Parameters
    ----------
    df : pd.DataFrame
        Input panel data with group and time keys.
    group_col : str
        Group identifier column name.
    time_col : str
        Time column name.

    Returns
    -------
    pd.DataFrame
        Copy of df with normalized dtypes.
    """
    g = df.copy()
    g[time_col] = pd.to_datetime(g[time_col], errors="coerce")
    try:
        g[time_col] = g[time_col].dt.tz_localize(None)
    except Exception:
        pass
    g[group_col] = g[group_col].astype(str).str.strip().str.zfill(5)
    return g


def make_lags(
    df: pd.DataFrame,
    group_col: str,
    time_col: str,
    x_col: str,
    K: int = 4,
) -> pd.DataFrame:
    """
    Create distributed lag features within each group for a weekly exposure.

    Parameters
    ----------
    df : pd.DataFrame
        Panel data sorted or sortable by (group, time).
    group_col : str
        Group identifier column name.
    time_col : str
        Time column name.
    x_col : str
        Exposure column to lag.
    K : int
        Maximum lag order to generate.

    Returns
    -------
    pd.DataFrame
        DataFrame with added columns f"{x_col}_lag{k}" for k in 0..K.
    """
    g = df.sort_values([group_col, time_col]).copy()
    for k in range(K + 1):
        g[f"{x_col}_lag{k}"] = g.groupby(group_col, group_keys=False)[x_col].shift(k)
    return g


def build_rainy_keep_mask(
    df: pd.DataFrame,
    group_col: str = "mpio_code",
    date_col: str = "date",
    precip_col: str = "precip_week_mm",
    month_threshold_mm: float = 5.0,
    max_lag: int = 4,
) -> pd.DataFrame:
    """
    Flag weeks to keep based on rainy months and a backward window for lagged carryover.
    A week is kept if its month has monthly precipitation >= threshold or if any of the
    previous K weeks belonged to a rainy month within the same group.

    Parameters
    ----------
    df : pd.DataFrame
        Weekly panel with precipitation and keys.
    group_col : str
        Group identifier column.
    date_col : str
        Time column (weekly dates).
    precip_col : str
        Weekly precipitation column in mm.
    month_threshold_mm : float
        Minimum monthly precipitation to consider a month as rainy.
    max_lag : int
        Number of previous weeks to include in the keep window.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns [group_col, date_col, precip_month_mm, rainy_month, keep_rainy_window].
    """
    g = df[[group_col, date_col, precip_col]].copy()
    g[date_col] = pd.to_datetime(g[date_col], errors="coerce")
    g["year"] = g[date_col].dt.year.astype(int)
    g["month"] = g[date_col].dt.month.astype(int)
    mon = (
        g.groupby([group_col, "year", "month"], as_index=False)[precip_col]
        .sum()
        .rename(columns={precip_col: "precip_month_mm"})
    )
    g = g.merge(mon, on=[group_col, "year", "month"], how="left")
    g["rainy_month"] = g["precip_month_mm"] >= float(month_threshold_mm)
    g = g.sort_values([group_col, date_col])

    def _keep_window(s: pd.Series) -> pd.Series:
        keep = s.copy()
        for L in range(1, max_lag + 1):
            keep = keep | s.shift(L, fill_value=False)
        return keep

    g["keep_rainy_window"] = (
        g.groupby(group_col, group_keys=False)["rainy_month"].apply(_keep_window)
    )
    return g[[group_col, date_col, "precip_month_mm", "rainy_month", "keep_rainy_window"]]


def merge_rainy_window(
    df: pd.DataFrame,
    rainy_mask: pd.DataFrame,
    group_col: str = "mpio_code",
    date_col: str = "date",
) -> pd.DataFrame:
    """
    Merge rainy-month window flags into the panel.

    Parameters
    ----------
    df : pd.DataFrame
        Base panel data.
    rainy_mask : pd.DataFrame
        Output from build_rainy_keep_mask with keys and flags.
    group_col : str
        Group identifier column.
    date_col : str
        Time column.

    Returns
    -------
    pd.DataFrame
        Panel with merged rainy-month fields.
    """
    cols = [group_col, date_col, "precip_month_mm", "rainy_month", "keep_rainy_window"]
    return df.merge(rainy_mask[cols], on=[group_col, date_col], how="left")


def _cum_effect_from_results(
    res: sm.regression.linear_model.RegressionResultsWrapper,
    lag_names: list[str],
) -> tuple[float, float, float, float]:
    """
    Compute cumulative effect, standard error, z-statistic, and p-value for a set of lag coefficients.

    Parameters
    ----------
    res : RegressionResultsWrapper
        Fitted OLS results with HAC covariance.
    lag_names : list of str
        Names of lag coefficient columns to be summed.

    Returns
    -------
    tuple
        (cum_effect, cum_se, cum_z, cum_p). Returns NaN if inputs are not estimable.
    """
    beta = res.params.reindex(lag_names)
    V = res.cov_params().reindex(index=lag_names, columns=lag_names)
    if beta.isnull().any() or V.isnull().any().any():
        return np.nan, np.nan, np.nan, np.nan
    c = np.ones(len(lag_names))
    cum = float(c @ beta.values)
    var = float(c @ V.values @ c)
    se = np.sqrt(var) if var >= 0 else np.nan
    z = (cum / se) if (se and np.isfinite(se) and se > 0) else np.nan
    try:
        from scipy.stats import norm
        p = 2 * (1 - norm.cdf(abs(z))) if np.isfinite(z) else np.nan
    except Exception:
        p = np.nan
    return cum, se, z, p


def fit_per_municipio_linear(
    df: pd.DataFrame,
    outcome: str,
    group_col: str = "mpio_code",
    time_col: str = "date",
    precip_col: str = "precip_week_mm",
    K: int = 4,
    hac_maxlags: int = 8,
    min_obs: int = 40,
    keep_col: str | None = None,
) -> pd.DataFrame:
    """
    Fit per-municipality OLS with weekly rainfall lags and HAC (Newey–West) standard errors.
    Model per group: y_t = α + Σ_{k=0..K} β_k * precip_{t-k} + ε_t.
    Rows can be restricted using a boolean keep column (e.g., rainy-season window).

    Parameters
    ----------
    df : pd.DataFrame
        Weekly panel with keys, outcome, and precipitation.
    outcome : str
        Outcome column to model.
    group_col : str
        Group identifier column.
    time_col : str
        Time column.
    precip_col : str
        Weekly precipitation column.
    K : int
        Maximum lag order included.
    hac_maxlags : int
        HAC lag parameter for Newey–West covariance.
    min_obs : int
        Minimum number of observations required to fit per group.
    keep_col : str or None
        Optional boolean column to filter rows before fitting.

    Returns
    -------
    pd.DataFrame
        Per-group estimates with cumulative effect and per-lag coefficients.
    """
    cols = [group_col, time_col, outcome, precip_col]
    if keep_col and keep_col in df.columns:
        cols.append(keep_col)
    use = df[cols].copy()
    use = make_lags(use, group_col, time_col, precip_col, K)
    lag_cols = [f"{precip_col}_lag{k}" for k in range(K + 1)]
    rows = []
    for mpio, g in use.groupby(group_col, sort=False):
        if keep_col and keep_col in g.columns:
            g = g[g[keep_col].fillna(False)].copy()
        g = g.dropna(subset=[outcome] + lag_cols)
        if len(g) < min_obs:
            rows.append({"mpio_code": mpio, "n_obs": len(g), "status": "too_few_obs"})
            continue
        X = g[lag_cols].astype(float)
        keep_mask = X.std(ddof=0) > 0
        X = X.loc[:, keep_mask]
        lag_kept = list(X.columns)
        if len(lag_kept) == 0:
            rows.append({"mpio_code": mpio, "n_obs": len(g), "status": "all_zero_lags"})
            continue
        X = sm.add_constant(X, has_constant="add")
        y = g[outcome].astype(float).values
        try:
            ols = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": hac_maxlags})
        except Exception as e:
            rows.append({"mpio_code": mpio, "n_obs": len(g), "status": f"fit_error: {e}"})
            continue
        cum, se, z, p = _cum_effect_from_results(ols, lag_kept)
        row = {
            "mpio_code": mpio,
            "n_obs": len(g),
            "status": "ok",
            "cum_effect": cum,
            "cum_se": se,
            "cum_z": z,
            "cum_p": p,
        }
        for k in range(K + 1):
            name = f"{precip_col}_lag{k}"
            row[f"beta_{k}"] = ols.params.get(name, np.nan)
            row[f"se_{k}"] = ols.bse.get(name, np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def classify_significance(
    df_res: pd.DataFrame,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Classify per-group cumulative effects by sign and significance.

    Parameters
    ----------
    df_res : pd.DataFrame
        Output from fit_per_municipio_linear with cumulative effect and p-value.
    alpha : float
        Significance threshold for two-sided tests.

    Returns
    -------
    pd.DataFrame
        Copy with a 'class' column in {'pos_sig','neg_sig','ns','no_fit'}.
    """
    out = df_res.copy()
    labels = []
    for _, r in out.iterrows():
        if r.get("status") != "ok" or not np.isfinite(r.get("cum_p", np.nan)):
            labels.append("no_fit")
        elif r["cum_p"] < alpha and r["cum_effect"] > 0:
            labels.append("pos_sig")
        elif r["cum_p"] < alpha and r["cum_effect"] < 0:
            labels.append("neg_sig")
        else:
            labels.append("ns")
    out["class"] = labels
    return out


def plot_map_by_outcome(
    gdf_shapes: gpd.GeoDataFrame,
    df_res: pd.DataFrame,
    outcome_name: str,
    code_col: str = "mpio_code",
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot a choropleth of per-municipality cumulative effects classified by sign and significance.

    Parameters
    ----------
    gdf_shapes : gpd.GeoDataFrame
        Geospatial layer with a municipal geometry and an ID column compatible with df_res.
    df_res : pd.DataFrame
        Classified per-group results including ['mpio_code','class'].
    outcome_name : str
        Label for the map title.
    code_col : str
        Column name of the municipal code in both inputs.

    Returns
    -------
    (Figure, Axes)
        Matplotlib figure and axes with the rendered map.
    """
    gdf = gdf_shapes.copy()
    gdf[code_col] = gdf[code_col].astype(str).str.zfill(5)
    res = df_res.copy()
    res[code_col] = res[code_col].astype(str).str.zfill(5)
    gm = gdf.merge(res, on=code_col, how="left")
    color_map = {"pos_sig": "#d73027", "neg_sig": "#4575b4", "ns": "#bdbdbd", "no_fit": "#ffffff"}
    gm["color"] = gm["class"].map(color_map).fillna("#ffffff")
    fig, ax = plt.subplots(1, 1, figsize=(9, 10))
    gm.plot(color=gm["color"], edgecolor="#666666", linewidth=0.3, ax=ax)
    ax.set_title(f"Cumulative rainfall effect (lags 0–4) — {outcome_name}", fontsize=13)
    ax.axis("off")
    handles = [
        Patch(facecolor=color_map["pos_sig"], edgecolor="none", label="Positive, p<0.05"),
        Patch(facecolor=color_map["neg_sig"], edgecolor="none", label="Negative, p<0.05"),
        Patch(facecolor=color_map["ns"], edgecolor="none", label="Not significant"),
        Patch(facecolor=color_map["no_fit"], edgecolor="none", label="No fit / too few obs"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=9, frameon=True)
    plt.tight_layout()
    return fig, ax


def run_per_mpio_with_rainy_filter(
    df: pd.DataFrame,
    outcomes: list[str],
    group_col: str = "mpio_code",
    time_col: str = "date",
    precip_col: str = "precip_week_mm",
    month_threshold_mm: float = 5.0,
    K: int = 4,
    hac_maxlags: int = 8,
    min_obs: int = 30,
    alpha: float = 0.05,
) -> dict[str, pd.DataFrame]:
    """
    End-to-end pipeline to estimate per-municipality linear DLMs only in rainy-season windows.
    Steps:
      1) Normalize keys and types.
      2) Build rainy-month window flags.
      3) Merge flags into the panel.
      4) Fit OLS per municipality with HAC errors using lags 0..K.
      5) Classify by sign and significance for each outcome.

    Parameters
    ----------
    df : pd.DataFrame
        Weekly panel with keys, precipitation, and outcomes.
    outcomes : list of str
        Outcome columns to estimate.
    group_col : str
        Group identifier column.
    time_col : str
        Time column.
    precip_col : str
        Weekly precipitation column.
    month_threshold_mm : float
        Monthly precipitation threshold to define rainy months.
    K : int
        Maximum lag order.
    hac_maxlags : int
        HAC lag parameter for Newey–West covariance.
    min_obs : int
        Minimum observations per group to fit a model.
    alpha : float
        Significance level for classification.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping from outcome name to classified per-municipality results.
    """
    base = ensure_panel_types(df, group_col=group_col, time_col=time_col)
    rainy = build_rainy_keep_mask(
        base,
        group_col=group_col,
        date_col=time_col,
        precip_col=precip_col,
        month_threshold_mm=month_threshold_mm,
        max_lag=K,
    )
    panel = merge_rainy_window(base, rainy, group_col=group_col, date_col=time_col)
    results = {}
    for y in outcomes:
        est = fit_per_municipio_linear(
            panel,
            outcome=y,
            group_col=group_col,
            time_col=time_col,
            precip_col=precip_col,
            K=K,
            hac_maxlags=hac_maxlags,
            min_obs=min_obs,
            keep_col="keep_rainy_window",
        )
        results[y] = classify_significance(est, alpha=alpha)
    return results


def summarize_class_counts(
    results_by_outcome: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Summarize classification counts across outcomes.

    Parameters
    ----------
    results_by_outcome : dict
        Output from run_per_mpio_with_rainy_filter.

    Returns
    -------
    pd.DataFrame
        Table with counts of classes per outcome.
    """
    rows = []
    for y, dfy in results_by_outcome.items():
        vc = dfy["class"].value_counts(dropna=False)
        rows.append(
            {
                "outcome": y,
                "pos_sig": int(vc.get("pos_sig", 0)),
                "neg_sig": int(vc.get("neg_sig", 0)),
                "ns": int(vc.get("ns", 0)),
                "no_fit": int(vc.get("no_fit", 0)),
                "total": int(len(dfy)),
            }
        )
    return pd.DataFrame(rows).sort_values("outcome")


def run_rainy_window_pipeline_with_map(
    df_time: pd.DataFrame,
    outcomes: list[str] = ("inc_total_pop", "inc_comp_pop", "inc_vivax_pop", "inc_falci_pop"),
    shapes_path: str = "data_raw/colombia_mpios/colombia_mpios_opendatasoft.shp",
    group_col: str = "mpio_code",
    time_col: str = "date",
    precip_col: str = "precip_week_mm",
    month_threshold_mm: float = 5.0,
    K: int = 4,
    hac_maxlags: int = 8,
    min_obs: int = 30,
    alpha: float = 0.05,
    map_outcome: str = "inc_total_pop",
    verbose: bool = True,
) -> tuple[dict[str, pd.DataFrame], float, plt.Figure, plt.Axes]:
    """
    Run the rainy-window panel pipeline and always render a map.
    Steps:
      1) Normalize keys and dtypes.
      2) Build rainy-month mask with a K-week carryover window.
      3) Merge flags into the panel.
      4) Fit per-municipality linear DLM with HAC errors for each outcome.
      5) Classify cumulative effects by sign/significance.
      6) Read shapes from `shapes_path` and plot a choropleth for `map_outcome`.
      7) Always call plt.show().

    Parameters
    ----------
    df_time : pd.DataFrame
        Weekly panel with [group_col, time_col, precip_col] and outcome columns.
    outcomes : list of str
        Outcome names to estimate.
    shapes_path : str
        Path to the municipal shapefile or geodata source.
    group_col : str
        Group identifier column name.
    time_col : str
        Weekly date column name.
    precip_col : str
        Weekly precipitation column in mm.
    month_threshold_mm : float
        Monthly precipitation threshold to define a rainy month.
    K : int
        Maximum rainfall lag order (0..K).
    hac_maxlags : int
        HAC lag parameter for Newey–West covariance.
    min_obs : int
        Minimum observations per municipality required to fit.
    alpha : float
        Significance level for classification.
    map_outcome : str
        Outcome to plot on the map; falls back to the first in `outcomes` if missing.
    verbose : bool
        If True, prints the share of kept weeks.

    Returns
    -------
    results_by_outcome : dict[str, pd.DataFrame]
        Classified per-municipality tables for each outcome.
    keep_rate : float
        Share of weeks retained under the rainy-window filter.
    fig : matplotlib.figure.Figure
        Figure handle of the rendered map.
    ax : matplotlib.axes.Axes
        Axes handle of the rendered map.
    """
    base = ensure_panel_types(df_time, group_col=group_col, time_col=time_col)
    rainy = build_rainy_keep_mask(
        base,
        group_col=group_col,
        date_col=time_col,
        precip_col=precip_col,
        month_threshold_mm=month_threshold_mm,
        max_lag=K,
    )
    panel = merge_rainy_window(base, rainy, group_col=group_col, date_col=time_col)
    keep_rate = float(panel["keep_rainy_window"].mean())
    if verbose:
        print(f"[INFO] Kept weeks under rainy-window (K={K}): {keep_rate:.1%}")

    results: dict[str, pd.DataFrame] = {}
    for y in outcomes:
        est = fit_per_municipio_linear(
            panel,
            outcome=y,
            group_col=group_col,
            time_col=time_col,
            precip_col=precip_col,
            K=K,
            hac_maxlags=hac_maxlags,
            min_obs=min_obs,
            keep_col="keep_rainy_window",
        )
        results[y] = classify_significance(est, alpha=alpha)

    gdf_mpios = gpd.read_file(shapes_path)
    candidates = ["mpio_code", "MPIO_CCNCT", "MPIO_CDPMP", "COD_DANE", "COD_MPIO", "mpio", "mpio_cd", "cod_mpio", "cod_dane"]
    code_col = next((c for c in candidates if c in gdf_mpios.columns), None)
    if code_col is None:
        raise ValueError(f"Could not find a municipal code column in shapes. Available columns: {list(gdf_mpios.columns)}")
    gdf_mpios = gdf_mpios.rename(columns={code_col: "mpio_code"}).copy()
    gdf_mpios["mpio_code"] = pd.to_numeric(gdf_mpios["mpio_code"], errors="coerce").astype("Int64").astype(str).str.zfill(5)
    gdf_mpios = gdf_mpios.dropna(subset=["geometry", "mpio_code"]).drop_duplicates(subset=["mpio_code"])

    plot_target = map_outcome if map_outcome in results else outcomes[0]
    fig, ax = plot_map_by_outcome(
        gdf_shapes=gdf_mpios,
        df_res=results[plot_target],
        outcome_name=plot_target,
        code_col="mpio_code",
    )
    plt.show()
    return results, keep_rate, fig, ax
