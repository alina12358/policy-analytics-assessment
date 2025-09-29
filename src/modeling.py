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