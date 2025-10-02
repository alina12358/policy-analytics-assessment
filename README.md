# Policy Analytics Challenge — README

## 1) Overview

This repository quantifies how weekly rainfall relates to malaria outcomes at the municipal level in Colombia, focusing on Antioquia, Córdoba, Chocó, Nariño, and Cauca. The pipeline is fully reproducible and produces: (i) a short policy narrative and visuals, (ii) per-municipality estimates for short-lag rainfall effects, and (iii) spatial diagnostics and a Spatial Durbin Model to separate local and spillover effects.

**Core idea:** we restrict estimation to a “rainy window” and model short lags (0–4 weeks) to capture epidemiologically relevant dynamics, then validate spatial clustering and cross-border effects.

---

## 2) Repository structure

```
/code
  01_ingest_clean.*            # load raw links, clean ids/dates, weekly panel merge
  02_feature_engineering.*     # rainy-window mask, lags, standardization, vulnerability index (PCA)
  03_modeling.*                # per-municipality DLM (0–4 lags) with HAC SEs, Moran/LISA, SDM
  04_eval_visuals.*            # tables, maps, policy-oriented figures and text blocks
  99_utils.*                   # helpers (spatial weights, plotting, IO, QC)
  
/data_raw/                     # empty; use /data_links.txt to fetch (do not commit raw data)
/data_links.txt                # public URLs + access steps + license notes

/data_processed/               # intermediate parquet/csv created by the pipeline

/outputs/
  figures/                     # PNG/SVG maps and plots
  tables/                      # CSVs with per-municipality results
  model_artifacts/             # SDM objects, spatial weights, logs
  logs/                        # run logs

/docs/
  slides.pptx                  # ≤ 15 slides
  README.md                    # this file
```

---

## 3) One-click reproduction

### Linux / macOS

```bash
git clone <REPO_URL>
cd <REPO_NAME>
bash run.sh
```

### Windows

```bat
git clone <REPO_URL>
cd <REPO_NAME>
run.bat
```

The run script will:

1. set up the environment,
2. download public datasets listed in `/data_links.txt`,
3. execute the pipeline end-to-end, and
4. write outputs to `/outputs`.

---

## 4) Environment

* Python ≥ 3.10
* Key packages: `pandas`, `numpy`, `geopandas`, `pyproj`, `shapely`, `libpysal`, `esda`, `spreg`, `matplotlib`, `statsmodels`, `linearmodels`, `scikit-learn`
* Reproducibility: versions pinned in `requirements.txt` or `environment.yml`; random seeds set where applicable

---

## 5) Data access

All inputs are public and fetched at runtime via `/data_links.txt`.
Do not commit raw or restricted files. The pipeline writes processed artifacts to `/data_processed/`.

---

## 6) Methods (succinct)

### 6.1 Data construction

* **Weekly panel:** join municipal malaria outcomes with weekly precipitation (CHIRPS) by `mpio_code` and week_end date.
* **Vulnerability index:** z-score the epidemiologic rates, run PCA on the correlation matrix, use sign-adjusted PC1 loadings as weights, and rescale to sum to one; stress-test against equal weights. Reference: [Decancq & Lugo 2008](https://ophi.org.uk/sites/default/files/OPHI-wp18.pdf).

### 6.2 Rainy-window filter

* **Monthly threshold:** compute municipal monthly rainfall, flag months with total ≥ threshold (default 5 mm).
* **Carryover:** keep all weeks in rainy months plus a 4-week carryover to retain short lags after the last rainy month.

### 6.3 Per-municipality distributed-lag model (time series)

* For each municipality estimate
  `y_t = α + Σ_{k=0..4} β_k * rain_{t-k} + ε_t`
* OLS with **Newey–West (HAC)** standard errors (default maxlags=8).
* **Cumulative effect:** β_cum = Σ β_k with delta-method SE using the full covariance.
* **Classification:** pos_sig / neg_sig / ns based on p-value for β_cum.
* Outputs: per-municipality table with β_k, SE_k, β_cum, SE, z, p, status, class; maps of significant municipalities.

### 6.4 Spatial autocorrelation and clusters

* **Weights:** row-standardized Queen contiguity; islands flagged.
* **Global Moran’s I:** permutation p-values.
* **LISA:** local clusters labeled HH, LL, HL, LH, masked by significance; grid of maps for malaria, climate, and vulnerability.

### 6.5 Spatial Durbin Model (cross-section, annualized, z-scores)

* SDM via `spreg.ML_Lag` with explicit WX terms; report:

  * ρ (spatial autoregressive parameter on Wy)
  * direct, indirect (WX), and total effects for each predictor
  * pseudo-R² as fit summary
* Used to separate local associations from cross-border spillovers for precipitation, temperature, and vulnerability.

---

## 7) Key outputs

* `/outputs/tables/`

  * `per_mpio_linear_<outcome>_rainywin.csv` (all)
  * `per_mpio_linear_<outcome>_rainywin_SIGONLY.csv` (significant only)
  * optional FDR-adjusted versions if enabled
* `/outputs/figures/`

  * LISA grid (malaria, climate, vulnerability)
  * SDM effect panels and summary bars
  * per-municipality significance maps for each outcome
* `/docs/slides.pptx` concise policy deck

---

## 8) How to run specific steps

### 8.1 Make the rainy-window panel and map it

```python
from code.03_modeling import run_rainy_window_pipeline_with_map

results, keep_rate, fig, ax = run_rainy_window_pipeline_with_map(
    df_time=df_time,
    outcomes=["inc_total_pop","inc_comp_pop","inc_vivax_pop","inc_falci_pop"],
    shapes_path="data_raw/colombia_mpios/colombia_mpios_opendatasoft.shp",
    month_threshold_mm=5.0,
    K=4,
    hac_maxlags=8,
    min_obs=30,
    alpha=0.05,
    map_outcome="inc_total_pop",
    verbose=True
)
```

Print only significant municipalities and export:

```python
from IPython.display import display
tbl_sig = (results["inc_total_pop"]
           .query("status=='ok' and class in ['pos_sig','neg_sig']")
           .sort_values(['class','cum_p','cum_effect']))
display(tbl_sig.head(25))
tbl_sig.to_csv("outputs/tables/per_mpio_linear_inc_total_pop_rainywin_SIGONLY.csv", index=False)
```

### 8.2 Spatial diagnostics and SDM

* Build Queen weights, run Moran/LISA for all variables, export cluster counts.
* Fit SDM on annualized, standardized predictors and report direct, indirect, total effects.

---

## 9) Interpretation guide

**Time-series DLM:** β_cum is the short-run change in the outcome per millimeter of cumulative rainfall over the 0–4 week window. To communicate magnitudes in cases per 100,000, multiply by 100,000 and by a scenario change in cumulative rainfall (e.g., 10 mm).

**Spatial diagnostics:** large positive Moran’s I confirms clustering. LISA maps show where HH and LL clusters are; few HL/LH outliers indicate smooth transitions.

**SDM:** rainfall typically shows positive direct effects within municipalities, while negative WX terms temper the total effect, consistent with cross-border dynamics. Vulnerability contributes directly for total and falciparum, indirectly for complicated cases. Temperature effects are mixed.

---

## 10) Quality assurance and robustness

* ID normalization to 5-digit `mpio_code`, week alignment by ISO week-end date
* Dry-season exclusion via rainy window with carryover
* HAC SEs to address serial correlation
* Island handling in spatial weights, optional k-NN sensitivity
* Alternative neighbor definitions (Queen/Rook/k-NN)
* Optional multiple-testing control via Benjamini–Hochberg FDR for per-municipality screening
* Sensitivity to lag length and rainy thresholds

---

## 11) Policy translation (one paragraph)

Results indicate that malaria risk clusters geographically and responds to short-run rainfall within municipalities, with neighboring conditions moderating net effects. This supports cluster-based targeting of surveillance and vector control in persistent HH areas and monitoring of rainfall-driven windows where short lags elevate risk. Cross-border coordination between adjacent municipalities is warranted since spillovers can attenuate or redirect local effects.

---

## 12) References

* Regional burden targeting: [Colombian Ministry of Health 2025](https://www.minsalud.gov.co/salud/publica/PET/Paginas/malaria.aspx)
* Data-driven weighting for composite indices: [Decancq & Lugo 2008](https://ophi.org.uk/sites/default/files/OPHI-wp18.pdf)

---

## 13) Contact

Questions or issues: open an issue in the repo or email **[Your Name]** at **[your.email@domain]**.
