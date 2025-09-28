# ---- SIMPLE: CNPV 2018 (DANE) -> solo PER/HOG CSVs, sin borrar tmp ----
import os, csv, time, zipfile, tempfile, shutil
from pathlib import Path
from typing import List, Tuple, Optional
import requests

# ---- SIMPLE: CNPV 2018 (DANE) -> only PER/HOG CSVs for selected IDs ----
import os, csv, time, zipfile, tempfile, shutil
from pathlib import Path
from typing import List, Tuple, Optional
import requests

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
    tables_root = base / "cnpv2018_tables"
    out_personas_path = base / out_personas
    out_hogares_path = base / out_hogares

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

from urllib.parse import urlencode
import pandas as pd

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