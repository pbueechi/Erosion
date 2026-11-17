"""
Assign a tillage type to every field (gdf) from farm-level data (csv).

Default tillage is "Pflug". For every farm in the CSV, its fields in the gdf are
assigned to the farm's tillage type(s) so that the summed field area per type
matches the CSV area (FLAECHE_M2) within +/- TOL. Fields not needed for any
CSV type stay "Pflug".

Outputs
  - GeoPackage with the new columns
        tillage          tillage type per field
        tillage_status   how it was assigned (see STATUS below)
        tillage_certain  True if the field gets the same type in every valid
                         combination, False if the combination is ambiguous
  - CSV log with one row per farm of the CSV file

STATUS (per farm)
  no_fields           farm from CSV has no fields in the gdf
  all_same            one type, all fields match the CSV area
  subset_match        fields > CSV area: a unique subset matches, rest Pflug
  split_match         several types, a unique split of the fields matches
  ambiguous           several field combinations match; the best one is used,
                      fields that differ between combinations are flagged
  fields_smaller_csv  fields sum to less than the CSV area (e.g. CSV area also
                      covers other crops / cantons); one type: all fields get
                      it, several types: fields set to "unklar"
  no_match            no combination within tolerance; fields set to "unklar"
  search_aborted      too many fields to search all combinations; best found
                      combination used if any, else "unklar"
"""
import pyogrio
import os
import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime
from itertools import combinations
from scipy.stats import ttest_ind
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, cohen_kappa_score, 
                            roc_auc_score)
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from xgboost import XGBClassifier

import glob
import multiprocessing as mp
import pickle
import re
import time
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import shapely
import xarray as xr
from pyproj import Transformer
import pyarrow.parquet as pq

# # ---------------------------------------------------------------------------
# # Settings
# # ---------------------------------------------------------------------------
# YEAR = 2022
# GPKG_IN = f'/mnt/eo-nas1/data/landuse/raw/lnf{YEAR}.gpkg'
# CSV_IN = '/mnt/Data-Labo-RE/27_Natural_Resources-RE/321.4_WAUM_protected/Daten/Erosionsrisiko/schonende_bodenbearbeitung.csv'

# GPKG_OUT = f"data/all/lnf_{YEAR}_tillage.gpkg"
# LOG_OUT = f"data/all/tillage_assignment_{YEAR}_log.csv"

# RESTRICT_PFLUG_TO_CSV_CROPS = True 

# CSV_FARM_COL = "betr_ID"
# CSV_CROP_COL = "CODE_KULTUR"
# CSV_AREA_COL = "FLAECHE_M2"
# CSV_TILLAGE_COL = "VERFAHREN"   # <- column with the tillage type, adjust
# GDF_FARM_COL = "betriebsnummer"
# GDF_CROP_COL = "lnf_code"
# GDF_AREA_COL = "groesse"
# GDF_AREA_TO_M2 = 1.0                   # factor if groesse is not in m2 (a: 100, ha: 10000)
 
# N_OTHER_PER_CSV_FARM = 50             # other (non-CSV) farms sampled per CSV farm
# SAMPLE_SEED = 42
# BASE_WHERE = None                      # optional extra filter, e.g. "kanton = 'AG'"

# # Pflug reference (non-CSV farms): only arable land, no forest, meadows etc.
# LNF_CLASS_FILE = "/mnt/eo-nas1/data/landuse/documentation/LNF_code_classification_20260217.xlsx"  # table with LNF_code and Crop_Label_lv3
# ARABLE_LABEL = "Arable Land"

# DEFAULT_TILLAGE = "Pflug"
# UNCLEAR = "unklar"
# TOL = 0.05                             # +/- 5 %
# MAX_SOLUTIONS = 20                    # stop counting combinations after this many
# MAX_NODES = 2_000_000                  # search budget per farm-crop
 
 
# def norm_id(s):
#     """Common id format (1234 vs '1234' vs 1234.0)."""
#     return s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
 
 
# # ---------------------------------------------------------------------------
# # Read only the needed farms from the GeoPackage
# # ---------------------------------------------------------------------------
# def load_arable_codes():
#     """LNF codes with Crop_Label_lv3 == ARABLE_LABEL."""
#     if LNF_CLASS_FILE.lower().endswith((".xlsx", ".xls")):
#         tab = pd.read_excel(LNF_CLASS_FILE)
#     else:
#         tab = pd.read_csv(LNF_CLASS_FILE, sep=None, engine="python")
#     sel = tab["Crop_Label_lv3"].astype(str).str.strip() == ARABLE_LABEL
#     codes = pd.to_numeric(tab.loc[sel, "LNF_code"], errors="coerce").dropna().astype(int)
#     codes = sorted(set(codes))
#     if not codes:
#         raise SystemExit(f"No LNF codes with Crop_Label_lv3 == '{ARABLE_LABEL}' in {LNF_CLASS_FILE}")
#     print(f"{len(codes)} arable LNF codes used for the Pflug reference")
#     return codes
 
 
# def _sql_in(col, values, numeric):
#     vals = ", ".join(str(v) for v in values) if numeric else ", ".join(f"'{v}'" for v in values)
#     return f"{col} IN ({vals})"
 
 
# def load_fields(csv_farms, pflug_codes):
#     """
#     All fields of the CSV farms, plus the fields with a crop in pflug_codes of
#     N_OTHER_PER_CSV_FARM x as many randomly chosen other farms (Pflug reference).
#     """
#     layer = pyogrio.list_layers(GPKG_IN)[0][0]
 
#     ids = pyogrio.read_dataframe(GPKG_IN, layer=layer, columns=[GDF_FARM_COL, GDF_CROP_COL],
#                                  read_geometry=False, where=BASE_WHERE)
#     farm_numeric = pd.api.types.is_numeric_dtype(ids[GDF_FARM_COL])
#     crop_numeric = pd.api.types.is_numeric_dtype(ids[GDF_CROP_COL])
#     ids["_farm"] = norm_id(ids[GDF_FARM_COL])
#     ids["_arable"] = pd.to_numeric(ids[GDF_CROP_COL], errors="coerce").isin(pflug_codes)
 
#     all_farms = pd.Series(ids["_farm"].unique())
#     in_csv = all_farms[all_farms.isin(csv_farms)]
#     # candidate Pflug farms: not in the CSV and with at least one arable field
#     arable_farms = set(ids.loc[ids["_arable"], "_farm"])
#     others = pd.Series(sorted(arable_farms - set(csv_farms)))
#     n_other = min(int(round(len(csv_farms) * N_OTHER_PER_CSV_FARM)), len(others))
#     sampled = others.sample(n=n_other, random_state=SAMPLE_SEED)
#     print(f"Farms in file: {len(all_farms)} | CSV farms: {len(csv_farms)} "
#           f"({len(in_csv)} found in file) | non-CSV farms with eligible crops: {len(others)} "
#           f"| sampled: {n_other}")
 
#     arable_sql = _sql_in(GDF_CROP_COL, pflug_codes if crop_numeric
#                          else [str(c) for c in pflug_codes], crop_numeric)
 
#     def read(farms, extra=None):
#         parts = []
#         farms = list(farms)
#         for i in range(0, len(farms), 1000):
#             where = _sql_in(GDF_FARM_COL, farms[i:i + 1000], farm_numeric)
#             for cond in (BASE_WHERE, extra):
#                 if cond:
#                     where = f"({cond}) AND {where}"
#             parts.append(gpd.read_file(GPKG_IN, layer=layer, where=where))
#         return parts
 
#     parts = read(in_csv) + read(sampled, extra=arable_sql)
#     parts = [p for p in parts if len(p)]
#     gdf = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True),
#                            geometry="geometry", crs=parts[0].crs)
#     print(f"Fields read: {len(gdf)} "
#           f"({gdf[norm_id(gdf[GDF_FARM_COL]).isin(csv_farms)].shape[0]} of CSV farms)")
#     return gdf
 
 
# # ---------------------------------------------------------------------------
# # Combination search
# # ---------------------------------------------------------------------------
# class _Abort(Exception):
#     pass
 
 
# def find_assignments(areas, targets, tol, allow_default):
#     """
#     Assign each field to one of the target types (0..k-1) or, if allow_default,
#     to the default type (k), so that every type's area sum lies within
#     target * (1 +/- tol).
 
#     Returns (solutions, aborted). Each solution is an int array of labels in the
#     original field order.
#     """
#     areas = np.asarray(areas, dtype=float)
#     n, k = len(areas), len(targets)
#     order = np.argsort(-areas)                 # large fields first -> better pruning
#     a = areas[order]
#     lo = [t * (1 - tol) for t in targets]
#     hi = [t * (1 + tol) for t in targets]
#     remaining = np.append(np.cumsum(a[::-1])[::-1], 0.0)
 
#     sums = [0.0] * k
#     labels = np.full(n, -1)
#     solutions = []
#     nodes = 0
 
#     def rec(i):
#         nonlocal nodes
#         nodes += 1
#         if nodes > MAX_NODES:
#             raise _Abort
#         if len(solutions) >= MAX_SOLUTIONS:
#             return
#         # can the remaining fields still fill every type up to its lower bound?
#         need = sum(max(0.0, lo[j] - sums[j]) for j in range(k))
#         if need > remaining[i] + 1e-6:
#             return
#         if i == n:
#             sol = np.empty(n, dtype=int)
#             sol[order] = labels
#             solutions.append(sol)
#             return
#         for j in range(k):
#             if sums[j] + a[i] <= hi[j] + 1e-6:
#                 sums[j] += a[i]
#                 labels[i] = j
#                 rec(i + 1)
#                 sums[j] -= a[i]
#         if allow_default:
#             labels[i] = k
#             rec(i + 1)
#         labels[i] = -1
 
#     try:
#         rec(0)
#         aborted = False
#     except _Abort:
#         aborted = True
#     return solutions, aborted
 
 
# def solution_error(sol, areas, targets):
#     """Sum of relative area errors over all types."""
#     return sum(
#         abs(areas[sol == j].sum() - t) / t for j, t in enumerate(targets)
#     )
 
 
# # ---------------------------------------------------------------------------
# # Main
# # ---------------------------------------------------------------------------
# def add_pflug():
#     # --- CSV ---
#     csv = pd.read_csv(CSV_IN, sep=None, engine="python", encoding="latin-1")
#     csv = csv[csv.JAHR==YEAR]
#     for col in (CSV_FARM_COL, CSV_CROP_COL, CSV_AREA_COL, CSV_TILLAGE_COL):
#         if col not in csv.columns:
#             raise SystemExit(f"Column '{col}' not in CSV. Columns: {list(csv.columns)}")
 
#     csv["_farm"] = norm_id(csv[CSV_FARM_COL])
#     csv["_crop"] = norm_id(csv[CSV_CROP_COL])
#     csv["_area"] = csv[CSV_AREA_COL].astype(float)
#     csv["_type"] = csv[CSV_TILLAGE_COL].astype(str).str.strip()
 
#     # --- fields ---
#     pflug_codes = set(load_arable_codes())
#     if RESTRICT_PFLUG_TO_CSV_CROPS:
#         csv_codes = set(pd.to_numeric(csv[CSV_CROP_COL], errors="coerce").dropna().astype(int))
#         print(f"{len(csv_codes)} crop codes in the CSV, "
#               f"{len(pflug_codes & csv_codes)} of them arable")
#         not_arable = sorted(csv_codes - pflug_codes)
#         if not_arable:
#             print(f"  CSV crop codes not classed as arable (ignored for Pflug): {not_arable}")
#         pflug_codes &= csv_codes
#     if not pflug_codes:
#         raise SystemExit("No crop codes left for the Pflug reference.")
#     print(f"Pflug reference sampled from {len(pflug_codes)} crop codes")
#     gdf = load_fields(set(csv["_farm"].unique()), sorted(pflug_codes))
#     gdf["_farm"] = norm_id(gdf[GDF_FARM_COL])
#     gdf["_crop"] = norm_id(gdf[GDF_CROP_COL])
#     gdf["_area"] = pd.to_numeric(gdf[GDF_AREA_COL], errors="coerce") * GDF_AREA_TO_M2
 
#     bad = gdf["_area"].isna() | (gdf["_area"] <= 0)
#     if bad.any():
#         print(f"{bad.sum()} field(s) without valid '{GDF_AREA_COL}' -> area from geometry")
#         if gdf.crs is None or not gdf.crs.is_projected:
#             raise SystemExit(f"CRS {gdf.crs} is not projected; cannot compute areas in m2.")
#         gdf.loc[bad, "_area"] = gdf.loc[bad].geometry.area
#     still_bad = gdf["_area"].isna()
#     if still_bad.any():
#         print(f"{still_bad.sum()} field(s) without area or geometry -> dropped")
#         gdf = gdf[~still_bad].copy()
#     gdf[GDF_AREA_COL] = gdf["_area"] / GDF_AREA_TO_M2
 
#     # --- defaults ---
#     csv_farm_set = set(csv["_farm"])
#     gdf["tillage"] = DEFAULT_TILLAGE
#     gdf["tillage_status"] = np.where(gdf["_farm"].isin(csv_farm_set),
#                                      "csv_farm_other_crop", "default")
#     gdf["tillage_certain"] = True
 
#     # CSV per farm and crop: area per tillage type
#     csv_groups = (
#         csv.groupby(["_farm", "_crop", "_type"], sort=False)["_area"].sum()
#         .reset_index()
#         .groupby(["_farm", "_crop"], sort=False)
#     )
#     fields_by = gdf.groupby(["_farm", "_crop"]).groups
 
#     log = []
#     for (farm, crop), rows in csv_groups:
#         types = rows["_type"].tolist()
#         targets = rows["_area"].tolist()
#         csv_total = float(sum(targets))
#         entry = dict(
#             betr_ID=farm, crop=crop,
#             csv_types="; ".join(f"{t}: {a:.0f}" for t, a in zip(types, targets)),
#             n_types=len(types), area_csv_m2=round(csv_total), n_fields=0,
#             area_fields_m2=0, diff_m2=np.nan, diff_pct=np.nan, status="",
#             n_combinations=0, assignment="", message="",
#         )
 
#         idx = fields_by.get((farm, crop))
#         if idx is None or len(idx) == 0:
#             entry.update(status="no_fields",
#                          message="no fields of this farm and crop in the gdf")
#             log.append(entry)
#             continue
 
#         areas = gdf.loc[idx, "_area"].to_numpy()
#         field_total = float(areas.sum())
#         entry.update(
#             n_fields=len(idx), area_fields_m2=round(field_total),
#             diff_m2=round(field_total - csv_total),
#             diff_pct=round(100 * (field_total - csv_total) / csv_total, 1) if csv_total else np.nan,
#         )
 
#         # fields sum to less than the CSV area
#         if field_total < csv_total * (1 - TOL):
#             if len(types) == 1:
#                 gdf.loc[idx, "tillage"] = types[0]
#                 gdf.loc[idx, "tillage_certain"] = True
#                 msg = f"all {len(idx)} fields -> {types[0]} (fields smaller than CSV area)"
#                 entry["assignment"] = f"{types[0]}: {len(idx)} fields"
#             else:
#                 gdf.loc[idx, "tillage"] = UNCLEAR
#                 gdf.loc[idx, "tillage_certain"] = False
#                 msg = "several types but fields smaller than CSV area -> unklar"
#             gdf.loc[idx, "tillage_status"] = "fields_smaller_csv"
#             entry.update(status="fields_smaller_csv", message=msg)
#             log.append(entry)
#             continue
 
#         # search field combinations
#         allow_default = field_total > csv_total * (1 + TOL)
#         sols, aborted = find_assignments(areas, targets, TOL, allow_default)
#         entry["n_combinations"] = len(sols)
 
#         if not sols:
#             status = "search_aborted" if aborted else "no_match"
#             gdf.loc[idx, "tillage"] = UNCLEAR
#             gdf.loc[idx, "tillage_status"] = status
#             gdf.loc[idx, "tillage_certain"] = False
#             entry.update(status=status,
#                          message=("too many fields for a full search" if aborted
#                                   else f"no field combination within +/-{100 * TOL:.0f} %"))
#             log.append(entry)
#             continue
 
#         errors = [solution_error(s, areas, targets) for s in sols]
#         best = sols[int(np.argmin(errors))]
#         all_labels = np.vstack(sols)
#         certain = (all_labels == all_labels[0]).all(axis=0)
 
#         names = types + [DEFAULT_TILLAGE]
#         gdf.loc[idx, "tillage"] = [names[l] for l in best]
#         gdf.loc[idx, "tillage_certain"] = certain
 
#         if aborted:
#             status = "search_aborted"
#         elif len(sols) > 1:
#             status = "ambiguous"
#         elif len(types) > 1:
#             status = "split_match"
#         elif allow_default:
#             status = "subset_match"
#         else:
#             status = "all_same"
#         gdf.loc[idx, "tillage_status"] = status
#         entry["status"] = status
 
#         entry["assignment"] = "; ".join(
#             f"{name}: {(best == j).sum()} fields / {areas[best == j].sum():.0f} m2"
#             for j, name in enumerate(names) if (best == j).any()
#         )
#         if status == "all_same":
#             entry["message"] = f"all fields same type ({types[0]})"
#         elif status in ("subset_match", "split_match"):
#             entry["message"] = "unique combination"
#         elif status == "ambiguous":
#             more = "+" if len(sols) >= MAX_SOLUTIONS else ""
#             entry["message"] = (f"{len(sols)}{more} combinations match; best used, "
#                                 f"{(~certain).sum()} of {len(idx)} fields uncertain")
#         else:
#             entry["message"] = (f"search stopped after {MAX_NODES} steps; "
#                                 f"best of {len(sols)} combinations used")
#         log.append(entry)
 
#     # --- outputs ---
#     log = pd.DataFrame(log)
#     log.to_csv(LOG_OUT, index=False)
#     if os.path.exists(GPKG_OUT):
#         os.remove(GPKG_OUT)            # overwriting would keep the old file size
#     gdf.drop(columns=["_farm", "_crop", "_area"]).to_file(GPKG_OUT, driver="GPKG")
 
#     print("\nFarm-crop pairs in CSV per status:")
#     print(log["status"].value_counts().to_string())
#     print("\nFields per tillage and status:")
#     print(pd.crosstab(gdf["tillage"], gdf["tillage_status"], margins=True))
#     print(f"\nUncertain fields: {(~gdf['tillage_certain']).sum()}")
#     print(f"\nWritten: {GPKG_OUT}, {LOG_OUT}")


"""
Extract Sentinel-1 and Sentinel-2 field statistics for the time window
August (YEAR-1) to June (YEAR) from the local yearly zarr tiles.

Sentinel-1 (S1_DIR / find_tiles, zarr groups asc / desc)
  VV, VH and cross ratio CR = VH/VV (dB) per field and acquisition, separately
  for ascending and descending.
  - pixels: centre inside the field shrunk by INNER_BUFFER_M, s1_valid > 0
  - high outliers removed in dB: absolute cap or > median + K_MAD * MAD
    (VV or VH -> pixel dropped for both)
  - CR from the means of the remaining pixels in linear units

Sentinel-2 (S2_DIR, same 1280 m tiles and pixel grid as S1)
  Mean of every available band B01 ... B12 (incl. B8A) per field and date.
  - pixels: same inner buffer; cloud mask from SCL (only S2_CLEAR_SCL classes)
  - outliers: a pixel is dropped if any band lies more than S2_K_MAD * MAD
    from the field median of that band (both directions: residual clouds,
    haze, shadows)
  - NaN if fewer than MIN_PIXELS or less than S2_MIN_CLEAR_FRAC of the field
    pixels remain

Speed: every tile is read once (no dask), tiles run in parallel processes,
statistics are vectorised over all dates, results are cached per tile so an
interrupted run resumes. Delete CACHE_DIR whenever settings or fields change.

Outputs (OUT_DIR)
  - wide parquet (CSV if pyarrow is missing): tillage, field_id, betriebs_id,
              asc_YYYYMMDD ..., desc_YYYYMMDD ... (S1 CR),
              B01_YYYYMMDD ..., B12_YYYYMMDD ... (S2 band means)
  - long parquet (CSV if pyarrow is missing): one row per field and
    acquisition, column 'sensor' = S1 / S2, S1 and S2 values in their own columns
  - dates table: sensor, direction, date, relative orbit, platform
  - shapefile of all selected fields
"""

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

# time window: August of the previous year to June
# YEAR = 2021
# T_START = pd.Timestamp(f"{YEAR - 1}-08-01")
# T_END = pd.Timestamp(f"{YEAR}-06-30 23:59:59")

# GPKG_IN = f"data/all/lnf_{YEAR}_tillage.gpkg"
# OUT_DIR = "data/all"
# CACHE_DIR = os.path.join(OUT_DIR, "tile_cache")      # delete to force a full rerun
# OUT_WIDE = os.path.join(OUT_DIR, f"s1_s2_cr_wide_{YEAR}.csv")
# OUT_LONG = os.path.join(OUT_DIR, f"s1_s2_cr_long_{YEAR}.parquet")
# OUT_DATES = os.path.join(OUT_DIR, f"s1_s2_cr_dates_{YEAR}.csv")
# # OUT_SHP = os.path.join(OUT_DIR, f"sampled_fields_{YEAR}.shp")

# FARM_COL = "betriebsnummer"
# AREA_COL = "groesse"
# FIELD_ID_COL = "id"              # None -> row index of the gpkg

# # field selection
# TILLAGE_PATTERNS = ["streifen", "mulch", "direkt"]
# GOOD_STATUS = ["all_same", "split_match", "subset_match"]
# USE_CERTAIN_AMBIGUOUS = False
# MIN_AREA_M2 = 5000
# N_PFLUG = 50000                   # None = all Pflug reference fields
# RANDOM_SEED = 42

# # tiles
# S1_DIR = '/mnt/eo-nas1/data/satellite/sentinel1/raw/CH'  # folder with all S1_*.zarr (fastest); None -> find_tiles
# S2_DIR = '/mnt/eo-nas1/data/satellite/sentinel2/raw/CH'  # <- folder with the S2 zarr tiles
# S2_PREFIX = "S2"                 # file names: <S2_PREFIX>_<x_left>_<y_top>_<start>_<end>.zarr
# S2_GROUP = None                  # zarr group of the S2 files, if any
# CRS = "EPSG:32632"
# TILE_M = 1280

# # S1
# DIRECTIONS = ["asc", "desc"]
# VV_VAR, VH_VAR, VALID_VAR = "s1_vv", "s1_vh", "s1_valid"
# K_MAD = 2.0
# MAX_VV_DB = 0.0
# MAX_VH_DB = -5.0
# MIN_VALID_FRAC = 0.5

# # S2
# S2_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A",
#             "B09", "B11", "B12"]          # missing bands (e.g. B10 in L2A) are skipped
# S2_CLEAR_SCL = [4, 5, 6]         # 4 vegetation, 5 bare soil, 6 water
# S2_K_MAD = 3.0
# S2_MIN_CLEAR_FRAC = 0.8
# # reflectance = (DN + offset) * S2_SCALE; offset S2_OFFSET only for dates >= S2_OFFSET_FROM
# # (processing baseline 04.00, 25 Jan 2022). Set S2_OFFSET_FROM = None to apply it to all
# # dates (fully reprocessed archive) or S2_OFFSET = 0 for already harmonized data.
# S2_SCALE = 0.0001
# S2_OFFSET = 0
# S2_OFFSET_FROM = None

# # both sensors
# INNER_BUFFER_M = 10
# MIN_PIXELS = 10
# N_WORKERS = 8


# ---------------------------------------------------------------------------
# 1. Field selection
# ---------------------------------------------------------------------------
def select_fields():
    gdf = gpd.read_file(GPKG_IN)
    gdf["field_id"] = gdf.index if FIELD_ID_COL is None else gdf[FIELD_ID_COL]
    if not gdf["field_id"].is_unique:
        raise SystemExit(f"field ids in '{FIELD_ID_COL}' are not unique")
    big = gdf[AREA_COL] >= MIN_AREA_M2

    good = gdf["tillage_status"].isin(GOOD_STATUS)
    if USE_CERTAIN_AMBIGUOUS:
        good |= (gdf["tillage_status"] == "ambiguous") & gdf["tillage_certain"].astype(bool)

    pattern = "|".join(TILLAGE_PATTERNS)
    alt = gdf[big & good & gdf["tillage"].str.contains(pattern, case=False, na=False)]

    pool = gdf[big & (gdf["tillage_status"] == "default") & (gdf["tillage"] == "Pflug")]
    n = len(pool) if N_PFLUG is None else min(N_PFLUG, len(pool))
    pflug = pool.sample(n=n, random_state=RANDOM_SEED) if n < len(pool) else pool

    sel = pd.concat([alt, pflug])
    print("Selected fields:")
    print(sel["tillage"].value_counts().to_string())
    print(f"({len(pool)} Pflug fields available, {n} used)")
    return sel


# ---------------------------------------------------------------------------
# 2. Tile index
# ---------------------------------------------------------------------------
def parse_tile(f, prefix):
    m = re.search(rf"{prefix}_(\d+)_(\d+)_(\d{{8}})_(\d{{8}})\.zarr",
                  os.path.basename(str(f).rstrip("/")))
    if m is None:
        return None
    return int(m[1]), int(m[2]), pd.Timestamp(m[3]), pd.Timestamp(m[4])


def in_window(info):
    return info[2] <= T_END and info[3] >= T_START


def index_dir(folder, prefix):
    index = defaultdict(list)
    for f in glob.glob(os.path.join(folder, f"{prefix}_*.zarr")):
        info = parse_tile(f, prefix)
        if info and in_window(info):
            index[(info[0], info[1])].append(f)
    return {k: sorted(v) for k, v in index.items()}


def build_tile_index(geoms):
    """geoms: dict field_id -> shrunk polygon in CRS."""
    s2_index = index_dir(S2_DIR, S2_PREFIX)
    print(f"S2 tiles in window: {len(s2_index)}")

    s1_index = index_dir(S1_DIR, "S1")
    s1_lookup = lambda key: s1_index.get(key, [])
    anchor = next(iter(s1_index or s2_index))

    ax0, ay0 = anchor
    field_keys = {}
    for fid, g in geoms.items():
        xmin, ymin, xmax, ymax = g.bounds
        kx = np.arange(np.floor((xmin - ax0) / TILE_M), np.floor((xmax - ax0) / TILE_M) + 1)
        ky = np.arange(np.ceil((ymin - ay0) / TILE_M), np.ceil((ymax - ay0) / TILE_M) + 1)
        field_keys[fid] = [(int(ax0 + i * TILE_M), int(ay0 + j * TILE_M)) for i in kx for j in ky]

    needed = sorted({k for ks in field_keys.values() for k in ks})
    tiles = {}
    for i, key in enumerate(needed, 1):
        s1, s2 = s1_lookup(key), s2_index.get(key, [])
        if s1 or s2:
            tiles[key] = (s1, s2)
        if i % 1000 == 0:
            print(f"  tile lookup {i}/{len(needed)}")
    n_s1 = sum(bool(v[0]) for v in tiles.values())
    n_s2 = sum(bool(v[1]) for v in tiles.values())
    print(f"Tiles needed: {len(needed)} | with S1: {n_s1} | with S2: {n_s2}")

    tile_fields, field_ntiles = defaultdict(list), {}
    for fid, keys in field_keys.items():
        keys = [k for k in keys if k in tiles]
        field_ntiles[fid] = len(keys)
        for k in keys:
            tile_fields[k].append(fid)
    return tiles, tile_fields, field_ntiles


# ---------------------------------------------------------------------------
# 3. Statistics (vectorised over acquisitions)
# ---------------------------------------------------------------------------
def s1_stats(V, H, ok, n_inside):
    """V, H, ok: (n_acq, n_pix)."""
    ok = ok & np.isfinite(V) & np.isfinite(H) & (V > 0) & (H > 0)
    with np.errstate(divide="ignore", invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        Vdb = np.where(ok, 10 * np.log10(V), np.nan)
        Hdb = np.where(ok, 10 * np.log10(H), np.nan)

        def keep_high(Xdb, cap):
            k = ok & (Xdb <= cap)
            Xk = np.where(k, Xdb, np.nan)
            med = np.nanmedian(Xk, axis=1, keepdims=True)
            mad = 1.4826 * np.nanmedian(np.abs(Xk - med), axis=1, keepdims=True)
            use = (mad > 0) & (k.sum(axis=1, keepdims=True) >= 3)
            return k & (Xdb <= np.where(use, med + K_MAD * mad, np.inf))

        keep = keep_high(Vdb, MAX_VV_DB) & keep_high(Hdb, MAX_VH_DB)
        n_used = keep.sum(axis=1)
        m_vv = np.where(keep, V, 0).sum(axis=1) / n_used
        m_vh = np.where(keep, H, 0).sum(axis=1) / n_used
        good = (n_used >= MIN_PIXELS) & (n_used >= MIN_VALID_FRAC * n_inside)
        vv_db = np.where(good, 10 * np.log10(m_vv), np.nan)
        vh_db = np.where(good, 10 * np.log10(m_vh), np.nan)
    return dict(n_inside=np.full(len(n_used), n_inside), n_valid=ok.sum(axis=1),
                n_used=n_used, vv_db=vv_db, vh_db=vh_db, cr_db=vh_db - vv_db)


def s2_stats(B, ok, n_inside, bands):
    """B: (n_band, n_date, n_pix) float, ok: (n_date, n_pix) clear-sky mask."""
    ok = ok & np.all(np.isfinite(B) & (B > 0), axis=0)
    with np.errstate(divide="ignore", invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        Bm = np.where(ok[None], B, np.nan)
        med = np.nanmedian(Bm, axis=2, keepdims=True)
        mad = 1.4826 * np.nanmedian(np.abs(Bm - med), axis=2, keepdims=True)
        use = (mad > 0) & (ok.sum(axis=1)[None, :, None] >= 3)
        outlier = use & (np.abs(B - med) > S2_K_MAD * mad)
        keep = ok & ~outlier.any(axis=0)
        n_used = keep.sum(axis=1)
        means = np.where(keep[None], B, 0).sum(axis=2) / n_used
        good = (n_used >= MIN_PIXELS) & (n_used >= S2_MIN_CLEAR_FRAC * n_inside)
        means = np.where(good[None], means, np.nan)        # still DN; scaled in main
    out = dict(n_inside=np.full(len(n_used), n_inside), n_clear=ok.sum(axis=1), n_used=n_used)
    out.update({b: means[i] for i, b in enumerate(bands)})
    return out


# ---------------------------------------------------------------------------
# 4. Reading tiles
# ---------------------------------------------------------------------------
def cache_path(key):
    return os.path.join(CACHE_DIR, f"{key[0]}_{key[1]}.pkl")


def _open(path, group):
    try:
        return xr.open_zarr(path, group=group, chunks=None) if group else \
            xr.open_zarr(path, chunks=None)
    except (FileNotFoundError, KeyError, ValueError, OSError):
        return None


def _time_sel(z):
    t = pd.DatetimeIndex(pd.to_datetime(z.time.values))
    sel = np.flatnonzero((t >= T_START) & (t <= T_END))
    return t, sel


def _check_grid(grid, z, path):
    lon, lat = z.lon.values, z.lat.values
    if grid is not None and not (np.array_equal(grid[0], lon) and np.array_equal(grid[1], lat)):
        raise RuntimeError(f"{path}: pixel grid differs from the other file of this tile")
    return lon, lat


def load_s1(paths, direction):
    parts, grid = [], None
    for p in paths:
        z = _open(p, direction)
        if z is None:
            continue
        miss = [v for v in (VV_VAR, VH_VAR, VALID_VAR) if v not in z.variables]
        if miss:
            raise RuntimeError(f"{p} [{direction}]: missing {miss}, has {list(z.data_vars)}")
        t, sel = _time_sel(z)
        if len(sel) == 0:
            continue
        z = z.isel(time=sel)
        grid = _check_grid(grid, z, p)
        n = len(sel)
        parts.append(dict(
            vv=z[VV_VAR].transpose("time", "lat", "lon").values.astype("float32"),
            vh=z[VH_VAR].transpose("time", "lat", "lon").values.astype("float32"),
            ok=z[VALID_VAR].transpose("time", "lat", "lon").values > 0,
            date=t[sel].floor("D").strftime("%Y%m%d").to_numpy(),
            orbit=(z["relative_orbit"].values.astype(int) if "relative_orbit" in z.variables
                   else np.full(n, -1)),
            platform=(z["platform"].values.astype(str) if "platform" in z.variables
                      else np.full(n, "")),
        ))
    if not parts:
        return None
    a = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    a["lon"], a["lat"] = grid

    keys = list(zip(a["date"], a["orbit"]))          # same day + orbit -> merge frames
    if len(set(keys)) < len(keys):
        groups = defaultdict(list)
        for i, k in enumerate(keys):
            groups[k].append(i)
        g = list(groups.values())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            vv = np.where(a["ok"], a["vv"], np.nan)
            vh = np.where(a["ok"], a["vh"], np.nan)
            a["vv"] = np.stack([np.nanmean(vv[i], axis=0) for i in g])
            a["vh"] = np.stack([np.nanmean(vh[i], axis=0) for i in g])
        a["ok"] = np.stack([a["ok"][i].any(axis=0) for i in g])
        for k in ("date", "orbit", "platform"):
            a[k] = np.array([a[k][i[0]] for i in g])
    return a


def _find_var(z, name):
    for c in (name, f"s2_{name}", name.lower(), f"s2_{name.lower()}"):
        if c in z.variables:
            return c
    return None


def load_s2(paths):
    parts, grid, bands = [], None, None
    for p in paths:
        z = _open(p, S2_GROUP)
        if z is None:
            continue
        scl = _find_var(z, "SCL")
        if scl is None:
            raise RuntimeError(f"{p}: no SCL band, has {list(z.data_vars)}")
        avail = [(b, v) for b in S2_BANDS if (v := _find_var(z, b))]
        if bands is None:
            bands = [b for b, _ in avail]
        elif bands != [b for b, _ in avail]:
            raise RuntimeError(f"{p}: band set differs from the other file of this tile")
        t, sel = _time_sel(z)
        if len(sel) == 0:
            continue
        z = z.isel(time=sel)
        grid = _check_grid(grid, z, p)
        parts.append(dict(
            B=np.stack([z[v].transpose("time", "lat", "lon").values for _, v in avail]),
            ok=np.isin(z[scl].transpose("time", "lat", "lon").values, S2_CLEAR_SCL),
            date=t[sel].floor("D").strftime("%Y%m%d").to_numpy(),
        ))
    if not parts:
        return None
    a = dict(B=np.concatenate([p["B"] for p in parts], axis=1),
             ok=np.concatenate([p["ok"] for p in parts]),
             date=np.concatenate([p["date"] for p in parts]),
             bands=bands, lon=grid[0], lat=grid[1])

    if len(set(a["date"])) < len(a["date"]):        # same day twice -> merge, clear pixels only
        groups = defaultdict(list)
        for i, d in enumerate(a["date"]):
            groups[d].append(i)
        g = list(groups.values())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            Bf = np.where(a["ok"][None], a["B"].astype("float32"), np.nan)
            a["B"] = np.stack([np.nanmean(Bf[:, i], axis=1) for i in g], axis=1)
        a["ok"] = np.stack([a["ok"][i].any(axis=0) for i in g])
        a["date"] = np.array([a["date"][i[0]] for i in g])
    return a


def field_pixels(lon, lat, geoms):
    """field_id -> flat indices of pixel centres inside the (shrunk) polygon."""
    X, Y = np.meshgrid(lon, lat)
    out = {}
    for fid, g, _ in geoms:
        xmin, ymin, xmax, ymax = g.bounds
        win = (X >= xmin) & (X <= xmax) & (Y >= ymin) & (Y <= ymax)
        if not win.any():
            continue
        inside = np.zeros(X.shape, dtype=bool)
        inside[win] = shapely.contains_xy(g, X[win], Y[win])
        idx = np.flatnonzero(inside)
        if len(idx):
            out[fid] = idx
    return out


# ---------------------------------------------------------------------------
# 5. Tile worker
# ---------------------------------------------------------------------------
def process_tile(key, s1_paths, s2_paths, fields):
    """fields: list of (field_id, polygon WKB, number of tiles the field touches)."""
    out = cache_path(key)
    if os.path.exists(out):
        return key
    geoms = [(fid, shapely.from_wkb(w), nt) for fid, w, nt in fields]
    ntiles = {fid: nt for fid, _, nt in geoms}
    s1_rec, s1_part, s2_rec, s2_part, bands = [], [], [], [], None

    # --- S1 ---
    for d in DIRECTIONS:
        a = load_s1(s1_paths, d) if s1_paths else None
        if a is None:
            continue
        nt = len(a["date"])
        V, H, OK = (a[k].reshape(nt, -1) for k in ("vv", "vh", "ok"))
        meta = pd.DataFrame({"direction": d, "date": a["date"],
                             "relative_orbit": a["orbit"], "platform": a["platform"]})
        for fid, idx in field_pixels(a["lon"], a["lat"], geoms).items():
            if ntiles[fid] == 1:
                df = meta.copy()
                df.insert(0, "field_id", fid)
                for k, v in s1_stats(V[:, idx], H[:, idx], OK[:, idx], len(idx)).items():
                    df[k] = v
                s1_rec.append(df)
            else:
                s1_part.append((fid, d, a["date"], a["orbit"], a["platform"],
                                V[:, idx], H[:, idx], OK[:, idx]))
        del a, V, H, OK

    # --- S2 ---
    a = load_s2(s2_paths) if s2_paths else None
    if a is not None:
        bands = a["bands"]
        nb, nt = a["B"].shape[:2]
        B = a["B"].reshape(nb, nt, -1)
        OK = a["ok"].reshape(nt, -1)
        for fid, idx in field_pixels(a["lon"], a["lat"], geoms).items():
            Bf = B[:, :, idx].astype("float32")
            if ntiles[fid] == 1:
                df = pd.DataFrame({"field_id": fid, "date": a["date"]})
                for k, v in s2_stats(Bf, OK[:, idx], len(idx), bands).items():
                    df[k] = v
                s2_rec.append(df)
            else:
                s2_part.append((fid, a["date"], Bf, OK[:, idx]))

    res = dict(
        s1=pd.concat(s1_rec, ignore_index=True) if s1_rec else None, s1_partial=s1_part,
        s2=pd.concat(s2_rec, ignore_index=True) if s2_rec else None, s2_partial=s2_part,
        bands=bands,
    )
    tmp = out + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, out)
    return key


# ---------------------------------------------------------------------------
# 6. Combine results
# ---------------------------------------------------------------------------
def collect(tile_keys):
    s1, s2, bands = [], [], None
    p1 = defaultdict(lambda: ([], [], []))
    plat = {}
    p2 = defaultdict(lambda: ([], []))
    for k in tile_keys:
        p = cache_path(k)
        if not os.path.exists(p):
            continue
        with open(p, "rb") as f:
            r = pickle.load(f)
        if r["s1"] is not None:
            s1.append(r["s1"])
        if r["s2"] is not None:
            s2.append(r["s2"])
        bands = bands or r["bands"]
        for fid, d, dates, orbits, plats, V, H, OK in r["s1_partial"]:
            for i in range(len(dates)):
                key = (fid, d, dates[i], int(orbits[i]))
                e = p1[key]
                e[0].append(V[i]); e[1].append(H[i]); e[2].append(OK[i])
                plat[key] = plats[i]
        for fid, dates, B, OK in r["s2_partial"]:
            for i in range(len(dates)):
                e = p2[(fid, dates[i])]
                e[0].append(B[:, i, :]); e[1].append(OK[i])

    rows = []
    for (fid, d, date, orb), (vs, hs, oks) in p1.items():
        v = np.concatenate(vs)[None]
        st = s1_stats(v, np.concatenate(hs)[None], np.concatenate(oks)[None], v.shape[1])
        rows.append(dict(field_id=fid, direction=d, date=date, relative_orbit=orb,
                         platform=plat[(fid, d, date, orb)], **{k: x[0] for k, x in st.items()}))
    if rows:
        s1.append(pd.DataFrame(rows))

    rows = []
    for (fid, date), (bs, oks) in p2.items():
        B = np.concatenate(bs, axis=1)[:, None, :]
        st = s2_stats(B, np.concatenate(oks)[None], B.shape[2], bands)
        rows.append(dict(field_id=fid, date=date, **{k: x[0] for k, x in st.items()}))
    if rows:
        s2.append(pd.DataFrame(rows))

    s1 = pd.concat(s1, ignore_index=True) if s1 else pd.DataFrame()
    s2 = pd.concat(s2, ignore_index=True) if s2 else pd.DataFrame()
    n_cross = len({k[0] for k in p1} | {k[0] for k in p2})
    return s1, s2, bands or [], n_cross


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def extract_s1s2():
    os.makedirs(CACHE_DIR, exist_ok=True)
    t0 = time.time()
    print(f"Time window: {T_START.date()} to {T_END.date()}")

    sel = select_fields()
    utm = sel.to_crs(CRS)
    geoms = {fid: g.buffer(-INNER_BUFFER_M) for fid, g in zip(utm["field_id"], utm.geometry)}
    empty = [fid for fid, g in geoms.items() if g.is_empty]
    for fid in empty:
        del geoms[fid]
    if empty:
        print(f"{len(empty)} field(s) vanish after the {INNER_BUFFER_M} m inner buffer")

    tiles, tile_fields, field_ntiles = build_tile_index(geoms)
    wkb = {fid: shapely.to_wkb(g) for fid, g in geoms.items()}

    todo = [k for k in tile_fields if not os.path.exists(cache_path(k))]
    print(f"{len(tile_fields)} tiles with fields, {len(tile_fields) - len(todo)} cached, "
          f"{len(todo)} to process")

    failed = []
    if todo:
        t1 = time.time()
        with ProcessPoolExecutor(max_workers=N_WORKERS, mp_context=mp.get_context("fork")) as ex:
            futs = {
                ex.submit(process_tile, k, tiles[k][0], tiles[k][1],
                          [(fid, wkb[fid], field_ntiles[fid]) for fid in tile_fields[k]]): k
                for k in todo
            }
            for i, fut in enumerate(as_completed(futs), 1):
                try:
                    fut.result()
                except Exception as e:
                    failed.append(futs[fut])
                    print(f"  tile {futs[fut]} failed: {e}")
                if i % 50 == 0 or i == len(futs):
                    el = time.time() - t1
                    print(f"  {i}/{len(futs)} tiles, {el / 60:.1f} min elapsed, "
                          f"~{el / i * (len(futs) - i) / 60:.1f} min left")
    if failed:
        print(f"{len(failed)} tile(s) failed; rerun the script to retry them.")

    s1, s2, bands, n_cross = collect(tile_fields)
    if s1.empty and s2.empty:
        raise SystemExit("No data extracted.")
    print(f"\nS1: {len(s1):,} field-acquisitions | S2: {len(s2):,} field-dates | "
          f"{n_cross:,} fields span several tiles | S2 bands: {bands}")

    meta = sel[["tillage", "field_id", FARM_COL]].rename(columns={FARM_COL: "betriebs_id"})
    wide_parts, date_rows = [], []

    if not s1.empty:
        s1["column"] = s1["direction"] + "_" + s1["date"].astype(str)
        dup = s1.groupby("column")["relative_orbit"].transform("nunique") > 1
        s1.loc[dup, "column"] += "_o" + s1.loc[dup, "relative_orbit"].astype(str)
        w = s1.pivot(index="field_id", columns="column", values="cr_db")
        wide_parts.append(w[sorted(w.columns, key=lambda c: (DIRECTIONS.index(c.split("_")[0]), c))])
        d = (s1.groupby("column")
             .agg(direction=("direction", "first"), date=("date", "first"),
                  relative_orbit=("relative_orbit", "first"),
                  platform=("platform", lambda p: ",".join(sorted(set(map(str, p))))),
                  n_fields_valid=("cr_db", lambda x: int(x.notna().sum())))
             .reset_index())
        d.insert(0, "sensor", "S1")
        date_rows.append(d)
        s1.insert(0, "sensor", "S1")

    if not s2.empty:
        # DN -> reflectance with date-dependent offset
        dates_s2 = pd.to_datetime(s2["date"].astype(str), format="%Y%m%d")
        offset = np.where(
            dates_s2 >= pd.Timestamp(S2_OFFSET_FROM) if S2_OFFSET_FROM else True, S2_OFFSET, 0)
        for b in bands:
            s2[b] = (s2[b] + offset) * S2_SCALE
        w = s2.pivot(index="field_id", columns="date", values=bands)
        w.columns = [f"{b}_{dt}" for b, dt in w.columns]
        wide_parts.append(w[sorted(w.columns, key=lambda c: (bands.index(c.split("_")[0]), c))])
        ref = "B04" if "B04" in bands else bands[0]
        d = (s2.groupby("date").agg(n_fields_valid=(ref, lambda x: int(x.notna().sum())))
             .reset_index())
        d.insert(0, "sensor", "S2")
        d["column"] = "<band>_" + d["date"].astype(str)
        date_rows.append(d)
        s2.insert(0, "sensor", "S2")
        s2["column"] = "<band>_" + s2["date"].astype(str)

    long = pd.concat([x for x in (s1, s2) if not x.empty], ignore_index=True)
    long = long.merge(meta, on="field_id", how="left")
    try:
        long.to_parquet(OUT_LONG, index=False)
        long_out = OUT_LONG
    except ImportError:
        long_out = os.path.splitext(OUT_LONG)[0] + ".csv"
        long.to_csv(long_out, index=False)

    wide = meta.set_index("field_id").join(pd.concat(wide_parts, axis=1)).reset_index()
    wide = wide[["tillage", "field_id", "betriebs_id"]
                + [c for c in wide.columns if c not in ("tillage", "field_id", "betriebs_id")]]
    try:
        wide.to_parquet(OUT_WIDE, index=False)
        wide_out = OUT_WIDE
    except ImportError:
        wide_out = os.path.splitext(OUT_WIDE)[0] + ".csv"
        wide.to_csv(wide_out, index=False)
    pd.concat(date_rows, ignore_index=True).to_csv(OUT_DATES, index=False)

    shp = sel[["tillage", "field_id", FARM_COL, AREA_COL, "tillage_status", "geometry"]].rename(
        columns={FARM_COL: "betr_id", AREA_COL: "groesse", "tillage_status": "status"})
    shp.to_file(OUT_SHP, encoding="utf-8")

    if not s1.empty:
        print(f"S1: {s1['cr_db'].notna().sum():,} valid CR values, "
              f"{100 * (1 - s1['n_used'].sum() / max(s1['n_valid'].sum(), 1)):.1f} % pixels "
              f"removed as high outliers")
    if not s2.empty:
        ref = "B04" if "B04" in bands else bands[0]
        print(f"S2: {s2[ref].notna().sum():,} of {len(s2):,} field-dates usable "
              f"(clear sky + outlier filter); {s2['date'].nunique()} dates")
    print(f"Wide table: {wide.shape[0]:,} fields x {wide.shape[1] - 3:,} value columns")
    print(f"\nWritten: {wide_out}\n         {long_out}\n         {OUT_DATES}\n         {OUT_SHP}")
    print(f"Total time: {(time.time() - t0) / 60:.1f} min")

def boxplot_s1_per_till():    
    LONG_CSV = "data/s1_cr_may2022_long.csv"
    OUT_PNG = "data/boxplot_asc88_may.png"
    
    DIRECTION = "asc"
    ORBIT = 88
    MIN_USED = 10
    TARGET_DAYS = {"mid May": 15, "end of May": 31}   # date of the orbit closest to these days
    
    # tillage groups: label -> substring to match in the tillage column
    GROUPS = {"Pflug": "pflug", "Mulchsaat": "mulch", "Streifensaat": "streifen"}
    ABBR = {"Pflug": "P", "Mulchsaat": "M", "Streifensaat": "S"}
    VARS = {"vv_db": "VV", "vh_db": "VH", "cr_db": "CR (VH/VV)"}
    PALETTE = {"Pflug": "0.65", "Mulchsaat": "tab:orange", "Streifensaat": "tab:green"}
    
    
    
    def fmt_p(p):
        if pd.isna(p):
            return "n/a"
        return "<0.001" if p < 0.001 else f"{p:.3f}"
    
    
    def ttest_label(d, var_label):
        """x tick label with Welch t-test p-values for all group pairs."""
        lines = [var_label, "", "p-value t-test"]
        for a, b in combinations(order, 2):
            xa = d.loc[(d["group"] == a) & (d["variable"] == var_label), "dB"].dropna()
            xb = d.loc[(d["group"] == b) & (d["variable"] == var_label), "dB"].dropna()
            p = ttest_ind(xa, xb, equal_var=False).pvalue if min(len(xa), len(xb)) >= 2 else float("nan")
            lines.append(f"{ABBR[a]}/{ABBR[b]}: {fmt_p(p)}")
        return "\n".join(lines)
    
    
    # ---------------------------------------------------------------------------
    df = pd.read_csv(LONG_CSV, dtype={"date": str})
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    
    df["group"] = pd.Series(pd.NA, index=df.index, dtype="object")
    for label, key in GROUPS.items():
        df.loc[df["tillage"].str.contains(key, case=False, na=False), "group"] = label
    print("Tillage values in file:", sorted(df["tillage"].dropna().unique()))
    
    sub = df[
        (df["direction"] == DIRECTION)
        & (df["relative_orbit"] == ORBIT)
        & (df["n_used"] >= MIN_USED)
        & df["group"].notna()
    ]
    dates = sorted(sub["date"].unique())
    print(f"{DIRECTION} {ORBIT} dates in May:", [pd.Timestamp(d).strftime("%Y-%m-%d") for d in dates])
    if not dates:
        raise SystemExit("No data for this orbit.")
    
    order = [g for g in GROUPS if g in sub["group"].unique()]
    missing = [g for g in GROUPS if g not in order]
    if missing:
        print(f"No fields for: {missing}")
    
    # long format: one row per field x variable
    long = sub.melt(
        id_vars=["field_id", "group", "date"], value_vars=list(VARS),
        var_name="variable", value_name="dB",
    )
    long["variable"] = long["variable"].map(VARS)
    
    fig, axes = plt.subplots(1, len(TARGET_DAYS), figsize=(14, 5.5), sharey=True)
    
    for ax, (name, day) in zip(axes, TARGET_DAYS.items()):
        target = pd.Timestamp(year=pd.Timestamp(dates[0]).year, month=5, day=day)
        date = min(dates, key=lambda d: abs(pd.Timestamp(d) - target))
        d = long[long["date"] == date]
    
        sns.boxplot(data=d, x="variable", y="dB", hue="group",
                    order=list(VARS.values()), hue_order=order, palette=PALETTE,
                    showfliers=False, width=0.75, ax=ax)
        sns.stripplot(data=d, x="variable", y="dB", hue="group",
                      order=list(VARS.values()), hue_order=order, palette=["k"] * len(order),
                      dodge=True, size=2, alpha=0.3, jitter=0.15, legend=False, ax=ax)
    
        counts = sub[sub["date"] == date].groupby("group")["field_id"].nunique()
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles[:len(order)], [f"{g} (n={counts.get(g, 0)})" for g in order],
                  fontsize=8, loc="lower right", frameon=False)
    
        date_str = pd.Timestamp(date).strftime("%Y-%m-%d")
        ax.set_title(f"{date_str} ({name})", fontsize=11)
        ax.set_xticks(range(len(VARS)))
        ax.set_xticklabels([ttest_label(d, v) for v in VARS.values()], fontsize=9)
        ax.set_xlabel("")
        ax.set_ylabel("Backscatter / ratio (dB)")
        ax.grid(axis="y", alpha=0.3)
    
        print(f"\n{name}: {date_str} (medians, dB)")
        print(sub[sub["date"] == date].groupby("group")[list(VARS)].median()
              .rename(columns=VARS).round(2).loc[order])
    
    fig.suptitle(f"Sentinel-1 {DIRECTION} {ORBIT}, fields with n_used ≥ {MIN_USED}")
    plt.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)


def xgb_till_s1():
    """
    Simple XGBoost classification of tillage type (Pflug / Mulchsaat / Streifensaat)
    from 6 Sentinel-1 predictors: VV, VH, CR of asc 88 at mid and end of May.
    
    Validation: repeated "leave farms out" splits. Farms (not fields) are shuffled
    and assigned to train / validation / test until ~60 / 20 / 20 % of the fields
    are reached, so no farm appears in more than one set. The validation set is
    used for early stopping, the test set only for the final scores.
    """

    LONG_CSV = "data/s1_cr_may2022_long.csv"
    OUT_PNG = "data/xgb_tillage_asc88.png"
    
    DIRECTION, ORBIT = "asc", 88
    MIN_USED = 10
    TARGET_DAYS = {"mid": 15, "end": 31}
    GROUPS = {"Pflug": "pflug", "Mulchsaat": "mulch", "Streifensaat": "streifen"}
    VARS = {"vv_db": "VV", "vh_db": "VH", "cr_db": "CR"}
    
    SPLIT = (0.6, 0.2, 0.2)          # train / validation / test share of fields
    N_REPEATS = 20                   # number of random farm splits
    SEED = 42
    
    XGB_PARAMS = dict(
        n_estimators=100, learning_rate=0.05, max_depth=3, min_child_weight=3,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        objective="multi:softprob", eval_metric="mlogloss",
        early_stopping_rounds=50, n_jobs=-1,
    )
    
    # ---------------------------------------------------------------------------
    # Feature table: one row per field, 6 predictors
    # ---------------------------------------------------------------------------
    df = pd.read_csv(LONG_CSV, dtype={"date": str})
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df["group"] = pd.Series(pd.NA, index=df.index, dtype="object")
    for label, key in GROUPS.items():
        df.loc[df["tillage"].str.contains(key, case=False, na=False), "group"] = label
    
    sub = df[(df["direction"] == DIRECTION) & (df["relative_orbit"] == ORBIT)
             & df["group"].notna()].copy()
    sub.loc[sub["n_used"] < MIN_USED, list(VARS)] = np.nan     # too few pixels -> missing
    
    dates = sorted(sub["date"].unique())
    chosen = {}
    for name, day in TARGET_DAYS.items():
        target = pd.Timestamp(year=pd.Timestamp(dates[0]).year, month=5, day=day)
        chosen[name] = min(dates, key=lambda d: abs(pd.Timestamp(d) - target))
    print("Dates:", {k: pd.Timestamp(v).strftime("%Y-%m-%d") for k, v in chosen.items()})
    
    parts = []
    for name, date in chosen.items():
        p = sub[sub["date"] == date].set_index("field_id")[list(VARS)]
        p.columns = [f"{VARS[c]}_{name}" for c in p.columns]
        parts.append(p)
    X = pd.concat(parts, axis=1)
    meta = sub.drop_duplicates("field_id").set_index("field_id")[["group", "betriebs_id"]]
    data = meta.join(X, how="inner")
    data = data[data[X.columns].notna().any(axis=1)]           # at least one predictor
    
    features = list(X.columns)
    classes = [g for g in GROUPS if g in data["group"].unique()]
    y_all = data["group"].map({c: i for i, c in enumerate(classes)}).to_numpy()
    farms_all = data["betriebs_id"].to_numpy()
    
    print(f"\n{len(data)} fields from {data['betriebs_id'].nunique()} farms")
    print(data.groupby("group").agg(fields=("betriebs_id", "size"),
                                    farms=("betriebs_id", "nunique")).loc[classes])
    print(f"Missing predictor values: {data[features].isna().mean().mean() * 100:.1f} %")
    
    
    # ---------------------------------------------------------------------------
    # Farm-grouped split
    # ---------------------------------------------------------------------------
    def farm_split(rng):
        """Assign whole farms to train/val/test so field shares are ~SPLIT."""
        counts = data.groupby("betriebs_id").size()
        farms = counts.index.to_numpy().copy()
        rng.shuffle(farms)
        cum = np.cumsum(counts.loc[farms].to_numpy()) / counts.sum()
        b1, b2 = SPLIT[0], SPLIT[0] + SPLIT[1]
        tr = set(farms[cum <= b1])
        va = set(farms[(cum > b1) & (cum <= b2)])
        te = set(farms[cum > b2])
        return (np.isin(farms_all, list(tr)), np.isin(farms_all, list(va)),
                np.isin(farms_all, list(te)))
    
    
    # ---------------------------------------------------------------------------
    # Repeated training
    # ---------------------------------------------------------------------------
    rng = np.random.default_rng(SEED)
    scores, importances = [], []
    cm_total = np.zeros((len(classes), len(classes)), dtype=int)
    Xv = data[features].to_numpy()
    
    for r in range(N_REPEATS):
        tr, va, te = farm_split(rng)
        if len(set(y_all[tr])) < len(classes) or len(set(y_all[va])) < len(classes):
            print(f"repeat {r}: a class is missing in train/val, skipped")
            continue
    
        model = XGBClassifier(**XGB_PARAMS, random_state=SEED + r)
        model.fit(
            Xv[tr], y_all[tr],
            sample_weight=compute_sample_weight("balanced", y_all[tr]),
            eval_set=[(Xv[va], y_all[va])], verbose=False,
        )
        pred = model.predict(Xv[te])
        yt = y_all[te]
    
        scores.append(dict(
            repeat=r, n_train=tr.sum(), n_val=va.sum(), n_test=te.sum(),
            best_iter=model.best_iteration,
            accuracy=accuracy_score(yt, pred),
            balanced_accuracy=balanced_accuracy_score(yt, pred),
            f1_macro=f1_score(yt, pred, average="macro"),
        ))
        cm_total += confusion_matrix(yt, pred, labels=range(len(classes)))
        gain = model.get_booster().get_score(importance_type="gain")
        importances.append([gain.get(f"f{i}", 0.0) for i in range(len(features))])
    
    scores = pd.DataFrame(scores)
    imp = pd.DataFrame(importances, columns=features)
    imp = imp.div(imp.sum(axis=1), axis=0)                     # relative gain per repeat
    
    print(f"\nTest scores over {len(scores)} farm splits (mean ± std):")
    for m in ["accuracy", "balanced_accuracy", "f1_macro"]:
        print(f"  {m:18s} {scores[m].mean():.3f} ± {scores[m].std():.3f}")
    print(f"  chance level (balanced accuracy) = {1 / len(classes):.3f}")
    print(f"  split sizes (mean): train {scores.n_train.mean():.0f}, "
          f"val {scores.n_val.mean():.0f}, test {scores.n_test.mean():.0f} fields")
    
    print("\nConfusion matrix (summed over test sets, rows = true):")
    print(pd.DataFrame(cm_total, index=classes, columns=classes))
    
    # ---------------------------------------------------------------------------
    # Plots: confusion matrix (row-normalised) and feature importance
    # ---------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1, 1.1]})
    
    cm_norm = cm_total / cm_total.sum(axis=1, keepdims=True)
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1,
                xticklabels=classes, yticklabels=classes, cbar=False, ax=axes[0])
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    axes[0].set_title(
        f"Confusion matrix (test, {len(scores)} farm splits)\n"
        f"balanced acc. {scores.balanced_accuracy.mean():.2f} ± {scores.balanced_accuracy.std():.2f}"
        f" (chance {1 / len(classes):.2f})", fontsize=10)
    
    order = imp.mean().sort_values(ascending=False).index
    sns.boxplot(data=imp[order], orient="h", color="tab:blue", showfliers=False, ax=axes[1])
    axes[1].set_xlabel("Relative importance (gain)")
    axes[1].set_title("Feature importance over farm splits", fontsize=10)
    axes[1].grid(axis="x", alpha=0.3)
    
    fig.suptitle(f"XGBoost tillage classification, S1 {DIRECTION} {ORBIT}, May")
    plt.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)

def add_crop_type(year):
    wide = pd.read_parquet(f"data/all/s1_s2_cr_wide_{year}.parquet")
    
    crops = pyogrio.read_dataframe(f"data/all/lnf_{year}_tillage.gpkg", columns=["id", "lnf_code"],
                                   read_geometry=False).rename(columns={"id": "field_id"})
    
    wide = wide.merge(crops, on="field_id", how="left")
    wide.insert(3, "lnf_code", wide.pop("lnf_code"))   # after betriebs_id
    
    print(wide["lnf_code"].isna().sum(), "fields without crop code")
    wide.to_parquet(f"data/all/s1_s2_cr_wide_{year}_ct.parquet", index=False)

def plot_tilltype_per_crop(year):
    TOP_N = 40                      # most frequent crops; the rest are grouped as "other"

    wide = pd.read_parquet(f"data/all/s1_s2_cr_wide_{year}_ct.parquet")
    lnf_classification_path = '/mnt/eo-nas1/data/landuse/documentation/LNF_code_classification_20260217.xlsx'
    crop_table = pd.read_excel(lnf_classification_path, sheet_name='label_sheet')
    
    crop_table = crop_table.dropna(subset=["LNF_code", "Crop_DE"])
    code_to_name = dict(zip(
        pd.to_numeric(crop_table["LNF_code"], errors="coerce").astype("Int64").astype(str),
        crop_table["Crop_DE"].astype(str).str.strip(),
    ))

    ct = pd.crosstab(wide["lnf_code"].astype("Int64").astype(str), wide["tillage"])
    ct = ct.loc[ct.sum(axis=1).sort_values(ascending=False).index]
    if len(ct) > TOP_N:
        ct = pd.concat([ct.iloc[:TOP_N], ct.iloc[TOP_N:].sum().to_frame("other").T])

    ct.index = [code_to_name.get(c, c) for c in ct.index]
    ct.index = [code_to_name.get(c, c)[:30] for c in ct.index]
    
    order = [t for t in ["Pflug", "Mulchsaat", "Streifensaat", "Direktsaat"] if t in ct.columns]
    order += [t for t in ct.columns if t not in order]
    colors = {"Pflug": "0.65", "Mulchsaat": "tab:orange",
              "Streifensaat": "tab:green", "Direktsaat": "tab:blue"}
    
    ax = ct[order].iloc[::-1].plot.barh(
        stacked=True, figsize=(9, 0.4 * len(ct) + 1.5),
        color=[colors.get(t, "tab:purple") for t in order], width=0.8,
    )
    for i, total in enumerate(ct.sum(axis=1).iloc[::-1]):
        ax.text(total, i, f" {total}", va="center", fontsize=8)
    
    ax.set_xlabel("Number of fields")
    ax.set_ylabel("Crop (lnf_code)")
    ax.set_title("Fields per crop and tillage type")
    ax.legend(title="Tillage", frameon=False, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"data/all/figs/till_dist_{year}.png", dpi=150)
    
    print(ct[order].assign(total=ct.sum(axis=1)))

def get_approx_till_time():
    path = "/mnt/Data-Labo-RE/27_Natural_Resources-RE/321.4_WAUM_protected/Daten/Feldkalender_AUI/Feldkalender_2022.txt"
    df = pd.read_csv(path, sep="\t", engine="python", encoding="latin-1")   # e.g. sep="\t" if you know it
    print(df.columns.tolist())
    df = df.loc[:,['datum', 'kultur', 'kulturnr', 'massnahme']]
    # sub = df[
    #     (df["kanton"] == "AG")                               # equal to a value
    #     & (df["lnf_code"].isin([513, 521, 522]))              # one of several values
    #     & (df["groesse"].between(5000, 50000))                # numeric range (inclusive)
    #     & (df["tillage"].str.contains("mulch", case=False, na=False))   # text contains
    #     & (df["betriebsnummer"].notna())                      # not missing
    # ]
    # print(len(sub), "rows")
    # valid_massnahmen = ["Mulchsaat", "Direktsaat", "Strip-Till", "Frässaat", "Kombinationssaat", "Pflügen", "Mulchen", "Wieseneggen"]
    valid_massnahmen = ["Mulchsaat", "Direktsaat", "Strip-Till", "Pflügen", "Mulchen"]
    sub_df = df[df["massnahme"].isin(valid_massnahmen)]
    print(sub_df)

    sub_df = sub_df.copy()
    sub_df["datum"] = pd.to_datetime(sub_df["datum"], dayfirst=True, errors="coerce")
    sub_df = sub_df[sub_df["datum"] <= "2022-06-30"]
    
    summary = (sub_df.groupby(["kultur", "massnahme"])["datum"]
           .agg(n="count", median="median", min="min", max="max")
           .sort_values(["kultur", "n"], ascending=[True, False]))
    summary["median"] = summary["median"].dt.strftime("%Y-%m-%d")
    summary["min"] = summary["min"].dt.strftime("%Y-%m-%d")
    summary["max"] = summary["max"].dt.strftime("%Y-%m-%d")

    print(summary.to_string())

def plot_trajectories_tilltype(year=2021):
    """
    Mean ± std trajectories of S1 VV, VH and S2 NDVI, NDTI per tillage class,
    for three main crops (rows) and four variables (columns).
    
      NDVI = (B08 - B04) / (B08 + B04)      vegetation
      NDTI = (B11 - B12) / (B11 + B12)      crop residue cover (tillage index)
    
    S1: one ascending orbit (different orbits have different incidence angles and
    are not mixed). Values are averaged per acquisition date.
    S2: dates differ between tiles, so values are averaged in BIN_DAYS bins.
    The shaded band marks the typical tillage / sowing period of each crop.
    """
    

    
    LONG = f"data/all/s1_s2_cr_long_{year}.parquet"
    GPKG = f"data/all/lnf_{year}_tillage.gpkg"
    FIELD_ID_COL = "id"                       # same as in the extraction
    CROP_TABLE = "/mnt/eo-nas1/data/landuse/documentation/LNF_code_classification_20260217.xlsx"
    CROP_SHEET = "label_sheet"
    OUT_PNG = f"data/all/figs/trajectories_tillage_{year}_med.png"
    MIN_SHARE = 0.05   

    # crop label -> (regex on Crop_DE, typical tillage / sowing period as month-day).
    # Periods from August on belong to the previous year, the others to the
    # harvest year; the harvest year is taken from the data (last date).
    CROPS = {
        "Winterweizen": ("winterweizen", ("09-25", "10-31")),
        "Silomais": ("silo", ("04-01", "04-30")),
        "Winterraps (Speiseöl)": ("winterraps.*speise", ("08-01", "08-31")),
    }
    CLASSES = {"Pflug": "pflug", "Mulchsaat": "mulch",
               "Streifensaat": "streifen", "Direktsaat": "direkt"}
    COLORS = {"Pflug": "brown", "Mulchsaat": "aqua",
              "Streifensaat": "blue", "Direktsaat": "darkblue"}
    MERGE = {"Pflug": "Pflug", "Mulchsaat": "Mulchsaat",
             "Streifensaat": "Streifen/Direkt", "Direktsaat": "Streifen/Direkt"}
    # MERGE = None
    COLORS["Streifen/Direkt"] = "darkblue"
    COLORS["Andere"] = "darkblue"

    DIRECTION = "asc"
    ORBIT = None              # None -> most frequent ascending orbit
    BIN_DAYS = 10             # S2 time bins
    MIN_FIELDS = 5            # hide points based on fewer fields
    MIN_USED = 10             # S1 pixel threshold (as in the extraction)
     
    VARS = [("vv_db", "VV (dB)"), ("vh_db", "VH (dB)"), ("cr_db", "CR = VH/VV (dB)"),
            ("ndvi", "NDVI"), ("ndti", "NDTI")]
    S1_VARS = ("vv_db", "vh_db", "cr_db")
     
     
    # ---------------------------------------------------------------------------
    # Data
    # ---------------------------------------------------------------------------
    def read_long(path, columns):
        if path.endswith(".parquet"):
            return pd.read_parquet(path, columns=columns)
        return pd.read_csv(path, usecols=columns)
     
     
    cols = ["sensor", "field_id", "tillage", "date", "direction", "relative_orbit",
            "n_used", "vv_db", "vh_db", "cr_db", "B04", "B08", "B11", "B12"]
    long = read_long(LONG, cols)
    long["date"] = pd.to_datetime(long["date"].astype(str), format="%Y%m%d")
     
    # tillage class
    long["class"] = pd.Series(pd.NA, index=long.index, dtype="object")
    for label, key in CLASSES.items():
        long.loc[long["tillage"].str.contains(key, case=False, na=False), "class"] = label
    long = long[long["class"].notna()]
    if MERGE:
        long["class"] = long["class"].map(MERGE)
        PLOT_CLASSES = list(dict.fromkeys(MERGE.values()))
    else:
        PLOT_CLASSES = list(CLASSES)
     
    # crop per field: gpkg (id -> lnf_code) + crop table (code -> name)
    crop_of_field = pyogrio.read_dataframe(GPKG, columns=[FIELD_ID_COL, "lnf_code"],
                                           read_geometry=False)
    crop_of_field = crop_of_field.rename(columns={FIELD_ID_COL: "field_id"})
    names = (pd.read_excel(CROP_TABLE, sheet_name=CROP_SHEET)
             .dropna(subset=["LNF_code", "Crop_DE"])[["LNF_code", "Crop_DE"]])
    names["LNF_code"] = pd.to_numeric(names["LNF_code"], errors="coerce")
    crop_of_field["lnf_code"] = pd.to_numeric(crop_of_field["lnf_code"], errors="coerce")
    crop_of_field = crop_of_field.merge(names, left_on="lnf_code", right_on="LNF_code", how="left")
     
    crop_label = pd.Series(pd.NA, index=crop_of_field.index, dtype="object")
    for label, (pattern, _) in CROPS.items():
        hit = crop_of_field["Crop_DE"].str.contains(pattern, case=False, na=False, regex=True)
        crop_label[hit & crop_label.isna()] = label
        print(f"{label}: matched crop names {sorted(crop_of_field.loc[hit, 'Crop_DE'].unique())}")
    crop_of_field["crop"] = crop_label
    long = long.merge(crop_of_field[["field_id", "crop"]], on="field_id", how="inner")
    long = long[long["crop"].notna()]
     
    # S1: one orbit
    s1 = long[(long["sensor"] == "S1") & (long["direction"] == DIRECTION)]
    if ORBIT is None:
        ORBIT = int(s1["relative_orbit"].value_counts().idxmax())
    s1 = s1[(s1["relative_orbit"] == ORBIT) & (s1["n_used"] >= MIN_USED)].copy()
    s1["t"] = s1["date"]
     
    # S2: indices + time bins
    s2 = long[long["sensor"] == "S2"].copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        s2["ndvi"] = (s2["B08"] - s2["B04"]) / (s2["B08"] + s2["B04"])
        s2["ndti"] = (s2["B11"] - s2["B12"]) / (s2["B11"] + s2["B12"])
    t0 = s2["date"].min()
    s2["t"] = t0 + pd.to_timedelta(((s2["date"] - t0).dt.days // BIN_DAYS) * BIN_DAYS
                                   + BIN_DAYS / 2, unit="D")
     
    YEAR = long["date"].max().year            # harvest year of the data
    T_MIN, T_MAX = long["date"].min(), long["date"].max()
     
     
    def period(md):
        """month-day -> Timestamp in the season Aug (YEAR-1) to Jul (YEAR)."""
        month = int(md.split("-")[0])
        return pd.Timestamp(f"{YEAR - 1 if month >= 8 else YEAR}-{md}")
     
     
    print(f"\nData from {T_MIN.date()} to {T_MAX.date()} (harvest year {YEAR})")
    print(f"S1 {DIRECTION} orbit {ORBIT}; S2 in {BIN_DAYS}-day bins")
    counts = (long.drop_duplicates("field_id").groupby(["crop", "class"]).size()
              .unstack(fill_value=0))
    print("\nFields per crop and class:")
    print(counts)
     
     
    # ---------------------------------------------------------------------------
    # Plot: rows = crops, columns = variables
    # ---------------------------------------------------------------------------
    def stats(df, var):
        per_field = df.groupby(["t", "field_id"])[var].mean().dropna()
        g = per_field.groupby(level="t")
        out = pd.DataFrame({"mean": g.median(), "lo": g.quantile(0.25),
                            "hi": g.quantile(0.75), "n": g.size()})
        n_fields = df["field_id"].nunique()
        return out[out["n"] >= max(MIN_FIELDS, MIN_SHARE * n_fields)]
     
     
    crops = [c for c in CROPS if c in long["crop"].unique()]
    fig, axes = plt.subplots(len(crops), len(VARS), figsize=(5 * len(VARS), 3.6 * len(crops)),
                             sharex=True, squeeze=False)
     
    for r, crop in enumerate(crops):
        start, end = (period(md) for md in CROPS[crop][1])
        for c, (var, label) in enumerate(VARS):
            ax = axes[r, c]
            src = s1 if var in S1_VARS else s2
            d = src[src["crop"] == crop]
            for cls in PLOT_CLASSES:
                dc = d[d["class"] == cls]
                if dc.empty:
                    continue
                st = stats(dc, var)
                if st.empty:
                    continue
                n = dc["field_id"].nunique()
                ax.plot(st.index, st["mean"], "-", lw=1.4, color=COLORS[cls], label=f"{cls} (n={n})")
                ax.fill_between(st.index, st["lo"], st["hi"], color=COLORS[cls], alpha=0.15, lw=0)
            ax.axvspan(start, end, color="gold", alpha=0.25, lw=0)
            ax.set_xlim(T_MIN - pd.Timedelta(days=5), T_MAX + pd.Timedelta(days=5))
            ax.grid(alpha=0.3)
            if r == 0:
                ax.set_title(label + (f"  (S1 {DIRECTION} {ORBIT})" if var in S1_VARS else ""),
                             fontsize=10)
            if c == 0:
                ax.set_ylabel(crop, fontsize=11)
            if c == len(VARS) - 1:
                ax.legend(fontsize=7, frameon=False, loc="best")
            ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[8, 10, 12, 2, 4, 6]))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
     
    fig.suptitle("Median and interquartile range per tillage class; shaded: typical tillage / sowing period", fontsize=12)
    plt.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)


"""
Build predictor table X and target table Y for the tillage classification.

Per field, the time series is reduced to two-week bins around the typical
tillage date X of its crop (median over all tillage measures, from
tillage_dates_per_crop.csv). A bin labelled by its end, e.g. "tp2" = X to X+2
weeks, holds the mean of all acquisitions in that window. S1 is averaged in
linear units (VV, VH) and converted back to dB; CR of a bin = VH_dB - VV_dB of
the bin means.

Predictors
  S1  VV, VH, CR (dB, averaged in linear units) for ascending and descending,
      see S1_MODE / S1_ANOMALY
  S2  B01 ... B12 (all available bands), NDVI, NDTI
  crop lnf_code (categorical, stored as text -> .astype("category") before training)

Targets
  Y: sample_id, year, field_id, betriebs_id, lnf_code, Crop_DE, tillage_date, tillage

Several years are combined; sample_id = "<year>_<field_id>" is unique across years.
"""
# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
# YEARS = {
#     2021: dict(long="data/all/s1_s2_cr_long_2021.parquet", gpkg="data/all/lnf_2021_tillage.gpkg"),
#     2022: dict(long="data/all/s1_s2_cr_long_2022.parquet", gpkg="data/all/lnf_2022_tillage.gpkg"),
# }
# FIELD_ID_COL = "id"                          # id column in the gpkg = field_id in the long file
# DATES_CSV = "tillage_dates_per_crop.csv"     # from tillage_dates_per_crop.py
# OUT_X = "data/all/X_ip.parquet"
# OUT_Y = "data/all/Y_ip.parquet"

# # two-week bins in weeks relative to the tillage date, labelled by their end:
# # t0 = X-2 to X weeks, tp2 = X to X+2, ..., tp8 = X+6 to X+8
# BINS = [(-2, 0), (0, 2), (2, 4), (4, 6), (6, 8)]
 
# CLASSES = {"Pflug": "pflug", "Mulchsaat": "mulch",
#            "Streifensaat": "streifen", "Direktsaat": "direkt"}
 
# # S1
# S1_MODE = "single"      # "single": most frequent orbit per direction and year
#                         # "per_field": best covered orbit per field and direction
# S1_ANOMALY = False      # divide by the mean of all fields of the same orbit and date
#                         # (linear), i.e. dB difference to that mean
# MIN_USED = 10
 
# # S2
# S2_INTERPOLATE = True   # fill empty bins by linear interpolation in time
# MAX_GAP_DAYS = 42       # max distance (each side) from bin centre to the acquisitions used
# S2_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A",
#             "B09", "B10", "B11", "B12"]
 
 
# def bin_label(hi):
#     return "t0" if hi == 0 else (f"tp{hi}" if hi > 0 else f"tm{-hi}")
 
 
# ---------------------------------------------------------------------------
# Tillage date per crop
# ---------------------------------------------------------------------------
def load_crop_dates():
    d = pd.read_csv(DATES_CSV, dtype={"LNF_code": str})
    d = d.dropna(subset=["median_date"])
    rows = []
    for _, r in d.iterrows():
        for code in str(r["LNF_code"]).split(","):
            code = code.strip()
            if code and code != "nan":
                rows.append(dict(lnf_code=int(float(code)), Crop_DE=r["Crop_DE"],
                                 median_md=r["median_date"], date_source=r["source"]))
    return pd.DataFrame(rows).drop_duplicates("lnf_code")
 
 
def tillage_date(md, year):
    """'MM-DD' -> date in the season Aug (year-1) to Jul (year)."""
    month = int(md[:2])
    return pd.Timestamp(f"{year - 1 if month >= 8 else year}-{md}")
 
 
# ---------------------------------------------------------------------------
# One year
# ---------------------------------------------------------------------------
def interpolate_s2(obs, quality, s2, s2vars, ref, fields):
    """
    Fill empty S2 bins by linear interpolation at the bin centre between the last
    clear acquisition before and the first after it (any date, not only bins).
    quality: 1 = both acquisitions lie in the neighbouring bins (within 2 weeks
    before / after the empty bin), 2 = at least one lies further away
    (max MAX_GAP_DAYS from the bin centre). Bins without data on both sides stay NaN.
    """
    edges = {bin_label(hi): (lo, hi) for lo, hi in BINS}
    miss = obs.index[obs[ref].isna()].to_frame(index=False)
    if miss.empty:
        return obs, quality
    td = fields.set_index("field_id")["tillage_date"]
    lo = miss["bin"].map(lambda b: edges[b][0]).to_numpy()
    hi = miss["bin"].map(lambda b: edges[b][1]).to_numpy()
    base = miss["field_id"].map(td).to_numpy()
    miss["start"] = base + pd.to_timedelta(lo * 7, unit="D")
    miss["end"] = base + pd.to_timedelta(hi * 7, unit="D")
    miss["centre"] = miss["start"] + (miss["end"] - miss["start"]) / 2
    miss = miss.sort_values("centre").reset_index(drop=True)
 
    acq = s2[["field_id", "date"] + s2vars].sort_values("date")
    tol = pd.Timedelta(days=MAX_GAP_DAYS)
    prev = pd.merge_asof(miss, acq.rename(columns={"date": "t_prev"}), left_on="centre",
                         right_on="t_prev", by="field_id", direction="backward", tolerance=tol)
    nxt = pd.merge_asof(miss, acq.rename(columns={"date": "t_next"}), left_on="centre",
                        right_on="t_next", by="field_id", direction="forward", tolerance=tol)
 
    ok = prev["t_prev"].notna() & nxt["t_next"].notna()
    span = (nxt["t_next"] - prev["t_prev"]).dt.total_seconds()
    frac = ((miss["centre"] - prev["t_prev"]).dt.total_seconds() / span).where(span > 0, 0.5)
    two_weeks = pd.Timedelta(days=14)
    adjacent = (prev["t_prev"] >= miss["start"] - two_weeks) & (nxt["t_next"] < miss["end"] + two_weeks)
 
    idx = pd.MultiIndex.from_frame(miss.loc[ok, ["field_id", "bin"]])
    for v in s2vars:
        val = prev[v] + (nxt[v] - prev[v]) * frac
        obs.loc[idx, v] = val[ok].to_numpy()
    quality.loc[idx] = np.where(adjacent[ok], 1.0, 2.0)
    return obs, quality
 
 
def build_year(year, cfg, crop_dates):
    print(f"\n=== {year} ===")
    cols = ["sensor", "field_id", "betriebs_id", "tillage", "date", "direction",
            "relative_orbit", "n_used", "vv_db", "vh_db", *S2_BANDS]
    schema = pq.read_schema(cfg["long"]).names
    long = pd.read_parquet(cfg["long"], columns=[c for c in cols if c in schema])
    bands = [b for b in S2_BANDS if b in long.columns]
    long["date"] = pd.to_datetime(long["date"].astype(str), format="%Y%m%d")
 
    # --- fields: class, crop, tillage date ---
    fields = long.drop_duplicates("field_id")[["field_id", "betriebs_id", "tillage"]].copy()
    fields["class"] = pd.Series(pd.NA, index=fields.index, dtype="object")
    for label, key in CLASSES.items():
        fields.loc[fields["tillage"].str.contains(key, case=False, na=False), "class"] = label
    n0 = len(fields)
    fields = fields[fields["class"].notna()]
 
    crops = pyogrio.read_dataframe(cfg["gpkg"], columns=[FIELD_ID_COL, "lnf_code"],
                                   read_geometry=False)
    crops = crops.rename(columns={FIELD_ID_COL: "field_id"})
    crops["lnf_code"] = pd.to_numeric(crops["lnf_code"], errors="coerce").astype("Int64")
    fields = fields.merge(crops, on="field_id", how="left")
    fields = fields.merge(crop_dates, on="lnf_code", how="left")
    no_date = fields["median_md"].isna()
    print(f"fields: {n0} | known class: {len(fields)} | without tillage date for crop: "
          f"{no_date.sum()} (excluded)")
    if no_date.any():
        print("  crops without date:", fields.loc[no_date, "lnf_code"].value_counts().head(10).to_dict())
    fields = fields[~no_date].copy()
    fields["tillage_date"] = [tillage_date(m, year) for m in fields["median_md"]]
 
    long = long.merge(fields[["field_id", "tillage_date"]], on="field_id", how="inner")
    off = (long["date"] - long["tillage_date"]).dt.days
    long["bin"] = pd.NA
    for lo, hi in BINS:
        long.loc[(off >= lo * 7) & (off < hi * 7), "bin"] = bin_label(hi)
    long_all = long                      # full series (S2 interpolation uses it)
    long = long[long["bin"].notna()]
 
    feats = []
 
    # --- S1 ---
    s1 = long[(long["sensor"] == "S1") & (long["n_used"] >= MIN_USED)].copy()
    for d in ("asc", "desc"):
        sd = s1[s1["direction"] == d].dropna(subset=["vv_db", "vh_db"])
        if sd.empty:
            continue
        # linear power
        sd["vv_lin"] = 10 ** (sd["vv_db"] / 10)
        sd["vh_lin"] = 10 ** (sd["vh_db"] / 10)
        if S1_ANOMALY:            # ratio to the orbit/date mean over all fields (no labels used)
            for v in ("vv_lin", "vh_lin"):
                sd[v] = sd[v] / sd.groupby(["relative_orbit", "date"])[v].transform("mean")
        if S1_MODE == "single":
            orbit = int(sd["relative_orbit"].value_counts().idxmax())
            sd = sd[sd["relative_orbit"] == orbit]
            n_cov = sd["field_id"].nunique()
            print(f"S1 {d}: orbit {orbit}, {n_cov} of {len(fields)} fields covered")
        else:                     # best covered orbit per field
            cov = sd.groupby(["field_id", "relative_orbit"]).size().rename("n").reset_index()
            best = cov.sort_values("n", ascending=False).drop_duplicates("field_id")
            sd = sd.merge(best[["field_id", "relative_orbit"]], on=["field_id", "relative_orbit"])
            orb = best.set_index("field_id")["relative_orbit"].rename(f"{d}_orbit")
            feats.append(orb.to_frame())
            print(f"S1 {d}: orbits used per field: {best['relative_orbit'].value_counts().to_dict()}")
        m = sd.groupby(["field_id", "bin"])[["vv_lin", "vh_lin"]].mean()
        m["VV"] = 10 * np.log10(m["vv_lin"])
        m["VH"] = 10 * np.log10(m["vh_lin"])
        m["CR"] = m["VH"] - m["VV"]
        w = m[["VV", "VH", "CR"]].unstack("bin")
        w.columns = [f"{d}_{v}_{b}" for v, b in w.columns]
        feats.append(w)
 
    # --- S2 ---
    s2 = long_all[long_all["sensor"] == "S2"].dropna(subset=bands, how="all").copy()
    if not s2.empty:
        with np.errstate(divide="ignore", invalid="ignore"):
            if {"B04", "B08"} <= set(bands):
                s2["NDVI"] = (s2["B08"] - s2["B04"]) / (s2["B08"] + s2["B04"])
            if {"B11", "B12"} <= set(bands):
                s2["NDTI"] = (s2["B11"] - s2["B12"]) / (s2["B11"] + s2["B12"])
        s2vars = bands + [v for v in ("NDVI", "NDTI") if v in s2.columns]
        ref = "B04" if "B04" in bands else bands[0]
        s2 = s2.dropna(subset=[ref])
 
        # observed bin means on the full field x bin grid
        labels = [bin_label(hi) for _, hi in BINS]
        grid = pd.MultiIndex.from_product([fields["field_id"], labels], names=["field_id", "bin"])
        obs = s2[s2["bin"].notna()].groupby(["field_id", "bin"])[s2vars].mean().reindex(grid)
        quality = pd.Series(np.where(obs[ref].notna(), 0.0, np.nan), index=grid)
 
        if S2_INTERPOLATE:
            obs, quality = interpolate_s2(obs, quality, s2, s2vars, ref, fields)
 
        w = obs.unstack("bin")
        w.columns = [f"S2_{v}_{b}" for v, b in w.columns]
        q = quality.unstack("bin")
        q.columns = [f"S2_quality_{b}" for b in q.columns]
        feats += [w, q]
 
        qc = quality.groupby(level="bin").value_counts(dropna=False).unstack(fill_value=0)
        qc = qc.reindex(labels)
        qc.columns = [{0.0: "observed (0)", 1.0: "interp. adjacent (1)",
                       2.0: "interp. further (2)"}.get(c, "missing") for c in qc.columns]
        print("S2 bins per quality class:")
        print(qc.to_string())
 
    X = pd.concat(feats, axis=1) if feats else pd.DataFrame(index=fields["field_id"])
    labels = [bin_label(hi) for _, hi in BINS]
    order = [c for pre in ("asc_", "desc_", "S2_") for c in X.columns
             if c.startswith(pre) and not c.endswith("_orbit")]
    groups = ("asc_", "desc_", "S2_")
    order = sorted(order, key=lambda c: (next(i for i, g in enumerate(groups) if c.startswith(g)),
                                         c.rsplit("_", 1)[0], labels.index(c.rsplit("_", 1)[1])))
    X = X[[c for c in X.columns if c.endswith("_orbit")] + order]
 
    X = fields.set_index("field_id")[["lnf_code"]].join(X)
    has_data = X.drop(columns=["lnf_code"]).filter(regex="^(asc_|desc_|S2_)").notna().any(axis=1)
    print(f"fields with at least one S1/S2 predictor: {has_data.sum()} of {len(X)}")
    X = X[has_data]
 
    Y = fields.set_index("field_id").loc[X.index, ["betriebs_id", "lnf_code", "Crop_DE",
                                                   "tillage_date", "date_source", "class"]]
    Y = Y.rename(columns={"class": "tillage"}).reset_index()
    Y.insert(0, "year", year)
    Y.insert(0, "sample_id", [f"{year}_{f}" for f in Y["field_id"]])
    X = X.reset_index()
    X.insert(0, "sample_id", Y["sample_id"].values)
    X.insert(1, "year", year)
    return X, Y
 
 
def prepare_xy():
    os.makedirs(os.path.dirname(OUT_X) or ".", exist_ok=True)
    crop_dates = load_crop_dates()
    print(f"Tillage dates for {len(crop_dates)} LNF codes; bins (weeks): {BINS}")
    print(f"S1_MODE = {S1_MODE}, S1_ANOMALY = {S1_ANOMALY}")
 
    Xs, Ys = [], []
    for year, cfg in YEARS.items():
        X, Y = build_year(year, cfg, crop_dates)
        Xs.append(X)
        Ys.append(Y)
    X = pd.concat(Xs, ignore_index=True)
    Y = pd.concat(Ys, ignore_index=True)
 
    # categorical predictors are stored as text (the category dtype does not survive
    # every parquet round trip); convert with .astype("category") before training
    X["lnf_code"] = X["lnf_code"].astype("Int64").astype(str)
    for c in [c for c in X.columns if c.endswith("_orbit")]:
        X[c] = X[c].astype("Int64").astype(str).replace("<NA>", pd.NA)
    X = X.drop(columns=["field_id"])          # identification via sample_id
    assert (X["sample_id"] == Y["sample_id"]).all()
 
    X.to_parquet(OUT_X, index=False)
    Y.to_parquet(OUT_Y, index=False)
 
    pred = [c for c in X.columns if c not in ("sample_id", "year")]
    print(f"\nX: {X.shape[0]} samples x {len(pred)} predictors | Y: {Y.shape[0]} samples")
    print("\nSamples per year and class:")
    print(pd.crosstab(Y["year"], Y["tillage"], margins=True))
    print("\nSamples per crop (top 10) and class:")
    top = Y["Crop_DE"].value_counts().index[:10]
    print(pd.crosstab(Y.loc[Y["Crop_DE"].isin(top), "Crop_DE"].str[:40],
                      Y.loc[Y["Crop_DE"].isin(top), "tillage"]))
    miss = X[pred].drop(columns=["lnf_code"]).isna().mean()
    groups = miss.groupby(lambda c: c.split("_")[0] if not c.startswith("S2") else "S2").mean()
    print("\nMissing values per predictor group (share):")
    print(groups.round(3).to_string())
    print(f"\nWritten: {OUT_X}, {OUT_Y}")


def field_orientation(geoms):
    """Long-axis orientation (0-180 deg from north) and elongation of polygons."""
    rect = shapely.oriented_envelope(np.asarray(geoms))
    orient = np.full(len(rect), np.nan)
    elong = np.full(len(rect), np.nan)
    for i, r in enumerate(rect):
        if r is None or r.is_empty or r.geom_type != "Polygon":
            continue
        xy = np.asarray(r.exterior.coords)[:4]
        e1, e2 = xy[1] - xy[0], xy[2] - xy[1]
        l1, l2 = np.hypot(*e1), np.hypot(*e2)
        long_edge = e1 if l1 >= l2 else e2
        # azimuth clockwise from north: atan2(dx, dy)
        orient[i] = np.degrees(np.arctan2(long_edge[0], long_edge[1])) % 180
        short = min(l1, l2)
        elong[i] = max(l1, l2) / short if short > 0 else np.nan
    return orient, elong


def relative_angle(orient_deg, look_deg):
    """Angle between an axis (0-180) and the look direction, folded to 0-90 deg."""
    d = (np.asarray(orient_deg) - look_deg) % 180
    return np.minimum(d, 180 - d)


def add_orientation(x_file="cfactor/data/all/X_ip.parquet", y_file="cfactor/data/all/Y_ip.parquet",
                    gpkg_pattern="cfactor/data/all/lnf_{year}_tillage.gpkg", gpkg_id="id", out_file=None):
    """
    Compute orientation predictors for every sample in X (via Y: year, field_id)
    and write them into X (out_file defaults to x_file).
    """
    X = pd.read_parquet(x_file)
    Y = pd.read_parquet(y_file)
    LOOK_DEG = {"asc": 77.0, "desc": 283.0}
    METRIC_CRS = "EPSG:2056"           # LV95; grid north deviates from true north by < 1 deg in CH
    assert (X["sample_id"] == Y["sample_id"]).all()

    parts = []
    for year, ids in Y.groupby("year")["field_id"]:
        g = pyogrio.read_dataframe(gpkg_pattern.format(year=year), columns=[gpkg_id])
        g = g.to_crs(METRIC_CRS)
        g["field_id"] = g[gpkg_id].astype(str).str.replace(r"\.0$", "", regex=True)
        g = g[g["field_id"].isin(set(ids.astype(str)))]
        o, e = field_orientation(g.geometry.values)
        parts.append(pd.DataFrame({"sample_id": [f"{year}_{f}" for f in g["field_id"]],
                                   "orient_deg": o, "elongation": e}))
    o = pd.concat(parts, ignore_index=True)
    for d, look in LOOK_DEG.items():
        o[f"rel_{d}_deg"] = relative_angle(o["orient_deg"], look)

    new_cols = ["orient_deg", "elongation", "rel_asc_deg", "rel_desc_deg"]
    X = X.drop(columns=[c for c in new_cols if c in X.columns])
    X = X.merge(o, on="sample_id", how="left")
    assert (X["sample_id"] == Y["sample_id"]).all()       # row order unchanged
    X.to_parquet(out_file or x_file, index=False)

    print(f"Orientation added for {X['orient_deg'].notna().sum():,} of {len(X):,} samples")
    print(X[new_cols].describe().round(2).to_string())
    return X



"""
Compare predictor setups for the tillage classification with XGBoost:
  1 crop            lnf_code only
  2 S1              asc/desc VV, VH, CR
  3 S2              S2 bands + NDVI, NDTI
  4 S1+S2
  5 crop+S1+S2

All setups use the same fields (complete S1 and S2 after interpolation) and the
same random stratified K-fold splits, so the scores are directly comparable.
"""

def compare_s1_s2():
    X_FILE = "data/all/X_ip.parquet"
    Y_FILE = "data/all/Y_ip.parquet"
    SCORE_ON = "test" 
    OUT_CSV = f"data/all/setup_scores_{SCORE_ON}.csv"
    OUT_PNG = f"data/all/setup_scores_{SCORE_ON}.png"
    BINARY = True                 # Pflug vs. other; False: all classes
    MAX_QUALITY = 2               # keep fields whose S2 bins are all <= this quality (0, 1 or 2)
    USE_QUALITY_AS_PREDICTOR = False
    N_FOLDS = 5
    SEED = 42
    
    PARAMS = dict(n_estimators=100, learning_rate=0.05, max_depth=6, subsample=0.8,
                  colsample_bytree=0.8, tree_method="hist", enable_categorical=True, n_jobs=-1)
    
    # ---------------------------------------------------------------------------
    X = pd.read_parquet(X_FILE)
    Y = pd.read_parquet(Y_FILE)
    assert (X["sample_id"] == Y["sample_id"]).all()
    
    s1 = [c for c in X.columns if c.startswith(("asc_", "desc_")) and not c.endswith("_orbit")]
    qual = [c for c in X.columns if c.startswith("S2_quality_")]
    s2 = [c for c in X.columns if c.startswith("S2_") and c not in qual]
    if USE_QUALITY_AS_PREDICTOR:
        s2 += qual
    
    keep = X[s1 + s2].notna().all(axis=1) & (X[qual] <= MAX_QUALITY).all(axis=1)
    X = X.loc[keep].reset_index(drop=True)
    Y = Y.loc[keep].reset_index(drop=True)
    X["lnf_code"] = X["lnf_code"].astype("category")
    
    SETUPS = {
        "1 crop": ["lnf_code"],
        "2 S1": s1,
        "3 S2": s2,
        "4 S1+S2": s1 + s2,
        "5 crop+S1+S2": ["lnf_code"] + s1 + s2,
    }
    
    if BINARY:
        target = np.where(Y["tillage"] == "Pflug", "Pflug", "Andere")
        classes = ["Andere", "Pflug"]
    else:
        target = Y["tillage"].to_numpy()
        classes = sorted(Y["tillage"].unique())
    y = pd.Series(target).map({c: i for i, c in enumerate(classes)}).to_numpy()
    
    print(f"{len(X):,} fields | S1: {len(s1)} predictors | S2: {len(s2)} predictors")
    print(pd.Series(target).value_counts().to_string(), "\n")
    
    obj = dict(objective="binary:logistic", eval_metric="logloss") if BINARY else \
        dict(objective="multi:softprob", eval_metric="mlogloss")
    folds = list(StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED).split(X, y))
    
    rows = []
    for name, cols in SETUPS.items():
        for k, (tr, te) in enumerate(folds, 1):
            m = XGBClassifier(**PARAMS, **obj, random_state=SEED + k)
            m.fit(X.loc[tr, cols], y[tr], sample_weight=compute_sample_weight("balanced", y[tr]))
            for part, idx in (("train", tr), ("test", te)):
                proba = m.predict_proba(X.loc[idx, cols])
                pred = proba.argmax(axis=1)
                auc = (roc_auc_score(y[idx], proba[:, 1]) if BINARY else
                       roc_auc_score(y[idx], proba, multi_class="ovr", average="macro"))
                rows.append(dict(setup=name, fold=k, part=part,
                                 f1_macro=f1_score(y[idx], pred, average="macro"),
                                 balanced_acc=balanced_accuracy_score(y[idx], pred),
                                 accuracy=accuracy_score(y[idx], pred),
                                 kappa=cohen_kappa_score(y[idx], pred),
                                 roc_auc=auc))
        r = pd.DataFrame(rows)
        r = r[(r["setup"] == name) & (r["part"] == SCORE_ON)]
        print(f"{name:14s} F1 macro {r.f1_macro.mean():.3f} ± {r.f1_macro.std():.3f} | "
              f"AUC {r.roc_auc.mean():.3f}")
    
    res = pd.DataFrame(rows)
    res.to_csv(OUT_CSV, index=False)
    res = res[res["part"] == SCORE_ON].drop(columns="part")
    summary = res.drop(columns="fold").groupby("setup").agg(["mean", "std"]).round(3)
    summary = res.drop(columns="fold").groupby("setup").agg(["mean", "std"]).round(3)
    res.to_csv(OUT_CSV, index=False)
    
    pd.set_option("display.width", 200)
    print(f"\nRandom {N_FOLDS}-fold CV, mean and std over folds "
          f"(chance: balanced acc / AUC {1 / len(classes):.2f} / 0.50):")
    print(summary.to_string())

    # --- plot: mean ± std per setup for F1 macro and AUC ---
    metrics = [("f1_macro", "F1 macro"), ("balanced_acc", "Balanced accuracy"), ("roc_auc", "ROC AUC")]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.5 * len(metrics), 4), sharey=False)
    names = list(SETUPS)
    colors = ["0.6", "tab:blue", "tab:green", "tab:purple", "tab:red"]
    for ax, (m, label) in zip(axes, metrics):
        mean = res.groupby("setup")[m].mean().reindex(names)
        std = res.groupby("setup")[m].std().reindex(names)
        ax.bar(range(len(names)), mean, yerr=std, color=colors, capsize=4)
        for i, v in enumerate(mean):
            ax.text(i, v + std.iloc[i] + 0.005, f"{v:.2f}", ha="center", fontsize=8)
        chance = 0.5 if m == "roc_auc" else (1 / len(classes) if m == "balanced_acc" else None)
        if chance:
            ax.axhline(chance, color="k", ls="--", lw=1)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n.split(" ", 1)[1] for n in names], rotation=30, ha="right")
        ax.set_title(label, fontsize=10)
        ax.set_ylim(max(0, mean.min() - 0.15), min(1, (mean + std).max() + 0.06))
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"Predictor setups, {len(X):,} fields, random {N_FOLDS}-fold CV, "
             f"{SCORE_ON} scores{' (Pflug vs. other)' if BINARY else ''}", fontsize=11)
    plt.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    plt.show()
    print(f"\nWritten: {OUT_CSV}, {OUT_PNG}")

def compare_s1_field_orientation():
    """
    Does field orientation improve the S1 backscatter tillage model?
    
    Two setups on exactly the same fields and folds:
      S1                 asc/desc VV, VH, CR in the biweekly bins
      S1 + orientation   + rel_asc_deg, rel_desc_deg, elongation (from add_orientation.py)
    
    Repeated stratified K-fold CV (N_REPEATS x N_FOLDS); scores are compared fold by
    fold (paired), so small but consistent gains become visible. Out-of-fold
    predictions are also scored for elongated fields only, where the orientation
    is meaningful.
    """    
    X_FILE, Y_FILE = "cfactor/data/all/X_ip.parquet", "cfactor/data/all/Y_ip.parquet"
    OUT_CSV = "cfactor/data/all/orientation_scores.csv"
    EXCLUDE_CODES = [601]
    # target grouping: three classes (None -> all four types; or Pflug / Andere for binary)
    MERGE = {"Pflug": "Pflug", "Mulchsaat": "Mulchsaat",
             "Streifensaat": "Streifen/Direkt", "Direktsaat": "Streifen/Direkt"}
    ORIENT_COLS = ["rel_asc_deg", "rel_desc_deg", "elongation"]
    ELONGATED = 1.5                 # subgroup: fields with long/short side >= this
    N_FOLDS, N_REPEATS, SEED = 5, 3, 42
    PARAMS = dict(n_estimators=400, learning_rate=0.05, max_depth=6, subsample=0.8,
                  colsample_bytree=0.8, tree_method="hist", n_jobs=-1)
    
    # ---------------------------------------------------------------------------
    X = pd.read_parquet(X_FILE)
    Y = pd.read_parquet(Y_FILE)
    assert (X["sample_id"] == Y["sample_id"]).all()
    missing = [c for c in ORIENT_COLS if c not in X.columns]
    if missing:
        raise SystemExit(f"{missing} not in X - run add_orientation.py first")
    
    drop = pd.to_numeric(X["lnf_code"], errors="coerce").isin(EXCLUDE_CODES)
    s1 = [c for c in X.columns if c.startswith(("asc_", "desc_")) and not c.endswith("_orbit")]
    keep = ~drop & X[s1].notna().all(axis=1) & X[ORIENT_COLS].notna().all(axis=1)
    X, Y = X.loc[keep].reset_index(drop=True), Y.loc[keep].reset_index(drop=True)
    
    target = Y["tillage"].map(MERGE) if MERGE else Y["tillage"]
    if target.isna().any():
        print(f"{target.isna().sum()} fields with tillage not in MERGE dropped: "
              f"{sorted(Y.loc[target.isna(), 'tillage'].unique())}")
        ok = target.notna().to_numpy()
        X, Y, target = X[ok].reset_index(drop=True), Y[ok].reset_index(drop=True), target[ok].reset_index(drop=True)
    classes = list(dict.fromkeys(MERGE.values())) if MERGE else sorted(target.unique())
    y = target.map({c: i for i, c in enumerate(classes)}).to_numpy()
    K = len(classes)
    obj = dict(objective="binary:logistic", eval_metric="logloss") if K == 2 else \
        dict(objective="multi:softprob", eval_metric="mlogloss")
    elong = X["elongation"].to_numpy() >= ELONGATED
    
    print(f"{len(X):,} fields ({elong.sum():,} elongated >= {ELONGATED}), {len(s1)} S1 predictors")
    print(pd.Series(target).value_counts().reindex(classes).to_string(), "\n")
    
    SETUPS = {"S1": s1, "S1 + orientation": s1 + ORIENT_COLS}
    
    
    def score(yt, proba):
        pred = proba.argmax(axis=1)
        auc = roc_auc_score(yt, proba[:, 1]) if K == 2 else \
            roc_auc_score(yt, proba, multi_class="ovr", average="macro", labels=range(K))
        return dict(f1_macro=f1_score(yt, pred, average="macro"),
                    balanced_acc=balanced_accuracy_score(yt, pred), roc_auc=auc)
    
    
    rows = []
    cv = RepeatedStratifiedKFold(n_splits=N_FOLDS, n_repeats=N_REPEATS, random_state=SEED)
    oof = {name: np.zeros((N_REPEATS, len(y), K)) for name in SETUPS}
    for f, (tr, te) in enumerate(cv.split(X, y)):
        rep = f // N_FOLDS
        for name, cols in SETUPS.items():
            m = XGBClassifier(**PARAMS, **obj, random_state=SEED + f)
            m.fit(X.loc[tr, cols], y[tr], sample_weight=compute_sample_weight("balanced", y[tr]))
            p = m.predict_proba(X.loc[te, cols])
            oof[name][rep, te] = p
            rows.append(dict(setup=name, fold=f, **score(y[te], p)))
        if (f + 1) % N_FOLDS == 0:
            print(f"repeat {rep + 1}/{N_REPEATS} done")
    
    res = pd.DataFrame(rows)
    res.to_csv(OUT_CSV, index=False)
    
    # --- overall, mean +- std over all folds ---
    pd.set_option("display.width", 200)
    print(f"\nAll fields, {N_REPEATS} x {N_FOLDS}-fold CV (chance: balanced acc {1 / K:.2f}, AUC 0.50):")
    print(res.drop(columns="fold").groupby("setup").agg(["mean", "std"]).round(3).to_string())
    
    # --- paired difference per fold ---
    wide = res.pivot(index="fold", columns="setup")
    print("\nGain from orientation (S1 + orientation minus S1), paired per fold:")
    for m in ("f1_macro", "balanced_acc", "roc_auc"):
        d = wide[(m, "S1 + orientation")] - wide[(m, "S1")]
        print(f"  {m:13s} mean {d.mean():+.4f}  std {d.std():.4f}  "
              f"better in {(d > 0).sum()}/{len(d)} folds")
    
    # --- elongated fields only (out-of-fold, averaged over repeats) ---
    print(f"\nOut-of-fold, elongated fields only (n={elong.sum():,}) vs. all fields:")
    sub = []
    for name in SETUPS:
        for label, mask in (("all", np.ones(len(y), bool)), (f"elongation >= {ELONGATED}", elong)):
            s = pd.DataFrame([score(y[mask], oof[name][r, mask]) for r in range(N_REPEATS)]).mean()
            sub.append(dict(setup=name, fields=label, **s.round(3)))
    print(pd.DataFrame(sub).to_string(index=False))
    
    # --- per class F1 ---
    print("\nF1 per class (out-of-fold, mean over repeats):")
    pc = {name: np.mean([f1_score(y, oof[name][r].argmax(axis=1), average=None, labels=range(K))
                         for r in range(N_REPEATS)], axis=0) for name in SETUPS}
    print(pd.DataFrame(pc, index=classes).round(3).to_string())
    print(f"\nWritten: {OUT_CSV}")


    
def plot_performance_per_ct():
    """
    Per-crop evaluation of the five predictor setups (crop, S1, S2, S1+S2,
    crop+S1+S2). Each setup is trained with the same random stratified K-fold
    splits on all fields; the out-of-fold test predictions are then scored
    separately within every crop.
    
    Within one crop the crop itself carries no information, so these scores show
    how well each sensor separates the tillage types for a given crop. Setup
    "1 crop" is therefore at chance level within every crop and serves as reference.
    """
    
    X_FILE = "data/all/X_ip.parquet"
    Y_FILE = "data/all/Y_ip.parquet"
    OUT_CSV = "data/all/setup_scores_per_crop_all_types.csv"
    OUT_PNG = "data/all/setup_scores_per_crop_all_types.png"
    BINARY = False
    EXCLUDE_CODES = [601]         # lnf_codes to leave out (601 = Kunstwiese)
    MAX_QUALITY = 2
    USE_QUALITY_AS_PREDICTOR = False
    N_FOLDS = 5
    SEED = 42
    MIN_FIELDS = 100              # crops with at least this many fields ...
    MIN_PER_CLASS = 10            # ... and at least this many fields per class
    
    PARAMS = dict(n_estimators=400, learning_rate=0.05, max_depth=6, subsample=0.8,
                  colsample_bytree=0.8, tree_method="hist", enable_categorical=True, n_jobs=-1)
    
    # ---------------------------------------------------------------------------
    X = pd.read_parquet(X_FILE)
    Y = pd.read_parquet(Y_FILE)
    assert (X["sample_id"] == Y["sample_id"]).all()
    
    drop = pd.to_numeric(X["lnf_code"], errors="coerce").isin(EXCLUDE_CODES)
    print(f"excluding {drop.sum():,} fields with lnf_code in {EXCLUDE_CODES}")
    X = X.loc[~drop].reset_index(drop=True)
    Y = Y.loc[~drop].reset_index(drop=True)
    
    s1 = [c for c in X.columns if c.startswith(("asc_", "desc_")) and not c.endswith("_orbit")]
    qual = [c for c in X.columns if c.startswith("S2_quality_")]
    s2 = [c for c in X.columns if c.startswith("S2_") and c not in qual]
    if USE_QUALITY_AS_PREDICTOR:
        s2 += qual
    
    keep = X[s1 + s2].notna().all(axis=1) & (X[qual] <= MAX_QUALITY).all(axis=1)
    X = X.loc[keep].reset_index(drop=True)
    Y = Y.loc[keep].reset_index(drop=True)
    X["lnf_code"] = X["lnf_code"].astype("category")
    
    SETUPS = {
        "1 crop": ["lnf_code"],
        "2 S1": s1,
        "3 S2": s2,
        "4 S1+S2": s1 + s2,
        "5 crop+S1+S2": ["lnf_code"] + s1 + s2,
    }
    
    if BINARY:
        target = np.where(Y["tillage"] == "Pflug", "Pflug", "Andere")
        classes = ["Andere", "Pflug"]
    else:
        target = Y["tillage"].to_numpy()
        classes = sorted(Y["tillage"].unique())
    y = pd.Series(target).map({c: i for i, c in enumerate(classes)}).to_numpy()
    K = len(classes)
    crop = Y["Crop_DE"].fillna("unknown").str.slice(0, 40).to_numpy()
    
    obj = dict(objective="binary:logistic", eval_metric="logloss") if BINARY else \
        dict(objective="multi:softprob", eval_metric="mlogloss")
    folds = list(StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED).split(X, y))
    
    # --- out-of-fold predictions per setup ---
    oof = {}
    for name, cols in SETUPS.items():
        proba = np.zeros((len(y), K))
        for k, (tr, te) in enumerate(folds, 1):
            m = XGBClassifier(**PARAMS, **obj, random_state=SEED + k)
            m.fit(X.loc[tr, cols], y[tr], sample_weight=compute_sample_weight("balanced", y[tr]))
            proba[te] = m.predict_proba(X.loc[te, cols])
        oof[name] = proba
        print(f"{name} done")
    
    
    def scores(yt, p):
        pred = p.argmax(axis=1)
        if len(np.unique(yt)) < 2:
            auc = np.nan
        elif BINARY:
            auc = roc_auc_score(yt, p[:, 1])
        else:
            auc = roc_auc_score(yt, p, multi_class="ovr", average="macro", labels=range(K))
        return dict(f1_macro=f1_score(yt, pred, average="macro"),
                    balanced_acc=balanced_accuracy_score(yt, pred), roc_auc=auc)
    
    
    # --- per crop ---
    counts = pd.crosstab(crop, y)
    eligible = counts[(counts.sum(axis=1) >= MIN_FIELDS) & (counts.min(axis=1) >= MIN_PER_CLASS)
                      & (counts.shape[1] == K)].index
    eligible = sorted(eligible, key=lambda c: -counts.loc[c].sum())
    print(f"\n{len(eligible)} crops with >= {MIN_FIELDS} fields and >= {MIN_PER_CLASS} per class")
    
    rows = []
    for c in ["all fields"] + eligible:
        idx = np.arange(len(y)) if c == "all fields" else np.flatnonzero(crop == c)
        for name, p in oof.items():
            row = dict(crop=c, setup=name, n=len(idx))
            if BINARY:
                row["share_pflug"] = round(y[idx].mean(), 2)
            row.update(scores(y[idx], p[idx]))
            rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(OUT_CSV, index=False)
    
    pd.set_option("display.width", 200)
    for metric in ("roc_auc", "f1_macro"):
        tab = res.pivot(index="crop", columns="setup", values=metric).reindex(["all fields"] + eligible)
        tab.insert(0, "n", res.drop_duplicates("crop").set_index("crop")["n"])
        print(f"\n{metric} per crop (out-of-fold, random {N_FOLDS}-fold CV):")
        print(tab.round(3).to_string())
    
    # --- plot: AUC per crop and setup ---
    setups = list(SETUPS)

    SENSOR_COLOR = {"none": "0.55", "S1": "tab:blue", "S2": "tab:red", "S1+S2": "tab:purple"}
    CROP_MARKER = {False: "o", True: "D"}
    
    def style(setup):
        cols = SETUPS[setup]
        has_s1 = any(c in s1 for c in cols)
        has_s2 = any(c in s2 for c in cols)
        sensor = "S1+S2" if has_s1 and has_s2 else "S1" if has_s1 else "S2" if has_s2 else "none"
        return SENSOR_COLOR[sensor], CROP_MARKER["lnf_code" in cols]
    
    colors = {s: style(s)[0] for s in setups}
    markers = {s: style(s)[1] for s in setups}
    
    crops_plot = ["all fields"] + eligible
    labels = []
    for c in crops_plot:
        r = res[(res["crop"] == c)].iloc[0]
        extra = f", {100 * r['share_pflug']:.0f}% Pflug" if BINARY else ""
        labels.append(f"{c[:32]} (n={r['n']:,}{extra})")
    
    fig, ax = plt.subplots(figsize=(10, 0.5 * len(crops_plot) + 1.8))
    ypos = np.arange(len(crops_plot))[::-1]
    ax.axvline(0.5, color="k", ls="--", lw=1)
    for c, yp in zip(crops_plot, ypos):
        v = res[res["crop"] == c].set_index("setup").reindex(setups)["roc_auc"]
        ax.hlines(yp, v.min(), v.max(), color="0.85", lw=4, zorder=1)
    for i, s in enumerate(setups):
        v = res[res["setup"] == s].set_index("crop").reindex(crops_plot)["roc_auc"]
        ax.scatter(v, ypos + (i - 2) * 0.08, color=colors[s], marker=markers[s], s=45,
                   zorder=3, label=s)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.axhline(ypos[0] - 0.5, color="0.5", lw=0.8)        # separate "all fields"
    ax.set_xlabel("ROC AUC  (0.5 = chance)")
    ax.set_title("Tillage classification per crop and predictor setup (out-of-fold test scores)",
                 fontsize=11)
    ax.grid(axis="x", alpha=0.3)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    plt.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    print(f"\nWritten: {OUT_CSV}, {OUT_PNG}")

def conf_matrix_till():   
    X_FILE = "data/all/X_ip.parquet"
    Y_FILE = "data/all/Y_ip.parquet"
    EXCLUDE_CODES = [601]
    MAX_QUALITY = 2
    N_FOLDS, SEED = 10, 42
    SETUPS_TO_PLOT = ["2 S1", "3 S2", "4 S1+S2"]   # or e.g. ["5 crop+S1+S2"]
    CLASS_ORDER = ["Pflug", "Mulchsaat", "Streifensaat", "Direktsaat"]
    PARAMS = dict(n_estimators=100, learning_rate=0.05, max_depth=6, subsample=0.8,
                  colsample_bytree=0.8, tree_method="hist", enable_categorical=True, n_jobs=-1,
                  objective="multi:softprob", eval_metric="mlogloss")
    
    X = pd.read_parquet(X_FILE)
    Y = pd.read_parquet(Y_FILE)
    drop = pd.to_numeric(X["lnf_code"], errors="coerce").isin(EXCLUDE_CODES)
    X, Y = X.loc[~drop].reset_index(drop=True), Y.loc[~drop].reset_index(drop=True)
    
    s1 = [c for c in X.columns if c.startswith(("asc_", "desc_")) and not c.endswith("_orbit")]
    qual = [c for c in X.columns if c.startswith("S2_quality_")]
    s2 = [c for c in X.columns if c.startswith("S2_") and c not in qual]
    keep = X[s1 + s2].notna().all(axis=1) & (X[qual] <= MAX_QUALITY).all(axis=1)
    X, Y = X.loc[keep].reset_index(drop=True), Y.loc[keep].reset_index(drop=True)
    X["lnf_code"] = X["lnf_code"].astype("category")
    
    SETUPS = {"1 crop": ["lnf_code"], "2 S1": s1, "3 S2": s2,
              "4 S1+S2": s1 + s2, "5 crop+S1+S2": ["lnf_code"] + s1 + s2}
    merge = {"Pflug": "Pflug", "Mulchsaat": "Mulchsaat",
             "Streifensaat": "Streifen/Direkt", "Direktsaat": "Streifen/Direkt"}
    classes = ["Pflug", "Mulchsaat", "Streifen/Direkt"]
    y = Y["tillage"].map(merge).map({c: i for i, c in enumerate(classes)}).to_numpy()
    folds = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(X, y))
    print(f"{len(X):,} fields:", Y["tillage"].value_counts().reindex(classes).to_dict())
    
    fig, axes = plt.subplots(1, len(SETUPS_TO_PLOT), figsize=(4.2 * len(SETUPS_TO_PLOT), 4.4),
                             squeeze=False)
    for ax, name in zip(axes[0], SETUPS_TO_PLOT):
        cols = SETUPS[name]
        pred = np.zeros(len(y), dtype=int)
        for k, (tr, te) in enumerate(folds, 1):
            m = XGBClassifier(**PARAMS, random_state=SEED + k)
            m.fit(X.loc[tr, cols], y[tr], sample_weight=compute_sample_weight("balanced", y[tr]))
            pred[te] = m.predict(X.loc[te, cols])
    
        cm = confusion_matrix(y, pred, labels=range(len(classes)))
        cm_norm = cm / cm.sum(axis=1, keepdims=True)
        annot = np.array([[f"{p:.2f}\n({n:,})" for p, n in zip(rp, rn)]
                          for rp, rn in zip(cm_norm, cm)])
        sns.heatmap(cm_norm, annot=annot, fmt="", cmap="Blues", vmin=0, vmax=1, cbar=False,
                    xticklabels=classes, yticklabels=classes, ax=ax, annot_kws={"fontsize": 8})
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True" if ax is axes[0][0] else "")
        ax.set_title(f"{name}\nF1 macro {f1_score(y, pred, average='macro'):.2f} | "
                     f"bal. acc {balanced_accuracy_score(y, pred):.2f}", fontsize=10)
        ax.tick_params(axis="x", rotation=30)
    
    fig.suptitle(f"Confusion matrices, all tillage types, out-of-fold (random {N_FOLDS}-fold CV), "
                 f"{len(X):,} fields", fontsize=11)
    plt.tight_layout()
    fig.savefig("data/all/figs/confusion_3types.png", dpi=150)

def plot_fi_till():
    X_FILE = "data/all/X_ip.parquet"
    Y_FILE = "data/all/Y_ip.parquet"
    EXCLUDE_CODES = [601]
    MAX_QUALITY = 2
    N_FOLDS, SEED = 5, 42
    TOP_N = 25
    # target grouping (None -> all four types)
    MERGE = {"Pflug": "Pflug", "Mulchsaat": "Mulchsaat",
             "Streifensaat": "Streifen/Direkt", "Direktsaat": "Streifen/Direkt"}
    PARAMS = dict(n_estimators=400, learning_rate=0.05, max_depth=6, subsample=0.8,
                  colsample_bytree=0.8, tree_method="hist", enable_categorical=True, n_jobs=-1,
                  objective="multi:softprob", eval_metric="mlogloss")
    
    # --- data, as in the setup comparison ---
    X = pd.read_parquet(X_FILE)
    Y = pd.read_parquet(Y_FILE)
    drop = pd.to_numeric(X["lnf_code"], errors="coerce").isin(EXCLUDE_CODES)
    X, Y = X.loc[~drop].reset_index(drop=True), Y.loc[~drop].reset_index(drop=True)
    s1 = [c for c in X.columns if c.startswith(("asc_", "desc_")) and not c.endswith("_orbit")]
    qual = [c for c in X.columns if c.startswith("S2_quality_")]
    s2 = [c for c in X.columns if c.startswith("S2_") and c not in qual]
    keep = X[s1 + s2].notna().all(axis=1) & (X[qual] <= MAX_QUALITY).all(axis=1)
    X, Y = X.loc[keep].reset_index(drop=True), Y.loc[keep].reset_index(drop=True)
    X["lnf_code"] = X["lnf_code"].astype("category")
    
    features = ["lnf_code"] + s1 + s2
    target = Y["tillage"].map(MERGE) if MERGE else Y["tillage"]
    classes = list(dict.fromkeys(MERGE.values())) if MERGE else sorted(target.unique())
    y = target.map({c: i for i, c in enumerate(classes)}).to_numpy()
    print(f"{len(X):,} fields, {len(features)} predictors, classes:",
          pd.Series(target).value_counts().reindex(classes).to_dict())
    
    # --- gain importance per fold, normalised to sum 1 ---
    imp = []
    folds = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(X, y)
    for k, (tr, te) in enumerate(folds, 1):
        m = XGBClassifier(**PARAMS, random_state=SEED + k)
        m.fit(X.loc[tr, features], y[tr], sample_weight=compute_sample_weight("balanced", y[tr]))
        gain = m.get_booster().get_score(importance_type="gain")
        imp.append(pd.Series({f: gain.get(f, 0.0) for f in features}))
    imp = pd.DataFrame(imp)
    imp = imp.div(imp.sum(axis=1), axis=0)
    
    
    # --- grouping: variable (without time bin) and time bin ---
    def variable(f):
        return "crop (lnf_code)" if f == "lnf_code" else f.rsplit("_", 1)[0]
    
    
    def time_bin(f):
        return "—" if f == "lnf_code" else f.rsplit("_", 1)[1]
    
    
    def sensor(f):
        return "crop" if f == "lnf_code" else ("S1" if f.startswith(("asc_", "desc_")) else "S2")
    
    
    by_var = imp.T.groupby(variable).sum().T
    by_bin = imp.T.groupby(time_bin).sum().T
    bin_order = [b for b in ["t0", "tp2", "tp4", "tp6", "tp8"] if b in by_bin.columns]
    colors = {"crop": "0.55", "S1": "tab:blue", "S2": "tab:red"}
    
    fig, axes = plt.subplots(1, 3, figsize=(17, 0.28 * TOP_N + 2.5),
                             gridspec_kw={"width_ratios": [1.3, 1.1, 0.8]})
    
    # 1: top single features
    top = imp.mean().sort_values(ascending=False).index[:TOP_N]
    sns.boxplot(data=imp[top], orient="h", showfliers=False, ax=axes[0],
                palette=[colors[sensor(f)] for f in top])
    axes[0].set_title(f"Top {TOP_N} single features", fontsize=10)
    axes[0].set_xlabel("Relative importance (gain)")
    
    # 2: per variable, summed over time bins
    var_order = by_var.mean().sort_values(ascending=False).index
    sns.boxplot(data=by_var[var_order], orient="h", showfliers=False, ax=axes[1],
                palette=[colors["crop" if v.startswith("crop") else
                                "S1" if v.startswith(("asc_", "desc_")) else "S2"] for v in var_order])
    axes[1].set_title("Per variable (all time bins)", fontsize=10)
    axes[1].set_xlabel("Relative importance (gain)")
    
    # 3: per time bin, summed over variables (crop excluded)
    sns.boxplot(data=by_bin[bin_order], showfliers=False, color="tab:purple", ax=axes[2])
    axes[2].set_title("Per time bin (all S1/S2 variables)", fontsize=10)
    axes[2].set_ylabel("Relative importance (gain)")
    axes[2].set_xlabel("Two-week bin relative to tillage date")
    
    for ax in axes:
        ax.grid(axis="x" if ax is not axes[2] else "y", alpha=0.3)
    axes[1].legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()],
                   labels=list(colors), fontsize=8, frameon=False, loc="lower right")
    fig.suptitle(f"Feature importance, crop + S1 + S2, {len(classes)} classes "
                 f"({', '.join(classes)}), {N_FOLDS} folds", fontsize=11)
    plt.tight_layout()
    fig.savefig("data/all/figs/feature_importance_crop_s1_s2.png", dpi=150)
    
    print("\nImportance per sensor (mean over folds):")
    print(imp.T.groupby(sensor).sum().T.mean().round(3).sort_values(ascending=False).to_string())

if __name__ == "__main__":
    start = datetime.now()  
    # compare_s1_field_orientation()
    # get_approx_till_time()
    # plot_trajectories_tilltype()
    # prepare_xy()
    # compare_s1_s2()
    # conf_matrix_till()
    
    end = datetime.now()
    print(f'Calculation done at {end}. Calculation took {end-start}')


