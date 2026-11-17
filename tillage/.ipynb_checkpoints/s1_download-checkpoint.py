"""
Sentinel-1 interferometric coherence per field via ASF HyP3 (burst InSAR, ISCE2).

Stages (run one after the other; each stage reads the output of the previous one):

  python s1_coherence_hyp3.py sample     # fields per crop and class, clustered in few bursts
  python s1_coherence_hyp3.py search     # all acquisitions of these bursts, build pairs
  python s1_coherence_hyp3.py submit     # dry run: number of jobs and credits
  python s1_coherence_hyp3.py submit --go --limit 5   # small test run first
  python s1_coherence_hyp3.py submit --go    # submit the remaining jobs
  python s1_coherence_hyp3.py download   # download finished products (coherence only)
  python s1_coherence_hyp3.py extract    # coherence statistics per field -> parquet

Cost scales with bursts x pairs, not with fields: one job (burst pair) gives
coherence for every field inside the burst. The sampling therefore picks fields
that meet the quotas per crop and tillage class with as few bursts as possible.

Pairs: consecutive acquisitions of the same burst, with temporal baselines in
BASELINES_DAYS (6 = S1A/S1B pairs, 12 = same-satellite pairs). Platform of both
acquisitions is stored, so 6- and 12-day coherence can be compared later.

Authentication: NASA Earthdata login. Put it in ~/.netrc as
  machine urs.earthdata.nasa.gov login <user> password <password>
or the HyP3 client will ask for it.

HyP3 products are deleted after about two weeks: run "download" in time.
Requires: asf_search, hyp3_sdk, geopandas, rasterio, shapely>=2, pandas, pyarrow
"""

import argparse
import glob
import os
import zipfile

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from datetime import datetime

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
FIELDS_SHP = "cfactor/data/all/sampled_fields_2021.shp"   # sampled_fields_2021.shp
FIELD_ID_COL = "field_id"

# crop: lnf_code per field from the gpkg (id = field_id)
CROP_GPKG = "cfactor/data/all/lnf_2021_tillage.gpkg"
CROP_GPKG_ID = "id"
CROP_CODES = {513: "Winterweizen", 502: "Wintergerste", 521: "Silo- und Grünmais",
              527: "Winterraps (Speiseöl)", 522: "Zuckerrüben"}
N_PER_CROP = 200
CLASS_SHARE = {"Pflug": 0.50, "Mulchsaat": 0.25, "Streifen/Direkt": 0.25}
CLASSES = {"Pflug": "pflug", "Mulchsaat": "mulch",
           "Streifen/Direkt": "streifen|direkt"}

# optional canton filter (column in the gpkg); None -> all of Switzerland
KANTON = None
KANTON_COL = "kanton"
SEED = 42

START, END = "2020-07-01", "2021-06-30"
POLARIZATION = "VV"
ORBITS = {"ASCENDING": None, "DESCENDING": None}   # None -> any orbit; or e.g. 15 to restrict
BASELINES_DAYS = [6]          # add 12 for same-satellite (S1A-S1A / S1B-S1B) pairs
BASELINE_TOL = 1              # days
LOOKS = "5x1"                 # "20x4" (~80 m), "10x2" (~40 m), "5x1" (~20 m); finer costs more
JOB_NAME = "tillage_coh_2021"

NETRC_FILE = "../../.netrc"

INNER_BUFFER_M = 10
METRIC_CRS = "EPSG:2056"

WORK_DIR = "tillage/coherence"
F_FIELDS = os.path.join(WORK_DIR, "fields_sample.gpkg")
F_BURSTS = os.path.join(WORK_DIR, "bursts.parquet")
F_FIELD_BURST = os.path.join(WORK_DIR, "field_burst.csv")
F_PAIRS = os.path.join(WORK_DIR, "pairs.csv")
F_JOBS = os.path.join(WORK_DIR, "jobs.csv")
COH_DIR = os.path.join(WORK_DIR, "coherence_tifs")
F_OUT = os.path.join(WORK_DIR, "coherence_long.parquet")


def norm_id(s):
    return s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


# ---------------------------------------------------------------------------
# 1. sample: fields per crop and class, clustered in as few bursts as possible
# ---------------------------------------------------------------------------
def _search(aoi_wkt, start, end, direction=None, orbit=None):
    import asf_search as asf
    kw = dict(dataset=asf.DATASET.SLC_BURST, intersectsWith=aoi_wkt, start=start, end=end,
              polarization=POLARIZATION)
    if direction:
        kw["flightDirection"] = direction
    if orbit:
        kw["relativeOrbit"] = orbit
    res = asf.search(**kw)
    rows = []
    for r in res:
        p = r.properties
        rows.append(dict(
            granule=p["sceneName"], burst_id=p["burst"]["fullBurstID"],
            time=pd.Timestamp(p["startTime"]), platform=p["platform"],
            direction=p["flightDirection"], orbit=int(p["pathNumber"]),
            geometry=shapely.geometry.shape(r.geometry),
        ))
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326") if rows else None


def _aoi_wkt(gdf_ll):
    aoi = shapely.convex_hull(shapely.union_all(gdf_ll.geometry.values)).buffer(0.01)
    return shapely.to_wkt(shapely.simplify(aoi, 0.005), rounding_precision=4)


def stage_sample():
    import pyogrio
    os.makedirs(WORK_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    f = gpd.read_file(FIELDS_SHP)
    f["field_id"] = norm_id(f[FIELD_ID_COL])
    cols = [CROP_GPKG_ID, "lnf_code"] + ([KANTON_COL] if KANTON else [])
    g = pyogrio.read_dataframe(CROP_GPKG, columns=cols, read_geometry=False)
    g["field_id"] = norm_id(g[CROP_GPKG_ID])
    f = f.merge(g.drop(columns=CROP_GPKG_ID), on="field_id", how="left")
    f["lnf_code"] = pd.to_numeric(f["lnf_code"], errors="coerce")
    f = f[f["lnf_code"].isin(list(CROP_CODES))]
    if KANTON:
        f = f[f[KANTON_COL].astype(str).str.strip() == KANTON]
    f["crop"] = f["lnf_code"].astype(int).map(CROP_CODES)
    f["group"] = pd.Series(pd.NA, index=f.index, dtype="object")
    for label, pat in CLASSES.items():
        f.loc[f["tillage"].str.contains(pat, case=False, na=False) & f["group"].isna(),
              "group"] = label
    f = f[f["group"].notna()].copy()
    print("Candidate fields per crop and class:")
    print(pd.crosstab(f["crop"], f["group"]))

    # --- burst footprints (two weeks are enough to see every burst once) ---
    buf = f.to_crs(METRIC_CRS).buffer(-INNER_BUFFER_M).to_crs("EPSG:4326")
    cand = gpd.GeoDataFrame(f[["field_id", "crop", "group"]], geometry=buf.values, crs="EPSG:4326")
    cand = cand[~cand.is_empty]
    probe = _search(_aoi_wkt(cand), START,
                    (pd.Timestamp(START) + pd.Timedelta(days=13)).date().isoformat())
    if probe is None:
        raise SystemExit("No bursts found.")
    for d, o in ORBITS.items():
        if o:
            probe = probe[(probe["direction"] != d) | (probe["orbit"] == o)]
    foot = probe.drop_duplicates("burst_id")[["burst_id", "direction", "orbit", "geometry"]]
    hit = gpd.sjoin(cand, foot, predicate="within")[["field_id", "burst_id", "direction"]]

    asc = hit[hit["direction"] == "ASCENDING"].groupby("field_id")["burst_id"].apply(list)
    desc = hit[hit["direction"] == "DESCENDING"].groupby("field_id")["burst_id"].apply(list)
    cand = cand[cand["field_id"].isin(asc.index) & cand["field_id"].isin(desc.index)]
    print(f"\n{len(cand):,} fields fully inside an ascending and a descending burst")

    # cells = (asc burst, desc burst) combinations; a field can lie in several
    rows = [(fid, a, d) for fid in cand["field_id"] for a in asc[fid] for d in desc[fid]]
    cells = pd.DataFrame(rows, columns=["field_id", "asc", "desc"]).merge(
        cand[["field_id", "crop", "group"]], on="field_id")

    # --- quotas per crop and class ---
    quota = {(c, gr): int(round(N_PER_CROP * sh)) for c in CROP_CODES.values()
             for gr, sh in CLASS_SHARE.items()}

    # --- greedy: add the cell with most useful fields per newly needed burst ---
    selected_bursts, chosen = set(), {}
    while True:
        open_q = {k: v for k, v in quota.items() if v > 0}
        if not open_q:
            break
        avail = cells[~cells["field_id"].isin(chosen)
                      & cells.set_index(["crop", "group"]).index.isin(list(open_q))]
        if avail.empty:
            break
        n = avail.groupby(["asc", "desc", "crop", "group"])["field_id"].nunique().reset_index()
        n["useful"] = [min(c, open_q[(cr, gr)]) for c, cr, gr in
                       zip(n["field_id"], n["crop"], n["group"])]
        score = n.groupby(["asc", "desc"])["useful"].sum().reset_index()
        score["cost"] = [(a not in selected_bursts) + (d not in selected_bursts)
                         for a, d in zip(score["asc"], score["desc"])]
        score["gain"] = score["useful"] / score["cost"].clip(lower=0.5)
        best = score.sort_values(["gain", "useful"], ascending=False).iloc[0]
        a, d = best["asc"], best["desc"]
        selected_bursts |= {a, d}
        cell = avail[(avail["asc"] == a) & (avail["desc"] == d)].drop_duplicates("field_id")
        for (cr, gr), grp in cell.groupby(["crop", "group"]):
            k = min(quota.get((cr, gr), 0), len(grp))
            for fid in rng.permutation(grp["field_id"].to_numpy())[:k]:
                chosen[fid] = (a, d)
            quota[(cr, gr)] -= k

    sel = f[f["field_id"].isin(chosen)].copy()
    sel["asc_burst"] = sel["field_id"].map(lambda x: chosen[x][0])
    sel["desc_burst"] = sel["field_id"].map(lambda x: chosen[x][1])
    sel.to_file(F_FIELDS, driver="GPKG")

    print("\nSelected fields per crop and class (target in brackets):")
    tab = pd.crosstab(sel["crop"], sel["group"]).reindex(columns=list(CLASS_SHARE), fill_value=0)
    print(tab.astype(str) + " (" + pd.DataFrame(
        {gr: [int(round(N_PER_CROP * CLASS_SHARE[gr]))] * len(tab) for gr in CLASS_SHARE},
        index=tab.index).astype(str) + ")")
    short = {k: v for k, v in quota.items() if v > 0}
    if short:
        print("Not filled (not enough fields available):", short)
    n_b = {d: sel[f"{d}_burst"].nunique() for d in ("asc", "desc")}
    pairs_per_burst = sum(365 // b for b in BASELINES_DAYS)
    print(f"Bursts: {n_b['asc']} ascending, {n_b['desc']} descending -> "
          f"about {(n_b['asc'] + n_b['desc']) * pairs_per_burst} jobs "
          f"({pairs_per_burst} pairs per burst)")
    print(f"Written: {F_FIELDS}")


# ---------------------------------------------------------------------------
# 2. search: all acquisitions of the selected bursts and the pairs
# ---------------------------------------------------------------------------
def stage_search():
    sel = gpd.read_file(F_FIELDS)
    sel["field_id"] = norm_id(sel["field_id"])
    aoi = _aoi_wkt(sel.to_crs("EPSG:4326"))
    fb = pd.concat([
        pd.DataFrame({"field_id": sel["field_id"], "direction": "ASCENDING",
                      "burst_id": sel["asc_burst"]}),
        pd.DataFrame({"field_id": sel["field_id"], "direction": "DESCENDING",
                      "burst_id": sel["desc_burst"]}),
    ])
    fb["orbit"] = fb["burst_id"].str.split("_").str[0].astype(int)
    fb.to_csv(F_FIELD_BURST, index=False)
    needed = set(fb["burst_id"])

    parts = []
    for (d, o), _ in fb.groupby(["direction", "orbit"]):
        r = _search(aoi, START, END, direction=d, orbit=int(o))
        if r is not None:
            parts.append(r[r["burst_id"].isin(needed)])
    bursts = pd.concat(parts, ignore_index=True).drop_duplicates("granule")
    pd.DataFrame(bursts.drop(columns="geometry")).to_parquet(F_BURSTS, index=False)

    pairs = []
    for bid, g in bursts.sort_values("time").groupby("burst_id"):
        t = g["time"].dt.normalize().to_numpy()
        rec = g.to_dict("records")
        for i in range(len(rec)):
            for j in range(i + 1, len(rec)):
                dt = (t[j] - t[i]) / np.timedelta64(1, "D")
                if dt > max(BASELINES_DAYS) + BASELINE_TOL:
                    break
                for b in BASELINES_DAYS:
                    if abs(dt - b) <= BASELINE_TOL:
                        pairs.append(dict(
                            burst_id=bid, direction=rec[i]["direction"], orbit=rec[i]["orbit"],
                            ref_granule=rec[i]["granule"], sec_granule=rec[j]["granule"],
                            ref_date=rec[i]["time"].date(), sec_date=rec[j]["time"].date(),
                            ref_platform=rec[i]["platform"], sec_platform=rec[j]["platform"],
                            baseline_days=int(round(dt)), nominal_baseline=b))
    pairs = pd.DataFrame(pairs).drop_duplicates(["ref_granule", "sec_granule"])
    pairs.insert(0, "pair_id", [f"p{i:05d}" for i in range(len(pairs))])
    pairs.to_csv(F_PAIRS, index=False)

    print(f"{len(needed)} bursts, {len(bursts)} burst acquisitions, {len(pairs)} pairs:")
    print(pairs.groupby(["direction", "nominal_baseline"]).size().to_string())
    print(f"Written: {F_BURSTS}, {F_FIELD_BURST}, {F_PAIRS}")


# ---------------------------------------------------------------------------
# 3. submit
# ---------------------------------------------------------------------------
def _hyp3():
    import netrc
    import hyp3_sdk as sdk
    auth = netrc.netrc(NETRC_FILE).authenticators("urs.earthdata.nasa.gov")
    if auth is None:
        raise SystemExit(f"No entry for urs.earthdata.nasa.gov in {NETRC_FILE}")
    username, _, password = auth
    return sdk.HyP3(username=username, password=password)


def _credits(hyp3):
    for fn in ("check_credits", "check_quota"):
        if hasattr(hyp3, fn):
            try:
                return getattr(hyp3, fn)()
            except Exception:
                pass
    return "unknown"


def stage_submit(go=False, limit=None):
    pairs = pd.read_csv(F_PAIRS)
    done = pd.read_csv(F_JOBS) if os.path.exists(F_JOBS) else pd.DataFrame(columns=["pair_id", "job_id"])
    todo = pairs[~pairs["pair_id"].isin(done["pair_id"])]
    if limit:
        todo = todo.head(limit)
    hyp3 = _hyp3()
    print(f"{len(pairs)} pairs, {len(done)} already submitted, {len(todo)} to submit "
          f"(looks {LOOKS}). Remaining credits/quota: {_credits(hyp3)}")
    if not go:
        print("Dry run. Check the numbers above, then run: submit --go")
        return

    new = []
    try:
        for i, r in enumerate(todo.itertuples(), 1):
            batch = hyp3.submit_insar_isce_burst_job(
                r.ref_granule, r.sec_granule, name=JOB_NAME, looks=LOOKS, apply_water_mask=False)
            new.append(dict(pair_id=r.pair_id, job_id=batch.jobs[0].job_id))
            if i % 50 == 0:
                print(f"  {i}/{len(todo)} submitted")
    finally:                       # keep track even if the loop stops halfway
        if new:
            pd.concat([done, pd.DataFrame(new)]).to_csv(F_JOBS, index=False)
    print(f"Submitted {len(new)} jobs. Written: {F_JOBS}")


# ---------------------------------------------------------------------------
# 4. download
# ---------------------------------------------------------------------------
def stage_download():
    os.makedirs(COH_DIR, exist_ok=True)
    jobs = pd.read_csv(F_JOBS)
    hyp3 = _hyp3()
    batch = hyp3.find_jobs(name=JOB_NAME)
    status = pd.Series({j.job_id: j.status_code for j in batch.jobs})
    print("Job status:", status.reindex(jobs["job_id"]).value_counts(dropna=False).to_dict())

    have = {os.path.basename(p).split("_corr")[0] for p in glob.glob(os.path.join(COH_DIR, "*_corr.tif"))}
    n_new = 0
    for j in batch.jobs:
        if j.job_id not in set(jobs["job_id"]) or j.status_code != "SUCCEEDED" or j.job_id in have:
            continue
        zip_path = j.download_files(WORK_DIR)[0]
        with zipfile.ZipFile(zip_path) as z:
            corr = [n for n in z.namelist() if n.endswith("_corr.tif")]
            if not corr:
                print(f"  {j.job_id}: no *_corr.tif in product")
                continue
            with z.open(corr[0]) as src, open(os.path.join(COH_DIR, f"{j.job_id}_corr.tif"), "wb") as dst:
                dst.write(src.read())
        os.remove(zip_path)
        n_new += 1
        if n_new % 20 == 0:
            print(f"  {n_new} downloaded")
    print(f"Downloaded {n_new} new coherence rasters to {COH_DIR}")


# ---------------------------------------------------------------------------
# 5. extract
# ---------------------------------------------------------------------------
def stage_extract():
    import rasterio
    from rasterio.windows import from_bounds

    fields = gpd.read_file(F_FIELDS)
    fields["field_id"] = norm_id(fields[FIELD_ID_COL])
    fields = fields.to_crs(METRIC_CRS)
    fields["geometry"] = fields.buffer(-INNER_BUFFER_M)
    fields = fields[~fields.is_empty]

    fb = pd.read_csv(F_FIELD_BURST, dtype={"field_id": str})
    pairs = pd.read_csv(F_PAIRS)
    jobs = pd.read_csv(F_JOBS)
    pairs = pairs.merge(jobs, on="pair_id")

    rows, cache = [], {}
    tifs = {os.path.basename(p).split("_corr")[0]: p
            for p in glob.glob(os.path.join(COH_DIR, "*_corr.tif"))}
    pairs = pairs[pairs["job_id"].isin(tifs)]
    print(f"{len(pairs)} pairs with a downloaded coherence raster")

    for i, pr in enumerate(pairs.itertuples(), 1):
        ids = fb.loc[(fb["burst_id"] == pr.burst_id), "field_id"]
        sub = fields[fields["field_id"].isin(ids)]
        if sub.empty:
            continue
        with rasterio.open(tifs[pr.job_id]) as src:
            crs = src.crs.to_string()
            if crs not in cache:
                cache[crs] = fields.to_crs(crs).set_index("field_id").geometry
            geoms = cache[crs].loc[sub["field_id"]]
            nodata = src.nodata
            for fid, g in geoms.items():
                try:
                    win = from_bounds(*g.bounds, transform=src.transform).round_offsets().round_lengths()
                    a = src.read(1, window=win, boundless=True, fill_value=np.nan).astype("float32")
                except Exception:
                    continue
                if a.size == 0:
                    continue
                tr = src.window_transform(win)
                cols, rws = np.meshgrid(np.arange(a.shape[1]) + 0.5, np.arange(a.shape[0]) + 0.5)
                xs, ys = tr * (cols, rws)
                inside = shapely.contains_xy(g, xs, ys)
                v = a[inside]
                n_inside = int(inside.sum())
                ok = np.isfinite(v) & (v > 0) & (v <= 1)
                if nodata is not None:
                    ok &= v != nodata
                v = v[ok]
                rows.append(dict(
                    field_id=fid, pair_id=pr.pair_id, burst_id=pr.burst_id,
                    direction=pr.direction, orbit=pr.orbit,
                    ref_date=pr.ref_date, sec_date=pr.sec_date,
                    mid_date=(pd.Timestamp(pr.ref_date)
                              + (pd.Timestamp(pr.sec_date) - pd.Timestamp(pr.ref_date)) / 2).date(),
                    baseline_days=pr.baseline_days,
                    ref_platform=pr.ref_platform, sec_platform=pr.sec_platform,
                    n_inside=n_inside, n_valid=int(len(v)),
                    coh_mean=float(v.mean()) if len(v) else np.nan,
                    coh_median=float(np.median(v)) if len(v) else np.nan,
                    coh_std=float(v.std()) if len(v) > 1 else np.nan,
                ))
        if i % 50 == 0:
            print(f"  {i}/{len(pairs)} rasters")

    out = pd.DataFrame(rows)
    meta = gpd.read_file(F_FIELDS)
    meta["field_id"] = norm_id(meta[FIELD_ID_COL])
    keep = [c for c in ("field_id", "tillage", "betr_id", "group", "crop", "lnf_code")
            if c in meta.columns]
    out = out.merge(pd.DataFrame(meta[keep]), on="field_id", how="left")
    out.to_parquet(F_OUT, index=False)

    print(f"\n{len(out):,} field-pair values, {out['field_id'].nunique():,} fields, "
          f"{out['pair_id'].nunique()} pairs")
    print("Pixels per field (median n_valid):", out.groupby("direction")["n_valid"].median().to_dict())
    print(f"Values without any valid pixel: {out['coh_mean'].isna().mean():.1%}")
    print(f"Written: {F_OUT}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    start = datetime.now()  
    
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["sample", "search", "submit", "download", "extract"])
    ap.add_argument("--go", action="store_true", help="really submit jobs (stage submit)")
    ap.add_argument("--limit", type=int, default=None, help="submit at most N pairs (test run)")
    a = ap.parse_args()
    {"sample": stage_sample, "search": stage_search,
     "submit": lambda: stage_submit(a.go, a.limit), "download": stage_download,
     "extract": stage_extract}[a.stage]()
    
    end = datetime.now()
    print(f'Calculation done at {end}. Calculation took {end-start}')
