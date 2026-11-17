import pyogrio
import os
import warnings
import pandas as pd
import numpy as np
import geopandas as gpd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

from calibrate_cfactor import get_ei_grid_offset, snap_to_ei_grid, load_ei_for_pixels, join_ei_to_fc

def get_betriebsnr():
    """
    Transfer the attribute "betriebsnummer" from lnf2022.gpkg to Georef.gpkg.
    
    Logic
    -----
    1. Intersect every Georef polygon with the lnf2022 polygons.
    2. Ignore tiny overlaps (edge slivers), because the polygons are not identical.
    3. If a Georef polygon overlaps several lnf2022 polygons, they must all have the
       same betriebsnummer. If not, a warning is raised and the betriebsnummer with
       the largest overlap area is used (and the row is flagged).
    4. Also warn if a Georef polygon has no match or is only poorly covered.
    """
    # ----------------------------------------------------------------------------
    # Settings
    # ----------------------------------------------------------------------------
    LNF_PATH = "/mnt/eo-nas1/data/landuse/raw/lnf2022.gpkg"
    GEOREF_PATH = "/mnt/Data-Labo-RE/27_Natural_Resources-RE/321.2_AUI_Monitoring_protected/Georeferenzierung/Georeferenzierung_AUI.gpkg"
    OUT_PATH = "test/Georeferenzierung_AUI_mit_betriebsnummer.gpkg"
    
    LNF_LAYER = None      # set a layer name if the gpkg contains several layers
    GEOREF_LAYER = None
    
    ID_COL = "betriebsnummer"
    
    # An lnf2022 polygon only counts if it covers at least this share of the
    # Georef polygon (filters out slivers along the borders).
    MIN_SHARE = 0.05
    
    # Warn if the Georef polygon is covered less than this by lnf2022 in total.
    MIN_TOTAL_COVERAGE = 0.5
    
    # Metric CRS used for area calculations (Swiss LV95).
    AREA_CRS = "EPSG:2056"
    
    # ----------------------------------------------------------------------------
    # Load data
    # ----------------------------------------------------------------------------
    lnf = gpd.read_file(LNF_PATH, layer=LNF_LAYER)
    georef = gpd.read_file(GEOREF_PATH, layer=GEOREF_LAYER)
    
    if ID_COL not in lnf.columns:
        raise KeyError(f"Column '{ID_COL}' not found in {LNF_PATH}. "
                       f"Available columns: {list(lnf.columns)}")
    
    georef = georef.reset_index(drop=True)
    georef["_gid"] = georef.index
    
    # Work on metric, valid copies for the overlay / area calculation
    geo_m = georef[["_gid", "geometry"]].to_crs(AREA_CRS)
    lnf_m = lnf[[ID_COL, "geometry"]].to_crs(AREA_CRS)
    geo_m["geometry"] = geo_m.geometry.make_valid()
    lnf_m["geometry"] = lnf_m.geometry.make_valid()
    lnf_m = lnf_m[lnf_m[ID_COL].notna()]
    
    # ----------------------------------------------------------------------------
    # Overlay and overlap shares
    # ----------------------------------------------------------------------------
    inter = gpd.overlay(geo_m, lnf_m, how="intersection", keep_geom_type=True)
    inter["ov_area"] = inter.geometry.area
    
    georef_area = geo_m.set_index("_gid").geometry.area
    inter["share"] = inter["ov_area"] / inter["_gid"].map(georef_area)
    
    # Sum overlap share per Georef polygon and betriebsnummer
    shares = (inter.groupby(["_gid", ID_COL], as_index=False)["share"].sum())
    total_coverage = shares.groupby("_gid")["share"].sum()
    
    relevant = shares[shares["share"] >= MIN_SHARE]
    
    # ----------------------------------------------------------------------------
    # Assign betriebsnummer and check consistency
    # ----------------------------------------------------------------------------
    results = []
    for gid in georef["_gid"]:
        sub = relevant[relevant["_gid"] == gid].sort_values("share", ascending=False)
        coverage = total_coverage.get(gid, 0.0)
    
        if sub.empty:
            warnings.warn(f"Georef polygon {gid}: no overlapping lnf2022 polygon found.")
            results.append((gid, None, 0, coverage, False))
            continue
    
        ids = sub[ID_COL].unique()
        conflict = len(ids) > 1
        if conflict:
            detail = ", ".join(f"{i} ({s:.0%})" for i, s in zip(sub[ID_COL], sub["share"]))
            warnings.warn(
                f"Georef polygon {gid}: overlaps lnf2022 polygons with different "
                f"{ID_COL} values: {detail}. Using the one with the largest overlap."
            )
    
        if coverage < MIN_TOTAL_COVERAGE:
            warnings.warn(f"Georef polygon {gid}: only {coverage:.0%} covered by lnf2022.")
    
        results.append((gid, sub.iloc[0][ID_COL], len(ids), coverage, conflict))
    
    res = pd.DataFrame(results, columns=["_gid", ID_COL, "n_betriebe",
                                         "lnf_coverage", "bnr_konflikt"])
    
    # ----------------------------------------------------------------------------
    # Join back to the original Georef (original CRS and attributes kept)
    # ----------------------------------------------------------------------------
    if ID_COL in georef.columns:
        georef = georef.drop(columns=ID_COL)
    
    out = georef.merge(res, on="_gid", how="left").drop(columns="_gid")
    out.to_file(OUT_PATH, driver="GPKG")

def merge_aui_cref(add_ei=False):
    # load reference data high quality C-factor estimated for various 
    results_dir = 'run_eb1'
    if add_ei:
        ei_path = '../../../../Erosion/erosivity_index/predictions/grid_EI_daily_avg_pred_20260424_nn3.parquet'
        df_fc = pd.read_parquet('samples_data_gpr_aui.parquet')
        df_ref = pd.read_pickle('samples_aui.pkl')
        x_off, y_off = get_ei_grid_offset(ei_path)
        print(f"EI grid offset: x={x_off}, y={y_off}")
        x_snap, y_snap = snap_to_ei_grid(df_fc['x'].values, df_fc['y'].values, x_off, y_off)
        x_unique = np.unique(x_snap)
        y_unique = np.unique(y_snap)
        df_ei = load_ei_for_pixels(ei_path, x_unique, y_unique)
        print(f"EI loaded: {len(df_ei)} rows for {df_ei[['x','y']].drop_duplicates().shape[0]} unique cells")
    
        # Join EI to FC timeseries
        print("Joining EI to FC timeseries...")
        df = join_ei_to_fc(df_fc, df_ei, x_off, y_off)
        print(df)
    
        df.to_csv(os.path.join(results_dir, 'df_ex.csv'), index=False)
        df_ref.to_csv(os.path.join(results_dir, 'df_ex_ref.csv'), index=False)

        x_ts = df
        y = df_ref

    else:
        x_ts = pd.read_csv(os.path.join(results_dir, 'df_ex.csv'))
        y = pd.read_csv(os.path.join(results_dir, 'df_ex_ref.csv'))

    # replace ZAAUI farm_id
    mapping_farm_path = '/mnt/Data-Labo-RE/27_Natural_Resources-RE/321.2_AUI_Monitoring_protected/Georeferenzierung/Mapping_IDs_ZAAUI_Georeferenzierung.csv'
    farm_map = pd.read_csv(mapping_farm_path, encoding="latin1", sep=";")
    farm_map = dict(zip(farm_map["ZAAUI_betrID"], farm_map["betr_ID"]))
    y["betr_ID"] = y["betr_ID"].map(farm_map).fillna(y["betr_ID"])

    # add tillage information
    path_bodenschonend = '/mnt/Data-Labo-RE/27_Natural_Resources-RE/321.4_WAUM_protected/Daten/Erosionsrisiko/schonende_bodenbearbeitung.csv'
    df_till = pd.read_csv(path_bodenschonend, encoding='latin-1', delimiter=';')
    till_map = (df_till.groupby("betr_ID")["VERFAHREN"].agg(lambda s: s.value_counts().idxmax()))
    replace_names = {'Mulchsaat':'Mulch', 'Streifensaat':'Direkt'}    
    y["tillage"] = y["betr_ID"].map(till_map).fillna("Pflug")
    y["tillage"] = y["tillage"].map(replace_names).fillna("Pflug")

    df = pd.read_csv('run_eb1/df.csv')
    df_ref = pd.read_csv('run_eb1/df_ref.csv')
    
    MONTHLY_VARS    = ['pv', 'npv', 'ei']
    KEY             = ['poly_id', 'yr']
    SEASON_START    = 7        # season for year yr runs from 1 July yr-1 to 30 June yr
    # monthly columns in season order: Jul ... Dec (of yr-1), Jan ... Jun (of yr)
    SEASON_MONTHS   = list(range(SEASON_START, 13)) + list(range(1, SEASON_START))
    DROP_DF_OVERLAP = True     # if a poly_id/yr exists in both sources, keep only x_ts (higher quality)
    
    
    # --- helpers ------------------------------------------------------------------
    def harmonise_keys(d):
        d = d.copy()
        d['poly_id'] = d['poly_id'].astype(str)
        d['yr'] = d['yr'].astype(int)
        return d
    
    
    def to_monthly(ts):
        """Monthly means per poly_id and season (1 July yr-1 to 30 June yr).
    
        The season year is derived from the date, not from the yr column, so the
        July-December values are taken from the rows of the previous year.
        """
        ts = ts.copy()
        t = pd.to_datetime(ts['time'])
        ts['month'] = t.dt.month
        ts['yr'] = t.dt.year + (t.dt.month >= SEASON_START).astype(int)   # season year
        m = ts.groupby(KEY + ['month'])[MONTHLY_VARS].mean().unstack('month')
        m = m.reindex(columns=pd.MultiIndex.from_product([MONTHLY_VARS, SEASON_MONTHS]))
        m.columns = [f'{v}_m{mo:02d}' for v, mo in m.columns]
        return m.reset_index()
    
    
    def report_incomplete(m, label):
        month_cols = [c for c in m.columns if c.startswith(tuple(f'{v}_m' for v in MONTHLY_VARS))]
        n_inc = m[month_cols].isna().any(axis=1).sum()
        if n_inc:
            print(f"Note ({label}): {n_inc} of {len(m)} poly_id/yr have months without data "
                  f"(e.g. first season, where July-December of the previous year is missing)")
    
    
    def first_per_key(d, cols, label):
        """One value per poly_id/yr for static attributes; warns if they are not unique."""
        out = d.groupby(KEY)[cols].first()
        n_conf = (d.groupby(KEY)[cols].nunique(dropna=True) > 1).sum()
        for c, n in n_conf.items():
            if n:
                print(f"Warning ({label}): {n} poly_id/yr with more than one '{c}', first value used")
        return out.reset_index()
    
    
    # --- 1. High-quality source: x_ts + y -----------------------------------------
    x_ts_h = harmonise_keys(x_ts)
    y_h    = harmonise_keys(y)
    
    # y: one row per poly_id/yr
    dup = y_h.duplicated(KEY, keep=False)
    if dup.any():
        n_conf = (y_h[dup].groupby(KEY)['C_Fakt'].nunique() > 1).sum()
        print(f"Warning (y): {dup.sum()} rows share a poly_id/yr "
              f"({n_conf} with different C_Fakt); C_Fakt averaged, other attributes from first row")
    y_attr = y_h.groupby(KEY).agg(C_Fakt=('C_Fakt', 'mean'),
                                  lnf_code=('lnf_code', 'first'),
                                  tillage=('tillage', 'first'),
                                  betr_ID=('betr_ID', 'first')).reset_index()
    
    # monthly features per season, then match with y
    x_monthly = to_monthly(x_ts_h)
    report_incomplete(x_monthly, 'x_ts')
    
    keys_x = set(map(tuple, x_monthly[KEY].to_numpy()))
    keys_y = set(map(tuple, y_attr[KEY].to_numpy()))
    print(f"x_ts: {len(keys_x)} poly_id/yr, y: {len(keys_y)} poly_id/yr, "
          f"matched: {len(keys_x & keys_y)}, only in x_ts: {len(keys_x - keys_y)}, "
          f"only in y: {len(keys_y - keys_x)}")
    
    hq = x_monthly.merge(y_attr, on=KEY, how='inner')
    
    # lnf_code consistency between x_ts and y
    lnf_x = first_per_key(x_ts_h, ['lnf_code'], 'x_ts').rename(columns={'lnf_code': 'lnf_x'})
    chk = hq[KEY + ['lnf_code']].merge(lnf_x, on=KEY)
    n_diff = (chk['lnf_code'].astype(str) != chk['lnf_x'].astype(str)).sum()
    if n_diff:
        print(f"Warning: {n_diff} poly_id/yr with different lnf_code in x_ts and y (y is used)")
    
    hq['source'] = 'x_ts'
    
    # --- 2. Lower-quality source: df + df_ref -------------------------------------
    df_h = harmonise_keys(df)
    
    df_monthly = to_monthly(df_h)
    report_incomplete(df_monthly, 'df')
    
    # region is only needed to look up C_ref, it is not kept in the output
    lq = df_monthly.merge(
        first_per_key(df_h, ['lnf_code', 'region', 'tillage', 'betr_ID'], 'df'), on=KEY, how='inner')
    
    merge_cols = ['lnf_code', 'region', 'tillage']
    ref = df_ref[merge_cols + ['C_ref']].copy()
    for c in merge_cols:
        lq[c] = lq[c].astype(str)
        ref[c] = ref[c].astype(str)
    ref = ref.drop_duplicates(subset=merge_cols)
    
    n_before = len(lq)
    lq = lq.merge(ref, on=merge_cols, how='inner').rename(columns={'C_ref': 'C_Fakt'})
    if len(lq) < n_before:
        print(f"Warning (df): {n_before - len(lq)} poly_id/yr without matching C_ref dropped")
    lq['source'] = 'df'
    
    # --- 3. Combine ---------------------------------------------------------------
    for c in ['lnf_code', 'tillage', 'betr_ID']:
        hq[c] = hq[c].astype(str).where(hq[c].notna())
        lq[c] = lq[c].astype(str).where(lq[c].notna())
    
    if DROP_DF_OVERLAP:
        overlap = lq.set_index(KEY).index.isin(hq.set_index(KEY).index)
        if overlap.any():
            print(f"{overlap.sum()} poly_id/yr exist in both sources; df rows dropped")
        lq = lq[~overlap]
    
    month_cols = [f'{v}_m{m:02d}' for v in MONTHLY_VARS for m in SEASON_MONTHS]
    cols = KEY + ['source', 'lnf_code', 'tillage', 'betr_ID', 'C_Fakt'] + month_cols
    data = pd.concat([hq[cols], lq[cols]], ignore_index=True)
    
    dups = data.duplicated(KEY).sum()
    if dups:
        print(f"Warning: {dups} duplicate poly_id/yr in the combined table "
              f"(set DROP_DF_OVERLAP = True, or check whether poly_ids differ between sources)")
    
    print(f"\nCombined table: {len(data)} rows")
    print(data['source'].value_counts().to_string())

    data.to_csv('XY.csv', index=False)

def calculate_c_factor():
    # ---------------------------------------------------------
    # Load data
    # ---------------------------------------------------------
    
    df = pd.read_csv("run_eb1/XY.csv")
    
    TARGET = "C_Fakt"
    YEAR_COL = "yr"
    
    # predictors
    spectral_cols = [
        c for c in df.columns
        if c.startswith("pv")
        or c.startswith("npv")
        or c.startswith("ei")
    ]
    
    # ---------------------------------------------------------
    # Function: hyperparameter tuning inside LOYO fold
    # ---------------------------------------------------------
    
    def train_predict_loyo(data, predictors):
    
        pred = pd.Series(np.nan, index=data.index)
    
        years = sorted(data[YEAR_COL].dropna().unique())
    
        param_grid = [
            {
                "max_depth": md,
                "learning_rate": lr,
                "n_estimators": ne,
            }
            for md in [3, 5, 7]
            for lr in [0.03, 0.1]
            for ne in [20, 35, 50]
        ]
    
        for test_year in years:
    
            train_idx = data[YEAR_COL] != test_year
            test_idx = data[YEAR_COL] == test_year
    
            train_data = data.loc[train_idx]
            test_data = data.loc[test_idx]
    
            X_train_full = train_data[predictors]
            y_train_full = train_data[TARGET]
    
            X_test = test_data[predictors]
    
            (
                X_train,
                X_val,
                y_train,
                y_val
            ) = train_test_split(
                X_train_full,
                y_train_full,
                test_size=0.2,
                random_state=42
            )
    
            best_r2 = -np.inf
            best_params = None
    
            for params in param_grid:
    
                model = XGBRegressor(
                    objective="reg:squarederror",
                    random_state=42,
                    **params
                )
    
                model.fit(X_train, y_train)
    
                pred_val = model.predict(X_val)
                pred_train = model.predict(X_train)
    
                r2 = r2_score(y_val, pred_val)
                r2_train = r2_score(y_train, pred_train)
    
                if r2 > best_r2:
                    best_r2 = r2
                    best_r2_train = r2_train
                    best_params = params
    
            final_model = XGBRegressor(
                objective="reg:squarederror",
                random_state=42,
                **best_params
            )
    
            final_model.fit(
                X_train_full,
                y_train_full
            )
    
            pred.loc[test_data.index] = final_model.predict(X_test)
    
            print(
                f"Year={test_year} "
                f"R2={best_r2:.3f}"
                f"R2 train={best_r2_train:.3f}"
            )
    
        return pred
    
    # ---------------------------------------------------------
    # Setup 1
    # all rows
    # ---------------------------------------------------------
    
    df["pred_setup1"] = train_predict_loyo(
        df,
        spectral_cols
    )
    
    # ---------------------------------------------------------
    # Setup 2
    # source == x_ts
    # ---------------------------------------------------------
    
    mask_xts = df["source"] == "x_ts"
    
    df["pred_setup2"] = np.nan
    
    df.loc[
        mask_xts,
        "pred_setup2"
    ] = train_predict_loyo(
        df.loc[mask_xts].copy(),
        spectral_cols
    )
    
    # ---------------------------------------------------------
    # Setup 3
    # source != x_ts
    # ---------------------------------------------------------
    
    mask_non_xts = df["source"] == "df"
    
    df["pred_setup3"] = np.nan
    
    df.loc[
        mask_non_xts,
        "pred_setup3"
    ] = train_predict_loyo(
        df.loc[mask_non_xts].copy(),
        spectral_cols
    )
    
    # ---------------------------------------------------------
    # Setup 4
    # add tillage
    # ---------------------------------------------------------
    
    df["tillage"] = (
        df["tillage"]
          .astype("category")
          .cat.codes
    )
    
    predictors4 = spectral_cols + ["tillage"]
    
    df["pred_setup4"] = train_predict_loyo(
        df,
        predictors4
    )
    
    # ---------------------------------------------------------
    # Setup 5
    # Formula approach
    # ---------------------------------------------------------
    
    # fvc_cols = []
    # slr_cols = []
    
    # for m in range(1, 13):
    
    #     month = f"{m:02d}"
    
    #     pv = f"pv_m{month}"
    #     npv = f"npv_m{month}"
    
    #     fvc = df[pv] + df[npv]
    
    #     slr = np.exp(-0.048 * fvc)
    
    #     ei = df[f"ei_m{month}"]
    
    #     slr_cols.append(
    #         slr * ei
    #     )
    
    # df["pred_setup5"] = (
    #     np.sum(slr_cols, axis=0) / 12
    # )
    # ---------------------------------------------------------
    # Setup 6
    # Formula approach
    # ---------------------------------------------------------
    mask_non_xts = df["source"] == "x_ts"

    # encode tillage in a separate column, so the original stays readable
    df["tillage_code"] = (
        df["tillage"]
          .astype("category")
          .cat.codes
    )
    
    predictors6 = spectral_cols + ["tillage_code"]
    
    df["pred_setup6"] = np.nan
    
    df.loc[mask_non_xts, "pred_setup6"] = train_predict_loyo(
        df.loc[mask_non_xts].copy(),
        predictors6
    )

    # ---------------------------------------------------------
    # Setup 7
    # Formula approach
    # ---------------------------------------------------------
    mask_non_xts = df["source"] == "df"

    # encode tillage in a separate column, so the original stays readable
    df["tillage_code"] = (
        df["tillage"]
          .astype("category")
          .cat.codes
    )
    
    predictors7 = spectral_cols + ["tillage_code"]
    
    df["pred_setup7"] = np.nan
    
    df.loc[mask_non_xts, "pred_setup7"] = train_predict_loyo(
        df.loc[mask_non_xts].copy(),
        predictors7
    )

    
    # ---------------------------------------------------------
    # Save results
    # ---------------------------------------------------------
    
    results = pd.DataFrame({
        "C_ref": df[TARGET],
        "all": df["pred_setup1"],
        "hq": df["pred_setup2"],
        "lq": df["pred_setup3"],
        "all_till": df["pred_setup4"],
        "hq_till": df["pred_setup6"],
        "lq_till": df["pred_setup7"]
    })
    
    results.to_csv(
        "run_eb1/XY_model_comparison_new.csv",
        index=False
    )
    
    print(results.head())
    print("Saved: XY_model_comparison_new.csv")

def boxplots_tillage():
    a = pd.read_csv('run_eb1/XY.csv')
    df = pd.read_csv('run_eb1/XY_model_comparison_new.csv')
    cols2add = a.columns[:6]
    df[cols2add] = a[cols2add]

    panels = [
        ("all", ["C_ref", "all", "all_till"], None),
        ("hq",  ["C_ref", "hq",  "hq_till"],  "hq"),
        ("lq",  ["C_ref", "lq",  "lq_till"],  "lq"),
    ]

    tillage_order = sorted(df["tillage"].dropna().unique())

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    for ax, (name, cols, filter_col) in zip(axes, panels):
        # hq / lq: only rows where that model has a prediction
        sub = df[df[filter_col].notna()] if filter_col else df

        df_long = sub.melt(
            id_vars="tillage",
            value_vars=cols,
            var_name="source",
            value_name="C",
        )

        sns.boxplot(
            data=df_long,
            x="tillage",
            y="C",
            hue="source",
            hue_order=cols,
            order=tillage_order,
            showfliers=True,
            ax=ax,
        )

        ax.set_xlabel("Tillage")
        ax.set_ylabel("C" if ax is axes[0] else "")
        ax.set_title(f"{name} (n = {len(sub)})")
        ax.legend(title="", frameon=False)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("C value per tillage")
    plt.tight_layout()
    plt.savefig('run_eb1/compare_tillage_n.png', dpi=150)

def scatter_setups():
    a = pd.read_csv('run_eb1/XY.csv')
    df = pd.read_csv('run_eb1/XY_model_comparison_new.csv')
    cols2add = a.columns[:6]
    df[cols2add] = a[cols2add]
    agg_col_name = "lnf_code"
    setups = ["all", "all_till", "hq",  "hq_till", "lq",  "lq_till"]
    
    cols = ["C_ref"] + setups
    
    g = df.groupby(agg_col_name)[cols]
    df_mean = g.mean()
    df_std = g.std()
    df_n = g.size()
    
    # shared axis limits (based on means ± std)
    lo = np.nanmin((df_mean - df_std).to_numpy())
    hi = np.nanmax((df_mean + df_std).to_numpy())
    pad = 0.05 * (hi - lo)
    lims = (lo - pad, hi + pad)
    
    codes = df_mean.index.tolist()
    palette = dict(zip(codes, sns.color_palette("tab10", len(codes))))
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 14), sharex=True, sharey=True)
    
    for ax, s in zip(axes.ravel(), setups):
        for c in codes:
            ax.errorbar(
                df_mean.loc[c, "C_ref"],
                df_mean.loc[c, s],
                xerr=df_std.loc[c, "C_ref"],
                yerr=df_std.loc[c, s],
                fmt="o",
                ms=7,
                color=palette[c],
                ecolor=palette[c],
                elinewidth=0.8,
                alpha=0.6,
                capsize=2,
            )
    
        ax.plot(lims, lims, "k--", lw=1)  # 1:1 line
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect("equal")
    
        # metrics on the class means
        d = df_mean[["C_ref", s]].dropna()
        rmse = np.sqrt(np.mean((d[s] - d["C_ref"]) ** 2))
        r2 = np.corrcoef(d["C_ref"], d[s])[0, 1] ** 2
        ax.text(
            0.04, 0.96,
            f"RMSE = {rmse:.2f}\nR² = {r2:.2f}\nn classes = {len(d)}",
            transform=ax.transAxes,
            va="top", fontsize=9,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )
    
        ax.set_title(s)
        ax.set_xlabel("C_ref (mean)")
        ax.set_ylabel(f"{s} (mean)")
        ax.grid(alpha=0.3)

        # shared legend: one entry per class, built from the palette
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", ms=7, color=palette[c], alpha=0.6,
                   label=f"{c} (n = {df_n[c]})")
        for c in codes
    ]
    fig.legend(
        handles=handles,
        title=agg_col_name,
        loc="center left",
        bbox_to_anchor=(0.86, 0.5),
        frameon=False,
    )
        
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(f'run_eb1/scatter_{agg_col_name}_new.png', dpi=150)

def boxplots_metrics_per_year(year_col="yr", agg_col_name="lnf_code", use_class_means=True):
    a = pd.read_csv('run_eb1/XY.csv')
    df = pd.read_csv('run_eb1/XY_model_comparison_new.csv')
    cols2add = a.columns[:6]
    df[cols2add] = a[cols2add]

    setups = ["all", "all_till", "hq", "hq_till", "lq", "lq_till"]

    records = []
    for yr, d_yr in df.groupby(year_col):
        for s in setups:
            d = d_yr[[agg_col_name, "C_ref", s]].dropna()
            if use_class_means:                      # same basis as the scatter plot
                d = d.groupby(agg_col_name)[["C_ref", s]].mean()
            if len(d) < 3:                           # too few points for a meaningful R²
                continue
            rmse = np.sqrt(np.mean((d[s] - d["C_ref"]) ** 2))
            r2 = np.corrcoef(d["C_ref"], d[s])[0, 1] ** 2
            records.append(dict(year=yr, setup=s, RMSE=rmse, R2=r2, n=len(d)))

    res = pd.DataFrame(records)
    print(res.pivot(index="year", columns="setup", values="R2").round(3))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, metric, label in zip(axes, ["RMSE", "R2"], ["RMSE", "R²"]):
        sns.boxplot(data=res, x="setup", y=metric, order=setups,
                    color="lightgrey", showfliers=False, ax=ax)
        sns.stripplot(data=res, x="setup", y=metric, order=setups,
                      hue="year", palette="viridis", size=6, jitter=0.15,
                      ax=ax, legend=(ax is axes[-1]))
        ax.set_xlabel("")
        ax.set_ylabel(label)
        ax.set_title(f"{label} per year")
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=30)

    axes[-1].legend(title=year_col, frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))

    basis = f"class means ({agg_col_name})" if use_class_means else "individual rows"
    fig.suptitle(f"Model performance per year, computed on {basis}")
    plt.tight_layout()
    plt.savefig('run_eb1/metrics_per_year.png', dpi=150, bbox_inches="tight")

if __name__ == "__main__":
    # scatter_setups()
    # boxplots_metrics_per_year()

    lnf_labels_path = '/mnt/eo-nas1/data/landuse/documentation/LNF_code_classification_20260217.xlsx'
    lnf_ignore_codes = [553, 554, 555, 556, 559, 572, 594, 595, 598, 618, 625]

    # Read Excel file
    df = pd.read_excel(lnf_labels_path)
    
    # Show crop_DE values for matching rows
    print(df.loc[df["LNF_code"].isin(lnf_ignore_codes), "Crop_DE"])
    
