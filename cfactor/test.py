# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import xgboost as xgb
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import r2_score, mean_absolute_error

# def sample_rows(csv_path, n=275, seed=42):
#     df = pd.read_csv(csv_path).dropna(subset=['C_ref', 'C_predicted'])
#     if len(df) < n:
#         raise ValueError(f"Only {len(df)} valid rows available, cannot sample {n}.")
#     return df.sample(n=n, random_state=seed).reset_index(drop=True)

# CALIB_CSV = 'run_eb1/calibration_results_stratified_per_pixel.csv'
# df = pd.read_csv('run_eb1/df.csv')
# df_ref = pd.read_csv('run_eb1/df_ref.csv')

# # --- 1. Aggregate: mean fc_total per field and harvest year -------------------
# group_cols = ['poly_id', 'yr', 'lnf_code', 'tillage', 'region']
# df_agg = (
#     df.groupby(group_cols, as_index=False)['fc_total']
#       .mean()
# )

# # --- 2. Attach reference C factor --------------------------------------------
# merge_cols = ['lnf_code', 'region', 'tillage']

# # make sure key dtypes match (e.g. lnf_code int vs str breaks the merge silently)
# for c in merge_cols:
#     df_agg[c] = df_agg[c].astype(str)
#     df_ref[c] = df_ref[c].astype(str)

# ref = df_ref[merge_cols + ['C_ref']].drop_duplicates(subset=merge_cols)
# data = df_agg.merge(ref, on=merge_cols, how='inner')

# n_lost = len(df_agg) - len(data)
# if n_lost:
#     print(f"Warning: {n_lost} field-years had no matching C_ref and were dropped.")

# data = data.dropna(subset=['fc_mean', 'C_ref'])

# # --- 3. Train/test split (80/20) ---------------------------------------------
# X = data[['fc_mean']]
# y = data['C_ref']

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# # --- 4. Model -----------------------------------------------------------------
# model = xgb.XGBRegressor(
#     n_estimators=300,
#     learning_rate=0.05,
#     max_depth=3,
#     subsample=0.8,
#     random_state=42,
# )
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)

# # --- 5. Calibration results from file -----------------------------------------
# calib = sample_rows(CALIB_CSV, n=len(y_test))
# obs_cal  = calib['C_ref'].to_numpy()
# pred_cal = calib['C_predicted'].to_numpy()

# # --- 6. Metrics ---------------------------------------------------------------
# def metrics(obs, pred):
#     obs, pred = np.asarray(obs), np.asarray(pred)
#     return {
#         'R²':   r2_score(obs, pred),
#         'MAE':  mean_absolute_error(obs, pred),
#         'Bias': np.mean(pred - obs),      # positive = overestimation
#         'n':    len(obs),
#     }

# m_xgb = metrics(y_test, y_pred)
# m_cal = metrics(obs_cal, pred_cal)

# for name, m in [('XGBoost (test)', m_xgb), ('Calibration', m_cal)]:
#     print(f"{name:15s}  R² = {m['R²']:.3f}  MAE = {m['MAE']:.4f}  "
#           f"Bias = {m['Bias']:.4f}  n = {m['n']}")

# # --- 7. Plot ------------------------------------------------------------------
# # shared limits so both panels are directly comparable
# all_vals = np.concatenate([y_test, y_pred, obs_cal, pred_cal])
# lims = [all_vals.min(), all_vals.max()]

# panels = [
#     (y_test, y_pred, m_xgb, 'XGBoost: mean fc_total (test set)'),
#     (obs_cal, pred_cal, m_cal, 'Calibration: stratified per pixel'),
# ]

# fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)

# for ax, (obs, pred, m, title) in zip(axes, panels):
#     ax.scatter(obs, pred, alpha=0.5, s=20, edgecolor='none')
#     ax.plot(lims, lims, 'k--', lw=1, label='1:1')
#     ax.set_xlim(lims)
#     ax.set_ylim(lims)
#     ax.set_aspect('equal')
#     ax.set_title(title)
#     ax.set_xlabel('Reference C factor (C_ref)')
#     ax.text(
#         0.05, 0.95,
#         f"R² = {m['R²']:.3f}\nMAE = {m['MAE']:.4f}\n"
#         f"Bias = {m['Bias']:.4f}\nn = {m['n']}",
#         transform=ax.transAxes, va='top',
#         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
#     )
#     ax.legend(loc='lower right')

# axes[0].set_ylabel('Predicted C factor')
# # plt.tight_layout()
# fig.subplots_adjust(top=0.92, bottom=0.08)
# plt.savefig('run_eb1/xgb_comp_pixel_random.png', dpi=300)


# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import xgboost as xgb
# from sklearn.metrics import r2_score, mean_absolute_error

# CALIB_CSV = 'run_eb1/calibration_results_stratified_per_pixel.csv'
# TEST_FRAC = 0.2
# SEED      = 42
# df = pd.read_csv('run_eb1/df.csv')
# df_ref = pd.read_csv('run_eb1/df_ref.csv')

# # --- 1. Aggregate: mean fc_total per field and harvest year -------------------
# # if df has a pixel ID column, add it here to keep pixels separate
# group_cols = ['poly_id', 'yr', 'lnf_code', 'tillage', 'region']
# df_agg = (
#     df.groupby(group_cols, as_index=False)['fc_total']
#       .mean()
#       .rename(columns={'fc_total': 'fc_mean'})
# )

# # --- 2. Attach reference C factor --------------------------------------------
# merge_cols = ['lnf_code', 'region', 'tillage']

# # make sure key dtypes match (e.g. int vs str breaks merges/filters silently)
# for c in merge_cols:
#     df_agg[c] = df_agg[c].astype(str)
#     df_ref[c] = df_ref[c].astype(str)
# df_agg['poly_id'] = df_agg['poly_id'].astype(str)

# ref = df_ref[merge_cols + ['C_ref']].drop_duplicates(subset=merge_cols)
# data = df_agg.merge(ref, on=merge_cols, how='inner')

# n_lost = len(df_agg) - len(data)
# if n_lost:
#     print(f"Warning: {n_lost} field-years had no matching C_ref and were dropped.")

# data = data.dropna(subset=['fc_mean', 'C_ref'])

# # --- 3. Load calibration results ---------------------------------------------
# calib = pd.read_csv(CALIB_CSV).dropna(subset=['C_ref', 'C_predicted'])
# calib['poly_id'] = calib['poly_id'].astype(str)

# # --- 4. Field-based split (20% of fields -> test) ----------------------------
# # draw test fields only from polygons present in BOTH tables,
# # so both panels are evaluated on exactly the same fields
# common_polys = np.intersect1d(data['poly_id'].unique(), calib['poly_id'].unique())
# n_only_xgb = data['poly_id'].nunique() - len(common_polys)
# if n_only_xgb:
#     print(f"Note: {n_only_xgb} fields in df are not in the calibration file "
#           f"(always used for training).")

# rng = np.random.default_rng(SEED)
# n_test = int(round(TEST_FRAC * len(common_polys)))
# test_polys = rng.choice(common_polys, size=n_test, replace=False)

# is_test = data['poly_id'].isin(test_polys)
# train, test = data[~is_test], data[is_test]
# calib_test = calib[calib['poly_id'].isin(test_polys)]

# print(f"Test fields: {n_test} of {len(common_polys)} shared fields")
# print(f"XGBoost:     {len(train)} train rows, {len(test)} test rows "
#       f"({len(test) / len(data):.1%})")
# print(f"Calibration: {len(calib_test)} test rows")

# # --- 5. Model -----------------------------------------------------------------
# X_train, y_train = train[['fc_mean']], train['C_ref']
# X_test,  y_test  = test[['fc_mean']],  test['C_ref']

# model = xgb.XGBRegressor(
#     n_estimators=300,
#     learning_rate=0.05,
#     max_depth=3,
#     subsample=0.8,
#     random_state=SEED,
# )
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)

# obs_cal  = calib_test['C_ref'].to_numpy()
# pred_cal = calib_test['C_predicted'].to_numpy()

# # --- 6. Metrics ---------------------------------------------------------------
# def metrics(obs, pred):
#     obs, pred = np.asarray(obs), np.asarray(pred)
#     return {
#         'R²':   r2_score(obs, pred),
#         'MAE':  mean_absolute_error(obs, pred),
#         'Bias': np.mean(pred - obs),      # positive = overestimation
#         'n':    len(obs),
#     }

# m_xgb = metrics(y_test, y_pred)
# m_cal = metrics(obs_cal, pred_cal)

# for name, m in [('XGBoost (test)', m_xgb), ('Calibration', m_cal)]:
#     print(f"{name:15s}  R² = {m['R²']:.3f}  MAE = {m['MAE']:.4f}  "
#           f"Bias = {m['Bias']:.4f}  n = {m['n']}")

# # --- 7. Plot ------------------------------------------------------------------
# def adjust_layout(fig, top=0.90, bottom=0.12, left=0.08, right=0.97, wspace=0.15):
#     fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right, wspace=wspace)

# # shared limits so both panels are directly comparable
# all_vals = np.concatenate([y_test, y_pred, obs_cal, pred_cal])
# lims = [all_vals.min(), all_vals.max()]

# panels = [
#     (y_test, y_pred, m_xgb, 'XGBoost: mean fc_total (test fields)'),
#     (obs_cal, pred_cal, m_cal, 'Calibration: stratified per pixel (test fields)'),
# ]

# fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)

# for ax, (obs, pred, m, title) in zip(axes, panels):
#     ax.scatter(obs, pred, alpha=0.5, s=20, edgecolor='none')
#     ax.plot(lims, lims, 'k--', lw=1, label='1:1')
#     ax.set_xlim(lims)
#     ax.set_ylim(lims)
#     ax.set_aspect('equal')
#     ax.set_title(title)
#     ax.set_xlabel('Reference C factor (C_ref)')
#     ax.text(
#         0.05, 0.95,
#         f"R² = {m['R²']:.3f}\nMAE = {m['MAE']:.4f}\n"
#         f"Bias = {m['Bias']:.4f}\nn = {m['n']}",
#         transform=ax.transAxes, va='top',
#         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
#     )
#     ax.legend(loc='lower right')

# axes[0].set_ylabel('Predicted C factor')
# adjust_layout(fig)
# plt.savefig('run_eb1/xgb_pixel_lofieldo.png', dpi=300)

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import xgboost as xgb
# from sklearn.metrics import r2_score, mean_absolute_error

# CALIB_CSV = 'run_eb1/calibration_results_stratified_per_pixel.csv'
# df = pd.read_csv('run_eb1/df.csv')
# df_ref = pd.read_csv('run_eb1/df_ref.csv')
# TEST_FRAC = 0.2
# SEED      = 42
# PIXEL_COL = None   # name of the pixel ID column in df (e.g. 'pixel_id'); None = no pixel column

# # --- 1. Aggregate: mean fc_total per pixel and harvest year -------------------
# group_cols = ['poly_id', 'yr', 'lnf_code', 'tillage', 'region']
# if PIXEL_COL:
#     group_cols.insert(1, PIXEL_COL)
# df_agg = (
#     df.groupby(group_cols, as_index=False)['fc_total']
#       .mean()
#       .rename(columns={'fc_total': 'fc_mean'})
# )

# # --- 2. Attach reference C factor --------------------------------------------
# merge_cols = ['lnf_code', 'region', 'tillage']

# # make sure key dtypes match (e.g. int vs str breaks merges/filters silently)
# for c in merge_cols:
#     df_agg[c] = df_agg[c].astype(str)
#     df_ref[c] = df_ref[c].astype(str)
# df_agg['poly_id'] = df_agg['poly_id'].astype(str)

# ref = df_ref[merge_cols + ['C_ref']].drop_duplicates(subset=merge_cols)
# data = df_agg.merge(ref, on=merge_cols, how='inner')

# n_lost = len(df_agg) - len(data)
# if n_lost:
#     print(f"Warning: {n_lost} field-years had no matching C_ref and were dropped.")

# data = data.dropna(subset=['fc_mean', 'C_ref'])

# # --- 3. Load calibration results ---------------------------------------------
# calib = pd.read_csv(CALIB_CSV).dropna(subset=['C_ref', 'C_predicted'])
# calib['poly_id'] = calib['poly_id'].astype(str)

# # --- 4. Field-based split (20% of fields -> test) ----------------------------
# # draw test fields only from polygons present in BOTH tables,
# # so both panels are evaluated on exactly the same fields
# common_polys = np.intersect1d(data['poly_id'].unique(), calib['poly_id'].unique())
# n_only_xgb = data['poly_id'].nunique() - len(common_polys)
# if n_only_xgb:
#     print(f"Note: {n_only_xgb} fields in df are not in the calibration file "
#           f"(always used for training).")

# rng = np.random.default_rng(SEED)
# n_test = int(round(TEST_FRAC * len(common_polys)))
# test_polys = rng.choice(common_polys, size=n_test, replace=False)

# is_test = data['poly_id'].isin(test_polys)
# train, test = data[~is_test], data[is_test]
# calib_test = calib[calib['poly_id'].isin(test_polys)]

# print(f"Test fields: {n_test} of {len(common_polys)} shared fields")
# print(f"XGBoost:     {len(train)} train rows, {len(test)} test rows "
#       f"({len(test) / len(data):.1%})")
# print(f"Calibration: {len(calib_test)} test rows")

# # --- 5. Model -----------------------------------------------------------------
# X_train, y_train = train[['fc_mean']], train['C_ref']
# X_test,  y_test  = test[['fc_mean']],  test['C_ref']

# model = xgb.XGBRegressor(
#     n_estimators=300,
#     learning_rate=0.05,
#     max_depth=3,
#     subsample=0.8,
#     random_state=SEED,
# )
# model.fit(X_train, y_train)

# # predictions at pixel level
# test = test.copy()
# test['C_predicted'] = model.predict(X_test)

# pix = {
#     'XGBoost':     test[['C_ref', 'C_predicted']],
#     'Calibration': calib_test[['C_ref', 'C_predicted']],
# }

# # --- 6. Metrics ---------------------------------------------------------------
# def metrics(obs, pred):
#     obs, pred = np.asarray(obs), np.asarray(pred)
#     return {
#         'R²':   r2_score(obs, pred),
#         'MAE':  mean_absolute_error(obs, pred),
#         'Bias': np.mean(pred - obs),      # positive = overestimation
#         'n':    len(obs),
#     }

# def metrics_text(m, n_label):
#     return (f"R² = {m['R²']:.3f}\nMAE = {m['MAE']:.4f}\n"
#             f"Bias = {m['Bias']:.4f}\n{n_label} = {m['n']}")

# # --- 7. Group pixels by identical C_ref ---------------------------------------
# def group_by_cref(d, decimals=6):
#     d = d.assign(C_ref=d['C_ref'].round(decimals))   # avoid float-equality issues
#     groups = {c: g['C_predicted'].to_numpy() for c, g in d.groupby('C_ref')}
#     medians = pd.DataFrame({
#         'C_ref':       list(groups.keys()),
#         'C_median':    [np.median(v) for v in groups.values()],
#         'n_pix':       [len(v) for v in groups.values()],
#     })
#     return groups, medians

# grouped = {name: group_by_cref(d) for name, d in pix.items()}

# for name in pix:
#     m_pix = metrics(pix[name]['C_ref'], pix[name]['C_predicted'])
#     med = grouped[name][1]
#     m_med = metrics(med['C_ref'], med['C_median'])
#     print(f"{name:12s} pixels:  R² = {m_pix['R²']:.3f}  MAE = {m_pix['MAE']:.4f}  "
#           f"Bias = {m_pix['Bias']:.4f}  n = {m_pix['n']}")
#     print(f"{name:12s} medians: R² = {m_med['R²']:.3f}  MAE = {m_med['MAE']:.4f}  "
#           f"Bias = {m_med['Bias']:.4f}  n C_ref classes = {m_med['n']}")

# # --- 8. Plot (2x2) ------------------------------------------------------------
# def adjust_layout(fig, top=0.94, bottom=0.07, left=0.08, right=0.97,
#                   wspace=0.15, hspace=0.25):
#     fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right,
#                         wspace=wspace, hspace=hspace)

# # shared limits over all pixel values so all four panels are comparable
# all_vals = np.concatenate([np.concatenate([d['C_ref'], d['C_predicted']])
#                            for d in pix.values()])
# pad = 0.03 * (all_vals.max() - all_vals.min())
# lims = [all_vals.min() - pad, all_vals.max() + pad]

# # box width: fraction of the smallest gap between C_ref classes
# all_cref = np.unique(np.concatenate([g[1]['C_ref'] for g in grouped.values()]))
# gaps = np.diff(all_cref)
# box_w = 0.6 * gaps.min() if len(gaps) else 0.02 * (lims[1] - lims[0])

# fig, axes = plt.subplots(2, 2, figsize=(12, 12), sharex=True, sharey=True)

# for col, name in enumerate(pix):
#     groups, med = grouped[name]
#     d = pix[name]

#     # top row: one median point per C_ref
#     ax = axes[0, col]
#     ax.scatter(med['C_ref'], med['C_median'], s=40, zorder=3)
#     m = metrics(med['C_ref'], med['C_median'])
#     ax.text(0.05, 0.95, metrics_text(m, 'n C_ref classes'),
#             transform=ax.transAxes, va='top',
#             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
#     ax.set_title(f'{name}: median prediction per C_ref')

#     # bottom row: boxplot of pixel predictions per C_ref
#     ax = axes[1, col]
#     ax.boxplot(list(groups.values()), positions=list(groups.keys()),
#                widths=box_w, manage_ticks=False,
#                flierprops=dict(markersize=3, alpha=0.5))
#     m = metrics(d['C_ref'], d['C_predicted'])
#     ax.text(0.05, 0.95, metrics_text(m, 'n pixels'),
#             transform=ax.transAxes, va='top',
#             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
#     ax.set_title(f'{name}: pixel predictions per C_ref')

# for ax in axes.flat:
#     ax.plot(lims, lims, 'k--', lw=1, label='1:1', zorder=1)
#     ax.set_xlim(lims)
#     ax.set_ylim(lims)
#     ax.set_aspect('equal')
#     ax.legend(loc='lower right')

# for ax in axes[1, :]:
#     ax.set_xlabel('Reference C factor (C_ref)')
# for ax in axes[:, 0]:
#     ax.set_ylabel('Predicted C factor')

# adjust_layout(fig)
# plt.savefig('run_eb1/xgb_field_lofieldo.png', dpi=300)


################################# Check FC_ts

def plt_ts_fc():
    import pandas as pd
    import matplotlib.pyplot as plt
    
    df = pd.read_csv('run_eb1/df.csv')
    TIME_COL  = 'time'
    VARIABLES = ['pv', 'npv', 'soil']
    
    # --- 1. Select first field and average over its pixels per date ---------------
    poly = df['poly_id'].iloc[0]
    ts = (
        df.loc[df['poly_id'] == poly, [TIME_COL] + VARIABLES]
          .assign(**{TIME_COL: lambda d: pd.to_datetime(d[TIME_COL])})
          .groupby(TIME_COL)[VARIABLES].mean()
          .sort_index()
    )
    
    # --- 2. Aggregate to coarser resolutions --------------------------------------
    # values are plotted at the centre of each aggregation window
    def resample_centered(d, rule):
        r = d.resample(rule).mean()
        ends = r.index.to_series().shift(-1).fillna(r.index[-1] + pd.tseries.frequencies.to_offset(rule))
        r.index = r.index + (ends - r.index) / 2
        return r
    
    ts_20d = resample_centered(ts, '20D')
    ts_mon = resample_centered(ts, 'MS')
    
    # --- 3. Plot (1x3) ------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)
    
    for ax, var in zip(axes, VARIABLES):
        ax.plot(ts.index, ts[var], 'o-', ms=3, lw=1, alpha=0.6, label='10-daily (original)')
        ax.plot(ts_20d.index, ts_20d[var], 's-', ms=4, lw=1.5, label='20-daily')
        ax.plot(ts_mon.index, ts_mon[var], 'd-', ms=5, lw=2, label='monthly')
        ax.set_title(var)
        ax.set_xlabel('Date')
        ax.grid(alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
    
    axes[0].set_ylabel('Fractional cover')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False)
    fig.suptitle(f'poly_id {poly}: temporal resolution comparison')
    fig.subplots_adjust(top=0.85, bottom=0.25, left=0.05, right=0.98, wspace=0.08)
    
    plt.savefig(f'run_eb1/fc/field_{poly}.png', dpi=300)

def check_fc_ts_10d():
    import pandas as pd
    import matplotlib.pyplot as plt
    
    df = pd.read_csv('run_eb1/df.csv')
    TIME_COL  = 'time'
    VARIABLES = ['pv', 'npv', 'soil']
    
    # --- 1. Select first field and average over its pixels per date ---------------
    poly = df['poly_id'].iloc[0]
    ts = (
        df.loc[df['poly_id'] == poly, [TIME_COL] + VARIABLES]
          .assign(**{TIME_COL: lambda d: pd.to_datetime(d[TIME_COL])})
          .groupby(TIME_COL)[VARIABLES].mean()
          .sort_index()
    )
    
    # --- 2. Plot (1x1, 10-daily) --------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    
    for var in VARIABLES:
        ax.plot(ts.index, ts[var], 'o-', ms=3, lw=1.2, label=var)
    
    ax.set_title(f'poly_id {poly}: fractional cover (10-daily)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Fractional cover')
    ax.grid(alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    ax.legend(loc='upper left')
    fig.subplots_adjust(top=0.9, bottom=0.2, left=0.08, right=0.97)
    plt.savefig(f'run_eb1/fc/field_{poly}_10d.png', dpi=300)
    

####################################################### CHECK Training curves

def training_curve_xgb():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import xgboost as xgb
    from sklearn.metrics import r2_score, mean_absolute_error
    
    # --- Settings -----------------------------------------------------------------
    df = pd.read_csv('run_eb1/df.csv')
    df_ref = pd.read_csv('run_eb1/df_ref.csv')
    
    TIME_COL     = 'time'
    PIXEL_COL    = None                    # pixel ID column in df (e.g. 'pixel_id'); None = poly_id is the sampling point
    FEATURE_VARS = ['pv', 'npv', 'ei']     # adjust to your column names
    FRACTIONS    = np.arange(0.1, 0.81, 0.1)
    N_REPEATS    = 5                       # random draws per fraction and year
    SEED         = 42
    
    XGB_PARAMS = dict(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        n_jobs=-1,
    )
    
    # --- 1. Monthly predictors per sampling point and year ------------------------
    d = df.copy()
    d[TIME_COL] = pd.to_datetime(d[TIME_COL])
    d['month'] = d[TIME_COL].dt.month
    
    id_cols = ['poly_id'] + ([PIXEL_COL] if PIXEL_COL else []) + ['yr', 'lnf_code', 'tillage', 'region']
    
    monthly = d.groupby(id_cols + ['month'])[FEATURE_VARS].mean().unstack('month')
    monthly.columns = [f'{var}_m{month:02d}' for var, month in monthly.columns]
    monthly = monthly.reset_index()
    feature_cols = [c for c in monthly.columns if c not in id_cols]
    
    # --- 2. Attach reference C factor --------------------------------------------
    merge_cols = ['lnf_code', 'region', 'tillage']
    for c in merge_cols:
        monthly[c] = monthly[c].astype(str)
        df_ref[c] = df_ref[c].astype(str)
    
    ref = df_ref[merge_cols + ['C_ref']].drop_duplicates(subset=merge_cols)
    data = monthly.merge(ref, on=merge_cols, how='inner').dropna(subset=['C_ref'])
    
    data['point_id'] = data['poly_id'].astype(str)
    if PIXEL_COL:
        data['point_id'] += '_' + data[PIXEL_COL].astype(str)
    
    years = sorted(data['yr'].unique())
    print(f"{len(data)} rows, {data['point_id'].nunique()} sampling points, "
          f"{len(years)} years, {len(feature_cols)} predictors")
    
    # --- 3. LOYO CV with increasing training fractions ----------------------------
    def scores(obs, pred):
        return r2_score(obs, pred), mean_absolute_error(obs, pred)
    
    records = []
    for frac in FRACTIONS:
        for rep in range(N_REPEATS):
            rng = np.random.default_rng(SEED + rep)
            for test_yr in years:
                pool = data[data['yr'] != test_yr]
                test = data[data['yr'] == test_yr]
    
                points = pool['point_id'].unique()
                n_sel = max(1, int(round(frac * len(points))))
                sel = rng.choice(points, size=n_sel, replace=False)
                train = pool[pool['point_id'].isin(sel)]
    
                model = xgb.XGBRegressor(**XGB_PARAMS, random_state=SEED + rep)
                model.fit(train[feature_cols], train['C_ref'])
    
                r2_tr, mae_tr = scores(train['C_ref'], model.predict(train[feature_cols]))
                r2_te, mae_te = scores(test['C_ref'],  model.predict(test[feature_cols]))
    
                records.append(dict(frac=frac, rep=rep, test_yr=test_yr, n_train=len(train),
                                    r2_train=r2_tr, mae_train=mae_tr,
                                    r2_test=r2_te, mae_test=mae_te))
        print(f"fraction {frac:.0%} done")
    
    res = pd.DataFrame(records)
    
    # mean and std over all held-out years and repeats
    summary = res.groupby('frac').agg(['mean', 'std'])
    print(summary[['n_train', 'r2_train', 'r2_test', 'mae_train', 'mae_test']].round(4))
    
    # --- 4. Plot (1x2) ------------------------------------------------------------
    x = summary.index * 100
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, metric, label in zip(axes, ['mae', 'r2'], ['MAE', 'R²']):
        for split, style in [('train', 'o--'), ('test', 's-')]:
            mean = summary[(f'{metric}_{split}', 'mean')]
            std  = summary[(f'{metric}_{split}', 'std')]
            ax.plot(x, mean, style, label=split)
            ax.fill_between(x, mean - std, mean + std, alpha=0.2)
        ax.set_xlabel('Training sampling points used (%)')
        ax.set_ylabel(label)
        ax.set_title(f'{label} vs. training data size (LOYO CV)')
        ax.set_xticks(x)
        ax.grid(alpha=0.3)
        ax.legend()
    
    fig.subplots_adjust(top=0.9, bottom=0.12, left=0.07, right=0.98, wspace=0.22)
    plt.savefig(f'run_eb1/fc/xgb_LOOCV.png', dpi=300)


##############################################################Same as before but with MLP from Selene
def training_curve_mlp():
    import os
    import sys
    
    import joblib
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.base import clone
    from sklearn.metrics import r2_score, mean_absolute_error
    
    # WeightedMLPRegressor must be importable before unpickling (as in compute_cfactor_pixels_ml.load_model)
    CFACTOR_DIR = os.path.expanduser('~/mnt/eo-nas1/eoa-share/projects/028_Erosion/Erosion/cfactor')
    sys.path.insert(0, CFACTOR_DIR)
    import weighted_mlp  # noqa: F401,E402
    
    # --- Settings -----------------------------------------------------------------
    df = pd.read_csv('run_eb1/df.csv')
    df_ref = pd.read_csv('run_eb1/df_ref.csv')
    
    MODEL_PATH   = os.path.join(CFACTOR_DIR, 'calibration_analysis_mlp_strat/nn_model.joblib')
    TIME_COL     = 'time'
    PIXEL_COL    = None                    # pixel ID column in df (e.g. 'pixel_id'); None = poly_id is the sampling point
    FEATURE_VARS = ['pv', 'npv', 'ei']     # adjust to your column names
    FRACTIONS    = np.arange(0.1, 0.81, 0.1)
    N_REPEATS    = 5                       # random draws per fraction and year
    SEED         = 42
    
    # --- 0. Load MLP bundle (architecture + hyperparameters + feature layout) -----
    bundle = joblib.load(MODEL_PATH)
    pipe_template   = bundle['pipeline']           # StandardScaler + WeightedMLPRegressor
    feature_cols    = bundle['feature_columns']    # pv_t000.., npv_t000.., ei_t000..
    n_steps         = int(bundle['n_steps'])
    mlp_step        = pipe_template.steps[-1][0]   # 'mlp' (or 'ridge')
    print(f"Loaded {bundle.get('model_kind', '?')} model: {n_steps} timesteps, "
          f"{len(feature_cols)} predictors, params={bundle.get('effective_params')}")
    
    # --- 1. Per-timestep predictors per sampling point and year (no resampling) ---
    d = df.copy()
    d[TIME_COL] = pd.to_datetime(d[TIME_COL])
    
    id_cols = ['poly_id'] + ([PIXEL_COL] if PIXEL_COL else []) + ['yr', 'lnf_code', 'tillage', 'region']
    
    # Same layout as train_cfactor_ml.build_pixel_features: chronological index per series
    d = d.dropna(subset=FEATURE_VARS).sort_values(id_cols + [TIME_COL])
    d['t_idx'] = d.groupby(id_cols).cumcount()
    
    lengths = d.groupby(id_cols)['t_idx'].size()
    bad = lengths[lengths != n_steps]
    if len(bad):
        print(f"[WARN] {len(bad)}/{len(lengths)} series do not have {n_steps} timesteps "
              f"(min={lengths.min()}, max={lengths.max()}) — they are dropped.")
    d = d.merge(lengths[lengths == n_steps].rename('n').reset_index()[id_cols], on=id_cols)
    
    wide = d.pivot_table(index=id_cols, columns='t_idx', values=FEATURE_VARS)
    wide.columns = [f'{var}_t{int(t):03d}' for var, t in wide.columns]
    wide = wide.reset_index()
    
    missing = [c for c in feature_cols if c not in wide.columns]
    if missing:
        raise KeyError(f"Predictors expected by the model are missing: {missing[:5]} ...")
    
    # --- 2. Attach reference C factor --------------------------------------------
    merge_cols = ['lnf_code', 'region', 'tillage']
    for c in merge_cols:
        wide[c] = wide[c].astype(str)
        df_ref[c] = df_ref[c].astype(str)
    
    ref = df_ref[merge_cols + ['C_ref']].drop_duplicates(subset=merge_cols)
    data = wide.merge(ref, on=merge_cols, how='inner').dropna(subset=['C_ref'] + feature_cols)
    
    data['point_id'] = data['poly_id'].astype(str)
    if PIXEL_COL:
        data['point_id'] += '_' + data[PIXEL_COL].astype(str)
    
    years = sorted(data['yr'].unique())
    print(f"{len(data)} rows, {data['point_id'].nunique()} sampling points, "
          f"{len(years)} years, {len(feature_cols)} predictors")
    
    # --- 3. LOYO CV with increasing training fractions ----------------------------
    def scores(obs, pred):
        return r2_score(obs, pred), mean_absolute_error(obs, pred)
    
    records = []
    for frac in FRACTIONS:
        for rep in range(N_REPEATS):
            rng = np.random.default_rng(SEED + rep)
            for test_yr in years:
                pool = data[data['yr'] != test_yr]
                test = data[data['yr'] == test_yr]
    
                points = pool['point_id'].unique()
                n_sel = max(1, int(round(frac * len(points))))
                sel = rng.choice(points, size=n_sel, replace=False)
                train = pool[pool['point_id'].isin(sel)]
    
                # Fresh, untrained copy of the saved pipeline (same scaler + MLP settings)
                model = clone(pipe_template)
                if 'random_state' in model.named_steps[mlp_step].get_params():
                    model.set_params(**{f'{mlp_step}__random_state': SEED + rep})
                X_tr = train[feature_cols].to_numpy(dtype=np.float32)
                X_te = test[feature_cols].to_numpy(dtype=np.float32)
                model.fit(X_tr, train['C_ref'].values)
    
                r2_tr, mae_tr = scores(train['C_ref'], model.predict(X_tr))
                r2_te, mae_te = scores(test['C_ref'],  model.predict(X_te))
    
                records.append(dict(frac=frac, rep=rep, test_yr=test_yr, n_train=len(train),
                                    r2_train=r2_tr, mae_train=mae_tr,
                                    r2_test=r2_te, mae_test=mae_te))
        print(f"fraction {frac:.0%} done")
    
    res = pd.DataFrame(records)
    
    # mean and std over all held-out years and repeats
    summary = res.groupby('frac').agg(['mean', 'std'])
    print(summary[['n_train', 'r2_train', 'r2_test', 'mae_train', 'mae_test']].round(4))
    
    # --- 4. Plot (1x2) ------------------------------------------------------------
    x = summary.index * 100
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, metric, label in zip(axes, ['mae', 'r2'], ['MAE', 'R²']):
        for split, style in [('train', 'o--'), ('test', 's-')]:
            mean = summary[(f'{metric}_{split}', 'mean')]
            std  = summary[(f'{metric}_{split}', 'std')]
            ax.plot(x, mean, style, label=split)
            ax.fill_between(x, mean - std, mean + std, alpha=0.2)
        ax.set_xlabel('Training sampling points used (%)')
        ax.set_ylabel(label)
        ax.set_title(f'{label} vs. training data size (LOYO CV, MLP)')
        ax.set_xticks(x)
        ax.grid(alpha=0.3)
        ax.legend()
    
    fig.subplots_adjust(top=0.9, bottom=0.12, left=0.07, right=0.98, wspace=0.22)
    plt.savefig('run_eb1/fc/mlp_LOOCV.png', dpi=300)


############################## XGB LOYO as above but with HP tuning and less complex structure
def training_curve_xgb_hp():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import xgboost as xgb
    from scipy.stats import randint, uniform, loguniform
    from sklearn.model_selection import RandomizedSearchCV, GroupKFold
    from sklearn.metrics import r2_score, mean_absolute_error
    
    df = pd.read_csv('run_eb1/df.csv')
    df_ref = pd.read_csv('run_eb1/df_ref.csv')
    
    # --- Settings -----------------------------------------------------------------
    TIME_COL     = 'time'
    PIXEL_COL    = None                    # pixel ID column in df; None = poly_id is the sampling point
    FEATURE_VARS = ['pv', 'npv', 'ei']     # adjust to your column names
    CV_MODE      = 'loyo'                  # 'loyo' = leave-one-year-out, 'fixed' = one fixed 20% test set
    TEST_FRAC    = 0.2                     # only used for CV_MODE = 'fixed'
    FRACTIONS    = np.arange(0.1, 0.81, 0.1)
    N_REPEATS    = 3                       # random training draws per fraction (and year)
    N_ITER       = 20                      # hyperparameter combinations tried per training set
    INNER_FOLDS  = 3                       # grouped inner CV folds for tuning
    SEED         = 42
    
    # conservative base model: shallow trees, few estimators, strong regularisation
    BASE_PARAMS = dict(objective='reg:squarederror', tree_method='hist', n_jobs=1)
    
    PARAM_SPACE = {
        'n_estimators':     randint(30, 200),
        'learning_rate':    loguniform(0.02, 0.2),
        'max_depth':        randint(2, 5),          # 2–4
        'min_child_weight': randint(3, 30),         # min. samples (hessian) per leaf
        'subsample':        uniform(0.5, 0.3),      # 0.5–0.8
        'colsample_bytree': uniform(0.4, 0.4),      # 0.4–0.8
        'gamma':            loguniform(1e-4, 1e-2), # min. loss reduction for a split
        'reg_lambda':       loguniform(1, 50),      # L2
        'reg_alpha':        loguniform(1e-3, 1),    # L1
    }
    
    # --- 1. Monthly predictors per sampling point and year ------------------------
    d = df.copy()
    d[TIME_COL] = pd.to_datetime(d[TIME_COL])
    d['month'] = d[TIME_COL].dt.month
    
    id_cols = ['poly_id'] + ([PIXEL_COL] if PIXEL_COL else []) + ['yr', 'lnf_code', 'tillage', 'region']
    
    monthly = d.groupby(id_cols + ['month'])[FEATURE_VARS].mean().unstack('month')
    monthly.columns = [f'{var}_m{month:02d}' for var, month in monthly.columns]
    monthly = monthly.reset_index()
    feature_cols = [c for c in monthly.columns if c not in id_cols]
    
    # --- 2. Attach reference C factor --------------------------------------------
    merge_cols = ['lnf_code', 'region', 'tillage']
    for c in merge_cols:
        monthly[c] = monthly[c].astype(str)
        df_ref[c] = df_ref[c].astype(str)
    
    ref = df_ref[merge_cols + ['C_ref']].drop_duplicates(subset=merge_cols)
    data = monthly.merge(ref, on=merge_cols, how='inner').dropna(subset=['C_ref'])
    
    data['point_id'] = data['poly_id'].astype(str)
    if PIXEL_COL:
        data['point_id'] += '_' + data[PIXEL_COL].astype(str)
    
    all_points = data['point_id'].unique()
    print(f"{len(data)} rows, {len(all_points)} sampling points, "
          f"{data['yr'].nunique()} years, {len(feature_cols)} predictors")
    
    # --- 3. Outer splits ----------------------------------------------------------
    def outer_splits():
        """Yields (label, training pool, test set)."""
        if CV_MODE == 'loyo':
            for yr in sorted(data['yr'].unique()):
                yield yr, data[data['yr'] != yr], data[data['yr'] == yr]
        else:
            rng = np.random.default_rng(SEED)
            test_pts = rng.choice(all_points, size=int(round(TEST_FRAC * len(all_points))),
                                  replace=False)
            is_test = data['point_id'].isin(test_pts)
            yield 'fixed', data[~is_test], data[is_test]
    
    # --- 4. Tuning: randomized search with grouped inner CV -----------------------
    def tune_and_fit(train, seed):
        X, y, groups = train[feature_cols], train['C_ref'], train['point_id']
        n_groups = groups.nunique()
        model = xgb.XGBRegressor(**BASE_PARAMS, random_state=seed)
    
        if n_groups < 2:                          # too little data to tune
            model.set_params(n_estimators=50, max_depth=2, min_child_weight=3)
            return model.fit(X, y), {}
    
        search = RandomizedSearchCV(
            model, PARAM_SPACE, n_iter=N_ITER,
            cv=GroupKFold(n_splits=min(INNER_FOLDS, n_groups)),
            scoring='neg_mean_absolute_error',
            random_state=seed, n_jobs=-1, refit=True,
        )
        search.fit(X, y, groups=groups)
        print(search.best_estimator_.n_estimators)
        return search.best_estimator_, search.best_params_
    
    # --- 5. Learning curve --------------------------------------------------------
    def scores(obs, pred):
        return r2_score(obs, pred), mean_absolute_error(obs, pred)
    
    records = []
    for frac in FRACTIONS:
        for label, pool, test in outer_splits():
            pool_points = pool['point_id'].unique()
            # fractions refer to all sampling points (capped at the size of the pool)
            n_sel = min(len(pool_points), max(1, int(round(frac * len(all_points)))))
    
            for rep in range(N_REPEATS):
                rng = np.random.default_rng(SEED + 1 + rep)
                sel = rng.choice(pool_points, size=n_sel, replace=False)
                train = pool[pool['point_id'].isin(sel)]
    
                model, best = tune_and_fit(train, seed=SEED + rep)
    
                r2_tr, mae_tr = scores(train['C_ref'], model.predict(train[feature_cols]))
                r2_te, mae_te = scores(test['C_ref'],  model.predict(test[feature_cols]))
    
                records.append(dict(frac=frac, split=label, rep=rep, n_train=len(train),
                                    r2_train=r2_tr, mae_train=mae_tr,
                                    r2_test=r2_te, mae_test=mae_te, **best))
        print(f"fraction {frac:.0%} done")
    
    res = pd.DataFrame(records)
    res.to_csv('learning_curve_results.csv', index=False)
    
    metric_cols = ['n_train', 'r2_train', 'r2_test', 'mae_train', 'mae_test']
    summary = res.groupby('frac')[metric_cols].agg(['mean', 'std'])
    print(summary.round(4))
    
    # typical tuned hyperparameters per fraction
    param_cols = [c for c in PARAM_SPACE if c in res.columns]
    if param_cols:
        print("\nMedian tuned hyperparameters per fraction:")
        print(res.groupby('frac')[param_cols].median().round(3))
    
    # --- 6. Plot (1x2) ------------------------------------------------------------
    x = summary.index * 100
    title_cv = 'LOYO CV' if CV_MODE == 'loyo' else 'fixed 20% test set'
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, metric, label in zip(axes, ['mae', 'r2'], ['MAE', 'R²']):
        for split, style in [('train', 'o--'), ('test', 's-')]:
            mean = summary[(f'{metric}_{split}', 'mean')]
            std  = summary[(f'{metric}_{split}', 'std')]
            ax.plot(x, mean, style, label=split)
            ax.fill_between(x, mean - std, mean + std, alpha=0.2)
        ax.set_xlabel('Sampling points used for training (% of all)')
        ax.set_ylabel(label)
        ax.set_title(f'{label} vs. training data size ({title_cv}, tuned)')
        ax.set_xticks(x)
        ax.grid(alpha=0.3)
        ax.legend()
    
    fig.subplots_adjust(top=0.9, bottom=0.12, left=0.07, right=0.98, wspace=0.22)
    plt.savefig('run_eb1/fc/xgb_hp_LOOCV.png', dpi=300)


####################################### Histograms of c factors for referenz fields and entire CH

def hist_cfactors():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    
    # --- Settings -----------------------------------------------------------------
    PARZ_CSV   = 'C_Faktoren_Parzellen.csv'
    DF_CSV     = 'run_eb1/df.csv'
    REF_CSV    = 'run_eb1/df_ref.csv'
    PARZ_C_COL = 'C_Fakt'          # name of the C factor column in C_Faktoren_Parzellen.csv
    PIXEL_COL  = None                # pixel ID column in df.csv; None = poly_id is the sampling point
    N_BINS     = 40
    
    # --- 1. C factors from C_Faktoren_Parzellen.csv -------------------------------
    parz = pd.read_csv(PARZ_CSV, sep=';', decimal=',', encoding='latin-1')
    c_parz = pd.to_numeric(parz[PARZ_C_COL], errors='coerce').dropna().to_numpy()
    
    # --- 2. C factors for df.csv, mapped from df_ref.csv (as for XGB) -------------
    df     = pd.read_csv(DF_CSV)
    df_ref = pd.read_csv(REF_CSV)
    
    # one row per sampling point and year (df holds full time series)
    id_cols = ['poly_id'] + ([PIXEL_COL] if PIXEL_COL else []) + ['yr', 'lnf_code', 'tillage', 'region']
    samples = df[id_cols].drop_duplicates()
    
    merge_cols = ['lnf_code', 'region', 'tillage']
    for c in merge_cols:
        samples[c] = samples[c].astype(str)
        df_ref[c] = df_ref[c].astype(str)
    
    ref = df_ref[merge_cols + ['C_ref']].drop_duplicates(subset=merge_cols)
    samples = samples.merge(ref, on=merge_cols, how='inner')
    c_df = samples['C_ref'].dropna().to_numpy()
    
    # --- 3. Summary ---------------------------------------------------------------
    def describe(c):
        return {'n': len(c), 'mean': c.mean(), 'median': np.median(c), 'std': c.std(),
                'min': c.min(), 'max': c.max()}
    
    sets = {'C_Faktoren_Parzellen': c_parz, 'df (mapped C_ref)': c_df}
    print(pd.DataFrame({k: describe(v) for k, v in sets.items()}).T.round(4))
    
    def standardise(c):
        return (c - c.mean()) / c.std()
    
    # --- 4. Plot (2x2) ------------------------------------------------------------
    C_THRESHOLD = 0.005
    
    sets_filt = {name: vals[vals >= C_THRESHOLD] for name, vals in sets.items()}
    for name in sets:
        n_removed = len(sets[name]) - len(sets_filt[name])
        print(f"{name}: {n_removed} values < {C_THRESHOLD} removed "
              f"({n_removed / len(sets[name]):.1%})")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = ['tab:blue', 'tab:orange']
    
    rows = [
        (sets,      'all values'),
        (sets_filt, f'C ≥ {C_THRESHOLD}'),
    ]
    
    for row, (data, subset_label) in enumerate(rows):
        all_vals = np.concatenate(list(data.values()))
        bins = np.linspace(all_vals.min(), all_vals.max(), N_BINS + 1)   # shared bins per row
    
        for col, relative in enumerate([False, True]):
            ax = axes[row, col]
            for (name, vals), color in zip(data.items(), colors):
                weights = np.full(len(vals), 100 / len(vals)) if relative else None
                ax.hist(vals, bins=bins, weights=weights, alpha=0.5, color=color,
                        edgecolor='white', linewidth=0.5, label=f'{name} (n = {len(vals)})')
            ax.set_xlabel('C factor')
            ax.set_ylabel('Share of samples (%)' if relative else 'Count')
            kind = 'Relative frequency' if relative else 'Absolute frequency'
            ax.set_title(f'{kind}, {subset_label}')
            ax.grid(alpha=0.3)
            ax.legend()
    
    fig.subplots_adjust(top=0.94, bottom=0.07, left=0.07, right=0.98, wspace=0.2, hspace=0.3)
    plt.savefig(f'run_eb1/fc/c_factor_comp.png', dpi=300)


def plot_cfact_spatially():
    import glob
    import os
    import re
     
    import numpy as np
    import pandas as pd
    import geopandas as gpd
    import xarray as xr
    import matplotlib.pyplot as plt

    # --- Settings -----------------------------------------------------------------
    GPKG_PATH     = '/mnt/Data-Labo-RE/27_Natural_Resources-RE/321.2_AUI_Monitoring_protected/Georeferenzierung/Georeferenzierung_AUI_buem.gpkg'
    FARM_COL      = 'farmname'
    FARM_NAME     = 'Ferme Bardet'

    S2_GRID_PATH  = '/mnt/eo-nas1/eoa-share/projects/012_EO_dataInfrastructure/Project layers/gridface_s2tiles_CH.shp'
    S2_DIR        = '/mnt/eo-nas1/data/satellite/sentinel2/raw/CH'

    YEAR          = 2016                 # year from which to take the scene

    # zarr stores are named S2_<left>_<bottom>_<YYYYMMDD>_<YYYYMMDD>.zarr,
    # left/bottom = lower-left corner of the tile in the CRS of the grid shapefile
    ZARR_REGEX    = r'S2_(?P<left>-?\d+(?:\.\d+)?)_(?P<bottom>-?\d+(?:\.\d+)?)_(?P<start>\d{8})_(?P<end>\d{8})\.zarr$'
    COORD_TOL     = 1.0                  # tolerance when matching left/bottom to the grid (map units)
    
    NODATA     = 65535                    # no-data value of bands and SCL
    X_DIM, Y_DIM, TIME_DIM = 'lon', 'lat', 'time'
    RGB_BANDS  = ['s2_B04', 's2_B03', 's2_B02']   # red, green, blue
    BAND_DIM   = None                    # None = one variable per band; else name of the band dimension
    DATA_VAR   = None                    # only needed if BAND_DIM is set (variable holding all bands)
    DATA_CRS   = None                    # e.g. 'EPSG:32632'; None = read from zarr attributes
    
    # cloud cover: a per-date coordinate/variable in %, or computed from the SCL band
    CLOUD_COORD = 'cloud_cover'          # used if present in the zarr store
    SCL_VAR     = 's2_SCL'                  # fallback: scene classification layer
    SCL_CLOUDY  = [3, 8, 9, 10, 11]          # cloud shadow, cloud medium/high probability, cirrus
    MAX_CLOUD   = 10                     # %
    DATE_CHOICE = 'least_cloudy'         # 'least_cloudy' or 'latest' (within YEAR)

    CLOUD_METHOD = 'auto'      # 'scl', 'brightness', or 'auto' (SCL if present, else brightness)
    BLUE_CLOUD   = 0.20        # blue reflectance above this counts as cloud (brightness method)
    MIN_COVERAGE = 95          # % of the area that must have valid data
    
    BUFFER_M    = 200                    # margin around the farm polygons (map units of the S2 data)
    
    # --- 1. Farm polygons ---------------------------------------------------------
    farm = gpd.read_file(GPKG_PATH)
    farm = farm[farm[FARM_COL] == FARM_NAME]
    if farm.empty:
        raise ValueError(f"No polygons with {FARM_COL} == '{FARM_NAME}'.")
    print(f"{len(farm)} polygons for farm '{FARM_NAME}'")
    
    # --- 2. Sentinel-2 tiles covering the farm -----------------------------------
    grid = gpd.read_file(S2_GRID_PATH)
    grid['tile_left'] = grid.geometry.bounds['minx']     # right edge (name kept for simplicity)
    grid['tile_bottom'] = grid.geometry.bounds['maxy']   # top edge
    
    farm_g = farm[['geometry']].to_crs(grid.crs).dissolve()
    hits = gpd.overlay(grid[['tile_left', 'tile_bottom', 'geometry']], farm_g, how='intersection')
    if hits.empty:
        raise ValueError("The farm does not intersect any tile in the S2 grid.")
    hits['area'] = hits.geometry.area
    hits = hits.sort_values('area', ascending=False).reset_index(drop=True)
    print(f"Farm intersects {len(hits)} tile(s), lower-left corners: "
          f"{list(zip(hits['tile_left'].round(1), hits['tile_bottom'].round(1)))}")
    
    # parse all zarr store names
    stores = []
    for path in glob.glob(os.path.join(S2_DIR, 'S2_*.zarr')):
        m = re.search(ZARR_REGEX, os.path.basename(path.rstrip('/')))
        if m:
            stores.append(dict(path=path, left=float(m['left']), bottom=float(m['bottom']),
                               start=pd.to_datetime(m['start'], format='%Y%m%d'),
                               end=pd.to_datetime(m['end'], format='%Y%m%d')))
    stores = pd.DataFrame(stores)
    if stores.empty:
        raise FileNotFoundError(f"No zarr stores matching the naming scheme found in {S2_DIR}.")
    
    # stores of the intersecting tiles whose time span overlaps YEAR
    year_start, year_end = pd.Timestamp(f'{YEAR}-01-01'), pd.Timestamp(f'{YEAR}-12-31')
    candidates = []
    for _, t in hits.iterrows():
        sel = stores[(abs(stores['left'] - t['tile_left']) <= COORD_TOL) &
                     (abs(stores['bottom'] - t['tile_bottom']) <= COORD_TOL) &
                     (stores['start'] <= year_end) & (stores['end'] >= year_start)]
        for _, st in sel.iterrows():
            candidates.append(dict(st, overlap=t['area']))
    candidates = pd.DataFrame(candidates)
    if candidates.empty:
        raise FileNotFoundError(f"No zarr store for the farm's tile(s) covering {YEAR}. "
                                f"Check COORD_TOL and that left/bottom use the grid's CRS.")
    print(f"{len(candidates)} zarr store(s) for {YEAR}:")
    for p_ in candidates['path']:
        print("  ", p_)
    
    # --- 3. CRS of the S2 data ----------------------------------------------------
    def get_crs(ds):
        if DATA_CRS:
            return DATA_CRS
        for src in [ds.attrs] + [ds[v].attrs for v in ds.variables]:
            for key in ('crs', 'crs_wkt', 'spatial_ref', 'epsg'):
                if key in src:
                    val = src[key]
                    return f'EPSG:{val}' if isinstance(val, (int, np.integer)) else val
        raise ValueError("CRS not found in the zarr store, please set DATA_CRS.")
    
    # --- 4. Cloud cover per date over the farm extent, across all stores ----------
    def farm_subset(ds):
        ds = ds.rio.write_crs(32632)
        crs = get_crs(ds)
        fd = farm.to_crs(crs)
        x0, y0, x1, y1 = fd.total_bounds
        bounds = (x0 - BUFFER_M, y0 - BUFFER_M, x1 + BUFFER_M, y1 + BUFFER_M)
        desc = bool(ds[Y_DIM][0] > ds[Y_DIM][-1])
        sub = ds.sel({X_DIM: slice(bounds[0], bounds[2]),
                      Y_DIM: slice(bounds[3], bounds[1]) if desc else slice(bounds[1], bounds[3])})
        sub = sub.sel({TIME_DIM: slice(str(year_start.date()), str(year_end.date()))})
        return sub, fd, bounds, desc
    
    def cloud_cover(sub):
        """One row per time entry: position, date, cloud % (of valid pixels), coverage %."""
        blue = sub[RGB_BANDS[2]]
        valid = blue.notnull() & (blue > 0) & (blue != NODATA)
        n_all = sub.sizes[X_DIM] * sub.sizes[Y_DIM]
    
        method = CLOUD_METHOD
        if method == 'auto':
            method = 'scl' if SCL_VAR in sub.variables else 'brightness'
    
        if method == 'scl':
            scl = sub[SCL_VAR]
            valid = valid & (scl != NODATA) & (scl > 0)
            cloudy = scl.isin(SCL_CLOUDY) & valid
        else:
            cloudy = (blue / 10000 > BLUE_CLOUD) & valid
    
        n_valid = valid.sum([X_DIM, Y_DIM]).compute().values
        n_cloud = cloudy.sum([X_DIM, Y_DIM]).compute().values
        with np.errstate(invalid='ignore', divide='ignore'):
            cloud = np.where(n_valid > 0, 100 * n_cloud / n_valid, np.nan)
        return pd.DataFrame({
            't_idx':    np.arange(sub.sizes[TIME_DIM]),
            'date':     pd.to_datetime(sub[TIME_DIM].values),
            'cloud':    cloud,
            'coverage': 100 * n_valid / n_all,
        }), method
    
    options = []
    for _, c in candidates.iterrows():
        ds_c = xr.open_zarr(c['path'])
        sub_c, fd_c, bounds_c, desc_c = farm_subset(ds_c)
        if sub_c.sizes[X_DIM] == 0 or sub_c.sizes[Y_DIM] == 0 or sub_c.sizes[TIME_DIM] == 0:
            continue
        cc, cloud_src = cloud_cover(sub_c)
        for _, row in cc.dropna(subset=['cloud']).iterrows():
            options.append(dict(date=row['date'], t_idx=int(row['t_idx']),
                                cloud=row['cloud'], coverage=row['coverage'],
                                overlap=c['overlap'], path=c['path'],
                                tile_key=(c['left'], c['bottom']),
                                sub=sub_c, farm_d=fd_c, bounds=bounds_c, y_desc=desc_c))
    
    options = pd.DataFrame(options)
    if options.empty:
        raise ValueError(f"No data over the farm extent in {YEAR} (check DATA_CRS).")
    
    clear = options[(options['cloud'] < MAX_CLOUD) & (options['coverage'] >= MIN_COVERAGE)]
    if clear.empty:
        raise ValueError(f"No date in {YEAR} with < {MAX_CLOUD} % cloud cover "
                         f"(minimum available: {options['cloud'].min():.1f} %).")
    
    # best scene per tile: least cloudy (or latest), ties -> later date
    sort_cols, asc = (['cloud', 'date'], [True, False]) if DATE_CHOICE == 'least_cloudy' \
        else (['date'], [False])
    chosen = (clear.sort_values(sort_cols, ascending=asc)
                   .groupby('tile_key', sort=False).head(1)
                   .sort_values('overlap'))          # largest overlap drawn last (on top)
    
    missing = set(options['tile_key']) - set(chosen['tile_key'])
    if missing:
        print(f"Warning: no scene < {MAX_CLOUD} % clouds in {YEAR} for tile(s) {missing}")
    
    for _, r in chosen.iterrows():
        print(f"  {os.path.basename(r['path'].rstrip('/'))}: "
              f"{r['date'].date()} ({r['cloud']:.1f} % clouds)")
    
    farm_d = chosen.iloc[-1]['farm_d']
    minx, miny, maxx, maxy = chosen.iloc[-1]['bounds']
    
    # --- 5. RGB composite per tile ------------------------------------------------
    def raw_rgb(r):
        scene = r['sub'].isel({TIME_DIM: r['t_idx']})     # exactly the selected entry
        if BAND_DIM:
            arr = scene[DATA_VAR].sel({BAND_DIM: RGB_BANDS}).transpose(Y_DIM, X_DIM, BAND_DIM).values
        else:
            arr = np.stack([scene[b].values for b in RGB_BANDS], axis=-1)
        arr = arr.astype(float)
        arr[(arr <= 0) | (arr == NODATA)] = np.nan         # no-data                        # no-data
    
        xs, ys = scene[X_DIM].values, scene[Y_DIM].values
        rx = abs(xs[1] - xs[0]) if len(xs) > 1 else 10
        ry = abs(ys[1] - ys[0]) if len(ys) > 1 else 10
        extent = [xs.min() - rx / 2, xs.max() + rx / 2, ys.min() - ry / 2, ys.max() + ry / 2]
        return arr, extent
    
    tiles = [(raw_rgb(r), r['y_desc']) for _, r in chosen.iterrows()]
    
    # one common stretch for all tiles, so colours match across tile borders
    lo, hi = [], []
    for i in range(3):
        vals = np.concatenate([arr[..., i].ravel() for (arr, _), _ in tiles])
        p_lo, p_hi = np.nanpercentile(vals, [2, 98])
        lo.append(p_lo); hi.append(p_hi)
    
    def stretch(arr):
        out = np.dstack([np.clip((arr[..., i] - lo[i]) / (hi[i] - lo[i]), 0, 1) for i in range(3)])
        alpha = (~np.isnan(arr).any(axis=-1)).astype(float)   # no-data transparent
        return np.dstack([np.nan_to_num(out), alpha])
    
    # --- 6. Plot ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 10))
    for (arr, extent), desc in tiles:
        ax.imshow(stretch(arr), extent=extent, origin='upper' if desc else 'lower')
    farm_d.boundary.plot(ax=ax, color='yellow', linewidth=1.5)
    
    dates = ', '.join(sorted({r['date'].strftime('%Y-%m-%d') for _, r in chosen.iterrows()}))
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_title(f"Farm '{FARM_NAME}' ({len(farm)} polygons)\n"
                 f"Sentinel-2, {len(chosen)} tile(s), {dates}")
    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')
    ax.ticklabel_format(style='plain', useOffset=False)
    fig.subplots_adjust(top=0.9, bottom=0.07, left=0.1, right=0.97)
    plt.savefig(f'run_eb1/fc/spatial_cfact_bardet_corr.png', dpi=300)

def check_cloud():
    import numpy as np
    import geopandas as gpd
    import xarray as xr
    import os
        
    GPKG_PATH     = '/mnt/Data-Labo-RE/27_Natural_Resources-RE/321.2_AUI_Monitoring_protected/Georeferenzierung/Georeferenzierung_AUI.gpkg'
    FARM_COL      = 'farmname'
    FARM_NAME     = 'Ferme Bardet'

    ZARR_PATH = '/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_400860_5182380_20160103_20161231.zarr'   # one store covering the farm
    DATA_CRS = 'EPSG:32632'
    X_DIM, Y_DIM, TIME_DIM = 'lon', 'lat', 'time'
    BLUE, SCL_VAR = 's2_B02', 's2_SCL'
    D_CLOUDY = '2016-07-15'                     # a date with visible clouds
    
    ds = xr.open_zarr(ZARR_PATH)
    print(ds)                                   # which variables exist?
    
    # subset to the farm extent
    farm = gpd.read_file(GPKG_PATH)
    farm = farm[farm[FARM_COL] == FARM_NAME].to_crs(DATA_CRS)
    minx, miny, maxx, maxy = farm.total_bounds
    y_desc = bool(ds[Y_DIM][0] > ds[Y_DIM][-1])
    sub = ds.sel({X_DIM: slice(minx, maxx),
                  Y_DIM: slice(maxy, miny) if y_desc else slice(miny, maxy)})
    print(sub.sizes)                            # x/y size 0 = farm not in this store
    
    times = sub[TIME_DIM].to_index()
    print(times.duplicated().sum(), "duplicate time stamps out of", len(times))
    
    target = np.datetime64(D_CLOUDY)
    i_near = int(np.argmin(np.abs(times.values - target)))       # nearest date
    same_day = np.where(times.values == times.values[i_near])[0]  # all entries on that date
    print(f"{len(same_day)} acquisition(s) on {str(times.values[i_near])[:10]}")
    
    for i in same_day:                                            # check each acquisition
        s = sub.isel({TIME_DIM: i})
        print(f"\n--- entry {i} ---")
        if SCL_VAR in s.variables:
            vals, counts = np.unique(s[SCL_VAR].values, return_counts=True)
            print("SCL classes:", dict(zip(vals, counts)))
        else:
            print(f"No '{SCL_VAR}' variable in the store")
        print(float(s[BLUE].max()), "max of blue band")
        print(int(s[BLUE].isnull().sum()), "NaN pixels,", int((s[BLUE] <= 0).sum()), "pixels <= 0")


def samples_from_gpkg(gpkg_path, layer=None,
                      id_col='betr_ID', yr_col='yr', code_col='lnf_code',
                      target_crs=32632):
    """Build the samples pickle expected by extract_s2_data_field from field polygons."""
    g = gpd.read_file(gpkg_path, layer=layer).to_crs(target_crs)
    print(g)
    g = g.rename(columns={id_col: 'poly_id', yr_col: 'yr', code_col: 'lnf_code'})
    g['poly_id'] = g['poly_id'].astype('int64')
    g['yr'] = g['yr'].astype(int).astype(str)          # same dtype as the original pipeline
    g = g[g.geometry.notnull() & ~g.geometry.is_empty & g.geometry.is_valid]

    dup = g.duplicated(['poly_id', 'yr'])
    if dup.any():
        raise ValueError(f"{dup.sum()} duplicated (poly_id, yr) rows in {gpkg_path}")

    # Sample point 10 m inside the field (as in the samplers); fall back to the
    # full polygon for fields too narrow for the inner buffer.
    inner = g.geometry.buffer(-10)
    inner = inner.where(~inner.is_empty, g.geometry)
    pts = inner.representative_point()

    samples = gpd.GeoDataFrame(g.drop(columns='geometry'), geometry=pts.values,
                               crs=target_crs).rename_geometry('point_geom')
    samples['polygon_geom'] = g.geometry.values
    samples['x'] = samples.point_geom.x
    samples['y'] = samples.point_geom.y
    for c in ('uuid', 'betr_ID', 'tillage_class'):
        if c not in samples.columns:
            samples[c] = np.nan
    print(samples.head())
    print(f"{len(samples)} field-years, years {sorted(samples['yr'].unique())}")
    return samples

def check_s1():
    import xarray as xr
    import matplotlib.pyplot as plt
    import numpy as np
    
    ds = xr.open_zarr("/mnt/eo-nas1/data/satellite/sentinel1/raw/CH/S1_586460_5140140_20220101_20221230.zarr", group="asc")
    
    # Every 2nd timestep, capped at 30 panels for a 5x6 grid
    # sub = ds[["s1_vh", "s1_vv"]].isel(time=slice(None, None, 2)).isel(time=slice(0, 30)).load()
    sub = ds[["s1_vh", "s1_vv"]].load()
    
    # Convert to dB if linear
    if float(sub["s1_vv"].mean()) > 0:
        sub = 10 * np.log10(sub.where(sub > 0))
    
    cr_db = sub["s1_vh"]
    
    g = cr_db.plot(
        x="lon", y="lat",
        col="time", col_wrap=10,          # 10 columns → 6 rows for 60 timesteps
        cmap="viridis",
        robust=True,
        figsize=(24, 16),
        cbar_kwargs={"label": "VH [dB]", "shrink": 0.6},
    )
    
    for ax, t in zip(g.axs.flat, cr_db.time.values):
        ax.set_title(str(t)[:10], fontsize=9)
        ax.set_xlabel(""); ax.set_ylabel("")
        ax.set_aspect("equal")
    
    plt.savefig("test/s1_asc_vh_camp.png", dpi=300)


"""
XGBoost C-factor estimation with leave-one-year-out (LOYO) cross-validation.

Inputs
  X: samples_data_gpr_aui.parquet  time series, one row per poly_id / yr / date, column fc_total
  y: samples_aui.pkl               one row per poly_id / yr, column C_Fakt

Steps
  1. Resample fc_total to monthly means and pivot to one column per month of the crop year,
     in chronological order (e.g. fc_prev_10, fc_prev_11, fc_prev_12, fc_cur_01 ... fc_cur_09)
  2. Merge with C_Fakt on poly_id + yr  ->  model table (C_Fakt is the last column)
  3. LOYO CV: each year is held out once, model trained on all other years
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut
from xgboost import XGBRegressor

# ---------------------------------------------------------------- config
X_FILE = 'samples_data_gpr_aui.parquet'
Y_FILE = 'samples_aui.pkl'
DATE_COL = 'time'          # sampling date
# yr is the crop (target) year; months are labelled relative to it:
# 'prev' = calendar year before yr, 'cur' = calendar year yr, 'next' = year after
YEAR_LABEL = {-1: 'prev', 0: 'cur', 1: 'next'}
KEYS = ['poly_id', 'yr']
FEATURE = 'fc_total'
TARGET = 'C_Fakt'

XGB_PARAMS = dict(
    n_estimators=50,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
)


# ---------------------------------------------------------------- data
def build_monthly_table():
    ts = pd.read_parquet(X_FILE, columns=KEYS + [DATE_COL, FEATURE])
    ts[DATE_COL] = pd.to_datetime(ts[DATE_COL])
    # month position relative to the crop year, e.g. yr=2020, Oct 2019 -> -2, Mar 2020 -> 3
    ts['rel_month'] = (ts[DATE_COL].dt.year - ts['yr'].astype(int)) * 12 + ts[DATE_COL].dt.month

    # monthly mean per poly_id / yr, then one column per month in chronological order
    monthly = (
        ts.groupby(KEYS + ['rel_month'])[FEATURE].mean()
          .unstack('rel_month')
          .sort_index(axis=1)
    )

    def col_name(rel):
        year_offset, month = divmod(rel - 1, 12)
        label = YEAR_LABEL.get(year_offset, f'y{year_offset:+d}')
        return f'fc_{label}_{month + 1:02d}'

    print(f'Crop-year months found: {col_name(monthly.columns.min())} ... '
          f'{col_name(monthly.columns.max())} ({monthly.shape[1]} columns)')
    monthly.columns = [col_name(c) for c in monthly.columns]
    monthly = monthly.reset_index()

    y = pd.read_pickle(Y_FILE)[KEYS + [TARGET]]

    table = monthly.merge(y, on=KEYS, how='inner', validate='one_to_one')
    print(f'X poly-years: {len(monthly)}, y rows: {len(y)}, merged: {len(table)}')

    n_missing = table.filter(like='fc_').isna().sum().sum()
    if n_missing:
        print(f'Note: {n_missing} missing monthly values (XGBoost handles NaN natively)')
    return table


# ---------------------------------------------------------------- CV
def loyo_cv(table):
    feat_cols = [c for c in table.columns if c.startswith('fc_')]
    X, y, groups = table[feat_cols], table[TARGET], table['yr']

    oof = np.full(len(table), np.nan)
    fold_rows = []

    for train_idx, test_idx in LeaveOneGroupOut().split(X, y, groups):
        year = groups.iloc[test_idx[0]]
        model = XGBRegressor(**XGB_PARAMS)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])

        pred = model.predict(X.iloc[test_idx])
        oof[test_idx] = pred
        y_true = y.iloc[test_idx]

        fold_rows.append({
            'year': year,
            'n_test': len(test_idx),
            'rmse': np.sqrt(mean_squared_error(y_true, pred)),
            'mae': mean_absolute_error(y_true, pred),
            'r2': r2_score(y_true, pred) if len(test_idx) > 1 else np.nan,
        })
        print(f"{year}: n={len(test_idx):5d}  RMSE={fold_rows[-1]['rmse']:.5f}  "
              f"MAE={fold_rows[-1]['mae']:.5f}  R2={fold_rows[-1]['r2']:.3f}")

    folds = pd.DataFrame(fold_rows)
    print('\nOverall (pooled out-of-fold predictions):')
    print(f'  RMSE = {np.sqrt(mean_squared_error(y, oof)):.5f}')
    print(f'  MAE  = {mean_absolute_error(y, oof):.5f}')
    print(f'  R2   = {r2_score(y, oof):.3f}')

    preds = table[KEYS + [TARGET]].copy()
    preds['C_pred'] = oof
    return folds, preds, feat_cols

def _norm_id(s):
    """Normalise a farm id to a clean integer-string so the AGIS<->VERFAHREN
    merge is robust to int/float/str representations (e.g. 10000000.0 vs
    10000000 vs '10000000')."""
    num = pd.to_numeric(s, errors='coerce')
    if num.notna().all():
        return num.astype('Int64').astype(str)
    return s.astype(str) 

"""
Typical tillage / sowing date per LNF crop (Crop_DE) from massnahmen.txt.

1. keep tillage measures (VALID_MASSNAHMEN)
2. map the crop names of the txt file to the Crop_DE names of the LNF table
   (manual mapping TXT_TO_LNF, several txt names can map to one LNF crop)
3. median date per LNF crop, computed on a season axis starting 1 August, so
   autumn (winter crops) and spring dates are not mixed up across the new year
4. LNF crops without records: derived from a closely related crop (DERIVED)
   or from a literature estimate (LITERATURE)

Output: CSV with one row per LNF crop: median date (MM-DD), 25 / 75 % quantile,
number of records, source (data / derived / literature / none).
"""

import numpy as np
import pandas as pd

MASSN_FILE = "/mnt/Data-Labo-RE/27_Natural_Resources-RE/321.4_WAUM_protected/Daten/Feldkalender_AUI/Feldkalender_2022.txt"
KULTUR_COL = "kultur"                     # crop name column in the txt file
LNF_TABLE = "/mnt/eo-nas1/data/landuse/documentation/LNF_code_classification_20260217.xlsx"
LNF_SHEET = "label_sheet"
OUT_CSV = "tillage_dates_per_crop.csv"
OUT_CSV_MEASURE = "tillage_dates_per_crop_and_measure.csv"

VALID_MASSNAHMEN = ["Mulchsaat", "Direktsaat", "Strip-Till", "Pflügen", "Mulchen"]

# ---------------------------------------------------------------------------
# txt crop name -> LNF Crop_DE
# ---------------------------------------------------------------------------
W = "Winterweizen (ohne Futterweizen der Sortenliste swiss granum)"
S = "Sommerweizen (ohne Futterweizen der Sortenliste swiss granum)"
VEG = "Einjährige Freilandgemüse, ohne Konservengemüse"
CAN = "Freiland-Konservengemüse"
KW = "Kunstwiesen (ohne Weiden)"

TXT_TO_LNF = {
    # cereals
    "Winterweizen": W,
    "Sommerweizen": S,
    "Futterweizen": "Futterweizen gemäss Sortenliste swiss granum",
    "Wintergerste": "Wintergerste",
    "Sommergerste": "Sommergerste",
    "Winterhafer": "Hafer",
    "Sommerhafer": "Hafer",
    "Wintertriticale": "Triticale",
    "Winterroggen (Hybridsorten)": "Roggen",
    "Winterroggen (Populationssorten)": "Roggen",
    "Dinkel (Winterkorn)": "Dinkel",
    "Winteremmer/-einkorn": "Emmer, Einkorn",
    "Getreide siliert": "Getreide siliert",
    "Getreide siliert mit Leguminosen": "Getreide siliert",
    # maize, millet, sorghum
    "Körnermais": "Körnermais",
    "Lieschkolbenschrot": "Körnermais",          # corn-cob mix = grain maize
    "Silomais": "Silo- und Grünmais",
    "Grünmais": "Silo- und Grünmais",
    "Saatmais": "Saatmais (Vertragsanbau)",
    "Hirse": "Hirse",
    "Sorghum": "Sorghum",
    "Buchweizen": "Buchweizen",
    # root crops
    "Zuckerrüben": "Zuckerrüben",
    "Futterrüben": "Futterrüben",
    "Speisekartoffeln": "Kartoffeln",
    "Industriekartoffeln": "Kartoffeln",
    "Frühkartoffeln": "Kartoffeln",
    "Futterkartoffeln": "Kartoffeln",
    "Saatkartoffeln": "Pflanzkartoffeln (Vertragsanbau)",
    # oil crops
    "Winterraps für Speiseöl": "Winterraps zur Speiseölgewinnung",
    "Sommerraps für Speiseöl": "Sommerraps zur Speiseölgewinnung",
    "Winterraps als NWR": "Winterraps als nachwachsender Rohstoff",
    "Sonnenblumen für Speiseöl": "Sonnenblumen zur Speiseölgewinnung",
    "Soja": "Soja",
    "Sommerlein (Ölgewinn.)": "Lein",
    "Kernenkürbis": "Ölkürbisse",
    # grain legumes
    "Sommerackerbohnen": "Bohnen und Wicken zur Körnergewinnung (z.B. Ackerbohnen) ",
    "Winterackerbohnen": "Bohnen und Wicken zur Körnergewinnung (z.B. Ackerbohnen) ",
    "Sommereiweisserbsen": "Erbsen zur Körnergewinnung (z.B. Eiweisserbsen)",
    "Wintereiweisserbsen": "Erbsen zur Körnergewinnung (z.B. Eiweisserbsen)",
    "Süsslupinen": "Lupinen",
    # vegetables
    "Freilandgemüse": VEG, "Zwiebeln": VEG, "Winterzwiebel": VEG, "Kabis Lager": VEG,
    "Kürbis": VEG, "Blumenkohl": VEG, "Winterspinat": VEG,
    "Spinat, n. überw.,1 Schnitt, Saat nach Mitte April": VEG,
    "Spinat, nicht überwinternd, 2 Schnitte": VEG,
    "Karotten, Lager-, hoher Ertrag": VEG, "Karotten, Lager-, mittlerer Ertrag": VEG,
    "Karotten Bund-, Früh-": VEG, "Karotten Pariser": VEG, "Rüben Herbst-, Mai-": VEG,
    "Rucola, ein Schnitt (G)": VEG, "Schnittsalat": VEG,
    "Salate, diverse, mittlerer Ertrag": VEG, "Zucchetti, Patisson (G)": VEG,
    "Wirz leicht": VEG, "Broccoli": VEG, "Lauch": VEG, "Sellerie, Knollen": VEG,
    "Kohlrabi": VEG, "Radies, 20 Bund/m2 (G)": VEG, "Randen": VEG, "Pastinaken": VEG,
    "Bohnen, Busch-": VEG,
    "Bohnen, Verarbeitung-": CAN, "Essiggurken": CAN,
    "Cichorien, Treib-": "Wurzeln der Treibzichorie",
    # other annual crops
    "Erdbeeren, einjährig, Ertrag mittel": "Einjährige Beeren (z.B. Erdbeeren)",
    "Kräuter, Gewürze (mittelgross)": "Einjährige Gewürz- und Medizinalpflanzen",
    "Schnittblumen, mittel": "Einjährige gärtnerische Freilandkulturen (Blumen, Rollrasen usw.)",
    # biodiversity areas on arable land
    "Buntbrache": "Buntbrache",
    "Rotationsbrache": "Rotationsbrache",
    "Ackerschonstreifen": "Ackerschonstreifen",
    "Saum auf Ackerflächen": "Saum auf Ackerflächen",
    "Blühstreifen für Bestäuber und andere Nützlinge": "Nützlingsstreifen auf offener Ackerfläche",
    # unlisted arable crops
    "Nicht aufgef. Kultur, Leguminosen": "Übrige offene Ackerfläche, beitragsberechtigt",
    "Nicht aufgef. Kultur, Nicht-Leg, Mischungen von Leg. u. Nicht-Leg.":
        "Übrige offene Ackerfläche, beitragsberechtigt",
    # temporary grassland (tillage = sowing / renewal)
    "Kunstwiese/Mähweide intensiv": KW,
    "Kunstwiese/Mähweide mittelintensiv": KW,
    "Kunstwiese/Mähweide wenig intensiv": KW,
    "Kunstwiese/Mähweide extensiv": KW,
    "Grassamenproduktion": "Futtergräser für die Samenproduktion (Vertragsanbau)",
}

# LNF crops without own records -> take the date of a closely related crop
DERIVED = {
    "Mischel Futtergetreide": "Triticale",
    "Mischel Brotgetreide": W,
    "Hartweizen": W,                                        # mostly autumn-sown durum in CH
    "Hirse zur Körnergewinnung": "Hirse",
    "Hirse zur Nutzung ganze Pflanze": "Hirse",
    "Sorghum zur Körnergewinnung": "Sorghum",
    "Sorghum zur Nutzung ganze Pflanze": "Sorghum",
    "Sommerraps als nachwachsender Rohstoff": "Sommerraps zur Speiseölgewinnung",
    "Sonnenblumen als nachwachsender Rohstoff": "Sonnenblumen zur Speiseölgewinnung",
    "Übrige Kunstwiese, beitragsberechtigt (z.B. Schweineweide, Geflügelweide)": KW,
}

# LNF crops without records and without close relative: typical sowing period in
# Switzerland (approximate, to be checked against Agroscope / AGRIDEA sowing guides)
LITERATURE = {
    "Reis": "05-10", "Trockenreis": "05-10", "Nassreis": "05-15",
    "Hanf": "05-01", "Hanf zur Nutzung der Samen": "05-01",
    "Hanf zur Fasernutzung": "05-01", "Anderer Hanf": "05-01",
    "Kichererbsen": "04-10", "Linsen": "04-01", "Saflor": "04-10",
    "Quinoa": "04-15", "Mohn": "03-20", "Leindotter": "04-01", "Senf": "04-01",
    "Tabak": "05-10",
    "Einjährige nachwachsende Rohstoffe (Kenaf, usw.)": "05-15",
    "Mischungen von Bohnen, Wicken, Erbsen, Kichererbsen und Lupinen mit Getreide oder "
    "Leindotter, mindestens 30 % Anteil Leguminosen bei der Ernte (zur Körnergewinnung)": "03-20",
    "Mischungen von Linsen mit Getreide oder Leindotter, mindestens 30 % Anteil Linsen bei "
    "der Ernte (zur Körnergewinnung)": "04-01",
    "Futterleguminosen für die Samenproduktion (Vertragsanbau)": "04-15",
}


# ---------------------------------------------------------------------------
def season_day(dates):
    """days since 1 August of the season start (Aug-Jul seasons)."""
    start_year = np.where(dates.dt.month >= 8, dates.dt.year, dates.dt.year - 1)
    start = pd.to_datetime(pd.Series(start_year, index=dates.index).astype(str) + "-08-01")
    return (dates - start).dt.days


def md(day):
    """season day -> 'MM-DD'."""
    if pd.isna(day):
        return None
    return (pd.Timestamp("2001-08-01") + pd.Timedelta(days=int(round(day)))).strftime("%m-%d")


def md_to_day(s):
    month = int(s[:2])
    year = 2001 if month >= 8 else 2002
    return (pd.Timestamp(f"{year}-{s}") - pd.Timestamp("2001-08-01")).days


def main():
    df = pd.read_csv(MASSN_FILE, sep=None, engine="python", encoding="latin-1")
    sub = df[df["massnahme"].isin(VALID_MASSNAHMEN)].copy()
    sub["datum"] = pd.to_datetime(sub["datum"], dayfirst=True, errors="coerce")
    sub = sub.dropna(subset=["datum"])
    sub[KULTUR_COL] = sub[KULTUR_COL].astype(str).str.strip()
    sub["Crop_DE"] = sub[KULTUR_COL].map({k.strip(): v for k, v in TXT_TO_LNF.items()})
    sub["sday"] = season_day(sub["datum"])

    lnf = pd.read_excel(LNF_TABLE, sheet_name=LNF_SHEET).dropna(subset=["LNF_code", "Crop_DE"])
    lnf_names = list(dict.fromkeys(lnf["Crop_DE"]))

    # --- txt crops that were not mapped (check if something relevant is missing) ---
    unmapped = sub.loc[sub["Crop_DE"].isna(), KULTUR_COL].value_counts()
    print(f"{len(sub)} tillage records, {sub['Crop_DE'].notna().sum()} mapped to an LNF crop")
    print("\nTxt crops with tillage records but no LNF match (not arable / not needed):")
    print(unmapped.to_string())
    bad_targets = sorted(set(TXT_TO_LNF.values()) - set(lnf_names))
    if bad_targets:
        print(f"\nWARNING: mapping targets not found in the LNF table: {bad_targets}")

    # --- median per LNF crop (and per crop x measure) ---
    m = sub.dropna(subset=["Crop_DE"])
    stats = m.groupby("Crop_DE")["sday"].agg(
        n="size", median="median", q25=lambda x: x.quantile(0.25), q75=lambda x: x.quantile(0.75))

    per_measure = (m.groupby(["Crop_DE", "massnahme"])["sday"]
                   .agg(n="size", median="median").reset_index())
    per_measure["median_date"] = per_measure["median"].map(md)
    per_measure.drop(columns="median").to_csv(OUT_CSV_MEASURE, index=False)

    rows = []
    for name in lnf_names:
        codes = ", ".join(str(int(c)) for c in lnf.loc[lnf["Crop_DE"] == name, "LNF_code"])
        if name in stats.index:
            s = stats.loc[name]
            rows.append(dict(Crop_DE=name, LNF_code=codes, source="data", n=int(s["n"]),
                             median_date=md(s["median"]), q25=md(s["q25"]), q75=md(s["q75"])))
        elif name in DERIVED and DERIVED[name] in stats.index:
            s = stats.loc[DERIVED[name]]
            rows.append(dict(Crop_DE=name, LNF_code=codes, source=f"derived from {DERIVED[name]}",
                             n=0, median_date=md(s["median"]), q25=md(s["q25"]), q75=md(s["q75"])))
        elif name in LITERATURE:
            rows.append(dict(Crop_DE=name, LNF_code=codes, source="literature (approx.)", n=0,
                             median_date=LITERATURE[name], q25=None, q75=None))
        else:
            rows.append(dict(Crop_DE=name, LNF_code=codes, source="none", n=0,
                             median_date=None, q25=None, q75=None))
    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)

    pd.set_option("display.width", 200, "display.max_colwidth", 60)
    print("\nTillage date per LNF crop:")
    print(out.to_string(index=False))
    print("\nLNF crops without own records:")
    print(out.loc[out["source"] != "data", ["Crop_DE", "source", "median_date"]].to_string(index=False))
    print(f"\nWritten: {OUT_CSV}, {OUT_CSV_MEASURE}")


# ---------------------------------------------------------------- main
if __name__ == '__main__':   
    # """
    # Per-crop validation of the S1 tillage model (Pflug vs. other, or all classes).
    
    # Checks whether the model separates tillage types WITHIN a crop, or mainly
    # learns which crops are usually ploughed.
    
    #   1. crop-only baseline: model with lnf_code as the only predictor
    #   2. global model (S1 + lnf_code), out-of-fold predictions from random K-fold CV,
    #      scored overall and separately within every crop
    #   3. crop-specific models: trained and cross-validated within one crop, S1 only
    
    # Within one crop, the crop cannot help, so per-crop scores of 2. and 3. measure
    # the S1 tillage signal only. Crops with >= MIN_FIELDS fields and at least
    # MIN_PER_CLASS fields of every class are evaluated.
    # """
    
    # import numpy as np
    # import pandas as pd
    # from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
    # from sklearn.model_selection import StratifiedKFold
    # from sklearn.utils.class_weight import compute_sample_weight
    # from xgboost import XGBClassifier
    
    # X_FILE = "data/all/X_tri.parquet"
    # Y_FILE = "data/all/Y_tri.parquet"
    # OUT_CSV = "data/all/per_crop_scores.csv"
    # BINARY = True                 # Pflug vs. other; False: all classes
    # N_FOLDS = 5
    # SEED = 42
    # MIN_FIELDS = 100
    # MIN_PER_CLASS = 10
    
    # PARAMS = dict(n_estimators=400, learning_rate=0.05, max_depth=6, subsample=0.8,
    #               colsample_bytree=0.8, tree_method="hist", enable_categorical=True, n_jobs=-1)
    
    # # ---------------------------------------------------------------------------
    # X = pd.read_parquet(X_FILE)
    # Y = pd.read_parquet(Y_FILE)
    # assert (X["sample_id"] == Y["sample_id"]).all()
    
    # s1_cols = [c for c in X.columns if c.startswith(("asc_", "desc_")) and not c.endswith("_orbit")]
    # keep = X[s1_cols].notna().all(axis=1)
    # X = X.loc[keep, s1_cols + ["lnf_code"]].reset_index(drop=True)
    # Y = Y.loc[keep].reset_index(drop=True)
    # X["lnf_code"] = X["lnf_code"].astype("category")
    
    # if BINARY:
    #     target = np.where(Y["tillage"] == "Pflug", "Pflug", "Andere")
    #     classes = ["Andere", "Pflug"]
    # else:
    #     target = Y["tillage"].to_numpy()
    #     classes = sorted(Y["tillage"].unique())
    # y = pd.Series(target).map({c: i for i, c in enumerate(classes)}).to_numpy()
    # K = len(classes)
    # crop = Y["Crop_DE"].fillna("unknown").str.slice(0, 45).to_numpy()
    
    
    # def make_model(seed):
    #     obj = dict(objective="binary:logistic", eval_metric="logloss") if BINARY else \
    #         dict(objective="multi:softprob", eval_metric="mlogloss")
    #     return XGBClassifier(**PARAMS, **obj, random_state=seed)
    
    
    # def oof_proba(Xd, yd):
    #     """Out-of-fold class probabilities from stratified K-fold CV."""
    #     proba = np.zeros((len(yd), K))
    #     skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    #     for k, (tr, te) in enumerate(skf.split(Xd, yd)):
    #         m = make_model(SEED + k)
    #         m.fit(Xd.iloc[tr], yd[tr], sample_weight=compute_sample_weight("balanced", yd[tr]))
    #         p = m.predict_proba(Xd.iloc[te])
    #         proba[te] = p if p.shape[1] == K else np.column_stack([1 - p[:, 0], p[:, 0]])
    #     return proba
    
    
    # def scores(yt, proba):
    #     pred = proba.argmax(axis=1)
    #     out = dict(f1_macro=f1_score(yt, pred, average="macro"),
    #                balanced_acc=balanced_accuracy_score(yt, pred))
    #     if len(np.unique(yt)) < 2:
    #         out["roc_auc"] = np.nan
    #     elif BINARY:
    #         out["roc_auc"] = roc_auc_score(yt, proba[:, 1])
    #     else:
    #         out["roc_auc"] = roc_auc_score(yt, proba, multi_class="ovr", average="macro",
    #                                        labels=range(K))
    #     return out
    
    
    # # ---------------------------------------------------------------------------
    # print(f"{len(X):,} fields, classes: {dict(zip(classes, np.bincount(y, minlength=K)))}")
    
    # print("\n[1] crop-only baseline ...")
    # p_crop = oof_proba(X[["lnf_code"]], y)
    # print("[2] global model (S1 + crop) ...")
    # p_glob = oof_proba(X, y)
    
    # overall = pd.DataFrame([dict(model="crop only", **scores(y, p_crop)),
    #                         dict(model="S1 + crop", **scores(y, p_glob))]).set_index("model")
    # print("\nOverall (random CV, out-of-fold):")
    # print(overall.round(3).to_string())
    
    # # --- per crop ---
    # counts = pd.crosstab(crop, y)
    # eligible = counts[(counts.sum(axis=1) >= MIN_FIELDS) & (counts.min(axis=1) >= MIN_PER_CLASS)
    #                   & (counts.shape[1] == K)].index
    # print(f"\n[3] crop-specific models for {len(eligible)} crops "
    #       f"(>= {MIN_FIELDS} fields, >= {MIN_PER_CLASS} per class) ...")
    
    # rows = []
    # for c in sorted(eligible, key=lambda c: -counts.loc[c].sum()):
    #     idx = np.flatnonzero(crop == c)
    #     yc = y[idx]
    #     g = scores(yc, p_glob[idx])
    #     p_own = oof_proba(X.loc[idx, s1_cols].reset_index(drop=True), yc)
    #     o = scores(yc, p_own)
    #     row = dict(crop=c, n=len(idx))
    #     if BINARY:
    #         row["share_pflug"] = round(yc.mean(), 2)
    #     row.update({f"global_{k}": v for k, v in g.items()})
    #     row.update({f"own_{k}": v for k, v in o.items()})
    #     rows.append(row)
    #     print(f"  {c[:40]:40s} n={len(idx):6d}  AUC global {g['roc_auc']:.3f} | own {o['roc_auc']:.3f}")
    
    # res = pd.DataFrame(rows)
    # res.to_csv(OUT_CSV, index=False)
    
    # pd.set_option("display.width", 220)
    # print("\nPer crop (random CV within the crop; chance: AUC 0.5, balanced acc "
    #       f"{1 / K:.2f}):")
    # print(res.round(3).to_string(index=False))
    
    # w = res["n"] / res["n"].sum()
    # print(f"\nField-weighted mean over evaluated crops: "
    #       f"AUC global {np.nansum(w * res['global_roc_auc']):.3f}, "
    #       f"AUC own {np.nansum(w * res['own_roc_auc']):.3f}, "
    #       f"F1 macro own {np.nansum(w * res['own_f1_macro']):.3f}")
    # print(f"\nWritten: {OUT_CSV}")
    # import matplotlib.pyplot as plt
    # import numpy as np
    # import pandas as pd
    
    # res = pd.read_csv("data/all/per_crop_scores.csv")
    # res = res.sort_values("own_roc_auc").reset_index(drop=True)
    # labels = [f"{c[:32]}  (n={n:,}, {100 * p:.0f}% Pflug)"
    #           for c, n, p in zip(res["crop"], res["n"], res["share_pflug"])]
    # ypos = np.arange(len(res))
    
    # fig, axes = plt.subplots(1, 2, figsize=(14, 0.45 * len(res) + 2), sharey=True,
    #                          gridspec_kw={"width_ratios": [1.3, 1]})
    
    # # --- left: ROC AUC, global vs. crop-specific model ---
    # ax = axes[0]
    # ax.axvspan(0.4, 0.5, color="0.92", zorder=0)
    # ax.axvline(0.5, color="k", lw=1, ls="--", zorder=1)
    # ax.hlines(ypos, res["global_roc_auc"], res["own_roc_auc"], color="0.6", lw=2, zorder=2)
    # ax.scatter(res["global_roc_auc"], ypos, s=60, color="tab:blue", zorder=3,
    #            label="global model (S1 + crop)")
    # ax.scatter(res["own_roc_auc"], ypos, s=60, color="tab:orange", zorder=3,
    #            label="crop-specific model (S1 only)")
    # ax.set_yticks(ypos)
    # ax.set_yticklabels(labels, fontsize=9)
    # ax.set_xlabel("ROC AUC within crop  (0.5 = chance)")
    # ax.set_title("Can S1 separate Pflug from other tillage within a crop?", fontsize=11)
    # ax.grid(axis="x", alpha=0.3)
    # ax.legend(loc="lower right", fontsize=8, frameon=False)
    # lo = min(0.4, res[["global_roc_auc", "own_roc_auc"]].min().min() - 0.02)
    # ax.set_xlim(lo, max(0.8, res[["global_roc_auc", "own_roc_auc"]].max().max() + 0.02))
    
    # # --- right: balanced accuracy of the crop-specific model, bar coloured by signal ---
    # ax = axes[1]
    # gain = res["own_balanced_acc"] - 0.5
    # ax.barh(ypos, gain, left=0.5, height=0.6,
    #         color=np.where(res["own_roc_auc"] >= 0.65, "tab:green",
    #                        np.where(res["own_roc_auc"] >= 0.55, "khaki", "lightgrey")))
    # ax.axvline(0.5, color="k", lw=1, ls="--")
    # ax.set_xlabel("Balanced accuracy, crop-specific model  (0.5 = chance)")
    # ax.set_title("Gain over chance", fontsize=11)
    # ax.grid(axis="x", alpha=0.3)
    # ax.set_xlim(0.4, max(0.8, res["own_balanced_acc"].max() + 0.02))
    # ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in
    #                    ("tab:green", "khaki", "lightgrey")],
    #           labels=["AUC ≥ 0.65 (useful)", "0.55–0.65 (weak)", "< 0.55 (none)"],
    #           loc="lower right", fontsize=8, frameon=False)
    
    # plt.tight_layout()
    # fig.savefig("data/all/figs/per_crop_scores.png", dpi=150)
    from datetime import datetime
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Done.")
