"""
Analysis of S1 interferometric coherence for tillage (output of s1_coherence_hyp3.py).

  python coherence_analysis.py plot    # median / IQR trajectories per crop, class, direction
  python coherence_analysis.py model   # XGBoost, random K-fold CV, several predictor setups

Coherence of a pair is assigned to the pair's midpoint date. For the model it is
averaged in two-week bins around the crop's typical tillage date (same bins as
build_xy.py: t0 = X-2 to X weeks ... tp8 = X+6 to X+8 weeks), per direction.
Optionally the existing S1 backscatter / S2 predictors (X.parquet) are joined.
"""

import argparse

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.lines import Line2D
    

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
YEAR = 2021
COH_FILE = "tillage/coherence/coherence_long.parquet"
DATES_CSV = "cfactor/tillage_dates_per_crop.csv"      # from tillage_dates_per_crop.py
OUT_DIR = "tillage/coherence/figs"

BASELINE = 6               # 6: S1A-S1B pairs, 12: same-satellite pairs
COH_STAT = "coh_mean"      # or "coh_median"
MIN_PIXELS = 3             # field-pair values with fewer valid pixels are dropped

MERGE = {"Pflug": "Pflug", "Mulchsaat": "Mulchsaat",
         "Streifensaat": "Streifen/Direkt", "Direktsaat": "Streifen/Direkt",
         "Streifen/Direkt": "Streifen/Direkt"}
COLORS = {"Pflug": "firebrick", "Mulchsaat": "cyan", "Streifen/Direkt": "navy",
          "Andere": "tab:orange"}

# plot
BIN_DAYS = 12
MIN_FIELDS = 5

# model
BINARY = False             # True: Pflug vs. other
BINS = [(-2, 0), (0, 2), (2, 4), (4, 6), (6, 8)]
N_FOLDS, SEED = 5, 42
X_FILE, Y_FILE = "cfactor/data/all/X_ip.parquet", "cfactor/data/all/Y_ip.parquet"   # existing predictors; None to skip
# X_FILE, Y_FILE = None, None
PARAMS = dict(n_estimators=400, learning_rate=0.05, max_depth=4, subsample=0.8,
              colsample_bytree=0.8, tree_method="hist", enable_categorical=True, n_jobs=-1)


# ---------------------------------------------------------------------------
def load():
    d = pd.read_parquet(COH_FILE)
    d = d[(d["baseline_days"].round() == BASELINE) & (d["n_valid"] >= MIN_PIXELS)]
    d = d.dropna(subset=[COH_STAT]).copy()
    d["mid_date"] = pd.to_datetime(d["mid_date"])
    d["field_id"] = d["field_id"].astype(str)

    cls_col = "group" if "group" in d.columns else "tillage"
    d["class"] = d[cls_col].map(MERGE)
    if d["class"].isna().any():          # raw tillage names, e.g. "Streifenfrässaat"
        for key, label in [("pflug", "Pflug"), ("mulch", "Mulchsaat"),
                           ("streifen", "Streifen/Direkt"), ("direkt", "Streifen/Direkt")]:
            hit = d["class"].isna() & d[cls_col].str.contains(key, case=False, na=False)
            d.loc[hit, "class"] = label
    d = d[d["class"].notna()]
    if BINARY:
        d["class"] = np.where(d["class"] == "Pflug", "Pflug", "Andere")
    d["direction"] = d["direction"].str.lower().str[:3].map({"asc": "asc", "des": "desc"})

    dates = pd.read_csv(DATES_CSV, dtype={"LNF_code": str})
    rows = []
    for _, r in dates.dropna(subset=["median_date"]).iterrows():
        for code in str(r["LNF_code"]).split(","):
            if code.strip() not in ("", "nan"):
                rows.append(dict(lnf_code=int(float(code)), median_md=r["median_date"],
                                 q25=r.get("q25"), q75=r.get("q75")))
    dates = pd.DataFrame(rows).drop_duplicates("lnf_code")
    d["lnf_code"] = pd.to_numeric(d["lnf_code"], errors="coerce")
    d = d.merge(dates, on="lnf_code", how="left")
    return d


def md_to_date(md):
    if not isinstance(md, str):
        return pd.NaT
    return pd.Timestamp(f"{YEAR - 1 if int(md[:2]) >= 8 else YEAR}-{md}")


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def stage_plot():
    d = load()
    t0 = d["mid_date"].min().normalize()
    d["t"] = t0 + pd.to_timedelta(((d["mid_date"] - t0).dt.days // BIN_DAYS) * BIN_DAYS
                                  + BIN_DAYS / 2, unit="D")
    crops = d["crop"].value_counts().index.tolist()
    dirs = [x for x in ("asc", "desc") if x in d["direction"].unique()]
    classes = [c for c in COLORS if c in d["class"].unique()]

    print(f"{d['field_id'].nunique()} fields, {BASELINE}-day pairs")
    print(d.drop_duplicates("field_id").groupby(["crop", "class"]).size().unstack(fill_value=0))

    fig, axes = plt.subplots(len(crops), len(dirs), figsize=(7 * len(dirs), 2.8 * len(crops)),
                             sharex=True, sharey=True, squeeze=False)
    for r, crop in enumerate(crops):
        dc = d[d["crop"] == crop]
        win = dc.iloc[0]
        start = md_to_date(win["q25"]) if isinstance(win["q25"], str) else \
            md_to_date(win["median_md"]) - pd.Timedelta(days=14)
        end = md_to_date(win["q75"]) if isinstance(win["q75"], str) else \
            md_to_date(win["median_md"]) + pd.Timedelta(days=14)
        for c, direction in enumerate(dirs):
            ax = axes[r, c]
            for cls in classes:
                x = dc[(dc["direction"] == direction) & (dc["class"] == cls)]
                if x.empty:
                    continue
                per_field = x.groupby(["t", "field_id"])[COH_STAT].mean()
                g = per_field.groupby(level="t")
                st = pd.DataFrame({"med": g.median(), "lo": g.quantile(0.25),
                                   "hi": g.quantile(0.75), "n": g.size()})
                st = st[st["n"] >= MIN_FIELDS]
                ax.plot(st.index, st["med"], color=COLORS[cls], lw=1.5,
                        label=f"{cls} (n={x['field_id'].nunique()})")
                ax.fill_between(st.index, st["lo"], st["hi"], color=COLORS[cls], alpha=0.15, lw=0)
            if pd.notna(start) and pd.notna(end):
                ax.axvspan(start, end, color="gold", alpha=0.3, lw=0)
            ax.grid(alpha=0.3)
            ax.set_ylim(0, 1)
            if r == 0:
                orb = sorted(d.loc[d["direction"] == direction, "orbit"].unique())
                ax.set_title(f"{direction}, orbit {', '.join(map(str, orb))}", fontsize=10)
            if c == 0:
                ax.set_ylabel(f"{crop[:22]}\ncoherence", fontsize=9)
            if c == len(dirs) - 1:
                ax.legend(fontsize=7, frameon=False, loc="upper left")
            ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[7, 9, 11, 1, 3, 5]))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    fig.suptitle(f"S1 VV coherence ({BASELINE}-day pairs, pair midpoint), median and IQR per class; "
                 f"shaded: tillage period (25-75 % of recorded dates)", fontsize=11)
    plt.tight_layout()
    out = f"{OUT_DIR}/coherence_trajectories_{BASELINE}d.png"
    fig.savefig(out, dpi=150)
    plt.show()
    print(f"Written: {out}")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def bin_label(hi):
    return "t0" if hi == 0 else (f"tp{hi}" if hi > 0 else f"tm{-hi}")


def coherence_features(d):
    d = d.dropna(subset=["median_md"]).copy()
    d["tillage_date"] = d["median_md"].map(md_to_date)
    off = (d["mid_date"] - d["tillage_date"]).dt.days
    d["bin"] = pd.NA
    for lo, hi in BINS:
        d.loc[(off >= lo * 7) & (off < hi * 7), "bin"] = bin_label(hi)
    d = d[d["bin"].notna()]
    w = d.groupby(["field_id", "direction", "bin"])[COH_STAT].mean().unstack(["direction", "bin"])
    labels = [bin_label(hi) for _, hi in BINS]
    w.columns = [f"coh_{di}_{b}" for di, b in w.columns]
    order = [f"coh_{di}_{b}" for di in ("asc", "desc") for b in labels]
    return w[[c for c in order if c in w.columns]]


def stage_model():
    from sklearn.metrics import (balanced_accuracy_score, cohen_kappa_score, f1_score,
                                 roc_auc_score)
    from sklearn.model_selection import StratifiedKFold
    from sklearn.utils.class_weight import compute_sample_weight
    from xgboost import XGBClassifier

    d = load()
    Xc = coherence_features(d)
    meta = d.drop_duplicates("field_id").set_index("field_id")[["class", "crop", "lnf_code"]]
    data = meta.join(Xc, how="inner")
    coh_cols = list(Xc.columns)

    s1_cols, s2_cols = [], []
    if X_FILE:
        X = pd.read_parquet(X_FILE)
        Y = pd.read_parquet(Y_FILE)
        X = X[Y["year"].to_numpy() == YEAR].copy()
        X["field_id"] = Y.loc[Y["year"] == YEAR, "field_id"].astype(str).to_numpy()
        s1_cols = [c for c in X.columns if c.startswith(("asc_", "desc_")) and not c.endswith("_orbit")]
        s2_cols = [c for c in X.columns if c.startswith("S2_") and "quality" not in c]
        data = data.join(X.set_index("field_id")[s1_cols + s2_cols], how="left")
        print(f"joined S1/S2 predictors for {data[s1_cols].notna().all(axis=1).sum()} of "
              f"{len(data)} fields")

    data = data[data[coh_cols].notna().all(axis=1)]          # complete coherence
    data["lnf_code"] = data["lnf_code"].astype("Int64").astype(str).astype("category")
    classes = sorted(data["class"].unique(), key=lambda c: list(COLORS).index(c))
    y = data["class"].map({c: i for i, c in enumerate(classes)}).to_numpy()
    K = len(classes)
    print(f"{len(data)} fields with complete coherence ({len(coh_cols)} predictors):",
          data["class"].value_counts().reindex(classes).to_dict())

    setups = {"crop": ["lnf_code"], "coherence": coh_cols, "coherence + crop": coh_cols + ["lnf_code"]}
    if s1_cols:
        ok = data[s1_cols].notna().all(axis=1)
        setups.update({"S1 backscatter": s1_cols, "coherence + S1": coh_cols + s1_cols,
                       "coherence + S1 + crop": coh_cols + s1_cols + ["lnf_code"]})
        if not ok.all():
            print(f"setups with S1 use the {ok.sum()} fields with complete backscatter")

    obj = dict(objective="binary:logistic", eval_metric="logloss") if K == 2 else \
        dict(objective="multi:softprob", eval_metric="mlogloss")
    rows, per_crop = [], []
    for name, cols in setups.items():
        sub = data if not any(c in s1_cols for c in cols) else data[data[s1_cols].notna().all(axis=1)]
        ys = sub["class"].map({c: i for i, c in enumerate(classes)}).to_numpy()
        proba = np.zeros((len(sub), K))
        for k, (tr, te) in enumerate(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
                                     .split(sub, ys), 1):
            m = XGBClassifier(**PARAMS, **obj, random_state=SEED + k)
            m.fit(sub.iloc[tr][cols], ys[tr], sample_weight=compute_sample_weight("balanced", ys[tr]))
            p = m.predict_proba(sub.iloc[te][cols])
            proba[te] = p
        pred = proba.argmax(axis=1)
        auc = roc_auc_score(ys, proba[:, 1]) if K == 2 else \
            roc_auc_score(ys, proba, multi_class="ovr", average="macro")
        rows.append(dict(setup=name, n=len(sub), f1_macro=f1_score(ys, pred, average="macro"),
                         balanced_acc=balanced_accuracy_score(ys, pred),
                         kappa=cohen_kappa_score(ys, pred), roc_auc=auc))
        for crop, idx in sub.reset_index().groupby("crop").groups.items():
            idx = np.asarray(idx)
            if len(np.unique(ys[idx])) < 2:
                continue
            per_crop.append(dict(setup=name, crop=crop, n=len(idx),
                                 f1_macro=f1_score(ys[idx], pred[idx], average="macro"),
                                 balanced_acc=balanced_accuracy_score(ys[idx], pred[idx])))

    res = pd.DataFrame(rows).set_index("setup")
    pc = pd.DataFrame(per_crop)
    pd.set_option("display.width", 200)
    print(f"\nOut-of-fold scores, random {N_FOLDS}-fold CV, {K} classes "
          f"(chance balanced acc {1 / K:.2f}, AUC 0.50):")
    print(res.round(3).to_string())
    print("\nBalanced accuracy per crop:")
    print(pc.pivot(index="crop", columns="setup", values="balanced_acc")
          .reindex(columns=list(setups)).round(3).to_string())
    res.to_csv(f"{OUT_DIR}/coherence_model_scores.csv")
    pc.to_csv(f"{OUT_DIR}/coherence_model_scores_per_crop.csv", index=False)

def plot_results():
    res = pd.read_csv("tillage/coherence/figs/coherence_model_scores.csv")
    METRIC = "f1_macro"
    N_CLASSES = 3                      # for the chance line (1 / number of classes)
    
    COL = {"coherence": "tab:orange", "backscatter": "tab:blue", "none": "0.55"}
    
    
    def style(setup):
        s = setup.lower()
        coh, bs, crop = "coherence" in s, ("s1" in s or "backscatter" in s), "crop" in s
        marker = "D" if crop else "o"
        if coh and bs:                  # two-tone marker for combined sensors
            return dict(marker=marker, color=COL["coherence"], markerfacecoloralt=COL["backscatter"],
                        fillstyle="left", markeredgecolor="k")
        c = COL["coherence"] if coh else COL["backscatter"] if bs else COL["none"]
        return dict(marker=marker, color=c, markeredgecolor="k")
    
    
    res = res.sort_values(METRIC).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(res) + 1.5))
    ax.axvline(1 / N_CLASSES, color="k", ls="--", lw=1)
    ax.text(1 / N_CLASSES, len(res) - 0.4, " chance", fontsize=8, va="bottom")
    
    for i, r in res.iterrows():
        ax.hlines(i, 1 / N_CLASSES, r[METRIC], color="0.85", lw=3, zorder=1)
        ax.plot(r[METRIC], i, ls="", markersize=13, zorder=3, **style(r["setup"]))
        ax.text(r[METRIC] + 0.012, i, f"{r[METRIC]:.2f}  (n={int(r['n'])})", va="center", fontsize=9)
    
    ax.set_yticks(range(len(res)))
    ax.set_yticklabels(res["setup"])
    ax.set_xlabel("F1 macro (out-of-fold, random CV)")
    ax.set_xlim(min(0.2, res[METRIC].min() - 0.05), min(1, res[METRIC].max() + 0.15))
    ax.set_ylim(-0.6, len(res) - 0.2)
    ax.grid(axis="x", alpha=0.3)
    
    handles = [
        Line2D([], [], marker="o", ls="", color=COL["coherence"], markeredgecolor="k", ms=9, label="coherence"),
        Line2D([], [], marker="o", ls="", color=COL["backscatter"], markeredgecolor="k", ms=9, label="backscatter (S1)"),
        Line2D([], [], marker="o", ls="", color=COL["coherence"], markerfacecoloralt=COL["backscatter"],
               fillstyle="left", markeredgecolor="k", ms=9, label="coherence + backscatter"),
        Line2D([], [], marker="o", ls="", color="white", markeredgecolor="k", ms=9, label="without crop"),
        Line2D([], [], marker="D", ls="", color="white", markeredgecolor="k", ms=9, label="with crop"),
    ]
    ax.legend(handles=handles, fontsize=8, frameon=False, loc="lower right")
    ax.set_title("Tillage classification: coherence vs. backscatter", fontsize=11)
    plt.tight_layout()
    plt.savefig("tillage/coherence/figs/coherence_model_f1.png", dpi=150)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["plot", "model", "plot_res"])
    {"plot": stage_plot, "model": stage_model, "plot_res": plot_results}[ap.parse_args().stage]()