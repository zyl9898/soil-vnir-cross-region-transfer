"""
Three evaluation protocols on identical folds-machinery, so the transfer penalty
can be decomposed without confusing two different sources of optimism.

  random        : plain K-fold. Horizons from the same soil profile can land in
                  both train and test, so this measures interpolation with the
                  most leakage.
  group_profile : GroupKFold on (ISO, ID). All horizons of a profile stay on one
                  side of the split, removing the within-profile leakage that
                  plain K-fold allows.
  group_region  : GroupKFold on REGION (leave-one-region-out). The deployment
                  case.

The gap random - group_profile isolates the optimism contributed by repeated
horizons of one profile; group_profile - group_region is the genuine
cross-region penalty.

Median is reported as the primary summary because the per-fold distribution is
skewed by a few regions where transfer breaks down; the mean is reported
alongside it. Earlier drafts compared a median against a mean, which this script
replaces with a single consistent table.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression

from common import N_JOBS, RANDOM_STATE, RESULTS, build_dataset, ensure_dirs, wavelength_columns

warnings.filterwarnings("ignore")

from scipy.signal import savgol_filter


def snv(X):
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    return (X - mu) / np.where(sd == 0, 1.0, sd)


def prep(X, kind):
    if kind == "raw": return X
    if kind == "snv": return snv(X)
    if kind == "sg1": return savgol_filter(X, 11, 2, deriv=1, axis=1)
    if kind == "snv_sg1": return savgol_filter(snv(X), 11, 2, deriv=1, axis=1)
    raise ValueError(kind)


def make_models():
    return {
        "PLSR": lambda: Pipeline([("sc", StandardScaler()),
                                  ("m", PLSRegression(n_components=15))]),
        "Ridge": lambda: Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=10.0))]),
        "RandomForest": lambda: RandomForestRegressor(n_estimators=300,
                                                      random_state=RANDOM_STATE, n_jobs=N_JOBS),
        "HistGBM": lambda: HistGradientBoostingRegressor(random_state=RANDOM_STATE),
        "MLP": lambda: Pipeline([("sc", StandardScaler()),
                                 ("m", MLPRegressor(hidden_layer_sizes=(256, 64), max_iter=600,
                                                    early_stopping=True,
                                                    random_state=RANDOM_STATE))]),
    }


def metrics(yt, yp):
    yt, yp = 10 ** yt, 10 ** yp
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    return {"rmse": rmse, "rpd": float(np.std(yt, ddof=1) / rmse) if rmse > 0 else np.nan}


def run(X, y, splits, model, kind, protocol, model_name, label=None, repeat=0):
    rows = []
    for i, (tr, te) in enumerate(splits):
        if len(tr) < 30 or len(te) < 10:
            continue
        Xtr, Xte = prep(X[tr], kind), prep(X[te], kind)
        m = model()
        t0 = time.perf_counter()
        m.fit(Xtr, y[tr])
        ft = time.perf_counter() - t0
        p = np.asarray(m.predict(Xte)).ravel()
        rec = {"protocol": protocol, "model": model_name, "prep": kind, "fold": i,
               "repeat": repeat, "group": label, "n_train": len(tr), "n_test": len(te),
               "r2_log": float(r2_score(y[te], p)), "fit_seconds": ft}
        rec.update(metrics(y[te], p))
        rows.append(rec)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    ensure_dirs()

    d = build_dataset()
    wl = wavelength_columns(d)
    X = d[wl].to_numpy(dtype=float)
    y = d["log_ORGC"].to_numpy(dtype=float)
    profile = (d["ISO"].astype(str) + "|" + d["ID"].astype(str)).to_numpy()
    region = d["REGION"].to_numpy()
    print(f"n={len(d)}  wavelengths={len(wl)}  profiles={len(np.unique(profile))}  "
          f"regions={len(np.unique(region))}", flush=True)

    preps = ["snv"] if args.smoke else ["raw", "snv", "sg1", "snv_sg1"]
    models = make_models()
    if args.smoke:
        models = {k: models[k] for k in ("PLSR", "MLP")}

    rows = []
    for kind in preps:
        for cur_model, factory in models.items():
            print(f"  --- {cur_model} / {kind} ---", flush=True)
            # random K-fold, 3 repeats with fresh partitions
            for r in range(1 if args.smoke else 3):
                kf = KFold(5, shuffle=True, random_state=RANDOM_STATE + r)
                rows += run(X, y, list(kf.split(X)), factory, kind, "random", cur_model, repeat=r)
            # leave-one-profile-out
            gk = GroupKFold(n_splits=min(5, len(np.unique(profile))))
            rows += run(X, y, list(gk.split(X, y, profile)), factory, kind, "group_profile", cur_model)
            # leave-one-region-out
            for g in np.unique(region):
                te = np.where(region == g)[0]; tr = np.where(region != g)[0]
                rows += run(X, y, [(tr, te)], factory, kind, "group_region", cur_model, label=str(g))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "E4_protocols.csv"), index=False)

    def agg(g):
        return pd.Series({
            "r2_log_median": g.r2_log.median(), "r2_log_mean": g.r2_log.mean(),
            "r2_log_sd": g.r2_log.std(), "rpd_median": g.rpd.median(),
            "n": len(g)})
    s = df.groupby(["protocol", "model", "prep"]).apply(agg).reset_index()
    s.to_csv(os.path.join(RESULTS, "E4_summary.csv"), index=False)
    print("\n=== summary (median primary) ===")
    for proto in ["random", "group_profile", "group_region"]:
        sub = s[s.protocol == proto].sort_values("r2_log_median", ascending=False)
        print(f"\n-- {proto} --")
        print(sub[["model", "prep", "r2_log_median", "r2_log_mean", "rpd_median"]]
              .to_string(index=False, float_format="%.4f"))

    # decomposition of the penalty
    piv = s.pivot_table(index=["model", "prep"], columns="protocol", values="r2_log_median")
    piv["leak_gain"] = piv["random"] - piv["group_profile"]
    piv["region_penalty"] = piv["group_profile"] - piv["group_region"]
    piv.to_csv(os.path.join(RESULTS, "E4_decomposition.csv"))
    print("\n=== decomposition (median R2) ===")
    print(piv.round(4).to_string())
    print(f"\nmean profile-leak gain : {piv.leak_gain.mean():+.4f}")
    print(f"mean region penalty    : {piv.region_penalty.mean():+.4f}")
