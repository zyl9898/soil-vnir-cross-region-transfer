"""
E1 — Baseline models for soil organic carbon (ORGC) from Vis-NIR reflectance,
under random cross-validation and under leave-one-region-out cross-validation.

The gap between the two protocols is the quantity of interest: it measures how
much accuracy is lost when a model must be transferred to a region it has never
seen, which is the practical situation for a global spectral library.

Preprocessing (all fitted per training fold / applied to both):
  raw     : reflectance as delivered
  snv     : standard normal variate (row-wise centring and scaling)
  sg1     : Savitzky-Golay first derivative (window 11, polyorder 2)
  snv_sg1 : SNV followed by SG first derivative

Usage: python3 01_baselines.py [--smoke]
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import N_JOBS, RANDOM_STATE, RESULTS, build_dataset, ensure_dirs, wavelength_columns

warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------
# preprocessing
# --------------------------------------------------------------------------
def snv(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    return (X - mu) / np.where(sd == 0, 1.0, sd)


def sg1(X: np.ndarray) -> np.ndarray:
    return savgol_filter(X, window_length=11, polyorder=2, deriv=1, axis=1)


def preprocess(X: np.ndarray, kind: str) -> np.ndarray:
    if kind == "raw":
        return X
    if kind == "snv":
        return snv(X)
    if kind == "sg1":
        return sg1(X)
    if kind == "snv_sg1":
        return sg1(snv(X))
    raise ValueError(kind)


PREPS = ["raw", "snv", "sg1", "snv_sg1"]


# --------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------
def make_models() -> dict:
    return {
        "PLSR": lambda: Pipeline([("sc", StandardScaler()),
                                  ("m", PLSRegression(n_components=15))]),
        "Ridge": lambda: Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=10.0))]),
        "RandomForest": lambda: RandomForestRegressor(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=N_JOBS),
        "HistGBM": lambda: HistGradientBoostingRegressor(random_state=RANDOM_STATE),
        "MLP": lambda: Pipeline([("sc", StandardScaler()),
                                 ("m", MLPRegressor(hidden_layer_sizes=(256, 64),
                                                    max_iter=600, early_stopping=True,
                                                    random_state=RANDOM_STATE))]),
    }


# --------------------------------------------------------------------------
# metrics for a regression on log10(ORGC)
# --------------------------------------------------------------------------
def metrics(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> dict:
    """Report in log space and back-transformed to % (the unit users care about)."""
    yt = 10 ** y_true_log
    yp = 10 ** y_pred_log
    rmse_log = float(np.sqrt(mean_squared_error(y_true_log, y_pred_log)))
    r2_log = float(r2_score(y_true_log, y_pred_log))
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    r2 = float(r2_score(yt, yp))
    # ratio of performance to deviation, the soil-spectroscopy convention
    rpd = float(np.std(yt, ddof=1) / rmse) if rmse > 0 else float("nan")
    return {"r2_log": r2_log, "rmse_log": rmse_log, "r2": r2, "rmse": rmse, "rpd": rpd}


def run_cv(X, y, groups, model_name, factory, splits, prep, protocol,
           region_label=None, repeat=0):
    rows = []
    for i, (tr, te) in enumerate(splits):
        if len(tr) < 30 or len(te) < 10:
            continue
        Xtr = preprocess(X[tr], prep)
        Xte = preprocess(X[te], prep)
        clf = factory()
        t0 = time.perf_counter()
        clf.fit(Xtr, y[tr])
        fit_s = time.perf_counter() - t0
        pred = np.asarray(clf.predict(Xte)).ravel()
        rec = {"model": model_name, "prep": prep, "protocol": protocol, "fold": i,
               "repeat": repeat, "region": region_label,
               "n_train": len(tr), "n_test": len(te),
               "fit_seconds": fit_s}
        rec.update(metrics(y[te], pred))
        rows.append(rec)
        print(f"    [{protocol}] {model_name:<13} {prep:<8} rep {repeat} fold {i:>2} "
              f"R2={rec['r2']:.3f} RMSE={rec['rmse']:.3f}% RPD={rec['rpd']:.2f} "
              f"({fit_s:.1f}s)", flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    ensure_dirs()

    n_repeats = 1 if args.smoke else 3
    preps = ["snv"] if args.smoke else PREPS
    models = make_models()
    if args.smoke:
        models = {k: models[k] for k in ("PLSR", "RandomForest")}

    d = build_dataset()
    wl = wavelength_columns(d)
    X = d[wl].to_numpy(dtype=float)
    y = d["log_ORGC"].to_numpy(dtype=float)
    groups = d["REGION"].to_numpy()
    print(f"data: n={len(d)}  wavelengths={len(wl)}  regions={len(np.unique(groups))}", flush=True)
    print(d.groupby("REGION_NAME").size().sort_values(ascending=False).head(12).to_string(), flush=True)

    rows = []
    # ---- random CV (reference protocol) ---------------------------------
    # fresh splits per repeat: reusing one split set would duplicate folds and
    # understate the variance of the random-CV estimate
    rand_split_sets = []
    for r in range(n_repeats):
        kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE + r)
        rand_split_sets.append(list(kf.split(X)))
    # ---- leave-one-region-out -------------------------------------------
    uniq = np.unique(groups)
    loo_splits = [(np.where(groups != g)[0], np.where(groups == g)[0]) for g in uniq]
    loo_labels = list(uniq)

    for prep in preps:
        for mname, factory in models.items():
            print(f"  --- {mname} / {prep} ---", flush=True)
            for r, sp in enumerate(rand_split_sets):
                rows += run_cv(X, y, groups, mname, factory, sp,
                               prep, "random", repeat=r)
            for i, (tr, te) in enumerate(loo_splits):
                if len(tr) < 30 or len(te) < 10:
                    continue
                rows += run_cv(X, y, groups, mname, factory, [(tr, te)],
                               prep, "leave_region_out", region_label=str(loo_labels[i]))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "E1_baselines.csv"), index=False)

    summary = (df.groupby(["protocol", "model", "prep"])
                 .agg(r2=("r2", "mean"), r2_sd=("r2", "std"),
                      rmse=("rmse", "mean"), rpd=("rpd", "mean"),
                      r2_log=("r2_log", "mean"), n_folds=("r2", "size"))
                 .reset_index())
    summary.to_csv(os.path.join(RESULTS, "E1_summary.csv"), index=False)
    print("\n=== summary ===", flush=True)
    for proto in summary.protocol.unique():
        s = summary[summary.protocol == proto].sort_values("r2", ascending=False)
        print(f"\n-- {proto} --")
        print(s.head(12).to_string(index=False, float_format="%.4f"))


if __name__ == "__main__":
    main()
