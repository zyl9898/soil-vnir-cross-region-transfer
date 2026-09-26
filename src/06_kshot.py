"""
Repeated k-shot sampling with confidence intervals.

The first version of this experiment drew one fixed subset of k target samples
per region, so the reported k-shot numbers carried no uncertainty. This script
draws R independent subsets per (region, k, design) and reports the median with a
95% confidence interval.

Uncertainty is reported at two levels, because they answer different questions:
  - within_region_sd : how much the score moves when a different k-sample subset
                       is drawn from the same region (sampling variability)
  - across-region CI : how much the median moves across the 13 held-out regions
                       (the quantity quoted in the paper)

Designs:
  target_only : the k samples are the entire training set
  pooled      : the k samples are added to the global source pool

target_only is repeated R=30 times because the claim depends on which samples are
drawn; pooled is repeated R=10 times because adding k samples to several thousand
leaves little room for the draw to matter.
"""

from __future__ import annotations

import argparse
import os
import warnings

import zlib
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import RANDOM_STATE, RESULTS, build_dataset, ensure_dirs, wavelength_columns

warnings.filterwarnings("ignore")
from scipy.signal import savgol_filter  # noqa: E402


def snv(X):
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    return (X - mu) / np.where(sd == 0, 1.0, sd)


def prep(X, kind="snv_sg1"):
    return savgol_filter(snv(X), 11, 2, deriv=1, axis=1) if kind == "snv_sg1" else X


def make_model(name, n_train):
    if name == "PLSR":
        return Pipeline([("sc", StandardScaler()),
                         ("m", PLSRegression(n_components=max(1, min(15, n_train - 1))))])
    if name == "Ridge":
        return Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=10.0))])
    if name == "MLP":
        small = n_train < 200
        return Pipeline([("sc", StandardScaler()),
                         ("m", MLPRegressor(hidden_layer_sizes=(256, 64),
                                            max_iter=300 if small else 600,
                                            early_stopping=not small,
                                            random_state=RANDOM_STATE))])
    raise ValueError(name)


def metrics(yt, yp):
    yt, yp = 10 ** yt, 10 ** yp
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    return {"rmse": rmse, "rpd": float(np.std(yt, ddof=1) / rmse) if rmse > 0 else np.nan}


def boot_ci(x, n_boot=2000, seed=RANDOM_STATE):
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    meds = np.median(rng.choice(x, size=(n_boot, len(x)), replace=True), axis=1)
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    ensure_dirs()

    ks = [10, 25] if args.smoke else [10, 25, 50, 100]
    models = ["PLSR", "MLP"] if args.smoke else ["PLSR", "Ridge", "MLP"]
    R_target = 3 if args.smoke else 30
    R_pooled = 2 if args.smoke else 10

    d = build_dataset()
    wl = wavelength_columns(d)
    Xraw = d[wl].to_numpy(dtype=float)
    y = d["log_ORGC"].to_numpy(dtype=float)
    regions = d["REGION"].to_numpy()
    region_names = d["REGION_NAME"].to_numpy()

    per_repeat = []
    for g in np.unique(regions):
        te_all = np.where(regions == g)[0]
        tr_all = np.where(regions != g)[0]
        if len(te_all) < 40 or len(tr_all) < 100:
            continue
        rname = region_names[te_all][0]
        sc = StandardScaler().fit(Xraw[tr_all])
        Xs = sc.transform(prep(Xraw[tr_all]))
        Xt = sc.transform(prep(Xraw[te_all]))
        ys, yt = y[tr_all], y[te_all]

        for k in ks:
            # a region smaller than k cannot supply k labelled samples; the
            # Middle East has 79, so k = 100 is skipped there and n_regions is
            # reported per k in the summary
            if k >= len(te_all):
                print(f"  {rname:<18} k={k:<4} skipped (region has {len(te_all)})", flush=True)
                continue
            rng = np.random.default_rng(RANDOM_STATE + (abs(hash((str(g), k))) % 10_000 if False else (abs(zlib.crc32(str(g).encode())) + k) % 10_000))
            for r in range(max(R_target, R_pooled)):
                idx = rng.choice(len(te_all), size=k, replace=False)
                for design, R in (("target_only", R_target), ("pooled", R_pooled)):
                    if r >= R:
                        continue
                    Xtr = Xt[idx] if design == "target_only" else np.vstack([Xs, Xt[idx]])
                    ytr = yt[idx] if design == "target_only" else np.concatenate([ys, yt[idx]])
                    for mname in models:
                        clf = make_model(mname, len(Xtr))
                        clf.fit(Xtr, ytr)
                        p = np.asarray(clf.predict(Xt)).ravel()
                        rec = {"region": g, "region_name": rname, "k": k,
                               "design": design, "model": mname, "repeat": r,
                               "r2_log": float(r2_score(yt, p))}
                        rec.update(metrics(yt, p))
                        per_repeat.append(rec)
            print(f"  {rname:<18} k={k:<4} done", flush=True)

    pr = pd.DataFrame(per_repeat)
    pr.to_csv(os.path.join(RESULTS, "E5_kshot_per_repeat.csv"), index=False)

    # within-region summary: median over repeats for each region
    reg = (pr.groupby(["model", "design", "k", "region", "region_name"])
             .agg(r2_log_median=("r2_log", "median"),
                  within_region_sd=("r2_log", "std"), n_repeats=("r2_log", "size"))
             .reset_index())
    reg.to_csv(os.path.join(RESULTS, "E5_kshot_per_region.csv"), index=False)

    # across-region summary: bootstrap CI over the 13 region medians
    rows = []
    for (m, ds, k), gdf in reg.groupby(["model", "design", "k"]):
        vals = gdf.r2_log_median.to_numpy()
        lo, hi = boot_ci(vals)
        rows.append({"model": m, "design": ds, "k": k,
                     "median": float(np.median(vals)), "ci_low": lo, "ci_high": hi,
                     "mean": float(vals.mean()), "across_region_sd": float(vals.std(ddof=1)),
                     "mean_within_region_sd": float(gdf.within_region_sd.mean()),
                     "n_regions": len(vals), "n_repeats": int(gdf.n_repeats.mean())})
    summ = pd.DataFrame(rows).sort_values(["model", "design", "k"])
    summ.to_csv(os.path.join(RESULTS, "E5_kshot_summary.csv"), index=False)

    print("\n=== k-shot, median over 13 regions with 95% bootstrap CI ===")
    for m in models:
        print(f"\n-- {m} --")
        print(summ[summ.model == m][["design", "k", "median", "ci_low", "ci_high",
                                     "mean_within_region_sd", "n_repeats"]]
              .to_string(index=False, float_format="%.4f"))


if __name__ == "__main__":
    main()
