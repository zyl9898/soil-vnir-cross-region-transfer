"""
E2 — Can the cross-region loss be recovered?

E1 showed that transferring a Vis-NIR calibration to an unseen world region costs
roughly a quarter to a third of the explained variance. This script tests three
increasingly demanding remedies under the same leave-one-region-out protocol:

  source_only      train on the other regions, predict the held-out region
  coral            unsupervised CORAL alignment of the target spectra to the
                   source distribution (no target labels used)
  k{k}             add k labelled samples from the held-out region
  coral_k{k}       CORAL alignment, then add k labelled target samples

The comparison separates two questions: how much of the loss is a pure
distribution shift that can be removed without labels, and how much remains
that only labels can fix.
"""

from __future__ import annotations

import argparse
import os
import warnings

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

# preprocessing reused from the baseline study (SNV + SG first derivative)
from scipy.signal import savgol_filter


def snv(X):
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    return (X - mu) / np.where(sd == 0, 1.0, sd)


def sg1(X):
    return savgol_filter(X, window_length=11, polyorder=2, deriv=1, axis=1)


def prep(X, kind="snv_sg1"):
    return sg1(snv(X)) if kind == "snv_sg1" else (sg1(X) if kind == "sg1" else X)


# --------------------------------------------------------------------------
# CORAL: align target covariance/second moment to the source
# --------------------------------------------------------------------------
def _sqrtm(C: np.ndarray, inverse: bool = False) -> np.ndarray:
    w, V = np.linalg.eigh(C)
    w = np.clip(w, 1e-9, None)
    w = 1.0 / np.sqrt(w) if inverse else np.sqrt(w)
    return (V * w) @ V.T


def coral(Xs: np.ndarray, Xt: np.ndarray) -> np.ndarray:
    """Return target features recoloured to match the source distribution."""
    ms, mt = Xs.mean(axis=0, keepdims=True), Xt.mean(axis=0, keepdims=True)
    Cs = np.cov(Xs, rowvar=False) + np.eye(Xs.shape[1]) * 1e-6
    Ct = np.cov(Xt, rowvar=False) + np.eye(Xt.shape[1]) * 1e-6
    A = _sqrtm(Ct, inverse=True)
    B = _sqrtm(Cs, inverse=False)
    return (Xt - mt) @ A @ B + ms


def make_model(name: str, n_train: int = 10_000):
    if name == "PLSR":
        # n_components must stay below the number of training samples, which
        # matters for the target-only k-shot conditions
        nc = max(1, min(15, n_train - 1))
        return Pipeline([("sc", StandardScaler()), ("m", PLSRegression(n_components=nc))])
    if name == "Ridge":
        return Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=10.0))])
    if name == "MLP":
        # early stopping needs a validation split, which does not exist for the
        # target-only k=10 condition; fall back to a fixed budget there
        small = n_train < 200
        return Pipeline([("sc", StandardScaler()),
                         ("m", MLPRegressor(hidden_layer_sizes=(256, 64),
                                            max_iter=300 if small else 600,
                                            early_stopping=not small,
                                            random_state=RANDOM_STATE))])
    raise ValueError(name)


def score(y_true, y_pred):
    yt, yp = 10 ** y_true, 10 ** y_pred
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    return {
        "r2_log": float(r2_score(y_true, y_pred)),
        "rmse_log": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(yt, yp)),
        "rmse": rmse,
        "rpd": float(np.std(yt, ddof=1) / rmse) if rmse > 0 else float("nan"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--prep", default="snv_sg1")
    args = ap.parse_args()
    ensure_dirs()

    shots = [10, 25] if args.smoke else [10, 25, 50, 100]
    models = ["PLSR", "Ridge"] if args.smoke else ["PLSR", "Ridge", "MLP"]

    d = build_dataset()
    wl = wavelength_columns(d)
    Xraw = d[wl].to_numpy(dtype=float)
    y = d["log_ORGC"].to_numpy(dtype=float)
    regions = d["REGION"].to_numpy()
    region_names = d["REGION_NAME"].to_numpy()
    uniq = np.unique(regions)

    rows = []
    rng = np.random.default_rng(RANDOM_STATE)

    for g in uniq:
        te = np.where(regions == g)[0]
        tr = np.where(regions != g)[0]
        if len(te) < 40 or len(tr) < 100:
            continue
        # standardise on the source statistics before any moment matching:
        # raw reflectance bands differ by orders of magnitude, which would
        # otherwise dominate the covariance CORAL tries to align
        sc = StandardScaler().fit(Xraw[tr])
        Xs_all = sc.transform(prep(Xraw[tr], args.prep))
        Xt_all = sc.transform(prep(Xraw[te], args.prep))
        ys, yt = y[tr], y[te]
        rname = region_names[te][0]

        Xt_coral = coral(Xs_all, Xt_all)
        Xt_mean = Xt_all - Xt_all.mean(axis=0, keepdims=True) + Xs_all.mean(axis=0, keepdims=True)

        # fixed shot subsets, drawn deterministically per region
        perm = rng.permutation(len(te))

        conditions = [("source_only", Xs_all, ys, Xt_all),
                      ("mean_only", Xs_all, ys, Xt_mean),
                      ("coral", Xs_all, ys, Xt_coral)]
        for k in shots:
            idx = perm[:k]
            conditions.append((f"target_only_k{k}", Xt_all[idx], yt[idx], Xt_all))
            conditions.append((f"k{k}", np.vstack([Xs_all, Xt_all[idx]]),
                               np.concatenate([ys, yt[idx]]), Xt_all))
            conditions.append((f"coral_k{k}", np.vstack([Xs_all, Xt_coral[idx]]),
                               np.concatenate([ys, yt[idx]]), Xt_coral))

        for cname, Xtr, ytr, Xte in conditions:
            for mname in models:
                clf = make_model(mname, len(Xtr))
                clf.fit(Xtr, ytr)
                pred = np.asarray(clf.predict(Xte)).ravel()
                rec = {"region": g, "region_name": rname, "condition": cname,
                       "model": mname, "prep": args.prep,
                       "n_train": len(Xtr), "n_test": len(te)}
                rec.update(score(yt, pred))
                rows.append(rec)
                print(f"  {rname:<18} {cname:<12} {mname:<6} "
                      f"R2_log={rec['r2_log']:+.3f} RMSE={rec['rmse']:.3f}% "
                      f"RPD={rec['rpd']:.2f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "E2_domain_adaptation.csv"), index=False)

    summ = (df.groupby(["model", "condition"])
              .agg(r2_log=("r2_log", "median"), r2_log_mean=("r2_log", "mean"),
                   r2_log_sd=("r2_log", "std"), rpd=("rpd", "median"),
                   rmse=("rmse", "median"), n=("r2_log", "size"))
              .reset_index())
    summ.to_csv(os.path.join(RESULTS, "E2_summary.csv"), index=False)
    print("\n=== median across held-out regions ===")
    for m in summ.model.unique():
        s = summ[summ.model == m]
        print(f"\n-- {m} --")
        print(s[["condition", "r2_log", "rpd", "rmse", "n"]]
              .to_string(index=False, float_format="%.4f"))


if __name__ == "__main__":
    main()
