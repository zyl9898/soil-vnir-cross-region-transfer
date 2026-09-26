"""
E3 — A 1D convolutional network on the raw spectral profile, with an optional
Deep CORAL penalty, under the same leave-one-region-out protocol.

The CNN sees the SG-preprocessed spectrum as a 1-D signal of 216 bands, so it can
exploit local spectral shape (absorption features) rather than treating bands as
independent tabular columns.

Two variants:
  cnn          plain regression CNN trained on the source regions
  cnn_coral    same network with a Deep CORAL penalty that matches the
               second-order statistics of the penultimate feature layer between
               source and (unlabelled) target regions

Runs on Apple MPS when available, else CPU.
"""

from __future__ import annotations

import argparse
import os
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.signal import savgol_filter
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from common import RANDOM_STATE, RESULTS, build_dataset, ensure_dirs, wavelength_columns

warnings.filterwarnings("ignore")

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def snv(X):
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    return (X - mu) / np.where(sd == 0, 1.0, sd)


def sg1(X):
    return savgol_filter(X, window_length=11, polyorder=2, deriv=1, axis=1)


class SpecCNN(nn.Module):
    def __init__(self, n_bands: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=11, padding=5), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=11, padding=5), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 32, kernel_size=7, padding=3), nn.ReLU(),
        )
        # Flatten rather than adaptive-pool: MPS rejects adaptive pooling when
        # the input length is not divisible by the output size.
        flat = 32 * (n_bands // 4)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(flat, 64), nn.ReLU(),
                                  nn.Linear(64, 1))

    def forward(self, x, return_feat=False):
        f = self.features(x)
        y = self.head(f)
        return (y, f) if return_feat else y


def coral_loss(fs: torch.Tensor, ft: torch.Tensor) -> torch.Tensor:
    """Deep CORAL: Frobenius norm between source and target feature covariances."""
    d = fs.shape[1]
    ns, nt = fs.shape[0], ft.shape[0]
    fs = fs.reshape(ns, -1)
    ft = ft.reshape(nt, -1)
    d = fs.shape[1]
    Cs = (fs.T @ fs) / (ns - 1) - fs.mean(0, keepdim=True).T @ fs.mean(0, keepdim=True)
    Ct = (ft.T @ ft) / (nt - 1) - ft.mean(0, keepdim=True).T @ ft.mean(0, keepdim=True)
    return ((Cs - Ct) ** 2).sum() / (4 * d * d)


def train_cnn(Xtr, ytr, Xte, use_coral, epochs=120, lr=2e-3, batch=128, seed=RANDOM_STATE):
    torch.manual_seed(seed)
    np.random.seed(seed)
    n_bands = Xtr.shape[1]
    model = SpecCNN(n_bands).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32).unsqueeze(1).to(DEVICE)
    ytr_t = torch.tensor(ytr, dtype=torch.float32).view(-1, 1).to(DEVICE)
    Xte_t = torch.tensor(Xte, dtype=torch.float32).unsqueeze(1).to(DEVICE)

    n = len(Xtr)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb, yb = Xtr_t[idx], ytr_t[idx]
            opt.zero_grad()
            if use_coral:
                # target batch drawn without labels, used only for the penalty
                tidx = torch.randint(0, Xte_t.shape[0], (min(batch, Xte_t.shape[0]),),
                                     device=DEVICE)
                _, fs = model(xb, return_feat=True)
                _, ft = model(Xte_t[tidx], return_feat=True)
                pred = model.head(fs)
                loss = nn.functional.mse_loss(pred, yb) + 1.0 * coral_loss(fs, ft)
            else:
                loss = nn.functional.mse_loss(model(xb), yb)
            loss.backward()
            opt.step()
        sched.step()

    model.eval()
    with torch.no_grad():
        pred = model(Xte_t).cpu().numpy().ravel()
    return pred


def score(y_true, y_pred):
    yt, yp = 10 ** y_true, 10 ** y_pred
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    return {"r2_log": float(r2_score(y_true, y_pred)),
            "rmse_log": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "r2": float(r2_score(yt, yp)), "rmse": rmse,
            "rpd": float(np.std(yt, ddof=1) / rmse) if rmse > 0 else float("nan")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--epochs", type=int, default=120)
    args = ap.parse_args()
    ensure_dirs()
    print(f"device: {DEVICE}", flush=True)

    d = build_dataset()
    wl = wavelength_columns(d)
    Xraw = d[wl].to_numpy(dtype=float)
    y = d["log_ORGC"].to_numpy(dtype=float)
    regions = d["REGION"].to_numpy()
    region_names = d["REGION_NAME"].to_numpy()

    rows = []
    uniq = np.unique(regions)
    todo = uniq[:2] if args.smoke else uniq
    for g in todo:
        te = np.where(regions == g)[0]
        tr = np.where(regions != g)[0]
        if len(te) < 40:
            continue
        sc = StandardScaler().fit(Xraw[tr])
        Xs = sc.transform(sg1(snv(Xraw[tr])))
        Xt = sc.transform(sg1(snv(Xraw[te])))
        ys, yt = y[tr], y[te]
        rname = region_names[te][0]
        for variant, use_coral in (("cnn", False), ("cnn_coral", True)):
            pred = train_cnn(Xs, ys, Xt, use_coral, epochs=args.epochs)
            rec = {"region": g, "region_name": rname, "variant": variant,
                   "n_train": len(tr), "n_test": len(te)}
            rec.update(score(yt, pred))
            rows.append(rec)
            print(f"  {rname:<18} {variant:<10} R2_log={rec['r2_log']:+.3f} "
                  f"RMSE={rec['rmse']:.3f}% RPD={rec['rpd']:.2f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "E3_cnn.csv"), index=False)
    s = (df.groupby("variant").agg(r2_log=("r2_log", "median"),
                                   rpd=("rpd", "median"), rmse=("rmse", "median"),
                                   n=("r2_log", "size")).reset_index())
    s.to_csv(os.path.join(RESULTS, "E3_summary.csv"), index=False)
    print("\n=== median across held-out regions ===")
    print(s.to_string(index=False, float_format="%.4f"))


if __name__ == "__main__":
    main()
