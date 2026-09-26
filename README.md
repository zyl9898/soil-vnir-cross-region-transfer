# Cross-region transfer of soil Vis-NIR organic-carbon calibrations

Analysis code and result tables for a study of how well a soil visible–near-infrared calibration
transfers to a world region it has never seen, and what recovers the loss.

**Data:** [ICRAF-ISRIC Soil VNIR Spectral Library](https://doi.org/10.34725/DVN/MFHA9C) — 3,980
horizons with reference organic carbon, 216 bands (350–2500 nm), 13 world regions, 58 countries.
Public, no registration required.

## Headline results

| Finding | Value |
|---|---|
| Genuine cross-region penalty (mean over 20 model × pre-treatment combinations) | **+0.094 R²** = 16.3% of the profile-grouped baseline (0.578) |
| Optimism contributed by profile reuse in plain K-fold | **+0.050 R²** = 7.9% of the random-K-fold baseline (0.629) |
| CORAL covariance alignment | does not recover the penalty (PLSR 0.451 → 0.362; ridge 0.524 → 0.371; MLP 0.538 → 0.203) |
| 100 labelled target samples used alone | median R² 0.69 (PLSR), 0.69 (ridge), 0.70 (MLP), against 0.45–0.54 for the whole global library |
| 1-D CNN under leave-one-region-out | median 0.514, mid-pack among classical families (0.451–0.565) |

The library holds several horizons per profile, so plain K-fold can place horizons of one profile in
both training and test partitions. Three protocols are therefore run side by side: random K-fold,
leave-one-profile-out (`GroupKFold` on `ISO`+`ID`), and leave-one-region-out. The difference between
the first two isolates profile leakage; the difference between the last two is the genuine
cross-region penalty.

## Quick start

```bash
git clone <this repository> && cd <repository>
bash download_data.sh                       # fetches the six library files (~11 MB total)

python3 -m pip install -r requirements.txt

python3 src/common.py                       # assemble and sanity-check the modelling table
python3 src/05_protocols.py                 # three-protocol decomposition (E4)
python3 src/01_baselines.py                 # model x pre-treatment benchmark (E1)
python3 src/02_domain_adaptation.py         # CORAL / mean-only / k-shot controls (E2)
python3 src/06_kshot.py                     # repeated subset sampling with CIs (E5)
python3 src/03_cnn.py --epochs 120          # 1-D CNN and Deep CORAL (E3, PyTorch)
python3 src/04_figures.py                   # figures + layout and PDF QA checks
```

Total runtime is a few hours on a laptop CPU. `03_cnn.py` uses Apple MPS or CUDA when available and
falls back to CPU automatically.

## Contents

```
download_data.sh   fetches the spectral library from the ICRAF Dataverse
requirements.txt   Python dependencies
src/
  common.py                  joins spectra, chemistry and region labels
  01_baselines.py            E1  five model families x four pre-treatments
  02_domain_adaptation.py    E2  CORAL, mean-only correction, k-shot, pooled
  03_cnn.py                  E3  1-D CNN with optional Deep CORAL penalty
  04_figures.py              all figures (panel-alignment gate + PDF audits)
  05_protocols.py            E4  random / profile-grouped / region-grouped
  06_kshot.py                E5  repeated subset sampling, median + bootstrap CI
results/           every derived table as CSV
figures/           figures as SVG (editable text) and PDF
```

## Data

`download_data.sh` writes the six library files into `./data`, which is where `src/common.py` looks
for them. To keep the data elsewhere, set `SPECTRA_DATA_DIR`:

```bash
SPECTRA_DATA_DIR=/path/to/library python3 src/05_protocols.py
```

The Dataverse server does not support byte-range requests, so an interrupted download must be
restarted rather than resumed. `data/` and `*.tab` are gitignored, so the library is not
redistributed here.

## Method notes

- **Preprocessing** (all fitted on the training fold only): raw reflectance; standard normal variate
  (SNV); Savitzky–Golay first derivative (window 11, order 2); SNV followed by the derivative.
- **Protocols**: random 5-fold x 3 repeats; `GroupKFold` on `(ISO, ID)`; leave-one-region-out.
- **Summary statistic**: the **median** across held-out regions is primary, because per-region scores
  are left-skewed by a few regions where transfer weakens. Means are reported alongside where useful.
- **k-shot**: 30 independent subset draws per region and k for the target-only design, 10 for the
  pooled design. Confidence intervals are bootstraps over region medians with 2,000 resamples. The
  k = 100 rows cover 12 regions, because the Middle East holds 79 samples and cannot supply 100.
- **Region-level results vary sharply**: per-region R² ranges from strongly negative to above 0.6, so
  a single global figure hides regions where the calibration is unusable.

## License

MIT (see `LICENSE`). The spectral library is distributed by World Agroforestry (ICRAF) under its own
terms; see the Dataverse record.

## Citation

If you use this code, please cite the paper once published, and the data source:

> World Agroforestry (ICRAF), International Soil Reference and Information Centre (ISRIC):
> ICRAF-ISRIC Soil VNIR Spectral Library. Dataverse, V1 (2020).
> https://doi.org/10.34725/DVN/MFHA9C
