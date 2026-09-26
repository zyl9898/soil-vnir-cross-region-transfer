"""
Data layer for the ICRAF-ISRIC Soil VNIR Spectral Library study.

Source (public, direct download, no registration):
  ICRAF-ISRIC Soil VNIR Spectral Library, World Agroforestry (ICRAF) Dataverse,
  doi:10.34725/DVN/MFHA9C.  Downloaded via the Dataverse access API:
      https://data.worldagroforestry.org/api/access/datafile/<id>

Files used
  11274  ASD Spectra.tab          Vis-NIR reflectance, 350-2500 nm at 10 nm  -> X
  11275  Chemical_properties.tab  reference chemistry (incl. ORGC)           -> y
  11283  Country.tab              ISO -> country -> macro REGION             -> groups
  11290  ICRAF sample codes.tab   Batch_Labid -> Sampleno -> country         -> join key
  11300  Site_description.tab     profile coordinates and altitude           -> covariates

Task: predict soil organic carbon (ORGC, %) from Vis-NIR reflectance, with
leave-one-region-out cross-validation to measure cross-region transfer.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def _resolve_data_dir() -> str:
    """Locate the downloaded library.

    Order: $SPECTRA_DATA_DIR, then ./data beside the repository root (what
    download_data.sh creates), then a sibling checkout used during development.
    """
    env = os.environ.get("SPECTRA_DATA_DIR")
    if env:
        return env
    local = os.path.join(ROOT, "data")
    if os.path.isdir(local):
        return local
    return os.path.join(os.path.dirname(ROOT), "soil-spectra-study", "icraf")


DATA = _resolve_data_dir()

RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "figures")
RANDOM_STATE = 42
N_JOBS = 4

F_SPECTRA = os.path.join(DATA, "ASD_Spectra.tab")
F_CHEM = os.path.join(DATA, "Chemical_properties.tab")
F_COUNTRY = os.path.join(DATA, "Country.tab")
F_CODES = os.path.join(DATA, "ICRAF_sample_codes.tab")
F_SITE = os.path.join(DATA, "Site_description.tab")

WAVELENGTH_RE = r"^W(\d+)$"


def _read_tab(path: str) -> pd.DataFrame:
    """Dataverse .tab files are tab-separated with a header row."""
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                       na_values=[""], quoting=3, engine="python")


def load_spectra() -> pd.DataFrame:
    """Wide reflectance table: Batch_Labid x wavelength columns (float)."""
    df = _read_tab(F_SPECTRA)
    df = df.rename(columns={df.columns[0]: "Batch_Labid"})
    df["Batch_Labid"] = df["Batch_Labid"].str.strip().str.strip('"')
    wl_cols = [c for c in df.columns if c.strip('"').startswith("W") and c.strip('"')[1:].isdigit()]
    out = pd.DataFrame({"Batch_Labid": df["Batch_Labid"]})
    for c in wl_cols:
        out[c.strip('"')] = pd.to_numeric(df[c], errors="coerce")
    return out


def wavelength_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("W")]


def load_targets() -> pd.DataFrame:
    """Reference chemistry, one row per (ISO, ID, HORI), with ORGC."""
    df = _read_tab(F_CHEM)
    for c in ("ISO", "ID", "HORI", "SAMPLENO"):
        if c in df.columns:
            df[c] = df[c].str.strip().str.strip('"')
    for c in ("ORGC", "PHH2O", "CECSOIL", "BTOP", "BBOT", "ORGN", "C/N"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_sample_codes() -> pd.DataFrame:
    """Bridge table: Batch_Labid <-> Sampleno <-> country."""
    df = _read_tab(F_CODES)
    df.columns = [c.strip().strip('"') for c in df.columns]
    ren = {}
    for c in df.columns:
        lc = c.lower()
        if "batch" in lc:
            ren[c] = "Batch_Labid"
        elif lc == "sampleno":
            ren[c] = "Sampleno"
        elif "country" in lc:
            ren[c] = "Country"
    df = df.rename(columns=ren)
    for c in ("Batch_Labid", "Sampleno", "Country"):
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().str.strip('"')
    return df


def load_regions() -> pd.DataFrame:
    """ISO -> COUNTRY -> REGION mapping, plus region names."""
    df = _read_tab(F_COUNTRY)
    df.columns = [c.strip().strip('"').lstrip("\ufeff") for c in df.columns]
    df = df.rename(columns={c: c.upper() for c in df.columns})
    for c in df.columns:
        df[c] = df[c].astype(str).str.strip().str.strip('"')
    regions = _read_tab(os.path.join(DATA, "Region.tab"))
    regions.columns = [c.strip().strip('"').lstrip("\ufeff") for c in regions.columns]
    regions = regions.rename(columns={c: c.upper() for c in regions.columns})
    for c in regions.columns:
        regions[c] = regions[c].astype(str).str.strip().str.strip('"')
    return df.merge(regions.rename(columns={"NAME": "REGION_NAME"}),
                    on="REGION", how="left")


def build_dataset(min_region_n: int = 40, drop_na: bool = True) -> pd.DataFrame:
    """Assemble the modelling table: spectra + ORGC + region + covariates.

    One row per measured horizon that has both a spectrum and an ORGC value.
    """
    spec = load_spectra()
    chem = load_targets()
    codes = load_sample_codes()
    regions = load_regions()

    # spectra -> sampleno via the bridge table
    m = spec.merge(codes[["Batch_Labid", "Sampleno", "Country"]].drop_duplicates("Batch_Labid"),
                   on="Batch_Labid", how="inner")
    m["Sampleno"] = m["Sampleno"].astype(str).str.strip()

    chem = chem.copy()
    chem["SAMPLENO"] = chem["SAMPLENO"].astype(str).str.strip()
    m = m.merge(chem[["ISO", "ID", "HORI", "SAMPLENO", "ORGC", "PHH2O", "CECSOIL",
                      "BTOP", "BBOT"]],
                left_on="Sampleno", right_on="SAMPLENO", how="inner")

    m = m.merge(regions[["ISO", "COUNTRY", "REGION", "REGION_NAME"]],
                on="ISO", how="left")

    if drop_na:
        m = m.dropna(subset=["ORGC"])
    m = m[m["ORGC"] > 0]                      # log-transform requires positive values
    m["log_ORGC"] = np.log10(m["ORGC"])

    if min_region_n:
        counts = m["REGION_NAME"].value_counts()
        keep = counts[counts >= min_region_n].index
        m = m[m["REGION_NAME"].isin(keep)]

    m = m.reset_index(drop=True)
    return m


def ensure_dirs() -> None:
    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(FIGURES, exist_ok=True)


if __name__ == "__main__":
    ensure_dirs()
    d = build_dataset()
    wl = wavelength_columns(d)
    print(f"rows={len(d)}  wavelengths={len(wl)}  ({wl[0]}..{wl[-1]} nm)")
    print(f"unique profiles (ISO,ID) = {d.groupby(['ISO','ID']).ngroups}")
    print(f"ORGC %: min={d.ORGC.min():.3f} median={d.ORGC.median():.3f} max={d.ORGC.max():.3f}")
    print("\nregions:")
    g = d.groupby("REGION_NAME").agg(n=("ORGC", "size"),
                                     orgc_median=("ORGC", "median"),
                                     countries=("COUNTRY", "nunique")).sort_values("n", ascending=False)
    print(g.to_string())
