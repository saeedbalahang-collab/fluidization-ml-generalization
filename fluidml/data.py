"""Data loading, unit handling and audit helpers."""
from __future__ import annotations
import numpy as np
import pandas as pd
from . import config as C

SOIL_LABEL = {0.64: "fine", 1.26: "medium", 2.1: "coarse", 2.0: "coarse"}


def load(legacy: bool = False, drop_suspect: bool = False) -> pd.DataFrame:
    """Load DataP.csv and derive all model variables.

    legacy=True reproduces the ORIGINAL submission's (erroneous) processing
    (d50 = 2.0 mm for the coarse sand; raw Af divided by d50[m]^2) for audit purposes only.
    """
    raw = pd.read_csv(C.DATA_CSV)
    raw.columns = [c.strip() for c in raw.columns]
    need = {"D50", "CD", "P", "t", "T", "Af"}
    assert need <= set(raw.columns), f"missing columns {need - set(raw.columns)}"
    df = raw.copy()
    df["row_id"] = np.arange(len(df))
    df["soil"] = df["D50"].map(SOIL_LABEL)
    df["bed"] = df["D50"].astype(str) + "mm_CD" + df["CD"].astype(str)   # one bed = soil x cover depth
    df["suspect"] = [(r.D50, r.CD, r.P, r.t) in C.SUSPECT_ROWS for r in df.itertuples()]

    d50_mm = df["D50"].replace(2.1, 2.0) if legacy else df["D50"]
    df["d50_mm"] = d50_mm.astype(float)
    d50_m, cd_m = df["d50_mm"] / 1e3, df["CD"] / 1e3
    df["X"] = df["P"] / np.sqrt(cd_m * d50_m)            # dimensionless (all lengths in the same unit)
    df["P_CD"] = df["P"] / cd_m
    df["CD_d50"] = cd_m / d50_m
    df["tau"] = df["t"] / df["T"]
    if legacy:
        df["Z"] = df["Af"] / d50_m**2                     # original (unit-inconsistent) definition
        df["Af_mm2"] = np.nan
    else:
        df["Af_mm2"] = df["Af"] * C.AF_TO_MM2
        df["Z"] = df["Af_mm2"] / df["d50_mm"] ** 2         # dimensionless
    df["Y"] = np.log10(df["Z"])
    df["domain"] = np.where(df["P"] < C.TRANSFER_P, "dev", "transfer")
    xmax_dev = df.loc[df.domain == "dev", "X"].max()
    df["x_in_dev_range"] = df["X"] <= xmax_dev
    if drop_suspect:
        df = df[~df.suspect].reset_index(drop=True)
    return df


def dev(df):      return df[df.domain == "dev"]
def transfer(df): return df[df.domain == "transfer"]


def area_from_Y(Y, d50_mm):
    """Back-transform log10(Z) to dimensional area in the dataset's unit (C.AF_UNIT)."""
    return (10.0 ** np.asarray(Y)) * np.asarray(d50_mm) ** 2 / C.AF_TO_MM2
