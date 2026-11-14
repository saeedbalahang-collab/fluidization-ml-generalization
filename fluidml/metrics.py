"""Metrics. All scores are computed from ONE predictions table so that tables cannot disagree.

Notation: Y = log10(Z), Z = Af/d50^2 (dimensionless), A = dimensional area (unit C.AF_UNIT).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from . import config as C
from .data import area_from_Y


def r2(y, yh):
    y, yh = np.asarray(y, float), np.asarray(yh, float)
    sst = ((y - y.mean()) ** 2).sum()
    return np.nan if sst == 0 else 1.0 - ((y - yh) ** 2).sum() / sst


def rmse(y, yh):
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(yh, float)) ** 2)))


def nrmse(y, yh):
    y = np.asarray(y, float)
    rng = y.max() - y.min()
    return np.nan if rng == 0 else rmse(y, yh) / rng


def mape(y, yh):
    y, yh = np.asarray(y, float), np.asarray(yh, float)
    return float(100 * np.mean(np.abs((y - yh) / y)))


def mdape(y, yh):
    y, yh = np.asarray(y, float), np.asarray(yh, float)
    return float(100 * np.median(np.abs((y - yh) / y)))


def within_factor(y, yh, f=C.TOLERANCE_FACTOR):
    ratio = np.asarray(yh, float) / np.asarray(y, float)
    return float(np.mean((ratio >= 1 / f) & (ratio <= f)))


def score(Y, Yh, d50_mm) -> dict:
    """Full metric set for one evaluation subset. Y, Yh in log10(Z) space."""
    Y, Yh, d50_mm = (np.asarray(a, float) for a in (Y, Yh, d50_mm))
    Z, Zh = 10 ** Y, 10 ** Yh
    A, Ah = area_from_Y(Y, d50_mm), area_from_Y(Yh, d50_mm)
    return {
        "n": len(Y),
        "R2_logZ": r2(Y, Yh), "RMSE_logZ": rmse(Y, Yh), "bias_logZ": float(np.mean(Yh - Y)),
        "R2_Z": r2(Z, Zh), "NRMSE_Z": nrmse(Z, Zh), "MAPE_Z": mape(Z, Zh),
        "R2_A": r2(A, Ah), "NRMSE_A": nrmse(A, Ah), "MAPE_A": mape(A, Ah), "MdAPE_A": mdape(A, Ah),
        "RMSE_A": rmse(A, Ah), "FF2": within_factor(A, Ah),
    }


def summarize(pred: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """pred needs columns: Y_true, Y_pred, d50_mm + grouping columns."""
    rows = []
    for key, g in pred.groupby(by, dropna=False, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        rows.append({**dict(zip(by, key)), **score(g.Y_true, g.Y_pred, g.d50_mm)})
    return pd.DataFrame(rows)
