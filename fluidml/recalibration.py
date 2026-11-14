"""Post-hoc recalibration of base predictions (log10 Z space).

none    : no correction
shift   : Y' = Yh + mean(Y - Yh)                           (1 parameter)
linear  : Y' = a + b*Yh   (OLS)                             (2 parameters)
diag    : Y' = Yh + k log10(X / Xc(tau))                     (additive, unit-invariant version of the original multiplicative Eq. 8,
                                                             Y*(1 + k log10(X/Xc)), whose effect depends on the arbitrary offset of log10 Z;
                                                             Xc(tau) from per-tau residual-vs-X zero crossings, Xc linear in tau,
                                                             k by least squares - closed form, no bounds or weights)
"""
from __future__ import annotations
import numpy as np


def _fit_diag(Yh, X, tau, Y):
    e = Y - Yh
    taus = np.unique(np.round(tau, 6))
    xc = []
    for t in taus:
        m = np.isclose(tau, t)
        b, a = np.polyfit(X[m], e[m], 1)
        xc.append(-a / b if b != 0 else np.nan)
    xc = np.array(xc)
    if not np.all(np.isfinite(xc)) or np.any(xc <= 0):
        return None, dict(status="undefined (no valid residual zero crossing)", Xc=list(map(float, xc)))
    s, c = np.polyfit(taus, xc, 1)                    # Xc(tau) = c + s*tau
    Xc = c + s * tau
    if np.any(Xc <= 0):
        return None, dict(status="undefined (Xc<=0)", Xc_c=float(c), Xc_s=float(s))
    g = np.log10(X / Xc)
    k = float((g * e).sum() / (g ** 2).sum())
    f = lambda Yh_, X_, tau_: Yh_ + k * np.log10(X_ / (c + s * tau_))
    return f, dict(status="ok", k=k, Xc_intercept=float(c), Xc_slope=float(s))


def fit(method, Yh, X, tau, Y):
    """Fit on calibration data; return (callable(Yh,X,tau)->Y', params dict)."""
    Yh, X, tau, Y = (np.asarray(a, float) for a in (Yh, X, tau, Y))
    if method == "none":
        return (lambda a, x, t: a), {}
    if method == "shift":
        d = float(np.mean(Y - Yh)); return (lambda a, x, t: a + d), dict(shift=d)
    if method == "linear":
        b, a0 = np.polyfit(Yh, Y, 1); return (lambda a, x, t: a0 + b * a), dict(a=float(a0), b=float(b))
    if method == "diag":
        f, p = _fit_diag(Yh, X, tau, Y)
        return (f if f else (lambda a, x, t: a)), p
    raise KeyError(method)

METHODS = ("none", "shift", "linear", "diag")
