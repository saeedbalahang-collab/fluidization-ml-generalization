"""Model factory: tuned ML models, log-log power-law benchmark, and the original affine benchmark."""
from __future__ import annotations
import numpy as np
from scipy.optimize import least_squares
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor
from . import config as C


# ---------------------------------------------------------------- search spaces (Optuna)
def suggest(name: str, trial) -> dict:
    if name == "xgb":
        return dict(n_estimators=trial.suggest_int("n_estimators", 10, 600),
                    max_depth=trial.suggest_int("max_depth", 2, 20),
                    learning_rate=trial.suggest_float("learning_rate", 1e-3, 0.5, log=True),
                    subsample=trial.suggest_float("subsample", 0.7, 1.0),
                    colsample_bytree=trial.suggest_float("colsample_bytree", 0.7, 1.0),
                    gamma=trial.suggest_float("gamma", 0.0, 1.0),
                    reg_alpha=trial.suggest_float("reg_alpha", 0.0, 1.0),
                    reg_lambda=trial.suggest_float("reg_lambda", 0.0, 1.0))
    if name == "rf":
        return dict(n_estimators=trial.suggest_int("n_estimators", 10, 600),
                    max_depth=trial.suggest_int("max_depth", 2, 20),
                    min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
                    min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
                    max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
                    bootstrap=trial.suggest_categorical("bootstrap", [True, False]))
    if name == "svr":
        kernel = trial.suggest_categorical("kernel", ["rbf", "poly"])
        p = dict(kernel=kernel,
                 C=trial.suggest_float("C", 0.1, 1e3, log=True),
                 epsilon=trial.suggest_float("epsilon", 0.01, 1.0),
                 gamma=trial.suggest_float("gamma", 0.1, 100.0, log=True))
        if kernel == "poly":
            p["degree"] = trial.suggest_int("degree", 2, 5)
        return p
    raise KeyError(name)


def make_ml(name: str, params: dict, seed: int = C.SEED):
    """Pipeline(MinMaxScaler -> estimator). The scaler is fitted inside .fit(), i.e. on training data only."""
    if name == "xgb":
        est = XGBRegressor(**params, random_state=seed, n_jobs=1, tree_method="hist", verbosity=0)
    elif name == "rf":
        est = RandomForestRegressor(**params, random_state=seed, n_jobs=1)
    elif name == "svr":
        est = SVR(**params, cache_size=500, max_iter=2_000_000)
    else:
        raise KeyError(name)
    return Pipeline([("scale", MinMaxScaler()), ("est", est)])


# ---------------------------------------------------------------- parametric benchmarks
class PowerLaw(BaseEstimator, RegressorMixin):
    """log10 Z = a + sum_i b_i log10(feature_i)   <=>   Z = 10^a * prod feature_i^b_i  (OLS).
    Columns are assumed strictly positive (X, P/CD, CD/d50, tau all are)."""
    def fit(self, X, y):
        self.lr_ = LinearRegression().fit(np.log10(np.asarray(X, float)), y)
        return self

    def predict(self, X):
        return self.lr_.predict(np.log10(np.asarray(X, float)))

    @property
    def params(self):
        return {"a": float(self.lr_.intercept_), **{f"b{i}": float(b) for i, b in enumerate(self.lr_.coef_)}}


class AffineBenchmark(BaseEstimator, RegressorMixin):
    """Original Eq.(7)-type benchmark: Z = a*X + b*tau + c, fitted by least squares on log10 Z.
    Columns must be [X, tau]."""
    def __init__(self, x0=(9510.31, 6.914, 9.768)):
        self.x0 = x0

    def _pred(self, p, X):
        z = p[0] * X[:, 0] + p[1] * X[:, 1] + p[2]
        return np.log10(np.maximum(z, 1e-12))

    def fit(self, X, y):
        X = np.asarray(X, float)
        # rescale so that the optimiser is well conditioned
        s = np.array([1e4, 1e2, 1e2])
        f = lambda q: self._pred(q * s, X) - y
        best = None
        # coefficients are constrained >= 0 (as in the original Eq. 7) so that Z-hat > 0 and log10 is defined
        for start in (np.array(self.x0) / s, np.array([2e3, 5.0, 10.0]) / s):
            r = least_squares(f, np.maximum(start, 1e-6), bounds=(0, np.inf), method="trf", max_nfev=5000)
            if best is None or r.cost < best.cost:
                best = r
        self.p_ = best.x * s
        return self

    def predict(self, X):
        return self._pred(self.p_, np.asarray(X, float))

    @property
    def params(self):
        return dict(zip("abc", map(float, self.p_)))


def make_param(name: str):
    return {"pl": PowerLaw, "affine": AffineBenchmark}[name]()
