"""Hyperparameter optimisation with Optuna (TPE, seeded). Validation folds respect soil beds."""
from __future__ import annotations
import numpy as np
import optuna
from sklearn.model_selection import GroupKFold, KFold
from . import config as C
from .metrics import rmse
from .models import make_ml, suggest

optuna.logging.set_verbosity(optuna.logging.WARNING)


def inner_splits(n, groups, grouped: bool, k=C.INNER_FOLDS, seed=C.SEED):
    if grouped:
        k = min(k, len(np.unique(groups)))
        return list(GroupKFold(n_splits=k).split(np.zeros(n), groups=groups))
    return list(KFold(n_splits=k, shuffle=True, random_state=seed).split(np.zeros(n)))


def tune(name, X, y, groups, grouped=True, n_trials=C.N_TRIALS, seed=C.SEED):
    """Return best params, best mean validation RMSE (log10 Z), and per-fold RMSE of the best trial."""
    X, y, groups = np.asarray(X, float), np.asarray(y, float), np.asarray(groups)
    splits = inner_splits(len(y), groups, grouped, seed=seed)

    def fold_scores(params):
        out = []
        for tr, va in splits:
            m = make_ml(name, params, seed).fit(X[tr], y[tr])
            out.append(rmse(y[va], m.predict(X[va])))
        return out

    def objective(trial):
        s = fold_scores(suggest(name, trial))
        trial.set_user_attr("folds", s)
        return float(np.mean(s))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_trial
    params = suggest(name, optuna.trial.FixedTrial(best.params))
    return params, float(best.value), list(best.user_attrs["folds"])
