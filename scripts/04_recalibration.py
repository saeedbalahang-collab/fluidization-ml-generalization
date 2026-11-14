"""Validation of the post-hoc correction.
forward_chain : base model trained on 15-35 m; correction calibrated on 45 m; tested on 55 m (independent)
lobo_calib    : base model trained on 15-45 m; correction calibrated on the 55 m rows of 8 beds, tested on the 9th bed
in_sample     : calibrated and scored on the same 55 m rows (what the original submission reported)
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fluidml import config as C
from fluidml.metrics import score
from fluidml.recalibration import fit, METHODS

pr = pd.read_csv(C.PREDS / "predictions_all.csv"); pr["tau"] = pr.t / 30
rows, outs, pars = [], [], []
def add(scheme, model, inputs, method, g, params):
    rows.append(dict(scheme=scheme, model=model, inputs=inputs, method=method, **score(g.Y_true, g.Y_pred, g.d50_mm)))
    pars.append(dict(scheme=scheme, model=model, inputs=inputs, method=method, params=json.dumps(params)))
    outs.append(g.assign(scheme=scheme, method=method))

for model in ("xgb", "rf", "svr"):
    for inputs in ("X", "sep"):
        # --- forward chaining
        fc = pr[(pr.protocol == "pressure_ahead") & (pr.fold == "trainmax35") & (pr.model == model) & (pr.inputs == inputs)]
        cal, tst = fc[fc.P == 45], fc[fc.P == 55]
        for m in METHODS:
            f, p = fit(m, cal.Y_pred, cal.X, cal.tau, cal.Y_true)
            add("forward_chain", model, inputs, m, tst.assign(Y_pred=f(tst.Y_pred.values, tst.X.values, tst.tau.values)), p)
        # --- in-sample and leave-one-bed-out calibration at 55 m
        t55 = pr[(pr.protocol == "transfer55") & (pr.model == model) & (pr.inputs == inputs)]
        for m in METHODS:
            f, p = fit(m, t55.Y_pred, t55.X, t55.tau, t55.Y_true)
            add("in_sample", model, inputs, m, t55.assign(Y_pred=f(t55.Y_pred.values, t55.X.values, t55.tau.values)), p)
            parts = []
            for b in t55.bed.unique():
                c, h = t55[t55.bed != b], t55[t55.bed == b]
                f, _ = fit(m, c.Y_pred, c.X, c.tau, c.Y_true)
                parts.append(h.assign(Y_pred=f(h.Y_pred.values, h.X.values, h.tau.values)))
            add("lobo_calib", model, inputs, m, pd.concat(parts), {})
res = pd.DataFrame(rows)
res.round(3).to_csv(C.TABLES / "T_recalibration.csv", index=False)
pd.DataFrame(pars).to_csv(C.TABLES / "recalibration_parameters.csv", index=False)
pd.concat(outs).to_csv(C.PREDS / "predictions_recalibrated.csv", index=False)
print(res[(res.model == "xgb")][["scheme", "inputs", "method", "R2_logZ", "RMSE_logZ", "R2_Z", "MAPE_A", "FF2"]].round(3).to_string())
print(pd.DataFrame(pars).query("model=='xgb' and inputs=='X' and method=='diag'").to_string())
