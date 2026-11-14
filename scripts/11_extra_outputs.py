"""Additional numbers/figures-data that mirror the diagnostics of the original analysis, under the revised protocols:
 - residual structure of the withheld-pressure predictions (per-tau linear trends, zero crossings)
 - input-perturbation sensitivity (X by +-3 %, tau by +-5 %) of the leave-one-bed-out models (log space)
 - forward-chaining correction: parameters, correction term range, effect on the fitting domain
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fluidml import config as C, data as D
from fluidml.metrics import r2, rmse, score
from fluidml.models import make_ml, make_param
from fluidml.recalibration import fit as fit_corr

df = D.load(); out = {}
pr = pd.read_csv(C.PREDS / "predictions_all.csv"); pr["tau"] = pr.t / 30
hp = pd.read_csv(C.TABLES / "selected_hyperparameters.csv")

# ---- residual structure at 55 m (XGBoost, composite inputs)
t = pr[(pr.protocol == "transfer55") & (pr.model == "xgb") & (pr.inputs == "X")].copy(); t["res"] = t.Y_true - t.Y_pred
out["res55_range"] = [float(t.res.min()), float(t.res.max())]
sl = np.polyfit(t.Y_pred, t.res, 1); out["res55_vs_pred_slope"] = float(sl[0])
xc = {}
for tau, g in t.groupby("tau"):
    b, a = np.polyfit(g.X, g.res, 1); xc[round(float(tau), 3)] = dict(slope=float(b), intercept=float(a), Xc=float(-a / b))
out["Xc_by_tau"] = xc
cfg = t[np.isclose(t.tau, 1.0)].sort_values("X")[["bed", "soil", "CD", "X", "res"]]; out["res55_by_config_tau1"] = cfg.round(3).to_dict("records")
neg = t.groupby("bed").res.apply(lambda s: (s < 0).all()); pos = t.groupby("bed").res.apply(lambda s: (s > 0).all())
out["beds_all_negative"] = list(neg[neg].index); out["beds_all_positive"] = list(pos[pos].index)

# ---- perturbation sensitivity in the learning domain (LOBO folds, tuned hyper-parameters, composite inputs)
dv = D.dev(df).set_index("row_id"); rows = []
for model in ("xgb", "rf", "svr", "affine"):
    P = {"base": [], "X+": [], "X-": [], "tau+": [], "tau-": []}; Yt = []
    for bed in sorted(dv.bed.unique()):
        tr, te = dv[dv.bed != bed], dv[dv.bed == bed]
        if model == "affine": m = make_param("affine").fit(tr[["X", "tau"]].values, tr.Y.values)
        else:
            r = hp[(hp.protocol == "lobo_dev") & (hp.fold == bed) & (hp.model == model) & (hp.inputs == "X")].iloc[0]
            m = make_ml(model, json.loads(r.params)).fit(tr[["X", "tau"]].values, tr.Y.values)
        Xte = te[["X", "tau"]].values.copy(); Yt += list(te.Y)
        for k, (dx, dt) in {"base": (1, 1), "X+": (1.03, 1), "X-": (0.97, 1), "tau+": (1, 1.05), "tau-": (1, 0.95)}.items():
            P[k] += list(m.predict(Xte * np.array([dx, dt])))
    Yt = np.array(Yt)
    rows.append(dict(model=model, base=r2(Yt, P["base"]), X_pert=np.mean([r2(Yt, P["X+"]), r2(Yt, P["X-"])]), tau_pert=np.mean([r2(Yt, P["tau+"]), r2(Yt, P["tau-"])])))
pd.DataFrame(rows).round(3).to_csv(C.TABLES / "T_perturbation.csv", index=False); print(pd.DataFrame(rows).round(3))

# ---- forward-chaining correction details (XGBoost, composite inputs)
fc = pr[(pr.protocol == "pressure_ahead") & (pr.fold == "trainmax35") & (pr.model == "xgb") & (pr.inputs == "X")]
cal, tst = fc[fc.P == 45], fc[fc.P == 55]
f, p = fit_corr("diag", cal.Y_pred, cal.X, cal.tau, cal.Y_true); out["fc_diag_params"] = p
corr_t = f(tst.Y_pred.values, tst.X.values, tst.tau.values) - tst.Y_pred.values
out["fc_term_range_log10"] = [float(corr_t.min()), float(corr_t.max())]; out["fc_term_ratio_range"] = [float(10 ** corr_t.min()), float(10 ** corr_t.max())]
# effect on the fitting domain (<=35 m, in-sample predictions of the base model)
r = hp[(hp.protocol == "pressure_ahead") & (hp.fold == "trainmax35") & (hp.model == "xgb") & (hp.inputs == "X")].iloc[0]
trn = df[df.P <= 35]; m = make_ml("xgb", json.loads(r.params)).fit(trn[["X", "tau"]].values, trn.Y.values)
yh = m.predict(trn[["X", "tau"]].values); yc = f(yh, trn.X.values, trn.tau.values)
out["fit_domain"] = dict(base=score(trn.Y, yh, trn.d50_mm), corrected=score(trn.Y, yc, trn.d50_mm))
# residual slope vs X before/after at 45 and 55 m
def slope(g, col): return float(np.polyfit(g.X, g.Y_true - g[col], 1)[0])
tst2 = tst.assign(Y_corr=f(tst.Y_pred.values, tst.X.values, tst.tau.values)); cal2 = cal.assign(Y_corr=f(cal.Y_pred.values, cal.X.values, cal.tau.values))
out["res_slope_vs_X"] = dict(cal45_before=slope(cal2, "Y_pred"), cal45_after=slope(cal2, "Y_corr"), test55_before=slope(tst2, "Y_pred"), test55_after=slope(tst2, "Y_corr"))
pd.concat([cal2.assign(set="calibration_45m"), tst2.assign(set="test_55m")]).to_csv(C.PREDS / "forward_chain_xgb_X.csv", index=False)
# inner-CV fold RMSE for Table 3 (transfer protocol)
ft = hp[(hp.protocol == "transfer55") & hp.model.isin(["xgb", "rf", "svr"]) & (hp.inputs == "X")]
out["inner_cv"] = {r.model: dict(folds=json.loads(r.inner_fold_rmse), mean=float(r.inner_cv_rmse), params=json.loads(r.params)) for r in ft.itertuples()}
Path(C.TABLES / "extra_outputs.json").write_text(json.dumps(out, indent=1, default=float)); print(json.dumps({k: out[k] for k in ("res55_range", "Xc_by_tau", "beds_all_negative", "beds_all_positive", "fc_diag_params", "fc_term_range_log10", "fc_term_ratio_range", "res_slope_vs_X")}, indent=1))
print(json.dumps({k: {m: round(v[m], 3) for m in ("R2_logZ", "RMSE_logZ", "R2_Z", "NRMSE_Z", "MAPE_Z")} for k, v in out["fit_domain"].items()}, indent=1))
print(json.dumps(out["inner_cv"], indent=0)[:1800])
