"""Bed-level (cluster) bootstrap, SHAP (development domain only) and partial dependence / extrapolation curves."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd, shap
from joblib import Parallel, delayed
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fluidml import config as C, data as D
from fluidml.models import make_ml, make_param

df = D.load(); dv, tr = D.dev(df), D.transfer(df)
hp = pd.read_csv(C.TABLES / "selected_hyperparameters.csv")
def params(model, inputs, proto="transfer55"):
    r = hp[(hp.protocol == proto) & (hp.model == model) & (hp.inputs == inputs)].iloc[0]
    return json.loads(r.params)

# ------------------------------------------------ bootstrap
def boot_once(b, unit, cols, p, seed):
    rng = np.random.default_rng(seed)
    if unit == "bed":
        beds = rng.choice(dv.bed.unique(), size=dv.bed.nunique(), replace=True)
        s = pd.concat([dv[dv.bed == x] for x in beds])
    else:
        s = dv.sample(len(dv), replace=True, random_state=int(rng.integers(1 << 31)))
    m = make_ml("xgb", p, seed=seed).fit(s[cols].values, s.Y.values)       # re-fit with fixed (tuned) hyper-parameters
    return m.predict(tr[cols].values)

rows = []
for inputs in ("X", "sep"):
    cols, p = C.INPUT_SETS[inputs], params("xgb", inputs)
    for unit in ("bed", "row"):
        B = np.array(Parallel(n_jobs=C.N_JOBS)(delayed(boot_once)(b, unit, cols, p, 1000 + b) for b in range(C.N_BOOT)))
        mu, sd = B.mean(0), B.std(0, ddof=1)
        lo, hi = np.percentile(B, 5, 0), np.percentile(B, 95, 0)
        y = tr.Y.values
        rows.append(dict(inputs=inputs, unit=unit, B=C.N_BOOT, cover_pm1sd=float(np.mean(np.abs(y - mu) <= sd)),
                         cover_90pct=float(np.mean((y >= lo) & (y <= hi))), mean_sd=float(sd.mean()), mean_abs_err=float(np.abs(y - mu).mean())))
        pd.DataFrame(dict(row_id=tr.row_id.values, bed=tr.bed.values, soil=tr.soil.values, X=tr.X.values, tau=tr.tau.values, Y=y,
                          mean=mu, sd=sd, p05=lo, p95=hi)).to_csv(C.PREDS / f"bootstrap_{inputs}_{unit}.csv", index=False)
pd.DataFrame(rows).round(3).to_csv(C.TABLES / "T_bootstrap.csv", index=False); print(pd.DataFrame(rows))

# ------------------------------------------------ SHAP (development domain only; no pressure-related claims)
sh = []
for inputs in ("X", "sep"):
    cols = C.INPUT_SETS[inputs]
    m = make_ml("xgb", params("xgb", inputs), seed=C.SEED).fit(dv[cols].values, dv.Y.values)
    scaled = m.named_steps["scale"].transform(dv[cols].values)
    sv = shap.TreeExplainer(m.named_steps["est"]).shap_values(scaled)
    for j, c in enumerate(cols):
        sh.append(dict(inputs=inputs, feature=c, mean_abs_shap=float(np.abs(sv[:, j]).mean())))
    np.save(C.PREDS / f"shap_values_{inputs}.npy", sv); dv[cols].to_csv(C.PREDS / f"shap_inputs_{inputs}.csv", index=False)
pd.DataFrame(sh).round(3).to_csv(C.TABLES / "T_shap_dev.csv", index=False); print(pd.DataFrame(sh))

# ------------------------------------------------ partial dependence on X (tau averaged) vs power law
cols = C.INPUT_SETS["X"]
xs = np.geomspace(df.X.min() * 0.9, df.X.max() * 1.05, 120)
taus = np.array([1 / 3, 2 / 3, 1.0])
grid = np.array([[x, t] for x in xs for t in taus])
xgbm = make_ml("xgb", params("xgb", "X"), seed=C.SEED).fit(dv[cols].values, dv.Y.values)
pl = make_param("pl").fit(dv[cols].values, dv.Y.values)
pdp = pd.DataFrame(dict(X=grid[:, 0], tau=grid[:, 1], xgb=xgbm.predict(grid), power_law=pl.predict(grid)))
pdp.groupby("X")[["xgb", "power_law"]].mean().reset_index().to_csv(C.TABLES / "pdp_X.csv", index=False)
json.dump(dict(power_law_X=pl.params, dev_X_max=float(dv.X.max())), open(C.TABLES / "power_law_dev_fit.json", "w"), indent=1)
