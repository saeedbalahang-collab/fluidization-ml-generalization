"""Tables for the row-level reference split (seed 0, 86 fitting / 22 test records): descriptive statistics,
fitting/test scores of the tuned models and of the affine benchmark, and bed-grouped inner-CV fold RMSE."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fluidml import config as C, data as D, models as M, metrics as Mx
from fluidml.protocols import row_split
T = C.TABLES; MD = T / "md"; MD.mkdir(exist_ok=True)
df = D.load(); sp = row_split(df, seeds=[0])[0]
tr, te = df.loc[sp["train"]], df.loc[sp["test"]]
wh = D.transfer(df)
def md(d, p):
    cols = list(d.columns); L = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] + ["---:"] * (len(cols) - 1)) + "|"]
    L += ["| " + " | ".join(str(v) for v in r) + " |" for r in d.values]; Path(p).write_text("\n".join(L) + "\n")
# ---- Table 2
rows = []
for v, lab in (("X", "X"), ("tau", "τ"), ("Y", "Y")):
    for nm, d in (("Fitting (n = 86)", tr), ("Test (n = 22)", te), ("Withheld 55 m (n = 27)", wh)):
        s = d[v]; f = (lambda x: f"{x:.2f}") if v == "X" else (lambda x: f"{x:.3f}")
        rows.append({"Variable": lab, "Dataset": nm, "Min": f(s.min()), "Max": f(s.max()), "Mean": f(s.mean()), "St. dev.": f(s.std()), "Variance": (f"{s.var():.1f}" if v == "X" else f"{s.var():.3f}")})
md(pd.DataFrame(rows), MD / "T2.md")
# ---- Table 4
hp = pd.read_csv(T / "selected_hyperparameters.csv"); hp = hp[(hp.protocol == "row_split") & (hp.fold == "seed0") & (hp.inputs == "X")]
feat = ["X", "tau"]; res = {}
for r in hp.itertuples():
    p = json.loads(r.params); m = M.make_ml(r.model, p).fit(tr[feat].values, tr.Y.values)
    res[r.model] = (m.predict(tr[feat].values), m.predict(te[feat].values))
a = M.make_param("affine").fit(tr[feat].values, tr.Y.values); res["affine"] = (a.predict(tr[feat].values), a.predict(te[feat].values))
NAME = {"xgb": "XGBoost", "rf": "RF", "svr": "SVR", "affine": "Regression benchmark"}
rows = []; out = []
for k in ("xgb", "rf", "svr", "affine"):
    ptr, pte = res[k]; st = Mx.score(tr.Y.values, ptr, tr.d50_mm.values); se = Mx.score(te.Y.values, pte, te.d50_mm.values)
    rows.append({"Model": NAME[k], "R² (Z), fitting": f"{st['R2_Z']:.3f}", "NRMSE (Z), fitting": f"{st['NRMSE_Z']:.3f}", "MAPE (Z, %), fitting": f"{st['MAPE_Z']:.1f}",
                 "R² (Z), test": f"{se['R2_Z']:.3f}", "NRMSE (Z), test": f"{se['NRMSE_Z']:.3f}", "MAPE (Z, %), test": f"{se['MAPE_Z']:.1f}", "R² (Y), test": f"{se['R2_logZ']:.3f}"})
    out.append(dict(model=k, **{f"fit_{a}": b for a, b in st.items()}, **{f"test_{a}": b for a, b in se.items()}))
md(pd.DataFrame(rows), MD / "T4.md"); pd.DataFrame(out).round(4).to_csv(T / "T_row_split_seed0_train_test.csv", index=False)
# ---- Table 3 (bed-grouped inner CV, withheld-pressure protocol, composite inputs)
ic = json.load(open(T / "extra_outputs.json"))["inner_cv"]
rows = [{"Model": NAME[k], **{f"Fold {i+1}": f"{v:.3f}" for i, v in enumerate(ic[k]["folds"])}, "Average": f"{ic[k]['mean']:.3f}"} for k in ("xgb", "rf", "svr")]
md(pd.DataFrame(rows), MD / "T3.md")
print(pd.DataFrame(rows).to_string()); print(pd.DataFrame(out).round(3).T.to_string())
